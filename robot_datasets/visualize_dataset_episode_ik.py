#!/usr/bin/env python3
"""
Standalone script to visualize an episode from a LeRobot dataset
using the SOARM101 URDF and Pinocchio Meshcat integration.

observation.state is a 10D array:
  [x, y, z, rot6d_0..rot6d_5, gripper]
  where gripper is in MOTOR UNITS (not degrees, not radians).

Unit reference (from SO101Control):
  motor units  <-- motor_to_rad -->  radians  <-- rad2deg/deg2rad -->  degrees
  inverse_kinematics : takes degrees, returns degrees
  wrist_roll_offset  : stored in motor units
  Meshcat/Pinocchio  : expects radians
"""

from pathlib import Path
import numpy as np
import sys
import os
import argparse
import time

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

LEROBOT_SRC = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from robot_control.so101_control import SO101Control
from lerobot.utils.robot_utils import precise_sleep

# --- Configuration ---
DATASET_ID = "edgarcancinoe/soarm101_pickplace_6d"
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
VISUALIZE_EPISODE = 40
# ---------------------

robot = None
_robot_wrist_roll_offset = 0.0  # motor units


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def create_target_pose(pos, rot_6d):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_6d_to_matrix(rot_6d)
    matrix[:3, 3] = pos
    return matrix


# ---------------------------------------------------------------------------
# Robot connection
# ---------------------------------------------------------------------------

def connect_robot():
    """Connect to real robot, return (robot, wrist_roll_offset in motor units)."""
    global robot, _robot_wrist_roll_offset
    import yaml
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    wrist_roll_offset = 0.0
    port = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        wrist_roll_offset = float(cfg.get("robot", {}).get("wrist_roll_offset", 0.0))
        port = cfg.get("robot", {}).get("port")

    if not port:
        raise ValueError("Error: 'port' not found in config/robot_config.yaml. Cannot connect to robot.")

    robot_config = SO101FollowerConfig(id="arm_follower", port=port)
    print(f"=== Initializing Robot ===")
    print(robot_config)
    robot = SO101Follower(robot_config)
    robot.connect()
    print(f"==========================")
    _robot_wrist_roll_offset = wrist_roll_offset
    return robot, wrist_roll_offset


def read_robot_seed_deg(kinematics):
    """
    Read current robot joint positions (motor units) and convert to degrees
    for the active joints only (no gripper), ready to seed inverse_kinematics().
    """
    obs = robot.get_observation()
    motor_vals = np.array([obs[f"{n}.pos"] for n in kinematics.JOINT_NAMES])
    seed_deg = np.rad2deg(kinematics.motor_to_rad(motor_vals))
    active_indices = [kinematics.JOINT_NAMES.index(j) for j in kinematics.active_joints]
    return seed_deg[active_indices]


# ---------------------------------------------------------------------------
# Sending commands to the real robot
# ---------------------------------------------------------------------------

def send_to_real_robot(arm_rad, gripper_motor, kinematics, should_log=True):
    """
    arm_rad      : np.array[len(active_joints)] in RADIANS  (IK output)
    gripper_motor: float in MOTOR UNITS  (taken directly from observation.state[9])

    Pipeline for arm joints:
      radians -> motor units -> send
    Pipeline for gripper:
      motor units -> send as-is
    """
    arm_motor = kinematics.rad_to_motor(arm_rad)

    action = {f"{n}.pos": float(arm_motor[i]) for i, n in enumerate(kinematics.active_joints)}
    action["gripper.pos"] = float(gripper_motor)

    if should_log:
        print("\n--- Sending to Real Robot ---")
        print(f"IK output (rad)        : {np.round(arm_rad, 4)}")
        print(f"Sent to Robot (motor)  : {np.round(arm_motor, 4)}")
        print(f"Sent Gripper (motor)   : {gripper_motor:.2f}")

    robot.send_action(action)


# ---------------------------------------------------------------------------
# Build trajectory from dataset
# ---------------------------------------------------------------------------

def build_trajectory(parquet_files, episode_idx, kinematics):
    """
    Returns:
      joints_motor_list : list of np.array[6] motor values (direct from observation.joint_positions)
      xyz_list          : list of np.array EEF positions (from observation.state)
    """
    import pyarrow.parquet as pq

    joints_motor_list = []
    xyz_list = []

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "observation.state" not in df.columns:
            continue

        episode_df = df[df["episode_index"] == episode_idx]
        if episode_df.empty:
            continue

        for i in range(len(episode_df)):
            eef_state = episode_df.iloc[i]["observation.state"]
            obs_joints = episode_df.iloc[i].get("observation.joint_positions")

            if len(eef_state) != 10:
                print(f"  Warning: state length {len(eef_state)}, expected 10. Skipping frame.")
                continue

            xyz_list.append(np.array(eef_state[0:3]))

            if obs_joints is not None:
                joints_motor_list.append(np.array(obs_joints))

    return joints_motor_list, xyz_list


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

def run_real(joints_motor_list, kinematics):
    """Direct joint replay: send motor values from dataset straight to robot."""
    if not joints_motor_list:
        print("No observation.joint_positions found — cannot replay.")
        return
    print("Executing on real robot (direct joint replay)...")
    try:
        for i, joints_motor in enumerate(joints_motor_list):
            action = {f"{n}.pos": float(joints_motor[j]) for j, n in enumerate(kinematics.JOINT_NAMES)}
            if i % 10 == 0:
                print(f"\nFrame {i}: motor={np.round(joints_motor, 2)}")
            robot.send_action(action)
            precise_sleep(1.0 / 30.0)
        print("Done. Disconnecting...")
        robot.disconnect()
    except KeyboardInterrupt:
        print("\n[SIGINT] Stopping early, disconnecting...")
        robot.disconnect()
        sys.exit(0)


def run_meshcat(joints_motor_list, xyz_list, kinematics, episode_idx):
    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
        import meshcat.geometry as g
        import meshcat.transformations as tf
    except ImportError:
        print("pinocchio or meshcat not installed.")
        return

    print(f"\nInitializing Meshcat for episode {episode_idx}...")
    package_dir = str(Path(kinematics.urdf_path).parent)
    model, cmod, vmod = pin.buildModelsFromUrdf(kinematics.urdf_path, package_dirs=package_dir)
    viz = MeshcatVisualizer(model, cmod, vmod)

    try:
        viz.initViewer(open=False)
    except Exception as e:
        print("Couldn't open meshcat automatically:", e)

    viz.loadViewerModel()
    
    print(f"=========================================================")
    print(f"Meshcat: {viz.viewer.url()}")
    print(f"=========================================================")

    # Draw EEF trajectory path
    if viz.viewer:
        for i in range(len(xyz_list) - 1):
            p1, p2 = xyz_list[i], xyz_list[i + 1]
            diff = p2 - p1
            L = np.linalg.norm(diff)
            if L > 1e-6:
                z = np.array([0., 0., 1.])
                axis = np.cross(z, diff / L)
                ang = np.arccos(np.clip(np.dot(z, diff / L), -1, 1))
                R = tf.rotation_matrix(ang, axis) if np.linalg.norm(axis) > 1e-6 else np.eye(4)
                viz.viewer[f"trajectory/seg_{i}"].set_object(
                    g.Cylinder(float(L), 0.002), g.MeshLambertMaterial(color=0xffa500))
                viz.viewer[f"trajectory/seg_{i}"].set_transform(
                    tf.translation_matrix((p1 + p2) / 2) @ R)
            viz.viewer[f"trajectory/pt_{i}"].set_object(
                g.Sphere(0.004), g.MeshLambertMaterial(color=0x00ff00))
            viz.viewer[f"trajectory/pt_{i}"].set_transform(tf.translation_matrix(p1))

        if xyz_list:
            viz.viewer[f"trajectory/pt_{len(xyz_list)-1}"].set_object(
                g.Sphere(0.006), g.MeshLambertMaterial(color=0xff0000))
            viz.viewer[f"trajectory/pt_{len(xyz_list)-1}"].set_transform(
                tf.translation_matrix(xyz_list[-1]))

    try:
        print(f"Playing back episode {episode_idx} (3 loops, direct joint replay)...")
        for _ in range(3):
            for i, joints_motor in enumerate(joints_motor_list):
                q_display = kinematics.motor_to_rad(joints_motor)  # motor -> rad, all 6 joints
                viz.display(q_display)

                if i % 30 == 0:  # print roughly once a second
                    print(f"\n--- Frame {i} ---")
                    print(f"Motor (raw) : {np.round(joints_motor, 2)}")
                    print(f"Joints (rad): {np.round(q_display, 4)}")
                    print(f"Joints (deg): {np.round(np.rad2deg(q_display), 2)}")

                time.sleep(1.0 / 30.0)
            time.sleep(1.0)

        print("Done. Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[SIGINT] Shutting down Meshcat.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize or Execute Episode")
    parser.add_argument("--real", action="store_true", help="Execute on real robot instead of Meshcat")
    parser.add_argument("--episode", type=int, default=VISUALIZE_EPISODE)
    args = parser.parse_args()

    import yaml
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    wrist_offset = 0.0
    home_pose = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        wrist_offset = float(cfg.get("robot", {}).get("wrist_roll_offset", 0.0))
        home_pose = cfg.get("robot", {}).get("home_pose")

    kinematics = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=wrist_offset, home_pose=home_pose)

    if args.real:
        connect_robot()

    data_dir = Path.home() / ".cache" / "lerobot" / DATASET_ID / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Building trajectory for episode {args.episode}...")
    joints_motor_list, xyz_list = build_trajectory(parquet_files, args.episode, kinematics)

    if not joints_motor_list:
        print(f"Episode {args.episode} not found or has no observation.joint_positions data.")
        return

    print(f"Loaded {len(joints_motor_list)} frames.")

    if args.real:
        run_real(joints_motor_list, kinematics)
    else:
        run_meshcat(joints_motor_list, xyz_list, kinematics, args.episode)


if __name__ == "__main__":
    main()