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

# Add the workspace root to sys.path to allow imports from robot_control
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from robot_control.so101_control import SO101Control
from lerobot.utils.robot_utils import precise_sleep

# --- Configuration ---
DATASET_ID = "edgarcancinoe/soarm101_pickplace_6d"
VISUALIZE_EPISODE = 40
# Interpolation parameters for moving to the first frame
START_MOVE_DURATION = 3.0  # seconds
START_MOVE_FPS = 10.0      # waypoints per second
# ---------------------

robot = None


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
    """Connect to real robot."""
    global robot
    import yaml
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    port = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        port = cfg.get("robot", {}).get("port")
        robot_name = cfg.get("robot", {}).get("name", "arm_follower")

    if not port:
        raise ValueError("Error: 'port' not found in config/robot_config.yaml. Cannot connect to robot.")

    calibration_dir = Path(__file__).parent.parent / ".cache" / "calibration"
    robot_config = SO101FollowerConfig(id=robot_name, port=port, calibration_dir=calibration_dir)
    print(f"=== Initializing Robot ===")
    print(robot_config)
    robot = SO101Follower(robot_config)
    robot.connect()
    print(f"==========================")
    return robot


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
# Debugging / Logging
# ---------------------------------------------------------------------------

def log_robot_state(frame_idx, joints_motor, kinematics, label="Robot State"):
    """Prints detailed information about the robot state/command."""
    joints_deg = kinematics.motor_to_deg(joints_motor)
    joints_rad = kinematics.motor_to_rad(joints_motor)
    xyz = kinematics.fk_xyz(joints_motor)
    
    print(f"\n--- {label} | Frame {frame_idx} ---")
    print(f"EEF Position (XYZ) : {np.round(xyz, 4)}")
    print(f"Joints Detail:")
    header = f"{'Joint Name':15s} | {'Motor':>8s} | {'Degrees':>8s} | {'Radians':>8s}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(kinematics.JOINT_NAMES):
        m = joints_motor[i]
        d = joints_deg[i]
        r = joints_rad[i]
        print(f"{name:15s} | {m:8.2f} | {d:8.2f} | {r:8.4f}")


# ---------------------------------------------------------------------------
# Build trajectory from dataset
# ---------------------------------------------------------------------------

def build_trajectory(parquet_files, episode_idx, kinematics, is_real):
    """
    Returns:
      joints_list : list of joint values (deg if is_real, else rad)
      xyz_list    : list of np.array EEF positions (from observation.state)
    """
    import pyarrow.parquet as pq

    joints_list = []
    xyz_list = []

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "observation.state" not in df.columns:
            continue

        episode_df = df[df["episode_index"] == episode_idx]
        if episode_df.empty:
            continue
        
        if "observation.joint_positions" in episode_df.columns:
            q_motor_chunk = np.stack(episode_df["observation.joint_positions"].values)
        elif "observation.state" in episode_df.columns and len(episode_df["observation.state"].iloc[0]) == 6:
            # Fallback for datasets where observation.state contains the 6 joints
            q_motor_chunk = np.stack(episode_df["observation.state"].values)
        else:
            continue

        # Dataset-specific wrist_roll fix (Apply to motor units directly)
        q_motor_chunk[:, 4] *= -1

        if is_real:
            # Need degrees for real robot
            joints_chunk = kinematics.motor_to_deg(q_motor_chunk)
        else:
            # Need radians for Meshcat (ignore polarities for sim to match URDF axes)
            joints_chunk = kinematics.motor_to_rad(q_motor_chunk, use_polarities=False)
        
        joints_list.extend(joints_chunk)

        # Recompute XYZ trajectory from joints to ensure it matches the current URDF/Kinematics
        # For simulation, we must use use_polarities=False to match the visual arm's joint axes
        xyz_chunk = kinematics.fk_xyz_chunk(q_motor_chunk, use_polarities=is_real)
        xyz_list.extend(xyz_chunk)

    return joints_list, xyz_list


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

def run_real(joints_deg_list, kinematics):
    """Direct joint replay: send joint degrees from dataset straight to robot."""
    if not joints_deg_list:
        print("No joint data found — cannot replay.")
        return
    print("Executing on real robot (direct joint replay)...")
    try:
        # Move to start position first
        print("Moving to home position...")
        kinematics.reset_to_home(robot, duration_s=3.0, fps=30.0)
        time.sleep(1.0)

        print("Moving to start position...")
        start_deg = kinematics.read_deg_real(robot, ignore_offset=True)
        steps = int(START_MOVE_DURATION * START_MOVE_FPS)
        waypoints_start = kinematics.interpolate_joint(start_deg, joints_deg_list[0], steps)
        kinematics.execute_joint_trajectory(robot, waypoints_start, fps=30.0, ignore_offset=True)
        time.sleep(1.0)

        print("Executing dataset trajectory...")
        kinematics.execute_joint_trajectory(robot, joints_deg_list, fps=30.0, ignore_offset=True)
        
        print("Done. Returning home and disconnecting...")
        kinematics.reset_to_home(robot, duration_s=3.0, fps=30.0)
        robot.disconnect()
    except KeyboardInterrupt:
        print("\n[SIGINT] Stopping early, returning home and disconnecting...")
        try:
            kinematics.reset_to_home(robot, duration_s=2.0)
        except: pass
        robot.disconnect()
        sys.exit(0)


def run_meshcat(joints_rad_list, xyz_list, kinematics, episode_idx):
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
            for i, q_display in enumerate(joints_rad_list):
                viz.display(q_display)

                if i % 100 == 0:  # print roughly once a second
                    # Convert back to motor for logging consistency
                    log_robot_state(i, kinematics.rad_to_motor(q_display, use_polarities=False), kinematics, label="Meshcat Visualization")

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
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument("--sim", action="store_true", help="Simulation (Meshcat) mode")
    exec_mode.add_argument("--real", action="store_true", help="Real robot mode")
    parser.add_argument("--episode", type=int, default=VISUALIZE_EPISODE)
    args = parser.parse_args()

    import yaml
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    wrist_offset = 0.0
    home_pose = None
    urdf_path = None
    
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        robot_cfg = cfg.get("robot", {})
        wrist_offset = float(robot_cfg.get("wrist_roll_offset", 0.0))
        home_pose = robot_cfg.get("home_pose")
        urdf_path = robot_cfg.get("urdf_path")

    if not urdf_path:
        raise ValueError("Error: 'urdf_path' not found in config/robot_config.yaml. This is required for kinematics.")

    kinematics = SO101Control(urdf_path=urdf_path, wrist_roll_offset=wrist_offset, home_pose=home_pose)

    if args.real:
        connect_robot()

    data_dir = Path.home() / ".cache" / "lerobot" / DATASET_ID / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Building trajectory for episode {args.episode}...")
    joints_list, xyz_list = build_trajectory(parquet_files, args.episode, kinematics, is_real=args.real)

    if not joints_list:
        print(f"Episode {args.episode} not found or has no observation.joint_positions data.")
        return

    print(f"Loaded {len(joints_list)} frames.")

    if args.real:
        run_real(joints_list, kinematics)
    else:
        run_meshcat(joints_list, xyz_list, kinematics, args.episode)


if __name__ == "__main__":
    main()