#!/usr/bin/env python3
"""
Standalone script to visualize an episode from a LeRobot dataset
using the SOARM101 URDF via Meshcat or real robot execution.
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

from lerobot.utils.robot_utils import precise_sleep
from robot_control.so101_control import SO101Control
from robot_sim.so101_meshcat import SO101Meshcat

# --- Configuration ---
DATASET_ID = "edgarcancinoe/soarm101_pickplace_orange_050e_fw_open"
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" 
VISUALIZE_EPISODE = 40
# ---------------------

def run_meshcat(q_rad_list, urdf_path, episode_idx):
    """
    Visualizes the robot's joint states over time using Meshcat.
    """
    print(f"\nInitializing Meshcat visualizer for episode {episode_idx}...")
    viz = SO101Meshcat(urdf_path)

    try:
        print(f"Playing back recorded episode {episode_idx}...")
        
        for _ in range(3): # Loop the playback 3 times
            for q_rad in q_rad_list:
                viz.display(q_rad)
                time.sleep(1.0 / 30.0)
            time.sleep(1.0)

        print("Visualization complete. Press Ctrl+C to exit and close the server...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[SIGINT caught] Shutting down Meshcat visualizer!")
        viz.disconnect()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def run_real(q_deg_list, kinematics):
    """
    Executes the joint trajectory on the real robot.
    """
    import yaml
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        robot_cfg = cfg.get("robot", {})
        port = robot_cfg.get("port")
        robot_name = robot_cfg.get("name", "arm_follower")
        calibration_dir = Path(__file__).parent.parent / ".cache" / "calibration"
    else:
        raise ValueError("Error: config/robot_config.yaml not found. Cannot connect to robot.")

    if not port:
        raise ValueError("Error: 'port' not found in config/robot_config.yaml. Cannot connect to robot.")

    print(f"Connecting to real robot '{robot_name}' on port {port}...")
    robot = SO101Follower(SO101FollowerConfig(id=robot_name, port=port, calibration_dir=calibration_dir))
    robot.connect()
    print("Robot connected.")

    print("Moving to home position first...")
    kinematics.reset_to_home(robot, duration_s=3.0, fps=30.0)
    time.sleep(1.0)

    print("Moving to start position...")
    start_deg = kinematics.read_deg_real(robot, ignore_offset=True)
    steps = max(1, int(round(3.0 * 30.0))) # 3 seconds to start pos at 30 fps
    waypoints_start = kinematics.interpolate_joint(start_deg, q_deg_list[0], steps)
    kinematics.execute_joint_trajectory(robot, waypoints_start, fps=30.0, ignore_offset=True)
    time.sleep(1.0)

    print("Executing dataset trajectory...")
    try:
        kinematics.execute_joint_trajectory(robot, q_deg_list, fps=30.0, ignore_offset=True)
        print("Trajectory complete.")
    except KeyboardInterrupt:
        print("\n[SIGINT caught] Stopping early!")
    finally:
        print("Returning to home...")
        kinematics.reset_to_home(robot, duration_s=3.0, fps=30.0)
        time.sleep(0.5)
        print("Disconnecting...")
        robot.disconnect()


def main():
    if not URDF_PATH:
        print("Please configure URDF_PATH at the top of the script.")
        return

    parser = argparse.ArgumentParser(description="Visualize Dataset Episode Joints")
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument("--sim", action="store_true", help="Simulation (Meshcat) mode")
    exec_mode.add_argument("--real", action="store_true", help="Real robot mode")
    parser.add_argument("--episode", type=int, default=VISUALIZE_EPISODE, help="Episode index to play")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("lerobot is not installed properly. Skipping dataset operations.")
        return

    # Initialize kinematics for unit conversions
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    wrist_roll_offset = 0.0
    home_pose = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        robot_cfg = cfg.get("robot", {})
        wrist_roll_offset = float(robot_cfg.get("wrist_roll_offset", 0.0))
        home_pose = robot_cfg.get("home_pose")

    kinematics = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=wrist_roll_offset, home_pose=home_pose)
    
    print(f"Loading dataset: {DATASET_ID} (This will download it if not present locally)")
    dataset = LeRobotDataset(DATASET_ID, download_videos=False)
    
    print(f"Reading frames for episode {args.episode}...")
    
    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(list(data_dir.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Could not find parquet files in {data_dir}")

    q_rad_list = []
    q_deg_list = []
    
    import pyarrow.parquet as pq

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        if "observation.state" not in df.columns and "observation.joint_positions" not in df.columns:
            continue
            
        episode_df = df[df["episode_index"] == args.episode]
        if episode_df.empty:
            continue

        col = "observation.joint_positions" if "observation.joint_positions" in episode_df.columns else "observation.state"
        print(f"Using {col} from {parquet_path.name}")
        
        # LeRobot datasets for SO101 log raw motor units directly.
        q_motor_chunk = np.stack(episode_df[col].values)
        
        if args.real:
            # We only need degrees for the real robot execution
            # TODO: Remove this when dataset is fixed
            #########################################
            q_motor_chunk[:, 4] *= -1 
            #########################################
            q_deg_chunk = kinematics.motor_to_deg(q_motor_chunk)
            q_deg_list.extend(q_deg_chunk)
        else:
            # We only need radians for Meshcat simulation
            q_motor_chunk[:, 4] *= -1             
            q_rad_chunk = kinematics.motor_to_rad(q_motor_chunk, use_polarities=False)
            q_rad_list.extend(q_rad_chunk)

    final_list = q_deg_list if args.real else q_rad_list
    if len(final_list) == 0:
        print(f"Episode {args.episode} not found or contains no state data.")
        return

    print(f"Loaded {len(final_list)} frames.")

    if args.real:
        run_real(q_deg_list, kinematics)
    else:
        run_meshcat(q_rad_list, URDF_PATH, args.episode)

if __name__ == "__main__":
    main()

