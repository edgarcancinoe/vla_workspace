#!/usr/bin/env python3
"""
Standalone script to visualize an episode from a LeRobot dataset
using the SOARM101 URDF and Pinocchio Meshcat integration.
This version interprets `observation.state` as a 10D array:
  [x, y, z, rot6d_0 ... rot6d_5, gripper]
and uses Inverse Kinematics to calculate the required joint angles
before sending them to the visualizer.
"""

from pathlib import Path
import numpy as np

import sys
import os

from lerobot.model.kinematics import RobotKinematics

# --- Configuration ---
DATASET_ID = "edgarcancinoe/soarm101_pickplace_6d"
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" 
VISUALIZE_EPISODE = 40
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
# ---------------------

def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation back to a 3x3 rotation matrix using Gram-Schmidt.
    """
    assert rot_6d.shape[-1] == 6, "Input must strictly be a 6D array."
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]
    
    b1 = a1 / np.linalg.norm(a1)
    
    dot = np.sum(b1 * a2)
    proj = dot * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2)
    
    b3 = np.cross(b1, b2)
    
    matrix = np.stack([b1, b2, b3], axis=-1)
    return matrix


def create_target_pose(pos, rot_6d):
    """
    Create a 4x4 transformation matrix for a target homogenous position given the positional
    coordinates (x, y, z) and the flattened 6D rotation vector.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_6d_to_matrix(rot_6d)
    matrix[:3, 3] = pos
    return matrix


def visualize_episode_trajectory(q_list, urdf_path, episode_idx):
    """
    Visualizes the robot's joint states over time using Meshcat and Pinocchio.
    """
    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
        import time
    except ImportError:
        print("pinocchio or meshcat not installed. Skipping visualization.")
        return

    print(f"\nInitializing Meshcat visualizer for episode {episode_idx}...")
    package_dir = str(Path(urdf_path).parent)
    model, cmod, vmod = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dir)
    viz = MeshcatVisualizer(model, cmod, vmod)
    
    try:
        viz.initViewer(open=False)
    except Exception as e:
        print("Couldn't open meshcat automatically:", e)
    
    viz.loadViewerModel()
    print(f"=========================================================")
    print(f"Meshcat running at: {viz.viewer.url()}")
    print(f"--> Open the URL in your browser to see the simulation!")
    print(f"=========================================================")

    try:
        print(f"Playing back recorded episode {episode_idx}...")
        
        for _ in range(3): # Loop the playback 3 times
            for q in q_list:
                viz.display(q)
                time.sleep(1.0 / 30.0) # Assume roughly 30 fps
            time.sleep(1.0)

        print("Visualization complete. Press Ctrl+C to exit and close the server...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[SIGINT caught] Shutting down Meshcat visualizer!")
        try:
            # We enforce sys.exit to explicitly demand python teardown all spawned children threads
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def main():
    if not URDF_PATH:
        print("Please configure URDF_PATH at the top of the script.")
        return

    print(f"Loading local dataset: {DATASET_ID}")
    dataset_root = Path.home() / ".cache" / "lerobot" / DATASET_ID
    
    print(f"Reading frames for episode {VISUALIZE_EPISODE}...")
    
    data_dir = dataset_root / "data"
    parquet_files = sorted(list(data_dir.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Could not find parquet files in {data_dir}")

    visualize_q_list = []
    
    import pyarrow.parquet as pq

    print("Initializing kinematics solver...")
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES[:-1], # Gripper is not actively inverted
    )

    # Use a neutral "seed" zero'd start or any pre-known static seed position internally in degrees
    current_joints_deg = np.zeros(len(JOINT_NAMES) - 1) 
    
    print("Reading data and solving Inverse Kinematics iteratively...")
    
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        if "observation.state" not in df.columns:
            continue
            
        episode_df = df[df["episode_index"] == VISUALIZE_EPISODE]
        
        for i in range(len(episode_df)):
            eef_state = episode_df.iloc[i]["observation.state"]
            
            if len(eef_state) != 10:
                print(f"Warning: observation.state is length {len(eef_state)}, expected 10. Skipping.")
                continue

            pos = eef_state[0:3]
            rot_6d = eef_state[3:9]
            gripper = eef_state[9]

            target_pose = create_target_pose(pos, rot_6d)

            # Solve IK frame by frame, seeding with the previous solution for path continuity
            target_joints_deg = kinematics.inverse_kinematics(
                current_joint_pos=current_joints_deg,
                desired_ee_pose=target_pose,
                position_weight=1.0,
                orientation_weight=1.0, 
            )

            current_joints_deg = target_joints_deg # Keep updating seed

            target_joints_rad = np.deg2rad(target_joints_deg)
            
            # Reconstruct the 6-motor action (Joints + Gripper) in Radians for Meshcat
            q = np.concatenate([target_joints_rad, [np.deg2rad(gripper)]])
            visualize_q_list.append(q)

    if len(visualize_q_list) > 0:
        visualize_episode_trajectory(visualize_q_list, URDF_PATH, VISUALIZE_EPISODE)
    else:
        print(f"Episode {VISUALIZE_EPISODE} not found or contains no compatible 10D EEF state data.")


if __name__ == "__main__":
    main()
