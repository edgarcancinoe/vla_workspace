#!/usr/bin/env python3
"""
Standalone script to visualize an episode from a LeRobot dataset
using the SOARM101 URDF and Pinocchio Meshcat integration.
"""

from pathlib import Path
import numpy as np

import sys
import os

# --- Configuration ---
DATASET_ID = "edgarcancinoe/soarm101_pickplace_orange_050e_fw_open"
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" 
VISUALIZE_EPISODE = 40
# ---------------------

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

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("lerobot is not installed properly. Skipping dataset operations.")
        return

    print(f"Loading dataset: {DATASET_ID} (This will download it if not present locally)")
    dataset = LeRobotDataset(DATASET_ID, download_videos=False)
    
    # We must restrict to the requested episode
    episodes = [VISUALIZE_EPISODE]
    print(f"Reading frames for episode {VISUALIZE_EPISODE}...")
    
    # Get all the parquet chunk files containing the trajectory data
    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(list(data_dir.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Could not find parquet files in {data_dir}")

    visualize_q_list = []
    
    import pyarrow.parquet as pq

    # Read the data to find the matching episode index
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        if "observation.state" not in df.columns:
            continue
            
        episode_df = df[df["episode_index"] == VISUALIZE_EPISODE]
        
        for i in range(len(episode_df)):
            if "observation.joint_positions" in episode_df.columns:
                state = episode_df.iloc[i]["observation.joint_positions"]
            else:
                state = episode_df.iloc[i]["observation.state"]
            # Convert degrees to radians for pinocchio
            visualize_q_list.append(np.deg2rad(state))

    if len(visualize_q_list) > 0:
        visualize_episode_trajectory(visualize_q_list, URDF_PATH, VISUALIZE_EPISODE)
    else:
        print(f"Episode {VISUALIZE_EPISODE} not found or contains no state data.")


if __name__ == "__main__":
    main()
