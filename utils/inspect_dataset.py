import sys
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd
from pathlib import Path

DATA_DIR = Path("outputs/datasets/SO101-1")
REPO_ID = "edgarcancinoe/soarm101_pickup_orange"

try:
    dataset = LeRobotDataset(repo_id=REPO_ID, root=DATA_DIR)
    print(f"Dataset Loaded: {REPO_ID}")
    print(f"Total Episodes: {dataset.num_episodes}")
    
    # Print last few episodes
    print("\nLast 5 Episodes:")
    if dataset.num_episodes > 0:
        meta = dataset.meta.episodes
        # It's a list of dicts, let's print the last few
        for i in range(max(0, dataset.num_episodes - 5), dataset.num_episodes):
            ep = meta[i]
            print(f"  Index: {ep['episode_index']}, Length: {ep['length']}, Path: {dataset.meta.get_data_file_path(i)}")
            
except Exception as e:
    print(f"Error loading dataset: {e}")
