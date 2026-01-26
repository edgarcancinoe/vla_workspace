
import sys
from pathlib import Path

# Adjust path to find lerobot
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "edgarcancinoe/soarm101_pickup_orange"
LOCAL_DIR = Path("outputs/datasets/SO101-1")

def main():
    ds = LeRobotDataset(repo_id=REPO_ID, root=LOCAL_DIR)
    print("Keys in meta.episodes[0]:")
    print(ds.meta.episodes[0].keys())
    print("\nValue of meta.episodes[0]:")
    print(ds.meta.episodes[0])

if __name__ == "__main__":
    main()
