import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

DATA_DIR = Path("outputs/datasets/soarm101_pickplace_local")
HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickplace"

def main():
    print(f"Verifying dataset at {DATA_DIR}...")
    
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    try:
        dataset = LeRobotDataset(
            repo_id=f"{HF_USER}/{HF_REPO_ID}",
            root=DATA_DIR,
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Dataset loaded successfully!")
    print(f"  Num Episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Check stats if available
    if hasattr(dataset, 'stats'):
        print(f"  Stats available: {list(dataset.stats.keys())}")
    else:
        print("  Stats NOT available (expected if skipped).")

    # Check a few frames
    print("Checking first episode frames...")
    try:
        item = dataset[0]
        print(f"  Frame 0 keys: {list(item.keys())}")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape} {v.dtype}")
    except Exception as e:
        print(f"  Error reading frame 0: {e}")

    print("Verification complete.")

if __name__ == "__main__":
    main()
