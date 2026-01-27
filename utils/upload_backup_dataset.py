#!/usr/bin/env python3
"""
Upload the backed-up dataset to a new HuggingFace repository.
This script uploads the 28-episode recording from SO101-F7-backup-20260121_191745
to a new repository to preserve it separately from the original dataset.
"""

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
BACKUP_DIR = Path("/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/outputs/datasets/SO101-F7-backup-20260121_191745")
HF_USER = "edgarcancinoe"
NEW_REPO_ID = "soarm101_pick_cubes_place_box_v2"  # New repository name

def main():
    print(f"Loading dataset from backup: {BACKUP_DIR}")
    
    # Load the dataset from the backup directory
    dataset = LeRobotDataset(
        repo_id=f"{HF_USER}/{NEW_REPO_ID}",
        root=BACKUP_DIR,
    )
    
    print(f"Dataset loaded:")
    print(f"  - Episodes: {dataset.num_episodes}")
    print(f"  - Total frames: {dataset.num_frames}")
    print(f"  - Repository: {HF_USER}/{NEW_REPO_ID}")
    print()
    
    # Push to HuggingFace
    print(f"Uploading to HuggingFace repository: {HF_USER}/{NEW_REPO_ID}")
    print("This may take several minutes...")
    
    dataset.push_to_hub()
    
    print()
    print("✅ Upload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{HF_USER}/{NEW_REPO_ID}")

if __name__ == "__main__":
    main()
