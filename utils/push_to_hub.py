
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
# This must match where record_episodes.py saves the data
DATA_DIR = Path("outputs/datasets/soarm101_pickplace_local")
HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickplace_top_wrist"

def main():
    print(f"Loading dataset from {DATA_DIR}...")
    
    # Instantiate dataset pointing to local files
    ds = LeRobotDataset(
        repo_id=f"{HF_USER}/{HF_REPO_ID}",
        root=DATA_DIR,
        video_backend="pyav"
    )
    
    print(f"Dataset loaded. Episodes: {ds.num_episodes}")
    print(f"Pushing to Hub: {HF_USER}/{HF_REPO_ID}")
    
    # Push
    ds.push_to_hub()
    print("Push completed successfully!")

if __name__ == "__main__":
    main()
