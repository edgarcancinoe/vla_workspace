import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
src_root = ROOT_DIR / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from thesis_vla.common.paths import DATASETS_OUTPUT_DIR

DATA_DIR = DATASETS_OUTPUT_DIR / "SO101-1"
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
