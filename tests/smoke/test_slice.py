import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "repos" / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset("edgarcancinoe/so101_joint")
    print(dataset.meta.features["action"])
    print(dataset.meta.stats["action"]["mean"].shape)
except Exception as e:
    print("Error:", e)
