import sys
sys.path.insert(0, "/home/jose/vla_workspace/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset("edgarcancinoe/so101_joint")
    print(dataset.meta.features["action"])
    print(dataset.meta.stats["action"]["mean"].shape)
except Exception as e:
    print("Error:", e)
