from lerobot.datasets.lerobot_dataset import LeRobotDataset
import logging

logging.basicConfig(level=logging.INFO)

dataset_name = "edgarcancinoe/soarm101_pickplace_orange_050e_fw_open"
dataset = LeRobotDataset(dataset_name)
print(f"Dataset features: {dataset.meta.features}")
print(f"Action shape: {dataset.meta.features['action']['shape']}")
