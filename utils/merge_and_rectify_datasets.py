import sys
from pathlib import Path
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.utils.utils import log_say # Silence TTS
def log_say(msg):
    print(msg)
import numpy as np

# Add project root to path to find utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera_calibration

HF_USER = "edgarcancinoe"
SOURCE_DATASETS = [
    f"{HF_USER}/soarm101_pickup_orange",
    f"{HF_USER}/soarm101_pick_cubes_place_box"
]
TARGET_DATASET = f"{HF_USER}/soarm101_pickplace"
TARGET_ROOT = Path(__file__).parent.parent / "outputs/datasets/soarm101_pickplace"

# Load config to check if rectification is globally enabled (optional, but good practice)
CONFIG_PATH = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config_data = yaml.safe_load(f)

# Allow overrides if needed, but defaults to config
RECTIFY_TOP = config_data.get("rectification", {}).get("top", True)
RECTIFY_WRIST = config_data.get("rectification", {}).get("wrist", True)

def main():
    log_say(f"Starting merge and rectification into {TARGET_DATASET}")
    log_say(f"Rectification Config: Top={RECTIFY_TOP}, Wrist={RECTIFY_WRIST}")

    # Create target dataset (initialized with features from first source)
    target_dataset = None
    
    for source_repo in SOURCE_DATASETS:
        log_say(f"Processing source: {source_repo}")
        
        # Load source dataset with pyav to avoid torchcodec issues
        source_ds = LeRobotDataset(source_repo, video_backend="pyav")
        
        if target_dataset is None:
            # Initialize target dataset using features from the first source
            if TARGET_ROOT.exists():
                import shutil
                shutil.rmtree(TARGET_ROOT)
            
            target_dataset = LeRobotDataset.create(
                repo_id=TARGET_DATASET,
                fps=source_ds.fps,
                features=source_ds.features,
                robot_type=source_ds.meta.robot_type,
                use_videos=True,
                root=TARGET_ROOT,
                video_backend="pyav"
            )
        
        # Iterate over episodes
        for ep_idx in range(source_ds.num_episodes):
            log_say(f"  > Processing episode {ep_idx} from {source_repo}")
            
            # Get episode range
            ep_info = source_ds.meta.episodes[ep_idx]
            start_idx = ep_info["dataset_from_index"]
            end_idx = ep_info["dataset_to_index"]
            
            for i in range(start_idx, end_idx):
                # Load frame (handles video decoding)
                frame = source_ds[i]
                
                # Prepare frame for add_frame
                # We need to build clean_frame with only valid features (no defaults like index, timestamp, etc.)
                # And we need to ensure all images are HWC numpy arrays.
                
                clean_frame = {}
                
                # We also need to add 'task'
                if "task" in frame:
                    clean_frame["task"] = frame["task"]

                for key, ft_spec in target_dataset.features.items():
                    # Skip if key not in frame
                    if key not in frame:
                        continue
                        
                    # Skip default features to allow auto-generation and validation pass
                    # These keys are often in 'features' but add_frame manages them
                    if key in ["index", "episode_index", "frame_index", "timestamp", "task_index", "next.done"]:
                         continue
                    
                    value = frame[key]
                    
                    # Handle Images and Videos
                    if ft_spec["dtype"] in ["image", "video"]:
                        # Value is likely Tensor (C, H, W) float32 [0,1]
                        if isinstance(value, torch.Tensor):
                            # Convert to Numpy (H, W, C) uint8
                            img_np = (value.permute(1, 2, 0).numpy() * 255).astype("uint8")
                        elif isinstance(value, np.ndarray):
                            # Assume it matches what we have, but ensure HWC uint8
                            if value.shape[0] == 3: # CHW
                                img_np = (np.transpose(value, (1, 2, 0)) * 255).astype("uint8")
                            else:
                                img_np = value.astype("uint8")
                        else:
                            # Fallback
                            img_np = np.array(value).astype("uint8")
                            
                        # Rectify if needed
                        # Check specific keys or partial matches
                        # specifically rectifying 'top' and 'wrist'/'lateral'
                        rectified = img_np
                        if RECTIFY_TOP and ("images.top" in key):
                            rectified = camera_calibration.rectify_image(img_np, "top")
                        elif RECTIFY_WRIST and ("images.wrist" in key or "images.lateral" in key):
                            rectified = camera_calibration.rectify_image(img_np, "wrist")
                            
                        clean_frame[key] = rectified
                    else:
                        # Copy other features as is
                        clean_frame[key] = value

                target_dataset.add_frame(clean_frame)
            
            target_dataset.save_episode()
            
    if target_dataset:
        log_say("Finalizing and Pushing to Hub...")
        target_dataset.finalize()
        target_dataset.push_to_hub()
        log_say("Done!")

if __name__ == "__main__":
    main()
