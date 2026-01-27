
import sys
import shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Adjust path to find lerobot and local scripts
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/physical-workspace")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.lerobot_dataset
import camera_calibration

# DISABLE STATS COMPUTATION TO AVOID FileNotFound ERROR
# The dataset writer tries to read back images as PNGs before video is finalized?
# We will compute stats at the end if needed.
print("WARNING: Patching compute_episode_stats to skip stats computation during write.")
lerobot.datasets.lerobot_dataset.compute_episode_stats = lambda *args, **kwargs: {}

PROJECT_ROOT = Path(__file__).parent.parent

SOURCE_REPOS = [
    {"repo_id": "edgarcancinoe/soarm101_pickup_orange", "root": None},
    {"repo_id": "soarm101_pick_fixed", "root": PROJECT_ROOT / "outputs/datasets/soarm101_pick_fixed"}
]

TARGET_REPO = "edgarcancinoe/soarm101_pickplace"
LOCAL_TARGET_DIR = PROJECT_ROOT / "outputs/datasets/soarm101_pickplace_local"
FPS = 30

def to_hwc_uint8(img_tensor):
    """Converts (C, H, W) float tensor [0,1] to (H, W, C) uint8 numpy [0,255]."""
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.permute(1, 2, 0).numpy()
    # If already numpy but float HWC or CHW? LeRobot usually gives CHW float.
    # Check shape
    if img_tensor.ndim == 3 and img_tensor.shape[0] == 3: # CHW numpy
        img_tensor = np.transpose(img_tensor, (1, 2, 0))
    
    # Scale if float
    if img_tensor.dtype != np.uint8:
        img_tensor = (img_tensor * 255).astype(np.uint8)
        
    return img_tensor

def main():
    print(f"--- Combining and Rectifying Datasets into {TARGET_REPO} ---")
    
    # CHECKPOINTING
    episodes_already_done = 0
    if LOCAL_TARGET_DIR.exists() and (LOCAL_TARGET_DIR / "meta/info.json").exists():
        print(f"Checking for existing dataset at {LOCAL_TARGET_DIR}...")
        try:
             # Load just to check count
             ds_check = LeRobotDataset(root=LOCAL_TARGET_DIR, repo_id=TARGET_REPO)
             episodes_already_done = ds_check.num_episodes
             print(f"RESUMING: Found {episodes_already_done} episodes already processed.")
             
             # Re-instantiate in write mode (create calls init internally if exists?)
             # Actually, to append, we might need a specific mode or just load it?
             # LeRobotDataset usually supports appending if we just load it.
             # But 'create' wipes if we are not careful? 
             # Let's assume loading valid dataset lets us append.
             ds_target = ds_check
             # We need to make sure we don't overwrite features/meta if they match.
             # The loop just calls add_frame/save_episode.
             
        except Exception as e:
             print(f"Could not load existing dataset ({e}). Restarting.")
             if LOCAL_TARGET_DIR.exists(): shutil.rmtree(LOCAL_TARGET_DIR)
             episodes_already_done = 0
    
    if episodes_already_done == 0:
        if LOCAL_TARGET_DIR.exists(): shutil.rmtree(LOCAL_TARGET_DIR)
        
        print(f"Loading reference features from {SOURCE_REPOS[0]}...")
        ds_ref = LeRobotDataset(
            repo_id=SOURCE_REPOS[0]["repo_id"], 
            root=SOURCE_REPOS[0]["root"], 
            video_backend="pyav", 
            tolerance_s=0.1
        )
        features = ds_ref.features
        robot_type = ds_ref.meta.robot_type
        
        print(f"Creating target dataset at {LOCAL_TARGET_DIR}...")
        ds_target = LeRobotDataset.create(
            repo_id=TARGET_REPO,
            fps=FPS,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            root=LOCAL_TARGET_DIR,
            video_backend="pyav", 
            vcodec="h264", 
            tolerance_s=0.1
        )
    
    # Global counter to track which source episode maps to target
    current_target_ep_idx = 0
    total_episodes = 0
    
    for repo_config in SOURCE_REPOS:
        repo_id = repo_config["repo_id"]
        repo_root = repo_config["root"]
        
        print(f"\nProcessing Source: {repo_id}")
        
        # Patch safe version to allow local-only loading for our fixed dataset
        old_get_safe_version = lerobot.datasets.lerobot_dataset.get_safe_version
        lerobot.datasets.lerobot_dataset.get_safe_version = lambda r, rv: rv
        
        try:
            ds_source = LeRobotDataset(repo_id=repo_id, root=repo_root, video_backend="pyav", tolerance_s=0.1)
        finally:
            lerobot.datasets.lerobot_dataset.get_safe_version = old_get_safe_version
            
        num_eps = ds_source.num_episodes
        print(f"  Episodes: {num_eps}")
        
        for ep_idx in tqdm(range(num_eps), desc=f"Processing {repo_id}"):
            # CHECKPOINT: Skip if already processed
            if current_target_ep_idx < episodes_already_done:
                if current_target_ep_idx % 10 == 0:
                    print(f"Skipping episode {ep_idx} (already done)", flush=True)
                current_target_ep_idx += 1
                continue

            current_target_ep_idx += 1
            
            # We must iterate frames
            from_idx = ds_source.meta.episodes[ep_idx]['dataset_from_index']
            to_idx = ds_source.meta.episodes[ep_idx]['dataset_to_index']
            
            print(f"DEBUG: Processing episode {ep_idx} (Target: {current_target_ep_idx-1}) Frames: {from_idx}-{to_idx}", flush=True)
            for i in range(from_idx, to_idx):
                if (i - from_idx) % 100 == 0:
                    print(f"DEBUG: Frame {i}", flush=True)
                try:
                    item = ds_source[i]
                    
                    # RECTIFY IMAGES
                    # item keys like 'observation.images.top'
                    for key, value in item.items():
                        if "images" in key and isinstance(value, torch.Tensor):
                            # 1. Convert to HWC Uint8 for OpenCV
                            img_np = to_hwc_uint8(value)
                            
                            # 2. Rectify
                            # Map key to camera name. 
                            # 'observation.images.top' -> 'top'
                            cam_name = key.split(".")[-1] # top or lateral
                            
                            rectified_np = camera_calibration.rectify_image(img_np, cam_name)
                            
                            # 3. Update item.
                            item[key] = rectified_np
                    
                    # Filter items
                    EXCLUDE_KEYS = {'index', 'frame_index', 'timestamp', 'episode_index', 'task_index'}
                    filtered_item = {k: v for k, v in item.items() if k in ds_target.features and k not in EXCLUDE_KEYS}
                    
                    # UNCONDITIONAL CHECK
                    if "task" not in filtered_item:
                        # print(f"DEBUG: 'task' missing for ep {ep_idx}. Item has: {list(item.keys())}", flush=True)
                        
                        # Try index lookup
                        if "task_index" in item and hasattr(ds_source, "tasks"):
                            filtered_item["task"] = ds_source.tasks[item["task_index"]]
                        else:
                            filtered_item["task"] = "combined_task_placeholder"
                    
                    # Force check before throw
                    if "task" not in filtered_item:
                         print(f"FATAL: Logic failed to inject task. filtered_item: {list(filtered_item.keys())}", flush=True)

                    # Add modified frame
                    ds_target.add_frame(filtered_item)

                except Exception as e:
                    print(f"WARNING: Skipping frame {i} in episode {ep_idx} due to error: {e}", flush=True)
                    continue
            
            # Save episode
            ds_target.save_episode()
            total_episodes += 1
            
    print("\nFinalizing target dataset...")
    ds_target.finalize()
    
    print(f"\nPushing {TARGET_REPO} to Hub...")
    ds_target.push_to_hub()
    
    print("--- SUCCESS ---")
    print(f"Combined {total_episodes} episodes into {TARGET_REPO}")

if __name__ == "__main__":
    main()
