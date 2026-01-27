
import sys
from pathlib import Path
import shutil
import torch
import numpy as np

# Adjust path to find lerobot
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
OLD_REPO_ID = "edgarcancinoe/soarm101_pickup_orange"
NEW_REPO_ID = "edgarcancinoe/soarm101_pickup_orange_fixed"  # Temporary new repo
LOCAL_DIR = Path("outputs/datasets/SO101-1")
NEW_LOCAL_DIR = Path("outputs/datasets/SO101-1-FIXED")
EPISODE_TO_REMOVE = 9
FPS = 30

def main():
    # 1. Load the original dataset
    print(f"Loading original dataset: {OLD_REPO_ID}")
    # Load from local or hub, relying on local cache first
    ds_old = LeRobotDataset(repo_id=OLD_REPO_ID, root=LOCAL_DIR)
    
    # 2. Setup the new dataset
    print(f"Creating new dataset at: {NEW_LOCAL_DIR}")
    if NEW_LOCAL_DIR.exists():
        shutil.rmtree(NEW_LOCAL_DIR)
    
    # We need to manually construct features because we can't easily clone them
    # But LeRobotDataset.create needs 'features' dict.
    # We can perform a trick: copy the features from the old dataset's info
    features = ds_old.features
    robot_type = ds_old.meta.robot_type

    ds_new = LeRobotDataset.create(
        repo_id=NEW_REPO_ID,
        fps=FPS,
        features=features,
        robot_type=robot_type,
        root=NEW_LOCAL_DIR,
        use_videos=True, 
    )

    print(f"Total episodes to process: {ds_old.num_episodes}")

    # 3. Iterate and Copy
    new_ep_idx = 0
    for i in range(ds_old.num_episodes):
        if i == EPISODE_TO_REMOVE:
            print(f"SKIPPING episode {i} (as requested)")
            continue
        
        print(f"Processing episode {i} -> new episode {new_ep_idx}")
        
        # Get data from old dataset
        # We need to extract all frames for this episode
        # ds_old.get_episode(i) returns a dict of tensors for the whole episode
        # BUT LeRobotDataset.add_episode is not a public API, usually we add frames.
        # However, looking at source, we can use `add_frame`.
        
        # More efficient way:
        # Load the episode data item by item? No, that's slow.
        # ds_old[index] gives one frame.
        
        # Let's get the range of indices for this episode
        ep_meta = ds_old.meta.episodes[i]
        frame_start = ep_meta['dataset_from_index']
        frame_end = ep_meta['dataset_to_index']
        
        # We can iterate through frames and add them
        # This is slow but safe.
        # Optimization: use slicing if possible, but add_frame expects single dicts.
        
        for idx in range(frame_start, frame_end):
            frame = ds_old[idx]
            
            # frame is a dict of items. 
            # We need to make sure they are in the format expected by add_frame.
            # Tensors need to be converted to numpy or kept as is?
            # add_frame usually expects numpy or simple types, but handles tensors?
            # Let's check source code logic...
            # Validation usually happens.
            
            # Simple fix: convert all tensors to numpy
            new_frame = {}
            for k, v in frame.items():
                if k == "index" or k == "episode_index" or k == "frame_index":
                    continue # specific indices are re-generated
                if isinstance(v, torch.Tensor):
                    new_frame[k] = v.cpu().numpy()
                else:
                    new_frame[k] = v
                    
            ds_new.add_frame(new_frame)
            
        # Manually save the episode after adding all its frames
        ds_new.save_episode(task=ds_old.tasks[i] if ds_old.tasks else "Task description") 
        new_ep_idx += 1

    # 4. Finalize
    print("Consolidating new dataset...")
    ds_new.consolidate()
    
    print("Done! You can now verify the new dataset in outputs/datasets/SO101-1-FIXED")
    print(f"To replace the old one, you can:")
    print(f"  rm -rf {LOCAL_DIR}")
    print(f"  mv {NEW_LOCAL_DIR} {LOCAL_DIR}")
    print(f"Then in your specific case, update the repo_id in the script back to '{OLD_REPO_ID}'")

if __name__ == "__main__":
    main()
