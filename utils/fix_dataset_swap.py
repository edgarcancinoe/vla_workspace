
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

# Adjust path to find lerobot
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# We will read from current SO101-1
SOURCE_ROOT = Path("outputs/datasets/SO101-1")
REPO_ID = "edgarcancinoe/soarm101_pickup_orange"

# We will write to SO101-corrected
TARGET_ROOT = Path("outputs/datasets/SO101-corrected")

FPS = 30

def main():
    if TARGET_ROOT.exists():
        print(f"Removing existing target directory: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)
        
    print(f"Loading source dataset from {SOURCE_ROOT}...")
    ds_source = LeRobotDataset(repo_id=REPO_ID, root=SOURCE_ROOT)
    
    print("Source Features:", ds_source.features.keys())
    
    # We want to swap these two
    KEY_TOP = "observation.images.top"
    KEY_LAT = "observation.images.lateral"
    
    # Verify keys exist
    if KEY_TOP not in ds_source.features or KEY_LAT not in ds_source.features:
        print(f"Error: Dataset must contain {KEY_TOP} and {KEY_LAT}")
        return

    # Create new dataset with SAME features (names are same, just content swapped)
    print(f"Creating new dataset at {TARGET_ROOT}...")
    ds_target = LeRobotDataset.create(
        repo_id=REPO_ID, # Keep same repo ID so we can push override later
        fps=FPS,
        features=ds_source.features,
        robot_type=ds_source.meta.robot_type,
        use_videos=True,
        root=TARGET_ROOT,
    )
    
    print(f"Processing {ds_source.num_episodes} episodes...")
    
    for ep_idx in tqdm(range(ds_source.num_episodes)):
        # Calculate frame range for this episode
        from_idx = ds_source.meta.episodes[ep_idx]['dataset_from_index']
        to_idx = ds_source.meta.episodes[ep_idx]['dataset_to_index']
        
        # We can iterate frames. Getting item by index is fast enough usually?
        # Or iterate range.
        
        for i in range(from_idx, to_idx):
            item = ds_source[i]
            
            # SWAP THE CONTENT
            # item is a dict of tensors
            content_top = item[KEY_TOP]
            content_lat = item[KEY_LAT]
            
            item[KEY_TOP] = content_lat
            item[KEY_LAT] = content_top
            
            # Add to target
            # Note: add_frame expects dict of values, not tensors usually? 
            # LeRobotDataset.create -> add_frame logic handles conversion if we pass tensors?
            # Wait, `add_frame` expects mostly numpy/PIL or tensors?
            # Usually it expects what comes from the robot processor.
            # But let's try passing the item dict directly.
            
            ds_target.add_frame(item)
            
        # Save episode
        ds_target.save_episode()
        
    print("Finalizing dataset...")
    ds_target.finalize()
    
    print(f"Done! Corrected dataset is at {TARGET_ROOT}")
    print("To replace your original dataset locally:")
    print(f"  rm -rf {SOURCE_ROOT}")
    print(f"  mv {TARGET_ROOT} {SOURCE_ROOT}")
    print("Then you can push to hub.")

if __name__ == "__main__":
    main()
