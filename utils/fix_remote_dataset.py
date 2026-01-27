
import json
import shutil
from pathlib import Path
import sys

# Adjust path to find lerobot
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "edgarcancinoe/soarm101_pick_cubes_place_box"
# Use a clear local name to avoid conflicts
LOCAL_DIR = Path("outputs/datasets/temp_fix_pick_cubes")

KEY_TOP = "observation.images.top"
KEY_LAT = "observation.images.lateral"

def main():
    print(f"--- Fixing Remote Dataset: {REPO_ID} ---")
    
    # 1. Download Dataset
    if LOCAL_DIR.exists():
        print(f"Cleaning previous temp dir: {LOCAL_DIR}")
        shutil.rmtree(LOCAL_DIR)
        
    print(f"Downloading to {LOCAL_DIR}...")
    ds = LeRobotDataset(repo_id=REPO_ID, root=LOCAL_DIR)
    # Trigger download of everything if not already
    # accessing features or checking length usually triggers metadata download.
    # We need the VIDEOS. Video files are downloaded on demand or we can force it.
    # However, LeRobotDataset usually lazily downloads videos?
    # No, for `push_to_hub` we need the files locally or it might re-upload.
    # But `LeRobotDataset` by default doesn't download all videos unless requested?
    # Wait, `root` is where it looks. 
    # If we want to manipulate "videos/" folder, we need them downloaded.
    # Let's force download by iterating or there might be a `download()` method?
    # Checking source: `LeRobotDataset` inherits from... usually handled in `__init__` if `download=True` (default is implicit?).
    # But files are often in `videos/`.
    
    # Let's assume metadata is there.
    # We need to ensure videos are there.
    # We can use `huggingface_hub` to snapshot download if we want to be sure?
    # Or just trust `LeRobotDataset`.
    # Let's try iterating one frame to trigger lazy download? No, that downloads one file.
    
    # Let's just use `snapshot_download` from huggingface_hub to be robust, seeing as we want to do file operations.
    from huggingface_hub import snapshot_download
    print("Force downloading all files...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR)
    
    # 2. Apply Fix (Swap Directories)
    VIDEOS_DIR = LOCAL_DIR / "videos"
    STATS_PATH = LOCAL_DIR / "meta/stats.json"
    
    dir_top = VIDEOS_DIR / KEY_TOP
    dir_lat = VIDEOS_DIR / KEY_LAT
    dir_temp = VIDEOS_DIR / "temp_swap_dir"
    
    print("Swapping Directory Content...")
    if dir_top.exists() and dir_lat.exists():
        dir_top.rename(dir_temp) # Top -> Temp
        dir_lat.rename(dir_top)  # Lat -> Top
        dir_temp.rename(dir_lat) # Temp -> Lat
        print("Done: Video folders swapped.")
    else:
        print(f"ERROR: Video folders not found! {dir_top}, {dir_lat}")
        return

    # 3. Apply Fix (Swap Stats)
    if STATS_PATH.exists():
        print("Swapping Stats...")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        
        if KEY_TOP in stats and KEY_LAT in stats:
            temp = stats[KEY_TOP]
            stats[KEY_TOP] = stats[KEY_LAT]
            stats[KEY_LAT] = temp
            
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f, indent=4)
            print("Done: Stats swapped.")
        else:
            print("Warning: Keys not in stats.")

    # 4. Push Back
    print("Pushing corrected dataset back to Hub...")
    # Re-init dataset from local modified files
    ds_fixed = LeRobotDataset(repo_id=REPO_ID, root=LOCAL_DIR)
    ds_fixed.push_to_hub()
    
    print("--- SUCCESS ---")
    print(f"Fixed {REPO_ID}. You can delete {LOCAL_DIR} if you wish.")

if __name__ == "__main__":
    main()
