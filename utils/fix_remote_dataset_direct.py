
import json
import shutil
from pathlib import Path
import sys
from huggingface_hub import snapshot_download, upload_folder

REPO_ID = "edgarcancinoe/soarm101_pick_cubes_place_box"
LOCAL_DIR = Path("outputs/datasets/temp_fix_pick_cubes")

KEY_TOP = "observation.images.top"
KEY_LAT = "observation.images.lateral"

def main():
    print(f"--- Fixing Remote Dataset: {REPO_ID} (Direct Mode) ---")
    
    # 1. Clean Start
    if LOCAL_DIR.exists():
        print(f"Propitiating clean slate: {LOCAL_DIR}")
        shutil.rmtree(LOCAL_DIR)
        
    # 2. Download
    print(f"Downloading to {LOCAL_DIR}...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR)
    
    # 3. Apply Fix (Swap Directories)
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

    # 4. Apply Fix (Swap Stats)
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

    # 5. Push Back using upload_folder (Bypassing LeRobotDataset validation)
    print("Pushing corrected dataset back to Hub (upload_folder)...")
    upload_folder(
        repo_id=REPO_ID,
        folder_path=LOCAL_DIR,
        repo_type="dataset",
        commit_message="Fix swapped Top and Lateral camera labels"
    )
    
    print("--- SUCCESS ---")
    print(f"Fixed {REPO_ID}. You can delete {LOCAL_DIR} if you wish.")

if __name__ == "__main__":
    main()
