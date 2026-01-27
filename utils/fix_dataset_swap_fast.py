
import json
import shutil
from pathlib import Path

# Paths
DATASET_ROOT = Path("outputs/datasets/SO101-1")
VIDEOS_DIR = DATASET_ROOT / "videos"
STATS_PATH = DATASET_ROOT / "meta/stats.json"
INFO_PATH = DATASET_ROOT / "meta/info.json"

KEY_TOP = "observation.images.top"
KEY_LAT = "observation.images.lateral"

def main():
    print(f"Fixing dataset at {DATASET_ROOT} by swapping directories and metadata...")
    
    # 1. Swap Video Directories
    # We have `videos/observation.images.top` and `videos/observation.images.lateral`
    # We want to swap them.
    dir_top = VIDEOS_DIR / KEY_TOP
    dir_lat = VIDEOS_DIR / KEY_LAT
    dir_temp = VIDEOS_DIR / "temp_swap_dir"
    
    if dir_top.exists() and dir_lat.exists():
        print("Swapping video directories...")
        # Top -> Temp
        dir_top.rename(dir_temp)
        # Lateral -> Top
        dir_lat.rename(dir_top)
        # Temp (Old Top) -> Lateral
        dir_temp.rename(dir_lat)
        print("Video directories swapped.")
    else:
        print(f"Warning: One or both video directories missing ({dir_top}, {dir_lat}). Skipping video swap.")

    # 2. Swap keys in meta/stats.json
    if STATS_PATH.exists():
        print(f"Updating {STATS_PATH}...")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
            
        if KEY_TOP in stats and KEY_LAT in stats:
            # Swap the contents associated with keys
            # Actually, stats might just contain min/max/mean/std.
            # If we swap the underlying data (videos), the stats associated with "Top" key now correspond to the new data (Old Lateral).
            # So if Stats[Top] described Old Top data.
            # And now "Top" key points to Old Lateral data.
            # We must swap the stats values too so they match the data.
            
            temp = stats[KEY_TOP]
            stats[KEY_TOP] = stats[KEY_LAT]
            stats[KEY_LAT] = temp
            
            with open(STATS_PATH, 'w') as f:
                json.dump(stats, f, indent=4)
            print("stats.json updated.")
        else:
            print(f"Keys not found in stats.json. Available: {list(stats.keys())}")

    # 3. Swap keys in meta/info.json ?
    # info.json describes features. "observation.images.top" info usually contains shape/fps.
    # Usually they are identical specs (640x480).
    # But strictly, yes, we should swap their definitions if they differed.
    # In this case, both are identical config, so no change needed in info.json features definition values, 
    # just ensuring the keys exist is enough.
    
    # 4. Parquet Files?
    # Parquet files contain columns "observation.images.top" etc. 
    # These columns usually store *file path references* (e.g. `videos/observation.images.top/chunk-000/file-000.mp4`).
    # If I blindly renamed the directories on disk, the Parquet file still points to `videos/observation.images.top/...`.
    # After rename: `videos/observation.images.top` now contains what used to be Lateral videos.
    # So `parquet["top"]` -> points to `videos/observation.images.top` -> which is now Lateral Video.
    # This means `parquet["top"]` loads Lateral Video.
    # THIS IS THE OPPOSITE OF WHAT WE WANT.
    # We want: `parquet["top"]` -> load Top Video.
    # Current Top Video is in `videos/observation.images.lateral` (after swap? No wait).
    
    # Original State:
    # "Top" Key -> `videos/top` -> Lateral Images.
    # "Lat" Key -> `videos/lat` -> Top Images.
    
    # Desired State:
    # "Top" Key -> Top Images.
    # "Lat" Key -> Lateral Images.
    
    # Approach A (Rename Directories Only):
    # Rename `videos/top` (Lat content) to `videos/lat`.
    # Rename `videos/lat` (Top content) to `videos/top`.
    # Now: `videos/top` contains Top content. `videos/lat` contains Lat content.
    # Parquet `observation.images.top` points to `videos/observation.images.top/...`.
    # Does logical path resolve? Yes.
    # So `parquet["top"]` -> `videos/top` -> New content (Top Images).
    # Correct!
    
    # So directory swap IS SUFFICIENT for the video file pointers in parquet.
    
    # But what about STATS?
    # Stats[Top] computed on Old Data (Lateral).
    # New Data for Top is Top Images.
    # So Stats[Top] must hold stats for Top Images.
    # Where are stats for Top Images? They are currently in Stats[Lat].
    # So yes, we MUST SWAP Stats keys.
    
    print("------------------------------------------------")
    print("Fix Complete! Please allow LeRobot to re-verify if needed.")
    print("You can now push the dataset.")

if __name__ == "__main__":
    main()
