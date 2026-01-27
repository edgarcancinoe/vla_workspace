
import pandas as pd
import json
from pathlib import Path

DATA_ROOT = Path("outputs/datasets/SO101-1")
META_EPISODES_DIR = DATA_ROOT / "meta/episodes/chunk-000"
INFO_PATH = DATA_ROOT / "meta/info.json"
EPISODE_TO_REMOVE = 9

def main():
    print(f"Starting Metadata Repair for {DATA_ROOT}...")
    
    # 1. Process Chunk 0 File 0 (Contains 0-9)
    file0 = META_EPISODES_DIR / "file-000.parquet"
    print(f"Processing {file0}...")
    df0 = pd.read_parquet(file0)
    
    # Get length of episode to remove for updating info.json
    removed_ep = df0[df0['episode_index'] == EPISODE_TO_REMOVE]
    if removed_ep.empty:
        print(f"WARNING: Episode {EPISODE_TO_REMOVE} not found in file-000!")
        removed_frames = 0
    else:
        removed_frames = removed_ep['length'].item()
        print(f"  Removing Episode {EPISODE_TO_REMOVE} (Length: {removed_frames} frames)")
        
        # Remove the row
        df0_clean = df0[df0['episode_index'] != EPISODE_TO_REMOVE]
        
        # Save back
        df0_clean.to_parquet(file0)
        print("  Saved updated file-000.parquet")

    # 2. Process Chunk 0 File 1 (Contains 10-19)
    # We need to shift indices down by 1
    file1 = META_EPISODES_DIR / "file-001.parquet"
    if file1.exists():
        print(f"Processing {file1}...")
        df1 = pd.read_parquet(file1)
        
        print("  Shifting episode indices down by 1...")
        df1['episode_index'] = df1['episode_index'] - 1
        
        # Save back
        df1.to_parquet(file1)
        print("  Saved updated file-001.parquet")
        
    # 3. Update info.json
    print(f"Updating {INFO_PATH}...")
    with open(INFO_PATH, 'r') as f:
        info = json.load(f)
        
    old_total = info['total_episodes']
    old_frames = info['total_frames']
    
    info['total_episodes'] = old_total - 1
    info['total_frames'] = old_frames - removed_frames
    
    # Update splits if present
    if "splits" in info:
        if "train" in info["splits"]:
            info["splits"]["train"] = f"0:{info['total_episodes']}"
            
    with open(INFO_PATH, 'w') as f:
        json.dump(info, f, indent=4)
        
    print("info.json updated:")
    print(f"  Total Episodes: {old_total} -> {info['total_episodes']}")
    print(f"  Total Frames:   {old_frames} -> {info['total_frames']}")
    
    print("\n✅ Repair Complete! Try verifying with scripts/inspect_dataset.py (but revert import fix first if needed)")

if __name__ == "__main__":
    main()
