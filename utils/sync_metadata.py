
import pandas as pd
import json
from pathlib import Path

DATA_ROOT = Path("outputs/datasets/SO101-1")
META_EPISODES_DIR = DATA_ROOT / "meta/episodes/chunk-000"
INFO_PATH = DATA_ROOT / "meta/info.json"

def main():
    print(f"Starting Metadata Synchronization for {DATA_ROOT}...")
    
    total_frames = 0
    max_episode_index = -1
    total_episodes_counted = 0
    
    # Iterate through all metadata files to count actual valid frames/episodes
    # We assume file-000.parquet, file-001.parquet exist and are valid from previous fix
    metadata_files = sorted(list(META_EPISODES_DIR.glob("file-*.parquet")))
    
    print(f"Found metadata files: {[f.name for f in metadata_files]}")
    
    for meta_file in metadata_files:
        try:
            df = pd.read_parquet(meta_file)
            if not df.empty:
                frames_in_file = df['length'].sum()
                episodes_in_file = len(df)
                max_idx = df['episode_index'].max()
                
                print(f"  {meta_file.name}: {episodes_in_file} eps, {frames_in_file} frames. Max Index: {max_idx}")
                
                total_frames += frames_in_file
                total_episodes_counted += episodes_in_file
                max_episode_index = max(max_episode_index, max_idx)
        except Exception as e:
            print(f"  ERROR reading {meta_file}: {e}")

    # Calculate expected total episodes (index + 1)
    # If indices are continuous 0..N, then total is N+1
    # But let's trust the count of rows
    
    print(f"\nCalculated from Metadata Files:")
    print(f"  Total Episodes: {total_episodes_counted}")
    print(f"  Total Frames:   {total_frames}")
    
    # Update info.json
    print(f"Updating {INFO_PATH}...")
    with open(INFO_PATH, 'r') as f:
        info = json.load(f)
        
    old_total_eps = info.get('total_episodes', 'UNKNOWN')
    old_total_frames = info.get('total_frames', 'UNKNOWN')
    
    info['total_episodes'] = int(total_episodes_counted)
    info['total_frames'] = int(total_frames)
    
    # Update splits
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{int(total_episodes_counted)}"

    with open(INFO_PATH, 'w') as f:
        json.dump(info, f, indent=4)
        
    print("info.json sync complete:")
    print(f"  Total Episodes: {old_total_eps} -> {info['total_episodes']}")
    print(f"  Total Frames:   {old_total_frames} -> {info['total_frames']}")
    
    print("\n✅ Sync Complete! The dataset is now truncated to the valid metadata.")

if __name__ == "__main__":
    main()
