
import shutil
from pathlib import Path
import pandas as pd
import json

DATASET_ROOT = Path("outputs/datasets/soarm101_pickplace_local")
EPISODE_TO_REMOVE = 28

def main():
    meta_dir = DATASET_ROOT / "meta"
    episodes_path = meta_dir / "episodes"
    
    # Load episodes
    try:
        df = pd.read_parquet(episodes_path)
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return

    print(f"Loaded {len(df)} episodes.")
    
    # Check if 28 exists
    if EPISODE_TO_REMOVE not in df["episode_index"].values:
        print(f"Episode {EPISODE_TO_REMOVE} not found!")
        return

    # Filter
    df_new = df[df["episode_index"] != EPISODE_TO_REMOVE].copy()
    
    # Verify no gaps or issues (since we remove the last one, it should be clean)
    expected_indices = list(range(len(df_new)))
    actual_indices = sorted(df_new["episode_index"].tolist())
    
    if expected_indices != actual_indices:
        print("Warning: Indices represent a gap or disorder!")
        print(f"Expected: {expected_indices}")
        print(f"Actual: {actual_indices}")
        # Re-index just in case
        print("Re-indexing...")
        df_new = df_new.sort_values("episode_index").reset_index(drop=True)
        df_new["episode_index"] = df_new.index.astype("int64")
    
    print(f"New episode count: {len(df_new)}")
    
    # Backup
    backup_path = meta_dir / "episodes_backup"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(episodes_path, backup_path)
    print(f"Backed up to {backup_path}")
    
    # Write back
    # We remove the directory and recreate it to ensure clean state
    shutil.rmtree(episodes_path)
    episodes_path.mkdir()
    
    # Write as single parquet file inside the directory
    output_file = episodes_path / "episodes.parquet"
    df_new.to_parquet(output_file)
    print(f"Written new metadata to {output_file}")
    
    # Update info.json
    info_path = meta_dir / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
        
    old_count = info.get("total_episodes", "Unknown")
    info["total_episodes"] = len(df_new)
    info["splits"]["train"] = f"0:{len(df_new)}" # update split info e.g. "0:29" -> "0:28"
    
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        
    print(f"Updated info.json. Total episodes: {old_count} -> {info['total_episodes']}")
    print("Done.")

if __name__ == "__main__":
    main()
