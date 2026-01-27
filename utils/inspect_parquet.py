
import pandas as pd
from pathlib import Path

FILE_PATH = "outputs/datasets/SO101-1/meta/episodes/chunk-000/file-000.parquet"

def main():
    print(f"Inspecting {FILE_PATH}...")
    try:
        df = pd.read_parquet(FILE_PATH)
        print("Columns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nEpisode Indices:")
        print(df['episode_index'].tolist())
        
        # Check if file-001 exists and inspect it too
        file1 = "outputs/datasets/SO101-1/meta/episodes/chunk-000/file-001.parquet"
        if Path(file1).exists():
            print(f"\nInspecting {file1}...")
            df1 = pd.read_parquet(file1)
            print("Episode Indices in file-001:")
            print(df1['episode_index'].tolist())
            
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    main()
