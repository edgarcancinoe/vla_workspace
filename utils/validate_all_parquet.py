
import os
import sys
import pyarrow.parquet as pq

def validate_all_parquet(root_dir):
    print(f"Scanning {root_dir} for parquet files...")
    
    parquet_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    print(f"Found {len(parquet_files)} parquet files. Starting validation...")
    print("-" * 50)
    
    valid_count = 0
    corrupted_files = []
    
    for i, file_path in enumerate(parquet_files):
        # Interactive progress
        sys.stdout.write(f"\rChecking file {i+1}/{len(parquet_files)}")
        sys.stdout.flush()
        
        try:
            pq.read_metadata(file_path)
            valid_count += 1
        except Exception as e:
            corrupted_files.append((file_path, str(e)))
            
    print("\n" + "-" * 50)
    print(f"Validation Complete.")
    print(f"Total Files: {len(parquet_files)}")
    print(f"Valid Files: {valid_count}")
    print(f"Corrupted Files: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\nList of Corrupted Files:")
        for path, error in corrupted_files:
            print(f"[X] {path}\n    Error: {error}")
    else:
        print("\nAll parquet files are valid! ✅")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_all_parquet.py <directory_to_scan>")
        sys.exit(1)
        
    validate_all_parquet(sys.argv[1])
