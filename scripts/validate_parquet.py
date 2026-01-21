
import sys
import pyarrow.parquet as pq
import os

def validate_parquet(file_path):
    print(f"Validating {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    try:
        # metadata reading checks for magic bytes and basic structure
        metadata = pq.read_metadata(file_path)
        print(f"Success! File is valid.")
        print(f"Rows: {metadata.num_rows}")
        print(f"Groups: {metadata.num_row_groups}")
        print(f"Serialized Size: {metadata.serialized_size} bytes")
        return True
    except Exception as e:
        print(f"Error: File is corrupted.")
        print(f"Details: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_parquet.py <path_to_parquet_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    validate_parquet(file_path)
