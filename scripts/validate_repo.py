
from datasets import load_dataset
import sys

def validate_repo(repo_id):
    print(f"Validating repository: {repo_id}")
    try:
        # Load in streaming mode to check file access without full download
        ds = load_dataset(repo_id, split="train", streaming=True)
        
        print("Dataset loaded successfully in streaming mode. Iterating through rows to verify data Integrity...")
        count = 0
        for i, item in enumerate(ds):
            count += 1
            if i % 100 == 0:
                print(f"Verified {i} rows...", end="\r")
        
        print(f"\nSuccessfully verified {count} rows in the dataset.")
        return True
    except Exception as e:
        print(f"\nError validating repository: {e}")
        return False

if __name__ == "__main__":
    repo_id = "edgarcancinoe/soarm101_pick_cubes_place_box"
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    
    validate_repo(repo_id)
