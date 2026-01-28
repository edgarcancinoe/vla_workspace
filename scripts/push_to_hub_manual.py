import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
CHECKPOINT_PATH = Path("/home/jose/vla_workspace/outputs/train/smolvla_finetuned_orange_20260127_202513/checkpoints/010000/pretrained_model")
REPO_ID = "edgarcancinoe/smolvla_finetuned_orange"
REVISION = "checkpoint-10000"

def main():
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint path does not exist: {CHECKPOINT_PATH}")
        sys.exit(1)

    print(f"Uploading checkpoint from {CHECKPOINT_PATH}...")
    print(f"Target Repo: {REPO_ID}")
    print(f"Target Branch/Revision: {REVISION}")

    api = HfApi()
    
    # 1. Ensure Repository Exists
    print("Verifying repository exists...")
    try:
        # This will create the repo if it doesn't exist, provided you are logged in
        create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
        print("Repository verified.")
    except Exception as e:
        print(f"\n[ERROR] Could not verify/create repository '{REPO_ID}'.")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("1. You are NOT logged in. Run: `huggingface-cli login`")
        print("2. You do not have permissions for this namespace.")
        sys.exit(1)

    # 2. Create the branch (revision)
    try:
        api.create_branch(repo_id=REPO_ID, branch=REVISION, exist_ok=True)
        print(f"Branch '{REVISION}' verified.")
    except Exception as e:
        print(f"Warning: Could not create branch (might already exist): {e}")

    # 3. Upload folder
    try:
        url = api.upload_folder(
            folder_path=str(CHECKPOINT_PATH),
            repo_id=REPO_ID,
            revision=REVISION,
            commit_message=f"Upload checkpoint 10000"
        )
        print("\n[SUCCESS] Pushed to Hugging Face Hub!")
        print(f"View here: {url}")
    except Exception as e:
        print(f"\n[ERROR] Failed to upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
