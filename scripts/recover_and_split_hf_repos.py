#!/usr/bin/env python3
"""
Recovery script: separates overwritten HF repos into action-mode-specific repos.

The ee6d run and joint run both pushed to the same HF repos. The joint run
overwrote the ee6d versions. This script recovers the ee6d versions from git
history and creates properly-named repos for both action modes.

Repos affected:
  - edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_10d          (final, step 60000)
  - edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_10d-step-20000
  - edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_10d-step-40000

For each, we need to:
  1. Download the OLD (ee6d) version from git history
  2. Upload to new repo with _so101_ee6d suffix
  3. Copy/download the CURRENT (joint) version
  4. Upload to new repo with _so101_joint suffix

Git history mapping (from commit analysis):
  step-20000: ee6d commits at ~11:04 (de4af21e), joint overwrite at ~17:49 (f966f9f6)
  step-40000: ee6d commits at ~12:16 (324283d2), joint overwrite at ~19:01 (b4bedeb8)
  final:      ee6d commit at ~15:25 (5eae5c1c), joint overwrite at ~20:11 (5e9d9a2b)
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, snapshot_download, upload_folder

api = HfApi()

BASE_NAME = "xvla-base_finetuned_soarm101_pickplace_10d"
HF_USER = "edgarcancinoe"

# Mapping: (old_repo_id, ee6d_commit, new_ee6d_repo, new_joint_repo)
REPOS = [
    {
        "old_repo": f"{HF_USER}/{BASE_NAME}-step-20000",
        "ee6d_commit": "de4af21e",  # ee6d upload at 11:04
        "new_ee6d_repo": f"{HF_USER}/{BASE_NAME}_so101_ee6d-step-20000",
        "new_joint_repo": f"{HF_USER}/{BASE_NAME}_so101_joint-step-20000",
    },
    {
        "old_repo": f"{HF_USER}/{BASE_NAME}-step-40000",
        "ee6d_commit": "324283d2",  # ee6d upload at 12:16
        "new_ee6d_repo": f"{HF_USER}/{BASE_NAME}_so101_ee6d-step-40000",
        "new_joint_repo": f"{HF_USER}/{BASE_NAME}_so101_joint-step-40000",
    },
    {
        "old_repo": f"{HF_USER}/{BASE_NAME}",
        "ee6d_commit": "5eae5c1c",  # ee6d final upload at 15:25
        "new_ee6d_repo": f"{HF_USER}/{BASE_NAME}_so101_ee6d",
        "new_joint_repo": f"{HF_USER}/{BASE_NAME}_so101_joint",
    },
]


def download_at_revision(repo_id: str, revision: str, local_dir: str):
    """Download a snapshot of a repo at a specific git revision."""
    print(f"  Downloading {repo_id} @ {revision} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def upload_to_new_repo(local_dir: str, new_repo_id: str, commit_message: str):
    """Create a new repo and upload the contents."""
    print(f"  Creating repo: {new_repo_id}")
    api.create_repo(repo_id=new_repo_id, exist_ok=True, repo_type="model")

    print(f"  Uploading to {new_repo_id}...")
    upload_folder(
        folder_path=local_dir,
        repo_id=new_repo_id,
        commit_message=commit_message,
        ignore_patterns=[".git", ".git/*", ".gitattributes"],
    )
    print(f"  ✓ Uploaded to {new_repo_id}")


def process_repo(entry):
    old_repo = entry["old_repo"]
    ee6d_commit = entry["ee6d_commit"]
    new_ee6d_repo = entry["new_ee6d_repo"]
    new_joint_repo = entry["new_joint_repo"]

    print(f"\n{'='*70}")
    print(f"Processing: {old_repo}")
    print(f"  ee6d commit:  {ee6d_commit}")
    print(f"  -> ee6d repo: {new_ee6d_repo}")
    print(f"  -> joint repo: {new_joint_repo}")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download ee6d version (old commit)
        ee6d_dir = os.path.join(tmpdir, "ee6d")
        download_at_revision(old_repo, ee6d_commit, ee6d_dir)
        upload_to_new_repo(ee6d_dir, new_ee6d_repo,
                           f"Recovered ee6d model from {old_repo} @ {ee6d_commit}")

        # 2. Download joint version (current/latest)
        joint_dir = os.path.join(tmpdir, "joint")
        download_at_revision(old_repo, "main", joint_dir)
        upload_to_new_repo(joint_dir, new_joint_repo,
                           f"Joint model from {old_repo} (latest)")


def main():
    print("=" * 70)
    print("HF Repo Recovery: Splitting overwritten repos by action mode")
    print("=" * 70)

    # Verify we can connect
    whoami = api.whoami()
    print(f"Authenticated as: {whoami['name']}")

    for entry in REPOS:
        process_repo(entry)

    # Also upload local-only checkpoints (step 30000 and 60000) that aren't on HF as intermediates
    # The local ee6d checkpoint at step 30000 is only saved locally
    local_ee6d_base = "/home/jose/vla_workspace/outputs/train/xvla-base_finetuned_soarm101_pickplace_10d_20260225_105259/checkpoints"
    local_joint_base = "/home/jose/vla_workspace/outputs/train/xvla-base_finetuned_soarm101_pickplace_10d_20260225_162713/checkpoints"

    for step in ["030000"]:
        ee6d_local = os.path.join(local_ee6d_base, step, "pretrained_model")
        joint_local = os.path.join(local_joint_base, step, "pretrained_model")

        if os.path.exists(ee6d_local):
            print(f"\n{'='*70}")
            print(f"Uploading local ee6d checkpoint step {step}")
            print(f"{'='*70}")
            upload_to_new_repo(
                ee6d_local,
                f"{HF_USER}/{BASE_NAME}_so101_ee6d-step-{int(step)}",
                f"Local ee6d checkpoint at step {step}"
            )

        if os.path.exists(joint_local):
            print(f"\n{'='*70}")
            print(f"Uploading local joint checkpoint step {step}")
            print(f"{'='*70}")
            upload_to_new_repo(
                joint_local,
                f"{HF_USER}/{BASE_NAME}_so101_joint-step-{int(step)}",
                f"Local joint checkpoint at step {step}"
            )

    print("\n" + "=" * 70)
    print("✓ All repos recovered and split successfully!")
    print("\nNew repos created:")
    for entry in REPOS:
        print(f"  - {entry['new_ee6d_repo']}")
        print(f"  - {entry['new_joint_repo']}")
    print(f"  - {HF_USER}/{BASE_NAME}_so101_ee6d-step-30000")
    print(f"  - {HF_USER}/{BASE_NAME}_so101_joint-step-30000")
    print("=" * 70)


if __name__ == "__main__":
    main()
