#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a visual-thought checkpoint directory to the Hugging Face Hub.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to visual-thought checkpoint directory (contains decoder.safetensors, trainer_state.pt, metadata.json, visual_thought_config.json, and policy/).")
    parser.add_argument("--repo_id", required=True, help="Hugging Face model repo id, e.g. edgarcancinoe/cedirnet_joint_stage.")
    parser.add_argument("--commit_message", default="Upload visual-thought checkpoint", help="Commit message for the upload.")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    required = ["decoder.safetensors", "trainer_state.pt", "metadata.json", "visual_thought_config.json", "policy"]
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if not checkpoint_dir.is_dir() or missing:
        logging.error(f"Checkpoint directory is invalid or missing required files: {checkpoint_dir} missing={missing}")
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(folder_path=str(checkpoint_dir), repo_id=args.repo_id, repo_type="model", commit_message=args.commit_message)
    logging.info(f"Success: {url}")


if __name__ == "__main__":
    main()
