#!/usr/bin/env python3
"""
Utility script to manually push a local LeRobot training checkpoint to the Hugging Face Hub.
This is useful if the automatic push at the end of training fails due to network issues.

Usage:
    python push_checkpoint.py --checkpoint_dir /path/to/checkpoints/060000/pretrained_model \
                              --repo_id edgarcancinoe/my_awesome_model \
                              --commit_message "Upload final checkpoint"

Optionally, you can generate a default model card by providing base model and action mode:
    python push_checkpoint.py ... --base_model lerobot/xvla-base --action_mode so101_ee6d
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_model_card(repo_id: str, base_model: str, action_mode: str) -> str:
    """Generates a default markdown string for the model card (README.md)."""
    return f"""---
library_name: lerobot
tags:
- lerobot
- robotics
- xvla
---

# Model Card for {repo_id.split('/')[-1]}

This model was trained using [LeRobot](https://github.com/huggingface/lerobot).

## Training Configuration
- **Base Model**: `{base_model}`
- **Action Mode**: `{action_mode}`

*This repository contains the weights and configuration files for the trained policy.*
"""


def main():
    parser = argparse.ArgumentParser(description="Push a local LeRobot checkpoint to the Hugging Face Hub.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the directory containing model.safetensors and config files (usually within 'pretrained_model').",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/model_name').",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model checkpoint",
        help="Commit message for the upload.",
    )
    # Optional arguments for the model card
    parser.add_argument(
        "--base_model",
        type=str,
        default="Unknown",
        help="Base model used for training (added to model card).",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="Unknown",
        help="Action mode used for training (added to model card).",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logging.error(f"Checkpoint directory does not exist or is not a directory: {checkpoint_path}")
        sys.exit(1)

    # Basic verification that it looks like a model directory
    if not (checkpoint_path / "model.safetensors").exists():
        logging.warning("model.safetensors not found in the specified directory. Are you sure this is the correct path? (Typically ends in 'pretrained_model')")
        # Ask for confirmation if running interactively, otherwise proceed with warning
        if sys.stdin.isatty():
            ans = input("Proceed anyway? (y/N): ")
            if ans.lower() != 'y':
                sys.exit(0)

    api = HfApi()

    logging.info(f"Ensuring repository '{args.repo_id}' exists on the Hub...")
    try:
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create/verify repository. Are you authenticated? ({e})")
        sys.exit(1)

    # Automatically generate and upload a model card if one doesn't exist locally
    local_readme = checkpoint_path / "README.md"
    if not local_readme.exists():
        logging.info("README.md not found in checkpoint. Generating a default model card...")
        readme_content = generate_model_card(args.repo_id, args.base_model, args.action_mode)
        # We avoid writing to the actual checkpoint dir just to be safe, save it temporarily
        temp_readme = Path("/tmp/lerobot_temp_readme.md")
        with open(temp_readme, "w") as f:
            f.write(readme_content)
        
        logging.info("Uploading model card...")
        try:
             api.upload_file(
                path_or_fileobj=str(temp_readme),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Add initial model card"
            )
        except Exception as e:
            logging.warning(f"Failed to upload model card: {e}")
        finally:
             if temp_readme.exists():
                 os.remove(temp_readme)

    logging.info(f"Uploading files from '{checkpoint_path}' to '{args.repo_id}'...")
    try:
        url = api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
        logging.info(f"Success! Model pushed to: {url}")
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
