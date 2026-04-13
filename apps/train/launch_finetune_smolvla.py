"""
Canonical SmolVLA finetune launcher.

Edit only `CONFIG` for day-to-day runs.

Example:
    python launch/launch_finetune_smolvla.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists():
    sys.path.insert(0, str(lerobot_src))

from thesis_vla.common.paths import PROJECT_ROOT
from thesis_vla.training.smolvla_finetune_launcher import SmolVLAConfig, run_training


WORKSPACE_DIR = PROJECT_ROOT


CONFIG = SmolVLAConfig(
    hf_user="edgarcancinoe",
    dataset_name="soarm101_pickplace_front",
    base_policy_path="lerobot/smolvla_base",
    policy_name="smolvla_finetuned_orange_50ep_open_gripper",
    batch_size=64,
    steps=25_000,
    log_freq=100,
    eval_freq=-1,
    save_freq=10_000,
    device="cuda",
    cuda_device="0",
    num_workers=4,
    enable_augmentation=False,
    augmentation_degrees="[-2, 2]",
    augmentation_translate="[-0.015, 0.015]",
    augmentation_backend="custom",
    augmentation_enable_photometric=True,
    augmentation_fill_mode="reflect",
    resume=False,
    policy_push_to_hub=True,
    wandb_enable=True,
    dry_run=False,
)


def main() -> None:
    run_training(CONFIG, WORKSPACE_DIR)


if __name__ == "__main__":
    main()
