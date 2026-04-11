"""
Canonical XVLA finetune launcher.

Edit only `DEFAULTS` and `EXPERIMENTS` for day-to-day experiment work.

Examples:
    python launch/launch_finetune_xvla.py
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
from thesis_vla.training.xvla_finetune_launcher import (
    ExperimentSpec,
    FreezeConfig,
    LaunchConfig,
    RuntimeConfig,
    run_experiments,
)

WORKSPACE_DIR = PROJECT_ROOT

DEFAULTS = LaunchConfig(
    hf_user="edgarcancinoe",
    version="v1",
    dataset_name="soarm101_pickplace_10d_7p5hz_resampled",
    dataset_revision="v3.0",
    base_model="lerobot/xvla-base",
    normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
    freeze=FreezeConfig(
        freeze_vision_encoder=False,
        freeze_language_encoder=False,
        train_policy_transformer=True,
        train_soft_prompts=True,
    ),
    runtime=RuntimeConfig(
        launch_mode="single",
        cuda_devices=(1,),
        num_workers=0,
        dry_run=False,
    ),
    batch_size=32,
    steps=12_500,
    log_freq=500,
    eval_freq=-1,
    save_freq=8_000,
    push_every=8_000,
    policy_push_to_hub=True,
    wandb_enable=True,
)


EXPERIMENTS = [
    ExperimentSpec(
        action_mode="so101_ee6d",
    ),
    ExperimentSpec(
        action_mode="so101_joint",
    ),
]


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)

if __name__ == "__main__":
    main()
