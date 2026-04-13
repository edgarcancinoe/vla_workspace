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

from thesis_vla.common.paths import PROJECT_ROOT #type: ignore
from thesis_vla.training.xvla_finetune_launcher import ExperimentSpec, FreezeConfig, LaunchConfig, RuntimeConfig, run_experiments #type: ignore

WORKSPACE_DIR = PROJECT_ROOT

DEFAULTS = LaunchConfig(
    # ----- General model and database settings ------
    hf_user="edgarcancinoe",
    version="v1",
    dataset_name="soarm101_pickplace_10d_7p5hz_resampled",
    dataset_revision="v3.0",
    base_model="lerobot/xvla-base",
    normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',

    # ----------- Runtime/Device settings -----------
    runtime=RuntimeConfig(
        launch_mode="single",
        cuda_devices=(1,),
        num_workers=0,
        dry_run=False,
    ),

    # ---------- Training layers settings -----------
    freeze=FreezeConfig(
        freeze_vision_encoder=False,
        freeze_language_encoder=False,
        train_policy_transformer=True,
        train_soft_prompts=True,
    ),

    # ------------ Optimization settings ------------
    batch_size=32,
    optimizer_lr=1e-4,
    steps=12_500,

    # ------- Logging and checkpoint settings -------
    log_freq=500,
    eval_freq=-1,
    save_freq=8_000,
    push_every=8_000,
    policy_push_to_hub=True,
    wandb_enable=True,

    # ------------ Augmentation settings ------------
    enable_augmentation=False,
    augmentation_degrees="[-2.5, 2.5]",
    augmentation_translate="[0.025, 0.025]",
    augmentation_backend="custom",
    augmentation_enable_photometric=True,
    augmentation_fill_mode="reflect",
)

# Experiment specs. Edit for overriding settings.
EXPERIMENTS = [
    ExperimentSpec(
        action_mode="so101_ee6d",
        enable_augmentation=False,
    ),
    ExperimentSpec(
        action_mode="so101_joint",
        enable_augmentation=False,
    ),
]


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)

if __name__ == "__main__":
    main()
