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
    action_mode="so101_ee6d", # so101_ee6d / so101_joint

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
    gradient_accumulation_steps=1,
    optimizer_lr=1e-4,
    scheduler_decay_lr=1e-5,
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

# FreezeConfig presets for convenience
train_all               = FreezeConfig(freeze_vision_encoder=False,     freeze_language_encoder=False,  train_policy_transformer=True,  train_soft_prompts=True)
train_domain_specific   = FreezeConfig(freeze_vision_encoder=True,      freeze_language_encoder=True,   train_policy_transformer=True,  train_soft_prompts=True)

# Base model presets for convenience
BASE_MODEL = ("lerobot/xvla-base", 'xvla-base')
BASE_ORANGE_196 = ("edgarcancinoe/xvla-base_soarm101_pickplace_10d_7p5hz_resampled_so101_ee6d_a-m_s-m_v1", "orange196")

# Dataset presets for convenience
DATASET_ORANGE = "soarm101_pickplace_10d_7p5hz"
DATASET_MULTICOLOR = "soarm101_pickplace_multicolor_v1_7p5hz"


# Experiment specs.
EXPERIMENTS = [
    # Simple Orange
    # ------------------------------------------------------------------
    # 0: [Base ->      Orange196]  [NoAug] [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,    dataset_name=DATASET_ORANGE, 
        batch_size=32,  optimizer_lr=1e-4,  steps=15_000,  scheduler_decay_lr=1e-5,
    ),
    # 1: [Base ->      Orange196]  [Aug]   [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,     dataset_name=DATASET_ORANGE,     enable_augmentation=True,
        batch_size=32,  optimizer_lr=1e-4,  steps=15_000,  scheduler_decay_lr=1e-5,
    ),
    # ------------------------------------------------------------------
    
    # Multicolor
    # ------------------------------------------------------------------
    # 2: [Base ->      Multicolor] [NoAug] [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=15_000,  scheduler_decay_lr=1e-5,
    ),
    # 3: [Orange196 -> Multicolor] [NoAug] [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=15_000,  scheduler_decay_lr=1e-5,
    ),
    # 4: [Orange196 -> Multicolor] [NoAug] [train_domain_specific]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=15_000,  scheduler_decay_lr=1e-5,
        freeze=train_domain_specific
    ),

]

EXPERIMENTS = [EXPERIMENTS[4]]

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)

if __name__ == "__main__":
    main()
