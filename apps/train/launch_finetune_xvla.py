"""
Canonical XVLA finetune launcher.

Edit only `DEFAULTS` and `EXPERIMENTS` for day-to-day experiment work.

Examples:
    python launch/launch_finetune_xvla.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
workspace_src = ROOT_DIR / "src"
sys.path.insert(0, str(workspace_src))
lerobot_src_candidates = [
    ROOT_DIR / "lerobot" / "src",               # monorepo layout on cluster
    ROOT_DIR.parent / "repos" / "lerobot" / "src",  # split repos layout
]
for lerobot_src in lerobot_src_candidates:
    if lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))

# Ensure spawned training subprocesses inherit thesis_vla importability.
pythonpath_parts = [str(workspace_src)]
for lerobot_src in lerobot_src_candidates:
    if lerobot_src.exists():
        pythonpath_parts.append(str(lerobot_src))
existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
if existing_pythonpath:
    pythonpath_parts.append(existing_pythonpath)
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

from thesis_vla.common.paths import PROJECT_ROOT #type: ignore
from thesis_vla.training.xvla_finetune_launcher import ExperimentSpec, FreezeConfig, LaunchConfig, RuntimeConfig, run_experiments #type: ignore

WORKSPACE_DIR = PROJECT_ROOT

DEFAULTS = LaunchConfig(
    # ----- General model and database settings ------
    hf_user="edgarcancinoe",
    version="v1",
    dataset_name="soarm101_pickplace_multicolor_v1_7p5hz",
    dataset_revision="v3.0",
    base_model="lerobot/xvla-base",
    normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
    action_mode="so101_ee6d", # so101_ee6d / so101_joint

    # ----------- Runtime/Device settings -----------
    runtime=RuntimeConfig(
        launch_mode="single", # single / accelerate
        cuda_devices=(3,),
        num_workers=4,
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
    steps=30_000,

    # ------- Logging and checkpoint settings -------
    log_freq=500,
    eval_freq=-1,
    save_freq=30_000,
    push_every=30_000,
    policy_push_to_hub=True,
    wandb_enable=True,
    wandb_project="lerobot",

    # ------------ Augmentation settings ------------
    enable_augmentation=False,
    augmentation_degrees="[-1., 1.]",
    augmentation_translate="[0.015, 0.015]",
    augmentation_backend="custom",
    augmentation_enable_photometric=True,
    augmentation_fill_mode="reflect",
    mix_enabled=False,
    mix_base_repo_id=None,
    mix_new_repo_id=None,
    mix_new_repeat=1,
    mix_output_repo_id=None,
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

DATASET_CLOTH_DROP = "soarm101_square_cloth_corner_to_box_7p5hz"

# Experiment specs.
CUBE_EXPERIMENTS = [
    # Simple Orange
    # ------------------------------------------------------------------
    # 0: [Base ->      Orange196]  [NoAug] [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,    dataset_name=DATASET_ORANGE,  dataset_revision="main", 
        batch_size=16,  optimizer_lr=1e-4,  steps=30_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
    ),
    # 1: [Base ->      Orange196]  [Aug]   [train_all]
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,     dataset_name=DATASET_ORANGE, dataset_revision="main",  enable_augmentation=True,
        batch_size=32,  optimizer_lr=1e-4,  steps=30_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
    ),
    # ------------------------------------------------------------------
    
    # Multicolor
    # ------------------------------------------------------------------
    # 2: [Base ->      Multicolor] [NoAug] [train_all] [bs64]               DONE
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=45_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
    ),
    # 3: [Orange196 -> Multicolor] [NoAug] [train_all] [bs32]               DONE
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=45_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=1,  
    ),
    # 4: [Orange196 -> Multicolor] [NoAug] [train_all] [bs64]               DONE
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=45_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
    ),
    # 5: [Orange196 -> Multicolor] [NoAug] [domain-specific] [bs64]         DONE  
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_MULTICOLOR,
        batch_size=32,  optimizer_lr=1e-4,  steps=45_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
        freeze=train_domain_specific
    ),
]

CLOTH_EXPERIMENTS = [
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_ORANGE_196,     dataset_name=DATASET_CLOTH_DROP,
        batch_size=32,  optimizer_lr=1e-4,  steps=40_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
        save_freq=25_000,   push_every=25_000,
        freeze=train_domain_specific
    ),
    ExperimentSpec(
        action_mode="so101_ee6d",   
        base_model=BASE_MODEL,          dataset_name=DATASET_CLOTH_DROP,
        batch_size=32,      optimizer_lr=1e-4,  steps=40_000,  scheduler_decay_lr=1e-5, gradient_accumulation_steps=2,
        save_freq=25_000,   push_every=25_000,
        freeze=train_domain_specific
    ),    # ------------------------------------------------------------------

    # OOD adaptation (safe default): mix base fixed-location data with boosted random-placement data.
    # 6: [Orange196 -> mixed(base + 4x random)] [NoAug] [train_all] [stable lr/steps]
    ExperimentSpec(
        action_mode="so101_ee6d",
        base_model=BASE_ORANGE_196,
        dataset_name=DATASET_ORANGE,  # fallback base if mix_base_repo_id is not set
        dataset_revision="main",
        mix_enabled=True,
        mix_base_repo_id=DATASET_ORANGE,
        mix_new_repo_id=DATASET_MULTICOLOR,
        mix_new_repeat=4,
        mix_output_repo_id="soarm101_pickplace_ood_mix_orange_multicolor_r4",
        batch_size=32,
        optimizer_lr=3e-5,
        steps=12_000,
        scheduler_decay_lr=1e-5,
        gradient_accumulation_steps=2,
        enable_augmentation=False,
    ),
]

EXPERIMENTS = [CUBE_EXPERIMENTS[0]]

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)

if __name__ == "__main__":
    main()
