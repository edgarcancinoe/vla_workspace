from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
workspace_src = ROOT_DIR / "src"
sys.path.insert(0, str(workspace_src))
lerobot_src_candidates = [ROOT_DIR / "lerobot" / "src", ROOT_DIR.parent / "repos" / "lerobot" / "src"]
for lerobot_src in lerobot_src_candidates:
    if lerobot_src.exists(): sys.path.insert(0, str(lerobot_src))
pythonpath_parts = [str(workspace_src)]
for lerobot_src in lerobot_src_candidates:
    if lerobot_src.exists(): pythonpath_parts.append(str(lerobot_src))
existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
if existing_pythonpath: pythonpath_parts.append(existing_pythonpath)
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

from thesis_vla.common.paths import CONFIG_ROOT, PROJECT_ROOT
from thesis_vla.training.xvla_guided_launcher import GuidedExperimentSpec, GuidedLaunchConfig, GuidedRuntimeConfig, run_experiments

WORKSPACE_DIR = PROJECT_ROOT
RUN_TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

RUNTIME_CONFIG = GuidedRuntimeConfig(launch_mode="single", cuda_devices=(1,), num_workers=0, dry_run=False)

DEFAULTS = GuidedLaunchConfig(
    hf_user="edgarcancinoe",
    dataset_name="soarm101_pickplace_multicolor_v1_7p5hz",
    dataset_revision="v3.0",
    runtime=RUNTIME_CONFIG,
    xvla_init_path="lerobot/xvla-base",
    action_mode="so101_ee6d",
    decoder_stack_config_path=str(CONFIG_ROOT / "visual_thought" / "cedirnet_stack.yaml"),
    decoder_task_config_path=str(CONFIG_ROOT / "visual_thought" / "cedirnet_head.yaml"),
    guided_stage_config_path=str(CONFIG_ROOT / "visual_thought" / "cedirnet_guided_policy.yaml"),
    batch_size=8,
    gradient_accumulation_steps=1,
    decoder_optimizer_lr=1e-4,
    xvla_optimizer_lr=1e-5,
    xvla_scheduler_decay_lr=2.5e-6,
    guidance_train_mode="train_from_start",
    freeze_xvla_vlm=False,
    steps=2500,
    log_every=20,
    save_every=500,
    wandb_enable=True,
    wandb_project="xvla-guided",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=500,
    validation_max_batches=10,
    validation_seed=1337,
    name_prefix=f"xvla-guided-{RUN_TS}",
)

# DATASETS
# =====================================================================================
CLOTH_FOLD_DS = ("cloth-corner-fold_7p5hz",     "main")
CLOTH_DROP_DS = ("cloth-corner-box_7p5hz",      "main")
# =====================================================================================

# EXP NAMING
# =====================================================================================
FOLD_CROSS_ATTN_NAME        = f"guided_cedirnet_cross_attention_{RUN_TS}_cloth_fold"
FOLD_GATED_CROSS_ATTN_NAME  = f"guided_cedirnet_gated_cross_attention_{RUN_TS}_cloth_fold"
# =====================================================================================

# JOINT PRETRAINED CEDIRNET FOLD CHECKPOINT
# =====================================================================================
OUT = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/"
# Set this to the checkpoint root of the joint CeDirNet fold run, e.g.
# .../cedirnet_joint_stage_<timestamp>_cloth_fold/checkpoint_final
JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT = OUT + "<set_joint_cedirnet_fold_checkpoint_root>"
JOINT_CEDIRNET_FOLD_XVLA_INIT       = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT + "/policy"
JOINT_CEDIRNET_FOLD_DECODER_INIT    = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT
# =====================================================================================

FOLD_CEDIRNET_GUIDANCE = [
    GuidedExperimentSpec(
        name=FOLD_CROSS_ATTN_NAME,
        wandb_run_name=FOLD_CROSS_ATTN_NAME,
        dataset_name=CLOTH_FOLD_DS[0],
        dataset_revision=CLOTH_FOLD_DS[1],
        xvla_init_path=JOINT_CEDIRNET_FOLD_XVLA_INIT,
        decoder_init_path=JOINT_CEDIRNET_FOLD_DECODER_INIT,
        fusion_mode="cross_attention",
    ),
    GuidedExperimentSpec(
        name=FOLD_GATED_CROSS_ATTN_NAME,
        wandb_run_name=FOLD_GATED_CROSS_ATTN_NAME,
        dataset_name=CLOTH_FOLD_DS[0],
        dataset_revision=CLOTH_FOLD_DS[1],
        xvla_init_path=JOINT_CEDIRNET_FOLD_XVLA_INIT,
        decoder_init_path=JOINT_CEDIRNET_FOLD_DECODER_INIT,
        fusion_mode="gated_cross_attention",
    ),
]
EXPERIMENTS = FOLD_CEDIRNET_GUIDANCE


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
