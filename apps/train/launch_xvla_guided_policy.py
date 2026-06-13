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

RUNTIME_CONFIG = GuidedRuntimeConfig(launch_mode="accelerate", cuda_devices=(2,3), num_workers=2, dry_run=False)

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
    guidance_train_mode="frozen", # frozen | train_from_start | warmup_freeze
    guidance_unfreeze_step = None,
    freeze_xvla_vlm=False,
    steps=8000,
    log_every=20,
    save_every=2000,
    wandb_enable=True,
    wandb_project="xvla-guided",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=10,
    validation_seed=1337,
    name_prefix=f"xvla-guided-{RUN_TS}",
)

# DATASETS
# =====================================================================================
CLOTH_FOLD_DS = ("cloth-corner-fold_7p5hz",     "main")
CLOTH_BOX_DS = ("cloth-corner-box_7p5hz",      "main")
# =====================================================================================

# EXP NAMING
# =====================================================================================
FOLD_CROSS_ATTN_NAME        = f"expl_cedir_ca_{RUN_TS}_cloth_fold"
FOLD_GATED_CROSS_ATTN_NAME  = f"expl_cedir_g-ca_{RUN_TS}_cloth_fold"
BOX_CROSS_ATTN_NAME        = f"expl_cedir_ca_{RUN_TS}_cloth_box"
BOX_GATED_CROSS_ATTN_NAME  = f"expl_cedir_g-ca_{RUN_TS}_cloth_box"
FOLD_BOTH_CROSS_ATTN_NAME        = f"expl_both_cedir_dino_ca_{RUN_TS}_cloth_fold"
FOLD_BOTH_GATED_CROSS_ATTN_NAME  = f"expl_both_cedir_dino_g-ca_{RUN_TS}_cloth_fold"
BOX_BOTH_CROSS_ATTN_NAME        = f"expl_both_cedir_dino_ca_{RUN_TS}_cloth_box"
BOX_BOTH_GATED_CROSS_ATTN_NAME  = f"expl_both_cedir_dino_g-ca_{RUN_TS}_cloth_box"
# =====================================================================================

# JOINT PRETRAINED CEDIRNET FOLD CHECKPOINT
# =====================================================================================
OUT = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/Implicit Models/Excel"

# -------------------------
# CEDIRNet joint: cloth fold
# -------------------------
JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT = (
    OUT + "/cedirnet_joint_stage_20260613_022141_cloth_fold/checkpoint_0004000"
)
JOINT_CEDIRNET_FOLD_XVLA_INIT = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT + "/policy"
JOINT_CEDIRNET_FOLD_DECODER_INIT = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT

# CEDIRNet joint: cloth box
# -------------------------
JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT = OUT + "/cedirnet_joint_stage_20260613_022141_cloth_box/checkpoint_0004000"
JOINT_CEDIRNET_BOX_XVLA_INIT = JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT + "/policy"
JOINT_CEDIRNET_BOX_DECODER_INIT = JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT

# BOTH: CEDIRNet + DINO joint, cloth fold
# -------------------------
BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT = OUT + "/both_cedirnet_dino_joint_clothfold_20260612_192204/checkpoint_0004000"
BOTH_CEDIRNET_DINO_FOLD_XVLA_INIT = BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT + "/policy"
BOTH_CEDIRNET_DINO_FOLD_DECODER_INIT = BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT

# BOTH: CEDIRNet + DINO joint, cloth box
# -------------------------
BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT = OUT + "/both_cedirnet_dino_joint_cloth_box_20260613_022649/checkpoint_0004000"
BOTH_CEDIRNET_DINO_BOX_XVLA_INIT = BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT + "/policy"
BOTH_CEDIRNET_DINO_BOX_DECODER_INIT = BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT
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

BOX_CEDIRNET_GUIDANCE = [
    GuidedExperimentSpec(
        name=BOX_CROSS_ATTN_NAME,
        wandb_run_name=BOX_CROSS_ATTN_NAME,
        dataset_name=CLOTH_BOX_DS[0],
        dataset_revision=CLOTH_BOX_DS[1],
        xvla_init_path=JOINT_CEDIRNET_BOX_XVLA_INIT,
        decoder_init_path=JOINT_CEDIRNET_BOX_DECODER_INIT,
        fusion_mode="cross_attention",
    ),
    GuidedExperimentSpec(
        name=BOX_GATED_CROSS_ATTN_NAME,
        wandb_run_name=BOX_GATED_CROSS_ATTN_NAME,
        dataset_name=CLOTH_BOX_DS[0],
        dataset_revision=CLOTH_BOX_DS[1],
        xvla_init_path=JOINT_CEDIRNET_BOX_XVLA_INIT,
        decoder_init_path=JOINT_CEDIRNET_BOX_DECODER_INIT,
        fusion_mode="gated_cross_attention",
    ),
]

FOLD_BOTH_CEDIRNET_DINO_GUIDANCE = [
    GuidedExperimentSpec(
        name=FOLD_BOTH_CROSS_ATTN_NAME,
        wandb_run_name=FOLD_BOTH_CROSS_ATTN_NAME,
        dataset_name=CLOTH_FOLD_DS[0],
        dataset_revision=CLOTH_FOLD_DS[1],
        xvla_init_path=BOTH_CEDIRNET_DINO_FOLD_XVLA_INIT,
        decoder_init_path=BOTH_CEDIRNET_DINO_FOLD_DECODER_INIT,
        fusion_mode="cross_attention",
    ),
    GuidedExperimentSpec(
        name=FOLD_BOTH_GATED_CROSS_ATTN_NAME,
        wandb_run_name=FOLD_BOTH_GATED_CROSS_ATTN_NAME,
        dataset_name=CLOTH_FOLD_DS[0],
        dataset_revision=CLOTH_FOLD_DS[1],
        xvla_init_path=BOTH_CEDIRNET_DINO_FOLD_XVLA_INIT,
        decoder_init_path=BOTH_CEDIRNET_DINO_FOLD_DECODER_INIT,
        fusion_mode="gated_cross_attention",
    ),
]

BOX_BOTH_CEDIRNET_DINO_GUIDANCE = [
    GuidedExperimentSpec(
        name=BOX_BOTH_CROSS_ATTN_NAME,
        wandb_run_name=BOX_BOTH_CROSS_ATTN_NAME,
        dataset_name=CLOTH_BOX_DS[0],
        dataset_revision=CLOTH_BOX_DS[1],
        xvla_init_path=BOTH_CEDIRNET_DINO_BOX_XVLA_INIT,
        decoder_init_path=BOTH_CEDIRNET_DINO_BOX_DECODER_INIT,
        fusion_mode="cross_attention",
    ),
    GuidedExperimentSpec(
        name=BOX_BOTH_GATED_CROSS_ATTN_NAME,
        wandb_run_name=BOX_BOTH_GATED_CROSS_ATTN_NAME,
        dataset_name=CLOTH_BOX_DS[0],
        dataset_revision=CLOTH_BOX_DS[1],
        xvla_init_path=BOTH_CEDIRNET_DINO_BOX_XVLA_INIT,
        decoder_init_path=BOTH_CEDIRNET_DINO_BOX_DECODER_INIT,
        fusion_mode="gated_cross_attention",
    ),
]

EXPERIMENTS = FOLD_CEDIRNET_GUIDANCE + BOX_CEDIRNET_GUIDANCE

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
