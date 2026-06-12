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
from thesis_vla.training.visual_thought_launcher import VisualThoughtExperimentSpec, VisualThoughtLaunchConfig, VisualThoughtRuntimeConfig, run_experiments

WORKSPACE_DIR = PROJECT_ROOT
RUN_TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

RUNTIME_CONFIG = VisualThoughtRuntimeConfig(launch_mode="single", cuda_devices=(1,), num_workers=3, dry_run=False)

DEFAULTS = VisualThoughtLaunchConfig(
    hf_user="edgarcancinoe",
    dataset_name="soarm101_pickplace_multicolor_v1_7p5hz",
    dataset_revision="v3.0",
    runtime=RUNTIME_CONFIG,
    training_stage="joint_multitask",
    expert_type="cedirnet",
    xvla_init_path="lerobot/xvla-base",
    decoder_stack_config_path=str(CONFIG_ROOT / "visual_thought" / "cedirnet_stack.yaml"),
    decoder_task_config_path=str(CONFIG_ROOT / "visual_thought" / "cedirnet_head.yaml"),
    batch_size=8,
    gradient_accumulation_steps=2,
    
    decoder_optimizer_lr=1e-4,
    xvla_adaptation_mode="staged_prompt_warmup",
    xvla_freeze_steps=0,
    xvla_warmup_steps=0,
    xvla_learning_coef=0.1,
    profile_step_time_every=20,
    xvla_scheduler_decay_steps=30000,
    xvla_scheduler_decay_lr=1e-5,

    wandb_enable=True,
    wandb_project="visual-thought",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=10,
    vis_every=500,
    vis_num_samples=3,
    vis_final=False,
    push_to_hub=True,
    push_repo_id=None,
    push_every=2000,
    action_loss_weight=1.0,
    expert_loss_weight=0.0,
    steps=8000,
    log_every=20,
    save_every=2000,
    name_prefix=f"visual-thought-{RUN_TS}",
)

# STARTING POINTS
# =====================================================================================
# XVLA -------------------------------------------------------------------------------
OUT = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/"
XVLA_INIT_CUBES                     = OUT + "orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model"
XVLA_INIT_CLOTHFOLD                 = OUT + "orange196_cloth-corner-fold_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_230620/checkpoints/015000/pretrained_model"  
XVLA_INIT_CLOTHDROP                 = OUT + "orange196_cloth-corner-box_7p5hz_so101_ee6d_am_sm_b8_ga4_eb64_full_adapt_stagedpw_v1_20260603_111556/checkpoints/030000/pretrained_model"

# CEDIRNET DECODER ----------------------------------------------------------------------
CEDIRNET_DECODER_INIT               = "/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32"
CEDIRNET_DECODER_INIT_CONFIG        = "/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_decoder.yaml"
CEDIRNET_DECODER_INIT_STACK_CONFIG  = "/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml"
CEDIRNET_DECODER_INIT_TASK_CONFIG   = "/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml"

# DINO DECODER --------------------------------------------------------------------------
DINO_CUBES_DECODER_INIT             = OUT + "dino_tokenseq_distill_cubes_20260611_005813/checkpoint_final"
DINO_CLOTH_DECODER_INIT             = OUT + "dino_tokenseq_distill_cloth_fold_20260611_000622/checkpoint_final"
DINO_CLOTH_DROP_DECODER_INIT        = OUT + "dino_tokenseq_distill_cloth_box_20260611_005813/checkpoint_final"

DINO_STACK_CONFIG                   = str(CONFIG_ROOT / "visual_thought" / "dino_stack.yaml")
DINO_TOKENSEQ_CONFIG                = str(CONFIG_ROOT / "visual_thought" / "dino_decoder.yaml")  # target_kind: token_sequence
# =====================================================================================


# EXP NAMING
# =====================================================================================
FOLD_CEDIRNET_NAME      = f"cedirnet_joint_stage_{RUN_TS}_cloth_fold"
DROP_CEDIRNET_NAME      = f"cedirnet_joint_stage_{RUN_TS}_cloth_box"

DINO_CUBES_NAME         = f"dino_tokenseq_joint_cubes_{RUN_TS}"
DINO_CLOTH_FOLD_NAME    = f"dino_tokenseq_joint_clothfold_{RUN_TS}"
DINO_CLOTH_DROP_NAME    = f"dino_tokenseq_joint_clothbox_{RUN_TS}"
BOTH_CLOTH_FOLD_NAME    = f"cedirnet_dino_joint_clothfold_{RUN_TS}"
# =====================================================================================

# DATASETS
# =====================================================================================
CLOTH_FOLD_DS = ("cloth-corner-fold_7p5hz",     "main")
CLOTH_DROP_DS = ("cloth-corner-box_7p5hz",      "main")
CUBES_DS      = ("pickplace-multicolor_7p5hz",  "main")
# =====================================================================================


FOLD_CEDIRNET = [
    VisualThoughtExperimentSpec(
        expert_type                 ="cedirnet",
        training_stage              ="joint_multitask",
        name                        =FOLD_CEDIRNET_NAME,
        wandb_run_name              =FOLD_CEDIRNET_NAME,
        dataset_name                =CLOTH_FOLD_DS[0],
        dataset_revision            =CLOTH_FOLD_DS[1],
        xvla_init_path              =XVLA_INIT_CLOTHFOLD,
        decoder_init_path           =CEDIRNET_DECODER_INIT,
        decoder_stack_config_path   =CEDIRNET_DECODER_INIT_STACK_CONFIG,
        decoder_task_config_path    =CEDIRNET_DECODER_INIT_TASK_CONFIG,
        # action_loss_weight=1.0,
        # expert_loss_weight=1.0,
    )
]

CLOTH_DROP_CEDIRNET = [
    VisualThoughtExperimentSpec(
        expert_type                 ="cedirnet",
        training_stage              ="joint_multitask",
        name                        =DROP_CEDIRNET_NAME,
        wandb_run_name              =DROP_CEDIRNET_NAME,
        dataset_name                =CLOTH_DROP_DS[0],
        dataset_revision            =CLOTH_DROP_DS[1],
        xvla_init_path              =XVLA_INIT_CLOTHDROP,
        decoder_init_path           =CEDIRNET_DECODER_INIT,
        decoder_stack_config_path   =CEDIRNET_DECODER_INIT_STACK_CONFIG,
        decoder_task_config_path    =CEDIRNET_DECODER_INIT_TASK_CONFIG,
        # action_loss_weight=1.0,
        # expert_loss_weight=1.0,
    )
]

DINO_CLOTH_FOLD = [
    VisualThoughtExperimentSpec(
        expert_type                 ="dino",
        training_stage              ="joint_multitask",
        name                        =DINO_CLOTH_FOLD_NAME,
        wandb_run_name              =DINO_CLOTH_FOLD_NAME,
        dataset_name                =CLOTH_FOLD_DS[0],
        dataset_revision            =CLOTH_FOLD_DS[1],
        xvla_init_path              =XVLA_INIT_CLOTHFOLD,
        decoder_init_path           =DINO_CLOTH_DECODER_INIT,
        decoder_stack_config_path   =DINO_STACK_CONFIG,
        decoder_task_config_path    =DINO_TOKENSEQ_CONFIG,
        # action_loss_weight=1.0,
        # expert_loss_weight=1.0,
    ),
]

DINO_CLOTH_DROP = [
    VisualThoughtExperimentSpec(
        expert_type                 ="dino",
        training_stage              ="joint_multitask",
        name                        =DINO_CLOTH_DROP_NAME,
        wandb_run_name              =DINO_CLOTH_DROP_NAME,
        dataset_name                =CLOTH_DROP_DS[0],
        dataset_revision            =CLOTH_DROP_DS[1],
        xvla_init_path              =XVLA_INIT_CLOTHDROP,
        decoder_init_path           =DINO_CLOTH_DROP_DECODER_INIT,
        decoder_stack_config_path   =DINO_STACK_CONFIG,
        decoder_task_config_path    =DINO_TOKENSEQ_CONFIG,
        # action_loss_weight=1.0,
        # expert_loss_weight=1.0,
    ),
]

BOTH_CLOTH_FOLD = [
    VisualThoughtExperimentSpec(
        expert_type                     ="cedirnet",
        expert_types                    =("cedirnet", "dino"),
        training_stage                  ="joint_multitask",
        name                            =BOTH_CLOTH_FOLD_NAME,
        dataset_name                    =CLOTH_FOLD_DS[0],
        dataset_revision                =CLOTH_FOLD_DS[1],
        xvla_init_path                  =XVLA_INIT_CLOTHFOLD,
        cedirnet_decoder_init_path      =CEDIRNET_DECODER_INIT,
        cedirnet_decoder_stack_config_path=CEDIRNET_DECODER_INIT_STACK_CONFIG,
        cedirnet_decoder_task_config_path=CEDIRNET_DECODER_INIT_TASK_CONFIG,
        dino_decoder_init_path          =DINO_CLOTH_DECODER_INIT,
        dino_decoder_stack_config_path  =DINO_STACK_CONFIG,
        dino_decoder_task_config_path   =DINO_TOKENSEQ_CONFIG,
        wandb_run_name                  =BOTH_CLOTH_FOLD_NAME,
        action_loss_weight=1.0,
        cedirnet_expert_loss_weight=1.0,
        dino_expert_loss_weight=0.25,
    ),
]

DINO_CUBES = [
    VisualThoughtExperimentSpec(
        expert_type                 ="dino",
        training_stage              ="joint_multitask",
        name                        =DINO_CUBES_NAME,
        wandb_run_name              =DINO_CUBES_NAME,
        dataset_name                =CUBES_DS[0],
        dataset_revision            =CUBES_DS[1],
        xvla_init_path              =XVLA_INIT_CUBES,
        decoder_init_path           =DINO_CUBES_DECODER_INIT,
        decoder_stack_config_path   =DINO_STACK_CONFIG,
        decoder_task_config_path    =DINO_TOKENSEQ_CONFIG,
        # action_loss_weight=1.0,
        # expert_loss_weight=1.0,
    ),
]

# Pendiente correr estos. Ajustar a bs 32. Dino antes habia corrido con un mal pre-trained decoder. Los otros hay q volver a correrlos solo extended
# EXPERIMENTS = FOLD_CEDIRNET + CLOTH_DROP_CEDIRNET
EXPERIMENTS = DINO_CLOTH_DROP

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
