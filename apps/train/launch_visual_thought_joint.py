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

RUNTIME_CONFIG = VisualThoughtRuntimeConfig(launch_mode="single", cuda_devices=(2,), num_workers=2, dry_run=False)

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
    decoder_optimizer_lr=1e-3,
    xvla_optimizer_lr=1e-5,
    wandb_enable=True,
    wandb_project="visual-thought",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=10,
    push_to_hub=False,
    push_repo_id=None,
    push_every=500,
    action_loss_weight=1.0,
    expert_loss_weight=1.0,
    steps=2400,
    log_every=20,
    save_every=1000,
    name_prefix=f"visual-thought-{RUN_TS}",
)

EXPERIMENTS = [
    VisualThoughtExperimentSpec(
        name                        =f"cedirnet_joint_stage_{RUN_TS}",
        dataset_name                ="cloth-corner-fold_7p5hz",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        wandb_run_name              =f"cedirnet_joint_stage_{RUN_TS}",
        action_loss_weight=1.0,
        expert_loss_weight=0.25,
    ),
    VisualThoughtExperimentSpec(
        name                        ="cedirnet_joint_stage",
        dataset_name                ="cloth-corner-fold_7p5hz",
        wandb_run_name              ="cedirnet_joint_stage",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        action_loss_weight=1.0,
        expert_loss_weight=0.5,
    ),
    VisualThoughtExperimentSpec(
        name                        ="cedirnet_joint_stage",
        dataset_name                ="cloth-corner-fold_7p5hz",
        wandb_run_name              ="cedirnet_joint_stage",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        action_loss_weight=1.0,
        expert_loss_weight=1.0,
    ),
    VisualThoughtExperimentSpec(
        name                        ="cedirnet_joint_stage",
        dataset_name                ="cloth-corner-fold_7p5hz",
        wandb_run_name              ="cedirnet_joint_stage",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        action_loss_weight=1.0,
        expert_loss_weight=0.5,
    ),
    VisualThoughtExperimentSpec(
        name                        ="cedirnet_joint_stage",
        dataset_name                ="cloth-corner-fold_7p5hz",
        wandb_run_name              ="cedirnet_joint_stage",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        action_loss_weight=1.0,
        expert_loss_weight=1.0,
    ),
]


# =====================================================================================
# DINO (token_sequence) joint experiments
# -------------------------------------------------------------------------------------
# Teacher = stock dinov2 vitb14 from torch.hub (configs/dino: checkpoint=null), identical
# to how XVLA-VisualThought/scripts/train/train_dino.py distilled the student decoder.
#
# decoder_init_path must point at a CONVERTED student decoder (a directory containing
# decoder.safetensors), produced from the train_dino.py output via:
#   python apps/train/convert_dino_checkpoint.py \
#       --checkpoint <train_dino .../checkpoint_final.pt> \
#       --config     <XVLA-VisualThought/configs/dino/train.yaml> \
#       --output-dir <runtime/outputs/train/visual_thought_imports/dino_...>
#
# IMPORTANT: the converted decoder's num_decoder_tokens is fixed by the dinov2 patch grid
# at distillation time, so use the cloth-fold decoder for the cloth-fold joint run and the
# cubes decoder for the cubes joint run (same camera resolution / teacher config on both
# sides), otherwise the strict decoder load will fail on a shape mismatch.
# =====================================================================================
DINO_STACK_CONFIG     = str(CONFIG_ROOT / "visual_thought" / "dino_stack.yaml")
DINO_TOKENSEQ_CONFIG  = str(CONFIG_ROOT / "visual_thought" / "dino_decoder.yaml")  # target_kind: token_sequence

# XVLA inits (cloth-fold reuses the same orange196 pickplace init as the cedirnet specs).
DINO_XVLA_INIT_CLOTHFOLD = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model"
DINO_XVLA_INIT_CUBES     = None  # TODO: provide the XVLA init checkpoint for the cubes joint run

# Converted student-decoder dirs (output of convert_dino_checkpoint.py).
DINO_DECODER_INIT_CLOTHFOLD = None  # TODO: provide converted cloth-fold DINO decoder dir
DINO_DECODER_INIT_CUBES     = None  # TODO: provide converted cubes DINO decoder dir

DINO_CLOTH_FOLD = [
    VisualThoughtExperimentSpec(
        name                        =f"dino_tokenseq_joint_clothfold_{RUN_TS}",
        dataset_name                ="cloth-corner-fold_7p5hz",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="dino",
        xvla_init_path              =DINO_XVLA_INIT_CLOTHFOLD,
        decoder_init_path           =DINO_DECODER_INIT_CLOTHFOLD,
        decoder_stack_config_path   =DINO_STACK_CONFIG,
        decoder_task_config_path    =DINO_TOKENSEQ_CONFIG,
        wandb_run_name              =f"dino_tokenseq_joint_clothfold_{RUN_TS}",
        action_loss_weight=1.0,
        expert_loss_weight=0.25,
    ),
]

DINO_CUBES = [
    VisualThoughtExperimentSpec(
        name                        =f"dino_tokenseq_joint_cubes_{RUN_TS}",
        dataset_name                ="pickplace-multicolor_7p5hz",
        dataset_revision            ="main",
        training_stage              ="joint_multitask",
        expert_type                 ="dino",
        xvla_init_path              =DINO_XVLA_INIT_CUBES,
        decoder_init_path           =DINO_DECODER_INIT_CUBES,
        decoder_stack_config_path   =DINO_STACK_CONFIG,
        decoder_task_config_path    =DINO_TOKENSEQ_CONFIG,
        wandb_run_name              =f"dino_tokenseq_joint_cubes_{RUN_TS}",
        action_loss_weight=1.0,
        expert_loss_weight=0.25,
    ),
]


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
