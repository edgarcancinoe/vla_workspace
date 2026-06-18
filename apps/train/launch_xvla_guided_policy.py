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

RUNTIME_CONFIG = GuidedRuntimeConfig(launch_mode="accelerate", cuda_devices=(0,1), num_workers=2, dry_run=False)

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

    # LEARNING CONFIUGRATIONS ----–-----–-----–-----–-
    batch_size=8,
    gradient_accumulation_steps=1,
    decoder_optimizer_lr=1e-4,
    xvla_optimizer_lr=1e-5,
    xvla_scheduler_decay_lr=2.5e-6,
    steps=8000,
    # ----–-----–-----–-----–-----–-----–-----–-----–-

    # Guidance use configuration ---------------------
    guidance_train_mode="train_from_start", # frozen | train_from_start | warmup_freeze
    guidance_unfreeze_step = 1000,
    guidance_dropout_prob=0.15,
    guidance_noise_prob=0.15,
    guidance_noise_std=0.10,
    freeze_xvla_vlm=False,
    # ----–-----–-----–-----–-----–-----–-----–-----–-

    # Saving and logging -----------------------------
    log_every=20,
    save_every=2000,
    wandb_enable=True,
    wandb_project="xvla-guided",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=10,
    validation_include_no_guidance=True,
    validation_seed=1337,
    name_prefix=f"xvla-guided-{RUN_TS}",
)

# DATASETS
# =====================================================================================
CLOTH_FOLD_DS = ("cloth-corner-fold_7p5hz",     "main")
CLOTH_BOX_DS = ("cloth-corner-box_7p5hz",      "main")
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

# RESOLVERS
# =====================================================================================
TASK_SUFFIX = {"fold": "cloth_fold", "box": "cloth_box"}
TASK_DATASET = {"fold": CLOTH_FOLD_DS, "box": CLOTH_BOX_DS}
NAME_PREFIX = {"cedirnet": "expl_cedir", "both": "expl_both_cedir_dino"}
MODE_TAG = {"concat_after": "concat_after_visual", "concat_before": "concat_before_vlm"}
XVLA_BASE = {
    ("cedirnet", "fold"): JOINT_CEDIRNET_FOLD_XVLA_INIT,
    ("cedirnet", "box"): JOINT_CEDIRNET_BOX_XVLA_INIT,
    ("both", "fold"): BOTH_CEDIRNET_DINO_FOLD_XVLA_INIT,
    ("both", "box"): BOTH_CEDIRNET_DINO_BOX_XVLA_INIT,
}
DECODER_BASE = {
    ("cedirnet", "fold"): JOINT_CEDIRNET_FOLD_DECODER_INIT,
    ("cedirnet", "box"): JOINT_CEDIRNET_BOX_DECODER_INIT,
    ("both", "fold"): BOTH_CEDIRNET_DINO_FOLD_DECODER_INIT,
    ("both", "box"): BOTH_CEDIRNET_DINO_BOX_DECODER_INIT,
}

def run_name(task: str, mode: str, family: str = "cedirnet") -> str:
    return f"{NAME_PREFIX[family]}_{MODE_TAG[mode]}_{RUN_TS}_{TASK_SUFFIX[task]}"

def fold_name(mode: str, family: str = "cedirnet") -> str:
    return run_name("fold", mode, family)

def box_name(mode: str, family: str = "cedirnet") -> str:
    return run_name("box", mode, family)

def resolve_ds(task: str = "fold") -> tuple[str, str]:
    return TASK_DATASET[task]

def resolve_xvla_base(task: str = "fold", family: str = "cedirnet") -> str:
    return XVLA_BASE[(family, task)]

def resolve_decoder_base(task: str = "fold", family: str = "cedirnet") -> str:
    return DECODER_BASE[(family, task)]

def guided_spec(task: str, mode: str, position: str, family: str = "cedirnet") -> GuidedExperimentSpec:
    name = run_name(task, mode, family)
    return GuidedExperimentSpec(
        name=name,
        wandb_run_name=name,
        dataset_name=resolve_ds(task)[0],
        dataset_revision=resolve_ds(task)[1],
        xvla_init_path=resolve_xvla_base(task, family),
        decoder_init_path=resolve_decoder_base(task, family),
        guidance_mode="concat",
        guidance_insertion_position=position,
    )

FOLD_CEDIRNET_GUIDANCE = [
    guided_spec("fold", "concat_after", "after_visual"),
    guided_spec("fold", "concat_before", "before_vlm"),
]

BOX_CEDIRNET_GUIDANCE = [
    guided_spec("box", "concat_after", "after_visual"),
    guided_spec("box", "concat_before", "before_vlm"),
]

FOLD_BOTH_CEDIRNET_DINO_GUIDANCE = [
    guided_spec("fold", "concat_after", "after_visual", family="both"),
    guided_spec("fold", "concat_before", "before_vlm", family="both"),
]

BOX_BOTH_CEDIRNET_DINO_GUIDANCE = [
    guided_spec("box", "concat_after", "after_visual", family="both"),
    guided_spec("box", "concat_before", "before_vlm", family="both"),
]

EXPERIMENTS = FOLD_CEDIRNET_GUIDANCE + BOX_CEDIRNET_GUIDANCE + FOLD_BOTH_CEDIRNET_DINO_GUIDANCE + BOX_BOTH_CEDIRNET_DINO_GUIDANCE

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
