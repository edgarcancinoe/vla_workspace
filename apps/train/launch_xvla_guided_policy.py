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


BS = 8
GA = 1
DEC_LR = 5e-5
XVLA_LR = 5e-6
MODE = "accelerate"
DEVICES = (0,1)

RUNTIME_CONFIG = GuidedRuntimeConfig(launch_mode=MODE, cuda_devices=DEVICES, num_workers=4, dry_run=False)
MULT = 1 if MODE == "single" else len(DEVICES)


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
    batch_size=BS,
    gradient_accumulation_steps=GA,
    decoder_optimizer_lr=DEC_LR,
    xvla_optimizer_lr=XVLA_LR,
    xvla_scheduler_decay_lr=2.5e-6,
    steps=4000,
    # ----–-----–-----–-----–-----–-----–-----–-----–-

    # Guidance use configuration ---------------------

    # Legacy-only decoder trainability knobs:
    # guidance_train_mode="frozen",
    # guidance_train_mode="train_from_start",
    # guidance_train_mode="warmup_freeze",
    guidance_train_mode="train_from_start", # frozen | train_from_start | warmup_freeze
    guidance_unfreeze_step = 1000,
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ Overriden if guidance_training_schedule="decoder_warmup_then_policy"

    guidance_training_schedule="decoder_warmup_then_policy",
    # Phase 1  (step <= guidance_warmup_steps): decoder-only, action loss off, expert loss = 1.0, corruption off
    guidance_warmup_steps=100,
    # Phase 2a (warmup < step < guidance_corruption_restore_step): decoder frozen, action loss on, phase2 expert loss on, corruption off
    guidance_phase2_expert_loss_weight=0.1,
    # Phase 2b (step >= guidance_corruption_restore_step): same as phase 2a, but dropout/noise restored
    guidance_corruption_restore_step=2500,
    guidance_dropout_prob=0.15,
    guidance_noise_prob=0.15,
    guidance_noise_std=0.10,
    freeze_xvla_vlm=True, # staged schedule requires True

    # Saving and logging -----------------------------
    log_every=20,
    save_every=2000,
    wandb_enable=True,
    wandb_project="xvla-guided-final",
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=1000000,
    validation_include_no_guidance=False,
    validation_seed=1337,
    name_prefix=f"xvla-guided-{RUN_TS}",
)

# DATASETS
# =====================================================================================
CLOTH_FOLD_DS = ("cloth-corner-fold_7p5hz",     "main")
CLOTH_BOX_DS  = ("cloth-corner-box_7p5hz",      "main")
# =====================================================================================

# JOINT PRETRAINED CEDIRNET FOLD CHECKPOINT
# =====================================================================================
OUT = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/Implicit Models/Best3k-Bases/"

# -------------------------
# CEDIRNet: cloth fold
# -------------------------
PATH = "1.0___cedirnet___fold_20260619_15451____1e-04_bs16"
CHECKPOINT = "/checkpoint_final"
JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT = OUT + PATH + CHECKPOINT
JOINT_CEDIRNET_FOLD_XVLA_INIT = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT + "/policy"
JOINT_CEDIRNET_FOLD_DECODER_INIT = JOINT_CEDIRNET_FOLD_CHECKPOINT_ROOT

# CEDIRNet: cloth box
# -------------------------
PATH = "1.0___cedirnet___box_20260619_154558___1e-04_bs16"
CHECKPOINT = "/checkpoint_final"
JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT = OUT + PATH + CHECKPOINT
JOINT_CEDIRNET_BOX_XVLA_INIT = JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT + "/policy"
JOINT_CEDIRNET_BOX_DECODER_INIT = JOINT_CEDIRNET_BOX_CHECKPOINT_ROOT

# BOTH: CEDIRNet + DINO joint, cloth fold
# -------------------------
PATH = "None" # "/both_cedirnet_dino_joint_clothfold_20260612_192204/checkpoint_0004000"
CHECKPOINT = "/checkpoint_final"
BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT = OUT + PATH + CHECKPOINT
BOTH_CEDIRNET_DINO_FOLD_XVLA_INIT = BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT + "/policy"
BOTH_CEDIRNET_DINO_FOLD_DECODER_INIT = BOTH_CEDIRNET_DINO_FOLD_CHECKPOINT_ROOT

# BOTH: CEDIRNet + DINO joint, cloth box
# -------------------------
PATH = "None" # "/both_cedirnet_dino_joint_cloth_box_20260613_022649/checkpoint_0004000"
CHECKPOINT = "/checkpoint_final"
BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT = OUT + PATH + CHECKPOINT
BOTH_CEDIRNET_DINO_BOX_XVLA_INIT = BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT + "/policy"
BOTH_CEDIRNET_DINO_BOX_DECODER_INIT = BOTH_CEDIRNET_DINO_BOX_CHECKPOINT_ROOT
# =====================================================================================

# RESOLVERS
# =====================================================================================
TASK_SUFFIX  = {"fold": "fold", "box": "box"}
TASK_DATASET = {"fold": CLOTH_FOLD_DS, "box": CLOTH_BOX_DS}
NAME_PREFIX  = {"cedirnet": "cedir", "both": "both"}

MODE_TAG = {
    "baseline": "baseline_control",
    "concat_after": "concat_after_visual",
    "concat_before": "concat_before_vlm",
    "iface_after": "concat_iface_after_visual",
    "iface_before": "concat_iface_before_vlm",
    "selected_after": "selected_layers_after_visual",
    "selected_before": "selected_layers_before_vlm",
    "gated_after": "concat_gated_after_visual",
    "gated_before": "concat_gated_before_vlm",
    "selected_after": "selected_layers_after_visual",
    "selected_before": "selected_layers_before_vlm",
}

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
    return f"{TASK_SUFFIX[task]}__{NAME_PREFIX[family]}__{MODE_TAG[mode]}__bs{MULT * GA * BS}_dec_lr{DEC_LR}_vla_lr{XVLA_LR}__{RUN_TS}"

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

def guided_spec(task: str, mode: str, position: str, family: str = "cedirnet", **overrides) -> GuidedExperimentSpec:
    name = run_name(task, mode, family)
    return GuidedExperimentSpec(
        name=name,
        wandb_run_name=name,
        dataset_name=resolve_ds(task)[0],
        dataset_revision=resolve_ds(task)[1],
        xvla_init_path=resolve_xvla_base(task, family),
        decoder_init_path=resolve_decoder_base(task, family),
        guidance_mode=overrides.pop("guidance_mode", "concat"),
        guidance_insertion_position=overrides.pop("guidance_insertion_position", position),
        **overrides,
    )

# ===== FOLD ==================================================
# ONLY CEDIRNET

FOLD_CEDIRNET_GUIDANCE = [
    guided_spec("fold", "concat_after", "after_visual"),
    guided_spec("fold", "concat_before", "before_vlm"),
] # RAN
FOLD_CEDIRNET_INTERFACE_GUIDANCE = [
    # guided_spec("fold", "iface_after", "after_visual", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
    guided_spec("fold", "iface_before", "before_vlm", guidance_use_interface_projection=True, guidance_interface_num_tokens=32),
] # RUNNING BEFORE_VLM
FOLD_CEDIRNET_GATED_GUIDANCE = [
    guided_spec("fold", "gated_after", "after_visual", guidance_concat_gating=True),
    guided_spec("fold", "gated_before", "before_vlm", guidance_concat_gating=True),
]
FOLD_CEDIRNET_SELECTED_GUIDANCE = [
    guided_spec("fold", "selected_before", "before_vlm", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
    guided_spec("fold", "selected_before", "before_vlm", guidance_mode="selected_layers", guidance_selected_layers=(6,12,18)),
] # Running Before VLM

FOLD_CEDIRNET_BASELINE_CONTROL = [
    guided_spec("fold", "baseline", "after_visual", guidance_mode="disabled", expert_loss_weight=0.0, validation_include_no_guidance=False),
]

# CEDIRNET AND DINO
FOLD_BOTH_CEDIRNET_DINO_GUIDANCE = [
    guided_spec("fold", "concat_after", "after_visual", family="both"),
    guided_spec("fold", "concat_before", "before_vlm", family="both"),
]
FOLD_BOTH_CEDIRNET_DINO_INTERFACE_GUIDANCE = [
    guided_spec("fold", "iface_after", "after_visual", family="both", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
    guided_spec("fold", "iface_before", "before_vlm", family="both", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
]
FOLD_BOTH_CEDIRNET_DINO_GATED_GUIDANCE = [
    guided_spec("fold", "gated_after", "after_visual", family="both", guidance_concat_gating=True),
    guided_spec("fold", "gated_before", "before_vlm", family="both", guidance_concat_gating=True),
]
FOLD_BOTH_CEDIRNET_DINO_SELECTED_GUIDANCE = [
    guided_spec("fold", "selected_after", "after_visual", family="both", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
    guided_spec("fold", "selected_before", "before_vlm", family="both", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
]
# =============================================================


# ===== BOX ===================================================
# ONLY CEDIRNET
BOX_CEDIRNET_GUIDANCE = [
    guided_spec("box", "concat_after", "after_visual"),
    guided_spec("box", "concat_before", "before_vlm"),
] # RAN
BOX_CEDIRNET_INTERFACE_GUIDANCE = [
    guided_spec("box", "iface_after", "after_visual", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
    guided_spec("box", "iface_before", "before_vlm", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
] # RUNNING BEFORE_VLM
BOX_CEDIRNET_GATED_GUIDANCE = [
    guided_spec("box", "gated_after", "after_visual", guidance_concat_gating=True),
    guided_spec("box", "gated_before", "before_vlm", guidance_concat_gating=True),
]
BOX_CEDIRNET_SELECTED_GUIDANCE = [
    guided_spec("box", "selected_before", "before_vlm", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
    guided_spec("box", "selected_before", "before_vlm", guidance_mode="selected_layers", guidance_selected_layers=(6,12,18)),
]

BOX_CEDIRNET_BASELINE_CONTROL = [
    guided_spec("box", "baseline", "after_visual", guidance_mode="disabled", expert_loss_weight=0.0, validation_include_no_guidance=False),
]

# CEDIRNET AND DINO
BOX_BOTH_CEDIRNET_DINO_INTERFACE_GUIDANCE = [
    guided_spec("box", "iface_after", "after_visual", family="both", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
    guided_spec("box", "iface_before", "before_vlm", family="both", guidance_use_interface_projection=True, guidance_interface_num_tokens=64),
]
BOX_BOTH_CEDIRNET_DINO_GATED_GUIDANCE = [
    guided_spec("box", "gated_after", "after_visual", family="both", guidance_concat_gating=True),
    guided_spec("box", "gated_before", "before_vlm", family="both", guidance_concat_gating=True),
]
BOX_BOTH_CEDIRNET_DINO_GUIDANCE = [
    guided_spec("box", "concat_after", "after_visual", family="both"),
    guided_spec("box", "concat_before", "before_vlm", family="both"),
]
BOX_BOTH_CEDIRNET_DINO_SELECTED_GUIDANCE = [
    guided_spec("box", "selected_after", "after_visual", family="both", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
    guided_spec("box", "selected_before", "before_vlm", family="both", guidance_mode="selected_layers", guidance_selected_layers=(11,)),
]
# =============================================================


# EXPERIMENTS = [FOLD_CEDIRNET_GUIDANCE[0]]
# EXPERIMENTS = [BOX_CEDIRNET_GUIDANCE[0]]
# EXPERIMENTS = [BOX_CEDIRNET_GUIDANCE[1]]
EXPERIMENTS = [FOLD_CEDIRNET_INTERFACE_GUIDANCE[0]]
def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
