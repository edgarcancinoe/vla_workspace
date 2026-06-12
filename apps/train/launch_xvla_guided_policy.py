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
FOLD_CEDIRNET_NAME      = f"explicit_cedirnet_{RUN_TS}_cloth_fold"
DROP_CEDIRNET_NAME      = f"explicit_cedirnet_{RUN_TS}_cloth_box"
# =====================================================================================

# XVLA 
# =====================================================================================
OUT = "/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/"
XVLA_INIT_CLOTHFOLD     = OUT + "orange196_cloth-corner-fold_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_230620/checkpoints/015000/pretrained_model"  
XVLA_INIT_CLOTHDROP     = OUT + "orange196_cloth-corner-box_7p5hz_so101_ee6d_am_sm_b8_ga4_eb64_full_adapt_stagedpw_v1_20260603_111556/checkpoints/030000/pretrained_model"
# =====================================================================================

FOLD_CEDIRNET_GUIDANCE = [
    GuidedLaunchConfig(
        name=FOLD_CEDIRNET_NAME,
        dataset_name=CLOTH_FOLD_DS[0],
        dataset_revision=CLOTH_FOLD_DS[1],
        xvla_init_path=None,
        decoder_stack_config_path=None,
        decoder_task_config_path=None,
        guided_stage_config_path=None
    )
]
EXPERIMENTS = []


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
