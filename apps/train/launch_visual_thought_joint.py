from __future__ import annotations

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

RUNTIME_CONFIG = VisualThoughtRuntimeConfig(launch_mode="single", cuda_devices=(0,), num_workers=0, dry_run=False)

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
    gradient_accumulation_steps=1,
    decoder_optimizer_lr=1e-4,
    xvla_optimizer_lr=1e-5,
    wandb_enable=True,
    wandb_project="visual-thought",
    action_loss_weight=1.0,
    expert_loss_weight=1.0,
    steps=2500,
    log_every=20,
    save_every=500,
    name_prefix="visual-thought",
)

EXPERIMENTS = [
    VisualThoughtExperimentSpec(
        name                        ="cedirnet_joint_stage",
        dataset_name                ="cloth-corner-fold_7p5hz",
        dataset_revision            ="v3.0",
        training_stage              ="joint_multitask",
        expert_type                 ="cedirnet",
        xvla_init_path              ="/home/jose/EMAI-Thesis/vla_workspace/runtime/outputs/train/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adapt_stagedpw_v1_20260604_141258/checkpoints/015000/pretrained_model",
        decoder_init_path           ="/home/jose/EMAI-Thesis/vla_workspace/models/cedirnet_legacy_32x32",
        decoder_stack_config_path   ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_stack.yaml",
        decoder_task_config_path    ="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/cedirnet_head.yaml",
        wandb_run_name              ="cedirnet_joint_stage",
        action_loss_weight=1.0,
        expert_loss_weight=0.25,
    ),
]



def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
