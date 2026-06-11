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

RUNTIME_CONFIG = VisualThoughtRuntimeConfig(launch_mode="accelerate", cuda_devices=(1,2,3,4,5,6,7,8), num_workers=2, dry_run=False)

DATASET_CLOTH_BOX   = ("cloth-corner-box_7p5hz",        "main", f"dino_tokenseq_distill_cloth_box_{RUN_TS}") 
DATASET_CUBES       = ("pickplace-multicolor_7p5hz",    "main", f"dino_tokenseq_distill_cubes_{RUN_TS}")
CLOTH_FOLD_DS       = ("cloth-corner-fold_7p5hz",       "main", f"dino_tokenseq_distill_cloth_fold_{RUN_TS}")

# Pending run increased or decreased ffn_mlp_ratio 
DATA = CLOTH_FOLD_DS
DEFAULTS = VisualThoughtLaunchConfig(
    hf_user     ="edgarcancinoe",
    dataset_name=DATA[0],
    dataset_revision=DATA[1],
    runtime=RUNTIME_CONFIG,
    training_stage="distill_only",
    expert_type="dino",
    xvla_init_path="lerobot/xvla-base",
    decoder_stack_config_path=str(CONFIG_ROOT / "visual_thought" / "dino_stack.yaml"),
    decoder_task_config_path=str(CONFIG_ROOT / "visual_thought" / "dino_decoder.yaml"),  # target_kind: token_sequence
    batch_size=8,
    gradient_accumulation_steps=1,
    decoder_optimizer_lr=1e-3,
    xvla_optimizer_lr=1e-5,
    wandb_enable=True,
    wandb_project="visual-thought",
    steps=5000,
    log_every=20,
    save_every=500,
    name_prefix=f"visual-thought-{RUN_TS}",
    vis_every=500,
    vis_num_samples=3,
    vis_final=False,
    validation_enable=True,
    validation_split_ratio=0.1,
    validation_freq=250,
    validation_max_batches=10,
)

# DINO distill_only: no target cache needed (teacher = stock dinov2 vitb14 from torch.hub).
EXPERIMENTS = [
    # VisualThoughtExperimentSpec(
    #     name=DATASET_CUBES[2],
    #     wandb_run_name=DATASET_CUBES[2],
    #     dataset_name=DATASET_CUBES[0],
    #     dataset_revision=DATASET_CUBES[1],
    # ),
    # VisualThoughtExperimentSpec(
    #     name=DATASET_CLOTH_BOX[2],
    #     wandb_run_name=DATASET_CLOTH_BOX[2],
    #     dataset_name=DATASET_CLOTH_BOX[0],
    #     dataset_revision=DATASET_CLOTH_BOX[1],
    # ),
    VisualThoughtExperimentSpec(
        name=DATASET_CLOTH_BOX[2],
        wandb_run_name=DATASET_CLOTH_BOX[2],
        dataset_name=DATASET_CLOTH_BOX[0],
        dataset_revision=DATASET_CLOTH_BOX[1],
        decoder_stack_config_path="/home/jose/EMAI-Thesis/vla_workspace/config/visual_thought/dino_stackfn4.yaml"
    )
]

from dataclasses import replace
EXPERIMENTS = [replace(exp, name=f"{exp.name}{i}", wandb_run_name=f"{exp.wandb_run_name}{i}" if exp.wandb_run_name else None) for i, exp in enumerate(EXPERIMENTS)]

def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
