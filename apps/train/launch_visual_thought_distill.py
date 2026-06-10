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

RUNTIME_CONFIG = VisualThoughtRuntimeConfig(launch_mode="single", cuda_devices=(0,), num_workers=0, dry_run=False)

DEFAULTS = VisualThoughtLaunchConfig(
    hf_user     ="edgarcancinoe",
    dataset_name="soarm101_pickplace_multicolor_v1_7p5hz",
    dataset_revision="v3.0",
    runtime=RUNTIME_CONFIG,
    training_stage="distill_only",
    expert_type="dino",
    xvla_init_path="lerobot/xvla-base",
    decoder_stack_config_path=str(CONFIG_ROOT / "visual_thought" / "dino_stack.yaml"),
    decoder_task_config_path=str(CONFIG_ROOT / "visual_thought" / "dino_decoder.yaml"),  # target_kind: token_sequence
    batch_size=8,
    gradient_accumulation_steps=1,
    decoder_optimizer_lr=1e-4,
    xvla_optimizer_lr=1e-5,
    steps=2500,
    log_every=20,
    save_every=500,
    name_prefix=f"visual-thought-{RUN_TS}",
)

# DINO distill_only: no target cache needed (teacher = stock dinov2 vitb14 from torch.hub).
# decoder_init_path is NOT required for distill_only — the student decoder is trained from scratch.
# The empty spec inherits expert_type/dataset/configs from DEFAULTS above.
EXPERIMENTS = [
    VisualThoughtExperimentSpec(
        name=f"dino_tokenseq_distill_cubes_{RUN_TS}",
        wandb_run_name=f"dino_tokenseq_distill_cubes_{RUN_TS}",
    ),
]


def main() -> None:
    run_experiments(workspace_dir=WORKSPACE_DIR, defaults=DEFAULTS, experiments=EXPERIMENTS)


if __name__ == "__main__":
    main()
