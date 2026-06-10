from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

from thesis_vla.common.paths import RUNTIME_CACHE_DIR, RUNTIME_TMP_DIR, TRAIN_OUTPUT_DIR


LaunchMode = Literal["single", "accelerate"]
TrainingStage = Literal["distill_only", "joint_multitask"]
ExpertType = Literal["cedirnet", "dino"]


@dataclass(frozen=True)
class VisualThoughtRuntimeConfig:
    launch_mode: LaunchMode = "single"
    cuda_devices: tuple[int, ...] = (0,)
    main_process_port: int = 45001
    device: str = "cuda"
    num_workers: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class VisualThoughtLaunchConfig:
    hf_user: str = os.environ.get("HF_USER", "edgarcancinoe")
    dataset_name: str = "soarm101_pickplace_multicolor_v1_7p5hz"
    dataset_revision: str = os.environ.get("DATASET_REVISION", "v3.0")
    dataset_root: str | None = None
    runtime: VisualThoughtRuntimeConfig = VisualThoughtRuntimeConfig()
    training_stage: TrainingStage = "distill_only"
    expert_type: ExpertType = "cedirnet"
    xvla_init_path: str = "lerobot/xvla-base"
    decoder_init_path: str | None = None
    decoder_stack_config_path: str = ""
    decoder_task_config_path: str = ""
    teacher_image_feature_key: str = "observation.images.image"
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    wandb_enable: bool = False
    wandb_project: str = "visual-thought"
    wandb_run_name: str | None = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    decoder_optimizer_lr: float = 1e-4
    xvla_optimizer_lr: float = 1e-5
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 1.0
    align_feature_until_step: int = 0
    steps: int = 2_500
    log_every: int = 20
    save_every: int = 500
    save_final_checkpoint: bool = True
    seed: int = 42
    name_prefix: str = "visual-thought"


@dataclass(frozen=True)
class VisualThoughtExperimentSpec:
    name: str | None = None
    output_dir: str | None = None
    dataset_name: str | None = None
    dataset_revision: str | None = None
    dataset_root: str | None = None
    training_stage: TrainingStage | None = None
    expert_type: ExpertType | None = None
    xvla_init_path: str | None = None
    decoder_init_path: str | None = None
    decoder_stack_config_path: str | None = None
    decoder_task_config_path: str | None = None
    teacher_image_feature_key: str | None = None
    dataset_video_backend: str | None = None
    dataset_tolerance_s: float | None = None
    wandb_enable: bool | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    decoder_optimizer_lr: float | None = None
    xvla_optimizer_lr: float | None = None
    action_loss_weight: float | None = None
    expert_loss_weight: float | None = None
    align_feature_until_step: int | None = None
    steps: int | None = None
    log_every: int | None = None
    save_every: int | None = None
    seed: int | None = None
    num_workers: int | None = None
    launch_mode: LaunchMode | None = None
    cuda_devices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ResolvedVisualThoughtExperiment:
    name: str
    output_dir: str
    dataset_repo_id: str
    dataset_revision: str | None
    dataset_root: str | None
    runtime: VisualThoughtRuntimeConfig
    training_stage: TrainingStage
    expert_type: ExpertType
    xvla_init_path: str
    decoder_init_path: str | None
    decoder_stack_config_path: str
    decoder_task_config_path: str
    teacher_image_feature_key: str
    dataset_video_backend: str
    dataset_tolerance_s: float
    wandb_enable: bool
    wandb_project: str
    wandb_run_name: str | None
    batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    decoder_optimizer_lr: float
    xvla_optimizer_lr: float
    action_loss_weight: float
    expert_loss_weight: float
    align_feature_until_step: int
    steps: int
    log_every: int
    save_every: int
    save_final_checkpoint: bool
    seed: int


def _resolve_hf_repo_id(repo_or_name: str, hf_user: str) -> str:
    repo_or_name = repo_or_name.strip()
    if "/" in repo_or_name: return repo_or_name
    return f"{hf_user}/{repo_or_name}"


def _slug(value: str) -> str:
    return value.split("/")[-1].replace("_", "-")


def _with_overrides(base, **overrides):
    valid = {field.name for field in fields(base)}
    payload = {key: value for key, value in overrides.items() if key in valid and value is not None}
    return replace(base, **payload)


def resolve_experiment(workspace_dir: Path, defaults: VisualThoughtLaunchConfig, experiment: VisualThoughtExperimentSpec, timestamp: str | None = None) -> ResolvedVisualThoughtExperiment:
    runtime = _with_overrides(defaults.runtime, launch_mode=experiment.launch_mode, cuda_devices=experiment.cuda_devices, num_workers=experiment.num_workers)
    dataset_repo_id = _resolve_hf_repo_id(experiment.dataset_name or defaults.dataset_name, defaults.hf_user)
    training_stage = experiment.training_stage or defaults.training_stage
    expert_type = experiment.expert_type or defaults.expert_type
    xvla_init_path = experiment.xvla_init_path or defaults.xvla_init_path
    decoder_init_path = experiment.decoder_init_path if experiment.decoder_init_path is not None else defaults.decoder_init_path
    decoder_stack_config_path = experiment.decoder_stack_config_path or defaults.decoder_stack_config_path
    decoder_task_config_path = experiment.decoder_task_config_path or defaults.decoder_task_config_path
    if not xvla_init_path: raise ValueError("xvla_init_path is required.")
    if not decoder_stack_config_path or not decoder_task_config_path: raise ValueError("decoder_stack_config_path and decoder_task_config_path are required.")
    if training_stage == "joint_multitask" and not decoder_init_path: raise ValueError("decoder_init_path is required for joint_multitask runs.")
    run_timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment.name or f"{defaults.name_prefix}_{training_stage}_{expert_type}_{_slug(dataset_repo_id)}_{run_timestamp}"
    output_dir = experiment.output_dir or str(TRAIN_OUTPUT_DIR / name)
    return ResolvedVisualThoughtExperiment(
        name=name,
        output_dir=output_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=experiment.dataset_revision or defaults.dataset_revision,
        dataset_root=experiment.dataset_root if experiment.dataset_root is not None else defaults.dataset_root,
        runtime=runtime,
        training_stage=training_stage,
        expert_type=expert_type,
        xvla_init_path=xvla_init_path,
        decoder_init_path=decoder_init_path,
        decoder_stack_config_path=decoder_stack_config_path,
        decoder_task_config_path=decoder_task_config_path,
        teacher_image_feature_key=experiment.teacher_image_feature_key or defaults.teacher_image_feature_key,
        dataset_video_backend=experiment.dataset_video_backend or defaults.dataset_video_backend,
        dataset_tolerance_s=experiment.dataset_tolerance_s if experiment.dataset_tolerance_s is not None else defaults.dataset_tolerance_s,
        wandb_enable=experiment.wandb_enable if experiment.wandb_enable is not None else defaults.wandb_enable,
        wandb_project=experiment.wandb_project or defaults.wandb_project,
        wandb_run_name=experiment.wandb_run_name or defaults.wandb_run_name or name,
        batch_size=experiment.batch_size if experiment.batch_size is not None else defaults.batch_size,
        gradient_accumulation_steps=experiment.gradient_accumulation_steps if experiment.gradient_accumulation_steps is not None else defaults.gradient_accumulation_steps,
        weight_decay=defaults.weight_decay,
        decoder_optimizer_lr=experiment.decoder_optimizer_lr if experiment.decoder_optimizer_lr is not None else defaults.decoder_optimizer_lr,
        xvla_optimizer_lr=experiment.xvla_optimizer_lr if experiment.xvla_optimizer_lr is not None else defaults.xvla_optimizer_lr,
        action_loss_weight=experiment.action_loss_weight if experiment.action_loss_weight is not None else defaults.action_loss_weight,
        expert_loss_weight=experiment.expert_loss_weight if experiment.expert_loss_weight is not None else defaults.expert_loss_weight,
        align_feature_until_step=experiment.align_feature_until_step if experiment.align_feature_until_step is not None else defaults.align_feature_until_step,
        steps=experiment.steps if experiment.steps is not None else defaults.steps,
        log_every=experiment.log_every if experiment.log_every is not None else defaults.log_every,
        save_every=experiment.save_every if experiment.save_every is not None else defaults.save_every,
        save_final_checkpoint=defaults.save_final_checkpoint,
        seed=experiment.seed if experiment.seed is not None else defaults.seed,
    )


def prepare_environment(workspace_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    pythonpath_parts = [str(workspace_dir / "src")]
    lerobot_src_candidates = [workspace_dir / "lerobot" / "src", workspace_dir.parent / "repos" / "lerobot" / "src"]
    for candidate in lerobot_src_candidates:
        if candidate.exists(): pythonpath_parts.append(str(candidate))
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath: pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    user = env.get("USER", "default_user")
    cache_root = RUNTIME_CACHE_DIR / f"xvla_{user}"
    os.makedirs(cache_root, exist_ok=True)
    env.pop("HF_HOME", None)
    env["HF_HUB_CACHE"] = str(cache_root / "hub")
    env["HF_ASSETS_CACHE"] = str(cache_root / "assets")
    env["HF_LEROBOT_HOME"] = str(cache_root / "lerobot")
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(env["HF_ASSETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_LEROBOT_HOME"], exist_ok=True)
    return env


def run_preflight_checks(experiments: list[ResolvedVisualThoughtExperiment], env: dict[str, str]) -> None:
    importlib.import_module("lerobot")
    probe = subprocess.run([sys.executable, "-c", "import thesis_vla"], env=env, capture_output=True, text=True, check=False)
    if probe.returncode != 0: raise SystemExit(f"ERROR: thesis_vla is not importable in subprocess environment.\n{(probe.stderr or probe.stdout).strip()}")
    if any(experiment.runtime.launch_mode != "single" for experiment in experiments): raise SystemExit("Visual-thought training currently supports launch_mode='single' only.")


def write_resolved_config(resolved: ResolvedVisualThoughtExperiment) -> Path:
    RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_TMP_DIR / f"{resolved.name}_visual_thought.json"
    payload = asdict(resolved)
    runtime = payload.pop("runtime")
    payload["device"] = runtime["device"]
    payload["num_workers"] = runtime["num_workers"]
    payload["dry_run"] = runtime["dry_run"]
    payload["cuda_visible_devices"] = list(runtime["cuda_devices"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def build_training_command(resolved: ResolvedVisualThoughtExperiment, config_path: Path) -> list[str]:
    return [sys.executable, "-m", "thesis_vla.training.visual_thought_trainer", f"--config_path={config_path}"]


def apply_runtime_environment(env: dict[str, str], runtime: VisualThoughtRuntimeConfig) -> dict[str, str]:
    updated = dict(env)
    updated["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in runtime.cuda_devices)
    return updated


def print_run_summary(index: int, total: int, resolved: ResolvedVisualThoughtExperiment, cmd: list[str]) -> None:
    print("=" * 88)
    print(f"Launching visual-thought experiment {index}/{total}")
    print("=" * 88)
    print(f"  Name:               {resolved.name}")
    print(f"  Stage:              {resolved.training_stage}")
    print(f"  Expert:             {resolved.expert_type}")
    print(f"  XVLA Init:          {resolved.xvla_init_path}")
    print(f"  Decoder Init:       {resolved.decoder_init_path}")
    print(f"  Dataset:            {resolved.dataset_repo_id}")
    print(f"  Video Backend:      {resolved.dataset_video_backend}")
    print(f"  Timestamp Tol:      {resolved.dataset_tolerance_s}")
    print(f"  W&B Enabled:        {resolved.wandb_enable}")
    print(f"  W&B Project:        {resolved.wandb_project}")
    print(f"  W&B Run:            {resolved.wandb_run_name}")
    print(f"  Output Dir:         {resolved.output_dir}")
    print(f"  Steps:              {resolved.steps}")
    print(f"  Batch Size:         {resolved.batch_size}")
    print(f"  Grad Accum:         {resolved.gradient_accumulation_steps}")
    print(f"  Decoder LR:         {resolved.decoder_optimizer_lr}")
    print(f"  XVLA LR:            {resolved.xvla_optimizer_lr}")
    print(f"  Dry Run:            {resolved.runtime.dry_run}")
    print(f"  Command:            {' '.join(cmd)}")
    print("=" * 88)


def run_experiments(workspace_dir: Path, defaults: VisualThoughtLaunchConfig, experiments: list[VisualThoughtExperimentSpec]) -> None:
    resolved = [resolve_experiment(workspace_dir, defaults, experiment) for experiment in experiments]
    env = prepare_environment(workspace_dir)
    run_preflight_checks(resolved, env)
    for index, experiment in enumerate(resolved, start=1):
        config_path = write_resolved_config(experiment)
        cmd = build_training_command(experiment, config_path)
        runtime_env = apply_runtime_environment(env, experiment.runtime)
        print_run_summary(index, len(resolved), experiment, cmd)
        if experiment.runtime.dry_run: continue
        exit_code = subprocess.call(cmd, env=runtime_env)
        if exit_code != 0: raise SystemExit(exit_code)
