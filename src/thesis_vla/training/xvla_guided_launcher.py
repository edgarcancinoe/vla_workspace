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

import yaml

from thesis_vla.common.paths import CONFIG_ROOT, RUNTIME_CACHE_DIR, RUNTIME_TMP_DIR, TRAIN_OUTPUT_DIR
from thesis_vla.policies.xvla_guided.configuration_xvla_guided import normalize_guidance_fusion_mode
from thesis_vla.training.xvla_finetune_launcher import validate_mean_std_normalization


LaunchMode = Literal["single", "accelerate"]


@dataclass(frozen=True)
class GuidedRuntimeConfig:
    launch_mode: LaunchMode = "single"
    cuda_devices: tuple[int, ...] = (0,)
    main_process_port: int = 45001
    mixed_precision: str = "no"
    device: str = "cuda"
    num_workers: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class GuidedLaunchConfig:
    hf_user: str = os.environ.get("HF_USER", "edgarcancinoe")
    dataset_name: str = "soarm101_pickplace_multicolor_v1_7p5hz"
    dataset_revision: str = os.environ.get("DATASET_REVISION", "v3.0")
    dataset_root: str | None = None
    runtime: GuidedRuntimeConfig = GuidedRuntimeConfig()
    xvla_init_path: str = "lerobot/xvla-base"
    action_mode: str | None = None
    decoder_init_path: str = ""
    decoder_stack_config_path: str = str(CONFIG_ROOT / "visual_thought" / "cedirnet_stack.yaml")
    decoder_task_config_path: str = str(CONFIG_ROOT / "visual_thought" / "cedirnet_head.yaml")
    guided_stage_config_path: str = str(CONFIG_ROOT / "visual_thought" / "cedirnet_guided_policy.yaml")
    teacher_image_feature_key: str = "observation.images.image"
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    normalization_mapping: str = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    decoder_optimizer_lr: float = 1e-4
    xvla_optimizer_lr: float = 1e-5
    xvla_scheduler_decay_lr: float | None = 2.5e-6
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 0.25
    fusion_mode: str = "concat"
    gated_fusion: bool | None = None
    guidance_train_mode: str = "frozen"
    guidance_unfreeze_step: int = 1_000
    freeze_xvla_vlm: bool = True
    steps: int = 2_500
    log_every: int = 20
    save_every: int = 500
    save_final_checkpoint: bool = True
    resume: bool = False
    resume_checkpoint_path: str | None = None
    wandb_enable: bool = True
    wandb_project: str = "xvla-guided"
    wandb_run_name: str | None = None
    validation_enable: bool = True
    validation_split_ratio: float = 0.1
    validation_freq: int = 500
    validation_max_batches: int = 10
    validation_seed: int = 1337
    seed: int = 42
    name_prefix: str = "xvla-guided"


@dataclass(frozen=True)
class GuidedExperimentSpec:
    name: str | None = None
    output_dir: str | None = None
    dataset_name: str | None = None
    dataset_revision: str | None = None
    dataset_root: str | None = None
    xvla_init_path: str | None = None
    action_mode: str | None = None
    decoder_init_path: str | None = None
    decoder_stack_config_path: str | None = None
    decoder_task_config_path: str | None = None
    teacher_image_feature_key: str | None = None
    dataset_video_backend: str | None = None
    dataset_tolerance_s: float | None = None
    normalization_mapping: str | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    decoder_optimizer_lr: float | None = None
    xvla_optimizer_lr: float | None = None
    xvla_scheduler_decay_lr: float | None = None
    action_loss_weight: float | None = None
    expert_loss_weight: float | None = None
    fusion_mode: str | None = None
    gated_fusion: bool | None = None
    guidance_train_mode: str | None = None
    guidance_unfreeze_step: int | None = None
    freeze_xvla_vlm: bool | None = None
    steps: int | None = None
    log_every: int | None = None
    save_every: int | None = None
    resume: bool | None = None
    resume_checkpoint_path: str | None = None
    wandb_enable: bool | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    validation_enable: bool | None = None
    validation_split_ratio: float | None = None
    validation_freq: int | None = None
    validation_max_batches: int | None = None
    validation_seed: int | None = None
    seed: int | None = None
    num_workers: int | None = None
    launch_mode: LaunchMode | None = None
    cuda_devices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ResolvedGuidedExperiment:
    name: str
    output_dir: str
    dataset_repo_id: str
    dataset_revision: str | None
    dataset_root: str | None
    runtime: GuidedRuntimeConfig
    xvla_init_path: str
    action_mode: str | None
    decoder_init_path: str
    decoder_stack_config_path: str
    decoder_task_config_path: str
    teacher_image_feature_key: str
    dataset_video_backend: str
    dataset_tolerance_s: float
    batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    decoder_optimizer_lr: float
    xvla_optimizer_lr: float
    xvla_scheduler_decay_lr: float | None
    action_loss_weight: float
    expert_loss_weight: float
    fusion_mode: str
    guidance_train_mode: str
    guidance_unfreeze_step: int
    freeze_xvla_vlm: bool
    steps: int
    log_every: int
    save_every: int
    save_final_checkpoint: bool
    wandb_enable: bool
    wandb_project: str
    wandb_run_name: str | None
    validation_enable: bool
    validation_split_ratio: float
    validation_freq: int
    validation_max_batches: int
    validation_seed: int
    seed: int
    normalization_mapping: str
    resume: bool
    resume_checkpoint_path: str | None


def _read_yaml(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict): raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _resolve_hf_repo_id(repo_or_name: str, hf_user: str) -> str:
    repo_or_name = repo_or_name.strip()
    if "/" in repo_or_name: return repo_or_name
    return f"{hf_user}/{repo_or_name}"


def _slug(value: str) -> str:
    return value.split("/")[-1].replace("_", "-")


def _with_overrides(base, **overrides):
    valid = {field.name for field in fields(base)}
    return replace(base, **{key: value for key, value in overrides.items() if key in valid and value is not None})


def _stage_defaults(path: str | Path) -> dict:
    payload = _read_yaml(path)
    return {key: payload[key] for key in ["fusion_mode", "guidance_fusion_mode", "gated_fusion", "guidance_train_mode", "guidance_unfreeze_step", "freeze_xvla_vlm", "action_loss_weight", "expert_loss_weight"] if key in payload}


def _resolve_fusion_mode(*, defaults: GuidedLaunchConfig, experiment: GuidedExperimentSpec, stage_defaults: dict) -> str:
    if experiment.fusion_mode is not None: return normalize_guidance_fusion_mode(experiment.fusion_mode, experiment.gated_fusion)
    if "guidance_fusion_mode" in stage_defaults: return normalize_guidance_fusion_mode(stage_defaults["guidance_fusion_mode"])
    if "fusion_mode" in stage_defaults: return normalize_guidance_fusion_mode(stage_defaults["fusion_mode"], stage_defaults.get("gated_fusion"))
    return normalize_guidance_fusion_mode(defaults.fusion_mode, defaults.gated_fusion)


def resolve_experiment(workspace_dir: Path, defaults: GuidedLaunchConfig, experiment: GuidedExperimentSpec, timestamp: str | None = None) -> ResolvedGuidedExperiment:
    runtime = _with_overrides(defaults.runtime, launch_mode=experiment.launch_mode, cuda_devices=experiment.cuda_devices, num_workers=experiment.num_workers)
    dataset_repo_id = _resolve_hf_repo_id(experiment.dataset_name or defaults.dataset_name, defaults.hf_user)
    stage_defaults = _stage_defaults(defaults.guided_stage_config_path)
    xvla_init_path = experiment.xvla_init_path or defaults.xvla_init_path
    decoder_init_path = experiment.decoder_init_path or defaults.decoder_init_path
    fusion_mode = _resolve_fusion_mode(defaults=defaults, experiment=experiment, stage_defaults=stage_defaults)
    normalization_mapping = experiment.normalization_mapping or defaults.normalization_mapping
    validate_mean_std_normalization(normalization_mapping)
    if not xvla_init_path: raise ValueError("xvla_init_path is required.")
    if not decoder_init_path: raise ValueError("decoder_init_path is required.")
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment.name or f"{defaults.name_prefix}_{_slug(dataset_repo_id)}_{timestamp}"
    output_dir = experiment.output_dir or str(TRAIN_OUTPUT_DIR / name)
    return ResolvedGuidedExperiment(
        name=name,
        output_dir=output_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=experiment.dataset_revision or defaults.dataset_revision,
        dataset_root=experiment.dataset_root if experiment.dataset_root is not None else defaults.dataset_root,
        runtime=runtime,
        xvla_init_path=xvla_init_path,
        action_mode=experiment.action_mode if experiment.action_mode is not None else defaults.action_mode,
        decoder_init_path=decoder_init_path,
        decoder_stack_config_path=experiment.decoder_stack_config_path or defaults.decoder_stack_config_path,
        decoder_task_config_path=experiment.decoder_task_config_path or defaults.decoder_task_config_path,
        teacher_image_feature_key=experiment.teacher_image_feature_key or defaults.teacher_image_feature_key,
        dataset_video_backend=experiment.dataset_video_backend or defaults.dataset_video_backend,
        dataset_tolerance_s=experiment.dataset_tolerance_s if experiment.dataset_tolerance_s is not None else defaults.dataset_tolerance_s,
        normalization_mapping=normalization_mapping,
        batch_size=experiment.batch_size if experiment.batch_size is not None else defaults.batch_size,
        gradient_accumulation_steps=experiment.gradient_accumulation_steps if experiment.gradient_accumulation_steps is not None else defaults.gradient_accumulation_steps,
        weight_decay=defaults.weight_decay,
        decoder_optimizer_lr=experiment.decoder_optimizer_lr if experiment.decoder_optimizer_lr is not None else defaults.decoder_optimizer_lr,
        xvla_optimizer_lr=experiment.xvla_optimizer_lr if experiment.xvla_optimizer_lr is not None else defaults.xvla_optimizer_lr,
        xvla_scheduler_decay_lr=experiment.xvla_scheduler_decay_lr if experiment.xvla_scheduler_decay_lr is not None else defaults.xvla_scheduler_decay_lr,
        action_loss_weight=experiment.action_loss_weight if experiment.action_loss_weight is not None else float(stage_defaults.get("action_loss_weight", defaults.action_loss_weight)),
        expert_loss_weight=experiment.expert_loss_weight if experiment.expert_loss_weight is not None else float(stage_defaults.get("expert_loss_weight", defaults.expert_loss_weight)),
        fusion_mode=fusion_mode,
        guidance_train_mode=experiment.guidance_train_mode or str(stage_defaults.get("guidance_train_mode", defaults.guidance_train_mode)),
        guidance_unfreeze_step=experiment.guidance_unfreeze_step if experiment.guidance_unfreeze_step is not None else int(stage_defaults.get("guidance_unfreeze_step", defaults.guidance_unfreeze_step)),
        freeze_xvla_vlm=experiment.freeze_xvla_vlm if experiment.freeze_xvla_vlm is not None else bool(stage_defaults.get("freeze_xvla_vlm", defaults.freeze_xvla_vlm)),
        steps=experiment.steps if experiment.steps is not None else defaults.steps,
        log_every=experiment.log_every if experiment.log_every is not None else defaults.log_every,
        save_every=experiment.save_every if experiment.save_every is not None else defaults.save_every,
        save_final_checkpoint=defaults.save_final_checkpoint,
        resume=experiment.resume if experiment.resume is not None else defaults.resume,
        resume_checkpoint_path=experiment.resume_checkpoint_path if experiment.resume_checkpoint_path is not None else defaults.resume_checkpoint_path,
        wandb_enable=experiment.wandb_enable if experiment.wandb_enable is not None else defaults.wandb_enable,
        wandb_project=experiment.wandb_project or defaults.wandb_project,
        wandb_run_name=experiment.wandb_run_name if experiment.wandb_run_name is not None else defaults.wandb_run_name,
        validation_enable=experiment.validation_enable if experiment.validation_enable is not None else defaults.validation_enable,
        validation_split_ratio=experiment.validation_split_ratio if experiment.validation_split_ratio is not None else defaults.validation_split_ratio,
        validation_freq=experiment.validation_freq if experiment.validation_freq is not None else defaults.validation_freq,
        validation_max_batches=experiment.validation_max_batches if experiment.validation_max_batches is not None else defaults.validation_max_batches,
        validation_seed=experiment.validation_seed if experiment.validation_seed is not None else defaults.validation_seed,
        seed=experiment.seed if experiment.seed is not None else defaults.seed,
    )


def prepare_environment(workspace_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    pythonpath_parts = [str(workspace_dir / "src")]
    for candidate in [workspace_dir / "lerobot" / "src", workspace_dir.parent / "repos" / "lerobot" / "src"]:
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


def run_preflight_checks(experiments: list[ResolvedGuidedExperiment], env: dict[str, str]) -> None:
    importlib.import_module("lerobot")
    if any(experiment.runtime.launch_mode != "single" for experiment in experiments): importlib.import_module("accelerate")
    probe = subprocess.run([sys.executable, "-c", "import thesis_vla"], env=env, capture_output=True, text=True, check=False)
    if probe.returncode != 0: raise SystemExit(f"ERROR: thesis_vla is not importable in subprocess environment.\n{(probe.stderr or probe.stdout).strip()}")


def write_resolved_config(resolved: ResolvedGuidedExperiment) -> Path:
    RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_TMP_DIR / f"{resolved.name}_xvla_guided.json"
    payload = asdict(resolved)
    runtime = payload.pop("runtime")
    payload["device"] = runtime["device"]
    payload["num_workers"] = runtime["num_workers"]
    payload["dry_run"] = runtime["dry_run"]
    payload["cuda_visible_devices"] = list(runtime["cuda_devices"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def build_training_command(resolved: ResolvedGuidedExperiment, config_path: Path) -> list[str]:
    trainer_module = "thesis_vla.training.xvla_guided_trainer"
    trainer_args = [f"--config_path={config_path}"]
    if resolved.runtime.launch_mode == "single": return [sys.executable, "-m", trainer_module, *trainer_args]
    num_processes = len(resolved.runtime.cuda_devices)
    if num_processes < 1: raise ValueError("Accelerate launch mode requires at least one CUDA device.")
    accelerate_cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        f"--num_processes={num_processes}", "--num_machines=1",
        f"--main_process_port={resolved.runtime.main_process_port}",
        f"--mixed_precision={resolved.runtime.mixed_precision}", "--dynamo_backend=no",
    ]
    if num_processes > 1: accelerate_cmd.append("--multi_gpu")
    return [*accelerate_cmd, "--module", trainer_module, *trainer_args]


def apply_runtime_environment(env: dict[str, str], runtime: GuidedRuntimeConfig) -> dict[str, str]:
    updated = dict(env)
    updated["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in runtime.cuda_devices)
    return updated


def print_run_summary(index: int, total: int, resolved: ResolvedGuidedExperiment, cmd: list[str]) -> None:
    print("=" * 88)
    print(f"Launching guided XVLA experiment {index}/{total}")
    print("=" * 88)
    print(f"  Name:               {resolved.name}")
    print(f"  XVLA Init:          {resolved.xvla_init_path}")
    print(f"  Action Mode:        {resolved.action_mode}")
    print(f"  Decoder Init:       {resolved.decoder_init_path}")
    print(f"  Fusion:             {resolved.fusion_mode}")
    print(f"  Guidance Train:     {resolved.guidance_train_mode} @ step {resolved.guidance_unfreeze_step}")
    print(f"  Freeze XVLA VLM:    {resolved.freeze_xvla_vlm}")
    print(f"  Dataset:            {resolved.dataset_repo_id}")
    print(f"  Video Backend:      {resolved.dataset_video_backend}")
    print(f"  Timestamp Tol:      {resolved.dataset_tolerance_s}")
    print(f"  Normalization:      {resolved.normalization_mapping}")
    print(f"  Output Dir:         {resolved.output_dir}")
    print(f"  Launch Mode:        {resolved.runtime.launch_mode}")
    print(f"  CUDA Devices:       {resolved.runtime.cuda_devices}")
    if resolved.runtime.launch_mode != "single":
        print(f"  Mixed Precision:    {resolved.runtime.mixed_precision}")
        print(f"  Main Process Port:  {resolved.runtime.main_process_port}")
    print(f"  Steps:              {resolved.steps}")
    print(f"  Batch Size:         {resolved.batch_size}")
    print(f"  Decoder LR:         {resolved.decoder_optimizer_lr}")
    print(f"  XVLA LR:            {resolved.xvla_optimizer_lr}")
    print(f"  XVLA Sched Decay:   {resolved.xvla_scheduler_decay_lr}")
    print(f"  Action W:           {resolved.action_loss_weight}")
    print(f"  Expert W:           {resolved.expert_loss_weight}")
    print(f"  Resume:             {resolved.resume}")
    if resolved.resume_checkpoint_path is not None: print(f"  Resume Checkpoint:  {resolved.resume_checkpoint_path}")
    print(f"  WandB:              {resolved.wandb_enable}")
    if resolved.wandb_enable: print(f"  WandB Project:      {resolved.wandb_project}")
    print(f"  Validation:         {resolved.validation_enable}")
    if resolved.validation_enable:
        print(f"  Val Split Ratio:    {resolved.validation_split_ratio}")
        print(f"  Val Frequency:      {resolved.validation_freq}")
        print(f"  Val Max Batches:    {resolved.validation_max_batches}")
        print(f"  Val Seed:           {resolved.validation_seed}")
    print(f"  Dry Run:            {resolved.runtime.dry_run}")
    print(f"  Command:            {' '.join(cmd)}")
    print("=" * 88)


def run_experiments(workspace_dir: Path, defaults: GuidedLaunchConfig, experiments: list[GuidedExperimentSpec]) -> None:
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
