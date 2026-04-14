from __future__ import annotations

import datetime as dt
import functools
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable, Literal, TypeAlias

from thesis_vla.common.paths import RUNTIME_CACHE_DIR, TRAIN_OUTPUT_DIR


LaunchMode = Literal["single", "accelerate"]
AugmentationBackend = Literal["custom", "lerobot"]
ModelRef: TypeAlias = str | tuple[str, str]


@dataclass(frozen=True)
class FreezeConfig:
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    launch_mode: LaunchMode = "single"
    cuda_devices: tuple[int, ...] = (1,)
    mixed_precision: str = "bf16"
    main_process_port: int = 45001
    device: str = "cuda"
    num_workers: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class LaunchConfig:
    hf_user: str = os.environ.get("HF_USER", "edgarcancinoe")
    version: str = os.environ.get("VERSION", "v1")
    dataset_name: str = "soarm101_pickplace_10d_7p5hz_resampled"
    dataset_revision: str = os.environ.get("DATASET_REVISION", "v3.0")
    base_model: ModelRef = "lerobot/xvla-base"
    action_mode: str = "so101_ee6d"
    normalization_mapping: str = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
    freeze: FreezeConfig = FreezeConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    optimizer_lr: float = 1e-4
    scheduler_decay_lr: float = 2.5e-6
    steps: int = 12_500
    log_freq: int = 500
    eval_freq: int = -1
    save_freq: int = 8_000
    push_every: int = 8_000
    policy_push_to_hub: bool = True
    wandb_enable: bool = True
    resume: bool = False
    policy_dtype: str = "bfloat16"
    enable_gripper_debug_stats: bool = True
    enable_augmentation: bool = False
    augmentation_degrees: str = "[-2.5, 2.5]"
    augmentation_translate: str = "[0.025, 0.025]"
    augmentation_backend: AugmentationBackend = "custom"
    augmentation_enable_photometric: bool = True
    augmentation_fill_mode: str = "reflect"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str | None = None
    name_suffix: str | None = None
    job_name: str | None = None
    output_dir: str | None = None
    policy_repo_id: str | None = None
    dataset_name: str | None = None
    dataset_revision: str | None = None
    base_model: ModelRef | None = None
    action_mode: str | None = None
    normalization_mapping: str | None = None
    freeze: FreezeConfig | None = None
    freeze_vision_encoder: bool | None = None
    freeze_language_encoder: bool | None = None
    train_policy_transformer: bool | None = None
    train_soft_prompts: bool | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    optimizer_lr: float | None = None
    scheduler_decay_lr: float | None = None
    steps: int | None = None
    log_freq: int | None = None
    eval_freq: int | None = None
    save_freq: int | None = None
    push_every: int | None = None
    num_workers: int | None = None
    launch_mode: LaunchMode | None = None
    cuda_devices: tuple[int, ...] | None = None
    enable_augmentation: bool | None = None
    augmentation_degrees: str | None = None
    augmentation_translate: str | None = None
    augmentation_backend: AugmentationBackend | None = None
    augmentation_enable_photometric: bool | None = None
    augmentation_fill_mode: str | None = None


@dataclass(frozen=True)
class ResolvedExperiment:
    name: str
    job_name: str
    output_dir: str
    policy_repo_id: str
    dataset_repo_id: str
    dataset_revision: str
    base_model: str
    action_mode: str
    normalization_mapping: str
    freeze: FreezeConfig
    runtime: RuntimeConfig
    batch_size: int
    gradient_accumulation_steps: int
    optimizer_lr: float
    scheduler_decay_lr: float
    steps: int
    log_freq: int
    eval_freq: int
    save_freq: int
    push_every: int
    policy_push_to_hub: bool
    wandb_enable: bool
    resume: bool
    policy_dtype: str
    enable_gripper_debug_stats: bool
    enable_augmentation: bool
    augmentation_degrees: str
    augmentation_translate: str
    augmentation_backend: AugmentationBackend
    augmentation_enable_photometric: bool
    augmentation_fill_mode: str


EMPTY_CAMERAS = 1
POLICY_NUM_IMAGE_VIEWS = 3
POLICY_TOKENIZER_MAX_LENGTH = 64
POLICY_MAX_LEN_SEQ = 1024
DATASET_VIDEO_BACKEND = "pyav"
RENAME_MAP = '{"observation.images.main": "observation.images.image", "observation.images.secondary": "observation.images.image2"}'
HF_REPO_ID_MAX_LEN = 96


def resolve_hf_repo_id(repo_or_name: str, hf_user: str) -> str:
    repo_or_name = repo_or_name.strip()
    if "/" in repo_or_name:
        return repo_or_name
    return f"{hf_user}/{repo_or_name}"


def repo_slug(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def bool_str(value: bool) -> str:
    return "true" if value else "false"


@functools.lru_cache(maxsize=1)
def lerobot_supports_gradient_accumulation() -> bool:
    """Best-effort check for gradient accumulation support in lerobot train config."""
    try:
        # Probe config dataclass directly instead of parsing --help output, which can vary.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from lerobot.configs.train import TrainPipelineConfig as C; "
                    "print('ok' if 'gradient_accumulation_steps' in C.__dataclass_fields__ else 'no')"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and "ok" in (result.stdout or "")
    except Exception:
        return False


def resolve_model_ref(model_ref: ModelRef, hf_user: str) -> tuple[str, str | None]:
    """Resolve a model reference into repo_id and optional alias."""
    if isinstance(model_ref, tuple):
        repo_or_name, alias = model_ref
        return resolve_hf_repo_id(repo_or_name, hf_user), alias.strip() if alias else None
    return resolve_hf_repo_id(model_ref, hf_user), None


def sanitize_name_part(part: str) -> str:
    """Normalize naming tokens for concise, Hub-safe policy names."""
    cleaned = part.replace("soarm101_", "").replace("soarm101", "")
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    cleaned = cleaned.strip("_.-")
    return cleaned


def _shorten_repo_name(name: str, max_len: int) -> str:
    """Shorten a Hub repo name while preserving readability and uniqueness."""
    if len(name) <= max_len:
        return name
    # Keep a readable prefix and add a stable hash suffix to avoid collisions.
    suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    keep = max(1, max_len - len(suffix) - 1)
    trimmed = name[:keep].rstrip("-.")
    if not trimmed:
        trimmed = "repo"
    return f"{trimmed}-{suffix}"


def normalize_repo_id(repo_or_name: str, hf_user: str) -> tuple[str, bool]:
    """Return a Hub-safe repo id and whether it had to be shortened."""
    repo_id = resolve_hf_repo_id(repo_or_name, hf_user)
    namespace, name = repo_id.split("/", 1)
    max_name_len = HF_REPO_ID_MAX_LEN - len(namespace) - 1
    safe_name = _shorten_repo_name(name, max_name_len)
    safe_repo_id = f"{namespace}/{safe_name}"
    return safe_repo_id, safe_repo_id != repo_id


def training_mode_suffix(freeze: FreezeConfig) -> str:
    fv = freeze.freeze_vision_encoder
    fl = freeze.freeze_language_encoder
    tp = freeze.train_policy_transformer
    ts = freeze.train_soft_prompts

    if fv and fl and (not tp) and ts:
        return "softprompts"
    if fv and fl and tp and ts:
        return "transformer_softprompts"
    if fv and fl and tp and (not ts):
        return "transformer_only"
    if (not fv) and (not fl) and tp and ts:
        return "full_adapt"
    return f"fv-{int(fv)}_fl-{int(fl)}_tp-{int(tp)}_ts-{int(ts)}"


def get_norm_suffix(mapping_str: str) -> str:
    try:
        mapping = json.loads(mapping_str)
        action_mode = str(mapping.get("ACTION", "ID")).replace("_", "").lower()[:1]
        state_mode = str(mapping.get("STATE", "ID")).replace("_", "").lower()[:1]
        return f"a{action_mode}_s{state_mode}"
    except Exception:
        return "custom"


def _normalize_mode_name(value: str) -> str:
    return str(value).replace("-", "_").strip().upper()


def validate_mean_std_normalization(mapping_str: str) -> None:
    """Enforce ACTION/STATE MEAN_STD normalization for XVLA finetuning."""
    try:
        mapping = json.loads(mapping_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "policy.normalization_mapping must be valid JSON. "
            f"Received: {mapping_str!r}"
        ) from exc

    action_mode = _normalize_mode_name(mapping.get("ACTION", ""))
    state_mode = _normalize_mode_name(mapping.get("STATE", ""))
    if action_mode != "MEAN_STD" or state_mode != "MEAN_STD":
        raise ValueError(
            "XVLA finetuning requires mean/std normalization for ACTION and STATE. "
            f"Received ACTION={action_mode or '<missing>'}, STATE={state_mode or '<missing>'}."
        )


def with_overrides(base, **overrides):
    valid = {field.name for field in fields(base)}
    payload = {key: value for key, value in overrides.items() if key in valid and value is not None}
    return replace(base, **payload)


def merge_defaults(defaults: LaunchConfig, experiment: ExperimentSpec) -> tuple[FreezeConfig, RuntimeConfig]:
    freeze_base = experiment.freeze if experiment.freeze is not None else defaults.freeze
    freeze = with_overrides(
        freeze_base,
        freeze_vision_encoder=experiment.freeze_vision_encoder,
        freeze_language_encoder=experiment.freeze_language_encoder,
        train_policy_transformer=experiment.train_policy_transformer,
        train_soft_prompts=experiment.train_soft_prompts,
    )
    runtime = with_overrides(
        defaults.runtime,
        launch_mode=experiment.launch_mode,
        cuda_devices=experiment.cuda_devices,
        num_workers=experiment.num_workers,
    )
    return freeze, runtime


def default_policy_name(
    version: str,
    base_model_repo_id: str,
    base_model_alias: str | None,
    dataset_repo_id: str,
    action_mode: str,
    normalization_mapping: str,
    freeze: FreezeConfig,
    enable_augmentation: bool,
    name_suffix: str | None,
) -> str:
    base_part = base_model_alias or repo_slug(base_model_repo_id)
    parts = [
        sanitize_name_part(base_part),
        sanitize_name_part(repo_slug(dataset_repo_id)),
        sanitize_name_part(action_mode),
        sanitize_name_part(get_norm_suffix(normalization_mapping)),
        sanitize_name_part(training_mode_suffix(freeze)),
    ]
    if enable_augmentation:
        parts.append("aug")
    if name_suffix:
        parts.append(sanitize_name_part(name_suffix))
    parts.append(sanitize_name_part(version))
    return "_".join(part for part in parts if part)


def resolve_experiment(
    workspace_dir: Path,
    defaults: LaunchConfig,
    experiment: ExperimentSpec,
    timestamp: str | None = None,
) -> ResolvedExperiment:
    freeze, runtime = merge_defaults(defaults, experiment)

    dataset_name = experiment.dataset_name or defaults.dataset_name
    dataset_revision = experiment.dataset_revision or defaults.dataset_revision
    base_model_ref = experiment.base_model or defaults.base_model
    base_model_repo_id, base_model_alias = resolve_model_ref(base_model_ref, defaults.hf_user)
    action_mode = experiment.action_mode or defaults.action_mode
    normalization_mapping = experiment.normalization_mapping or defaults.normalization_mapping
    validate_mean_std_normalization(normalization_mapping)
    enable_augmentation = (
        experiment.enable_augmentation
        if experiment.enable_augmentation is not None
        else defaults.enable_augmentation
    )
    augmentation_degrees = experiment.augmentation_degrees or defaults.augmentation_degrees
    augmentation_translate = experiment.augmentation_translate or defaults.augmentation_translate
    augmentation_backend = experiment.augmentation_backend or defaults.augmentation_backend
    augmentation_enable_photometric = (
        experiment.augmentation_enable_photometric
        if experiment.augmentation_enable_photometric is not None
        else defaults.augmentation_enable_photometric
    )
    augmentation_fill_mode = experiment.augmentation_fill_mode or defaults.augmentation_fill_mode
    dataset_repo_id = resolve_hf_repo_id(dataset_name, defaults.hf_user)
    run_timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    generated_name = experiment.name or default_policy_name(
        version=defaults.version,
        base_model_repo_id=base_model_repo_id,
        base_model_alias=base_model_alias,
        dataset_repo_id=dataset_repo_id,
        action_mode=action_mode,
        normalization_mapping=normalization_mapping,
        freeze=freeze,
        enable_augmentation=enable_augmentation,
        name_suffix=experiment.name_suffix,
    )
    job_name = experiment.job_name or f"{generated_name}_{run_timestamp}"
    output_dir = experiment.output_dir or str(TRAIN_OUTPUT_DIR / f"{generated_name}_{run_timestamp}")
    requested_policy_repo = experiment.policy_repo_id or f"{defaults.hf_user}/{generated_name}"
    policy_repo_id, was_shortened = normalize_repo_id(requested_policy_repo, defaults.hf_user)
    if was_shortened:
        print(
            "WARNING: policy.repo_id exceeded Hugging Face length constraints and was shortened to: "
            f"{policy_repo_id}"
        )

    return ResolvedExperiment(
        name=generated_name,
        job_name=job_name,
        output_dir=output_dir,
        policy_repo_id=policy_repo_id,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        base_model=base_model_repo_id,
        action_mode=action_mode,
        normalization_mapping=normalization_mapping,
        freeze=freeze,
        runtime=runtime,
        batch_size=experiment.batch_size if experiment.batch_size is not None else defaults.batch_size,
        gradient_accumulation_steps=(
            experiment.gradient_accumulation_steps
            if experiment.gradient_accumulation_steps is not None
            else defaults.gradient_accumulation_steps
        ),
        optimizer_lr=experiment.optimizer_lr if experiment.optimizer_lr is not None else defaults.optimizer_lr,
        scheduler_decay_lr=(
            experiment.scheduler_decay_lr
            if experiment.scheduler_decay_lr is not None
            else defaults.scheduler_decay_lr
        ),
        steps=experiment.steps if experiment.steps is not None else defaults.steps,
        log_freq=experiment.log_freq if experiment.log_freq is not None else defaults.log_freq,
        eval_freq=experiment.eval_freq if experiment.eval_freq is not None else defaults.eval_freq,
        save_freq=experiment.save_freq if experiment.save_freq is not None else defaults.save_freq,
        push_every=experiment.push_every if experiment.push_every is not None else defaults.push_every,
        policy_push_to_hub=defaults.policy_push_to_hub,
        wandb_enable=defaults.wandb_enable,
        resume=defaults.resume,
        policy_dtype=defaults.policy_dtype,
        enable_gripper_debug_stats=defaults.enable_gripper_debug_stats,
        enable_augmentation=enable_augmentation,
        augmentation_degrees=augmentation_degrees,
        augmentation_translate=augmentation_translate,
        augmentation_backend=augmentation_backend,
        augmentation_enable_photometric=augmentation_enable_photometric,
        augmentation_fill_mode=augmentation_fill_mode,
    )


def prepare_environment(workspace_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    pythonpath_parts: list[str] = []
    workspace_src = workspace_dir / "src"
    if workspace_src.exists():
        pythonpath_parts.append(str(workspace_src))
    lerobot_src_candidates = [
        workspace_dir / "lerobot" / "src",             # monorepo layout on cluster
        workspace_dir.parent / "repos" / "lerobot" / "src",  # split repos layout
    ]
    for local_lerobot_src in lerobot_src_candidates:
        if local_lerobot_src.exists():
            pythonpath_parts.append(str(local_lerobot_src))
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    if pythonpath_parts:
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


def run_preflight_checks(defaults: LaunchConfig, experiments: Iterable[ExperimentSpec], env: dict[str, str]) -> None:
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if not active_env.startswith("vla"):
        print(
            "WARNING: expected a VLA conda environment (e.g. 'vla' or 'vla2'). "
            f"Current env: '{active_env or 'none'}'."
        )

    try:
        importlib.import_module("lerobot")
    except ImportError as exc:
        raise SystemExit(
            "ERROR: lerobot Python package not found.\n"
            "Please install lerobot first or ensure the correct environment is active."
        ) from exc

    # Hard-fail early if thesis_vla is not importable in the subprocess environment.
    probe = subprocess.run(
        [sys.executable, "-c", "import thesis_vla"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "").strip()
        raise SystemExit(
            "ERROR: thesis_vla Python package is not importable in the training subprocess environment.\n"
            "Please verify PYTHONPATH includes your workspace src directory.\n"
            f"Details: {details}"
        )

    requires_accelerate = any((experiment.launch_mode or defaults.runtime.launch_mode) == "accelerate" for experiment in experiments)
    if requires_accelerate:
        try:
            importlib.import_module("accelerate")
        except ImportError as exc:
            raise SystemExit(
                "ERROR: accelerate package not found.\n"
                "Install it in the active environment with: python -m pip install accelerate"
            ) from exc


def augmentation_transforms(experiment: ResolvedExperiment) -> str:
    return (
        f'{{"affine": {{"type": "RandomAffine", "kwargs": {{"degrees": {experiment.augmentation_degrees},'
        f' "translate": {experiment.augmentation_translate}}}}}}}'
    )


def build_training_command(experiment: ResolvedExperiment) -> list[str]:
    base_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={experiment.base_model}",
        f"--policy.repo_id={experiment.policy_repo_id}",
        f"--policy.push_to_hub={bool_str(experiment.policy_push_to_hub)}",
        f"--dataset.repo_id={experiment.dataset_repo_id}",
        f"--dataset.revision={experiment.dataset_revision}",
        f"--dataset.video_backend={DATASET_VIDEO_BACKEND}",
        f"--rename_map={RENAME_MAP}",
        f"--dataset.image_transforms.enable={bool_str(experiment.enable_augmentation)}",
        f"--dataset.image_transforms.tfs={augmentation_transforms(experiment)}",
        f"--batch_size={experiment.batch_size}",
        f"--policy.optimizer_lr={experiment.optimizer_lr}",
        f"--policy.scheduler_decay_lr={experiment.scheduler_decay_lr}",
        f"--steps={experiment.steps}",
        f"--log_freq={experiment.log_freq}",
        f"--eval_freq={experiment.eval_freq}",
        f"--save_freq={experiment.save_freq}",
        f"--push_every={experiment.push_every}",
        f"--output_dir={experiment.output_dir}",
        f"--job_name={experiment.job_name}",
        f"--policy.device={experiment.runtime.device}",
        f"--wandb.enable={bool_str(experiment.wandb_enable)}",
        f"--num_workers={experiment.runtime.num_workers}",
        f"--resume={bool_str(experiment.resume)}",
        f"--policy.dtype={experiment.policy_dtype}",
        f"--policy.action_mode={experiment.action_mode}",
        f"--policy.empty_cameras={EMPTY_CAMERAS}",
        f"--policy.freeze_vision_encoder={bool_str(experiment.freeze.freeze_vision_encoder)}",
        f"--policy.freeze_language_encoder={bool_str(experiment.freeze.freeze_language_encoder)}",
        f"--policy.train_policy_transformer={bool_str(experiment.freeze.train_policy_transformer)}",
        f"--policy.train_soft_prompts={bool_str(experiment.freeze.train_soft_prompts)}",
        f"--policy.num_image_views={POLICY_NUM_IMAGE_VIEWS}",
        f"--policy.tokenizer_max_length={POLICY_TOKENIZER_MAX_LENGTH}",
        f"--policy.max_len_seq={POLICY_MAX_LEN_SEQ}",
        f"--policy.normalization_mapping={experiment.normalization_mapping}",
        f"--policy.enable_gripper_debug_stats={bool_str(experiment.enable_gripper_debug_stats)}",
        f"--gradient_accumulation_steps={experiment.gradient_accumulation_steps}",
    ]

    if experiment.runtime.launch_mode == "single":
        return base_cmd

    num_processes = len(experiment.runtime.cuda_devices)
    if num_processes < 1:
        raise ValueError("Accelerate launch mode requires at least one CUDA device.")

    accelerate_cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--num_processes={num_processes}",
        "--num_machines=1",
        f"--main_process_port={experiment.runtime.main_process_port}",
        f"--mixed_precision={experiment.runtime.mixed_precision}",
        "--dynamo_backend=no",
    ]
    if num_processes > 1:
        accelerate_cmd.append("--multi_gpu")
    accelerate_cmd += ["--module", "lerobot.scripts.lerobot_train"]
    accelerate_cmd += base_cmd[3:]
    return accelerate_cmd


def apply_runtime_environment(env: dict[str, str], runtime: RuntimeConfig) -> dict[str, str]:
    updated = dict(env)
    updated["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in runtime.cuda_devices)
    return updated


def apply_experiment_environment(env: dict[str, str], experiment: ResolvedExperiment) -> dict[str, str]:
    updated = dict(env)
    updated["THESIS_AUGMENTATION_BACKEND"] = experiment.augmentation_backend
    updated["THESIS_AUG_ENABLE_PHOTOMETRIC"] = bool_str(experiment.augmentation_enable_photometric)
    updated["THESIS_AUG_FILL_MODE"] = experiment.augmentation_fill_mode
    return updated


def print_run_summary(index: int, total: int, experiment: ResolvedExperiment, cmd: list[str]) -> None:
    num_processes = len(experiment.runtime.cuda_devices)
    print("\n" + "=" * 88)
    print(f"Launching experiment {index}/{total}")
    print("=" * 88)
    print(f"  Name:               {experiment.name}")
    print(f"  Launch Mode:        {experiment.runtime.launch_mode}")
    print(f"  Base Model:         {experiment.base_model}")
    print(f"  Dataset:            {experiment.dataset_repo_id}")
    print(f"  Action Mode:        {experiment.action_mode}")
    print(f"  Freeze Vision:      {experiment.freeze.freeze_vision_encoder}")
    print(f"  Freeze Language:    {experiment.freeze.freeze_language_encoder}")
    print(f"  Train Transformer:  {experiment.freeze.train_policy_transformer}")
    print(f"  Train Soft Prompts: {experiment.freeze.train_soft_prompts}")
    print(f"  Batch Size:         {experiment.batch_size}")
    print(f"  Grad Accum Steps:   {experiment.gradient_accumulation_steps}")
    print(f"  Steps:              {experiment.steps}")
    print(f"  Augmentation:       {experiment.enable_augmentation}")
    print(f"  Aug Backend:        {experiment.augmentation_backend}")
    print(f"  Aug Photometric:    {experiment.augmentation_enable_photometric}")
    print(f"  Aug Fill Mode:      {experiment.augmentation_fill_mode}")
    print(f"  Output Dir:         {experiment.output_dir}")
    print(f"  Policy Repo ID:     {experiment.policy_repo_id}")
    print(f"  CUDA Devices:       {experiment.runtime.cuda_devices}")
    if experiment.runtime.launch_mode == "accelerate":
        print(f"  Num Processes:      {num_processes}")
        effective = experiment.batch_size * num_processes * experiment.gradient_accumulation_steps
        print(
            "  Effective Batch:    "
            f"{experiment.batch_size} x {num_processes} x {experiment.gradient_accumulation_steps} = {effective}"
        )
    print(f"  Dry Run:            {experiment.runtime.dry_run}")
    print("  Command:")
    print(f"    {' '.join(cmd)}")
    print("=" * 88)


def run_experiments(
    workspace_dir: Path,
    defaults: LaunchConfig,
    experiments: list[ExperimentSpec],
) -> None:
    env = prepare_environment(workspace_dir)
    run_preflight_checks(defaults, experiments, env)

    if not experiments:
        print("No experiments configured. Nothing to launch.")
        return

    print("=" * 88)
    print(f"Starting {len(experiments)} XVLA experiment(s)")
    print("=" * 88)

    for index, experiment in enumerate(experiments, start=1):
        resolved = resolve_experiment(workspace_dir=workspace_dir, defaults=defaults, experiment=experiment)
        cmd = build_training_command(resolved)
        runtime_env = apply_runtime_environment(env, resolved.runtime)
        runtime_env = apply_experiment_environment(runtime_env, resolved)
        print_run_summary(index=index, total=len(experiments), experiment=resolved, cmd=cmd)

        if resolved.runtime.dry_run:
            print(f"Dry run enabled for '{resolved.name}', skipping process launch.")
            continue

        try:
            exit_code = subprocess.call(cmd, env=runtime_env)
        except KeyboardInterrupt:
            raise SystemExit(130)
        except Exception as exc:
            print(f"Error launching training: {exc}")
            exit_code = 1

        if exit_code != 0:
            print(f"Training failed for '{resolved.name}' with exit code {exit_code}.")
            print("Stopping remaining experiments.")
            raise SystemExit(exit_code)

        print(f"Completed run '{resolved.name}'.")

    print("\n" + "=" * 88)
    print("All requested XVLA experiments completed successfully.")
    print("=" * 88)
