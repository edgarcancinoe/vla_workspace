from __future__ import annotations

import datetime as dt
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from thesis_vla.common.paths import RUNTIME_CACHE_DIR, TRAIN_OUTPUT_DIR


RENAME_MAP = '{"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}'
AugmentationBackend = Literal["custom", "lerobot"]


@dataclass(frozen=True)
class SmolVLAConfig:
    hf_user: str = os.environ.get("HF_USER", "edgarcancinoe")
    dataset_name: str = "soarm101_pickplace_front"
    base_policy_path: str = "lerobot/smolvla_base"
    policy_name: str = "smolvla_finetuned_orange_50ep_open_gripper"
    batch_size: int = 64
    scheduler_decay_lr: float = 1e-5
    steps: int = 25_000
    log_freq: int = 100
    eval_freq: int = -1
    save_freq: int = 10_000
    device: str = "cuda"
    cuda_device: str = "0"
    num_workers: int = 4
    enable_augmentation: bool = False
    augmentation_degrees: str = "[-2, 2]"
    augmentation_translate: str = "[-0.015, 0.015]"
    augmentation_backend: AugmentationBackend = "custom"
    augmentation_enable_photometric: bool = True
    augmentation_fill_mode: str = "reflect"
    resume: bool = False
    policy_push_to_hub: bool = True
    wandb_enable: bool = True
    dry_run: bool = False
    job_name: str | None = None
    output_dir: str | None = None
    policy_repo_id: str | None = None


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def prepare_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    user = env.get("USER", "default_user")
    cache_root = RUNTIME_CACHE_DIR / f"smolvla_{user}"
    os.makedirs(cache_root, exist_ok=True)

    env["HF_HOME"] = str(cache_root)
    env["HF_LEROBOT_HOME"] = str(cache_root / "lerobot")
    os.makedirs(env["HF_LEROBOT_HOME"], exist_ok=True)

    return env


def run_preflight_checks(config: SmolVLAConfig) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "vla":
        print("WARNING: 'vla' conda environment is not activated.")

    try:
        importlib.import_module("lerobot")
        print("✓ lerobot package found")
    except ImportError as exc:
        raise SystemExit(
            "ERROR: lerobot Python package not found.\n"
            "Please install lerobot first or ensure the correct environment is active."
        ) from exc

    if config.wandb_enable:
        try:
            importlib.import_module("wandb")
        except ImportError:
            print("WARNING: wandb package not found.")
            print("Install with: pip install wandb")


def resolve_launch(config: SmolVLAConfig, workspace_dir: Path) -> tuple[str, str, str, str]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_repo_id = f"{config.hf_user}/{config.dataset_name}"
    job_name = config.job_name or f"{config.policy_name}_{timestamp}"
    output_dir = config.output_dir or str(TRAIN_OUTPUT_DIR / f"{config.policy_name}_{timestamp}")
    policy_repo_id = config.policy_repo_id or f"{config.hf_user}/{config.policy_name}"
    return dataset_repo_id, job_name, output_dir, policy_repo_id


def augmentation_transforms(config: SmolVLAConfig) -> str:
    return (
        f'{{"affine": {{"type": "RandomAffine", "kwargs": {{"degrees": {config.augmentation_degrees},'
        f' "translate": {config.augmentation_translate}}}}}}}'
    )


def build_command(config: SmolVLAConfig, workspace_dir: Path) -> tuple[list[str], dict[str, str], dict[str, str]]:
    dataset_repo_id, job_name, output_dir, policy_repo_id = resolve_launch(config, workspace_dir)
    env = prepare_environment()

    if config.cuda_device:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    env["THESIS_AUGMENTATION_BACKEND"] = config.augmentation_backend
    env["THESIS_AUG_ENABLE_PHOTOMETRIC"] = bool_str(config.augmentation_enable_photometric)
    env["THESIS_AUG_FILL_MODE"] = config.augmentation_fill_mode

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={config.base_policy_path}",
        f"--policy.repo_id={policy_repo_id}",
        f"--policy.push_to_hub={bool_str(config.policy_push_to_hub)}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--rename_map={RENAME_MAP}",
        f"--dataset.image_transforms.enable={bool_str(config.enable_augmentation)}",
        f"--dataset.image_transforms.tfs={augmentation_transforms(config)}",
        f"--batch_size={config.batch_size}",
        f"--policy.scheduler_decay_lr={config.scheduler_decay_lr}",
        f"--steps={config.steps}",
        f"--log_freq={config.log_freq}",
        f"--eval_freq={config.eval_freq}",
        f"--save_freq={config.save_freq}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={config.device}",
        f"--wandb.enable={bool_str(config.wandb_enable)}",
        f"--num_workers={config.num_workers}",
        f"--resume={bool_str(config.resume)}",
    ]

    summary = {
        "dataset_repo_id": dataset_repo_id,
        "job_name": job_name,
        "output_dir": output_dir,
        "policy_repo_id": policy_repo_id,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
    }
    return cmd, env, summary


def print_summary(config: SmolVLAConfig, summary: dict[str, str], cmd: list[str]) -> None:
    print("=" * 76)
    print("SmolVLA Finetuning Launch Script")
    print("=" * 76)
    print(f"  Dataset:              {summary['dataset_repo_id']}")
    print(f"  Batch Size:           {config.batch_size}")
    print(f"  Steps:                {config.steps}")
    print(f"  Scheduler Decay LR:   {config.scheduler_decay_lr}")
    print(f"  Save Freq:            {config.save_freq}")
    print(f"  Output Dir:           {summary['output_dir']}")
    print(f"  Job Name:             {summary['job_name']}")
    print(f"  Device:               {config.device}")
    print(f"  CUDA_VISIBLE_DEVICES: {summary['cuda_visible_devices']}")
    print(f"  W&B Enabled:          {config.wandb_enable}")
    print(f"  Augmentation:         {config.enable_augmentation}")
    print(f"  Aug Backend:          {config.augmentation_backend}")
    print(f"  Aug Photometric:      {config.augmentation_enable_photometric}")
    print(f"  Aug Fill Mode:        {config.augmentation_fill_mode}")
    print(f"  Base Policy:          {config.base_policy_path}")
    print(f"  Policy Repo ID:       {summary['policy_repo_id']}")
    print(f"  Push to Hub:          {config.policy_push_to_hub}")
    print(f"  Dry Run:              {config.dry_run}")
    print("  Command:")
    print(f"    {' '.join(cmd)}")
    print("=" * 76)


def run_training(config: SmolVLAConfig, workspace_dir: Path) -> None:
    run_preflight_checks(config)
    cmd, env, summary = build_command(config, workspace_dir)
    print_summary(config, summary, cmd)

    if config.dry_run:
        print("Dry run enabled, skipping process launch.")
        return

    exit_code = subprocess.call(cmd, env=env)
    if exit_code != 0:
        raise SystemExit(exit_code)
