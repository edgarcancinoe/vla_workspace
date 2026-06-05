#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
workspace_src = str(ROOT_DIR / "src")
if workspace_src in sys.path: sys.path.remove(workspace_src)
sys.path.insert(0, workspace_src)

lerobot_src_candidates = [ROOT_DIR / "lerobot" / "src", ROOT_DIR.parent / "repos" / "lerobot" / "src"]
for lerobot_src in reversed([str(p) for p in lerobot_src_candidates if p.exists()]):
    if lerobot_src in sys.path: sys.path.remove(lerobot_src)
    sys.path.insert(0, lerobot_src)
    
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.xvla.action_contract import build_slice_map, get_so101_slice_spec, slice_dataset_meta_in_place
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.slice_processor import SliceProcessorStep
from lerobot.scripts.lerobot_train import (
    GRIPPER_DEBUG_COUNT_KEYS,
    _compute_policy_loss,
    _enforce_xvla_finetune_contract,
    _make_xvla_validation_corruption,
    _patch_xvla_gripper_stats_for_overrides,
    _rebuild_xvla_visual_input_features,
    build_dataset_frame_indices,
    make_dataset_with_episodes,
    split_train_validation_episodes,
    FixedIndexSampler,
)
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device


CHECKPOINTS: list[str | dict[str, str]] = [
    # "/abs/path/to/run/checkpoints/010000/pretrained_model",
    # {"name": "run_010000", "checkpoint": "/abs/path/to/run/checkpoints/010000/pretrained_model"},
    "edgarcancinoe/orange196_pickplace-multicolor_7p5hz_so101_ee6d_am_sm_b8_ga2_eb32_full_ad-02a199f2-step-30000",
]

DEVICE = "cuda"
BATCH_SIZE = None
NUM_WORKERS = None
MAX_BATCHES = None
OUTPUT_JSON = ""

import torch


class _EvalAccelerator:
    def __init__(self, device: torch.device):
        self.device = device

    def autocast(self):
        return nullcontext()

    def unwrap_model(self, model, keep_fp32_wrapper: bool = False):
        return model


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    name: str
    checkpoint: str


def resolve_pretrained_ref(checkpoint: str) -> tuple[str, Path | None]:
    candidate = Path(checkpoint).expanduser()
    if not candidate.exists():
        return checkpoint, None
    resolved = candidate.resolve()
    if resolved.is_file():
        if resolved.name != TRAIN_CONFIG_NAME:
            raise ValueError(f"Unsupported checkpoint file: {resolved}")
        return str(resolved.parent), resolved.parent
    if (resolved / TRAIN_CONFIG_NAME).exists():
        return str(resolved), resolved
    pretrained_dir = resolved / PRETRAINED_MODEL_DIR
    if (pretrained_dir / TRAIN_CONFIG_NAME).exists():
        return str(pretrained_dir), pretrained_dir
    raise ValueError(f"Could not locate {TRAIN_CONFIG_NAME} under {resolved}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return auto_select_torch_device()
    return get_safe_torch_device(device_arg, log=True)


def normalize_eval_specs(raw_specs: list[str | dict[str, str]]) -> list[EvalSpec]:
    normalized_specs: list[EvalSpec] = []
    for index, raw_spec in enumerate(raw_specs, start=1):
        if isinstance(raw_spec, str):
            checkpoint = raw_spec.strip()
            if not checkpoint:
                continue
            normalized_specs.append(EvalSpec(name=Path(checkpoint).name or f"checkpoint_{index}", checkpoint=checkpoint))
            continue
        if isinstance(raw_spec, dict):
            checkpoint = raw_spec.get("checkpoint", "").strip()
            if not checkpoint:
                raise ValueError(f"CHECKPOINTS[{index - 1}] is missing a non-empty 'checkpoint' value.")
            name = raw_spec.get("name", "").strip() or Path(checkpoint).name or f"checkpoint_{index}"
            normalized_specs.append(EvalSpec(name=name, checkpoint=checkpoint))
            continue
        raise TypeError(f"CHECKPOINTS[{index - 1}] must be a string path/repo id or a dict with 'checkpoint' and optional 'name'.")
    if not normalized_specs:
        raise ValueError("CHECKPOINTS is empty. Populate it in this file before running the script.")
    return normalized_specs


def to_scalar_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    elif not isinstance(value, (int, float)):
        return None
    return float(value)


def make_eval_dataloader(dataset, policy_cfg, batch_size: int, num_workers: int, device_type: str):
    sampler = None
    if hasattr(policy_cfg, "drop_n_last_frames"):
        sampler = FixedIndexSampler(build_dataset_frame_indices(dataset, drop_n_last_frames=policy_cfg.drop_n_last_frames), shuffle=False)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=device_type == "cuda", drop_last=False, prefetch_factor=2 if num_workers > 0 else None)


def ensure_xvla_slice_step(preprocessor, slice_spec) -> None:
    if slice_spec is None:
        return
    slice_map = build_slice_map(slice_spec)
    has_slice_step = any(isinstance(step, SliceProcessorStep) and getattr(step, "slice_map", None) == slice_map for step in preprocessor.steps)
    if has_slice_step:
        return
    insert_idx = len(preprocessor.steps)
    for idx, step in enumerate(preprocessor.steps):
        if isinstance(step, NormalizerProcessorStep):
            insert_idx = idx
            break
    preprocessor.steps.insert(insert_idx, SliceProcessorStep(slice_map=slice_map))


def evaluate_split(name: str, policy, dataloader, preprocessor, validation_seed: int, max_batches: int | None) -> dict[str, float]:
    fake_accelerator = _EvalAccelerator(next(policy.parameters()).device)
    was_training = policy.training
    policy.eval()
    local_loss_sum = 0.0
    local_sample_count = 0.0
    local_batch_count = 0.0
    local_component_sums: dict[str, float] = {}
    local_component_counts: dict[str, float] = {}
    local_gripper_counts = {key: 0.0 for key in GRIPPER_DEBUG_COUNT_KEYS}
    base_policy = policy
    try:
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = preprocessor(raw_batch)
                deterministic_validation_corruption = _make_xvla_validation_corruption(base_policy, batch, batch_index, validation_seed)
                loss, output_dict, batch_size = _compute_policy_loss(policy, batch, fake_accelerator, deterministic_validation_corruption=deterministic_validation_corruption)
                local_loss_sum += float(loss.detach().item()) * batch_size
                local_sample_count += batch_size
                local_batch_count += 1
                for key, value in output_dict.items():
                    if not key.endswith("_loss"):
                        continue
                    scalar_value = to_scalar_float(value)
                    if scalar_value is None:
                        continue
                    local_component_sums[key] = local_component_sums.get(key, 0.0) + scalar_value * batch_size
                    local_component_counts[key] = local_component_counts.get(key, 0.0) + batch_size
                gripper_debug_counts = output_dict.get("gripper_debug_counts")
                if gripper_debug_counts is not None:
                    for key in GRIPPER_DEBUG_COUNT_KEYS:
                        value = gripper_debug_counts[key]
                        if isinstance(value, torch.Tensor):
                            value = value.detach().item()
                        local_gripper_counts[key] += float(value)
    finally:
        policy.train(was_training)
    if local_batch_count < 1 or local_sample_count <= 0:
        raise RuntimeError(f"{name} split evaluation ran with zero batches.")
    metrics = {"loss": local_loss_sum / local_sample_count, "num_batches": float(local_batch_count), "num_samples": float(local_sample_count)}
    for key, component_sum in local_component_sums.items():
        denom = local_component_counts[key]
        if denom > 0:
            metrics[key] = component_sum / denom
    total = local_gripper_counts["true_negative_count"] + local_gripper_counts["true_positive_count"] + local_gripper_counts["false_positive_count"] + local_gripper_counts["false_negative_count"]
    if total > 0:
        class0_total = local_gripper_counts["target_zero_count"]
        class1_total = local_gripper_counts["target_one_count"]
        metrics["gripper_accuracy"] = (local_gripper_counts["true_negative_count"] + local_gripper_counts["true_positive_count"]) / total
        metrics["gripper_class0_accuracy"] = local_gripper_counts["true_negative_count"] / class0_total if class0_total > 0 else 0.0
        metrics["gripper_class1_accuracy"] = local_gripper_counts["true_positive_count"] / class1_total if class1_total > 0 else 0.0
    return metrics


def ordered_metric_items(metrics: dict[str, float]) -> list[tuple[str, float]]:
    preferred = ["loss", "position_loss", "rotate6D_loss", "joints_loss", "gripper_loss", "gripper_accuracy", "gripper_class0_accuracy", "gripper_class1_accuracy", "num_batches", "num_samples"]
    keys = [key for key in preferred if key in metrics]
    keys += [key for key in sorted(metrics) if key not in keys]
    return [(key, metrics[key]) for key in keys]


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}:")
    for key, value in ordered_metric_items(metrics):
        if key in {"num_batches", "num_samples"}:
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.6f}")


def evaluate_checkpoint(spec: EvalSpec, device: torch.device, batch_size_override: int | None, num_workers_override: int | None, max_batches: int | None) -> dict[str, Any]:
    pretrained_ref, local_pretrained_dir = resolve_pretrained_ref(spec.checkpoint)
    cfg = TrainPipelineConfig.from_pretrained(pretrained_ref)
    if not cfg.validation.enable:
        raise ValueError(f"{spec.name}: saved training config has validation disabled, so there is no training-time split to reproduce.")
    cfg.policy.pretrained_path = local_pretrained_dir if local_pretrained_dir is not None else pretrained_ref
    cfg.policy.device = str(device)
    cfg.dataset = dataclasses.replace(cfg.dataset, image_transforms=dataclasses.replace(cfg.dataset.image_transforms, enable=False))
    eval_batch_size = batch_size_override if batch_size_override is not None else cfg.batch_size
    eval_num_workers = num_workers_override if num_workers_override is not None else cfg.num_workers
    base_dataset = make_dataset(cfg)
    train_episodes, val_episodes = split_train_validation_episodes(base_dataset, cfg.validation.split_ratio, cfg.validation.seed)
    train_dataset = make_dataset_with_episodes(cfg, train_episodes, disable_augmentation=True)
    val_dataset = make_dataset_with_episodes(cfg, val_episodes, disable_augmentation=True)
    xvla_slice_spec = get_so101_slice_spec(getattr(cfg.policy, "action_mode", "")) if cfg.policy.type == "xvla" else None
    if xvla_slice_spec is not None:
        slice_dataset_meta_in_place(train_dataset.meta, xvla_slice_spec)
        slice_dataset_meta_in_place(val_dataset.meta, xvla_slice_spec)
    features = dataset_to_policy_features(train_dataset.meta.features)
    cfg.policy.output_features = {key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION}
    if not cfg.policy.input_features:
        cfg.policy.input_features = {key: feature for key, feature in features.items() if key not in cfg.policy.output_features}
    if cfg.policy.type == "xvla":
        _enforce_xvla_finetune_contract(cfg.policy)
        _rebuild_xvla_visual_input_features(cfg.policy, train_dataset.meta, cfg.rename_map)
    if cfg.policy.type != "xvla":
        raise ValueError(f"Expected an XVLA checkpoint, got policy type {cfg.policy.type!r}")
    policy = XVLAPolicy.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.config.normalization_mapping = cfg.policy.normalization_mapping
    if hasattr(policy.config, "enable_gripper_debug_stats"):
        policy.config.enable_gripper_debug_stats = True
    processor_stats = _patch_xvla_gripper_stats_for_overrides(cfg.policy, train_dataset.meta.stats)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.policy.pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
            "normalizer_processor": {"stats": processor_stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping},
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    ensure_xvla_slice_step(preprocessor, xvla_slice_spec)
    train_dataloader = make_eval_dataloader(train_dataset, cfg.policy, eval_batch_size, eval_num_workers, device.type)
    val_dataloader = make_eval_dataloader(val_dataset, cfg.policy, eval_batch_size, eval_num_workers, device.type)
    train_metrics = evaluate_split("train", policy, train_dataloader, preprocessor, cfg.validation.seed, max_batches)
    val_metrics = evaluate_split("validation", policy, val_dataloader, preprocessor, cfg.validation.seed, max_batches)
    return {
        "name": spec.name,
        "checkpoint": pretrained_ref,
        "device": str(device),
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_revision": cfg.dataset.revision,
        "validation_split_ratio": cfg.validation.split_ratio,
        "validation_seed": cfg.validation.seed,
        "eval_batch_size": eval_batch_size,
        "eval_num_workers": eval_num_workers,
        "max_batches": max_batches,
        "train_episode_count": len(train_episodes),
        "val_episode_count": len(val_episodes),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }


def print_payload(payload: dict[str, Any], index: int, total: int) -> None:
    print(f"\n=== [{index}/{total}] {payload['name']} ===")
    print(f"Checkpoint: {payload['checkpoint']}")
    print(f"Dataset: {payload['dataset_repo_id']} @ {payload['dataset_revision']}")
    print(f"Device: {payload['device']}")
    print(f"Split: ratio={payload['validation_split_ratio']} seed={payload['validation_seed']} train_episodes={payload['train_episode_count']} val_episodes={payload['val_episode_count']}")
    max_batches = payload["max_batches"]
    print(f"Evaluation: batch_size={payload['eval_batch_size']} num_workers={payload['eval_num_workers']} max_batches={'all' if max_batches is None else max_batches}")
    print_metric_block("Train Split Metrics", payload["train_metrics"])
    print_metric_block("Validation Split Metrics", payload["validation_metrics"])


def main() -> None:
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    eval_specs = normalize_eval_specs(CHECKPOINTS)
    device = resolve_device(DEVICE)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = device.type == "cuda"
    payloads = [evaluate_checkpoint(spec, device, BATCH_SIZE, NUM_WORKERS, MAX_BATCHES) for spec in eval_specs]
    for index, payload in enumerate(payloads, start=1):
        print_payload(payload, index, len(payloads))
    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payloads if len(payloads) > 1 else payloads[0], indent=2))
        print(f"\nWrote metrics JSON to {output_path}")


if __name__ == "__main__":
    main()
