#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
import logging
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

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Subset

from lerobot.configs.default import ValidationConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.xvla.action_contract import build_slice_map, get_so101_slice_spec, slice_dataset_meta_in_place
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.slice_processor import SliceProcessorStep
from lerobot.scripts.lerobot_train import GRIPPER_DEBUG_COUNT_KEYS, _compute_policy_loss, _make_xvla_validation_corruption, build_dataset_frame_indices, make_dataset_with_episodes, split_train_validation_episodes, FixedIndexSampler
from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME, PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config
from thesis_vla.training.visual_thought_trainer import VisualThoughtTrainConfig


CHECKPOINTS: list[str | dict[str, str]] = [
    # "edgarcancinoe/orange196_cloth-corner-fold_7p5hz_so101_ee6d_am_sm_b16_ga2_eb64_full_adap-d345f6ab",
    # "edgarcancinoe/cedirnet_joint_stage_20260611_195850_cloth_fold::step_0000500",
]

DEVICE = "cuda"
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_BATCHES = 10
OUTPUT_JSON = ""
EVALUATE_TRAIN_SPLIT = False
DEAD_POLICY_THRESHOLD_MM = 5.0
PROGRESS_EVERY = 10
DATASET_TOLERANCE_S_OVERRIDE: float | None = 0.2

VISUAL_THOUGHT_CONFIG_NAME = "visual_thought_config.json"
METADATA_FILENAME = "metadata.json"


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    name: str
    checkpoint: str


@dataclasses.dataclass(frozen=True)
class ResolvedCheckpoint:
    name: str
    checkpoint: str
    checkpoint_root: Path
    policy_path: str
    config_kind: str
    config_source_root: Path | None = None
    train_cfg: TrainPipelineConfig | None = None
    vt_cfg: VisualThoughtTrainConfig | None = None


class _EvalAccelerator:
    def __init__(self, device: torch.device):
        self.device = device

    def autocast(self):
        return nullcontext()

    def unwrap_model(self, model, keep_fp32_wrapper: bool = False):
        return model


def normalize_eval_specs(raw_specs: list[str | dict[str, str]]) -> list[EvalSpec]:
    normalized_specs: list[EvalSpec] = []
    for index, raw_spec in enumerate(raw_specs, start=1):
        if isinstance(raw_spec, str):
            checkpoint = raw_spec.strip()
            if not checkpoint: continue
            normalized_specs.append(EvalSpec(name=Path(checkpoint.split("::", 1)[0]).name or f"checkpoint_{index}", checkpoint=checkpoint))
            continue
        if isinstance(raw_spec, dict):
            checkpoint = raw_spec.get("checkpoint", "").strip()
            if not checkpoint: raise ValueError(f"CHECKPOINTS[{index - 1}] is missing a non-empty 'checkpoint' value.")
            name = raw_spec.get("name", "").strip() or Path(checkpoint.split("::", 1)[0]).name or f"checkpoint_{index}"
            normalized_specs.append(EvalSpec(name=name, checkpoint=checkpoint))
            continue
        raise TypeError(f"CHECKPOINTS[{index - 1}] must be a string path/repo id or a dict with 'checkpoint' and optional 'name'.")
    if not normalized_specs: raise ValueError("CHECKPOINTS is empty. Populate it in this file before running the script.")
    return normalized_specs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto": return auto_select_torch_device()
    return get_safe_torch_device(device_arg, log=True)


def _parse_remote_ref(checkpoint: str) -> tuple[str, str | None, str | None]:
    revision, subdir, repo_id = None, None, checkpoint
    if "::" in repo_id:
        repo_id, subdir = repo_id.split("::", 1)
        subdir = subdir.strip("/") or None
    if repo_id.count("/") == 1 and "@" in repo_id.split("/", 1)[1]:
        repo_id, revision = repo_id.rsplit("@", 1)
    return repo_id, revision, subdir


def _download_checkpoint_root(checkpoint: str) -> Path:
    repo_id, revision, subdir = _parse_remote_ref(checkpoint)
    allow_patterns = [
        TRAIN_CONFIG_NAME,
        VISUAL_THOUGHT_CONFIG_NAME,
        METADATA_FILENAME,
        "config.json",
        "model.safetensors",
        f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        f"{POLICY_PREPROCESSOR_DEFAULT_NAME}_*",
        f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}_*",
        "policy/**",
        "step_*/policy/**",
        f"step_*/{VISUAL_THOUGHT_CONFIG_NAME}",
        f"step_*/{METADATA_FILENAME}",
        f"step_*/{TRAIN_CONFIG_NAME}",
    ]
    if subdir is not None: allow_patterns = [f"{subdir}/**"]
    local_root = Path(snapshot_download(repo_id=repo_id, repo_type="model", revision=revision, allow_patterns=allow_patterns))
    if subdir is not None: return local_root / subdir
    if (local_root / VISUAL_THOUGHT_CONFIG_NAME).exists() or (local_root / TRAIN_CONFIG_NAME).exists(): return local_root
    staged_candidates = sorted([child for child in local_root.iterdir() if child.is_dir() and child.name.startswith("step_") and ((child / VISUAL_THOUGHT_CONFIG_NAME).exists() or (child / TRAIN_CONFIG_NAME).exists() or (child / "policy" / "config.json").exists())])
    if len(staged_candidates) == 1: return staged_candidates[0]
    if staged_candidates:
        example = ", ".join(child.name for child in staged_candidates[:5])
        raise ValueError(f"{checkpoint}: remote repo contains staged checkpoints ({example}). Use a staged ref like '{repo_id}::{staged_candidates[-1].name}'.")
    return local_root


def _resolve_local_checkpoint_root(candidate: Path) -> Path:
    if candidate.is_file():
        if candidate.name in {TRAIN_CONFIG_NAME, VISUAL_THOUGHT_CONFIG_NAME}: return candidate.parent
        raise ValueError(f"Unsupported checkpoint file: {candidate}")
    if candidate.name == "policy" and (candidate.parent / VISUAL_THOUGHT_CONFIG_NAME).exists(): return candidate.parent
    if (candidate / VISUAL_THOUGHT_CONFIG_NAME).exists() or (candidate / TRAIN_CONFIG_NAME).exists(): return candidate
    if (candidate / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME).exists(): return candidate / PRETRAINED_MODEL_DIR
    checkpoints_dir = candidate / CHECKPOINTS_DIR
    last_pretrained_dir = checkpoints_dir / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR
    if (last_pretrained_dir / TRAIN_CONFIG_NAME).exists(): return last_pretrained_dir
    if checkpoints_dir.exists():
        checkpoint_candidates = sorted([path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()], key=lambda path: int(path.name))
        if checkpoint_candidates and (checkpoint_candidates[-1] / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME).exists(): return checkpoint_candidates[-1] / PRETRAINED_MODEL_DIR
    if (candidate / "policy" / "config.json").exists() and (candidate / METADATA_FILENAME).exists(): return candidate
    raise ValueError(f"Could not resolve checkpoint root from {candidate}")


def _find_visual_thought_config_source(candidate: Path) -> Path | None:
    if (candidate / VISUAL_THOUGHT_CONFIG_NAME).exists(): return candidate
    parent = candidate.parent
    sibling_roots = sorted([path for path in parent.iterdir() if path.is_dir() and path.name.startswith("checkpoint_") and (path / VISUAL_THOUGHT_CONFIG_NAME).exists()]) if parent.exists() else []
    return sibling_roots[-1] if sibling_roots else None


def resolve_checkpoint(spec: EvalSpec) -> ResolvedCheckpoint:
    candidate = Path(spec.checkpoint).expanduser()
    checkpoint_root = _resolve_local_checkpoint_root(candidate.resolve()) if candidate.exists() else _download_checkpoint_root(spec.checkpoint)
    config_source_root = checkpoint_root if (checkpoint_root / VISUAL_THOUGHT_CONFIG_NAME).exists() else _find_visual_thought_config_source(checkpoint_root)
    if config_source_root is not None:
        if config_source_root != checkpoint_root: logging.warning("%s: checkpoint %s is missing %s; reusing config from sibling checkpoint %s", spec.name, checkpoint_root, VISUAL_THOUGHT_CONFIG_NAME, config_source_root)
        vt_cfg = VisualThoughtTrainConfig.from_json(config_source_root / VISUAL_THOUGHT_CONFIG_NAME)
        policy_path = str((checkpoint_root / "policy").resolve()) if (checkpoint_root / "policy" / "config.json").exists() else str(checkpoint_root.resolve())
        return ResolvedCheckpoint(name=spec.name, checkpoint=spec.checkpoint, checkpoint_root=checkpoint_root, policy_path=policy_path, config_kind="visual_thought", config_source_root=config_source_root, vt_cfg=vt_cfg)
    if (checkpoint_root / TRAIN_CONFIG_NAME).exists():
        train_cfg = TrainPipelineConfig.from_pretrained(str(checkpoint_root))
        train_cfg.policy.pretrained_path = str(checkpoint_root)
        return ResolvedCheckpoint(name=spec.name, checkpoint=spec.checkpoint, checkpoint_root=checkpoint_root, policy_path=str(checkpoint_root), config_kind="lerobot_train", config_source_root=checkpoint_root, train_cfg=train_cfg)
    raise ValueError(f"{spec.checkpoint}: checkpoint root {checkpoint_root} does not contain {TRAIN_CONFIG_NAME} or {VISUAL_THOUGHT_CONFIG_NAME}.")


def resolve_validation_config(name: str, resolved: ResolvedCheckpoint) -> tuple[ValidationConfig, list[str]]:
    warnings = []
    if resolved.config_kind == "lerobot_train":
        if resolved.train_cfg is None: raise RuntimeError("Missing train config.")
        if resolved.train_cfg.validation.enable: return resolved.train_cfg.validation, warnings
        fallback = ValidationConfig(enable=True)
        warnings.append(f"{name}: saved training config has validation disabled; using default validation split settings (split_ratio={fallback.split_ratio}, seed={fallback.seed}, max_batches={fallback.max_batches}).")
        logging.warning(warnings[-1])
        return fallback, warnings
    if resolved.vt_cfg is None: raise RuntimeError("Missing visual-thought config.")
    if bool(resolved.vt_cfg.validation_enable) and float(resolved.vt_cfg.validation_split_ratio) > 0.0:
        return ValidationConfig(enable=True, split_ratio=float(resolved.vt_cfg.validation_split_ratio), seed=int(resolved.vt_cfg.validation_seed), max_batches=int(resolved.vt_cfg.validation_max_batches)), warnings
    fallback = ValidationConfig(enable=True)
    warnings.append(f"{name}: visual-thought config has validation disabled; using default validation split settings (split_ratio={fallback.split_ratio}, seed={fallback.seed}, max_batches={fallback.max_batches}).")
    logging.warning(warnings[-1])
    return fallback, warnings


def make_visual_thought_dataset(cfg: VisualThoughtTrainConfig):
    tolerance_s = float(DATASET_TOLERANCE_S_OVERRIDE) if DATASET_TOLERANCE_S_OVERRIDE is not None else float(cfg.dataset_tolerance_s)
    return LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, revision=cfg.dataset_revision, video_backend=cfg.dataset_video_backend, tolerance_s=tolerance_s)


def split_visual_thought_dataset(dataset, validation_cfg: ValidationConfig) -> tuple[Subset, Subset, list[int], list[int]]:
    dataset_len = len(dataset)
    val_len = min(max(int(round(dataset_len * float(validation_cfg.split_ratio))), 1), max(dataset_len - 1, 1))
    if dataset_len < 2 or val_len >= dataset_len: raise ValueError("Dataset too small to build a validation split.")
    generator = torch.Generator().manual_seed(int(validation_cfg.seed))
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    val_indices, train_indices = permutation[:val_len], permutation[val_len:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices


def prepare_datasets_and_policy_config(resolved: ResolvedCheckpoint, validation_cfg: ValidationConfig):
    if resolved.config_kind == "lerobot_train":
        if resolved.train_cfg is None: raise RuntimeError("Missing train config.")
        cfg = resolved.train_cfg
        cfg.policy.device = str(resolve_device(DEVICE))
        cfg.dataset = dataclasses.replace(cfg.dataset, image_transforms=dataclasses.replace(cfg.dataset.image_transforms, enable=False))
        if DATASET_TOLERANCE_S_OVERRIDE is not None: cfg.dataset.tolerance_s = float(DATASET_TOLERANCE_S_OVERRIDE)
        base_dataset = make_dataset(cfg)
        train_episodes, val_episodes = split_train_validation_episodes(base_dataset, validation_cfg.split_ratio, validation_cfg.seed)
        train_dataset = make_dataset_with_episodes(cfg, train_episodes, disable_augmentation=True)
        val_dataset = make_dataset_with_episodes(cfg, val_episodes, disable_augmentation=True)
        split_info = {"split_mode": "episode", "train_episode_count": len(train_episodes), "val_episode_count": len(val_episodes), "train_episodes": train_episodes, "val_episodes": val_episodes}
        policy_cfg = cfg.policy
        return train_dataset, val_dataset, split_info, policy_cfg
    if resolved.vt_cfg is None: raise RuntimeError("Missing visual-thought config.")
    base_dataset = make_visual_thought_dataset(resolved.vt_cfg)
    train_dataset, val_dataset, train_indices, val_indices = split_visual_thought_dataset(base_dataset, validation_cfg)
    episode_column = getattr(getattr(base_dataset, "hf_dataset", None), "__getitem__", None)
    episode_counts = {"train_episode_count": 0, "val_episode_count": 0}
    if episode_column is not None:
        episodes = base_dataset.hf_dataset["episode_index"]
        train_eps = sorted({int(episodes[index].item()) if torch.is_tensor(episodes[index]) else int(episodes[index]) for index in train_indices})
        val_eps = sorted({int(episodes[index].item()) if torch.is_tensor(episodes[index]) else int(episodes[index]) for index in val_indices})
        episode_counts = {"train_episode_count": len(train_eps), "val_episode_count": len(val_eps), "train_episodes": train_eps, "val_episodes": val_eps}
    split_info = {"split_mode": "frame", "train_index_count": len(train_indices), "val_index_count": len(val_indices), **episode_counts}
    policy_cfg = XVLAPolicy.from_pretrained(resolved.policy_path).config
    return train_dataset, val_dataset, split_info, policy_cfg


def make_eval_dataloader(dataset, policy_cfg, batch_size: int, num_workers: int, device_type: str):
    sampler = None
    if hasattr(policy_cfg, "drop_n_last_frames"): sampler = FixedIndexSampler(build_dataset_frame_indices(dataset, drop_n_last_frames=policy_cfg.drop_n_last_frames), shuffle=False)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=device_type == "cuda", drop_last=False, prefetch_factor=2 if num_workers > 0 else None)


def ensure_xvla_slice_step(preprocessor, slice_spec) -> None:
    if slice_spec is None: return
    slice_map = build_slice_map(slice_spec)
    has_slice_step = any(isinstance(step, SliceProcessorStep) and getattr(step, "slice_map", None) == slice_map for step in preprocessor.steps)
    if has_slice_step: return
    insert_idx = len(preprocessor.steps)
    for idx, step in enumerate(preprocessor.steps):
        if isinstance(step, NormalizerProcessorStep):
            insert_idx = idx
            break
    preprocessor.steps.insert(insert_idx, SliceProcessorStep(slice_map=slice_map))


def load_policy_and_processors(resolved: ResolvedCheckpoint, dataset, device: torch.device):
    policy_cfg = XVLAPolicy.from_pretrained(resolved.policy_path).config
    if policy_cfg.type != "xvla": raise ValueError(f"{resolved.name}: expected an XVLA checkpoint, got policy type {policy_cfg.type!r}")
    policy_cfg.device = str(device)
    slice_spec = get_so101_slice_spec(getattr(policy_cfg, "action_mode", None))
    if slice_spec is not None: slice_dataset_meta_in_place(dataset.meta, slice_spec)
    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    policy = XVLAPolicy.from_pretrained(resolved.policy_path, config=policy_cfg, device=str(device))
    if hasattr(policy.config, "enable_gripper_debug_stats"): policy.config.enable_gripper_debug_stats = True
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=resolved.policy_path, device=device.type, rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=False)
    ensure_xvla_slice_step(preprocessor, slice_spec)
    return policy, preprocessor, postprocessor, slice_spec


def to_scalar_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1: return None
        value = value.detach().item()
    elif not isinstance(value, (int, float)):
        return None
    return float(value)


def _slice_tensor(x: Any, slice_spec) -> torch.Tensor:
    tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if slice_spec is None: return tensor
    return slice_spec.slice_tensor(tensor)


def _ensure_batched_chunk_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim == 1: return tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
    if tensor.ndim == 2:
        if tensor.shape[0] == batch_size: return tensor.unsqueeze(1)
        return tensor.unsqueeze(0).expand(batch_size, -1, -1)
    return tensor


def _valid_chunk_length(pad_mask: torch.Tensor | None, chunk_len: int) -> int:
    if pad_mask is None: return int(chunk_len)
    valid = int((~pad_mask.bool()).sum().item())
    return max(valid, 0)


def _first_gripper_threshold(policy) -> tuple[int, float]:
    idx = int(policy.model.action_space.gripper_idx[0])
    threshold = float(getattr(policy.config, "gripper_open_threshold", 0.0))
    return idx, threshold


def _behavior_metrics_from_batch(policy, raw_batch: dict[str, Any], pred_chunk: torch.Tensor, slice_spec, dead_threshold_mm: float) -> dict[str, float]:
    gripper_idx, gripper_threshold = _first_gripper_threshold(policy)
    if pred_chunk.ndim == 2: pred_chunk = pred_chunk.unsqueeze(1)
    target_chunk = _slice_tensor(raw_batch["action"], slice_spec).detach().to(dtype=torch.float32).cpu()
    state = _slice_tensor(raw_batch["observation.state"], slice_spec).detach().to(dtype=torch.float32).cpu()
    pad_mask = raw_batch.get("action_is_pad")
    if pad_mask is not None: pad_mask = torch.as_tensor(pad_mask).detach().cpu().bool()
    batch_size, chunk_size = int(pred_chunk.shape[0]), int(pred_chunk.shape[1])
    target_chunk = _ensure_batched_chunk_tensor(target_chunk, batch_size)
    if target_chunk.shape[1] < chunk_size: target_chunk = torch.cat([target_chunk, target_chunk[:, -1:, :].expand(-1, chunk_size - target_chunk.shape[1], -1)], dim=1)
    elif target_chunk.shape[1] > chunk_size: target_chunk = target_chunk[:, :chunk_size]
    if state.ndim == 1: state = state.unsqueeze(0).expand(batch_size, -1)
    chunk_disp_mm, first_disp_mm, first_err_mm, first_gripper_vals, closed_flags = [], [], [], [], []
    for batch_index in range(batch_size):
        valid_len = _valid_chunk_length(None if pad_mask is None else pad_mask[batch_index], chunk_size)
        if valid_len < 1: continue
        pred_valid = pred_chunk[batch_index, :valid_len]
        target_valid = target_chunk[batch_index, :valid_len]
        current_xyz = state[batch_index, :3]
        first_pred_xyz = pred_valid[0, :3]
        last_pred_xyz = pred_valid[-1, :3]
        first_target_xyz = target_valid[0, :3]
        disp_chunk = float(torch.linalg.vector_norm(last_pred_xyz - first_pred_xyz).item()) * 1000.0
        disp_first = float(torch.linalg.vector_norm(first_pred_xyz - current_xyz).item()) * 1000.0
        err_first = float(torch.linalg.vector_norm(first_pred_xyz - first_target_xyz).item()) * 1000.0
        gripper_value = float(pred_valid[0, gripper_idx].item())
        chunk_disp_mm.append(disp_chunk)
        first_disp_mm.append(disp_first)
        first_err_mm.append(err_first)
        first_gripper_vals.append(gripper_value)
        closed_flags.append(gripper_value < gripper_threshold)
    if not chunk_disp_mm:
        return {"mean_chunk_disp_mm": float("nan"), "median_chunk_disp_mm": float("nan"), "dead_policy_rate": float("nan"), "mean_first_action_disp_mm": float("nan"), "median_first_action_disp_mm": float("nan"), "first_action_pos_error_mm": float("nan"), "mean_gripper_value": float("nan"), "closed_gripper_rate": float("nan"), "behavior_samples": 0.0}
    return {
        "mean_chunk_disp_mm": float(np.mean(chunk_disp_mm)),
        "median_chunk_disp_mm": float(np.median(chunk_disp_mm)),
        "dead_policy_rate": float(np.mean(np.asarray(chunk_disp_mm) < float(dead_threshold_mm))),
        "mean_first_action_disp_mm": float(np.mean(first_disp_mm)),
        "median_first_action_disp_mm": float(np.median(first_disp_mm)),
        "first_action_pos_error_mm": float(np.mean(first_err_mm)),
        "mean_gripper_value": float(np.mean(first_gripper_vals)),
        "closed_gripper_rate": float(np.mean(closed_flags)),
        "behavior_samples": float(len(chunk_disp_mm)),
    }


def evaluate_split(name: str, policy, dataloader, preprocessor, postprocessor, slice_spec, validation_seed: int, max_batches: int | None, progress_every: int, dead_threshold_mm: float) -> dict[str, float]:
    fake_accelerator = _EvalAccelerator(next(policy.parameters()).device)
    was_training = policy.training
    policy.eval()
    local_loss_sum, local_sample_count, local_batch_count = 0.0, 0.0, 0.0
    local_component_sums: dict[str, float] = {}
    local_component_counts: dict[str, float] = {}
    local_gripper_counts = {key: 0.0 for key in GRIPPER_DEBUG_COUNT_KEYS}
    behavior_sums: dict[str, float] = {}
    behavior_counts: dict[str, float] = {}
    total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    print(f"[progress] split={name} starting batches={total_batches}", flush=True)
    try:
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches: break
                if batch_index == 0 or (progress_every > 0 and (batch_index + 1) % progress_every == 0) or (batch_index + 1 == total_batches): print(f"[progress] split={name} batch={batch_index + 1}/{total_batches}", flush=True)
                batch = preprocessor(raw_batch)
                deterministic_validation_corruption = _make_xvla_validation_corruption(policy, batch, batch_index, validation_seed)
                loss, output_dict, batch_size = _compute_policy_loss(policy, batch, fake_accelerator, deterministic_validation_corruption=deterministic_validation_corruption)
                local_loss_sum += float(loss.detach().item()) * batch_size
                local_sample_count += batch_size
                local_batch_count += 1
                for key, value in output_dict.items():
                    if not key.endswith("_loss"): continue
                    scalar_value = to_scalar_float(value)
                    if scalar_value is None: continue
                    local_component_sums[key] = local_component_sums.get(key, 0.0) + scalar_value * batch_size
                    local_component_counts[key] = local_component_counts.get(key, 0.0) + batch_size
                gripper_debug_counts = output_dict.get("gripper_debug_counts")
                if gripper_debug_counts is not None:
                    for key in GRIPPER_DEBUG_COUNT_KEYS:
                        value = gripper_debug_counts[key]
                        if isinstance(value, torch.Tensor): value = value.detach().item()
                        local_gripper_counts[key] += float(value)
                policy.reset()
                pred_chunk = postprocessor(policy.predict_action_chunk(batch)).detach().to(dtype=torch.float32).cpu()
                behavior_metrics = _behavior_metrics_from_batch(policy, raw_batch, pred_chunk, slice_spec, dead_threshold_mm)
                sample_count = behavior_metrics.pop("behavior_samples", 0.0)
                if sample_count > 0:
                    for key, value in behavior_metrics.items():
                        behavior_sums[key] = behavior_sums.get(key, 0.0) + float(value) * sample_count
                        behavior_counts[key] = behavior_counts.get(key, 0.0) + sample_count
    finally:
        policy.train(was_training)
    if local_batch_count < 1 or local_sample_count <= 0: raise RuntimeError(f"{name} split evaluation ran with zero batches.")
    print(f"[progress] split={name} done batches={int(local_batch_count)} samples={int(local_sample_count)}", flush=True)
    metrics = {"action_total": local_loss_sum / local_sample_count, "num_batches": float(local_batch_count), "num_samples": float(local_sample_count)}
    for key, component_sum in local_component_sums.items():
        denom = local_component_counts[key]
        if denom > 0: metrics[key] = component_sum / denom
    total = local_gripper_counts["true_negative_count"] + local_gripper_counts["true_positive_count"] + local_gripper_counts["false_positive_count"] + local_gripper_counts["false_negative_count"]
    if total > 0:
        class0_total = local_gripper_counts["target_zero_count"]
        class1_total = local_gripper_counts["target_one_count"]
        metrics["gripper_accuracy"] = (local_gripper_counts["true_negative_count"] + local_gripper_counts["true_positive_count"]) / total
        metrics["gripper_class0_accuracy"] = local_gripper_counts["true_negative_count"] / class0_total if class0_total > 0 else 0.0
        metrics["gripper_class1_accuracy"] = local_gripper_counts["true_positive_count"] / class1_total if class1_total > 0 else 0.0
    for key, weighted_sum in behavior_sums.items():
        denom = behavior_counts.get(key, 0.0)
        if denom > 0: metrics[key] = weighted_sum / denom
    return metrics


def ordered_metric_items(metrics: dict[str, float]) -> list[tuple[str, float]]:
    preferred = ["action_total", "position_loss", "rotate6D_loss", "gripper_loss", "gripper_accuracy", "gripper_class0_accuracy", "gripper_class1_accuracy", "mean_chunk_disp_mm", "median_chunk_disp_mm", "dead_policy_rate", "mean_first_action_disp_mm", "median_first_action_disp_mm", "first_action_pos_error_mm", "mean_gripper_value", "closed_gripper_rate", "num_batches", "num_samples"]
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
    print(f"[progress] checkpoint={spec.name} resolving", flush=True)
    resolved = resolve_checkpoint(spec)
    validation_cfg, warnings = resolve_validation_config(spec.name, resolved)
    print(f"[progress] checkpoint={spec.name} building datasets", flush=True)
    train_dataset, val_dataset, split_info, policy_cfg = prepare_datasets_and_policy_config(resolved, validation_cfg)
    eval_batch_size = batch_size_override if batch_size_override is not None else (resolved.vt_cfg.batch_size if resolved.vt_cfg is not None else resolved.train_cfg.batch_size)
    eval_num_workers = num_workers_override if num_workers_override is not None else (resolved.vt_cfg.num_workers if resolved.vt_cfg is not None else resolved.train_cfg.num_workers)
    reference_dataset = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
    print(f"[progress] checkpoint={spec.name} loading policy", flush=True)
    policy, preprocessor, postprocessor, slice_spec = load_policy_and_processors(resolved, reference_dataset, device)
    train_dataloader = make_eval_dataloader(train_dataset, policy_cfg, eval_batch_size, eval_num_workers, device.type) if EVALUATE_TRAIN_SPLIT else None
    val_dataloader = make_eval_dataloader(val_dataset, policy_cfg, eval_batch_size, eval_num_workers, device.type)
    train_metrics = None
    if train_dataloader is not None:
        print(f"[progress] checkpoint={spec.name} evaluating train_split", flush=True)
        train_metrics = evaluate_split("train", policy, train_dataloader, preprocessor, postprocessor, slice_spec, validation_cfg.seed, max_batches, PROGRESS_EVERY, DEAD_POLICY_THRESHOLD_MM)
    print(f"[progress] checkpoint={spec.name} evaluating validation_split", flush=True)
    val_metrics = evaluate_split("validation", policy, val_dataloader, preprocessor, postprocessor, slice_spec, validation_cfg.seed, max_batches, PROGRESS_EVERY, DEAD_POLICY_THRESHOLD_MM)
    print(f"[progress] checkpoint={spec.name} complete", flush=True)
    return {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "resolved_checkpoint_root": str(resolved.checkpoint_root),
        "config_source_root": str((resolved.config_source_root or resolved.checkpoint_root)),
        "policy_path": resolved.policy_path,
        "config_kind": resolved.config_kind,
        "device": str(device),
        "validation_split_ratio": validation_cfg.split_ratio,
        "validation_seed": validation_cfg.seed,
        "warnings": warnings,
        "eval_batch_size": eval_batch_size,
        "eval_num_workers": eval_num_workers,
        "max_batches": max_batches,
        "dead_policy_threshold_mm": float(DEAD_POLICY_THRESHOLD_MM),
        "dataset_tolerance_s_override": DATASET_TOLERANCE_S_OVERRIDE,
        **split_info,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }


def print_payload(payload: dict[str, Any], index: int, total: int) -> None:
    print(f"\n=== [{index}/{total}] {payload['name']} ===")
    print(f"Checkpoint: {payload['checkpoint']}")
    print(f"Resolved root: {payload['resolved_checkpoint_root']}")
    print(f"Config source: {payload['config_source_root']}")
    print(f"Policy path: {payload['policy_path']}")
    print(f"Config kind: {payload['config_kind']}")
    print(f"Device: {payload['device']}")
    print(f"Split: ratio={payload['validation_split_ratio']} seed={payload['validation_seed']} mode={payload['split_mode']}")
    if "train_episode_count" in payload or "val_episode_count" in payload: print(f"Episodes: train={payload.get('train_episode_count', 0)} val={payload.get('val_episode_count', 0)}")
    if "train_index_count" in payload or "val_index_count" in payload: print(f"Indices: train={payload.get('train_index_count', 0)} val={payload.get('val_index_count', 0)}")
    for warning in payload.get("warnings", []): print(f"Warning: {warning}")
    print(f"Evaluation: batch_size={payload['eval_batch_size']} num_workers={payload['eval_num_workers']} max_batches={'all' if payload['max_batches'] is None else payload['max_batches']} dead_mm<{payload['dead_policy_threshold_mm']} tol_override={payload['dataset_tolerance_s_override']}")
    print_metric_block("Validation Split Metrics", payload["validation_metrics"])
    if payload.get("train_metrics") is not None: print_metric_block("Train Split Metrics", payload["train_metrics"])


def print_ranking(payloads: list[dict[str, Any]]) -> None:
    ranked = sorted(payloads, key=lambda payload: (payload["validation_metrics"].get("dead_policy_rate", float("inf")), -payload["validation_metrics"].get("mean_chunk_disp_mm", float("-inf")), payload["validation_metrics"].get("action_total", float("inf"))))
    print("\n=== Ranking (best first) ===")
    for index, payload in enumerate(ranked, start=1):
        metrics = payload["validation_metrics"]
        print(f"{index:02d}. {payload['name']} | dead_policy_rate={metrics.get('dead_policy_rate', float('nan')):.6f} | mean_chunk_disp_mm={metrics.get('mean_chunk_disp_mm', float('nan')):.3f} | action_total={metrics.get('action_total', float('nan')):.6f}")


def main() -> None:
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    eval_specs = normalize_eval_specs(CHECKPOINTS)
    device = resolve_device(DEVICE)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = device.type == "cuda"
    payloads = [evaluate_checkpoint(spec, device, BATCH_SIZE, NUM_WORKERS, MAX_BATCHES) for spec in eval_specs]
    for index, payload in enumerate(payloads, start=1): print_payload(payload, index, len(payloads))
    if len(payloads) > 1: print_ranking(payloads)
    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payloads if len(payloads) > 1 else payloads[0], indent=2))
        print(f"\nWrote metrics JSON to {output_path}")


if __name__ == "__main__":
    main()
