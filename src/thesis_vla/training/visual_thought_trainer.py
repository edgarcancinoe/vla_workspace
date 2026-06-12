from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.xvla.action_contract import build_slice_map, get_so101_slice_spec
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.slice_processor import SliceProcessorStep
from thesis_vla.common.hf_hub import HubUploadConfig, clear_hub_upload_failure_marker, push_folder_to_hub, write_hub_upload_failure_marker
from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config
from thesis_vla.visual_thought import CeDirNetDistillationModel, DinoFeatureAlignmentModel, DinoTokenSequenceModel, compute_feature_alignment_loss, load_cedirnet_decoder_config, load_dino_decoder_config
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache
from thesis_vla.visual_thought.checkpoints import CONFIG_FILENAME, DECODER_STATE_FILENAME, DECODER_STATE_TEMPLATE, POLICY_DIRNAME, TRAINER_STATE_FILENAME, load_decoder_state, load_visual_thought_checkpoint_metadata, load_visual_thought_config_snapshot, save_visual_thought_checkpoint
from thesis_vla.visual_thought.targets import TeacherTarget, compute_teacher_loss
from thesis_vla.visual_thought.teachers import DinoV2Teacher


TrainingStage = Literal["distill_only", "joint_multitask"]
ExpertType = Literal["cedirnet", "dino"]

# Match the normal XVLA finetune loop (lerobot uses accelerator.accumulate, which averages
# the loss across the accumulation window and clips grad norm at the boundary). XVLA's
# optimizer default is grad_clip_norm=10.0 (configuration_xvla.optimizer_grad_clip_norm).
XVLA_GRAD_CLIP_NORM = 10.0
TRAINING_HUB_UPLOAD_CONFIG = HubUploadConfig(max_retries=5, retry_backoff_s=5.0)
DEFAULT_NORMALIZATION_MAPPING = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
FORCED_VALIDATION_STEPS = frozenset({0, 1})


@dataclass
class VisualThoughtTrainConfig:
    name: str
    training_stage: TrainingStage
    expert_type: ExpertType
    xvla_init_path: str
    decoder_init_path: str | None
    decoder_stack_config_path: str
    decoder_task_config_path: str
    dataset_repo_id: str
    dataset_revision: str | None
    dataset_root: str | None
    output_dir: str
    device: str
    expert_types: tuple[ExpertType, ...] | None = None
    cedirnet_decoder_init_path: str | None = None
    cedirnet_decoder_stack_config_path: str | None = None
    cedirnet_decoder_task_config_path: str | None = None
    dino_decoder_init_path: str | None = None
    dino_decoder_stack_config_path: str | None = None
    dino_decoder_task_config_path: str | None = None
    cuda_visible_devices: tuple[int, ...] = (0,)
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    steps: int = 1_000
    log_every: int = 20
    save_every: int = 500
    weight_decay: float = 0.01
    xvla_adaptation_mode: str | None = None
    xvla_freeze_steps: int | None = None
    xvla_warmup_steps: int | None = None
    xvla_learning_coef: float | None = None
    xvla_optimizer_lr: float | None = None
    xvla_optimizer_soft_prompt_lr_scale: float | None = None
    xvla_optimizer_soft_prompt_warmup_lr_scale: float | None = None
    xvla_scheduler_warmup_steps: int | None = None
    xvla_scheduler_decay_steps: int | None = None
    xvla_scheduler_decay_lr: float | None = None
    decoder_optimizer_lr: float = 1e-4
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 1.0
    cedirnet_expert_loss_weight: float = 1.0
    dino_expert_loss_weight: float = 1.0
    teacher_image_feature_key: str = "observation.images.image"
    teacher_target_cache_root: str | None = None
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    normalization_mapping: str = DEFAULT_NORMALIZATION_MAPPING
    wandb_enable: bool = False
    wandb_project: str = "visual-thought"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    validation_enable: bool = False
    validation_split_ratio: float = 0.1
    validation_freq: int = 500
    validation_max_batches: int = 10
    validation_seed: int = 1337
    vis_every: int = 0
    vis_num_samples: int = 4
    vis_final: bool = True
    profile_step_time_every: int = 0
    cutout_enable: bool = False
    cutout_prob: float = 0.3
    cutout_num_patches: int = 1
    cutout_area_min: float = 0.05
    cutout_area_max: float = 0.15
    cutout_aspect_min: float = 0.75
    cutout_aspect_max: float = 1.5
    cutout_fill: float = 0.0
    push_to_hub: bool = False
    push_repo_id: str | None = None
    push_every: int = 0
    align_feature_until_step: int = 0
    save_final_checkpoint: bool = True
    resume: bool = False
    resume_checkpoint_path: str | None = None
    seed: int = 42
    dry_run: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "VisualThoughtTrainConfig":
        payload = json.loads(Path(path).read_text())
        if "cuda_visible_devices" in payload: payload["cuda_visible_devices"] = tuple(int(device) for device in payload["cuda_visible_devices"])
        if "expert_types" in payload and payload["expert_types"] is not None: payload["expert_types"] = tuple(str(expert) for expert in payload["expert_types"])
        return cls(**payload)

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class XVLARuntime:
    policy: Any
    dataset: Any
    preprocessor: Any
    postprocessor: Any
    rename_map: dict[str, str]
    teacher_image_key: str
    policy_device: str
    vlm_device: str
    vlm_only_distill: bool


@dataclass
class JointTrainingState:
    policy_optimizer: torch.optim.Optimizer
    decoder_optimizer: torch.optim.Optimizer
    policy_scheduler: LRScheduler | None = None


def resolve_expert_types(config: VisualThoughtTrainConfig) -> tuple[ExpertType, ...]:
    if config.expert_types is not None:
        expert_types = tuple(str(expert) for expert in config.expert_types)
        if not expert_types: raise ValueError("expert_types must be non-empty when provided.")
        invalid = [expert for expert in expert_types if expert not in {"cedirnet", "dino"}]
        if invalid: raise ValueError(f"Unsupported expert_types={invalid}.")
        return expert_types  # type: ignore[return-value]
    return (config.expert_type,)


def is_combined_expert_run(config: VisualThoughtTrainConfig) -> bool:
    return len(resolve_expert_types(config)) > 1


def expert_loss_weight_for(config: VisualThoughtTrainConfig, expert_type: ExpertType) -> float:
    return float(config.cedirnet_expert_loss_weight if expert_type == "cedirnet" else config.dino_expert_loss_weight)


def decoder_init_path_for(config: VisualThoughtTrainConfig, expert_type: ExpertType) -> str | None:
    if not is_combined_expert_run(config): return config.decoder_init_path
    return config.cedirnet_decoder_init_path if expert_type == "cedirnet" else config.dino_decoder_init_path


def decoder_config_paths_for(config: VisualThoughtTrainConfig, expert_type: ExpertType) -> tuple[str, str]:
    if not is_combined_expert_run(config): return config.decoder_stack_config_path, config.decoder_task_config_path
    if expert_type == "cedirnet": return str(config.cedirnet_decoder_stack_config_path), str(config.cedirnet_decoder_task_config_path)
    return str(config.dino_decoder_stack_config_path), str(config.dino_decoder_task_config_path)


def _validate_config(config: VisualThoughtTrainConfig) -> None:
    expert_types = resolve_expert_types(config)
    if len(expert_types) == 1: return
    if config.training_stage != "joint_multitask": raise ValueError("Combined expert_types mode is supported for joint_multitask only.")
    if tuple(expert_types) != ("cedirnet", "dino"): raise ValueError(f"Combined expert_types must be ('cedirnet', 'dino'), got {expert_types!r}.")
    required = {
        "cedirnet_decoder_init_path": config.cedirnet_decoder_init_path,
        "cedirnet_decoder_stack_config_path": config.cedirnet_decoder_stack_config_path,
        "cedirnet_decoder_task_config_path": config.cedirnet_decoder_task_config_path,
        "dino_decoder_init_path": config.dino_decoder_init_path,
        "dino_decoder_stack_config_path": config.dino_decoder_stack_config_path,
        "dino_decoder_task_config_path": config.dino_decoder_task_config_path,
    }
    missing = [key for key, value in required.items() if not value]
    if missing: raise ValueError(f"Combined expert_types mode requires {', '.join(missing)}.")


def _resolve_grid_hw(target: TeacherTarget) -> tuple[int, int]:
    grid_hw = target.aux.get("grid_hw") if isinstance(target.aux, dict) else None
    if isinstance(grid_hw, (tuple, list)) and len(grid_hw) == 2: return int(grid_hw[0]), int(grid_hw[1])
    num_tokens = int(target.tensor.shape[1]); side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens: raise ValueError(f"Unable to infer token grid from N={num_tokens}.")
    return side, side


def _to_uint8_image(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().cpu().float()
    if image.ndim != 3: raise ValueError(f"Expected CHW image, got shape={tuple(image.shape)}.")
    image = image.permute(1, 2, 0).numpy()
    if image.max() <= 1.0: image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _draw_patch_borders(image_hwc: np.ndarray, gh: int, gw: int, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    canvas = image_hwc.copy(); height, width = canvas.shape[:2]
    for row in range(1, gh):
        y = int(round(row * height / gh))
        canvas[max(0, y - 1):min(height, y + 1), :] = color
    for col in range(1, gw):
        x = int(round(col * width / gw))
        canvas[:, max(0, x - 1):min(width, x + 1)] = color
    return canvas


def _compute_token_pca(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    token_np = tokens.detach().float().cpu().numpy(); norms = np.linalg.norm(token_np, axis=-1, keepdims=True)
    token_norm = token_np / np.clip(norms, 1e-8, None); centered = token_norm - token_norm.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False); pc1 = centered @ vt[0]; hi_mask = pc1 > np.median(pc1); flat_norms = norms.squeeze(-1)
    fg_mask = hi_mask if flat_norms[hi_mask].mean() >= flat_norms[~hi_mask].mean() else ~hi_mask
    if fg_mask.sum() > 3:
        fg_tokens = centered[fg_mask]; _, _, vt_fg = np.linalg.svd(fg_tokens - fg_tokens.mean(axis=0, keepdims=True), full_matrices=False); projected = (centered - fg_tokens.mean(axis=0, keepdims=True)) @ vt_fg[:3].T
    else:
        projected = centered @ vt[1:4].T
    for channel in range(3):
        lo = np.percentile(projected[fg_mask, channel], 2) if fg_mask.sum() > 1 else float(projected[:, channel].min())
        hi = np.percentile(projected[fg_mask, channel], 98) if fg_mask.sum() > 1 else float(projected[:, channel].max())
        projected[:, channel] = np.clip((projected[:, channel] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    pca_rgb = projected * fg_mask[:, None].astype(np.float32) + 0.3 * (~fg_mask)[:, None].astype(np.float32)
    return pca_rgb, fg_mask


def _select_fixed_vis_indices(n: int, seed: int, count: int) -> list[int]:
    return random.Random(int(seed)).sample(list(range(n)), min(max(int(count), 0), int(n))) if int(n) > 0 and int(count) > 0 else []


def _overlay_map(image_chw: torch.Tensor, map_hw: torch.Tensor, alpha: float = 0.45, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    image = _to_uint8_image(image_chw).astype(np.float32)
    if tuple(map_hw.shape[-2:]) != tuple(image.shape[:2]): map_hw = F.interpolate(map_hw.detach().float().unsqueeze(0).unsqueeze(0), size=image.shape[:2], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    map_np = map_hw.detach().cpu().float().numpy()
    lo = float(map_np.min()) if vmin is None else float(vmin); hi = float(map_np.max()) if vmax is None else float(vmax)
    denom = max(hi - lo, 1e-8); norm = np.clip((map_np - lo) / denom, 0.0, 1.0)
    heat = np.stack([norm, 1.0 - np.abs(2.0 * norm - 1.0), 1.0 - norm], axis=-1) * 255.0
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 255.0).astype(np.uint8)


def _channel_stem(index: int) -> str:
    return f"ch{int(index):02d}"


@torch.no_grad()
def run_visualization(config: VisualThoughtTrainConfig, runtime: XVLARuntime, teacher, decoder: torch.nn.Module, loader: DataLoader, step: int) -> dict[str, Any]:
    if int(config.vis_num_samples) <= 0: return {"vis_skipped": "vis_num_samples"}
    if is_combined_expert_run(config): return {"vis_skipped": "combined_expert_run"}
    decoder_was_training, policy_was_training = decoder.training, runtime.policy.training
    decoder.eval(); runtime.policy.eval()
    try:
        raw_batch = next(iter(loader))
        processed_batch = preprocess_batch(runtime, raw_batch)
        _, enc = build_xvla_inputs(runtime, processed_batch, config)
        target = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
        vis_root = Path(config.output_dir) / "visualizations" / f"step_{int(step):07d}"; vis_root.mkdir(parents=True, exist_ok=True)
        teacher_images = get_teacher_images(raw_batch, runtime.teacher_image_key); gallery = []
        sample_ids = raw_batch.get("index")
        if config.expert_type == "cedirnet" and target.kind == "dense_map":
            prediction = decoder(enc["vlm_features"], target_map=target.tensor)
            sample_count = min(int(config.vis_num_samples), int(prediction.shape[0]))
            target_maps = target.tensor.detach().cpu(); pred_maps = prediction.detach().cpu(); image_paths = []
            for sample_idx in range(sample_count):
                sample_id = int(sample_ids[sample_idx].item()) if torch.is_tensor(sample_ids) else int(sample_idx)
                stem = vis_root / f"sample{sample_idx:02d}_{sample_id}"
                image_path = stem.with_name(stem.name + "_image.png")
                Image.fromarray(_to_uint8_image(teacher_images[sample_idx])).save(image_path, compress_level=1)
                image_paths.append(str(image_path))
                channels = []
                for channel_idx in range(int(pred_maps.shape[1])):
                    target_ch = target_maps[sample_idx, channel_idx]; pred_ch = pred_maps[sample_idx, channel_idx]; error_ch = (pred_ch - target_ch).abs()
                    vmax = max(float(target_ch.max().item()), float(pred_ch.max().item())); vmin = min(float(target_ch.min().item()), float(pred_ch.min().item()))
                    error_vmax = float(error_ch.max().item()) if float(error_ch.max().item()) > 0.0 else 1.0
                    ch_name = _channel_stem(channel_idx)
                    teacher_overlay_path = stem.with_name(stem.name + f"_{ch_name}_teacher.png")
                    pred_overlay_path = stem.with_name(stem.name + f"_{ch_name}_pred.png")
                    error_overlay_path = stem.with_name(stem.name + f"_{ch_name}_error.png")
                    Image.fromarray(_overlay_map(teacher_images[sample_idx], target_ch, vmin=vmin, vmax=vmax)).save(teacher_overlay_path, compress_level=1)
                    Image.fromarray(_overlay_map(teacher_images[sample_idx], pred_ch, vmin=vmin, vmax=vmax)).save(pred_overlay_path, compress_level=1)
                    Image.fromarray(_overlay_map(teacher_images[sample_idx], error_ch, vmin=0.0, vmax=error_vmax)).save(error_overlay_path, compress_level=1)
                    channels.append({"channel": channel_idx, "teacher_overlay": str(teacher_overlay_path), "pred_overlay": str(pred_overlay_path), "error_overlay": str(error_overlay_path), "channel_mse": float(torch.mean((pred_ch - target_ch) ** 2).item()), "channel_l1": float(torch.mean(error_ch).item())})
                gallery.append({"sample_id": sample_id, "image": str(image_path), "num_channels": int(pred_maps.shape[1]), "map_mse": float(torch.mean((pred_maps[sample_idx] - target_maps[sample_idx]) ** 2).item()), "map_l1": float(torch.mean(torch.abs(pred_maps[sample_idx] - target_maps[sample_idx])).item()), "channels": channels})
            gallery_path = vis_root / "gallery.json"; gallery_path.write_text(json.dumps(gallery, indent=2))
            return {"vis_path": str(vis_root), "vis_gallery": str(gallery_path), "vis_samples": len(gallery)}
        if config.expert_type != "dino" or target.kind != "token_sequence": return {"vis_skipped": f"{config.expert_type}:{target.kind}"}
        prediction = decoder(enc["vlm_features"])
        gh, gw = _resolve_grid_hw(target); sample_count = min(int(config.vis_num_samples), int(prediction.shape[0]))
        teacher_pca_root = Path(config.output_dir) / "teacher_pca"; teacher_pca_root.mkdir(parents=True, exist_ok=True)
        pred_tokens = prediction.detach().cpu(); target_tokens = target.tensor.detach().cpu()
        cosine_maps = F.cosine_similarity(pred_tokens, target_tokens, dim=-1).view(pred_tokens.shape[0], gh, gw)
        error_maps = ((pred_tokens - target_tokens) ** 2).mean(dim=-1).view(pred_tokens.shape[0], gh, gw)
        for sample_idx in range(sample_count):
            sample_id = int(sample_ids[sample_idx].item()) if torch.is_tensor(sample_ids) else int(sample_idx)
            stem = vis_root / f"sample{sample_idx:02d}_{sample_id}"
            image_uint8 = _to_uint8_image(teacher_images[sample_idx]); Image.fromarray(_draw_patch_borders(image_uint8, gh, gw)).save(stem.with_name(stem.name + "_image.png"), compress_level=1)
            student_pca, fg_mask = _compute_token_pca(pred_tokens[sample_idx])
            student_pca_img = np.array(Image.fromarray((student_pca.reshape(gh, gw, 3) * 255).astype(np.uint8)).resize((image_uint8.shape[1], image_uint8.shape[0]), Image.NEAREST))
            Image.fromarray(_draw_patch_borders(student_pca_img, gh, gw)).save(stem.with_name(stem.name + "_pca_student.png"), compress_level=1)
            teacher_pca_path = teacher_pca_root / f"sample{sample_id}_pca_teacher.png"
            if not teacher_pca_path.exists():
                teacher_pca, _ = _compute_token_pca(target_tokens[sample_idx])
                teacher_pca_img = np.array(Image.fromarray((teacher_pca.reshape(gh, gw, 3) * 255).astype(np.uint8)).resize((image_uint8.shape[1], image_uint8.shape[0]), Image.NEAREST))
                Image.fromarray(_draw_patch_borders(teacher_pca_img, gh, gw)).save(teacher_pca_path, compress_level=1)
            fg_up = F.interpolate(torch.from_numpy(fg_mask.reshape(gh, gw).astype(np.float32)).unsqueeze(0).unsqueeze(0), size=teacher_images[sample_idx].shape[-2:], mode="nearest").squeeze(0).squeeze(0)
            fg_overlay = _overlay_map(teacher_images[sample_idx], fg_up)
            Image.fromarray(_draw_patch_borders(fg_overlay, gh, gw)).save(stem.with_name(stem.name + "_fg_mask.png"), compress_level=1)
            cos_overlay = _overlay_map(teacher_images[sample_idx], F.interpolate(cosine_maps[sample_idx].unsqueeze(0).unsqueeze(0), size=teacher_images[sample_idx].shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0), vmin=-1.0, vmax=1.0)
            err_up = F.interpolate(error_maps[sample_idx].unsqueeze(0).unsqueeze(0), size=teacher_images[sample_idx].shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            err_overlay = _overlay_map(teacher_images[sample_idx], err_up, vmin=0.0, vmax=float(err_up.max().item()) if float(err_up.max().item()) > 0.0 else 1.0)
            Image.fromarray(cos_overlay).save(stem.with_name(stem.name + "_cosine_overlay.png"), compress_level=1)
            Image.fromarray(err_overlay).save(stem.with_name(stem.name + "_error_overlay.png"), compress_level=1)
            gallery.append({"sample_id": sample_id, "mean_cosine": float(cosine_maps[sample_idx].mean().item()), "token_mse": float(error_maps[sample_idx].mean().item()), "image": str(stem.with_name(stem.name + "_image.png")), "pca_student": str(stem.with_name(stem.name + "_pca_student.png")), "pca_teacher": str(teacher_pca_path), "fg_mask": str(stem.with_name(stem.name + "_fg_mask.png")), "cosine_overlay": str(stem.with_name(stem.name + "_cosine_overlay.png")), "error_overlay": str(stem.with_name(stem.name + "_error_overlay.png"))})
        gallery_path = vis_root / "gallery.json"; gallery_path.write_text(json.dumps(gallery, indent=2))
        return {"vis_path": str(vis_root), "vis_gallery": str(gallery_path), "vis_samples": len(gallery)}
    finally:
        if decoder_was_training: decoder.train()
        if policy_was_training: runtime.policy.train()


def _as_device(config: VisualThoughtTrainConfig) -> str:
    if config.device == "cuda" and torch.cuda.is_available(): return config.device
    return "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def _default_lerobot_home() -> Path:
    user = os.environ.get("USER", "default_user")
    cache_root = RUNTIME_CACHE_DIR / f"xvla_{user}"
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(cache_root / "assets"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(cache_root / "lerobot"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_ASSETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_LEROBOT_HOME"]).mkdir(parents=True, exist_ok=True)
    return Path(os.environ["HF_LEROBOT_HOME"])


def _resolve_dataset_root(config: VisualThoughtTrainConfig) -> Path | None:
    return Path(config.dataset_root) / config.dataset_repo_id if config.dataset_root else _default_lerobot_home() / config.dataset_repo_id


def _resolve_teacher_image_key(camera_keys: list[str], rename_map: dict[str, str], requested_key: str) -> str:
    reverse = {dst: src for src, dst in rename_map.items()}
    if requested_key in reverse: return reverse[requested_key]
    if requested_key in camera_keys: return requested_key
    if reverse.get("observation.images.image") is not None: return reverse["observation.images.image"]
    if "observation.images.image" in camera_keys: return "observation.images.image"
    return camera_keys[0]


def _freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters(): parameter.requires_grad = False


def _resolve_policy_device(config: VisualThoughtTrainConfig) -> str:
    return "cpu" if config.training_stage == "distill_only" else _as_device(config)


def _normalize_mode_name(value: str) -> str:
    return str(value).replace("-", "_").strip().upper()


def validate_mean_std_normalization(mapping_str: str) -> None:
    try:
        mapping = json.loads(mapping_str)
    except json.JSONDecodeError as exc:
        raise ValueError("policy.normalization_mapping must be valid JSON. " f"Received: {mapping_str!r}") from exc
    action_mode = _normalize_mode_name(mapping.get("ACTION", ""))
    state_mode = _normalize_mode_name(mapping.get("STATE", ""))
    if action_mode != "MEAN_STD" or state_mode != "MEAN_STD":
        raise ValueError("XVLA training requires mean/std normalization for ACTION and STATE. " f"Received ACTION={action_mode or '<missing>'}, STATE={state_mode or '<missing>'}.")


def resolve_normalization_mapping(mapping_str: str | None) -> dict[str, str] | None:
    if mapping_str is None: return None
    validate_mean_std_normalization(mapping_str)
    mapping = json.loads(mapping_str)
    if not isinstance(mapping, dict): raise ValueError(f"policy.normalization_mapping must decode to a JSON object, got {type(mapping).__name__}.")
    return {str(key): str(value) for key, value in mapping.items()}


def apply_normalization_mapping_override(policy_cfg, mapping_str: str | None) -> None:
    mapping = resolve_normalization_mapping(mapping_str)
    if mapping is not None: policy_cfg.normalization_mapping = mapping


def resolve_resume_checkpoint(output_dir: str | Path, resume: bool = False, resume_checkpoint_path: str | Path | None = None) -> Path | None:
    if resume_checkpoint_path is not None:
        checkpoint_dir = Path(resume_checkpoint_path)
        if not checkpoint_dir.exists(): raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_dir}")
        return checkpoint_dir
    if not bool(resume): return None
    root = Path(output_dir)
    if not root.exists(): raise FileNotFoundError(f"Resume requested but output_dir does not exist: {root}")
    candidates = [path for path in root.glob("checkpoint_*") if path.is_dir()]
    if not candidates: raise FileNotFoundError(f"Resume requested but no checkpoint_* directories were found under {root}")

    def _sort_key(path: Path) -> tuple[int, int]:
        if path.name == "checkpoint_final": return (1, 0)
        suffix = path.name.removeprefix("checkpoint_")
        return (0, int(suffix)) if suffix.isdigit() else (-1, -1)

    return max(candidates, key=_sort_key)


def load_saved_trainer_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    trainer_state_path = Path(checkpoint_dir) / TRAINER_STATE_FILENAME
    if not trainer_state_path.is_file(): raise FileNotFoundError(f"Missing trainer state file: {trainer_state_path}")
    return torch.load(trainer_state_path, map_location="cpu")


def restore_optimizer_state(optimizer: torch.optim.Optimizer | JointTrainingState, trainer_state: dict[str, Any]) -> None:
    state_format = trainer_state.get("format")
    if isinstance(optimizer, JointTrainingState):
        if state_format == "joint_policy_decoder_v1":
            optimizer.policy_optimizer.load_state_dict(trainer_state["optimizers"]["policy"])
            optimizer.decoder_optimizer.load_state_dict(trainer_state["optimizers"]["decoder"])
            if optimizer.policy_scheduler is not None and "schedulers" in trainer_state and "policy" in trainer_state["schedulers"]: optimizer.policy_scheduler.load_state_dict(trainer_state["schedulers"]["policy"])
            return
        raise ValueError(f"Unsupported joint trainer state format: {state_format!r}")
    if state_format in {None, "single_optimizer_v1"} and "optimizer" in trainer_state:
        optimizer.load_state_dict(trainer_state["optimizer"])
        return
    raise ValueError(f"Unsupported single-optimizer trainer state format: {state_format!r}")


def should_run_validation_step(step: int, total_steps: int, validation_freq: int, emitted_steps: set[int]) -> bool:
    if step in emitted_steps: return False
    if step in FORCED_VALIDATION_STEPS: return True
    return step == int(total_steps) or step % max(int(validation_freq), 1) == 0


def _resume_check(field_name: str, current_value, saved_value) -> None:
    if saved_value is None or current_value is None: return
    if field_name == "normalization_mapping":
        current_norm = resolve_normalization_mapping(str(current_value))
        saved_norm = resolve_normalization_mapping(str(saved_value))
    else:
        current_norm = tuple(current_value) if isinstance(current_value, (list, tuple)) else current_value
        saved_norm = tuple(saved_value) if isinstance(saved_value, (list, tuple)) else saved_value
    if current_norm != saved_norm: raise ValueError(f"Resume checkpoint is incompatible for {field_name}: current={current_norm!r}, saved={saved_norm!r}.")


def assert_visual_thought_resume_compatible(config: VisualThoughtTrainConfig, snapshot: dict[str, Any]) -> None:
    _resume_check("training_stage", config.training_stage, snapshot.get("training_stage"))
    _resume_check("expert_type", config.expert_type, snapshot.get("expert_type"))
    _resume_check("expert_types", resolve_expert_types(config), snapshot.get("expert_types"))
    _resume_check("normalization_mapping", config.normalization_mapping, snapshot.get("normalization_mapping"))


def _apply_xvla_training_overrides(policy_cfg, config: VisualThoughtTrainConfig) -> None:
    overrides = {
        "adaptation_mode": config.xvla_adaptation_mode,
        "freeze_steps": config.xvla_freeze_steps,
        "warmup_steps": config.xvla_warmup_steps,
        "learning_coef": config.xvla_learning_coef,
        "optimizer_lr": config.xvla_optimizer_lr,
        "optimizer_soft_prompt_lr_scale": config.xvla_optimizer_soft_prompt_lr_scale,
        "optimizer_soft_prompt_warmup_lr_scale": config.xvla_optimizer_soft_prompt_warmup_lr_scale,
        "scheduler_warmup_steps": config.xvla_scheduler_warmup_steps,
        "scheduler_decay_steps": config.xvla_scheduler_decay_steps,
        "scheduler_decay_lr": config.xvla_scheduler_decay_lr,
    }
    for key, value in overrides.items():
        if value is not None: setattr(policy_cfg, key, value)


def _move_inputs_for_vlm(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _apply_student_cutout(image_input: torch.Tensor, config: VisualThoughtTrainConfig) -> torch.Tensor:
    if not bool(config.cutout_enable) or float(config.cutout_prob) <= 0.0 or int(config.cutout_num_patches) <= 0: return image_input
    out = image_input.clone(); _, _, _, height, width = out.shape
    area_min, area_max = float(config.cutout_area_min), float(config.cutout_area_max)
    aspect_min, aspect_max = float(config.cutout_aspect_min), float(config.cutout_aspect_max)
    fill = float(config.cutout_fill)
    if area_min <= 0.0 and area_max <= 0.0: return out
    area_min, area_max = sorted((max(area_min, 0.0), max(area_max, 0.0)))
    aspect_min, aspect_max = sorted((max(aspect_min, 1e-6), max(aspect_max, 1e-6)))
    for batch_idx in range(int(out.shape[0])):
        for view_idx in range(int(out.shape[1])):
            if torch.rand((), device=out.device).item() > float(config.cutout_prob): continue
            for _ in range(max(int(config.cutout_num_patches), 0)):
                patch_area = torch.empty((), device=out.device).uniform_(area_min, area_max).item() * float(height * width)
                patch_aspect = torch.empty((), device=out.device).uniform_(aspect_min, aspect_max).item()
                patch_h = max(1, min(int(round((patch_area / max(patch_aspect, 1e-8)) ** 0.5)), height))
                patch_w = max(1, min(int(round((patch_area * patch_aspect) ** 0.5)), width))
                y0 = 0 if patch_h >= height else int(torch.randint(0, height - patch_h + 1, (), device=out.device).item())
                x0 = 0 if patch_w >= width else int(torch.randint(0, width - patch_w + 1, (), device=out.device).item())
                out[batch_idx, view_idx, :, y0:y0 + patch_h, x0:x0 + patch_w] = fill
    return out


def _extract_vlm_features(runtime: XVLARuntime, processed_batch: dict[str, Any], config: VisualThoughtTrainConfig, apply_cutout: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = runtime.policy._build_model_inputs(processed_batch)
    if apply_cutout and runtime.vlm_only_distill and config.training_stage == "distill_only": inputs = {**inputs, "image_input": _apply_student_cutout(inputs["image_input"], config)}
    if not runtime.vlm_only_distill: return inputs, runtime.policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
    vlm_inputs = _move_inputs_for_vlm({key: inputs[key] for key in ("input_ids", "image_input", "image_mask")}, runtime.vlm_device)
    with torch.no_grad():
        enc = runtime.policy.model.forward_vlm(input_ids=vlm_inputs["input_ids"], pixel_values=vlm_inputs["image_input"], image_mask=vlm_inputs["image_mask"])
    return inputs, enc


def build_xvla_runtime(config: VisualThoughtTrainConfig, policy_source_path: str | Path | None = None) -> XVLARuntime:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    dataset_root = _resolve_dataset_root(config)
    pretrained_path = str(policy_source_path or config.xvla_init_path)
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    apply_normalization_mapping_override(policy_cfg, config.normalization_mapping)
    ds_meta = LeRobotDatasetMetadata(config.dataset_repo_id, root=dataset_root, revision=config.dataset_revision)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(config.dataset_repo_id, root=dataset_root, revision=config.dataset_revision, delta_timestamps=delta_timestamps, video_backend=config.dataset_video_backend, tolerance_s=float(config.dataset_tolerance_s))
    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    _apply_xvla_training_overrides(policy_cfg, config)
    policy_device = _resolve_policy_device(config)
    if hasattr(policy_cfg, "device"): policy_cfg.device = policy_device
    policy = XVLAPolicy.from_pretrained(pretrained_path, config=policy_cfg, device=policy_device)
    vlm_only_distill = config.training_stage == "distill_only"
    if vlm_only_distill:
        policy.eval()
        _freeze_module(policy)
        policy.model.vlm.to(_as_device(config))
        policy.model.vlm.eval()
        vlm_device = _as_device(config)
        processor_device = "cpu"
    else:
        policy = policy.to(dtype=torch.float32)
        vlm_device = _as_device(config)
        processor_device = _as_device(config)
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=pretrained_path, device=processor_device, rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True)
    ensure_xvla_slice_step(preprocessor, get_so101_slice_spec(getattr(policy.config, "action_mode", None)))
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), rename_map, config.teacher_image_feature_key)
    return XVLARuntime(policy=policy, dataset=dataset, preprocessor=preprocessor, postprocessor=postprocessor, rename_map=rename_map, teacher_image_key=teacher_image_key, policy_device=policy_device, vlm_device=vlm_device, vlm_only_distill=vlm_only_distill)


def build_dataloader(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> DataLoader:
    return DataLoader(runtime.dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)


def _build_loader_for_subset(dataset, config: VisualThoughtTrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)


def _episode_split_indices(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> tuple[list[int], list[int]] | None:
    dataset = runtime.dataset
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None: return None
    episode_column = hf_dataset["episode_index"]
    episode_indices = [int(value.item()) if torch.is_tensor(value) else int(value) for value in episode_column]
    unique_episodes = sorted(set(episode_indices))
    if len(unique_episodes) < 2: return None
    val_episode_len = int(len(unique_episodes) * float(config.validation_split_ratio))
    if val_episode_len <= 0 or val_episode_len >= len(unique_episodes): return None
    generator = torch.Generator().manual_seed(int(config.validation_seed))
    permutation = torch.randperm(len(unique_episodes), generator=generator).tolist()
    val_episode_ids = {unique_episodes[index] for index in permutation[:val_episode_len]}
    train_indices, val_indices = [], []
    for sample_index, episode_index in enumerate(episode_indices):
        (val_indices if episode_index in val_episode_ids else train_indices).append(sample_index)
    if not train_indices or not val_indices: return None
    return train_indices, val_indices


def build_train_val_dataloaders(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DataLoader | None, list[int]]:
    if not config.validation_enable or float(config.validation_split_ratio) <= 0.0:
        loader = build_dataloader(runtime, config)
        vis_indices = _select_fixed_vis_indices(len(runtime.dataset), config.seed, config.vis_num_samples)
        vis_loader = DataLoader(Subset(runtime.dataset, vis_indices), batch_size=max(1, min(int(config.batch_size), len(vis_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_indices else None
        return loader, None, None, vis_loader, vis_indices
    episode_split = _episode_split_indices(runtime, config)
    if episode_split is None:
        loader = build_dataloader(runtime, config)
        vis_indices = _select_fixed_vis_indices(len(runtime.dataset), config.seed, config.vis_num_samples)
        vis_loader = DataLoader(Subset(runtime.dataset, vis_indices), batch_size=max(1, min(int(config.batch_size), len(vis_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_indices else None
        return loader, None, None, vis_loader, vis_indices
    train_indices, val_indices = episode_split
    train_dataset, val_dataset = Subset(runtime.dataset, train_indices), Subset(runtime.dataset, val_indices)
    train_loader = _build_loader_for_subset(train_dataset, config, shuffle=True)
    val_loader = _build_loader_for_subset(val_dataset, config, shuffle=False)
    vis_subset_indices = _select_fixed_vis_indices(len(val_dataset), config.seed, config.vis_num_samples)
    vis_loader = DataLoader(Subset(val_dataset, vis_subset_indices), batch_size=max(1, min(int(config.batch_size), len(vis_subset_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_subset_indices else None
    return train_loader, val_loader, None, vis_loader, [val_indices[index] for index in vis_subset_indices]


def preprocess_batch(runtime: XVLARuntime, raw_batch: dict[str, Any]) -> dict[str, Any]:
    return runtime.preprocessor(raw_batch)


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


def get_teacher_images(raw_batch: dict[str, Any], teacher_image_key: str) -> torch.Tensor:
    images = raw_batch[teacher_image_key]
    return images[:, -1] if getattr(images, "ndim", 0) == 5 else images


def set_policy_trainability(policy, training_stage: TrainingStage) -> None:
    requires_grad = training_stage == "joint_multitask"
    for parameter in policy.parameters(): parameter.requires_grad = requires_grad
    if requires_grad: policy.train()
    else: policy.eval()


def build_teacher(config: VisualThoughtTrainConfig, runtime: XVLARuntime | None = None, expert_type: ExpertType | None = None):
    expert_type = expert_type or config.expert_type
    stack_path, task_path = decoder_config_paths_for(config, expert_type)
    if expert_type == "cedirnet":
        task_cfg = load_cedirnet_decoder_config(stack_path, task_path)
        dataset_length = len(runtime.dataset) if runtime is not None else None
        if dataset_length is None: raise ValueError("CeDiRNet cached teacher resolution requires an initialized runtime dataset.")
        cache = CeDiRNetTargetCache.resolve(dataset_repo_id=config.dataset_repo_id, dataset_revision=config.dataset_revision, dataset_root=_resolve_dataset_root(config), dataset_length=dataset_length, teacher_cfg=task_cfg.teacher, cache_root=config.teacher_target_cache_root)
        return task_cfg, cache
    task_cfg = load_dino_decoder_config(stack_path, task_path)
    if is_combined_expert_run(config) and getattr(task_cfg.teacher, "target_kind", None) != "token_sequence": raise ValueError("Combined CeDirNet+DINO runs require DINO teacher.target_kind='token_sequence'.")
    return task_cfg, DinoV2Teacher(task_cfg.teacher)


def build_teacher_bundle(config: VisualThoughtTrainConfig, runtime: XVLARuntime) -> dict[ExpertType, tuple[Any, Any]]:
    return {expert_type: build_teacher(config, runtime, expert_type=expert_type) for expert_type in resolve_expert_types(config)}


def load_teacher_target(config: VisualThoughtTrainConfig, teacher_source, raw_batch: dict[str, Any], teacher_image_key: str, expert_type: ExpertType | None = None) -> TeacherTarget:
    expert_type = expert_type or config.expert_type
    if expert_type == "cedirnet": return teacher_source.target_for_batch(raw_batch, device=_as_device(config))
    return teacher_source.predict(get_teacher_images(raw_batch, teacher_image_key).to(_as_device(config)))


def build_decoder(config: VisualThoughtTrainConfig, task_cfg, target: TeacherTarget, student_vlm_dim: int, expert_type: ExpertType | None = None):
    expert_type = expert_type or config.expert_type
    if expert_type == "cedirnet": return CeDirNetDistillationModel.from_config(student_vlm_dim=student_vlm_dim, cfg=task_cfg)
    if target.kind == "token_sequence": return DinoTokenSequenceModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)
    return DinoFeatureAlignmentModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)


def load_decoder_init_if_present(decoder: torch.nn.Module, decoder_init_path: str | None) -> None:
    if not decoder_init_path: return
    root = Path(decoder_init_path)
    state = load_decoder_state(root / DECODER_STATE_FILENAME if root.is_dir() else root)
    decoder.load_state_dict(state, strict=True)


def _visual_thought_policy_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    policy_dir = Path(checkpoint_dir) / POLICY_DIRNAME
    if not policy_dir.is_dir(): raise FileNotFoundError(f"Missing visual-thought policy directory: {policy_dir}")
    return policy_dir


def load_visual_thought_decoder_checkpoint(decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module], checkpoint_dir: str | Path) -> None:
    root = Path(checkpoint_dir)
    if isinstance(decoder, dict):
        for expert_type, module in decoder.items():
            state_path = root / DECODER_STATE_TEMPLATE.format(expert=str(expert_type))
            if not state_path.is_file(): raise FileNotFoundError(f"Missing decoder checkpoint for expert={expert_type}: {state_path}")
            module.load_state_dict(load_decoder_state(state_path), strict=True)
        return
    state_path = root / DECODER_STATE_FILENAME
    if not state_path.is_file(): raise FileNotFoundError(f"Missing decoder checkpoint: {state_path}")
    decoder.load_state_dict(load_decoder_state(state_path), strict=True)

def build_decoder_bundle(config: VisualThoughtTrainConfig, runtime: XVLARuntime, teacher_bundle: dict[ExpertType, tuple[Any, Any]]) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DataLoader | None, list[int], dict[ExpertType, torch.nn.Module], dict[ExpertType, TeacherTarget]]:
    loader, val_loader, val_loader_episode, vis_loader, vis_indices = build_train_val_dataloaders(runtime, config)
    first_raw_batch = next(iter(loader))
    processed = preprocess_batch(runtime, first_raw_batch)
    _, enc = build_xvla_inputs(runtime, processed, config)
    decoders, targets = {}, {}
    for expert_type, (task_cfg, teacher_source) in teacher_bundle.items():
        target = load_teacher_target(config, teacher_source, first_raw_batch, runtime.teacher_image_key, expert_type=expert_type)
        decoder = build_decoder(config, task_cfg, target, student_vlm_dim=int(enc["vlm_features"].shape[-1]), expert_type=expert_type).to(_as_device(config))
        load_decoder_init_if_present(decoder, decoder_init_path_for(config, expert_type))
        decoders[expert_type], targets[expert_type] = decoder, target
    return loader, val_loader, val_loader_episode, vis_loader, vis_indices, decoders, targets


def _decoder_parameters(decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module]) -> list[torch.nn.Parameter]:
    if isinstance(decoder, dict): decoder_params = [parameter for module in decoder.values() for parameter in module.parameters() if parameter.requires_grad]
    else: decoder_params = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    return decoder_params


def build_optimizer(config: VisualThoughtTrainConfig, policy, decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module]) -> torch.optim.Optimizer | JointTrainingState:
    decoder_params = _decoder_parameters(decoder)
    decoder_optimizer = torch.optim.AdamW(decoder_params, lr=config.decoder_optimizer_lr, weight_decay=config.weight_decay)
    if config.training_stage == "distill_only": return decoder_optimizer
    policy_optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    return JointTrainingState(policy_optimizer=policy_optimizer, decoder_optimizer=decoder_optimizer)


def build_policy_scheduler(config: VisualThoughtTrainConfig, policy, policy_optimizer: torch.optim.Optimizer) -> LRScheduler | None:
    if config.training_stage != "joint_multitask": return None
    return policy.config.get_scheduler_preset().build(policy_optimizer, config.steps)


def zero_optimizer_grad(optimizer: torch.optim.Optimizer | JointTrainingState) -> None:
    if isinstance(optimizer, JointTrainingState):
        optimizer.policy_optimizer.zero_grad(set_to_none=True)
        optimizer.decoder_optimizer.zero_grad(set_to_none=True)
        return
    optimizer.zero_grad(set_to_none=True)


def step_optimizer(optimizer: torch.optim.Optimizer | JointTrainingState) -> None:
    if isinstance(optimizer, JointTrainingState):
        optimizer.policy_optimizer.step()
        optimizer.decoder_optimizer.step()
        if optimizer.policy_scheduler is not None: optimizer.policy_scheduler.step()
        return
    optimizer.step()


def trainer_state_dict(optimizer: torch.optim.Optimizer | JointTrainingState) -> dict[str, Any]:
    if isinstance(optimizer, JointTrainingState):
        state = {
            "format": "joint_policy_decoder_v1",
            "optimizers": {
                "policy": optimizer.policy_optimizer.state_dict(),
                "decoder": optimizer.decoder_optimizer.state_dict(),
            },
        }
        if optimizer.policy_scheduler is not None: state["schedulers"] = {"policy": optimizer.policy_scheduler.state_dict()}
        return state
    return {"format": "single_optimizer_v1", "optimizer": optimizer.state_dict()}


def optimizer_metrics(optimizer: torch.optim.Optimizer | JointTrainingState) -> dict[str, float]:
    if isinstance(optimizer, JointTrainingState):
        metrics = {
            "policy_lr": float(optimizer.policy_optimizer.param_groups[0]["lr"]),
            "decoder_lr": float(optimizer.decoder_optimizer.param_groups[0]["lr"]),
        }
        for group in optimizer.policy_optimizer.param_groups:
            name = group.get("name")
            if name: metrics[f"policy_lr_{name}"] = float(group["lr"])
        return metrics
    return {"decoder_lr": float(optimizer.param_groups[0]["lr"])}


def build_xvla_inputs(runtime: XVLARuntime, processed_batch: dict[str, Any], config: VisualThoughtTrainConfig, apply_cutout: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return _extract_vlm_features(runtime, processed_batch, config, apply_cutout=apply_cutout)


def compute_xvla_action_loss_from_encoder(policy, processed_batch: dict[str, Any], inputs: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    targets = policy._prepare_action_targets(processed_batch)
    target_dtype = policy.model._get_target_dtype()
    t, action_noisy = policy.model._build_corrupted_action(action=targets, device=inputs["input_ids"].device, target_dtype=target_dtype)
    proprio_m, action_noisy_m = policy.model.action_space.preprocess(inputs["proprio"].to(dtype=target_dtype), action_noisy)
    transformer_parameters = policy.model.transformer.parameters() if hasattr(policy.model.transformer, "parameters") else iter(())
    transformer_parameter = next(transformer_parameters, None)
    transformer_dtype = transformer_parameter.dtype if transformer_parameter is not None else action_noisy_m.dtype
    action_noisy_m = action_noisy_m.to(dtype=transformer_dtype)
    proprio_m = proprio_m.to(dtype=transformer_dtype)
    t = t.to(dtype=transformer_dtype)
    enc = {key: value.to(dtype=transformer_dtype) if torch.is_tensor(value) and value.is_floating_point() else value for key, value in enc.items()}
    pred_action = policy.model.transformer(domain_id=inputs["domain_id"], action_with_noise=action_noisy_m, t=t, proprio=proprio_m, **enc)
    loss_dict = policy.model.action_space.compute_loss(pred_action, targets)
    action_loss = sum(loss_dict.values())
    stats = {key: float(value.detach().item()) for key, value in loss_dict.items()}
    stats["action_total"] = float(action_loss.detach().item())
    return action_loss, stats


def compute_expert_loss(config: VisualThoughtTrainConfig, decoder: torch.nn.Module, task_cfg, target: TeacherTarget, vlm_features: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
    # In joint_multitask the expert loss MUST flow gradient back into the VLM so the
    # auxiliary task actually shapes the backbone features. Only detach in distill_only,
    # where the policy is frozen anyway and the decoder is the sole thing being trained.
    if config.training_stage == "distill_only": vlm_features = vlm_features.detach()
    
    if config.expert_type == "cedirnet":
        prediction  = decoder(vlm_features, target_map=target.tensor)
        loss_target = TeacherTarget(name=target.name, tensor=target.tensor.to(dtype=prediction.dtype), kind=target.kind, loss_type=target.loss_type, weight=target.weight, aux=target.aux)
        loss        = compute_teacher_loss(prediction, loss_target).float()
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 0.0}
    
    if target.kind == "token_sequence":
        prediction  = decoder(vlm_features)
        loss        = compute_teacher_loss(prediction, target)
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 0.0}
    
    query_tokens = decoder.query_tokens(vlm_features)
    if int(step) <= int(config.align_feature_until_step):
        attended_stu, teacher_aligned_exp = decoder.query_align_features(query_tokens, target)
        loss = compute_feature_alignment_loss(attended_stu, teacher_aligned_exp, task_cfg.head.align_weight)
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 1.0}
    
    if target.tensor.ndim < 4: 
        raise ValueError("DINO reconstruction phase requires a spatial target.tensor. Keep align_feature_until_step >= steps when using alignment-only teacher targets.")
    
    _, prediction = decoder.query_reconstruct(query_tokens, target)
    recon = compute_teacher_loss(prediction, target)
    loss = float(task_cfg.head.recon_weight) * float(task_cfg.head.recon_scale) * recon
    return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 2.0}


def compute_expert_losses(config: VisualThoughtTrainConfig, decoders: dict[ExpertType, torch.nn.Module], task_cfgs: dict[ExpertType, Any], targets: dict[ExpertType, TeacherTarget], vlm_features: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = vlm_features.new_zeros(())
    stats: dict[str, float] = {}
    combined_total = 0.0
    for expert_type in resolve_expert_types(config):
        expert_loss, expert_stats = compute_expert_loss(VisualThoughtTrainConfig(**{**config.to_json_dict(), "expert_type": expert_type, "expert_types": None}), decoders[expert_type], task_cfgs[expert_type], targets[expert_type], vlm_features, step)
        total_loss = total_loss + expert_loss_weight_for(config, expert_type) * expert_loss
        combined_total += float(expert_loss.detach().item())
        prefix = f"{expert_type}_"
        for key, value in expert_stats.items(): stats[f"{prefix}{key}"] = float(value)
    stats["expert_total_combined"] = combined_total
    return total_loss, stats


def prepare_models_and_target(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher):
    loader, val_loader, val_loader_episode, vis_loader, vis_indices, decoders, targets = build_decoder_bundle(config, runtime, {config.expert_type: (task_cfg, teacher)})
    return loader, val_loader, val_loader_episode, vis_loader, vis_indices, decoders[config.expert_type], targets[config.expert_type]


def _checkpoint_metadata(config: VisualThoughtTrainConfig, step: int) -> dict[str, Any]:
    payload = {"name": config.name, "training_stage": config.training_stage, "expert_type": config.expert_type, "expert_types": list(resolve_expert_types(config)), "global_step": int(step), "xvla_init_path": config.xvla_init_path, "decoder_init_path": config.decoder_init_path, "normalization_mapping": config.normalization_mapping, "wandb_run_id": config.wandb_run_id}
    if is_combined_expert_run(config):
        payload["decoder_init_paths"] = {"cedirnet": config.cedirnet_decoder_init_path, "dino": config.dino_decoder_init_path}
    return payload


def _hub_step_dir(step: int) -> str:
    return f"step_{int(step):07d}"


def _push_checkpoint_to_hub(checkpoint_dir: Path, repo_id: str, step: int, commit_message: str) -> bool:
    path_in_repo = _hub_step_dir(step)
    result = push_folder_to_hub(folder_path=checkpoint_dir, repo_id=repo_id, repo_type="model", path_in_repo=path_in_repo, commit_message=commit_message, ignore_patterns=[TRAINER_STATE_FILENAME], upload_config=TRAINING_HUB_UPLOAD_CONFIG)
    if result.ok:
        clear_hub_upload_failure_marker(checkpoint_dir)
        return True
    marker_path = write_hub_upload_failure_marker(folder_path=checkpoint_dir, repo_id=repo_id, repo_type="model", path_in_repo=path_in_repo, commit_message=commit_message, result=result)
    print(f"[push] WARNING: failed to upload checkpoint to {repo_id} after {result.attempts} attempts. Training will continue. Retry metadata saved to {marker_path}")
    return False


def _save_checkpoint_if_needed(config: VisualThoughtTrainConfig, runtime: XVLARuntime, decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module], optimizer: torch.optim.Optimizer | JointTrainingState, step: int, final: bool = False) -> None:
    if not final and (config.save_every <= 0 or step % config.save_every != 0): return
    checkpoint_name = "checkpoint_final" if final else f"checkpoint_{step:07d}"
    checkpoint_dir = Path(config.output_dir) / checkpoint_name
    stack_cfg_path = {"cedirnet": config.cedirnet_decoder_stack_config_path, "dino": config.dino_decoder_stack_config_path} if is_combined_expert_run(config) else config.decoder_stack_config_path
    task_cfg_path = {"cedirnet": config.cedirnet_decoder_task_config_path, "dino": config.dino_decoder_task_config_path} if is_combined_expert_run(config) else config.decoder_task_config_path
    save_visual_thought_checkpoint(checkpoint_dir=checkpoint_dir, policy=runtime.policy, decoder=decoder, trainer_state={"step": int(step), "wandb_run_id": config.wandb_run_id, **trainer_state_dict(optimizer)}, metadata=_checkpoint_metadata(config, step), config_snapshot=config.to_json_dict(), preprocessor=runtime.preprocessor, postprocessor=runtime.postprocessor, decoder_stack_config_path=stack_cfg_path, decoder_task_config_path=task_cfg_path)
    should_push = bool(config.push_to_hub) and bool(config.push_repo_id) and (final or (int(config.push_every) > 0 and step % int(config.push_every) == 0))
    if should_push: _push_checkpoint_to_hub(checkpoint_dir, str(config.push_repo_id), step, f"Upload visual-thought checkpoint {checkpoint_name}")


def _maybe_init_wandb(config: VisualThoughtTrainConfig):
    if not config.wandb_enable: return None
    import wandb

    kwargs = {"project": config.wandb_project, "name": config.wandb_run_name or config.name, "config": config.to_json_dict(), "dir": config.output_dir}
    if config.wandb_run_id is not None: kwargs.update({"id": config.wandb_run_id, "resume": "must"})
    run = wandb.init(**kwargs)
    config.wandb_run_id = getattr(run, "id", config.wandb_run_id)
    return run


@torch.no_grad()
def run_validation(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher, decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module], val_loader: DataLoader, prefix: str = "val") -> dict[str, float]:
    decoder_was_training = {expert_type: module.training for expert_type, module in decoder.items()} if isinstance(decoder, dict) else decoder.training
    policy_was_training = runtime.policy.training
    if isinstance(decoder, dict):
        for module in decoder.values(): module.eval()
    else:
        decoder.eval()
    runtime.policy.eval()
    total_loss, total_action, total_expert, batches = 0.0, 0.0, 0.0, 0
    component_totals: dict[str, float] = {}
    for raw_batch in val_loader:
        if batches >= max(int(config.validation_max_batches), 1): break
        processed_batch = preprocess_batch(runtime, raw_batch)
        inputs, enc = build_xvla_inputs(runtime, processed_batch, config)
        if isinstance(decoder, dict):
            targets = {expert_type: load_teacher_target(config, teacher[expert_type], raw_batch, runtime.teacher_image_key, expert_type=expert_type) for expert_type in resolve_expert_types(config)}
            expert_loss, expert_stats = compute_expert_losses(config, decoder, task_cfg, targets, enc["vlm_features"], step=0)
            total = expert_loss
        else:
            target = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
            expert_loss, expert_stats = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], step=0)
            total = float(config.expert_loss_weight) * expert_loss
        action_value = 0.0
        for key, value in expert_stats.items(): component_totals[f"{prefix}_{key}"] = component_totals.get(f"{prefix}_{key}", 0.0) + float(value)
        if config.training_stage == "joint_multitask":
            action_loss, action_stats = compute_xvla_action_loss_from_encoder(runtime.policy, processed_batch, inputs, enc)
            action_value = float(action_loss.detach().item()); total = float(config.action_loss_weight) * action_loss + total
            for key, value in action_stats.items(): component_totals[f"{prefix}_{key}"] = component_totals.get(f"{prefix}_{key}", 0.0) + float(value)
        total_loss += float(total.detach().item())
        total_action += action_value
        total_expert += float(expert_loss.detach().item())
        batches += 1
    if isinstance(decoder, dict):
        for expert_type, module in decoder.items():
            if decoder_was_training[expert_type]: module.train()
    elif decoder_was_training:
        decoder.train()
    if policy_was_training: runtime.policy.train()
    denom = max(batches, 1)
    metrics = {f"{prefix}_loss": total_loss / denom, f"{prefix}_expert_total": total_expert / denom, f"{prefix}_action_total": total_action / denom, f"{prefix}_batches": float(batches)}
    metrics.update({key: value / denom for key, value in component_totals.items()})
    return metrics


def load_visual_thought_resume_payload(config: VisualThoughtTrainConfig) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    checkpoint_dir = resolve_resume_checkpoint(config.output_dir, resume=config.resume, resume_checkpoint_path=config.resume_checkpoint_path)
    if checkpoint_dir is None: return None
    snapshot = load_visual_thought_config_snapshot(checkpoint_dir) if (Path(checkpoint_dir) / CONFIG_FILENAME).is_file() else {}
    assert_visual_thought_resume_compatible(config, snapshot)
    trainer_state = load_saved_trainer_state(checkpoint_dir)
    saved_run_id = trainer_state.get("wandb_run_id") or snapshot.get("wandb_run_id")
    if saved_run_id is not None and config.wandb_run_id is None: config.wandb_run_id = str(saved_run_id)
    return Path(checkpoint_dir), snapshot, trainer_state


def restore_visual_thought_training_state(decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module], optimizer: torch.optim.Optimizer | JointTrainingState, checkpoint_dir: str | Path, trainer_state: dict[str, Any]) -> int:
    load_visual_thought_decoder_checkpoint(decoder, checkpoint_dir)
    restore_optimizer_state(optimizer, trainer_state)
    return int(trainer_state.get("step", 0))


class VisualThoughtModule(torch.nn.Module):
    """Bundles policy + decoder so a single DDP-wrapped forward touches every trainable
    parameter. Required for joint_multitask multi-GPU: the trainer calls policy submodules
    directly (forward_vlm / transformer), so without one wrapped forward the policy grads
    would never all-reduce across ranks."""

    def __init__(self, policy, decoder: torch.nn.Module | dict[ExpertType, torch.nn.Module], config: VisualThoughtTrainConfig, task_cfg) -> None:
        super().__init__()
        self.policy = policy
        self.decoder = torch.nn.ModuleDict(decoder) if isinstance(decoder, dict) else decoder
        self._config = config
        self._task_cfg = task_cfg

    def forward(self, processed_batch: dict[str, Any], target: TeacherTarget | dict[ExpertType, TeacherTarget], step: int):
        inputs = self.policy._build_model_inputs(processed_batch)
        enc = self.policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
        if isinstance(target, dict):
            expert_loss, expert_stats = compute_expert_losses(self._config, dict(self.decoder.items()), self._task_cfg, target, enc["vlm_features"], step)
        else:
            expert_loss, expert_stats = compute_expert_loss(self._config, self.decoder, self._task_cfg, target, enc["vlm_features"], step)
        action_loss, action_stats = compute_xvla_action_loss_from_encoder(self.policy, processed_batch, inputs, enc)
        total_loss = float(self._config.action_loss_weight) * action_loss + expert_loss if isinstance(target, dict) else float(self._config.action_loss_weight) * action_loss + float(self._config.expert_loss_weight) * expert_loss
        return total_loss, expert_stats, action_stats


def train_visual_thought(config: VisualThoughtTrainConfig) -> None:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    _validate_config(config)
    accum_steps = max(int(config.gradient_accumulation_steps), 1)
    find_unused_parameters = not (config.training_stage == "joint_multitask" and not is_combined_expert_run(config))
    accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)], gradient_accumulation_steps=accum_steps)
    # Pin this rank's CUDA device so the existing `_as_device()` -> "cuda" resolves to
    # cuda:local_rank (Accelerator already set it, but be explicit for the policy/VLM loads).
    if torch.cuda.is_available(): torch.cuda.set_device(accelerator.local_process_index)
    is_main = accelerator.is_main_process
    is_joint = config.training_stage == "joint_multitask"

    _set_seed(config.seed)
    if is_main: Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    resume_payload = load_visual_thought_resume_payload(config)
    wandb_run = _maybe_init_wandb(config) if is_main else None

    resume_checkpoint_dir = resume_payload[0] if resume_payload is not None else None
    policy_source_path = _visual_thought_policy_checkpoint_dir(resume_checkpoint_dir) if resume_checkpoint_dir is not None else None
    runtime = build_xvla_runtime(config, policy_source_path=policy_source_path)
    accelerator.wait_for_everyone()

    set_policy_trainability(runtime.policy, config.training_stage)
    if is_combined_expert_run(config):
        teacher_bundle = build_teacher_bundle(config, runtime)
        loader, val_loader, val_loader_episode, vis_loader, vis_indices, decoder, _ = build_decoder_bundle(config, runtime, teacher_bundle)
        task_cfg, teacher = {expert_type: bundle[0] for expert_type, bundle in teacher_bundle.items()}, {expert_type: bundle[1] for expert_type, bundle in teacher_bundle.items()}
    else:
        task_cfg, teacher = build_teacher(config, runtime)
        loader, val_loader, val_loader_episode, vis_loader, vis_indices, decoder, _ = prepare_models_and_target(config, runtime, task_cfg, teacher)
    if is_main and vis_loader is not None: print(json.dumps({"event": "visualization_config", "vis_fixed_indices": vis_indices, "vis_num_samples": int(config.vis_num_samples), "vis_every": int(config.vis_every)}))
    if is_joint:
        if isinstance(decoder, dict):
            for module in decoder.values():
                for parameter in module.parameters(): parameter.requires_grad = True
        else:
            for parameter in decoder.parameters(): parameter.requires_grad = True
        runtime.policy.train()
    else:
        if isinstance(decoder, dict):
            for module in decoder.values(): module.train()
        else:
            decoder.train()

    # Distill multi-GPU DDP-wraps the decoder, so it must be driven through `forward`.
    # cedirnet (dense_map) and dino token_sequence are forward-based; dino feature-alignment
    # uses custom decoder methods that a DDP wrapper would hide -> unsupported on >1 GPU.
    if (not is_joint) and accelerator.num_processes > 1 and config.expert_type == "dino" and getattr(task_cfg.teacher, "target_kind", None) != "token_sequence":
        raise SystemExit("Multi-GPU distill supports dino target_kind='token_sequence' only (feature-alignment uses custom decoder methods incompatible with DDP-wrapping the decoder). Use launch_mode='single' for that expert.")

    optimizer = build_optimizer(config, runtime.policy, decoder)

    if config.dry_run:
        if wandb_run is not None: wandb_run.finish()
        return

    if is_joint:
        train_module = VisualThoughtModule(runtime.policy, decoder, config, task_cfg)
        if not isinstance(optimizer, JointTrainingState): raise RuntimeError("joint_multitask requires split policy/decoder optimizers.")
        train_module, optimizer.policy_optimizer, optimizer.decoder_optimizer, loader = accelerator.prepare(train_module, optimizer.policy_optimizer, optimizer.decoder_optimizer, loader)
        optimizer.policy_scheduler = build_policy_scheduler(config, accelerator.unwrap_model(train_module).policy, optimizer.policy_optimizer)
        unwrapped_decoder = lambda: dict(accelerator.unwrap_model(train_module).decoder.items()) if is_combined_expert_run(config) else accelerator.unwrap_model(train_module).decoder
    else:
        if isinstance(decoder, dict): raise SystemExit("Combined expert_types mode is supported for joint_multitask only.")
        decoder, optimizer, loader = accelerator.prepare(decoder, optimizer, loader); train_module = decoder; unwrapped_decoder = lambda: accelerator.unwrap_model(decoder)

    step = restore_visual_thought_training_state(unwrapped_decoder(), optimizer, resume_checkpoint_dir, resume_payload[2]) if resume_payload is not None else 0
    zero_optimizer_grad(optimizer)
    progress = tqdm(total=max(int(config.steps) - int(step), 0), desc=config.name, disable=not is_main)
    if is_main: print(json.dumps({"event": "ddp_config", "find_unused_parameters": bool(find_unused_parameters), "gradient_accumulation_steps": int(accum_steps), "num_processes": int(accelerator.num_processes)}))
    emitted_validation_steps: set[int] = set()
    if is_main and step == 0 and val_loader is not None and should_run_validation_step(0, config.steps, config.validation_freq, emitted_validation_steps):
        val_metrics = {"event": "validation_step", "step": 0, **run_validation(config, runtime, task_cfg, teacher, unwrapped_decoder(), val_loader, prefix="val")}
        if val_loader_episode is not None: val_metrics.update(run_validation(config, runtime, task_cfg, teacher, unwrapped_decoder(), val_loader_episode, prefix="val_ep"))
        print(json.dumps(val_metrics))
        if wandb_run is not None: wandb_run.log({key: value for key, value in val_metrics.items() if key != "event"}, step=0)
        emitted_validation_steps.add(0)
    while step < config.steps:
        for raw_batch in loader:
            if step >= config.steps: break
            current_step = step + 1
            t0 = time.perf_counter()
            processed_batch = preprocess_batch(runtime, raw_batch)
            t1 = time.perf_counter()
            target = {expert_type: load_teacher_target(config, teacher[expert_type], raw_batch, runtime.teacher_image_key, expert_type=expert_type) for expert_type in resolve_expert_types(config)} if isinstance(decoder, dict) else load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
            t2 = time.perf_counter()
            with accelerator.accumulate(train_module):
                if is_joint:
                    total_loss, expert_stats, action_stats = train_module(processed_batch, target, current_step)
                else:
                    _, enc = build_xvla_inputs(runtime, processed_batch, config, apply_cutout=True)
                    expert_loss, expert_stats = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], current_step); total_loss, action_stats = float(config.expert_loss_weight) * expert_loss, {}
                accelerator.backward(total_loss)
                if not accelerator.sync_gradients: continue
                if torch.cuda.is_available() and int(config.profile_step_time_every) > 0 and current_step % max(int(config.profile_step_time_every), 1) == 0: torch.cuda.synchronize()
                t3 = time.perf_counter()
                accelerator.clip_grad_norm_(train_module.parameters(), XVLA_GRAD_CLIP_NORM)
                step_optimizer(optimizer)
                zero_optimizer_grad(optimizer)
                if torch.cuda.is_available() and int(config.profile_step_time_every) > 0 and current_step % max(int(config.profile_step_time_every), 1) == 0: torch.cuda.synchronize()
                t4 = time.perf_counter()

            step = current_step
            if is_main and step % max(int(config.log_every), 1) == 0:
                metrics = {"event": "train_step", "step": int(step), "loss": float(total_loss.detach().item()), **expert_stats, **action_stats, **optimizer_metrics(optimizer)}
                if int(config.profile_step_time_every) > 0 and step % max(int(config.profile_step_time_every), 1) == 0: metrics.update({"time_preprocess_s": t1 - t0, "time_teacher_s": t2 - t1, "time_forward_backward_s": t3 - t2, "time_step_s": t4 - t3, "time_total_s": t4 - t0})
                print(json.dumps(metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in metrics.items() if key != "event"}, step=int(step))
            if is_main and val_loader is not None and should_run_validation_step(step, config.steps, config.validation_freq, emitted_validation_steps):
                val_metrics = {"event": "validation_step", "step": int(step), **run_validation(config, runtime, task_cfg, teacher, unwrapped_decoder(), val_loader, prefix="val")}
                if val_loader_episode is not None: val_metrics.update(run_validation(config, runtime, task_cfg, teacher, unwrapped_decoder(), val_loader_episode, prefix="val_ep"))
                print(json.dumps(val_metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in val_metrics.items() if key != "event"}, step=int(step))
                emitted_validation_steps.add(int(step))
            if is_main and vis_loader is not None and int(config.vis_every) > 0 and step % max(int(config.vis_every), 1) == 0:
                vis_metrics = {"event": "visualization_step", "step": int(step), **run_visualization(config, runtime, teacher, unwrapped_decoder(), vis_loader, step)}
                print(json.dumps(vis_metrics))
            if is_main: _save_checkpoint_if_needed(config, runtime, unwrapped_decoder(), optimizer, step, final=False)
            progress.update(1)
    progress.close()
    accelerator.wait_for_everyone()
    if is_main and bool(config.vis_final) and vis_loader is not None:
        vis_metrics = {"event": "visualization_final", "step": int(step), **run_visualization(config, runtime, teacher, unwrapped_decoder(), vis_loader, step)}
        print(json.dumps(vis_metrics))
    if is_main and config.save_final_checkpoint:
        _save_checkpoint_if_needed(config, runtime, unwrapped_decoder(), optimizer, step, final=True)
    accelerator.wait_for_everyone()
    if wandb_run is not None: wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    train_visual_thought(VisualThoughtTrainConfig.from_json(args.config_path))


if __name__ == "__main__":
    main()
