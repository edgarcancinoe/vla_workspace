from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config
from thesis_vla.visual_thought import CeDirNetDistillationModel, DinoFeatureAlignmentModel, DinoTokenSequenceModel, compute_feature_alignment_loss, load_cedirnet_decoder_config, load_dino_decoder_config
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache
from thesis_vla.visual_thought.checkpoints import DECODER_STATE_FILENAME, POLICY_DIRNAME, load_decoder_state, load_visual_thought_checkpoint_metadata, save_visual_thought_checkpoint
from thesis_vla.visual_thought.targets import TeacherTarget, compute_teacher_loss
from thesis_vla.visual_thought.teachers import DinoV2Teacher


TrainingStage = Literal["distill_only", "joint_multitask"]
ExpertType = Literal["cedirnet", "dino"]

# Match the normal XVLA finetune loop (lerobot uses accelerator.accumulate, which averages
# the loss across the accumulation window and clips grad norm at the boundary). XVLA's
# optimizer default is grad_clip_norm=10.0 (configuration_xvla.optimizer_grad_clip_norm).
XVLA_GRAD_CLIP_NORM = 10.0


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
    cuda_visible_devices: tuple[int, ...] = (0,)
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    steps: int = 1_000
    log_every: int = 20
    save_every: int = 500
    weight_decay: float = 0.01
    decoder_optimizer_lr: float = 1e-4
    xvla_optimizer_lr: float = 1e-5
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 1.0
    teacher_image_feature_key: str = "observation.images.image"
    teacher_target_cache_root: str | None = None
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    wandb_enable: bool = False
    wandb_project: str = "visual-thought"
    wandb_run_name: str | None = None
    validation_enable: bool = False
    validation_split_ratio: float = 0.1
    validation_freq: int = 500
    validation_max_batches: int = 10
    validation_seed: int = 1337
    vis_every: int = 0
    vis_num_samples: int = 4
    vis_final: bool = True
    push_to_hub: bool = False
    push_repo_id: str | None = None
    push_every: int = 0
    align_feature_until_step: int = 0
    save_final_checkpoint: bool = True
    seed: int = 42
    dry_run: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "VisualThoughtTrainConfig":
        payload = json.loads(Path(path).read_text())
        if "cuda_visible_devices" in payload: payload["cuda_visible_devices"] = tuple(int(device) for device in payload["cuda_visible_devices"])
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
    map_np = map_hw.detach().cpu().float().numpy()
    lo = float(map_np.min()) if vmin is None else float(vmin); hi = float(map_np.max()) if vmax is None else float(vmax)
    denom = max(hi - lo, 1e-8); norm = np.clip((map_np - lo) / denom, 0.0, 1.0)
    heat = np.stack([norm, 1.0 - np.abs(2.0 * norm - 1.0), 1.0 - norm], axis=-1) * 255.0
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 255.0).astype(np.uint8)


@torch.no_grad()
def run_visualization(config: VisualThoughtTrainConfig, runtime: XVLARuntime, teacher, decoder: torch.nn.Module, loader: DataLoader, step: int) -> dict[str, Any]:
    if int(config.vis_num_samples) <= 0: return {"vis_skipped": "vis_num_samples"}
    decoder_was_training, policy_was_training = decoder.training, runtime.policy.training
    decoder.eval(); runtime.policy.eval()
    try:
        raw_batch = next(iter(loader))
        processed_batch = preprocess_batch(runtime, raw_batch)
        _, enc = build_xvla_inputs(runtime, processed_batch)
        target = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
        if config.expert_type != "dino" or target.kind != "token_sequence": return {"vis_skipped": f"{config.expert_type}:{target.kind}"}
        prediction = decoder(enc["vlm_features"])
        gh, gw = _resolve_grid_hw(target); sample_count = min(int(config.vis_num_samples), int(prediction.shape[0]))
        vis_root = Path(config.output_dir) / "visualizations" / f"step_{int(step):07d}"; vis_root.mkdir(parents=True, exist_ok=True)
        teacher_pca_root = Path(config.output_dir) / "teacher_pca"; teacher_pca_root.mkdir(parents=True, exist_ok=True)
        teacher_images = get_teacher_images(raw_batch, runtime.teacher_image_key); gallery = []
        pred_tokens = prediction.detach().cpu(); target_tokens = target.tensor.detach().cpu()
        cosine_maps = F.cosine_similarity(pred_tokens, target_tokens, dim=-1).view(pred_tokens.shape[0], gh, gw)
        error_maps = ((pred_tokens - target_tokens) ** 2).mean(dim=-1).view(pred_tokens.shape[0], gh, gw)
        sample_ids = raw_batch.get("index")
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


def _move_inputs_for_vlm(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _extract_vlm_features(runtime: XVLARuntime, processed_batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = runtime.policy._build_model_inputs(processed_batch)
    if not runtime.vlm_only_distill: return inputs, runtime.policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
    vlm_inputs = _move_inputs_for_vlm({key: inputs[key] for key in ("input_ids", "image_input", "image_mask")}, runtime.vlm_device)
    with torch.no_grad():
        enc = runtime.policy.model.forward_vlm(input_ids=vlm_inputs["input_ids"], pixel_values=vlm_inputs["image_input"], image_mask=vlm_inputs["image_mask"])
    return inputs, enc


def build_xvla_runtime(config: VisualThoughtTrainConfig) -> XVLARuntime:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    dataset = LeRobotDataset(config.dataset_repo_id, root=_resolve_dataset_root(config), revision=config.dataset_revision, video_backend=config.dataset_video_backend, tolerance_s=float(config.dataset_tolerance_s))
    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    policy_cfg = PreTrainedConfig.from_pretrained(config.xvla_init_path)
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    policy_device = _resolve_policy_device(config)
    if hasattr(policy_cfg, "device"): policy_cfg.device = policy_device
    policy = XVLAPolicy.from_pretrained(config.xvla_init_path, config=policy_cfg, device=policy_device)
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
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=config.xvla_init_path, device=processor_device, rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True)
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), rename_map, config.teacher_image_feature_key)
    return XVLARuntime(policy=policy, dataset=dataset, preprocessor=preprocessor, postprocessor=postprocessor, rename_map=rename_map, teacher_image_key=teacher_image_key, policy_device=policy_device, vlm_device=vlm_device, vlm_only_distill=vlm_only_distill)


def build_dataloader(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> DataLoader:
    return DataLoader(runtime.dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)


def build_train_val_dataloaders(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> tuple[DataLoader, DataLoader | None, DataLoader | None, list[int]]:
    if not config.validation_enable or float(config.validation_split_ratio) <= 0.0:
        loader = build_dataloader(runtime, config)
        vis_indices = _select_fixed_vis_indices(len(runtime.dataset), config.seed, config.vis_num_samples)
        vis_loader = DataLoader(Subset(runtime.dataset, vis_indices), batch_size=max(1, min(int(config.batch_size), len(vis_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_indices else None
        return loader, None, vis_loader, vis_indices
    dataset_len = len(runtime.dataset)
    val_len = min(max(int(round(dataset_len * float(config.validation_split_ratio))), 1), max(dataset_len - 1, 1))
    if dataset_len < 2 or val_len >= dataset_len:
        loader = build_dataloader(runtime, config)
        vis_indices = _select_fixed_vis_indices(len(runtime.dataset), config.seed, config.vis_num_samples)
        vis_loader = DataLoader(Subset(runtime.dataset, vis_indices), batch_size=max(1, min(int(config.batch_size), len(vis_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_indices else None
        return loader, None, vis_loader, vis_indices
    generator = torch.Generator().manual_seed(int(config.validation_seed))
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    val_indices, train_indices = permutation[:val_len], permutation[val_len:]
    train_dataset, val_dataset = Subset(runtime.dataset, train_indices), Subset(runtime.dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)
    vis_subset_indices = _select_fixed_vis_indices(len(val_dataset), config.seed, config.vis_num_samples)
    vis_loader = DataLoader(Subset(val_dataset, vis_subset_indices), batch_size=max(1, min(int(config.batch_size), len(vis_subset_indices))), shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False) if vis_subset_indices else None
    return train_loader, val_loader, vis_loader, [val_indices[index] for index in vis_subset_indices]


def preprocess_batch(runtime: XVLARuntime, raw_batch: dict[str, Any]) -> dict[str, Any]:
    return runtime.preprocessor(raw_batch)


def get_teacher_images(raw_batch: dict[str, Any], teacher_image_key: str) -> torch.Tensor:
    images = raw_batch[teacher_image_key]
    return images[:, -1] if getattr(images, "ndim", 0) == 5 else images


def set_policy_trainability(policy, training_stage: TrainingStage) -> None:
    requires_grad = training_stage == "joint_multitask"
    for parameter in policy.parameters(): parameter.requires_grad = requires_grad
    if requires_grad: policy.train()
    else: policy.eval()


def build_teacher(config: VisualThoughtTrainConfig, runtime: XVLARuntime | None = None):
    if config.expert_type == "cedirnet":
        task_cfg = load_cedirnet_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
        dataset_length = len(runtime.dataset) if runtime is not None else None
        if dataset_length is None: raise ValueError("CeDiRNet cached teacher resolution requires an initialized runtime dataset.")
        cache = CeDiRNetTargetCache.resolve(dataset_repo_id=config.dataset_repo_id, dataset_revision=config.dataset_revision, dataset_root=_resolve_dataset_root(config), dataset_length=dataset_length, teacher_cfg=task_cfg.teacher, cache_root=config.teacher_target_cache_root)
        return task_cfg, cache
    task_cfg = load_dino_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
    return task_cfg, DinoV2Teacher(task_cfg.teacher)


def load_teacher_target(config: VisualThoughtTrainConfig, teacher_source, raw_batch: dict[str, Any], teacher_image_key: str) -> TeacherTarget:
    if config.expert_type == "cedirnet": return teacher_source.target_for_batch(raw_batch, device=_as_device(config))
    return teacher_source.predict(get_teacher_images(raw_batch, teacher_image_key).to(_as_device(config)))


def build_decoder(config: VisualThoughtTrainConfig, task_cfg, target: TeacherTarget, student_vlm_dim: int):
    if config.expert_type == "cedirnet": 
        return CeDirNetDistillationModel.from_config(student_vlm_dim=student_vlm_dim, cfg=task_cfg)
    if target.kind == "token_sequence": 
        return DinoTokenSequenceModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)
    return DinoFeatureAlignmentModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)


def load_decoder_init_if_present(decoder: torch.nn.Module, decoder_init_path: str | None) -> None:
    if not decoder_init_path: return
    root = Path(decoder_init_path)
    state = load_decoder_state(root / DECODER_STATE_FILENAME if root.is_dir() else root)
    decoder.load_state_dict(state, strict=True)


def build_optimizer(config: VisualThoughtTrainConfig, policy, decoder: torch.nn.Module) -> torch.optim.Optimizer:
    if config.training_stage == "distill_only": return torch.optim.AdamW(decoder.parameters(), lr=config.decoder_optimizer_lr, weight_decay=config.weight_decay)
    decoder_params = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    xvla_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    return torch.optim.AdamW([{"params": decoder_params, "lr": config.decoder_optimizer_lr}, {"params": xvla_params, "lr": config.xvla_optimizer_lr}], weight_decay=config.weight_decay)


def build_xvla_inputs(runtime: XVLARuntime, processed_batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return _extract_vlm_features(runtime, processed_batch)


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


def prepare_models_and_target(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher):
    loader, val_loader, vis_loader, vis_indices = build_train_val_dataloaders(runtime, config)
    first_raw_batch = next(iter(loader))
    processed = preprocess_batch(runtime, first_raw_batch)
    inputs, enc = build_xvla_inputs(runtime, processed)
    target = load_teacher_target(config, teacher, first_raw_batch, runtime.teacher_image_key)
    decoder = build_decoder(config, task_cfg, target, student_vlm_dim=int(enc["vlm_features"].shape[-1])).to(_as_device(config))
    load_decoder_init_if_present(decoder, config.decoder_init_path)
    return loader, val_loader, vis_loader, vis_indices, decoder, target


def _checkpoint_metadata(config: VisualThoughtTrainConfig, step: int) -> dict[str, Any]:
    return {"name": config.name, "training_stage": config.training_stage, "expert_type": config.expert_type, "global_step": int(step), "xvla_init_path": config.xvla_init_path, "decoder_init_path": config.decoder_init_path}


def _push_checkpoint_to_hub(checkpoint_dir: Path, repo_id: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model", commit_message=commit_message)


def _save_checkpoint_if_needed(config: VisualThoughtTrainConfig, runtime: XVLARuntime, decoder: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, final: bool = False) -> None:
    if not final and (config.save_every <= 0 or step % config.save_every != 0): return
    checkpoint_name = "checkpoint_final" if final else f"checkpoint_{step:07d}"
    checkpoint_dir = Path(config.output_dir) / checkpoint_name
    save_visual_thought_checkpoint(checkpoint_dir=checkpoint_dir, policy=runtime.policy, decoder=decoder, trainer_state={"step": int(step), "optimizer": optimizer.state_dict()}, metadata=_checkpoint_metadata(config, step), config_snapshot=config.to_json_dict(), preprocessor=runtime.preprocessor, postprocessor=runtime.postprocessor)
    should_push = bool(config.push_to_hub) and bool(config.push_repo_id) and (final or (int(config.push_every) > 0 and step % int(config.push_every) == 0))
    if should_push: _push_checkpoint_to_hub(checkpoint_dir, str(config.push_repo_id), f"Upload visual-thought checkpoint {checkpoint_name}")


def _maybe_init_wandb(config: VisualThoughtTrainConfig):
    if not config.wandb_enable: return None
    import wandb

    return wandb.init(project=config.wandb_project, name=config.wandb_run_name or config.name, config=config.to_json_dict(), dir=config.output_dir)


@torch.no_grad()
def run_validation(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher, decoder: torch.nn.Module, val_loader: DataLoader) -> dict[str, float]:
    decoder_was_training = decoder.training
    policy_was_training = runtime.policy.training
    decoder.eval()
    runtime.policy.eval()
    total_loss, total_action, total_expert, batches = 0.0, 0.0, 0.0, 0
    component_totals: dict[str, float] = {}
    for raw_batch in val_loader:
        if batches >= max(int(config.validation_max_batches), 1): break
        processed_batch = preprocess_batch(runtime, raw_batch)
        inputs, enc = build_xvla_inputs(runtime, processed_batch)
        target = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
        expert_loss, expert_stats = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], step=0)
        total = float(config.expert_loss_weight) * expert_loss
        action_value = 0.0
        for key, value in expert_stats.items(): component_totals[f"val_{key}"] = component_totals.get(f"val_{key}", 0.0) + float(value)
        if config.training_stage == "joint_multitask":
            action_loss, action_stats = compute_xvla_action_loss_from_encoder(runtime.policy, processed_batch, inputs, enc)
            action_value = float(action_loss.detach().item())
            total = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
            for key, value in action_stats.items(): component_totals[f"val_{key}"] = component_totals.get(f"val_{key}", 0.0) + float(value)
        total_loss += float(total.detach().item())
        total_action += action_value
        total_expert += float(expert_loss.detach().item())
        batches += 1
    if decoder_was_training: decoder.train()
    if policy_was_training: runtime.policy.train()
    denom = max(batches, 1)
    metrics = {"val_loss": total_loss / denom, "val_expert_total": total_expert / denom, "val_action_total": total_action / denom, "val_batches": float(batches)}
    metrics.update({key: value / denom for key, value in component_totals.items()})
    return metrics


def train_visual_thought(config: VisualThoughtTrainConfig) -> None:
    _set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(config)

    runtime = build_xvla_runtime(config)

    set_policy_trainability(runtime.policy, config.training_stage)
    task_cfg, teacher = build_teacher(config, runtime)
    loader, val_loader, vis_loader, vis_indices, decoder, _ = prepare_models_and_target(config, runtime, task_cfg, teacher)
    if vis_loader is not None: print(json.dumps({"event": "visualization_config", "vis_fixed_indices": vis_indices, "vis_num_samples": int(config.vis_num_samples), "vis_every": int(config.vis_every)}))
    if config.training_stage == "joint_multitask":
        for parameter in decoder.parameters(): parameter.requires_grad = True
        runtime.policy.train()
    else:
        decoder.train()
        
    optimizer = build_optimizer(config, runtime.policy, decoder)
    
    if config.dry_run: return

    step = 0
    accum_steps = max(int(config.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=config.steps, desc=config.name)
    while step < config.steps:
        for raw_batch in loader:
            if step >= config.steps: break
            step += 1

            processed_batch = preprocess_batch(runtime, raw_batch)
            inputs, enc     = build_xvla_inputs(runtime, processed_batch)
            target          = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
            expert_loss, expert_stats = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], step)
            total_loss      = float(config.expert_loss_weight) * expert_loss
            
            action_stats = {}
            if config.training_stage == "joint_multitask":
                action_loss, action_stats = compute_xvla_action_loss_from_encoder(runtime.policy, processed_batch, inputs, enc)
                total_loss = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
            
            (total_loss / accum_steps).backward()
            if step % accum_steps == 0 or step == config.steps:
                torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], XVLA_GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if step % max(int(config.log_every), 1) == 0:
                metrics = {"event": "train_step", "step": int(step), "loss": float(total_loss.detach().item()), **expert_stats, **action_stats}
                print(json.dumps(metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in metrics.items() if key != "event"}, step=int(step))
            if val_loader is not None and step % max(int(config.validation_freq), 1) == 0:
                val_metrics = {"event": "validation_step", "step": int(step), **run_validation(config, runtime, task_cfg, teacher, decoder, val_loader)}
                print(json.dumps(val_metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in val_metrics.items() if key != "event"}, step=int(step))
            if vis_loader is not None and int(config.vis_every) > 0 and step % max(int(config.vis_every), 1) == 0:
                vis_metrics = {"event": "visualization_step", "step": int(step), **run_visualization(config, runtime, teacher, decoder, vis_loader, step)}
                print(json.dumps(vis_metrics))
            _save_checkpoint_if_needed(config, runtime, decoder, optimizer, step, final=False)
            progress.update(1)
    progress.close()
    if bool(config.vis_final) and vis_loader is not None:
        vis_metrics = {"event": "visualization_final", "step": int(step), **run_visualization(config, runtime, teacher, decoder, vis_loader, step)}
        print(json.dumps(vis_metrics))
    if config.save_final_checkpoint: 
        _save_checkpoint_if_needed(config, runtime, decoder, optimizer, step, final=True)
    if wandb_run is not None: wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    train_visual_thought(VisualThoughtTrainConfig.from_json(args.config_path))


if __name__ == "__main__":
    main()
