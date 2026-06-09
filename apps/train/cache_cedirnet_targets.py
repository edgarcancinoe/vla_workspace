from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
workspace_src = ROOT_DIR / "src"
sys.path.insert(0, str(workspace_src))
lerobot_src_candidates = [ROOT_DIR / "lerobot" / "src", ROOT_DIR.parent / "repos" / "lerobot" / "src"]
for lerobot_src in lerobot_src_candidates:
    if lerobot_src.exists(): sys.path.insert(0, str(lerobot_src))

from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.visual_thought import load_cedirnet_decoder_config
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache, CeDiRNetTargetCacheError, collect_batch_indices, resolve_cedirnet_cache_dir, write_cedirnet_target_cache
from thesis_vla.visual_thought.teachers import CeDiRNetTeacher


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


def _resolve_dataset_root(dataset_repo_id: str, dataset_root: str | None) -> Path:
    return Path(dataset_root).expanduser() / dataset_repo_id if dataset_root else _default_lerobot_home() / dataset_repo_id


def _resolve_teacher_image_key(camera_keys: list[str], requested_key: str) -> str:
    if requested_key in camera_keys: return requested_key
    if "observation.images.image" in camera_keys: return "observation.images.image"
    if "observation.images.main" in camera_keys: return "observation.images.main"
    if not camera_keys: raise CeDiRNetTargetCacheError("Dataset does not expose any camera keys for CeDiRNet target generation.")
    return camera_keys[0]


def _get_teacher_images(raw_batch: dict[str, object], teacher_image_key: str) -> torch.Tensor:
    images = raw_batch[teacher_image_key]
    return images[:, -1] if getattr(images, "ndim", 0) == 5 else images


def _build_dataset(args) -> tuple[object, Path, str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved_root = _resolve_dataset_root(args.dataset_repo_id, args.dataset_root)
    dataset = LeRobotDataset(args.dataset_repo_id, root=resolved_root, revision=args.dataset_revision, video_backend=args.dataset_video_backend, tolerance_s=float(args.dataset_tolerance_s))
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), args.teacher_image_feature_key)
    return dataset, resolved_root, teacher_image_key


def _cache_dir(args, teacher_cfg) -> Path:
    return resolve_cedirnet_cache_dir(dataset_repo_id=args.dataset_repo_id, dataset_revision=args.dataset_revision, teacher_cfg=teacher_cfg, cache_root=args.cache_root)


def build_cache(args) -> None:
    task_cfg = load_cedirnet_decoder_config(args.decoder_stack_config_path, args.decoder_task_config_path)
    if task_cfg.teacher.target_kind != "dense_map": raise CeDiRNetTargetCacheError(f"Only CeDiRNet dense_map caching is supported, got {task_cfg.teacher.target_kind!r}.")
    dataset, resolved_root, teacher_image_key = _build_dataset(args)
    cache_dir = _cache_dir(args, task_cfg.teacher)
    if cache_dir.exists():
        if not args.overwrite: raise CeDiRNetTargetCacheError(f"CeDiRNet cache already exists at {cache_dir}. Pass --overwrite to rebuild it.")
        shutil.rmtree(cache_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=default_collate, drop_last=False)
    teacher = CeDiRNetTeacher(task_cfg.teacher)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    absolute_indices, episode_indices, tensors, target_aux = [], [], [], None
    total = min(len(dataset), int(args.limit)) if int(args.limit) > 0 else len(dataset)
    progress = tqdm(total=total, desc=f"cache:{args.dataset_repo_id}")
    with torch.no_grad():
        for raw_batch in loader:
            if int(args.limit) > 0 and len(tensors) >= int(args.limit): break
            target = teacher.predict(_get_teacher_images(raw_batch, teacher_image_key).to(device))
            batch_indices, batch_episodes = collect_batch_indices(raw_batch)
            batch_tensors = target.tensor.detach().cpu().numpy()
            if batch_tensors.shape[0] != batch_indices.shape[0]: raise CeDiRNetTargetCacheError(f"Target batch size {batch_tensors.shape[0]} does not match dataset index batch size {batch_indices.shape[0]}.")
            if target_aux is None: target_aux = dict(target.aux)
            for batch_row, absolute_index in enumerate(batch_indices.tolist()):
                if int(args.limit) > 0 and len(tensors) >= int(args.limit): break
                absolute_indices.append(int(absolute_index))
                episode_indices.append(int(batch_episodes[batch_row]))
                tensors.append(np.asarray(batch_tensors[batch_row], dtype=np.float32))
                progress.update(1)
    progress.close()
    if not tensors: raise CeDiRNetTargetCacheError("CeDiRNet cache generation did not produce any targets.")
    ordered = sorted(zip(absolute_indices, episode_indices, tensors), key=lambda item: item[0])
    ordered_absolute = np.asarray([item[0] for item in ordered], dtype=np.int64)
    ordered_episode = np.asarray([item[1] for item in ordered], dtype=np.int64)
    ordered_tensors = [item[2] for item in ordered]
    write_cedirnet_target_cache(cache_dir=cache_dir, dataset_repo_id=args.dataset_repo_id, dataset_revision=args.dataset_revision, dataset_root=resolved_root, teacher_cfg=task_cfg.teacher, absolute_indices=ordered_absolute, episode_indices=ordered_episode, tensors=ordered_tensors, target_aux=target_aux or {}, chunk_size=args.chunk_size, batch_size=args.batch_size, device=str(device))
    print(json.dumps({"event": "cedirnet_cache_built", "cache_dir": str(cache_dir), "frame_count": len(ordered_tensors), "dataset_root": str(resolved_root), "chunk_size": int(args.chunk_size)}, indent=2, sort_keys=True))


def inspect_cache(args) -> None:
    task_cfg = load_cedirnet_decoder_config(args.decoder_stack_config_path, args.decoder_task_config_path)
    dataset, resolved_root, _ = _build_dataset(args)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id=args.dataset_repo_id, dataset_revision=args.dataset_revision, dataset_root=resolved_root, dataset_length=len(dataset), teacher_cfg=task_cfg.teacher, cache_root=args.cache_root)
    summary = cache.manifest_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.spot_check_indices: return
    indices = [int(item.strip()) for item in str(args.spot_check_indices).split(",") if item.strip()]
    if not indices: return
    target = cache.target_for_absolute_indices(indices, device="cpu")
    print(json.dumps({"event": "cedirnet_cache_spot_check", "indices": indices, "target_shape": list(target.tensor.shape), "target_mean": float(target.tensor.mean().item()), "target_std": float(target.tensor.std().item())}, indent=2, sort_keys=True))


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0: return np.zeros(channel.shape, dtype=np.uint8)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if abs(hi - lo) < 1e-8: return np.zeros(channel.shape, dtype=np.uint8)
    return (np.clip((channel - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _channel_image(channel: np.ndarray) -> Image.Image:
    normalized = _normalize_channel(channel)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def _tile_with_label(image: Image.Image, label: str, panel_size: int, header_height: int = 28) -> Image.Image:
    resized = image.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)
    tile = Image.new("RGB", (panel_size, panel_size + header_height), color=(16, 16, 16))
    tile.paste(resized, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 6), label, fill=(255, 255, 255))
    return tile


def visualize_cache(args) -> None:
    task_cfg = load_cedirnet_decoder_config(args.decoder_stack_config_path, args.decoder_task_config_path)
    dataset, resolved_root, _ = _build_dataset(args)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id=args.dataset_repo_id, dataset_revision=args.dataset_revision, dataset_root=resolved_root, dataset_length=len(dataset), teacher_cfg=task_cfg.teacher, cache_root=args.cache_root)
    indices = [int(item.strip()) for item in str(args.indices).split(",") if item.strip()]
    if not indices: raise CeDiRNetTargetCacheError("visualize requires at least one --indices entry.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_size = int(args.panel_size)
    for index in indices:
        target = cache.target_for_absolute_indices([index], device="cpu")
        tensor = target.tensor[0].detach().cpu().numpy()
        panels = [_tile_with_label(_channel_image(tensor[channel]), f"idx={index} ch={channel}", panel_size) for channel in range(int(tensor.shape[0]))]
        canvas = Image.new("RGB", (panel_size * len(panels), panels[0].height), color=(0, 0, 0))
        for i, panel in enumerate(panels): canvas.paste(panel, (i * panel_size, 0))
        out_path = output_dir / f"cedirnet_cache_{index:06d}.png"
        canvas.save(out_path)
        print(json.dumps({"event": "cedirnet_cache_visualized", "index": index, "output_path": str(out_path), "target_shape": list(target.tensor.shape)}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or inspect offline CeDiRNet dense-map caches for visual-thought training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "inspect", "visualize"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset_repo_id", required=True)
        sub.add_argument("--dataset_revision", default=None)
        sub.add_argument("--dataset_root", default=None)
        sub.add_argument("--decoder_stack_config_path", required=True)
        sub.add_argument("--decoder_task_config_path", required=True)
        sub.add_argument("--cache_root", default=None)
        sub.add_argument("--teacher_image_feature_key", default="observation.images.image")
        sub.add_argument("--dataset_video_backend", default="pyav")
        sub.add_argument("--dataset_tolerance_s", type=float, default=1e-4)
    build = subparsers.choices["build"]
    build.add_argument("--device", default="cuda")
    build.add_argument("--batch_size", type=int, default=1)
    build.add_argument("--num_workers", type=int, default=0)
    build.add_argument("--chunk_size", type=int, default=256)
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--overwrite", action="store_true")
    inspect = subparsers.choices["inspect"]
    inspect.add_argument("--spot_check_indices", default="0")
    visualize = subparsers.choices["visualize"]
    visualize.add_argument("--indices", default="0")
    visualize.add_argument("--output_dir", default=str(ROOT_DIR / "runtime" / "outputs" / "visualizations" / "cedirnet_cache"))
    visualize.add_argument("--panel_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build": build_cache(args)
    elif args.command == "inspect": inspect_cache(args)
    else: visualize_cache(args)


if __name__ == "__main__":
    main()
