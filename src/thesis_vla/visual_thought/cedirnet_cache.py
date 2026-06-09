from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.visual_thought.config import CeDirNetTeacherConfig
from thesis_vla.visual_thought.targets import TeacherTarget


MANIFEST_FILENAME = "manifest.json"
ABSOLUTE_INDICES_FILENAME = "absolute_indices.npy"
EPISODE_INDICES_FILENAME = "episode_indices.npy"
TARGET_CHUNK_PREFIX = "targets"
CACHE_SCHEMA_VERSION = 1


class CeDiRNetTargetCacheError(RuntimeError):
    pass


def _slug(value: str | None) -> str:
    text = (value or "main").strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "main"


def _serialize_jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, tuple): return [_serialize_jsonable(item) for item in value]
    if isinstance(value, list): return [_serialize_jsonable(item) for item in value]
    if isinstance(value, dict): return {str(key): _serialize_jsonable(item) for key, item in value.items()}
    return value


def _teacher_fingerprint_payload(cfg: CeDirNetTeacherConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "target_kind": cfg.target_kind,
        "loss_type": cfg.loss_type,
        "weight": float(cfg.weight),
        "model_type": cfg.model_type,
        "image_size": int(cfg.image_size),
        "checkpoint": cfg.checkpoint,
        "repo_src": cfg.repo_src,
        "config_path": cfg.config_path,
        "localization_checkpoint": cfg.localization_checkpoint,
        "target_channel_indices": [int(index) for index in cfg.target_channel_indices],
        "resize": bool(cfg.resize),
    }


def teacher_fingerprint(cfg: CeDirNetTeacherConfig) -> str:
    payload = json.dumps(_teacher_fingerprint_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cedirnet_cache_root() -> Path:
    user = os.environ.get("USER", "default_user")
    root = RUNTIME_CACHE_DIR / f"xvla_{user}" / "visual_thought_teacher_targets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_cedirnet_cache_dir(dataset_repo_id: str, dataset_revision: str | None, teacher_cfg: CeDirNetTeacherConfig, cache_root: str | Path | None = None) -> Path:
    base = Path(cache_root).expanduser() if cache_root else default_cedirnet_cache_root()
    fingerprint = teacher_fingerprint(teacher_cfg)
    return base / _slug(dataset_repo_id) / _slug(dataset_revision) / f"{teacher_cfg.name}_{teacher_cfg.target_kind}" / fingerprint[:16]


def _chunk_path(cache_dir: Path, chunk_id: int) -> Path:
    return cache_dir / f"{TARGET_CHUNK_PREFIX}-{chunk_id:05d}.npy"


def _ensure_tensor_shape(tensor: torch.Tensor) -> tuple[int, int, int]:
    if tensor.ndim != 4: raise CeDiRNetTargetCacheError(f"Expected cached dense maps shaped [B,C,H,W], got {tuple(tensor.shape)}.")
    return int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])


def _target_from_cache_tensor(tensor: torch.Tensor, manifest: dict[str, Any]) -> TeacherTarget:
    teacher = manifest["teacher"]
    aux = dict(manifest.get("target_aux", {}))
    if "prepped_hw" in aux and isinstance(aux["prepped_hw"], list): aux["prepped_hw"] = tuple(int(item) for item in aux["prepped_hw"])
    if "map_hw" in aux and isinstance(aux["map_hw"], list): aux["map_hw"] = tuple(int(item) for item in aux["map_hw"])
    return TeacherTarget(name=str(teacher["name"]), tensor=tensor, kind=str(teacher["target_kind"]), loss_type=str(teacher["loss_type"]), weight=float(teacher["weight"]), aux=aux)


def _dataset_root_string(dataset_root: str | Path | None) -> str | None:
    return None if dataset_root is None else str(Path(dataset_root).expanduser().resolve())


@dataclass(frozen=True)
class CeDiRNetTargetCacheManifest:
    payload: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.payload["frame_count"])

    @property
    def target_shape(self) -> tuple[int, int, int]:
        shape = self.payload["target"]["shape"]
        return int(shape[0]), int(shape[1]), int(shape[2])

    @property
    def chunk_size(self) -> int:
        return int(self.payload["storage"]["chunk_size"])

    @property
    def chunk_files(self) -> list[str]:
        return [str(name) for name in self.payload["storage"]["chunk_files"]]

    @property
    def absolute_indices_file(self) -> str:
        return str(self.payload["index"]["absolute_indices_file"])

    @property
    def episode_indices_file(self) -> str:
        return str(self.payload["index"]["episode_indices_file"])

    def validate_for_dataset(self, *, dataset_repo_id: str, dataset_revision: str | None, dataset_root: str | Path | None, dataset_length: int, teacher_cfg: CeDirNetTeacherConfig) -> None:
        dataset_info = self.payload["dataset"]
        teacher_info = self.payload["teacher"]
        if int(self.payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION: raise CeDiRNetTargetCacheError(f"Unsupported CeDiRNet cache schema version {self.payload.get('schema_version')}. Expected {CACHE_SCHEMA_VERSION}.")
        if str(dataset_info["repo_id"]) != str(dataset_repo_id): raise CeDiRNetTargetCacheError(f"CeDiRNet cache repo mismatch: expected {dataset_repo_id!r}, found {dataset_info['repo_id']!r}.")
        if (dataset_info.get("revision") or None) != (dataset_revision or None): raise CeDiRNetTargetCacheError(f"CeDiRNet cache revision mismatch: expected {dataset_revision!r}, found {dataset_info.get('revision')!r}.")
        if _dataset_root_string(dataset_root) != (dataset_info.get("resolved_root") or None): raise CeDiRNetTargetCacheError(f"CeDiRNet cache root mismatch: expected {_dataset_root_string(dataset_root)!r}, found {dataset_info.get('resolved_root')!r}.")
        if int(dataset_info["length"]) != int(dataset_length): raise CeDiRNetTargetCacheError(f"CeDiRNet cache length mismatch: expected {dataset_length}, found {dataset_info['length']}.")
        if str(teacher_info["type"]) != "cedirnet": raise CeDiRNetTargetCacheError(f"CeDiRNet cache teacher type mismatch: expected 'cedirnet', found {teacher_info['type']!r}.")
        if str(teacher_info["target_kind"]) != "dense_map": raise CeDiRNetTargetCacheError(f"CeDiRNet cache target kind mismatch: expected 'dense_map', found {teacher_info['target_kind']!r}.")
        expected_fingerprint = teacher_fingerprint(teacher_cfg)
        if str(teacher_info["fingerprint"]) != expected_fingerprint: raise CeDiRNetTargetCacheError("CeDiRNet cache fingerprint mismatch. The cached targets do not match the configured checkpoint/config/target settings.")

    @classmethod
    def load(cls, path: str | Path) -> "CeDiRNetTargetCacheManifest":
        return cls(json.loads(Path(path).read_text()))


class CeDiRNetTargetCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / MANIFEST_FILENAME
        if not manifest_path.is_file(): raise CeDiRNetTargetCacheError(f"Missing CeDiRNet cache manifest: {manifest_path}")
        self.manifest = CeDiRNetTargetCacheManifest.load(manifest_path)
        self._absolute_indices = np.load(self.cache_dir / self.manifest.absolute_indices_file)
        self._episode_indices = np.load(self.cache_dir / self.manifest.episode_indices_file)
        if int(self._absolute_indices.shape[0]) != self.manifest.frame_count: raise CeDiRNetTargetCacheError(f"CeDiRNet cache absolute index count mismatch: expected {self.manifest.frame_count}, found {self._absolute_indices.shape[0]}.")
        if int(self._episode_indices.shape[0]) != self.manifest.frame_count: raise CeDiRNetTargetCacheError(f"CeDiRNet cache episode index count mismatch: expected {self.manifest.frame_count}, found {self._episode_indices.shape[0]}.")
        self._row_by_absolute_index = {int(index): row for row, index in enumerate(self._absolute_indices.tolist())}
        self._chunks: dict[int, np.ndarray] = {}

    @classmethod
    def resolve(cls, *, dataset_repo_id: str, dataset_revision: str | None, dataset_root: str | Path | None, dataset_length: int, teacher_cfg: CeDirNetTeacherConfig, cache_root: str | Path | None = None) -> "CeDiRNetTargetCache":
        cache = cls(resolve_cedirnet_cache_dir(dataset_repo_id=dataset_repo_id, dataset_revision=dataset_revision, teacher_cfg=teacher_cfg, cache_root=cache_root))
        cache.manifest.validate_for_dataset(dataset_repo_id=dataset_repo_id, dataset_revision=dataset_revision, dataset_root=dataset_root, dataset_length=dataset_length, teacher_cfg=teacher_cfg)
        return cache

    def _load_chunk(self, chunk_id: int) -> np.ndarray:
        if chunk_id not in self._chunks:
            chunk_path = self.cache_dir / self.manifest.chunk_files[chunk_id]
            if not chunk_path.is_file(): raise CeDiRNetTargetCacheError(f"Missing CeDiRNet cache shard: {chunk_path}")
            self._chunks[chunk_id] = np.load(chunk_path, mmap_mode="r")
        return self._chunks[chunk_id]

    def _row_to_tensor(self, row: int) -> np.ndarray:
        chunk_size = self.manifest.chunk_size
        chunk_id, offset = divmod(int(row), chunk_size)
        chunk = self._load_chunk(chunk_id)
        return np.asarray(chunk[offset], dtype=np.float32)

    def target_for_absolute_indices(self, indices: torch.Tensor | np.ndarray | list[int] | tuple[int, ...], *, device: str | torch.device | None = None) -> TeacherTarget:
        if isinstance(indices, torch.Tensor): raw = indices.detach().cpu().reshape(-1).tolist()
        elif isinstance(indices, np.ndarray): raw = indices.reshape(-1).tolist()
        else: raw = list(indices)
        rows = []
        for index in raw:
            key = int(index)
            if key not in self._row_by_absolute_index: raise CeDiRNetTargetCacheError(f"CeDiRNet cache does not contain dataset index {key}.")
            rows.append(self._row_by_absolute_index[key])
        stacked = np.stack([self._row_to_tensor(row) for row in rows], axis=0)
        tensor = torch.from_numpy(stacked)
        if device is not None: tensor = tensor.to(device)
        return _target_from_cache_tensor(tensor, self.manifest.payload)

    def target_for_batch(self, raw_batch: dict[str, Any], *, device: str | torch.device | None = None) -> TeacherTarget:
        if "index" not in raw_batch: raise CeDiRNetTargetCacheError("CeDiRNet cache lookup requires raw_batch['index'] from the LeRobot dataset.")
        return self.target_for_absolute_indices(raw_batch["index"], device=device)

    def manifest_summary(self) -> dict[str, Any]:
        payload = dict(self.manifest.payload)
        payload["cache_dir"] = str(self.cache_dir)
        payload["absolute_index_min"] = int(np.min(self._absolute_indices)) if self._absolute_indices.size else None
        payload["absolute_index_max"] = int(np.max(self._absolute_indices)) if self._absolute_indices.size else None
        return payload


def build_cedirnet_cache_manifest(*, dataset_repo_id: str, dataset_revision: str | None, dataset_root: str | Path | None, dataset_length: int, teacher_cfg: CeDirNetTeacherConfig, target_shape: tuple[int, int, int], target_aux: dict[str, Any], chunk_size: int, chunk_files: list[str], batch_size: int, device: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": int(dataset_length),
        "dataset": {"repo_id": dataset_repo_id, "revision": dataset_revision, "resolved_root": _dataset_root_string(dataset_root), "length": int(dataset_length)},
        "teacher": {
            "type": "cedirnet",
            "fingerprint": teacher_fingerprint(teacher_cfg),
            **_teacher_fingerprint_payload(teacher_cfg),
        },
        "target": {"dtype": "float32", "shape": [int(target_shape[0]), int(target_shape[1]), int(target_shape[2])], "num_channels": int(target_shape[0]), "spatial_hw": [int(target_shape[1]), int(target_shape[2])]},
        "target_aux": _serialize_jsonable(target_aux),
        "storage": {"chunk_size": int(chunk_size), "chunk_files": [str(name) for name in chunk_files]},
        "index": {"absolute_indices_file": ABSOLUTE_INDICES_FILENAME, "episode_indices_file": EPISODE_INDICES_FILENAME},
        "generation": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device), "batch_size": int(batch_size)},
    }


def write_cedirnet_target_cache(*, cache_dir: str | Path, dataset_repo_id: str, dataset_revision: str | None, dataset_root: str | Path | None, teacher_cfg: CeDirNetTeacherConfig, absolute_indices: np.ndarray, episode_indices: np.ndarray, tensors: list[np.ndarray], target_aux: dict[str, Any], chunk_size: int, batch_size: int, device: str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if len(tensors) != int(absolute_indices.shape[0]): raise CeDiRNetTargetCacheError(f"CeDiRNet cache write expected {absolute_indices.shape[0]} tensors, got {len(tensors)}.")
    if len(tensors) != int(episode_indices.shape[0]): raise CeDiRNetTargetCacheError(f"CeDiRNet cache write expected {episode_indices.shape[0]} episode ids, got {len(episode_indices)}.")
    target_shape = None
    chunk_files: list[str] = []
    for chunk_id, start in enumerate(range(0, len(tensors), int(chunk_size))):
        batch = np.stack(tensors[start:start + int(chunk_size)], axis=0).astype(np.float32, copy=False)
        if batch.ndim != 4: raise CeDiRNetTargetCacheError(f"CeDiRNet cache shards must be [N,C,H,W], got {batch.shape}.")
        target_shape = target_shape or (int(batch.shape[1]), int(batch.shape[2]), int(batch.shape[3]))
        if target_shape != (int(batch.shape[1]), int(batch.shape[2]), int(batch.shape[3])): raise CeDiRNetTargetCacheError("CeDiRNet cache tensors must all share the same [C,H,W] shape.")
        filename = _chunk_path(cache_dir, chunk_id).name
        np.save(cache_dir / filename, batch)
        chunk_files.append(filename)
    if target_shape is None: raise CeDiRNetTargetCacheError("CeDiRNet cache write requires at least one target tensor.")
    np.save(cache_dir / ABSOLUTE_INDICES_FILENAME, absolute_indices.astype(np.int64, copy=False))
    np.save(cache_dir / EPISODE_INDICES_FILENAME, episode_indices.astype(np.int64, copy=False))
    manifest = build_cedirnet_cache_manifest(dataset_repo_id=dataset_repo_id, dataset_revision=dataset_revision, dataset_root=dataset_root, dataset_length=len(tensors), teacher_cfg=teacher_cfg, target_shape=target_shape, target_aux=target_aux, chunk_size=chunk_size, chunk_files=chunk_files, batch_size=batch_size, device=device)
    (cache_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return cache_dir


def collect_batch_indices(raw_batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "index" not in raw_batch: raise CeDiRNetTargetCacheError("LeRobot raw batch is missing 'index'; CeDiRNet offline caching requires stable dataset indices.")
    if "episode_index" not in raw_batch: raise CeDiRNetTargetCacheError("LeRobot raw batch is missing 'episode_index'; CeDiRNet offline caching requires episode index metadata.")
    absolute = raw_batch["index"].detach().cpu().numpy() if isinstance(raw_batch["index"], torch.Tensor) else np.asarray(raw_batch["index"])
    episode = raw_batch["episode_index"].detach().cpu().numpy() if isinstance(raw_batch["episode_index"], torch.Tensor) else np.asarray(raw_batch["episode_index"])
    return absolute.astype(np.int64, copy=False).reshape(-1), episode.astype(np.int64, copy=False).reshape(-1)
