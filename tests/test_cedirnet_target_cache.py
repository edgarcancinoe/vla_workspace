from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.training.visual_thought_trainer import VisualThoughtTrainConfig, load_teacher_target
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache, CeDiRNetTargetCacheError, resolve_cedirnet_cache_dir, write_cedirnet_target_cache
from thesis_vla.visual_thought.config import CeDirNetTeacherConfig


def _teacher_cfg() -> CeDirNetTeacherConfig:
    return CeDirNetTeacherConfig(name="cedirnet", target_kind="dense_map", loss_type="mse", weight=1.0, model_type="ConvNext-B-RGB", image_size=768, checkpoint="/tmp/checkpoint.pth", repo_src="/tmp/repo/src", config_path="/tmp/config.json", localization_checkpoint="/tmp/localization.pth", target_channel_indices=(0, 1, 2), resize=True)


def _write_cache(tmp_path: Path, revision: str | None = "main") -> tuple[CeDirNetTeacherConfig, Path]:
    teacher_cfg = _teacher_cfg()
    cache_dir = resolve_cedirnet_cache_dir(dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision=revision, teacher_cfg=teacher_cfg, cache_root=tmp_path)
    tensors = [np.full((3, 4, 5), fill_value=value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]
    write_cedirnet_target_cache(cache_dir=cache_dir, dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision=revision, dataset_root=Path("/tmp/datasets") / "edgarcancinoe/cloth-corner-fold_7p5hz", teacher_cfg=teacher_cfg, absolute_indices=np.asarray([10, 11, 12], dtype=np.int64), episode_indices=np.asarray([0, 0, 1], dtype=np.int64), tensors=tensors, target_aux={"prepped_hw": (768, 768), "map_hw": (4, 5), "num_channels": 3, "full_num_channels": 5, "localization_checkpoint": teacher_cfg.localization_checkpoint}, chunk_size=2, batch_size=1, device="cpu")
    return teacher_cfg, cache_dir


def test_cedirnet_cache_roundtrip_and_validation(tmp_path):
    teacher_cfg, _ = _write_cache(tmp_path)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision="main", dataset_root=Path("/tmp/datasets") / "edgarcancinoe/cloth-corner-fold_7p5hz", dataset_length=3, teacher_cfg=teacher_cfg, cache_root=tmp_path)
    target = cache.target_for_absolute_indices([12, 10], device="cpu")
    assert target.name == "cedirnet"
    assert target.kind == "dense_map"
    assert target.loss_type == "mse"
    assert target.tensor.shape == (2, 3, 4, 5)
    assert float(target.tensor[0, 0, 0, 0].item()) == 3.0
    assert float(target.tensor[1, 0, 0, 0].item()) == 1.0
    assert target.aux["prepped_hw"] == (768, 768)
    assert target.aux["map_hw"] == (4, 5)


def test_cedirnet_cache_rejects_revision_mismatch(tmp_path):
    teacher_cfg, cache_dir = _write_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset"]["revision"] = "v1"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(CeDiRNetTargetCacheError, match="revision mismatch"):
        CeDiRNetTargetCache.resolve(dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision="main", dataset_root=Path("/tmp/datasets") / "edgarcancinoe/cloth-corner-fold_7p5hz", dataset_length=3, teacher_cfg=teacher_cfg, cache_root=tmp_path)


def test_cedirnet_cache_missing_shard_fails_fast(tmp_path):
    teacher_cfg, cache_dir = _write_cache(tmp_path)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision="main", dataset_root=Path("/tmp/datasets") / "edgarcancinoe/cloth-corner-fold_7p5hz", dataset_length=3, teacher_cfg=teacher_cfg, cache_root=tmp_path)
    (cache_dir / "targets-00000.npy").unlink()
    with pytest.raises(CeDiRNetTargetCacheError, match="Missing CeDiRNet cache shard"):
        cache.target_for_absolute_indices([10], device="cpu")


def test_load_teacher_target_uses_cache_for_cedirnet_without_images(tmp_path):
    teacher_cfg, _ = _write_cache(tmp_path)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision="main", dataset_root=Path("/tmp/datasets") / "edgarcancinoe/cloth-corner-fold_7p5hz", dataset_length=3, teacher_cfg=teacher_cfg, cache_root=tmp_path)
    config = VisualThoughtTrainConfig(name="demo", training_stage="distill_only", expert_type="cedirnet", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", dataset_repo_id="edgarcancinoe/cloth-corner-fold_7p5hz", dataset_revision="main", dataset_root=str(Path("/tmp/datasets")), output_dir="/tmp/out", device="cpu", teacher_target_cache_root=str(tmp_path))
    target = load_teacher_target(config, cache, {"index": torch.tensor([11])}, "missing.image.key")
    assert target.tensor.shape == (1, 3, 4, 5)
    assert float(target.tensor[0, 0, 0, 0].item()) == 2.0
