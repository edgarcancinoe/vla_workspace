from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
workspace_src = ROOT_DIR / "src"
if str(workspace_src) not in sys.path: sys.path.insert(0, str(workspace_src))

from thesis_vla.visual_thought import CeDirNetDistillationModel
from thesis_vla.visual_thought.checkpoints import CONFIG_FILENAME, DECODER_STATE_FILENAME, METADATA_FILENAME, save_decoder_state
from thesis_vla.visual_thought.config import CeDirNetDecoderConfig, CeDirNetTeacherConfig, DecoderStackConfig, DenseMapHeadConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict): raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _infer_student_vlm_dim(state_dict: dict[str, torch.Tensor], projection_mode: str) -> int:
    mode = str(projection_mode).lower()
    if mode == "linear": return int(state_dict["strategy.student_projection.proj.weight"].shape[1])
    if mode == "mlp": return int(state_dict["strategy.student_projection.proj.1.weight"].shape[1])
    if mode == "res_mlp": return int(state_dict["strategy.student_projection.skip.weight"].shape[1])
    raise ValueError(f"Unsupported decoder_student_projection_mode: {projection_mode!r}.")


def _load_legacy_state_dict(checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    blob = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict): raise ValueError(f"Legacy checkpoint must be a dict, got {type(blob).__name__}.")
    state_dict = blob.get("model", blob)
    if not isinstance(state_dict, dict): raise ValueError("Legacy checkpoint does not contain a valid model state dict.")
    return state_dict, blob


def _legacy_stack_config(payload: dict[str, Any], teacher_extra: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> DecoderStackConfig:
    return DecoderStackConfig(
        decoder_dim=int(teacher_extra.get("decoder_dim") or state_dict["strategy.query_vectors"].shape[1]),
        num_decoder_tokens=int(teacher_extra.get("decoder_num_tokens") or (int(teacher_extra["decoder_grid_hw"][0]) * int(teacher_extra["decoder_grid_hw"][1]))),
        num_heads=int(payload.get("token_decoder_num_heads", 8)),
        num_layers=int(payload.get("decoder_num_layers", 1)),
        ffn_enabled=bool(payload.get("token_decoder_ffn_enabled", True)),
        ffn_mlp_ratio=float(payload.get("token_decoder_ffn_mlp_ratio", 4.0)),
        ffn_dropout=float(payload.get("token_decoder_ffn_dropout", 0.0)),
        self_attn_queries=bool(payload.get("decoder_self_attn_queries", True)),
        self_attn_student=bool(payload.get("decoder_self_attn_student", False)),
        gating_enabled=bool(payload.get("decoder_gating_enabled", False)),
        gating_mode=str(payload.get("decoder_gating_mode", "none")),
        cross_attn_residual=bool(payload.get("decoder_cross_attn_residual", False)),
        student_projection_mode=str(payload.get("decoder_student_projection_mode", "linear")),
        student_projection_mlp_ratio=float(payload.get("decoder_student_projection_mlp_ratio", 4.0)),
        student_projection_dropout=float(payload.get("decoder_student_projection_dropout", 0.0)),
        positional_encodings=bool(payload.get("decoder_positional_encodings", False)),
    )


def _legacy_head_config(teacher_extra: dict[str, Any]) -> DenseMapHeadConfig:
    return DenseMapHeadConfig(
        grid_hw=(int(teacher_extra["decoder_grid_hw"][0]), int(teacher_extra["decoder_grid_hw"][1])),
        hidden_dim=None if teacher_extra.get("dense_map_hidden_dim") is None else int(teacher_extra["dense_map_hidden_dim"]),
        projection_mode=str(teacher_extra.get("dense_map_projection_mode", "linear")),
        projection_mlp_ratio=float(teacher_extra.get("dense_map_projection_mlp_ratio", 4.0)),
        projection_dropout=float(teacher_extra.get("dense_map_projection_dropout", 0.0)),
        refine_layers=int(teacher_extra.get("dense_map_refine_layers", 2)),
        refine_kernel_size=int(teacher_extra.get("dense_map_refine_kernel_size", 3)),
        refine_dropout=float(teacher_extra.get("dense_map_refine_dropout", 0.0)),
        out_layers=int(teacher_extra.get("dense_map_out_layers", 1)),
        out_hidden_dim=None if teacher_extra.get("dense_map_out_hidden_dim") is None else int(teacher_extra["dense_map_out_hidden_dim"]),
        out_dropout=float(teacher_extra.get("dense_map_out_dropout", 0.0)),
        resize_mode="bilinear",
        align_corners=False,
    )


def _legacy_teacher_config(teacher_payload: dict[str, Any], teacher_extra: dict[str, Any]) -> CeDirNetTeacherConfig:
    return CeDirNetTeacherConfig(
        name=str(teacher_payload.get("name", "cedirnet")),
        target_kind=str(teacher_payload.get("target_kind", "dense_map")),
        loss_type=str(teacher_payload.get("loss_type", "mse")),
        weight=float(teacher_payload.get("weight", 1.0)),
        model_type=None if teacher_payload.get("model_type") is None else str(teacher_payload["model_type"]),
        image_size=int(teacher_payload.get("image_size", 768)),
        checkpoint=None if teacher_payload.get("checkpoint") is None else str(teacher_payload["checkpoint"]),
        repo_src=None if teacher_extra.get("repo_src") is None else str(teacher_extra["repo_src"]),
        config_path=None if teacher_extra.get("config_path") is None else str(teacher_extra["config_path"]),
        localization_checkpoint=None if teacher_extra.get("localization_checkpoint") is None else str(teacher_extra["localization_checkpoint"]),
        target_channel_indices=tuple(int(v) for v in teacher_extra.get("target_channel_indices", [0, 1, 2])),
        resize=bool(teacher_extra.get("resize", True)),
    )


def convert_legacy_cedirnet_checkpoint(legacy_checkpoint_path: str | Path, legacy_config_path: str | Path, output_dir: str | Path) -> Path:
    state_dict, blob = _load_legacy_state_dict(legacy_checkpoint_path)
    payload = _read_yaml(legacy_config_path)
    teacher_payload = dict(payload.get("teacher") or {})
    if str(teacher_payload.get("target_kind", "dense_map")) != "dense_map": raise ValueError(f"Expected legacy CeDirNet config with teacher.target_kind='dense_map', got {teacher_payload.get('target_kind')!r}.")
    teacher_extra = dict(teacher_payload.get("extra") or {})
    stack = _legacy_stack_config(payload, teacher_extra, state_dict)
    head = _legacy_head_config(teacher_extra)
    teacher = _legacy_teacher_config(teacher_payload, teacher_extra)
    cfg = CeDirNetDecoderConfig(stack=stack, head=head, teacher=teacher).validate()
    model = CeDirNetDistillationModel(student_vlm_dim=_infer_student_vlm_dim(state_dict, stack.student_projection_mode), stack_cfg=cfg.stack, head_cfg=cfg.head, out_channels=cfg.teacher.out_channels)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected: raise RuntimeError(f"Legacy checkpoint mismatch. missing={list(missing)} unexpected={list(unexpected)}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_decoder_state(model, root / DECODER_STATE_FILENAME)
    (root / CONFIG_FILENAME).write_text(json.dumps({"stack": cfg.stack.__dict__, "head": cfg.head.__dict__, "teacher": cfg.teacher.__dict__, "legacy": {"checkpoint_path": str(Path(legacy_checkpoint_path).expanduser()), "config_path": str(Path(legacy_config_path).expanduser()), "legacy_step": int(blob.get("step", 0))}}, indent=2, sort_keys=True))
    (root / METADATA_FILENAME).write_text(json.dumps({"expert_type": "cedirnet", "source_format": "xvla_visualthought_legacy", "legacy_checkpoint_path": str(Path(legacy_checkpoint_path).expanduser()), "legacy_config_path": str(Path(legacy_config_path).expanduser()), "legacy_step": int(blob.get("step", 0))}, indent=2, sort_keys=True))
    return root


def _default_output_dir(checkpoint_path: Path) -> Path:
    stem = checkpoint_path.parent.name if checkpoint_path.name == "checkpoint_final.pt" else checkpoint_path.stem
    return ROOT_DIR / "runtime" / "outputs" / "train" / "visual_thought_imports" / stem


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a legacy XVLA-VisualThought CeDirNet checkpoint into the decoder export format used by vla_workspace joint visual-thought training.")
    p.add_argument("--legacy-checkpoint", required=True, help="Path to legacy checkpoint_final.pt or equivalent checkpoint containing blob['model'].")
    p.add_argument("--legacy-config", required=True, help="Path to the legacy XVLA-VisualThought CeDirNet YAML used to train that checkpoint.")
    p.add_argument("--output-dir", default=None, help="Directory to write decoder.safetensors plus config snapshot. Defaults under runtime/outputs/train/visual_thought_imports/.")
    args = p.parse_args()
    legacy_checkpoint = Path(args.legacy_checkpoint).expanduser().resolve()
    legacy_config = Path(args.legacy_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(legacy_checkpoint)
    out = convert_legacy_cedirnet_checkpoint(legacy_checkpoint_path=legacy_checkpoint, legacy_config_path=legacy_config, output_dir=output_dir)
    print(json.dumps({"event": "legacy_cedirnet_checkpoint_converted", "output_dir": str(out), "decoder_state_path": str(out / DECODER_STATE_FILENAME), "config_snapshot_path": str(out / CONFIG_FILENAME), "metadata_path": str(out / METADATA_FILENAME)}, indent=2))


if __name__ == "__main__":
    main()
