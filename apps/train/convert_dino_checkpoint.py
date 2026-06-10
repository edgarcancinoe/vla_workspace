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

from thesis_vla.visual_thought import DinoTokenSequenceModel
from thesis_vla.visual_thought.checkpoints import CONFIG_FILENAME, DECODER_STATE_FILENAME, METADATA_FILENAME, save_decoder_state
from thesis_vla.visual_thought.config import DecoderStackConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict): raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _load_state_dict(checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    blob = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict): raise ValueError(f"DINO checkpoint must be a dict, got {type(blob).__name__}.")
    state_dict = blob.get("model", blob)
    if not isinstance(state_dict, dict): raise ValueError("DINO checkpoint does not contain a valid model state dict.")
    # token_sequence DistillationModel keeps the decoder under `strategy.*` and uses an
    # Identity head (no params). Keep only the strategy weights so the state dict maps
    # cleanly onto DinoTokenSequenceModel (which is just `self.strategy`).
    strategy_state = {key: value for key, value in state_dict.items() if key.startswith("strategy.")}
    if not strategy_state: raise ValueError("DINO checkpoint has no 'strategy.*' weights; is this a token_sequence run?")
    dropped = sorted(key for key in state_dict if not key.startswith("strategy."))
    if dropped: print(json.dumps({"event": "dropped_non_strategy_keys", "keys": dropped}))
    return strategy_state, blob


def _infer_student_vlm_dim(state_dict: dict[str, torch.Tensor], projection_mode: str) -> int:
    mode = str(projection_mode).lower()
    if mode == "linear": return int(state_dict["strategy.student_projection.proj.weight"].shape[1])
    if mode == "mlp": return int(state_dict["strategy.student_projection.proj.1.weight"].shape[1])
    if mode == "res_mlp": return int(state_dict["strategy.student_projection.skip.weight"].shape[1])
    raise ValueError(f"Unsupported decoder_student_projection_mode: {projection_mode!r}.")


def _stack_config(payload: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> DecoderStackConfig:
    # For token_sequence, decoder_dim and num_decoder_tokens are fixed by the teacher token
    # grid at distillation time, so read them straight off the saved query_vectors.
    query_vectors = state_dict["strategy.query_vectors"]
    return DecoderStackConfig(
        decoder_dim=int(query_vectors.shape[1]),
        num_decoder_tokens=int(query_vectors.shape[0]),
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


def convert_dino_checkpoint(checkpoint_path: str | Path, config_path: str | Path, output_dir: str | Path) -> Path:
    state_dict, blob = _load_state_dict(checkpoint_path)
    payload = _read_yaml(config_path)
    teacher_payload = dict(payload.get("teacher") or {})
    if str(teacher_payload.get("target_kind", "token_sequence")) != "token_sequence": raise ValueError(f"convert_dino_checkpoint only supports teacher.target_kind='token_sequence', got {teacher_payload.get('target_kind')!r}.")
    stack = _stack_config(payload, state_dict)
    model = DinoTokenSequenceModel(student_vlm_dim=_infer_student_vlm_dim(state_dict, stack.student_projection_mode), stack_cfg=stack)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected: raise RuntimeError(f"DINO checkpoint mismatch. missing={list(missing)} unexpected={list(unexpected)}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_decoder_state(model, root / DECODER_STATE_FILENAME)
    (root / CONFIG_FILENAME).write_text(json.dumps({"stack": stack.__dict__, "teacher": teacher_payload, "source": {"checkpoint_path": str(Path(checkpoint_path).expanduser()), "config_path": str(Path(config_path).expanduser()), "source_step": int(blob.get("step", 0))}}, indent=2, sort_keys=True))
    (root / METADATA_FILENAME).write_text(json.dumps({"expert_type": "dino", "target_kind": "token_sequence", "source_format": "xvla_visualthought_distill", "source_checkpoint_path": str(Path(checkpoint_path).expanduser()), "source_config_path": str(Path(config_path).expanduser()), "source_step": int(blob.get("step", 0)), "decoder_dim": stack.decoder_dim, "num_decoder_tokens": stack.num_decoder_tokens}, indent=2, sort_keys=True))
    return root


def _default_output_dir(checkpoint_path: Path) -> Path:
    stem = checkpoint_path.parent.name if checkpoint_path.name == "checkpoint_final.pt" else checkpoint_path.stem
    return ROOT_DIR / "runtime" / "outputs" / "train" / "visual_thought_imports" / f"dino_{stem}"


def main() -> None:
    p = argparse.ArgumentParser(description="Convert an XVLA-VisualThought DINO token_sequence distillation checkpoint into the decoder export format used by vla_workspace joint visual-thought training.")
    p.add_argument("--checkpoint", required=True, help="Path to the train_dino.py checkpoint (e.g. checkpoint_final.pt) containing blob['model'].")
    p.add_argument("--config", required=True, help="Path to the XVLA-VisualThought DINO train.yaml used to produce that checkpoint.")
    p.add_argument("--output-dir", default=None, help="Directory to write decoder.safetensors plus config/metadata snapshot. Defaults under runtime/outputs/train/visual_thought_imports/.")
    args = p.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(checkpoint)
    out = convert_dino_checkpoint(checkpoint_path=checkpoint, config_path=config, output_dir=output_dir)
    print(json.dumps({"event": "dino_checkpoint_converted", "output_dir": str(out), "decoder_state_path": str(out / DECODER_STATE_FILENAME), "config_snapshot_path": str(out / CONFIG_FILENAME), "metadata_path": str(out / METADATA_FILENAME)}, indent=2))


if __name__ == "__main__":
    main()
