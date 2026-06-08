from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from thesis_vla.common.paths import CONFIG_ROOT


VISUAL_THOUGHT_CONFIG_ROOT = CONFIG_ROOT / "visual_thought"
DEFAULT_CEDIRNET_STACK_CONFIG_PATH = VISUAL_THOUGHT_CONFIG_ROOT / "cedirnet_stack.yaml"
DEFAULT_CEDIRNET_HEAD_CONFIG_PATH = VISUAL_THOUGHT_CONFIG_ROOT / "cedirnet_head.yaml"
DEFAULT_DINO_STACK_CONFIG_PATH = VISUAL_THOUGHT_CONFIG_ROOT / "dino_stack.yaml"
DEFAULT_DINO_FEATURE_ALIGNMENT_CONFIG_PATH = VISUAL_THOUGHT_CONFIG_ROOT / "dino_feature_alignment.yaml"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict): raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _as_tuple2(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2: raise ValueError(f"{field_name} must be a two-item list or tuple.")
    return int(value[0]), int(value[1])


def _as_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value: raise ValueError(f"{field_name} must be a non-empty list or tuple.")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class DecoderStackConfig:
    decoder_dim: int
    num_decoder_tokens: int
    num_heads: int = 8
    num_layers: int = 1
    ffn_enabled: bool = True
    ffn_mlp_ratio: float = 4.0
    ffn_dropout: float = 0.0
    self_attn_queries: bool = True
    self_attn_student: bool = False
    gating_enabled: bool = False
    gating_mode: str = "none"
    cross_attn_residual: bool = False
    student_projection_mode: str = "linear"
    student_projection_mlp_ratio: float = 4.0
    student_projection_dropout: float = 0.0
    positional_encodings: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecoderStackConfig":
        data = dict(payload)
        gating_enabled = bool(data.get("gating_enabled", False))
        gating_mode = str(data.get("gating_mode", "none")).lower()
        if gating_enabled and gating_mode == "none": gating_mode = "flamingo"
        if not gating_enabled and gating_mode != "none": raise ValueError("Set gating_mode='none' when gating_enabled is false.")
        return cls(
            decoder_dim=int(data["decoder_dim"]),
            num_decoder_tokens=int(data["num_decoder_tokens"]),
            num_heads=int(data.get("num_heads", 8)),
            num_layers=int(data.get("num_layers", 1)),
            ffn_enabled=bool(data.get("ffn_enabled", True)),
            ffn_mlp_ratio=float(data.get("ffn_mlp_ratio", 4.0)),
            ffn_dropout=float(data.get("ffn_dropout", 0.0)),
            self_attn_queries=bool(data.get("self_attn_queries", True)),
            self_attn_student=bool(data.get("self_attn_student", False)),
            gating_enabled=gating_enabled,
            gating_mode=gating_mode,
            cross_attn_residual=bool(data.get("cross_attn_residual", False)),
            student_projection_mode=str(data.get("student_projection_mode", "linear")),
            student_projection_mlp_ratio=float(data.get("student_projection_mlp_ratio", 4.0)),
            student_projection_dropout=float(data.get("student_projection_dropout", 0.0)),
            positional_encodings=bool(data.get("positional_encodings", False)),
        )


@dataclass(frozen=True)
class DenseMapHeadConfig:
    grid_hw: tuple[int, int]
    hidden_dim: int | None = None
    projection_mode: str = "linear"
    projection_mlp_ratio: float = 4.0
    projection_dropout: float = 0.0
    refine_layers: int = 2
    refine_kernel_size: int = 3
    refine_dropout: float = 0.0
    out_layers: int = 1
    out_hidden_dim: int | None = None
    out_dropout: float = 0.0
    resize_mode: str = "bilinear"
    align_corners: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseMapHeadConfig":
        data = dict(payload)
        return cls(
            grid_hw=_as_tuple2(data["grid_hw"], "head.grid_hw"),
            hidden_dim=None if data.get("hidden_dim") is None else int(data["hidden_dim"]),
            projection_mode=str(data.get("projection_mode", "linear")),
            projection_mlp_ratio=float(data.get("projection_mlp_ratio", 4.0)),
            projection_dropout=float(data.get("projection_dropout", 0.0)),
            refine_layers=int(data.get("refine_layers", 2)),
            refine_kernel_size=int(data.get("refine_kernel_size", 3)),
            refine_dropout=float(data.get("refine_dropout", 0.0)),
            out_layers=int(data.get("out_layers", 1)),
            out_hidden_dim=None if data.get("out_hidden_dim") is None else int(data["out_hidden_dim"]),
            out_dropout=float(data.get("out_dropout", 0.0)),
            resize_mode=str(data.get("resize_mode", "bilinear")),
            align_corners=bool(data.get("align_corners", False)),
        )


@dataclass(frozen=True)
class CeDirNetTeacherConfig:
    name: str = "cedirnet"
    target_kind: str = "dense_map"
    loss_type: str = "mse"
    weight: float = 1.0
    model_type: str | None = None
    image_size: int = 768
    checkpoint: str | None = None
    repo_src: str | None = None
    config_path: str | None = None
    localization_checkpoint: str | None = None
    target_channel_indices: tuple[int, ...] = (0, 1, 2)
    resize: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CeDirNetTeacherConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "cedirnet")),
            target_kind=str(data.get("target_kind", "dense_map")),
            loss_type=str(data.get("loss_type", "mse")),
            weight=float(data.get("weight", 1.0)),
            model_type=None if data.get("model_type") is None else str(data["model_type"]),
            image_size=int(data.get("image_size", 768)),
            checkpoint=None if data.get("checkpoint") is None else str(data["checkpoint"]),
            repo_src=None if data.get("repo_src") is None else str(data["repo_src"]),
            config_path=None if data.get("config_path") is None else str(data["config_path"]),
            localization_checkpoint=None if data.get("localization_checkpoint") is None else str(data["localization_checkpoint"]),
            target_channel_indices=_as_int_tuple(data.get("target_channel_indices", [0, 1, 2]), "teacher.target_channel_indices"),
            resize=bool(data.get("resize", True)),
        )

    @property
    def out_channels(self) -> int:
        return len(self.target_channel_indices)

@dataclass(frozen=True)
class CeDirNetDecoderConfig:
    stack: DecoderStackConfig
    head: DenseMapHeadConfig
    teacher: CeDirNetTeacherConfig

    def validate(self) -> "CeDirNetDecoderConfig":
        expected_tokens = self.head.grid_hw[0] * self.head.grid_hw[1]
        if self.stack.num_decoder_tokens != expected_tokens: raise ValueError(f"num_decoder_tokens={self.stack.num_decoder_tokens} does not match head.grid_hw product {expected_tokens}.")
        if self.teacher.target_kind != "dense_map": raise ValueError(f"CeDirNet currently expects target_kind='dense_map', got {self.teacher.target_kind!r}.")
        return self

def load_cedirnet_decoder_config(stack_path: str | Path = DEFAULT_CEDIRNET_STACK_CONFIG_PATH, head_path: str | Path = DEFAULT_CEDIRNET_HEAD_CONFIG_PATH) -> CeDirNetDecoderConfig:
    stack_payload = _read_yaml(stack_path)
    head_payload = _read_yaml(head_path)
    stack = DecoderStackConfig.from_dict(stack_payload.get("stack", stack_payload))
    head = DenseMapHeadConfig.from_dict(head_payload.get("head", head_payload))
    teacher = CeDirNetTeacherConfig.from_dict(head_payload.get("teacher", head_payload))
    return CeDirNetDecoderConfig(stack=stack, head=head, teacher=teacher).validate()

@dataclass(frozen=True)
class ExpertQueryHeadConfig:
    query_projection_mode: str = "mlp"
    query_projection_mlp_ratio: float = 1.0
    query_projection_dropout: float = 0.0
    query_aggregation_mode: str = "mean"
    align_weight: float = 1.0
    recon_weight: float = 1.0
    recon_scale: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertQueryHeadConfig":
        data = dict(payload)
        return cls(
            query_projection_mode=str(data.get("query_projection_mode", "mlp")),
            query_projection_mlp_ratio=float(data.get("query_projection_mlp_ratio", 1.0)),
            query_projection_dropout=float(data.get("query_projection_dropout", 0.0)),
            query_aggregation_mode=str(data.get("query_aggregation_mode", "mean")),
            align_weight=float(data.get("align_weight", 1.0)),
            recon_weight=float(data.get("recon_weight", 1.0)),
            recon_scale=float(data.get("recon_scale", 1.0)),
        )

@dataclass(frozen=True)
class DinoTeacherConfig:
    name: str = "dinov2"
    target_kind: str = "expert_feature_query"
    loss_type: str = "mse"
    weight: float = 1.0
    model_type: str | None = None
    image_size: int = 476
    checkpoint: str | None = None
    resize_mode: str = "aspect_patch14"
    patch_multiple: int = 14
    size_ref: str = "long_side"
    round_mode: str = "floor"
    layer_indices: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DinoTeacherConfig":
        data = dict(payload)
        raw_layer_indices = data.get("layer_indices")
        layer_indices = None if raw_layer_indices is None else _as_int_tuple(raw_layer_indices, "teacher.layer_indices")
        return cls(
            name=str(data.get("name", "dinov2")),
            target_kind=str(data.get("target_kind", "expert_feature_query")),
            loss_type=str(data.get("loss_type", "mse")),
            weight=float(data.get("weight", 1.0)),
            model_type=None if data.get("model_type") is None else str(data["model_type"]),
            image_size=int(data.get("image_size", 476)),
            checkpoint=None if data.get("checkpoint") is None else str(data["checkpoint"]),
            resize_mode=str(data.get("resize_mode", "aspect_patch14")),
            patch_multiple=int(data.get("patch_multiple", 14)),
            size_ref=str(data.get("size_ref", "long_side")),
            round_mode=str(data.get("round_mode", "floor")),
            layer_indices=layer_indices,
        )


@dataclass(frozen=True)
class DinoFeatureAlignmentConfig:
    stack: DecoderStackConfig
    head: ExpertQueryHeadConfig
    teacher: DinoTeacherConfig

    def validate(self) -> "DinoFeatureAlignmentConfig":
        if self.teacher.target_kind != "expert_feature_query": raise ValueError(f"DINO feature alignment expects target_kind='expert_feature_query', got {self.teacher.target_kind!r}.")
        return self


def load_dino_feature_alignment_config(stack_path: str | Path = DEFAULT_DINO_STACK_CONFIG_PATH, feature_alignment_path: str | Path = DEFAULT_DINO_FEATURE_ALIGNMENT_CONFIG_PATH) -> DinoFeatureAlignmentConfig:
    stack_payload = _read_yaml(stack_path)
    alignment_payload = _read_yaml(feature_alignment_path)
    stack = DecoderStackConfig.from_dict(stack_payload.get("stack", stack_payload))
    head = ExpertQueryHeadConfig.from_dict(alignment_payload.get("head", alignment_payload))
    teacher = DinoTeacherConfig.from_dict(alignment_payload.get("teacher", alignment_payload))
    return DinoFeatureAlignmentConfig(stack=stack, head=head, teacher=teacher).validate()
