import dataclasses
from dataclasses import dataclass, field
import enum
import json
from pathlib import Path
from typing import Any

from huggingface_hub.constants import CONFIG_NAME
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from thesis_vla.visual_thought.config import CeDirNetTeacherConfig, DinoTeacherConfig


GUIDANCE_MODES = frozenset({"concat", "selected_layers"})
GUIDANCE_INSERTION_POSITIONS = frozenset({"before_vlm", "after_visual"})
GUIDANCE_FUSION_ALIASES = frozenset({"concat", "gated_concat", "selected_layers"})
LEGACY_REMOVED_FUSION_ALIASES = frozenset({"cross_attention", "gated_cross_attention", "cross_attn"})
GUIDANCE_TRAINING_SCHEDULES = frozenset({"legacy", "decoder_warmup_then_policy"})


def normalize_guidance_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in GUIDANCE_MODES: raise ValueError(f"guidance_mode must be one of: {', '.join(sorted(GUIDANCE_MODES))}. Got {mode!r}.")
    return mode


def normalize_guidance_insertion_position(position: str) -> str:
    position = str(position).strip().lower()
    if position not in GUIDANCE_INSERTION_POSITIONS: raise ValueError(f"guidance_insertion_position must be one of: {', '.join(sorted(GUIDANCE_INSERTION_POSITIONS))}. Got {position!r}.")
    return position


def normalize_guidance_fusion_mode(mode: str, gated: bool | None = None) -> str:
    mode = str(mode).strip().lower()
    if mode in LEGACY_REMOVED_FUSION_ALIASES: raise ValueError("Legacy cross-attention guidance modes are no longer supported. Use concat with ordering/interface options or selected_layers.")
    if mode == "concat": return "gated_concat" if bool(gated) else "concat"
    if mode == "selected_layers":
        if bool(gated): raise ValueError("selected_layers does not support gated_fusion in v1.")
        return "selected_layers"
    if mode == "gated_concat": return "gated_concat"
    raise ValueError(f"guidance_fusion_mode must be one of: {', '.join(sorted(GUIDANCE_FUSION_ALIASES))}. Got {mode!r}.")


def guidance_mode_from_fusion_alias(mode: str, gated: bool | None = None) -> tuple[str, bool]:
    normalized = normalize_guidance_fusion_mode(mode, gated)
    if normalized == "gated_concat": return "concat", True
    return normalized, False


@PreTrainedConfig.register_subclass("xvla_guided")
@dataclass
class XVLAGuidedConfig(XVLAConfig):
    guidance_expert_type: str = "cedirnet"
    guidance_source: str = "decoder_tokens"
    guidance_train_mode: str = "warmup_freeze"
    guidance_unfreeze_step: int = 1_000
    guidance_training_schedule: str = "legacy"
    guidance_warmup_steps: int = 1_000
    guidance_num_heads: int | None = None
    guidance_mode: str = "concat"
    guidance_insertion_position: str = "after_visual"
    guidance_use_interface_projection: bool = False
    guidance_interface_num_tokens: int | None = None
    guidance_concat_gating: bool = False
    guidance_selected_layers: tuple[int, ...] = ()
    guidance_decoder_stack: dict[str, Any] = field(default_factory=dict)
    guidance_decoder_head: dict[str, Any] = field(default_factory=dict)
    guidance_decoder_teacher: dict[str, Any] = field(default_factory=dict)
    guidance_fusion_mode: str | None = None
    guidance_gated: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.guidance_fusion_mode is not None:
            self.guidance_mode, self.guidance_concat_gating = guidance_mode_from_fusion_alias(self.guidance_fusion_mode, self.guidance_gated)
            if self.guidance_mode == "concat":
                self.guidance_insertion_position = "after_visual"; self.guidance_use_interface_projection = False; self.guidance_interface_num_tokens = None
        self.guidance_mode = normalize_guidance_mode(self.guidance_mode)
        self.guidance_insertion_position = normalize_guidance_insertion_position(self.guidance_insertion_position)
        self.guidance_selected_layers = tuple(sorted(set(int(idx) for idx in self.guidance_selected_layers)))
        if self.guidance_interface_num_tokens is not None and int(self.guidance_interface_num_tokens) <= 0: raise ValueError("guidance_interface_num_tokens must be > 0 when provided.")
        self.guidance_concat_gating = bool(self.guidance_concat_gating)
        self.guidance_fusion_mode = "gated_concat" if self.guidance_mode == "concat" and self.guidance_concat_gating else self.guidance_mode
        self.guidance_gated = bool(self.guidance_concat_gating)
        if self.guidance_expert_type not in {"cedirnet", "dino"}: raise ValueError(f"guidance_expert_type must be one of: cedirnet, dino. Got {self.guidance_expert_type!r}.")
        if self.guidance_source != "decoder_tokens": raise ValueError(f"Only decoder_tokens guidance_source is supported in v1, got {self.guidance_source!r}.")
        if self.guidance_train_mode not in {"warmup_freeze", "train_from_start", "frozen"}: raise ValueError(f"guidance_train_mode must be one of: warmup_freeze, train_from_start, frozen. Got {self.guidance_train_mode!r}.")
        if int(self.guidance_unfreeze_step) < 0: raise ValueError("guidance_unfreeze_step must be >= 0.")
        self.guidance_training_schedule = str(self.guidance_training_schedule).strip().lower()
        if self.guidance_training_schedule not in GUIDANCE_TRAINING_SCHEDULES: raise ValueError(f"guidance_training_schedule must be one of: {', '.join(sorted(GUIDANCE_TRAINING_SCHEDULES))}. Got {self.guidance_training_schedule!r}.")
        if int(self.guidance_warmup_steps) < 0: raise ValueError("guidance_warmup_steps must be >= 0.")
        if not isinstance(self.guidance_decoder_stack, dict) or not self.guidance_decoder_stack: raise ValueError("guidance_decoder_stack must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_head, dict) or not self.guidance_decoder_head: raise ValueError("guidance_decoder_head must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_teacher, dict) or not self.guidance_decoder_teacher: raise ValueError("guidance_decoder_teacher must be a non-empty mapping.")
        if self.guidance_mode == "selected_layers":
            if self.guidance_use_interface_projection: raise ValueError("selected_layers mode does not support guidance_use_interface_projection in v1.")
            if self.guidance_concat_gating: raise ValueError("selected_layers mode does not support guidance_concat_gating in v1.")
            if len(self.guidance_selected_layers) == 0: raise ValueError("selected_layers mode requires a non-empty guidance_selected_layers tuple.")
        target_kind = str(self.guidance_decoder_teacher.get("target_kind", "")).strip()
        if self.guidance_expert_type == "cedirnet":
            CeDirNetTeacherConfig.from_dict(self.guidance_decoder_teacher)
            if target_kind != "dense_map": raise ValueError(f"CeDirNet guided policy expects teacher.target_kind='dense_map', got {target_kind!r}.")
        else:
            DinoTeacherConfig.from_dict(self.guidance_decoder_teacher)
            if target_kind != "token_sequence": raise ValueError(f"Guided DINO v1 supports teacher.target_kind='token_sequence' only, got {target_kind!r}.")

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if dataclasses.is_dataclass(value): return {field.name: XVLAGuidedConfig._to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, enum.Enum): return value.value
        if isinstance(value, Path): return str(value)
        if isinstance(value, tuple): return [XVLAGuidedConfig._to_jsonable(item) for item in value]
        if isinstance(value, list): return [XVLAGuidedConfig._to_jsonable(item) for item in value]
        if isinstance(value, dict): return {str(key): XVLAGuidedConfig._to_jsonable(item) for key, item in value.items()}
        return value

    def _save_pretrained(self, save_directory: Path) -> None:
        payload = self._to_jsonable(self)
        payload["type"] = self.type
        (save_directory / CONFIG_NAME).write_text(json.dumps(payload, indent=4))

    @property
    def resolved_guidance_num_heads(self) -> int:
        return int(self.guidance_num_heads or self.num_heads)

    @property
    def guidance_num_tokens(self) -> int:
        return int(self.guidance_decoder_stack["num_decoder_tokens"])

    @property
    def guidance_appended_num_tokens(self) -> int:
        return int(self.guidance_interface_num_tokens or self.guidance_num_tokens) if bool(self.guidance_use_interface_projection) else int(self.guidance_num_tokens)

    @classmethod
    def from_xvla_config(
        cls,
        base: XVLAConfig,
        *,
        guidance_expert_type: str = "cedirnet",
        guidance_decoder_stack: dict[str, Any],
        guidance_decoder_head: dict[str, Any],
        guidance_decoder_teacher: dict[str, Any],
        guidance_mode: str = "concat",
        guidance_insertion_position: str = "after_visual",
        guidance_use_interface_projection: bool = False,
        guidance_interface_num_tokens: int | None = None,
        guidance_concat_gating: bool = False,
        guidance_selected_layers: tuple[int, ...] | list[int] = (),
        guidance_train_mode: str = "warmup_freeze",
        guidance_unfreeze_step: int = 1_000,
        guidance_training_schedule: str = "legacy",
        guidance_warmup_steps: int = 1_000,
        guidance_num_heads: int | None = None,
        guidance_fusion_mode: str | None = None,
        guidance_gated: bool | None = None,
    ) -> "XVLAGuidedConfig":
        payload = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        if guidance_fusion_mode is not None:
            guidance_mode, guidance_concat_gating = guidance_mode_from_fusion_alias(guidance_fusion_mode, guidance_gated)
            if guidance_mode == "concat":
                guidance_insertion_position = "after_visual"; guidance_use_interface_projection = False; guidance_interface_num_tokens = None
        guidance_mode = normalize_guidance_mode(guidance_mode)
        guidance_insertion_position = normalize_guidance_insertion_position(guidance_insertion_position)
        appended_tokens = int(guidance_interface_num_tokens or guidance_decoder_stack["num_decoder_tokens"]) if bool(guidance_use_interface_projection) else int(guidance_decoder_stack["num_decoder_tokens"])
        payload["max_len_seq"] = int(base.max_len_seq) + appended_tokens
        return cls(
            **payload,
            guidance_expert_type=str(guidance_expert_type),
            guidance_mode=guidance_mode,
            guidance_insertion_position=guidance_insertion_position,
            guidance_use_interface_projection=bool(guidance_use_interface_projection),
            guidance_interface_num_tokens=guidance_interface_num_tokens,
            guidance_concat_gating=bool(guidance_concat_gating),
            guidance_selected_layers=tuple(int(idx) for idx in guidance_selected_layers),
            guidance_train_mode=str(guidance_train_mode),
            guidance_unfreeze_step=int(guidance_unfreeze_step),
            guidance_training_schedule=str(guidance_training_schedule),
            guidance_warmup_steps=int(guidance_warmup_steps),
            guidance_num_heads=guidance_num_heads,
            guidance_decoder_stack=dict(guidance_decoder_stack),
            guidance_decoder_head=dict(guidance_decoder_head),
            guidance_decoder_teacher=dict(guidance_decoder_teacher),
            guidance_fusion_mode=None,
            guidance_gated=None,
        )
