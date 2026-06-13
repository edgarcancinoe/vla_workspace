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


GUIDANCE_FUSION_MODES = frozenset({"concat", "gated_concat", "cross_attention", "gated_cross_attention"})


def normalize_guidance_fusion_mode(mode: str, gated: bool | None = None) -> str:
    mode = str(mode).strip()
    if mode == "concat": return "gated_concat" if bool(gated) else "concat"
    if mode == "cross_attention": return "gated_cross_attention" if bool(gated) else "cross_attention"
    if mode == "cross_attn": return "gated_cross_attention" if bool(gated) else "cross_attention"
    if mode in {"gated_concat", "gated_cross_attention"}: return mode
    raise ValueError(f"guidance_fusion_mode must be one of: {', '.join(sorted(GUIDANCE_FUSION_MODES))}. Got {mode!r}.")


@PreTrainedConfig.register_subclass("xvla_guided")
@dataclass
class XVLAGuidedConfig(XVLAConfig):
    guidance_expert_type: str = "cedirnet"
    guidance_source: str = "decoder_tokens"
    guidance_fusion_mode: str = "concat"
    guidance_gated: bool = False
    guidance_train_mode: str = "warmup_freeze"
    guidance_unfreeze_step: int = 1_000
    guidance_num_heads: int | None = None
    guidance_decoder_stack: dict[str, Any] = field(default_factory=dict)
    guidance_decoder_head: dict[str, Any] = field(default_factory=dict)
    guidance_decoder_teacher: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.guidance_fusion_mode = normalize_guidance_fusion_mode(self.guidance_fusion_mode, self.guidance_gated)
        self.guidance_gated = self.guidance_fusion_mode.startswith("gated_")
        if self.guidance_expert_type not in {"cedirnet", "dino"}: raise ValueError(f"guidance_expert_type must be one of: cedirnet, dino. Got {self.guidance_expert_type!r}.")
        if self.guidance_source != "decoder_tokens": raise ValueError(f"Only decoder_tokens guidance_source is supported in v1, got {self.guidance_source!r}.")
        if self.guidance_train_mode not in {"warmup_freeze", "train_from_start", "frozen"}: raise ValueError(f"guidance_train_mode must be one of: warmup_freeze, train_from_start, frozen. Got {self.guidance_train_mode!r}.")
        if int(self.guidance_unfreeze_step) < 0: raise ValueError("guidance_unfreeze_step must be >= 0.")
        if not isinstance(self.guidance_decoder_stack, dict) or not self.guidance_decoder_stack: raise ValueError("guidance_decoder_stack must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_head, dict) or not self.guidance_decoder_head: raise ValueError("guidance_decoder_head must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_teacher, dict) or not self.guidance_decoder_teacher: raise ValueError("guidance_decoder_teacher must be a non-empty mapping.")
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

    @classmethod
    def from_xvla_config(
        cls,
        base: XVLAConfig,
        *,
        guidance_expert_type: str = "cedirnet",
        guidance_decoder_stack: dict[str, Any],
        guidance_decoder_head: dict[str, Any],
        guidance_decoder_teacher: dict[str, Any],
        guidance_fusion_mode: str = "concat",
        guidance_gated: bool = False,
        guidance_train_mode: str = "warmup_freeze",
        guidance_unfreeze_step: int = 1_000,
        guidance_num_heads: int | None = None,
    ) -> "XVLAGuidedConfig":
        payload = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        guidance_fusion_mode = normalize_guidance_fusion_mode(guidance_fusion_mode, guidance_gated)
        if guidance_fusion_mode in {"concat", "gated_concat"}: payload["max_len_seq"] = int(base.max_len_seq) + int(guidance_decoder_stack["num_decoder_tokens"])
        return cls(
            **payload,
            guidance_expert_type=str(guidance_expert_type),
            guidance_fusion_mode=guidance_fusion_mode,
            guidance_gated=guidance_fusion_mode.startswith("gated_"),
            guidance_train_mode=str(guidance_train_mode),
            guidance_unfreeze_step=int(guidance_unfreeze_step),
            guidance_num_heads=guidance_num_heads,
            guidance_decoder_stack=dict(guidance_decoder_stack),
            guidance_decoder_head=dict(guidance_decoder_head),
            guidance_decoder_teacher=dict(guidance_decoder_teacher),
        )
