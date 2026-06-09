import dataclasses
from dataclasses import dataclass, field
import enum
import json
from pathlib import Path
from typing import Any

from huggingface_hub.constants import CONFIG_NAME
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig

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
        if self.guidance_expert_type != "cedirnet": raise ValueError(f"Only CeDirNet guidance is supported in v1, got {self.guidance_expert_type!r}.")
        if self.guidance_source != "decoder_tokens": raise ValueError(f"Only decoder_tokens guidance_source is supported in v1, got {self.guidance_source!r}.")
        if self.guidance_fusion_mode not in {"concat", "cross_attn"}: raise ValueError(f"guidance_fusion_mode must be one of: concat, cross_attn. Got {self.guidance_fusion_mode!r}.")
        if self.guidance_train_mode not in {"warmup_freeze", "train_from_start", "frozen"}: raise ValueError(f"guidance_train_mode must be one of: warmup_freeze, train_from_start, frozen. Got {self.guidance_train_mode!r}.")
        if int(self.guidance_unfreeze_step) < 0: raise ValueError("guidance_unfreeze_step must be >= 0.")
        if not isinstance(self.guidance_decoder_stack, dict) or not self.guidance_decoder_stack: raise ValueError("guidance_decoder_stack must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_head, dict) or not self.guidance_decoder_head: raise ValueError("guidance_decoder_head must be a non-empty mapping.")
        if not isinstance(self.guidance_decoder_teacher, dict) or not self.guidance_decoder_teacher: raise ValueError("guidance_decoder_teacher must be a non-empty mapping.")

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
        if str(guidance_fusion_mode) == "concat": payload["max_len_seq"] = int(base.max_len_seq) + int(guidance_decoder_stack["num_decoder_tokens"])
        return cls(
            **payload,
            guidance_fusion_mode=str(guidance_fusion_mode),
            guidance_gated=bool(guidance_gated),
            guidance_train_mode=str(guidance_train_mode),
            guidance_unfreeze_step=int(guidance_unfreeze_step),
            guidance_num_heads=guidance_num_heads,
            guidance_decoder_stack=dict(guidance_decoder_stack),
            guidance_decoder_head=dict(guidance_decoder_head),
            guidance_decoder_teacher=dict(guidance_decoder_teacher),
        )
