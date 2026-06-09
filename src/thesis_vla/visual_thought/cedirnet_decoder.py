from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thesis_vla.visual_thought.config import CeDirNetDecoderConfig, DecoderStackConfig, DenseMapHeadConfig
from thesis_vla.visual_thought.decoder_stack import StackedDecoderStrategy, StudentProjection


class DenseMapDecoderHead(nn.Module):
    def __init__(self, input_dim: int, out_channels: int, cfg: DenseMapHeadConfig) -> None:
        super().__init__()
        hidden_dim = int(cfg.hidden_dim or input_dim)
        self.grid_hw = tuple(int(v) for v in cfg.grid_hw)
        self.resize_mode = cfg.resize_mode
        self.align_corners = cfg.align_corners
        self.proj = StudentProjection(input_dim, hidden_dim, mode=cfg.projection_mode, mlp_ratio=cfg.projection_mlp_ratio, dropout=cfg.projection_dropout)
        self.refine = self._make_refine(hidden_dim, refine_layers=cfg.refine_layers, kernel_size=cfg.refine_kernel_size, dropout=cfg.refine_dropout)
        self.out = self._make_out(hidden_dim, int(out_channels), out_layers=cfg.out_layers, out_hidden_dim=cfg.out_hidden_dim, dropout=cfg.out_dropout)

    @staticmethod
    def _make_refine(hidden_dim: int, refine_layers: int, kernel_size: int, dropout: float) -> nn.Module:
        layers = []
        pad = int(kernel_size) // 2
        for _ in range(max(int(refine_layers), 0)):
            layers.extend([nn.Conv2d(hidden_dim, hidden_dim, int(kernel_size), padding=pad), nn.GELU()])
            if float(dropout) > 0.0: layers.append(nn.Dropout2d(float(dropout)))
        return nn.Sequential(*layers) if layers else nn.Identity()

    @staticmethod
    def _make_out(hidden_dim: int, out_channels: int, out_layers: int, out_hidden_dim: int | None, dropout: float) -> nn.Module:
        out_layers = max(int(out_layers), 1)
        if out_layers == 1: return nn.Conv2d(hidden_dim, out_channels, 1)
        hidden = int(out_hidden_dim or hidden_dim)
        layers = []
        for i in range(out_layers - 1):
            layers.extend([nn.Conv2d(hidden_dim if i == 0 else hidden, hidden, 1), nn.GELU()])
            if float(dropout) > 0.0: layers.append(nn.Dropout2d(float(dropout)))
        layers.append(nn.Conv2d(hidden, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, student_tokens: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        gh, gw = self.grid_hw
        if int(student_tokens.shape[1]) != gh * gw: raise ValueError(f"Expected {gh * gw} tokens for grid_hw={self.grid_hw}, got {int(student_tokens.shape[1])}.")
        feat = self.proj(student_tokens).transpose(1, 2).reshape(student_tokens.shape[0], -1, gh, gw)
        out = self.out(self.refine(feat))
        if target_hw is not None and out.shape[-2:] != target_hw: out = F.interpolate(out, size=target_hw, mode=self.resize_mode, align_corners=self.align_corners if self.resize_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None)
        return out


class CeDirNetDistillationModel(nn.Module):
    def __init__(self, student_vlm_dim: int, stack_cfg: DecoderStackConfig, head_cfg: DenseMapHeadConfig, out_channels: int) -> None:
        super().__init__()
        expected_tokens = head_cfg.grid_hw[0] * head_cfg.grid_hw[1]
        if stack_cfg.num_decoder_tokens != expected_tokens: raise ValueError(f"num_decoder_tokens={stack_cfg.num_decoder_tokens} must equal grid product {expected_tokens}.")
        self.strategy = StackedDecoderStrategy(student_vlm_dim=student_vlm_dim, cfg=stack_cfg)
        self.head = DenseMapDecoderHead(input_dim=stack_cfg.decoder_dim, out_channels=out_channels, cfg=head_cfg)

    @classmethod
    def from_config(cls, student_vlm_dim: int, cfg: CeDirNetDecoderConfig) -> "CeDirNetDistillationModel":
        return cls(student_vlm_dim=student_vlm_dim, stack_cfg=cfg.stack, head_cfg=cfg.head, out_channels=cfg.teacher.out_channels)

    def decoder_tokens(self, vlm_features: torch.Tensor) -> torch.Tensor:
        return self.strategy(vlm_features)

    def predict_from_tokens(self, student_tokens: torch.Tensor, target_map: torch.Tensor | None = None, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        target_hw = tuple(int(v) for v in target_map.shape[-2:]) if target_map is not None else output_size
        return self.head(student_tokens, target_hw=target_hw)

    def forward(self, vlm_features: torch.Tensor, target_map: torch.Tensor | None = None, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        return self.predict_from_tokens(self.decoder_tokens(vlm_features), target_map=target_map, output_size=output_size)
