from __future__ import annotations

import math

import torch
from torch import nn

from thesis_vla.visual_thought.config import DecoderStackConfig


class StudentProjection(nn.Module):
    def __init__(self, student_vlm_dim: int, decoder_dim: int, mode: str = "linear", mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"linear", "mlp", "res_mlp"}: raise ValueError("student projection mode must be one of: linear, mlp, res_mlp.")
        hidden_dim = int(student_vlm_dim * mlp_ratio)
        if self.mode == "linear":
            self.proj = nn.Linear(student_vlm_dim, decoder_dim)
            return
        if self.mode == "mlp":
            self.proj = nn.Sequential(nn.LayerNorm(student_vlm_dim), nn.Linear(student_vlm_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, decoder_dim), nn.Dropout(dropout))
            return
        self.skip = nn.Linear(student_vlm_dim, decoder_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(student_vlm_dim), nn.Linear(student_vlm_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, decoder_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters(), None)
        if parameter is not None and (x.dtype != parameter.dtype or x.device != parameter.device): x = x.to(device=parameter.device, dtype=parameter.dtype)
        if self.mode == "res_mlp": return self.skip(x) + self.mlp(x)
        return self.proj(x)


class StackedDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, ffn_enabled: bool = True, ffn_mlp_ratio: float = 4.0, ffn_dropout: float = 0.0, self_attn_queries: bool = True, self_attn_student: bool = False, gating_enabled: bool = False, gating_mode: str | None = None, cross_attn_residual: bool = False) -> None:
        super().__init__()
        self.ffn_enabled = bool(ffn_enabled)
        self.self_attn_queries = bool(self_attn_queries)
        self.self_attn_student = bool(self_attn_student)
        self.cross_attn_residual = bool(cross_attn_residual)
        self.gating_mode = ("flamingo" if gating_enabled else "none") if gating_mode is None else str(gating_mode).lower()
        if self.gating_mode not in {"none", "flamingo", "linear_sigmoid"}: raise ValueError("gating_mode must be one of: none, flamingo, linear_sigmoid.")
        if not self.ffn_enabled: self.gating_mode = "none"
        if self.self_attn_student:
            self.student_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=ffn_dropout, batch_first=True)
            self.student_self_attn_norm = nn.LayerNorm(hidden_dim)
        if self.self_attn_queries:
            self.query_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=ffn_dropout, batch_first=True)
            self.query_self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=ffn_dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim) if self.ffn_enabled else None
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim * ffn_mlp_ratio)), nn.GELU(), nn.Dropout(ffn_dropout), nn.Linear(int(hidden_dim * ffn_mlp_ratio), hidden_dim), nn.Dropout(ffn_dropout)) if self.ffn_enabled else None
        if self.gating_mode == "flamingo": self.ffn_gate = nn.Parameter(torch.zeros(1))
        if self.gating_mode == "linear_sigmoid": self.gate_proj = nn.Linear(hidden_dim, 1)

    def forward(self, queries: torch.Tensor, student_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.self_attn_student:
            student_norm = self.student_self_attn_norm(student_features)
            student_att, _ = self.student_self_attn(student_norm, student_norm, student_norm, need_weights=False)
            student_features = student_features + student_att
        if self.self_attn_queries:
            query_norm = self.query_self_attn_norm(queries)
            query_att, _ = self.query_self_attn(query_norm, query_norm, query_norm, need_weights=False)
            queries = queries + query_att
        cross_att, _ = self.cross_attn(queries, student_features, student_features, need_weights=False)
        ffn_in = queries + cross_att if self.cross_attn_residual else cross_att
        if self.ffn is None: return ffn_in, student_features
        ffn_out = self.ffn(self.ffn_norm(ffn_in))
        if self.gating_mode == "flamingo": return ffn_in + torch.tanh(self.ffn_gate) * ffn_out, student_features
        if self.gating_mode == "linear_sigmoid": return ffn_in + torch.sigmoid(self.gate_proj(ffn_out)) * ffn_out, student_features
        return ffn_in + ffn_out, student_features


class StackedDecoderStrategy(nn.Module):
    @staticmethod
    def _sinusoidal_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
        if int(dim) <= 0: return positions.new_zeros((positions.numel(), 0))
        scale = -math.log(10000.0) / max(int(dim) // 2 - 1, 1)
        freqs = torch.exp(torch.arange(0, int(dim), 2, device=positions.device, dtype=torch.float32) * scale)
        angles = positions.to(dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = positions.new_zeros((positions.numel(), int(dim)), dtype=torch.float32)
        emb[:, 0::2] = torch.sin(angles)
        if int(dim) > 1: emb[:, 1::2] = torch.cos(angles[:, : emb[:, 1::2].shape[1]])
        return emb

    @classmethod
    def _sinusoidal_2d(cls, side: int, dim: int) -> torch.Tensor:
        ys = torch.arange(int(side), dtype=torch.float32)
        xs = torch.arange(int(side), dtype=torch.float32)
        y_dim = int(dim) // 2
        x_dim = int(dim) - y_dim
        y_emb = cls._sinusoidal_1d(ys, y_dim).unsqueeze(1).expand(-1, int(side), -1)
        x_emb = cls._sinusoidal_1d(xs, x_dim).unsqueeze(0).expand(int(side), -1, -1)
        return torch.cat([y_emb, x_emb], dim=-1).reshape(int(side) * int(side), int(dim))

    def __init__(self, student_vlm_dim: int, cfg: DecoderStackConfig) -> None:
        super().__init__()
        if cfg.num_layers < 1: 
            raise ValueError("num_layers must be >= 1.")
        if cfg.decoder_dim % cfg.num_heads != 0: 
            raise ValueError("decoder_dim must be divisible by num_heads.")


        self.student_projection = StudentProjection(student_vlm_dim, cfg.decoder_dim, mode=cfg.student_projection_mode, mlp_ratio=cfg.student_projection_mlp_ratio, dropout=cfg.student_projection_dropout)
        self.query_vectors = nn.Parameter(torch.randn(cfg.num_decoder_tokens, cfg.decoder_dim) * 0.02)
        if cfg.positional_encodings:
            side = int(math.isqrt(cfg.num_decoder_tokens))
            if side * side != cfg.num_decoder_tokens: 
                raise ValueError("positional_encodings requires num_decoder_tokens to be a perfect square.")
            self.register_buffer("pos_emb", self._sinusoidal_2d(side, cfg.decoder_dim), persistent=True)
        else:
            self.register_buffer("pos_emb", None, persistent=True)
        self.blocks = nn.ModuleList([StackedDecoderBlock(cfg.decoder_dim, num_heads=cfg.num_heads, ffn_enabled=cfg.ffn_enabled, ffn_mlp_ratio=cfg.ffn_mlp_ratio, ffn_dropout=cfg.ffn_dropout, self_attn_queries=cfg.self_attn_queries, self_attn_student=cfg.self_attn_student, gating_enabled=cfg.gating_enabled, gating_mode=cfg.gating_mode, cross_attn_residual=cfg.cross_attn_residual) for _ in range(cfg.num_layers)])

    def forward(self, vlm_features: torch.Tensor) -> torch.Tensor:
        student_features = self.student_projection(vlm_features)
        queries = self.query_vectors.unsqueeze(0).expand(vlm_features.shape[0], -1, -1)
        
        if self.pos_emb is not None: 
            queries = queries + self.pos_emb.unsqueeze(0)
        
        for block in self.blocks: 
            queries, student_features = block(queries, student_features)
        return queries
