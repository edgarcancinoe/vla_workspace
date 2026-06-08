from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from thesis_vla.visual_thought.config import DecoderStackConfig, DinoDecoderConfig, ExpertQueryHeadConfig
from thesis_vla.visual_thought.decoder_stack import StackedDecoderStrategy, StudentProjection
from thesis_vla.visual_thought.targets import TeacherTarget


class ExpertFeatureQueryHead(nn.Module):
    def __init__(self, input_dim: int, expert_dim: int, num_query_tokens: int = 4, projection_mode: str = "mlp", projection_mlp_ratio: float = 1.0, projection_dropout: float = 0.0, aggregation_mode: str = "mean") -> None:
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.expert_dim = int(expert_dim)
        self.aggregation_mode = str(aggregation_mode).lower()
        if self.aggregation_mode not in {"mean", "sum", "max", "none"}: raise ValueError("query_aggregation_mode must be one of: mean, sum, max, none.")
        self.query_projection_mode = str(projection_mode).lower()
        if self.query_projection_mode == "identity":
            if int(input_dim) != self.expert_dim: raise ValueError("query_projection_mode='identity' requires input_dim == expert_dim.")
            self.query_token_generator = nn.Identity()
        else:
            self.query_token_generator = StudentProjection(input_dim, self.expert_dim, mode=self.query_projection_mode, mlp_ratio=projection_mlp_ratio, dropout=projection_dropout)

    def aggregate(self, token_maps_stu: torch.Tensor) -> torch.Tensor:
        if self.aggregation_mode == "none": return token_maps_stu
        if self.aggregation_mode == "mean": return token_maps_stu.mean(dim=1, keepdim=True)
        if self.aggregation_mode == "sum": return token_maps_stu.sum(dim=1, keepdim=True)
        if self.aggregation_mode == "max": return token_maps_stu.max(dim=1, keepdim=True).values
        raise ValueError(f"Unsupported query_aggregation_mode: {self.aggregation_mode}")

    def _validate_student_tokens(self, student_tokens_stu: torch.Tensor):
        if int(student_tokens_stu.shape[1]) != self.num_query_tokens: raise ValueError(f"Expert feature reconstruction expected {self.num_query_tokens} student tokens, got {int(student_tokens_stu.shape[1])}.")

    def _resolve_patch_grid(self, feat_exp: torch.Tensor, target: TeacherTarget) -> tuple[torch.Tensor, int, int]:
        n = int(feat_exp.shape[1])
        aux_hw = target.aux.get("expert_spatial_hw", target.aux.get("patch_hw"))
        if not isinstance(aux_hw, (tuple, list)) or len(aux_hw) != 2: raise ValueError("Patch expert features require aux['expert_spatial_hw'] or aux['patch_hw'].")
        ph, pw = int(aux_hw[0]), int(aux_hw[1])
        if ph * pw < n: return feat_exp[:, -ph * pw:, :], ph, pw
        if ph * pw != n: raise ValueError(f"Patch expert spatial_hw={aux_hw} expects {ph * pw} tokens, got {n}.")
        return feat_exp, ph, pw

    def _expert_features(self, target: TeacherTarget) -> tuple[str, torch.Tensor]:
        layout = str(target.aux.get("expert_feature_layout", "patch")).lower()
        features_exp = target.aux.get("expert_features", target.aux.get("patch_feats"))
        if features_exp is None: raise ValueError("ExpertFeatureQueryHead requires aux['expert_features'] or aux['patch_feats'].")
        if layout not in {"patch", "channel_map"}: raise ValueError("expert_feature_layout must be one of: patch, channel_map.")
        return layout, features_exp

    def _query_patch_features(self, query_tokens_stu: torch.Tensor, features_exp: torch.Tensor, target: TeacherTarget) -> torch.Tensor:
        if features_exp.ndim != 4: raise ValueError(f"Patch expert features must have shape [B,L,N_s,d_exp], got {tuple(features_exp.shape)}.")
        if int(features_exp.shape[1]) != self.num_query_tokens: raise ValueError(f"Patch expert features expected L={self.num_query_tokens}, got {int(features_exp.shape[1])}.")
        if int(features_exp.shape[-1]) != self.expert_dim: raise ValueError(f"Patch expert features expected d_exp={self.expert_dim}, got {int(features_exp.shape[-1])}.")
        batch_size = int(query_tokens_stu.shape[0])
        out_h, out_w = int(target.tensor.shape[-2]), int(target.tensor.shape[-1])
        response_maps_stu = []
        for i in range(self.num_query_tokens):
            query_token_stu = query_tokens_stu[:, i, :].unsqueeze(1).to(dtype=features_exp.dtype)
            feat_exp, ph, pw = self._resolve_patch_grid(features_exp[:, i], target)
            response_map_stu = torch.bmm(query_token_stu, feat_exp.transpose(1, 2)).squeeze(1).view(batch_size, 1, ph, pw)
            response_maps_stu.append(F.interpolate(response_map_stu, size=(out_h, out_w), mode="bilinear", align_corners=False))
        return torch.cat(response_maps_stu, dim=1)

    def _query_channel_map_features(self, query_tokens_stu: torch.Tensor, features_exp: torch.Tensor, target: TeacherTarget) -> torch.Tensor:
        if features_exp.ndim == 4:
            if int(features_exp.shape[1]) != self.expert_dim: raise ValueError(f"Channel-map expert features expected d_exp={self.expert_dim}, got {int(features_exp.shape[1])}.")
            grouped = False
        elif features_exp.ndim == 5:
            if int(features_exp.shape[1]) != self.num_query_tokens: raise ValueError(f"Grouped channel-map expert features expected L={self.num_query_tokens}, got {int(features_exp.shape[1])}.")
            if int(features_exp.shape[2]) != self.expert_dim: raise ValueError(f"Grouped channel-map expert features expected d_exp={self.expert_dim}, got {int(features_exp.shape[2])}.")
            grouped = True
        else:
            raise ValueError(f"Channel-map expert features must have shape [B,d_exp,H,W] or [B,L,d_exp,H,W], got {tuple(features_exp.shape)}.")
        out_h, out_w = int(target.tensor.shape[-2]), int(target.tensor.shape[-1])
        response_maps_stu = []
        for i in range(self.num_query_tokens):
            per_query_maps_stu = []
            for b in range(int(query_tokens_stu.shape[0])):
                fmap_exp = features_exp[b : b + 1, i] if grouped else features_exp[b : b + 1]
                weight_stu = query_tokens_stu[b, i, :].to(dtype=features_exp.dtype).view(1, self.expert_dim, 1, 1)
                per_query_maps_stu.append(F.conv2d(fmap_exp, weight_stu))
            response_map_stu = torch.cat(per_query_maps_stu, dim=0)
            if response_map_stu.shape[-2:] != (out_h, out_w): response_map_stu = F.interpolate(response_map_stu, size=(out_h, out_w), mode="bilinear", align_corners=False)
            response_maps_stu.append(response_map_stu)
        return torch.cat(response_maps_stu, dim=1)

    def project_to_query_tokens(self, student_tokens_stu: torch.Tensor) -> torch.Tensor:
        self._validate_student_tokens(student_tokens_stu)
        return self.query_token_generator(student_tokens_stu)

    def reconstruct_from_query_tokens(self, query_tokens_stu: torch.Tensor, target: TeacherTarget) -> tuple[torch.Tensor, torch.Tensor]:
        layout, features_exp = self._expert_features(target)
        token_maps_stu = self._query_patch_features(query_tokens_stu, features_exp, target) if layout == "patch" else self._query_channel_map_features(query_tokens_stu, features_exp, target)
        return token_maps_stu, self.aggregate(token_maps_stu)

    def align_features_from_query_tokens(self, query_tokens_stu: torch.Tensor, target: TeacherTarget) -> tuple[torch.Tensor, torch.Tensor]:
        layout, patch_feats_exp = self._expert_features(target)
        if layout != "patch": raise ValueError("align_features_from_query_tokens only supports patch expert_feature_layout.")
        attended_stu, teacher_aligned_exp = [], []
        for i in range(self.num_query_tokens):
            query_token_stu = query_tokens_stu[:, i, :].unsqueeze(1).to(dtype=patch_feats_exp.dtype)
            feat_exp, _, _ = self._resolve_patch_grid(patch_feats_exp[:, i], target)
            score_stu = torch.bmm(query_token_stu, feat_exp.transpose(1, 2)) / float(max(self.expert_dim, 1) ** 0.5)
            weights_stu = torch.softmax(score_stu, dim=-1)
            attended_stu.append(torch.bmm(weights_stu, feat_exp).squeeze(1))
            teacher_aligned_exp.append(feat_exp.mean(dim=1))
        return torch.stack(attended_stu, dim=1), torch.stack(teacher_aligned_exp, dim=1)

    def forward(self, student_tokens_stu: torch.Tensor, target: TeacherTarget) -> torch.Tensor:
        query_tokens_stu = self.project_to_query_tokens(student_tokens_stu)
        _, final_map = self.reconstruct_from_query_tokens(query_tokens_stu, target)
        return final_map


def resolve_expert_query_metadata(target: TeacherTarget) -> tuple[int, int]:
    layout = str(target.aux.get("expert_feature_layout", "patch")).lower()
    features_exp = target.aux.get("expert_features", target.aux.get("patch_feats"))
    
    if features_exp is None: 
        raise ValueError("expert_feature_query requires aux['expert_features'] or aux['patch_feats'].")
    
    if layout == "patch":
        if features_exp.ndim != 4: raise ValueError(f"Patch expert features must have shape [B,L,N_s,d_exp], got {tuple(features_exp.shape)}.")
        return int(features_exp.shape[1]), int(features_exp.shape[-1])
    
    if layout == "channel_map":
        if features_exp.ndim == 4:
            num_query_tokens = target.aux.get("num_query_tokens")
            if num_query_tokens is None: 
                raise ValueError("Shared channel_map expert features require aux['num_query_tokens'].")
            return int(num_query_tokens), int(features_exp.shape[1])
        if features_exp.ndim == 5: 
            return int(features_exp.shape[1]), int(features_exp.shape[2])
        raise ValueError(f"Channel-map expert features must have shape [B,d_exp,H,W] or [B,L,d_exp,H,W], got {tuple(features_exp.shape)}.")
    raise ValueError("expert_feature_layout must be one of: patch, channel_map.")


def compute_feature_alignment_loss(attended_stu: torch.Tensor, teacher_aligned_exp: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return float(weight) * F.mse_loss(attended_stu, teacher_aligned_exp)


class DinoTokenSequenceModel(nn.Module):
    def __init__(self, student_vlm_dim: int, stack_cfg: DecoderStackConfig) -> None:
        super().__init__()
        self.strategy = StackedDecoderStrategy(student_vlm_dim=student_vlm_dim, cfg=stack_cfg)

    @staticmethod
    def resolved_stack_config(target: TeacherTarget, stack_cfg: DecoderStackConfig) -> DecoderStackConfig:
        if target.tensor.ndim != 3: raise ValueError(f"DINO token_sequence target must have shape [B,N,D], got {tuple(target.tensor.shape)}.")
        return dataclasses.replace(stack_cfg, decoder_dim=int(target.tensor.shape[-1]), num_decoder_tokens=int(target.tensor.shape[1]))

    @classmethod
    def from_config(cls, student_vlm_dim: int, target: TeacherTarget, cfg: DinoDecoderConfig) -> "DinoTokenSequenceModel":
        return cls(student_vlm_dim=student_vlm_dim, stack_cfg=cls.resolved_stack_config(target, cfg.stack))

    def forward(self, vlm_features: torch.Tensor) -> torch.Tensor:
        return self.strategy(vlm_features)


class DinoFeatureAlignmentModel(nn.Module):
    def __init__(self, student_vlm_dim: int, num_query_tokens: int, expert_dim: int, stack_cfg: DecoderStackConfig, head_cfg: ExpertQueryHeadConfig) -> None:
        super().__init__()
        if int(stack_cfg.num_decoder_tokens) != int(num_query_tokens): raise ValueError(f"num_decoder_tokens={stack_cfg.num_decoder_tokens} must equal num_query_tokens={num_query_tokens} for expert feature alignment.")
        self.strategy = StackedDecoderStrategy(student_vlm_dim=student_vlm_dim, cfg=stack_cfg)
        self.head = ExpertFeatureQueryHead(input_dim=stack_cfg.decoder_dim, expert_dim=expert_dim, num_query_tokens=num_query_tokens, projection_mode=head_cfg.query_projection_mode, projection_mlp_ratio=head_cfg.query_projection_mlp_ratio, projection_dropout=head_cfg.query_projection_dropout, aggregation_mode=head_cfg.query_aggregation_mode)

    @classmethod
    def from_config(cls, student_vlm_dim: int, target: TeacherTarget, cfg: DinoDecoderConfig) -> "DinoFeatureAlignmentModel":
        num_query_tokens, expert_dim = resolve_expert_query_metadata(target)
        return cls(student_vlm_dim=student_vlm_dim, num_query_tokens=num_query_tokens, expert_dim=expert_dim, stack_cfg=cfg.stack, head_cfg=cfg.head)

    def query_tokens(self, vlm_features: torch.Tensor) -> torch.Tensor:
        return self.head.project_to_query_tokens(self.strategy(vlm_features))

    def query_reconstruct(self, query_tokens_stu: torch.Tensor, target: TeacherTarget) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head.reconstruct_from_query_tokens(query_tokens_stu, target)

    def query_align_features(self, query_tokens_stu: torch.Tensor, target: TeacherTarget) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head.align_features_from_query_tokens(query_tokens_stu, target)
