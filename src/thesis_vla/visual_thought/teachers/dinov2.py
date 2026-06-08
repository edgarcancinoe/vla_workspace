from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from thesis_vla.visual_thought.config import DinoTeacherConfig
from thesis_vla.visual_thought.targets import TeacherTarget


class _DinoFeatureModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        return self.backbone.forward_features(x)


class DinoV2Teacher:
    def __init__(self, cfg: DinoTeacherConfig):
        self.cfg = cfg
        self.hub_name = f"dinov2_{cfg.model_type or 'vitb14'}"
        self.model = None
        self.backbone = None

    def _load(self, device: torch.device):
        if self.model is None:
            backbone = torch.hub.load("facebookresearch/dinov2", self.hub_name).to(device)
            backbone.eval()
            for parameter in backbone.parameters(): parameter.requires_grad = False
            self.backbone = backbone
            self.model = _DinoFeatureModel(backbone).to(device)
            self.model.eval()
        elif next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
            self.backbone = self.backbone.to(device)

    def _prep(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        x = x / 255.0 if x.max() > 1.0 else x
        x = x.clamp(0.0, 1.0)
        size = int(self.cfg.image_size or 224)
        mode = str(self.cfg.resize_mode)
        if mode == "square":
            if x.shape[-2:] != (size, size): x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            return x
        patch = max(int(self.cfg.patch_multiple), 1)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if mode == "native_patch14":
            nh = max((h // patch) * patch, patch)
            nw = max((w // patch) * patch, patch)
            if x.shape[-2:] != (nh, nw): x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            return x
        ref = max(h, w) if str(self.cfg.size_ref) == "long_side" else max(h, w)
        scale = float(size) / float(ref)
        nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
        if str(self.cfg.round_mode) == "floor":
            nh = max((nh // patch) * patch, patch)
            nw = max((nw // patch) * patch, patch)
        else:
            nh = max(int(round(nh / patch) * patch), patch)
            nw = max(int(round(nw / patch) * patch), patch)
        if x.shape[-2:] != (nh, nw): x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _norm(images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def _extract_patch_tokens(self, feats: dict) -> torch.Tensor:
        return feats["x_prenorm"][:, 1:] if "x_prenorm" in feats else feats["x_norm_patchtokens"]

    def _extract_intermediate_layers(self, images: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        if not self.cfg.layer_indices: raise ValueError("DINO expert_feature_query requires teacher.layer_indices.")
        if self.backbone is None: raise ValueError("DINO backbone has not been loaded.")
        getter = getattr(self.backbone, "get_intermediate_layers", None)
        if getter is None: raise ValueError("Loaded DINO backbone does not expose get_intermediate_layers.")
        layer_indices = tuple(int(idx) for idx in self.cfg.layer_indices)
        signature = inspect.signature(getter)
        kwargs = {}
        if "n" in signature.parameters: kwargs["n"] = layer_indices
        else: kwargs["layers"] = layer_indices
        if "reshape" in signature.parameters: kwargs["reshape"] = False
        if "return_class_token" in signature.parameters: kwargs["return_class_token"] = False
        if "norm" in signature.parameters: kwargs["norm"] = True
        outputs = getter(images, **kwargs)
        if not isinstance(outputs, (tuple, list)) or not outputs: raise ValueError("DINO intermediate layer extraction returned no layers.")
        tokens = []
        for output in outputs:
            layer = output[0] if isinstance(output, tuple) else output
            if layer.ndim != 3: raise ValueError(f"DINO intermediate layer must be [B,N,D], got {tuple(layer.shape)}.")
            if int(layer.shape[1]) == patch_h * patch_w + 1: layer = layer[:, 1:]
            tokens.append(layer)
        return torch.stack(tokens, dim=1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> TeacherTarget:
        self._load(images.device)
        prepped = self._prep(images)
        feats = self.model(self._norm(prepped))
        tokens = self._extract_patch_tokens(feats)
        patch = max(int(self.cfg.patch_multiple), 1)
        gh, gw = prepped.shape[-2] // patch, prepped.shape[-1] // patch
        if self.cfg.target_kind == "expert_feature_query":
            patch_feats = self._extract_intermediate_layers(self._norm(prepped), gh, gw)
            return TeacherTarget(name=self.cfg.name, tensor=tokens.detach(), kind=self.cfg.target_kind, loss_type=self.cfg.loss_type, weight=self.cfg.weight, aux={"grid_hw": (gh, gw), "prepped_hw": (int(prepped.shape[-2]), int(prepped.shape[-1])), "expert_feature_layout": "patch", "expert_features": patch_feats.detach(), "patch_feats": patch_feats.detach(), "patch_hw": (gh, gw), "expert_spatial_hw": (gh, gw), "layer_indices": tuple(int(idx) for idx in self.cfg.layer_indices or ())})
        return TeacherTarget(name=self.cfg.name, tensor=tokens.detach(), kind=self.cfg.target_kind, loss_type=self.cfg.loss_type, weight=self.cfg.weight, aux={"grid_hw": (gh, gw), "prepped_hw": (int(prepped.shape[-2]), int(prepped.shape[-1]))})
