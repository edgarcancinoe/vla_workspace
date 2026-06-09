from __future__ import annotations

import inspect
import io
import json
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F

from thesis_vla.visual_thought.config import CeDirNetTeacherConfig
from thesis_vla.visual_thought.targets import TeacherTarget


def _convert_booleans(data):
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if isinstance(value, str) and value.lower() in {"false", "true", "yes", "no"}: data[key] = value.lower() in {"true", "yes"}
            else: _convert_booleans(value)
    elif isinstance(data, list):
        for value in data: _convert_booleans(value)
    return data


def _patch_smp_fpn_decoder_forward():
    try:
        from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
    except Exception:
        return
    if getattr(FPNDecoder.forward, "_cedirnet_varargs_compat", False): return
    original_forward = FPNDecoder.forward
    params = list(inspect.signature(original_forward).parameters.values())[1:]
    is_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)

    def forward_compat(self, *features):
        if is_varargs:
            if len(features) == 1 and isinstance(features[0], (list, tuple)): return original_forward(self, *features[0])
            return original_forward(self, *features)
        if len(features) == 1: return original_forward(self, features[0])
        return original_forward(self, list(features))

    forward_compat._cedirnet_varargs_compat = True
    FPNDecoder.forward = forward_compat


class CeDiRNetTeacher:
    def __init__(self, cfg: CeDirNetTeacherConfig):
        self.cfg = cfg
        self.args = None
        self.model = None
        self.center_model = None

    def _use_data_parallel(self) -> bool:
        return int(os.environ.get("WORLD_SIZE", "1")) <= 1 and torch.cuda.device_count() > 1

    def _external_log_context(self):
        return redirect_stdout(io.StringIO())

    def _repo_src(self) -> Path:
        if not self.cfg.repo_src: raise ValueError("CeDiRNet teacher requires repo_src.")
        return Path(self.cfg.repo_src).expanduser()

    def _config_path(self) -> Path:
        if not self.cfg.config_path: raise ValueError("CeDiRNet teacher requires config_path.")
        path = Path(self.cfg.config_path).expanduser()
        if path.is_dir(): raise ValueError(f"CeDiRNet teacher config_path must be a JSON file, not a directory: {path}")
        return path

    def _load_args(self) -> dict:
        if self.args is None:
            with self._config_path().open("r") as handle: self.args = _convert_booleans(json.load(handle))
            self.args["checkpoint_path"] = self.cfg.checkpoint
            if self.cfg.localization_checkpoint: self.args["center_checkpoint_path"] = self.cfg.localization_checkpoint
        return self.args

    def _target_channel_indices(self) -> list[int]:
        return [int(x) for x in self.cfg.target_channel_indices]

    def _num_vector_fields_for_model(self, args: dict) -> int:
        return int(args["num_vector_fields"])

    def _select_target_channels(self, maps: torch.Tensor) -> torch.Tensor:
        indices = self._target_channel_indices()
        if int(maps.shape[1]) == len(indices) and indices == list(range(len(indices))): return maps
        if max(indices) >= int(maps.shape[1]): raise ValueError(f"CeDiRNet target channel index {max(indices)} is out of range for output with {int(maps.shape[1])} channels.")
        return maps[:, indices, :, :]

    @staticmethod
    def _freeze_eval(model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad = False
        return model

    @staticmethod
    def _state_with_module_prefix(state_dict: dict, model: torch.nn.Module) -> dict:
        model_keys = model.state_dict().keys()
        if any(str(key).startswith("module.") for key in model_keys) and not any(str(key).startswith("module.") for key in state_dict): return {f"module.{key}": value for key, value in state_dict.items()}
        if not any(str(key).startswith("module.") for key in model_keys) and any(str(key).startswith("module.") for key in state_dict): return {str(key).removeprefix("module."): value for key, value in state_dict.items()}
        return state_dict

    @staticmethod
    def _adapt_cedirnet_state_shapes(state_dict: dict, model: torch.nn.Module, *, center: bool):
        unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
        center_key = "module.instance_center_estimator.conv_start.0.weight" if "module.instance_center_estimator.conv_start.0.weight" in state_dict else "instance_center_estimator.conv_start.0.weight"
        if center and center_key in state_dict:
            checkpoint_input_weights = state_dict[center_key]
            center_input_weights = unwrapped.instance_center_estimator.conv_start[0].weight
            if checkpoint_input_weights.shape != center_input_weights.shape: state_dict[center_key] = checkpoint_input_weights[:, :center_input_weights.shape[1], :, :]
        seg_key = "module.model.segmentation_head.2.weight" if "module.model.segmentation_head.2.weight" in state_dict else "model.segmentation_head.2.weight"
        bias_key = "module.model.segmentation_head.2.bias" if "module.model.segmentation_head.2.bias" in state_dict else "model.segmentation_head.2.bias"
        if not center and seg_key in state_dict:
            checkpoint_input_weights = state_dict[seg_key]
            checkpoint_input_bias = state_dict[bias_key]
            model_output_weights = unwrapped.model.segmentation_head[2].weight
            if checkpoint_input_weights.shape != model_output_weights.shape:
                state_dict[seg_key] = checkpoint_input_weights[:model_output_weights.shape[0], :, :, :]
                state_dict[bias_key] = checkpoint_input_bias[:model_output_weights.shape[0]]

    @staticmethod
    def _load_state_like_cedirnet_infer(model: torch.nn.Module, state: dict, key: str, *, strict: bool, center: bool):
        if key not in state: return False
        state_dict = CeDiRNetTeacher._state_with_module_prefix(state[key], model)
        CeDiRNetTeacher._adapt_cedirnet_state_shapes(state_dict, model, center=center)
        model.load_state_dict(state_dict, strict=strict)
        return True

    def _load(self, device: torch.device):
        if self.model is None:
            repo_src = self._repo_src()
            if str(repo_src) not in sys.path: sys.path.insert(0, str(repo_src))
            _patch_smp_fpn_decoder_forward()
            from models import get_model

            args = self._load_args()
            with self._external_log_context():
                model = get_model(args["model"]["name"], args["model"]["kwargs"])
                model.init_output(self._num_vector_fields_for_model(args))
            if self._use_data_parallel(): model = torch.nn.DataParallel(model)
            model = model.to(device)
            if not args.get("checkpoint_path"): raise ValueError("CeDiRNet teacher requires checkpoint.")
            state = torch.load(args["checkpoint_path"], map_location=device, weights_only=False)
            if not isinstance(state, dict) or not self._load_state_like_cedirnet_infer(model, state, "model_state_dict", strict=True, center=False): model.load_state_dict(self._state_with_module_prefix(state, model), strict=True)
            self.model = self._freeze_eval(model)
        elif next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

    def _load_center_model(self, device: torch.device):
        if self.center_model is None:
            repo_src = self._repo_src()
            if str(repo_src) not in sys.path: sys.path.insert(0, str(repo_src))
            _patch_smp_fpn_decoder_forward()
            from models import get_center_model

            args = self._load_args()
            center_cfg = args.get("center_model")
            if not center_cfg: raise ValueError("CeDiRNet config is missing center_model settings.")
            with self._external_log_context():
                center_model = get_center_model(center_cfg["name"], center_cfg["kwargs"], is_learnable=True)
                center_model.init_output(self._num_vector_fields_for_model(args))
            if self._use_data_parallel(): center_model = torch.nn.DataParallel(center_model)
            center_model = center_model.to(device)
            center_model_loaded = False
            if args.get("checkpoint_path"):
                state = torch.load(args["checkpoint_path"], map_location=device, weights_only=False)
                if not args.get("center_checkpoint_path") and isinstance(state, dict): center_model_loaded = self._load_state_like_cedirnet_infer(center_model, state, "center_model_state_dict", strict=False, center=True)
            if args.get("center_checkpoint_path"):
                state = torch.load(args["center_checkpoint_path"], map_location=device, weights_only=False)
                if isinstance(state, dict): center_model_loaded = self._load_state_like_cedirnet_infer(center_model, state, "center_model_state_dict", strict=False, center=True)
            if not center_model_loaded: raise ValueError("Missing CeDiRNet center model checkpoint.")
            self.center_model = self._freeze_eval(center_model)
        elif next(self.center_model.parameters()).device != device:
            self.center_model = self.center_model.to(device)

    def _prep(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        x = x / 255.0 if x.max() > 1.0 else x
        x = x.clamp(0.0, 1.0)
        infer_size = (self._load_args() or {}).get("size")
        target_hw = (int(infer_size[1]), int(infer_size[0])) if isinstance(infer_size, (tuple, list)) and len(infer_size) == 2 else ((int(self.cfg.image_size), int(self.cfg.image_size)) if int(self.cfg.image_size) > 0 else None)
        if self.cfg.resize and target_hw and x.shape[-2:] != target_hw: x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x

    def _target_from_maps(self, maps: torch.Tensor, prepped: torch.Tensor) -> TeacherTarget:
        selected = self._select_target_channels(maps)
        return TeacherTarget(name=self.cfg.name, tensor=selected.detach(), kind=str(self.cfg.target_kind), loss_type=self.cfg.loss_type, weight=self.cfg.weight, aux={"prepped_hw": (int(prepped.shape[-2]), int(prepped.shape[-1])), "map_hw": (int(selected.shape[-2]), int(selected.shape[-1])), "num_channels": int(selected.shape[1]), "full_num_channels": int(maps.shape[1]), "localization_checkpoint": self.cfg.localization_checkpoint})

    def _target_from_maps_and_features(self, maps: torch.Tensor, features: torch.Tensor, prepped: torch.Tensor) -> TeacherTarget:
        target = self._target_from_maps(maps, prepped)
        if target.kind == "expert_feature_query": target.aux.update({"expert_feature_layout": "channel_map", "expert_features": features.detach(), "num_query_tokens": int(target.tensor.shape[1]), "expert_spatial_hw": (int(features.shape[-2]), int(features.shape[-1]))})
        return target

    @torch.no_grad()
    def _predict_full_maps(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._load(images.device)
        prepped = self._prep(images)
        maps = self.model(prepped)
        if isinstance(maps, (tuple, list)): maps = maps[0]
        if maps.ndim != 4: raise ValueError(f"CeDiRNet teacher expected dense maps shaped [B,C,H,W], got {tuple(maps.shape)}.")
        return prepped, maps

    @torch.no_grad()
    def _predict_full_maps_and_decoder_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._load(images.device)
        prepped = self._prep(images)
        captured = []
        core = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        decoder = getattr(getattr(core, "model", None), "decoder", None)
        if decoder is None: raise ValueError("CeDiRNet expert_feature_query requires a model with .model.decoder for decoder feature extraction.")
        handle = decoder.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
        try:
            maps = self.model(prepped)
        finally:
            handle.remove()
        if isinstance(maps, (tuple, list)): maps = maps[0]
        if maps.ndim != 4: raise ValueError(f"CeDiRNet teacher expected dense maps shaped [B,C,H,W], got {tuple(maps.shape)}.")
        if not captured or not isinstance(captured[-1], torch.Tensor): raise ValueError("CeDiRNet decoder feature hook did not capture a tensor.")
        return prepped, maps, captured[-1]

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> TeacherTarget:
        if str(self.cfg.target_kind) == "expert_feature_query":
            prepped, maps, features = self._predict_full_maps_and_decoder_features(images)
            return self._target_from_maps_and_features(maps, features, prepped)
        prepped, maps = self._predict_full_maps(images)
        return self._target_from_maps(maps, prepped)

    @torch.no_grad()
    def localize(self, maps: torch.Tensor) -> dict:
        self._load_center_model(maps.device)
        try:
            out = self.center_model(maps, detect_centers=True)
        except TypeError:
            out = self.center_model(maps)
        if isinstance(out, dict): return {key: (value.detach() if isinstance(value, torch.Tensor) else value) for key, value in out.items()}
        return {"localization": out.detach() if isinstance(out, torch.Tensor) else out}
