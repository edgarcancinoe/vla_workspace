# ------------------------------------------------------------------------------
# Custom XVLA Pipeline Copy
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle, mat_to_rotate6d
from lerobot.processor import (
    ActionProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition, 
    transition_to_policy_action
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

def make_custom_xvla_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA (Custom Copy).
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    if getattr(config, 'use_delta', False):
        input_steps.append(XVLADeltaActionProcessorStep())

    input_steps.append(
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        )
    )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    if getattr(config, 'use_delta', False):
        output_steps.append(XVLAAbsoluteActionProcessorStep())

    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# --- Custom Processor Steps ---

@dataclass
@ProcessorStepRegistry.register(name="custom_xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition
        obs = obs.copy()
        keys_to_convert = self.image_keys or [k for k in obs if k.startswith(OBS_IMAGES)]
        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]
                max_val = tensor.max().item()
                if max_val <= 1.0:
                    obs[key] = tensor.float()
                    continue
                obs[key] = tensor.float() / 255.0
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features): return features
    def get_config(self) -> dict[str, Any]: return {"image_keys": self.image_keys, "validate_range": self.validate_range}

@dataclass
@ProcessorStepRegistry.register(name="custom_xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None: return new_transition
        obs = obs.copy()
        keys_to_normalize = self.image_keys or [k for k in obs if k.startswith(OBS_IMAGES)]
        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)
                while mean.dim() < tensor.dim():
                    mean, std = mean.unsqueeze(0), std.unsqueeze(0)
                obs[key] = (tensor - mean) / std
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features): return features
    def get_config(self) -> dict[str, Any]: return {"image_keys": self.image_keys}

@dataclass
@ProcessorStepRegistry.register(name="custom_xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    domain_id: int = 0
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        comp = comp.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features): return features
    def get_config(self) -> dict[str, Any]: return {"domain_id": self.domain_id}

@dataclass
@ProcessorStepRegistry.register(name="custom_xvla_delta_action")
class XVLADeltaActionProcessorStep(ActionProcessorStep):
    state_key: str = OBS_STATE
    delta_indices: list[int] | None = None
    def action(self, action: PolicyAction | torch.Tensor) -> PolicyAction | torch.Tensor:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if obs is None or self.state_key not in obs: return action
        proprio = obs[self.state_key]
        if proprio.ndim == 2: proprio = proprio.unsqueeze(1)
        proprio = proprio.to(action.device)
        if proprio.shape[-1] < action.shape[-1]:
            padding = torch.zeros((*proprio.shape[:-1], action.shape[-1] - proprio.shape[-1]), device=proprio.device, dtype=proprio.dtype)
            proprio = torch.cat([proprio, padding], dim=-1)
        elif proprio.shape[-1] > action.shape[-1]:
            proprio = proprio[..., : action.shape[-1]]
        new_action = action.clone()
        if self.delta_indices is not None:
            new_action[..., self.delta_indices] = action[..., self.delta_indices] - proprio[..., self.delta_indices]
        else:
            new_action[..., :-1] = action[..., :-1] - proprio[..., :-1]
        return new_action
    def transform_features(self, features): return features

@dataclass
@ProcessorStepRegistry.register(name="custom_xvla_absolute_action")
class XVLAAbsoluteActionProcessorStep(ActionProcessorStep):
    state_key: str = OBS_STATE
    delta_indices: list[int] | None = None
    def action(self, action: PolicyAction | torch.Tensor) -> PolicyAction | torch.Tensor:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if obs is None or self.state_key not in obs: return action
        proprio = obs[self.state_key]
        if proprio.ndim == 2: proprio = proprio.unsqueeze(1)
        proprio = proprio.to(action.device)
        if proprio.shape[-1] < action.shape[-1]:
            padding = torch.zeros((*proprio.shape[:-1], action.shape[-1] - proprio.shape[-1]), device=proprio.device, dtype=proprio.dtype)
            proprio = torch.cat([proprio, padding], dim=-1)
        elif proprio.shape[-1] > action.shape[-1]:
            proprio = proprio[..., : action.shape[-1]]
        new_action = action.clone()
        if self.delta_indices is not None:
            new_action[..., self.delta_indices] = action[..., self.delta_indices] + proprio[..., self.delta_indices]
        else:
            new_action[..., :-1] = action[..., :-1] + proprio[..., :-1]
        return new_action
    def transform_features(self, features): return features
