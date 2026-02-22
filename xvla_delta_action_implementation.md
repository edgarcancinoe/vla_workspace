# XVLA Delta Action Support Implementation

This document details the changes made to the `lerobot` repository to add optional support for delta action processing in the XVLA policy. These changes allow the model to learn to predict relative movement (deltas) instead of absolute poses.

## Files Modified

1. `src/lerobot/policies/xvla/configuration_xvla.py`
2. `src/lerobot/policies/xvla/processor_xvla.py`

---

## 1. Configuration Change
**File:** `src/lerobot/policies/xvla/configuration_xvla.py`

Added the `use_delta` flag to the `XVLAConfig` class.

```python
@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    # ... previous fields ...
    
    # Action & proprioception
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    use_delta: bool = False  # <--- Added this line
    max_state_dim: int = 32
    # ...
```

---

## 2. Processor Implementation
**File:** `src/lerobot/policies/xvla/processor_xvla.py`

### Imports
Added `ActionProcessorStep` to the imports from `lerobot.processor`.

### New Processor Steps
Implemented two new classes at the end of the file (before `make_xvla_libero_pre_post_processors`):

```python
@dataclass
@ProcessorStepRegistry.register(name="xvla_delta_action")
class XVLADeltaActionProcessorStep(ActionProcessorStep):
    """
    Computes delta actions relative to the current proprioception.
    action = action - proprio
    """
    state_key: str = OBS_STATE
    delta_indices: list[int] | None = None

    def action(self, action: PolicyAction | torch.Tensor) -> PolicyAction | torch.Tensor:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if obs is None or self.state_key not in obs:
            return action

        proprio = obs[self.state_key]
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1) # (B, 1, D)

        proprio = proprio.to(action.device)

        # Handle dimension mismatches (padding)
        if proprio.shape[-1] < action.shape[-1]:
            padding = torch.zeros((*proprio.shape[:-1], action.shape[-1] - proprio.shape[-1]), device=proprio.device, dtype=proprio.dtype)
            proprio = torch.cat([proprio, padding], dim=-1)
        elif proprio.shape[-1] > action.shape[-1]:
            proprio = proprio[..., : action.shape[-1]]

        new_action = action.clone()
        if self.delta_indices is not None:
            new_action[..., self.delta_indices] = action[..., self.delta_indices] - proprio[..., self.delta_indices]
        else:
            # Default: Apply delta to all but the last dimension (assuming last is gripper)
            new_action[..., :-1] = action[..., :-1] - proprio[..., :-1]
        return new_action

    def transform_features(self, features): return features

@dataclass
@ProcessorStepRegistry.register(name="xvla_absolute_action")
class XVLAAbsoluteActionProcessorStep(ActionProcessorStep):
    """
    Computes absolute actions from predicted deltas and current proprioception.
    action = predicted_delta + proprio
    """
    state_key: str = OBS_STATE
    delta_indices: list[int] | None = None

    def action(self, action: PolicyAction | torch.Tensor) -> PolicyAction | torch.Tensor:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if obs is None or self.state_key not in obs:
            return action

        proprio = obs[self.state_key]
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)

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
```

### Pipeline Integration
Updated `make_xvla_pre_post_processors` to inject these steps conditionally.

```python
def make_xvla_pre_post_processors(config: XVLAConfig, dataset_stats=None):
    features = {**config.input_features, **config.output_features}
    input_steps = [
        # ... standard steps ...
        DeviceProcessorStep(device=config.device),
    ]

    # Inject Delta Step for training
    if config.use_delta:
        input_steps.append(XVLADeltaActionProcessorStep())

    input_steps.append(
        NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping, stats=dataset_stats)
    )

    output_steps = [
        UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats),
    ]

    # Inject Absolute Step for inference
    if config.use_delta:
        output_steps.append(XVLAAbsoluteActionProcessorStep())

    output_steps.append(DeviceProcessorStep(device="cpu"))
    # ...
```

---

## 3. Usage

To enable delta processing during training, pass the following flag to the `lerobot_train.py` script:

```bash
python lerobot/scripts/lerobot_train.py \
    --policy.type=xvla \
    --policy.use_delta=True \
    ...
```

This will ensure the model is trained on deltas and correctly outputs absolute poses during evaluation.
