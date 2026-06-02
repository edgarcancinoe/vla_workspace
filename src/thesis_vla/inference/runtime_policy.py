from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.xvla.action_contract import get_so101_slice_spec

from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, sync_xvla_policy_config


@dataclass
class RuntimePolicyOverrides:
    chunk_size: int | None = None
    n_action_steps: int | None = None
    max_action_tokens: int | None = None
    num_xvla_obs_steps: int = 1
    binary_gripper_inference: bool = False


def load_runtime_policy(
    policy_type: str,
    pretrained_path: str,
    device: str,
    overrides: RuntimePolicyOverrides | None = None,
):
    overrides = overrides or RuntimePolicyOverrides()
    policy_cls = get_policy_class(policy_type)
    include_eef_state = False

    if policy_type == "xvla":
        config = PreTrainedConfig.from_pretrained(pretrained_path)
        config.device = device
        if overrides.chunk_size is not None:
            config.chunk_size = overrides.chunk_size
        if overrides.n_action_steps is not None:
            config.n_action_steps = overrides.n_action_steps
        config.n_obs_steps = overrides.num_xvla_obs_steps
        config.binary_gripper_inference = overrides.binary_gripper_inference
        policy = policy_cls.from_pretrained(pretrained_path, config=config, device=device)
        include_eef_state = getattr(policy.config, "action_mode", None) == "so101_ee6d"
        if getattr(policy.model, "chunk_size", None) != getattr(policy.config, "chunk_size", None):
            raise ValueError(
                f"MISMATCH: model.chunk_size={getattr(policy.model, 'chunk_size', None)} != "
                f"config.chunk_size={getattr(policy.config, 'chunk_size', None)}."
            )
    else:
        policy = policy_cls.from_pretrained(pretrained_path, device=device)
        if overrides.chunk_size is not None and hasattr(policy.config, "chunk_size"):
            policy.config.chunk_size = overrides.chunk_size
        if overrides.n_action_steps is not None and hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = overrides.n_action_steps
        if overrides.max_action_tokens is not None and hasattr(policy.config, "max_action_tokens"):
            policy.config.max_action_tokens = overrides.max_action_tokens

    policy.reset()
    return policy, include_eef_state


def resolve_action_slice_spec(policy) -> Any:
    return get_so101_slice_spec(getattr(policy.config, "action_mode", None))


def sync_xvla_policy_with_features(policy_or_config, dataset_features: dict[str, Any], rename_map: dict[str, str]) -> None:
    sync_xvla_policy_config(policy_or_config, SimpleNamespace(features=dataset_features), rename_map)


def build_runtime_policy_processors(
    policy,
    pretrained_path: str,
    device: str,
    rename_map: dict[str, str] | None = None,
    stats: dict | None = None,
    use_dataset_stats: bool = False,
):
    rename_map = rename_map or {}
    if getattr(policy.config, "type", None) == "xvla":
        return make_xvla_runtime_processors(
            policy=policy,
            pretrained_path=pretrained_path,
            device=device,
            rename_map=rename_map,
            dataset_stats=stats,
            use_dataset_stats=use_dataset_stats,
        )
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained_path,
        dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": rename_map}},
    )
