from __future__ import annotations

from collections.abc import Iterable


XVLA_RUNTIME_RENAME_MAPS: tuple[dict[str, str], ...] = (
    {
        "observation.images.main": "observation.images.image",
        "observation.images.secondary": "observation.images.image2",
    },
    {
        "observation.images.wrist": "observation.images.image",
        "observation.images.top": "observation.images.image2",
    },
)


def resolve_xvla_rename_map(camera_keys: Iterable[str] | None) -> dict[str, str]:
    keys = set(camera_keys or [])
    for mapping in XVLA_RUNTIME_RENAME_MAPS:
        if set(mapping).issubset(keys):
            return dict(mapping)
    return {}


def _sync_xvla_action_contract_features(policy_or_config) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xvla.action_contract import get_so101_slice_spec

    config = getattr(policy_or_config, "config", policy_or_config)
    slice_spec = get_so101_slice_spec(getattr(config, "action_mode", None))
    if slice_spec is None:
        return
    if "observation.state" in config.input_features:
        config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(slice_spec.real_dim,))
    if "action" in config.output_features:
        config.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(slice_spec.real_dim,))


def sync_xvla_policy_config(policy_or_config, dataset_meta, rename_map: dict[str, str]) -> None:
    from lerobot.scripts.lerobot_train import (
        _assert_xvla_finetune_contract,
        _enforce_xvla_finetune_contract,
        _rebuild_xvla_visual_input_features,
    )

    config = getattr(policy_or_config, "config", policy_or_config)
    _enforce_xvla_finetune_contract(config)
    _rebuild_xvla_visual_input_features(config, dataset_meta, rename_map)
    _sync_xvla_action_contract_features(config)
    _assert_xvla_finetune_contract(config)


def make_xvla_runtime_processors(
    policy,
    pretrained_path: str,
    device: str,
    rename_map: dict[str, str],
    dataset_stats: dict | None = None,
    use_dataset_stats: bool = False,
    load_pretrained_processors: bool = True,
):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    from lerobot.scripts.lerobot_train import _patch_xvla_gripper_stats_for_overrides

    _sync_xvla_action_contract_features(policy)
    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename_map},
        "normalizer_processor": {
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }
    processor_stats = None

    if use_dataset_stats and dataset_stats is not None:
        processor_stats = _patch_xvla_gripper_stats_for_overrides(policy.config, dataset_stats)
        preprocessor_overrides["normalizer_processor"]["stats"] = processor_stats
        postprocessor_overrides["unnormalizer_processor"]["stats"] = processor_stats

    if not load_pretrained_processors:
        preprocessor, postprocessor = make_xvla_pre_post_processors(config=policy.config, dataset_stats=processor_stats if use_dataset_stats and dataset_stats is not None else None)
        for step in preprocessor.steps:
            if getattr(step.__class__, "_registry_name", "") == "rename_observations_processor": step.rename_map = dict(rename_map)
            if getattr(step.__class__, "_registry_name", "") == "device_processor": step.device = device
        for step in postprocessor.steps:
            if getattr(step.__class__, "_registry_name", "") == "device_processor": step.device = "cpu"
        return preprocessor, postprocessor

    return make_pre_post_processors(policy_cfg=policy.config, pretrained_path=pretrained_path, preprocessor_overrides=preprocessor_overrides, postprocessor_overrides=postprocessor_overrides)
