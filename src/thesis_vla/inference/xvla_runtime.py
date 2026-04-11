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


def sync_xvla_policy_config(policy_or_config, dataset_meta, rename_map: dict[str, str]) -> None:
    from lerobot.scripts.lerobot_train import (
        _assert_xvla_finetune_contract,
        _enforce_xvla_finetune_contract,
        _rebuild_xvla_visual_input_features,
    )

    config = getattr(policy_or_config, "config", policy_or_config)
    _enforce_xvla_finetune_contract(config)
    _rebuild_xvla_visual_input_features(config, dataset_meta, rename_map)
    _assert_xvla_finetune_contract(config)


def make_xvla_runtime_processors(
    policy,
    pretrained_path: str,
    device: str,
    rename_map: dict[str, str],
    dataset_stats: dict | None = None,
    use_dataset_stats: bool = False,
):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _patch_xvla_gripper_stats_for_overrides

    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    postprocessor_overrides = {}

    if use_dataset_stats and dataset_stats is not None:
        processor_stats = _patch_xvla_gripper_stats_for_overrides(policy.config, dataset_stats)
        preprocessor_overrides["normalizer_processor"] = {
            "stats": processor_stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        }
        postprocessor_overrides["unnormalizer_processor"] = {
            "stats": processor_stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }

    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
