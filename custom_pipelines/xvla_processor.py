from __future__ import annotations

import torch

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_custom_xvla_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, object], dict[str, object]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Alias the local custom path to the canonical XVLA processor implementation."""
    return make_xvla_pre_post_processors(config=config, dataset_stats=dataset_stats)
