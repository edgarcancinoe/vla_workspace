from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


DECODER_STATE_FILENAME = "decoder.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"
METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "visual_thought_config.json"
POLICY_DIRNAME = "policy"


def save_decoder_state(model: torch.nn.Module, path: str | Path) -> None:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, str(path))


def load_decoder_state(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def save_visual_thought_checkpoint(checkpoint_dir: str | Path, policy, decoder: torch.nn.Module, trainer_state: dict[str, Any], metadata: dict[str, Any], config_snapshot: dict[str, Any], preprocessor=None, postprocessor=None) -> Path:
    root = Path(checkpoint_dir)
    policy_dir = root / POLICY_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(policy_dir)
    if preprocessor is not None: preprocessor.save_pretrained(policy_dir)
    if postprocessor is not None: postprocessor.save_pretrained(policy_dir)
    save_decoder_state(decoder, root / DECODER_STATE_FILENAME)
    torch.save(trainer_state, root / TRAINER_STATE_FILENAME)
    (root / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (root / CONFIG_FILENAME).write_text(json.dumps(config_snapshot, indent=2, sort_keys=True))
    return root


def load_visual_thought_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(checkpoint_dir) / METADATA_FILENAME).read_text())


def load_visual_thought_config_snapshot(checkpoint_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(checkpoint_dir) / CONFIG_FILENAME).read_text())
