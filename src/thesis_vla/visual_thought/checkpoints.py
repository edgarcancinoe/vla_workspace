from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


DECODER_STATE_FILENAME = "decoder.safetensors"
DECODER_STATE_TEMPLATE = "decoder_{expert}.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"
METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "visual_thought_config.json"
POLICY_DIRNAME = "policy"
DECODER_STACK_CONFIG_FILENAME = "decoder_stack_config.yaml"
DECODER_TASK_CONFIG_FILENAME = "decoder_task_config.yaml"
DECODER_STACK_CONFIG_TEMPLATE = "decoder_stack_config_{expert}.yaml"
DECODER_TASK_CONFIG_TEMPLATE = "decoder_task_config_{expert}.yaml"


def save_decoder_state(model: torch.nn.Module, path: str | Path) -> None:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, str(path))


def load_decoder_state(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def _write_config_copy(path: Path, source_path: str | Path) -> None:
    path.write_text(Path(source_path).read_text())


def _save_decoder_payload(root: Path, decoder: torch.nn.Module | dict[str, torch.nn.Module]) -> None:
    if isinstance(decoder, dict):
        for expert, module in decoder.items(): save_decoder_state(module, root / DECODER_STATE_TEMPLATE.format(expert=str(expert)))
        return
    save_decoder_state(decoder, root / DECODER_STATE_FILENAME)


def _save_decoder_config_payload(root: Path, decoder_stack_config_path: str | Path | dict[str, str | Path] | None = None, decoder_task_config_path: str | Path | dict[str, str | Path] | None = None) -> None:
    if isinstance(decoder_stack_config_path, dict):
        for expert, source in decoder_stack_config_path.items(): _write_config_copy(root / DECODER_STACK_CONFIG_TEMPLATE.format(expert=str(expert)), source)
    elif decoder_stack_config_path is not None:
        _write_config_copy(root / DECODER_STACK_CONFIG_FILENAME, decoder_stack_config_path)
    if isinstance(decoder_task_config_path, dict):
        for expert, source in decoder_task_config_path.items(): _write_config_copy(root / DECODER_TASK_CONFIG_TEMPLATE.format(expert=str(expert)), source)
    elif decoder_task_config_path is not None:
        _write_config_copy(root / DECODER_TASK_CONFIG_FILENAME, decoder_task_config_path)


def save_visual_thought_checkpoint(checkpoint_dir: str | Path, policy, decoder: torch.nn.Module | dict[str, torch.nn.Module], trainer_state: dict[str, Any], metadata: dict[str, Any], config_snapshot: dict[str, Any], preprocessor=None, postprocessor=None, decoder_stack_config_path: str | Path | dict[str, str | Path] | None = None, decoder_task_config_path: str | Path | dict[str, str | Path] | None = None) -> Path:
    root = Path(checkpoint_dir)
    policy_dir = root / POLICY_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(policy_dir)
    if preprocessor is not None: preprocessor.save_pretrained(policy_dir)
    if postprocessor is not None: postprocessor.save_pretrained(policy_dir)
    _save_decoder_payload(root, decoder)
    torch.save(trainer_state, root / TRAINER_STATE_FILENAME)
    (root / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (root / CONFIG_FILENAME).write_text(json.dumps(config_snapshot, indent=2, sort_keys=True))
    _save_decoder_config_payload(root, decoder_stack_config_path=decoder_stack_config_path, decoder_task_config_path=decoder_task_config_path)
    return root


def load_visual_thought_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(checkpoint_dir) / METADATA_FILENAME).read_text())


def load_visual_thought_config_snapshot(checkpoint_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(checkpoint_dir) / CONFIG_FILENAME).read_text())
