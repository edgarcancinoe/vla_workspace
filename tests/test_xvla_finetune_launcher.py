import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.training.xvla_finetune_launcher import ExperimentSpec, FreezeConfig, LaunchConfig, build_training_command, resolve_experiment


def test_resolve_experiment_keeps_joint_mode_by_default():
    defaults = LaunchConfig(hf_user="tester", dataset_name="dataset", base_model=("lerobot/xvla-base", "xvla-base"))
    resolved = resolve_experiment(Path.cwd(), defaults, ExperimentSpec(name="demo"), timestamp="20260101_000000")
    assert resolved.adaptation.mode == "joint"
    assert resolved.freeze == defaults.freeze
    assert "stagedpw" not in resolved.name
    cmd = build_training_command(resolved)
    assert "--policy.adaptation_mode=joint" in cmd


def test_resolve_experiment_normalizes_staged_mode_and_emits_cli():
    defaults = LaunchConfig(hf_user="tester", dataset_name="dataset", base_model=("lerobot/xvla-base", "xvla-base"))
    experiment = ExperimentSpec(
        name="staged-demo",
        freeze=FreezeConfig(freeze_vision_encoder=True, freeze_language_encoder=True, train_policy_transformer=False, train_soft_prompts=True),
        adaptation_mode="staged_prompt_warmup",
        freeze_steps=123,
        warmup_steps=456,
        learning_coef=0.25,
    )
    resolved = resolve_experiment(Path.cwd(), defaults, experiment, timestamp="20260101_000000")
    assert resolved.freeze == FreezeConfig(freeze_vision_encoder=False, freeze_language_encoder=False, train_policy_transformer=True, train_soft_prompts=True)
    assert resolved.adaptation.mode == "staged_prompt_warmup"
    assert resolved.adaptation.freeze_steps == 123
    assert resolved.adaptation.warmup_steps == 456
    assert resolved.adaptation.learning_coef == 0.25
    cmd = build_training_command(resolved)
    assert "--policy.adaptation_mode=staged_prompt_warmup" in cmd
    assert "--policy.freeze_steps=123" in cmd
    assert "--policy.warmup_steps=456" in cmd
    assert "--policy.learning_coef=0.25" in cmd
    assert "--policy.freeze_vision_encoder=false" in cmd
    assert "--policy.train_policy_transformer=true" in cmd
