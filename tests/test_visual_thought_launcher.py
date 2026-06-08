from pathlib import Path

from thesis_vla.training.visual_thought_launcher import VisualThoughtExperimentSpec, VisualThoughtLaunchConfig, resolve_experiment


def test_resolve_visual_thought_distill_only_does_not_require_decoder_init():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", training_stage="distill_only")
    resolved = resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="demo"), timestamp="20260101_000000")
    assert resolved.training_stage == "distill_only"
    assert resolved.decoder_init_path is None
    assert resolved.xvla_init_path == "lerobot/xvla-base"


def test_resolve_visual_thought_joint_requires_decoder_init():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", training_stage="joint_multitask")
    try:
        resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="joint"), timestamp="20260101_000000")
    except ValueError as exc:
        assert "decoder_init_path is required" in str(exc)
    else:
        raise AssertionError("Expected joint_multitask resolve_experiment to require decoder_init_path.")
