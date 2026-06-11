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


def test_resolve_visual_thought_joint_combined_requires_per_expert_paths():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", training_stage="joint_multitask")
    try:
        resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="joint-both", expert_types=("cedirnet", "dino")), timestamp="20260101_000000")
    except ValueError as exc:
        assert "Combined expert_types mode requires" in str(exc)
    else:
        raise AssertionError("Expected combined resolve_experiment to require per-expert decoder paths.")


def test_resolve_visual_thought_joint_combined_succeeds():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", training_stage="joint_multitask", cedirnet_decoder_init_path="c", cedirnet_decoder_stack_config_path="cs", cedirnet_decoder_task_config_path="ct", dino_decoder_init_path="d", dino_decoder_stack_config_path="ds", dino_decoder_task_config_path="dt")
    resolved = resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="joint-both", expert_types=("cedirnet", "dino")), timestamp="20260101_000000")
    assert resolved.expert_types == ("cedirnet", "dino")
    assert resolved.cedirnet_decoder_init_path == "c"
    assert resolved.dino_decoder_init_path == "d"
