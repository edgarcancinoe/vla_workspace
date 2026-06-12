from pathlib import Path
import sys

from thesis_vla.training.visual_thought_launcher import VisualThoughtExperimentSpec, VisualThoughtLaunchConfig, VisualThoughtRuntimeConfig, build_training_command, resolve_experiment


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


def test_resolve_visual_thought_normalization_and_resume_fields():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml")
    resolved = resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="demo", normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}', resume=True, resume_checkpoint_path="/tmp/checkpoint_0000007"), timestamp="20260101_000000")
    assert resolved.normalization_mapping == '{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
    assert resolved.resume is True
    assert resolved.resume_checkpoint_path == "/tmp/checkpoint_0000007"


def test_visual_thought_build_training_command_uses_accelerate():
    defaults = VisualThoughtLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", runtime=VisualThoughtRuntimeConfig(launch_mode="accelerate", cuda_devices=(0, 1), mixed_precision="bf16"))
    resolved = resolve_experiment(Path.cwd(), defaults, VisualThoughtExperimentSpec(name="demo-accel"), timestamp="20260101_000000")
    cmd = build_training_command(resolved, Path("/tmp/demo.json"))
    assert cmd[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert "--num_processes=2" in cmd
    assert "--mixed_precision=bf16" in cmd
    assert "thesis_vla.training.visual_thought_trainer" in cmd
