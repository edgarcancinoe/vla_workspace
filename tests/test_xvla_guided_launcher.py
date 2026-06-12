from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.training.xvla_guided_launcher import GuidedExperimentSpec, GuidedLaunchConfig, resolve_experiment


def test_guided_launcher_resolves_stage_defaults(tmp_path):
    defaults = GuidedLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_init_path="/tmp/decoder", guided_stage_config_path=str(Path(__file__).resolve().parents[1] / "config" / "visual_thought" / "cedirnet_guided_policy.yaml"))
    resolved = resolve_experiment(tmp_path, defaults, GuidedExperimentSpec())
    assert resolved.dataset_repo_id == "tester/dataset"
    assert resolved.action_mode is None
    assert resolved.fusion_mode == "concat"
    assert resolved.guidance_train_mode == "frozen"
    assert resolved.guidance_unfreeze_step == 1000
    assert resolved.freeze_xvla_vlm is True
    assert resolved.decoder_init_path == "/tmp/decoder"
    assert resolved.wandb_enable is True
    assert resolved.validation_enable is True
    assert resolved.validation_split_ratio == 0.1
    assert resolved.xvla_scheduler_decay_lr == 2.5e-6


def test_guided_launcher_maps_legacy_gated_fusion(tmp_path, tmp_path_factory):
    stage_cfg = tmp_path / "guided_stage.yaml"
    stage_cfg.write_text("fusion_mode: cross_attn\ngated_fusion: true\n")
    defaults = GuidedLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_init_path="/tmp/decoder", guided_stage_config_path=str(stage_cfg))
    resolved = resolve_experiment(tmp_path_factory.mktemp("workspace"), defaults, GuidedExperimentSpec())
    assert resolved.fusion_mode == "gated_cross_attention"


def test_guided_launcher_allows_wandb_override(tmp_path):
    defaults = GuidedLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_init_path="/tmp/decoder", wandb_enable=True, wandb_project="default-proj")
    resolved = resolve_experiment(tmp_path, defaults, GuidedExperimentSpec(wandb_enable=False, wandb_project="exp-proj", wandb_run_name="exp-run"))
    assert resolved.wandb_enable is False
    assert resolved.wandb_project == "exp-proj"
    assert resolved.wandb_run_name == "exp-run"


def test_guided_launcher_allows_action_mode_and_scheduler_override(tmp_path):
    defaults = GuidedLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_init_path="/tmp/decoder", action_mode="so101_ee6d", xvla_scheduler_decay_lr=2.5e-6)
    resolved = resolve_experiment(tmp_path, defaults, GuidedExperimentSpec(action_mode="so101_joint", xvla_scheduler_decay_lr=1e-6))
    assert resolved.action_mode == "so101_joint"
    assert resolved.xvla_scheduler_decay_lr == 1e-6
