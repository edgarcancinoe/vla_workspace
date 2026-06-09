from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.training.xvla_guided_launcher import GuidedExperimentSpec, GuidedLaunchConfig, resolve_experiment


def test_guided_launcher_resolves_stage_defaults(tmp_path):
    defaults = GuidedLaunchConfig(hf_user="tester", dataset_name="dataset", xvla_init_path="lerobot/xvla-base", decoder_init_path="/tmp/decoder", guided_stage_config_path=str(Path(__file__).resolve().parents[1] / "config" / "visual_thought" / "cedirnet_guided_policy.yaml"))
    resolved = resolve_experiment(tmp_path, defaults, GuidedExperimentSpec())
    assert resolved.dataset_repo_id == "tester/dataset"
    assert resolved.fusion_mode == "concat"
    assert resolved.gated_fusion is False
    assert resolved.guidance_train_mode == "warmup_freeze"
    assert resolved.guidance_unfreeze_step == 1000
    assert resolved.decoder_init_path == "/tmp/decoder"
