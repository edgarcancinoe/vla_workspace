import json
from pathlib import Path
import sys
import tempfile

import torch
from torch.optim.lr_scheduler import LambdaLR
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.common.hf_hub import FAILED_HUB_UPLOAD_FILENAME, HubUploadConfig, HubUploadResult, push_folder_to_hub
from thesis_vla.training.visual_thought_trainer import JointTrainingState, VisualThoughtTrainConfig, _hub_step_dir, _push_checkpoint_to_hub, build_optimizer, build_policy_scheduler, compute_expert_loss, compute_expert_losses, compute_xvla_action_loss_from_encoder, load_decoder_init_if_present, optimizer_metrics, set_policy_trainability, trainer_state_dict
from thesis_vla.visual_thought import CeDirNetDistillationModel, DinoFeatureAlignmentModel, DinoTokenSequenceModel, TeacherTarget, load_cedirnet_decoder_config, load_dino_decoder_config
from thesis_vla.visual_thought.checkpoints import DECODER_STATE_FILENAME, save_decoder_state, save_visual_thought_checkpoint


class _FakeActionSpace:
    def preprocess(self, proprio, action_noisy):
        return proprio, action_noisy

    def compute_loss(self, pred_action, action):
        return {"mse": torch.mean((pred_action - action) ** 2)}


class _FakeModel:
    def __init__(self):
        self.action_space = _FakeActionSpace()
        self.transformer = lambda domain_id, action_with_noise, t, proprio, **enc: action_with_noise

    def _get_target_dtype(self):
        return torch.float32

    def _build_corrupted_action(self, action, device, target_dtype, t=None, action_noise=None):
        batch_size = action.shape[0]
        if t is None: t = torch.zeros(batch_size, device=device, dtype=target_dtype)
        if action_noise is None: action_noise = torch.zeros_like(action)
        return t, action_noise * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)


class _FakePolicy:
    def __init__(self):
        self.model = _FakeModel()

    def _prepare_action_targets(self, batch):
        return batch["action"]


class _TrainablePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeModel()
        self.linear = torch.nn.Linear(4, 4)


class _FakeOptimizerPreset:
    def build(self, params):
        assert isinstance(params, dict)
        return torch.optim.AdamW([
            {"params": [params["vlm.weight"]], "lr": 1e-4, "name": "vlm"},
            {"params": [params["transformer.weight"]], "lr": 1e-3, "name": "transformer_core"},
        ])


class _FakeSchedulerPreset:
    def build(self, optimizer, steps):
        return LambdaLR(optimizer, lambda current_step: 1.0)


class _JointPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Linear(2, 2, bias=False)
        self.transformer = torch.nn.Linear(2, 2, bias=False)
        self.config = type("Cfg", (), {"get_optimizer_preset": lambda _self: _FakeOptimizerPreset(), "get_scheduler_preset": lambda _self: _FakeSchedulerPreset()})()

    def get_optim_params(self):
        return {"vlm.weight": self.vlm.weight, "transformer.weight": self.transformer.weight}


def test_cedirnet_distill_step_smoke():
    cfg = load_cedirnet_decoder_config()
    decoder = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
    trainer_cfg = VisualThoughtTrainConfig(name="demo", training_stage="distill_only", expert_type="cedirnet", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu")
    target = TeacherTarget(name="cedirnet", tensor=torch.randn(2, cfg.teacher.out_channels, 64, 64), kind="dense_map", loss_type="mse", weight=1.0)
    loss, stats = compute_expert_loss(trainer_cfg, decoder, cfg, target, torch.randn(2, 256, 64), step=1)
    assert loss.ndim == 0
    assert "expert_total" in stats


def test_joint_expert_loss_propagates_gradient_to_vlm_features():
    cfg = load_cedirnet_decoder_config()
    decoder = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
    target = TeacherTarget(name="cedirnet", tensor=torch.randn(2, cfg.teacher.out_channels, 64, 64), kind="dense_map", loss_type="mse", weight=1.0)

    base_kwargs = dict(name="demo", expert_type="cedirnet", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu")

    joint_cfg = VisualThoughtTrainConfig(training_stage="joint_multitask", **base_kwargs)
    vlm_features = torch.randn(2, 256, 64, requires_grad=True)
    loss, _ = compute_expert_loss(joint_cfg, decoder, cfg, target, vlm_features, step=1)
    loss.backward()
    assert vlm_features.grad is not None, "joint_multitask expert loss must shape the VLM backbone"
    assert torch.any(vlm_features.grad != 0)

    distill_cfg = VisualThoughtTrainConfig(training_stage="distill_only", **base_kwargs)
    vlm_features_frozen = torch.randn(2, 256, 64, requires_grad=True)
    loss_frozen, _ = compute_expert_loss(distill_cfg, decoder, cfg, target, vlm_features_frozen, step=1)
    loss_frozen.backward()
    assert vlm_features_frozen.grad is None, "distill_only must detach the VLM backbone"


def test_dino_alignment_step_smoke():
    cfg = load_dino_decoder_config(Path(__file__).resolve().parents[1] / "config" / "visual_thought" / "dino_stack.yaml", Path(__file__).resolve().parents[1] / "config" / "visual_thought" / "dino_expert_query.yaml")
    target = TeacherTarget(name="dinov2", tensor=torch.randn(2, 64, 768), kind="expert_feature_query", loss_type="mse", weight=1.0, aux={"expert_feature_layout": "patch", "expert_features": torch.randn(2, 4, 64, 768), "patch_hw": (8, 8), "expert_spatial_hw": (8, 8)})
    decoder = DinoFeatureAlignmentModel.from_config(student_vlm_dim=64, target=target, cfg=cfg)
    trainer_cfg = VisualThoughtTrainConfig(name="demo", training_stage="distill_only", expert_type="dino", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", align_feature_until_step=10, steps=10)
    loss, stats = compute_expert_loss(trainer_cfg, decoder, cfg, target, torch.randn(2, 256, 64), step=1)
    assert loss.ndim == 0
    assert stats["expert_stage"] == 1.0


def test_dino_token_sequence_default_step_smoke():
    cfg = load_dino_decoder_config()
    target = TeacherTarget(name="dinov2", tensor=torch.randn(2, 64, 768), kind="token_sequence", loss_type="mse", weight=1.0)
    decoder = DinoTokenSequenceModel.from_config(student_vlm_dim=64, target=target, cfg=cfg)
    trainer_cfg = VisualThoughtTrainConfig(name="demo", training_stage="distill_only", expert_type="dino", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu")
    loss, stats = compute_expert_loss(trainer_cfg, decoder, cfg, target, torch.randn(2, 256, 64), step=1)
    assert loss.ndim == 0
    assert stats["expert_stage"] == 0.0


def test_combined_cedirnet_dino_expert_losses_smoke():
    cedirnet_cfg = load_cedirnet_decoder_config()
    dino_cfg = load_dino_decoder_config()
    trainer_cfg = VisualThoughtTrainConfig(name="demo", training_stage="joint_multitask", expert_type="cedirnet", expert_types=("cedirnet", "dino"), xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", cedirnet_decoder_init_path="c", cedirnet_decoder_stack_config_path="cs", cedirnet_decoder_task_config_path="ct", dino_decoder_init_path="d", dino_decoder_stack_config_path="ds", dino_decoder_task_config_path="dt", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", cedirnet_expert_loss_weight=0.5, dino_expert_loss_weight=0.25)
    cedirnet_decoder = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cedirnet_cfg)
    cedirnet_target = TeacherTarget(name="cedirnet", tensor=torch.randn(2, cedirnet_cfg.teacher.out_channels, 64, 64), kind="dense_map", loss_type="mse", weight=1.0)
    dino_target = TeacherTarget(name="dinov2", tensor=torch.randn(2, 64, 768), kind="token_sequence", loss_type="mse", weight=1.0)
    dino_decoder = DinoTokenSequenceModel.from_config(student_vlm_dim=64, target=dino_target, cfg=dino_cfg)
    loss, stats = compute_expert_losses(trainer_cfg, {"cedirnet": cedirnet_decoder, "dino": dino_decoder}, {"cedirnet": cedirnet_cfg, "dino": dino_cfg}, {"cedirnet": cedirnet_target, "dino": dino_target}, torch.randn(2, 256, 64), step=1)
    assert loss.ndim == 0
    assert "cedirnet_expert_total" in stats
    assert "dino_expert_total" in stats
    assert "expert_total_combined" in stats


def test_joint_step_reports_action_and_expert_losses():
    policy = _FakePolicy()
    processed_batch = {"action": torch.randn(2, 4, 6)}
    inputs = {"input_ids": torch.zeros(2, 3, dtype=torch.long), "proprio": torch.randn(2, 5), "domain_id": torch.zeros(2, dtype=torch.long)}
    enc = {"vlm_features": torch.randn(2, 256, 64), "aux_visual_inputs": torch.randn(2, 10, 64)}
    action_loss, action_stats = compute_xvla_action_loss_from_encoder(policy, processed_batch, inputs, enc)
    assert action_loss.ndim == 0
    assert "action_total" in action_stats


def test_set_policy_trainability_matches_stage_contract():
    policy = _TrainablePolicy()
    set_policy_trainability(policy, "distill_only")
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    set_policy_trainability(policy, "joint_multitask")
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_joint_training_uses_split_policy_and_decoder_optimizers():
    policy = _JointPolicy()
    decoder = torch.nn.Linear(2, 2, bias=False)
    cfg = VisualThoughtTrainConfig(name="demo", training_stage="joint_multitask", expert_type="cedirnet", xvla_init_path="x", decoder_init_path=None, decoder_stack_config_path="a", decoder_task_config_path="b", dataset_repo_id="repo", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", steps=12, decoder_optimizer_lr=2e-4, weight_decay=0.01)
    optimizers = build_optimizer(cfg, policy, decoder)
    assert isinstance(optimizers, JointTrainingState)
    assert isinstance(optimizers.policy_optimizer, torch.optim.AdamW)
    assert isinstance(optimizers.decoder_optimizer, torch.optim.AdamW)
    policy_param_ids = {id(parameter) for group in optimizers.policy_optimizer.param_groups for parameter in group["params"]}
    decoder_param_ids = {id(parameter) for group in optimizers.decoder_optimizer.param_groups for parameter in group["params"]}
    assert id(policy.vlm.weight) in policy_param_ids
    assert id(policy.transformer.weight) in policy_param_ids
    assert id(decoder.weight) in decoder_param_ids
    assert id(decoder.weight) not in policy_param_ids
    scheduler = build_policy_scheduler(cfg, policy, optimizers.policy_optimizer)
    assert isinstance(scheduler, LambdaLR)
    optimizers.policy_scheduler = scheduler
    state = trainer_state_dict(optimizers)
    assert state["format"] == "joint_policy_decoder_v1"
    assert "policy" in state["optimizers"]
    assert "decoder" in state["optimizers"]
    assert "policy" in state["schedulers"]
    metrics = optimizer_metrics(optimizers)
    assert metrics["policy_lr"] == 1e-4
    assert metrics["decoder_lr"] == 2e-4
    assert metrics["policy_lr_vlm"] == 1e-4
    assert metrics["policy_lr_transformer_core"] == 1e-3


def test_decoder_checkpoint_roundtrip():
    cfg = load_cedirnet_decoder_config()
    model = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_root = Path(tmpdir)
        save_decoder_state(model, checkpoint_root / DECODER_STATE_FILENAME)
        twin = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
        load_decoder_init_if_present(twin, str(checkpoint_root))
        for key, value in model.state_dict().items():
            assert torch.equal(value, twin.state_dict()[key])


class _FakeProcessor:
    def __init__(self, filename: str):
        self.filename = filename

    def save_pretrained(self, save_dir):
        Path(save_dir, self.filename).write_text("{}")


class _FakeSavePolicy:
    def save_pretrained(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(save_dir, "config.json").write_text("{}")
        Path(save_dir, "model.safetensors").write_bytes(b"fake")


def test_visual_thought_checkpoint_saves_processors():
    cfg = load_cedirnet_decoder_config()
    model = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "ckpt"
        save_visual_thought_checkpoint(checkpoint_dir=root, policy=_FakeSavePolicy(), decoder=model, trainer_state={"step": 1}, metadata={"name": "demo"}, config_snapshot={"device": "cpu"}, preprocessor=_FakeProcessor("policy_preprocessor.json"), postprocessor=_FakeProcessor("policy_postprocessor.json"))
        assert (root / "policy" / "policy_preprocessor.json").is_file()
        assert (root / "policy" / "policy_postprocessor.json").is_file()


def test_visual_thought_checkpoint_saves_multi_decoder_states():
    cedirnet_cfg = load_cedirnet_decoder_config()
    dino_cfg = load_dino_decoder_config()
    cedirnet_model = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cedirnet_cfg)
    dino_target = TeacherTarget(name="dinov2", tensor=torch.randn(2, 64, 768), kind="token_sequence", loss_type="mse", weight=1.0)
    dino_model = DinoTokenSequenceModel.from_config(student_vlm_dim=64, target=dino_target, cfg=dino_cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "ckpt"
        save_visual_thought_checkpoint(checkpoint_dir=root, policy=_FakeSavePolicy(), decoder={"cedirnet": cedirnet_model, "dino": dino_model}, trainer_state={"step": 1}, metadata={"name": "demo", "expert_types": ["cedirnet", "dino"]}, config_snapshot={"device": "cpu"})
        assert (root / "decoder_cedirnet.safetensors").is_file()
        assert (root / "decoder_dino.safetensors").is_file()


def test_push_folder_to_hub_retries_until_success(monkeypatch):
    calls, sleeps, logs, upload_kwargs = {"create": 0, "upload": 0}, [], [], {}

    class _FakeApi:
        def create_repo(self, **kwargs):
            calls["create"] += 1

        def upload_folder(self, **kwargs):
            calls["upload"] += 1
            upload_kwargs.update(kwargs)
            if calls["upload"] < 3: raise ConnectionError("network is unreachable")

    monkeypatch.setattr("thesis_vla.common.hf_hub.HfApi", _FakeApi)
    monkeypatch.setattr("thesis_vla.common.hf_hub.time.sleep", lambda seconds: sleeps.append(seconds))
    result = push_folder_to_hub(folder_path="/tmp/checkpoint", repo_id="user/repo", repo_type="model", path_in_repo="step_0000123", commit_message="push", upload_config=HubUploadConfig(max_retries=5, retry_backoff_s=2.0), logger=logs.append)
    assert result.ok is True
    assert result.attempts == 3
    assert calls == {"create": 3, "upload": 3}
    assert sleeps == [2.0, 4.0]
    assert upload_kwargs["path_in_repo"] == "step_0000123"
    assert any("Upload attempt 1/5" in line for line in logs)


def test_push_checkpoint_to_hub_writes_and_clears_failure_marker(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        marker_path = checkpoint_dir / FAILED_HUB_UPLOAD_FILENAME
        monkeypatch.setattr("thesis_vla.training.visual_thought_trainer.push_folder_to_hub", lambda **kwargs: HubUploadResult(ok=False, attempts=5, error="ConnectionError: network is unreachable"))
        assert _push_checkpoint_to_hub(checkpoint_dir, "user/repo", 123, "push") is False
        assert marker_path.is_file()
        payload = json.loads(marker_path.read_text())
        assert payload["repo_id"] == "user/repo"
        assert payload["path_in_repo"] == "step_0000123"
        assert payload["attempts"] == 5
        assert "ConnectionError" in payload["error"]
        monkeypatch.setattr("thesis_vla.training.visual_thought_trainer.push_folder_to_hub", lambda **kwargs: HubUploadResult(ok=True, attempts=1))
        assert _push_checkpoint_to_hub(checkpoint_dir, "user/repo", 123, "push") is True
        assert not marker_path.exists()


def test_hub_step_dir_uses_zero_padded_step_number():
    assert _hub_step_dir(7) == "step_0000007"
    assert _hub_step_dir(1234) == "step_0001234"
