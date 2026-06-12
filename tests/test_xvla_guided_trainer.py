from pathlib import Path
import sys

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.policies.xvla.action_contract import build_slice_map, get_so101_slice_spec
from lerobot.policies.xvla.modeling_xvla import pad_tensor_along_dim, pad_vector
from lerobot.processor.slice_processor import SliceProcessorStep
from thesis_vla.training.xvla_guided_trainer import GuidedXVLATrainConfig, _configure_explicit_stage_trainability, _episode_split_indices, _maybe_init_wandb, build_xvla_runtime, compute_guidance_loss, compute_guided_action_loss_from_encoder
from thesis_vla.visual_thought.targets import TeacherTarget


class _FakeActionSpace:
    def preprocess(self, proprio, action_noisy):
        return proprio, action_noisy

    def compute_loss(self, pred_action, action):
        return {"mse": torch.mean((pred_action - action) ** 2)}


class _FakeGuidanceDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(6, 3)

    def decoder_tokens(self, vlm_features):
        return vlm_features[:, :4, :6]

    def predict_from_tokens(self, guidance_tokens, target_map=None, output_size=None):
        b = guidance_tokens.shape[0]
        return self.proj(guidance_tokens).transpose(1, 2).reshape(b, 3, 2, 2)


class _FakeGuidedModel:
    def __init__(self):
        self.action_space = _FakeActionSpace()
        self.guidance_decoder = _FakeGuidanceDecoder()
        self.transformer = lambda domain_id, action_with_noise, t, proprio, guidance_tokens, **enc: action_with_noise + 0.0 * guidance_tokens.mean()

    def _get_target_dtype(self):
        return torch.float32

    def _build_corrupted_action(self, action, device, target_dtype, t=None, action_noise=None):
        batch_size = action.shape[0]
        if t is None: t = torch.zeros(batch_size, device=device, dtype=target_dtype)
        if action_noise is None: action_noise = torch.zeros_like(action)
        return t, action_noise * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)

    def guidance_tokens(self, vlm_features):
        return self.guidance_decoder.decoder_tokens(vlm_features)


class _FakeGuidedPolicy:
    def __init__(self):
        self.model = _FakeGuidedModel()

    def _prepare_action_targets(self, batch):
        return batch["action"]


def test_guided_action_loss_helper_smoke():
    policy = _FakeGuidedPolicy()
    processed_batch = {"action": torch.randn(2, 4, 6)}
    inputs = {"input_ids": torch.zeros(2, 3, dtype=torch.long), "proprio": torch.randn(2, 5), "domain_id": torch.zeros(2, dtype=torch.long)}
    enc = {"vlm_features": torch.randn(2, 8, 6), "aux_visual_inputs": torch.randn(2, 10, 6)}
    action_loss, action_stats, guidance_tokens = compute_guided_action_loss_from_encoder(policy, processed_batch, inputs, enc)
    assert action_loss.ndim == 0
    assert guidance_tokens.shape == (2, 4, 6)
    assert "action_total" in action_stats


def test_guidance_loss_helper_smoke():
    policy = _FakeGuidedPolicy()
    target = TeacherTarget(name="cedirnet", tensor=torch.randn(2, 3, 2, 2), kind="dense_map", loss_type="mse", weight=1.0)
    loss, stats = compute_guidance_loss(policy, target, torch.randn(2, 4, 6))
    assert loss.ndim == 0
    assert "expert_total" in stats


def test_explicit_stage_freezes_xvla_vlm_by_default():
    policy = type("Policy", (), {"model": type("Model", (), {"vlm": nn.Linear(4, 4)})()})()
    config = GuidedXVLATrainConfig(name="guided", xvla_init_path="base", decoder_init_path="decoder", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", dataset_repo_id="user/dataset", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu")
    _configure_explicit_stage_trainability(policy, config)
    assert all(not parameter.requires_grad for parameter in policy.model.vlm.parameters())
    assert policy.model.vlm.training is False


class _FakePipeline:
    def __init__(self):
        self.steps = []

    def __call__(self, transition):
        out = transition
        for step in self.steps: out = step(out)
        return out


class _FakePolicyCfg:
    def __init__(self, action_mode="so101_ee6d", chunk_size=32):
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.optimizer_lr = 1e-4
        self.scheduler_decay_lr = 2.5e-6
        self.device = "cpu"


class _FakeRuntimePolicy:
    def __init__(self, cfg):
        self.config = cfg


class _FakeGuidedBatchPolicy:
    def __init__(self, chunk_size=32, dim_action=20):
        self.config = type("Cfg", (), {"chunk_size": chunk_size})()
        self.model = type("Model", (), {"dim_action": dim_action})()

    def _prepare_action_targets(self, batch):
        actions = batch["action"]
        if actions.ndim == 2: actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action: actions = pad_vector(actions, self.model.dim_action)
        return actions


def test_build_xvla_runtime_uses_delta_timestamps_and_slice_step(monkeypatch):
    captured = {}
    fake_policy_cfg = _FakePolicyCfg()
    fake_meta = type("Meta", (), {"camera_keys": ["observation.images.main", "observation.images.secondary"], "stats": {"action": {}, "observation.state": {}}})()
    fake_dataset = type("Dataset", (), {"meta": fake_meta})()
    fake_preprocessor, fake_postprocessor = _FakePipeline(), object()

    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.PreTrainedConfig.from_pretrained", lambda path: fake_policy_cfg)
    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.LeRobotDatasetMetadata", lambda repo_id, root=None, revision=None: fake_meta)

    def _fake_dataset_ctor(repo_id, root=None, revision=None, delta_timestamps=None, video_backend=None, tolerance_s=None):
        captured["delta_timestamps"] = delta_timestamps
        captured["dataset_root"] = root
        return fake_dataset

    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.LeRobotDataset", _fake_dataset_ctor)
    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.resolve_delta_timestamps", lambda policy_cfg, ds_meta: {"action": [0.0, 0.1], "observation.state": [0.0]})
    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.sync_xvla_policy_config", lambda policy_cfg, dataset_meta, rename_map: None)
    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.make_xvla_runtime_processors", lambda **kwargs: (fake_preprocessor, fake_postprocessor))
    monkeypatch.setattr("thesis_vla.training.xvla_guided_trainer.XVLAPolicy.from_pretrained", lambda pretrained_name_or_path, config=None, device=None: _FakeRuntimePolicy(config))

    config = GuidedXVLATrainConfig(name="guided", xvla_init_path="base", decoder_init_path="decoder", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", dataset_repo_id="user/dataset", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", action_mode="so101_joint", xvla_optimizer_lr=3e-5, xvla_scheduler_decay_lr=1e-6)
    runtime = build_xvla_runtime(config)

    assert captured["delta_timestamps"] == {"action": [0.0, 0.1], "observation.state": [0.0]}
    assert runtime.policy_device == "cpu"
    assert runtime.vlm_device == "cpu"
    assert runtime.vlm_only_distill is False
    assert runtime.policy.config.action_mode == "so101_joint"
    assert runtime.policy.config.optimizer_lr == 3e-5
    assert runtime.policy.config.scheduler_decay_lr == 1e-6
    expected_slice_map = build_slice_map(get_so101_slice_spec("so101_joint"))
    assert any(isinstance(step, SliceProcessorStep) and step.slice_map == expected_slice_map for step in runtime.preprocessor.steps)


def test_guided_contract_slices_real_dims_before_policy_padding():
    slice_step = SliceProcessorStep(slice_map=build_slice_map(get_so101_slice_spec("so101_ee6d")))
    raw_batch = {
        "action": torch.randn(2, 4, 16),
        "observation": {"observation.state": torch.randn(2, 16)},
    }

    processed = slice_step(raw_batch)

    assert processed["action"].shape == (2, 4, 10)
    assert processed["observation"]["observation.state"].shape == (2, 10)
    policy = _FakeGuidedBatchPolicy(chunk_size=32, dim_action=20)
    targets = policy._prepare_action_targets({"action": processed["action"]})
    assert targets.shape == (2, 32, 20)
    assert torch.equal(targets[:, :4, :10], processed["action"])
    assert torch.count_nonzero(targets[:, :4, 10:]) == 0
    assert torch.count_nonzero(targets[:, 4:, :]) == 0


def test_episode_split_indices_hold_out_full_episodes():
    runtime = type("Runtime", (), {"dataset": type("Dataset", (), {"hf_dataset": {"episode_index": [0, 0, 1, 1, 2, 2, 3, 3]}})()})()
    config = GuidedXVLATrainConfig(name="guided", xvla_init_path="base", decoder_init_path="decoder", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", dataset_repo_id="user/dataset", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", validation_enable=True, validation_split_ratio=0.25, validation_seed=7)
    split = _episode_split_indices(runtime, config)

    assert split is not None
    train_indices, val_indices = split
    train_episodes = {runtime.dataset.hf_dataset["episode_index"][index] for index in train_indices}
    val_episodes = {runtime.dataset.hf_dataset["episode_index"][index] for index in val_indices}
    assert train_episodes.isdisjoint(val_episodes)
    assert len(val_episodes) == 1


def test_maybe_init_wandb_uses_config_name(monkeypatch):
    captured = {}

    class _FakeWandb:
        @staticmethod
        def init(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    config = GuidedXVLATrainConfig(name="guided-run", xvla_init_path="base", decoder_init_path="decoder", decoder_stack_config_path="stack.yaml", decoder_task_config_path="task.yaml", dataset_repo_id="user/dataset", dataset_revision="main", dataset_root=None, output_dir="/tmp/out", device="cpu", wandb_enable=True, wandb_project="proj")
    run = _maybe_init_wandb(config)

    assert run is not None
    assert captured["project"] == "proj"
    assert captured["name"] == "guided-run"
