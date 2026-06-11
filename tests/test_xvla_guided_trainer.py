from pathlib import Path
import sys

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from thesis_vla.training.xvla_guided_trainer import GuidedXVLATrainConfig, _configure_explicit_stage_trainability, compute_guidance_loss, compute_guided_action_loss_from_encoder
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
