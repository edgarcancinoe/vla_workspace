from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types

import torch
from torch import nn

for candidate in [Path(__file__).resolve().parents[2] / "repos" / "lerobot" / "src", Path(__file__).resolve().parents[1] / "lerobot" / "src", Path(__file__).resolve().parents[1] / "src"]:
    if candidate.exists(): sys.path.insert(0, str(candidate))

if "gymnasium" not in sys.modules: sys.modules["gymnasium"] = types.SimpleNamespace(Env=object, Wrapper=object, vector=types.SimpleNamespace(VectorEnv=object))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xvla.soft_transformer import Attention
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from thesis_vla.inference.runtime_policy import load_runtime_policy
from thesis_vla.policies.xvla_guided import XVLAGuidedConfig, XVLAGuidedPolicy
from thesis_vla.policies.xvla_guided.modeling_xvla_guided import GuidedSoftPromptedTransformer


class _DummyEncoder(nn.Module):
    def forward(self, attention_mask=None, inputs_embeds=None):
        return (inputs_embeds,)


class _DummyLanguageModelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyEncoder()
        self.decoder = nn.Identity()
        self.shared = nn.Embedding(32, 16)


class _DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyLanguageModelModel()
        self.lm_head = nn.Linear(16, 16)


class _DummyFlorence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=16)
        self.language_model = _DummyLanguageModel()
        self.emb = nn.Embedding(64, 16)
        self.vision_proj = nn.Linear(3, 16)

    def _encode_image(self, valid_images):
        pooled = valid_images.float().mean(dim=(-1, -2)).unsqueeze(1)
        return self.vision_proj(pooled).expand(-1, 4, -1)

    def get_input_embeddings(self):
        return self.emb

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
        return inputs_embeds, torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)


def _patch_dummy_vlm(monkeypatch):
    import lerobot.policies.xvla.modeling_xvla as base_modeling
    monkeypatch.setattr(base_modeling, "Florence2ForConditionalGeneration", _DummyFlorence)
    monkeypatch.setattr(XVLAGuidedConfig, "get_florence_config", lambda self: SimpleNamespace(projection_dim=16))


def _make_guided_config(guidance_expert_type="cedirnet", **overrides):
    guidance_head = {"grid_hw": [2, 2], "projection_mode": "linear", "projection_mlp_ratio": 2.0, "projection_dropout": 0.0, "refine_layers": 1, "refine_kernel_size": 3, "refine_dropout": 0.0, "out_layers": 1, "out_dropout": 0.0, "resize_mode": "bilinear", "align_corners": False} if guidance_expert_type == "cedirnet" else {"query_projection_mode": "mlp", "query_projection_mlp_ratio": 1.0, "query_projection_dropout": 0.0, "query_aggregation_mode": "mean", "align_weight": 1.0, "recon_weight": 1.0, "recon_scale": 1.0}
    guidance_teacher = {"name": "cedirnet", "target_kind": "dense_map", "loss_type": "mse", "weight": 1.0, "target_channel_indices": [0, 1, 2]} if guidance_expert_type == "cedirnet" else {"name": "dinov2", "target_kind": "token_sequence", "loss_type": "mse", "weight": 1.0, "model_type": "vitb14"}
    payload = dict(
        input_features={f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)), f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)), OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        florence_config={"vision_config": {}, "text_config": {}},
        guidance_expert_type=guidance_expert_type,
        hidden_size=16,
        depth=2,
        num_heads=4,
        max_len_seq=48,
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        num_image_views=2,
        action_mode="auto",
        guidance_decoder_stack={"decoder_dim": 8, "num_decoder_tokens": 4, "num_heads": 4, "num_layers": 1, "ffn_enabled": True, "ffn_mlp_ratio": 2.0, "ffn_dropout": 0.0, "self_attn_queries": True, "self_attn_student": False, "gating_enabled": False, "gating_mode": "none", "cross_attn_residual": False, "student_projection_mode": "linear", "student_projection_mlp_ratio": 2.0, "student_projection_dropout": 0.0, "positional_encodings": False},
        guidance_decoder_head=guidance_head,
        guidance_decoder_teacher=guidance_teacher,
    )
    payload.update(overrides)
    return XVLAGuidedConfig(**payload)


def test_guided_transformer_supports_concat_and_selected_layer_variants():
    variants = [
        dict(guidance_mode="concat", guidance_insertion_position="after_visual", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=()),
        dict(guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=()),
        dict(guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=True, guidance_interface_num_tokens=3, guidance_concat_gating=False, guidance_selected_layers=()),
        dict(guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=True, guidance_selected_layers=()),
        dict(guidance_mode="selected_layers", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=(1,)),
        dict(guidance_mode="selected_layers", guidance_insertion_position="after_visual", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=(0, 1)),
    ]
    for kwargs in variants:
        model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, **kwargs)
        out = model(domain_id=torch.zeros(2, dtype=torch.long), vlm_features=torch.randn(2, 6, 16), aux_visual_inputs=torch.randn(2, 4, 16), guidance_tokens=torch.randn(2, 5, 8), action_with_noise=torch.randn(2, 4, 4), proprio=torch.randn(2, 4), t=torch.rand(2))
        assert out.shape == (2, 4, 4)


def test_guided_transformer_supports_legacy_blocks_without_token_keep_mask():
    class _LegacyBlock(nn.Module):
        def forward(self, x):
            return x

    model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=1, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_mode="concat", guidance_insertion_position="after_visual", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=())
    model.blocks = nn.ModuleList([_LegacyBlock()])
    out = model(domain_id=torch.zeros(2, dtype=torch.long), vlm_features=torch.randn(2, 6, 16), aux_visual_inputs=torch.randn(2, 4, 16), guidance_tokens=torch.randn(2, 5, 8), guidance_available=torch.ones(2, 1, 1), action_with_noise=torch.randn(2, 4, 4), proprio=torch.randn(2, 4), t=torch.rand(2))
    assert out.shape == (2, 4, 4)


def test_attention_all_true_mask_preserves_default_behavior():
    attn = Attention(dim=16, num_heads=4)
    x = torch.randn(2, 6, 16)
    keep_mask = torch.ones(2, 6, dtype=torch.bool)
    attn.eval()
    with torch.no_grad():
        baseline = attn(x)
        masked = attn(x, token_keep_mask=keep_mask)
    assert torch.allclose(baseline, masked, atol=1e-6)


def test_guided_dino_rejects_expert_feature_query():
    try:
        base = _make_guided_config(guidance_expert_type="dino")
        payload = {name: getattr(base, name) for name in base.__dataclass_fields__}
        payload["guidance_decoder_teacher"] = {"name": "dinov2", "target_kind": "expert_feature_query", "loss_type": "mse", "weight": 1.0, "model_type": "vitb14"}
        _ = XVLAGuidedConfig(**payload)
    except ValueError as exc:
        assert "token_sequence" in str(exc)
    else:
        raise AssertionError("Expected guided DINO config to reject expert_feature_query.")


def test_guided_transformer_concat_sequence_shape_rules():
    domain_id = torch.zeros(2, dtype=torch.long)
    vlm_features, aux_visual_inputs, guidance_tokens = torch.randn(2, 6, 16), torch.randn(2, 4, 16), torch.randn(2, 5, 8)
    action_with_noise, proprio, t = torch.randn(2, 4, 4), torch.randn(2, 4), torch.rand(2)
    for use_interface, gating in [(False, False), (True, False), (False, True)]:
        model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=use_interface, guidance_interface_num_tokens=3 if use_interface else None, guidance_concat_gating=gating, guidance_selected_layers=())
        action_proj, z_proj, aux_proj = model._project_native_tokens(domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t)
        guidance_proj = model._prepare_guidance_context(guidance_tokens, domain_id)
        fused = model._apply_concat_fusion(action_proj, z_proj, aux_proj, guidance_proj)
        assert fused.shape[1] == action_proj.shape[1] + z_proj.shape[1] + aux_proj.shape[1] + guidance_proj.shape[1]


def test_guided_transformer_no_guidance_is_invariant_for_concat_and_selected_layers():
    domain_id = torch.zeros(2, dtype=torch.long)
    vlm_features, aux_visual_inputs = torch.randn(2, 6, 16), torch.randn(2, 4, 16)
    action_with_noise, proprio, t = torch.randn(2, 4, 4), torch.randn(2, 4), torch.rand(2)
    guidance_a, guidance_b = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    guidance_off = torch.zeros(2, 1, 1)
    for kwargs in [
        dict(guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=()),
        dict(guidance_mode="selected_layers", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=(1,)),
    ]:
        model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, **kwargs)
        model.eval()
        with torch.no_grad():
            out_a = model(domain_id=domain_id, vlm_features=vlm_features, aux_visual_inputs=aux_visual_inputs, guidance_tokens=guidance_a, guidance_available=guidance_off, action_with_noise=action_with_noise, proprio=proprio, t=t)
            out_b = model(domain_id=domain_id, vlm_features=vlm_features, aux_visual_inputs=aux_visual_inputs, guidance_tokens=guidance_b, guidance_available=guidance_off, action_with_noise=action_with_noise, proprio=proprio, t=t)
        assert torch.allclose(out_a, out_b, atol=1e-6)


def test_guided_transformer_selected_layers_only_expand_selected_blocks():
    class _RecordingBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, token_keep_mask=None):  # noqa: ARG002
            self.seen.append(int(x.shape[1]))
            return x

    model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=3, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_mode="selected_layers", guidance_insertion_position="before_vlm", guidance_use_interface_projection=False, guidance_interface_num_tokens=None, guidance_concat_gating=False, guidance_selected_layers=(1,))
    model.blocks = nn.ModuleList([_RecordingBlock(), _RecordingBlock(), _RecordingBlock()])
    _ = model(domain_id=torch.zeros(2, dtype=torch.long), vlm_features=torch.randn(2, 6, 16), aux_visual_inputs=torch.randn(2, 4, 16), guidance_tokens=torch.randn(2, 5, 8), action_with_noise=torch.randn(2, 4, 4), proprio=torch.randn(2, 4), t=torch.rand(2))
    assert model.blocks[0].seen == [16]
    assert model.blocks[1].seen == [21]
    assert model.blocks[2].seen == [16]


def test_guided_policy_save_load_and_runtime_resolution(monkeypatch):
    _patch_dummy_vlm(monkeypatch)
    config = _make_guided_config(guidance_mode="concat", guidance_insertion_position="before_vlm", guidance_use_interface_projection=True, guidance_interface_num_tokens=3, guidance_concat_gating=True)
    policy = XVLAGuidedPolicy(config)
    assert all(not parameter.requires_grad for parameter in policy.model.guidance_decoder.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        policy.save_pretrained(save_dir)
        loaded = XVLAGuidedPolicy.from_pretrained(save_dir, device="cpu")
        actions = loaded.model.generate_actions(input_ids=torch.zeros(2, 5, dtype=torch.long), image_input=torch.randn(2, 2, 3, 8, 8), image_mask=torch.ones(2, 2, dtype=torch.bool), domain_id=torch.zeros(2, dtype=torch.long), proprio=torch.randn(2, 4), steps=2)
        runtime_policy, include_eef_state = load_runtime_policy("xvla_guided", str(save_dir), "cpu")
        assert actions.shape == (2, 4, 4)
        assert runtime_policy.config.type == "xvla_guided"
        assert include_eef_state is False


def test_guided_policy_staged_schedule_switches_decoder_trainability(monkeypatch):
    _patch_dummy_vlm(monkeypatch)
    config = _make_guided_config(guidance_training_schedule="decoder_warmup_then_policy", guidance_warmup_steps=5)
    policy = XVLAGuidedPolicy(config)
    policy.model.set_guidance_trainability(5)
    assert all(parameter.requires_grad for parameter in policy.model.guidance_decoder.parameters())
    policy.model.set_guidance_trainability(6)
    assert all(not parameter.requires_grad for parameter in policy.model.guidance_decoder.parameters())


def test_guided_dino_policy_save_load_and_runtime_resolution(monkeypatch):
    _patch_dummy_vlm(monkeypatch)
    config = _make_guided_config(guidance_expert_type="dino", guidance_mode="selected_layers", guidance_selected_layers=(1,))
    policy = XVLAGuidedPolicy(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        policy.save_pretrained(save_dir)
        loaded = XVLAGuidedPolicy.from_pretrained(save_dir, device="cpu")
        actions = loaded.model.generate_actions(input_ids=torch.zeros(2, 5, dtype=torch.long), image_input=torch.randn(2, 2, 3, 8, 8), image_mask=torch.ones(2, 2, dtype=torch.bool), domain_id=torch.zeros(2, dtype=torch.long), proprio=torch.randn(2, 4), steps=2)
        runtime_policy, _ = load_runtime_policy("xvla_guided", str(save_dir), "cpu")
        assert actions.shape == (2, 4, 4)
        assert runtime_policy.config.guidance_expert_type == "dino"
        try:
            runtime_policy.model.guidance_map(torch.randn(1, 4, 16))
        except ValueError as exc:
            assert "CeDirNet guidance only" in str(exc)
        else:
            raise AssertionError("Expected DINO guided policy to reject guidance_map().")
