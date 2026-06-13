from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

if "gymnasium" not in sys.modules: sys.modules["gymnasium"] = types.SimpleNamespace(Env=object, Wrapper=object, vector=types.SimpleNamespace(VectorEnv=object))

from lerobot.configs.types import FeatureType, PolicyFeature
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


def _make_guided_config(guidance_expert_type="cedirnet"):
    guidance_head = {"grid_hw": [2, 2], "projection_mode": "linear", "projection_mlp_ratio": 2.0, "projection_dropout": 0.0, "refine_layers": 1, "refine_kernel_size": 3, "refine_dropout": 0.0, "out_layers": 1, "out_dropout": 0.0, "resize_mode": "bilinear", "align_corners": False} if guidance_expert_type == "cedirnet" else {"query_projection_mode": "mlp", "query_projection_mlp_ratio": 1.0, "query_projection_dropout": 0.0, "query_aggregation_mode": "mean", "align_weight": 1.0, "recon_weight": 1.0, "recon_scale": 1.0}
    guidance_teacher = {"name": "cedirnet", "target_kind": "dense_map", "loss_type": "mse", "weight": 1.0, "target_channel_indices": [0, 1, 2]} if guidance_expert_type == "cedirnet" else {"name": "dinov2", "target_kind": "token_sequence", "loss_type": "mse", "weight": 1.0, "model_type": "vitb14"}
    return XVLAGuidedConfig(
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
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


def test_guided_transformer_fusion_variants():
    for fusion_mode in ["concat", "gated_concat", "cross_attention", "gated_cross_attention"]:
        model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_fusion_mode=fusion_mode, guidance_gated=False)
        out = model(domain_id=torch.zeros(2, dtype=torch.long), vlm_features=torch.randn(2, 6, 16), aux_visual_inputs=torch.randn(2, 4, 16), guidance_tokens=torch.randn(2, 5, 8), action_with_noise=torch.randn(2, 4, 4), proprio=torch.randn(2, 4), t=torch.rand(2))
        assert out.shape == (2, 4, 4)


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


def test_guided_transformer_sequence_shape_rules():
    domain_id = torch.zeros(2, dtype=torch.long)
    vlm_features, aux_visual_inputs, guidance_tokens = torch.randn(2, 6, 16), torch.randn(2, 4, 16), torch.randn(2, 5, 8)
    action_with_noise, proprio, t = torch.randn(2, 4, 4), torch.randn(2, 4), torch.rand(2)
    for fusion_mode in ["concat", "gated_concat", "cross_attention", "gated_cross_attention"]:
        model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_fusion_mode=fusion_mode, guidance_gated=False)
        action_proj, z_proj, aux_proj = model._project_native_tokens(domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t)
        guidance_proj = model._project_guidance(guidance_tokens, domain_id)
        native_len = action_proj.shape[1] + z_proj.shape[1] + aux_proj.shape[1]
        if fusion_mode in {"concat", "gated_concat"}:
            fused = model._apply_concat_fusion(torch.cat([action_proj, z_proj, aux_proj], dim=1), guidance_proj)
            assert fused.shape[1] == native_len + guidance_proj.shape[1]
        else:
            z_guided = model._apply_cross_attention_fusion(z_proj, guidance_proj)
            fused = torch.cat([action_proj, z_guided, aux_proj], dim=1)
            assert fused.shape[1] == native_len


def test_gated_cross_attention_starts_with_near_zero_effect():
    model = GuidedSoftPromptedTransformer(hidden_size=16, multi_modal_input_size=16, guidance_input_size=8, depth=2, num_heads=4, guidance_num_heads=4, mlp_ratio=2.0, num_domains=3, dim_action=4, dim_propio=4, dim_time=8, len_soft_prompts=2, max_len_seq=64, use_hetero_proj=False, guidance_fusion_mode="gated_cross_attention", guidance_gated=False)
    z_proj, guidance_proj = torch.randn(2, 6, 16), torch.randn(2, 5, 16)
    assert torch.allclose(model._apply_cross_attention_fusion(z_proj, guidance_proj), z_proj)
    assert float(model.cross_attn_gamma.detach().item()) == 0.0


def test_guided_policy_save_load_and_runtime_resolution(monkeypatch):
    _patch_dummy_vlm(monkeypatch)
    config = _make_guided_config()
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


def test_guided_dino_policy_save_load_and_runtime_resolution(monkeypatch):
    _patch_dummy_vlm(monkeypatch)
    config = _make_guided_config(guidance_expert_type="dino")
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
