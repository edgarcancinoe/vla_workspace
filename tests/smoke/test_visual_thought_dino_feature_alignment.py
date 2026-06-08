from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from thesis_vla.common.paths import PROJECT_ROOT
from thesis_vla.visual_thought import DinoFeatureAlignmentModel, TeacherTarget, compute_feature_alignment_loss, load_dino_feature_alignment_config


def test_dino_feature_alignment_smoke():
    cfg = load_dino_feature_alignment_config(PROJECT_ROOT / "config" / "visual_thought" / "dino_stack.yaml", PROJECT_ROOT / "config" / "visual_thought" / "dino_feature_alignment.yaml")
    batch_size, num_query_tokens, num_spatial_tokens, student_vlm_dim, expert_dim = 2, 4, 64, 64, 768
    target = TeacherTarget(name="dinov2", tensor=torch.randn(batch_size, 1, 32, 32), kind="expert_feature_query", loss_type="mse", weight=1.0, aux={"expert_feature_layout": "patch", "expert_features": torch.randn(batch_size, num_query_tokens, num_spatial_tokens, expert_dim), "patch_hw": (8, 8), "expert_spatial_hw": (8, 8)})
    model = DinoFeatureAlignmentModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=cfg)
    query_tokens = model.query_tokens(torch.randn(batch_size, 256, student_vlm_dim))
    token_maps, final_map = model.query_reconstruct(query_tokens, target)
    attended_stu, teacher_aligned_exp = model.query_align_features(query_tokens, target)
    loss = compute_feature_alignment_loss(attended_stu, teacher_aligned_exp, cfg.head.align_weight)
    assert query_tokens.shape == (batch_size, num_query_tokens, expert_dim)
    assert token_maps.shape == (batch_size, num_query_tokens, 32, 32)
    assert final_map.shape == (batch_size, 1, 32, 32)
    assert attended_stu.shape == (batch_size, num_query_tokens, expert_dim)
    assert teacher_aligned_exp.shape == (batch_size, num_query_tokens, expert_dim)
    assert loss.ndim == 0
