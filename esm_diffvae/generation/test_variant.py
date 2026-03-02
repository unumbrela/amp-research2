import torch

from generation.variant import (
    enforce_edit_constraints,
    sample_logits,
    score_variant,
    summarize_variants,
)


def test_sample_logits_shape_and_range():
    torch.manual_seed(42)
    logits = torch.randn(4, 7, 21)
    sampled = sample_logits(logits, temperature=1.1, top_k=5, top_p=0.95)
    assert sampled.shape == (4, 7)
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < 21


def test_score_variant_prefers_target_band():
    ideal = score_variant(identity=0.75, edit_distance=5)
    weak = score_variant(identity=0.30, edit_distance=15)
    assert ideal > weak


def test_summarize_variants_basic_stats():
    variants = [
        {"sequence": "AAAAK", "identity": 0.70, "edit_distance": 4},
        {"sequence": "AAAAR", "identity": 0.80, "edit_distance": 5},
        {"sequence": "AAAAR", "identity": 0.80, "edit_distance": 5},
    ]
    summary = summarize_variants(variants)
    assert summary["count"] == 3
    assert summary["unique_count"] == 2
    assert 0 < summary["uniqueness_rate"] < 1
    assert 0.7 <= summary["mean_identity"] <= 0.8


def test_enforce_edit_constraints_hits_target_band():
    parent = torch.tensor([1, 2, 3, 4, 5])
    sampled = torch.tensor([6, 7, 8, 9, 10])
    logits = torch.randn(5, 21)
    out = enforce_edit_constraints(
        sampled_idx=sampled,
        logits_row=logits,
        parent_idx=parent,
        min_edits=2,
        max_edits=3,
        pad_idx=20,
    )
    edits = int((out != parent).sum().item())
    assert 2 <= edits <= 3
