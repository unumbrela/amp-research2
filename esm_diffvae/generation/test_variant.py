"""Basic tests for variant generation utilities."""

from generation.variant import (
    edit_distance,
    highlight_mutations,
    sequence_identity,
    _as_int_choices,
    _normalize_mode_weights,
    _allocate_counts,
)


def test_edit_distance():
    assert edit_distance("ABC", "ABC") == 0
    assert edit_distance("ABC", "ABD") == 1
    assert edit_distance("ABC", "ABCD") == 1
    assert edit_distance("", "ABC") == 3


def test_sequence_identity():
    assert sequence_identity("AAAA", "AAAA") == 1.0
    assert sequence_identity("AAAA", "AABB") == 0.5
    assert sequence_identity("AA", "AABB") == 0.5  # 2 matches / 4 max_len


def test_highlight_mutations():
    assert highlight_mutations("ABC", "ABD") == "ABd"
    assert highlight_mutations("ABC", "ABCDE") == "ABC|de"


def test_as_int_choices():
    assert _as_int_choices([1, 5], []) == [1, 2, 3, 4, 5]
    assert _as_int_choices([1, 2, 3], []) == [1, 2, 3]
    assert _as_int_choices(3, []) == [3]


def test_normalize_mode_weights():
    w = _normalize_mode_weights(["a", "b"], {"a": 1.0, "b": 1.0})
    assert abs(w["a"] - 0.5) < 1e-6
    assert abs(w["b"] - 0.5) < 1e-6


def test_allocate_counts():
    counts = _allocate_counts(10, {"a": 0.7, "b": 0.3})
    assert counts["a"] + counts["b"] == 10
    assert counts["a"] == 7
    assert counts["b"] == 3
