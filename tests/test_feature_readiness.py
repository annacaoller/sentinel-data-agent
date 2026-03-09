from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.feature_readiness import assess_feature_readiness


def test_feature_readiness_basic() -> None:
    result = assess_feature_readiness("examples/example.csv", "category").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["selected_target"] == "category"
    assert result["target_present"] is True
    assert result["target_dtype"] == "object"

    assert result["candidate_feature_columns"] == ["id", "value"]
    assert result["numeric_feature_columns"] == ["id", "value"]
    assert result["categorical_feature_columns"] == []

    assert result["blocked_reasons"] == []
    assert result["modeling_ready"] is True