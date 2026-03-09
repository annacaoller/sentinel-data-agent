from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.baseline_classification import run_baseline_classification


def test_baseline_classification_blocked_small_class() -> None:
    result = run_baseline_classification("examples/example.csv", "category").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["selected_target"] == "category"

    assert result["modeling_executed"] is False
    assert "least_populated_class_has_less_than_two_samples" in result["blocked_reasons"]

    assert result["num_rows_used"] == 4
    assert result["num_features_used"] == 2

    assert result["metrics"] == {}

    assert set(result["class_labels"]) == {"A", "B", "C"}