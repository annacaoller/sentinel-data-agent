from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.target_inference import infer_target_column


def test_target_inference_basic() -> None:
    result = infer_target_column("examples/example.csv").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["selected_target"] == "category"
    assert result["confidence"] == "medium"
    assert result["classification_compatible"] is True

    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) == 3

    best_candidate = result["candidates"][0]
    assert best_candidate["column"] == "category"
    assert "categorical_small_cardinality" in best_candidate["reasons"]