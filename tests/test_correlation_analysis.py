from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.correlation_analysis import analyze_correlations


def test_correlation_analysis_basic() -> None:
    result = analyze_correlations("examples/example.csv").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["num_numeric_columns"] == 2
    assert result["correlation_method"] == "pearson"

    matrix = result["correlation_matrix"]

    assert "id" in matrix
    assert "value" in matrix["id"]

    strongest = result["strongest_pair"]

    assert strongest["column_a"] in ["id", "value"]
    assert strongest["column_b"] in ["id", "value"]

    assert strongest["absolute_correlation"] > 0