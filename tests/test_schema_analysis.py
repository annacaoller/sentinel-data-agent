from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.schema_analysis import analyze_schema


def test_schema_analysis_basic() -> None:
    result = analyze_schema("examples/example.csv").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["num_rows"] == 4
    assert result["num_columns"] == 3

    assert "id" in result["numeric_columns"]
    assert "value" in result["numeric_columns"]

    assert "category" in result["categorical_columns"]

    assert result["boolean_columns"] == []
    assert result["datetime_columns"] == []