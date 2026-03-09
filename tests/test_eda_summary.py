from __future__ import annotations

from pathlib import Path

from sentinel.capabilities.eda_summary import generate_eda_summary


def test_eda_summary_basic() -> None:
    result = generate_eda_summary("examples/example.csv").to_dict()

    expected_path = str(Path("examples/example.csv").resolve())
    assert result["dataset_path"] == expected_path

    assert result["num_rows"] == 4
    assert result["num_columns"] == 3

    numeric = result["numeric_summary"]
    categorical = result["categorical_summary"]

    assert "id" in numeric
    assert "value" in numeric

    assert numeric["id"]["min"] == 1.0
    assert numeric["id"]["max"] == 4.0

    assert "category" in categorical
    assert categorical["category"]["unique_values"] == 3