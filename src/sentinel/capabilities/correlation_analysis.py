from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


CorrelationMethod = Literal["pearson", "kendall", "spearman"]


@dataclass(frozen=True)
class CorrelationAnalysisResult:
    dataset_path: str
    num_numeric_columns: int
    correlation_method: CorrelationMethod
    correlation_matrix: Dict[str, Dict[str, float]]
    strongest_pair: Dict[str, Any] | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "num_numeric_columns": self.num_numeric_columns,
            "correlation_method": self.correlation_method,
            "correlation_matrix": self.correlation_matrix,
            "strongest_pair": self.strongest_pair,
        }


def _to_python_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def analyze_correlations(
    dataset_path: str | Path,
    method: CorrelationMethod = "pearson",
) -> CorrelationAnalysisResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty or numeric_df.shape[1] < 2:
        return CorrelationAnalysisResult(
            dataset_path=str(path),
            num_numeric_columns=int(numeric_df.shape[1]),
            correlation_method=method,
            correlation_matrix={},
            strongest_pair=None,
        )

    corr_df = numeric_df.corr(method=method)

    correlation_matrix: Dict[str, Dict[str, float]] = {
        str(row): {
            str(col): _to_python_float(corr_df.loc[row, col])
            for col in corr_df.columns
        }
        for row in corr_df.index
    }

    strongest_pair: Dict[str, Any] | None = None
    strongest_abs = -1.0

    columns = list(corr_df.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = str(columns[i])
            col_b = str(columns[j])
            value = _to_python_float(corr_df.iloc[i, j])
            abs_value = abs(value)

            if abs_value > strongest_abs:
                strongest_abs = abs_value
                strongest_pair = {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": value,
                    "absolute_correlation": abs_value,
                }

    return CorrelationAnalysisResult(
        dataset_path=str(path),
        num_numeric_columns=int(numeric_df.shape[1]),
        correlation_method=method,
        correlation_matrix=correlation_matrix,
        strongest_pair=strongest_pair,
    )