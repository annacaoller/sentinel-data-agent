from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


@dataclass(frozen=True)
class EdaSummaryResult:
    dataset_path: str
    num_rows: int
    num_columns: int
    numeric_summary: Dict[str, Dict[str, float]]
    categorical_summary: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "numeric_summary": self.numeric_summary,
            "categorical_summary": self.categorical_summary,
        }


def generate_eda_summary(dataset_path: str | Path) -> EdaSummaryResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    numeric_summary: Dict[str, Dict[str, float]] = {}
    categorical_summary: Dict[str, Dict[str, Any]] = {}

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in numeric_columns:
        series = df[col].dropna()
        numeric_summary[str(col)] = {
            "mean": float(series.mean()) if not series.empty else 0.0,
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()) if not series.empty else 0.0,
            "max": float(series.max()) if not series.empty else 0.0,
            "median": float(series.median()) if not series.empty else 0.0,
        }

    for col in categorical_columns:
        series = df[col].dropna().astype(str)
        top_values = series.value_counts().head(5).to_dict()
        categorical_summary[str(col)] = {
            "unique_values": int(series.nunique()),
            "top_values": {str(k): int(v) for k, v in top_values.items()},
        }

    return EdaSummaryResult(
        dataset_path=str(path),
        num_rows=int(df.shape[0]),
        num_columns=int(df.shape[1]),
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
    )