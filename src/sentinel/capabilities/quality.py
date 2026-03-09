from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


@dataclass(frozen=True)
class DatasetQualityReport:
    dataset_path: str
    num_rows: int
    num_columns: int
    duplicate_rows: int
    duplicate_rate: float
    missing_by_column: Dict[str, int]
    missing_rate_by_column: Dict[str, float]
    constant_columns: List[str]
    high_missing_columns: List[str]
    empty_columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_rate": self.duplicate_rate,
            "missing_by_column": self.missing_by_column,
            "missing_rate_by_column": self.missing_rate_by_column,
            "constant_columns": self.constant_columns,
            "high_missing_columns": self.high_missing_columns,
            "empty_columns": self.empty_columns,
        }


def assess_dataset_quality(
    dataset_path: str | Path,
    high_missing_threshold: float = 0.5,
) -> DatasetQualityReport:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    num_rows = int(df.shape[0])
    num_columns = int(df.shape[1])

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = float(duplicate_rows / num_rows) if num_rows > 0 else 0.0

    missing_by_column = {str(col): int(df[col].isna().sum()) for col in df.columns}
    missing_rate_by_column = {
        str(col): float(df[col].isna().mean()) if num_rows > 0 else 0.0
        for col in df.columns
    }

    constant_columns = [
        str(col)
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    high_missing_columns = [
        str(col)
        for col in df.columns
        if missing_rate_by_column[str(col)] >= high_missing_threshold
    ]

    empty_columns = [
        str(col)
        for col in df.columns
        if missing_by_column[str(col)] == num_rows and num_rows > 0
    ]

    return DatasetQualityReport(
        dataset_path=str(path),
        num_rows=num_rows,
        num_columns=num_columns,
        duplicate_rows=duplicate_rows,
        duplicate_rate=duplicate_rate,
        missing_by_column=missing_by_column,
        missing_rate_by_column=missing_rate_by_column,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        empty_columns=empty_columns,
    )