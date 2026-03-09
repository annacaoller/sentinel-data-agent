from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


@dataclass(frozen=True)
class SchemaAnalysisResult:
    dataset_path: str
    num_rows: int
    num_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    datetime_columns: List[str]
    text_like_columns: List[str]
    likely_identifier_columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "boolean_columns": self.boolean_columns,
            "datetime_columns": self.datetime_columns,
            "text_like_columns": self.text_like_columns,
            "likely_identifier_columns": self.likely_identifier_columns,
        }


def _is_text_like(series: pd.Series) -> bool:
    if str(series.dtype) not in {"object", "string"}:
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    avg_len = non_null.astype(str).str.len().mean()
    unique_ratio = float(non_null.nunique(dropna=True) / len(non_null)) if len(non_null) > 0 else 0.0

    return bool(avg_len >= 20 or unique_ratio > 0.8)


def _is_likely_identifier(series: pd.Series, column_name: str) -> bool:
    normalized = column_name.strip().lower().replace("-", "_").replace(" ", "_")
    name_hints = {"id", "uuid", "key", "hash", "index"}

    if normalized in name_hints or any(hint in normalized for hint in name_hints):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    if len(non_null) < 20:
        return False

    unique_ratio = float(non_null.nunique(dropna=True) / len(non_null)) if len(non_null) > 0 else 0.0
    return unique_ratio > 0.98


def analyze_schema(dataset_path: str | Path) -> SchemaAnalysisResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    boolean_columns: List[str] = []
    datetime_columns: List[str] = []
    text_like_columns: List[str] = []
    likely_identifier_columns: List[str] = []

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        dtype = str(series.dtype)

        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(col_name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(col_name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col_name)
        else:
            categorical_columns.append(col_name)

        if _is_text_like(series):
            text_like_columns.append(col_name)

        if _is_likely_identifier(series, col_name):
            likely_identifier_columns.append(col_name)

    return SchemaAnalysisResult(
        dataset_path=str(path),
        num_rows=int(df.shape[0]),
        num_columns=int(df.shape[1]),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
        datetime_columns=datetime_columns,
        text_like_columns=text_like_columns,
        likely_identifier_columns=likely_identifier_columns,
    )