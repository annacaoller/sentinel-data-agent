from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


@dataclass(frozen=True)
class FeatureReadinessResult:
    dataset_path: str
    selected_target: Optional[str]
    target_present: bool
    target_dtype: Optional[str]
    candidate_feature_columns: List[str]
    numeric_feature_columns: List[str]
    categorical_feature_columns: List[str]
    blocked_reasons: List[str]
    modeling_ready: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "selected_target": self.selected_target,
            "target_present": self.target_present,
            "target_dtype": self.target_dtype,
            "candidate_feature_columns": self.candidate_feature_columns,
            "numeric_feature_columns": self.numeric_feature_columns,
            "categorical_feature_columns": self.categorical_feature_columns,
            "blocked_reasons": self.blocked_reasons,
            "modeling_ready": self.modeling_ready,
        }


def assess_feature_readiness(
    dataset_path: str | Path,
    selected_target: Optional[str],
) -> FeatureReadinessResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    blocked_reasons: List[str] = []

    target_present = bool(selected_target) and selected_target in df.columns
    target_dtype: Optional[str] = None

    if not selected_target:
        blocked_reasons.append("no_target_selected")
    elif selected_target not in df.columns:
        blocked_reasons.append("selected_target_not_found")

    if target_present:
        target_dtype = str(df[selected_target].dtype)

    candidate_feature_columns = [
        str(col)
        for col in df.columns
        if str(col) != str(selected_target)
    ]

    if not candidate_feature_columns:
        blocked_reasons.append("no_feature_columns_available")

    numeric_feature_columns = [
        str(col)
        for col in candidate_feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_feature_columns = [
        str(col)
        for col in candidate_feature_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if target_present and df[selected_target].nunique(dropna=False) <= 1:
        blocked_reasons.append("target_is_constant")

    modeling_ready = len(blocked_reasons) == 0

    return FeatureReadinessResult(
        dataset_path=str(path),
        selected_target=selected_target,
        target_present=target_present,
        target_dtype=target_dtype,
        candidate_feature_columns=candidate_feature_columns,
        numeric_feature_columns=numeric_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        blocked_reasons=blocked_reasons,
        modeling_ready=modeling_ready,
    )