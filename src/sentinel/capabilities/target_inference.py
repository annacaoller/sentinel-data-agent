from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


POSITIVE_NAME_HINTS = {
    "target",
    "label",
    "class",
    "outcome",
    "fraud",
    "default",
    "churn",
    "approved",
    "response",
}

NEGATIVE_NAME_HINTS = {
    "id",
    "uuid",
    "key",
    "hash",
    "index",
    "timestamp",
    "date",
    "time",
    "name",
}


@dataclass(frozen=True)
class TargetCandidate:
    column: str
    score: float
    reasons: List[str]
    dtype: str
    unique_values: int
    unique_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "score": self.score,
            "reasons": self.reasons,
            "dtype": self.dtype,
            "unique_values": self.unique_values,
            "unique_ratio": self.unique_ratio,
        }


@dataclass(frozen=True)
class TargetInferenceResult:
    dataset_path: str
    selected_target: Optional[str]
    confidence: str
    score: float
    candidates: List[Dict[str, Any]]
    reasoning: List[str]
    classification_compatible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "selected_target": self.selected_target,
            "confidence": self.confidence,
            "score": self.score,
            "candidates": self.candidates,
            "reasoning": self.reasoning,
            "classification_compatible": self.classification_compatible,
        }


def _normalized_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _score_column_as_target(df: pd.DataFrame, column: str) -> TargetCandidate:
    series = df[column]
    col_name = _normalized_name(str(column))
    tokens = _name_tokens(str(column))
    dtype = str(series.dtype)

    num_rows = len(df)
    unique_values = int(series.nunique(dropna=False))
    unique_ratio = float(unique_values / num_rows) if num_rows > 0 else 0.0

    score = 0.0
    reasons: List[str] = []

    if col_name in POSITIVE_NAME_HINTS or bool(tokens & POSITIVE_NAME_HINTS):
        score += 4.0
        reasons.append("positive_name_hint")

    if col_name in NEGATIVE_NAME_HINTS or bool(tokens & NEGATIVE_NAME_HINTS):
        score -= 5.0
        reasons.append("negative_name_hint")

    if dtype in {"bool"}:
        score += 4.0
        reasons.append("boolean_dtype")

    if dtype in {"object", "category"} and 2 <= unique_values <= 10:
        score += 3.0
        reasons.append("categorical_small_cardinality")

    if "int" in dtype and 2 <= unique_values <= 10:
        score += 2.0
        reasons.append("integer_small_cardinality")

    if unique_values <= 1:
        score -= 5.0
        reasons.append("constant_column")

    if unique_ratio > 0.9:
        score -= 4.0
        reasons.append("high_unique_ratio")

    if series.isna().mean() > 0.5:
        score -= 2.0
        reasons.append("high_missing_rate")

    return TargetCandidate(
        column=str(column),
        score=float(score),
        reasons=reasons,
        dtype=dtype,
        unique_values=unique_values,
        unique_ratio=unique_ratio,
    )


def _confidence_from_score(score: float) -> str:
    if score >= 5.0:
        return "high"
    if score >= 3.0:
        return "medium"
    if score >= 1.5:
        return "low"
    return "none"


def _is_classification_compatible(candidate: TargetCandidate) -> bool:
    if candidate.dtype in {"bool", "object", "category"} and 2 <= candidate.unique_values <= 20:
        return True
    if "int" in candidate.dtype and 2 <= candidate.unique_values <= 20:
        return True
    return False

def _name_tokens(name: str) -> set[str]:
    normalized = _normalized_name(name)
    return {token for token in normalized.split("_") if token}


def infer_target_column(dataset_path: str | Path) -> TargetInferenceResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    candidates = [_score_column_as_target(df, str(col)) for col in df.columns]
    candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)

    if not candidates_sorted:
        return TargetInferenceResult(
            dataset_path=str(path),
            selected_target=None,
            confidence="none",
            score=0.0,
            candidates=[],
            reasoning=["no_columns_available"],
            classification_compatible=False,
        )

    best = candidates_sorted[0]
    confidence = _confidence_from_score(best.score)
    compatible = _is_classification_compatible(best)

    reasoning: List[str] = []
    selected_target: Optional[str] = None

    if confidence == "none":
        reasoning.append("no_candidate_reached_minimum_score")
    elif not compatible:
        reasoning.append("best_candidate_not_classification_compatible")
    else:
        selected_target = best.column
        reasoning.append("selected_highest_scoring_candidate")

    return TargetInferenceResult(
        dataset_path=str(path),
        selected_target=selected_target,
        confidence=confidence,
        score=best.score,
        candidates=[c.to_dict() for c in candidates_sorted],
        reasoning=reasoning,
        classification_compatible=compatible,
    )