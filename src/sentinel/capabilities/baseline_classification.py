from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sentinel.capabilities.ingestion import load_dataset, validate_dataset_path


@dataclass(frozen=True)
class BaselineClassificationResult:
    dataset_path: str
    selected_target: str
    modeling_executed: bool
    blocked_reasons: List[str]
    num_rows_used: int
    num_features_used: int
    train_size: int
    test_size: int
    metrics: Dict[str, float]
    class_labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "selected_target": self.selected_target,
            "modeling_executed": self.modeling_executed,
            "blocked_reasons": self.blocked_reasons,
            "num_rows_used": self.num_rows_used,
            "num_features_used": self.num_features_used,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics,
            "class_labels": self.class_labels,
        }


def run_baseline_classification(
    dataset_path: str | Path,
    selected_target: Optional[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> BaselineClassificationResult:
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    blocked_reasons: List[str] = []

    if not selected_target:
        blocked_reasons.append("no_target_selected")
    elif selected_target not in df.columns:
        blocked_reasons.append("selected_target_not_found")

    if blocked_reasons:
        return BaselineClassificationResult(
            dataset_path=str(path),
            selected_target=str(selected_target),
            modeling_executed=False,
            blocked_reasons=blocked_reasons,
            num_rows_used=0,
            num_features_used=0,
            train_size=0,
            test_size=0,
            metrics={},
            class_labels=[],
        )

    working_df = df.dropna(subset=[selected_target]).copy()

    if working_df.empty:
        blocked_reasons.append("no_rows_after_target_dropna")

    y = working_df[selected_target]
    X = working_df.drop(columns=[selected_target])

    if X.shape[1] == 0:
        blocked_reasons.append("no_feature_columns_available")

    unique_classes = int(y.nunique(dropna=False))
    if unique_classes < 2:
        blocked_reasons.append("target_has_less_than_two_classes")

    if len(working_df) < 4:
        blocked_reasons.append("dataset_too_small_for_split")

    class_counts = y.astype(str).value_counts()
    if not class_counts.empty and int(class_counts.min()) < 2:
        blocked_reasons.append("least_populated_class_has_less_than_two_samples")

    if blocked_reasons:
        return BaselineClassificationResult(
            dataset_path=str(path),
            selected_target=str(selected_target),
            modeling_executed=False,
            blocked_reasons=blocked_reasons,
            num_rows_used=int(len(working_df)),
            num_features_used=int(X.shape[1]) if "X" in locals() else 0,
            train_size=0,
            test_size=0,
            metrics={},
            class_labels=[str(v) for v in sorted(y.astype(str).unique())] if "y" in locals() else [],
        )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    y_as_str = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_as_str,
        test_size=test_size,
        random_state=random_state,
        stratify=y_as_str,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    return BaselineClassificationResult(
        dataset_path=str(path),
        selected_target=str(selected_target),
        modeling_executed=True,
        blocked_reasons=[],
        num_rows_used=int(len(working_df)),
        num_features_used=int(X.shape[1]),
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        metrics=metrics,
        class_labels=[str(v) for v in sorted(y_as_str.unique())],
    )