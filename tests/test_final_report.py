from __future__ import annotations

from sentinel.capabilities.final_report import build_final_report


def test_final_report_basic() -> None:
    run_state = {
        "run_id": "test_run_001",
        "status": "report_completed",
        "objective": "auto",
        "dataset_summary": {
            "file_type": "csv",
            "num_rows": 4,
            "num_columns": 3,
        },
        "quality_summary": {
            "duplicate_rows": 0,
            "duplicate_rate": 0.0,
            "constant_columns": [],
            "high_missing_columns": [],
            "empty_columns": [],
        },
        "target_summary": {
            "selected_target": "category",
            "confidence": "medium",
            "score": 3.0,
            "classification_compatible": True,
        },
        "schema_summary": {
            "numeric_columns": ["id", "value"],
            "categorical_columns": ["category"],
            "boolean_columns": [],
            "datetime_columns": [],
            "text_like_columns": [],
            "likely_identifier_columns": ["id"],
        },
        "plan_summary": {
            "num_nodes": 5,
            "enabled_nodes": 5,
            "selected_target": "category",
            "target_confidence": "medium",
        },
        "eda_summary_overview": {
            "numeric_columns_analyzed": ["id", "value"],
            "categorical_columns_analyzed": ["category"],
        },
        "correlation_summary": {
            "num_numeric_columns": 2,
            "correlation_method": "pearson",
            "strongest_pair": {
                "column_a": "id",
                "column_b": "value",
                "correlation": 0.83,
                "absolute_correlation": 0.83,
            },
        },
        "feature_readiness_summary": {
            "selected_target": "category",
            "target_present": True,
            "target_dtype": "object",
            "candidate_feature_columns": ["id", "value"],
            "numeric_feature_columns": ["id", "value"],
            "categorical_feature_columns": [],
            "blocked_reasons": [],
            "modeling_ready": True,
        },
        "baseline_summary": {
            "modeling_executed": False,
            "blocked_reasons": ["least_populated_class_has_less_than_two_samples"],
            "num_rows_used": 4,
            "num_features_used": 2,
            "train_size": 0,
            "test_size": 0,
            "metrics": {},
            "class_labels": ["A", "B", "C"],
        },
    }

    result = build_final_report(run_state).to_dict()

    assert result["run_id"] == "test_run_001"
    assert result["status"] == "report_completed"
    assert result["objective"] == "auto"

    assert result["target_summary"]["selected_target"] == "category"
    assert result["feature_readiness_summary"]["modeling_ready"] is True
    assert result["baseline_summary"]["modeling_executed"] is False

    overall = result["overall_assessment"]
    assert overall["target_selected"] == "category"
    assert overall["target_confidence"] == "medium"
    assert overall["modeling_ready"] is True
    assert overall["modeling_executed"] is False
    assert "least_populated_class_has_less_than_two_samples" in overall["blocked_reasons"]