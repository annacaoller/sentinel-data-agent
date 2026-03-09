from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FinalReportResult:
    run_id: str
    status: str
    objective: str
    dataset_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    target_summary: Dict[str, Any]
    schema_summary: Dict[str, Any]
    plan_summary: Dict[str, Any]
    eda_summary_overview: Dict[str, Any]
    correlation_summary: Dict[str, Any]
    feature_readiness_summary: Dict[str, Any]
    baseline_summary: Dict[str, Any]
    overall_assessment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "objective": self.objective,
            "dataset_summary": self.dataset_summary,
            "quality_summary": self.quality_summary,
            "target_summary": self.target_summary,
            "schema_summary": self.schema_summary,
            "plan_summary": self.plan_summary,
            "eda_summary_overview": self.eda_summary_overview,
            "correlation_summary": self.correlation_summary,
            "feature_readiness_summary": self.feature_readiness_summary,
            "baseline_summary": self.baseline_summary,
            "overall_assessment": self.overall_assessment,
        }


def build_final_report(run_state: Dict[str, Any]) -> FinalReportResult:
    baseline_summary = run_state.get("baseline_summary", {})
    feature_readiness_summary = run_state.get("feature_readiness_summary", {})
    target_summary = run_state.get("target_summary", {})

    modeling_executed = bool(baseline_summary.get("modeling_executed", False))
    modeling_ready = bool(feature_readiness_summary.get("modeling_ready", False))
    target_selected = target_summary.get("selected_target")

    overall_assessment = {
        "target_selected": target_selected,
        "target_confidence": target_summary.get("confidence"),
        "modeling_ready": modeling_ready,
        "modeling_executed": modeling_executed,
        "blocked_reasons": baseline_summary.get("blocked_reasons", []),
        "summary_text": (
            "Bootstrap, planning, EDA, correlation analysis and feature readiness completed successfully. "
            "Baseline classification was executed."
            if modeling_executed
            else "Bootstrap, planning, EDA, correlation analysis and feature readiness completed successfully. "
                 "Baseline classification was not executed due to blocking conditions."
        ),
    }

    return FinalReportResult(
        run_id=str(run_state["run_id"]),
        status=str(run_state["status"]),
        objective=str(run_state.get("objective", "auto")),
        dataset_summary=dict(run_state.get("dataset_summary", {})),
        quality_summary=dict(run_state.get("quality_summary", {})),
        target_summary=dict(run_state.get("target_summary", {})),
        schema_summary=dict(run_state.get("schema_summary", {})),
        plan_summary=dict(run_state.get("plan_summary", {})),
        eda_summary_overview=dict(run_state.get("eda_summary_overview", {})),
        correlation_summary=dict(run_state.get("correlation_summary", {})),
        feature_readiness_summary=dict(run_state.get("feature_readiness_summary", {})),
        baseline_summary=dict(run_state.get("baseline_summary", {})),
        overall_assessment=overall_assessment,
    )