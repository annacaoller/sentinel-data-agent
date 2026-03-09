from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sentinel.capabilities.baseline_classification import run_baseline_classification
from sentinel.capabilities.correlation_analysis import analyze_correlations
from sentinel.capabilities.eda_summary import generate_eda_summary
from sentinel.capabilities.feature_readiness import assess_feature_readiness
from sentinel.capabilities.final_report import build_final_report
from sentinel.capabilities.ingestion import profile_dataset
from sentinel.capabilities.quality import assess_dataset_quality
from sentinel.capabilities.schema_analysis import analyze_schema
from sentinel.capabilities.target_inference import infer_target_column
from sentinel.planner.dag_planner import DagPlanner
from sentinel.state.store import JsonRunStore


@dataclass(frozen=True)
class BootstrapResult:
    run_id: str
    state: Dict[str, Any]
    profile: Dict[str, Any]
    quality: Dict[str, Any]
    target_inference: Dict[str, Any]
    schema: Dict[str, Any]
    execution_plan: Dict[str, Any]
    eda_summary: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    feature_readiness: Dict[str, Any]
    baseline_classification: Dict[str, Any]
    final_report: Dict[str, Any]
    profile_path: str
    quality_path: str
    target_inference_path: str
    schema_path: str
    execution_plan_path: str
    eda_summary_path: str
    correlation_analysis_path: str
    feature_readiness_path: str
    baseline_classification_path: str
    final_report_path: str
    checkpoint_path: str


class BootstrapOrchestrator:
    """
    Executes the deterministic bootstrap sequence for a Sentinel run.

    Current phases:
    1. create run state
    2. dataset profiling
    3. dataset quality assessment
    4. deterministic target inference
    5. schema analysis
    6. deterministic plan generation
    7. execute eda_summary
    8. execute correlation_analysis
    9. execute feature_readiness
    10. execute baseline_classification (conditional)
    11. build final_report
    12. update run state with summaries
    13. persist checkpoint
    """

    def __init__(self, store: JsonRunStore) -> None:
        self.store = store
        self.planner = DagPlanner()

    def execute(self, dataset: str, objective: str) -> BootstrapResult:
        state = self.store.create_run_state(dataset_path=dataset, objective=objective)
        run_id = state["run_id"]

        try:
            profile = profile_dataset(dataset)
            profile_path = self.store.write_dataset_profile(run_id, profile)

            quality = assess_dataset_quality(dataset).to_dict()
            quality_path = self.store.write_dataset_quality(run_id, quality)

            target_inference = infer_target_column(dataset).to_dict()
            target_inference_path = self.store.write_target_inference(run_id, target_inference)

            schema = analyze_schema(dataset).to_dict()
            schema_path = self.store.write_schema_analysis(run_id, schema)

            enriched_state = self.store.update_run_state(
                run_id,
                {
                    "status": "schema_analyzed",
                    "current_phase": "schema_analysis",
                    "progress_pct": 40,
                    "dataset_path": profile["dataset_path"],
                    "dataset_summary": {
                        "file_type": profile["file_type"],
                        "num_rows": profile["num_rows"],
                        "num_columns": profile["num_columns"],
                    },
                    "quality_summary": {
                        "duplicate_rows": quality["duplicate_rows"],
                        "duplicate_rate": quality["duplicate_rate"],
                        "constant_columns": quality["constant_columns"],
                        "high_missing_columns": quality["high_missing_columns"],
                        "empty_columns": quality["empty_columns"],
                    },
                    "target_summary": {
                        "selected_target": target_inference["selected_target"],
                        "confidence": target_inference["confidence"],
                        "score": target_inference["score"],
                        "classification_compatible": target_inference["classification_compatible"],
                    },
                    "schema_summary": {
                        "numeric_columns": schema["numeric_columns"],
                        "categorical_columns": schema["categorical_columns"],
                        "boolean_columns": schema["boolean_columns"],
                        "datetime_columns": schema["datetime_columns"],
                        "text_like_columns": schema["text_like_columns"],
                        "likely_identifier_columns": schema["likely_identifier_columns"],
                    },
                    "last_error": None,
                },
            )

            execution_plan = self.planner.build_plan(enriched_state)
            execution_plan_dict = execution_plan.to_dict()
            execution_plan_path = self.store.write_execution_plan(run_id, execution_plan_dict)

            eda_summary = generate_eda_summary(dataset).to_dict()
            eda_summary_path = self.store.write_eda_summary(run_id, eda_summary)

            correlation_analysis = analyze_correlations(dataset).to_dict()
            correlation_analysis_path = self.store.write_correlation_analysis(run_id, correlation_analysis)

            selected_target = target_inference["selected_target"]
            feature_readiness = assess_feature_readiness(dataset, selected_target).to_dict()
            feature_readiness_path = self.store.write_feature_readiness(run_id, feature_readiness)

            baseline_classification = run_baseline_classification(dataset, selected_target).to_dict()
            baseline_classification_path = self.store.write_baseline_classification(run_id, baseline_classification)

            pre_report_state = self.store.update_run_state(
                run_id,
                {
                    "status": "baseline_completed",
                    "current_phase": "baseline_classification",
                    "progress_pct": 90,
                    "plan_summary": {
                        "num_nodes": len(execution_plan.nodes),
                        "enabled_nodes": sum(1 for n in execution_plan.nodes if n["enabled"]),
                        "selected_target": execution_plan.selected_target,
                        "target_confidence": execution_plan.target_confidence,
                    },
                    "eda_summary_overview": {
                        "numeric_columns_analyzed": list(eda_summary["numeric_summary"].keys()),
                        "categorical_columns_analyzed": list(eda_summary["categorical_summary"].keys()),
                    },
                    "correlation_summary": {
                        "num_numeric_columns": correlation_analysis["num_numeric_columns"],
                        "correlation_method": correlation_analysis["correlation_method"],
                        "strongest_pair": correlation_analysis["strongest_pair"],
                    },
                    "feature_readiness_summary": {
                        "selected_target": feature_readiness["selected_target"],
                        "target_present": feature_readiness["target_present"],
                        "target_dtype": feature_readiness["target_dtype"],
                        "candidate_feature_columns": feature_readiness["candidate_feature_columns"],
                        "numeric_feature_columns": feature_readiness["numeric_feature_columns"],
                        "categorical_feature_columns": feature_readiness["categorical_feature_columns"],
                        "blocked_reasons": feature_readiness["blocked_reasons"],
                        "modeling_ready": feature_readiness["modeling_ready"],
                    },
                    "baseline_summary": {
                        "modeling_executed": baseline_classification["modeling_executed"],
                        "blocked_reasons": baseline_classification["blocked_reasons"],
                        "num_rows_used": baseline_classification["num_rows_used"],
                        "num_features_used": baseline_classification["num_features_used"],
                        "train_size": baseline_classification["train_size"],
                        "test_size": baseline_classification["test_size"],
                        "metrics": baseline_classification["metrics"],
                        "class_labels": baseline_classification["class_labels"],
                    },
                },
            )

            initial_final_report = build_final_report(pre_report_state).to_dict()

            updated_state = self.store.update_run_state(
                run_id,
                {
                    "status": "report_completed",
                    "current_phase": "final_report",
                    "progress_pct": 100,
                    "final_report_summary": {
                        "target_selected": initial_final_report["overall_assessment"]["target_selected"],
                        "target_confidence": initial_final_report["overall_assessment"]["target_confidence"],
                        "modeling_ready": initial_final_report["overall_assessment"]["modeling_ready"],
                        "modeling_executed": initial_final_report["overall_assessment"]["modeling_executed"],
                        "blocked_reasons": initial_final_report["overall_assessment"]["blocked_reasons"],
                    },
                },
            )

            final_report = build_final_report(updated_state).to_dict()
            final_report_path = self.store.write_final_report(run_id, final_report)

            checkpoint_path = self.store.write_checkpoint(
                run_id,
                "bootstrap_complete",
                {
                    "state": updated_state,
                    "profile": profile,
                    "quality": quality,
                    "target_inference": target_inference,
                    "schema": schema,
                    "execution_plan": execution_plan_dict,
                    "eda_summary": eda_summary,
                    "correlation_analysis": correlation_analysis,
                    "feature_readiness": feature_readiness,
                    "baseline_classification": baseline_classification,
                    "final_report": final_report,
                },
            )

            return BootstrapResult(
                run_id=run_id,
                state=updated_state,
                profile=profile,
                quality=quality,
                target_inference=target_inference,
                schema=schema,
                execution_plan=execution_plan_dict,
                eda_summary=eda_summary,
                correlation_analysis=correlation_analysis,
                feature_readiness=feature_readiness,
                baseline_classification=baseline_classification,
                final_report=final_report,
                profile_path=str(profile_path),
                quality_path=str(quality_path),
                target_inference_path=str(target_inference_path),
                schema_path=str(schema_path),
                execution_plan_path=str(execution_plan_path),
                eda_summary_path=str(eda_summary_path),
                correlation_analysis_path=str(correlation_analysis_path),
                feature_readiness_path=str(feature_readiness_path),
                baseline_classification_path=str(baseline_classification_path),
                final_report_path=str(final_report_path),
                checkpoint_path=str(checkpoint_path),
            )

        except Exception as exc:
            failed_state = self.store.update_run_state(
                run_id,
                {
                    "status": "failed",
                    "current_phase": "final_report",
                    "progress_pct": 0,
                    "last_error": str(exc),
                },
            )

            self.store.write_checkpoint(
                run_id,
                "bootstrap_failed",
                {
                    "state": failed_state,
                    "error": str(exc),
                },
            )

            raise RuntimeError(f"Bootstrap failed for run_id={run_id}: {exc}") from exc