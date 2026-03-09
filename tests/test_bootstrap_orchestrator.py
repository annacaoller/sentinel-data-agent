from __future__ import annotations

from pathlib import Path

from sentinel.orchestrator.bootstrap import BootstrapOrchestrator
from sentinel.state.store import JsonRunStore, SentinelPaths


def test_bootstrap_orchestrator_full_run(tmp_path: Path) -> None:
    paths = SentinelPaths(root=tmp_path)
    store = JsonRunStore(paths=paths)
    orchestrator = BootstrapOrchestrator(store)

    result = orchestrator.execute("examples/example.csv", "auto")

    assert result.run_id
    assert result.state["status"] == "report_completed"
    assert result.state["current_phase"] == "final_report"
    assert result.state["progress_pct"] == 100

    run_dir = paths.run_dir(result.run_id)

    assert (run_dir / "dataset_profile.json").exists()
    assert (run_dir / "dataset_quality.json").exists()
    assert (run_dir / "target_inference.json").exists()
    assert (run_dir / "schema_analysis.json").exists()
    assert (run_dir / "execution_plan.json").exists()
    assert (run_dir / "eda_summary.json").exists()
    assert (run_dir / "correlation_analysis.json").exists()
    assert (run_dir / "feature_readiness.json").exists()
    assert (run_dir / "baseline_classification.json").exists()
    assert (run_dir / "final_report.json").exists()

    assert (run_dir / "checkpoints" / "bootstrap_complete.json").exists()

    final_report = store._read_json(run_dir / "final_report.json")
    assert final_report["run_id"] == result.run_id
    assert final_report["status"] == "report_completed"