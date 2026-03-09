from __future__ import annotations

from pathlib import Path

from sentinel.state.store import JsonRunStore, SentinelPaths


def test_update_run_state(tmp_path: Path) -> None:
    paths = SentinelPaths(root=tmp_path)
    store = JsonRunStore(paths=paths)

    state = store.create_run_state(
        dataset_path="examples/example.csv",
        objective="auto",
    )

    run_id = state["run_id"]

    updated_state = store.update_run_state(
        run_id,
        {
            "status": "schema_analyzed",
            "current_phase": "schema_analysis",
            "progress_pct": 40,
        },
    )

    assert updated_state["status"] == "schema_analyzed"
    assert updated_state["current_phase"] == "schema_analysis"
    assert updated_state["progress_pct"] == 40

    reloaded_state = store.read_run_state(run_id)

    assert reloaded_state["status"] == "schema_analyzed"
    assert reloaded_state["current_phase"] == "schema_analysis"
    assert reloaded_state["progress_pct"] == 40