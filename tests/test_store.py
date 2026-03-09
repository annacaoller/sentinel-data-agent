from __future__ import annotations

from pathlib import Path

from sentinel.state.store import JsonRunStore, SentinelPaths


def test_create_and_read_run_state(tmp_path: Path) -> None:
    paths = SentinelPaths(root=tmp_path)
    store = JsonRunStore(paths=paths)

    state = store.create_run_state(
        dataset_path="examples/example.csv",
        objective="auto",
    )

    run_id = state["run_id"]
    run_state_path = paths.run_state_path(run_id)

    assert run_state_path.exists()

    loaded_state = store.read_run_state(run_id)

    assert loaded_state["run_id"] == run_id
    assert loaded_state["dataset_path"] == "examples/example.csv"
    assert loaded_state["objective"] == "auto"
    assert loaded_state["status"] == "initialized"
    assert loaded_state["current_phase"] == "bootstrap"
    assert loaded_state["progress_pct"] == 0
    assert loaded_state["last_error"] is None
    assert isinstance(loaded_state["notes"], list)
    assert len(loaded_state["notes"]) > 0