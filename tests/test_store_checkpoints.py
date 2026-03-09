from __future__ import annotations

import json
from pathlib import Path

from sentinel.state.store import JsonRunStore, SentinelPaths


def test_write_checkpoint(tmp_path: Path) -> None:
    paths = SentinelPaths(root=tmp_path)
    store = JsonRunStore(paths=paths)

    state = store.create_run_state(
        dataset_path="examples/example.csv",
        objective="auto",
    )

    run_id = state["run_id"]

    payload = {
        "state": {
            "run_id": run_id,
            "status": "schema_analyzed"
        },
        "note": "checkpoint test"
    }

    checkpoint_path = store.write_checkpoint(
        run_id,
        "test_checkpoint",
        payload,
    )

    assert checkpoint_path.exists()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        stored = json.load(f)

    assert stored["run_id"] == run_id
    assert stored["checkpoint_name"] == "test_checkpoint"
    assert stored["payload"]["note"] == "checkpoint test"