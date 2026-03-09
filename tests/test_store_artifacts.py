from __future__ import annotations

import json
from pathlib import Path

from sentinel.state.store import JsonRunStore, SentinelPaths


def test_write_artifacts(tmp_path: Path) -> None:
    paths = SentinelPaths(root=tmp_path)
    store = JsonRunStore(paths=paths)

    state = store.create_run_state(
        dataset_path="examples/example.csv",
        objective="auto",
    )

    run_id = state["run_id"]

    profile = {
        "dataset_path": "examples/example.csv",
        "file_type": "csv",
        "num_rows": 4,
        "num_columns": 3,
    }

    quality = {
        "dataset_path": "examples/example.csv",
        "num_rows": 4,
        "num_columns": 3,
        "duplicate_rows": 0,
        "duplicate_rate": 0.0,
    }

    profile_path = store.write_dataset_profile(run_id, profile)
    quality_path = store.write_dataset_quality(run_id, quality)

    assert profile_path.exists()
    assert quality_path.exists()

    with open(profile_path, "r", encoding="utf-8") as f:
        stored_profile = json.load(f)

    with open(quality_path, "r", encoding="utf-8") as f:
        stored_quality = json.load(f)

    assert stored_profile["dataset_path"] == "examples/example.csv"
    assert stored_profile["num_rows"] == 4

    assert stored_quality["duplicate_rows"] == 0
    assert stored_quality["duplicate_rate"] == 0.0