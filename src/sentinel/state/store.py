from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SentinelPaths:
    """
    Centralizes filesystem layout conventions for Sentinel runs.

    Root is either:
    - SENTINEL_PROJECT_ROOT env var, or
    - current working directory (repo root when developing).
    """

    root: Path

    @staticmethod
    def from_env_or_cwd() -> "SentinelPaths":
        env_root = os.getenv("SENTINEL_PROJECT_ROOT")
        root = Path(env_root).expanduser().resolve() if env_root else Path.cwd().resolve()
        return SentinelPaths(root=root)

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def run_dir(self, run_id: str) -> Path:
        return self.artifacts_dir / run_id

    def run_state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run_state.json"

    def checkpoints_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoints"

    def report_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "report"

    def ensure_base_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


class JsonRunStore:
    """
    Minimal deterministic run store using JSON files under artifacts/<run_id>/run_state.json.

    This is intentionally simple and auditable.
    Later we can swap to SQLite behind the same interface.
    """

    def __init__(self, paths: Optional[SentinelPaths] = None) -> None:
        self.paths = paths or SentinelPaths.from_env_or_cwd()
        self.paths.ensure_base_dirs()

    def generate_run_id(self) -> str:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%d_%H%M%S")
        mmm = f"{int(now.microsecond / 1000):03d}"
        return f"sda_{stamp}_{mmm}_{os.getpid()}"

    def create_run_state(
        self,
        dataset_path: str,
        objective: str,
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = run_id or self.generate_run_id()
        run_dir = self.paths.run_dir(rid)
        run_dir.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "run_id": rid,
            "status": "initialized",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "dataset_path": dataset_path,
            "objective": objective,
            "current_phase": "bootstrap",
            "progress_pct": 0,
            "last_error": None,
            "notes": [
                "Run initialized by store. Orchestrator will populate DAG, observations, checkpoints, and artifacts."
            ],
        }

        self.write_run_state(rid, state)
        return state

    def write_run_state(self, run_id: str, state: Dict[str, Any]) -> None:
        state = dict(state)
        state["run_id"] = run_id
        state["updated_at"] = utc_now_iso()

        path = self.paths.run_state_path(run_id)
        self._atomic_write_json(path, state)

    def read_run_state(self, run_id: str) -> Dict[str, Any]:
        path = self.paths.run_state_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return self._read_json(path)

    def update_run_state(self, run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        state = self.read_run_state(run_id)
        state.update(patch)
        self.write_run_state(run_id, state)
        return state

    def list_runs(self, limit: int = 20) -> Iterable[Dict[str, Any]]:
        runs_root = self.paths.artifacts_dir
        if not runs_root.exists():
            return []

        run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        run_dirs.sort(key=lambda p: p.name, reverse=True)
        run_dirs = run_dirs[: max(1, limit)]

        results: list[Dict[str, Any]] = []
        for d in run_dirs:
            state_path = d / "run_state.json"
            if not state_path.exists():
                results.append({"run_id": d.name, "status": "unknown"})
                continue
            results.append(self._read_json(state_path))

        return results

    def expected_report_paths(self, run_id: str) -> Dict[str, Path]:
        run_dir = self.paths.run_dir(run_id)
        report_dir = self.paths.report_dir(run_id)
        return {
            "run_dir": run_dir,
            "final_report_json": run_dir / "final_report.json",
            "report_dir": report_dir,
            "report_html": report_dir / "report.html",
            "report_json": report_dir / "report.json",
        }

    def dataset_profile_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "dataset_profile.json"

    def write_dataset_profile(self, run_id: str, profile: Dict[str, Any]) -> Path:
        path = self.dataset_profile_path(run_id)
        self._atomic_write_json(path, profile)
        return path

    def dataset_quality_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "dataset_quality.json"

    def write_dataset_quality(self, run_id: str, quality_report: Dict[str, Any]) -> Path:
        path = self.dataset_quality_path(run_id)
        self._atomic_write_json(path, quality_report)
        return path

    def target_inference_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "target_inference.json"

    def write_target_inference(self, run_id: str, target_inference: Dict[str, Any]) -> Path:
        path = self.target_inference_path(run_id)
        self._atomic_write_json(path, target_inference)
        return path

    def checkpoint_path(self, run_id: str, checkpoint_name: str) -> Path:
        safe_name = checkpoint_name.strip().replace(" ", "_").lower()
        return self.paths.checkpoints_dir(run_id) / f"{safe_name}.json"

    def write_checkpoint(self, run_id: str, checkpoint_name: str, payload: Dict[str, Any]) -> Path:
        path = self.checkpoint_path(run_id, checkpoint_name)
        checkpoint_payload = {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "created_at": utc_now_iso(),
            "payload": payload,
        }
        self._atomic_write_json(path, checkpoint_payload)
        return path
        
        
    def schema_analysis_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "schema_analysis.json"

    def write_schema_analysis(self, run_id: str, schema: Dict[str, Any]) -> Path:
        path = self.schema_analysis_path(run_id)
        self._atomic_write_json(path, schema)
        return path
    
    def execution_plan_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "execution_plan.json"

    def write_execution_plan(self, run_id: str, plan: Dict[str, Any]) -> Path:
        path = self.execution_plan_path(run_id)
        self._atomic_write_json(path, plan)
        return path
    
    def eda_summary_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "eda_summary.json"

    def write_eda_summary(self, run_id: str, eda_summary: Dict[str, Any]) -> Path:
        path = self.eda_summary_path(run_id)
        self._atomic_write_json(path, eda_summary)
        return path
    
    def correlation_analysis_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "correlation_analysis.json"

    def write_correlation_analysis(self, run_id: str, correlation_analysis: Dict[str, Any]) -> Path:
        path = self.correlation_analysis_path(run_id)
        self._atomic_write_json(path, correlation_analysis)
        return path
    
    def feature_readiness_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "feature_readiness.json"

    def write_feature_readiness(self, run_id: str, feature_readiness: Dict[str, Any]) -> Path:
        path = self.feature_readiness_path(run_id)
        self._atomic_write_json(path, feature_readiness)
        return path
    
    def baseline_classification_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "baseline_classification.json"

    def write_baseline_classification(self, run_id: str, baseline_classification: Dict[str, Any]) -> Path:
        path = self.baseline_classification_path(run_id)
        self._atomic_write_json(path, baseline_classification)
        return path
    
    def final_report_path(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id) / "final_report.json"

    def write_final_report(self, run_id: str, final_report: Dict[str, Any]) -> Path:
        path = self.final_report_path(run_id)
        self._atomic_write_json(path, final_report)
        return path

    def latest_checkpoint_path(self, run_id: str) -> Optional[Path]:
        cdir = self.paths.checkpoints_dir(run_id)
        if not cdir.exists():
            return None
        checkpoints = sorted(cdir.glob("*.json"), key=lambda p: p.name, reverse=True)
        return checkpoints[0] if checkpoints else None

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))