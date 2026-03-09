from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    capability: str
    depends_on: List[str]
    enabled: bool
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "depends_on": self.depends_on,
            "enabled": self.enabled,
            "params": self.params,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    objective: str
    selected_target: Optional[str]
    target_confidence: str
    nodes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "selected_target": self.selected_target,
            "target_confidence": self.target_confidence,
            "nodes": self.nodes,
        }


class DagPlanner:
    """
    Builds a deterministic execution plan from the bootstrap state.

    For now, the planner instantiates a conservative DAG template.
    Execution of these nodes will be implemented later.
    """

    def build_plan(self, run_state: Dict[str, Any]) -> ExecutionPlan:
        run_id = run_state["run_id"]
        objective = str(run_state.get("objective", "auto"))

        target_summary = run_state.get("target_summary", {})
        selected_target = target_summary.get("selected_target")
        target_confidence = str(target_summary.get("confidence", "none"))
        classification_compatible = bool(target_summary.get("classification_compatible", False))

        nodes: List[PlanNode] = [
            PlanNode(
                node_id="eda_summary",
                capability="eda_summary",
                depends_on=[],
                enabled=True,
                params={},
            ),
            PlanNode(
                node_id="correlation_analysis",
                capability="correlation_analysis",
                depends_on=["eda_summary"],
                enabled=True,
                params={},
            ),
            PlanNode(
                node_id="feature_readiness",
                capability="feature_readiness",
                depends_on=["eda_summary"],
                enabled=True,
                params={"selected_target": selected_target},
            ),
            PlanNode(
                node_id="baseline_classification",
                capability="baseline_classification",
                depends_on=["feature_readiness", "correlation_analysis"],
                enabled=bool(selected_target) and classification_compatible and target_confidence in {"medium", "high"},
                params={
                    "selected_target": selected_target,
                    "target_confidence": target_confidence,
                },
            ),
            PlanNode(
                node_id="final_report",
                capability="final_report",
                depends_on=["eda_summary", "correlation_analysis", "feature_readiness"],
                enabled=True,
                params={"include_modeling": bool(selected_target) and classification_compatible},
            ),
        ]

        return ExecutionPlan(
            run_id=run_id,
            objective=objective,
            selected_target=selected_target,
            target_confidence=target_confidence,
            nodes=[node.to_dict() for node in nodes],
        )