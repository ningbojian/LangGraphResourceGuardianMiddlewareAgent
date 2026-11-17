"""Policy proposals and a deterministic rule-based proposer.

The rule proposer is the offline optimizer agent's default "brain" and a
fallback when no LLM key is configured. It only proposes; the Pareto gate and a
human decide whether a proposal is adopted.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from resguard.batch import BatchInsights
from resguard.policy import GuardPolicy


@dataclass
class ParameterChange:
    path: str
    old: Any
    new: Any
    rationale: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterChange":
        return cls(
            path=data["path"],
            old=data.get("old"),
            new=data.get("new"),
            rationale=data.get("rationale", ""),
        )


@dataclass
class WorkflowChange:
    kind: str
    target: str
    config: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowChange":
        return cls(
            kind=data["kind"],
            target=data.get("target", ""),
            config=dict(data.get("config") or {}),
            rationale=data.get("rationale", ""),
        )


@dataclass
class PolicyProposal:
    name: str
    description: str
    parameter_changes: list[ParameterChange] = field(default_factory=list)
    workflow_changes: list[WorkflowChange] = field(default_factory=list)
    expected_effect: dict[str, str] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    risk: str = "low"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyProposal":
        return cls(
            id=data.get("id") or uuid.uuid4().hex,
            name=data["name"],
            description=data.get("description", ""),
            parameter_changes=[
                ParameterChange.from_dict(item) for item in data.get("parameter_changes", [])
            ],
            workflow_changes=[
                WorkflowChange.from_dict(item) for item in data.get("workflow_changes", [])
            ],
            expected_effect=dict(data.get("expected_effect") or {}),
            evidence=list(data.get("evidence") or []),
            risk=data.get("risk", "low"),
        )


class RuleProposer:
    """Deterministic proposals from batch hotspots."""

    def propose(
        self,
        insights: BatchInsights,
        policy: GuardPolicy | None = None,
    ) -> list[PolicyProposal]:
        proposals: list[PolicyProposal] = []
        hotspots = set(insights.hotspots)
        evidence = [f"session_count={insights.session_count}"] + sorted(hotspots)

        if {"db_cu_exhausted", "db_queries_exhausted"} & hotspots or "duplicate_tool_calls" in hotspots:
            proposals.append(
                PolicyProposal(
                    name="cache_hot_db_reads",
                    description="Cache idempotent DB reads for hot query types and lower DB CU pressure.",
                    parameter_changes=[
                        ParameterChange(
                            path="profiles.default.cache_ttl_seconds",
                            old=None,
                            new=300,
                            rationale="reuse repeated DB reads",
                        )
                    ],
                    workflow_changes=[
                        WorkflowChange(
                            kind="add_cache_node",
                            target="db_tools",
                            config={"ttl_seconds": 300},
                            rationale="avoid duplicate DB queries/CU",
                        )
                    ],
                    expected_effect={"db_cu": "down", "db_queries": "down", "usd": "down"},
                    evidence=evidence,
                    risk="low",
                )
            )

        if "usd_exhausted" in hotspots:
            proposals.append(
                PolicyProposal(
                    name="route_cheaper_model",
                    description="Route high-volume query types to a cheaper model tier.",
                    workflow_changes=[
                        WorkflowChange(
                            kind="model_routing",
                            target="default",
                            config={"model_tier": "cheap"},
                            rationale="reduce LLM USD while preserving success",
                        )
                    ],
                    expected_effect={"usd": "down"},
                    evidence=evidence,
                    risk="medium",
                )
            )

        if "concurrency" in hotspots:
            proposals.append(
                PolicyProposal(
                    name="cap_parallel_db_calls",
                    description="Cap parallel DB tool calls to the connection-pool limit.",
                    parameter_changes=[
                        ParameterChange(
                            path="limits.concurrency",
                            old=None,
                            new=2,
                            rationale="avoid connection-pool contention",
                        )
                    ],
                    expected_effect={"concurrency": "down"},
                    evidence=evidence,
                    risk="low",
                )
            )

        return proposals
