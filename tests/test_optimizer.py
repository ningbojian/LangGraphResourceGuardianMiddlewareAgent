from __future__ import annotations

import json

import pytest

from resguard.batch import BatchInsights
from resguard.optimizer import AuditLog, ParetoGate
from resguard.proposer import (
    ParameterChange,
    PolicyProposal,
    RuleProposer,
    WorkflowChange,
)


def insights(
    *,
    usd: float = 0.10,
    db_cu: float = 2.0,
    success_rate: float = 1.0,
    hotspots: list[str] | None = None,
) -> BatchInsights:
    return BatchInsights(
        session_count=10,
        success_rate=success_rate,
        labeled_sessions=10,
        resource_per_success={
            "usd": usd,
            "db_cu": db_cu,
            "db_queries": 10.0,
            "api_quota": 1.0,
            "tool_calls": 5.0,
        },
        hotspots=hotspots or [],
    )


def test_policy_proposal_round_trip() -> None:
    proposal = PolicyProposal(
        id="p1",
        name="cache-db",
        description="Cache hot DB reads",
        parameter_changes=[
            ParameterChange(
                path="profiles.refund.limits.db_cu",
                old=1.0,
                new=0.8,
                rationale="lower DB CU budget",
            )
        ],
        workflow_changes=[
            WorkflowChange(
                kind="add_cache_node",
                target="search_docs",
                config={"ttl_seconds": 300},
                rationale="avoid repeated DB reads",
            )
        ],
        expected_effect={"db_cu": "down"},
        evidence=["s1", "s2"],
        risk="low",
    )

    restored = PolicyProposal.from_dict(proposal.to_dict())

    assert restored == proposal


def test_rule_proposer_for_db_hotspot() -> None:
    proposals = RuleProposer().propose(
        insights(hotspots=["db_cu_exhausted"]),
        policy=None,
    )

    assert proposals
    assert proposals[0].id
    assert proposals[0].parameter_changes or proposals[0].workflow_changes
    assert proposals[0].evidence


def test_rule_proposer_for_duplicate_calls() -> None:
    proposals = RuleProposer().propose(
        insights(hotspots=["duplicate_tool_calls"]),
        policy=None,
    )

    assert any(
        any(change.kind == "add_cache_node" for change in proposal.workflow_changes)
        for proposal in proposals
    )


def test_rule_proposer_without_hotspots_returns_empty() -> None:
    assert RuleProposer().propose(insights(), policy=None) == []


def test_gate_accepts_cost_reduction_with_same_success() -> None:
    result = ParetoGate().evaluate(
        "p1", baseline=insights(usd=0.10, db_cu=2.0), candidate=insights(usd=0.08, db_cu=1.9)
    )

    assert result.accepted is True
    assert result.reasons == []
    assert result.improvements["usd"] == pytest.approx(0.2)


def test_gate_rejects_success_drop() -> None:
    result = ParetoGate().evaluate(
        "p1",
        baseline=insights(success_rate=1.0),
        candidate=insights(usd=0.08, success_rate=0.9),
    )

    assert result.accepted is False
    assert any("success_rate" in reason for reason in result.reasons)


def test_gate_rejects_no_resource_improvement() -> None:
    result = ParetoGate().evaluate(
        "p1",
        baseline=insights(usd=0.10, db_cu=2.0),
        candidate=insights(usd=0.099, db_cu=1.99),
    )

    assert result.accepted is False
    assert any("no_resource" in reason for reason in result.reasons)


def test_gate_rejects_regression_in_another_dimension() -> None:
    result = ParetoGate().evaluate(
        "p1",
        baseline=insights(usd=0.10, db_cu=2.0),
        candidate=insights(usd=0.08, db_cu=2.5),
    )

    assert result.accepted is False
    assert any("regression" in reason for reason in result.reasons)


def test_audit_log_append(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append({"proposal_id": "p1", "accepted": True})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["proposal_id"] == "p1"
