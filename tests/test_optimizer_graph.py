from __future__ import annotations

import pytest
from langgraph.types import Command

from resguard.metrics import SessionMetrics
from resguard.optimizer_agent import OptimizerAgent, ScriptedBrain, SyntheticEvaluator
from resguard.proposer import PolicyProposal, WorkflowChange
from resguard.session import JsonlSessionStore, SessionRecord


def record(session_id: str, *, success: bool = True) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        query_type="refund",
        metrics=SessionMetrics(tool_calls=2, db_cu=2.0, llm_usd=0.01),
        success=success,
        label_source="human",
    )


def cache_proposal(proposal_id: str = "p-cache") -> PolicyProposal:
    return PolicyProposal(
        id=proposal_id,
        name="cache-db",
        description="Cache hot DB reads",
        workflow_changes=[
            WorkflowChange(
                kind="add_cache_node",
                target="db_tools",
                config={"ttl_seconds": 300},
                rationale="avoid duplicate DB reads",
            )
        ],
        expected_effect={"db_cu": "down"},
        evidence=["db_cu_exhausted"],
        risk="low",
    )


def noop_proposal(proposal_id: str = "p-noop") -> PolicyProposal:
    return PolicyProposal(
        id=proposal_id,
        name="noop",
        description="No recognized resource change",
        evidence=["manual"],
        risk="low",
    )


def make_agent(tmp_path, proposals, *, require_approval: bool = False) -> OptimizerAgent:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    store.append(record("s1"))
    store.append(record("s2"))
    return OptimizerAgent(
        store=store,
        brain=ScriptedBrain(focus="db_cu", proposals=proposals),
        evaluator=SyntheticEvaluator(),
        require_approval=require_approval,
    )


def test_graph_runs_to_accepted(tmp_path) -> None:
    agent = make_agent(tmp_path, [cache_proposal()])

    result = agent.graph.invoke(
        {"batch_id": "b1"}, config={"configurable": {"thread_id": "t1"}}
    )

    assert result["termination_reason"] == "accepted"
    assert result["accepted_proposal_id"] == "p-cache"
    assert result["approved_proposal_id"] == "p-cache"
    assert len(result["evaluations"]) == 1
    assert result["focus"] == "db_cu"
    assert result["insights"]["session_count"] == 2


def test_graph_streams_node_updates(tmp_path) -> None:
    agent = make_agent(tmp_path, [cache_proposal()])

    updates = list(
        agent.graph.stream(
            {"batch_id": "b2"},
            config={"configurable": {"thread_id": "t2"}},
            stream_mode="updates",
        )
    )

    node_names = {name for update in updates for name in update}
    assert {
        "observe",
        "diagnose",
        "hypothesize",
        "plan_experiment",
        "execute_experiment",
        "evaluate",
        "reflect",
        "finalize",
    } <= node_names


def test_graph_no_proposals(tmp_path) -> None:
    agent = make_agent(tmp_path, [])

    result = agent.graph.invoke(
        {"batch_id": "b3"}, config={"configurable": {"thread_id": "t3"}}
    )

    assert result["termination_reason"] == "no_proposals"
    assert result["evaluations"] == []


def test_graph_loops_rejected_then_accepted(tmp_path) -> None:
    agent = make_agent(tmp_path, [noop_proposal(), cache_proposal()])

    result = agent.graph.invoke(
        {"batch_id": "b4"}, config={"configurable": {"thread_id": "t4"}}
    )

    assert result["termination_reason"] == "accepted"
    assert result["accepted_proposal_id"] == "p-cache"
    assert len(result["evaluations"]) == 2
    assert result["evaluations"][0]["accepted"] is False
    assert result["evaluations"][1]["accepted"] is True


def test_graph_interrupt_and_resume(tmp_path) -> None:
    agent = make_agent(tmp_path, [cache_proposal()], require_approval=True)

    paused = agent.graph.invoke(
        {"batch_id": "b5"}, config={"configurable": {"thread_id": "t5"}}
    )

    assert "__interrupt__" in paused
    assert paused["accepted_proposal_id"] == "p-cache"

    resumed = agent.graph.invoke(
        Command(resume="approve"),
        config={"configurable": {"thread_id": "t5"}},
    )

    assert resumed["termination_reason"] == "approved"
    assert resumed["approved_proposal_id"] == "p-cache"


def test_graph_interrupt_reject(tmp_path) -> None:
    agent = make_agent(tmp_path, [cache_proposal()], require_approval=True)
    agent.graph.invoke({"batch_id": "b6"}, config={"configurable": {"thread_id": "t6"}})

    resumed = agent.graph.invoke(
        Command(resume="reject"),
        config={"configurable": {"thread_id": "t6"}},
    )

    assert resumed["termination_reason"] == "rejected"
    assert resumed.get("approved_proposal_id") is None


def test_synthetic_evaluator_reduces_cached_resources() -> None:
    from resguard.batch import BatchInsights

    baseline = BatchInsights(
        session_count=1,
        success_rate=1.0,
        labeled_sessions=1,
        resource_per_success={"usd": 0.01, "db_cu": 2.0, "db_queries": 10.0, "api_quota": 1.0, "tool_calls": 2.0},
    )

    candidate = SyntheticEvaluator().evaluate(cache_proposal(), baseline)

    assert candidate.resource_per_success["db_cu"] == pytest.approx(1.6)
    assert candidate.success_rate == 1.0
