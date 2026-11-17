from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from resguard.batch import BatchInsights
from resguard.metrics import SessionMetrics
from resguard.optimizer_agent import LLMBrain, OptimizerAgent, SyntheticEvaluator
from resguard.session import JsonlSessionStore, SessionRecord
from tests.fakes import ScriptedChatModel


def insights() -> BatchInsights:
    return BatchInsights(
        session_count=3,
        success_rate=1.0,
        labeled_sessions=3,
        resource_per_success={
            "usd": 0.01,
            "db_cu": 2.0,
            "db_queries": 10.0,
            "api_quota": 1.0,
            "tool_calls": 2.0,
        },
        hotspots=["db_cu_exhausted"],
    )


def proposal_json() -> str:
    return json.dumps(
        [
            {
                "name": "cache-db",
                "description": "Cache hot DB reads",
                "parameter_changes": [],
                "workflow_changes": [
                    {
                        "kind": "add_cache_node",
                        "target": "db_tools",
                        "config": {"ttl_seconds": 300},
                        "rationale": "avoid repeated DB reads",
                    }
                ],
                "expected_effect": {"db_cu": "down"},
                "evidence": ["db_cu_exhausted"],
                "risk": "low",
            }
        ]
    )


def test_llm_brain_diagnose_and_propose() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="db_cu_exhausted"),
            AIMessage(content=f"```json\n{proposal_json()}\n```"),
        ]
    )
    brain = LLMBrain(model)

    focus = brain.diagnose(insights())
    proposals = brain.propose(insights(), focus)

    assert focus == "db_cu_exhausted"
    assert len(proposals) == 1
    assert proposals[0].name == "cache-db"
    assert proposals[0].workflow_changes[0].kind == "add_cache_node"


def test_llm_brain_invalid_json_falls_back_to_rule_brain() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="not json at all")])
    brain = LLMBrain(model)

    proposals = brain.propose(insights(), "db_cu_exhausted")

    assert proposals
    assert any(
        change.kind == "add_cache_node"
        for proposal in proposals
        for change in proposal.workflow_changes
    )


def test_llm_brain_strict_mode_raises() -> None:
    class RaisingModel:
        def invoke(self, prompt):
            raise RuntimeError("auth failed")

    brain = LLMBrain(RaisingModel(), raise_on_error=True)

    import pytest

    with pytest.raises(RuntimeError):
        brain.diagnose(insights())


def test_llm_brain_runs_inside_optimizer_graph(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    for index in range(3):
        store.append(
            SessionRecord(
                session_id=f"s{index}",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:01+00:00",
                query_type="refund",
                metrics=SessionMetrics(tool_calls=2, db_cu=2.0, llm_usd=0.01),
                success=True,
                label_source="human",
                termination_reason="db_cu_exhausted",
            )
        )
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="db_cu_exhausted"),
            AIMessage(content=proposal_json()),
        ]
    )
    agent = OptimizerAgent(store=store, brain=LLMBrain(model), evaluator=SyntheticEvaluator())

    result = agent.graph.invoke(
        {"batch_id": "llm-brain"}, config={"configurable": {"thread_id": "llm-brain"}}
    )

    assert result["termination_reason"] == "accepted"
    assert result["focus"] == "db_cu_exhausted"
    assert len(result["evaluations"]) == 1
