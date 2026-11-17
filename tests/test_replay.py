from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict

from resguard.metrics import SessionMetrics
from resguard.proposer import PolicyProposal, WorkflowChange
from resguard.replay import (
    MessageReplayEvaluator,
    ReplayEvaluator,
    replay_messages,
    replay_spec_from_record,
)
from resguard.resources import ToolResourceProfile
from resguard.session import SessionRecord


def usage() -> dict[str, int]:
    return {"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500}


def message_record() -> SessionRecord:
    messages = [
        HumanMessage("refund?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_docs",
                    "args": {"text": "refund"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
            usage_metadata=usage(),
        ),
        ToolMessage(content="doc", tool_call_id="c1", name="search_docs"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_docs",
                    "args": {"text": "refund"},
                    "id": "c2",
                    "type": "tool_call",
                }
            ],
            usage_metadata=usage(),
        ),
        ToolMessage(content="doc", tool_call_id="c2", name="search_docs"),
        AIMessage(content="done", usage_metadata=usage()),
    ]
    return SessionRecord(
        session_id="messages-1",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        query_type="refund",
        model_version="fake-model",
        metrics=SessionMetrics(tool_calls=2, tool_usd=0.02),
        success=True,
        label_source="human",
        messages=messages_to_dict(messages),
    )


def traced_record() -> SessionRecord:
    return SessionRecord(
        session_id="trace-1",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        query_type="refund",
        metrics=SessionMetrics(tool_calls=5, tool_usd=0.05),
        success=True,
        label_source="human",
        trace=[
            {"name": "search_docs", "args": {"text": "refund"}, "result": "doc"}
            for _ in range(5)
        ],
    )


def cache_proposal() -> PolicyProposal:
    return PolicyProposal(
        id="p-cache",
        name="cache",
        description="cache repeated reads",
        workflow_changes=[
            WorkflowChange("add_cache_node", "search_docs", {"ttl_seconds": 300}, "x")
        ],
        evidence=["trace-1"],
    )


def test_replay_spec_from_record_preserves_calls_and_results() -> None:
    record = traced_record()

    spec = replay_spec_from_record(
        record,
        {"search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)},
        budget_usd=1.0,
    )

    assert len(spec.calls) == 5
    assert spec.calls[0] == ("search_docs", {"text": "refund"})
    assert spec.recorded_results == {"refund": "doc"}
    assert spec.legitimate is True


def test_replay_evaluator_applies_candidate_cache() -> None:
    record = traced_record()
    spec = replay_spec_from_record(
        record,
        {"search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)},
        budget_usd=1.0,
    )
    evaluator = ReplayEvaluator([spec])
    baseline = evaluator.baseline_insights

    candidate = evaluator.evaluate(cache_proposal(), baseline)

    assert baseline.total_metrics.tool_calls == 5
    assert candidate.total_metrics.tool_calls == 1
    assert candidate.total_metrics.cache_hits == 4
    assert candidate.resource_per_success["usd"] < baseline.resource_per_success["usd"]


def test_session_record_trace_round_trip() -> None:
    record = traced_record()

    restored = SessionRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.trace[0]["result"] == "doc"


def test_full_message_replay_baseline_and_cache() -> None:
    record = message_record()
    profiles = {"search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)}
    evaluator = MessageReplayEvaluator([record], profiles, budget_usd=1.0)
    baseline = evaluator.baseline_insights

    candidate = evaluator.evaluate(cache_proposal(), baseline)

    assert baseline.total_metrics.tool_calls == 2
    assert candidate.total_metrics.tool_calls == 1
    assert candidate.total_metrics.cache_hits == 1
    assert candidate.quality_rate == 1.0


def test_full_message_replay_returns_recorded_result() -> None:
    record = message_record()

    result = replay_messages(
        record,
        {"search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)},
    )

    assert result.executed_tool_calls == 2
    assert result.termination == "completed"
    assert result.metrics.answer_quality == 1.0


def test_session_record_messages_round_trip() -> None:
    record = message_record()

    restored = SessionRecord.from_dict(record.to_dict())

    assert restored == record
    assert len(restored.messages) == 6

