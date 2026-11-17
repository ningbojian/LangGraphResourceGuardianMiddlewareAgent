from __future__ import annotations

from decimal import Decimal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.recorder import SessionRecorder
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile
from resguard.session import JsonlSessionStore
from tests.fakes import ScriptedChatModel, ai_final, ai_tool_call, make_echo_tool


def make_guard(
    *,
    budget_usd: float,
    tool_profiles: dict[str, ToolResourceProfile],
    cache: ToolCache | None = None,
    pricing: ModelPricing | None = None,
) -> ResourceGuardMiddleware:
    return ResourceGuardMiddleware(
        ledger=InMemoryLedger(ResourceLimits({ResourceKind.USD: budget_usd})),
        policy=GuardPolicy(soft_thresholds=(0.5, 0.8), hard_limit_action=DegradationAction.STOP),
        tool_profiles=tool_profiles,
        pricing=ModelPricingTable(
            {"fake-model": pricing or ModelPricing(Decimal("0"), Decimal("0"))}
        ),
        cache=cache,
        cache_context=CacheContext(user_id="test-user"),
        estimate_output_tokens=10,
        run_id="recorder-run",
    )


def test_recorder_writes_session_record(tmp_path) -> None:
    echo, _ = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_final("done"),
        ]
    )
    guard = make_guard(
        budget_usd=1.0,
        tool_profiles={"echo": ToolResourceProfile(usd_per_call=0.01)},
    )
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    recorder = SessionRecorder(
        guard=guard,
        store=store,
        session_id="s1",
        query_type="refund",
        model_version="fake-model",
    )
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]}, config={"recursion_limit": 20})
    record = recorder.finalize(success=True)

    assert record.session_id == "s1"
    assert record.query_type == "refund"
    assert record.success is True
    assert record.label_source == "human"
    assert record.metrics.tool_calls == 2
    assert record.metrics.tool_usd == pytest.approx(0.02)
    assert record.metrics.llm_usd == pytest.approx(guard.llm_usd)
    assert record.metrics.wall_clock_ms >= 0
    stored = list(store.iter_sessions())
    assert len(stored) == 1
    assert stored[0].session_id == "s1"


def test_recorder_captures_cache_hits(tmp_path) -> None:
    echo, calls = make_echo_tool()
    cache = ToolCache()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1", args={"text": "hi"}),
            ai_tool_call(call_id="call_2", args={"text": "hi"}),
            ai_final("done"),
        ]
    )
    guard = make_guard(
        budget_usd=1.0,
        tool_profiles={
            "echo": ToolResourceProfile(
                usd_per_call=0.01, cacheable=True, idempotent=True, ttl_seconds=300
            )
        },
        cache=cache,
    )
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    recorder = SessionRecorder(guard=guard, store=store, session_id="s2")
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]}, config={"recursion_limit": 20})
    record = recorder.finalize(success=True)

    assert len(calls) == 1
    assert record.metrics.cache_hits == 1
    assert record.metrics.tool_calls == 1


def test_recorder_records_binding_limit(tmp_path) -> None:
    echo, _ = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_final("done"),
        ]
    )
    guard = make_guard(
        budget_usd=0.015,
        tool_profiles={"echo": ToolResourceProfile(usd_per_call=0.01)},
    )
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    recorder = SessionRecorder(guard=guard, store=store, session_id="s3")
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]}, config={"recursion_limit": 20})
    record = recorder.finalize(success=False, termination_reason="usd_exhausted")

    assert record.metrics.tool_calls == 1
    assert record.binding_limit == "usd"
    assert record.termination_reason == "usd_exhausted"


def test_recorder_model_cost_and_unlabeled_session(tmp_path) -> None:
    model = ScriptedChatModel(responses=[ai_final("done")])
    guard = make_guard(
        budget_usd=1.0,
        tool_profiles={},
        pricing=ModelPricing(Decimal("1"), Decimal("2")),
    )
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    recorder = SessionRecorder(guard=guard, store=store, session_id="s4")
    agent = create_agent(model=model, tools=[], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]})
    record = recorder.finalize()

    assert record.metrics.llm_usd == pytest.approx(0.002)
    assert record.success is None
    assert record.label_source == "none"


def test_recorder_includes_trace(tmp_path) -> None:
    model = ScriptedChatModel(responses=[ai_final("done")])
    guard = make_guard(budget_usd=1.0, tool_profiles={})
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    trace = [{"name": "search_docs", "args": {"text": "refund"}, "result": "doc"}]
    recorder = SessionRecorder(
        guard=guard, store=store, session_id="s5", trace=trace
    )
    agent = create_agent(model=model, tools=[], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]})
    record = recorder.finalize(success=True)

    assert record.trace == trace

