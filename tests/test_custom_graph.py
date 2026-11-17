from __future__ import annotations

from decimal import Decimal

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from resguard.custom import CustomGraphGuard, CustomGuardState
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.policy import DegradationAction, GuardPolicy
from resguard.recorder import SessionRecorder
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile
from resguard.session import JsonlSessionStore
from tests.fakes import ScriptedChatModel, ai_final, ai_tool_call, make_echo_tool


def build_graph(model, guard, echo):
    def model_node(state):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(CustomGuardState)
    builder.add_node("model", guard.wrap_model_node(model_node, model_name="fake-model"))
    builder.add_node("guard", guard.guard_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", guard.route_after_model, {"guard": "guard", "end": END}
    )
    builder.add_conditional_edges(
        "guard", guard.route_after_guard, {"model": "model", "end": END}
    )
    return builder.compile()


def make_guard(*, budget: float, echo, cache=None) -> CustomGraphGuard:
    return CustomGraphGuard(
        ledger=InMemoryLedger(ResourceLimits({ResourceKind.USD: budget})),
        policy=GuardPolicy(
            soft_thresholds=(0.5, 0.8),
            hard_limit_action=DegradationAction.STOP,
        ),
        tools=[echo],
        tool_profiles={"echo": ToolResourceProfile(usd_per_call=0.01, cacheable=True)},
        pricing=ModelPricingTable(
            {"fake-model": ModelPricing(Decimal("0"), Decimal("0"))}
        ),
        cache=cache,
        estimate_output_tokens=10,
        run_id="custom-run",
    )


def test_custom_graph_enforces_budget() -> None:
    echo, calls = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_final("done"),
        ]
    )
    guard = make_guard(budget=0.015, echo=echo)
    graph = build_graph(model, guard, echo)

    result = graph.invoke({"messages": [HumanMessage("hi")]})

    assert len(calls) == 1
    assert guard.tool_calls == 1
    assert any("[budget]" in str(message.content) for message in result["messages"])
    assert result["resguard_stop"] is True


def test_custom_graph_ephemeral_reminder() -> None:
    echo, _ = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_final("done"),
        ]
    )
    guard = make_guard(budget=0.02, echo=echo)
    graph = build_graph(model, guard, echo)

    result = graph.invoke({"messages": [HumanMessage("hi")]})

    assert any("[budget]" in str(message.content) for message in model.requests[1])
    assert all("scope=run" not in str(message.content) for message in result["messages"])
    assert result["resguard_stop"] is True


def test_custom_graph_cache_hit() -> None:
    from resguard.cache import CacheContext, ToolCache

    echo, calls = make_echo_tool()
    cache = ToolCache()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1", args={"text": "hi"}),
            ai_tool_call(call_id="call_2", args={"text": "hi"}),
            ai_final("done"),
        ]
    )
    guard = make_guard(budget=1.0, echo=echo, cache=cache)
    guard.cache_context = CacheContext(user_id="test-user")
    graph = build_graph(model, guard, echo)

    graph.invoke({"messages": [HumanMessage("hi")]})

    assert len(calls) == 1
    assert guard.cache_hits == 1
    assert guard.tool_calls == 1


def test_custom_graph_recorder_integration(tmp_path) -> None:
    echo, _ = make_echo_tool()
    model = ScriptedChatModel(
        responses=[ai_tool_call(call_id="call_1"), ai_final("done")]
    )
    guard = make_guard(budget=1.0, echo=echo)
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    recorder = SessionRecorder(guard=guard, store=store, session_id="custom-1")
    graph = build_graph(model, guard, echo)

    graph.invoke({"messages": [HumanMessage("hi")]})
    record = recorder.finalize(success=True)

    assert record.metrics.tool_calls == 1
    assert record.metrics.tool_usd == pytest.approx(0.01)
    assert list(store.iter_sessions())[0].session_id == "custom-1"
