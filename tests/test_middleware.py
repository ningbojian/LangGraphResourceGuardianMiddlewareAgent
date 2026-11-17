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
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile
from tests.fakes import ScriptedChatModel, ai_final, ai_tool_call, make_echo_tool


def make_guard(
    *,
    budget_usd: float,
    tool_profiles: dict[str, ToolResourceProfile],
    cache: ToolCache | None = None,
    hard_action: DegradationAction = DegradationAction.STOP,
    pricing: ModelPricing | None = None,
) -> tuple[ResourceGuardMiddleware, InMemoryLedger]:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: budget_usd}))
    guard = ResourceGuardMiddleware(
        ledger=ledger,
        policy=GuardPolicy(soft_thresholds=(0.5, 0.8), hard_limit_action=hard_action),
        tool_profiles=tool_profiles,
        pricing=ModelPricingTable(
            {"fake-model": pricing or ModelPricing(Decimal("0"), Decimal("0"))}
        ),
        cache=cache,
        cache_context=CacheContext(user_id="test-user"),
        estimate_output_tokens=10,
        run_id="test-run",
    )
    return guard, ledger


def test_tool_budget_stops_runaway_loop() -> None:
    echo, calls = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_tool_call(call_id="call_3"),
            ai_final("done"),
        ]
    )
    guard, ledger = make_guard(
        budget_usd=0.025,
        tool_profiles={"echo": ToolResourceProfile(usd_per_call=0.01)},
    )
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    result = agent.invoke(
        {"messages": [HumanMessage("hi")]},
        config={"recursion_limit": 20},
    )

    assert len(calls) == 2
    assert ledger.snapshot().used.get(ResourceKind.USD) == pytest.approx(0.02)
    assert any("budget" in str(message.content).lower() for message in result["messages"])


def test_model_cost_is_committed_from_usage_metadata() -> None:
    model = ScriptedChatModel(
        responses=[ai_final("done", input_tokens=1_000, output_tokens=500)]
    )
    guard, ledger = make_guard(
        budget_usd=1.0,
        tool_profiles={},
        pricing=ModelPricing(Decimal("1"), Decimal("2")),
    )
    agent = create_agent(model=model, tools=[], middleware=[guard])

    agent.invoke({"messages": [HumanMessage("hi")]})

    assert ledger.snapshot().used.get(ResourceKind.USD) == pytest.approx(0.002)


def test_soft_reminder_is_ephemeral() -> None:
    echo, calls = make_echo_tool()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1"),
            ai_tool_call(call_id="call_2"),
            ai_final("done"),
        ]
    )
    guard, ledger = make_guard(
        budget_usd=0.02,
        tool_profiles={"echo": ToolResourceProfile(usd_per_call=0.01)},
    )
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    result = agent.invoke(
        {"messages": [HumanMessage("hi")]},
        config={"recursion_limit": 20},
    )

    assert any("[budget]" in str(message.content) for message in model.requests[1])
    # The ephemeral reminder is identifiable by "scope=run"; it must not be
    # persisted. The final stop AIMessage also starts with "[budget]" and is a
    # legitimate persisted message.
    assert all("scope=run" not in str(message.content) for message in result["messages"])
    assert any("limit reached" in str(message.content) for message in result["messages"])
    assert len(calls) == 2


def test_cache_hit_avoids_tool_execution() -> None:
    echo, calls = make_echo_tool()
    cache = ToolCache()
    model = ScriptedChatModel(
        responses=[
            ai_tool_call(call_id="call_1", args={"text": "hi"}),
            ai_tool_call(call_id="call_2", args={"text": "hi"}),
            ai_final("done"),
        ]
    )
    guard, ledger = make_guard(
        budget_usd=1.0,
        tool_profiles={
            "echo": ToolResourceProfile(
                usd_per_call=0.01, cacheable=True, idempotent=True, ttl_seconds=300
            )
        },
        cache=cache,
    )
    agent = create_agent(model=model, tools=[echo], middleware=[guard])

    agent.invoke(
        {"messages": [HumanMessage("hi")]},
        config={"recursion_limit": 20},
    )

    assert len(calls) == 1
    assert cache.stats.hits == 1
    assert ledger.snapshot().used.get(ResourceKind.USD) == pytest.approx(0.01)
