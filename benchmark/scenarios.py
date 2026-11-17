from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from benchmark.fake_model import ScriptedChatModel, ai_final, ai_tool_call
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile


@dataclass
class ScenarioResult:
    scenario: str
    baseline: str
    executed_tool_calls: int
    attempted_tool_calls: int
    declared_usd: float
    termination: str
    elapsed_ms: float
    budget_usd: float = 0.0
    within_budget: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _echo_tool():
    calls: list[str] = []

    @tool
    def echo(text: str) -> str:
        """Echo text."""
        calls.append(text)
        return f"echo:{text}"

    return echo, calls


def run_case(
    scenario: str,
    baseline: str,
    *,
    calls: int,
    tool_cost: float,
    budget: float,
    official_limit: int,
) -> ScenarioResult:
    echo, executed = _echo_tool()
    responses = [ai_tool_call(call_id=f"call_{index}") for index in range(calls)]
    responses.append(ai_final("done"))
    model = ScriptedChatModel(responses=responses)

    middleware: list[Any] = []
    ledger: InMemoryLedger | None = None
    if baseline == "official":
        middleware = [
            ToolCallLimitMiddleware(run_limit=official_limit, exit_behavior="end")
        ]
    elif baseline == "full":
        ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: budget}))
        middleware = [
            ResourceGuardMiddleware(
                ledger=ledger,
                policy=GuardPolicy(
                    soft_thresholds=(0.5, 0.8),
                    hard_limit_action=DegradationAction.STOP,
                ),
                tool_profiles={"echo": ToolResourceProfile(usd_per_call=tool_cost)},
                pricing=ModelPricingTable(
                    {"fake-model": ModelPricing(Decimal("0"), Decimal("0"))}
                ),
                estimate_output_tokens=10,
                run_id=f"{scenario}:{baseline}",
            )
        ]
    elif baseline != "none":
        raise ValueError(f"unknown baseline {baseline}")

    agent = create_agent(model=model, tools=[echo], middleware=middleware)
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage("do the loop")]},
        config={"recursion_limit": 50},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    text = " ".join(str(message.content) for message in result["messages"]).lower()
    termination = "guard_stop" if ("budget" in text or "limit" in text) else "completed"
    declared_usd = float(ledger.snapshot().used.get(ResourceKind.USD)) if ledger else (
        len(executed) * tool_cost
    )
    return ScenarioResult(
        scenario=scenario,
        baseline=baseline,
        executed_tool_calls=len(executed),
        attempted_tool_calls=calls,
        declared_usd=declared_usd,
        termination=termination,
        elapsed_ms=elapsed_ms,
        budget_usd=budget,
        within_budget=declared_usd <= budget + 1e-9,
    )


def _named_tool(name: str):
    calls: list[str] = []

    def body(text: str) -> str:
        calls.append(text)
        return f"{name}:{text}"

    body.__name__ = name
    body.__doc__ = f"Deterministic {name} tool."
    return tool(name)(body), calls


def run_mixed_case(
    baseline: str,
    *,
    cheap_calls: int = 8,
    expensive_calls: int = 2,
    cheap_cost: float = 0.001,
    expensive_cost: float = 0.20,
    budget: float = 0.25,
    official_limit: int = 10,
) -> ScenarioResult:
    cheap, cheap_executed = _named_tool("cheap_lookup")
    expensive, expensive_executed = _named_tool("expensive_parse")
    responses = [
        ai_tool_call(name="cheap_lookup", call_id=f"cheap_{index}")
        for index in range(cheap_calls)
    ] + [
        ai_tool_call(name="expensive_parse", call_id=f"expensive_{index}")
        for index in range(expensive_calls)
    ] + [ai_final("done")]
    model = ScriptedChatModel(responses=responses)

    middleware: list[Any] = []
    ledger: InMemoryLedger | None = None
    if baseline == "official":
        # A call-count limit of 10 allows the whole script; it cannot see that
        # two of those calls are 200x more expensive than the others.
        middleware = [ToolCallLimitMiddleware(run_limit=official_limit, exit_behavior="end")]
    elif baseline == "full":
        ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: budget}))
        middleware = [
            ResourceGuardMiddleware(
                ledger=ledger,
                policy=GuardPolicy(
                    soft_thresholds=(0.5, 0.8),
                    hard_limit_action=DegradationAction.STOP,
                ),
                tool_profiles={
                    "cheap_lookup": ToolResourceProfile(usd_per_call=cheap_cost),
                    "expensive_parse": ToolResourceProfile(usd_per_call=expensive_cost),
                },
                pricing=ModelPricingTable(
                    {"fake-model": ModelPricing(Decimal("0"), Decimal("0"))}
                ),
                estimate_output_tokens=10,
                run_id="mixed:full",
            )
        ]
    elif baseline != "none":
        raise ValueError(f"unknown baseline {baseline}")

    agent = create_agent(model=model, tools=[cheap, expensive], middleware=middleware)
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage("do the mixed workload")]},
        config={"recursion_limit": 50},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    text = " ".join(str(message.content) for message in result["messages"]).lower()
    termination = "guard_stop" if ("budget" in text or "limit" in text) else "completed"
    executed_total = len(cheap_executed) + len(expensive_executed)
    declared_usd = (
        float(ledger.snapshot().used.get(ResourceKind.USD))
        if ledger
        else len(cheap_executed) * cheap_cost + len(expensive_executed) * expensive_cost
    )
    return ScenarioResult(
        scenario="mixed_cost_spike",
        baseline=baseline,
        executed_tool_calls=executed_total,
        attempted_tool_calls=cheap_calls + expensive_calls,
        declared_usd=declared_usd,
        termination=termination,
        elapsed_ms=elapsed_ms,
        budget_usd=budget,
        within_budget=declared_usd <= budget + 1e-9,
    )
