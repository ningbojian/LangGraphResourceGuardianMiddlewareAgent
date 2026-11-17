"""Executable scenario evaluator for the offline optimizer.

Unlike ``SyntheticEvaluator`` (a factor-based stand-in), this evaluator actually
runs the guard on scripted scenarios under a candidate policy and aggregates
multi-dimensional metrics. It is still a mechanics benchmark, not a replay of
production sessions; label its results accordingly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.batch import BatchInsights
from resguard.ledger import InMemoryLedger
from resguard.metrics import SessionMetrics
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.proposer import PolicyProposal
from resguard.resources import (
    ResourceKind,
    ResourceLimits,
    ToolResourceProfile,
)
from resguard.testing import (
    ScriptedChatModel,
    ai_final,
    ai_parallel_tool_calls,
    make_counting_tool,
    make_replay_tool,
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    calls: tuple[tuple[str, dict[str, Any]], ...]
    tool_profiles: Mapping[str, ToolResourceProfile]
    budget_usd: float = 1.0
    concurrency_limit: int | None = None
    api_quota_limit: float | None = None
    db_cu_limit: float | None = None
    legitimate: bool = False
    parallel_group: int = 1
    expected_marker: str | None = "done"
    recorded_results: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": [[name, dict(args)] for name, args in self.calls],
            "tool_profiles": {
                name: asdict(profile) for name, profile in self.tool_profiles.items()
            },
            "budget_usd": self.budget_usd,
            "concurrency_limit": self.concurrency_limit,
            "api_quota_limit": self.api_quota_limit,
            "db_cu_limit": self.db_cu_limit,
            "legitimate": self.legitimate,
            "parallel_group": self.parallel_group,
            "expected_marker": self.expected_marker,
            "recorded_results": dict(self.recorded_results) if self.recorded_results else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSpec":
        return cls(
            name=str(data["name"]),
            calls=tuple(
                (str(name), dict(args)) for name, args in data.get("calls", [])
            ),
            tool_profiles={
                name: ToolResourceProfile(**profile)
                for name, profile in (data.get("tool_profiles") or {}).items()
            },
            budget_usd=float(data.get("budget_usd", 1.0)),
            concurrency_limit=data.get("concurrency_limit"),
            api_quota_limit=data.get("api_quota_limit"),
            db_cu_limit=data.get("db_cu_limit"),
            legitimate=bool(data.get("legitimate", False)),
            parallel_group=int(data.get("parallel_group", 1)),
            expected_marker=data.get("expected_marker", "done"),
            recorded_results=data.get("recorded_results"),
        )


@dataclass(frozen=True)
class PolicyConfig:
    cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    model_cost_factor: Decimal = Decimal("1")
    max_concurrency: int | None = None
    budget_usd: float | None = None
    api_quota_limit: float | None = None
    db_cu_limit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "model_cost_factor": str(self.model_cost_factor),
            "max_concurrency": self.max_concurrency,
            "budget_usd": self.budget_usd,
            "api_quota_limit": self.api_quota_limit,
            "db_cu_limit": self.db_cu_limit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyConfig":
        return cls(
            cache_enabled=bool(data.get("cache_enabled", False)),
            cache_ttl_seconds=int(data.get("cache_ttl_seconds", 300)),
            model_cost_factor=Decimal(str(data.get("model_cost_factor", "1"))),
            max_concurrency=data.get("max_concurrency"),
            budget_usd=data.get("budget_usd"),
            api_quota_limit=data.get("api_quota_limit"),
            db_cu_limit=data.get("db_cu_limit"),
        )

    def with_proposal(self, proposal: PolicyProposal) -> "PolicyConfig":
        cache_enabled = self.cache_enabled
        model_cost_factor = self.model_cost_factor
        max_concurrency = self.max_concurrency
        budget_usd = self.budget_usd
        api_quota_limit = self.api_quota_limit
        db_cu_limit = self.db_cu_limit

        for change in proposal.workflow_changes:
            if change.kind == "add_cache_node":
                cache_enabled = True
            elif change.kind == "model_routing":
                model_cost_factor = Decimal("0.7")
            elif change.kind == "set_max_concurrency":
                value = change.config.get("max_concurrency")
                if isinstance(value, (int, float)) and value > 0:
                    max_concurrency = int(value)
        for change in proposal.parameter_changes:
            path = change.path.lower()
            if "cache" in path:
                cache_enabled = True
            if "concurrency" in path and isinstance(change.new, (int, float)) and change.new > 0:
                max_concurrency = int(change.new)
            if ("budget" in path or "usd" in path) and isinstance(change.new, (int, float)):
                budget_usd = float(change.new)
            if ("api_quota" in path or "quota" in path) and isinstance(change.new, (int, float)):
                api_quota_limit = float(change.new)
            if "db_cu" in path and isinstance(change.new, (int, float)):
                db_cu_limit = float(change.new)

        return PolicyConfig(
            cache_enabled=cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            model_cost_factor=model_cost_factor,
            max_concurrency=max_concurrency,
            budget_usd=budget_usd,
            api_quota_limit=api_quota_limit,
            db_cu_limit=db_cu_limit,
        )


@dataclass
class ScenarioResult:
    name: str
    executed_tool_calls: int
    metrics: SessionMetrics
    termination: str
    success: bool
    false_positive: bool = False


def _default_profile() -> ToolResourceProfile:
    return ToolResourceProfile(usd_per_call=0.001, cacheable=True, idempotent=True)


def run_scenario(
    spec: ScenarioSpec,
    config: PolicyConfig,
    *,
    judge: Any | None = None,
) -> ScenarioResult:
    profiles = dict(spec.tool_profiles)
    for name, _ in spec.calls:
        profiles.setdefault(name, _default_profile())
    calls_by_tool: dict[str, list[str]] = {name: [] for name in profiles}
    if spec.recorded_results is not None:
        tools = [
            make_replay_tool(name, dict(spec.recorded_results))
            for name in profiles
        ]
    else:
        tools = [make_counting_tool(name, calls_by_tool[name]) for name in profiles]

    responses = []
    group_size = max(1, spec.parallel_group)
    for start in range(0, len(spec.calls), group_size):
        responses.append(
            ai_parallel_tool_calls(
                list(spec.calls[start : start + group_size]),
                prefix=f"g{start}",
            )
        )
    responses.append(ai_final("done"))
    model = ScriptedChatModel(responses=responses)

    budget = config.budget_usd if config.budget_usd is not None else spec.budget_usd
    limits: dict[ResourceKind, float] = {ResourceKind.USD: budget}
    concurrency = config.max_concurrency or spec.concurrency_limit
    if concurrency is not None:
        limits[ResourceKind.CONCURRENCY] = concurrency
    api_quota_limit = (
        config.api_quota_limit
        if config.api_quota_limit is not None
        else spec.api_quota_limit
    )
    if api_quota_limit is not None:
        limits[ResourceKind.API_QUOTA] = api_quota_limit
    db_cu_limit = config.db_cu_limit if config.db_cu_limit is not None else spec.db_cu_limit
    if db_cu_limit is not None:
        limits[ResourceKind.DB_CU] = db_cu_limit
    ledger = InMemoryLedger(ResourceLimits(limits))
    guard = ResourceGuardMiddleware(
        ledger=ledger,
        policy=GuardPolicy(
            soft_thresholds=(0.5, 0.8),
            hard_limit_action=DegradationAction.STOP,
        ),
        tool_profiles=profiles,
        pricing=ModelPricingTable(
            {
                "fake-model": ModelPricing(
                    Decimal("1") * config.model_cost_factor,
                    Decimal("2") * config.model_cost_factor,
                )
            }
        ),
        cache=ToolCache() if config.cache_enabled else None,
        cache_context=CacheContext(user_id="scenario"),
        estimate_output_tokens=10,
        run_id=f"scenario:{spec.name}",
    )
    agent = create_agent(model=model, tools=tools, middleware=[guard])
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage("run scenario")]},
        config={"recursion_limit": 50},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    snapshot = guard.snapshot()
    text = " ".join(str(message.content) for message in result["messages"]).lower()
    termination = "guard_stop" if ("budget" in text or "limit" in text) else "completed"
    false_positive = spec.legitimate and termination == "guard_stop"
    quality = (
        judge.score(text)
        if judge is not None
        else (
            1.0
            if spec.expected_marker and spec.expected_marker.lower() in text
            else 0.0
        )
    )
    quality = max(0.0, min(1.0, float(quality)))
    success = quality >= 0.5 and termination != "error"
    metrics = SessionMetrics(
        tool_calls=guard.tool_calls,
        cache_hits=guard.cache_hits,
        db_queries=int(snapshot.used.get(ResourceKind.DB_QUERIES)),
        db_cu=snapshot.used.get(ResourceKind.DB_CU),
        api_quota=snapshot.used.get(ResourceKind.API_QUOTA),
        peak_concurrency=guard.peak_concurrency,
        concurrency_denials=guard.concurrency_denials,
        false_positive_stops=1 if false_positive else 0,
        answer_quality=quality,
        llm_usd=guard.llm_usd,
        tool_usd=guard.tool_usd,
        wall_clock_ms=elapsed_ms,
    )
    return ScenarioResult(
        name=spec.name,
        executed_tool_calls=guard.tool_calls,
        metrics=metrics,
        termination=termination,
        success=success,
        false_positive=false_positive,
    )


def aggregate(results: Iterable[ScenarioResult]) -> BatchInsights:
    items = list(results)
    total = SessionMetrics()
    for item in items:
        total = total.merge(item.metrics)
    successes = sum(1 for item in items if item.success)
    false_positives = sum(1 for item in items if item.false_positive)
    quality = sum(item.metrics.answer_quality for item in items)
    vector = total.resource_amount()
    divisor = successes or 1
    return BatchInsights(
        session_count=len(items),
        total_metrics=total,
        success_rate=(successes / len(items)) if items else 0.0,
        false_positive_rate=(false_positives / len(items)) if items else 0.0,
        quality_rate=(quality / len(items)) if items else 0.0,
        labeled_sessions=len(items),
        proxy_label=True,
        resource_per_success={
            kind.value: vector.get(kind) / divisor
            for kind in (
                ResourceKind.USD,
                ResourceKind.DB_CU,
                ResourceKind.DB_QUERIES,
                ResourceKind.API_QUOTA,
                ResourceKind.TOOL_CALLS,
            )
        },
        hotspots=sorted(
            {item.termination for item in items if item.termination != "completed"}
        ),
    )


class ScenarioEvaluator:
    """Runs scripted scenarios under the baseline and candidate policy configs."""

    def __init__(
        self,
        specs: Iterable[ScenarioSpec],
        *,
        judge: Any | None = None,
    ) -> None:
        self.specs = list(specs)
        self.judge = judge
        self._baseline: BatchInsights | None = None

    @property
    def baseline_insights(self) -> BatchInsights:
        if self._baseline is None:
            self._baseline = aggregate(
                run_scenario(spec, PolicyConfig(), judge=self.judge) for spec in self.specs
            )
        return self._baseline

    def evaluate(
        self,
        proposal: PolicyProposal,
        baseline: BatchInsights,
    ) -> BatchInsights:
        config = PolicyConfig().with_proposal(proposal)
        return aggregate(
            run_scenario(spec, config, judge=self.judge) for spec in self.specs
        )
