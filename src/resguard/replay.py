"""Recorded-trace replay.

Two levels:

- ``ReplayEvaluator`` / ``replay_spec_from_record`` replay a **tool-call sequence**
  with recorded results (``SessionRecord.trace``).
- ``MessageReplayEvaluator`` / ``replay_messages`` replay the full recorded
  **message sequence**: recorded AI messages are fed back by ``ReplayChatModel``
  and recorded tool results by ``QueuedReplayTool``. This is a single-run
  replay; multi-turn session replay is a future extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.evaluator import PolicyConfig, ScenarioEvaluator, ScenarioResult, ScenarioSpec, aggregate
from resguard.ledger import InMemoryLedger
from resguard.metrics import SessionMetrics
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile
from resguard.session import SessionRecord


def replay_spec_from_record(
    record: SessionRecord,
    tool_profiles: Mapping[str, ToolResourceProfile],
    *,
    budget_usd: float = 1.0,
) -> ScenarioSpec:
    calls: list[tuple[str, dict]] = []
    recorded: dict[str, str] = {}
    for entry in record.trace:
        name = str(entry.get("name", "unknown"))
        args = dict(entry.get("args") or {})
        calls.append((name, args))
        text = str(args.get("text", ""))
        recorded[text] = str(entry.get("result", ""))
    return ScenarioSpec(
        name=f"replay:{record.session_id}",
        calls=tuple(calls),
        tool_profiles=tool_profiles,
        budget_usd=budget_usd,
        legitimate=bool(record.success),
        recorded_results=recorded,
    )


def replay_specs(
    records: Iterable[SessionRecord],
    tool_profiles: Mapping[str, ToolResourceProfile],
    *,
    budget_usd: float = 1.0,
) -> list[ScenarioSpec]:
    return [
        replay_spec_from_record(record, tool_profiles, budget_usd=budget_usd)
        for record in records
    ]


class ReplayEvaluator(ScenarioEvaluator):
    """ScenarioEvaluator over recorded-trace tool-call sequences."""


class ReplayChatModel(BaseChatModel):
    """Returns recorded AI messages in order; records each request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[AIMessage] = Field(default_factory=list)
    model_name: str = "replay-model"
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "replay"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(list(messages))
        response = self.responses.pop(0) if self.responses else AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ReplayChatModel":
        return self


class _LooseArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class QueuedReplayTool(BaseTool):
    """Returns recorded results in call order, without executing side effects."""

    name: str
    description: str = "Replay tool"
    results: list[str] = Field(default_factory=list)
    args_schema: type[BaseModel] = _LooseArgs

    def _run(self, **kwargs: Any) -> str:
        return self.results.pop(0) if self.results else "replayed"


def _tool_queues(messages: list[BaseMessage]) -> dict[str, list[str]]:
    tool_by_id = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    queues: dict[str, list[str]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            name = tool_call.get("name", "unknown")
            result = tool_by_id.get(tool_call.get("id", ""))
            queues.setdefault(name, []).append(
                str(result.content) if result is not None else ""
            )
    return queues


def replay_messages(
    record: SessionRecord,
    tool_profiles: Mapping[str, ToolResourceProfile],
    *,
    config: PolicyConfig | None = None,
    judge: Any | None = None,
) -> ScenarioResult:
    config = config or PolicyConfig()
    messages = messages_from_dict(record.messages) if record.messages else []
    initial = next(
        (message for message in messages if isinstance(message, HumanMessage)), None
    )
    if initial is None:
        metrics = SessionMetrics()
        return ScenarioResult(
            name=f"replay:{record.session_id}",
            executed_tool_calls=0,
            metrics=metrics,
            termination="no_input",
            success=False,
        )

    responses = [message for message in messages if isinstance(message, AIMessage)]
    queues = _tool_queues(messages)
    profiles = dict(tool_profiles)
    for name in queues:
        profiles.setdefault(
            name, ToolResourceProfile(usd_per_call=0.001, cacheable=True)
        )
    tools = [
        QueuedReplayTool(name=name, results=queues.get(name, []))
        for name in profiles
        if name in queues
    ]

    model_name = (
        record.model_version
        if record.model_version and record.model_version != "unknown"
        else "replay-model"
    )
    budget = config.budget_usd if config.budget_usd is not None else 1.0
    limits: dict[ResourceKind, float] = {ResourceKind.USD: budget}
    if config.max_concurrency is not None:
        limits[ResourceKind.CONCURRENCY] = config.max_concurrency
    if config.api_quota_limit is not None:
        limits[ResourceKind.API_QUOTA] = config.api_quota_limit
    if config.db_cu_limit is not None:
        limits[ResourceKind.DB_CU] = config.db_cu_limit
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
                model_name: ModelPricing(
                    Decimal("1") * config.model_cost_factor,
                    Decimal("2") * config.model_cost_factor,
                )
            }
        ),
        cache=ToolCache() if config.cache_enabled else None,
        cache_context=CacheContext(user_id=record.user_id_hash or "replay"),
        estimate_output_tokens=10,
        run_id=f"replay:{record.session_id}",
    )
    agent = create_agent(
        model=ReplayChatModel(responses=responses, model_name=model_name),
        tools=tools,
        middleware=[guard],
    )
    result = agent.invoke(
        {"messages": [initial]},
        config={"recursion_limit": 50},
    )
    snapshot = guard.snapshot()
    text = " ".join(str(message.content) for message in result["messages"]).lower()
    termination = "guard_stop" if ("budget" in text or "limit" in text) else "completed"
    false_positive = record.success is True and termination == "guard_stop"
    success = termination == "completed" and record.success is not False
    quality = 1.0 if success else 0.0
    if judge is not None:
        quality = max(0.0, min(1.0, float(judge.score(text))))
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
    )
    return ScenarioResult(
        name=f"replay:{record.session_id}",
        executed_tool_calls=guard.tool_calls,
        metrics=metrics,
        termination=termination,
        success=success,
        false_positive=false_positive,
    )


class MessageReplayEvaluator:
    """Replays full recorded message sequences under a candidate policy."""

    def __init__(
        self,
        records: Iterable[SessionRecord],
        tool_profiles: Mapping[str, ToolResourceProfile],
        *,
        budget_usd: float = 1.0,
        judge: Any | None = None,
    ) -> None:
        self.records = list(records)
        self.tool_profiles = dict(tool_profiles)
        self.budget_usd = budget_usd
        self.judge = judge
        self._baseline = None

    @property
    def baseline_insights(self):
        if self._baseline is None:
            self._baseline = self.evaluate_config(
                PolicyConfig(budget_usd=self.budget_usd)
            )
        return self._baseline

    def evaluate_config(self, config: PolicyConfig):
        return aggregate(
            replay_messages(
                record,
                self.tool_profiles,
                config=config,
                judge=self.judge,
            )
            for record in self.records
        )

    def evaluate(self, proposal, baseline):
        config = PolicyConfig(budget_usd=self.budget_usd).with_proposal(proposal)
        return self.evaluate_config(config)
