"""Custom LangGraph StateGraph adapter.

``AgentMiddleware`` only plugs into ``create_agent``. This module provides the
same guard semantics for a hand-built graph:

- ``wrap_model_node(node, model_name=...)`` — budget check, ephemeral reminder,
  model-cost reserve/commit;
- ``guard_node(state)`` — preflight/execute tool calls with multi-resource
  reserve/commit, cache, and concurrency lease;
- ``route_after_model`` / ``route_after_guard`` — conditional-edge helpers.

The guard node executes tools itself (replacing a plain ``ToolNode``), so a
custom graph gets guard semantics without LangChain middleware internals.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Mapping
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricingTable, TokenUsage, cost_for_usage, estimate_request_cost
from resguard.ledger import ConcurrencyDenied, InMemoryLedger, ReservationDenied
from resguard.metrics import estimate_history_tokens, estimate_tool_schema_tokens
from resguard.policy import DegradationAction, GuardPolicy, decide, tool_decision
from resguard.reminder import ReminderState, format_reminder, should_remind
from resguard.resources import ResourceAmount, ResourceKind, ToolResourceProfile


class CustomGuardState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    resguard_stop: bool


class CustomGraphGuard:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        policy: GuardPolicy,
        tools: list[BaseTool],
        tool_profiles: Mapping[str, ToolResourceProfile],
        pricing: ModelPricingTable,
        cache: ToolCache | None = None,
        cache_context: CacheContext | None = None,
        estimate_output_tokens: int = 512,
        run_id: str = "default",
        unknown_tool_profile: ToolResourceProfile | None = None,
    ) -> None:
        self.ledger = ledger
        self.policy = policy
        self.tools = list(tools)
        self.tool_map = {tool.name: tool for tool in tools}
        self.tool_profiles = dict(tool_profiles)
        self.pricing = pricing
        self.cache = cache
        self.cache_context = cache_context or CacheContext()
        self.estimate_output_tokens = estimate_output_tokens
        self.run_id = run_id
        self.unknown_tool_profile = unknown_tool_profile or ToolResourceProfile(
            idempotent=False, cacheable=False
        )
        self.reminder_state = ReminderState()
        self.model_calls = 0
        self.tool_calls = 0
        self.cache_hits = 0
        self.llm_usd = 0.0
        self.tool_usd = 0.0
        self.peak_concurrency = 0
        self.concurrency_denials = 0
        self.last_binding: ResourceKind | None = None
        self.per_tool_counts: dict[str, int] = {}

    def snapshot(self):
        return self.ledger.snapshot()

    # ------------------------------------------------------------------ model
    def wrap_model_node(
        self,
        node: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        model_name: str,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def wrapped(state: dict[str, Any]) -> dict[str, Any]:
            snapshot = self.ledger.snapshot()
            decision = decide(snapshot, self.policy)
            if decision.binding_kind is not None and decision.action in (
                DegradationAction.STOP,
                DegradationAction.ASK_USER,
            ):
                self.last_binding = decision.binding_kind
                return {
                    "messages": [AIMessage(content=self._stop_message(decision))],
                    "resguard_stop": True,
                }

            state_for_model = state
            if should_remind(
                decision,
                snapshot,
                self.policy,
                self.reminder_state,
                model_call_count=self.model_calls,
            ):
                notice = format_reminder(snapshot, decision, scope="run")
                state_for_model = {
                    **state,
                    "messages": [
                        *state.get("messages", []),
                        HumanMessage(content=notice.message, name="resguard_budget"),
                    ],
                }
                self.reminder_state.mark(
                    decision, snapshot, model_call_count=self.model_calls
                )

            pricing = self.pricing.resolve(model_name)
            input_tokens = estimate_history_tokens(state.get("messages", []))
            input_tokens += estimate_tool_schema_tokens(self.tools)
            estimated_usd = estimate_request_cost(
                pricing,
                input_tokens=input_tokens,
                max_output_tokens=self.estimate_output_tokens,
            )
            amount = ResourceAmount(
                {
                    ResourceKind.USD: float(estimated_usd),
                    ResourceKind.LLM_INPUT_TOKENS: input_tokens,
                    ResourceKind.LLM_OUTPUT_TOKENS: self.estimate_output_tokens,
                }
            )
            try:
                reservation = self.ledger.reserve(
                    amount, reservation_id=f"model:{self.run_id}:{self.model_calls}"
                )
            except ReservationDenied as exc:
                self.last_binding = exc.binding
                return {
                    "messages": [AIMessage(content=self._denied_message(exc))],
                    "resguard_stop": True,
                }

            try:
                result = node(state_for_model)
            except Exception:
                self.ledger.release(reservation)
                raise

            actual = self._actual_model_amount(result, pricing)
            self.ledger.commit(reservation, actual)
            self.llm_usd += (actual or amount).get(ResourceKind.USD)
            self.model_calls += 1
            return result

        return wrapped

    # ------------------------------------------------------------------- tool
    def guard_node(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return {"messages": [], "resguard_stop": False}

        existing_ids = {
            message.tool_call_id for message in messages if isinstance(message, ToolMessage)
        }
        pending = [
            tool_call
            for tool_call in last_ai.tool_calls
            if tool_call.get("id") not in existing_ids
        ]

        out: list[ToolMessage] = []
        stop = False
        for tool_call in pending:
            name = tool_call.get("name", "")
            args = tool_call.get("args", {}) or {}
            call_id = tool_call.get("id") or uuid4().hex
            profile = self.tool_profiles.get(name, self.unknown_tool_profile)

            decision = tool_decision(
                self.policy,
                total_tool_calls=self.tool_calls,
                tool_name=name,
                per_tool_calls=self.per_tool_counts.get(name, 0),
            )
            if decision.action is DegradationAction.STOP:
                stop = True
                out.append(
                    ToolMessage(
                        content="[budget] tool-call limit reached; the call was not executed.",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            if self.cache is not None and profile.validate_cacheable():
                cached = self.cache.get(name, args, ctx=self.cache_context)
                if cached is not None:
                    self.cache_hits += 1
                    out.append(
                        ToolMessage(content=str(cached), tool_call_id=call_id, name=name)
                    )
                    continue

            amount = profile.per_call_amount()
            try:
                reservation = self.ledger.reserve(
                    amount, reservation_id=f"tool:{self.run_id}:{call_id}"
                )
            except ReservationDenied as exc:
                self.last_binding = exc.binding
                stop = True
                out.append(
                    ToolMessage(
                        content=self._denied_message(exc),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            try:
                lease = self.ledger.acquire_concurrency(profile.concurrency_slots)
            except ConcurrencyDenied:
                self.ledger.release(reservation)
                self.last_binding = ResourceKind.CONCURRENCY
                self.concurrency_denials += 1
                stop = True
                out.append(
                    ToolMessage(
                        content="[budget] concurrency limit reached; the call was not executed.",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            self.peak_concurrency = max(
                self.peak_concurrency, self.ledger.snapshot().concurrency_in_use
            )

            try:
                tool = self.tool_map[name]
                result = tool.invoke(args)
                self.ledger.commit(reservation, amount)
                lease.release()
                self.tool_calls += 1
                self.per_tool_counts[name] = self.per_tool_counts.get(name, 0) + 1
                self.tool_usd += amount.get(ResourceKind.USD)
                out.append(
                    ToolMessage(content=str(result), tool_call_id=call_id, name=name)
                )
                if self.cache is not None and profile.validate_cacheable():
                    self.cache.set(
                        name,
                        args,
                        result,
                        ctx=self.cache_context,
                        ttl_seconds=profile.ttl_seconds,
                    )
            except Exception as exc:  # tool errors become observations, not crashes
                self.ledger.release(reservation)
                lease.release()
                out.append(
                    ToolMessage(
                        content=f"[tool-error] {exc}",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )

        return {"messages": out, "resguard_stop": stop}

    # ---------------------------------------------------------------- routing
    def route_after_model(self, state: dict[str, Any]) -> str:
        if state.get("resguard_stop"):
            return "end"
        messages = state.get("messages", [])
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return "end"
        return "guard"

    def route_after_guard(self, state: dict[str, Any]) -> str:
        return "end" if state.get("resguard_stop") else "model"

    # ---------------------------------------------------------------- helpers
    def _actual_model_amount(
        self, result: dict[str, Any], pricing: Any
    ) -> ResourceAmount | None:
        messages = result.get("messages", [])
        message = next(
            (item for item in reversed(messages) if isinstance(item, AIMessage)), None
        )
        if message is None:
            return None
        usage = TokenUsage.from_usage_metadata(getattr(message, "usage_metadata", None))
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            return None
        cost = cost_for_usage(pricing, usage)
        return ResourceAmount(
            {
                ResourceKind.USD: float(cost),
                ResourceKind.LLM_INPUT_TOKENS: usage.input_tokens,
                ResourceKind.LLM_OUTPUT_TOKENS: usage.output_tokens,
            }
        )

    def _stop_message(self, decision: Any) -> str:
        binding = decision.binding_kind.value if decision.binding_kind else "resource"
        return (
            f"[budget] {binding} limit reached for run {self.run_id}; stopping "
            "before the next model call."
        )

    @staticmethod
    def _denied_message(exc: ReservationDenied) -> str:
        return (
            f"[budget] resource {exc.binding.value} exhausted in scope "
            f"{exc.scope_key}; the call was not executed."
        )
