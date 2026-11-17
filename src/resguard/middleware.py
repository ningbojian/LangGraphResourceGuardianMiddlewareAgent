"""Runtime resource guard middleware for LangChain v1 ``create_agent``.

This is the lightweight online half of resguard: it records/enforces the
currently approved static policy and never calls an LLM itself.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricingTable, TokenUsage, cost_for_usage, estimate_request_cost
from resguard.ledger import (
    BudgetScope,
    CompositeLedger,
    CompositeSnapshot,
    ConcurrencyDenied,
    InMemoryLedger,
    ReservationDenied,
    ResourceSnapshot,
)
from resguard.metrics import (
    estimate_history_tokens,
    estimate_tokens,
    estimate_tool_schema_tokens,
)
from resguard.policy import DegradationAction, GuardPolicy, decide
from resguard.reminder import ReminderState, format_reminder, should_remind
from resguard.resources import ResourceAmount, ResourceKind, ToolResourceProfile


class ResourceGuardMiddleware(AgentMiddleware):
    """Enforce an approved static policy over cumulative resources.

    One middleware instance is intended per run/thread. For run+thread+user
    composition, pass a :class:`CompositeLedger`.
    """

    def __init__(
        self,
        *,
        ledger: InMemoryLedger | CompositeLedger,
        policy: GuardPolicy,
        tool_profiles: dict[str, ToolResourceProfile],
        pricing: ModelPricingTable,
        cache: ToolCache | None = None,
        cache_context: CacheContext | None = None,
        estimate_output_tokens: int = 512,
        run_id: str = "default",
        unknown_tool_profile: ToolResourceProfile | None = None,
    ) -> None:
        super().__init__()
        self.ledger = ledger
        self.policy = policy
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
        self.denied_calls = 0
        self.llm_usd = 0.0
        self.tool_usd = 0.0
        self.peak_concurrency = 0
        self.concurrency_denials = 0
        self.last_binding: ResourceKind | None = None

    # ------------------------------------------------------------------ model
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        snapshot = self._primary_snapshot()
        decision = decide(snapshot, self.policy)

        if decision.binding_kind is not None and decision.action in (
            DegradationAction.STOP,
            DegradationAction.ASK_USER,
        ):
            self.last_binding = decision.binding_kind
            return ModelResponse(result=[AIMessage(content=self._stop_message(decision))])

        if should_remind(
            decision,
            snapshot,
            self.policy,
            self.reminder_state,
            model_call_count=self.model_calls,
        ):
            notice = format_reminder(snapshot, decision, scope="run")
            request = request.override(
                messages=[
                    *request.messages,
                    HumanMessage(content=notice.message, name="resguard_budget"),
                ]
            )
            self.reminder_state.mark(decision, snapshot, model_call_count=self.model_calls)

        pricing = self.pricing.resolve(self._model_name(request.model))
        input_tokens = self._estimate_input_tokens(request)
        estimated_usd = estimate_request_cost(
            pricing,
            input_tokens=input_tokens,
            max_output_tokens=self.estimate_output_tokens,
        )
        reservation_amount = ResourceAmount(
            {
                ResourceKind.USD: float(estimated_usd),
                ResourceKind.LLM_INPUT_TOKENS: input_tokens,
                ResourceKind.LLM_OUTPUT_TOKENS: self.estimate_output_tokens,
            }
        )
        try:
            reservation = self.ledger.reserve(
                reservation_amount,
                reservation_id=f"model:{self.run_id}:{self.model_calls}",
            )
        except ReservationDenied as exc:
            self.denied_calls += 1
            self.last_binding = exc.binding
            return ModelResponse(result=[AIMessage(content=self._denied_message(exc))])

        try:
            response = handler(request)
        except Exception:
            self.ledger.release(reservation)
            raise

        actual = self._actual_model_amount(response, pricing)
        self.ledger.commit(reservation, actual)
        self.llm_usd += (actual or reservation_amount).get(ResourceKind.USD)
        self.model_calls += 1
        return response

    # ------------------------------------------------------------------- tool
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Any],
    ) -> ToolMessage | Any:
        tool_call = request.tool_call
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        call_id = tool_call.get("id", "")
        profile = self.tool_profiles.get(name, self.unknown_tool_profile)

        if self.cache is not None and profile.validate_cacheable():
            cached = self.cache.get(name, args, ctx=self.cache_context)
            if cached is not None:
                self.cache_hits += 1
                return ToolMessage(content=str(cached), tool_call_id=call_id, name=name)

        amount = profile.per_call_amount()
        try:
            reservation = self.ledger.reserve(
                amount,
                reservation_id=f"tool:{self.run_id}:{call_id}",
            )
        except ReservationDenied as exc:
            self.denied_calls += 1
            self.last_binding = exc.binding
            return ToolMessage(
                content=self._denied_message(exc),
                tool_call_id=call_id,
                name=name,
                status="error",
            )

        try:
            lease = self.ledger.acquire_concurrency(profile.concurrency_slots)
        except ConcurrencyDenied as exc:
            self.ledger.release(reservation)
            self.denied_calls += 1
            self.concurrency_denials += 1
            self.last_binding = ResourceKind.CONCURRENCY
            return ToolMessage(
                content=f"[budget] concurrency limit reached: {exc}",
                tool_call_id=call_id,
                name=name,
                status="error",
            )

        self.peak_concurrency = max(
            self.peak_concurrency, self._primary_snapshot().concurrency_in_use
        )

        try:
            result = handler(request)
        except Exception:
            self.ledger.release(reservation)
            lease.release()
            raise

        actual = amount
        if isinstance(result, ToolMessage):
            reported = result.additional_kwargs.get("resource_usage")
            if isinstance(reported, dict):
                actual = ResourceAmount(
                    {
                        **amount.amounts,
                        **{
                            kind: float(value)
                            for kind, value in reported.items()
                            if isinstance(kind, ResourceKind)
                        },
                    }
                )
        self.ledger.commit(reservation, actual)
        lease.release()
        self.tool_calls += 1
        self.tool_usd += actual.get(ResourceKind.USD)

        if (
            self.cache is not None
            and profile.validate_cacheable()
            and isinstance(result, ToolMessage)
            and result.status != "error"
        ):
            self.cache.set(
                name,
                args,
                result.content,
                ctx=self.cache_context,
                ttl_seconds=profile.ttl_seconds,
            )
        return result

    # ---------------------------------------------------------------- helpers
    def snapshot(self) -> ResourceSnapshot:
        """Public snapshot for the recorder/demo; the middleware is per run."""
        return self._primary_snapshot()

    def _primary_snapshot(self) -> ResourceSnapshot:
        snapshot = self.ledger.snapshot()
        if isinstance(snapshot, CompositeSnapshot):
            if BudgetScope.RUN in snapshot.scopes:
                return snapshot.scopes[BudgetScope.RUN]
            return next(iter(snapshot.scopes.values()))
        return snapshot

    @staticmethod
    def _model_name(model: Any) -> str:
        name = getattr(model, "model_name", None)
        if isinstance(name, str) and name:
            return name
        bound = getattr(model, "model", None)
        bound_name = getattr(bound, "model_name", None)
        if isinstance(bound_name, str) and bound_name:
            return bound_name
        return "unknown"

    @staticmethod
    def _estimate_input_tokens(request: ModelRequest) -> int:
        total = estimate_history_tokens(request.messages)
        if request.system_message is not None:
            total += estimate_tokens(str(request.system_message.content))
        total += estimate_tool_schema_tokens(request.tools or [])
        return total

    @staticmethod
    def _response_message(response: Any) -> AIMessage | None:
        if isinstance(response, AIMessage):
            return response
        if isinstance(response, ModelResponse):
            for message in reversed(response.result):
                if isinstance(message, AIMessage):
                    return message
            return None
        extended = getattr(response, "model_response", None)
        if isinstance(extended, ModelResponse):
            for message in reversed(extended.result):
                if isinstance(message, AIMessage):
                    return message
        return None

    def _actual_model_amount(
        self, response: Any, pricing: Any
    ) -> ResourceAmount | None:
        message = self._response_message(response)
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
            "before the next model call. Raise the budget or ask the user to "
            "continue with a larger budget."
        )

    @staticmethod
    def _denied_message(exc: ReservationDenied) -> str:
        return (
            f"[budget] resource {exc.binding.value} exhausted in scope "
            f"{exc.scope_key}; the call was not executed. Prefer cached "
            "results or ask the user to continue with a larger budget."
        )
