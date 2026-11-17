"""Soft, ephemeral multi-resource reminders.

The reminder is injected into the model request only; it is never persisted in
agent state. Format is compact and stable so the large prompt prefix remains
cacheable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resguard.ledger import ResourceSnapshot
from resguard.policy import DegradationAction, GuardPolicy, PolicyDecision
from resguard.resources import RESOURCE_SPECS, ResourceKind, ResourceMode


@dataclass(frozen=True)
class BudgetNotice:
    level: str
    scope: str
    used_fraction: float
    binding_kind: ResourceKind | None
    message: str
    suggested_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "scope": self.scope,
            "used_fraction": self.used_fraction,
            "binding_kind": self.binding_kind.value if self.binding_kind else None,
            "message": self.message,
            "suggested_actions": list(self.suggested_actions),
        }


@dataclass
class ReminderState:
    last_threshold: float | None = None
    last_spent_usd: float = 0.0
    last_model_call: int = -10_000

    def mark(
        self,
        decision: PolicyDecision,
        snapshot: ResourceSnapshot,
        *,
        model_call_count: int,
    ) -> None:
        if decision.threshold is not None:
            self.last_threshold = decision.threshold
        self.last_spent_usd = snapshot.total.get(ResourceKind.USD)
        self.last_model_call = model_call_count


def should_remind(
    decision: PolicyDecision,
    snapshot: ResourceSnapshot,
    policy: GuardPolicy,
    state: ReminderState,
    *,
    model_call_count: int,
) -> bool:
    if decision.action is not DegradationAction.REMIND or decision.threshold is None:
        return False
    threshold = decision.threshold
    if state.last_threshold is None:
        return True
    if threshold > state.last_threshold:
        return True
    if threshold < state.last_threshold:
        return False
    if model_call_count - state.last_model_call < policy.reminder_cooldown_model_calls:
        return False
    spent_usd = snapshot.total.get(ResourceKind.USD)
    return spent_usd - state.last_spent_usd >= policy.reminder_re_notice_delta


def _format_limit_value(kind: ResourceKind, value: float) -> str:
    if kind in (ResourceKind.USD, ResourceKind.DB_CU):
        return f"{value:.2f}"
    if kind in (
        ResourceKind.DB_QUERIES,
        ResourceKind.API_QUOTA,
        ResourceKind.TOOL_CALLS,
        ResourceKind.LLM_INPUT_TOKENS,
        ResourceKind.LLM_OUTPUT_TOKENS,
        ResourceKind.WALL_CLOCK_MS,
        ResourceKind.CHECKPOINT_BYTES,
    ):
        return f"{value:.0f}"
    return f"{value:g}"


_ORDER = (
    ResourceKind.USD,
    ResourceKind.LLM_INPUT_TOKENS,
    ResourceKind.LLM_OUTPUT_TOKENS,
    ResourceKind.TOOL_CALLS,
    ResourceKind.DB_QUERIES,
    ResourceKind.DB_CU,
    ResourceKind.API_QUOTA,
    ResourceKind.WALL_CLOCK_MS,
    ResourceKind.CHECKPOINT_BYTES,
)


def _suggested_actions(binding: ResourceKind | None) -> tuple[str, ...]:
    if binding is ResourceKind.CONCURRENCY:
        return ("wait_for_slot", "reduce_parallelism", "ask_user_or_stop")
    if binding in (ResourceKind.USD, ResourceKind.DB_CU, ResourceKind.DB_QUERIES, ResourceKind.API_QUOTA):
        return ("reuse_cached_results", "avoid_expensive_tool", "ask_user_or_stop")
    return ("reuse_cached_results", "ask_user_or_stop")


def format_reminder(
    snapshot: ResourceSnapshot,
    decision: PolicyDecision,
    *,
    scope: str = "run",
) -> BudgetNotice:
    parts: list[str] = []
    for kind in _ORDER:
        if RESOURCE_SPECS[kind].mode is not ResourceMode.CUMULATIVE:
            continue
        limit = snapshot.limits.get(kind)
        if limit is None:
            continue
        used = snapshot.total.get(kind)
        if used == 0 and kind is not ResourceKind.USD:
            continue
        parts.append(
            f"{kind.value}={_format_limit_value(kind, used)}/"
            f"{_format_limit_value(kind, limit)}"
        )

    concurrency_limit = snapshot.limits.get(ResourceKind.CONCURRENCY)
    if concurrency_limit is not None:
        parts.append(
            f"concurrency={snapshot.concurrency_in_use}/{concurrency_limit:g}"
        )

    binding = decision.binding_kind.value if decision.binding_kind else "none"
    message = (
        f"[budget] scope={scope} level={decision.level} "
        f"used={decision.utilization:.1%} binding={binding}; "
        + "; ".join(parts)
        + ". Prefer cached results or ask the user before more expensive calls."
    )
    return BudgetNotice(
        level=decision.level,
        scope=scope,
        used_fraction=decision.utilization,
        binding_kind=decision.binding_kind,
        message=message,
        suggested_actions=_suggested_actions(decision.binding_kind),
    )
