"""Static guard policy: thresholds, hard-limit actions, and degradation choices.

The online guard only executes an approved policy. No LLM decision happens here;
the offline optimizer proposes new policies, and a human approves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from resguard.ledger import ResourceSnapshot
from resguard.resources import RESOURCE_SPECS, ResourceKind, ResourceMode


class DegradationAction(str, Enum):
    CONTINUE = "continue"
    REMIND = "remind"
    CACHE_FIRST = "cache_first"
    CHEAPER_MODEL = "cheaper_model"
    ASK_USER = "ask_user"
    STOP = "stop"


@dataclass(frozen=True)
class PolicyDecision:
    action: DegradationAction
    level: str
    utilization: float
    binding_kind: ResourceKind | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class GuardPolicy:
    soft_thresholds: tuple[float, ...] = (0.5, 0.8, 0.95)
    hard_limit_action: DegradationAction = DegradationAction.ASK_USER
    max_tool_calls: int | None = None
    per_tool_limits: Mapping[str, int] = field(default_factory=dict)
    allow_cheaper_model: bool = False
    cheaper_model_after_fraction: float = 0.8
    cheaper_model: str | None = None
    wrapup_reserve_fraction: float = 0.1
    reminder_re_notice_delta: float = 0.02
    reminder_cooldown_model_calls: int = 2

    def __post_init__(self) -> None:
        thresholds = tuple(float(t) for t in self.soft_thresholds)
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError("soft_thresholds must be sorted ascending")
        if any(t < 0 or t > 1 for t in thresholds):
            raise ValueError("soft_thresholds must be within [0, 1]")
        if not 0 <= self.cheaper_model_after_fraction <= 1:
            raise ValueError("cheaper_model_after_fraction must be within [0, 1]")
        if not 0 <= self.wrapup_reserve_fraction <= 1:
            raise ValueError("wrapup_reserve_fraction must be within [0, 1]")
        if self.max_tool_calls is not None and self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")
        for tool, limit in self.per_tool_limits.items():
            if limit < 0:
                raise ValueError(f"per-tool limit for {tool} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "soft_thresholds": list(self.soft_thresholds),
            "hard_limit_action": self.hard_limit_action.value,
            "max_tool_calls": self.max_tool_calls,
            "per_tool_limits": dict(self.per_tool_limits),
            "allow_cheaper_model": self.allow_cheaper_model,
            "cheaper_model_after_fraction": self.cheaper_model_after_fraction,
            "cheaper_model": self.cheaper_model,
            "wrapup_reserve_fraction": self.wrapup_reserve_fraction,
            "reminder_re_notice_delta": self.reminder_re_notice_delta,
            "reminder_cooldown_model_calls": self.reminder_cooldown_model_calls,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GuardPolicy":
        return cls(
            soft_thresholds=tuple(data.get("soft_thresholds", (0.5, 0.8, 0.95))),
            hard_limit_action=DegradationAction(
                data.get("hard_limit_action", DegradationAction.ASK_USER.value)
            ),
            max_tool_calls=data.get("max_tool_calls"),
            per_tool_limits=dict(data.get("per_tool_limits", {})),
            allow_cheaper_model=bool(data.get("allow_cheaper_model", False)),
            cheaper_model_after_fraction=float(
                data.get("cheaper_model_after_fraction", 0.8)
            ),
            cheaper_model=data.get("cheaper_model"),
            wrapup_reserve_fraction=float(data.get("wrapup_reserve_fraction", 0.1)),
            reminder_re_notice_delta=float(data.get("reminder_re_notice_delta", 0.02)),
            reminder_cooldown_model_calls=int(
                data.get("reminder_cooldown_model_calls", 2)
            ),
        )


def _max_utilization(snapshot: ResourceSnapshot) -> float:
    maximum = 0.0
    for kind in ResourceKind:
        if RESOURCE_SPECS[kind].mode is not ResourceMode.CUMULATIVE:
            continue
        if snapshot.limits.get(kind) is None:
            continue
        maximum = max(maximum, snapshot.utilization(kind))
    return maximum


def decide(snapshot: ResourceSnapshot, policy: GuardPolicy) -> PolicyDecision:
    binding = snapshot.binding()
    if binding is not None:
        return PolicyDecision(
            action=policy.hard_limit_action,
            level="hard",
            utilization=snapshot.utilization(binding),
            binding_kind=binding,
        )

    utilization = _max_utilization(snapshot)
    if policy.allow_cheaper_model and utilization >= policy.cheaper_model_after_fraction:
        return PolicyDecision(
            action=DegradationAction.CHEAPER_MODEL,
            level="critical",
            utilization=utilization,
            threshold=policy.cheaper_model_after_fraction,
        )

    crossed: float | None = None
    for threshold in policy.soft_thresholds:
        if utilization >= threshold:
            crossed = threshold
    if crossed is None:
        return PolicyDecision(
            action=DegradationAction.CONTINUE,
            level="ok",
            utilization=utilization,
        )

    level = "warning" if crossed == policy.soft_thresholds[0] else "critical"
    return PolicyDecision(
        action=DegradationAction.REMIND,
        level=level,
        utilization=utilization,
        threshold=crossed,
    )


def tool_decision(
    policy: GuardPolicy,
    *,
    total_tool_calls: int,
    tool_name: str,
    per_tool_calls: int,
) -> PolicyDecision:
    if policy.max_tool_calls is not None and total_tool_calls >= policy.max_tool_calls:
        utilization = (
            total_tool_calls / policy.max_tool_calls if policy.max_tool_calls else 1.0
        )
        return PolicyDecision(
            action=DegradationAction.STOP,
            level="hard",
            utilization=utilization,
            threshold=float(policy.max_tool_calls),
        )
    limit = policy.per_tool_limits.get(tool_name)
    if limit is not None and per_tool_calls >= limit:
        utilization = per_tool_calls / limit if limit else 1.0
        return PolicyDecision(
            action=DegradationAction.STOP,
            level="hard",
            utilization=utilization,
            threshold=float(limit),
        )
    return PolicyDecision(action=DegradationAction.CONTINUE, level="ok", utilization=0.0)
