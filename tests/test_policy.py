from __future__ import annotations

import pytest

from resguard.ledger import InMemoryLedger
from resguard.policy import DegradationAction, GuardPolicy, decide, tool_decision
from resguard.resources import ResourceAmount, ResourceKind, ResourceLimits


def snapshot(usd: float, *, limit: float = 1.0):
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: limit}))
    if usd:
        ledger.reserve(ResourceAmount({ResourceKind.USD: usd}))
    return ledger.snapshot()


def test_policy_defaults() -> None:
    policy = GuardPolicy()

    assert policy.soft_thresholds == (0.5, 0.8, 0.95)
    assert policy.hard_limit_action is DegradationAction.ASK_USER
    assert policy.allow_cheaper_model is False


def test_policy_rejects_unsorted_thresholds() -> None:
    with pytest.raises(ValueError):
        GuardPolicy(soft_thresholds=(0.8, 0.5))


def test_policy_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(ValueError):
        GuardPolicy(soft_thresholds=(0.5, 1.2))


def test_decide_continue_below_first_threshold() -> None:
    decision = decide(snapshot(0.3), GuardPolicy())

    assert decision.action is DegradationAction.CONTINUE
    assert decision.level == "ok"
    assert decision.utilization == pytest.approx(0.3)


def test_decide_remind_at_first_threshold() -> None:
    decision = decide(snapshot(0.5), GuardPolicy())

    assert decision.action is DegradationAction.REMIND
    assert decision.level == "warning"
    assert decision.threshold == 0.5


def test_decide_remind_at_critical_threshold() -> None:
    decision = decide(snapshot(0.8), GuardPolicy())

    assert decision.action is DegradationAction.REMIND
    assert decision.level == "critical"


def test_decide_hard_limit_uses_policy_action() -> None:
    decision = decide(snapshot(1.0), GuardPolicy(hard_limit_action=DegradationAction.STOP))

    assert decision.action is DegradationAction.STOP
    assert decision.level == "hard"
    assert decision.binding_kind is ResourceKind.USD


def test_decide_cheaper_model_when_enabled() -> None:
    policy = GuardPolicy(allow_cheaper_model=True, cheaper_model_after_fraction=0.8)
    decision = decide(snapshot(0.8), policy)

    assert decision.action is DegradationAction.CHEAPER_MODEL
    assert decision.level == "critical"


def test_tool_decision_global_limit() -> None:
    policy = GuardPolicy(max_tool_calls=5)

    assert tool_decision(policy, total_tool_calls=4, tool_name="search", per_tool_calls=4).action is DegradationAction.CONTINUE
    assert tool_decision(policy, total_tool_calls=5, tool_name="search", per_tool_calls=5).action is DegradationAction.STOP


def test_tool_decision_per_tool_limit() -> None:
    policy = GuardPolicy(per_tool_limits={"parse_pdf": 2})

    decision = tool_decision(
        policy, total_tool_calls=1, tool_name="parse_pdf", per_tool_calls=2
    )

    assert decision.action is DegradationAction.STOP
    assert decision.binding_kind is None


def test_policy_to_dict_round_trip() -> None:
    policy = GuardPolicy(
        soft_thresholds=(0.6, 0.9),
        hard_limit_action=DegradationAction.STOP,
        max_tool_calls=10,
        per_tool_limits={"parse_pdf": 2},
        allow_cheaper_model=True,
        cheaper_model="gpt-4o-mini",
    )

    restored = GuardPolicy.from_dict(policy.to_dict())

    assert restored == policy
