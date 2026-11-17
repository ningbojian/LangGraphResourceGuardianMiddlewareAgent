from __future__ import annotations

import pytest

from resguard.ledger import InMemoryLedger
from resguard.policy import DegradationAction, GuardPolicy, PolicyDecision
from resguard.reminder import (
    BudgetNotice,
    ReminderState,
    format_reminder,
    should_remind,
)
from resguard.resources import ResourceAmount, ResourceKind, ResourceLimits


def make_snapshot(
    *,
    usd: float = 0.0,
    db_cu: float = 0.0,
    db_queries: int = 0,
    concurrency: int = 0,
):
    ledger = InMemoryLedger(
        ResourceLimits(
            {
                ResourceKind.USD: 1.0,
                ResourceKind.DB_CU: 10.0,
                ResourceKind.DB_QUERIES: 20,
                ResourceKind.CONCURRENCY: 4,
            }
        )
    )
    ledger.reserve(
        ResourceAmount(
            {
                ResourceKind.USD: usd,
                ResourceKind.DB_CU: db_cu,
                ResourceKind.DB_QUERIES: db_queries,
            }
        )
    )
    if concurrency:
        ledger.acquire_concurrency(concurrency)
    return ledger.snapshot()


def reminder_decision(threshold: float = 0.8) -> PolicyDecision:
    return PolicyDecision(
        action=DegradationAction.REMIND,
        level="warning",
        utilization=threshold,
        binding_kind=ResourceKind.DB_CU,
        threshold=threshold,
    )


def test_format_reminder_includes_multiple_dimensions() -> None:
    snapshot = make_snapshot(usd=0.42, db_cu=8.0, db_queries=12, concurrency=4)

    notice = format_reminder(snapshot, reminder_decision(), scope="run")

    assert isinstance(notice, BudgetNotice)
    assert notice.level == "warning"
    assert "usd=0.42/1.00" in notice.message
    assert "db_cu=8.00/10.00" in notice.message
    assert "db_queries=12/20" in notice.message
    assert "concurrency=4/4" in notice.message
    assert "binding=db_cu" in notice.message
    assert "reuse_cached_results" in notice.suggested_actions


def test_should_remind_first_time() -> None:
    snapshot = make_snapshot(usd=0.42, db_cu=8.0)
    decision = reminder_decision()

    assert should_remind(
        decision, snapshot, GuardPolicy(), ReminderState(), model_call_count=1
    )


def test_should_not_remind_when_decision_is_continue() -> None:
    snapshot = make_snapshot(usd=0.2)
    decision = PolicyDecision(
        action=DegradationAction.CONTINUE, level="ok", utilization=0.2
    )

    assert not should_remind(
        decision, snapshot, GuardPolicy(), ReminderState(), model_call_count=1
    )


def test_should_remind_on_higher_threshold() -> None:
    snapshot = make_snapshot(usd=0.85, db_cu=8.5)
    state = ReminderState(last_threshold=0.5, last_spent_usd=0.5, last_model_call=1)

    assert should_remind(
        reminder_decision(0.8), snapshot, GuardPolicy(), state, model_call_count=2
    )


def test_should_not_remind_same_threshold_within_cooldown() -> None:
    snapshot = make_snapshot(usd=0.405, db_cu=8.0)
    state = ReminderState(last_threshold=0.8, last_spent_usd=0.4, last_model_call=10)
    policy = GuardPolicy(reminder_cooldown_model_calls=2, reminder_re_notice_delta=0.02)

    assert not should_remind(
        reminder_decision(0.8), snapshot, policy, state, model_call_count=11
    )


def test_should_remind_same_threshold_after_cooldown_and_delta() -> None:
    snapshot = make_snapshot(usd=0.45, db_cu=8.5)
    state = ReminderState(last_threshold=0.8, last_spent_usd=0.4, last_model_call=10)
    policy = GuardPolicy(reminder_cooldown_model_calls=2, reminder_re_notice_delta=0.02)

    assert should_remind(
        reminder_decision(0.8), snapshot, policy, state, model_call_count=12
    )


def test_reminder_state_mark_updates_fields() -> None:
    state = ReminderState()
    snapshot = make_snapshot(usd=0.42)

    state.mark(reminder_decision(0.8), snapshot, model_call_count=7)

    assert state.last_threshold == 0.8
    assert state.last_spent_usd == pytest.approx(0.42)
    assert state.last_model_call == 7


def test_notice_to_dict_is_serializable() -> None:
    snapshot = make_snapshot(usd=0.42, db_cu=8.0)
    notice = format_reminder(snapshot, reminder_decision(), scope="run")

    data = notice.to_dict()

    assert data["level"] == "warning"
    assert data["scope"] == "run"
    assert data["binding_kind"] == "db_cu"
    assert isinstance(data["suggested_actions"], list)
