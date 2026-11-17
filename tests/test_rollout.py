from __future__ import annotations

from resguard.batch import BatchInsights
from resguard.optimizer import EvaluationResult
from resguard.proposer import PolicyProposal
from resguard.rollout import build_canary_plan, detect_drift


def insights(
    *,
    success_rate: float = 1.0,
    false_positive_rate: float = 0.0,
    quality_rate: float = 1.0,
    usd: float = 0.1,
) -> BatchInsights:
    return BatchInsights(
        session_count=10,
        success_rate=success_rate,
        false_positive_rate=false_positive_rate,
        quality_rate=quality_rate,
        resource_per_success={"usd": usd, "db_cu": 1.0, "tool_calls": 2.0},
    )


def test_build_canary_plan() -> None:
    proposal = PolicyProposal(id="p1", name="cache", description="")
    evaluation = EvaluationResult(proposal_id="p1", accepted=True, reasons=["r"])

    plan = build_canary_plan(proposal, evaluation)

    assert plan.proposal_id == "p1"
    assert plan.traffic_percent == 0.05
    assert "success_rate" in plan.observe_metrics
    assert "r" in plan.rollback_conditions
    assert plan.to_dict()["duration_hours"] == 24


def test_detect_drift_no_drift() -> None:
    report = detect_drift(insights(), insights())

    assert report.drift_detected is False
    assert report.reasons == []


def test_detect_drift_success_drop() -> None:
    report = detect_drift(insights(success_rate=1.0), insights(success_rate=0.9))

    assert report.drift_detected is True
    assert "success_rate_drift" in report.reasons


def test_detect_drift_quality_drop() -> None:
    report = detect_drift(insights(quality_rate=1.0), insights(quality_rate=0.8))

    assert "quality_rate_drift" in report.reasons


def test_detect_drift_resource_regression() -> None:
    report = detect_drift(insights(usd=0.1), insights(usd=0.2))

    assert "usd_drift" in report.reasons
    assert report.deltas["usd"] == 1.0


def test_detect_drift_false_positive_increase() -> None:
    report = detect_drift(
        insights(false_positive_rate=0.0),
        insights(false_positive_rate=0.1),
    )

    assert "false_positive_rate_drift" in report.reasons
