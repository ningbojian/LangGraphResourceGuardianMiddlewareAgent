"""Canary plans and drift detection for offline resource-policy changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from resguard.batch import BatchInsights
from resguard.optimizer import EvaluationResult
from resguard.proposer import PolicyProposal


@dataclass(frozen=True)
class CanaryPlan:
    proposal_id: str
    traffic_percent: float
    duration_hours: int
    observe_metrics: tuple[str, ...]
    rollback_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "traffic_percent": self.traffic_percent,
            "duration_hours": self.duration_hours,
            "observe_metrics": list(self.observe_metrics),
            "rollback_conditions": list(self.rollback_conditions),
        }


def build_canary_plan(
    proposal: PolicyProposal,
    evaluation: EvaluationResult | None = None,
    *,
    traffic_percent: float = 0.05,
    duration_hours: int = 24,
) -> CanaryPlan:
    conditions = [
        "success_rate drop > 2 percentage points",
        "false_positive_rate increase > 2 percentage points",
        "quality_rate drop > 2 percentage points",
        "any resource dimension regression > 2%",
    ]
    if evaluation is not None:
        conditions.extend(evaluation.reasons)
    return CanaryPlan(
        proposal_id=proposal.id,
        traffic_percent=traffic_percent,
        duration_hours=duration_hours,
        observe_metrics=(
            "success_rate",
            "false_positive_rate",
            "quality_rate",
            "resource_per_success",
            "guard_overhead_ms",
        ),
        rollback_conditions=tuple(dict.fromkeys(conditions)),
    )


@dataclass(frozen=True)
class DriftThresholds:
    success_drop: float = 0.02
    false_positive_increase: float = 0.02
    quality_drop: float = 0.02
    resource_regression: float = 0.20


@dataclass
class DriftReport:
    drift_detected: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
        }


def detect_drift(
    baseline: BatchInsights,
    recent: BatchInsights,
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    thresholds = thresholds or DriftThresholds()
    reasons: list[str] = []
    deltas = {
        "success_rate": recent.success_rate - baseline.success_rate,
        "false_positive_rate": recent.false_positive_rate - baseline.false_positive_rate,
        "quality_rate": recent.quality_rate - baseline.quality_rate,
    }
    if recent.success_rate < baseline.success_rate - thresholds.success_drop:
        reasons.append("success_rate_drift")
    if (
        recent.false_positive_rate
        > baseline.false_positive_rate + thresholds.false_positive_increase
    ):
        reasons.append("false_positive_rate_drift")
    if recent.quality_rate < baseline.quality_rate - thresholds.quality_drop:
        reasons.append("quality_rate_drift")

    for dimension in ("usd", "db_cu", "db_queries", "api_quota", "tool_calls"):
        before = baseline.resource_per_success.get(dimension, 0.0)
        after = recent.resource_per_success.get(dimension, 0.0)
        if before <= 0:
            continue
        delta = (after - before) / before
        deltas[dimension] = delta
        if delta > thresholds.resource_regression:
            reasons.append(f"{dimension}_drift")

    return DriftReport(drift_detected=bool(reasons), reasons=reasons, deltas=deltas)
