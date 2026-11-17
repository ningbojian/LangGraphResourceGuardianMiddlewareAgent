"""Pareto gate and audit log for proposals.

The gate is deterministic and is the only authority that can turn an LLM
proposal into an accepted candidate. A human still approves deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from resguard.batch import BatchInsights


@dataclass
class EvaluationResult:
    proposal_id: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    improvements: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "improvements": dict(self.improvements),
            "regressions": dict(self.regressions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        return cls(
            proposal_id=str(data["proposal_id"]),
            accepted=bool(data.get("accepted", False)),
            reasons=list(data.get("reasons") or []),
            improvements=dict(data.get("improvements") or {}),
            regressions=dict(data.get("regressions") or {}),
        )


@dataclass(frozen=True)
class ParetoGate:
    min_improvement: float = 0.10
    success_tolerance: float = 0.02
    false_positive_tolerance: float = 0.02
    quality_tolerance: float = 0.02
    regression_tolerance: float = 0.02
    dimensions: tuple[str, ...] = ("usd", "db_cu", "db_queries", "api_quota", "tool_calls")

    def evaluate(
        self,
        proposal_id: str,
        *,
        baseline: BatchInsights,
        candidate: BatchInsights,
    ) -> EvaluationResult:
        reasons: list[str] = []
        improvements: dict[str, float] = {}
        regressions: dict[str, float] = {}

        if candidate.success_rate < baseline.success_rate - self.success_tolerance:
            reasons.append("success_rate_regression")
        if (
            candidate.false_positive_rate
            > baseline.false_positive_rate + self.false_positive_tolerance
        ):
            reasons.append("false_positive_rate_regression")
        if candidate.quality_rate < baseline.quality_rate - self.quality_tolerance:
            reasons.append("quality_regression")

        for dimension in self.dimensions:
            before = baseline.resource_per_success.get(dimension, 0.0)
            after = candidate.resource_per_success.get(dimension, 0.0)
            if before <= 0:
                continue
            delta = (before - after) / before
            if delta >= self.min_improvement:
                improvements[dimension] = delta
            elif delta <= -self.regression_tolerance:
                regressions[dimension] = delta

        if not improvements:
            reasons.append("no_resource_improvement")
        if regressions:
            reasons.append("regression:" + ",".join(sorted(regressions)))

        return EvaluationResult(
            proposal_id=proposal_id,
            accepted=not reasons,
            reasons=reasons,
            improvements=improvements,
            regressions=regressions,
        )


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
