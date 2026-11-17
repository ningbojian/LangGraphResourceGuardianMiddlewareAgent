"""Deterministic batch analysis of completed sessions.

The analyzer is the optimizer agent's first tool: it turns a batch of
``SessionRecord`` objects into ``BatchInsights`` without calling an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from resguard.metrics import SessionMetrics
from resguard.resources import ResourceKind
from resguard.session import SessionRecord


@dataclass
class BatchInsights:
    session_count: int = 0
    total_metrics: SessionMetrics = field(default_factory=SessionMetrics)
    by_query_type: dict[str, SessionMetrics] = field(default_factory=dict)
    termination_reasons: dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    false_positive_rate: float = 0.0
    quality_rate: float = 0.0
    labeled_sessions: int = 0
    proxy_label: bool = False
    resource_per_success: dict[str, float] = field(default_factory=dict)
    hotspots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_count": self.session_count,
            "total_metrics": self.total_metrics.to_dict(),
            "by_query_type": {
                query_type: metrics.to_dict()
                for query_type, metrics in self.by_query_type.items()
            },
            "termination_reasons": dict(self.termination_reasons),
            "success_rate": self.success_rate,
            "false_positive_rate": self.false_positive_rate,
            "quality_rate": self.quality_rate,
            "labeled_sessions": self.labeled_sessions,
            "proxy_label": self.proxy_label,
            "resource_per_success": dict(self.resource_per_success),
            "hotspots": list(self.hotspots),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchInsights":
        return cls(
            session_count=int(data.get("session_count", 0)),
            total_metrics=SessionMetrics.from_dict(data.get("total_metrics") or {}),
            by_query_type={
                query_type: SessionMetrics.from_dict(metrics)
                for query_type, metrics in (data.get("by_query_type") or {}).items()
            },
            termination_reasons=dict(data.get("termination_reasons") or {}),
            success_rate=float(data.get("success_rate", 0.0)),
            false_positive_rate=float(data.get("false_positive_rate", 0.0)),
            quality_rate=float(data.get("quality_rate", 0.0)),
            labeled_sessions=int(data.get("labeled_sessions", 0)),
            proxy_label=bool(data.get("proxy_label", False)),
            resource_per_success={
                key: float(value)
                for key, value in (data.get("resource_per_success") or {}).items()
            },
            hotspots=list(data.get("hotspots") or []),
        )


class BatchAnalyzer:
    def analyze(self, records: Iterable[SessionRecord]) -> BatchInsights:
        insights = BatchInsights()
        successes = 0
        labeled = 0
        for record in records:
            insights.session_count += 1
            insights.total_metrics = insights.total_metrics.merge(record.metrics)
            insights.by_query_type[record.query_type] = insights.by_query_type.get(
                record.query_type, SessionMetrics()
            ).merge(record.metrics)
            insights.termination_reasons[record.termination_reason] = (
                insights.termination_reasons.get(record.termination_reason, 0) + 1
            )
            if record.success is not None:
                labeled += 1
                successes += 1 if record.success else 0

        insights.labeled_sessions = labeled
        insights.proxy_label = labeled < insights.session_count
        insights.success_rate = successes / labeled if labeled else 0.0

        vectors = insights.total_metrics.resource_amount()
        divisor = successes if successes else 1
        insights.resource_per_success = {
            kind.value: vectors.get(kind) / divisor
            for kind in (
                ResourceKind.USD,
                ResourceKind.DB_CU,
                ResourceKind.DB_QUERIES,
                ResourceKind.API_QUOTA,
                ResourceKind.TOOL_CALLS,
            )
        }

        hotspots: list[str] = []
        for reason, count in insights.termination_reasons.items():
            if reason != "completed" and count > 0:
                hotspots.append(reason)
        if insights.total_metrics.duplicate_tool_calls > 0:
            hotspots.append("duplicate_tool_calls")
        if insights.total_metrics.peak_concurrency > 0:
            hotspots.append("concurrency")
        if insights.total_metrics.checkpoint_bytes > 0:
            hotspots.append("checkpoint_bytes")
        insights.hotspots = hotspots
        return insights
