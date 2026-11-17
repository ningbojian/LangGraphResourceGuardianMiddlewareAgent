from __future__ import annotations

import pytest

from resguard.batch import BatchAnalyzer
from resguard.metrics import SessionMetrics
from resguard.session import SessionRecord


def record(
    session_id: str,
    *,
    query_type: str = "refund",
    success: bool | None = True,
    termination_reason: str = "completed",
    metrics: SessionMetrics | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        query_type=query_type,
        metrics=metrics or SessionMetrics(tool_calls=2, llm_usd=0.01, tool_usd=0.01),
        success=success,
        label_source="human" if success is not None else "none",
        termination_reason=termination_reason,
    )


def test_analyze_aggregates_counts_and_costs() -> None:
    insights = BatchAnalyzer().analyze(
        [
            record("s1", metrics=SessionMetrics(tool_calls=2, llm_usd=0.01, tool_usd=0.01)),
            record("s2", metrics=SessionMetrics(tool_calls=3, llm_usd=0.02, tool_usd=0.02)),
        ]
    )

    assert insights.session_count == 2
    assert insights.total_metrics.tool_calls == 5
    assert insights.total_metrics.llm_usd == 0.03
    assert insights.total_metrics.tool_usd == 0.03


def test_success_rate_and_proxy_label() -> None:
    insights = BatchAnalyzer().analyze(
        [record("s1", success=True), record("s2", success=False), record("s3", success=None)]
    )

    assert insights.success_rate == 0.5
    assert insights.labeled_sessions == 2
    assert insights.proxy_label is True


def test_resource_per_success() -> None:
    insights = BatchAnalyzer().analyze(
        [
            record("s1", success=True, metrics=SessionMetrics(llm_usd=0.10, db_cu=2.0)),
            record("s2", success=True, metrics=SessionMetrics(llm_usd=0.20, db_cu=4.0)),
        ]
    )

    assert insights.resource_per_success["usd"] == pytest.approx(0.15)
    assert insights.resource_per_success["db_cu"] == pytest.approx(3.0)


def test_by_query_type_and_termination_reason() -> None:
    insights = BatchAnalyzer().analyze(
        [
            record("s1", query_type="refund", termination_reason="completed"),
            record("s2", query_type="refund", termination_reason="usd_exhausted"),
            record("s3", query_type="pdf", termination_reason="completed"),
        ]
    )

    assert insights.by_query_type["refund"].tool_calls == 4
    assert insights.by_query_type["pdf"].tool_calls == 2
    assert insights.termination_reasons["completed"] == 2
    assert insights.termination_reasons["usd_exhausted"] == 1


def test_hotspots_are_deterministic() -> None:
    insights = BatchAnalyzer().analyze(
        [
            record("s1", termination_reason="usd_exhausted"),
            record("s2", termination_reason="db_cu_exhausted"),
            record("s3", metrics=SessionMetrics(duplicate_tool_calls=3)),
        ]
    )

    assert "usd_exhausted" in insights.hotspots
    assert "db_cu_exhausted" in insights.hotspots
    assert "duplicate_tool_calls" in insights.hotspots


def test_to_dict_is_serializable() -> None:
    insights = BatchAnalyzer().analyze([record("s1")])

    data = insights.to_dict()

    assert data["session_count"] == 1
    assert isinstance(data["total_metrics"], dict)
    assert data["proxy_label"] is False


def test_batch_insights_round_trip() -> None:
    original = BatchAnalyzer().analyze(
        [record("s1", query_type="refund"), record("s2", query_type="pdf")]
    )

    restored = type(original).from_dict(original.to_dict())

    assert restored == original

