from __future__ import annotations

import pytest

from resguard.metrics import (
    SessionMetrics,
    estimate_checkpoint_bytes,
    estimate_history_tokens,
    estimate_rehydrate_tokens,
    estimate_tokens,
    estimate_tool_schema_tokens,
)
from resguard.resources import ResourceAmount, ResourceKind


def test_session_metrics_defaults_are_zero() -> None:
    metrics = SessionMetrics()

    assert metrics.prompt_tokens == 0
    assert metrics.tool_calls == 0
    assert metrics.checkpoint_bytes == 0
    assert metrics.llm_usd == 0.0


def test_to_dict_round_trip() -> None:
    metrics = SessionMetrics(prompt_tokens=10, db_cu=1.5, peak_concurrency=3)

    restored = SessionMetrics.from_dict(metrics.to_dict())

    assert restored == metrics


def test_resource_amount_exports_enforceable_dimensions() -> None:
    metrics = SessionMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        tool_calls=4,
        db_queries=2,
        db_cu=0.5,
        api_quota=1,
        peak_concurrency=3,
        wall_clock_ms=1200,
        checkpoint_bytes=4096,
        llm_usd=0.01,
        tool_usd=0.02,
        planner_usd=0.003,
    )

    amount = metrics.resource_amount()

    assert isinstance(amount, ResourceAmount)
    assert amount.get(ResourceKind.USD) == pytest.approx(0.033)
    assert amount.get(ResourceKind.LLM_INPUT_TOKENS) == 100
    assert amount.get(ResourceKind.LLM_OUTPUT_TOKENS) == 50
    assert amount.get(ResourceKind.TOOL_CALLS) == 4
    assert amount.get(ResourceKind.DB_QUERIES) == 2
    assert amount.get(ResourceKind.DB_CU) == 0.5
    assert amount.get(ResourceKind.API_QUOTA) == 1
    assert amount.get(ResourceKind.CONCURRENCY) == 3
    assert amount.get(ResourceKind.WALL_CLOCK_MS) == 1200
    assert amount.get(ResourceKind.CHECKPOINT_BYTES) == 4096


def test_estimate_tokens_char_fallback() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tool_schema_tokens() -> None:
    tools = [
        {"name": "search_docs", "description": "Search documents", "args": {"query": "string"}},
        {"name": "parse_pdf", "description": "Parse a PDF", "args": {"path": "string"}},
    ]

    tokens = estimate_tool_schema_tokens(tools)

    assert tokens > 0
    assert tokens == estimate_tool_schema_tokens(tools)


def test_estimate_history_tokens() -> None:
    messages = [
        {"role": "user", "content": "hello " * 40},
        {"role": "assistant", "content": "world " * 40},
    ]

    tokens = estimate_history_tokens(messages)

    assert tokens > 0
    assert tokens == estimate_history_tokens(messages)


def test_estimate_checkpoint_bytes_and_rehydrate_tokens() -> None:
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    size = estimate_checkpoint_bytes(payload)
    tokens = estimate_rehydrate_tokens(size)

    assert size > 0
    assert tokens > 0


def test_merge_sums_counters_and_takes_peak_concurrency() -> None:
    left = SessionMetrics(tool_calls=2, db_cu=1.0, peak_concurrency=3, llm_usd=0.01)
    right = SessionMetrics(tool_calls=3, db_cu=2.0, peak_concurrency=5, llm_usd=0.02)

    merged = left.merge(right)

    assert merged.tool_calls == 5
    assert merged.db_cu == 3.0
    assert merged.peak_concurrency == 5
    assert merged.llm_usd == pytest.approx(0.03)


def test_negative_metrics_rejected() -> None:
    with pytest.raises(ValueError):
        SessionMetrics(tool_calls=-1)
