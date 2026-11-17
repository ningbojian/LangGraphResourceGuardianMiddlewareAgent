from __future__ import annotations

import pytest

from resguard.integrations.langfuse import (
    LangfuseSessionStore,
    record_to_trace_metadata,
    trace_to_record,
)
from resguard.metrics import SessionMetrics
from resguard.session import SessionRecord


def record() -> SessionRecord:
    return SessionRecord(
        session_id="s1",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        query_type="refund",
        metrics=SessionMetrics(tool_calls=2, llm_usd=0.01),
        success=True,
        label_source="human",
    )


def test_trace_to_record_from_metadata() -> None:
    item = record()
    trace = {"metadata": record_to_trace_metadata(item)}

    assert trace_to_record(trace) == item


def test_trace_to_record_from_output() -> None:
    item = record()
    trace = {"metadata": {}, "output": {"resguard_session": item.to_dict()}}

    assert trace_to_record(trace) == item


def test_trace_to_record_skips_unrelated_trace() -> None:
    assert trace_to_record({"metadata": {"other": "value"}}) is None


def test_store_reads_fetched_traces() -> None:
    item = record()
    traces = [{"metadata": record_to_trace_metadata(item)}]
    store = LangfuseSessionStore(lambda **kwargs: traces)

    assert list(store.iter_sessions()) == [item]


def test_store_append_uses_create_trace() -> None:
    created: list[dict] = []
    store = LangfuseSessionStore(
        lambda **kwargs: [],
        create_trace=lambda **kwargs: created.append(kwargs),
    )

    store.append(record())

    assert created[0]["name"] == "resguard.session"
    assert created[0]["metadata"]["resguard_session"]["session_id"] == "s1"


def test_store_append_requires_create_trace() -> None:
    store = LangfuseSessionStore(lambda **kwargs: [])

    with pytest.raises(NotImplementedError):
        store.append(record())
