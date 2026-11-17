from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from resguard.metrics import SessionMetrics
from resguard.resources import ResourceKind
from resguard.session import (
    JsonlSessionStore,
    LabelStore,
    SessionRecord,
    SqliteSessionStore,
)


def record(
    session_id: str,
    *,
    started_at: str = "2026-01-01T00:00:00+00:00",
    metrics: SessionMetrics | None = None,
    success: bool | None = True,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        started_at=started_at,
        finished_at=started_at,
        query_type="refund",
        run_id=f"run-{session_id}",
        thread_id="thread-1",
        user_id_hash="hash-user",
        metrics=metrics or SessionMetrics(tool_calls=2, llm_usd=0.01),
        success=success,
        label_source="human" if success is not None else "none",
    )


def test_session_record_round_trip() -> None:
    original = record("s1", metrics=SessionMetrics(tool_calls=3, db_cu=1.5))

    restored = SessionRecord.from_dict(original.to_dict())

    assert restored == original


def test_resource_vector_is_derived_from_metrics() -> None:
    item = record("s1", metrics=SessionMetrics(tool_calls=3, llm_usd=0.02, tool_usd=0.01))

    vector = item.resource_vector()

    assert vector.get(ResourceKind.TOOL_CALLS) == 3
    assert vector.get(ResourceKind.USD) == pytest.approx(0.03)


def test_jsonl_store_append_and_iter(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    store.append(record("s1", started_at="2026-01-01T00:00:00+00:00"))
    store.append(record("s2", started_at="2026-01-02T00:00:00+00:00"))
    store.append(record("s3", started_at="2026-01-03T00:00:00+00:00"))

    assert [s.session_id for s in store.iter_sessions()] == ["s1", "s2", "s3"]
    assert [s.session_id for s in store.iter_sessions(limit=2)] == ["s1", "s2"]
    assert [s.session_id for s in store.iter_sessions(since="2026-01-02T00:00:00+00:00")] == ["s2", "s3"]
    assert [s.session_id for s in store.iter_sessions(until="2026-01-02T00:00:00+00:00")] == ["s1", "s2"]


def test_jsonl_store_sample_is_deterministic(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    for index in range(10):
        store.append(record(f"s{index}"))

    assert list(store.iter_sessions(sample=0.0)) == []
    assert len(list(store.iter_sessions(sample=1.0))) == 10
    first = [s.session_id for s in store.iter_sessions(sample=0.5)]
    second = [s.session_id for s in store.iter_sessions(sample=0.5)]
    assert first == second


def test_sqlite_store_append_and_iter(tmp_path) -> None:
    store = SqliteSessionStore(tmp_path / "sessions.sqlite")
    store.append(record("s1", started_at="2026-01-01T00:00:00+00:00"))
    store.append(record("s2", started_at="2026-01-02T00:00:00+00:00"))

    assert [s.session_id for s in store.iter_sessions()] == ["s1", "s2"]
    assert [s.session_id for s in store.iter_sessions(limit=1)] == ["s1"]


def test_label_store_set_and_get(tmp_path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.set_label("s1", True, source="human")

    assert store.get_label("s1") == (True, "human")
    assert store.get_label("missing") is None


def test_label_store_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    first = LabelStore(path)
    first.set_label("s1", False, source="rule")

    second = LabelStore(path)

    assert second.get_label("s1") == (False, "rule")


def test_now_helper_is_iso() -> None:
    assert datetime.now(timezone.utc).isoformat().startswith("20")
    assert json.loads(json.dumps({"ok": True})) == {"ok": True}
