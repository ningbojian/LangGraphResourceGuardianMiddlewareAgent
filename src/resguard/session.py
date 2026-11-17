"""Session records and local stores for the offline optimizer.

The online recorder writes one ``SessionRecord`` per run. The optimizer reads a
batch, never a live user request. JSONL is the default local format; SQLite is
provided for larger local batches.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from resguard.metrics import SessionMetrics
from resguard.resources import ResourceAmount


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else value


@dataclass
class SessionRecord:
    session_id: str
    started_at: str
    finished_at: str
    query_type: str = "unknown"
    run_id: str = ""
    thread_id: str | None = None
    user_id_hash: str | None = None
    agent_version: str = "0"
    model_version: str = "unknown"
    prompt_version: str = "0"
    tool_versions: dict[str, str] = field(default_factory=dict)
    workflow_version: str = "0"
    policy_version: str = "0"
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    success: bool | None = None
    label_source: str = "none"
    termination_reason: str = "completed"
    binding_limit: str | None = None
    replay_safe: bool = True
    trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def resource_vector(self) -> ResourceAmount:
        return self.metrics.resource_amount()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionRecord":
        known = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        payload["metrics"] = SessionMetrics.from_dict(data.get("metrics") or {})
        payload["tool_versions"] = dict(data.get("tool_versions") or {})
        return cls(**payload)


class SessionStore(Protocol):
    def append(self, record: SessionRecord) -> None: ...
    def iter_sessions(
        self,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> Iterator[SessionRecord]: ...


def _filter_and_sample(
    records: Iterator[SessionRecord],
    *,
    since: datetime | str | None,
    until: datetime | str | None,
    limit: int | None,
    sample: float,
) -> Iterator[SessionRecord]:
    if sample <= 0:
        return
    since_iso = _iso(since)
    until_iso = _iso(until)
    rng = random.Random(0)
    emitted = 0
    for record in records:
        if since_iso is not None and record.started_at < since_iso:
            continue
        if until_iso is not None and record.started_at > until_iso:
            continue
        if sample < 1.0 and rng.random() >= sample:
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return


class JsonlSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: SessionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _iter_all(self) -> Iterator[SessionRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield SessionRecord.from_dict(json.loads(line))

    def iter_sessions(
        self,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> Iterator[SessionRecord]:
        yield from _filter_and_sample(
            self._iter_all(), since=since, until=until, limit=limit, sample=sample
        )


class SqliteSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def append(self, record: SessionRecord) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO sessions (session_id, started_at, payload) VALUES (?, ?, ?)",
            (
                record.session_id,
                record.started_at,
                json.dumps(record.to_dict(), ensure_ascii=False),
            ),
        )
        self._connection.commit()

    def _iter_all(self) -> Iterator[SessionRecord]:
        cursor = self._connection.execute(
            "SELECT payload FROM sessions ORDER BY started_at, session_id"
        )
        for (payload,) in cursor:
            yield SessionRecord.from_dict(json.loads(payload))

    def iter_sessions(
        self,
        *,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> Iterator[SessionRecord]:
        yield from _filter_and_sample(
            self._iter_all(), since=since, until=until, limit=limit, sample=sample
        )


class LabelStore:
    """Small label store; the latest label per session wins."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._labels: dict[str, tuple[bool, str]] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        self._labels[item["session_id"]] = (
                            bool(item["success"]),
                            item.get("source", "human"),
                        )

    def set_label(self, session_id: str, success: bool, *, source: str = "human") -> None:
        self._labels[session_id] = (bool(success), source)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"session_id": session_id, "success": bool(success), "source": source},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def get_label(self, session_id: str) -> tuple[bool, str] | None:
        return self._labels.get(session_id)

    def apply_to(self, record: SessionRecord) -> SessionRecord:
        label = self.get_label(record.session_id)
        if label is None:
            record.label_source = "none"
            return record
        record.success, record.label_source = label
        return record
