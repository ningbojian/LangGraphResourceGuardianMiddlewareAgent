"""Langfuse session adapter.

Langfuse is an optional read source. The developer stores a serialized
``SessionRecord`` in trace metadata under ``resguard_session``; this adapter
reads traces via a caller-provided ``fetch_traces`` callable and maps them back
to ``SessionRecord`` objects. A ``create_trace`` callable enables writing.

The adapter does not import the Langfuse SDK; wrap your client:

```python
from langfuse import Langfuse

client = Langfuse()
store = LangfuseSessionStore(
    fetch_traces=lambda **kwargs: client.api.trace.list(**kwargs),
    create_trace=lambda **kwargs: client.trace(**kwargs),
)
```

The exact Langfuse client API differs by version; the callable boundary keeps
this adapter stable.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, Mapping

from resguard.session import SessionRecord, _filter_and_sample

TraceFetcher = Callable[..., Iterable[Any]]
TraceCreator = Callable[..., Any]


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _trace_parts(trace: Any) -> tuple[Mapping[str, Any], Any]:
    if isinstance(trace, Mapping):
        return trace.get("metadata") or {}, trace.get("output")
    return getattr(trace, "metadata", None) or {}, getattr(trace, "output", None)


def trace_to_record(trace: Any) -> SessionRecord | None:
    metadata, output = _trace_parts(trace)
    payload = None
    for key in ("resguard_session", "resguard", "session_record"):
        payload = _as_mapping(metadata.get(key))
        if payload is not None:
            break
    if payload is None:
        output_payload = _as_mapping(output)
        if output_payload is not None:
            payload = _as_mapping(output_payload.get("resguard_session"))
    if payload is None:
        return None
    try:
        return SessionRecord.from_dict(payload)
    except Exception:
        return None


def record_to_trace_metadata(record: SessionRecord) -> dict[str, Any]:
    return {"resguard_session": record.to_dict()}


class LangfuseSessionStore:
    def __init__(
        self,
        fetch_traces: TraceFetcher,
        *,
        create_trace: TraceCreator | None = None,
    ) -> None:
        self._fetch_traces = fetch_traces
        self._create_trace = create_trace

    def append(self, record: SessionRecord) -> None:
        if self._create_trace is None:
            raise NotImplementedError(
                "Langfuse writing requires a create_trace callable; "
                "use JSONL/SQLite for the default write path."
            )
        self._create_trace(
            name="resguard.session",
            metadata=record_to_trace_metadata(record),
        )

    def iter_sessions(
        self,
        *,
        since=None,
        until=None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> Iterator[SessionRecord]:
        traces = self._fetch_traces(since=since, until=until, limit=limit)
        records = (trace_to_record(trace) for trace in traces)
        yield from _filter_and_sample(
            (record for record in records if record is not None),
            since=since,
            until=until,
            limit=limit,
            sample=sample,
        )
