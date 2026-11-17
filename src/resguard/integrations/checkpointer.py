"""LangGraph checkpointer session adapter.

Reads a thread's state history from a compiled graph or a checkpointer and turns
the latest state into a ``SessionRecord``. The checkpointer does not store
resource counters, so metrics are derived from messages/state: tool calls, token
usage, checkpoint count, and estimated checkpoint bytes. Success labels are
optional via ``label_fn``.

``thread_ids`` must be supplied; a checkpointer has no generic "list all
threads" API. Full graph synthesis/reordering is out of scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable, Iterator, Sequence

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict

from resguard.metrics import SessionMetrics, estimate_checkpoint_bytes
from resguard.session import SessionRecord, _filter_and_sample, now_iso

LabelFn = Callable[[str, Sequence[Any]], bool | None]
QueryTypeFn = Callable[[str], str]


def _trace_from_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    tool_by_id = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    trace: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            result = tool_by_id.get(tool_call.get("id", ""))
            trace.append(
                {
                    "name": tool_call.get("name", "unknown"),
                    "args": dict(tool_call.get("args") or {}),
                    "result": str(result.content) if result is not None else "",
                }
            )
    return trace


def _snapshot_values(snapshot: Any) -> dict[str, Any]:
    values = getattr(snapshot, "values", None)
    if values is not None:
        return dict(values)
    checkpoint = getattr(snapshot, "checkpoint", None) or {}
    if isinstance(checkpoint, dict):
        return dict(checkpoint.get("channel_values") or {})
    return {}


def _snapshot_time(snapshot: Any) -> str:
    value = getattr(snapshot, "created_at", None)
    if value is None:
        checkpoint = getattr(snapshot, "checkpoint", None) or {}
        if isinstance(checkpoint, dict):
            value = checkpoint.get("ts")
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return now_iso()


class CheckpointerSessionStore:
    def __init__(
        self,
        graph: Any | None = None,
        *,
        checkpointer: Any | None = None,
        thread_ids: Iterable[str],
        label_fn: LabelFn | None = None,
        query_type_fn: QueryTypeFn | None = None,
    ) -> None:
        if graph is None and checkpointer is None:
            raise ValueError("provide a compiled graph or a checkpointer")
        self.graph = graph
        self.checkpointer = checkpointer
        self.thread_ids = list(thread_ids)
        self.label_fn = label_fn
        self.query_type_fn = query_type_fn

    def append(self, record: SessionRecord) -> None:
        raise NotImplementedError(
            "CheckpointerSessionStore is read-only; use JSONL/SQLite for writes."
        )

    def _history(self, thread_id: str) -> Iterator[Any]:
        config = {"configurable": {"thread_id": thread_id}}
        if self.graph is not None:
            return self.graph.get_state_history(config)
        if self.checkpointer is not None and hasattr(self.checkpointer, "list"):
            return self.checkpointer.list(config)
        raise TypeError("graph/checkpointer does not expose state history")

    def _to_record(self, thread_id: str, history: Sequence[Any]) -> SessionRecord:
        latest = max(history, key=_snapshot_time)
        values = _snapshot_values(latest)
        messages = values.get("messages") or []
        ai_messages = [message for message in messages if isinstance(message, AIMessage)]
        input_tokens = sum(
            int((getattr(message, "usage_metadata", None) or {}).get("input_tokens", 0) or 0)
            for message in ai_messages
        )
        output_tokens = sum(
            int((getattr(message, "usage_metadata", None) or {}).get("output_tokens", 0) or 0)
            for message in ai_messages
        )
        times = sorted(_snapshot_time(snapshot) for snapshot in history)
        try:
            serialized_messages = messages_to_dict(messages)
        except Exception:
            serialized_messages = []
        record = SessionRecord(
            session_id=f"thread:{thread_id}",
            started_at=times[0],
            finished_at=times[-1],
            query_type=(
                self.query_type_fn(thread_id) if self.query_type_fn is not None else "unknown"
            ),
            model_version="checkpointer",
            metrics=SessionMetrics(
                tool_calls=sum(len(message.tool_calls) for message in ai_messages),
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                checkpoint_writes=len(history),
                checkpoint_bytes=estimate_checkpoint_bytes(values),
                rehydrate_tokens=estimate_checkpoint_bytes(values) // 4,
            ),
            messages=serialized_messages,
            trace=_trace_from_messages(messages),
            replay_safe=True,
        )
        if self.label_fn is not None:
            success = self.label_fn(thread_id, messages)
            record.success = success
            record.label_source = "human" if success is not None else "none"
        return record

    def iter_sessions(
        self,
        *,
        since=None,
        until=None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> Iterator[SessionRecord]:
        records: list[SessionRecord] = []
        for thread_id in self.thread_ids:
            history = list(self._history(thread_id))
            if history:
                records.append(self._to_record(thread_id, history))
        yield from _filter_and_sample(
            iter(records),
            since=since,
            until=until,
            limit=limit,
            sample=sample,
        )
