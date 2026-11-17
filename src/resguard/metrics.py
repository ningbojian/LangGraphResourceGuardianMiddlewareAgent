"""Session-level observed metrics and cheap token/byte estimators.

These metrics are *observed* rather than hard-enforced. The optimizer uses them
to find token/context, persistence, retry, and guard-overhead regressions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

from resguard.resources import ResourceAmount, ResourceKind


@dataclass
class SessionMetrics:
    # token/context
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    tool_schema_tokens: int = 0
    history_tokens: int = 0
    # tool/db/api
    tool_calls: int = 0
    cache_hits: int = 0
    duplicate_tool_calls: int = 0
    tool_errors: int = 0
    tool_retries: int = 0
    db_queries: int = 0
    db_cu: float = 0.0
    vector_queries: int = 0
    api_quota: float = 0.0
    # graph/persistence
    supersteps: int = 0
    node_visits: int = 0
    checkpoint_writes: int = 0
    checkpoint_bytes: int = 0
    rehydrate_tokens: int = 0
    # concurrency/quality/optimizer
    peak_concurrency: int = 0
    concurrency_denials: int = 0
    queue_wait_ms: float = 0.0
    rate_limit_429s: int = 0
    false_positive_stops: int = 0
    answer_quality: float = 0.0
    user_retries: int = 0
    optimizer_tokens: int = 0
    # subagent/observability
    subagent_calls: int = 0
    subagent_prompt_tokens: int = 0
    parent_child_message_tokens: int = 0
    trace_spans: int = 0
    trace_bytes: int = 0
    export_ms: float = 0.0
    # guard overhead / latency / cost breakdown
    middleware_ms: float = 0.0
    callback_ms: float = 0.0
    wall_clock_ms: float = 0.0
    llm_usd: float = 0.0
    tool_usd: float = 0.0
    planner_usd: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value < 0:
                raise ValueError(f"metric {field.name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionMetrics":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def resource_amount(self) -> ResourceAmount:
        return ResourceAmount(
            {
                ResourceKind.USD: self.llm_usd + self.tool_usd + self.planner_usd,
                ResourceKind.LLM_INPUT_TOKENS: self.prompt_tokens,
                ResourceKind.LLM_OUTPUT_TOKENS: self.completion_tokens,
                ResourceKind.TOOL_CALLS: self.tool_calls,
                ResourceKind.DB_QUERIES: self.db_queries,
                ResourceKind.DB_CU: self.db_cu,
                ResourceKind.API_QUOTA: self.api_quota,
                ResourceKind.CONCURRENCY: self.peak_concurrency,
                ResourceKind.WALL_CLOCK_MS: self.wall_clock_ms,
                ResourceKind.CHECKPOINT_BYTES: self.checkpoint_bytes,
            }
        )

    def merge(self, other: "SessionMetrics") -> "SessionMetrics":
        merged: dict[str, Any] = {}
        for field in fields(self):
            left = getattr(self, field.name)
            right = getattr(other, field.name)
            merged[field.name] = max(left, right) if field.name == "peak_concurrency" else left + right
        return SessionMetrics(**merged)


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic token estimate (roughly 4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _tool_payload(tool: Any) -> Any:
    if isinstance(tool, Mapping):
        return tool
    payload = {}
    for attr in ("name", "description", "args"):
        if hasattr(tool, attr):
            payload[attr] = getattr(tool, attr)
    return payload or repr(tool)


def estimate_tool_schema_tokens(tools: Iterable[Any]) -> int:
    total = 0
    for tool in tools:
        serialized = json.dumps(_tool_payload(tool), default=str, sort_keys=True)
        total += estimate_tokens(serialized)
    return total


def _message_text(message: Any) -> str:
    if isinstance(message, Mapping):
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
    else:
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str, sort_keys=True)
    if tool_calls:
        content += json.dumps(tool_calls, default=str, sort_keys=True)
    return content


def estimate_history_tokens(messages: Iterable[Any]) -> int:
    return sum(estimate_tokens(_message_text(message)) for message in messages)


def estimate_checkpoint_bytes(payload: Any) -> int:
    try:
        serialized = json.dumps(payload, default=str, sort_keys=True)
    except (TypeError, ValueError):
        serialized = repr(payload)
    return len(serialized.encode("utf-8"))


def estimate_rehydrate_tokens(payload_bytes: int) -> int:
    if payload_bytes <= 0:
        return 0
    return max(1, payload_bytes // 4)
