"""Small per-tool TTL cache with tenant/user/doc-version isolation.

This is intentionally a thin adapter, not a cache store. A production system can
replace it with LangGraph ``BaseCache``/Redis while keeping the same key policy.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class CacheContext:
    user_id: str | None = None
    tenant_id: str | None = None
    doc_version: str | None = None


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float | None
    tool_name: str
    user_id: str | None
    tenant_id: str | None
    doc_version: str | None


def canonical_key(
    tool_name: str,
    args: Mapping[str, Any],
    ctx: CacheContext,
) -> str:
    payload = {
        "tool": tool_name,
        "args": args,
        "tenant_id": ctx.tenant_id,
        "user_id": ctx.user_id,
        "doc_version": ctx.doc_version,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ToolCache:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        default_ttl_seconds: int | None = None,
    ) -> None:
        self._clock = clock
        self._default_ttl = default_ttl_seconds
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def get(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        ctx: CacheContext,
    ) -> Any | None:
        key = canonical_key(tool_name, args, ctx)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at is not None and self._clock() >= entry.expires_at:
                del self._entries[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        value: Any,
        *,
        ctx: CacheContext,
        ttl_seconds: int | None = None,
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = None if ttl is None else self._clock() + ttl
        key = canonical_key(tool_name, args, ctx)
        with self._lock:
            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=expires_at,
                tool_name=tool_name,
                user_id=ctx.user_id,
                tenant_id=ctx.tenant_id,
                doc_version=ctx.doc_version,
            )
            self._sets += 1

    def invalidate_tool(self, tool_name: str) -> int:
        """Invalidate every cached result for a tool (e.g. after a data refresh)."""
        with self._lock:
            keys = [
                key
                for key, entry in self._entries.items()
                if entry.tool_name == tool_name
            ]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def invalidate_context(self, ctx: CacheContext) -> int:
        """Invalidate entries matching a tenant/user/doc-version context."""
        with self._lock:
            keys = [
                key
                for key, entry in self._entries.items()
                if entry.user_id == ctx.user_id
                and entry.tenant_id == ctx.tenant_id
                and entry.doc_version == ctx.doc_version
            ]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(hits=self._hits, misses=self._misses, sets=self._sets)
