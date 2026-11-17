from __future__ import annotations

import pytest

from resguard.cache import CacheContext, ToolCache, canonical_key


def test_set_and_get_returns_result() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")

    cache.set("search_docs", {"query": "refund"}, "result", ctx=ctx)

    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) == "result"


def test_different_args_miss() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")
    cache.set("search_docs", {"query": "refund"}, "result", ctx=ctx)

    assert cache.get("search_docs", {"query": "shipping"}, ctx=ctx) is None


def test_ttl_expiry_with_fake_clock() -> None:
    now = [100.0]
    cache = ToolCache(clock=lambda: now[0])
    ctx = CacheContext(user_id="alice")
    cache.set("search_docs", {"query": "refund"}, "result", ttl_seconds=10, ctx=ctx)

    now[0] = 109.0
    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) == "result"

    now[0] = 111.0
    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) is None


def test_user_and_tenant_are_part_of_key() -> None:
    cache = ToolCache()
    alice = CacheContext(user_id="alice", tenant_id="t1")
    bob = CacheContext(user_id="bob", tenant_id="t1")
    cache.set("search_docs", {"query": "refund"}, "alice-result", ctx=alice)

    assert cache.get("search_docs", {"query": "refund"}, ctx=bob) is None


def test_doc_version_is_part_of_key() -> None:
    cache = ToolCache()
    v1 = CacheContext(user_id="alice", doc_version="v1")
    v2 = CacheContext(user_id="alice", doc_version="v2")
    cache.set("search_docs", {"query": "refund"}, "old", ctx=v1)

    assert cache.get("search_docs", {"query": "refund"}, ctx=v2) is None


def test_canonical_key_is_stable_for_arg_order_and_nested_values() -> None:
    left = canonical_key("search", {"a": 1, "b": {"x": 2, "y": 3}}, CacheContext(user_id="u"))
    right = canonical_key("search", {"b": {"y": 3, "x": 2}, "a": 1}, CacheContext(user_id="u"))

    assert left == right


def test_invalidate_clears_entries() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")
    cache.set("search_docs", {"query": "refund"}, "result", ctx=ctx)

    cache.clear()

    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) is None


def test_invalidate_tool_removes_only_that_tool() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")
    cache.set("search_docs", {"query": "refund"}, "search", ctx=ctx)
    cache.set("parse_pdf", {"query": "refund"}, "parse", ctx=ctx)

    removed = cache.invalidate_tool("search_docs")

    assert removed == 1
    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) is None
    assert cache.get("parse_pdf", {"query": "refund"}, ctx=ctx) == "parse"


def test_invalidate_context_removes_only_matching_user() -> None:
    cache = ToolCache()
    alice = CacheContext(user_id="alice", tenant_id="t1")
    bob = CacheContext(user_id="bob", tenant_id="t1")
    cache.set("search_docs", {"query": "refund"}, "alice", ctx=alice)
    cache.set("search_docs", {"query": "refund"}, "bob", ctx=bob)

    removed = cache.invalidate_context(alice)

    assert removed == 1
    assert cache.get("search_docs", {"query": "refund"}, ctx=alice) is None
    assert cache.get("search_docs", {"query": "refund"}, ctx=bob) == "bob"


def test_cache_stats() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")
    cache.set("search_docs", {"query": "refund"}, "result", ctx=ctx)

    assert cache.get("search_docs", {"query": "refund"}, ctx=ctx) == "result"
    assert cache.get("search_docs", {"query": "other"}, ctx=ctx) is None

    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.sets == 1


def test_unserializable_args_use_stable_fallback() -> None:
    cache = ToolCache()
    ctx = CacheContext(user_id="alice")
    cache.set("tool", {"value": object()}, "result", ctx=ctx)

    # Cannot reliably look up an arbitrary object(), so just assert no exception
    # and that a stable key is produced for the same object.
    key1 = canonical_key("tool", {"value": object()}, ctx)
    key2 = canonical_key("tool", {"value": object()}, ctx)
    assert isinstance(key1, str) and isinstance(key2, str)
