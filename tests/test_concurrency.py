from __future__ import annotations

import threading
from decimal import Decimal
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile


def make_guard() -> ResourceGuardMiddleware:
    return ResourceGuardMiddleware(
        ledger=InMemoryLedger(ResourceLimits({ResourceKind.CONCURRENCY: 2})),
        policy=GuardPolicy(hard_limit_action=DegradationAction.STOP),
        tool_profiles={
            "echo": ToolResourceProfile(usd_per_call=0.0, concurrency_slots=1)
        },
        pricing=ModelPricingTable(
            {"fake-model": ModelPricing(Decimal("0"), Decimal("0"))}
        ),
        estimate_output_tokens=10,
        run_id="concurrency-test",
    )


def test_guard_enforces_concurrency_under_real_overlap() -> None:
    guard = make_guard()
    barrier = threading.Barrier(2)
    overlap = threading.Event()
    release = threading.Event()
    results: list[ToolMessage] = []

    def handler(request) -> ToolMessage:
        barrier.wait(timeout=2)
        overlap.set()
        release.wait(timeout=2)
        return ToolMessage(
            content="ok",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    def worker(index: int) -> None:
        request = SimpleNamespace(
            tool_call={"name": "echo", "args": {}, "id": f"call_{index}"}
        )
        results.append(guard.wrap_tool_call(request, handler))

    first = threading.Thread(target=worker, args=(1,))
    second = threading.Thread(target=worker, args=(2,))
    first.start()
    second.start()

    assert overlap.wait(timeout=2), "the two tool calls did not overlap"
    assert guard.peak_concurrency == 2

    # Both leases are held; a third call must be rejected without running.
    third = SimpleNamespace(
        tool_call={"name": "echo", "args": {}, "id": "call_3"}
    )
    third_result = guard.wrap_tool_call(third, handler)

    assert isinstance(third_result, ToolMessage)
    assert third_result.status == "error"
    assert "concurrency" in str(third_result.content).lower()
    assert guard.concurrency_denials == 1

    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 2
    assert guard.snapshot().concurrency_in_use == 0
