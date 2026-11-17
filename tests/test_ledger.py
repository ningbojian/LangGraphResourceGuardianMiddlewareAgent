from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from resguard.ledger import (
    BudgetScope,
    CompositeLedger,
    ConcurrencyDenied,
    InMemoryLedger,
    ReservationDenied,
    ReservationStatus,
)
from resguard.resources import ResourceAmount, ResourceKind, ResourceLimits


def test_reserve_and_commit() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))

    reservation = ledger.reserve(ResourceAmount({ResourceKind.USD: 0.4}))
    assert ledger.snapshot().reserved.get(ResourceKind.USD) == pytest.approx(0.4)

    ledger.commit(reservation, ResourceAmount({ResourceKind.USD: 0.3}))

    snapshot = ledger.snapshot()
    assert snapshot.used.get(ResourceKind.USD) == pytest.approx(0.3)
    assert snapshot.reserved.get(ResourceKind.USD) == 0
    assert reservation.status is ReservationStatus.COMMITTED


def test_reserve_denied_reports_binding_kind_and_leaves_state_unchanged() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 0.5}))

    with pytest.raises(ReservationDenied) as exc:
        ledger.reserve(ResourceAmount({ResourceKind.USD: 0.6}))

    assert exc.value.binding is ResourceKind.USD
    assert ledger.snapshot().reserved.get(ResourceKind.USD) == 0


def test_reserve_checks_all_cumulative_dimensions() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.DB_CU: 1.0}))

    with pytest.raises(ReservationDenied) as exc:
        ledger.reserve(ResourceAmount({ResourceKind.DB_CU: 1.5}))

    assert exc.value.binding is ResourceKind.DB_CU


def test_gauge_limit_is_not_part_of_cumulative_reservation() -> None:
    ledger = InMemoryLedger(
        ResourceLimits({ResourceKind.CONCURRENCY: 1, ResourceKind.USD: 1.0})
    )

    reservation = ledger.reserve(
        ResourceAmount({ResourceKind.USD: 0.1, ResourceKind.CONCURRENCY: 1})
    )

    assert reservation.status is ReservationStatus.RESERVED


def test_release_returns_reserved_amount() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    reservation = ledger.reserve(ResourceAmount({ResourceKind.USD: 0.4}))

    ledger.release(reservation)

    snapshot = ledger.snapshot()
    assert snapshot.reserved.get(ResourceKind.USD) == 0
    assert snapshot.used.get(ResourceKind.USD) == 0
    assert reservation.status is ReservationStatus.RELEASED


def test_commit_is_idempotent() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    reservation = ledger.reserve(ResourceAmount({ResourceKind.USD: 0.4}))

    ledger.commit(reservation, ResourceAmount({ResourceKind.USD: 0.3}))
    ledger.commit(reservation, ResourceAmount({ResourceKind.USD: 0.3}))

    assert ledger.snapshot().used.get(ResourceKind.USD) == pytest.approx(0.3)


def test_release_after_commit_is_noop() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    reservation = ledger.reserve(ResourceAmount({ResourceKind.USD: 0.4}))
    ledger.commit(reservation, ResourceAmount({ResourceKind.USD: 0.4}))

    ledger.release(reservation)

    snapshot = ledger.snapshot()
    assert snapshot.used.get(ResourceKind.USD) == pytest.approx(0.4)
    assert snapshot.reserved.get(ResourceKind.USD) == 0


def test_concurrency_lease_enforces_gauge() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.CONCURRENCY: 2}))

    first = ledger.acquire_concurrency(1)
    second = ledger.acquire_concurrency(1)
    assert ledger.snapshot().concurrency_in_use == 2

    with pytest.raises(ConcurrencyDenied):
        ledger.acquire_concurrency(1)

    first.release()
    third = ledger.acquire_concurrency(1)
    assert ledger.snapshot().concurrency_in_use == 2

    second.release()
    third.release()
    assert ledger.snapshot().concurrency_in_use == 0


def test_concurrency_lease_release_is_idempotent() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.CONCURRENCY: 1}))
    lease = ledger.acquire_concurrency(1)

    lease.release()
    lease.release()

    assert ledger.snapshot().concurrency_in_use == 0


def test_composite_reserve_rolls_back_other_scopes_on_denial() -> None:
    run_ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 0.5}))
    user_ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    # User is first so it succeeds before the run ledger denies; rollback must
    # release the already-acquired user reservation.
    composite = CompositeLedger(
        {BudgetScope.USER: user_ledger, BudgetScope.RUN: run_ledger}
    )

    reservation = composite.reserve(ResourceAmount({ResourceKind.USD: 0.4}))
    assert user_ledger.snapshot().reserved.get(ResourceKind.USD) == pytest.approx(0.4)

    with pytest.raises(ReservationDenied):
        composite.reserve(ResourceAmount({ResourceKind.USD: 0.3}))

    # The user scope must be rolled back when the run scope denies.
    assert user_ledger.snapshot().reserved.get(ResourceKind.USD) == pytest.approx(0.4)

    composite.release(reservation)
    assert user_ledger.snapshot().reserved.get(ResourceKind.USD) == 0


def test_composite_commit_and_snapshot() -> None:
    run_ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    user_ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 2.0}))
    composite = CompositeLedger(
        {BudgetScope.RUN: run_ledger, BudgetScope.USER: user_ledger}
    )

    reservation = composite.reserve(ResourceAmount({ResourceKind.USD: 0.4}))
    composite.commit(reservation, ResourceAmount({ResourceKind.USD: 0.25}))

    snapshot = composite.snapshot()
    assert snapshot.scopes[BudgetScope.RUN].used.get(ResourceKind.USD) == pytest.approx(0.25)
    assert snapshot.scopes[BudgetScope.USER].used.get(ResourceKind.USD) == pytest.approx(0.25)


def test_snapshot_available_and_utilization() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))
    ledger.reserve(ResourceAmount({ResourceKind.USD: 0.25}))

    snapshot = ledger.snapshot()

    assert snapshot.available(ResourceKind.USD) == pytest.approx(0.75)
    assert snapshot.utilization(ResourceKind.USD) == pytest.approx(0.25)
    assert snapshot.binding() is None


def test_concurrent_reservations_do_not_oversubscribe() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0}))

    def try_reserve(_: int) -> bool:
        try:
            ledger.reserve(ResourceAmount({ResourceKind.USD: 0.2}))
            return True
        except ReservationDenied:
            return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(try_reserve, range(20)))

    assert sum(results) == 5
    assert ledger.snapshot().reserved.get(ResourceKind.USD) <= 1.0


def test_snapshot_binding_reports_exhausted_kind() -> None:
    ledger = InMemoryLedger(ResourceLimits({ResourceKind.DB_CU: 1.0}))
    ledger.reserve(ResourceAmount({ResourceKind.DB_CU: 1.0}))

    assert ledger.snapshot().binding() is ResourceKind.DB_CU
