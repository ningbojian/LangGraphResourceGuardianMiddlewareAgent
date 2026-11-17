"""Multi-dimensional resource ledger.

The ledger is the authoritative accounting layer. ``ResourceAmount`` vectors are
reserved before an operation and committed/released afterwards. ``CONCURRENCY``
is a gauge handled by leases, not by cumulative reservation.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol

from resguard.resources import (
    RESOURCE_SPECS,
    ResourceAmount,
    ResourceKind,
    ResourceLimits,
    ResourceMode,
)


class BudgetScope(str, Enum):
    RUN = "run"
    THREAD = "thread"
    USER = "user"


class ReservationStatus(str, Enum):
    RESERVED = "reserved"
    COMMITTED = "committed"
    RELEASED = "released"


class ReservationDenied(Exception):
    def __init__(
        self,
        *,
        binding: ResourceKind,
        scope_key: str,
        requested: ResourceAmount,
        remaining: float | None,
    ) -> None:
        self.binding = binding
        self.scope_key = scope_key
        self.requested = requested
        self.remaining = remaining
        message = (
            f"resource {binding.value} exhausted in scope {scope_key}: "
            f"requested {requested.get(binding):g}, remaining "
            f"{'unlimited' if remaining is None else f'{remaining:g}'}"
        )
        super().__init__(message)


class ConcurrencyDenied(Exception):
    def __init__(
        self,
        *,
        scope_key: str,
        requested_slots: int,
        in_use: int,
        limit: float | None,
    ) -> None:
        self.scope_key = scope_key
        self.requested_slots = requested_slots
        self.in_use = in_use
        self.limit = limit
        super().__init__(
            f"concurrency limit in scope {scope_key}: in_use={in_use}, "
            f"requested={requested_slots}, limit={limit}"
        )


def _cumulative_only(amount: ResourceAmount) -> ResourceAmount:
    return ResourceAmount(
        {
            kind: value
            for kind, value in amount.amounts.items()
            if RESOURCE_SPECS[kind].mode is ResourceMode.CUMULATIVE
        }
    )


def _subtract_clamped(left: ResourceAmount, right: ResourceAmount) -> ResourceAmount:
    """Per-kind subtraction clamped at zero.

    Resource vectors are measurements in floats; subtraction can leave a tiny
    negative residue (e.g. 0.4 + 0.3 - 0.3 - 0.4). Clamping keeps the ledger in
    the non-negative domain instead of leaking a ``ResourceDelta`` into state.
    """
    keys = set(left.amounts) | set(right.amounts)
    return ResourceAmount({kind: max(0.0, left.get(kind) - right.get(kind)) for kind in keys})


@dataclass
class Reservation:
    id: str
    amount: ResourceAmount
    scope_key: str = "default"
    status: ReservationStatus = ReservationStatus.RESERVED
    actual: ResourceAmount | None = None


@dataclass(frozen=True)
class ResourceSnapshot:
    scope_key: str
    used: ResourceAmount
    reserved: ResourceAmount
    limits: ResourceLimits
    concurrency_in_use: int = 0

    @property
    def total(self) -> ResourceAmount:
        return self.used + self.reserved

    def available(self, kind: ResourceKind) -> float | None:
        limit = self.limits.get(kind)
        if limit is None:
            return None
        if RESOURCE_SPECS[kind].mode is ResourceMode.GAUGE:
            return limit - self.concurrency_in_use
        return limit - self.total.get(kind)

    def utilization(self, kind: ResourceKind) -> float:
        limit = self.limits.get(kind)
        if limit is None or limit == 0:
            return 0.0
        if RESOURCE_SPECS[kind].mode is ResourceMode.GAUGE:
            return self.concurrency_in_use / limit
        return self.total.get(kind) / limit

    def binding(self) -> ResourceKind | None:
        """First cumulative kind with no remaining budget (>= limit)."""
        for kind in ResourceKind:
            if RESOURCE_SPECS[kind].mode is not ResourceMode.CUMULATIVE:
                continue
            limit = self.limits.get(kind)
            if limit is not None and self.total.get(kind) >= limit:
                return kind
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "scope_key": self.scope_key,
            "used": self.used.to_dict(),
            "reserved": self.reserved.to_dict(),
            "limits": {kind.value: limit for kind, limit in self.limits.limits.items()},
            "concurrency_in_use": self.concurrency_in_use,
            "binding": self.binding().value if self.binding() else None,
        }


class ResourceLedger(Protocol):
    def reserve(self, amount: ResourceAmount, *, reservation_id: str | None = None) -> Reservation: ...
    def commit(self, reservation: Reservation, actual: ResourceAmount | None = None) -> Reservation: ...
    def release(self, reservation: Reservation) -> Reservation: ...
    def acquire_concurrency(self, slots: int = 1) -> "ConcurrencyLease | CompositeLease": ...
    def release_concurrency(self, lease: "ConcurrencyLease | CompositeLease") -> None: ...
    def snapshot(self) -> ResourceSnapshot | "CompositeSnapshot": ...


@dataclass
class ConcurrencyLease:
    ledger: "InMemoryLedger"
    slots: int
    _released: bool = False

    def release(self) -> None:
        self.ledger.release_concurrency(self)


class InMemoryLedger:
    """Thread-safe single-scope ledger."""

    def __init__(self, limits: ResourceLimits, *, scope_key: str = "default") -> None:
        self.limits = limits
        self.scope_key = scope_key
        self._used = ResourceAmount.zero()
        self._reserved = ResourceAmount.zero()
        self._concurrency_in_use = 0
        self._reservations: dict[str, Reservation] = {}
        self._lock = threading.RLock()

    def reserve(self, amount: ResourceAmount, *, reservation_id: str | None = None) -> Reservation:
        cumulative = _cumulative_only(amount)
        with self._lock:
            total = self._used + self._reserved + cumulative
            binding = self.limits.binding(total)
            if binding is not None:
                limit = self.limits.get(binding)
                remaining = None if limit is None else limit - (self._used + self._reserved).get(binding)
                raise ReservationDenied(
                    binding=binding,
                    scope_key=self.scope_key,
                    requested=cumulative,
                    remaining=remaining,
                )
            self._reserved = self._reserved + cumulative
            reservation = Reservation(
                id=reservation_id or uuid.uuid4().hex,
                amount=cumulative,
                scope_key=self.scope_key,
            )
            self._reservations[reservation.id] = reservation
            return reservation

    def commit(self, reservation: Reservation, actual: ResourceAmount | None = None) -> Reservation:
        with self._lock:
            if reservation.status is not ReservationStatus.RESERVED:
                return reservation
            committed = _cumulative_only(actual) if actual is not None else reservation.amount
            self._reserved = _subtract_clamped(self._reserved, reservation.amount)
            self._used = self._used + committed
            reservation.status = ReservationStatus.COMMITTED
            reservation.actual = committed
            return reservation

    def release(self, reservation: Reservation) -> Reservation:
        with self._lock:
            if reservation.status is not ReservationStatus.RESERVED:
                return reservation
            self._reserved = _subtract_clamped(self._reserved, reservation.amount)
            reservation.status = ReservationStatus.RELEASED
            return reservation

    def acquire_concurrency(self, slots: int = 1) -> ConcurrencyLease:
        limit = self.limits.get(ResourceKind.CONCURRENCY)
        with self._lock:
            if limit is not None and self._concurrency_in_use + slots > limit:
                raise ConcurrencyDenied(
                    scope_key=self.scope_key,
                    requested_slots=slots,
                    in_use=self._concurrency_in_use,
                    limit=limit,
                )
            self._concurrency_in_use += slots
            return ConcurrencyLease(self, slots)

    def release_concurrency(self, lease: ConcurrencyLease) -> None:
        with self._lock:
            if lease._released:
                return
            self._concurrency_in_use = max(0, self._concurrency_in_use - lease.slots)
            lease._released = True

    def snapshot(self) -> ResourceSnapshot:
        with self._lock:
            return ResourceSnapshot(
                scope_key=self.scope_key,
                used=self._used,
                reserved=self._reserved,
                limits=self.limits,
                concurrency_in_use=self._concurrency_in_use,
            )


@dataclass
class CompositeReservation:
    entries: tuple[tuple["InMemoryLedger", Reservation], ...]

    @property
    def reservations(self) -> tuple[Reservation, ...]:
        return tuple(reservation for _, reservation in self.entries)


@dataclass
class CompositeLease:
    leases: tuple[ConcurrencyLease, ...]
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        for lease in self.leases:
            lease.release()
        self._released = True


@dataclass(frozen=True)
class CompositeSnapshot:
    scopes: Mapping[BudgetScope, ResourceSnapshot]

    @property
    def used(self) -> ResourceAmount:
        total = ResourceAmount.zero()
        for snapshot in self.scopes.values():
            total = total + snapshot.used
        return total

    @property
    def reserved(self) -> ResourceAmount:
        total = ResourceAmount.zero()
        for snapshot in self.scopes.values():
            total = total + snapshot.reserved
        return total

    def binding_scope(self) -> BudgetScope | None:
        for scope, snapshot in self.scopes.items():
            if snapshot.binding() is not None:
                return scope
        return None

    def to_dict(self) -> dict[str, object]:
        return {scope.value: snapshot.to_dict() for scope, snapshot in self.scopes.items()}


class CompositeLedger:
    """Reserve atomically across run/thread/user scopes."""

    def __init__(self, ledgers: Mapping[BudgetScope, InMemoryLedger]) -> None:
        self.ledgers = dict(ledgers)

    def reserve(self, amount: ResourceAmount, *, reservation_id: str | None = None) -> CompositeReservation:
        acquired: list[tuple[InMemoryLedger, Reservation]] = []
        for ledger in self.ledgers.values():
            try:
                reservation = ledger.reserve(amount, reservation_id=reservation_id)
                acquired.append((ledger, reservation))
            except ReservationDenied:
                for prior_ledger, prior_reservation in reversed(acquired):
                    prior_ledger.release(prior_reservation)
                raise
        return CompositeReservation(tuple(acquired))

    def commit(self, reservation: CompositeReservation, actual: ResourceAmount | None = None) -> CompositeReservation:
        for ledger, sub_reservation in reservation.entries:
            ledger.commit(sub_reservation, actual)
        return reservation

    def release(self, reservation: CompositeReservation) -> CompositeReservation:
        for ledger, sub_reservation in reservation.entries:
            ledger.release(sub_reservation)
        return reservation

    def acquire_concurrency(self, slots: int = 1) -> CompositeLease:
        leases: list[ConcurrencyLease] = []
        for ledger in self.ledgers.values():
            try:
                leases.append(ledger.acquire_concurrency(slots))
            except ConcurrencyDenied:
                for lease in reversed(leases):
                    lease.release()
                raise
        return CompositeLease(tuple(leases))

    def release_concurrency(self, lease: CompositeLease) -> None:
        lease.release()

    def snapshot(self) -> CompositeSnapshot:
        return CompositeSnapshot({scope: ledger.snapshot() for scope, ledger in self.ledgers.items()})
