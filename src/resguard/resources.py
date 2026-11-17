"""Multi-dimensional resource primitives.

``ResourceAmount`` is a non-negative *usage* vector. ``ResourceDelta`` is a
signed difference vector. Monetary accounting in the ledger is done in integer
micro-USD; this module is the shared measurement vocabulary used by the guard,
the recorder, and the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


class ResourceKind(str, Enum):
    USD = "usd"
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"
    TOOL_CALLS = "tool_calls"
    DB_QUERIES = "db_queries"
    DB_CU = "db_cu"
    API_QUOTA = "api_quota"
    CONCURRENCY = "concurrency"
    WALL_CLOCK_MS = "wall_clock_ms"
    CHECKPOINT_BYTES = "checkpoint_bytes"
    RATE_REQUESTS = "rate_requests"
    RATE_TOKENS = "rate_tokens"


class ResourceMode(str, Enum):
    CUMULATIVE = "cumulative"
    GAUGE = "gauge"
    RATE_WINDOW = "rate_window"


@dataclass(frozen=True)
class ResourceSpec:
    kind: ResourceKind
    unit: str
    mode: ResourceMode


RESOURCE_SPECS: Mapping[ResourceKind, ResourceSpec] = MappingProxyType(
    {
        ResourceKind.USD: ResourceSpec(ResourceKind.USD, "$", ResourceMode.CUMULATIVE),
        ResourceKind.LLM_INPUT_TOKENS: ResourceSpec(
            ResourceKind.LLM_INPUT_TOKENS, "tokens", ResourceMode.CUMULATIVE
        ),
        ResourceKind.LLM_OUTPUT_TOKENS: ResourceSpec(
            ResourceKind.LLM_OUTPUT_TOKENS, "tokens", ResourceMode.CUMULATIVE
        ),
        ResourceKind.TOOL_CALLS: ResourceSpec(
            ResourceKind.TOOL_CALLS, "calls", ResourceMode.CUMULATIVE
        ),
        ResourceKind.DB_QUERIES: ResourceSpec(
            ResourceKind.DB_QUERIES, "queries", ResourceMode.CUMULATIVE
        ),
        ResourceKind.DB_CU: ResourceSpec(ResourceKind.DB_CU, "cu", ResourceMode.CUMULATIVE),
        ResourceKind.API_QUOTA: ResourceSpec(
            ResourceKind.API_QUOTA, "units", ResourceMode.CUMULATIVE
        ),
        ResourceKind.CONCURRENCY: ResourceSpec(
            ResourceKind.CONCURRENCY, "slots", ResourceMode.GAUGE
        ),
        ResourceKind.WALL_CLOCK_MS: ResourceSpec(
            ResourceKind.WALL_CLOCK_MS, "ms", ResourceMode.CUMULATIVE
        ),
        ResourceKind.CHECKPOINT_BYTES: ResourceSpec(
            ResourceKind.CHECKPOINT_BYTES, "bytes", ResourceMode.CUMULATIVE
        ),
        ResourceKind.RATE_REQUESTS: ResourceSpec(
            ResourceKind.RATE_REQUESTS, "requests/window", ResourceMode.RATE_WINDOW
        ),
        ResourceKind.RATE_TOKENS: ResourceSpec(
            ResourceKind.RATE_TOKENS, "tokens/window", ResourceMode.RATE_WINDOW
        ),
    }
)


def _freeze(amounts: Mapping[ResourceKind, float]) -> Mapping[ResourceKind, float]:
    return MappingProxyType({kind: float(value) for kind, value in amounts.items()})


def _combine(
    left: Mapping[ResourceKind, float],
    right: Mapping[ResourceKind, float],
    sign: float,
) -> dict[ResourceKind, float]:
    keys: set[ResourceKind] = set(left) | set(right)
    return {kind: float(left.get(kind, 0.0)) + sign * float(right.get(kind, 0.0)) for kind in keys}


@dataclass(frozen=True, eq=False)
class ResourceAmount:
    """A non-negative usage vector. Missing kinds are zero."""

    amounts: Mapping[ResourceKind, float]

    def __post_init__(self) -> None:
        for kind, value in self.amounts.items():
            if not isinstance(kind, ResourceKind):
                raise TypeError(f"expected ResourceKind, got {type(kind)!r}")
            if float(value) < 0:
                raise ValueError(f"resource amount for {kind.value} must be >= 0")
        object.__setattr__(self, "amounts", _freeze(self.amounts))

    def get(self, kind: ResourceKind) -> float:
        return float(self.amounts.get(kind, 0.0))

    def to_dict(self) -> dict[str, float]:
        return {kind.value: self.get(kind) for kind in ResourceKind}

    @classmethod
    def zero(cls) -> "ResourceAmount":
        return cls({})

    def __add__(self, other: "ResourceAmount") -> "ResourceAmount":
        return ResourceAmount(_combine(self.amounts, other.amounts, 1.0))

    def __sub__(self, other: "ResourceAmount") -> "ResourceDelta":
        return ResourceDelta(_combine(self.amounts, other.amounts, -1.0))

    def __le__(self, other: "ResourceAmount") -> bool:
        keys: set[ResourceKind] = set(self.amounts) | set(other.amounts)
        return all(self.get(kind) <= other.get(kind) for kind in keys)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceAmount):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        items = ", ".join(f"{k.value}={v:g}" for k, v in sorted(self.amounts.items(), key=lambda x: x[0].value))
        return f"ResourceAmount({items})"


@dataclass(frozen=True, eq=False)
class ResourceDelta:
    """A signed difference vector, used for savings/regression reporting."""

    amounts: Mapping[ResourceKind, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "amounts", _freeze(self.amounts))

    def get(self, kind: ResourceKind) -> float:
        return float(self.amounts.get(kind, 0.0))

    def to_dict(self) -> dict[str, float]:
        return {kind.value: self.get(kind) for kind in ResourceKind}

    def __add__(self, other: "ResourceDelta") -> "ResourceDelta":
        return ResourceDelta(_combine(self.amounts, other.amounts, 1.0))

    def __sub__(self, other: "ResourceDelta") -> "ResourceDelta":
        return ResourceDelta(_combine(self.amounts, other.amounts, -1.0))

    def __repr__(self) -> str:
        items = ", ".join(f"{k.value}={v:+g}" for k, v in sorted(self.amounts.items(), key=lambda x: x[0].value))
        return f"ResourceDelta({items})"


@dataclass(frozen=True, eq=False)
class ResourceLimits:
    """Per-kind upper bounds. A missing kind is unlimited."""

    limits: Mapping[ResourceKind, float]

    def __post_init__(self) -> None:
        for kind, value in self.limits.items():
            if not isinstance(kind, ResourceKind):
                raise TypeError(f"expected ResourceKind, got {type(kind)!r}")
            if float(value) < 0:
                raise ValueError(f"resource limit for {kind.value} must be >= 0")
        object.__setattr__(self, "limits", _freeze(self.limits))

    def get(self, kind: ResourceKind) -> float | None:
        value = self.limits.get(kind)
        return None if value is None else float(value)

    def binding(self, amount: ResourceAmount) -> ResourceKind | None:
        """Return the first exhausted *cumulative* kind in enum order.

        Gauge kinds (``CONCURRENCY``) and rate-window kinds are handled by the
        concurrency lease / rate limiter, not by cumulative reservation.
        """
        for kind in ResourceKind:
            spec = RESOURCE_SPECS.get(kind)
            if spec is None or spec.mode is not ResourceMode.CUMULATIVE:
                continue
            limit = self.get(kind)
            if limit is not None and amount.get(kind) > limit:
                return kind
        return None

    def allows(self, amount: ResourceAmount) -> bool:
        return self.binding(amount) is None

    @classmethod
    def unlimited(cls) -> "ResourceLimits":
        return cls({})

    def __repr__(self) -> str:
        items = ", ".join(f"{k.value}<={v:g}" for k, v in sorted(self.limits.items(), key=lambda x: x[0].value))
        return f"ResourceLimits({items})"


@dataclass(frozen=True)
class ToolResourceProfile:
    """Declared per-call resource usage for a tool.

    Values are *declared* estimates. Tools can report actual usage through
    ``ToolMessage.additional_kwargs["resource_usage"]``; the ledger/report
    distinguishes declared, estimated, and actual.
    """

    usd_per_call: float = 0.0
    input_tokens_per_call: int = 0
    output_tokens_per_call: int = 0
    db_queries_per_call: int = 0
    db_cu_per_call: float = 0.0
    api_quota_per_call: float = 0.0
    concurrency_slots: int = 1
    latency_ms: int = 0
    idempotent: bool = True
    cacheable: bool = True
    ttl_seconds: int | None = None

    def per_call_amount(self) -> ResourceAmount:
        return ResourceAmount(
            {
                ResourceKind.USD: self.usd_per_call,
                ResourceKind.LLM_INPUT_TOKENS: self.input_tokens_per_call,
                ResourceKind.LLM_OUTPUT_TOKENS: self.output_tokens_per_call,
                ResourceKind.DB_QUERIES: self.db_queries_per_call,
                ResourceKind.DB_CU: self.db_cu_per_call,
                ResourceKind.API_QUOTA: self.api_quota_per_call,
                ResourceKind.CONCURRENCY: self.concurrency_slots,
            }
        )

    def validate_cacheable(self) -> bool:
        """A profile may only be cached if it is idempotent and cacheable."""
        return bool(self.idempotent and self.cacheable)
