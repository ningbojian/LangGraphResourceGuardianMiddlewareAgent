"""Model pricing and token-cost accounting.

Money is represented with :class:`decimal.Decimal` throughout this module. The
ledger stores integer micro-USD; this module is the calculation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

_MILLION = Decimal("1000000")


class UnknownModelError(KeyError):
    """Raised when strict pricing is requested for an unregistered model."""


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal | None = None
    cache_write_usd_per_million: Decimal | None = None
    reasoning_usd_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_usd_per_million",
            "output_usd_per_million",
            "cached_input_usd_per_million",
            "cache_write_usd_per_million",
            "reasoning_usd_per_million",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def zero(cls) -> "ModelPricing":
        return cls(Decimal("0"), Decimal("0"))


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "reasoning_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def from_usage_metadata(cls, metadata: Mapping[str, Any] | None) -> "TokenUsage":
        if not metadata:
            return cls()
        input_details = metadata.get("input_token_details") or {}
        output_details = metadata.get("output_token_details") or {}
        return cls(
            input_tokens=int(metadata.get("input_tokens") or 0),
            output_tokens=int(metadata.get("output_tokens") or 0),
            cached_input_tokens=int(input_details.get("cache_read") or 0),
            cache_creation_tokens=int(input_details.get("cache_creation") or 0),
            reasoning_tokens=int(output_details.get("reasoning") or 0),
        )


def cost_for_usage(pricing: ModelPricing, usage: TokenUsage) -> Decimal:
    """Compute USD cost for a token usage record.

    ``cached_input_tokens`` and ``cache_creation_tokens`` are subsets of
    ``input_tokens``. ``reasoning_tokens`` is a subset of ``output_tokens``.
    """
    uncached_input = max(
        0, usage.input_tokens - usage.cached_input_tokens - usage.cache_creation_tokens
    )
    cached_price = pricing.cached_input_usd_per_million
    if cached_price is None:
        cached_price = pricing.input_usd_per_million
    cache_write_price = pricing.cache_write_usd_per_million
    if cache_write_price is None:
        cache_write_price = pricing.input_usd_per_million
    reasoning_price = pricing.reasoning_usd_per_million
    if reasoning_price is None:
        reasoning_price = pricing.output_usd_per_million

    input_cost = Decimal(uncached_input) * pricing.input_usd_per_million
    cached_cost = Decimal(usage.cached_input_tokens) * cached_price
    cache_write_cost = Decimal(usage.cache_creation_tokens) * cache_write_price

    non_reasoning_output = max(0, usage.output_tokens - usage.reasoning_tokens)
    output_cost = Decimal(non_reasoning_output) * pricing.output_usd_per_million
    reasoning_cost = Decimal(usage.reasoning_tokens) * reasoning_price

    return (input_cost + cached_cost + cache_write_cost + output_cost + reasoning_cost) / _MILLION


def estimate_request_cost(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    max_output_tokens: int,
    safety_factor: Decimal | float = Decimal("1"),
) -> Decimal:
    """Conservative pre-call estimate used for reservation."""
    factor = safety_factor if isinstance(safety_factor, Decimal) else Decimal(str(safety_factor))
    cost = (
        Decimal(input_tokens) * pricing.input_usd_per_million
        + Decimal(max_output_tokens) * pricing.output_usd_per_million
    ) / _MILLION
    return cost * factor


@dataclass(frozen=True)
class ModelPricingTable:
    pricing: Mapping[str, ModelPricing]
    strict: bool = True

    def resolve(self, model_name: str) -> ModelPricing:
        if model_name in self.pricing:
            return self.pricing[model_name]
        for prefix in sorted(self.pricing, key=len, reverse=True):
            if model_name.startswith(prefix):
                return self.pricing[prefix]
        if self.strict:
            raise UnknownModelError(model_name)
        return ModelPricing.zero()
