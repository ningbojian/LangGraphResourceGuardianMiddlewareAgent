from __future__ import annotations

from decimal import Decimal

import pytest

from resguard.costs import (
    ModelPricing,
    ModelPricingTable,
    TokenUsage,
    UnknownModelError,
    cost_for_usage,
    estimate_request_cost,
)


def test_cost_for_usage_basic() -> None:
    pricing = ModelPricing(
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("2"),
    )

    cost = cost_for_usage(pricing, TokenUsage(input_tokens=1_000, output_tokens=500))

    assert cost == Decimal("0.002")


def test_cached_input_is_billed_at_cached_price() -> None:
    pricing = ModelPricing(
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("0"),
        cached_input_usd_per_million=Decimal("0.1"),
    )

    cost = cost_for_usage(
        pricing,
        TokenUsage(input_tokens=1_000, cached_input_tokens=800, output_tokens=0),
    )

    assert cost == Decimal("0.00028")


def test_cache_creation_is_billed_at_write_price() -> None:
    pricing = ModelPricing(
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("0"),
        cache_write_usd_per_million=Decimal("1.25"),
    )

    cost = cost_for_usage(
        pricing,
        TokenUsage(input_tokens=1_000, cache_creation_tokens=1_000),
    )

    assert cost == Decimal("0.00125")


def test_reasoning_tokens_use_reasoning_price_when_configured() -> None:
    pricing = ModelPricing(
        input_usd_per_million=Decimal("0"),
        output_usd_per_million=Decimal("2"),
        reasoning_usd_per_million=Decimal("4"),
    )

    cost = cost_for_usage(
        pricing,
        TokenUsage(input_tokens=0, output_tokens=1_000, reasoning_tokens=400),
    )

    # 600 normal output @ $2/M + 400 reasoning @ $4/M
    assert cost == Decimal("0.0028")


def test_usage_from_message_metadata() -> None:
    usage = TokenUsage.from_usage_metadata(
        {
            "input_tokens": 1_200,
            "output_tokens": 300,
            "input_token_details": {"cache_read": 500, "cache_creation": 100},
            "output_token_details": {"reasoning": 50},
        }
    )

    assert usage.input_tokens == 1_200
    assert usage.output_tokens == 300
    assert usage.cached_input_tokens == 500
    assert usage.cache_creation_tokens == 100
    assert usage.reasoning_tokens == 50


def test_usage_from_missing_metadata_is_zero() -> None:
    usage = TokenUsage.from_usage_metadata(None)

    assert usage == TokenUsage()


def test_pricing_table_exact_and_longest_prefix() -> None:
    table = ModelPricingTable(
        {
            "gpt-4o": ModelPricing(Decimal("1"), Decimal("2")),
            "gpt-4o-2024-08-06": ModelPricing(Decimal("0.5"), Decimal("1")),
        }
    )

    assert table.resolve("gpt-4o").input_usd_per_million == Decimal("1")
    assert table.resolve("gpt-4o-2024-08-06").input_usd_per_million == Decimal("0.5")
    assert table.resolve("gpt-4o-mini").input_usd_per_million == Decimal("1")


def test_unknown_model_strict_raises() -> None:
    table = ModelPricingTable({}, strict=True)

    with pytest.raises(UnknownModelError):
        table.resolve("unknown-model")


def test_unknown_model_non_strict_returns_zero_pricing() -> None:
    table = ModelPricingTable({}, strict=False)

    pricing = table.resolve("unknown-model")

    assert pricing.input_usd_per_million == Decimal("0")
    assert pricing.output_usd_per_million == Decimal("0")


def test_estimate_request_cost_applies_safety_factor() -> None:
    pricing = ModelPricing(Decimal("1"), Decimal("2"))

    estimate = estimate_request_cost(
        pricing,
        input_tokens=1_000,
        max_output_tokens=500,
        safety_factor=Decimal("1.1"),
    )

    assert estimate == Decimal("0.0022")


def test_zero_usage_is_free() -> None:
    pricing = ModelPricing(Decimal("3"), Decimal("9"))

    assert cost_for_usage(pricing, TokenUsage()) == Decimal("0")
