from __future__ import annotations

import pytest

from resguard.resources import (
    RESOURCE_SPECS,
    ResourceAmount,
    ResourceDelta,
    ResourceKind,
    ResourceLimits,
    ResourceMode,
    ToolResourceProfile,
)


def test_missing_kinds_default_to_zero() -> None:
    amount = ResourceAmount({ResourceKind.USD: 1.5})

    assert amount.get(ResourceKind.USD) == 1.5
    assert amount.get(ResourceKind.DB_CU) == 0.0
    assert amount.get(ResourceKind.CONCURRENCY) == 0.0


def test_amount_add_and_subtract() -> None:
    left = ResourceAmount({ResourceKind.USD: 1.0, ResourceKind.DB_QUERIES: 2})
    right = ResourceAmount({ResourceKind.USD: 0.5, ResourceKind.DB_CU: 3.0})

    total = left + right
    delta = left - right

    assert total.get(ResourceKind.USD) == 1.5
    assert total.get(ResourceKind.DB_QUERIES) == 2
    assert total.get(ResourceKind.DB_CU) == 3.0
    assert isinstance(delta, ResourceDelta)
    assert delta.get(ResourceKind.USD) == 0.5
    assert delta.get(ResourceKind.DB_CU) == -3.0


def test_amount_le_compares_every_dimension_with_missing_as_zero() -> None:
    small = ResourceAmount({ResourceKind.USD: 1.0})
    large = ResourceAmount({ResourceKind.USD: 1.0, ResourceKind.DB_CU: 5.0})

    assert small <= large
    assert not (large <= small)


def test_limits_binding_returns_the_first_exhausted_kind() -> None:
    limits = ResourceLimits({ResourceKind.USD: 1.0, ResourceKind.DB_CU: 10.0})

    assert limits.binding(ResourceAmount({ResourceKind.USD: 0.99})) is None
    assert limits.binding(ResourceAmount({ResourceKind.USD: 1.01})) is ResourceKind.USD
    assert limits.binding(ResourceAmount({ResourceKind.DB_CU: 11.0})) is ResourceKind.DB_CU


def test_limits_unbounded_kind() -> None:
    limits = ResourceLimits({ResourceKind.USD: 1.0})

    assert limits.get(ResourceKind.USD) == 1.0
    assert limits.get(ResourceKind.API_QUOTA) is None


def test_tool_profile_per_call_amount() -> None:
    profile = ToolResourceProfile(
        usd_per_call=0.02,
        db_queries_per_call=2,
        db_cu_per_call=0.5,
        api_quota_per_call=1,
        concurrency_slots=1,
    )

    amount = profile.per_call_amount()

    assert amount.get(ResourceKind.USD) == 0.02
    assert amount.get(ResourceKind.DB_QUERIES) == 2
    assert amount.get(ResourceKind.DB_CU) == 0.5
    assert amount.get(ResourceKind.API_QUOTA) == 1
    assert amount.get(ResourceKind.CONCURRENCY) == 1


def test_tool_profile_defaults_are_conservative() -> None:
    profile = ToolResourceProfile()

    assert profile.usd_per_call == 0.0
    assert profile.db_queries_per_call == 0
    assert profile.idempotent is True
    assert profile.cacheable is True


def test_resource_specs_declare_modes() -> None:
    assert RESOURCE_SPECS[ResourceKind.USD].mode is ResourceMode.CUMULATIVE
    assert RESOURCE_SPECS[ResourceKind.CONCURRENCY].mode is ResourceMode.GAUGE
    assert RESOURCE_SPECS[ResourceKind.DB_CU].unit == "cu"


def test_amount_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        ResourceAmount({ResourceKind.USD: -0.01})
