from __future__ import annotations

import pytest

from benchmark.scenarios import run_case, run_mixed_case


def test_full_guard_stops_loop_at_budget() -> None:
    result = run_case(
        "identical_loop",
        "full",
        calls=5,
        tool_cost=0.01,
        budget=0.025,
        official_limit=2,
    )

    assert result.executed_tool_calls == 2
    assert result.attempted_tool_calls == 5
    assert result.declared_usd == 0.02
    assert result.termination == "guard_stop"


def test_no_guard_executes_all_scripted_calls() -> None:
    result = run_case(
        "identical_loop",
        "none",
        calls=5,
        tool_cost=0.01,
        budget=0.025,
        official_limit=2,
    )

    assert result.executed_tool_calls == 5
    assert result.declared_usd == 0.05


def test_official_limit_stops_loop() -> None:
    result = run_case(
        "identical_loop",
        "official",
        calls=5,
        tool_cost=0.01,
        budget=0.025,
        official_limit=2,
    )

    assert result.executed_tool_calls == 2
    assert result.termination == "guard_stop"


def test_mixed_full_guard_respects_budget() -> None:
    result = run_mixed_case("full")

    assert result.executed_tool_calls == 9
    assert result.declared_usd == pytest.approx(0.208)
    assert result.within_budget is True
    assert result.termination == "guard_stop"


def test_mixed_official_limit_cannot_see_cost() -> None:
    result = run_mixed_case("official")

    assert result.executed_tool_calls == 10
    assert result.declared_usd == pytest.approx(0.408)
    assert result.within_budget is False

