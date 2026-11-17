from __future__ import annotations

from decimal import Decimal

import pytest

from resguard.evaluator import (
    PolicyConfig,
    ScenarioEvaluator,
    ScenarioSpec,
    aggregate,
    run_scenario,
)
from resguard.metrics import SessionMetrics
from resguard.optimizer import ParetoGate
from resguard.optimizer_agent import OptimizerAgent, ScriptedBrain
from resguard.proposer import ParameterChange, PolicyProposal, WorkflowChange
from resguard.resources import ToolResourceProfile
from resguard.session import JsonlSessionStore


def cache_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="cache_repeat",
        calls=(("search_docs", {"text": "refund"}),) * 5,
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)
        },
        budget_usd=1.0,
    )


def model_routing_spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="single_call",
        calls=(("search_docs", {"text": "refund"}),),
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=False)
        },
        budget_usd=1.0,
    )


def cache_proposal() -> PolicyProposal:
    return PolicyProposal(
        id="p-cache",
        name="cache",
        description="cache repeated reads",
        workflow_changes=[
            WorkflowChange("add_cache_node", "search_docs", {"ttl_seconds": 300}, "x")
        ],
        evidence=["cache_repeat"],
    )


def routing_proposal() -> PolicyProposal:
    return PolicyProposal(
        id="p-routing",
        name="route",
        description="use cheaper model",
        workflow_changes=[WorkflowChange("model_routing", "default", {}, "x")],
        evidence=["single_call"],
    )


def test_cache_proposal_reduces_tool_calls_and_usd() -> None:
    evaluator = ScenarioEvaluator([cache_spec()])
    baseline = evaluator.baseline_insights

    candidate = evaluator.evaluate(cache_proposal(), baseline)

    assert baseline.total_metrics.tool_calls == 5
    assert candidate.total_metrics.tool_calls == 1
    assert candidate.resource_per_success["usd"] < baseline.resource_per_success["usd"]
    assert candidate.proxy_label is True


def test_model_routing_reduces_model_usd() -> None:
    evaluator = ScenarioEvaluator([model_routing_spec()])
    baseline = evaluator.baseline_insights

    candidate = evaluator.evaluate(routing_proposal(), baseline)

    assert candidate.resource_per_success["usd"] < baseline.resource_per_success["usd"]


def test_policy_config_maps_proposal_changes() -> None:
    proposal = PolicyProposal(
        id="p",
        name="combo",
        description="",
        workflow_changes=[
            WorkflowChange("add_cache_node", "x", {}, ""),
            WorkflowChange("model_routing", "x", {}, ""),
            WorkflowChange("set_max_concurrency", "x", {"max_concurrency": 2}, ""),
        ],
        parameter_changes=[
            ParameterChange("limits.concurrency", None, 3, ""),
            ParameterChange("limits.budget_usd", None, 0.5, ""),
        ],
        evidence=["e"],
    )

    config = PolicyConfig().with_proposal(proposal)

    assert config.cache_enabled is True
    assert config.model_cost_factor == Decimal("0.7")
    assert config.max_concurrency == 3
    assert config.budget_usd == 0.5


def test_scenario_evaluator_in_optimizer_graph(tmp_path) -> None:
    evaluator = ScenarioEvaluator([cache_spec()])
    store = JsonlSessionStore(tmp_path / "empty.jsonl")
    agent = OptimizerAgent(
        store=store,
        brain=ScriptedBrain(focus="cache", proposals=[cache_proposal()]),
        evaluator=evaluator,
    )

    result = agent.graph.invoke(
        {"batch_id": "scenario"}, config={"configurable": {"thread_id": "scenario"}}
    )

    assert result["termination_reason"] == "accepted"
    assert result["accepted_proposal_id"] == "p-cache"
    assert len(result["evaluations"]) == 1
    assert result["evaluations"][0]["accepted"] is True


def test_run_scenario_reports_metrics() -> None:
    result = run_scenario(model_routing_spec(), PolicyConfig())

    assert result.executed_tool_calls == 1
    assert result.metrics.tool_usd == pytest.approx(0.01)
    assert result.metrics.llm_usd > 0
    assert result.success is True


def test_api_quota_limit_stops_calls() -> None:
    spec = ScenarioSpec(
        name="quota_burst",
        calls=(("parse_pdf", {"text": "doc"}),) * 5,
        tool_profiles={
            "parse_pdf": ToolResourceProfile(usd_per_call=0.0, api_quota_per_call=1)
        },
        budget_usd=1.0,
        api_quota_limit=2,
    )

    result = run_scenario(spec, PolicyConfig())

    assert result.executed_tool_calls == 2
    assert result.metrics.api_quota == 2
    assert result.false_positive is False


def test_parallel_tool_calls_respect_concurrency_limit() -> None:
    spec = ScenarioSpec(
        name="parallel_db",
        calls=tuple(
            ("search_docs", {"text": f"query-{index}"}) for index in range(4)
        ),
        tool_profiles={
            "search_docs": ToolResourceProfile(
                usd_per_call=0.0, concurrency_slots=1
            )
        },
        budget_usd=10.0,
        concurrency_limit=2,
        parallel_group=4,
    )

    result = run_scenario(spec, PolicyConfig())

    assert result.metrics.peak_concurrency <= 2
    assert result.executed_tool_calls + result.metrics.concurrency_denials == 4



def test_legitimate_long_task_has_no_false_positive() -> None:
    spec = ScenarioSpec(
        name="legitimate",
        calls=tuple(
            ("search_docs", {"text": f"query-{index}"}) for index in range(8)
        ),
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.001, cacheable=False)
        },
        budget_usd=10.0,
        legitimate=True,
    )

    result = run_scenario(spec, PolicyConfig())
    insights = aggregate([result])

    assert result.executed_tool_calls == 8
    assert result.false_positive is False
    assert insights.false_positive_rate == 0.0


def test_cache_version_key_prevents_stale_hits() -> None:
    spec = ScenarioSpec(
        name="cache_version",
        calls=(
            ("search_docs", {"text": "refund", "doc_version": "v1"}),
            ("search_docs", {"text": "refund", "doc_version": "v2"}),
        ),
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)
        },
        budget_usd=1.0,
    )
    evaluator = ScenarioEvaluator([spec])
    baseline = evaluator.baseline_insights

    candidate = evaluator.evaluate(cache_proposal(), baseline)

    assert baseline.total_metrics.tool_calls == 2
    assert candidate.total_metrics.tool_calls == 2
    assert candidate.total_metrics.cache_hits == 0


def test_pareto_gate_rejects_false_positive_regression() -> None:
    from resguard.batch import BatchInsights

    baseline = BatchInsights(session_count=1, success_rate=1.0, false_positive_rate=0.0)
    candidate = BatchInsights(session_count=1, success_rate=1.0, false_positive_rate=0.2)

    result = ParetoGate().evaluate("p", baseline=baseline, candidate=candidate)

    assert result.accepted is False
    assert any("false_positive" in reason for reason in result.reasons)


def test_answer_quality_marker() -> None:
    good = run_scenario(model_routing_spec(), PolicyConfig())
    bad_spec = ScenarioSpec(
        name="quality_miss",
        calls=(("search_docs", {"text": "refund"}),),
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=False)
        },
        budget_usd=1.0,
        expected_marker="never-present",
    )
    bad = run_scenario(bad_spec, PolicyConfig())

    assert good.metrics.answer_quality == 1.0
    assert good.success is True
    assert bad.metrics.answer_quality == 0.0
    assert bad.success is False


def test_pareto_gate_rejects_quality_regression() -> None:
    from resguard.batch import BatchInsights

    baseline = BatchInsights(session_count=1, success_rate=1.0, quality_rate=1.0)
    candidate = BatchInsights(session_count=1, success_rate=1.0, quality_rate=0.5)

    result = ParetoGate().evaluate("p", baseline=baseline, candidate=candidate)

    assert result.accepted is False
    assert "quality_regression" in result.reasons


