from __future__ import annotations

from langchain_core.messages import AIMessage

from resguard.evaluator import PolicyConfig, ScenarioEvaluator, ScenarioSpec, run_scenario
from resguard.judge import DeterministicQualityJudge, LLMQualityJudge
from resguard.resources import ToolResourceProfile
from resguard.testing import ScriptedChatModel


def spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="judge",
        calls=(("search_docs", {"text": "refund"}),),
        tool_profiles={
            "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=False)
        },
        budget_usd=1.0,
        expected_marker="never-present",
    )


def test_llm_judge_parses_json_score() -> None:
    judge = LLMQualityJudge(
        ScriptedChatModel(responses=[AIMessage(content='{"score": 0.8}')])
    )

    assert judge.score("answer") == 0.8


def test_llm_judge_parses_plain_float() -> None:
    judge = LLMQualityJudge(ScriptedChatModel(responses=[AIMessage(content="0.6")]))

    assert judge.score("answer") == 0.6


def test_llm_judge_invalid_response_falls_back() -> None:
    judge = LLMQualityJudge(
        ScriptedChatModel(responses=[AIMessage(content="no score here")]),
        fallback=DeterministicQualityJudge("answer"),
    )

    assert judge.score("answer") == 1.0


def test_llm_judge_invalid_response_without_fallback_is_zero() -> None:
    judge = LLMQualityJudge(
        ScriptedChatModel(responses=[AIMessage(content="no score here")])
    )

    assert judge.score("answer") == 0.0


def test_llm_judge_in_scenario_evaluator() -> None:
    judge = LLMQualityJudge(
        ScriptedChatModel(
            responses=[
                AIMessage(content='{"score": 0.8}'),
                AIMessage(content='{"score": 0.8}'),
            ]
        )
    )

    result = run_scenario(spec(), PolicyConfig(), judge=judge)
    evaluator = ScenarioEvaluator([spec()], judge=judge)

    assert result.metrics.answer_quality == 0.8
    assert result.success is True
    assert evaluator.baseline_insights.quality_rate == 0.8
