"""Small end-to-end demo: sessions → OptimizerAgent → proposal → evaluation.

Runs without an API key. The optimizer is a real LangGraph StateGraph;
``RuleBrain`` is a deterministic stand-in for an LLM, and ``ScenarioEvaluator``
executes the real guard on scripted scenarios. It is still a mechanics demo,
not a replay of production traces. Pass ``--interactive`` to see the HITL
interrupt and resume path.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from langgraph.types import Command

from resguard.evaluator import ScenarioEvaluator, ScenarioSpec
from resguard.metrics import SessionMetrics
from resguard.optimizer_agent import OptimizerAgent, RuleBrain
from resguard.resources import ToolResourceProfile
from resguard.session import JsonlSessionStore, SessionRecord


def scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="cache_repeat",
            calls=tuple(("search_docs", {"text": "refund"}) for _ in range(5)),
            tool_profiles={
                "search_docs": ToolResourceProfile(usd_per_call=0.01, cacheable=True)
            },
            budget_usd=1.0,
        ),
        ScenarioSpec(
            name="api_quota_burst",
            calls=tuple(("parse_pdf", {"text": "doc"}) for _ in range(5)),
            tool_profiles={
                "parse_pdf": ToolResourceProfile(
                    usd_per_call=0.0, api_quota_per_call=1
                )
            },
            budget_usd=1.0,
            api_quota_limit=2,
        ),
        ScenarioSpec(
            name="legitimate_long_task",
            calls=tuple(
                ("search_docs", {"text": f"query-{index}"}) for index in range(8)
            ),
            tool_profiles={
                "search_docs": ToolResourceProfile(
                    usd_per_call=0.001, cacheable=False
                )
            },
            budget_usd=10.0,
            legitimate=True,
        ),
    ]


def seed_sessions(path: Path) -> JsonlSessionStore:
    store = JsonlSessionStore(path)
    for index in range(3):
        store.append(
            SessionRecord(
                session_id=f"demo-{index}",
                started_at=f"2026-01-0{index + 1}T00:00:00+00:00",
                finished_at=f"2026-01-0{index + 1}T00:00:01+00:00",
                query_type="refund",
                metrics=SessionMetrics(
                    tool_calls=2,
                    db_queries=4,
                    db_cu=2.0,
                    llm_usd=0.01,
                    checkpoint_bytes=4096,
                ),
                success=True,
                label_source="human",
                termination_reason="db_cu_exhausted",
            )
        )
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="resguard optimizer agent demo")
    parser.add_argument("--interactive", action="store_true", help="pause for human approval")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        store = seed_sessions(Path(tmp) / "sessions.jsonl")
        evaluator = ScenarioEvaluator(scenario_specs())
        agent = OptimizerAgent(
            store=store,
            brain=RuleBrain(),
            evaluator=evaluator,
            require_approval=args.interactive,
        )
        config = {"configurable": {"thread_id": "demo-thread"}}
        initial = {"batch_id": "demo-batch"}

        if args.interactive:
            state = dict(agent.graph.invoke(initial, config))
            if "__interrupt__" in state:
                print("Optimizer paused for human approval:")
                print(state["__interrupt__"])
                decision = input("approve/reject [approve]: ").strip() or "approve"
                state = dict(agent.graph.invoke(Command(resume=decision), config))
        else:
            for update in agent.graph.stream(initial, config, stream_mode="updates"):
                for node, payload in update.items():
                    print(f"[{node}]")
                    if node == "diagnose":
                        print("  focus:", payload.get("focus"))
                    if node == "hypothesize":
                        print("  proposals:", [item.get("name") for item in payload.get("proposals", [])])
                    if node == "evaluate":
                        print("  evaluation:", payload.get("evaluations"))
            state = dict(agent.graph.get_state(config).values)

        print()
        print("termination_reason:", state.get("termination_reason"))
        print("accepted_proposal_id:", state.get("accepted_proposal_id"))
        print("approved_proposal_id:", state.get("approved_proposal_id"))
        print("evaluations:", len(state.get("evaluations", [])))
        print("trace:", state.get("trace"))
        print()
        baseline = evaluator.baseline_insights
        print("baseline resource_per_success:", baseline.resource_per_success)
        print("baseline false_positive_rate:", baseline.false_positive_rate)
        candidate = state.get("candidate_insights")
        if isinstance(candidate, dict):
            print("candidate resource_per_success:", candidate.get("resource_per_success"))
            print("candidate false_positive_rate:", candidate.get("false_positive_rate"))
            print("candidate total tool_calls:", candidate.get("total_metrics", {}).get("tool_calls"))


if __name__ == "__main__":
    main()
