"""Real-LLM brain demo for the offline optimizer.

Run with a real provider:

```bash
uv sync --extra llm
export OPENAI_API_KEY=...
uv run python -m demo.llm_brain --model openai:gpt-4o-mini --stream
```

For a no-key smoke test of the wiring, pass ``--fake``. The fake path uses the
same ``LLMBrain`` adapter and the same ``OptimizerAgent`` graph; only the chat
model is scripted.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.types import Command

from demo.run import scenario_specs, seed_sessions
from resguard.evaluator import ScenarioEvaluator
from resguard.optimizer_agent import LLMBrain, OptimizerAgent
from resguard.testing import ScriptedChatModel


def _fake_proposal() -> dict:
    return {
        "name": "cache_hot_db_reads",
        "description": "Cache repeated DB reads",
        "parameter_changes": [],
        "workflow_changes": [
            {
                "kind": "add_cache_node",
                "target": "search_docs",
                "config": {"ttl_seconds": 300},
                "rationale": "avoid duplicate DB reads",
            }
        ],
        "expected_effect": {"db_cu": "down", "usd": "down"},
        "evidence": ["db_cu_exhausted"],
        "risk": "low",
    }


def fake_brain() -> LLMBrain:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="db_cu_exhausted"),
            AIMessage(content=json.dumps([_fake_proposal()])),
        ]
    )
    return LLMBrain(model)


def real_brain(model_name: str) -> LLMBrain:
    from langchain.chat_models import init_chat_model

    return LLMBrain(init_chat_model(model_name), raise_on_error=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="resguard LLM brain demo")
    parser.add_argument("--model", default=os.getenv("RESGUARD_MODEL", "openai:gpt-4o-mini"))
    parser.add_argument("--fake", action="store_true", help="use a scripted model (no key)")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)

    if args.fake:
        brain = fake_brain()
    else:
        try:
            brain = real_brain(args.model)
        except Exception as exc:
            print(f"Could not initialize model {args.model!r}: {exc}")
            print("Install a provider package with `uv sync --extra llm` and set an API key,")
            print("or run `uv run python -m demo.llm_brain --fake` for a no-key smoke test.")
            return 2

    with tempfile.TemporaryDirectory() as tmp:
        store = seed_sessions(Path(tmp) / "sessions.jsonl")
        evaluator = ScenarioEvaluator(scenario_specs())
        agent = OptimizerAgent(
            store=store,
            brain=brain,
            evaluator=evaluator,
            require_approval=args.interactive,
        )
        config = {"configurable": {"thread_id": "llm-brain-demo"}}
        initial = {"batch_id": "llm-brain-demo"}

        if args.interactive:
            state = dict(agent.graph.invoke(initial, config))
            if "__interrupt__" in state:
                print("Optimizer paused for human approval.")
                decision = input("approve/reject [approve]: ").strip() or "approve"
                state = dict(agent.graph.invoke(Command(resume=decision), config))
        else:
            if args.stream:
                for update in agent.graph.stream(initial, config, stream_mode="updates"):
                    for node, payload in update.items():
                        print(f"[{node}]")
                        if node == "diagnose":
                            print("  focus:", payload.get("focus"))
                        if node == "hypothesize":
                            print(
                                "  proposals:",
                                [item.get("name") for item in payload.get("proposals", [])],
                            )
                        if node == "evaluate":
                            print("  evaluation:", payload.get("evaluations"))
                state = dict(agent.graph.get_state(config).values)
            else:
                state = dict(agent.graph.invoke(initial, config))

    print("termination_reason:", state.get("termination_reason"))
    print("accepted_proposal_id:", state.get("accepted_proposal_id"))
    print("approved_proposal_id:", state.get("approved_proposal_id"))
    print("evaluations:", len(state.get("evaluations", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
