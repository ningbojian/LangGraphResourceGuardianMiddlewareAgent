from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langgraph.types import Command

from resguard.batch import BatchAnalyzer
from resguard.evaluator import PolicyConfig, ScenarioSpec, aggregate, run_scenario
from resguard.optimizer import AuditLog, EvaluationResult, ParetoGate
from resguard.optimizer_agent import LLMBrain, OptimizerAgent, RuleBrain
from resguard.proposer import PolicyProposal
from resguard.replay import MessageReplayEvaluator
from resguard.report import build_html_report
from resguard.resources import ToolResourceProfile
from resguard.rollout import build_canary_plan, detect_drift
from resguard.session import JsonlSessionStore
from resguard.workflow import (
    WorkflowSpec,
    apply_workflow_changes,
    workflow_spec_to_policy_config,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_since(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if len(text) > 1 and text[-1] in "dhm" and text[:-1].isdigit():
        amount = int(text[:-1])
        unit = text[-1]
        delta = {
            "d": timedelta(days=amount),
            "h": timedelta(hours=amount),
            "m": timedelta(minutes=amount),
        }[unit]
        return (datetime.now(timezone.utc) - delta).isoformat()
    return text


def _load_specs(path: str | Path) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                specs.append(ScenarioSpec.from_dict(json.loads(line)))
    return specs


def _load_policy(path: str | Path) -> PolicyConfig:
    return PolicyConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_tool_profiles(path: str | None) -> dict[str, ToolResourceProfile]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: ToolResourceProfile(**profile)
        for name, profile in data.items()
    }


def _find_proposal(state: dict[str, Any]) -> PolicyProposal | None:
    proposal_id = state.get("approved_proposal_id") or state.get("accepted_proposal_id")
    for item in state.get("proposals", []):
        if item.get("id") == proposal_id:
            return PolicyProposal.from_dict(item)
    return None


def _find_evaluation(state: dict[str, Any], proposal_id: str) -> EvaluationResult | None:
    for item in state.get("evaluations", []):
        if item.get("proposal_id") == proposal_id:
            return EvaluationResult.from_dict(item)
    return None


def _write_report(out: Path, final_state: dict[str, Any], session_count: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(
        json.dumps(_jsonable(final_state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    proposals = final_state.get("proposals", [])
    evaluations = final_state.get("evaluations", [])
    lines = [
        "# resguard optimization report",
        "",
        f"- batch: {final_state.get('batch_id', '')}",
        f"- sessions: {session_count}",
        f"- focus: {final_state.get('focus', '')}",
        f"- proposals: {len(proposals)}",
        f"- accepted proposal: {final_state.get('accepted_proposal_id')}",
        f"- approved proposal: {final_state.get('approved_proposal_id')}",
        f"- termination: {final_state.get('termination_reason')}",
        f"- evaluations: {len(evaluations)}",
        "",
        "## Proposals",
    ]
    for proposal in proposals:
        lines.append(
            f"- `{proposal.get('id')}` {proposal.get('name')}: "
            f"{proposal.get('description', '')} (risk={proposal.get('risk', 'low')})"
        )
    lines.extend(["", "## Evaluations"])
    for evaluation in evaluations:
        lines.append(
            f"- `{evaluation.get('proposal_id')}` accepted={evaluation.get('accepted')} "
            f"reasons={evaluation.get('reasons')} improvements={evaluation.get('improvements')}"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_policy_candidate(final_state: dict[str, Any]) -> Path:
    policy_dir = Path("policies") / "versions"
    policy_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(policy_dir.glob("v*.json"))
    version = len(existing) + 1
    path = policy_dir / f"v{version}.json"
    proposal = _find_proposal(final_state)
    path.write_text(
        json.dumps(
            {
                "version": version,
                "batch_id": final_state.get("batch_id"),
                "proposal": proposal.to_dict() if proposal else None,
                "evaluations": final_state.get("evaluations", []),
                "applied": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _build_brain(args: argparse.Namespace):
    if args.brain == "rule":
        return RuleBrain()
    if not args.model:
        raise SystemExit("--model is required with --brain llm")
    from langchain.chat_models import init_chat_model

    return LLMBrain(init_chat_model(args.model))


def _run_optimize(args: argparse.Namespace) -> int:
    store = JsonlSessionStore(args.sessions)
    since = _parse_since(args.since)
    limit = args.limit if args.limit is not None else args.batch_size
    session_count = len(list(store.iter_sessions(since=since, limit=limit)))
    if session_count == 0:
        print("No sessions found; nothing to optimize.")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "audit.jsonl")
    brain = _build_brain(args)
    require_approval = bool(args.interactive and not args.auto_approve)
    agent = OptimizerAgent(
        store=store,
        brain=brain,
        audit_log=audit,
        require_approval=require_approval,
        since=since,
        limit=limit,
    )
    config = {"configurable": {"thread_id": f"resguard-{args.batch_id}"}}
    initial = {"batch_id": args.batch_id}

    final_state: dict[str, Any]
    if args.interactive:
        if args.stream:
            print("--interactive uses invoke for HITL; --stream is ignored.")
        final_state = dict(agent.graph.invoke(initial, config))
        if "__interrupt__" in final_state:
            print("Optimizer paused for human approval.")
            print(json.dumps(_jsonable(final_state.get("evaluations", [])), indent=2))
            decision = input("approve/reject [approve]: ").strip() or "approve"
            final_state = dict(agent.graph.invoke(Command(resume=decision), config))
    elif args.stream:
        for update in agent.graph.stream(initial, config, stream_mode="updates"):
            for node in update:
                print(f"[{node}]")
        final_state = dict(agent.graph.get_state(config).values)
    else:
        final_state = dict(agent.graph.invoke(initial, config))

    _write_report(out, final_state, session_count)
    approved = bool(final_state.get("approved_proposal_id")) or final_state.get(
        "termination_reason"
    ) in ("accepted", "approved")
    print(f"report: {out / 'report.md'}")
    print(f"termination: {final_state.get('termination_reason')}")

    if args.apply:
        if not approved:
            print("Not applied: no approved proposal.")
            return 1
        if not (args.auto_approve or args.interactive):
            print("Refusing to apply without --interactive or --auto-approve.")
            return 1
        path = _write_policy_candidate(final_state)
        print(f"policy candidate written: {path}")

    if args.canary and approved:
        proposal = _find_proposal(final_state)
        if proposal is not None:
            evaluation = _find_evaluation(final_state, proposal.id)
            plan = build_canary_plan(proposal, evaluation)
            canary_path = out / "canary_plan.json"
            canary_path.write_text(
                json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"canary plan written: {canary_path}")
    return 0


def _run_batch_run(args: argparse.Namespace) -> int:
    specs = _load_specs(args.dataset)
    config = _load_policy(args.policy)
    results = [run_scenario(spec, config) for spec in specs]
    insights = aggregate(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": config.to_dict(),
        "insights": insights.to_dict(),
        "results": [
            {
                "name": result.name,
                "executed_tool_calls": result.executed_tool_calls,
                "termination": result.termination,
                "success": result.success,
                "false_positive": result.false_positive,
                "metrics": result.metrics.to_dict(),
            }
            for result in results
        ],
    }
    (out / "batch_run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# resguard batch run",
        "",
        f"- scenarios: {len(specs)}",
        f"- success_rate: {insights.success_rate:.3f}",
        f"- false_positive_rate: {insights.false_positive_rate:.3f}",
        f"- quality_rate: {insights.quality_rate:.3f}",
        f"- resource_per_success: {insights.resource_per_success}",
    ]
    (out / "batch_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out / 'batch_run.json'}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    baseline_config = _load_policy(args.baseline)
    candidate_config = _load_policy(args.candidate)
    if args.sessions:
        records = list(JsonlSessionStore(args.sessions).iter_sessions())
        profiles = _load_tool_profiles(args.tool_profiles)
        evaluator = MessageReplayEvaluator(
            records,
            profiles,
            budget_usd=baseline_config.budget_usd or 1.0,
        )
        baseline = evaluator.evaluate_config(baseline_config)
        candidate = evaluator.evaluate_config(candidate_config)
    else:
        specs = _load_specs(args.dataset)
        baseline = aggregate(run_scenario(spec, baseline_config) for spec in specs)
        candidate = aggregate(run_scenario(spec, candidate_config) for spec in specs)
    result = ParetoGate().evaluate("evaluate", baseline=baseline, candidate=candidate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_config": baseline_config.to_dict(),
        "candidate_config": candidate_config.to_dict(),
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "result": result.to_dict(),
    }
    (out / "evaluate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html = build_html_report(
        payload,
        sessions_per_month=args.sessions_per_month,
        optimizer_cost_usd=args.optimizer_cost_usd,
    )
    (out / "evaluate.html").write_text(html, encoding="utf-8")
    lines = [
        "# resguard policy evaluation",
        "",
        f"- accepted: {result.accepted}",
        f"- reasons: {result.reasons}",
        f"- improvements: {result.improvements}",
        f"- regressions: {result.regressions}",
        f"- baseline resource_per_success: {baseline.resource_per_success}",
        f"- candidate resource_per_success: {candidate.resource_per_success}",
    ]
    (out / "evaluate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out / 'evaluate.json'}")
    print(f"wrote {out / 'evaluate.html'}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    evaluation = json.loads(Path(args.input).read_text(encoding="utf-8"))
    html = build_html_report(
        evaluation,
        sessions_per_month=args.sessions_per_month,
        optimizer_cost_usd=args.optimizer_cost_usd,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _run_drift(args: argparse.Namespace) -> int:
    baseline = BatchAnalyzer().analyze(
        list(JsonlSessionStore(args.baseline_sessions).iter_sessions())
    )
    recent = BatchAnalyzer().analyze(
        list(JsonlSessionStore(args.recent_sessions).iter_sessions())
    )
    report = detect_drift(baseline, recent)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"drift_detected: {report.drift_detected}")
    print(f"reasons: {report.reasons}")
    print(f"wrote {out / 'drift.json'}")
    return 0


def _run_workflow(args: argparse.Namespace) -> int:
    spec = WorkflowSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    proposal = PolicyProposal.from_dict(
        json.loads(Path(args.proposal).read_text(encoding="utf-8"))
    )
    updated = apply_workflow_changes(spec, proposal)
    config = workflow_spec_to_policy_config(updated)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "before": spec.to_dict(),
        "after": updated.to_dict(),
        "policy_config": config.to_dict(),
        "changes": [change.__dict__ for change in proposal.workflow_changes],
    }
    (out / "workflow_spec.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# resguard workflow change",
        "",
        f"- proposal: {proposal.id} {proposal.name}",
        f"- before: {spec.to_dict()}",
        f"- after: {updated.to_dict()}",
        f"- policy_config: {config.to_dict()}",
    ]
    (out / "workflow_diff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out / 'workflow_spec.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resguard")
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="run the offline optimizer agent on a session batch")
    optimize.add_argument("--sessions", required=True, help="JSONL session store")
    optimize.add_argument("--since", default=None)
    optimize.add_argument("--limit", type=int, default=None)
    optimize.add_argument("--batch-size", type=int, default=None, help="alias for --limit")
    optimize.add_argument("--out", default="proposals")
    optimize.add_argument("--batch-id", default="batch")
    optimize.add_argument("--brain", choices=["rule", "llm"], default="rule")
    optimize.add_argument("--model", default=None)
    optimize.add_argument("--stream", action="store_true", help="print node state transitions")
    optimize.add_argument("--interactive", action="store_true", help="pause for human approval")
    optimize.add_argument("--auto-approve", action="store_true", help="CI only; skips approval")
    optimize.add_argument("--apply", action="store_true", help="write a policy candidate")
    optimize.add_argument("--canary", action="store_true", help="write a canary plan")
    optimize.set_defaults(func=_run_optimize)

    batch_run = sub.add_parser("batch-run", help="run scenario specs through a policy")
    batch_run.add_argument("--dataset", required=True, help="JSONL scenario specs")
    batch_run.add_argument("--policy", required=True, help="policy JSON")
    batch_run.add_argument("--out", default="runs")
    batch_run.set_defaults(func=_run_batch_run)

    evaluate = sub.add_parser("evaluate", help="compare two policies on scenario specs or sessions")
    evaluate.add_argument("--baseline", required=True, help="baseline policy JSON")
    evaluate.add_argument("--candidate", required=True, help="candidate policy JSON")
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="JSONL scenario specs")
    source.add_argument("--sessions", help="JSONL session store with recorded messages")
    evaluate.add_argument("--tool-profiles", default=None, help="JSON tool cost profiles for --sessions")
    evaluate.add_argument("--out", default="reports/eval")
    evaluate.add_argument("--sessions-per-month", type=int, default=10_000)
    evaluate.add_argument("--optimizer-cost-usd", type=float, default=0.0)
    evaluate.set_defaults(func=_run_evaluate)

    drift = sub.add_parser("drift", help="compare recent sessions to a baseline batch")
    drift.add_argument("--baseline-sessions", required=True)
    drift.add_argument("--recent-sessions", required=True)
    drift.add_argument("--out", default="reports/drift")
    drift.set_defaults(func=_run_drift)

    workflow = sub.add_parser("workflow", help="apply a proposal to a reference workflow spec")
    workflow.add_argument("--spec", required=True, help="baseline WorkflowSpec JSON")
    workflow.add_argument("--proposal", required=True, help="PolicyProposal JSON")
    workflow.add_argument("--out", default="reports/workflow")
    workflow.set_defaults(func=_run_workflow)

    report = sub.add_parser("report", help="render an evaluate.json into a self-contained HTML report")
    report.add_argument("--input", required=True, help="evaluate.json")
    report.add_argument("--out", default="reports/report.html")
    report.add_argument("--sessions-per-month", type=int, default=10_000)
    report.add_argument("--optimizer-cost-usd", type=float, default=0.0)
    report.set_defaults(func=_run_report)

    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
