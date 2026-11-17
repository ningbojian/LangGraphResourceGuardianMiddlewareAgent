from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict

from resguard.cli import main
from resguard.metrics import SessionMetrics
from resguard.session import JsonlSessionStore, SessionRecord, now_iso


def write_sessions(tmp_path):
    path = tmp_path / "sessions.jsonl"
    store = JsonlSessionStore(path)
    for index in range(3):
        store.append(
            SessionRecord(
                session_id=f"s{index}",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:01+00:00",
                query_type="refund",
                metrics=SessionMetrics(tool_calls=2, db_cu=2.0, llm_usd=0.01),
                success=True,
                label_source="human",
                termination_reason="db_cu_exhausted",
            )
        )
    return path


def test_cli_stream_dry_run(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    sessions = write_sessions(tmp_path)
    out = tmp_path / "out"

    exit_code = main(
        ["optimize", "--sessions", str(sessions), "--out", str(out), "--stream"]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "[observe]" in captured
    assert "[finalize]" in captured
    assert (out / "report.md").exists()
    assert (out / "raw.json").exists()
    assert (out / "audit.jsonl").exists()
    assert not (tmp_path / "policies" / "versions").exists()


def test_cli_apply_auto_approve(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sessions = write_sessions(tmp_path)
    out = tmp_path / "out"

    exit_code = main(
        [
            "optimize",
            "--sessions",
            str(sessions),
            "--out",
            str(out),
            "--apply",
            "--auto-approve",
            "--canary",
        ]
    )

    assert exit_code == 0
    versions = list((tmp_path / "policies" / "versions").glob("v*.json"))
    assert len(versions) == 1
    assert (out / "canary_plan.json").exists()


def write_dataset(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    case = {
        "name": "cache_repeat",
        "calls": [
            ["search_docs", {"text": "refund"}],
            ["search_docs", {"text": "refund"}],
        ],
        "tool_profiles": {"search_docs": {"usd_per_call": 0.01, "cacheable": True}},
        "budget_usd": 1.0,
    }
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    return path


def write_policy(tmp_path, name: str, payload: dict):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_batch_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = write_dataset(tmp_path)
    policy = write_policy(tmp_path, "policy.json", {"budget_usd": 1.0})
    out = tmp_path / "runs"

    exit_code = main(
        ["batch-run", "--dataset", str(dataset), "--policy", str(policy), "--out", str(out)]
    )

    assert exit_code == 0
    assert (out / "batch_run.json").exists()
    assert (out / "batch_run.md").exists()


def test_cli_evaluate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = write_dataset(tmp_path)
    baseline = write_policy(
        tmp_path, "baseline.json", {"budget_usd": 1.0, "cache_enabled": False}
    )
    candidate = write_policy(
        tmp_path, "candidate.json", {"budget_usd": 1.0, "cache_enabled": True}
    )
    out = tmp_path / "eval"

    exit_code = main(
        [
            "evaluate",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--dataset",
            str(dataset),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "evaluate.json").read_text(encoding="utf-8"))
    assert report["result"]["accepted"] is True
    assert report["candidate"]["resource_per_success"]["usd"] < report["baseline"][
        "resource_per_success"
    ]["usd"]
    assert (out / "evaluate.html").exists()


def test_cli_evaluate_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def usage():
        return {"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500}

    messages = [
        HumanMessage("refund?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_docs",
                    "args": {"text": "refund"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
            usage_metadata=usage(),
        ),
        ToolMessage(content="doc", tool_call_id="c1", name="search_docs"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_docs",
                    "args": {"text": "refund"},
                    "id": "c2",
                    "type": "tool_call",
                }
            ],
            usage_metadata=usage(),
        ),
        ToolMessage(content="doc", tool_call_id="c2", name="search_docs"),
        AIMessage(content="done", usage_metadata=usage()),
    ]
    sessions = tmp_path / "sessions.jsonl"
    store = JsonlSessionStore(sessions)
    store.append(
        SessionRecord(
            session_id="recorded-1",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            query_type="refund",
            model_version="fake-model",
            metrics=SessionMetrics(tool_calls=2, tool_usd=0.02),
            success=True,
            label_source="human",
            messages=messages_to_dict(messages),
        )
    )
    profiles = write_policy(
        tmp_path,
        "profiles.json",
        {"search_docs": {"usd_per_call": 0.01, "cacheable": True}},
    )
    baseline = write_policy(
        tmp_path, "baseline.json", {"budget_usd": 1.0, "cache_enabled": False}
    )
    candidate = write_policy(
        tmp_path, "candidate.json", {"budget_usd": 1.0, "cache_enabled": True}
    )
    out = tmp_path / "eval-sessions"

    exit_code = main(
        [
            "evaluate",
            "--sessions",
            str(sessions),
            "--tool-profiles",
            str(profiles),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--out",
            str(out),
            "--sessions-per-month",
            "1000",
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "evaluate.json").read_text(encoding="utf-8"))
    assert report["result"]["accepted"] is True
    assert (out / "evaluate.html").exists()
    assert "resguard optimization report" in (out / "evaluate.html").read_text(
        encoding="utf-8"
    )


def test_cli_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    evaluate_path = tmp_path / "evaluate.json"
    evaluate_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "resource_per_success": {"usd": 0.04},
                    "success_rate": 1.0,
                    "quality_rate": 1.0,
                    "false_positive_rate": 0.0,
                },
                "candidate": {
                    "resource_per_success": {"usd": 0.02},
                    "success_rate": 1.0,
                    "quality_rate": 1.0,
                    "false_positive_rate": 0.0,
                },
                "result": {"accepted": True, "reasons": [], "improvements": {}},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.html"

    exit_code = main(["report", "--input", str(evaluate_path), "--out", str(out)])

    assert exit_code == 0
    assert out.exists()
    assert "resguard optimization report" in out.read_text(encoding="utf-8")



def write_drift_sessions(tmp_path, name: str, success: bool):
    path = tmp_path / name
    store = JsonlSessionStore(path)
    for index in range(5):
        store.append(
            SessionRecord(
                session_id=f"{name}-{index}",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:01+00:00",
                query_type="refund",
                metrics=SessionMetrics(tool_calls=2, llm_usd=0.01),
                success=success,
                label_source="human",
            )
        )
    return path


def test_cli_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    baseline = write_drift_sessions(tmp_path, "baseline.jsonl", True)
    recent = write_drift_sessions(tmp_path, "recent.jsonl", False)
    out = tmp_path / "drift"

    exit_code = main(
        [
            "drift",
            "--baseline-sessions",
            str(baseline),
            "--recent-sessions",
            str(recent),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "drift.json").read_text(encoding="utf-8"))
    assert report["drift_detected"] is True
    assert "success_rate_drift" in report["reasons"]


def test_cli_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = write_policy(tmp_path, "workflow.json", {"name": "reference"})
    proposal = write_policy(
        tmp_path,
        "proposal.json",
        {
            "id": "p1",
            "name": "cache",
            "description": "",
            "workflow_changes": [
                {
                    "kind": "add_cache_node",
                    "target": "search_docs",
                    "config": {"ttl_seconds": 300},
                    "rationale": "x",
                },
                {
                    "kind": "model_routing",
                    "target": "default",
                    "config": {"model_tier": "cheap"},
                    "rationale": "x",
                },
            ],
            "evidence": ["e"],
        },
    )
    out = tmp_path / "workflow-out"

    exit_code = main(
        [
            "workflow",
            "--spec",
            str(spec),
            "--proposal",
            str(proposal),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads((out / "workflow_spec.json").read_text(encoding="utf-8"))
    assert payload["after"]["cache_enabled"] is True
    assert payload["after"]["model_tier"] == "cheap"
    assert payload["policy_config"]["cache_enabled"] is True




def test_cli_interactive_approve(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "approve")
    sessions = write_sessions(tmp_path)
    out = tmp_path / "out"

    exit_code = main(
        [
            "optimize",
            "--sessions",
            str(sessions),
            "--out",
            str(out),
            "--interactive",
            "--apply",
        ]
    )

    assert exit_code == 0
    assert list((tmp_path / "policies" / "versions").glob("v*.json"))


def test_cli_interactive_reject(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "reject")
    sessions = write_sessions(tmp_path)
    out = tmp_path / "out"

    exit_code = main(
        [
            "optimize",
            "--sessions",
            str(sessions),
            "--out",
            str(out),
            "--interactive",
            "--apply",
        ]
    )

    assert exit_code == 1
    assert not (tmp_path / "policies" / "versions").exists()


def test_cli_apply_requires_approval(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sessions = write_sessions(tmp_path)
    out = tmp_path / "out"

    exit_code = main(
        ["optimize", "--sessions", str(sessions), "--out", str(out), "--apply"]
    )

    assert exit_code == 1
    assert not (tmp_path / "policies" / "versions").exists()


def test_cli_no_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sessions = tmp_path / "empty.jsonl"
    sessions.write_text("", encoding="utf-8")
    out = tmp_path / "out"

    exit_code = main(["optimize", "--sessions", str(sessions), "--out", str(out)])

    assert exit_code == 1


def test_cli_relative_since_keeps_recent_sessions(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "recent.jsonl"
    store = JsonlSessionStore(path)
    store.append(
        SessionRecord(
            session_id="old",
            started_at="2020-01-01T00:00:00+00:00",
            finished_at="2020-01-01T00:00:01+00:00",
            query_type="refund",
            metrics=SessionMetrics(tool_calls=2, db_cu=2.0, llm_usd=0.01),
            success=True,
            label_source="human",
            termination_reason="db_cu_exhausted",
        )
    )
    store.append(
        SessionRecord(
            session_id="recent",
            started_at=now_iso(),
            finished_at=now_iso(),
            query_type="refund",
            metrics=SessionMetrics(tool_calls=2, db_cu=2.0, llm_usd=0.01),
            success=True,
            label_source="human",
            termination_reason="db_cu_exhausted",
        )
    )
    out = tmp_path / "out"

    exit_code = main(
        ["optimize", "--sessions", str(path), "--since", "1d", "--out", str(out), "--stream"]
    )

    assert exit_code == 0
    assert "[observe]" in capsys.readouterr().out
    import json

    raw = json.loads((out / "raw.json").read_text(encoding="utf-8"))
    assert raw["insights"]["session_count"] == 1


