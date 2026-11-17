from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from demo.real_agent import main, trace_from_messages
from resguard.session import JsonlSessionStore


def test_trace_from_messages() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_docs",
                    "args": {"query": "refund"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="30 days", tool_call_id="c1", name="search_docs"),
    ]

    trace = trace_from_messages(messages)

    assert trace == [
        {"name": "search_docs", "args": {"query": "refund"}, "result": "30 days"}
    ]


def test_real_agent_demo_fake_mode(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--fake",
            "--data",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "sessions.jsonl"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "sessions written" in output
    records = list(JsonlSessionStore(tmp_path / "sessions.jsonl").iter_sessions())
    assert len(records) == 3
    assert all(record.messages for record in records)
    assert all(record.trace for record in records)
    assert all(record.success for record in records)
    assert all(record.metrics.tool_calls >= 1 for record in records)
