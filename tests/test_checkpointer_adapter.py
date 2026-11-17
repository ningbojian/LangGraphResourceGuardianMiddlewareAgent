from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from resguard.integrations.checkpointer import CheckpointerSessionStore
from tests.fakes import ScriptedChatModel, ai_final, ai_tool_call, make_echo_tool


def build_agent() -> tuple[object, InMemorySaver, list[str]]:
    echo, calls = make_echo_tool()
    model = ScriptedChatModel(
        responses=[ai_tool_call(call_id="c1"), ai_final("done")]
    )
    saver = InMemorySaver()
    agent = create_agent(
        model=model, tools=[echo], middleware=[], checkpointer=saver
    )
    agent.invoke(
        {"messages": [HumanMessage("hi")]},
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": 20},
    )
    return agent, saver, calls


def test_checkpointer_store_reads_final_state() -> None:
    agent, _, _ = build_agent()
    store = CheckpointerSessionStore(graph=agent, thread_ids=["t1"])

    records = list(store.iter_sessions())

    assert len(records) == 1
    record = records[0]
    assert record.session_id == "thread:t1"
    assert record.messages
    assert record.metrics.tool_calls == 1
    assert record.metrics.checkpoint_writes >= 1
    assert record.metrics.checkpoint_bytes > 0
    assert record.trace[0]["name"] == "echo"
    assert record.success is None


def test_checkpointer_store_accepts_checkpointer_and_label_fn() -> None:
    _, saver, _ = build_agent()
    store = CheckpointerSessionStore(
        checkpointer=saver,
        thread_ids=["t1"],
        label_fn=lambda thread_id, messages: True,
        query_type_fn=lambda thread_id: "refund",
    )

    record = next(store.iter_sessions())

    assert record.success is True
    assert record.label_source == "human"
    assert record.query_type == "refund"
