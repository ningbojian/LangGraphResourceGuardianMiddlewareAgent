"""Real end-to-end demo: Anthropic model + two real tools + SessionRecorder.

- ``search_docs`` reads a real local policy markdown file.
- ``lookup_order`` queries a real local SQLite database.
- Each run is guarded by ``ResourceGuardMiddleware`` and recorded by
  ``SessionRecorder`` (messages + tool trace) into a JSONL session store.

Then:

```bash
uv run resguard evaluate --sessions examples/real_agent/sessions.jsonl \
  --tool-profiles examples/real_agent/tool_profiles.json \
  --baseline examples/real_agent/policy_baseline.json \
  --candidate examples/real_agent/policy_candidate.json \
  --out reports/real-agent/ --sessions-per-month 10000 --optimizer-cost-usd 5
```

The candidate policy models a 30%-cheaper model tier and cache-enabled reads;
the report shows the counterfactual resource delta. Use ``--fake`` for a no-key
CI smoke test; it still uses the real tools and recorder.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict
from langchain_core.tools import BaseTool, tool

from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.recorder import SessionRecorder
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile
from resguard.session import JsonlSessionStore
from resguard.testing import ScriptedChatModel, ai_final, ai_parallel_tool_calls

POLICY_MD = """# Refund Policy

Customers may request a refund within 30 days of delivery.
Late deliveries over 7 days qualify for a full refund.

# Returns

Opened items can be returned within 14 days with a receipt.
Unopened items can be returned within 30 days.

# Shipping

Standard shipping takes 3-5 business days.
Express shipping takes 1-2 business days.
"""


def seed_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "policy.md").write_text(POLICY_MD, encoding="utf-8")
    database = data_dir / "orders.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS orders ("
        "order_id TEXT PRIMARY KEY, status TEXT, delivered_days_ago INTEGER, total REAL)"
    )
    connection.execute("DELETE FROM orders")
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [
            ("A1001", "shipped", 0, 49.90),
            ("B2002", "delivered", 10, 79.00),
            ("C3003", "delivered", 40, 120.00),
        ],
    )
    connection.commit()
    connection.close()


def build_tools(data_dir: Path) -> tuple[BaseTool, BaseTool]:
    @tool
    def search_docs(query: str) -> str:
        """Search the local policy documents for a query."""
        text = (data_dir / "policy.md").read_text(encoding="utf-8")
        sections = [section.strip() for section in text.split("\n\n") if section.strip()]
        terms = [term.lower() for term in query.split() if term]
        matches = [
            section
            for section in sections
            if any(term in section.lower() for term in terms)
        ]
        return "\n\n".join(matches[:3]) or "No matching policy section."

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order status in the local SQLite database."""
        connection = sqlite3.connect(data_dir / "orders.db")
        row = connection.execute(
            "SELECT order_id, status, delivered_days_ago, total FROM orders WHERE order_id = ?",
            (order_id.upper(),),
        ).fetchone()
        connection.close()
        if row is None:
            return json.dumps({"order_id": order_id, "error": "not found"})
        return json.dumps(
            {
                "order_id": row[0],
                "status": row[1],
                "delivered_days_ago": row[2],
                "total": row[3],
            }
        )

    return search_docs, lookup_order


def queries() -> list[dict[str, Any]]:
    return [
        {
            "query": "Use search_docs to find the refund window. Then answer briefly.",
            "query_type": "refund",
            "expected": ["30"],
        },
        {
            "query": "Use lookup_order to check order A1001 and answer briefly.",
            "query_type": "order_status",
            "expected": ["A1001"],
        },
        {
            "query": "Use search_docs to find the return policy for opened items.",
            "query_type": "returns",
            "expected": ["14"],
        },
    ]


def trace_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    tool_by_id = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    trace: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            result = tool_by_id.get(tool_call.get("id", ""))
            trace.append(
                {
                    "name": tool_call.get("name", "unknown"),
                    "args": dict(tool_call.get("args") or {}),
                    "result": str(result.content) if result is not None else "",
                }
            )
    return trace


def _model_name(model: Any, fallback: str) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return fallback


def _build_real_model(args: argparse.Namespace) -> Any:
    """Build a real model.

    If ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN are set, use ChatAnthropic
    against that Anthropic-compatible endpoint. The endpoint in this environment
    serves DeepSeek model names (e.g. ``--model deepseek-flash``), not Anthropic
    model names; a real Anthropic key can be used without the proxy.
    """
    _provider, _sep, model_name = args.model.partition(":")
    if not model_name:
        model_name = args.model
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    if not args.direct and base_url and auth_token:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            base_url=base_url,
            api_key=auth_token,
            max_tokens=512,
            timeout=60,
        )
    from langchain.chat_models import init_chat_model

    return init_chat_model(args.model)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part)))
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(content)


def _invoke_real(agent: Any, item: dict[str, Any]) -> Any | None:
    try:
        return agent.invoke(
            {"messages": [HumanMessage(item["query"])]},
            config={"recursion_limit": 20},
        )
    except Exception as exc:
        print(f"Real model call failed: {type(exc).__name__}: {exc}")
        print(
            "Set a valid ANTHROPIC_API_KEY (or another provider/model), "
            "or run with --fake for the no-key pipeline."
        )
        return None


def _fake_model_for(query: dict[str, Any]) -> ScriptedChatModel:
    if query["query_type"] == "order_status":
        call = ("lookup_order", {"order_id": "A1001"})
        final = "Order A1001 is shipped."
    elif query["query_type"] == "returns":
        call = ("search_docs", {"query": "return opened items"})
        final = "Opened items can be returned within 14 days with a receipt."
    else:
        call = ("search_docs", {"query": "refund window"})
        final = "Refunds are available within 30 days of delivery."
    return ScriptedChatModel(
        responses=[
            ai_parallel_tool_calls([call], prefix=query["query_type"]),
            ai_final(final),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="resguard real agent demo")
    parser.add_argument("--model", default="anthropic:claude-haiku-5-5")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="ignore ANTHROPIC_BASE_URL/AUTH_TOKEN and use init_chat_model directly",
    )
    parser.add_argument("--fake", action="store_true", help="no-key CI smoke test")
    parser.add_argument("--data", default="examples/real_agent/data")
    parser.add_argument("--out", default="examples/real_agent/sessions.jsonl")
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--budget", type=float, default=0.10)
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    seed_data(data_dir)
    search_docs, lookup_order = build_tools(data_dir)
    tool_profiles = {
        "search_docs": ToolResourceProfile(
            usd_per_call=0.001,
            db_queries_per_call=1,
            db_cu_per_call=0.01,
            cacheable=True,
            idempotent=True,
            ttl_seconds=300,
        ),
        "lookup_order": ToolResourceProfile(
            usd_per_call=0.002,
            db_queries_per_call=1,
            db_cu_per_call=0.02,
            cacheable=True,
            idempotent=True,
            ttl_seconds=60,
        ),
    }
    store = JsonlSessionStore(args.out)
    selected = queries()[: args.max_sessions]

    if args.fake:
        model = _fake_model_for(selected[0]) if selected else None
        model_name = "fake-model"
    else:
        model = _build_real_model(args)
        model_name = _model_name(model, args.model)

    pricing = ModelPricingTable(
        {
            model_name: ModelPricing(Decimal("0.8"), Decimal("4")),
            args.model: ModelPricing(Decimal("0.8"), Decimal("4")),
        },
        strict=False,
    )

    for index, item in enumerate(selected):
        session_model = _fake_model_for(item) if args.fake else model
        guard = ResourceGuardMiddleware(
            ledger=InMemoryLedger(
                ResourceLimits(
                    {ResourceKind.USD: args.budget, ResourceKind.CONCURRENCY: 2}
                )
            ),
            policy=GuardPolicy(
                soft_thresholds=(0.5, 0.8),
                hard_limit_action=DegradationAction.STOP,
            ),
            tool_profiles=tool_profiles,
            pricing=pricing,
            cache=ToolCache(),
            cache_context=CacheContext(user_id="demo-user"),
            estimate_output_tokens=512,
            run_id=f"real-agent-{index}",
        )
        agent = create_agent(
            model=session_model,
            tools=[search_docs, lookup_order],
            middleware=[guard],
        )
        result = agent.invoke(
            {"messages": [HumanMessage(item["query"])]},
            config={"recursion_limit": 20},
        ) if args.fake else _invoke_real(agent, item)
        if result is None:
            return 2
        messages = result["messages"]
        final_text = _content_text(messages[-1].content)
        expected = [str(value).lower() for value in item.get("expected", [])]
        success = bool(final_text.strip()) and (
            not expected or any(value in final_text.lower() for value in expected)
        )
        recorder = SessionRecorder(
            guard=guard,
            store=store,
            session_id=f"real-agent-{index}",
            query_type=item["query_type"],
            model_version=model_name,
            policy_version="baseline",
            messages=messages_to_dict(messages),
            trace=trace_from_messages(messages),
        )
        record = recorder.finalize(
            success=success,
            termination_reason="completed",
        )
        print(
            f"[{index}] type={item['query_type']} success={success} "
            f"tools={record.metrics.tool_calls} cache_hits={record.metrics.cache_hits} "
            f"usd={record.metrics.llm_usd + record.metrics.tool_usd:.6f}"
        )
        print("  answer:", final_text[:180].replace("\n", " "))

    print(f"\nsessions written: {args.out}")
    print(
        "next: uv run resguard evaluate --sessions "
        f"{args.out} --tool-profiles examples/real_agent/tool_profiles.json "
        "--baseline examples/real_agent/policy_baseline.json "
        "--candidate examples/real_agent/policy_candidate.json "
        "--out reports/real-agent/ --sessions-per-month 10000 --optimizer-cost-usd 5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
