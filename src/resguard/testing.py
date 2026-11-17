"""Deterministic test doubles for scenarios and tests.

This module is packaged so the evaluator can run without API keys. It is not a
production model/tool implementation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from pydantic import ConfigDict, Field


def usage(input_tokens: int = 1_000, output_tokens: int = 500) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def ai_tool_call(
    *,
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {"text": "hi"},
                "id": call_id,
                "type": "tool_call",
            }
        ],
        usage_metadata=usage(),
    )


def ai_final(content: str = "done") -> AIMessage:
    return AIMessage(content=content, usage_metadata=usage())


def ai_parallel_tool_calls(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    prefix: str = "call",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"{prefix}_{index}",
                "type": "tool_call",
            }
            for index, (name, args) in enumerate(calls)
        ],
        usage_metadata=usage(),
    )


class ScriptedChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[AIMessage] = Field(default_factory=list)
    model_name: str = "fake-model"
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(list(messages))
        response = self.responses.pop(0) if self.responses else ai_final("done")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


def make_counting_tool(name: str, calls: list[str]) -> BaseTool:
    def body(text: str) -> str:
        calls.append(text)
        return f"{name}:{text}"

    body.__name__ = name
    body.__doc__ = f"Deterministic {name} tool."
    return tool(name)(body)


def make_replay_tool(name: str, results: dict[str, str]) -> BaseTool:
    def body(text: str) -> str:
        return str(results.get(text, f"replayed:{name}"))

    body.__name__ = name
    body.__doc__ = f"Replay tool {name} returning recorded results."
    return tool(name)(body)
