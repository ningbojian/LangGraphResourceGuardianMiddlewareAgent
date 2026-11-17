from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import ConfigDict, Field


def usage(input_tokens: int = 1_000, output_tokens: int = 500) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def ai_tool_call(
    *,
    name: str = "echo",
    args: dict[str, Any] | None = None,
    call_id: str = "call_1",
    input_tokens: int = 1_000,
    output_tokens: int = 500,
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
        usage_metadata=usage(input_tokens, output_tokens),
    )


def ai_final(
    content: str = "done",
    *,
    input_tokens: int = 1_000,
    output_tokens: int = 500,
) -> AIMessage:
    return AIMessage(content=content, usage_metadata=usage(input_tokens, output_tokens))


def make_echo_tool():
    calls: list[str] = []

    @tool
    def echo(text: str) -> str:
        """Echo text back."""
        calls.append(text)
        return f"echo:{text}"

    return echo, calls


class ScriptedChatModel(BaseChatModel):
    """Deterministic fake chat model for middleware integration tests."""

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
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = ai_final("done")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self
