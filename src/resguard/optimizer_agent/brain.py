from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from resguard.batch import BatchInsights
from resguard.proposer import PolicyProposal, RuleProposer


class OptimizerBrain(Protocol):
    def diagnose(self, insights: BatchInsights) -> str: ...
    def propose(self, insights: BatchInsights, focus: str) -> list[PolicyProposal]: ...


@dataclass
class ScriptedBrain:
    focus: str = "general"
    proposals: list[PolicyProposal] = field(default_factory=list)

    def diagnose(self, insights: BatchInsights) -> str:
        return self.focus

    def propose(self, insights: BatchInsights, focus: str) -> list[PolicyProposal]:
        return list(self.proposals)


class RuleBrain:
    """Deterministic fallback brain; no LLM call."""

    def diagnose(self, insights: BatchInsights) -> str:
        return insights.hotspots[0] if insights.hotspots else "general"

    def propose(self, insights: BatchInsights, focus: str) -> list[PolicyProposal]:
        return RuleProposer().propose(insights)


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = min(
        [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0],
        default=-1,
    )
    if start < 0:
        raise ValueError("no JSON found")
    return json.loads(cleaned[start:])


class LLMBrain:
    """Thin LLM adapter with deterministic fallback.

    The LLM only runs offline inside the optimizer agent. Tests inject a fake
    chat model; production can pass ``init_chat_model(...)``.
    """

    def __init__(
        self,
        model: Any,
        *,
        fallback: OptimizerBrain | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.model = model
        self.fallback = fallback or RuleBrain()
        self.raise_on_error = raise_on_error

    def diagnose(self, insights: BatchInsights) -> str:
        prompt = (
            "You are an agent resource optimizer. Given this batch summary, "
            "choose ONE focus hotspot (for example db_cu_exhausted, usd_exhausted, "
            "duplicate_tool_calls, concurrency). Return only the hotspot name.\n"
            f"summary={json.dumps(insights.to_dict(), sort_keys=True)}"
        )
        try:
            content = str(self.model.invoke(prompt).content).strip()
            return content.splitlines()[0][:200] or self.fallback.diagnose(insights)
        except Exception:
            if self.raise_on_error:
                raise
            return self.fallback.diagnose(insights)

    def propose(self, insights: BatchInsights, focus: str) -> list[PolicyProposal]:
        prompt = (
            "You are an agent resource optimizer. Propose 1-3 resource policy "
            "changes as a JSON array. Each item must have: name, description, "
            "parameter_changes (path/old/new/rationale), workflow_changes "
            "(kind/target/config/rationale), expected_effect, evidence, risk. "
            f"focus={focus}; summary={json.dumps(insights.to_dict(), sort_keys=True)}"
        )
        try:
            payload = _extract_json(str(self.model.invoke(prompt).content))
            if not isinstance(payload, list):
                raise ValueError("expected a list")
            return [PolicyProposal.from_dict(item) for item in payload]
        except Exception:
            if self.raise_on_error:
                raise
            return self.fallback.propose(insights, focus)
