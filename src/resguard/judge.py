"""Answer-quality judges for the offline evaluator.

The deterministic marker judge is the default; the LLM judge is optional and
runs offline with a developer-provided model. Both return a score in [0, 1].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class QualityJudge(Protocol):
    def score(self, answer: str) -> float: ...


@dataclass
class DeterministicQualityJudge:
    expected_marker: str

    def score(self, answer: str) -> float:
        return 1.0 if self.expected_marker.lower() in answer.lower() else 0.0


class LLMQualityJudge:
    """LLM-as-judge with a deterministic fallback.

    The judge only runs offline. Production callers should pass a cheap model
    and keep the answer text redacted/minimized.
    """

    def __init__(self, model: Any, *, fallback: QualityJudge | None = None) -> None:
        self.model = model
        self.fallback = fallback

    def score(self, answer: str) -> float:
        prompt = (
            "You are evaluating an agent answer. Score whether the answer "
            "completed the task, from 0.0 to 1.0. Return only JSON like "
            '{"score": 0.8}.\n\nAnswer:\n'
            f"{answer[:4000]}"
        )
        try:
            content = str(self.model.invoke(prompt).content)
            return self._parse(content)
        except Exception:
            return self.fallback.score(answer) if self.fallback is not None else 0.0

    @staticmethod
    def _parse(text: str) -> float:
        try:
            data = json.loads(text)
            value = data.get("score", data.get("quality")) if isinstance(data, dict) else data
            return max(0.0, min(1.0, float(value)))
        except Exception:
            match = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
            if match is None:
                raise ValueError("no score found")
            return max(0.0, min(1.0, float(match.group())))
