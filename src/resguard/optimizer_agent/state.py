from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class OptimizerState(TypedDict, total=False):
    batch_id: str
    insights: dict[str, Any]
    focus: str
    proposals: Annotated[list[dict[str, Any]], operator.add]
    current_index: int
    current_proposal: dict[str, Any]
    candidate_insights: dict[str, Any]
    evaluations: Annotated[list[dict[str, Any]], operator.add]
    accepted_proposal_id: str | None
    approved_proposal_id: str | None
    should_continue: bool
    termination_reason: str | None
    trace: Annotated[list[str], operator.add]
