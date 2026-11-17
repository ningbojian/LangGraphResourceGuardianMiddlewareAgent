from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from resguard.batch import BatchAnalyzer, BatchInsights
from resguard.optimizer import AuditLog, ParetoGate
from resguard.optimizer_agent.brain import OptimizerBrain
from resguard.optimizer_agent.evaluator import SyntheticEvaluator
from resguard.optimizer_agent.state import OptimizerState
from resguard.proposer import PolicyProposal
from resguard.session import SessionStore


class OptimizerAgent:
    """Offline LangGraph agent:

    observe → diagnose → hypothesize → plan_experiment → execute_experiment
    → evaluate → reflect → (loop or finalize) → END

    The graph is the product: it has its own state, conditional edges, a loop,
    a checkpointer, and an optional human-approval interrupt. It runs offline;
    the online guard never calls an LLM.
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        brain: OptimizerBrain,
        evaluator: SyntheticEvaluator | Any | None = None,
        gate: ParetoGate | None = None,
        audit_log: AuditLog | None = None,
        require_approval: bool = False,
        since: str | None = None,
        limit: int | None = None,
        sample: float = 1.0,
    ) -> None:
        self.store = store
        self.brain = brain
        self.evaluator = evaluator or SyntheticEvaluator()
        self.gate = gate or ParetoGate()
        self.audit_log = audit_log
        self.require_approval = require_approval
        self.since = since
        self.limit = limit
        self.sample = sample
        self.graph = self._build()

    def _build(self) -> Any:
        builder = StateGraph(OptimizerState)
        builder.add_node("observe", self._observe)
        builder.add_node("diagnose", self._diagnose)
        builder.add_node("hypothesize", self._hypothesize)
        builder.add_node("plan_experiment", self._plan_experiment)
        builder.add_node("execute_experiment", self._execute_experiment)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("reflect", self._reflect)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "observe")
        builder.add_edge("observe", "diagnose")
        builder.add_edge("diagnose", "hypothesize")
        builder.add_edge("hypothesize", "plan_experiment")
        builder.add_edge("plan_experiment", "execute_experiment")
        builder.add_edge("execute_experiment", "evaluate")
        builder.add_edge("evaluate", "reflect")
        builder.add_conditional_edges(
            "reflect",
            self._route_reflect,
            {"continue": "plan_experiment", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------ nodes
    def _observe(self, state: OptimizerState) -> dict[str, Any]:
        records = list(
            self.store.iter_sessions(since=self.since, limit=self.limit, sample=self.sample)
        )
        insights = BatchAnalyzer().analyze(records)
        return {"insights": insights.to_dict(), "trace": ["observe"]}

    def _diagnose(self, state: OptimizerState) -> dict[str, Any]:
        insights = BatchInsights.from_dict(state.get("insights", {}))
        focus = self.brain.diagnose(insights)
        return {"focus": focus, "trace": ["diagnose"]}

    def _hypothesize(self, state: OptimizerState) -> dict[str, Any]:
        insights = BatchInsights.from_dict(state.get("insights", {}))
        proposals = self.brain.propose(insights, state.get("focus", "general"))
        serialized = [
            proposal.to_dict() if isinstance(proposal, PolicyProposal) else dict(proposal)
            for proposal in proposals
        ]
        return {"proposals": serialized, "trace": ["hypothesize"]}

    def _plan_experiment(self, state: OptimizerState) -> dict[str, Any]:
        proposals = state.get("proposals", [])
        index = state.get("current_index", 0)
        if index >= len(proposals):
            return {
                "should_continue": False,
                "termination_reason": "no_proposals" if not proposals else "no_more_proposals",
                "trace": ["plan_experiment"],
            }
        return {
            "current_index": index + 1,
            "current_proposal": proposals[index],
            "should_continue": True,
            "trace": ["plan_experiment"],
        }

    def _execute_experiment(self, state: OptimizerState) -> dict[str, Any]:
        proposal_data = state.get("current_proposal")
        if not proposal_data:
            return {"trace": ["execute_experiment"]}
        proposal = PolicyProposal.from_dict(proposal_data)
        baseline = BatchInsights.from_dict(state.get("insights", {}))
        candidate = self.evaluator.evaluate(proposal, baseline)
        return {
            "candidate_insights": candidate.to_dict(),
            "trace": ["execute_experiment"],
        }

    def _evaluate(self, state: OptimizerState) -> dict[str, Any]:
        proposal_data = state.get("current_proposal")
        candidate_data = state.get("candidate_insights")
        if not proposal_data or not candidate_data:
            return {"trace": ["evaluate"]}
        proposal = PolicyProposal.from_dict(proposal_data)
        baseline = BatchInsights.from_dict(state.get("insights", {}))
        evaluator_baseline = getattr(self.evaluator, "baseline_insights", None)
        if isinstance(evaluator_baseline, BatchInsights):
            baseline = evaluator_baseline
        candidate = BatchInsights.from_dict(candidate_data)
        result = self.gate.evaluate(proposal.id, baseline=baseline, candidate=candidate)
        updates: dict[str, Any] = {
            "evaluations": [result.to_dict()],
            "trace": ["evaluate"],
        }
        if result.accepted:
            updates["accepted_proposal_id"] = proposal.id
        if self.audit_log is not None:
            self.audit_log.append(
                {
                    "event": "evaluate",
                    "proposal_id": proposal.id,
                    **result.to_dict(),
                }
            )
        return updates

    def _reflect(self, state: OptimizerState) -> dict[str, Any]:
        if state.get("accepted_proposal_id"):
            return {
                "should_continue": False,
                "termination_reason": "accepted",
                "trace": ["reflect"],
            }
        if state.get("termination_reason") == "no_proposals":
            return {
                "should_continue": False,
                "termination_reason": "no_proposals",
                "trace": ["reflect"],
            }
        index = state.get("current_index", 0)
        total = len(state.get("proposals", []))
        if index < total:
            return {"should_continue": True, "trace": ["reflect"]}
        return {
            "should_continue": False,
            "termination_reason": "no_accepted_proposal",
            "trace": ["reflect"],
        }

    def _finalize(self, state: OptimizerState) -> dict[str, Any]:
        accepted = state.get("accepted_proposal_id")
        if not accepted:
            return {
                "termination_reason": state.get("termination_reason") or "no_accepted_proposal",
                "trace": ["finalize"],
            }
        if self.require_approval:
            decision = interrupt(
                {
                    "proposal_id": accepted,
                    "evaluations": state.get("evaluations", []),
                }
            )
            if decision == "approve":
                return {
                    "approved_proposal_id": accepted,
                    "termination_reason": "approved",
                    "trace": ["finalize"],
                }
            return {"termination_reason": "rejected", "trace": ["finalize"]}
        return {
            "approved_proposal_id": accepted,
            "termination_reason": "accepted",
            "trace": ["finalize"],
        }

    @staticmethod
    def _route_reflect(state: OptimizerState) -> str:
        return "continue" if state.get("should_continue") else "finalize"
