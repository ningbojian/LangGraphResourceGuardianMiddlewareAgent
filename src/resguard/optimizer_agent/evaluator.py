from __future__ import annotations

from resguard.batch import BatchInsights
from resguard.proposer import PolicyProposal


class SyntheticEvaluator:
    """Deterministic stand-in for replay/batch-run.

    This is a *simulation*, not evidence. The real evaluator should replay
    recorded sessions or run a held-out batch. It is used for graph mechanics
    and CI so the optimizer can run without API keys.
    """

    def evaluate(self, proposal: PolicyProposal, baseline: BatchInsights) -> BatchInsights:
        factor = 1.0
        for change in proposal.workflow_changes:
            if change.kind == "add_cache_node":
                factor *= 0.8
            elif change.kind == "model_routing":
                factor *= 0.7
            elif change.kind == "set_max_concurrency":
                factor *= 0.9
        for change in proposal.parameter_changes:
            if "cache_ttl" in change.path:
                factor *= 0.9
        candidate = BatchInsights.from_dict(baseline.to_dict())
        candidate.resource_per_success = {
            dimension: value * factor
            for dimension, value in baseline.resource_per_success.items()
        }
        return candidate
