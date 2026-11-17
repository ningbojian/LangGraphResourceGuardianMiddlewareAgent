"""Limited workflow-change execution for a reference template.

Full graph synthesis is out of scope. The supported changes are config-level
and map to a reference workflow spec; the evaluator uses the same changes via
``PolicyConfig.with_proposal``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import Any, Mapping

from resguard.evaluator import PolicyConfig
from resguard.proposer import PolicyProposal, WorkflowChange


@dataclass(frozen=True)
class WorkflowSpec:
    name: str = "reference"
    cache_enabled: bool = False
    model_tier: str = "default"
    max_concurrency: int | None = None
    excluded_tools: tuple[str, ...] = ()
    delta_channel: bool = False
    context_editing: bool = False
    offload_tool_outputs: bool = False
    durability: str = "async"
    guard_node: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowSpec":
        return cls(
            name=data.get("name", "reference"),
            cache_enabled=bool(data.get("cache_enabled", False)),
            model_tier=data.get("model_tier", "default"),
            max_concurrency=data.get("max_concurrency"),
            excluded_tools=tuple(data.get("excluded_tools") or ()),
            delta_channel=bool(data.get("delta_channel", False)),
            context_editing=bool(data.get("context_editing", False)),
            offload_tool_outputs=bool(data.get("offload_tool_outputs", False)),
            durability=data.get("durability", "async"),
            guard_node=bool(data.get("guard_node", False)),
        )


def apply_workflow_change(spec: WorkflowSpec, change: WorkflowChange) -> WorkflowSpec:
    if change.kind == "add_cache_node":
        return replace(spec, cache_enabled=True)
    if change.kind == "model_routing":
        return replace(spec, model_tier=str(change.config.get("model_tier", "cheap")))
    if change.kind == "set_max_concurrency":
        value = change.config.get("max_concurrency")
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("set_max_concurrency requires a positive max_concurrency")
        return replace(spec, max_concurrency=int(value))
    if change.kind == "add_guard_node":
        return replace(spec, guard_node=True)
    if change.kind == "exclude_tools":
        tools = [change.target, *(change.config.get("tools") or [])]
        excluded = tuple(dict.fromkeys([*spec.excluded_tools, *(t for t in tools if t)]))
        return replace(spec, excluded_tools=excluded)
    if change.kind == "enable_delta_channel":
        return replace(spec, delta_channel=True)
    if change.kind == "switch_context_editing":
        return replace(spec, context_editing=True)
    if change.kind == "offload_tool_outputs":
        return replace(spec, offload_tool_outputs=True)
    if change.kind == "set_durability":
        return replace(spec, durability=str(change.config.get("mode", "async")))
    raise ValueError(f"unsupported workflow change: {change.kind}")


def apply_workflow_changes(
    spec: WorkflowSpec, proposal: PolicyProposal
) -> WorkflowSpec:
    updated = spec
    for change in proposal.workflow_changes:
        updated = apply_workflow_change(updated, change)
    return updated


def workflow_spec_to_policy_config(
    spec: WorkflowSpec, base: PolicyConfig | None = None
) -> PolicyConfig:
    base = base or PolicyConfig()
    factor = base.model_cost_factor
    if spec.model_tier == "cheap":
        factor = min(factor, Decimal("0.7"))
    return PolicyConfig(
        cache_enabled=base.cache_enabled or spec.cache_enabled,
        cache_ttl_seconds=base.cache_ttl_seconds,
        model_cost_factor=factor,
        max_concurrency=spec.max_concurrency or base.max_concurrency,
        budget_usd=base.budget_usd,
        api_quota_limit=base.api_quota_limit,
        db_cu_limit=base.db_cu_limit,
    )
