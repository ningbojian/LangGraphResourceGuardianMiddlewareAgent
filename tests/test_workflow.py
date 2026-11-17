from __future__ import annotations

import pytest

from resguard.proposer import PolicyProposal, WorkflowChange
from resguard.workflow import (
    WorkflowSpec,
    apply_workflow_change,
    apply_workflow_changes,
    workflow_spec_to_policy_config,
)


def test_workflow_spec_round_trip() -> None:
    spec = WorkflowSpec(
        cache_enabled=True,
        model_tier="cheap",
        max_concurrency=2,
        excluded_tools=("write_file",),
        delta_channel=True,
        context_editing=True,
        offload_tool_outputs=True,
        durability="sync",
        guard_node=True,
    )

    assert WorkflowSpec.from_dict(spec.to_dict()) == spec


def test_apply_cache_and_concurrency_changes() -> None:
    spec = WorkflowSpec()
    spec = apply_workflow_change(
        spec, WorkflowChange("add_cache_node", "search_docs", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("set_max_concurrency", "db", {"max_concurrency": 2}, "")
    )

    assert spec.cache_enabled is True
    assert spec.max_concurrency == 2


def test_apply_model_routing_and_structural_flags() -> None:
    spec = WorkflowSpec()
    spec = apply_workflow_change(
        spec, WorkflowChange("model_routing", "default", {"model_tier": "cheap"}, "")
    )
    spec = apply_workflow_change(spec, WorkflowChange("add_guard_node", "x", {}, ""))
    spec = apply_workflow_change(
        spec, WorkflowChange("enable_delta_channel", "x", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("switch_context_editing", "x", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("offload_tool_outputs", "x", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("set_durability", "x", {"mode": "sync"}, "")
    )

    assert spec.model_tier == "cheap"
    assert spec.guard_node is True
    assert spec.delta_channel is True
    assert spec.context_editing is True
    assert spec.offload_tool_outputs is True
    assert spec.durability == "sync"


def test_exclude_tools_deduplicates() -> None:
    spec = apply_workflow_change(
        WorkflowSpec(), WorkflowChange("exclude_tools", "write_file", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("exclude_tools", "write_file", {}, "")
    )
    spec = apply_workflow_change(
        spec, WorkflowChange("exclude_tools", "delete_db", {}, "")
    )

    assert spec.excluded_tools == ("write_file", "delete_db")


def test_unknown_change_raises() -> None:
    with pytest.raises(ValueError):
        apply_workflow_change(WorkflowSpec(), WorkflowChange("unknown", "x", {}, ""))


def test_apply_workflow_changes_from_proposal() -> None:
    proposal = PolicyProposal(
        name="combo",
        description="",
        workflow_changes=[
            WorkflowChange("add_cache_node", "search_docs", {}, ""),
            WorkflowChange("model_routing", "default", {"model_tier": "cheap"}, ""),
        ],
    )

    spec = apply_workflow_changes(WorkflowSpec(), proposal)

    assert spec.cache_enabled is True
    assert spec.model_tier == "cheap"


def test_workflow_spec_to_policy_config() -> None:
    spec = WorkflowSpec(cache_enabled=True, model_tier="cheap", max_concurrency=2)

    config = workflow_spec_to_policy_config(spec)

    assert config.cache_enabled is True
    assert config.max_concurrency == 2
    assert float(config.model_cost_factor) == pytest.approx(0.7)
