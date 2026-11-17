"""resguard — developer-facing multi-resource guard and batch optimizer."""

from resguard.custom import CustomGraphGuard, CustomGuardState
from resguard.evaluator import ScenarioEvaluator, ScenarioSpec
from resguard.integrations import CheckpointerSessionStore, LangfuseSessionStore
from resguard.judge import DeterministicQualityJudge, LLMQualityJudge
from resguard.middleware import ResourceGuardMiddleware
from resguard.recorder import SessionRecorder
from resguard.replay import (
    MessageReplayEvaluator,
    ReplayChatModel,
    ReplayEvaluator,
)
from resguard.rollout import CanaryPlan, DriftReport, detect_drift
from resguard.workflow import WorkflowSpec, apply_workflow_changes

__version__ = "0.1.0"

__all__ = [
    "CanaryPlan",
    "CheckpointerSessionStore",
    "CustomGraphGuard",
    "CustomGuardState",
    "DeterministicQualityJudge",
    "DriftReport",
    "LLMQualityJudge",
    "LangfuseSessionStore",
    "MessageReplayEvaluator",
    "ReplayChatModel",
    "ReplayEvaluator",
    "ResourceGuardMiddleware",
    "ScenarioEvaluator",
    "ScenarioSpec",
    "SessionRecorder",
    "WorkflowSpec",
    "apply_workflow_changes",
    "detect_drift",
    "__version__",
]
