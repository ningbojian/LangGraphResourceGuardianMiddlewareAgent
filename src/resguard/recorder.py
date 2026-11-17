"""Online session recorder helper.

The guard tracks counters and the ledger; the recorder turns one finished run
into a ``SessionRecord`` and appends it to a local store. It is deliberately a
helper rather than an extra middleware so it can be used with any runner.
"""

from __future__ import annotations

import time
from typing import Mapping, Protocol

from resguard.ledger import ResourceSnapshot
from resguard.metrics import SessionMetrics
from resguard.resources import ResourceKind
from resguard.session import SessionRecord, SessionStore, now_iso


class GuardCounters(Protocol):
    """Minimal surface shared by ResourceGuardMiddleware and CustomGraphGuard."""

    run_id: str
    tool_calls: int
    cache_hits: int
    peak_concurrency: int
    concurrency_denials: int
    llm_usd: float
    tool_usd: float
    last_binding: ResourceKind | None

    def snapshot(self) -> ResourceSnapshot: ...


class SessionRecorder:
    def __init__(
        self,
        *,
        guard: GuardCounters,
        store: SessionStore,
        session_id: str,
        query_type: str = "unknown",
        agent_version: str = "0",
        model_version: str = "unknown",
        prompt_version: str = "0",
        tool_versions: Mapping[str, str] | None = None,
        workflow_version: str = "0",
        policy_version: str = "0",
        started_at: str | None = None,
        trace: list[dict] | None = None,
        messages: list[dict] | None = None,
    ) -> None:
        self.guard = guard
        self.store = store
        self.session_id = session_id
        self.query_type = query_type
        self.agent_version = agent_version
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.tool_versions = dict(tool_versions or {})
        self.workflow_version = workflow_version
        self.policy_version = policy_version
        self.started_at = started_at or now_iso()
        self.trace = list(trace or [])
        self.messages = list(messages or [])
        self._timer = time.perf_counter()

    def finalize(
        self,
        *,
        success: bool | None = None,
        termination_reason: str = "completed",
        label_source: str | None = None,
        finished_at: str | None = None,
    ) -> SessionRecord:
        snapshot = self.guard.snapshot()
        binding = self.guard.last_binding or snapshot.binding()
        metrics = SessionMetrics(
            tool_calls=self.guard.tool_calls,
            cache_hits=self.guard.cache_hits,
            db_queries=int(snapshot.used.get(ResourceKind.DB_QUERIES)),
            db_cu=snapshot.used.get(ResourceKind.DB_CU),
            api_quota=snapshot.used.get(ResourceKind.API_QUOTA),
            peak_concurrency=self.guard.peak_concurrency,
            concurrency_denials=self.guard.concurrency_denials,
            llm_usd=self.guard.llm_usd,
            tool_usd=self.guard.tool_usd,
            wall_clock_ms=(time.perf_counter() - self._timer) * 1000.0,
        )
        record = SessionRecord(
            session_id=self.session_id,
            started_at=self.started_at,
            finished_at=finished_at or now_iso(),
            query_type=self.query_type,
            run_id=self.guard.run_id,
            agent_version=self.agent_version,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            tool_versions=dict(self.tool_versions),
            workflow_version=self.workflow_version,
            policy_version=self.policy_version,
            metrics=metrics,
            success=success,
            label_source=label_source
            or ("human" if success is not None else "none"),
            termination_reason=termination_reason,
            binding_limit=binding.value if binding else None,
            replay_safe=True,
            trace=list(self.trace),
            messages=list(self.messages),
        )
        self.store.append(record)
        return record
