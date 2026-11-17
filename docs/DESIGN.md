# Design

## Problem

LangGraph agents are bounded by `recursion_limit` and can use call-count
middleware, but those mechanisms are **cost-blind**:

- A 10-call loop of a cheap tool is fine; one extra PDF-parse call can blow a
  small USD budget.
- LLM gateways (LiteLLM/Portkey) see tokens, not your vector DB CU, MCP tools,
  document-parsing quota, DB connections, or concurrency limits.
- Observability platforms (Langfuse/LangSmith) show traces, but they do not
  enforce a per-run resource policy.

The mixed-cost mechanics benchmark makes the gap concrete: an official
call-count limit of 10 executes all 10 calls and spends **$0.408** against a
**$0.25** budget; the USD-aware guard stops at 9 calls / **$0.208**.

## Two halves

### Online guard (no per-query LLM)

`ResourceGuardMiddleware` is an `AgentMiddleware` for `create_agent`. It:

- reserves/commits multi-dimensional resources before/after model and tool
  calls;
- acquires/releases concurrency leases (gauge);
- serves idempotent tool results from a tenant/user/doc-version-scoped TTL cache;
- injects an **ephemeral** multi-dimensional reminder at the model-request tail
  (`request.override(messages=...)`), never in the system prompt and never
  persisted;
- returns a `ToolMessage` denial or a graceful stop when a resource is exhausted;
- exposes counters for `SessionRecorder`.

`SessionRecorder.finalize()` turns the real counters into a `SessionRecord` and
appends it to a JSONL/SQLite store. That is the online → offline handoff.

For a hand-built `StateGraph`, `CustomGraphGuard` provides the same semantics:
`wrap_model_node` wraps the model node (budget check, ephemeral reminder,
model-cost reserve/commit), `guard_node` executes tool calls with
reserve/commit/cache/concurrency, and `route_after_model` / `route_after_guard`
are the conditional-edge helpers.

### Offline optimizer agent (experimental)

`OptimizerAgent` is a real LangGraph `StateGraph`:

```
observe → diagnose → hypothesize → plan_experiment → execute_experiment
   ↑                                                    ↓
   └──────────── reflect ← evaluate ←───────────────────┘
                         ↓
                      finalize → human approval (interrupt)
```

- `observe` loads a session batch and runs `BatchAnalyzer`.
- `diagnose` / `hypothesize` are LLM decision points (`LLMBrain`) with a
  deterministic `RuleBrain` fallback.
- `execute_experiment` uses an evaluator; `SyntheticEvaluator` is a stand-in
  for replay/batch-run.
- `evaluate` applies a deterministic Pareto gate: success must not drop, at
  least one resource dimension must improve by a threshold, and others must not
  regress.
- `reflect` loops to the next proposal or finalizes.
- `finalize` can `interrupt` for developer approval; `--apply` writes a
  versioned policy candidate.

The optimizer runs offline, in a separate process, on completed sessions. It
does not run on the user-query hot path and does not change parameters after
every interaction.

## Resource model

Enforceable cumulative resources: USD, LLM input/output tokens, tool calls, DB
queries, DB CU, API quota, wall clock, checkpoint bytes. `CONCURRENCY` is a
gauge handled by leases. `SessionMetrics` also records observed metrics: prompt
/ cached / tool-schema / history tokens, duplicate calls, super-steps,
checkpoint writes/bytes, rehydration tokens, queue wait, false-positive stops,
subagent/trace overhead, and guard overhead.

The reported evidence distinguishes **declared**, **estimated**, and **actual**
values. Tool costs are declared by the developer; real DB/API adapters can
report actual usage.

## Evaluation

Primary endpoint: held-out batch `resource_per_success` per dimension.
Constraints: success rate not lower by more than 2 percentage points, no
subgroup regression, false-positive stop rate acceptable. Per-dimension results
include mean ± std, per-session deltas, and paired bootstrap 95% CIs.

Validity threats:

- Replay is a counterfactual approximation. `ReplayEvaluator` replays tool calls
  with recorded results; `MessageReplayEvaluator` replays the recorded
  model-message sequence with `ReplayChatModel` and `QueuedReplayTool`. These
  are not a production canary.
- Fake models/mock tools can overfit; a real-model smoke test and held-out
  batches are required.
- Tool/DB/API resource values are declared or estimated; label them.
- Small synthetic samples; conclusions are benchmark-bound, not production
  guarantees.
- LLM proposers and `LLMQualityJudge` are stochastic; use temperature 0 where
  possible and report variance.

## Non-goals

- No per-query LLM planner.
- No full graph synthesis/reordering; workflow changes are limited and
  reference-template based.
- No custom dashboard/log DB/eval framework; reuse Langfuse/OTel, LangGraph
  cache/checkpointers, and existing eval tools.
- No prompt optimization; that is Opik/DSPy/LangSmith territory.

## Status

Stable-ish: resources, costs, metrics, ledger, policy, reminder, cache,
middleware, recorder, session stores, batch analyzer, proposer, Pareto gate,
CLI, mechanics benchmark.

Experimental: `OptimizerAgent`, `LLMBrain`, `SyntheticEvaluator`,
`MessageReplayEvaluator`, `LLMQualityJudge`, `WorkflowSpec` execution,
`CanaryPlan`, `detect_drift`. APIs may change.

See `README.md` for quickstart and `ROADMAP.md` for the demand-validation plan.
