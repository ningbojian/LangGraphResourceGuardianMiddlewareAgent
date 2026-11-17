# resguard

[English](README.md) | [简体中文](README.zh-CN.md)

**Developer-facing multi-resource guard and batch optimizer for LangGraph agents.**

## Why this exists

LangGraph's `recursion_limit` and call-count middleware are necessary but
**cost-blind**. A 10-call loop of a cheap tool is fine; one extra PDF-parse call
can blow a small USD budget. LLM gateways see tokens, not your vector DB CU, MCP
tools, document-parsing quota, DB connections, or concurrency limits.
Observability platforms show traces but do not enforce a per-run resource
policy.

The mechanics benchmark makes the gap concrete: an official call-count limit of
10 executes all 10 calls and spends **$0.408** against a **$0.25** budget; the
USD-aware guard stops at 9 calls / **$0.208**.

resguard makes those resources first-class: a lightweight online guard plus an
offline batch optimizer that proposes policy changes, evaluated by a Pareto gate
and reviewed by a developer.

## Who it's for / not for

**For:** LangGraph developers whose agents call expensive or private tools
(vector DB, document parsing, serverless DB, MCP), especially teams that already
have session logs and want a per-run guard plus an offline report.

**Not for:** teams that only need LLM token budgets at the gateway (use
LiteLLM/Portkey); teams without session logs or success labels (the optimizer
cannot prove improvements); teams that want a full observability platform (use
Langfuse/LangSmith).

## Stable vs experimental

- **Stable-ish:** resource model, ledger, policy/reminder, cache, middleware,
  `SessionRecorder`, session stores, batch analyzer, proposer, Pareto gate, CLI,
  mechanics benchmark. Covered by tests.
- **Experimental:** `OptimizerAgent`, `LLMBrain`, `SyntheticEvaluator`,
  `MessageReplayEvaluator`, `LLMQualityJudge`, `WorkflowSpec` execution,
  `CanaryPlan`, `detect_drift`. APIs may change; the synthetic evaluator is not
  evidence.

`resguard` has two halves:

- **Online (lightweight):** `ResourceGuardMiddleware` enforces an approved
  static resource policy and exposes counters. It never calls an LLM, so user
  queries are not slowed down by the optimizer. `SessionRecorder` turns the
  counters into a `SessionRecord`.
- **Offline (experimental agentic optimizer):** `OptimizerAgent` is a LangGraph
  `StateGraph` that reads a batch of completed sessions, diagnoses hotspots,
  proposes resource-policy changes, evaluates them with a Pareto gate, and
  pauses for human approval before anything is versioned.

> The adoption wedge is the guard + benchmark. The offline optimizer is the
> research/preview half that makes the project more than another middleware.
> This keeps the "agent" feeling (state transitions, conditional edges, a loop,
> tools, checkpointer, HITL, streaming) without adding per-query LLM cost.

## Why this is an agent, not a cron script

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and
[ROADMAP.md](ROADMAP.md) for the demand-validation plan and kill criteria.

```
observe → diagnose → hypothesize → plan_experiment → execute_experiment
   ↑                                                    ↓
   └──────────── reflect ← evaluate ←───────────────────┘
                         ↓
                      finalize → human approval (interrupt) → versioned policy
```

- It is a real LangGraph `StateGraph` with conditional edges and a loop.
- The LLM can decide at `diagnose` (which hotspot), `hypothesize` (which
  proposal), and `reflect` (continue or finalize). `RuleBrain` is the
  deterministic fallback; `LLMBrain` is the pluggable LLM adapter.
- It uses a checkpointer and `interrupt` for developer approval/resume.
- `stream_mode="updates"` exposes the live state transitions to the developer.
- It runs **offline**, on completed sessions, in a separate process. It does
  not run on the user-query hot path, and it does not change parameters after
  every interaction.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m demo.run
uv run python -m demo.run --interactive   # see the HITL interrupt/resume path
uv run python -m demo.llm_brain --fake --stream
uv run python -m benchmark.run --out reports/

# real LLM brain (requires a valid provider key; this environment's
# ANTHROPIC_API_KEY returned 401, so the verified run uses --fake)
uv sync --extra dev --extra llm
export ANTHROPIC_API_KEY=...
uv run python -m demo.llm_brain --model anthropic:claude-haiku-5-5 --stream

# real Anthropic (requires a valid ANTHROPIC_API_KEY; --direct bypasses the
# local ANTHROPIC_BASE_URL proxy, which in this environment serves DeepSeek)
uv run python -m demo.real_agent --direct \
  --model anthropic:claude-haiku-5-5 \
  --data examples/real_agent/data \
  --out examples/real_agent/sessions_real.jsonl

# real model available in this environment (DeepSeek via the Anthropic-compatible proxy)
uv run python -m demo.real_agent --model deepseek-flash \
  --data examples/real_agent/data \
  --out examples/real_agent/sessions_real.jsonl
uv run resguard evaluate --sessions examples/real_agent/sessions_real.jsonl \
  --tool-profiles examples/real_agent/tool_profiles.json \
  --baseline examples/real_agent/policy_baseline.json \
  --candidate examples/real_agent/policy_candidate.json \
  --out reports/real-agent/ \
  --sessions-per-month 10000 --optimizer-cost-usd 5

# scenario-level CLI
uv run resguard batch-run --dataset examples/eval/scenarios.jsonl \
  --policy examples/eval/policy_baseline.json --out runs/
uv run resguard evaluate --baseline examples/eval/policy_baseline.json \
  --candidate examples/eval/policy_candidate.json \
  --dataset examples/eval/scenarios.jsonl --out reports/eval/

# visualize before/after: self-contained HTML with inline SVG, no dashboard
uv run resguard report --input reports/eval/evaluate.json \
  --out reports/eval/report.html \
  --sessions-per-month 10000 --optimizer-cost-usd 5

# evaluate your own recorded sessions with full message replay
uv run resguard evaluate --sessions ./sessions \
  --tool-profiles examples/eval/tool_profiles.json \
  --baseline examples/eval/policy_baseline.json \
  --candidate examples/eval/policy_candidate.json \
  --out reports/eval-sessions/
```

`uv run python -m demo.run` uses `ScenarioEvaluator` (real guard execution on
scripted scenarios) and prints baseline vs candidate `resource_per_success`,
false-positive rate, and candidate tool calls. `--interactive` shows the HITL
interrupt/resume path.

The benchmark is a **mechanics** benchmark (scripted model + mock tools). It
writes `reports/benchmark.json` and `reports/benchmark.md`. It is not a quality
or production benchmark.

- On uniform-cost loops, an official call-count limiter and the full guard tie;
  call counting is enough there.
- On `mixed_cost_spike` (8 cheap calls + 2 calls that are 200x more expensive),
  the official limit of 10 executes all 10 calls and spends **$0.408**, over the
  **$0.25** budget; the USD-aware full guard stops at 9 calls / **$0.208**,
  within budget. That is the cost-awareness differentiator. Numbers are
  generated by the harness; see `reports/benchmark.md`.
- Do not read the current table as a production quality claim.

CLI on a JSONL session batch:

```bash
uv run resguard optimize \
  --sessions ./sessions \
  --since 7d \
  --out ./proposals/latest \
  --stream \
  --dry-run

# after reviewing report.md / raw.json / audit.jsonl:
uv run resguard optimize --sessions ./sessions --out ./proposals/latest --interactive --apply
```

`--apply` refuses to run without `--interactive` or the CI-only
`--auto-approve`; the default is a dry-run report.

## Online SDK

```python
from decimal import Decimal

from langchain.agents import create_agent
from resguard.cache import CacheContext, ToolCache
from resguard.costs import ModelPricing, ModelPricingTable
from resguard.ledger import InMemoryLedger
from resguard.middleware import ResourceGuardMiddleware
from resguard.policy import DegradationAction, GuardPolicy
from resguard.resources import ResourceKind, ResourceLimits, ToolResourceProfile

guard = ResourceGuardMiddleware(
    ledger=InMemoryLedger(ResourceLimits({ResourceKind.USD: 1.0})),
    policy=GuardPolicy(hard_limit_action=DegradationAction.STOP),
    tool_profiles={
        "search_docs": ToolResourceProfile(
            usd_per_call=0.002,
            db_queries_per_call=1,
            db_cu_per_call=0.1,
            cacheable=True,
            idempotent=True,
            ttl_seconds=300,
        )
    },
    pricing=ModelPricingTable({"your-model": ModelPricing(Decimal("1"), Decimal("2"))}),
    cache=ToolCache(),
    cache_context=CacheContext(user_id="alice"),
)

agent = create_agent(model=model, tools=[search_docs], middleware=[guard])

# After a run, persist the guard's real counters as a SessionRecord for the
# offline optimizer. This is the online → offline handoff.
from resguard.recorder import SessionRecorder
from resguard.session import JsonlSessionStore

recorder = SessionRecorder(
    guard=guard,
    store=JsonlSessionStore("sessions/sessions.jsonl"),
    session_id="run-1",
    query_type="refund",
)
agent.invoke({"messages": [HumanMessage("refund policy?")]})
recorder.finalize(success=True)
```

The guard enforces cumulative resources, injects ephemeral multi-dimensional
reminders at the model-call tail (never in the system prompt), returns
`ToolMessage` denials, and releases reservations on cache hits/errors.

### Custom StateGraph

`AgentMiddleware` only plugs into `create_agent`. For a hand-built graph,
`CustomGraphGuard` provides the same guard semantics:

```python
from langgraph.graph import END, START, StateGraph
from resguard.custom import CustomGraphGuard, CustomGuardState

guard = CustomGraphGuard(
    ledger=ledger,
    policy=policy,
    tools=[search_docs],
    tool_profiles={"search_docs": ToolResourceProfile(usd_per_call=0.002)},
    pricing=pricing,
)

builder = StateGraph(CustomGuardState)
builder.add_node("model", guard.wrap_model_node(model_node, model_name="your-model"))
builder.add_node("guard", guard.guard_node)
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model", guard.route_after_model, {"guard": "guard", "end": END}
)
builder.add_conditional_edges(
    "guard", guard.route_after_guard, {"model": "model", "end": END}
)
graph = builder.compile()
```

`guard_node` executes the tool calls itself (replacing a plain `ToolNode`) so
it can reserve/commit multi-resource costs, serve cache hits, and honor
concurrency. The wrapped model node checks the budget, injects the ephemeral
reminder into a copy of the state, and commits actual model usage.

### Integration support

| Integration | Recording | Guard | Workflow optimization |
|---|---|---|---|
| `create_agent` + middleware | `SessionRecorder` | `ResourceGuardMiddleware` | reference template only |
| custom `StateGraph` | `SessionRecorder` | `CustomGraphGuard` | patch suggestions only |
| LangGraph checkpointer | `CheckpointerSessionStore` (reads `get_state_history`; metrics derived; optional `label_fn`) | — | — |
| Langfuse | `LangfuseSessionStore` (reads trace metadata `resguard_session`; optional `create_trace`) | — | — |

Both adapters are read-only sources. `CheckpointerSessionStore` needs explicit
`thread_ids`; it derives tool calls/token usage/checkpoint metrics from state and
accepts an optional success `label_fn`. `LangfuseSessionStore` expects a
`fetch_traces` callable (wrap your Langfuse client); the SDK is not imported.

## Resource model

Observed and enforceable dimensions include:

| Dimension | Unit | Mode |
|---|---|---|
| `USD` | $ | cumulative |
| `LLM_INPUT_TOKENS` / `LLM_OUTPUT_TOKENS` | tokens | cumulative |
| `TOOL_CALLS` | calls | cumulative |
| `DB_QUERIES` | queries | cumulative |
| `DB_CU` | CU | cumulative |
| `API_QUOTA` | units | cumulative |
| `CONCURRENCY` | slots | gauge (lease) |
| `WALL_CLOCK_MS` | ms | cumulative |
| `CHECKPOINT_BYTES` | bytes | cumulative |

`SessionMetrics` also records token/context, checkpoint/persistence,
retry/duplicate, subagent/observability, and guard-overhead metrics. The
optimizer uses observed metrics for diagnosis; only enforceable resources are
hard-limited.

## Repository map

```
src/resguard/
  resources.py        multi-dimensional resource vectors
  costs.py            Decimal token pricing
  metrics.py          SessionMetrics + token/schema/checkpoint estimators
  ledger.py           reserve/commit/release + concurrency lease + composite scopes
  policy.py           thresholds, hard-limit actions, degradation choices
  reminder.py         multi-dimensional ephemeral reminders + dedup
  cache.py            per-tool TTL cache with tenant/user/doc-version keys
  middleware.py       ResourceGuardMiddleware (online actuator)
  custom.py           CustomGraphGuard (custom StateGraph adapter)
  recorder.py         SessionRecorder (online → offline handoff)
  session.py          SessionRecord, JSONL/SQLite stores, LabelStore
  batch.py            deterministic BatchAnalyzer
  evaluator.py        ScenarioEvaluator (real guard execution on scripted scenarios)
  report.py           self-contained before/after HTML report (inline SVG)
  replay.py           tool-call + full model-message replay (ReplayEvaluator/MessageReplayEvaluator)
  judge.py            deterministic marker + optional LLM-as-judge quality
  integrations/
    langfuse.py       LangfuseSessionStore (trace metadata source)
    checkpointer.py   CheckpointerSessionStore (LangGraph state-history source)
  workflow.py         WorkflowSpec + supported WorkflowChange execution
  rollout.py          CanaryPlan + drift detection
  testing.py          ScriptedChatModel + counting/replay tools (packaged test doubles)
  proposer.py         PolicyProposal, RuleProposer
  optimizer.py        Pareto gate + audit log
  optimizer_agent/    LangGraph StateGraph, brains, synthetic evaluator
  cli.py              optimize / batch-run / evaluate / drift / workflow
tests/                unit/integration tests
demo/run.py           no-API-key end-to-end demo
demo/llm_brain.py     real-provider LLM brain demo (+ --fake no-key smoke test)
demo/real_agent.py    real file-search + SQLite tools + SessionRecorder (--fake model)
examples/real_agent/  real-agent data/policies/tool profiles + generated sessions
examples/scheduling/  cron + GitHub Actions examples
docs/DESIGN.md        architecture + validity threats
ROADMAP.md            demand validation + kill criteria
LICENSE               Apache-2.0
```

## Quantified before/after report

`resguard evaluate` writes `evaluate.json` + `evaluate.html`. The HTML report is
self-contained (inline CSS + SVG, no dashboard or JS dependency) and shows:

- **Cost / success** (`resource_per_success["usd"]`) baseline → candidate, with
  reduction %.
- **Tool calls / DB queries / DB CU / API quota per successful task**.
- **Success rate / quality rate / false-positive stop rate** deltas in
  percentage points.
- **Projected monthly savings** = `(usd_before - usd_after) × sessions_per_month
  − optimizer_cost_usd`.
- The Pareto gate's accepted/rejected status, improvements, and reasons.

The report is generated from a benchmark/replay evaluation, not a production
canary; treat it as evidence about guard mechanics and policy trade-offs, not a
guaranteed production saving.

## Evaluation methodology (summary)

The optimizer is only considered useful if it improves a resource dimension
**without lowering success rate, raising false-positive stops, or lowering
answer quality**. The Pareto gate checks success, false-positive rate, quality,
per-dimension improvement, and per-dimension regression. The planned gate
reports, per dimension: used/limit, `resource_per_success`, mean ± std, paired
per-session deltas, and bootstrap 95% CIs. Research questions include tool/USD
reduction, DB queries/CU, concurrency violations, API quota events,
soft-reminder ablation, per-query-type policies, optimizer out-of-sample gain,
workflow-change value, token/context overhead, checkpoint/persistence overhead,
retry/duplicate waste, and guard overhead.

No percentages are hardcoded in this README. `ScenarioEvaluator` executes the
real guard on scripted scenarios; `ReplayEvaluator` replays tool-call sequences
with recorded results; `MessageReplayEvaluator` replays full recorded message
sequences with `ReplayChatModel` + `QueuedReplayTool`; `LLMQualityJudge` is an
optional offline judge. The benchmark is still synthetic, so results are
benchmark-bound, not production guarantees.

## Status

Implemented and tested:

- Multi-resource vectors, Decimal pricing, session metrics/estimators
- Reserve/commit/release ledger, composite run/thread/user scopes, concurrency leases
- Guard policy, multi-dimensional reminders, TTL cache, online middleware
- `CustomGraphGuard` for hand-built `StateGraph` (wrapped model node + guard node + routing helpers)
- `SessionRecorder` online → offline handoff (real guard counters → `SessionRecord`)
- SessionRecord + JSONL/SQLite stores + labels
- BatchAnalyzer, PolicyProposal, RuleProposer, Pareto gate, audit log
- `ScenarioEvaluator` executes the real guard on scripted scenarios for a
  candidate policy; `SyntheticEvaluator` remains a quick factor-based stand-in.
  Covered scenarios: repeated-query cache, model routing, API quota limits,
  DB concurrency burst, legitimate long task / false-positive rate, and
  cache-version key safety. `tests/test_concurrency.py` additionally forces two
  `wrap_tool_call` invocations to overlap (barrier) and verifies the third is
  denied.
- Cache invalidation events (`invalidate_tool` / `invalidate_context`).
- Answer quality: deterministic marker plus optional `LLMQualityJudge` with
  fallback; quality-regression gate.
- Recorded-trace replay at two levels: tool-call-level (`ReplayEvaluator`) and
  full model-message replay (`MessageReplayEvaluator` + `ReplayChatModel` +
  `QueuedReplayTool`). Recorded tool results are returned without re-executing
  side effects.
- **OptimizerAgent LangGraph StateGraph** (observe → … → reflect → finalize),
  streaming, checkpointer, HITL interrupt/resume
- `WorkflowSpec` execution for supported config-level changes, `CanaryPlan`, and
  `detect_drift`.
- CLI: `optimize --stream/--interactive/--dry-run/--apply/--canary`,
  `batch-run`, `evaluate`, `drift`, `workflow`; demo `demo/run.py`,
  `demo/llm_brain.py`, and `demo/real_agent.py` (real file-search + SQLite tools +
  `SessionRecorder`; `--fake` model for the no-key pipeline).
- Real-agent pipeline verified with a real model (`deepseek-flash` via the
  environment's Anthropic-compatible proxy), two real tools (file search +
  SQLite), `SessionRecorder`, `evaluate --sessions`, and an HTML report.
  The user accepted the DeepSeek real-model result as the substitute for
  Anthropic because the local `ANTHROPIC_API_KEY` returns 401 and the proxy
  serves only DeepSeek model names. Real Anthropic can be run with `--direct`
  and a valid key; `LLMBrain` supports `raise_on_error=True` so a real-provider
  failure does not silently fall back to `RuleBrain`.
- Real-model artifacts: `examples/real_agent/sessions_real.jsonl` and
  `reports/real-agent/evaluate.html` (provider: DeepSeek).
- Session adapters: `LangfuseSessionStore` (trace metadata source) and
  `CheckpointerSessionStore` (LangGraph `get_state_history` source); both are
  read-only and covered by tests.
- Mechanics benchmark harness (uniform + heterogeneous-cost scenarios) with
  generated JSON + Markdown reports
- `LLMBrain` adapter (diagnose/propose) tested with a scripted model and inside
  the full optimizer graph.

Planned for the full version:

- Real-provider verification with a valid key (the code path exists; the local
  `ANTHROPIC_API_KEY` returned 401, so `--fake` is the verified path)
- Full graph synthesis/reordering remains out of scope

See [ROADMAP.md](ROADMAP.md) for the open-source demand-validation plan and
explicit kill/pivot criteria. If the open-source pilot does not show demand, the
project pivots to a standalone benchmark/portfolio artifact rather than adding
more workflow-optimization features.

## Related work and differentiation

- **Opik Agent Optimizer** optimizes prompts/tool schemas/agent workflows. We
  optimize **resource policies** (USD, DB CU, concurrency, API quota, cache,
  model routing) and enforce them at runtime.
- **Trace** does general generative agent-system optimization. We focus on
  resource constraints and a reproducible resource Pareto gate.
- **Langfuse** provides sessions/traces/datasets/experiments. We reuse it as an
  optional storage/eval backend rather than rebuilding observability.
- **LoopBuster / agent-cost-guardrails / Traccia** provide online hard limits.
  We complement them with **offline batch multi-resource optimization**.

## License

Apache-2.0. See [LICENSE](LICENSE).
