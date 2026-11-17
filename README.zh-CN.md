# resguard

[English](README.md) | 简体中文

[![CI](https://github.com/ningbojian/LangGraphResourceGuardianMiddlewareAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/ningbojian/LangGraphResourceGuardianMiddlewareAgent/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: developer preview](https://img.shields.io/badge/status-developer%20preview-orange.svg)](ROADMAP.md)
[![Reports](https://img.shields.io/badge/reports-GitHub%20Pages-0f766e.svg)](https://ningbojian.github.io/LangGraphResourceGuardianMiddlewareAgent/)

**面向 LangGraph Agent 开发者的多资源护栏 + 批量优化工具。**

## 为什么需要它

LangGraph 的 `recursion_limit` 和按调用次数限制的 middleware 是必要的，但它们是**成本盲**的：

- 一个便宜工具循环 10 次可能没事；多一次 PDF 解析调用就可能超掉一个小额美元预算。
- LLM 网关只能看到 token，看不到你的向量库 CU、MCP 工具、文档解析配额、数据库连接数、并发上限。
- 观测平台能看到 trace，但不会执行 per-run 的资源策略。

mechanics benchmark 把这个问题量化了：官方 call-count limit 执行全部 10 次调用、花掉 **$0.408**，超过 **$0.25** 预算；USD-aware guard 停在 9 次 / **$0.208**。

resguard 把这些资源变成一等公民：一个轻量在线 guard，加一个离线批量 optimizer，由 Pareto gate 评估、开发者审核后再版本化上线。

## 适用 / 不适用

**适用：** 调用昂贵或私有资源的 LangGraph Agent 开发者（向量库、文档解析、Serverless DB、MCP），尤其是已经有 session 日志、想要 per-run guard + 离线报告的团队。

**不适用：** 只需要网关层 LLM token 预算的团队（用 LiteLLM/Portkey）；没有 session 日志或 success label 的团队（optimizer 无法证明改进）；想要完整可观测平台的团队（用 Langfuse/LangSmith）。

## 稳定 vs 实验

- **相对稳定：** 资源模型、ledger、policy/reminder、cache、middleware、`SessionRecorder`、session stores、BatchAnalyzer、proposer、Pareto gate、CLI、mechanics benchmark。有测试覆盖。
- **实验性：** `OptimizerAgent`、`LLMBrain`、`SyntheticEvaluator`、`MessageReplayEvaluator`、`LLMQualityJudge`、`WorkflowSpec` 执行、`CanaryPlan`、`detect_drift`。API 可能变化；`SyntheticEvaluator` 不是证据。

## 两部分架构

- **在线（轻量）：** `ResourceGuardMiddleware` 执行已批准的静态资源策略并暴露 counters；**不调用 LLM**，所以不会增加用户 query 的智能决策开销。`SessionRecorder` 把 counters 写成 `SessionRecord`。
- **离线（实验性 Agentic Optimizer）：** `OptimizerAgent` 是一个 LangGraph `StateGraph`，批量读取已结束的 session，诊断资源/失败热点，提出资源策略改动，用 Pareto gate 评估，并在版本化之前暂停等待人工确认。

> 开源 adoption 的楔子是 **guard + benchmark**；离线 optimizer 是 research/preview 部分，让项目不只是又一个 middleware。它保留了 Agent 感（state transitions、条件边、循环、tools、checkpointer、HITL、streaming），同时不增加每 query 的 LLM 成本。

## 为什么它是 Agent，不是 cron script

架构与效度威胁见 [docs/DESIGN.md](docs/DESIGN.md)，需求验证与 kill criteria 见 [ROADMAP.md](ROADMAP.md)。

```
observe → diagnose → hypothesize → plan_experiment → execute_experiment
   ↑                                                    ↓
   └──────────── reflect ← evaluate ←───────────────────┘
                         ↓
                      finalize → 人工审批（interrupt）→ 版本化策略
```

- 它是真正的 LangGraph `StateGraph`，有条件边和循环。
- LLM 可以在 `diagnose`（先查哪个热点）、`hypothesize`（提什么提案）、`reflect`（继续还是 finalize）做决策；`RuleBrain` 是确定性 fallback，`LLMBrain` 是可插拔 LLM adapter。
- 使用 checkpointer + `interrupt` 支持开发者审批/恢复。
- `stream_mode="updates"` 把实时状态变化暴露给开发者。
- **离线运行**，读取已完成 session，独立进程；不在用户 query 热路径上，也不会每次交互后改参数。

## 快速开始

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m demo.run
uv run python -m demo.run --interactive   # 查看 HITL interrupt/resume
uv run python -m demo.llm_brain --fake --stream
uv run python -m benchmark.run --out reports/

# 真实 LLM brain（需要有效 provider key；本环境 ANTHROPIC_API_KEY 返回 401，
# 因此已验证的是 --fake 路径）
uv sync --extra dev --extra llm
export ANTHROPIC_API_KEY=...
uv run python -m demo.llm_brain --model anthropic:claude-haiku-5-5 --stream

# 真实 Anthropic（需要有效 ANTHROPIC_API_KEY；--direct 会绕过本地
# ANTHROPIC_BASE_URL 代理——本环境该代理实际服务 DeepSeek）
uv run python -m demo.real_agent --direct \
  --model anthropic:claude-haiku-5-5 \
  --data examples/real_agent/data \
  --out examples/real_agent/sessions_real.jsonl

# 本环境可用的真实模型（DeepSeek，通过 Anthropic-compatible 代理）
uv run python -m demo.real_agent --model deepseek-flash \
  --data examples/real_agent/data \
  --out examples/real_agent/sessions_real.jsonl
uv run resguard evaluate --sessions examples/real_agent/sessions_real.jsonl \
  --tool-profiles examples/real_agent/tool_profiles.json \
  --baseline examples/real_agent/policy_baseline.json \
  --candidate examples/real_agent/policy_candidate.json \
  --out reports/real-agent/ \
  --sessions-per-month 10000 --optimizer-cost-usd 5

# 场景级 CLI
uv run resguard batch-run --dataset examples/eval/scenarios.jsonl \
  --policy examples/eval/policy_baseline.json --out runs/
uv run resguard evaluate --baseline examples/eval/policy_baseline.json \
  --candidate examples/eval/policy_candidate.json \
  --dataset examples/eval/scenarios.jsonl --out reports/eval/

# 可视化 before/after：自包含 HTML（inline SVG，无 dashboard 依赖）
uv run resguard report --input reports/eval/evaluate.json \
  --out reports/eval/report.html \
  --sessions-per-month 10000 --optimizer-cost-usd 5

# 用你自己的 recorded sessions 做 full message replay
uv run resguard evaluate --sessions ./sessions \
  --tool-profiles examples/eval/tool_profiles.json \
  --baseline examples/eval/policy_baseline.json \
  --candidate examples/eval/policy_candidate.json \
  --out reports/eval-sessions/
```

`uv run python -m demo.run` 使用 `ScenarioEvaluator`（在脚本化场景上真实执行 guard），打印 baseline vs candidate 的 `resource_per_success`、false-positive rate 和 candidate tool calls。`--interactive` 展示 HITL interrupt/resume。

benchmark 是 **mechanics** benchmark（scripted model + mock tools），写入 `reports/benchmark.json` / `reports/benchmark.md`，不是质量或生产 benchmark：

- 在 uniform-cost loop 上，官方 call-count limit 和 full guard 打平；这种场景次数限制已经够用。
- 在 `mixed_cost_spike`（8 次便宜调用 + 2 次贵 200 倍的调用）上，官方 limit 10 执行全部 10 次、花 **$0.408**，超过 **$0.25** 预算；USD-aware full guard 停在 9 次 / **$0.208**，在预算内。这是成本感知的差异化。数字由 harness 生成，见 `reports/benchmark.md`。
- 不要把当前表格当作生产质量声明。

对 JSONL session 批次运行 optimizer：

```bash
uv run resguard optimize \
  --sessions ./sessions \
  --since 7d \
  --out ./proposals/latest \
  --stream \
  --dry-run

# 审阅 report.md / raw.json / audit.jsonl 之后：
uv run resguard optimize --sessions ./sessions --out ./proposals/latest --interactive --apply
```

`--apply` 如果不带 `--interactive` 或 CI-only 的 `--auto-approve` 会拒绝执行；默认只出 dry-run 报告。

## 在线 SDK

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

# agent run 之后，把 guard 的真实 counters 写成 SessionRecord，交给离线 optimizer。
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

guard 会执行 cumulative 资源限制，在 model-call 末尾注入 ephemeral 多维提醒（**绝不写 system prompt**），返回 `ToolMessage` denial，并在 cache hit/error 时释放 reservation。

### 自定义 StateGraph

`AgentMiddleware` 只接入 `create_agent`。手写图可以用 `CustomGraphGuard` 获得同样的 guard 语义：

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

`guard_node` 自己执行 tool calls（替代普通 `ToolNode`），因此可以做多维 reserve/commit、cache hit、concurrency lease。wrapped model node 会检查预算、把 ephemeral reminder 注入 state 的副本，并 commit 实际 model usage。

### 集成支持

| 集成方式 | 记录 | Guard | Workflow 优化 |
|---|---|---|---|
| `create_agent` + middleware | `SessionRecorder` | `ResourceGuardMiddleware` | 仅 reference template |
| 手写 `StateGraph` | `SessionRecorder` | `CustomGraphGuard` | 仅 patch 建议 |
| LangGraph checkpointer | `CheckpointerSessionStore`（读 `get_state_history`；派生 metrics；可选 `label_fn`） | — | — |
| Langfuse | `LangfuseSessionStore`（读 trace metadata `resguard_session`；可选 `create_trace`） | — | — |

两个 adapter 都是只读来源。`CheckpointerSessionStore` 需要显式传 `thread_ids`；它从 state 派生 tool calls/token usage/checkpoint 指标，并支持可选 success `label_fn`。`LangfuseSessionStore` 需要传入 `fetch_traces` callable（包装你的 Langfuse client）；adapter 本身不 import Langfuse SDK。

## 资源模型

可观测 / 可执行维度：

| 维度 | 单位 | 模式 |
|---|---|---|
| `USD` | $ | cumulative |
| `LLM_INPUT_TOKENS` / `LLM_OUTPUT_TOKENS` | tokens | cumulative |
| `TOOL_CALLS` | calls | cumulative |
| `DB_QUERIES` | queries | cumulative |
| `DB_CU` | CU | cumulative |
| `API_QUOTA` | units | cumulative |
| `CONCURRENCY` | slots | gauge（lease） |
| `WALL_CLOCK_MS` | ms | cumulative |
| `CHECKPOINT_BYTES` | bytes | cumulative |

`SessionMetrics` 还记录 token/context、checkpoint/persistence、retry/duplicate、subagent/observability、guard overhead 等观测指标。optimizer 用观测指标做诊断；只有可执行资源会被硬限制。

## 仓库结构

```
src/resguard/
  resources.py        多维资源向量
  costs.py            Decimal token 定价
  metrics.py          SessionMetrics + token/schema/checkpoint 估算
  ledger.py           reserve/commit/release + concurrency lease + composite scopes
  policy.py           阈值、硬限动作、降级选择
  reminder.py         多维 ephemeral 提醒 + 去重
  cache.py            per-tool TTL cache（tenant/user/doc-version key）
  middleware.py       ResourceGuardMiddleware（在线执行器）
  custom.py           CustomGraphGuard（自定义 StateGraph adapter）
  recorder.py         SessionRecorder（在线 → 离线 handoff）
  session.py          SessionRecord、JSONL/SQLite stores、LabelStore
  batch.py            确定性 BatchAnalyzer
  evaluator.py        ScenarioEvaluator（在脚本化场景上真实执行 guard）
  report.py           自包含 before/after HTML 报告（inline SVG）
  replay.py           tool-call + full model-message replay
  judge.py            deterministic marker + 可选 LLM-as-judge
  integrations/
    langfuse.py       LangfuseSessionStore（trace metadata 来源）
    checkpointer.py   CheckpointerSessionStore（LangGraph state-history 来源）
  workflow.py         WorkflowSpec + 受支持的 WorkflowChange 执行
  rollout.py          CanaryPlan + drift detection
  testing.py          ScriptedChatModel + counting/replay tools
  proposer.py         PolicyProposal、RuleProposer
  optimizer.py        Pareto gate + audit log
  optimizer_agent/    LangGraph StateGraph、brains、synthetic evaluator
  cli.py              optimize / batch-run / evaluate / drift / workflow
tests/                单元/集成测试
demo/run.py           无 API key 的 end-to-end demo
demo/llm_brain.py     真实 provider 的 LLM brain demo（+ --fake 无 key smoke test）
demo/real_agent.py    真实文件检索 + SQLite 工具 + SessionRecorder
examples/real_agent/  real-agent 数据/policy/tool profiles + 生成的 sessions
examples/scheduling/  cron + GitHub Actions 示例
docs/DESIGN.md        架构 + 效度威胁
ROADMAP.md            需求验证 + kill criteria
LICENSE               Apache-2.0
```

## 量化 before/after 报告

`resguard evaluate` 会写 `evaluate.json` + `evaluate.html`。HTML 报告自包含（inline CSS + SVG，无 dashboard / JS 依赖），包含：

- **Cost / success**（`resource_per_success["usd"]`）baseline → candidate，以及下降百分比。
- **每个成功任务的 tool calls / DB queries / DB CU / API quota**。
- **Success rate / quality rate / false-positive stop rate** 的百分点变化。
- **预计月节省** = `(usd_before - usd_after) × sessions_per_month − optimizer_cost_usd`。
- Pareto gate 的 accepted/rejected、improvements、reasons。

报告来自 benchmark/replay 评测，不是 production canary；把它当作 guard mechanics 和策略 trade-off 的证据，而不是保证的生产节省。

## 评测方法（摘要）

只有在**不降低 success rate、不提高 false-positive stop、不降低 answer quality** 的前提下改善至少一个资源维度，optimizer 才被认为有用。Pareto gate 检查 success、false-positive、quality、逐维度改善和逐维度回退。完整 gate 计划按维度报告 used/limit、`resource_per_success`、mean ± std、per-session delta、paired bootstrap 95% CI。研究问题包括 tool/USD 下降、DB queries/CU、concurrency violations、API quota events、soft-reminder 消融、per-query-type policy、optimizer out-of-sample gain、workflow change 价值、token/context 开销、checkpoint/persistence 开销、retry/duplicate 浪费、guard overhead。

README 不写死百分比。`ScenarioEvaluator` 在脚本化场景上真实执行 guard；`ReplayEvaluator` 用 recorded results 回放 tool-call 序列；`MessageReplayEvaluator` 用 `ReplayChatModel` + `QueuedReplayTool` 回放完整 recorded message 序列；`LLMQualityJudge` 是可选离线 judge。benchmark 仍是 synthetic，所以结论限定在 benchmark，不外推生产。

## 当前状态

已实现并测试：

- 多维资源向量、Decimal 定价、session metrics/estimators
- reserve/commit/release ledger、run/thread/user composite scopes、concurrency lease
- Guard policy、多维提醒、TTL cache、在线 middleware
- `CustomGraphGuard`（wrapped model node + guard node + routing helpers）
- `SessionRecorder` 在线 → 离线 handoff（真实 guard counters → `SessionRecord`）
- SessionRecord + JSONL/SQLite stores + labels
- BatchAnalyzer、PolicyProposal、RuleProposer、Pareto gate、audit log
- `ScenarioEvaluator` 在脚本化场景上真实执行 guard；覆盖 repeated-query cache、model routing、API quota、DB concurrency burst、legitimate long task / false-positive、cache-version key safety。`tests/test_concurrency.py` 还用 barrier 强制两个 `wrap_tool_call` 重叠，并验证第三个被拒绝。
- Cache invalidation（`invalidate_tool` / `invalidate_context`）。
- Answer quality：deterministic marker + 可选 `LLMQualityJudge` fallback；quality-regression gate。
- 两级 recorded-trace replay：tool-call 级（`ReplayEvaluator`）和 full model-message 级（`MessageReplayEvaluator` + `ReplayChatModel` + `QueuedReplayTool`）。recorded tool results 直接返回，不重复执行副作用。
- **OptimizerAgent LangGraph StateGraph**（observe → … → reflect → finalize）、streaming、checkpointer、HITL interrupt/resume
- `WorkflowSpec` 受支持的 config-level 变更执行、`CanaryPlan`、`detect_drift`
- CLI：`optimize --stream/--interactive/--dry-run/--apply/--canary`、`batch-run`、`evaluate`、`drift`、`workflow`；demo `demo/run.py`、`demo/llm_brain.py`、`demo/real_agent.py`
- 真实模型 pipeline 已验证：`deepseek-flash`（通过本环境的 Anthropic-compatible 代理）+ 2 个真实工具（文件检索 + SQLite）+ `SessionRecorder` + `evaluate --sessions` + HTML 报告
- 真实模型产物：`examples/real_agent/sessions_real.jsonl`、`reports/real-agent/evaluate.html`（provider: DeepSeek）
- Session adapters：`LangfuseSessionStore`（trace metadata 来源）和 `CheckpointerSessionStore`（LangGraph `get_state_history` 来源）；两者只读，有测试覆盖
- Mechanics benchmark（uniform + heterogeneous-cost scenarios），生成 JSON + Markdown 报告
- `LLMBrain` adapter（diagnose/propose）用 scripted model 和完整 optimizer graph 测试

**真实模型结果：** 3 个 session，`deepseek-flash`，tools 1/1/2，success=true；baseline USD/success **0.003251** → candidate **0.002775**（**-14.6%**，accepted）。当前 **181 tests passed**。

**Anthropic 说明：** 本环境 `ANTHROPIC_API_KEY` 返回 401，本地 `ANTHROPIC_BASE_URL` 代理只服务 DeepSeek 模型名，Claude Code CLI 也走 `ccs-claude-opencode--deepseek-v4.1-flash`。因此真实 Anthropic 在本环境未验证；有有效 key 时用 `--direct --model anthropic:claude-haiku-5-5` 即可。`LLMBrain` 支持 `raise_on_error=True`，不会再静默 fallback 到 `RuleBrain`。

后续 full version：

- 有有效 key 时做真实 Anthropic/OpenAI 验证（代码路径已存在）
- 完整 graph synthesis/reordering 仍 out of scope

开源需求验证与 kill/pivot criteria 见 [ROADMAP.md](ROADMAP.md)。如果开源 pilot 没有需求信号，项目转为 standalone benchmark/portfolio artifact，而不是继续堆 workflow 优化功能。

## 相关工作与差异化

- **Opik Agent Optimizer** 优化 prompt、tool schema、agent workflow；我们优化**资源策略**（USD、DB CU、concurrency、API quota、cache、model routing），并在运行时执行。
- **Trace** 做通用生成式 agent-system 优化；我们聚焦资源约束和可复现的资源 Pareto gate。
- **Langfuse** 提供 sessions/traces/datasets/experiments；我们把它作为可选存储/评测后端，而不是重建可观测平台。
- **LoopBuster / agent-cost-guardrails / Traccia** 提供在线硬限；我们用**离线批量多资源优化**做互补。

## License

Apache-2.0。见 [LICENSE](LICENSE)。
