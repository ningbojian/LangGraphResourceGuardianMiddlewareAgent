# Roadmap and demand validation

resguard is **open-source-first**. Open source is the first pilot: if developers
with expensive tools/DB/API do not try it, share sessions, or open issues, the
product hypothesis is not validated.

## Positioning

- **Wedge:** multi-resource guard (USD, DB CU, DB queries, API quota,
  concurrency, wall clock, checkpoint bytes) + a reproducible mixed-cost
  benchmark.
- **Experimental:** the Agentic Optimizer graph and LLM brain. They are a
  research/preview feature and a portfolio differentiator, not the adoption
  wedge.

## Phase 0 — OSS readiness (now)

- [x] installable package, tests, demo without API keys
- [x] mechanics benchmark with generated reports
- [x] Apache-2.0 LICENSE, py.typed, CI matrix (Linux/Windows)
- [x] custom `StateGraph` adapter (`CustomGraphGuard`: wrapped model node + guard node + routing)
- [ ] publish to PyPI
- [ ] docs site / quickstart video

## Phase 1 — demand validation (2-4 weeks)

Goal: find out whether LangGraph developers with expensive/private resources
have a recurring pain, and whether they will act on a report.

- 5-10 problem interviews. Ask about the last cost/quota/loop incident, not
  opinions.
- 3-5 concierge session audits: run our `BatchAnalyzer` on their (redacted)
  sessions and deliver a report.
- Launch in 5 places: LangChain forum, LangGraph GitHub discussions,
  r/LangChain, Show HN, Chinese developer communities.
- Metrics:
  - PyPI installs / unique users
  - GitHub stars, forks, unsolicited issues/PRs from non-friends
  - teams willing to share a second batch of sessions
  - teams that act on a concierge report

### Kill / pivot criteria

Pivot away from a product (keep it as a benchmark/portfolio) if, after 8 weeks:

- fewer than 3 of 10 interviewed developers had a cost/quota/loop incident in
  the last month;
- fewer than 3 teams share logs or act on a report;
- no unsolicited issue/PR from a non-friend.

## Phase 2 — if traction

- Real-provider verification with a valid key (the code path exists; the local
  `ANTHROPIC_API_KEY` returned 401, so `--fake` is the verified path)
- Full graph synthesis/reordering remains out of scope

Already implemented in the current slice: full recorded **model-message**
replay, LLM-judge quality with deterministic fallback, supported
`WorkflowChange` execution, canary plans, drift detection, the
`batch-run` / `evaluate` / `drift` / `workflow` CLI commands, and the
`LangfuseSessionStore` / `CheckpointerSessionStore` read-only session adapters.

## Phase 3 — if no traction

- Publish the benchmark as a standalone open artifact ("Agent Resource Bench")
- Keep the guard + optimizer as a portfolio project; document negative results
  and validity threats
- Do not keep adding workflow-optimization features without demand
