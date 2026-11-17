# Scheduling the optimizer

The optimizer is offline by design: it reads a batch of completed sessions and
never sits on the user-query hot path. Run it nightly or weekly.

## cron

```cron
# Every day at 02:00, write a dry-run report; a human reviews and applies.
0 2 * * * cd /path/to/your/agent && \
  uv run resguard optimize --sessions ./sessions --since 7d --batch-size 500 \
  --out ./proposals/$(date +\%F) --stream --dry-run
```

## GitHub Actions

```yaml
name: resguard-optimize
on:
  schedule:
    - cron: "0 2 * * *"
  workflow_dispatch: {}
jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - name: Run optimizer dry-run
        run: |
          uv run resguard optimize \
            --sessions ./sessions \
            --since 7d \
            --out ./proposals/latest \
            --stream --dry-run
      - uses: actions/upload-artifact@v4
        with:
          name: resguard-proposals
          path: proposals/latest
```

A human reviews `report.md`, `raw.json`, and `audit.jsonl`; applying is a
separate explicit step (`--interactive` or a reviewed `--apply` in CI). The
online guard keeps using the pinned policy version until the new version is
deployed.
