"""Standalone HTML before/after report for resource-policy evaluation.

No dashboard, no external JS, no extra dependency: the chart is inline SVG and
the page is self-contained. It is generated from ``resguard evaluate`` JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping

_RESOURCE_DIMENSIONS = (
    ("usd", "USD"),
    ("tool_calls", "Tool calls"),
    ("db_queries", "DB queries"),
    ("db_cu", "DB CU"),
    ("api_quota", "API quota"),
)


def _get(insights: Mapping[str, Any], dimension: str) -> float:
    return float((insights.get("resource_per_success") or {}).get(dimension, 0.0))


def _pct_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def _pp_delta(before: float, after: float) -> float:
    return (after - before) * 100.0


def _resource_status(before: float, after: float) -> str:
    reduction = _pct_reduction(before, after)
    if reduction >= 10:
        return "improved"
    if reduction <= -2:
        return "regressed"
    return "flat"


def _metric_status(before: float, after: float) -> str:
    delta = _pp_delta(before, after)
    if delta >= 2:
        return "improved"
    if delta <= -2:
        return "regressed"
    return "flat"


def _false_positive_status(before: float, after: float) -> str:
    delta = _pp_delta(before, after)
    if delta <= -2:
        return "improved"
    if delta >= 2:
        return "regressed"
    return "flat"


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _svg_bars(rows: list[tuple[str, float, float]]) -> str:
    if not rows:
        return ""
    width = 760
    row_height = 54
    height = 30 + row_height * len(rows)
    max_value = max([max(before, after) for _, before, after in rows] + [1e-9])
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Before and after resource usage" '
        'style="width:100%;height:auto">'
    ]
    for index, (label, before, after) in enumerate(rows):
        y = 20 + index * row_height
        before_width = max(1.0, before / max_value * 380)
        after_width = max(1.0, after / max_value * 380)
        parts.append(
            f'<text x="8" y="{y + 24}" font-size="13" fill="#334155">{escape(label)}</text>'
        )
        parts.append(
            f'<rect x="170" y="{y + 6}" width="{before_width:.1f}" height="16" '
            'rx="3" fill="#94a3b8"/>'
        )
        parts.append(
            f'<text x="{176 + before_width:.1f}" y="{y + 18}" font-size="11" '
            f'fill="#475569">baseline {_fmt(before)}</text>'
        )
        parts.append(
            f'<rect x="170" y="{y + 26}" width="{after_width:.1f}" height="16" '
            'rx="3" fill="#0f766e"/>'
        )
        parts.append(
            f'<text x="{176 + after_width:.1f}" y="{y + 38}" font-size="11" '
            f'fill="#0f766e">candidate {_fmt(after)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _kpi(label: str, value: str, detail: str, status: str = "flat") -> str:
    return (
        f'<div class="kpi {status}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-detail">{escape(detail)}</div></div>'
    )


def build_html_report(
    evaluation: Mapping[str, Any],
    *,
    sessions_per_month: int = 10_000,
    optimizer_cost_usd: float = 0.0,
) -> str:
    baseline = evaluation["baseline"]
    candidate = evaluation["candidate"]
    result = evaluation.get("result", {})
    usd_before = _get(baseline, "usd")
    usd_after = _get(candidate, "usd")
    usd_saved_pct = _pct_reduction(usd_before, usd_after)
    usd_saved_abs = usd_before - usd_after
    monthly_savings = usd_saved_abs * sessions_per_month - optimizer_cost_usd
    tool_before = _get(baseline, "tool_calls")
    tool_after = _get(candidate, "tool_calls")
    api_before = _get(baseline, "api_quota")
    api_after = _get(candidate, "api_quota")
    success_before = float(baseline.get("success_rate", 0.0))
    success_after = float(candidate.get("success_rate", 0.0))
    quality_before = float(baseline.get("quality_rate", 0.0))
    quality_after = float(candidate.get("quality_rate", 0.0))
    fp_before = float(baseline.get("false_positive_rate", 0.0))
    fp_after = float(candidate.get("false_positive_rate", 0.0))

    rows = [
        (label, _get(baseline, dimension), _get(candidate, dimension))
        for dimension, label in _RESOURCE_DIMENSIONS
    ]
    rows = [row for row in rows if row[1] > 0 or row[2] > 0]

    table_rows = []
    for label, before, after in rows:
        reduction = _pct_reduction(before, after)
        status = _resource_status(before, after)
        table_rows.append(
            f"<tr class='{status}'><td>{escape(label)}</td><td>{_fmt(before)}</td>"
            f"<td>{_fmt(after)}</td><td>{reduction:+.1f}%</td>"
            f"<td>{escape(status)}</td></tr>"
        )
    table_rows.append(
        f"<tr class='{_metric_status(success_before, success_after)}'><td>Success rate</td>"
        f"<td>{success_before:.1%}</td><td>{success_after:.1%}</td>"
        f"<td>{_pp_delta(success_before, success_after):+.1f} pp</td>"
        f"<td>{_metric_status(success_before, success_after)}</td></tr>"
    )
    table_rows.append(
        f"<tr class='{_metric_status(quality_before, quality_after)}'><td>Quality rate</td>"
        f"<td>{quality_before:.1%}</td><td>{quality_after:.1%}</td>"
        f"<td>{_pp_delta(quality_before, quality_after):+.1f} pp</td>"
        f"<td>{_metric_status(quality_before, quality_after)}</td></tr>"
    )
    table_rows.append(
        f"<tr class='{_false_positive_status(fp_before, fp_after)}'><td>False-positive stop rate</td>"
        f"<td>{fp_before:.1%}</td><td>{fp_after:.1%}</td>"
        f"<td>{_pp_delta(fp_before, fp_after):+.1f} pp</td>"
        f"<td>{_false_positive_status(fp_before, fp_after)}</td></tr>"
    )

    reasons = result.get("reasons") or []
    improvements = result.get("improvements") or {}
    status_text = "ACCEPTED" if result.get("accepted") else "REJECTED"
    status_class = "accepted" if result.get("accepted") else "rejected"
    generated = datetime.now(timezone.utc).isoformat()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>resguard optimization report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #f8fafc; color: #0f172a; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }}
  .muted {{ color: #64748b; font-size: 13px; }}
  .status {{ display:inline-block; padding: 4px 10px; border-radius: 999px;
             font-size: 12px; font-weight: 700; letter-spacing: .04em; }}
  .status.accepted {{ background:#dcfce7; color:#166534; }}
  .status.rejected {{ background:#fee2e2; color:#991b1b; }}
  .kpis {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr));
           gap: 12px; margin: 22px 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px; }}
  .kpi.improved {{ border-color:#86efac; }}
  .kpi.regressed {{ border-color:#fca5a5; }}
  .kpi-label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.04em; }}
  .kpi-value {{ font-size:24px; font-weight:700; margin-top:6px; }}
  .kpi-detail {{ font-size:12px; color:#64748b; margin-top:4px; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:18px;
           margin:18px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ text-align:left; padding:9px 8px; border-bottom:1px solid #e2e8f0; }}
  th {{ color:#475569; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  tr.improved td:nth-child(5) {{ color:#166534; font-weight:600; }}
  tr.regressed td:nth-child(5) {{ color:#991b1b; font-weight:600; }}
  .note {{ font-size:13px; color:#475569; line-height:1.6; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body><main>
  <h1>resguard optimization report</h1>
  <div class="muted">generated {escape(generated)} · benchmark-bound comparison</div>
  <p><span class="status {status_class}">{status_text}</span></p>

  <div class="kpis">
    {_kpi("Cost / success", f"{usd_saved_pct:+.1f}%", f"baseline {_fmt(usd_before)} → candidate {_fmt(usd_after)} USD", "improved" if usd_saved_pct >= 10 else "flat")}
    {_kpi("Tool calls", f"{(_pct_reduction(tool_before, tool_after)):+.1f}%", f"{_fmt(tool_before)} → {_fmt(tool_after)} per success", "improved" if _pct_reduction(tool_before, tool_after) >= 10 else "flat")}
    {_kpi("API quota", f"{(_pct_reduction(api_before, api_after)):+.1f}%", f"{_fmt(api_before)} → {_fmt(api_after)} per success", "improved" if _pct_reduction(api_before, api_after) >= 10 else "flat")}
    {_kpi("Success / quality", f"{_pp_delta(success_before, success_after):+.1f} / {_pp_delta(quality_before, quality_after):+.1f} pp", "change must stay within tolerance", _metric_status(success_before, success_after))}
    {_kpi("Projected monthly savings", f"${monthly_savings:,.2f}", f"{sessions_per_month:,} sessions/month, optimizer cost ${optimizer_cost_usd:,.2f}", "improved" if monthly_savings > 0 else "regressed")}
  </div>

  <div class="card">
    <h2 style="margin-top:0;font-size:18px">Resource usage per successful task</h2>
    {_svg_bars(rows)}
  </div>

  <div class="card">
    <h2 style="margin-top:0;font-size:18px">Before / after</h2>
    <table>
      <thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Change</th><th>Status</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0;font-size:18px">Gate</h2>
    <p class="note"><strong>Improvements:</strong> {escape(json.dumps(improvements))}<br>
    <strong>Reasons:</strong> {escape(json.dumps(reasons))}</p>
    <p class="note">This report is generated from a benchmark/replay evaluation, not a production canary.
    Treat the numbers as evidence about guard mechanics and policy trade-offs, not as a guaranteed
    production saving. See <code>docs/DESIGN.md</code> for validity threats.</p>
  </div>
</main></body></html>"""


def write_html_report(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sessions_per_month: int = 10_000,
    optimizer_cost_usd: float = 0.0,
) -> Path:
    evaluation = json.loads(Path(input_path).read_text(encoding="utf-8"))
    html = build_html_report(
        evaluation,
        sessions_per_month=sessions_per_month,
        optimizer_cost_usd=optimizer_cost_usd,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
