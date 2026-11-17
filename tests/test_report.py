from __future__ import annotations

import json

from resguard.report import build_html_report, write_html_report


def evaluation() -> dict:
    return {
        "baseline": {
            "resource_per_success": {
                "usd": 0.04,
                "tool_calls": 4.0,
                "db_cu": 1.0,
                "db_queries": 2.0,
                "api_quota": 1.0,
            },
            "success_rate": 1.0,
            "quality_rate": 1.0,
            "false_positive_rate": 0.0,
        },
        "candidate": {
            "resource_per_success": {
                "usd": 0.02,
                "tool_calls": 2.0,
                "db_cu": 0.8,
                "db_queries": 1.5,
                "api_quota": 0.5,
            },
            "success_rate": 1.0,
            "quality_rate": 1.0,
            "false_positive_rate": 0.0,
        },
        "result": {
            "accepted": True,
            "reasons": [],
            "improvements": {"usd": 0.5},
            "regressions": {},
        },
    }


def test_html_report_contains_kpis_table_and_svg() -> None:
    html = build_html_report(evaluation(), sessions_per_month=1000, optimizer_cost_usd=1.0)

    assert "resguard optimization report" in html
    assert "ACCEPTED" in html
    assert "<svg" in html
    assert "50.0%" in html
    assert "Projected monthly savings" in html
    assert "USD" in html
    assert "Success rate" in html


def test_write_html_report(tmp_path) -> None:
    input_path = tmp_path / "evaluate.json"
    input_path.write_text(json.dumps(evaluation()), encoding="utf-8")
    output_path = tmp_path / "report.html"

    written = write_html_report(input_path, output_path, sessions_per_month=500)

    assert written == output_path
    assert output_path.exists()
    assert "<html" in output_path.read_text(encoding="utf-8")
