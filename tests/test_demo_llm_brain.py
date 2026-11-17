from __future__ import annotations

from demo.llm_brain import main


def test_fake_llm_brain_demo_runs(capsys) -> None:
    exit_code = main(["--fake", "--stream"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[diagnose]" in output
    assert "focus:" in output
    assert "termination_reason: accepted" in output
