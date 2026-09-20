"""The health checker is the only automated watchdog over a 15-hour run.

It had no tests, and two of its judgements were wrong in ways that mattered:

* A flat attribution softmax was filed as a PROBLEM. That is the paper's own
  predicted behaviour at this reward scale (notes 2.2), so on a 250-step run it
  would fire 250 times and bury a real fault -- and make the exit code useless.
* Any two consecutive rollouts were enough to declare a "trend", so ordinary
  step-to-step variation in length or entropy was reported as a collapse. Both
  fired on the smoke run while it was healthy.

These tests pin the distinction between an expected observation and a fault.
"""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts" / "check_run_health.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_run_health", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _healthy(rollout=1, **overrides):
    row = {
        "rollout": rollout,
        "loss": -0.05,
        "grad_norm": 0.7,
        "optimizer_steps": 1,
        "reward_count": 64,
        "raw_reward_mean": 0.55,
        "mean_response_tokens": 293.0,
        "response_entropy": 0.94,
        "kl_to_init": 0.001,
        "group_sigma_mean": 3.0,
        "truncated_responses": 0,
        "rollout_logp_abs_error_p50": 0.003,
        "rollout_logp_abs_error_mean": 0.012,
        "rollout_is_ess_ratio": 0.999,
        "initial_hf_logp_max_abs_error": 0.0,
        "credit_ess_ratio": 0.5,
        "p2t_flat_response_fraction": 0.0,
        "p2t_varying_bonus_over_advantage": 1e-5,
    }
    row.update(overrides)
    return row


def _report(tmp_path, rows):
    report = tmp_path / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return report


def test_flat_attribution_softmax_is_a_note_not_a_problem(tmp_path, capsys):
    """The documented inert-token-term result must not look like a malfunction."""
    module = _load()
    report = _report(tmp_path, [_healthy(credit_ess_ratio=0.9942,
                                        p2t_flat_response_fraction=1.0)])
    code = module.main(["--report", str(report)])
    out = capsys.readouterr().out
    assert code == 0, "an expected result must not make the exit code non-zero"
    assert "PROBLEM" not in out
    assert "note:" in out and "expected" in out


def test_a_real_fault_is_still_a_problem(tmp_path, capsys):
    module = _load()
    report = _report(tmp_path, [_healthy(loss=float("nan"))])
    assert module.main(["--report", str(report)]) == 1
    assert "PROBLEM" in capsys.readouterr().out


@pytest.mark.parametrize("field,value,fragment", [
    ("optimizer_steps", 0, "no optimizer step"),
    ("initial_hf_logp_max_abs_error", 0.5, "own cache"),
    ("rollout_is_ess_ratio", 0.5, "degenerate"),
    ("rollout_logp_abs_error_p50", 0.9, "systematic"),
    ("kl_to_init", 9.0, "drifted"),
    ("group_sigma_mean", 0.0, "advantages are ~0"),
])
def test_each_fault_condition_is_detected(tmp_path, field, value, fragment):
    module = _load()
    report = _report(tmp_path, [_healthy(**{field: value})])
    assert module.main(["--report", str(report)]) == 1


def test_two_noisy_rollouts_are_not_a_collapse(tmp_path, capsys):
    """Step-to-step variation is large; two points cannot establish a trend."""
    module = _load()
    rows = [_healthy(1, mean_response_tokens=700.0, response_entropy=1.8),
            _healthy(2, mean_response_tokens=549.0, response_entropy=1.07)]
    code = module.main(["--report", str(_report(tmp_path, rows))])
    out = capsys.readouterr().out
    assert code == 0, f"a two-point difference must not be a fault: {out}"
    assert "falling" not in out and "collapse" not in out


def test_a_sustained_decline_is_still_caught(tmp_path, capsys):
    """Enough points, and a genuine collapse must still be reported."""
    module = _load()
    rows = [_healthy(i + 1, mean_response_tokens=700.0 - 60 * i,
                     response_entropy=1.8 - 0.1 * i) for i in range(8)]
    assert module.main(["--report", str(_report(tmp_path, rows))]) == 1
    out = capsys.readouterr().out
    assert "falling" in out


def test_event_rows_before_the_first_step_are_not_read_as_a_rollout(tmp_path, capsys):
    """`prompt_filter` has no "rollout" key and must not be mistaken for a step."""
    module = _load()
    report = _report(tmp_path, [{"event": "prompt_filter", "kept_prompts": 59057}])
    assert module.main(["--report", str(report)]) == 0
    assert "PROBLEM" not in capsys.readouterr().out


def test_missing_metrics_file_is_not_a_fault(tmp_path):
    module = _load()
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert module.main(["--report", str(empty)]) == 0
