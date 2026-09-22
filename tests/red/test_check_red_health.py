"""The RED health checker's collapse trip-wires, and its mirrored stamps.

``check_red_health.py`` is a standalone script rather than a module, so it
duplicates the protocol stamps instead of importing them (running it puts
``red/scripts`` on sys.path, not the repository root).  That mirror is asserted
here: drift between the two would let a run from a retired rule be certified as
healthy, which is the one failure this checker exists to prevent.

The trip-wire thresholds are calibrated against what ``red250`` actually
reported, so they are tested against those numbers rather than against
made-up ones wherever the real artifact is available.
"""
from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from red.reward import (RED_PROTOCOL, RETIRED_RLOO_ADVANTAGE_RULE,
                        RLOO_ADVANTAGE_RULE)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "red" / "scripts" / "check_red_health.py"
RED250_METRICS = ROOT / "reports" / "red250" / "metrics.jsonl"

_spec = importlib.util.spec_from_file_location("check_red_health", SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def _row(**overrides):
    """A metrics row that passes every check, before overrides."""
    row = {
        "rollout": 5, "optimizer_steps": 1, "loss": -1.0, "grad_norm": 0.5,
        "raw_reward_mean": 4.0, "response_entropy": 1.2, "kl_to_init": 0.05,
        "mean_response_tokens": 400.0, "red_token_reward_abs_mean": 1.0,
        "credit_ess_ratio": 0.3, "rloo_baseline_mean": 4.0, "advantage_abs_mean": 0.8,
        "group_sigma_mean": 3.0, "reward_count": 64, "truncated_responses": 0,
        "rollout_logp_abs_error_p50": 0.001, "rollout_is_ess_ratio": 0.999,
        "initial_hf_logp_max_abs_error": 0.0,
        "red_protocol": RED_PROTOCOL, "red_advantage_rule": RLOO_ADVANTAGE_RULE,
        "red_positive_advantage_fraction": 0.5, "red_bonus_over_advantage": 1.2,
        "red_advantage_flip_fraction": 0.08,
    }
    row.update(overrides)
    return row


def _run(tmp_path, monkeypatch, rows, *extra, capsys=None):
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["check_red_health.py", "--report", str(tmp_path),
                                      *extra])
    code = checker.main()
    return code, (capsys.readouterr().out if capsys else "")


def test_mirrored_stamps_match_the_package():
    assert checker.RED_PROTOCOL == RED_PROTOCOL
    assert checker.RLOO_ADVANTAGE_RULE == RLOO_ADVANTAGE_RULE
    assert checker.RETIRED_RLOO_ADVANTAGE_RULE == RETIRED_RLOO_ADVANTAGE_RULE


def test_a_healthy_row_passes(tmp_path, monkeypatch, capsys):
    code, out = _run(tmp_path, monkeypatch, [_row()], capsys=capsys)
    assert code == 0, out
    assert "ok" in out


def test_the_retired_rule_is_a_problem_not_an_unknown_string(tmp_path, monkeypatch, capsys):
    """A red250-era artifact reaching this check must not be certified."""
    code, out = _run(tmp_path, monkeypatch,
                     [_row(red_advantage_rule=RETIRED_RLOO_ADVANTAGE_RULE)], capsys=capsys)
    assert code == 1
    assert "retired" in out
    assert "uncertain" not in out.lower()


@pytest.mark.parametrize("pushed", [1.0, 0.0])
def test_advantages_collapsed_onto_one_sign_are_a_problem(pushed, tmp_path, monkeypatch, capsys):
    """``red250`` reported exactly 1.000 from rollout 26 for the rest of the run."""
    code, out = _run(tmp_path, monkeypatch,
                     [_row(red_positive_advantage_fraction=pushed)], capsys=capsys)
    assert code == 1
    assert "collapsed onto one sign" in out


def test_a_redistributed_term_that_stopped_mattering_is_a_problem(tmp_path, monkeypatch, capsys):
    code, out = _run(tmp_path, monkeypatch, [_row(red_bonus_over_advantage=1e-6)], capsys=capsys)
    assert code == 1
    assert "stopped mattering" in out


def test_a_flip_fraction_flat_at_zero_over_a_window_is_a_problem(tmp_path, monkeypatch, capsys):
    """The signature red250 carried for 40 rollouts before it crashed."""
    rows = [_row(rollout=index, red_advantage_flip_fraction=0.0) for index in range(1, 11)]
    code, out = _run(tmp_path, monkeypatch, rows, capsys=capsys)
    assert code == 1
    assert "red250 collapse signature" in out


def test_one_quiet_rollout_is_only_a_note(tmp_path, monkeypatch, capsys):
    """A single low flip fraction is ordinary; the window is what makes it a problem."""
    rows = [_row(rollout=index, red_advantage_flip_fraction=0.09) for index in range(1, 10)]
    rows.append(_row(rollout=10, red_advantage_flip_fraction=0.0))
    code, out = _run(tmp_path, monkeypatch, rows, capsys=capsys)
    assert code == 0, out
    assert "note:" in out


def test_stop_on_problem_requires_a_pidfile(monkeypatch, capsys):
    """It must not be possible to SIGTERM whatever happens to be at pid 0."""
    monkeypatch.setattr(sys, "argv", ["check_red_health.py", "--report", "x",
                                      "--stop-on-problem"])
    with pytest.raises(SystemExit) as exit_info:
        checker.main()
    assert exit_info.value.code == 2
    assert "--pidfile" in capsys.readouterr().err


def test_stop_on_problem_signals_only_the_named_process(tmp_path, monkeypatch):
    """A real child process, so the signal path is exercised rather than mocked."""
    child = subprocess.Popen(["sleep", "60"])
    try:
        pidfile = tmp_path / "train.pid"
        pidfile.write_text(str(child.pid))
        checker.stop_run(pidfile, ["a problem worth stopping for"])
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()


def test_stop_on_problem_leaves_an_absent_run_alone(tmp_path, capsys):
    """The watcher races the run's own exit, so a dead pid is expected, not an error."""
    child = subprocess.Popen(["true"])
    child.wait(timeout=10)
    pidfile = tmp_path / "train.pid"
    pidfile.write_text(str(child.pid))
    checker.stop_run(pidfile, ["a problem"])
    assert "already gone" in capsys.readouterr().out
    checker.stop_run(tmp_path / "missing.pid", ["a problem"])
    assert "no usable pid file" in capsys.readouterr().out


@pytest.mark.skipif(not RED250_METRICS.is_file(),
                    reason="the red250 artifacts are not present")
def test_the_real_collapsed_run_is_rejected(tmp_path, monkeypatch, capsys):
    """The end-to-end calibration: the actual failed run must not pass.

    It is rejected by the *stamp*, and that is the point of keeping the retired
    rule as a named constant rather than treating an unknown string as unknown:
    those rows predate every trip-wire added here, so they carry no
    ``red_positive_advantage_fraction`` to catch them by.  The measurement that
    would have caught the collapse in flight has to be present to be checked, so
    the stamp is the only thing standing between a retired estimator and a
    healthy verdict for every artifact already on disk.
    """
    rows = [json.loads(line) for line in RED250_METRICS.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "rollout" in row][-10:]
    assert rows, "the red250 metrics should hold completed rollouts"
    assert not any("red_positive_advantage_fraction" in row for row in rows), \
        "these rows predate the metric; if that changes, tighten this test"
    code, out = _run(tmp_path, monkeypatch, rows, capsys=capsys)
    assert code == 1, out
    assert "retired" in out
