"""The standalone launcher preserves experiment semantics and GPU ownership."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("ssh_launcher", ROOT / "scripts/run_ssh_rl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_gpu_vpo_command_uses_credit_chunks_without_changing_logical_batch(tmp_path):
    m = launcher()
    args = m.build_parser().parse_args(["--arm", "lam4", "--output-dir", str(tmp_path / "run"),
                                       "--gpus", "4,5,6,7", "--sigma0", "2.5"])
    command, environment = m.build_command(args)
    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--method") == "vpo_rm" and value("--credit-lambda") == "4"
    assert value("--credit-microbatch-responses") == "1"
    assert value("--vllm-tensor-parallel-size") == "2"
    assert value("--optimizer-minibatch-responses") == "64"
    assert value("--max-response-tokens") == "2048"
    assert value("--learning-rate") == "5e-5" and value("--beta") == "0.03"
    assert value("--length-reward-sigma0") == "2.5"
    assert "--freeze-stop-tokens" in command and "--freeze-structural" in command
    assert Path(value("--init-adapter")).name == "sft-native-eos-clean2k5e2"


def test_legacy_command_preserves_old_short_reward_and_grpo_freezing(tmp_path):
    m = launcher()
    args = m.build_parser().parse_args(["--arm", "grpo", "--output-dir", str(tmp_path),
                                       "--length-reward-mode", "legacy", "--gpus", "0,1,2"])
    command, _ = m.build_command(args)
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--min-response-tokens") == "64"
    assert value("--length-penalty-slope") == "0.00506"
    assert value("--length-penalty-anchor") == "600"
    assert value("--vllm-tensor-parallel-size") == "1"
    assert "--freeze-stop-tokens" not in command


@pytest.mark.parametrize("gpus", ["0,0,1,2", "0,1", "0,1,2,3,4", "0,,2", ""])
def test_invalid_gpu_assignment_is_rejected_before_launch(tmp_path, gpus):
    m = launcher()
    args = m.build_parser().parse_args(["--arm", "grpo", "--output-dir", str(tmp_path), "--gpus", gpus])
    with pytest.raises(ValueError):
        m.build_command(args)


def test_dry_run_needs_no_models_gpu_or_training_environment(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_ssh_rl.py"), "--arm", "lam2",
                             "--output-dir", str(tmp_path / "unused"), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"CUDA_VISIBLE_DEVICES": "0,1,2,3"' in result.stdout
    assert not (tmp_path / "unused").exists()
