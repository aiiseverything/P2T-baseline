"""Execute launcher control flow with external Python/bash work replaced by recorders."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    trace = tmp_path / "trace.jsonl"
    env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
               PROJECT_ROOT=str(project), SHELL_TEST_ROOT=str(project),
               SHELL_TEST_TRACE=str(trace), JOB_ID="launcher-test")
    # The pre-fix launchers hard-code this checkout. Redirect only that cd to
    # the temporary project; all shell branching and argument expansion is real.
    bash_env = tmp_path / "bash_env"
    bash_env.write_text('cd() { if [[ "${1:-}" = "' + str(ROOT) + '" ]]; then '
                        'builtin cd "$SHELL_TEST_ROOT"; else builtin cd "$@"; fi; }\n')
    env["BASH_ENV"] = str(bash_env)
    recorder = f'''#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
entry = pathlib.Path(args[0]).name if args else ""
keys = ["MODEL", "RM", "INIT_ADAPTER", "EVAL_MODEL", "EVAL_RM", "IFEVAL_MODEL",
        "CALIB_INIT", "SFT_MODEL", "SFT_OUTPUT", "SFT_RESPONSE_EOS", "TVT_MODEL"]
with open(os.environ["SHELL_TEST_TRACE"], "a") as out:
    out.write(json.dumps({{"command": pathlib.Path(sys.argv[0]).name, "args": args,
                         "env": {{k: os.environ.get(k) for k in keys}}}}) + "\\n")
if entry == os.environ.get("STUB_FAIL_SCRIPT"):
    raise SystemExit(17)
if entry == "run_skywork_500.sh":
    out = pathlib.Path(os.environ["SHELL_TEST_ROOT"]) / ("runs/formal-skywork-" + args[1] + "-" + os.environ["JOB_ID"])
    (out / "vllm-adapters/step-50").mkdir(parents=True, exist_ok=True)
    (out / "profile_summary.json").write_text("{{}}")
    (out / "vllm-adapters/step-50/adapter_config.json").write_text("{{}}")
'''
    for name in ("bash", "python3"):
        executable = commands / name
        executable.write_text(recorder)
        executable.chmod(0o755)

    def run(script, overrides=None, args=()):
        trace.write_text("")
        result = subprocess.run(["/bin/bash", str(ROOT / "scripts" / script), *args],
                                env=env | (overrides or {}), capture_output=True, text=True,
                                timeout=15)
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        return result, records

    run.project = project
    run.commands = commands
    return run


def called(records, entry):
    return [r for r in records if r["args"] and Path(r["args"][0]).name == entry]


def option(record, name):
    args = record["args"]
    return args[args.index(name) + 1] if name in args else None


def chain_env(project):
    for model in ("policy-8b", "reward-4b", "init-8b"):
        (project / model).mkdir(exist_ok=True)
        filename = "adapter_config.json" if model == "init-8b" else "config.json"
        (project / model / filename).write_text("{}")
    return dict(CHAIN_NAME="arm-8b", CHAIN_METHOD="grpo",
                CHAIN_MODEL=str(project / "policy-8b"), CHAIN_RM=str(project / "reward-4b"),
                CHAIN_INIT=str(project / "init-8b"))


def test_chain_propagates_actor_and_reward_models_to_evaluation(launcher):
    config = chain_env(launcher.project)
    result, records = launcher("run_train_eval_chain.sh", config)
    assert result.returncode == 0, result.stderr
    evaluation = called(records, "run_arm_eval_combo.sh")[0]
    assert evaluation["env"]["EVAL_MODEL"] == config["CHAIN_MODEL"]
    assert evaluation["env"]["IFEVAL_MODEL"] == config["CHAIN_MODEL"]
    assert evaluation["env"]["EVAL_RM"] == config["CHAIN_RM"]


@pytest.mark.parametrize("stage", ["run_skywork_500.sh", "run_arm_eval_combo.sh"])
def test_chain_propagates_failures_even_with_stale_summary(launcher, stage):
    config = chain_env(launcher.project)
    stale = launcher.project / "runs/formal-skywork-grpo-launcher-test"
    stale.mkdir(parents=True)
    (stale / "profile_summary.json").write_text("{}")
    result, records = launcher("run_train_eval_chain.sh", config | {"STUB_FAIL_SCRIPT": stage})
    assert result.returncode == 17
    if stage == "run_skywork_500.sh":
        assert not called(records, "run_arm_eval_combo.sh")


def test_rm_eval_expands_semicolon_runs_and_preserves_model_arguments(launcher):
    result, records = launcher("run_eval.sh", {"EVAL_RUNS": "a=/run one;b=/run two",
                              "EVAL_OUT": "/output", "EVAL_MODEL": "/policy 8b", "EVAL_RM": "/reward 4b"})
    assert result.returncode == 0, result.stderr
    invocation = called(records, "eval_checkpoints.py")[0]
    args = invocation["args"]
    assert [args[i + 1] for i, value in enumerate(args) if value == "--run"] == ["a=/run one", "b=/run two"]
    assert option(invocation, "--model") == "/policy 8b"
    assert option(invocation, "--rm") == "/reward 4b"


@pytest.mark.parametrize("benchmark", ["ifeval", "gsm8k", "alpaca"])
def test_benchmark_launchers_forward_the_selected_actor(launcher, benchmark):
    prefix = benchmark.upper()
    result, records = launcher(f"run_{benchmark}.sh", {f"{prefix}_OUT": "/output",
                               f"{prefix}_ADAPTERS": "init=/adapter", f"{prefix}_MODEL": "/policy-8b"})
    assert result.returncode == 0, result.stderr
    invocation = [r for r in called(records, f"eval_{benchmark}.py") if "--selftest" not in r["args"]][0]
    assert option(invocation, "--model") == "/policy-8b"


def test_calibration_passes_the_requested_init_adapter(launcher):
    config = chain_env(launcher.project)
    env = dict(CALIB_MODEL=config["CHAIN_MODEL"], CALIB_RM=config["CHAIN_RM"],
               CALIB_INIT=config["CHAIN_INIT"], CALIB_OUT=str(launcher.project / "calibration"))
    result, records = launcher("run_rm_calib.sh", env)
    assert result.returncode == 0, result.stderr
    assert called(records, "run_skywork_500.sh")[0]["env"]["INIT_ADAPTER"] == env["CALIB_INIT"]


def test_default_training_init_is_native_14b(launcher):
    result, records = launcher("run_skywork_500.sh", args=("grpo",))
    assert result.returncode == 0, result.stderr
    assert option(called(records, "profile_vllm_full.py")[0], "--init-adapter") == "models/sft-native-eos-clean2k5e2"


def test_non_14b_training_requires_an_explicit_matching_init(launcher):
    result, records = launcher("run_skywork_500.sh", {"MODEL": "models/Qwen3-8B-Base"}, args=("grpo",))
    assert result.returncode != 0
    assert not called(records, "profile_vllm_full.py")


@pytest.mark.parametrize("script", ["run_sft_init.sh", "run_sft_v2.sh"])
def test_sft_smoke_honors_actor_and_uses_native_eos(launcher, script):
    result, records = launcher(script, {"SFT_MODE": "smoke", "SFT_MODEL": "models/Qwen3-8B-Base"})
    assert result.returncode == 0, result.stderr
    invocation = called(records, "sft_init.py")[0]
    assert option(invocation, "--model") == "models/Qwen3-8B-Base"
    assert option(invocation, "--response-eos") == "native"


def test_train_vs_test_forwards_the_sft_base_model(launcher):
    result, records = launcher("run_train_vs_test.sh", {"TVT_ADAPTER": "/adapter-8b",
                               "TVT_MODEL": "/policy-8b", "TVT_OUT": "/output"})
    assert result.returncode == 0, result.stderr
    evaluations = called(records, "eval_alpaca.py")
    assert len(evaluations) == 2
    assert all(option(record, "--model") == "/policy-8b" for record in evaluations)


def test_matrix_does_not_reuse_old_non_native_sft_job(launcher):
    scheduler = launcher.commands / "rjob"
    state = launcher.project / "scheduler-state.json"
    scheduler.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
state = pathlib.Path({str(state)!r})
names = json.loads(state.read_text()) if state.exists() else ["sft-init-8b"]
args = sys.argv[1:]
if args[0] == "list":
    print("\\n".join("showname=" + name + ")" for name in names))
elif args[0] == "submit":
    names.append(args[args.index("--name") + 1]); state.write_text(json.dumps(names))
    with open(os.environ["SHELL_TEST_TRACE"], "a") as out:
        out.write(json.dumps({{"command": "rjob", "args": args}}) + "\\n")
    print("stubbed submission")
else:
    raise SystemExit(2)
''')
    scheduler.chmod(0o755)
    result, records = launcher("submit_matrix_batch.sh")
    assert result.returncode == 0, result.stderr
    sft_jobs = [r for r in records if r["command"] == "rjob"
                and any(arg.startswith("SFT_OUTPUT=") for arg in r["args"])]
    assert len(sft_jobs) == 1, "An old non-native SFT job must not suppress the new native prerequisite"
    assert any(arg.endswith("/models/sft-native-eos-qwen3-8b-base") for arg in sft_jobs[0]["args"])
