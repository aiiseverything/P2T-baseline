"""Portable GPU topology is checked before loading any model."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import profile_vllm_full as profile
from scripts import vllm_generate_server as server


def test_default_cli_preserves_three_gpu_h200_behavior():
    args = profile.parse_args(["--output-dir", "unused"])
    assert args.vllm_gpu_memory_utilization == .45
    assert args.vllm_tensor_parallel_size == 1
    assert args.credit_microbatch_responses == 0
    assert args.generation_microbatch == 32
    cfg = profile.build_trainer_config(args, "unused")
    assert cfg.allocated_gpu_count == 3
    assert (cfg.prompts_per_rollout, cfg.group_size) == (8, 8)
    assert cfg.optimizer_minibatch_responses == 64
    assert cfg.credit_microbatch_responses == 0
    assert cfg.checkpoint_interval == 100


def test_profile_can_save_only_the_final_training_checkpoint():
    args = profile.parse_args(["--output-dir", "unused", "--checkpoint-interval", "250"])
    assert profile.build_trainer_config(args, "unused").checkpoint_interval == 250
    with pytest.raises(SystemExit):
        profile.parse_args(["--output-dir", "unused", "--checkpoint-interval", "0"])


def test_portable_cli_reaches_trainer_and_real_server_parser():
    args = profile.parse_args(["--output-dir", "unused", "--vllm-gpu-memory-utilization", ".85",
                               "--vllm-tensor-parallel-size", "2", "--credit-microbatch-responses", "1",
                               "--generation-microbatch", "4", "--generation-seed", "19"])
    cfg = profile.build_trainer_config(args, "unused")
    assert cfg.allocated_gpu_count == 4
    assert cfg.credit_microbatch_responses == 1
    assert (cfg.prompts_per_rollout, cfg.group_size, cfg.optimizer_minibatch_responses) == (8, 8, 64)
    argv = profile.vllm_server_command(args, "/tmp/test.sock")
    parsed = server.parse_args(argv[2:])
    assert parsed.gpu_memory_utilization == .85
    assert parsed.tensor_parallel_size == 2
    assert parsed.max_num_seqs == 4
    assert parsed.seed == 19
    assert parsed.socket == "/tmp/test.sock"


@pytest.mark.parametrize("visible,expected", [
    ("0,1,2,3", "2,3"),
    ("5,6,2,7", "2,7"),
    ("GPU-actor,GPU-rm,GPU-gen-a,GPU-gen-b", "GPU-gen-a,GPU-gen-b"),
    (None, "2,3"),
])
def test_child_gpu_mapping_preserves_physical_numbers_and_uuids(visible, expected):
    original = {"KEEP_ME": "yes"}
    if visible is not None:
        original["CUDA_VISIBLE_DEVICES"] = visible
    snapshot = dict(original)
    env = profile.vllm_subprocess_environment(2, 4, original)
    assert env["CUDA_VISIBLE_DEVICES"] == expected
    assert env["KEEP_ME"] == "yes"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert original == snapshot


def test_default_child_mapping_uses_only_third_visible_gpu():
    env = profile.vllm_subprocess_environment(1, 3, {"CUDA_VISIBLE_DEVICES": "3,5,7"})
    assert env["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize("count", [0, 3, 5, 8])
def test_tensor_parallel_requires_exactly_actor_rm_plus_tp_visible_gpus(count):
    with pytest.raises(RuntimeError, match="4 visible GPUs"):
        profile.vllm_subprocess_environment(2, count, {})


def test_gpu_environment_length_must_match_detected_devices():
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        profile.vllm_subprocess_environment(2, 4, {"CUDA_VISIBLE_DEVICES": "0,1,2"})


@pytest.mark.parametrize("flags", [
    ["--vllm-gpu-memory-utilization", "0"],
    ["--vllm-gpu-memory-utilization", "1.1"],
    ["--vllm-gpu-memory-utilization", "nan"],
    ["--vllm-tensor-parallel-size", "0"],
    ["--credit-microbatch-responses", "-1"],
    ["--generation-microbatch", "0"],
])
def test_profile_rejects_invalid_runtime_parameters(flags):
    with pytest.raises(SystemExit):
        profile.parse_args(["--output-dir", "unused", *flags])


@pytest.mark.parametrize("flags", [
    ["--gpu-memory-utilization", "0"], ["--gpu-memory-utilization", "inf"],
    ["--tensor-parallel-size", "0"], ["--max-num-seqs", "0"],
])
def test_server_rejects_invalid_runtime_parameters(flags):
    with pytest.raises(SystemExit):
        server.parse_args(["--model", "unused", "--socket", "/tmp/unused", *flags])


def test_server_default_engine_allocation_is_unchanged():
    args = server.parse_args(["--model", "unused", "--socket", "/tmp/unused"])
    assert args.gpu_memory_utilization == .45
    assert args.tensor_parallel_size == 1
    assert args.max_num_seqs == 32


def test_server_main_forwards_allocation_to_actual_llm_constructor(monkeypatch, tmp_path):
    captured = {}
    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = lambda **kwargs: captured.update(kwargs)
    fake_vllm.SamplingParams = object
    fake_lora = ModuleType("vllm.lora.request")
    fake_lora.LoRARequest = object
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=0))
    fake_transformers.AutoConfig = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(vocab_size=12))
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("PRESENCE_PENALTY", "0")
    monkeypatch.setattr(sys, "argv", ["server", "--model", "test-model", "--socket", str(tmp_path / "socket"),
                                    "--gpu-memory-utilization", ".85", "--tensor-parallel-size", "2",
                                    "--max-num-seqs", "4"])

    class ReachedListener(Exception):
        pass

    def stop_at_listener(*args):
        raise ReachedListener()

    monkeypatch.setattr(server.socket, "socket", stop_at_listener)
    with pytest.raises(ReachedListener):
        server.main()
    assert captured["gpu_memory_utilization"] == .85
    assert captured["tensor_parallel_size"] == 2
    assert captured["max_num_seqs"] == 4
    assert captured["max_model_len"] == 4096
    assert captured["generation_config"] == "vllm"
