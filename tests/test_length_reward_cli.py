import argparse
import hashlib
import json
from unittest.mock import patch

import pytest

from vpo_rm.length_reward_cli import add_length_reward_args, length_reward_config_kwargs


def parse(*argv):
    parser = argparse.ArgumentParser()
    add_length_reward_args(parser)
    return length_reward_config_kwargs(parser.parse_args(argv))


@pytest.fixture
def isolated_benchmarks(tmp_path, monkeypatch):
    from vpo_rm import data

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"instruction": "held-out benchmark prompt"}) + "\n")
    monkeypatch.setattr(data, "DEFAULT_BENCHMARK_PATHS", {"synthetic": benchmark})


def test_length_cli_defaults_preserve_legacy_and_defer_minimum_resolution():
    assert parse() == {
        "length_reward_mode": "legacy",
        "length_reward_sigma0": None,
        "short_response_threshold": 8,
        "long_response_threshold": 1024,
        "short_penalty_strength": .5,
        "long_penalty_strength": 2.,
        "advantage_std_floor_fraction": .5,
        "length_calibration_prompts": 128,
        "degenerate_newline_run": 32,
        "min_response_tokens": None,
        "length_penalty_slope": 0.,
        "length_penalty_anchor": 600,
    }


def test_length_cli_passes_explicit_shared_sigma_and_all_soft_controls():
    assert parse("--length-reward-mode", "soft", "--length-reward-sigma0", "3.25",
                 "--short-response-threshold", "7", "--long-response-threshold", "1280",
                 "--short-penalty-strength", ".7", "--long-penalty-strength", "1.4",
                 "--advantage-std-floor-fraction", ".3",
                 "--length-calibration-prompts", "64", "--degenerate-newline-run", "40",
                 "--min-response-tokens", "0") == {
        "length_reward_mode": "soft",
        "length_reward_sigma0": 3.25,
        "short_response_threshold": 7,
        "long_response_threshold": 1280,
        "short_penalty_strength": .7,
        "long_penalty_strength": 1.4,
        "advantage_std_floor_fraction": .3,
        "length_calibration_prompts": 64,
        "degenerate_newline_run": 40,
        "min_response_tokens": 0,
        "length_penalty_slope": 0.,
        "length_penalty_anchor": 600,
    }


def test_length_cli_keeps_legacy_slope_and_anchor_options():
    values = parse("--length-penalty-slope", ".00506", "--length-penalty-anchor", "600")
    assert values["length_penalty_slope"] == .00506
    assert values["length_penalty_anchor"] == 600


def test_train_skywork_passes_init_adapter_and_soft_config_before_model_load(tmp_path, isolated_benchmarks):
    from scripts import train_skywork

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\nsecond prompt\n")
    output = tmp_path / "run"
    captured = {}

    def stop_at_model_load(config):
        captured["config"] = config
        raise RuntimeError("reached model loading")

    with patch.object(train_skywork.VPOTrainer, "from_pretrained", side_effect=stop_at_model_load):
        with pytest.raises(RuntimeError, match="reached model loading"):
            train_skywork.main(["--prompts-file", str(prompts), "--output-dir", str(output),
                                "--length-reward-mode", "soft", "--length-reward-sigma0", "2.5",
                                "--init-adapter", "models/sft-native-eos-clean2k5e2"])
    config = captured["config"]
    assert config.length_reward_mode == "soft"
    assert config.length_reward_sigma0 == 2.5
    assert config.min_response_tokens == 0
    assert config.init_adapter == "models/sft-native-eos-clean2k5e2"


def test_trainer_module_cli_forwards_soft_mode_and_init_adapter(tmp_path, isolated_benchmarks):
    from vpo_rm import trainer

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\nsecond prompt\n")
    captured = {}

    def stop_at_model_load(config):
        captured["config"] = config
        raise RuntimeError("reached model loading")

    with patch.object(trainer.VPOTrainer, "from_pretrained", side_effect=stop_at_model_load):
        with pytest.raises(RuntimeError, match="reached model loading"):
            trainer.main(["--prompts-file", str(prompts), "--output-dir", str(tmp_path / "run"),
                          "--length-reward-mode", "soft", "--length-reward-sigma0", "2.5",
                          "--init-adapter", "models/sft-native-eos-clean2k5e2"])
    config = captured["config"]
    assert config.length_reward_mode == "soft"
    assert config.length_reward_sigma0 == 2.5
    assert config.init_adapter == "models/sft-native-eos-clean2k5e2"


def test_profile_manifest_records_calibrated_trainer_config_not_precalibration_copy(tmp_path):
    from scripts.profile_vllm_full import write_profile_calibration_manifest
    from vpo_rm.trainer import TrainerConfig

    manifest_path = tmp_path / "profile_manifest.json"
    manifest_path.write_text(json.dumps({"config": {"length_reward_sigma0": None}, "other": "kept"}))
    calibration_path = tmp_path / "length_reward_calibration.json"
    calibration_path.write_text(json.dumps({"sigma0": 2.5, "source": "initial_policy"}))
    actual_trainer_config = TrainerConfig(length_reward_mode="soft",
                                          length_reward_sigma0=2.5).resolved()

    write_profile_calibration_manifest(manifest_path, calibration_path, actual_trainer_config)

    saved = json.loads(manifest_path.read_text())
    assert saved["config"]["length_reward_sigma0"] == 2.5
    assert saved["other"] == "kept"
    assert saved["length_reward_calibration"]["data"] == {
        "sigma0": 2.5, "source": "initial_policy"}
    assert saved["length_reward_calibration"]["sha256"] == hashlib.sha256(
        calibration_path.read_bytes()).hexdigest()
