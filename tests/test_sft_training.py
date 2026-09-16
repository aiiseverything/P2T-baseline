import math

import pytest

from scripts.sft_init import (
    accumulation_windows,
    build_microbatch_schedule,
    default_sft_output,
    effective_batch_denominator,
    monitor_stop_token_ids,
    prepare_sft_output,
    response_token_weights,
    step_is_due,
)


def test_monitor_uses_shared_registered_stop_token_policy():
    class Tokenizer:
        eos_token_id = 3

        @staticmethod
        def get_vocab():
            return {"<eos>": 3, "<|im_end|>": 4}

    assert monitor_stop_token_ids(Tokenizer()) == (3, 4)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/Qwen3-14B-Base", "models/sft-native-eos-qwen3-14b-base"),
        ("/checkpoints/Qwen3-8B-Base/", "models/sft-native-eos-qwen3-8b-base"),
    ],
)
def test_default_output_names_a_fresh_native_eos_adapter(model, expected):
    assert default_sft_output(model) == expected


@pytest.mark.parametrize(
    "marker",
    ["sft_metrics.jsonl", "sft_manifest.json", "adapter_config.json", "adapter_model.safetensors"],
)
def test_sft_output_rejects_existing_training_artifacts_before_model_load(tmp_path, marker):
    output = tmp_path / "adapter"
    output.mkdir()
    (output / marker).write_text("existing")

    with pytest.raises(FileExistsError, match="fresh output"):
        prepare_sft_output(output)


def test_sft_output_allows_an_empty_directory(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()

    assert prepare_sft_output(output) == output


def test_partial_accumulation_window_produces_the_declared_final_update():
    schedule = build_microbatch_schedule(num_batches=625, epochs=2.0, seed=42)
    windows = accumulation_windows(schedule, grad_accum=8)

    assert len(schedule) == 1250
    assert len(windows) == 157
    assert all(len(window) == 8 for window in windows[:-1])
    assert len(windows[-1]) == 2


def test_fractional_epochs_have_an_explicit_microbatch_count_and_epoch_shuffle():
    schedule = build_microbatch_schedule(num_batches=10, epochs=1.5, seed=7)

    assert len(schedule) == 15
    assert sorted(schedule[:10]) == list(range(10))
    assert schedule[:5] != schedule[10:]


@pytest.mark.parametrize("epochs", [0, -0.5, math.inf, math.nan])
def test_invalid_epoch_counts_fail_fast(epochs):
    with pytest.raises(ValueError, match="epochs"):
        build_microbatch_schedule(num_batches=10, epochs=epochs, seed=1)


def test_weighted_response_tokens_use_one_global_effective_batch_denominator():
    first = [-100, 10, 99]
    second = [-100, 20, 21, 99]

    first_weights = response_token_weights(first, terminator_id=99, eos_weight=3.0)
    second_weights = response_token_weights(second, terminator_id=99, eos_weight=3.0)

    assert first_weights == [0.0, 1.0, 3.0]
    assert second_weights == [0.0, 1.0, 1.0, 3.0]
    assert sum(first_weights) + sum(second_weights) == 9.0


def test_weighted_response_tokens_reject_missing_final_terminator():
    with pytest.raises(ValueError, match="terminator"):
        response_token_weights([-100, 10, 11], terminator_id=99, eos_weight=2.0)


def test_partial_tail_uses_its_own_weighted_token_denominator():
    class Dataset:
        examples = [
            ([1, 2], [-100, 10, 99]),
            ([1, 2], [-100, 20, 21, 99]),
        ]

    denominator = effective_batch_denominator(
        Dataset(), batches=[[0], [1]], batch_indices=[0, 1],
        terminator_id=99, eos_weight=3.0,
    )

    assert denominator == 9.0


def test_final_step_always_emits_metrics_and_monitor_events():
    assert step_is_due(step=157, total_steps=157, every=20)
    assert step_is_due(step=157, total_steps=157, every=100)
    assert not step_is_due(step=156, total_steps=157, every=20)
    assert not step_is_due(step=156, total_steps=157, every=100)
