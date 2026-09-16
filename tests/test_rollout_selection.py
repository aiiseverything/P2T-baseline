"""Whole-prompt retry/skip behavior, including independently padded samples."""
from types import SimpleNamespace

import pytest
import torch

from vpo_rm.alignment import check_response_tokens
from vpo_rm.rollout_selection import select_training_rollout


class Tokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        text = {0: "<eos>", 1: "a", 2: "b", 3: " \t", 4: "\n" * 32,
                5: "\n" * 16, 6: "<end>"}
        return "".join(text.get(i, "x") for i in ids)


def sample(prefixes, answers, reasons, *, prompt_width=None, response_width=None):
    pwidth = prompt_width or max(map(len, prefixes))
    twidth = response_width or max(map(len, answers))
    batch = len(answers)
    inputs = torch.zeros((batch, pwidth + twidth), dtype=torch.long)
    attention = torch.zeros_like(inputs)
    responses = torch.zeros((batch, twidth), dtype=torch.long)
    mask = torch.zeros_like(responses, dtype=torch.bool)
    for i, (prefix, answer) in enumerate(zip(prefixes, answers)):
        inputs[i, pwidth - len(prefix):pwidth] = torch.tensor(prefix)
        attention[i, pwidth - len(prefix):pwidth] = 1
        inputs[i, pwidth:pwidth + len(answer)] = torch.tensor(answer)
        attention[i, pwidth:pwidth + len(answer)] = 1
        responses[i, :len(answer)] = torch.tensor(answer)
        mask[i, :len(answer)] = True
    positions = torch.arange(pwidth, pwidth + twidth).expand(batch, -1).clone()
    rendered = [str(prefix) for prefix in prefixes]
    return inputs, attention, positions, responses, mask, rendered, list(reasons)


class Trainer:
    def __init__(self, *samples):
        self.cfg = SimpleNamespace(group_size=2, max_response_tokens=5,
                                   degenerate_newline_run=32)
        self.actor_tokenizer = Tokenizer()
        self.stop_token_ids = (0, 6)
        self.output_mask = torch.ones(12, dtype=torch.bool)
        self.output_mask[9] = False
        self.samples = list(samples)
        self.calls = []

    def rollout(self, prompts):
        self.calls.append(list(prompts))
        return self.samples.pop(0)


def answers_of(rollout):
    return [row[mask].tolist() for row, mask in zip(rollout[3], rollout[4])]


def prefixes_of(rollout):
    ids, attention, positions, _, mask, _, _ = rollout
    return [ids[i, :positions[i, mask[i]][0]][attention[i, :positions[i, mask[i]][0]].bool()].tolist()
            for i in range(ids.shape[0])]


def test_good_groups_are_returned_without_resampling_or_mutation():
    original = sample([[7], [7], [7, 8], [7, 8]],
                      [[1, 0], [2, 6], [1, 2, 0], [2, 0]], ["stop"] * 4)
    before = [x.clone() if torch.is_tensor(x) else list(x) for x in original]
    trainer = Trainer(original)
    result, prompts, stats = select_training_rollout(trainer, ["short", "long"])
    assert result is original
    assert prompts == ["short", "long"]
    assert trainer.calls == [["short", "long"]]
    assert stats == {"input_prompt_groups": 2, "kept_prompt_groups": 2,
                     "resampled_groups": 0, "skipped_groups": 0,
                     "generated_response_tokens": 9}
    for lhs, rhs in zip(original, before):
        assert torch.equal(lhs, rhs) if torch.is_tensor(lhs) else lhs == rhs


def test_retry_only_bad_group_and_repack_different_prompt_and_response_padding():
    original = sample([[7], [7], [7, 8, 8], [7, 8, 8]],
                      [[1] * 5, [2] * 5, [1, 0], [2, 6]],
                      ["length", "length", "stop", "stop"], prompt_width=5)
    retry = sample([[7], [7]], [[2, 1, 0], [1, 6]], ["stop", "stop"],
                   prompt_width=2, response_width=4)
    trainer = Trainer(original, retry)
    result, prompts, stats = select_training_rollout(trainer, ["short", "long"])
    assert trainer.calls == [["short", "long"], ["short"]]
    assert prompts == ["short", "long"]
    assert answers_of(result) == [[2, 1, 0], [1, 6], [1, 0], [2, 6]]
    assert prefixes_of(result) == [[7], [7], [7, 8, 8], [7, 8, 8]]
    assert result[0].shape == (4, 6)
    assert result[5] == retry[5] + original[5][2:]
    assert stats["generated_response_tokens"] == 19
    assert stats["resampled_groups"] == 1
    assert stats["skipped_groups"] == 0
    check_response_tokens(*result[:5])


def test_degeneracy_removes_both_stop_ids_and_decodes_newline_runs_across_tokens():
    original = sample([[7]] * 4, [[0], [3, 6], [1, 5, 5, 0], [2, 4, 6]], ["stop"] * 4)
    retry = sample([[7]] * 4, [[1, 0], [2, 6], [2, 0], [1, 6]], ["stop"] * 4)
    trainer = Trainer(original, retry)
    result, prompts, stats = select_training_rollout(trainer, ["blank", "newlines"])
    assert trainer.calls == [["blank", "newlines"], ["blank", "newlines"]]
    assert prompts == ["blank", "newlines"]
    assert answers_of(result) == answers_of(retry)
    assert stats["resampled_groups"] == 2
    assert stats["generated_response_tokens"] == 18


def test_mixed_truncated_and_degenerate_has_no_valid_completion_and_retries():
    original = sample([[7], [7]], [[1] * 5, [0]], ["length", "stop"])
    retry = sample([[7], [7]], [[1, 0], [2, 0]], ["stop", "stop"])
    trainer = Trainer(original, retry)
    result, prompts, stats = select_training_rollout(trainer, ["mixed"])
    assert answers_of(result) == answers_of(retry)
    assert prompts == ["mixed"]
    assert stats["resampled_groups"] == 1
    assert stats["generated_response_tokens"] == 10


def test_one_valid_completion_keeps_a_group_containing_a_bad_completion():
    original = sample([[7], [7]], [[1] * 5, [2, 0]], ["length", "stop"])
    trainer = Trainer(original)
    result, prompts, stats = select_training_rollout(trainer, ["mixed"])
    assert result is original
    assert prompts == ["mixed"]
    assert stats["resampled_groups"] == 0


def test_still_bad_groups_are_skipped_whole_and_original_order_is_kept():
    prefixes = [[7], [7], [8], [8], [7, 8], [7, 8]]
    original = sample(prefixes, [[1] * 5, [2] * 5, [1, 0], [2, 0], [0], [3, 0]],
                      ["length", "length", "stop", "stop", "stop", "stop"])
    retry = sample(prefixes[:2] + prefixes[4:], [[0], [3, 0], [2, 0], [1, 6]], ["stop"] * 4)
    trainer = Trainer(original, retry)
    result, prompts, stats = select_training_rollout(trainer, ["skip", "original", "replace"])
    assert trainer.calls[-1] == ["skip", "replace"]
    assert prompts == ["original", "replace"]
    assert answers_of(result) == [[1, 0], [2, 0], [2, 0], [1, 6]]
    assert prefixes_of(result) == [[8], [8], [7, 8], [7, 8]]
    assert stats == {"input_prompt_groups": 3, "kept_prompt_groups": 2,
                     "resampled_groups": 2, "skipped_groups": 1,
                     "generated_response_tokens": 24}


def test_all_skipped_returns_none_without_losing_generated_token_counts():
    original = sample([[7], [7]], [[0], [0]], ["stop", "stop"])
    retry = sample([[7], [7]], [[1] * 5, [2] * 5], ["length", "length"])
    result, prompts, stats = select_training_rollout(Trainer(original, retry), ["bad"])
    assert result is None
    assert prompts == []
    assert stats == {"input_prompt_groups": 1, "kept_prompt_groups": 0,
                     "resampled_groups": 1, "skipped_groups": 1,
                     "generated_response_tokens": 12}


@pytest.mark.parametrize("corruption", ["support", "mapping", "reason", "reason_count",
                                        "rows", "empty", "over_cap", "mask_gap"])
def test_original_samples_are_validated_before_discard_or_retry(corruption):
    original = list(sample([[7], [7]], [[1] * 5, [2] * 5], ["length", "length"]))
    if corruption == "support":
        original[0][0, 1] = original[3][0, 0] = 9
    elif corruption == "mapping":
        original[0][0, 1] = 2
    elif corruption == "reason":
        original[6][0] = "abort"
    elif corruption == "reason_count":
        original[6].pop()
    elif corruption == "rows":
        original = [x[:1] for x in original]
    elif corruption == "empty":
        original[4][0] = False
    elif corruption == "over_cap":
        original = sample([[7], [7]], [[1] * 6, [2] * 6], ["length", "length"])
    elif corruption == "mask_gap":
        original[4][0, 1] = False
    trainer = Trainer(tuple(original))
    with pytest.raises(ValueError):
        select_training_rollout(trainer, ["bad"])
    assert len(trainer.calls) == 1


def test_invalid_retry_cannot_be_hidden_by_skipping_group():
    original = sample([[7], [7]], [[0], [0]], ["stop", "stop"])
    retry = list(sample([[7], [7]], [[0], [0]], ["stop", "stop"]))
    retry[2][0, 0] = 0
    with pytest.raises(ValueError):
        select_training_rollout(Trainer(original, tuple(retry)), ["bad"])


def test_retry_must_keep_same_prompt_token_prefix():
    original = sample([[7], [7]], [[0], [0]], ["stop", "stop"])
    retry = sample([[8], [8]], [[1, 0], [2, 0]], ["stop", "stop"])
    with pytest.raises(ValueError, match="prefix"):
        select_training_rollout(Trainer(original, retry), ["bad"])
