"""Whole-prompt retry/skip behavior, including independently padded samples."""
from types import SimpleNamespace

import pytest
import torch

from vpo_rm.alignment import check_response_tokens
from vpo_rm.rollout_selection import select_training_rollout, validate_rollout_logprobs


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
    ids, attention, positions, _, mask, _, _ = rollout[:7]
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


@pytest.mark.parametrize('answer,reason', [
    ([1, 2], 'stop'),
    ([1, 0, 2, 0], 'stop'),
    ([1, 2], 'length'),
    ([1, 2, 1, 2, 0], 'length'),
])
def test_inconsistent_termination_is_rejected_before_selecting_or_retrying(answer, reason):
    original = sample([[7], [7]], [answer, [2, 0]], [reason, 'stop'])
    trainer = Trainer(original)
    with pytest.raises(ValueError, match='stop|EOS|length|termination'):
        select_training_rollout(trainer, ['bad'])
    assert trainer.calls == [['bad']]


def attach_logp(rollout, values):
    logp = torch.full(rollout[3].shape, float('nan'))
    for row, entries in zip(logp, values):
        row[:len(entries)] = torch.tensor(entries)
    return (*rollout, logp)


def test_rollout_logp_tracks_each_response_through_retry_drop_and_repadding():
    prefixes = [[7], [7], [8], [8], [7, 8], [7, 8]]
    original = attach_logp(sample(prefixes,
        [[1] * 5, [2] * 5, [1, 0], [2, 1, 0], [0], [3, 0]],
        ['length', 'length', 'stop', 'stop', 'stop', 'stop'],
        prompt_width=5, response_width=7),
        [[-.1] * 5, [-.2] * 5, [-.31, -.32], [-.41, -.42, -.43], [-.5], [-.6, -.61]])
    retry = attach_logp(sample(prefixes[:2] + prefixes[4:],
        [[0], [3, 0], [2, 1, 2, 0], [1, 6]], ['stop'] * 4,
        prompt_width=3, response_width=6),
        [[-.7], [-.8, -.81], [-.91, -.92, -.93, -.94], [-1.01, -1.02]])
    trainer = Trainer(original, retry)
    trainer.cfg.rollout_importance_correction = True
    # A global latest-rollout side channel must not override the original good group.
    trainer.last_rollout_logp = torch.full_like(retry[7], -99.)
    result, prompts, stats = select_training_rollout(trainer, ['drop', 'keep', 'replace'])
    assert prompts == ['keep', 'replace']
    assert trainer.calls == [['drop', 'keep', 'replace'], ['drop', 'replace']]
    assert len(result) == 8 and result[7].shape == result[3].shape == (4, 4)
    assert answers_of(result) == [[1, 0], [2, 1, 0], [2, 1, 2, 0], [1, 6]]
    torch.testing.assert_close(result[7], torch.tensor([
        [-.31, -.32, 0, 0], [-.41, -.42, -.43, 0],
        [-.91, -.92, -.93, -.94], [-1.01, -1.02, 0, 0]]))
    assert stats['skipped_groups'] == 1
    check_response_tokens(*result[:5])


def test_valid_eighth_field_returns_unmodified_and_ignores_padding_nan():
    original = attach_logp(sample([[7], [7]], [[1, 0], [2, 1, 6]], ['stop', 'stop']),
                           [[0., -.1], [-.2, -.3, -.4]])
    trainer = Trainer(original)
    trainer.cfg.rollout_importance_correction = True
    result, _, _ = select_training_rollout(trainer, ['good'])
    assert result is original
    assert torch.isnan(result[7][0, 2])


@pytest.mark.parametrize('corruption', ['missing', 'none', 'shape', 'integer', 'bool', 'complex',
                                       'not_tensor', 'device', 'nan', 'inf', 'positive'])
def test_rollout_logp_is_validated_before_bad_group_is_discarded(corruption):
    original = attach_logp(sample([[7], [7]], [[0], [0]], ['stop', 'stop']), [[-.1], [-.2]])
    values = list(original)
    if corruption == 'missing':
        values.pop()
    elif corruption == 'none':
        values[7] = None
    elif corruption == 'shape':
        values[7] = torch.zeros(2, 2)
    elif corruption in ('integer', 'bool', 'complex'):
        values[7] = values[7].to({'integer': torch.long, 'bool': torch.bool, 'complex': torch.complex64}[corruption])
    elif corruption == 'not_tensor':
        values[7] = [[-.1], [-.2]]
    elif corruption == 'device':
        values[7] = torch.empty((2, 1), device='meta')
    else:
        values[7][0, 0] = {'nan': float('nan'), 'inf': -float('inf'), 'positive': .01}[corruption]
    trainer = Trainer(tuple(values))
    trainer.cfg.rollout_importance_correction = True
    with pytest.raises(ValueError, match='logp|probabilit'):
        select_training_rollout(trainer, ['bad'])
    assert trainer.calls == [['bad']]


def test_retry_requires_its_own_logp_when_correction_is_enabled():
    original = attach_logp(sample([[7], [7]], [[0], [0]], ['stop', 'stop']), [[-.1], [-.2]])
    retry = sample([[7], [7]], [[1, 0], [2, 0]], ['stop', 'stop'])
    trainer = Trainer(original, retry)
    trainer.cfg.rollout_importance_correction = True
    with pytest.raises(ValueError, match='logp|probabilit'):
        select_training_rollout(trainer, ['retry'])
    assert trainer.calls == [['retry'], ['retry']]


def test_optional_none_logp_is_supported_without_correction():
    original = (*sample([[7], [7]], [[0], [0]], ['stop', 'stop']), None)
    retry = (*sample([[7], [7]], [[1, 0], [2, 0]], ['stop', 'stop']), None)
    result, _, _ = select_training_rollout(Trainer(original, retry), ['retry'])
    assert len(result) == 8 and result[7] is None


def test_mixed_present_and_missing_logp_cannot_be_silently_repacked():
    original = sample([[7], [7], [8], [8]], [[1, 0], [2, 0], [0], [0]], ['stop'] * 4)
    retry = attach_logp(sample([[8], [8]], [[1, 0], [2, 0]], ['stop', 'stop']),
                        [[-.1, -.2], [-.3, -.4]])
    with pytest.raises(ValueError, match='logp|probabilit'):
        select_training_rollout(Trainer(original, retry), ['keep', 'replace'])


def test_public_validator_checks_provided_values_even_when_not_required():
    original = sample([[7], [7]], [[1, 0], [2, 0]], ['stop', 'stop'])
    assert validate_rollout_logprobs(original) is None
    assert validate_rollout_logprobs((*original, None)) is None
    with pytest.raises(ValueError, match='finite and nonpositive'):
        validate_rollout_logprobs((*original, torch.ones_like(original[3], dtype=torch.float)))
    provided = attach_logp(original, [[-.1, -.2], [-.3, -.4]])
    assert validate_rollout_logprobs(provided) is provided[7]


def test_repacking_promotes_logp_dtype_without_rounding_original_probabilities():
    original = attach_logp(sample([[7], [7], [8], [8]],
        [[1, 0], [2, 0], [0], [0]], ['stop'] * 4),
        [[-.1, -.2], [-.3, -.4], [-.5], [-.6]])
    original = (*original[:7], original[7].double())
    original[7][0, 0] = -.1234567890123
    retry = attach_logp(sample([[8], [8]], [[1, 0], [2, 0]], ['stop', 'stop']),
                        [[-.7, -.8], [-.9, -1.]])
    result, _, _ = select_training_rollout(Trainer(original, retry), ['keep', 'replace'])
    assert result[7].dtype == torch.float64
    assert result[7][0, 0] == original[7][0, 0]
    torch.testing.assert_close(result[7][2:], retry[7].double())
