"""Rollout selection and termination validation.

A prompt group whose responses are all truncated or degenerate carries no usable
signal; training on it would also make this arm see a different prompt
population than the GRPO and VPO-RM arms it is compared against.
"""
import pytest
import torch

from p2t.rollout import select_training_rollout, validate_response_termination

GROUP, WIDTH, PAD = 2, 5, 0


def _rollout(prompt_count, length, reason, degenerate=False):
    """A rollout tuple for ``prompt_count`` groups; every row the same shape."""
    count = prompt_count * GROUP
    responses = torch.full((count, WIDTH), PAD, dtype=torch.long)
    mask = torch.zeros((count, WIDTH), dtype=torch.long)
    logprobs = torch.zeros((count, WIDTH))
    for index in range(count):
        responses[index, :length] = torch.arange(1, length + 1)
        if reason == "stop":
            responses[index, length - 1] = 2  # a registered stop id
        mask[index, :length] = 1
        logprobs[index, :length] = -1.0
    prompt_width = 3
    input_ids = torch.cat([torch.ones((count, prompt_width), dtype=torch.long), responses], dim=1)
    full_mask = torch.cat([torch.ones((count, prompt_width), dtype=torch.long), mask], dim=1)
    positions = torch.arange(prompt_width, input_ids.shape[1]).expand(count, -1)
    rendered = [f"p{index // GROUP}" for index in range(count)]
    return (input_ids, full_mask, positions, responses, mask, rendered,
            [reason] * count, logprobs)


def _flags(degenerate_groups=()):
    """A degeneracy flagger sized from its argument, as the real one is."""

    def flag_degenerate(responses, mask):
        rows = responses.shape[0]
        empty = torch.zeros(rows, dtype=torch.bool)
        for group in degenerate_groups:
            empty[group * GROUP:(group + 1) * GROUP] = True
        return empty, torch.zeros(rows, dtype=torch.bool)

    return flag_degenerate


def test_good_groups_pass_through_untouched():
    rollout = _rollout(2, 4, "stop")
    result, prompts, stats = select_training_rollout(
        lambda ps: rollout, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(), device=torch.device("cpu"))
    assert result is not None and prompts == ["a", "b"]
    assert stats["resampled_groups"] == 0 and stats["skipped_groups"] == 0
    assert stats["kept_prompt_groups"] == 2
    torch.testing.assert_close(result[4].sum(-1), rollout[4].sum(-1))


def test_padding_is_trimmed_to_the_longest_survivor():
    rollout = _rollout(2, 4, "stop")
    result, _, _ = select_training_rollout(
        lambda ps: rollout, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(), device=torch.device("cpu"))
    assert result[3].shape == (2 * GROUP, 4)


def test_wholly_bad_group_is_resampled_once_and_kept_when_it_recovers():
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # first pass: everything truncated. retry: everything stops properly.
        return _rollout(len(prompts), 4, "length" if len(calls) == 1 else "stop")

    result, prompts, stats = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(), device=torch.device("cpu"))
    assert calls == [2, 2], "a whole round of groups is resampled together"
    assert result is not None and prompts == ["a", "b"]
    assert stats["resampled_groups"] == 2 and stats["skipped_groups"] == 0


def test_group_that_stays_bad_is_dropped_and_nothing_survives():
    result, prompts, stats = select_training_rollout(
        lambda ps: _rollout(len(ps), 4, "length"), ["a", "b"], group_size=GROUP,
        pad_token_id=PAD, flag_degenerate=_flags(), device=torch.device("cpu"))
    assert result is None, "nothing survived, so the caller must skip the update"
    assert prompts == [] and stats["kept_prompt_groups"] == 0


def test_degenerate_group_is_dropped_but_a_healthy_one_survives():
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        return _rollout(len(prompts), 4, "stop")

    result, prompts, stats = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(degenerate_groups=(0,)), device=torch.device("cpu"))
    assert calls == [2, 1], "only the bad group is resampled"
    assert prompts == ["b"], "group a is degenerate twice over, so it is dropped"
    assert result is not None and result[3].shape[0] == GROUP
    assert stats["resampled_groups"] == 1 and stats["skipped_groups"] == 1


def test_pieces_of_different_widths_merge_into_one_batch():
    """Group 'a' comes back from the retry and group 'b' from the first pass.

    Their rollouts have different response widths, so the merged batch must be
    re-padded to the wider of the two rather than concatenated directly.
    """
    calls = []
    degenerate_group_zero = [True]

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # pass 1 is short; the retry is wider and succeeds
        return _rollout(len(prompts), 2 if len(calls) == 1 else 5, "stop")

    def flag_degenerate(responses, mask):
        rows = responses.shape[0]
        empty = torch.zeros(rows, dtype=torch.bool)
        if degenerate_group_zero[0]:
            empty[:GROUP] = True
            degenerate_group_zero[0] = False  # only the first pass flags group a
        return empty, torch.zeros(rows, dtype=torch.bool)

    result, prompts, stats = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=flag_degenerate, device=torch.device("cpu"))
    assert calls == [2, 1], "only group a is resampled"
    # Survivors of the first pass keep their order; resampled groups follow.
    # Group order inside a rollout does not change any group-relative quantity.
    assert prompts == ["b", "a"]
    assert result is not None
    assert result[3].shape[0] == 2 * GROUP
    assert result[3].shape[1] == 5, "re-padded to the wider piece"
    lengths = result[4].sum(-1)
    torch.testing.assert_close(lengths[:GROUP], torch.full((GROUP,), 2))   # group b, pass 1
    torch.testing.assert_close(lengths[GROUP:], torch.full((GROUP,), 5))   # group a, retry
    # The narrow rows must be zero-padded, not left holding stale tokens.
    torch.testing.assert_close(result[3][:GROUP, 2:], torch.zeros(GROUP, 3, dtype=torch.long))


def test_termination_validation_accepts_consistent_metadata():
    responses = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    validate_response_termination(responses, mask, ["stop", "length"], (2,), 3)


def test_termination_validation_rejects_a_stop_that_did_not_stop():
    responses = torch.tensor([[1, 5, 0]])
    mask = torch.tensor([[1, 1, 0]])
    with pytest.raises(ValueError, match="termination"):
        validate_response_termination(responses, mask, ["stop"], (2,), 3)


def test_termination_validation_rejects_a_length_row_below_the_cap():
    responses = torch.tensor([[1, 5, 0]])
    mask = torch.tensor([[1, 1, 0]])
    with pytest.raises(ValueError, match="termination"):
        validate_response_termination(responses, mask, ["length"], (2,), 3)


def test_termination_validation_rejects_a_padded_hole():
    responses = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 0, 1]])
    with pytest.raises(ValueError, match="right-padded"):
        validate_response_termination(responses, mask, ["stop"], (3,), 3)
