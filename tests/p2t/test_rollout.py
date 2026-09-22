"""Rollout selection and termination validation.

A prompt group whose responses are all truncated or degenerate carries no usable
signal; training on it would also make this arm see a different prompt
population than the GRPO and VPO-RM arms it is compared against.
"""
import pytest
import torch

from p2t.rollout import select_training_rollout, validate_response_termination

GROUP, WIDTH, PAD = 2, 5, 0


def _rollout(prompt_count, length, reason, degenerate=False, block=WIDTH):
    """A rollout tuple for ``prompt_count`` groups; every row the same shape.

    ``block`` is the response block's padded width, i.e. the batch's longest
    response.  It is normally ``WIDTH``, but a batch whose responses all collapsed
    to a couple of tokens produces a correspondingly narrow block, and that is the
    case the resample path has to survive.
    """
    count = prompt_count * GROUP
    responses = torch.full((count, block), PAD, dtype=torch.long)
    mask = torch.zeros((count, block), dtype=torch.long)
    logprobs = torch.zeros((count, block))
    for index in range(count):
        # Values start above the stop id so the terminal marker is the only one.
        responses[index, :length] = torch.arange(3, length + 3)
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
        flag_degenerate=_flags(), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert result is not None and prompts == ["a", "b"]
    assert stats["resampled_groups"] == 0 and stats["skipped_groups"] == 0
    assert stats["kept_prompt_groups"] == 2
    torch.testing.assert_close(result[4].sum(-1), rollout[4].sum(-1))


def test_padding_is_trimmed_to_the_longest_survivor():
    rollout = _rollout(2, 4, "stop")
    result, _, _ = select_training_rollout(
        lambda ps: rollout, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert result[3].shape == (2 * GROUP, 4)


def test_wholly_bad_group_is_resampled_once_and_kept_when_it_recovers():
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # first pass: everything truncated. retry: everything stops properly.
        return _rollout(len(prompts), WIDTH if len(calls) == 1 else 4,
                        "length" if len(calls) == 1 else "stop")

    result, prompts, stats = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert calls == [2, 2], "a whole round of groups is resampled together"
    assert result is not None and prompts == ["a", "b"]
    assert stats["resampled_groups"] == 2 and stats["skipped_groups"] == 0


def test_group_that_stays_bad_is_dropped_and_nothing_survives():
    result, prompts, stats = select_training_rollout(
        lambda ps: _rollout(len(ps), WIDTH, "length"), ["a", "b"], group_size=GROUP,
        pad_token_id=PAD, flag_degenerate=_flags(), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert result is None, "nothing survived, so the caller must skip the update"
    assert prompts == [] and stats["kept_prompt_groups"] == 0


def test_degenerate_group_is_dropped_but_a_healthy_one_survives():
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        return _rollout(len(prompts), 4, "stop")

    result, prompts, stats = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=_flags(degenerate_groups=(0,)), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
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
        flag_degenerate=flag_degenerate, device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
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


def test_a_narrow_response_block_merged_with_a_wider_piece_is_padded_not_sliced():
    """The piece's *block* width, not its row lengths, is what has to fit.

    ``test_pieces_of_different_widths_merge_into_one_batch`` varies the valid
    lengths but both pieces keep a five-column block, so the merge never has to
    pad a piece whose own block is narrower than the merged width.  A collapsing
    policy does produce such a piece -- a batch of two-token responses is two
    columns wide -- and then ``_pack`` used to index a two-column slice against a
    five-column ``keep`` mask and raise

        The size of tensor a (2) must match the size of tensor b (5)

    This arm shares ``_pack`` with the RED arm verbatim; the bug fired on that
    arm's red250 run at rollout 68.  It is latent here, not absent.
    """
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # pass 1 collapsed to a two-column block; the retry did not
        return _rollout(len(prompts), 2 if len(calls) == 1 else WIDTH, "stop",
                        block=2 if len(calls) == 1 else WIDTH)

    def flag_degenerate(responses, mask):
        rows = responses.shape[0]
        empty = torch.zeros(rows, dtype=torch.bool)
        if len(calls) == 1:
            empty[:GROUP] = True
        return empty, torch.zeros(rows, dtype=torch.bool)

    result, _, _ = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=flag_degenerate, device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert calls == [2, 1]
    assert result is not None, "the merged batch must survive"
    assert result[3].shape[1] == WIDTH, "re-padded to the wider piece"
    assert result[4].shape == result[3].shape, "the mask travels with the responses"
    lengths = result[4].sum(-1)
    torch.testing.assert_close(lengths[:GROUP], torch.full((GROUP,), 2))          # pass 1
    torch.testing.assert_close(lengths[GROUP:], torch.full((GROUP,), WIDTH))      # retry
    # The narrow rows are zero-padded, so no stale token sits past the mask.
    torch.testing.assert_close(result[3][:GROUP, 2:], torch.zeros(GROUP, WIDTH - 2,
                                                                 dtype=torch.long))


def test_retry_batch_with_a_narrower_prompt_block_still_merges():
    """A retry batch holds fewer prompts, so its chat padding is narrower.

    The pieces must be re-padded to a common prompt width before concatenation,
    otherwise the correct-looking single-pass path hides a crash that only the
    resample path reaches.
    """
    calls = []
    degenerate_first = [True]

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # the retry is built from a single prompt, so its prompt block is narrower
        base = _rollout(len(prompts), 4, "stop")
        if len(calls) > 1:
            input_ids, full_mask, positions, responses, mask, rendered, reasons, logprobs = base
            narrower = input_ids[:, 1:]
            base = (narrower, full_mask[:, 1:], positions - 1, responses, mask,
                    rendered, reasons, logprobs)
        return base

    def flag_degenerate(responses, mask):
        rows = responses.shape[0]
        empty = torch.zeros(rows, dtype=torch.bool)
        if degenerate_first[0]:
            empty[:GROUP] = True
            degenerate_first[0] = False
        return empty, torch.zeros(rows, dtype=torch.bool)

    result, prompts, _ = select_training_rollout(
        rollout_fn, ["a", "b"], group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=flag_degenerate, device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)
    assert calls == [2, 1]
    assert result is not None
    assert result[0].shape[0] == 2 * GROUP
    assert result[1].shape == result[0].shape
    # positions index the response columns only
    assert result[2].shape == result[3].shape
    # and must stay anchored at the common prompt width after the re-pad
    prompt_width = result[0].shape[1] - result[3].shape[1]
    torch.testing.assert_close(result[2][:, 0], torch.full((2 * GROUP,), prompt_width,
                                                           dtype=result[2].dtype))
    # the narrow piece's prompt block was left-padded, so its mask starts at zero
    assert (result[1][:, 0] == 0).any(), "left padding must be unattended"


def test_source_termination_is_validated_before_anything_is_dropped():
    """Packing re-densifies the mask, so a bad source must be rejected first."""
    rollout = _rollout(1, 4, "length")  # truncated rows must fill the cap

    def rollout_fn(prompts):
        input_ids, full_mask, positions, responses, mask, rendered, reasons, logprobs = rollout
        return (input_ids, full_mask, positions, responses, mask, rendered,
                ["stop"] * responses.shape[0], logprobs)

    with pytest.raises(ValueError, match="termination"):
        select_training_rollout(rollout_fn, ["a"], group_size=GROUP, pad_token_id=PAD,
                                flag_degenerate=_flags(), device=torch.device("cpu"),
                                stop_token_ids=(2,), max_response_tokens=5)


def test_rollout_fn_must_return_the_bare_tuple():
    """The trainer's `rollout` returns a pair; the selector takes the tuple.

    Passing the pair through would index a summary dict as if it were a rollout,
    which is exactly the wiring mistake this asserts against.
    """
    rollout = _rollout(1, 4, "stop")
    with pytest.raises((IndexError, TypeError, KeyError)):
        select_training_rollout(lambda ps: (rollout, {"summary": 1}), ["a"],
                                group_size=GROUP, pad_token_id=PAD,
                                flag_degenerate=_flags(), device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=WIDTH)


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
