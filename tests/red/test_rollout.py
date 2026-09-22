"""Rollout selection, and the response-block width the resample path can vary.

A prompt group whose responses are all truncated or degenerate carries no usable
signal, so it is resampled once and dropped if it stays bad.  Resampling is the
only path that merges two *separately generated* rollouts into one batch, and the
two can have different response-block widths: a batch's block is as wide as its
longest response, so a batch whose groups all collapsed to two tokens is two
columns wide while the retry is not.

``_pack`` used to slice the narrow piece by the merged width and index it against
a ``keep`` mask of that same merged width, which is a shape error rather than a
trim.  It surfaced on the real ``red250`` run as

    RuntimeError: The size of tensor a (2) must match the size of tensor b (3)

at rollout 68, after the policy had collapsed to ~2-token responses -- the first
rollout in the whole run where the resample path met two different block widths.
The sibling P2T arm shares ``_pack`` verbatim and carries the same latent bug; it
never fired there only because ``p2t250`` never resampled a single group.
"""
import pytest
import torch

from red.rollout import select_training_rollout, validate_response_termination

GROUP, PAD = 2, 0


def _rollout(prompt_count, length, block, reason="stop"):
    """A rollout tuple whose response *block* is ``block`` columns wide.

    ``block`` is the padded width the batch actually produced, i.e. its longest
    response; ``length`` is how many of those columns carry valid tokens.
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


def _flags(degenerate_groups=(), only_first_pass=False):
    """A degeneracy flagger; ``only_first_pass`` models a group that recovers."""
    state = {"first": True}

    def flag_degenerate(responses, mask):
        rows = responses.shape[0]
        empty = torch.zeros(rows, dtype=torch.bool)
        if only_first_pass and not state["first"]:
            return empty, torch.zeros(rows, dtype=torch.bool)
        state["first"] = False
        for group in degenerate_groups:
            empty[group * GROUP:(group + 1) * GROUP] = True
        return empty, torch.zeros(rows, dtype=torch.bool)

    return flag_degenerate


def _select(rollout_fn, prompts, flag_degenerate, max_response_tokens):
    return select_training_rollout(
        rollout_fn, prompts, group_size=GROUP, pad_token_id=PAD,
        flag_degenerate=flag_degenerate, device=torch.device("cpu"),
        stop_token_ids=(2,), max_response_tokens=max_response_tokens)


def test_good_groups_pass_through_untouched():
    rollout = _rollout(2, 4, block=5)
    result, prompts, stats = _select(lambda ps: rollout, ["a", "b"], _flags(), 5)
    assert result is not None and prompts == ["a", "b"]
    assert stats["resampled_groups"] == 0 and stats["kept_prompt_groups"] == 2
    torch.testing.assert_close(result[4].sum(-1), rollout[4].sum(-1))


def test_a_narrow_piece_merged_with_a_wide_retry_is_padded_not_sliced():
    """The ``red250`` crash: pass 1 collapsed to two tokens, the retry did not.

    Both rows of group ``a`` are flagged in the first pass, so the retry holds
    group ``a`` alone; group ``b`` survives from a two-column block.  The merged
    width is the retry's, so the narrow piece has to be padded up to it.
    """
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        # pass 1: every response is two tokens, so the block is two columns wide
        return _rollout(len(prompts), 2 if len(calls) == 1 else 5,
                        2 if len(calls) == 1 else 5)

    result, prompts, stats = _select(rollout_fn, ["a", "b"], _flags((0,), True), 5)
    assert calls == [2, 1], "only the bad group is resampled"
    assert result is not None, "the merged batch must survive"
    assert result[3].shape[1] == 5, "re-padded to the wider piece"
    assert result[4].shape == result[3].shape, "the mask travels with the responses"
    # Every piece's mask is the merged width, or the concatenation is ragged.
    assert result[4].shape[1] == 5
    lengths = result[4].sum(-1)
    torch.testing.assert_close(lengths[:GROUP], torch.full((GROUP,), 2))   # group b, pass 1
    torch.testing.assert_close(lengths[GROUP:], torch.full((GROUP,), 5))   # group a, retry
    # The narrow rows are zero-padded: no stale token may sit past the mask.
    torch.testing.assert_close(result[3][:GROUP, 2:], torch.zeros(GROUP, 3, dtype=torch.long))


def test_a_narrow_retry_merged_with_a_wide_first_pass_is_padded_not_sliced():
    """The same mismatch with the pieces the other way round.

    Here pass 1 is the wide one and the collapsed group comes back narrow, which
    is the ordering a run reaches *after* the policy collapses: the retry holds
    only the failed groups, and those are the short ones.
    """
    calls = []

    def rollout_fn(prompts):
        calls.append(len(prompts))
        return _rollout(len(prompts), 5 if len(calls) == 1 else 2,
                        5 if len(calls) == 1 else 2)

    result, prompts, stats = _select(rollout_fn, ["a", "b"], _flags((0,), True), 5)
    assert calls == [2, 1]
    assert result is not None
    assert result[3].shape[1] == 5, "the first pass sets the merged width"
    assert result[4].shape == result[3].shape
    lengths = result[4].sum(-1)
    torch.testing.assert_close(lengths[:GROUP], torch.full((GROUP,), 5))   # group b, pass 1
    torch.testing.assert_close(lengths[GROUP:], torch.full((GROUP,), 2))   # group a, retry
    torch.testing.assert_close(result[3][GROUP:, 2:], torch.zeros(GROUP, 3, dtype=torch.long))


def test_an_uneven_block_still_reports_the_widest_row_it_carries():
    """A block wider than every row's mask is ordinary: vLLM pads to the batch's
    longest response, and the pack trims that down to what the rows actually
    need rather than carrying the slack into the update."""
    rollout = _rollout(1, 3, block=5)
    result, _, _ = _select(lambda ps: rollout, ["a"], _flags(), 5)
    assert result[3].shape[1] == 3, "trimmed to the longest surviving row"
    torch.testing.assert_close(result[4].sum(-1), torch.full((GROUP,), 3))


def test_group_that_stays_bad_is_dropped_and_nothing_survives():
    result, prompts, stats = _select(lambda ps: _rollout(len(ps), 5, 5, "length"),
                                     ["a", "b"], _flags(), 5)
    assert result is None, "nothing survived, so the caller must skip the update"
    assert prompts == [] and stats["kept_prompt_groups"] == 0


def test_source_termination_is_validated_before_anything_is_dropped():
    """Packing re-densifies the mask, so a bad source must be rejected first."""
    rollout = _rollout(1, 4, block=5, reason="length")  # truncated: below the cap

    def rollout_fn(prompts):
        input_ids, full_mask, positions, responses, mask, rendered, reasons, logprobs = rollout
        return (input_ids, full_mask, positions, responses, mask, rendered,
                ["stop"] * responses.shape[0], logprobs)

    with pytest.raises(ValueError, match="termination"):
        _select(rollout_fn, ["a"], _flags(), 5)


def test_termination_validation_rejects_a_padded_hole():
    responses = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 0, 1]])
    with pytest.raises(ValueError, match="right-padded"):
        validate_response_termination(responses, mask, ["stop"], (3,), 3)
