"""Cross-checks against the parent project's own implementations.

The RED package is standalone -- it never imports ``vpo_rm`` at runtime -- but a
baseline is only useful if it shares the project's protocol.  What RED must share
with its siblings is deliberately narrower than what P2T shares: the byte-exact
actor-to-reward-model token mapping, the soft length window, the degeneracy flags
and the reward-scale calibration.  The surrogate and the standardisation are
*not* shared, because RED does not standardise and does not clip; those
divergences are asserted absent here so they cannot creep back in unnoticed.
"""
from pathlib import Path

import pytest
import torch

vpo_rm = pytest.importorskip("vpo_rm", reason="parent project not importable from this checkout")


def test_reward_input_mapping_matches_the_project():
    """Same actor->RM token attribution as the VPO-RM arms, on a real tokenizer."""
    pytest.importorskip("transformers", reason="transformers is not installed here")
    model_dir = Path(__file__).resolve().parents[2] / "models/Qwen3-14B-Base"
    if not model_dir.is_dir():
        pytest.skip(f"local actor tokenizer not present at {model_dir}")
    from transformers import AutoTokenizer
    from vpo_rm.reward_inputs import build_reward_input as reference
    from red.mapping import build_reward_input as ours

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = "Explain why the sky is blue, briefly."
    response_ids = tokenizer("Because short wavelengths scatter more.",
                             add_special_tokens=False)["input_ids"]
    response_ids = response_ids + [tokenizer.eos_token_id]
    first, second = ours(tokenizer, tokenizer, prompt, response_ids), \
        reference(tokenizer, tokenizer, prompt, response_ids)
    assert first.input_ids == second.input_ids
    assert first.response_positions == second.response_positions
    assert first.response_text == second.response_text


def test_soft_length_window_matches_the_project():
    from vpo_rm.length_reward import soft_length_penalties as reference
    from red.length_reward import soft_length_penalties as ours
    lengths = torch.tensor([1, 5, 8, 600, 1024, 1500, 2048], dtype=torch.float32)
    kwargs = dict(short_threshold=8, long_threshold=1024, max_length=2048,
                  short_strength=.5, long_strength=2.)
    ours_short, ours_long = ours(lengths, 3.0, **kwargs)
    theirs_short, theirs_long = reference(lengths, 3.0, **kwargs)
    torch.testing.assert_close(ours_short, theirs_short, atol=0, rtol=0)
    torch.testing.assert_close(ours_long, theirs_long, atol=0, rtol=0)


def test_degeneracy_flags_match_the_project():
    from vpo_rm.length_reward import response_degeneracy as reference
    from red.length_reward import response_degeneracy as ours
    texts = ["", "   ", "hello", "a\n" * 32, "a\n" * 31 + "b"]
    assert ours(texts, 32) == reference(texts, 32)


def test_length_calibration_matches_the_project():
    from vpo_rm.length_reward import calibrate_reward_scale as reference
    from red.length_reward import calibrate_reward_scale as ours
    torch.manual_seed(31)
    rewards = torch.randn(16) * 3
    groups = torch.arange(16) // 4
    valid = torch.ones(16, dtype=torch.bool)
    valid[3] = False
    torch.testing.assert_close(ours(rewards, groups, valid), reference(rewards, groups, valid),
                               atol=0, rtol=0)


def test_group_sigma_is_the_projects_group_scale():
    """RED reports (but never divides by) the sibling arms' group spread."""
    from vpo_rm.core import group_advantages as reference
    from red.reward import group_sigma
    torch.manual_seed(37)
    rewards = torch.randn(16) * 4
    groups = torch.arange(16) // 4
    ours = group_sigma(rewards, groups)
    # std_floor=0 makes the project return std + eps; the eps is the only gap.
    _, theirs = reference(rewards, groups, std_floor=0.0)
    torch.testing.assert_close(ours, theirs, atol=2e-6, rtol=1e-6)


def test_red_does_not_standardise_or_clip():
    """The two deliberate divergences, asserted as absences."""
    import inspect
    from red import loss, reward
    assert not hasattr(reward, "group_advantages"), \
        "RED must not standardise the advantage by the group spread"
    parameters = inspect.signature(loss.rloo_policy_loss).parameters
    assert "clip_eps" not in parameters and "old_logp" not in parameters, \
        "RLOO has no importance ratio, so there is nothing to clip"
