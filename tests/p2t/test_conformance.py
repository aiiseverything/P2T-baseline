"""Cross-checks against the parent project's own implementations.

The P2T package is standalone -- it never imports ``vpo_rm`` at runtime -- but a
baseline is only useful if it shares the project's protocol.  These tests import
``vpo_rm`` from the same checkout and assert that the pieces which must be
identical really are: group standardisation, the clipped surrogate, and the
byte-exact actor-to-reward-model token mapping.
"""
from pathlib import Path

import pytest
import torch

vpo_rm = pytest.importorskip("vpo_rm", reason="parent project not importable from this checkout")

from p2t.loss import grpo_policy_loss as p2t_policy_loss  # noqa: E402
from p2t.reward import group_advantages as p2t_group_advantages  # noqa: E402


def test_group_advantages_matches_the_project():
    from vpo_rm.core import group_advantages as reference
    torch.manual_seed(17)
    rewards = torch.randn(16) * 4
    groups = torch.arange(16) // 4
    for floor in (0.0, 0.7):
        ours = p2t_group_advantages(rewards, groups, std_floor=floor)
        theirs = reference(rewards, groups, std_floor=floor)
        torch.testing.assert_close(ours[0], theirs[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ours[1], theirs[1], atol=1e-6, rtol=1e-6)


def test_policy_loss_matches_the_project_including_gradients():
    from vpo_rm.core import grpo_policy_loss as reference
    torch.manual_seed(23)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    advantage = torch.randn(3, 4) * 2
    old = torch.randn(3, 4) * 0.5 - 2.0
    new = (old + torch.randn(3, 4) * 0.3).requires_grad_(True)
    ours = p2t_policy_loss(new, old, advantage, mask, 0.2)
    theirs = reference(new, old, advantage, mask, 0.2)
    torch.testing.assert_close(ours, theirs, atol=0, rtol=0)
    ours_grad = torch.autograd.grad(ours, new, retain_graph=True)[0]
    theirs_grad = torch.autograd.grad(theirs, new)[0]
    torch.testing.assert_close(ours_grad, theirs_grad, atol=0, rtol=0)


def test_policy_loss_matches_with_rollout_importance_weights():
    from vpo_rm.core import grpo_policy_loss as reference
    torch.manual_seed(29)
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.bool)
    advantage = torch.randn(2, 3)
    old = torch.randn(2, 3) - 2.0
    new = (old + 0.1).requires_grad_(True)
    importance = torch.rand(2, 3) + 0.5
    ours = p2t_policy_loss(new, old, advantage, mask, 0.2, importance_weights=importance)
    theirs = reference(new, old, advantage, mask, 0.2, importance_weights=importance)
    torch.testing.assert_close(ours, theirs, atol=0, rtol=0)


def test_reward_input_mapping_matches_the_project():
    """Same actor->RM token attribution as the VPO-RM arms, on a real tokenizer."""
    pytest.importorskip("transformers", reason="transformers is not installed here")
    model_dir = Path(__file__).resolve().parents[2] / "models/Qwen3-14B-Base"
    if not model_dir.is_dir():
        pytest.skip(f"local actor tokenizer not present at {model_dir}")
    from transformers import AutoTokenizer
    from vpo_rm.reward_inputs import build_reward_input as reference
    from p2t.mapping import build_reward_input as ours

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = "Explain why the sky is blue, briefly."
    response_ids = tokenizer("Because short wavelengths scatter more.", add_special_tokens=False)["input_ids"]
    response_ids = response_ids + [tokenizer.eos_token_id]
    first, second = ours(tokenizer, tokenizer, prompt, response_ids), reference(tokenizer, tokenizer, prompt, response_ids)
    assert first.input_ids == second.input_ids
    assert first.response_positions == second.response_positions
    assert first.response_text == second.response_text


def test_soft_length_window_matches_the_project():
    from vpo_rm.length_reward import soft_length_penalties as reference
    from p2t.length_reward import soft_length_penalties as ours
    lengths = torch.tensor([1, 5, 8, 600, 1024, 1500, 2048], dtype=torch.float32)
    kwargs = dict(short_threshold=8, long_threshold=1024, max_length=2048,
                  short_strength=.5, long_strength=2.)
    ours_short, ours_long = ours(lengths, 3.0, **kwargs)
    theirs_short, theirs_long = reference(lengths, 3.0, **kwargs)
    torch.testing.assert_close(ours_short, theirs_short, atol=0, rtol=0)
    torch.testing.assert_close(ours_long, theirs_long, atol=0, rtol=0)


def test_degeneracy_flags_match_the_project():
    from vpo_rm.length_reward import response_degeneracy as reference
    from p2t.length_reward import response_degeneracy as ours
    texts = ["", "   ", "hello", "a\n" * 32, "a\n" * 31 + "b"]
    assert ours(texts, 32) == reference(texts, 32)


def test_length_calibration_matches_the_project():
    from vpo_rm.length_reward import calibrate_reward_scale as reference
    from p2t.length_reward import calibrate_reward_scale as ours
    torch.manual_seed(31)
    rewards = torch.randn(16) * 3
    groups = torch.arange(16) // 4
    valid = torch.ones(16, dtype=torch.bool)
    valid[3] = False
    torch.testing.assert_close(ours(rewards, groups, valid), reference(rewards, groups, valid),
                               atol=0, rtol=0)
