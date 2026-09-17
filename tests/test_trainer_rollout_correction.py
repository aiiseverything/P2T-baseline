"""Actual optimizer checks for a separately frozen sampler importance ratio."""
import math
from unittest.mock import patch

import pytest
import torch

from test_trainer_regressions import trainer, fixed_rollout


def attach_probabilities(t, factor):
    fixed_rollout(t)
    original = t.rollout
    values = original(["p"])
    old = t._old_logp_microbatch(*values[:5])
    q = (old - math.log(factor)).detach().requires_grad_(True)
    t.rollout = lambda prompts: original(prompts) + (q,)
    return q


def test_sampler_ratio_scales_policy_and_existing_kl_gradients(tmp_path):
    baseline = trainer(tmp_path / "baseline", beta=.7, max_grad_norm=1000.)
    corrected = trainer(tmp_path / "corrected", beta=.7, max_grad_norm=1000.,
                        rollout_importance_correction=True)
    for t in (baseline, corrected):
        with torch.no_grad():
            t.actor.logits[2].add_(.5)
    fixed_rollout(baseline)
    q = attach_probabilities(corrected, 2.)
    initial = baseline.actor.logits.detach().clone()
    plain = baseline.train_rollout(["p"])
    result = corrected.train_rollout(["p"])
    torch.testing.assert_close(initial - corrected.actor.logits,
                               2 * (initial - baseline.actor.logits), atol=1e-7, rtol=1e-5)
    assert result["loss"] == pytest.approx(2 * plain["loss"], abs=1e-6)
    assert result["rollout_is_mean"] == pytest.approx(2.)
    assert result["rollout_is_ess_ratio"] == pytest.approx(1.)
    assert q.grad is None


def test_required_sampling_probabilities_fail_before_reward_or_update(tmp_path):
    t = trainer(tmp_path, rollout_importance_correction=True)
    fixed_rollout(t)
    with patch.object(t, "_reward_batch") as reward, patch.object(t.optimizer, "step") as step:
        with pytest.raises(ValueError, match="logprob|probabilit"):
            t.train_rollout(["p"])
    reward.assert_not_called()
    step.assert_not_called()


def test_recorded_sampler_probabilities_follow_selected_token_artifact(tmp_path):
    t = trainer(tmp_path, rollout_importance_correction=True)
    q = attach_probabilities(t, 1.5)
    t.train_rollout(["p"])
    record = torch.load(tmp_path / "rollout-1-probabilities.pt", weights_only=True)
    torch.testing.assert_close(record["rollout_logp"], q.detach())
    torch.testing.assert_close(record["importance_weights"], torch.full_like(q, 1.5))
    assert record["response_mask"].all()
    assert not record["old_logp"].requires_grad


def test_importance_configuration_requires_a_boolean(tmp_path):
    with pytest.raises(ValueError, match="rollout_importance_correction"):
        trainer(tmp_path, rollout_importance_correction="true")
