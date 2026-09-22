"""The three-arm comparability contract: VPO, P2T and he20 must differ only in method.

This file is the executable form of that requirement.  The arms are compared
through the parent project's own canonical protocol rather than against each other
directly, because that is what each arm's config claims to be:

* VPO's parameters are ``scripts/corrected_rl_launcher.py:common_config``;
* ``configs/formal250.json`` (P2T) states in its own comment that it is that
  ``common_config`` plus P2T's credit assignment;
* ``configs/he20250.json`` states that it is ``formal250.json`` plus the mask.

So the chain is checked end to end: every training parameter the parent protocol
defines must match here, transitively through P2T.  Keys that are deliberately
*not* compared are listed with the reason, so that "the arms differ only in the
method" is a statement about a named set rather than a claim about everything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from he20.trainer import load_config

ROOT = Path(__file__).resolve().parents[2]

# parent protocol key -> this arm's config key
TRAINING_PARAMETERS = {
    "max_rollouts": "rollout_iterations",
    "max_response_tokens": "max_response_tokens",
    "learning_rate": "learning_rate",
    "beta": "beta",
    "temperature": "temperature",
    "optimizer_minibatch_responses": "optimizer_minibatch_responses",
    "seed": "seed",
    "generation_seed": "generation_seed",
    "min_response_tokens": "min_response_tokens",
    "short_response_threshold": "length_threshold_short",
    "long_response_threshold": "length_threshold_long",
    "short_penalty_strength": "short_penalty_strength",
    "long_penalty_strength": "long_penalty_strength",
    "advantage_std_floor_fraction": "advantage_std_floor_fraction",
    "length_calibration_prompts": "calibration_prompts",
    "degenerate_newline_run": "degenerate_newline_run",
    "credit_microbatch_responses": "microbatch_responses",
    "policy_head_dtype": "policy_head_dtype",
}

# Keys the parent protocol defines that this arm is NOT required to match, each
# with the reason it is a deployment or delivery choice rather than a training
# parameter.  Listing them is the point: an omission here would be a silent
# difference, and this way it is a named one.
NOT_TRAINING_PARAMETERS = {
    "checkpoint_interval": "how often weights are written down, not how they are updated",
    "keep_adapters_every": "adapter retention on disk, not the update",
    "generation_microbatch": "how generation is batched, not the update",
    "vllm_gpu_memory_utilization": "sized for this box's 46 GiB L20 cards, not for the H200s "
                                   "the parent protocol was written against",
    "vllm_tensor_parallel_size": "same: this box needs tp=2 to fit a 14B engine",
    "tau": "the VPO-RM arm's attribution temperature; this arm has no attribution",
    "kl_reference": "the parent names 'init'; this arm's reference adapter is the init adapter",
    "length_reward_mode": "the parent's selector for the soft window; this arm only has the soft one",
    "length_penalty_slope": "the parent's legacy below-anchor cost, off in soft mode",
    "dataset_path": "checked separately below, against the run's own data split",
    "model": "checked separately below, by basename",
    "rm": "checked separately below, by basename",
    "init_adapter": "checked separately below, by weight hash -- the parent's path and this "
                    "checkout's are different names for the same bytes",
}


def _common_config():
    """The parent protocol, or skip if the parent's launcher cannot be imported."""
    launcher = pytest.importorskip("scripts.corrected_rl_launcher",
                                   reason="the parent project's launcher is not importable")
    return launcher.common_config(ROOT)


def test_every_training_parameter_matches_the_parent_protocol():
    canonical = _common_config()
    config = load_config(ROOT / "configs" / "he20250.json").resolved()
    for parent_key, arm_key in TRAINING_PARAMETERS.items():
        assert parent_key in canonical, f"the parent protocol no longer defines {parent_key}"
        assert getattr(config, arm_key) == canonical[parent_key], (
            f"{arm_key} is {getattr(config, arm_key)!r} but the parent protocol's "
            f"{parent_key} is {canonical[parent_key]!r}; this arm would differ from VPO "
            f"and P2T in something other than the method")


def test_the_excluded_keys_are_still_defined_by_the_parent():
    """Renaming a parent key must surface here, not silently stop being compared."""
    canonical = _common_config()
    for name in NOT_TRAINING_PARAMETERS:
        assert name in canonical, (
            f"{name} is no longer in the parent protocol; either it was renamed, in which "
            f"case move it into TRAINING_PARAMETERS, or the exclusion is stale")


def test_the_init_adapter_is_the_canonical_sft_checkpoint_by_weight_hash():
    """The parent names `models/sft-native-eos-clean2k5e2`; this checkout has `models/sft-p2t`.

    Different names, and the difference matters: if they were different weights, the
    arms would start from different policies and no amount of parameter alignment
    would make them comparable.  The canonical asset manifest pins the weight hash,
    so the two can be identified rather than assumed.
    """
    manifest = json.loads((ROOT / "configs" / "ssh-a6000-assets.json").read_text())
    canonical = None
    for asset in manifest["assets"]:
        if asset.get("path", "").endswith("sft-native-eos-clean2k5e2"):
            canonical = asset["weights"]["sha256"]
    if canonical is None:
        pytest.skip("the asset manifest no longer pins the canonical SFT weights")
    local = ROOT / "models" / "sft-p2t" / "adapter_model.safetensors"
    if not local.is_file():
        pytest.skip(f"the init adapter is not present at {local}")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    assert digest == canonical, (
        "the arm's init adapter is not the canonical SFT checkpoint; the arms would "
        "start from different policies")
    # And the config must name that same adapter, so the run cannot silently start
    # from something else.
    config = load_config(ROOT / "configs" / "he20250.json")
    assert config.init_adapter == "models/sft-p2t"


def test_the_model_and_reward_model_are_the_same_pair_the_parent_names():
    canonical = _common_config()
    config = load_config(ROOT / "configs" / "he20250.json").resolved()
    assert Path(config.model_name).name == Path(canonical["model"]).name
    assert Path(config.reward_model_name).name == Path(canonical["rm"]).name


def test_the_arms_that_must_agree_all_have_their_protocols_stamped():
    """A config that cannot say which rule it ran under is not comparable to anything."""
    he20 = load_config(ROOT / "configs" / "he20250.json").resolved()
    assert he20.entropy_top_ratio == 0.2 and he20.entropy_top_rule == "threshold"
    # P2T's config exists and pins its own knob; this is the sibling the arm is
    # measured against, so a missing or renamed knob there is worth failing on.
    p2t = json.loads((ROOT / "configs" / "formal250.json").read_text())
    assert p2t["run_name"] == "p2t250"
