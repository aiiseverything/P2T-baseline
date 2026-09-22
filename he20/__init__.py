"""he20: a high-entropy-minority-token baseline for the VPO-RM project.

Reproduces "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
Reinforcement Learning for LLM Reasoning" (Wang et al., NeurIPS 2025,
arXiv 2506.01939) under this project's actor, reward model, data and
hyperparameters.  The paper's Eq. (6) is the whole of the method: the
policy-gradient loss is computed only over the tokens whose per-token entropy
falls in the top ``entropy_top_ratio`` fraction of the population, and the
token-count normaliser is restricted to those same tokens.  The RL algorithm is
otherwise this project's own GRPO -- the same clipped surrogate and KL that the
sibling ``p2t`` and ``vpo_rm`` arms run -- with no reward redistribution, no
attribution and no token-level credit.  See ``HE20_REPRO_NOTES.md`` for the
equation-to-code map and for the two things the paper leaves open about the
population, which ``he20/mask.py`` records rather than settles silently.
"""

from .mask import (ENTROPY_TOP_RATIO_DEFAULT, ENTROPY_TOP_RULES, entropy_threshold,
                   entropy_top_mask, kept_fraction)
from .reward import (Credit, HE20_PROTOCOL, group_advantages, group_sigma,
                     grpo_token_advantage, he20_credit)

__all__ = [
    "ENTROPY_TOP_RATIO_DEFAULT", "ENTROPY_TOP_RULES", "entropy_threshold",
    "entropy_top_mask", "kept_fraction",
    "Credit", "HE20_PROTOCOL", "group_advantages", "group_sigma",
    "grpo_token_advantage", "he20_credit",
]
