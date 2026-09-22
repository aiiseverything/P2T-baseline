"""RED: a reward-redistribution token-reward baseline for the VPO-RM project.

Reproduces "RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward
Redistribution" (Jiahui Li et al., EMNLP 2025) under this project's actor, reward
model, data and hyperparameters, in its **RLOO** variant.  See
``RED_REPRO_NOTES.md`` for the equation-to-code map, for the place where the paper
leaves the RLOO baseline undefined, and for the four contradictions it records
rather than repairs.
"""

from .reward import (Credit, RED_ALPHA_DEFAULT, RED_BETA_C_DEFAULT, RED_PROTOCOL,
                     RETIRED_RLOO_ADVANTAGE_RULE, RLOO_ADVANTAGE_RULE, credit_share,
                     group_sigma, prefix_boundaries, prefix_token_rewards,
                     red_convex_combination, red_final_reward, red_kl_reward,
                     rloo_baseline, rloo_red_credit, sequence_returns,
                     sequence_reward_at_eos)

__all__ = [
    "Credit", "RED_ALPHA_DEFAULT", "RED_BETA_C_DEFAULT", "RED_PROTOCOL",
    "RETIRED_RLOO_ADVANTAGE_RULE", "RLOO_ADVANTAGE_RULE",
    "credit_share", "group_sigma", "prefix_boundaries", "prefix_token_rewards",
    "red_convex_combination", "red_final_reward", "red_kl_reward",
    "rloo_baseline", "rloo_red_credit", "sequence_returns",
    "sequence_reward_at_eos",
]
