"""P2T: a training-free token-reward baseline for the VPO-RM project.

Reproduces "Unlocking Token Rewards via Training-Free Reward Attribution"
(Sitong Wu et al., CVPR) under this project's actor, reward model, data and
hyperparameters.  See ``P2T_REPRO_NOTES.md`` for the equation-to-code map and
for the three places where the paper's text and its own equations disagree.
"""

from .attribution import null_token_attribution, P2T_APPROXIMATION
from .reward import (Credit, group_advantages, p2t_credit, p2t_token_advantage,
                     p2t_token_reward, P2T_ALPHA_SHORT_COT, P2T_OMEGA)

__all__ = [
    "null_token_attribution", "P2T_APPROXIMATION",
    "Credit", "group_advantages", "p2t_credit", "p2t_token_advantage",
    "p2t_token_reward", "P2T_ALPHA_SHORT_COT", "P2T_OMEGA",
]
