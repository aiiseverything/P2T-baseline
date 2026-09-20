from .alignment import check_tokenizers, check_response_tokens, gather_response, shared_output_mask
from .core import Credit, allocate, compute_credit, group_advantages, grpo_policy_loss, \
    guard_degenerate_rewards, random_credit
from .reward import LastTokenReward, reward_input_gradients
from .integration import (RolloutCache, actor_response_logits, build_credit_cache,
                          response_reward_gradients, actor_policy_loss)
from .trainer import TrainerConfig, VPOTrainer, split_prompts, normalize_prompt, load_prompt_dataset

__all__ = ["Credit", "RolloutCache", "allocate", "compute_credit", "group_advantages",
           "grpo_policy_loss", "guard_degenerate_rewards", "random_credit",
           "LastTokenReward", "reward_input_gradients",
           "check_tokenizers", "check_response_tokens", "gather_response",
           "actor_response_logits", "build_credit_cache", "response_reward_gradients",
           "actor_policy_loss", "shared_output_mask"]
__all__ += ["TrainerConfig", "VPOTrainer", "split_prompts", "normalize_prompt", "load_prompt_dataset"]
