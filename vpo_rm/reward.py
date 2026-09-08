"""Differentiable reward adapters with explicit, padding-aware pooling."""
import torch
from torch import Tensor, nn


def position_ids_from_mask(attention_mask: Tensor) -> Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)


class LastTokenReward(nn.Module):
    """Decoder backbone + scalar score head; explicit score position or last valid.

    For HF Qwen3 sequence classifiers:
        LastTokenReward(model.base_model, model.score)
    Check checkpoint-specific reward normalization separately.
    """
    def __init__(self, backbone: nn.Module, score_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.score_head = score_head

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, *, inputs_embeds: Tensor, attention_mask: Tensor,
                position_ids: Tensor | None = None,
                score_positions: Tensor | None = None) -> Tensor:
        if not attention_mask.bool().any(-1).all():
            raise ValueError("Each RM input must contain a valid token")
        if position_ids is None:
            position_ids = position_ids_from_mask(attention_mask)
        hidden = self.backbone(inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask, position_ids=position_ids,
                               use_cache=False, return_dict=True).last_hidden_state
        batch, length = attention_mask.shape
        rows = torch.arange(batch, device=hidden.device)
        if score_positions is None:
            indexes = torch.arange(length, device=hidden.device).expand(batch, -1)
            score_positions = indexes.masked_fill(~attention_mask.bool(), -1).amax(-1)
        if score_positions.shape != (batch,) or score_positions.dtype != torch.long:
            raise ValueError("score_positions must be int64 [B]")
        if ((score_positions < 0) | (score_positions >= length)).any():
            raise ValueError("score_positions outside input sequence")
        if not attention_mask[rows, score_positions].bool().all():
            raise ValueError("Reward pooling must select valid positions")
        score = self.score_head(hidden[rows, score_positions])
        if score.shape != (batch, 1):
            raise ValueError("LastTokenReward requires a scalar head")
        return score[:, 0].float()


def reward_input_gradients(reward_model: nn.Module, input_ids: Tensor,
                           attention_mask: Tensor, **score_kwargs) -> tuple[Tensor, Tensor]:
    """One scalar reward and one input gradient per response, no batch rescaling."""
    reward_model.requires_grad_(False)
    reward_model.eval()
    # inference_mode(False) also supports callers whose rollout uses inference_mode.
    with torch.inference_mode(False), torch.enable_grad():
        ids, mask = input_ids.clone(), attention_mask.clone()
        kwargs = {k: v.clone() if isinstance(v, Tensor) else v
                  for k, v in score_kwargs.items()}
        e = reward_model.get_input_embeddings()(ids).detach().requires_grad_(True)
        rewards = reward_model(inputs_embeds=e, attention_mask=mask, **kwargs)
        if rewards.shape != (ids.shape[0],):
            raise ValueError("Reward adapter must return a scalar per response [B]")
        if not torch.isfinite(rewards).all():
            raise ValueError("RM produced non-finite rewards")
        grads = torch.autograd.grad(rewards.sum(), e)[0]
    return rewards.detach(), grads.detach()
