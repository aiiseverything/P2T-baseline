"""Paper Eq. (1)-(2): training-free token attribution from RM input gradients.

The paper defines a token's attribution as the reward lost when its embedding is
replaced by a semantically inert null token,

    I_i = R(E) - R(E_{i<-null})                                    (Eq. 1)

and approximates that difference with a first-order Taylor expansion around the
token's own embedding,

    I_i ~= grad_{e_i} R(E)^T (e_i - e_null).                       (Eq. 2)

One forward and one backward pass of the reward model yields every I_i at once,
which is the whole point of the method: no token-level reward model is trained.

The null token is the reward model's padding token (paper Table 3(a): pad 53.8,
mean-of-vocab 52.5, zero embedding 51.4, EOS 50.3 on MATH-500 pass@1).

Sign convention.  ``e_i - e_null`` (Eq. 2), not ``e_null - e_i``.  The paper's
introductory prose says "the difference between the null token embedding and the
token's original embedding", which is the negation of its own Eq. (2); the
equation is the one that is consistent with Eq. (1), and Eq. (1) is the
definition.  We implement Eq. (2).
"""
from __future__ import annotations

import torch
from torch import Tensor

P2T_APPROXIMATION = "taylor_first_order_eq2"


def _binary_mask(mask: Tensor) -> Tensor:
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError("response_mask must be a binary [B, T] tensor")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("response_mask must be a binary [B, T] tensor")
    mask = mask.bool()
    if not mask.any(-1).all():
        raise ValueError("Each response must have at least one valid token")
    return mask


@torch.no_grad()
def null_token_attribution(input_grads: Tensor, token_ids: Tensor,
                           rm_embedding_weight: Tensor, null_token_id: int,
                           response_mask: Tensor, token_chunk_size: int = 128) -> Tensor:
    """Eq. (2) for every valid response position, in float32.

    ``input_grads`` [B, T, D] is the reward model's gradient with respect to the
    token embedding at each *response* position, already gathered onto the
    actor's token positions.  ``rm_embedding_weight`` [V, D] is the reward
    model's input embedding matrix in the same token-ID order as the actor
    output vocabulary.  Positions outside ``response_mask`` are zero, and so are
    positions the reward model never saw: the caller's gather step zeroes the
    gradient wherever an actor token has no byte-exact reward-model counterpart,
    which makes I = 0 there rather than an invented value.

    A zero attribution is the paper's own "negligible marginal effect" case, so
    those positions keep participating in the Eq. (3) normaliser.
    """
    mask = _binary_mask(response_mask)
    if input_grads.ndim != 3 or input_grads.shape[:2] != mask.shape:
        raise ValueError("input_grads must have shape [B, T, D] matching the response mask")
    if not input_grads.is_floating_point():
        raise ValueError("input_grads must be a floating tensor")
    if token_ids.shape != mask.shape:
        raise ValueError("token_ids must match the response mask shape [B, T]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must be int64")
    if rm_embedding_weight.ndim != 2 or not rm_embedding_weight.is_floating_point():
        raise ValueError("rm_embedding_weight must be a floating [V, D] matrix")
    if rm_embedding_weight.shape[1] != input_grads.shape[-1]:
        raise ValueError("RM embedding width must match the RM input-gradient width")
    if input_grads.device != rm_embedding_weight.device or token_ids.device != input_grads.device:
        raise ValueError("Attribution tensors must share one device")
    if isinstance(null_token_id, bool) or not isinstance(null_token_id, int):
        raise ValueError("null_token_id must be a Python int")
    vocab = rm_embedding_weight.shape[0]
    if not 0 <= null_token_id < vocab:
        raise ValueError("null_token_id must index the RM embedding matrix")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    valid_ids = token_ids[mask]
    if ((valid_ids < 0) | (valid_ids >= vocab)).any():
        raise ValueError("Valid response token IDs must lie inside the RM vocabulary")
    if not torch.isfinite(input_grads[mask]).all():
        raise ValueError("RM input gradients must be finite on valid response positions")

    attribution = torch.zeros(mask.shape, device=input_grads.device, dtype=torch.float32)
    null_row = rm_embedding_weight[null_token_id].float()
    rows, positions = mask.nonzero(as_tuple=True)
    for start in range(0, rows.numel(), token_chunk_size):
        r = rows[start:start + token_chunk_size]
        t = positions[start:start + token_chunk_size]
        gradient = input_grads[r, t].float()
        delta = rm_embedding_weight[token_ids[r, t]].float() - null_row
        attribution[r, t] = (gradient * delta).sum(-1)
    if not torch.isfinite(attribution).all():
        raise ValueError("Token attribution overflowed float32")
    return attribution
