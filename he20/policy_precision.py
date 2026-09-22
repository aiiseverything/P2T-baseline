"""FP32 policy output projection, ported from ``vpo_rm/policy_precision.py``.

The project's canonical protocol sets ``policy_head_dtype='float32'`` and its
launcher refuses runs whose manifest disagrees, so the vLLM sampler and the HF
trainer forward both project the vocabulary in FP32.

This matters more for he20 than for the VPO-RM arms.  he20 anchors its importance
ratio on the sampler's chosen-token log probability, so any head-precision
difference between the sampler and the trainer lands directly in
``importance = exp(old_logp - rollout_logprob)`` instead of cancelling out.
Keeping both sides in FP32 removes the discrepancy rather than absorbing it.

The projection runs in FP32 with ordinary autograd on the hidden states; the
head itself stays frozen, which is what a LoRA run wants anyway.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _require_frozen_linear(head):
    if type(head) not in (nn.Linear, FrozenFP32OutputHead):
        raise ValueError("FP32 output head requires a plain nn.Linear without head LoRA or wrappers")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("FP32 output head weight and bias must already be frozen")
    if head.weight.device.type == "meta":
        raise ValueError("FP32 output head requires materialized frozen weights")


class FrozenFP32OutputHead(nn.Linear):
    """Frozen Linear preserving ``weight``/``bias`` keys and hidden autograd."""

    def __init__(self, head: nn.Linear):
        _require_frozen_linear(head)
        # Avoid initializing another vocabulary-sized random weight matrix.
        nn.Module.__init__(self)
        self.in_features, self.out_features = head.in_features, head.out_features
        self.weight = nn.Parameter(head.weight.detach().to(torch.float32), requires_grad=False)
        self.bias = (nn.Parameter(head.bias.detach().to(torch.float32), requires_grad=False)
                     if head.bias is not None else None)
        self.train(head.training)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if any(p.dtype != torch.float32 or p.requires_grad for p in self.parameters()):
            raise ValueError("FP32 output head parameters must remain frozen float32")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.weight, self.bias)


def enable_fp32_output_head(actor) -> FrozenFP32OutputHead:
    """Install an untied, frozen FP32 head; call again after an adapter reload."""
    head = actor.get_output_embeddings()
    _require_frozen_linear(head)
    embedding = actor.get_input_embeddings()
    embedding_weight = getattr(embedding, "weight", None)
    if embedding_weight is None:
        raise ValueError("Cannot establish whether the output head is tied to input embeddings")
    if (head.weight is embedding_weight
            or (head.weight.device == embedding_weight.device
                and head.weight.untyped_storage().data_ptr()
                == embedding_weight.untyped_storage().data_ptr())):
        raise ValueError("FP32 output head must not be tied to input embeddings")
    if isinstance(head, FrozenFP32OutputHead):
        if any(p.dtype != torch.float32 for p in head.parameters()):
            raise ValueError("FP32 output head parameters must remain float32")
        return head
    converted = FrozenFP32OutputHead(head)
    actor.set_output_embeddings(converted)
    return converted
