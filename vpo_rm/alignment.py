"""Token-ID identity and explicit mappings between Actor and RM positions."""
import json
import torch
from torch import Tensor


def check_tokenizers(actor_tokenizer, rm_tokenizer,
                     actor_vocab_size: int, rm_embedding_rows: int) -> None:
    """Run once after loading the tokenizers and model heads.

    Equal vocabulary size alone does not establish token identity. Fast-tokenizer
    serialization also checks segmentation/normalization/added-token behavior.
    Padding side and chat templates may differ; positions are mapped explicitly.
    """
    av, rv = actor_tokenizer.get_vocab(), rm_tokenizer.get_vocab()
    if av != rv:
        raise ValueError("Actor and RM must share the complete token-to-ID mapping")
    if actor_vocab_size != rm_embedding_rows:
        raise ValueError("Actor output vocabulary and RM embedding rows must match")
    if not av or max(av.values()) >= actor_vocab_size:
        raise ValueError("Tokenizer IDs exceed model vocabulary capacity")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(actor_tokenizer, name, None) != getattr(rm_tokenizer, name, None):
            raise ValueError(f"Tokenizer special-token mismatch: {name}")
    if hasattr(actor_tokenizer, "backend_tokenizer") and hasattr(rm_tokenizer, "backend_tokenizer"):
        def spec(tok):
            config = json.loads(tok.backend_tokenizer.to_str())
            config.pop("padding", None)
            config.pop("truncation", None)
            return config
        if spec(actor_tokenizer) != spec(rm_tokenizer):
            raise ValueError("Tokenizer segmentation or normalization differs")
    elif actor_tokenizer is not rm_tokenizer:
        raise ValueError("For slow tokenizers, supply the same tokenizer instance")


def gather_response(values: Tensor, positions: Tensor, response_mask: Tensor) -> Tensor:
    """Gather [B,L,...] at explicit [B,T] positions; invalid entries may be -1."""
    if positions.shape != response_mask.shape or values.shape[0] != positions.shape[0]:
        raise ValueError("Batch dimensions and response position shapes must match")
    if positions.dtype != torch.long:
        raise ValueError("Response positions must be int64")
    valid = response_mask.bool()
    if ((positions[valid] < 0) | (positions[valid] >= values.shape[1])).any():
        raise ValueError("Valid response position outside source sequence")
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    result = values[rows, positions.masked_fill(~valid, 0)]
    expanded = valid.reshape(*valid.shape, *((1,) * (result.ndim - 2)))
    return result.masked_fill(~expanded, 0)


def check_response_tokens(input_ids: Tensor, attention_mask: Tensor,
                          positions: Tensor, token_ids: Tensor,
                          response_mask: Tensor) -> None:
    valid = response_mask.bool()
    gathered = gather_response(input_ids, positions, valid)
    attended = gather_response(attention_mask, positions, valid)
    if not attended[valid].bool().all() or not torch.equal(gathered[valid], token_ids[valid]):
        raise ValueError("Response IDs or valid positions differ between Actor and RM")
    for row in range(positions.shape[0]):
        selected = positions[row, valid[row]]
        if selected.numel() > 1 and not (selected[1:] > selected[:-1]).all():
            raise ValueError("Response positions must be strictly increasing")


def shared_output_mask(tokenizer, vocab_size: int, device=None) -> Tensor:
    """Output support: registered token IDs, with padding excluded unless EOS.

    Use this same support during generation, old-policy attribution and updates.
    The corresponding HF generate argument is suppress_tokens=(~mask).nonzero().
    """
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ids = list(tokenizer.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError("Tokenizer IDs must fit the Actor output vocabulary")
    allowed[ids] = True
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if pad is not None and pad != eos:
        allowed[pad] = False
    if not allowed.any():
        raise ValueError("Output support must contain at least one token")
    return allowed
