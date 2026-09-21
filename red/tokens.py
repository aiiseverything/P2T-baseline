"""Tokenizer-derived token sets, mirrored from ``vpo_rm/token_policy.py``.

The actor and the reward model must share one token-to-ID mapping for Eq. (2) to
be meaningful: ``W[a_t]`` is only the reward model's embedding of the sampled
token if ID ``a_t`` means the same thing on both sides.  ``vpo_rm.alignment``
enforces that; we re-assert it here so the standalone package cannot silently
run with mismatched tokenizers.
"""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import torch

# Stop tokens are the registered end-of-conversation IDs.  Qwen3-Base's EOS is
# <|endoftext|>; an SFT adapter trained on a chat template also emits <|im_end|>,
# so generation must stop on both or a rollout runs past the answer (the
# 2026-09-16 audit in the parent project).
_STOP_TOKEN_NAMES = ("<|endoftext|>", "<|im_end|>",
                     "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>")


def get_stop_token_ids(tokenizer) -> tuple[int, ...]:
    vocab = tokenizer.get_vocab()
    registered = set(vocab.values())
    eos = getattr(tokenizer, "eos_token_id", None)
    candidates = list(eos) if isinstance(eos, Iterable) and not isinstance(eos, (str, bytes)) else [eos]
    candidates.extend(vocab.get(name) for name in _STOP_TOKEN_NAMES)
    return tuple(sorted({int(token_id) for token_id in candidates
                         if isinstance(token_id, Integral) and token_id >= 0
                         and token_id in registered}))


def get_special_token_ids(tokenizer) -> tuple[int, ...]:
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    special.update(token_id for token_id, token
                   in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).items()
                   if getattr(token, "special", False))
    special.update(get_stop_token_ids(tokenizer))
    for name in ("bos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name, None)
        if isinstance(token_id, Integral):
            special.add(int(token_id))
    return tuple(sorted(special))


def configure_model_padding(tokenizer, *, fallback_token=None) -> int:
    """Registered Llama pad if present, else the shared actor-EOS fallback."""
    dedicated = "<|finetune_right_pad_id|>"
    vocab = tokenizer.get_vocab()
    pad = dedicated if dedicated in vocab else (fallback_token or tokenizer.eos_token)
    if pad not in vocab:
        raise ValueError("Model padding requires a registered pad or EOS token")
    tokenizer.pad_token = pad
    tokenizer.pad_token_id = vocab[pad]
    return tokenizer.pad_token_id


def shared_output_mask(tokenizer, vocab_size: int, device=None) -> torch.Tensor:
    """Registered token IDs, with padding excluded unless it is also the EOS."""
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ids = list(tokenizer.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError("Tokenizer IDs must fit the actor output vocabulary")
    allowed[ids] = True
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if pad is not None and pad != eos:
        allowed[pad] = False
    if not allowed.any():
        raise ValueError("Output support must contain at least one token")
    return allowed


def check_tokenizer_identity(actor_tokenizer, rm_tokenizer,
                             actor_vocab_size: int, rm_embedding_rows: int) -> None:
    """Actor and RM must agree on the complete token-to-ID mapping.

    Special-token *semantics* may differ (Qwen3-Base's EOS is <|endoftext|>
    while an instruct-derived RM may register <|im_end|>); padding may not,
    because the trainer shares it by construction when it pads reward-model rows.
    """
    import json
    av, rv = actor_tokenizer.get_vocab(), rm_tokenizer.get_vocab()
    if av != rv:
        raise ValueError("Actor and RM must share the complete token-to-ID mapping")
    if actor_vocab_size != rm_embedding_rows:
        raise ValueError("Actor output vocabulary and RM embedding rows must match")
    if not av or max(av.values()) >= actor_vocab_size:
        raise ValueError("Tokenizer IDs exceed model vocabulary capacity")
    if getattr(actor_tokenizer, "pad_token_id", None) != getattr(rm_tokenizer, "pad_token_id", None):
        raise ValueError("Tokenizer special-token mismatch: pad_token_id")
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


def resolve_actor_tokenizer_source(model_name, init_adapter="", tokenizer_name="") -> str:
    """Prefer explicit or saved SFT tokenizer artifacts; only absence falls back."""
    from pathlib import Path
    if tokenizer_name:
        return str(tokenizer_name)
    if init_adapter and any((Path(init_adapter) / name).exists() for name in (
            "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "vocab.json")):
        return str(init_adapter)
    return str(model_name)


def load_actor_tokenizer(model_name, init_adapter="", tokenizer_name="", *,
                         padding_side="left"):
    """Load the actor tokenizer and verify its recorded SFT protocol if present.

    Saved-but-invalid tokenizer artifacts must fail loading: silently replacing a
    newer export with the base tokenizer would change the token-to-ID mapping
    that Eq. (2) depends on.
    """
    import hashlib
    import json
    from pathlib import Path
    from transformers import AutoTokenizer

    source = resolve_actor_tokenizer_source(model_name, init_adapter, tokenizer_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, padding_side=padding_side,
                                                  trust_remote_code=True)
    except ValueError as error:
        # Transformers 4 cannot resolve the class name Transformers 5 writes into
        # a saved tokenizer_config.json; the fast loader reads the same backend.
        if "TokenizersBackend" not in str(error):
            raise
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(source, padding_side=padding_side)
    configure_model_padding(tokenizer)
    manifest_path = Path(source) / "sft_manifest.json"
    if manifest_path.is_file():
        protocol = json.loads(manifest_path.read_text()).get("token_protocol")
        if protocol is not None:
            actual = {"bos_token_id": tokenizer.bos_token_id, "pad_token_id": tokenizer.pad_token_id,
                      "response_eos_id": tokenizer.eos_token_id,
                      "stop_token_ids": list(get_stop_token_ids(tokenizer)),
                      "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()}
            if not isinstance(protocol, dict) or any(protocol.get(key) != value
                                                     for key, value in actual.items()):
                raise ValueError("Saved actor tokenizer differs from its SFT token protocol")
    return tokenizer
