"""Byte-exact mapping from actor response tokens to reward-model positions.

Eq. (2) attributes a gradient to *the token that was sampled*.  The reward model
never sees the actor's token IDs directly -- it sees its own rendering of the
decoded text -- so an actor token only has a reward-model gradient if its bytes
survive the round trip unchanged.  This module finds those correspondences.

This is a port of ``vpo_rm/reward_inputs.py`` from the parent project, kept
behaviourally identical so the two credit assignment schemes attribute gradients
to the same positions: a token that cannot be mapped gets position -1 and,
downstream, a zero reward-model gradient.

Two consequences worth stating explicitly:

* Special tokens are decoded away by ``skip_special_tokens=True``, so they are
  never mapped.  In particular the reward model pools at the last valid position
  of the canonical chat, which *is* a special token -- so the last prefix score
  the reward model actually reads sits past the last mapped response token.  RED
  carries that trailing difference on the final mapped response token so the
  telescoping identity ``sum_t r~_t = R(x,y) - R(x, empty)`` stays exact; see
  ``reward.prefix_token_rewards``.  The trainer logs ``red_unmapped_share_mean``
  so the unmapped share stays visible.
* Tokens that a BPE merge/split rewrites receive -1.  A redistributed reward is
  never invented for them; they contribute exactly zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json

from .tokens import get_special_token_ids

REWARD_INPUT_PROTOCOL = "canonical_chat_v1"


def _render(tokenizer, prompt, response_text):
    if not isinstance(prompt, str) or not isinstance(response_text, str):
        raise TypeError("Reward prompt and response must be strings")
    # No generation prefix and no fallback: the checkpoint owns scoring format.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response_text}],
        tokenize=False, add_generation_prompt=False)


def canonical_reward_input(reward_tokenizer, prompt: str, response_text: str) -> list[int]:
    """Follow the RM model-card full-conversation serialization exactly."""
    rendered = _render(reward_tokenizer, prompt, response_text)
    # Chat templates already contain their own BOS/end markers. Keep that exact
    # text for byte attribution and do not add a second BOS here.
    ids = list(reward_tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("Reward chat must contain valid token IDs")
    return ids


@dataclass(frozen=True)
class RewardInput:
    input_ids: list[int]
    response_positions: list[int]
    response_text: str


@lru_cache(maxsize=8)
def _byte_decoder(tokenizer):
    """ByteLevel BPE byte alphabet; reject other decoder families explicitly."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or backend.decoder is None:
        return None
    decoder = json.loads(backend.decoder.__getstate__())
    if decoder.get("type") != "ByteLevel":
        return None
    values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = values[:]
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + len(codepoints) - 188)
    # The original alphabet contains 188 visible byte values.
    return {chr(character): byte for byte, character in zip(values, codepoints)}


def _byte_spans(tokenizer, ids, text, *, skip_special_tokens):
    """Byte intervals, only when the entire reconstructed stream agrees.

    Byte offsets avoid ambiguity from repeated words and UTF-8 tokens that split
    one character.  Unsupported tokenizers get no token-specific credit rather
    than silently attributing a gradient to a different token.
    """
    decoder = _byte_decoder(tokenizer)
    if decoder is None:
        return None
    specials = set(get_special_token_ids(tokenizer))
    added = set(getattr(tokenizer, "added_tokens_decoder", {}))
    pieces = tokenizer.convert_ids_to_tokens(ids)
    offset, spans, chunks = 0, [], []
    for token_id, piece in zip(ids, pieces):
        if skip_special_tokens and token_id in specials:
            spans.append(None)
            continue
        if token_id in added:
            raw = piece.encode("utf-8")
        else:
            try:
                raw = bytes(decoder[c] for c in piece)
            except KeyError:
                return None
        spans.append((offset, offset + len(raw)))
        chunks.append(raw)
        offset += len(raw)
    return spans if b"".join(chunks) == text.encode("utf-8") else None


def build_reward_input(actor_tokenizer, reward_tokenizer, prompt: str,
                       response_ids: list[int]) -> RewardInput:
    """Canonical score input and exact positions for unchanged response tokens.

    Removed specials, stop tokens, BPE merges/splits and template-rewritten text
    receive -1.
    """
    ids = list(response_ids)
    text = actor_tokenizer.decode(ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    rendered = _render(reward_tokenizer, prompt, text)
    canonical = canonical_reward_input(reward_tokenizer, prompt, text)
    positions = [-1] * len(ids)
    marker = "__VPO_RM_RESPONSE_BOUNDARY_9f174__"
    while marker in prompt or marker in text:
        marker += "x"
    probe = _render(reward_tokenizer, prompt, marker)
    if probe.count(marker) != 1:
        return RewardInput(canonical, positions, text)
    prefix, suffix = probe.split(marker)
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        return RewardInput(canonical, positions, text)
    body_end = len(rendered) - len(suffix) if suffix else len(rendered)
    visible = rendered[len(prefix):body_end]
    # Qwen removes leading newlines; Llama trims whitespace at both ends.
    # Accept only an unchanged contiguous body with whitespace outside it.
    if not visible or (visible.isspace() and visible != text):
        return RewardInput(canonical, positions, text)
    body_start = text.find(visible)
    if body_start < 0:
        return RewardInput(canonical, positions, text)
    removed = text[:body_start]
    if removed.strip() or text[body_start + len(visible):].strip():
        return RewardInput(canonical, positions, text)
    source_spans = _byte_spans(actor_tokenizer, ids, text, skip_special_tokens=True)
    target_spans = _byte_spans(reward_tokenizer, canonical, rendered, skip_special_tokens=False)
    if source_spans is None or target_spans is None:
        return RewardInput(canonical, positions, text)
    trimmed_bytes, prefix_bytes = len(removed.encode()), len(prefix.encode())
    visible_end = trimmed_bytes + len(visible.encode())
    target = {(span[0], span[1], token): index
              for index, (token, span) in enumerate(zip(canonical, target_spans)) if span}
    for i, (token, span) in enumerate(zip(ids, source_spans)):
        if span is None or span[0] < trimmed_bytes or span[1] > visible_end:
            continue
        key = (prefix_bytes + span[0] - trimmed_bytes,
               prefix_bytes + span[1] - trimmed_bytes, token)
        positions[i] = target.get(key, -1)
    return RewardInput(canonical, positions, text)
