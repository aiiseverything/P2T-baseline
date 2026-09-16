"""Tokenizer-derived stop and structural token identities shared by all callers."""
from collections.abc import Iterable
from numbers import Integral


def get_stop_token_ids(tokenizer) -> tuple[int, ...]:
    """Return registered EOS/chat-end IDs, without an unknown-token fallback.

    Qwen base and instruct checkpoints disagree about which of endoftext and
    im_end is designated EOS. Include either spelling only when it is actually
    registered; never assume its integer ID in an unrelated tokenizer.
    """
    vocab = tokenizer.get_vocab()
    registered = set(vocab.values())
    eos = getattr(tokenizer, "eos_token_id", None)
    candidates = list(eos) if isinstance(eos, Iterable) and not isinstance(eos, (str, bytes)) else [eos]
    candidates.extend(vocab.get(token) for token in ("<|endoftext|>", "<|im_end|>"))
    return tuple(sorted({int(token_id) for token_id in candidates
                         if isinstance(token_id, Integral) and token_id >= 0
                         and token_id in registered}))


def get_structural_token_ids(tokenizer) -> tuple[int, ...]:
    """Return non-special whitespace and period-with-newline token IDs.

    Classify the actual decoded text, with cleanup disabled. A structural token
    is nonempty pure whitespace, or contains a newline and strips to exactly a
    period. Ordinary words, emoji, and other punctuation are content tokens.
    """
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    special.update(get_stop_token_ids(tokenizer))
    for name in ("bos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name, None)
        if isinstance(token_id, Integral):
            special.add(int(token_id))
    structural = []
    for token_id in sorted(set(tokenizer.get_vocab().values()) - special):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded and (decoded.isspace() or ("\n" in decoded and decoded.strip() == ".")):
            structural.append(token_id)
    return tuple(structural)
