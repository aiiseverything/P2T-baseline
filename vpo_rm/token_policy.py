"""Tokenizer-derived stop and structural token identities shared by all callers."""
from collections.abc import Iterable
from numbers import Integral


def get_stop_token_ids(tokenizer) -> tuple[int, ...]:
    """Return registered EOS/chat-end IDs, without an unknown-token fallback.

    Include the Qwen and Llama conversation endings only when registered;
    never assume an integer ID in an unrelated tokenizer. Llama's end-of-text
    and end-of-message markers supplement its tokenizer EOS (end-of-turn).
    """
    vocab = tokenizer.get_vocab()
    registered = set(vocab.values())
    eos = getattr(tokenizer, "eos_token_id", None)
    candidates = list(eos) if isinstance(eos, Iterable) and not isinstance(eos, (str, bytes)) else [eos]
    candidates.extend(vocab.get(token) for token in (
        "<|endoftext|>", "<|im_end|>",
        "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>",
    ))
    return tuple(sorted({int(token_id) for token_id in candidates
                         if isinstance(token_id, Integral) and token_id >= 0
                         and token_id in registered}))


def get_special_token_ids(tokenizer) -> tuple[int, ...]:
    """Include backend specials omitted from some exported tokenizer attributes."""
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    special.update(token_id for token_id, token in
                   (getattr(tokenizer, "added_tokens_decoder", {}) or {}).items()
                   if getattr(token, "special", False))
    special.update(get_stop_token_ids(tokenizer))
    for name in ("bos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name, None)
        if isinstance(token_id, Integral):
            special.add(int(token_id))
    return tuple(sorted(special))


def get_structural_token_ids(tokenizer) -> tuple[int, ...]:
    """Return non-special whitespace and period-with-newline token IDs.

    Classify the actual decoded text, with cleanup disabled. A structural token
    is nonempty pure whitespace, or contains a newline and strips to exactly a
    period. Ordinary words, emoji, and other punctuation are content tokens.
    """
    special = set(get_special_token_ids(tokenizer))
    structural = []
    for token_id in sorted(set(tokenizer.get_vocab().values()) - special):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded and (decoded.isspace() or ("\n" in decoded and decoded.strip() == ".")):
            structural.append(token_id)
    return tuple(structural)


def configure_model_padding(tokenizer, *, fallback_token=None) -> int:
    """Use model-owned Llama padding, or the established shared EOS fallback.

    Qwen actor/RM callers pass the actor EOS as ``fallback_token`` so their
    historical shared-pad contract remains unchanged. A registered Llama pad
    always takes precedence, independently for each model.
    """
    dedicated = '<|finetune_right_pad_id|>'
    vocab = tokenizer.get_vocab()
    pad = dedicated if dedicated in vocab else (fallback_token or tokenizer.eos_token)
    if pad not in vocab:
        raise ValueError('Model padding requires a registered pad or EOS token')
    tokenizer.pad_token = pad
    tokenizer.pad_token_id = vocab[pad]
    return tokenizer.pad_token_id


def resolve_actor_tokenizer_source(model_name, init_adapter='', tokenizer_name='') -> str:
    """Prefer explicit or saved SFT tokenizer artifacts; only absence falls back."""
    from pathlib import Path

    if tokenizer_name:
        return str(tokenizer_name)
    if init_adapter and any((Path(init_adapter) / name).exists() for name in (
            'tokenizer_config.json', 'tokenizer.json', 'tokenizer.model', 'vocab.json')):
        return str(init_adapter)
    return str(model_name)


def load_actor_tokenizer(model_name, init_adapter='', tokenizer_name='', *, padding_side='left'):
    """Load the chosen actor tokenizer and verify its recorded SFT protocol.

    Saved-but-invalid tokenizer artifacts must fail loading. In particular, do
    not silently replace a newer tokenizer export with the base tokenizer.
    """
    import hashlib
    import json
    from pathlib import Path
    from transformers import AutoTokenizer

    source = resolve_actor_tokenizer_source(model_name, init_adapter, tokenizer_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, padding_side=padding_side, trust_remote_code=True)
    except ValueError as error:
        # Transformers 4 cannot resolve the class name that Transformers 5 writes
        # into a saved tokenizer_config.json. The fast loader reads the identical
        # saved backend and chat template; the protocol check below still applies.
        if 'TokenizersBackend' not in str(error):
            raise
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(source, padding_side=padding_side)
    configure_model_padding(tokenizer)
    manifest_path = Path(source) / 'sft_manifest.json'
    if manifest_path.is_file():
        protocol = json.loads(manifest_path.read_text()).get('token_protocol')
        if protocol is not None:
            actual = {'bos_token_id': tokenizer.bos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                      'response_eos_id': tokenizer.eos_token_id,
                      'stop_token_ids': list(get_stop_token_ids(tokenizer)),
                      'chat_template_sha256': hashlib.sha256((tokenizer.chat_template or '').encode()).hexdigest()}
            if not isinstance(protocol, dict) or any(protocol.get(key) != value
                                                      for key, value in actual.items()):
                raise ValueError('Saved actor tokenizer differs from its SFT token protocol')
    return tokenizer


def tokenize_rendered_prompts(tokenizer, prompts, prompt_token_ids=None) -> list[dict]:
    """Bind explicit vLLM token inputs to rendered chats without another BOS."""
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError('Generation prompts must be a nonempty list of rendered strings')
    encoded = [tokenizer(prompt, add_special_tokens=False)['input_ids'] for prompt in prompts]
    if any(not row for row in encoded):
        raise ValueError('Generation prompt tokens must be nonempty')
    if prompt_token_ids is not None:
        if (not isinstance(prompt_token_ids, list) or len(prompt_token_ids) != len(encoded)
                or any(not isinstance(row, list) or not row
                       or any(type(token) is not int or token < 0 for token in row)
                       for row in prompt_token_ids)
                or prompt_token_ids != encoded):
            raise ValueError('Explicit prompt token IDs differ from the rendered chat protocol')
    return [{'prompt_token_ids': row} for row in encoded]
