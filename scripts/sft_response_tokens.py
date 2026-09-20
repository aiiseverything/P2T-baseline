"""Pure token helpers for choosing the supervised SFT response terminator."""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ResponseEosSpec:
    mode: str
    template_eos_id: int
    actual_eos_id: int
    trailing_ids: tuple[int, ...]


def configure_sft_pad_token(tokenizer) -> int:
    """Keep existing padding, otherwise prefer Llama's dedicated training pad."""
    if tokenizer.pad_token_id is None:
        dedicated_pad = "<|finetune_right_pad_id|>"
        dedicated_id = tokenizer.get_vocab().get(dedicated_pad)
        if isinstance(dedicated_id, int) and dedicated_id >= 0:
            tokenizer.pad_token = dedicated_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("SFT padding requires a registered pad token or native EOS")
    return tokenizer.pad_token_id


def install_chat_template(tokenizer, template: str, expected_sha256: str | None = None) -> str:
    """Install a pinned chat template on a tokenizer that ships without one.

    A base checkpoint such as Llama-3.1-8B has no chat template; the SFT must
    render with the exact native template. An existing different template is
    never silently replaced. Returns the installed template's SHA256.
    """
    if not isinstance(template, str) or not template.strip():
        raise ValueError("chat template text must be a nonempty string")
    actual_sha256 = hashlib.sha256(template.encode()).hexdigest()
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256).lower():
        raise ValueError("chat template sha256 does not match the pinned value")
    existing = getattr(tokenizer, "chat_template", None)
    if isinstance(existing, str) and existing and existing != template:
        raise ValueError("tokenizer already carries a different chat template")
    tokenizer.chat_template = template
    return actual_sha256


def build_response_eos_spec(tokenizer, mode: str) -> ResponseEosSpec:
    if mode not in {"chat_template", "native"}:
        raise ValueError(f"unsupported response EOS mode: {mode}")

    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        template_eos_id = vocab["<|im_end|>"]
        trailing_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
        if not trailing_ids or not all(isinstance(token_id, int) and token_id >= 0
                                      for token_id in trailing_ids):
            raise ValueError("chat template trailing newline has no valid token IDs")
    else:
        # Llama 3.1 ends ordinary assistant messages at eot, with no newline.
        template_eos_id = vocab.get("<|eot_id|>")
        trailing_ids = []
    if not isinstance(template_eos_id, int) or template_eos_id < 0:
        raise ValueError("chat template EOS token <|im_end|> or <|eot_id|> is missing")

    actual_eos_id = template_eos_id
    if mode == "native":
        actual_eos_id = tokenizer.eos_token_id
        if not isinstance(actual_eos_id, int) or actual_eos_id < 0:
            raise ValueError("tokenizer native EOS token is missing")
        if actual_eos_id not in vocab.values():
            raise ValueError("tokenizer native EOS ID is outside the registered vocabulary")

    return ResponseEosSpec(mode, template_eos_id, actual_eos_id,
                           tuple(trailing_ids))


def rewrite_response_eos(full_ids, prefix_length: int,
                         spec: ResponseEosSpec) -> list[int]:
    ids = list(full_ids)
    if not 0 <= prefix_length <= len(ids):
        raise ValueError("prefix length is outside the token sequence")

    expected_tail = [spec.template_eos_id, *spec.trailing_ids]
    eos_position = len(ids) - len(expected_tail)
    if eos_position < prefix_length or ids[eos_position:] != expected_tail:
        raise ValueError(
            "assistant chat template tail does not match its registered ending"
        )
    ids[eos_position] = spec.actual_eos_id
    return ids


def response_labels(full_ids, prefix_length: int) -> list[int]:
    ids = list(full_ids)
    if not 0 <= prefix_length <= len(ids):
        raise ValueError("prefix length is outside the token sequence")
    return [-100] * prefix_length + ids[prefix_length:]


def supervised_token_sha256(label_rows) -> str:
    """Stable hash of supervised token IDs, preserving example boundaries."""
    digest = hashlib.sha256()
    for labels in label_rows:
        target = [int(token_id) for token_id in labels if token_id != -100]
        digest.update(len(target).to_bytes(8, "little"))
        for token_id in target:
            if not 0 <= token_id < 2**32:
                raise ValueError("supervised token ID is outside uint32 range")
            digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()
