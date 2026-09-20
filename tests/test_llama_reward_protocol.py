"""Real Llama tokenizers expose BOS and trim failures hidden by Qwen fixtures."""
from copy import deepcopy
from pathlib import Path

import pytest

from vpo_rm.reward_inputs import build_reward_input, canonical_reward_input


@pytest.fixture(scope="module")
def llama_tokenizers():
    from transformers import PreTrainedTokenizerFast
    root = Path("/data/VPO-RM/models")
    paths = (root / "sft-llama31-8b-instruct-clean2k5e2-20260917",
             root / "Skywork-Reward-Llama-3.1-8B-v0.2")
    if not all((path / "tokenizer.json").is_file() for path in paths):
        pytest.skip("Local Llama tokenizer artifacts are required")
    # TF4 cannot AutoTokenizer-load TF5's TokenizersBackend class name. The
    # explicit fast loader reads the exact saved backend/template, not a substitute.
    return tuple(PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
                 for path in paths)


@pytest.mark.parametrize("text", ["Hello world!", "café 中文🙂", "repeat repeat repeat"])
def test_llama_bos_preserves_exact_content_gradient_positions(llama_tokenizers, text):
    actor, rm = llama_tokenizers
    ids = actor.encode(text, add_special_tokens=False) + [128009]
    result = build_reward_input(actor, rm, "Say hello", ids)
    assert all(position >= 0 for position in result.response_positions[:-1])
    assert result.response_positions[-1] == -1
    assert [result.input_ids[p] for p in result.response_positions[:-1]] == ids[:-1]


@pytest.mark.parametrize("leading,trailing", [("", "\n"), ("\t", ""),
                                               ("\n", "\n"), ("  ", "  ")])
def test_llama_trim_only_unmaps_removed_whitespace(llama_tokenizers, leading, trailing):
    actor, rm = llama_tokenizers
    before = actor.encode(leading, add_special_tokens=False)
    content = actor.encode("Hello world!", add_special_tokens=False)
    after = actor.encode(trailing, add_special_tokens=False)
    result = build_reward_input(actor, rm, "Repeat", before + content + after + [128009])
    assert result.response_positions[:len(before)] == [-1] * len(before)
    mapped = result.response_positions[len(before):len(before) + len(content)]
    assert all(position >= 0 for position in mapped)
    assert [result.input_ids[p] for p in mapped] == content
    assert result.response_positions[len(before) + len(content):] == [-1] * (len(after) + 1)


def test_llama_trim_does_not_map_partial_token(llama_tokenizers):
    actor, rm = llama_tokenizers
    ids = actor.encode(" Hello world! ", add_special_tokens=False) + [128009]
    result = build_reward_input(actor, rm, "Repeat", ids)
    assert ids[0] == 22691  # The leading space belongs to the Hello token.
    assert result.response_positions[0] == -1
    assert result.response_positions[1] >= 0  # Unchanged " world" survives.
    assert result.response_positions[-2:] == [-1, -1]


@pytest.mark.parametrize("text", ["", "  ", "\n\nHello\n", "café 中文🙂",
                                  "__VPO_RM_RESPONSE_BOUNDARY_9f174__"])
@pytest.mark.parametrize("stop", [128001, 128008, 128009])
def test_llama_complete_chat_equals_model_card(llama_tokenizers, text, stop):
    actor, rm = llama_tokenizers
    ids = actor.encode(text, add_special_tokens=False) + [stop]
    result = build_reward_input(actor, rm, "Repeat", ids)
    official = rm.apply_chat_template([
        {"role": "user", "content": "Repeat"},
        {"role": "assistant", "content": result.response_text}], tokenize=True, return_dict=False)
    assert result.input_ids == official
    assert result.input_ids.count(128000) == 1
    assert result.input_ids[-1] == 128009
    assert result.response_positions[-1] == -1


def test_llama_mapping_uses_reward_template_when_actor_template_differs(llama_tokenizers):
    original, rm = llama_tokenizers
    actor = deepcopy(original)
    actor.chat_template = "{{ bos_token }}{{ messages[0]['content'] }}"
    ids = actor.encode("Hello world!", add_special_tokens=False) + [128009]
    result = build_reward_input(actor, rm, "Hello world!", ids)
    assert all(p >= 0 for p in result.response_positions[:-1])
    assert result.response_positions[-1] == -1
    decoded = rm.decode(result.input_ids)
    assert "Cutting Knowledge Date: December 2023" in decoded
    assert "Today Date: 26 Jul 2024" in decoded


@pytest.mark.parametrize("special", [128001, 128006, 128008, 128009])
def test_llama_backend_specials_do_not_disable_other_token_gradients(llama_tokenizers, special):
    actor, rm = llama_tokenizers
    result = build_reward_input(actor, rm, "Say hello", [9906, special])
    assert result.response_text == "Hello"
    assert result.response_positions == [37, -1]


def test_llama_serialization_does_not_silently_truncate_at_tokenizer_limit(llama_tokenizers):
    _, rm = llama_tokenizers
    text = " word" * 4096
    ids = canonical_reward_input(rm, "Repeat", text)
    official = rm.apply_chat_template([{"role": "user", "content": "Repeat"},
                                      {"role": "assistant", "content": text}], tokenize=True, return_dict=False)
    assert len(ids) > 4096
    assert ids == official
    assert ids[-1] == 128009


def test_qwen_style_template_retains_leading_newline_mapping():
    from bytelevel_fixtures import ByteLevelTestTokenizer
    tok = ByteLevelTestTokenizer({"q": 0, "H": 1, "i": 2, "\n": 3,
                                 "<|endoftext|>": 4, "<|im_end|>": 5})
    tok.chat_template = ("{{ messages[0]['content'] }}"
                         "{{ messages[1]['content'].lstrip('\\n') }}<|im_end|>")
    result = build_reward_input(tok, tok, "q", [3, 3, 1, 2, 4])
    assert result.input_ids == [0, 1, 2, 5]
    assert result.response_positions == [-1, -1, 1, 2, -1]
