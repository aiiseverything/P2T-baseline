"""SFT contracts for Llama's native assistant ending and dedicated padding."""
import copy
import hashlib
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scripts import sft_response_tokens
from scripts.sft_init import SFTDataset, collate
from vpo_rm.token_policy import get_stop_token_ids


@pytest.fixture(scope="module")
def llama_tokenizer():
    source = Path(os.environ.get(
        "LLAMA_SFT_TOKENIZER_PATH", "/data/VPO-RM/models/Llama-3.1-8B-Instruct"))
    if not (source / "tokenizer_config.json").exists():
        pytest.skip("Local Llama 3.1 tokenizer is not available")
    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def small_tokenizer(*, dedicated_pad=False, existing_pad=False):
    vocab = {"<unk>": 0, "<|endoftext|>": 1, "<|im_end|>": 2,
             "\n": 3, "<pad>": 4}
    if dedicated_pad:
        vocab["<|finetune_right_pad_id|>"] = 5
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<|endoftext|>",
        pad_token="<pad>" if existing_pad else None)


@pytest.mark.parametrize("mode", ["native", "chat_template"])
def test_llama_response_uses_eot_without_added_newline(llama_tokenizer, mode):
    spec = sft_response_tokens.build_response_eos_spec(llama_tokenizer, mode)
    assert (spec.template_eos_id, spec.actual_eos_id, spec.trailing_ids) == (
        128009, 128009, ())
    full = [128000, 128009, 128006, 78191, 128007, 271, 9906, 128009]
    assert sft_response_tokens.rewrite_response_eos(full, 6, spec) == full
    with pytest.raises(ValueError, match="template tail"):
        sft_response_tokens.rewrite_response_eos(full + [198], 6, spec)


def test_llama_generation_stops_on_all_registered_native_endings(llama_tokenizer):
    assert get_stop_token_ids(llama_tokenizer) == (128001, 128008, 128009)


@pytest.mark.parametrize(
    ("dedicated_pad", "existing_pad", "expected"),
    [(True, True, 4), (True, False, 5), (False, False, 1)],
)
def test_sft_padding_preserves_existing_then_prefers_dedicated_token(
        dedicated_pad, existing_pad, expected):
    tokenizer = small_tokenizer(dedicated_pad=dedicated_pad, existing_pad=existing_pad)
    configure = getattr(sft_response_tokens, "configure_sft_pad_token", None)
    assert callable(configure), "SFT padding configuration helper is missing"
    assert configure(tokenizer) == expected
    assert tokenizer.pad_token_id == expected
    assert tokenizer.eos_token_id == 1


def test_sft_padding_rejects_missing_pad_and_eos():
    tokenizer = small_tokenizer()
    tokenizer.eos_token = None
    configure = getattr(sft_response_tokens, "configure_sft_pad_token", None)
    assert callable(configure), "SFT padding configuration helper is missing"
    with pytest.raises(ValueError, match="padding"):
        configure(tokenizer)


@pytest.mark.parametrize(("mode", "last_id"), [("native", 1), ("chat_template", 2)])
def test_qwen_ending_rewrite_still_preserves_prefix_content_and_newline(mode, last_id):
    tokenizer = small_tokenizer()
    spec = sft_response_tokens.build_response_eos_spec(tokenizer, mode)
    assert sft_response_tokens.rewrite_response_eos([2, 7, 2, 8, 2, 3], 2, spec) == [
        2, 7, 2, 8, last_id, 3]


def test_llama_dataset_masks_headers_preserves_eot_and_drops_long_rows(llama_tokenizer):
    tokenizer = copy.deepcopy(llama_tokenizer)
    configure = getattr(sft_response_tokens, "configure_sft_pad_token", None)
    assert callable(configure), "SFT padding configuration helper is missing"
    assert configure(tokenizer) == 128004
    spec = sft_response_tokens.build_response_eos_spec(tokenizer, "native")
    pairs = [("Say hello.", "Hello."), ("Say hi.", "42")]
    ds = SFTDataset(pairs, tokenizer, 4096, spec)
    assert len(ds) == 2 and ds.stats["skipped_boundary"] == 0
    for (prompt, answer), (ids, labels) in zip(pairs, ds.examples):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        assert ids.count(128000) == 1
        assert ids[:len(prefix)] == prefix
        assert labels[:len(prefix)] == [-100] * len(prefix)
        assert tokenizer.decode(labels[len(prefix):-1]) == answer
        assert labels[-1] == 128009
    ids, labels, attention = collate([ds[0], ds[1]], tokenizer.pad_token_id)
    assert bool((attention == 0).any())
    assert bool((ids[attention == 0] == 128004).all())
    assert bool((labels[attention == 0] == -100).all())
    assert int((labels == 128009).sum()) == 2
    cut = SFTDataset(pairs[:1], tokenizer, len(ds[0][0]) - 1, spec)
    assert len(cut) == 0 and cut.stats["skipped_length"] == 1


def test_llama_tokenizer_save_reload_keeps_native_template_and_pad(llama_tokenizer, tmp_path):
    tokenizer = copy.deepcopy(llama_tokenizer)
    configure = getattr(sft_response_tokens, "configure_sft_pad_token", None)
    assert callable(configure), "SFT padding configuration helper is missing"
    configure(tokenizer)
    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.chat_template == tokenizer.chat_template
    assert (reloaded.bos_token_id, reloaded.eos_token_id, reloaded.pad_token_id) == (
        128000, 128009, 128004)
    assert get_stop_token_ids(reloaded) == (128001, 128008, 128009)


def test_tiny_llama_sft_lora_backward_and_adapter_reload(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    base = LlamaForCausalLM(config).to(torch.bfloat16)
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    base = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
        target_modules=targets))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batch = [([1, 3, 4, 5, 2], [-100, -100, -100, 5, 2]),
             ([1, 6, 7, 2], [-100, -100, 7, 2])]
    ids, labels, attention = collate(batch, pad_id=0)
    model.train()
    output = model(input_ids=ids, labels=labels, attention_mask=attention, use_cache=False)
    assert output.logits.dtype == torch.bfloat16
    assert output.loss.dtype == torch.float32 and torch.isfinite(output.loss)
    output.loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert {name.split(".lora_")[0].split(".")[-1] for name, _ in trainable} == set(targets)
    assert all(parameter.dtype == torch.float32 and parameter.grad is not None
               and torch.isfinite(parameter.grad).all() for _, parameter in trainable)
    assert any(bool(parameter.grad.abs().sum() > 0) for _, parameter in trainable)
    torch.optim.AdamW((parameter for _, parameter in trainable), lr=1e-3).step()
    model.eval()
    with torch.no_grad():
        expected = model(ids, attention_mask=attention, use_cache=False).logits
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(adapter_path)
    assert all("lora_" in name for name in load_file(adapter_path / "adapter_model.safetensors"))
    reloaded_base = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_path).eval()
    with torch.no_grad():
        actual = reloaded(ids, attention_mask=attention, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def llama_base_tokenizer():
    source = Path(os.environ.get(
        "LLAMA_BASE_TOKENIZER_PATH", "/data/VPO-RM/models/Llama-3.1-8B"))
    if not (source / "tokenizer_config.json").exists():
        pytest.skip("Local Llama 3.1 base tokenizer is not available")
    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def test_install_chat_template_pins_sha_and_never_replaces_a_different_template():
    install = getattr(sft_response_tokens, "install_chat_template", None)
    assert callable(install), "pinned chat template installer is missing"
    tokenizer = small_tokenizer()
    assert not tokenizer.chat_template
    template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    digest = hashlib.sha256(template.encode()).hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        install(tokenizer, template, "0" * 64)
    assert not tokenizer.chat_template
    assert install(tokenizer, template, digest.upper()) == digest
    assert tokenizer.chat_template == template
    assert install(tokenizer, template) == digest
    with pytest.raises(ValueError, match="different chat template"):
        install(tokenizer, template + " ")
    assert tokenizer.chat_template == template
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="nonempty"):
            install(small_tokenizer(), bad)


def test_llama_base_with_pinned_native_template_supervises_end_of_text(
        llama_tokenizer, llama_base_tokenizer, tmp_path):
    base = copy.deepcopy(llama_base_tokenizer)
    assert not base.chat_template
    assert (base.bos_token_id, base.eos_token_id, base.pad_token_id) == (128000, 128001, None)
    template = llama_tokenizer.chat_template
    digest = hashlib.sha256(template.encode()).hexdigest()
    assert sft_response_tokens.install_chat_template(base, template, digest) == digest
    assert sft_response_tokens.configure_sft_pad_token(base) == 128004
    assert get_stop_token_ids(base) == (128001, 128008, 128009)
    spec = sft_response_tokens.build_response_eos_spec(base, "native")
    assert (spec.template_eos_id, spec.actual_eos_id, spec.trailing_ids) == (128009, 128001, ())

    instruct = copy.deepcopy(llama_tokenizer)
    sft_response_tokens.configure_sft_pad_token(instruct)
    pairs = [("Say hello.", "Hello."), ("Say hi.", "42")]
    ds_base = SFTDataset(pairs, base, 4096, spec)
    ds_inst = SFTDataset(pairs, instruct, 4096,
                         sft_response_tokens.build_response_eos_spec(instruct, "native"))
    assert len(ds_base) == len(ds_inst) == 2
    assert ds_base.stats["response_eos_replaced"] == 2 and ds_base.stats["response_eos_id"] == 128001
    assert ds_base.stats["target_token_sha256"] != ds_inst.stats["target_token_sha256"]
    for (prompt, answer), (ids, labels), (ids_i, labels_i) in zip(pairs, ds_base.examples, ds_inst.examples):
        # Byte-identical native rendering: only the supervised terminator differs.
        assert ids[:-1] == ids_i[:-1] and labels[:-1] == labels_i[:-1]
        assert ids[-1] == labels[-1] == 128001 and ids_i[-1] == 128009
        assert ids.count(128000) == 1 and ids.count(128001) == 1
        prefix = base.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        assert ids[:len(prefix)] == prefix and labels[:len(prefix)] == [-100] * len(prefix)
        assert base.decode(labels[len(prefix):-1]) == answer

    base.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.chat_template == template
    assert (reloaded.bos_token_id, reloaded.eos_token_id, reloaded.pad_token_id) == (
        128000, 128001, 128004)
    assert get_stop_token_ids(reloaded) == (128001, 128008, 128009)
    messages = [{"role": "user", "content": "Say hello."}]
    rendered = reloaded.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = llama_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    assert reloaded(rendered, add_special_tokens=False)["input_ids"] == expected
    assert expected[0] == 128000 and expected.count(128000) == 1
    # The tokenizer post-processor would add a second BOS; renders must never use it.
    assert reloaded(rendered)["input_ids"].count(128000) == 2
