"""Rendered chat control tokens must reach vLLM without another BOS pass."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import eval_alpaca


@pytest.mark.parametrize("with_bos", [True, False], ids=["llama-bos", "qwen-no-bos"])
@pytest.mark.parametrize('echo', ['correct', 'missing', 'changed'])
def test_alpaca_passes_chat_token_ids_without_retokenization(tmp_path, monkeypatch, with_bos, echo):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[USER]": 2, "[ASSISTANT]": 3,
                  "[END]": 4, "Hello": 5}
    if with_bos:
        vocabulary['<|finetune_right_pad_id|>'] = 6
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_bos:
        backend.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
        eos_token="[END]", additional_special_tokens=["[USER]", "[ASSISTANT]"])
    tokenizer.chat_template = (
        ("{{ bos_token }}" if with_bos else "")
        + "{% for m in messages %}[USER]{{ m['content'] }}[END]{% endfor %}"
          "{% if add_generation_prompt %}[ASSISTANT]{% endif %}")
    expected = [1, 2, 5, 4, 3] if with_bos else [2, 5, 4, 3]
    prompts_seen, sampling_seen = [], []

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params, **kwargs):
            prompts_seen.extend(prompts)
            sampling_seen.append(params)
            rows = []
            for prompt in prompts:
                row = SimpleNamespace(outputs=[SimpleNamespace(
                    text="Hello", token_ids=[5, 4], finish_reason="stop", stop_reason=4)])
                if echo != 'missing':
                    row.prompt_token_ids = (prompt['prompt_token_ids'] if echo == 'correct'
                                            else [1, *prompt['prompt_token_ids']])
                rows.append(row)
            return rows

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(
        LoRARequest=lambda *args: args))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(vocab_size=len(vocabulary)))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"instruction": "Hello"}) + "\n")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["eval_alpaca", "--model", str(model),
        "--dataset", str(dataset), "--output", str(output), "--max-tokens", "8"])
    if echo != 'correct':
        with pytest.raises(ValueError, match='prompt token'):
            eval_alpaca.main()
        assert not (output / 'model/manifest_t1.0_n1.json').exists()
        return
    eval_alpaca.main()
    assert prompts_seen == [{"prompt_token_ids": expected}]
    # Dedicated Llama PAD is banned; Qwen's EOS fallback must remain allowed.
    assert sampling_seen[0].logit_bias == ({6: -float('inf')} if with_bos else {})
    cache = json.loads((output / "model/manifest_t1.0_n1.json").read_text())
    assert cache["config"]["sources"]["scripts/eval_alpaca.py"] == hashlib.sha256(
        Path(eval_alpaca.__file__).read_bytes()).hexdigest()
    from scripts.eval_artifacts import digest
    assert cache['config']['prompt_token_ids_sha256'] == digest([expected])
    assert cache['config']['tokenizer'] == {
        'source': str(model), 'fingerprint': cache['config']['model']}
    eval_alpaca.main()
    assert len(prompts_seen) == 1


def test_alpaca_real_llama_padding_128004_is_suppressed(tmp_path, monkeypatch):
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
    base = Path('/data/VPO-RM/models/Llama-3.1-8B-Instruct')
    if not (base / 'tokenizer.json').is_file():
        pytest.skip('Local production Llama tokenizer artifacts required')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base, local_files_only=True)
    assert tokenizer.pad_token_id is None
    sampling = []
    class Engine:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params, **kwargs):
            sampling.append(params)
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'], outputs=[SimpleNamespace(
                text='Hello', token_ids=[9906, 128009], finish_reason='stop', stop_reason=128009)])
                for p in prompts]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *args: args))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(vocab_size=128256))
    dataset = tmp_path / 'data.jsonl'
    dataset.write_text(json.dumps({'instruction': 'Hello'}) + '\n')
    model = tmp_path / 'model'; model.mkdir(); (model / 'config.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['eval_alpaca', '--model', str(model), '--dataset', str(dataset),
        '--output', str(tmp_path / 'out'), '--policy-head-dtype', 'float32'])
    eval_alpaca.main()
    assert sampling[0].logit_bias == {128004: -float('inf')}
    assert sampling[0].stop_token_ids == [128001, 128008, 128009]


def test_alpaca_tokenizer_override_renders_prompts_and_records_its_source(tmp_path, monkeypatch):
    """A base checkpoint without a chat template renders with the saved SFT tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
    from scripts.eval_artifacts import fingerprint

    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[USER]": 2, "[ASSISTANT]": 3, "[END]": 4, "Hello": 5,
                  "<|finetune_right_pad_id|>": 6}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
        eos_token="[END]", additional_special_tokens=["[USER]", "[ASSISTANT]"])
    tokenizer.chat_template = ("{{ bos_token }}{% for m in messages %}[USER]{{ m['content'] }}[END]"
                               "{% endfor %}{% if add_generation_prompt %}[ASSISTANT]{% endif %}")
    sources_seen, prompts_seen = [], []

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params, **kwargs):
            prompts_seen.extend(prompts)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                text="Hello", token_ids=[5, 4], finish_reason="stop", stop_reason=4)]) for p in prompts]

    def from_pretrained(source, *args, **kwargs):
        sources_seen.append(str(source))
        return tokenizer

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *args: args))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(vocab_size=len(vocabulary)))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    saved_tokenizer = tmp_path / "sft-adapter"
    saved_tokenizer.mkdir()
    (saved_tokenizer / "tokenizer_config.json").write_text('{"chat_template": "pinned"}')
    (saved_tokenizer / "adapter_model.safetensors").write_bytes(b"sft weights")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"instruction": "Hello"}) + "\n")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["eval_alpaca", "--model", str(model), "--tokenizer", str(saved_tokenizer),
        "--dataset", str(dataset), "--output", str(output), "--max-tokens", "8"])
    eval_alpaca.main()
    assert sources_seen == [str(saved_tokenizer)]
    assert prompts_seen == [{"prompt_token_ids": [1, 2, 5, 4, 3]}]
    cache = json.loads((output / "model/manifest_t1.0_n1.json").read_text())
    expected = fingerprint(saved_tokenizer, full_weights=False)
    assert cache["config"]["tokenizer"] == {"source": str(saved_tokenizer), "fingerprint": expected}
    assert [row["name"] for row in expected["files"]] == ["adapter_model.safetensors", "tokenizer_config.json"]
    # Adapter weights beside the tokenizer bind by stat identity, like eval_ifeval/eval_checkpoints.
    assert "sha256" not in expected["files"][0] and "sha256" in expected["files"][1]
    assert cache["config"]["tokenizer"]["source"] != cache["config"]["model"]["path"]
