import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_gsm8k, judge_alpaca


@pytest.mark.parametrize('pred,gold', [(71995, 72000), (5600.4, 5600), (1.00001, 1)])
def test_gsm8k_requires_exact_numeric_answer(pred, gold):
    assert not eval_gsm8k.is_correct(pred, gold)


def test_gsm8k_does_not_round_large_integers():
    pred = eval_gsm8k.extract_number('#### 9007199254740993')
    gold = eval_gsm8k.gold_number('#### 9007199254740992')
    assert not eval_gsm8k.is_correct(pred, gold)
    assert eval_gsm8k.is_correct('1234.00', 1234)


@pytest.mark.parametrize('module_name', ['eval_alpaca', 'eval_gsm8k', 'eval_ifeval'])
def test_eval_recipes_reject_temperatures_vllm_would_clamp_or_reject(module_name):
    import importlib
    parse = importlib.import_module(f'scripts.{module_name}').parse_recipe

    assert parse('0:1:1:-1')['temp'] == 0
    assert parse('.01:1:1:-1')['temp'] == 0.01
    assert parse('2:1:1:-1')['temp'] == 2
    for invalid in ('.001:1:1:-1', '2.1:1:1:-1'):
        with pytest.raises(ValueError, match='out-of-range'):
            parse(invalid)


def test_checkpoint_eval_rejects_temperatures_vllm_would_silently_change():
    from scripts.eval_checkpoints import validate_temperature

    assert validate_temperature(0) == 0
    assert validate_temperature(0.01) == 0.01
    assert validate_temperature(2) == 2
    for invalid in (0.001, 2.1, float('nan')):
        with pytest.raises(ValueError, match='temperature'):
            validate_temperature(invalid)


def test_judge_stops_dispatch_at_budget_and_resumes(tmp_path):
    refs = [{'instruction': str(i), 'reference_output': 'ref'} for i in range(12)]
    path = tmp_path / 'generations.jsonl'
    path.write_text('\n'.join(json.dumps({'idx': i, 'instruction': str(i), 'response': 'answer'})
                              for i in range(12)))

    class Relay:
        budget_cny = 2
        calls = 0

        def spent_since_start(self):
            return self.calls

        def judge_call(self, messages):
            self.calls += 1
            return 0.5, {}

    relay = Relay()
    with pytest.raises(RuntimeError, match='BUDGET'):
        judge_alpaca.judge_tag('toy', path, refs, '{instruction} {output_1} {output_2}', relay, None, 2)
    assert relay.calls == 2
    relay.budget_cny = 100
    result = judge_alpaca.judge_tag('toy', path, refs, '{instruction} {output_1} {output_2}', relay, None, 2)
    assert relay.calls == 12  # completed rows are never sent again
    assert result['n_judged'] == 12


def test_judge_rejects_missing_candidate_before_calls(tmp_path):
    path = tmp_path / 'generations.jsonl'
    path.write_text(json.dumps({'idx': 0, 'instruction': 'a', 'response': 'answer'}))
    relay = SimpleNamespace(judge_call=lambda _: pytest.fail('must validate before API calls'))
    refs = [{'instruction': x, 'reference_output': 'ref'} for x in ('a', 'b')]
    with pytest.raises(ValueError, match='coverage'):
        judge_alpaca.judge_tag('toy', path, refs, 'template', relay, None, 1)


def test_judge_template_substitution_never_rewrites_candidate_text():
    template = '{instruction}|{output_1}|{output_2}'
    filled = judge_alpaca.fill_template(
        template, 'ask about {output_1}', 'candidate says {output_2}',
        'reference says {instruction}',
    )
    assert filled == (
        'ask about {output_1}|candidate says {output_2}|reference says {instruction}'
    )


def test_judge_rejects_corrupt_resume_checkpoint(tmp_path):
    refs = [{'instruction': 'a', 'reference_output': 'ref'}]
    path = tmp_path / 'generations.jsonl'
    path.write_text(json.dumps({'idx': 0, 'instruction': 'a', 'response': 'answer'}))

    class Relay:
        budget_cny = 10

        @staticmethod
        def spent_since_start():
            return 0

        @staticmethod
        def judge_call(messages):
            return 0.5, {}

    result = judge_alpaca.judge_tag(
        'toy', path, refs, '{instruction} {output_1} {output_2}', Relay(), None, 1)
    checkpoint = Path(result['annotations'])
    saved = json.loads(checkpoint.read_text())
    saved['rows'][next(iter(saved['rows']))]['preference'] = 2.0
    checkpoint.write_text(json.dumps(saved))

    with pytest.raises(ValueError, match='checkpoint'):
        judge_alpaca.judge_tag(
            'toy', path, refs, '{instruction} {output_1} {output_2}', Relay(), None, 1)


def test_judge_verified_cache_needs_no_relay_and_preserves_prior_verified_tags(
        tmp_path, monkeypatch):
    from scripts.eval_artifacts import commit_cache

    refs = [{'instruction': str(i), 'reference_output': 'ref'} for i in range(2)]
    refs_path = tmp_path / 'refs.jsonl'
    refs_path.write_text('\n'.join(json.dumps(row) for row in refs))
    root = tmp_path / 'gens'
    template = '{instruction} {output_1} {output_2}'
    for tag in ('a', 'b'):
        tag_dir = root / tag
        tag_dir.mkdir(parents=True)
        gen_path = tag_dir / 'generations_t1.0_n1.jsonl'
        gen_path.write_text('\n'.join(json.dumps({
            'idx': i, 'sample_idx': 0, 'instruction': str(i), 'response': tag,
        }) for i in range(2)))
        result_path = tag_dir / 'results_judged.json'
        result_path.write_text(json.dumps({'tag': tag, 'weighted_win_rate': None}))
        config = judge_alpaca.judge_result_config(
            gen_path, refs, template, 0, 'generations_t1.0_n1.jsonl')
        commit_cache(result_path.with_suffix('.manifest.json'), config, [result_path])

    monkeypatch.setattr(judge_alpaca, 'load_template', lambda: template)
    monkeypatch.setattr(
        judge_alpaca, 'Relay',
        lambda *args: pytest.fail('verified cache must not contact the relay'),
    )
    monkeypatch.setenv('LINKAPI_KEY', 'unused')
    common = ['judge', '--gens-root', str(root), '--refs', str(refs_path), '--tags']
    monkeypatch.setattr('sys.argv', [*common, 'a'])
    judge_alpaca.main()
    assert set(json.loads((root / 'judged_summary.json').read_text())) == {'a'}

    monkeypatch.setattr('sys.argv', [*common, 'b'])
    judge_alpaca.main()
    assert set(json.loads((root / 'judged_summary.json').read_text())) == {'a', 'b'}


def test_all_alpaca_samples_survive_serialization():
    from scripts.eval_alpaca import generation_rows
    sample = lambda text, ids: SimpleNamespace(text=text, token_ids=ids, finish_reason='stop', stop_reason=7)
    outputs = [SimpleNamespace(outputs=[sample('long', [1, 2, 7]), sample('short', [7])])]
    rows = generation_rows([{'instruction': 'a'}], outputs, 2)
    assert [(r['idx'], r['sample_idx'], r['response'], r['response_tokens']) for r in rows] == [
        (0, 0, 'long', 3), (0, 1, 'short', 1)]


def test_eval_cache_rejects_changed_config_or_partial_files(tmp_path):
    from scripts.eval_artifacts import cache_matches, commit_cache
    manifest = tmp_path / 'manifest.json'
    output = tmp_path / 'generations.jsonl'
    config = {'recipe': {'top_p': 1}, 'adapter': 'hash1'}
    assert not cache_matches(manifest, config, [output])
    output.write_text('complete')
    commit_cache(manifest, config, [output])
    assert cache_matches(manifest, config, [output])
    with pytest.raises(ValueError, match='different'):
        cache_matches(manifest, {**config, 'adapter': 'hash2'}, [output])
    output.write_text('partial')
    with pytest.raises(ValueError, match='incomplete|changed'):
        cache_matches(manifest, config, [output])


def test_eval_provenance_tracks_sampling_support_implementation(tmp_path):
    from scripts.eval_artifacts import eval_config

    dataset = tmp_path / 'data.jsonl'
    dataset.write_text('{}\n')
    args = SimpleNamespace(dataset=str(dataset), seed=1, max_tokens=8)
    config = eval_config(
        args, {'temp': 1}, {'model': 1}, None, (2,), 'alpaca',
        {'suppressed_token_count': 1, 'suppressed_token_ids_sha256': 'abc'},
    )

    assert {'vpo_rm/integration.py', 'vpo_rm/alignment.py'} <= set(config['sources'])
    assert set(config['runtime_versions']) == {'transformers', 'vllm'}


@pytest.mark.parametrize('benchmark', ['alpaca', 'gsm8k', 'ifeval'])
def test_real_eval_cli_all_samples_and_cache(benchmark, monkeypatch, tmp_path):
    import importlib
    import sys
    module = importlib.import_module('scripts.eval_' + benchmark)
    calls = []
    class Engine:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params, **kwargs):
            calls.append(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text='#### 42', token_ids=[1] * (3 - s),
                         finish_reason='stop', stop_reason=2) for s in range(params.n)]) for _ in prompts]
    class Tokenizer:
        eos_token_id = 2
        all_special_ids = [2]
        def get_vocab(self):
            return {'a': 1, '<eos>': 2}
        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)
    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *args: args))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(vocab_size=3))
    from vpo_rm.trainer import VPOTrainer
    monkeypatch.setattr(VPOTrainer, '_render_chat_prompt', lambda tok, prompt: prompt)
    checker = lambda inp, responses: SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True])
    monkeypatch.setitem(sys.modules, 'instruction_following_eval', SimpleNamespace(
        instructions_registry=SimpleNamespace(INSTRUCTION_DICT={'a': 1}), evaluation_lib=SimpleNamespace(
            test_instruction_following_strict=checker, test_instruction_following_loose=checker)))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'data.jsonl'
    dataset.write_text('\n'.join(json.dumps({'key': i, 'instruction': str(i), 'prompt': str(i),
                                            'question': str(i), 'answer': '#### 42',
                                            'instruction_id_list': ['a:b'], 'kwargs': [{}]}) for i in range(2)))
    output = tmp_path / 'results'
    argv = ['eval', '--model', str(model), '--dataset', str(dataset), '--output', str(output),
            '--recipes', '1:2:1:-1', '--max-tokens', '8']
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    rows = [json.loads(line) for line in (output / 'model/generations_t1.0_n2.jsonl').read_text().splitlines()]
    if benchmark == 'alpaca':
        assert len(rows) == 4 and [r['sample_idx'] for r in rows] == [0, 1, 0, 1]
    else:
        assert len(rows) == 2 and rows[0]['response_tokens'] == [3, 2]
    assert not hasattr(calls[0], 'allowed_token_ids')
    assert calls[0].logit_bias == {0: -float('inf')}
    module.main()
    assert len(calls) == 1
    monkeypatch.setattr(sys, 'argv', argv[:-1] + ['9'])
    with pytest.raises(ValueError, match='different'):
        module.main()
    assert len(calls) == 1


def test_usage_api_errors_fail_closed(monkeypatch):
    relay = object.__new__(judge_alpaca.Relay)
    relay.usage_start = '2026-09-01'
    requests, sampling = [], []
    def get(url, params):
        requests.append(params)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'error': 'denied'})
    relay.client = SimpleNamespace(get=get)
    monkeypatch.setattr(judge_alpaca.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='Cannot verify usage'):
        relay.usage()
    from datetime import datetime, timezone
    assert requests[-1]['end_date'] == datetime.now(timezone.utc).date().isoformat()


def test_checkpoint_adapter_ids_unique_across_runs(monkeypatch, tmp_path):
    import sys
    from scripts import eval_checkpoints as module
    requests, sampling = [], []
    class Engine:
        def __init__(self, **kwargs): pass
        def generate(self, prompts, params, lora_request):
            requests.append(lora_request)
            sampling.append(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])])]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    monkeypatch.setattr(module, 'discover_adapters', lambda path: [(50, path / 'step-50')])
    monkeypatch.setattr(module, '_banned_ids', lambda _: {})
    monkeypatch.setattr(module, '_stop_ids', lambda _: [2], raising=False)
    monkeypatch.setattr(module, 'validate_adapter_base', lambda *a: None, raising=False)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(get_vocab=lambda: {'a': 1, 'b': 2}))
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(vocab_size=3))
    args = SimpleNamespace(model='toy', max_num_seqs=1, seed=1, max_tokens=8)
    module.generate_all([('a', tmp_path / 'a'), ('b', tmp_path / 'b')], ['prompt'], [1.0], args)
    assert len({request[1] for request in requests}) == 2
    assert all(not hasattr(params, 'allowed_token_ids') for params in sampling)
    assert all(params.logit_bias == {0: -float('inf')} for params in sampling)


def test_reject_wrong_adapter_base(tmp_path):
    from scripts.eval_artifacts import validate_adapter_base
    model_a, model_b, adapter = [tmp_path / p for p in ('8B', '14B', 'adapter')]
    for path in (model_a, model_b, adapter): path.mkdir()
    (model_a / 'config.json').write_text(json.dumps({'hidden_size': 4096}))
    (model_b / 'config.json').write_text(json.dumps({'hidden_size': 5120}))
    (adapter / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': str(model_a)}))
    with pytest.raises(ValueError, match='incompatible'):
        validate_adapter_base(adapter, model_b)
    validate_adapter_base(adapter, model_a)
