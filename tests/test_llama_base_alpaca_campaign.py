"""Offline gates for the four-policy Llama-base Alpaca campaign."""
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.eval_artifacts import commit_cache, digest, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head
from scripts import judge_alpaca as judge

TAGS = ('base', 'sft', 'grpo', 'lam4')
STOPS = [128001, 128008, 128009]
TEMPLATE_SHA = 'e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65'
ROOT = Path(__file__).resolve().parents[1]
GEN_SOURCES = ('scripts/eval_artifacts.py', 'scripts/eval_alpaca.py',
               'vpo_rm/token_policy.py', 'vpo_rm/model_identity.py',
               'vpo_rm/trainer.py', 'vpo_rm/integration.py', 'vpo_rm/alignment.py')


def module():
    return importlib.import_module('scripts.llama_base_alpaca_campaign')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


@pytest.fixture
def campaign(tmp_path):
    suite = tmp_path / 'suite'; suite.mkdir()
    refs = [{'instruction': f'question {i}', 'reference_output': 'reference'} for i in range(805)]
    jsonl(suite / 'references.jsonl', refs)
    (suite / 'judge_template.txt').write_text('{instruction}|{output_1}|{output_2}')
    files = set(GEN_SOURCES) | {'scripts/llama_base_alpaca_campaign.py',
        'scripts/llama_base_eval_worker.py', 'scripts/eval_policy.py', 'scripts/judge_alpaca.py'}
    for name in files:
        target = suite / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    base = tmp_path / 'base'; base.mkdir(); save(base / 'config.json', {'model_type': 'llama'})
    sft = tmp_path / 'sft-adapter'
    models = {}
    for tag in TAGS:
        adapter = 'none'; bound = {}
        if tag != 'base':
            path = sft if tag == 'sft' else tmp_path / tag / 'checkpoint-250'
            path.mkdir(parents=True)
            save(path / 'adapter_config.json', {'base_model_name_or_path': str(base)})
            (path / 'adapter_model.safetensors').write_bytes(tag.encode())
            if tag == 'sft':
                save(path / 'tokenizer_config.json', {'chat_template': 'pinned'})
                save(path / 'sft_manifest.json', {'token_protocol': {'bos_token_id': 128000, 'pad_token_id': 128004,
                     'response_eos_id': 128001, 'stop_token_ids': STOPS, 'chat_template_sha256': TEMPLATE_SHA}})
            else:
                save(path / 'run_manifest.json', {'step': 250, 'resolved_config': {'model_name': str(base),
                     'init_adapter': str(sft), 'policy_head_dtype': 'float32',
                     'method': 'grpo' if tag == 'grpo' else 'vpo_rm', 'credit_lambda': 1.0 if tag == 'grpo' else 4.0}})
            adapter = str(path)
            bound = {str(p): file_hash(p) for p in path.iterdir()}
        models[tag] = {'adapter': adapter, 'files_sha256': bound,
                       'policy': resolve_policy_head(adapter, 'float32')}
    manifest = {'benchmark': 'alpaca805', 'base_model': str(base),
        'base_model_fingerprint': fingerprint(base, full_weights=False),
        'tokenizer': str(sft), 'tokenizer_fingerprint': fingerprint(sft, full_weights=False),
        'models': models, 'seeds': [42], 'expected_stop_token_ids': STOPS,
        'actor_pad_token_id': 128004, 'actor_response_eos_id': 128001, 'chat_template_sha256': TEMPLATE_SHA,
        'prompt_token_ids_sha256': 'a' * 64,
        'gpu_versions': {'torch': 'test', 'transformers': 'test', 'vllm': 'test'},
        'judge': {'model': 'gpt-4.1', 'workers': 16, 'budget_cny': 30,
                  'source_sha256': file_hash(ROOT / 'scripts/judge_alpaca.py')},
        'files_sha256': {str(p.relative_to(suite)): file_hash(p)
                        for p in suite.rglob('*') if p.is_file()}}
    save(suite / 'experiment.json', manifest)
    for tag in TAGS:
        item = models[tag]; directory = suite / 'generations' / tag
        rows = [{'idx': i, 'sample_idx': 0, 'instruction': ref['instruction'],
                 'response': 'answer', 'response_tokens': 2, 'finish_reason': 'stop',
                 'stop_reason': 128001, 'last_token_id': 128001} for i, ref in enumerate(refs)]
        path = directory / 'generations_t1.0_n1.jsonl'; jsonl(path, rows)
        cfg = {'protocol': 2, 'scorer': 'alpaca', 'recipe': {'temp': 1., 'n': 1, 'top_p': 1., 'top_k': -1},
            'seed': 42, 'max_tokens': 2048, 'engine': {'dtype': 'bfloat16', 'max_model_len': 4096},
            'model': manifest['base_model_fingerprint'],
            'adapter': None if item['adapter'] == 'none' else fingerprint(item['adapter']),
            'policy': item['policy'], 'dataset': fingerprint(suite / 'references.jsonl'),
            'stop_token_ids': STOPS,
            'tokenizer': {'source': str(sft), 'fingerprint': manifest['tokenizer_fingerprint']},
            'prompt_token_ids_sha256': manifest['prompt_token_ids_sha256'],
            'runtime_versions': {'transformers': 'test', 'vllm': 'test'},
            'sources': {name: file_hash(suite / 'source' / name) for name in GEN_SOURCES}}
        commit_cache(directory / 'manifest_t1.0_n1.json', cfg, [path])
        save(suite / 'job' / tag / 'status.json', {'state': 'complete',
             'experiment_sha256': file_hash(suite / 'experiment.json')})
        (suite / 'job' / tag / 'exit_code').write_text('0\n')
    return SimpleNamespace(suite=suite, manifest=manifest, refs=refs, sft=sft)


def make_judgments(campaign):
    summary = {}
    for tag in TAGS:
        directory = campaign.suite / 'generations' / tag
        gen = directory / 'generations_t1.0_n1.jsonl'
        rows = [json.loads(line) for line in gen.open()]
        template = (campaign.suite / 'judge_template.txt').read_text()
        protocol = digest({'rows': rows, 'refs': campaign.refs, 'template': template,
            'judge': 'gpt-4.1', 'protocol': 'md5-order-logprob-v2',
            'source': file_hash(ROOT / 'scripts/judge_alpaca.py')})
        records = {digest(row): {'instruction': row['instruction'], 'sample_idx': 0,
            'preference': None if i == 0 else (.75 if i % 2 else .25),
            'chars': len(row['response']), 'usage': {}} for i, row in enumerate(rows)}
        annotation = directory / f'annotations_{protocol[:16]}.json'
        save(annotation, {'protocol': protocol, 'rows': records})
        result = {'tag': tag, 'n_judged': 804, 'n_failed_parse': 1,
                  'weighted_win_rate': .5, 'win_rate': .5, 'mean_candidate_chars': 6.,
                  'judge_model': 'gpt-4.1', 'spent_cny': 1., 'limit': None,
                  'annotations': str(annotation), 'judge_protocol': protocol}
        result_path = directory / 'results_judged.json'; save(result_path, result)
        commit_cache(result_path.with_suffix('.manifest.json'),
                     judge.judge_result_config(gen, campaign.refs, template, 0, gen.name), [result_path])
        summary[tag] = result
    save(campaign.suite / 'generations/judged_summary.json', summary)


@pytest.mark.parametrize('tag', TAGS)
def test_commands_render_with_sft_tokenizer_and_bind_one_judge_budget(campaign, tag):
    m = module(); command = m.generation_command(campaign.suite, campaign.manifest, tag)
    assert command[command.index('--adapters') + 1] == f"{tag}={campaign.manifest['models'][tag]['adapter']}"
    assert command[command.index('--tokenizer') + 1] == str(campaign.sft)
    assert command[command.index('--max-tokens') + 1] == '2048'
    assert command[command.index('--policy-head-dtype') + 1] == 'float32'
    command = m.judge_command(campaign.suite, '/judge-python')
    assert command[command.index('--tags') + 1:command.index('--tags') + 5] == list(TAGS)
    assert command[command.index('--workers') + 1] == '16'
    assert command[command.index('--budget-cny') + 1] == '30'
    env = m.judge_environment({'HTTPS_PROXY': 'wrong', 'LINKAPI_KEY': 'keep-private'})
    assert all(env[key] == 'http://127.0.0.1:17891' for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'))
    assert env['NO_PROXY'] == env['no_proxy'] == '127.0.0.1,localhost'
    assert env['LINKAPI_KEY'] == 'keep-private'


@pytest.mark.parametrize('damage', ['tokenizer_is_base', 'base_with_adapter', 'eos', 'template', 'judge_budget'])
def test_manifest_gate_rejects_protocol_drift(campaign, damage):
    manifest = campaign.manifest
    if damage == 'tokenizer_is_base': manifest['tokenizer'] = manifest['base_model']
    elif damage == 'base_with_adapter': manifest['models']['base']['adapter'] = manifest['models']['sft']['adapter']
    elif damage == 'eos': manifest['actor_response_eos_id'] = 128009
    elif damage == 'template': manifest['chat_template_sha256'] = 'c' * 64
    else: manifest['judge']['budget_cny'] = 60
    with pytest.raises(ValueError): module().validate_manifest(manifest)


@pytest.mark.parametrize('tag', TAGS)
def test_generation_gate_accepts_805_and_null_base_adapter(campaign, tag):
    result = module().validate_generation(campaign.suite, tag, campaign.manifest)
    assert result['n'] == 805 and result['mean_tokens'] == 2 and result['capped'] == 0


def test_verify_inputs_binds_training_identity(campaign):
    m = module(); m.verify_inputs(campaign.suite, campaign.manifest)
    adapter = Path(campaign.manifest['models']['lam4']['adapter'])
    record = json.loads((adapter / 'run_manifest.json').read_text())
    record['resolved_config']['init_adapter'] = ''
    save(adapter / 'run_manifest.json', record)
    campaign.manifest['models']['lam4']['files_sha256'][str(adapter / 'run_manifest.json')] = file_hash(adapter / 'run_manifest.json')
    # The policy-metadata hash binding fires first; either gate rejects the changed initialization.
    with pytest.raises(ValueError, match='SFT|precision'): m.verify_inputs(campaign.suite, campaign.manifest, 'lam4')


@pytest.mark.parametrize('damage', ['hash', 'missing_row', 'idx', 'sample', 'empty', 'metadata', 'vocab',
                                  'recipe', 'head', 'tokenizer', 'prompt_hash', 'model', 'sources'])
def test_generation_gate_rejects_corruption_even_when_rehashed(campaign, damage):
    m = module(); directory = campaign.suite / 'generations/base'
    path = directory / 'generations_t1.0_n1.jsonl'
    rows = [json.loads(line) for line in path.open()]
    cache_path = directory / 'manifest_t1.0_n1.json'; cfg = json.loads(cache_path.read_text())['config']
    if damage == 'missing_row': rows.pop()
    elif damage == 'idx': rows[0]['idx'] = 1
    elif damage == 'sample': rows[0]['sample_idx'] = 1
    elif damage == 'empty': rows[0]['response'] = None
    elif damage == 'metadata': rows[0].pop('finish_reason')
    elif damage == 'vocab': rows[0]['last_token_id'] = 128256
    elif damage == 'recipe': cfg['recipe']['temp'] = .7
    elif damage == 'head': cfg['policy']['policy_head_dtype'] = 'native'
    elif damage == 'tokenizer': cfg['tokenizer']['source'] = campaign.manifest['base_model']
    elif damage == 'prompt_hash': cfg['prompt_token_ids_sha256'] = 'b' * 64
    elif damage == 'model': cfg['model'] = {}
    elif damage == 'sources': cfg['sources']['scripts/eval_alpaca.py'] = 'b' * 64
    jsonl(path, rows)
    if damage != 'hash': commit_cache(cache_path, cfg, [path])
    else: path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError): m.validate_generation(campaign.suite, 'base', campaign.manifest)


def test_supplied_expected_output_support_is_checked(campaign):
    campaign.manifest['expected_output_support'] = {'suppressed_count': 1, 'suppressed_sha256': 'expected'}
    with pytest.raises(ValueError, match='support'):
        module().validate_generation(campaign.suite, 'base', campaign.manifest)


def test_input_hash_failure_records_worker_failed_without_gpu_or_child(campaign, monkeypatch):
    m = module(); (campaign.suite / 'references.jsonl').write_text('changed')
    monkeypatch.setattr(m, '__file__', str(campaign.suite / 'source/scripts/llama_base_alpaca_campaign.py'))
    monkeypatch.setattr(m, 'runtime_preflight', lambda *a: pytest.fail('no GPU after hash failure'))
    with pytest.raises(ValueError): m.worker(campaign.suite, 'base')
    state = json.loads((campaign.suite / 'job/base/status.json').read_text())
    assert state['state'] == 'failed'


@pytest.mark.parametrize('initial', ['judge_started', 'failed'])
def test_watcher_never_restarts_a_previously_started_or_failed_judge(campaign, monkeypatch, initial):
    m = module(); save(campaign.suite / 'judge/state.json', {'state': initial})
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('must not repeat paid invocation'))
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == initial


def test_watcher_waits_for_all_four_exit_codes(campaign, monkeypatch):
    m = module(); (campaign.suite / 'job/lam4/exit_code').unlink()
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('incomplete generation'))
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == 'waiting'


@pytest.mark.parametrize('damage', ['worker_exit', 'generation_hash', 'source_hash'])
def test_watcher_fails_before_paid_call_on_bad_generation_or_source(campaign, monkeypatch, damage):
    m = module()
    if damage == 'worker_exit': (campaign.suite / 'job/base/exit_code').write_text('1')
    elif damage == 'generation_hash': (campaign.suite / 'generations/base/generations_t1.0_n1.jsonl').write_text('bad')
    else: (campaign.suite / 'source/scripts/eval_alpaca.py').write_text('bad')
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('bad inputs reached paid call'))
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == 'failed'


def test_watcher_persists_started_before_one_paid_call_and_verifies_all_results(campaign, monkeypatch):
    m = module(); observed = []
    def run(command, **kwargs):
        assert json.loads((campaign.suite / 'judge/state.json').read_text())['state'] == 'judge_started'
        observed.append(command); make_judgments(campaign)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, 'run', run)
    state = m.watch_once(campaign.suite, '/judge-python')
    assert state['state'] == 'complete' and len(observed) == 1
    summary = json.loads((campaign.suite / 'summary.json').read_text())
    assert summary['n_annotations'] == 4 * 805 and summary['n_failed_parse'] == 4
    assert all(row['weighted_win_rate'] == .5 and row['n_judged'] == 804 for row in summary['models'])
    assert (campaign.suite / 'summary.csv').is_file() and (campaign.suite / 'completion.json').is_file()
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == 'complete'
    assert len(observed) == 1


def test_judge_failure_is_terminal_and_preserves_progress(campaign, monkeypatch):
    m = module(); calls = []
    def run(*args, **kwargs):
        calls.append(1); save(campaign.suite / 'generations/base/annotations_partial.json', {'paid': True})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == 'failed'
    assert m.watch_once(campaign.suite, '/judge-python')['state'] == 'failed'
    assert len(calls) == 1 and (campaign.suite / 'generations/base/annotations_partial.json').exists()
    assert not (campaign.suite / 'completion.json').exists()


@pytest.mark.parametrize('damage', ['coverage', 'preference', 'chars', 'mean', 'count', 'annotation_hash'])
def test_result_gate_recomputes_annotations_and_rejects_bad_results(campaign, damage):
    m = module(); make_judgments(campaign)
    directory = campaign.suite / 'generations/base'; result_path = directory / 'results_judged.json'
    result = json.loads(result_path.read_text()); annotation = Path(result['annotations'])
    annotations = json.loads(annotation.read_text()); key = next(iter(annotations['rows']))
    if damage == 'coverage': del annotations['rows'][key]
    elif damage == 'preference': annotations['rows'][key]['preference'] = 2
    elif damage == 'chars': annotations['rows'][key]['chars'] = 0
    elif damage == 'mean': result['weighted_win_rate'] = .7
    elif damage == 'count': result['n_failed_parse'] = 0
    else: annotations['protocol'] = 'changed'
    save(annotation, annotations); save(result_path, result)
    cache = json.loads(result_path.with_suffix('.manifest.json').read_text())
    commit_cache(result_path.with_suffix('.manifest.json'), cache['config'], [result_path])
    with pytest.raises(ValueError): m.validate_judgment(campaign.suite, 'base', campaign.manifest)
