import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def assets(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.direct_rl_launcher')
    actor, reward = tmp_path / 'Qwen3-14B', tmp_path / 'reward'
    actor.mkdir(); reward.mkdir()
    (reward / 'config.json').write_text('{"unchanged": true}')
    names = [f'model-{i:05d}-of-00008.safetensors' for i in range(1, 9)]
    content = {'config.json': json.dumps({'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM']}),
        'generation_config.json': '{}', 'tokenizer.json': '{}', 'tokenizer_config.json': '{}',
        'merges.txt': '#version 0.2', 'vocab.json': '{}', 'LICENSE': 'fixture license',
        'model.safetensors.index.json': json.dumps({'weight_map': {f'layer.{i}': n for i, n in enumerate(names)}})}
    content.update({name: 'weight-fixture-' + name for name in names})
    files = []
    for name, text in content.items():
        p = actor / name; p.write_text(text)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({'path': name, 'size': p.stat().st_size, 'local_sha256': h,
            'source_sha256': h, 'verified_against_upstream': True,
            'upstream_identity_kind': 'sha256', 'upstream_identity': h,
            'source_file_revision': 'mirror-revision'})
    record = {'upstream_repo': 'Qwen/Qwen3-14B',
        'upstream_revision': '40c069824f4251a91eefaf281ebe4c544efd3e18',
        'destination': str(actor), 'verified': True, 'files': files}
    path = actor / 'DOWNLOAD_MANIFEST.json'; path.write_text(json.dumps(record))
    monkeypatch.setattr(m, 'QWEN_MODEL', actor, raising=False)
    config = {'model': str(actor), 'rm': str(reward)}
    baseline = {'validation_identity': {'input_fingerprints': {
        'actor': {'path': 'old-Qwen3-14B-Base', 'files': []},
        'reward': m.fingerprint(reward, full_weights=False)}}}
    return m, config, baseline, path, record


def test_new_posttrained_actor_is_verified_against_actual_shards_not_old_base(assets):
    m, config, baseline, path, record = assets
    result = m.verify_input_assets(config, baseline)
    evidence = result['actor_download']
    assert evidence['upstream_repo'] == 'Qwen/Qwen3-14B'
    assert evidence['upstream_revision'] == '40c069824f4251a91eefaf281ebe4c544efd3e18'
    assert evidence['weight_shards'] == 8
    assert evidence['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['input_fingerprints']['actor']['path'] == config['model']
    assert result['input_fingerprints']['reward'] == baseline['validation_identity']['input_fingerprints']['reward']
    m.validate_actor_download_evidence(config, evidence)


@pytest.mark.parametrize('field,value', [('verified', False), ('upstream_repo', 'Qwen/Qwen3-14B-Base'),
    ('upstream_revision', 'b' * 40), ('destination', '/another/model')])
def test_unverified_or_wrong_upstream_receipt_is_rejected(assets, field, value):
    m, config, baseline, path, record = assets
    record[field] = value; path.write_text(json.dumps(record))
    with pytest.raises(ValueError): m.verify_input_assets(config, baseline)


def test_changed_file_with_same_length_fails_full_hash_audit(assets):
    m, config, baseline, path, record = assets
    shard = Path(config['model']) / 'model-00001-of-00008.safetensors'
    shard.write_bytes(b'X' * shard.stat().st_size)
    with pytest.raises(ValueError, match='hash|SHA'): m.verify_input_assets(config, baseline)


def test_omitted_index_shard_is_rejected_even_when_every_receipt_file_exists(assets):
    m, config, baseline, path, record = assets
    record['files'] = [x for x in record['files'] if x['path'] != 'model-00008-of-00008.safetensors']
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): m.verify_input_assets(config, baseline)


def test_download_receipt_cannot_escape_actor_directory(assets):
    m, config, baseline, path, record = assets
    record['files'][0]['path'] = '../outside'; path.write_text(json.dumps(record))
    with pytest.raises(ValueError): m.verify_input_assets(config, baseline)


def test_wrong_model_architecture_cannot_pass_using_valid_file_hashes(assets):
    m, config, baseline, path, record = assets
    p = Path(config['model']) / 'config.json'
    p.write_text(json.dumps({'architectures': ['LlamaForCausalLM'], 'model_type': 'llama'}))
    row = next(x for x in record['files'] if x['path'] == 'config.json')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    row.update(size=p.stat().st_size, local_sha256=h, source_sha256=h, upstream_identity=h)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='architecture'): m.verify_input_assets(config, baseline)


def test_changed_reward_is_not_implicitly_authorized_by_actor_replacement(assets):
    m, config, baseline, path, record = assets
    (Path(config['rm']) / 'config.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='reward|RM'): m.verify_input_assets(config, baseline)


def test_runtime_download_evidence_detects_changed_manifest(assets):
    m, config, baseline, path, record = assets
    result = m.verify_input_assets(config, baseline)
    record['verified'] = False; path.write_text(json.dumps(record))
    with pytest.raises(ValueError): m.validate_actor_download_evidence(config, result['actor_download'])


def test_runtime_audit_detects_weight_mtime_change_without_rehashing_entire_model(assets):
    m, config, baseline, path, record = assets
    result = m.verify_input_assets(config, baseline)
    shard = Path(config['model']) / 'model-00001-of-00008.safetensors'
    before = shard.stat()
    shard.write_bytes(b'X' * before.st_size)
    # tmpfs timestamps can remain unchanged for writes within one kernel tick.
    # Exercise the runtime metadata guard with an explicitly changed mtime.
    os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert shard.stat().st_mtime_ns != before.st_mtime_ns
    with pytest.raises(ValueError, match='changed since full hashing'):
        m.validate_actor_download_evidence(config, result['actor_download'])


def test_git_blob_metadata_identity_is_checked_against_actual_bytes(assets):
    m, config, baseline, path, record = assets
    row = next(x for x in record['files'] if x['path'] == 'config.json')
    data = (Path(config['model']) / 'config.json').read_bytes()
    row.update(upstream_identity_kind='git_blob_sha1',
        upstream_identity=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
    path.write_text(json.dumps(record))
    assert m.verify_input_assets(config, baseline)['actor_download']['weight_shards'] == 8
    row['upstream_identity'] = '0' * 40; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='upstream identity'):
        m.verify_input_assets(config, baseline)


def test_replacement_rejects_llama_and_old_base_model():
    m = importlib.import_module('scripts.direct_rl_launcher')
    assert set(m.BASELINES) == {'qwen'}
    with pytest.raises(ValueError): m.build_config('llama')
    config = m.build_config('qwen')
    config['model'] = str(m.SHARED / 'models/Qwen3-14B-Base')
    with pytest.raises(ValueError, match='actor|Qwen'): m.validate_config(config)


def test_preparation_creates_only_qwen_two_jobs_with_download_bound_to_manifest(assets, monkeypatch, tmp_path):
    m, asset_config, baseline, path, record = assets
    config = m.build_config('qwen')
    config.update(asset_config)
    dataset = tmp_path / 'dataset.parquet'; dataset.write_bytes(b'fixed dataset fixture')
    config['dataset_path'] = str(dataset)
    baseline['common_config'] = config
    baseline_path = tmp_path / 'baseline.json'; baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(m, 'BASELINES', {'qwen': baseline_path})
    monkeypatch.setattr(m, 'cpu_protocol', lambda actual: {'status': 'passed',
        'formal_prompt_count': 2000, 'formal_prompt_sha256': m.PROMPT_SHA})
    suite, outputs = tmp_path / 'suite', tmp_path / 'outputs'
    prepared = m.prepare(suite, outputs)
    assert set(prepared) == {'qwen'}
    assert not (suite / 'llama').exists()
    plan = json.loads((suite / 'submission-plan.json').read_text())
    assert plan['max_jobs'] == 2 and set(plan['families']) == {'qwen'}
    manifest = m.read_manifest(suite / 'qwen')
    assert manifest['actor_download']['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['common_config']['model'] == asset_config['model']
    assert manifest['common_config']['init_adapter'] == ''
    assert manifest['actor_variant'] == 'official_posttrained'
    for arm in ['grpo', 'lam4']:
        command = json.loads((suite / 'qwen' / arm / 'submit_command.json').read_text())
        assert command[-2:] == ['qwen', arm]
        assert command[command.index('--gpu') + 1] == '3'
