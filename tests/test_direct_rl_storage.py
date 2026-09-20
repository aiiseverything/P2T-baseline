import hashlib
import json

import pytest


def final_run(root, step=250):
    """Small on-disk fixture; these bytes stand for already validated tensors."""
    checkpoint = root / f'checkpoint-{step}'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'validated adapter tensors')
    (checkpoint / 'adapter_config.json').write_text(json.dumps({'r': 64, 'lora_alpha': 128}))
    (checkpoint / 'tokenizer.json').write_text('{"version": "1.0"}')
    (checkpoint / 'tokenizer_config.json').write_text('{"tokenizer_class": "TestTokenizer"}')
    (checkpoint / 'trainer_state.pt').write_bytes(b'validated Adam state')
    (checkpoint / 'run_manifest.json').write_text(json.dumps({
        'step': step, 'resolved_config': {'output_dir': str(root), 'init_adapter': ''}}))
    exports = root / 'vllm-adapters'
    for number in (0, step):
        exported = exports / f'step-{number}'
        exported.mkdir(parents=True)
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            (exported / name).write_bytes((checkpoint / name).read_bytes())
    (root / 'profile_summary.json').write_text('{"completed": true}')
    return checkpoint, exports


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file() and not path.is_symlink()}


def test_cleanup_removes_only_exports_after_verifying_final_copy(tmp_path):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    before = snapshot(checkpoint)
    removed_bytes = sum(path.stat().st_size for path in exports.rglob('*') if path.is_file())

    result = verify_and_cleanup_exports(root, 250)

    assert not exports.exists()
    assert snapshot(checkpoint) == before
    assert (root / 'profile_summary.json').read_text() == '{"completed": true}'
    assert result['checkpoint'] == str(checkpoint)
    assert result['step'] == 250
    assert result['adapter_sha256'] == hashlib.sha256(b'validated adapter tensors').hexdigest()
    assert result['removed_paths'] == [str(exports)]
    assert result['removed_bytes'] == removed_bytes
    json.dumps(result)


@pytest.mark.parametrize('name', ['adapter_model.safetensors', 'adapter_config.json',
                                 'tokenizer.json', 'tokenizer_config.json',
                                 'trainer_state.pt', 'run_manifest.json'])
@pytest.mark.parametrize('damage', ['missing', 'empty'])
def test_incomplete_final_never_deletes_exports(tmp_path, name, damage):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    if damage == 'missing':
        (checkpoint / name).unlink()
    else:
        (checkpoint / name).write_bytes(b'')
    before = snapshot(exports)

    with pytest.raises(ValueError, match='checkpoint|nonempty'):
        verify_and_cleanup_exports(root, 250)

    assert snapshot(exports) == before


@pytest.mark.parametrize('step', [249, '250', True])
def test_final_manifest_wrong_step_never_deletes_exports(tmp_path, step):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    manifest = json.loads((checkpoint / 'run_manifest.json').read_text())
    manifest['step'] = step
    (checkpoint / 'run_manifest.json').write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match='step'):
        verify_and_cleanup_exports(root, 250)

    assert (exports / 'step-250/adapter_model.safetensors').exists()


def test_additional_checkpoint_never_deletes_any_output(tmp_path):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    _, exports = final_run(root)
    (root / 'checkpoint-50').mkdir()
    before = snapshot(root)

    with pytest.raises(ValueError, match='checkpoint'):
        verify_and_cleanup_exports(root, 250)

    assert snapshot(root) == before
    assert exports.exists()
    assert (root / 'checkpoint-50').is_dir()


@pytest.mark.parametrize('name', ['adapter_model.safetensors', 'adapter_config.json'])
def test_final_export_mismatch_never_deletes_exports(tmp_path, name):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    _, exports = final_run(root)
    (exports / 'step-250' / name).write_bytes(b'{}' if name.endswith('.json') else b'different tensors')
    before = snapshot(exports)

    with pytest.raises(ValueError, match='match|differ'):
        verify_and_cleanup_exports(root, 250)

    assert snapshot(exports) == before


def test_equivalent_adapter_config_json_is_accepted(tmp_path):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    _, exports = final_run(root)
    (exports / 'step-250/adapter_config.json').write_text('{"lora_alpha":128,"r":64}')

    verify_and_cleanup_exports(root, 250)

    assert not exports.exists()


@pytest.mark.parametrize('location', ['root', 'ancestor', 'exports', 'nested', 'checkpoint'])
def test_symlinks_are_rejected_without_deleting_targets(tmp_path, location):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'keep.txt'
    sentinel.write_text('untouched')
    supplied = root
    if location == 'root':
        supplied = tmp_path / 'train-link'
        supplied.symlink_to(root, target_is_directory=True)
    elif location == 'ancestor':
        alias = tmp_path / 'alias'
        alias.symlink_to(tmp_path, target_is_directory=True)
        supplied = alias / 'train'
    elif location == 'exports':
        exports.rename(root / 'saved-exports')
        exports.symlink_to(outside, target_is_directory=True)
    elif location == 'nested':
        (exports / 'step-250/external').symlink_to(outside, target_is_directory=True)
    else:
        original = checkpoint / 'trainer_state.pt'
        original.unlink()
        original.symlink_to(sentinel)

    with pytest.raises(ValueError, match='symlink'):
        verify_and_cleanup_exports(supplied, 250)

    assert sentinel.read_text() == 'untouched'
    assert exports.exists()
    assert checkpoint.exists()


def test_manifest_output_directory_must_match_cleanup_root(tmp_path):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    manifest = json.loads((checkpoint / 'run_manifest.json').read_text())
    manifest['resolved_config']['output_dir'] = str(tmp_path / 'outside')
    (checkpoint / 'run_manifest.json').write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match='output|outside'):
        verify_and_cleanup_exports(root, 250)

    assert exports.exists()


@pytest.mark.parametrize('config', [None, [], 'not a config'])
def test_malformed_manifest_config_fails_before_deletion(tmp_path, config):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    checkpoint, exports = final_run(root)
    (checkpoint / 'run_manifest.json').write_text(json.dumps({
        'step': 250, 'resolved_config': config}))

    with pytest.raises(ValueError, match='config'):
        verify_and_cleanup_exports(root, 250)

    assert exports.exists()


@pytest.mark.parametrize('rollouts', [0, -1, True, '250'])
def test_invalid_expected_step_never_deletes_exports(tmp_path, rollouts):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    root = tmp_path / 'train'
    _, exports = final_run(root)

    with pytest.raises(ValueError, match='positive integer'):
        verify_and_cleanup_exports(root, rollouts)

    assert exports.exists()


def preflight_run(suite):
    preflight = suite / 'gpu-preflight'
    for arm in ('grpo', 'lam4'):
        final_run(preflight / arm, step=2)
        (preflight / arm / 'preflight-arm.json').write_text('{"status": "passed"}')
    report = {'status': 'passed', 'arms': {
        'grpo': {'status': 'passed', 'rollouts': 2},
        'lam4': {'status': 'passed', 'rollouts': 2}}}
    (preflight / 'gpu-validation.json').write_text(json.dumps(report))
    (preflight / 'grpo/length_reward_calibration.json').write_text('{"sigma0": 0.2}')
    return preflight


def test_preflight_cleanup_preserves_evidence_and_formal_runs(tmp_path):
    from scripts.direct_rl_storage import cleanup_preflight_artifacts
    suite = tmp_path / 'family'
    preflight = preflight_run(suite)
    formal, _ = final_run(suite / 'grpo/train')
    before = snapshot(suite)
    removed = [preflight / arm / name for arm in ('grpo', 'lam4')
               for name in ('checkpoint-2', 'vllm-adapters')]

    result = cleanup_preflight_artifacts(preflight, suite_dir=suite)

    assert result['removed_paths'] == [str(path) for path in removed]
    assert result['removed_bytes'] > 0
    assert all(not path.exists() for path in removed)
    for name, contents in before.items():
        path = suite / name
        if not any(directory in path.parents for directory in removed):
            assert path.read_bytes() == contents
    assert formal.exists()
    assert result['validation_sha256'] == hashlib.sha256(
        (preflight / 'gpu-validation.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['failed', 'missing_arm', 'extra_arm', 'failed_arm'])
def test_invalid_preflight_report_does_not_delete_any_artifacts(tmp_path, damage):
    from scripts.direct_rl_storage import cleanup_preflight_artifacts
    suite = tmp_path / 'family'
    preflight = preflight_run(suite)
    path = preflight / 'gpu-validation.json'
    report = json.loads(path.read_text())
    if damage == 'failed':
        report['status'] = 'failed'
    elif damage == 'missing_arm':
        del report['arms']['lam4']
    elif damage == 'extra_arm':
        report['arms']['lam8'] = {'status': 'passed'}
    else:
        report['arms']['lam4']['status'] = 'failed'
    path.write_text(json.dumps(report))
    before = snapshot(preflight)

    with pytest.raises(ValueError, match='preflight|arm'):
        cleanup_preflight_artifacts(preflight, suite_dir=suite)

    assert snapshot(preflight) == before


def test_preflight_outside_suite_is_rejected_without_deletion(tmp_path):
    from scripts.direct_rl_storage import cleanup_preflight_artifacts
    suite = tmp_path / 'family'
    suite.mkdir()
    outside = preflight_run(tmp_path / 'other-family')
    before = snapshot(outside)

    with pytest.raises(ValueError, match='outside|suite'):
        cleanup_preflight_artifacts(outside, suite_dir=suite)

    assert snapshot(outside) == before


def test_bad_second_preflight_arm_preserves_first_arm_artifacts(tmp_path):
    from scripts.direct_rl_storage import cleanup_preflight_artifacts
    suite = tmp_path / 'family'
    preflight = preflight_run(suite)
    outside = tmp_path / 'keep.bin'
    outside.write_bytes(b'outside evidence')
    (preflight / 'lam4/vllm-adapters/outside').symlink_to(outside)
    before = snapshot(preflight)

    with pytest.raises(ValueError, match='symlink'):
        cleanup_preflight_artifacts(preflight, suite_dir=suite)

    assert snapshot(preflight) == before
    assert outside.read_bytes() == b'outside evidence'
    assert (preflight / 'grpo/checkpoint-2').exists()
