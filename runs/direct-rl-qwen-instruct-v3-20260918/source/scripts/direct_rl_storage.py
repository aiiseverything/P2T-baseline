"""Remove this direct-RL run's redundant weights after successful validation.

The launcher owns process shutdown and full tensor/optimizer validation. These
helpers independently check the on-disk checkpoint and export before deletion.
They never remove a formal checkpoint or the reports needed to audit a run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat


def _directory(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError(f'Parent traversal is not allowed in cleanup paths: {path}')
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f'Cleanup path contains a symlink: {path}')
    if not path.is_dir():
        raise ValueError(f'Cleanup directory does not exist: {path}')
    return path


def _tree_bytes(root):
    """Validate the entire tree before any caller starts deleting directories."""
    total = 0
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f'Cleanup tree contains a symlink: {path}')
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Cleanup path is outside its root: {path}')
        if stat.S_ISREG(mode):
            total += path.stat().st_size
        elif not stat.S_ISDIR(mode):
            raise ValueError(f'Cleanup tree contains a non-regular file: {path}')
    return total


def _nonempty(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Expected nonempty checkpoint/export file: {path}')


def _json_object(path):
    _nonempty(path)
    try:
        result = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid checkpoint/report JSON: {path}') from error
    if not isinstance(result, dict):
        raise ValueError(f'Expected a checkpoint/report JSON object: {path}')
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exports(train_dir, expected_rollouts):
    if type(expected_rollouts) is not int or expected_rollouts < 1:
        raise ValueError('expected_rollouts must be a positive integer')
    root = _directory(train_dir)
    _tree_bytes(root)
    checkpoint = root / f'checkpoint-{expected_rollouts}'
    if set(root.glob('checkpoint-*')) != {checkpoint} or not checkpoint.is_dir():
        raise ValueError('Expected exactly the final checkpoint and no intermediate checkpoints')
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'trainer_state.pt', 'run_manifest.json'):
        _nonempty(checkpoint / name)
    manifest = _json_object(checkpoint / 'run_manifest.json')
    if type(manifest.get('step')) is not int or manifest['step'] != expected_rollouts:
        raise ValueError('Final checkpoint manifest step differs from expected_rollouts')
    config = manifest.get('resolved_config')
    if not isinstance(config, dict):
        raise ValueError('Final checkpoint resolved_config must be an object')
    output = config.get('output_dir')
    if not isinstance(output, str) or Path(output).absolute() != root:
        raise ValueError('Final checkpoint output_dir differs from the cleanup root')
    exports = _directory(root / 'vllm-adapters')
    exported = _directory(exports / f'step-{expected_rollouts}')
    _nonempty(exported / 'adapter_model.safetensors')
    adapter_sha256 = _sha256(checkpoint / 'adapter_model.safetensors')
    if _sha256(exported / 'adapter_model.safetensors') != adapter_sha256:
        raise ValueError('Final exported adapter tensors differ from the checkpoint')
    if _json_object(exported / 'adapter_config.json') != _json_object(checkpoint / 'adapter_config.json'):
        raise ValueError('Final exported adapter config differs from the checkpoint')
    return root, checkpoint, exports, adapter_sha256


def verify_and_cleanup_exports(train_dir, expected_rollouts):
    """After a successful profile process, retain its verified final checkpoint.

    Call only after the launcher has validated completion and the vLLM process
    has exited. A verification failure deletes nothing. Filesystem deletion
    errors propagate, so the launcher must not mark such a run complete.
    """
    _, checkpoint, exports, digest = _verify_exports(train_dir, expected_rollouts)
    removed_bytes = _tree_bytes(exports)
    shutil.rmtree(exports)
    return {'checkpoint': str(checkpoint), 'step': expected_rollouts,
            'adapter_sha256': digest, 'removed_paths': [str(exports)],
            'removed_bytes': removed_bytes}


def cleanup_preflight_artifacts(preflight_dir, *, suite_dir):
    """Remove only the two successful arms' checkpoint-2 and adapter exports.

    The launcher verifies the report's provenance/hash before calling this.
    Validate every deletion target first; preserve JSON reports, calibration,
    rollout evidence and logs, and do not touch the formal training directories.
    """
    suite = _directory(suite_dir)
    preflight = _directory(preflight_dir)
    if preflight != suite / 'gpu-preflight':
        raise ValueError('Preflight cleanup must use this suite/gpu-preflight directory')
    _tree_bytes(preflight)
    report_path = preflight / 'gpu-validation.json'
    report = _json_object(report_path)
    arms = report.get('arms')
    if (report.get('status') != 'passed' or not isinstance(arms, dict)
            or set(arms) != {'grpo', 'lam4'}
            or any(not isinstance(arm, dict) or arm.get('status') != 'passed'
                   for arm in arms.values())):
        raise ValueError('Preflight report must pass exactly the grpo and lam4 arms')
    targets = []
    for arm in ('grpo', 'lam4'):
        _, checkpoint, exports, _ = _verify_exports(preflight / arm, 2)
        targets.extend((checkpoint, exports))
    removed_bytes = sum(_tree_bytes(path) for path in targets)
    validation_sha256 = _sha256(report_path)
    for path in targets:
        shutil.rmtree(path)
    return {'validation_sha256': validation_sha256,
            'removed_paths': [str(path) for path in targets], 'removed_bytes': removed_bytes}
