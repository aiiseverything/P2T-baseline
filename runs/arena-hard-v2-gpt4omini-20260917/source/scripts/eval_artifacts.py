"""Evaluation provenance and atomic files. Pure stdlib: usable by CPU judges."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_versions(packages=('transformers', 'vllm')):
    """Installed generation-stack versions, with a stable unavailable marker."""
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    return versions


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path, *, full_weights=True):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    records = []
    for p in files:
        # Base weights are tens of GB: record filesystem identity, size and both
        # change timestamps. Adapters, data, configs and tokenizer use full SHA256.
        stat = p.stat()
        record = {'name': str(p.relative_to(path)) if path.is_dir() else p.name,
                  'size': stat.st_size}
        if full_weights or p.suffix not in ('.safetensors', '.bin', '.pt'):
            record['sha256'] = file_hash(p)
        else:
            record.update(mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                          inode=stat.st_ino, resolved=str(p.resolve()))
        records.append(record)
    return {'path': str(path), 'files': records}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def cache_matches(manifest, config, outputs):
    manifest = Path(manifest)
    outputs = [Path(p) for p in outputs]
    if not manifest.exists():
        if any(p.exists() for p in outputs):
            raise ValueError(f'Unverified or incomplete legacy cache at {manifest.parent}; use a new output directory')
        return False
    stored = json.loads(manifest.read_text())
    if stored.get('config') != config:
        raise ValueError(f'Cache was made with different inputs/configuration: {manifest}; use a new output directory')
    expected = stored.get('outputs', {})
    if set(expected) != {p.name for p in outputs} or any(
            not p.is_file() or file_hash(p) != expected[p.name] for p in outputs):
        raise ValueError(f'Cache outputs incomplete or changed: {manifest}')
    return True


def commit_cache(manifest, config, outputs):
    atomic_text(manifest, json.dumps({'config': config,
                'outputs': {Path(p).name: file_hash(p) for p in outputs}}, indent=2,
                ensure_ascii=False, allow_nan=False))


def eval_config(args, recipe, model_fingerprint, adapter_fingerprint, stop_ids,
                scorer, output_support):
    sources = ['scripts/eval_artifacts.py', f'scripts/eval_{scorer}.py',
               'vpo_rm/token_policy.py', 'vpo_rm/model_identity.py', 'vpo_rm/trainer.py',
               'vpo_rm/integration.py', 'vpo_rm/alignment.py']
    if scorer == 'ifeval':
        sources.extend(str(p.relative_to(ROOT)) for p in
                       (ROOT / 'third_party/ifeval').rglob('*.py'))
    return {'protocol': 2, 'scorer': scorer, 'recipe': recipe,
            'model': model_fingerprint, 'adapter': adapter_fingerprint,
            'dataset': fingerprint(args.dataset), 'seed': args.seed,
            'max_tokens': args.max_tokens, 'stop_token_ids': list(stop_ids),
            'output_support': output_support,
            'engine': {'dtype': 'bfloat16', 'max_model_len': 4096},
            'runtime_versions': runtime_versions(),
            'sources': {p: file_hash(ROOT / p) for p in sources}}


def validate_outputs(outputs, n_prompts, n_samples):
    if len(outputs) != n_prompts or any(len(o.outputs) != n_samples for o in outputs):
        raise ValueError(f'Incomplete generation: expected {n_prompts} prompts x {n_samples} samples')


def validate_adapter_base(adapter, model):
    # Keep stdlib-only judge/cache imports usable without loading torch.
    from vpo_rm.model_identity import validate_adapter_base as validate
    return validate(adapter, model)
