"""Resolve generation-head precision from adapter provenance (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HEAD_DTYPES = ('native', 'float32')


def resolve_policy_head(adapter, requested='auto'):
    """Fail closed on conflicting declarations; undeclared legacy heads are native.

    Final checkpoints carry run_manifest.json. Temporary step adapters inherit
    train/profile_manifest.json through their vllm-adapters parent. A final
    checkpoint's adjacent training profile is also checked when present.
    Metadata content hashes bind caches even when the declared dtype is unchanged.
    """
    if requested not in ('auto', *HEAD_DTYPES):
        raise ValueError(f'Unknown policy_head_dtype request: {requested!r}')
    paths = []
    if adapter is not None and str(adapter) != 'none':
        adapter = Path(adapter).resolve()
        paths.append(adapter / 'run_manifest.json')
        if adapter.parent.name == 'vllm-adapters' and adapter.name.startswith('step-'):
            paths.append(adapter.parent.parent / 'profile_manifest.json')
        elif adapter.name.startswith('checkpoint-'):
            paths.append(adapter.parent / 'profile_manifest.json')
    metadata, declared = [], set()
    for path in paths:
        if not path.exists():
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'Invalid policy metadata object: {path}')
        metadata.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        sections = [value]
        for key in ('resolved_config', 'config', 'sampling', 'vllm_engine'):
            if key in value:
                if not isinstance(value[key], dict):
                    raise ValueError(f'Invalid policy metadata {key}: {path}')
                sections.append(value[key])
        for section in sections:
            if 'policy_head_dtype' not in section:
                continue
            dtype = section['policy_head_dtype']
            if dtype not in HEAD_DTYPES:
                raise ValueError(f'Unknown policy_head_dtype {dtype!r} in {path}')
            declared.add(dtype)
    if len(declared) > 1 or (declared and requested != 'auto' and requested not in declared):
        raise ValueError(f'policy_head_dtype conflict for {adapter}: declared={sorted(declared)}, requested={requested}')
    head = next(iter(declared)) if declared else ('native' if requested == 'auto' else requested)
    return {'policy_head_dtype': head, 'metadata': metadata,
            'resolver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def resolve_shared_policy(adapters, requested='auto'):
    policies = [resolve_policy_head(adapter, requested) for adapter in adapters]
    modes = {policy['policy_head_dtype'] for policy in policies}
    if len(modes) > 1:
        raise ValueError('Mixed native/float32 policy heads require separate evaluation invocations; '
                         'use --policy-head-dtype float32 for undeclared legacy baselines if intended')
    return (next(iter(modes)) if modes else 'native'), policies


def policy_engine_kwargs(head):
    """Actor vLLM head override only; never apply this to a reward model."""
    if head not in HEAD_DTYPES:
        raise ValueError(f'Unknown resolved policy_head_dtype: {head!r}')
    return {'hf_overrides': {'head_dtype': 'float32'}} if head == 'float32' else {}
