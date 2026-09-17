"""Bind an adapter to its recorded base, including byte-identical relocations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_base(adapter, model):
    """Reject changed weights/tokenizers; architecture compatibility is insufficient.

    An exact recorded identifier also supports Hugging Face model IDs without
    network access. A changed path requires matching actual model artifacts.
    """
    if str(adapter) == 'none':
        return
    config = json.loads((Path(adapter) / 'adapter_config.json').read_text())
    recorded = config.get('base_model_name_or_path')
    if not recorded:
        raise ValueError(f'Adapter does not identify its base model: {adapter}')
    if recorded == str(model):
        return
    recorded_path = Path(recorded)
    if not recorded_path.exists():
        recorded_path = ROOT / recorded_path
    model_path = Path(model)
    if recorded_path.resolve() == model_path.resolve():
        return
    if (recorded_path / 'config.json').is_file() and (model_path / 'config.json').is_file():
        a = json.loads((recorded_path / 'config.json').read_text())
        b = json.loads((model_path / 'config.json').read_text())
        fields = ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                  'num_key_value_heads', 'intermediate_size', 'vocab_size')
        if all(a.get(key) == b.get(key) for key in fields):
            def artifacts(directory):
                directory = Path(directory)
                names = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                         'special_tokens_map.json', 'added_tokens.json',
                         'vocab.json', 'merges.txt', 'tokenizer.model',
                         'chat_template.jinja'}
                return {str(path.relative_to(directory)): path for path in directory.rglob('*')
                        if path.is_file() and (path.name in names or path.suffix in ('.safetensors', '.bin')
                                               or path.name.endswith('.index.json'))}

            left, right = artifacts(recorded_path), artifacts(model_path)
            if (left.keys() == right.keys()
                    and any(Path(name).suffix in ('.safetensors', '.bin') for name in left)
                    and all(left[name].stat().st_size == right[name].stat().st_size
                            and (left[name].samefile(right[name])
                                 or _file_hash(left[name]) == _file_hash(right[name])) for name in left)):
                return
    raise ValueError(f'Adapter base model {recorded!r} is incompatible with {str(model)!r}')
