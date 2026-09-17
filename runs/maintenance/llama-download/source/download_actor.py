"""Download the public ModelScope distribution and verify Meta file identities."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/data/VPO-RM')
DEST = ROOT / 'models/Llama-3.1-8B-Instruct'
CACHE = ROOT / '.cache/modelscope'
TMP = ROOT / '.cache/tmp'
for path in (DEST, CACHE, TMP):
    path.mkdir(parents=True, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = str(CACHE)
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['TMPDIR'] = str(TMP)
os.environ['TQDM_DISABLE'] = '1'

from huggingface_hub import HfApi
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

UPSTREAM = 'meta-llama/Llama-3.1-8B-Instruct'
UPSTREAM_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
SOURCE = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
api = HubApi()
hf = HfApi().model_info(UPSTREAM, revision=UPSTREAM_REVISION,
                      files_metadata=True, timeout=45)
upstream_files = {item.rfilename: item for item in hf.siblings}
source_files = api.get_model_files(SOURCE, revision='master', recursive=True)
selected = [item for item in source_files
            if item.get('Type') != 'tree' and '/' not in item['Path']
            and item['Path'] in upstream_files]
required = {'config.json', 'generation_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json',
            'model.safetensors.index.json', 'LICENSE', 'USE_POLICY.md'}
assert required <= {item['Path'] for item in selected}
weights = [item for item in selected if item['Path'].endswith('.safetensors')]
assert len(weights) == 4
for item in selected:
    original = upstream_files[item['Path']]
    assert item['Size'] == original.size, item['Path']
    if original.lfs:
        assert item['Sha256'] == original.lfs.sha256, item['Path']

manifest = {
    'upstream_repo': UPSTREAM,
    'upstream_revision': hf.sha,
    'download_source': 'modelscope',
    'source_repo': SOURCE,
    'source_revision_requested': 'master',
    'source_version_note': 'Expected per-file commit, size and SHA256 captured before download; all selected files must also match pinned upstream Git/LFS identities.',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'destination': str(DEST),
    'files': [{
        'path': item['Path'], 'size': item['Size'],
        'source_file_revision': item['Revision'],
        'source_sha256': item['Sha256'],
        'upstream_identity_kind': 'sha256' if upstream_files[item['Path']].lfs else 'git_blob_sha1',
        'upstream_identity': (upstream_files[item['Path']].lfs.sha256
                              if upstream_files[item['Path']].lfs
                              else upstream_files[item['Path']].blob_id),
    } for item in selected],
    'verified': False,
}
manifest_path = DEST / 'DOWNLOAD_MANIFEST.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'event': 'START', 'files': len(selected),
                  'total_bytes': sum(item['Size'] for item in selected),
                  'destination': str(DEST)}, ensure_ascii=False), flush=True)
snapshot_download(SOURCE, revision='master', cache_dir=str(CACHE),
                  local_dir=str(DEST),
                  allow_patterns=[item['Path'] for item in selected], max_workers=4)

for item in manifest['files']:
    path = DEST / item['path']
    assert path.is_file() and path.stat().st_size == item['size'], item['path']
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {item['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            sha256.update(chunk)
            if item['upstream_identity_kind'] == 'git_blob_sha1':
                blob.update(chunk)
    actual = sha256.hexdigest()
    assert actual == item['source_sha256'], item['path']
    upstream_actual = actual if item['upstream_identity_kind'] == 'sha256' else blob.hexdigest()
    assert upstream_actual == item['upstream_identity'], item['path']
    item['local_sha256'] = actual
    item['verified_against_upstream'] = True
    print(json.dumps({'event': 'VERIFIED', 'file': item['path'],
                      'sha256': actual}), flush=True)

index = json.loads((DEST / 'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values()) == {item['Path'] for item in weights}
config = json.loads((DEST / 'config.json').read_text())
assert config['architectures'] == ['LlamaForCausalLM']
assert config['model_type'] == 'llama'
manifest['architecture'] = config['architectures']
manifest['index_tensor_count'] = len(index['weight_map'])
manifest['verified'] = True
manifest['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
manifest['total_file_bytes'] = sum(item['size'] for item in manifest['files'])
temporary = manifest_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(manifest, indent=2) + '\n')
temporary.replace(manifest_path)
print(json.dumps({'event': 'COMPLETE', 'verified': True,
                  'total_bytes': manifest['total_file_bytes']}), flush=True)
