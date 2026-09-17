import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import time

ROOT = Path('/data/VPO-RM')
DEST = ROOT / 'models/Skywork-Reward-Llama-3.1-8B-v0.2'
REPO = 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'
REVISION = 'd4117fbfd81b72f41b96341238baa1e3e90a4ce1'

# Explicitly keep model, transfer state, and temporary data on /data.
os.environ['HF_HOME'] = str(ROOT / '.download-cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(ROOT / '.download-cache/huggingface/hub')
os.environ['HF_XET_CACHE'] = str(ROOT / '.download-cache/huggingface/xet')
os.environ['TMPDIR'] = str(ROOT / '.download-tmp')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
os.environ['MODELSCOPE_CACHE'] = str(ROOT / '.download-cache/modelscope')
os.environ['MODELSCOPE_DOWNLOAD_PARALLELS'] = '8'
os.environ['TQDM_DISABLE'] = '1'
logging.disable(logging.CRITICAL)

from huggingface_hub import HfApi, hf_hub_download
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

def emit(event, **fields):
    print(json.dumps({'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **fields}), flush=True)

def safe_error(exc):
    response = getattr(exc, 'response', None)
    return {'type': type(exc).__name__, 'http_status': getattr(response, 'status_code', None)}

def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    assert info.sha == REVISION
    selected = list(info.siblings)
    source_manifest = {'repo_id': REPO, 'revision': REVISION, 'source': 'https://huggingface.co/' + REPO,
                       'files': [{'path': f.rfilename, 'size': f.size, 'git_blob_sha1': f.blob_id,
                                  'lfs_sha256': f.lfs.sha256 if f.lfs else None} for f in selected]}
    mirror_repo = 'AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2'
    mirror_files = {f['Path']: f for f in HubApi().get_model_files(mirror_repo, revision='master', recursive=True)
                    if f.get('Type') != 'tree'}
    mirror_paths = [f.rfilename for f in selected if f.rfilename in mirror_files]
    for f in selected:
        if f.rfilename in mirror_files:
            mf = mirror_files[f.rfilename]
            assert mf['Size'] == f.size, f.rfilename
            if f.lfs:
                assert mf['Sha256'] == f.lfs.sha256, f.rfilename
    assert all(f.rfilename in mirror_files for f in selected if f.lfs)
    source_manifest['download_source'] = 'modelscope (mirrored files verified against pinned Hugging Face identities)'
    source_manifest['mirror_repo'] = mirror_repo
    source_manifest['mirror_files'] = {name: {'sha256': mirror_files[name]['Sha256'],
                                            'revision': mirror_files[name]['Revision'],
                                            'size': mirror_files[name]['Size']} for name in mirror_paths}
    (DEST / 'DOWNLOAD_SOURCE.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    emit('metadata', repo=REPO, revision=REVISION, file_count=len(selected), total_bytes=sum(f.size for f in selected))

    def download(f):
        for attempt in range(1, 5):
            try:
                path = hf_hub_download(repo_id=REPO, filename=f.rfilename, revision=REVISION,
                                       local_dir=str(DEST), token=False)
                assert Path(path).stat().st_size == f.size
                emit('downloaded', file=f.rfilename, bytes=f.size)
                return path
            except Exception as exc:
                emit('download_error', file=f.rfilename, attempt=attempt, **safe_error(exc))
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))

    snapshot_download(mirror_repo, revision='master',
                      cache_dir=os.environ['MODELSCOPE_CACHE'], local_dir=str(DEST),
                      allow_patterns=mirror_paths, max_workers=4)
    for f in selected:
        if f.rfilename not in mirror_files:
            download(f)

    verified = []
    for f in selected:
        path = DEST / f.rfilename
        size = path.stat().st_size
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(b'blob ' + str(size).encode() + b'\x00')
        with path.open('rb') as fp:
            while block := fp.read(16 * 1024 * 1024):
                sha256.update(block)
                git_sha1.update(block)
        actual_sha256 = sha256.hexdigest()
        assert size == f.size, f.rfilename + ': size mismatch'
        if f.rfilename in mirror_files:
            assert actual_sha256 == mirror_files[f.rfilename]['Sha256'], f.rfilename + ': mirror SHA256 mismatch'
        if f.lfs:
            assert actual_sha256 == f.lfs.sha256, f.rfilename + ': LFS SHA256 mismatch'
        else:
            assert git_sha1.hexdigest() == f.blob_id, f.rfilename + ': git blob SHA1 mismatch'
        verified.append({'path': f.rfilename, 'size': size, 'sha256': actual_sha256,
                         'authoritative_check': 'LFS SHA256' if f.lfs else 'git blob SHA1', 'verified': True})
        emit('verified', file=f.rfilename, bytes=size)

    config = json.loads((DEST / 'config.json').read_text())
    assert config['architectures'] == ['LlamaForSequenceClassification'], config['architectures']
    assert config['model_type'] == 'llama'
    index = json.loads((DEST / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    indexed_shards = sorted(set(weight_map.values()))
    selected_shards = sorted(f.rfilename for f in selected if f.rfilename.endswith('.safetensors'))
    assert indexed_shards == selected_shards
    seen = set()
    tensor_bytes = 0
    for shard in indexed_shards:
        with (DEST / shard).open('rb') as fp:
            header_size = struct.unpack('<Q', fp.read(8))[0]
            assert header_size < 100_000_000
            header = json.loads(fp.read(header_size))
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            assert name not in seen
            assert weight_map[name] == shard
            seen.add(name)
            start, end = tensor['data_offsets']
            assert 0 <= start <= end <= (DEST / shard).stat().st_size - 8 - header_size
            tensor_bytes += end - start
    assert seen == set(weight_map)
    assert tensor_bytes == index['metadata']['total_size']
    report = {'status': 'complete_verified', 'repo_id': REPO, 'revision': REVISION,
              'download_source': source_manifest['download_source'], 'mirror_repo': mirror_repo,
              'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'architectures': config['architectures'], 'model_type': config['model_type'],
              'tensor_count': len(seen), 'tensor_bytes': tensor_bytes,
              'total_downloaded_bytes': sum(f.size for f in selected), 'files': verified}
    tmp = DEST / 'DOWNLOAD_VERIFIED.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(DEST / 'DOWNLOAD_VERIFIED.json')
    emit('complete_verified', destination=str(DEST), tensor_count=len(seen), total_bytes=report['total_downloaded_bytes'])

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit('fatal_error', **safe_error(exc))
        raise SystemExit(1)
