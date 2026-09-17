"""Keep the two explicitly requested downloads running independently of a UI turn."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path('/data/VPO-RM')
LOGS = ROOT / '.download-logs'
PYTHON = '/root/miniconda3/envs/sml/bin/python'
STATUS = LOGS / 'background_status.json'
LOCK = threading.Lock()
state = {'supervisor_pid': os.getpid(), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'models': {}}

def update(tag, **fields):
    with LOCK:
        state['models'].setdefault(tag, {}).update(fields)
        state['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temp = STATUS.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(STATUS)

def worker(tag, script, model_dir, manifest, successful):
    log = LOGS / (tag + '.background.log')
    dest = ROOT / 'models' / model_dir
    environment = os.environ.copy()
    environment['MODELSCOPE_DOWNLOAD_PARALLELS'] = '8'
    environment['HF_HOME'] = str(ROOT / '.download-cache/huggingface')
    environment['TMPDIR'] = str(ROOT / '.download-tmp')
    update(tag, status='starting', destination=str(dest), log=str(log))
    for attempt in range(1, 11):
        with log.open('ab', buffering=0) as output:
            process = subprocess.Popen([PYTHON, '-u', str(LOGS / script)], env=environment,
                                       cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            update(tag, status='downloading_or_verifying', attempt=attempt, worker_pid=process.pid)
            returncode = process.wait()
        ok = False
        if returncode == 0 and (dest / manifest).is_file():
            report = json.loads((dest / manifest).read_text())
            ok = successful(report)
        if ok:
            # Remove only obsolete transfer fragments belonging to this model.
            temporary = dest / '._____temp'
            if temporary.is_dir():
                shutil.rmtree(temporary)
            for path in (dest / '.cache/huggingface/download').glob('*.incomplete'):
                path.unlink()
            if tag == 'rm':
                probe = ROOT / '.download-tmp/rm-xet-probe'
                if probe.is_dir():
                    shutil.rmtree(probe)
            update(tag, status='complete_verified', worker_pid=None, returncode=returncode,
                   manifest=str(dest / manifest), completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            return
        update(tag, status='retry_pending' if attempt < 10 else 'failed', worker_pid=None, returncode=returncode)
        if attempt < 10:
            time.sleep(min(10 * attempt, 60))

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(worker, 'actor', 'download_actor.py', 'Llama-3.1-8B-Instruct',
                           'DOWNLOAD_MANIFEST.json', lambda r: r.get('verified') is True),
               pool.submit(worker, 'rm', 'download_rm.py', 'Skywork-Reward-Llama-3.1-8B-v0.2',
                           'DOWNLOAD_VERIFIED.json', lambda r: r.get('status') == 'complete_verified')]
    for future in futures:
        future.result()
