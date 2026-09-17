"""Resume the two requested, hash-verified downloads through the confirmed proxy."""
from pathlib import Path
import datetime
import json
import os
import shutil
import subprocess
import psutil

root = Path('/data/VPO-RM')
logs = root / '.download-logs'
names = {'download_supervisor.py', 'download_actor.py', 'download_rm.py'}
for process in psutil.process_iter(['pid', 'cmdline']):
    try:
        args = process.info['cmdline'] or []
        if args and 'python' in Path(args[0]).name and any(Path(arg).name in names for arg in args[1:]):
            raise RuntimeError('Download process already exists: ' + str(process.pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = logs / ('restart-' + stamp)
backup.mkdir()
for path in (logs / 'background_status.json', root / 'models/Llama-3.1-8B-Instruct/DOWNLOAD_MANIFEST.json',
             root / 'models/Skywork-Reward-Llama-3.1-8B-v0.2/DOWNLOAD_SOURCE.json'):
    if path.exists():
        shutil.copy2(path, backup / path.name)
environment = os.environ.copy()
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    environment[key] = 'http://127.0.0.1:17891'
for key in ('ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
    environment.pop(key, None)
with (logs / 'supervisor.log').open('ab') as stream:
    child = subprocess.Popen(['/root/miniconda3/envs/sml/bin/python', '-u', str(logs / 'download_supervisor.py')],
        cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
record = {'pid': child.pid, 'started_at_utc': stamp, 'proxy': 'http://127.0.0.1:17891',
          'backup': str(backup), 'resumes_existing_fragments': True}
(backup / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
