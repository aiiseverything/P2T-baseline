"""Launch the bound Arena continuation through the confirmed17891 proxy."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess

suite = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_prelaunch_retry_binding', suite / 'retry_resilient_judgments.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.binding(suite)
status = suite / 'resilient_status.json'
if status.exists():
    previous = helper.read_json(status)
    if previous.get('state') == 'running':
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous continuation is still running')
environment = os.environ.copy()
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    environment[key] = 'http://127.0.0.1:17891'
for key in ('ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
    environment.pop(key, None)
environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                   TIKTOKEN_CACHE_DIR=str(suite / '.tiktoken-cache'))
with (suite / 'job/resilient_continuation.log').open('ab') as stream:
    child = subprocess.Popen(['/root/.venvs/alpacaeval/bin/python', '-u', str(suite / 'continue_resilient.py')],
        cwd=suite, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
record = {'pid': child.pid, 'launched_at': helper.now(), 'proxy': 'http://127.0.0.1:17891',
          'resilient_policy_sha256': helper.file_hash(suite / 'resilient_policy.json')}
helper.judge_module(suite).atomic_json(suite / 'job/resilient_launch.json', record)
print(json.dumps(record))
