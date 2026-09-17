#!/usr/bin/env python3
"""Score HumanEval in a minimal chroot with new user/mount/network/PID namespaces.

The official execution helper runs only inside the isolated child. No fallback
executes candidate code on the host. Linux unshare/mount/chroot are required.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, file_hash
from scripts.eval_humaneval import load_problems

WORKER = r'''
import ctypes, json, math, resource, sys
sys.path.insert(0, '/')
from execution import unsafe_execute
request = json.load(sys.stdin)
timeout = request['timeout']
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout)+1, math.ceil(timeout)+1))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sec = ctypes.CDLL('/lib/x86_64-linux-gnu/libseccomp.so.2')
sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
sec.seccomp_load.argtypes=[ctypes.c_void_p]
sec.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=sec.seccomp_init(0x7fff0000)
if not ctx: raise RuntimeError('seccomp initialization failed')
for name in ['clone','clone3','fork','vfork','execve','execveat','mount','umount2',
             'unshare','setns','ptrace','open_by_handle_at','process_vm_readv',
             'process_vm_writev','socket','socketpair','bpf','userfaultfd']:
    nr=sec.seccomp_syscall_resolve_name(name.encode())
    if nr>=0 and sec.seccomp_rule_add(ctx,0x00050001,nr,0)!=0:
        raise RuntimeError('seccomp rule failed: '+name)
if sec.seccomp_load(ctx)!=0: raise RuntimeError('seccomp load failed')
sec.seccomp_release(ctx)
result=[]
unsafe_execute(request['problem'], request['completion'], timeout, result)
print(json.dumps({'result':result[0], 'passed':result[0]=='passed'}))
'''


def validate_samples(problems, samples):
    ids = [p["task_id"] for p in problems]
    actual = [s.get("task_id") for s in samples]
    if len(actual) != len(ids) or len(set(actual)) != len(actual) or set(actual) != set(ids):
        raise ValueError("Each HumanEval task must occur exactly once for greedy pass@1")
    by_id = {s["task_id"]:s for s in samples}
    if any(not isinstance(s.get("completion"), str) for s in samples):
        raise ValueError("Each completion must be a string")
    return [by_id[task_id] for task_id in ids]


def prepare_sandbox(root):
    """Copy only system Python/stdlib/runtime libraries, never the project or home."""
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f"Use a fresh sandbox root: {root}")
    root.mkdir(parents=True)
    python = Path('/usr/bin/python3').resolve()
    version = subprocess.check_output([str(python), '-c', 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    stdlib = Path('/usr/lib') / f'python{version}'
    shutil.copytree(stdlib, root / stdlib.relative_to('/'), ignore=shutil.ignore_patterns('__pycache__', 'test', 'tests'))
    dst = root / python.relative_to('/'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(python, dst)
    (root/'usr/bin/python3').symlink_to(python.name)
    libraries = {Path('/lib/x86_64-linux-gnu/libseccomp.so.2')}
    for binary in [python, *stdlib.rglob('*.so')]:
        proc = subprocess.run(['ldd', str(binary)], text=True, capture_output=True)
        libraries.update(Path(p) for p in re.findall(r'(/[^\s()]+)', proc.stdout) if Path(p).is_file())
    for lib in libraries:
        dst = root / lib.relative_to('/'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(lib, dst)
    official = ROOT / 'third_party/human_eval/execution.py'
    upstream = json.loads((official.parent/'UPSTREAM.json').read_text())
    if file_hash(official) != upstream['vendored_files']['execution.py']:
        raise ValueError('Official HumanEval execution helper changed')
    shutil.copy2(official, root/'execution.py')
    (root/'worker.py').write_text(WORKER)
    (root/'tmp').mkdir()
    return root


def sandbox_check(root, problem, completion, timeout=3.0):
    # Paths are positional shell arguments, never interpolated into shell code.
    shell = ('set -eu; mount --make-rprivate /; mount --bind "$1" "$1"; '
             'mount -o remount,bind,ro "$1"; '
             'mount -t tmpfs -o size=64m,nosuid,nodev,noexec tmpfs "$1/tmp"; '
             'exec chroot "$1" /usr/bin/python3 -I /worker.py')
    command = ['unshare','--user','--map-root-user','--mount','--net','--pid','--fork','--kill-child',
               '/bin/sh','-c',shell,'humaneval-sandbox',str(Path(root).resolve())]
    request = json.dumps({'problem':problem, 'completion':completion, 'timeout':timeout})
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C.UTF-8'})
    try:
        stdout, stderr = proc.communicate(request, timeout=timeout+5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {'task_id':problem['task_id'], 'passed':False, 'result':'timed out (outer watchdog)'}
    if proc.returncode:
        raise RuntimeError(f'HumanEval sandbox failed (no unsafe fallback): {stderr[-1500:]}')
    result = json.loads(stdout)
    if not isinstance(result.get('passed'), bool) or not isinstance(result.get('result'), str):
        raise ValueError('Invalid sandbox result')
    return {'task_id':problem['task_id'], **result}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/humaneval/HumanEval.jsonl.gz')
    p.add_argument('--samples', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sandbox-root', required=True)
    p.add_argument('--timeout', type=float, default=3.0)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(argv)
    if not 0 < args.timeout <= 30 or not 1 <= args.workers <= 8:
        p.error('timeout must be in (0,30], workers in [1,8]')
    problems = load_problems(args.dataset)
    samples = validate_samples(problems, [json.loads(x) for x in Path(args.samples).read_text().splitlines() if x.strip()])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = prepare_sandbox(args.sandbox_root)
    probe = {'task_id':'selftest','prompt':'def f():\n','entry_point':'f','test':'def check(candidate):\n    assert candidate() == 7\n'}
    if not sandbox_check(root, probe, '    return 7\n')['passed']:
        raise RuntimeError('Sandbox canonical self-test failed')
    if sandbox_check(root, probe, '    return 0\n')['passed']:
        raise RuntimeError('Sandbox negative self-test failed')
    def evaluate(pair):
        problem, sample = pair
        return sandbox_check(root, problem, sample['completion'], args.timeout)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, zip(problems, samples)))
    passed = sum(r['passed'] for r in results)
    atomic_text(output, json.dumps({'benchmark':'HumanEval','protocol':'native_function_completion_raw_v1',
        'n_tasks':len(results),'n_samples_per_task':1,'passed':passed,'pass@1':passed/len(results),
        'timeout_seconds':args.timeout,'sandbox':'user+mount+net+pid namespaces; minimal read-only chroot; seccomp; rlimits',
        'dataset_sha256':file_hash(args.dataset),'samples_sha256':file_hash(args.samples),
        'official_execution_sha256':file_hash(ROOT/'third_party/human_eval/execution.py'),
        'results':results},indent=2))
    print(f'HumanEval pass@1: {passed}/{len(results)} = {passed/len(results):.4%}', flush=True)


if __name__ == '__main__':
    main()
