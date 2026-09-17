"""Run six frozen Arena policies on six isolated visible GPUs."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SUITE = Path(__file__).resolve().parent
SOURCE = SUITE / 'source'
_paths = [str(SOURCE), str(SUITE / '.arena-extra'), str(SOURCE / '.vllm-extra')]
sys.path[:0] = _paths
_spec = importlib.util.spec_from_file_location('_arena_existing_driver', SUITE / 'run_evaluation.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
# The existing driver prepends its own paths; retain the new dependency priority.
sys.path[:] = _paths + [p for p in sys.path if p not in _paths]
TAGS = _legacy.TAGS
verify_inputs, validate_results, save = _legacy.verify_inputs, _legacy.validate_results, _legacy.save


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_tokens(visible_count, inherited_mask):
    if visible_count != 6:
        raise ValueError(f'Expected six visible GPUs, got {visible_count}')
    tokens = list(map(str, range(6))) if inherited_mask is None else [p.strip() for p in inherited_mask.split(',')]
    if len(tokens) != 6 or len(set(tokens)) != 6 or any(not p or p == '-1' for p in tokens):
        raise ValueError('CUDA_VISIBLE_DEVICES must expose six distinct device tokens')
    return tokens


def runtime_preflight(manifest, expected_devices):
    versions = {p: importlib.metadata.version(p) for p in manifest['gpu_versions']}
    if versions != manifest['gpu_versions']:
        raise RuntimeError(f'GPU runtime versions differ: {versions}')
    # Import the actual extension modules before any model allocation.
    modules = {p: importlib.import_module(p) for p in ('pandas', 'tiktoken', 'peft')}
    from scripts import eval_arena_hard
    eval_arena_hard._style_tools()
    import torch
    count = torch.cuda.device_count()
    if count != expected_devices:
        raise RuntimeError(f'Expected {expected_devices} visible GPUs, got {count}')
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if any('H200' not in name for name in names):
        raise RuntimeError(f'Expected H200 GPUs, got {names}')
    return {'versions': versions, 'device_count': count, 'gpus': names,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'extra_versions': {p: importlib.metadata.version(p) for p in modules},
            'dependency_modules': {p: getattr(module, '__file__', None) for p, module in modules.items()},
            'python': sys.version, 'checked_at': now()}


def generation_command(manifest, tag):
    if tag not in TAGS:
        raise ValueError(f'Unknown policy {tag}')
    return [sys.executable, str(SOURCE / 'scripts/eval_arena_hard.py'),
            '--model', manifest['base_model'], '--dataset', str(SUITE / 'question.jsonl'),
            '--output', str(SUITE), '--max-tokens', '4096', '--max-model-len', '16384',
            '--max-num-seqs', '32', '--seed', '42', '--policy-head-dtype', 'float32',
            '--adapters', f"{tag}={manifest['models'][tag]['adapter']}"]


def run_worker(tag):
    if tag not in TAGS:
        raise ValueError(f'Unknown policy {tag}')
    directory = SUITE / 'job' / tag
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    try:
        runtime = runtime_preflight(manifest, expected_devices=1)
        runtime.update(tag=tag, pid=os.getpid(), local_device='cuda:0')
        save(directory / 'runtime.json', runtime)
        command = generation_command(manifest, tag)
        save(directory / 'command.json', command)
        save(directory / 'status.json', {'state': 'generating', 'tag': tag, 'started_at': now()})
        print(json.dumps({'worker': tag, 'runtime': runtime, 'command': command}), flush=True)
        # Replace this process: six workers remain six model processes, not twelve.
        os.execv(sys.executable, command)
    except Exception as error:
        save(directory / 'status.json', {'state': 'preflight_failed', 'tag': tag,
                                         'error': repr(error), 'failed_at': now()})
        raise


def run_parent():
    state, processes, streams, exit_codes = {}, {}, [], {}
    try:
        if (SUITE / 'status.json').exists():
            state = json.loads((SUITE / 'status.json').read_text())
        state.update(state='parallel_preflight', started_at=now())
        save(SUITE / 'status.json', state)
        manifest = json.loads((SUITE / 'experiment.json').read_text())
        questions = verify_inputs(manifest)
        runtime = runtime_preflight(manifest, expected_devices=6)
        tokens = gpu_tokens(runtime['device_count'], os.environ.get('CUDA_VISIBLE_DEVICES'))
        runtime['assignments'] = dict(zip(TAGS, tokens))
        save(SUITE / 'job/runtime.json', runtime)
        state.update(state='generating', mode='six_gpu_parallel', assignments=runtime['assignments'])
        save(SUITE / 'status.json', state)
        for tag, token in zip(TAGS, tokens):
            directory = SUITE / 'job' / tag
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(SUITE / 'run_parallel_evaluation.py'), '--worker', tag]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=token, OMP_NUM_THREADS='8')
            cache_root = f'/tmp/arena-{SUITE.name}-{tag}'
            environment.update(TORCHINDUCTOR_COMPILE_THREADS='1',
                               VLLM_CACHE_ROOT=f'{cache_root}/vllm',
                               TORCHINDUCTOR_CACHE_DIR=f'{cache_root}/inductor')
            environment.pop('VLLM_PORT', None)
            save(directory / 'launcher.json', {'command': command, 'cuda_visible_devices': token,
                'environment': {key: environment[key] for key in ('OMP_NUM_THREADS',
                    'TORCHINDUCTOR_COMPILE_THREADS', 'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR')}})
            save(directory / 'status.json', {'state': 'starting', 'tag': tag, 'started_at': now()})
            stream = (directory / 'job.log').open('a', buffering=1)
            streams.append(stream)
            try:
                processes[tag] = subprocess.Popen(command, env=environment,
                    cwd=str(SUITE.parents[1]), stdout=stream, stderr=subprocess.STDOUT)
            except Exception as error:
                exit_codes[tag] = None
                save(directory / 'status.json', {'state': 'spawn_failed', 'tag': tag, 'error': repr(error)})
        pending = dict(processes)
        while pending:
            for tag, process in list(pending.items()):
                if process.poll() is None:
                    continue
                code = process.wait()
                exit_codes[tag] = code
                directory = SUITE / 'job' / tag
                _legacy.atomic_text(directory / 'exit_code', f'{code}\n')
                save(directory / 'status.json', {'state': 'exited' if code == 0 else 'failed',
                                                 'tag': tag, 'exit_code': code, 'finished_at': now()})
                print(json.dumps({'tag': tag, 'exit_code': code}), flush=True)
                del pending[tag]
            if pending:
                time.sleep(1)
        state['exit_codes'] = exit_codes
        failed = {tag: exit_codes.get(tag) for tag in TAGS if exit_codes.get(tag) != 0}
        if failed:
            raise RuntimeError(f'Parallel generation failed: {failed}')
        summary = validate_results(manifest, questions)
        save(SUITE / 'generation_summary.json', summary)
        state.update(state='generation_complete', generation_finished_at=now())
        save(SUITE / 'generation_complete.json', state)
        print('All six sets of 500 answers verified.', flush=True)
        return summary
    except BaseException as error:
        state.update(state='generation_failed', error=repr(error), exit_codes=exit_codes, failed_at=now())
        raise
    finally:
        # A parent interruption must not leave model children running unowned.
        for tag, process in processes.items():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()
        save(SUITE / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=TAGS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
    else:
        run_parent()


if __name__ == '__main__':
    main()
