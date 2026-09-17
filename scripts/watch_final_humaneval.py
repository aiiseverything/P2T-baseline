#!/usr/bin/env python3
"""Submit one HumanEval generation job per successfully completed final adapter.

The lightweight watcher runs on the login host. It allocates no GPU while
waiting. Generated completions are scored on the host only via the isolated
score_humaneval subprocess, after the generation job reports success.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, file_hash

ARMS = ('grpo', 'lam2', 'lam4', 'lam8')


def final_checkpoint_ready(arm_dir, expected=250):
    arm_dir = Path(arm_dir)
    checkpoint = arm_dir / f'train/checkpoint-{expected}'
    required = [arm_dir/'completion.json', arm_dir/'stage', arm_dir/'exit_code',
                arm_dir/'train/profile_summary.json', checkpoint/'run_manifest.json',
                checkpoint/'adapter_config.json', checkpoint/'adapter_model.safetensors']
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        return None
    try:
        if (arm_dir/'stage').read_text().strip() != 'complete' or (arm_dir/'exit_code').read_text().strip() != '0':
            return None
        if json.loads((arm_dir/'completion.json').read_text()).get('rollouts') != expected:
            return None
        if json.loads((checkpoint/'run_manifest.json').read_text()).get('step') != expected:
            return None
        rows = json.loads((arm_dir/'train/profile_summary.json').read_text())['rollouts']
        if [r['rollout'] for r in rows] != list(range(1, expected+1)):
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return checkpoint


def submission_command(template, suite, arm):
    command = list(template[:template.index('--')])
    for flag, value in [('--name', f'heval-final-{arm}-0917'), ('--gpu','1'), ('--cpu','16'), ('--memory','160000')]:
        command[command.index(flag)+1] = value
    return command + ['--','bash',str(suite/'run_generation.sh'),arm]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-suite', type=Path, required=True)
    p.add_argument('--eval-suite', type=Path, required=True)
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--max-hours', type=float, default=48)
    p.add_argument('--once', action='store_true')
    args = p.parse_args(argv)
    if args.poll_seconds < 10 or args.max_hours <= 0:
        p.error('poll-seconds >=10 and positive max-hours required')
    suite = args.eval_suite.resolve(); suite.mkdir(parents=True, exist_ok=True)
    lock = (suite/'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = suite/'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'arms':{a:{'state':'waiting_for_final_checkpoint'} for a in ARMS}}
    training = args.training_suite.resolve()
    template = json.loads((training/'grpo/submit_command.json').read_text())
    started = time.monotonic()
    while True:
        for arm in ARMS:
            entry = state['arms'][arm]
            out = suite/arm; out.mkdir(exist_ok=True)
            if entry['state'] == 'waiting_for_final_checkpoint':
                checkpoint = final_checkpoint_ready(training/arm)
                if checkpoint is not None:
                    command = submission_command(template,suite,arm)
                    atomic_text(out/'submit_command.json',json.dumps(command,indent=2))
                    # Persist an in-flight marker before the external mutation.
                    # Ambiguous interruptions require inspection, never auto-resubmission.
                    entry.update(state='submitting',checkpoint=str(checkpoint))
                    atomic_text(state_path,json.dumps(state,indent=2))
                    try:
                        result = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                        atomic_text(out/'submit.log',result.stdout)
                        match = re.search(r'created rjob_name:\s*(\S+)', result.stdout)
                        if result.returncode or not match:
                            raise RuntimeError(f'Submission did not confirm success: {result.stdout[-1000:]}')
                        entry.update(state='generation_submitted',job=match.group(1),submitted_at=datetime.now().astimezone().isoformat())
                        print(f'{arm}: submitted {match.group(1)}',flush=True)
                    except Exception as exc:
                        entry.update(state='submission_needs_inspection',error=repr(exc))
                elif (training/arm/'exit_code').exists() and (training/arm/'exit_code').read_text().strip() != '0':
                    entry.update(state='training_failed_no_final_eval')
            if entry['state'] == 'generation_submitted':
                exit_path = out/'exit_code'
                if exit_path.exists():
                    if exit_path.read_text().strip() != '0':
                        entry.update(state='generation_failed')
                    elif (out/'generation/generation_complete.json').exists():
                        generated = out/'generation'
                        marker = json.loads((generated/'generation_complete.json').read_text())
                        if marker['tasks'] != 164 or file_hash(generated/'samples.jsonl') != marker['samples_sha256']:
                            raise RuntimeError(f'{arm}: invalid generation completion marker')
                        entry['state']='scoring'
                        atomic_text(state_path,json.dumps(state,indent=2))
                        score_command = [sys.executable,str(suite/'source/scripts/score_humaneval.py'),
                            '--dataset',str(suite/'HumanEval.jsonl.gz'), '--samples',str(generated/'samples.jsonl'),
                            '--output',str(out/'results.json'), '--sandbox-root',str(out/'sandbox'), '--workers','4']
                        try:
                            with (out/'scoring.log').open('w') as log:
                                result=subprocess.run(score_command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                            if result.returncode:
                                raise RuntimeError(f'Scorer exited with {result.returncode}; inspect scoring.log')
                            scores=json.loads((out/'results.json').read_text())
                            entry.update(state='complete',passed=scores['passed'],tasks=scores['n_tasks'],pass_at_1=scores['pass@1'])
                            print(f"{arm}: HumanEval {scores['passed']}/{scores['n_tasks']}",flush=True)
                        except Exception as exc:
                            entry.update(state='scoring_failed',error=repr(exc))
        state['checked_at']=datetime.now().astimezone().isoformat()
        atomic_text(state_path,json.dumps(state,indent=2))
        terminal={'complete','training_failed_no_final_eval','generation_failed','scoring_failed','submission_needs_inspection','submitting','scoring'}
        if args.once or all(e['state'] in terminal for e in state['arms'].values()):
            return
        if time.monotonic()-started > args.max_hours*3600:
            atomic_text(suite/'watcher_timeout.json',json.dumps(state,indent=2))
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
