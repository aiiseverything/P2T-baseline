#!/usr/bin/env python3
"""Evaluate completed final RL checkpoints with the existing Alpaca protocol."""
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
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head
from scripts.watch_final_humaneval import final_checkpoint_ready
from scripts.judge_alpaca import candidate_rows

ARMS = ('grpo', 'lam2', 'lam4', 'lam8')


def generation_command(source, model, checkpoint, refs, output, tag, policy_head_dtype='auto'):
    policy = resolve_policy_head(checkpoint, policy_head_dtype)
    return [sys.executable, str(Path(source)/'scripts/eval_alpaca.py'),
        '--model',str(model), '--dataset',str(refs), '--output',str(output),
        '--adapters',f'{tag}={checkpoint}', '--recipes','1.0:1:1.0:-1',
        '--max-tokens','2048','--seed','42', '--policy-head-dtype',policy['policy_head_dtype']]


def judge_command(python, source, output, refs, template, tag):
    return [str(python),str(Path(source)/'scripts/judge_alpaca.py'),
        '--gens-root',str(output), '--refs',str(refs), '--template',str(template),
        '--tags',tag, '--workers','4','--budget-cny','5']


def validate_generation(directory, refs, checkpoint, policy_head_dtype='auto'):
    path = Path(directory)/'generations_t1.0_n1.jsonl'
    manifest = json.loads((Path(directory)/'manifest_t1.0_n1.json').read_text())
    if manifest['outputs'] != {path.name:file_hash(path)}:
        raise ValueError('Alpaca generation hash mismatch')
    config = manifest['config']
    if (Path(config['adapter']['path']).resolve() != Path(checkpoint).resolve()
            or config['adapter'] != fingerprint(checkpoint)
            or config['recipe'] != {'temp':1.0, 'n':1, 'top_p':1.0, 'top_k':-1}
            or config['max_tokens'] != 2048 or config['seed'] != 42
            or config['dataset']['files'][0]['sha256'] != file_hash(refs)):
        raise ValueError('Alpaca generation protocol/checkpoint mismatch')
    if config.get('policy') != resolve_policy_head(checkpoint, policy_head_dtype):
        raise ValueError('Alpaca generation policy head/metadata protocol mismatch')
    references = [json.loads(x) for x in Path(refs).read_text().splitlines() if x.strip()]
    rows, _ = candidate_rows(path, references)
    if len(references) != 805 or len(rows) != 805 or any(r.get('sample_idx',0) != 0 for r in rows):
        raise ValueError('Expected exactly 805 Alpaca responses, one per instruction')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-suite', type=Path, required=True)
    p.add_argument('--eval-suite', type=Path, required=True)
    p.add_argument('--judge-python', default='/root/.venvs/alpacaeval/bin/python')
    p.add_argument('--once', action='store_true')
    args = p.parse_args(argv)
    suite = args.eval_suite.resolve(); training = args.training_suite.resolve()
    source = suite/'source'; refs = suite/'references.jsonl'
    output = suite/'generations'; template = suite/'judge_template.txt'
    lock = (suite/'watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path = suite/'status.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'benchmark':'AlpacaEval 805; internal GPT-4.1 judge, not official LC',
        'judge_budget_policy':'One invocation per arm, dispatch guard 5 CNY; no automatic retry/reset',
        'arms':{a:{'state':'waiting_for_final_checkpoint'} for a in ARMS}}
    submit_template = json.loads((training/'grpo/submit_command.json').read_text())
    start = time.monotonic()
    while True:
        for arm in ARMS:
            entry = state['arms'][arm]; arm_dir = suite/arm
            if entry['state'] == 'waiting_for_final_checkpoint':
                checkpoint = final_checkpoint_ready(training/arm)
                if checkpoint:
                    command = submit_template[:submit_template.index('--')]
                    for flag, value in [('--name',f'alpaca-final-{arm}-0917'),('--gpu','1'),('--cpu','16'),('--memory','160000')]:
                        command[command.index(flag)+1] = value
                    command += ['--','bash',str(suite/'run_generation.sh'),arm]
                    atomic_text(arm_dir/'submit_command.json',json.dumps(command,indent=2))
                    entry.update(state='submitting',checkpoint=str(checkpoint))
                    atomic_text(path,json.dumps(state,indent=2))
                    try:
                        proc = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                        atomic_text(arm_dir/'submit.log',proc.stdout)
                        match = re.search(r'created rjob_name:\s*(\S+)',proc.stdout)
                        if proc.returncode or not match:
                            raise RuntimeError('Submission not confirmed; inspect submit.log before retrying')
                        entry.update(state='generation_submitted',job=match.group(1),submitted_at=datetime.now().astimezone().isoformat())
                        print(f'{arm}: submitted {match.group(1)}',flush=True)
                    except Exception as exc:
                        entry.update(state='submission_needs_inspection',error=repr(exc))
                elif (training/arm/'exit_code').exists() and (training/arm/'exit_code').read_text().strip() != '0':
                    entry['state']='training_failed_no_final_eval'
            if entry['state']=='generation_submitted' and (arm_dir/'exit_code').exists():
                if (arm_dir/'exit_code').read_text().strip() != '0':
                    entry['state']='generation_failed'
                else:
                    try:
                        validate_generation(output/arm,refs,entry['checkpoint'])
                        # Judges run serially, so their common summary cannot race.
                        entry['state']='judging'
                        atomic_text(path,json.dumps(state,indent=2))
                        command=judge_command(args.judge_python,source,output,refs,template,arm)
                        atomic_text(arm_dir/'judge-command.json',json.dumps(command,indent=2))
                        with (arm_dir/'judge.log').open('w') as log:
                            proc=subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                        if proc.returncode:
                            raise RuntimeError('Judge failed or budget guard reached; inspect judge.log; no automatic paid retry')
                        result=json.loads((output/arm/'results_judged.json').read_text())
                        if result['n_judged']+result['n_failed_parse'] != 805:
                            raise ValueError('Judge result coverage mismatch')
                        entry.update(state='complete' if result['n_failed_parse']==0 else 'complete_with_parse_failures',
                            n_judged=result['n_judged'],n_failed_parse=result['n_failed_parse'],
                            win_rate=result['win_rate'],weighted_win_rate=result['weighted_win_rate'],spent_cny=result['spent_cny'])
                        print(f"{arm}: Alpaca WWR={result['weighted_win_rate']} WR={result['win_rate']}",flush=True)
                    except Exception as exc:
                        entry.update(state='evaluation_needs_inspection',error=repr(exc))
            state['checked_at']=datetime.now().astimezone().isoformat()
            atomic_text(path,json.dumps(state,indent=2))
        active={'waiting_for_final_checkpoint','generation_submitted'}
        if args.once or not any(e['state'] in active for e in state['arms'].values()):
            return
        if time.monotonic()-start > 48*3600:
            atomic_text(suite/'watcher_timeout.json',json.dumps(state,indent=2))
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
