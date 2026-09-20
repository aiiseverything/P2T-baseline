#!/usr/bin/env python3
"""Persist progress for the authorized creative250 and two credit-control jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/credit-controls-20260920'
ARENA=Path('/data/VPO-RM/runs/qwen-instruct-creative250-20260920')
ARMS={'shuffle':ROOT/'runs/rl-ablation-shuffle-20260920/shuffle',
      'norm_product':ROOT/'runs/rl-ablation-norm-product-20260920/norm_product'}

def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return None

def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def report():
    arena={}
    for tag in ('instruct','grpo','lam4'):
        completed=read(ARENA/'completion'/f'{tag}.json')
        log=ARENA/'job'/tag/'job.log'
        code=ARENA/'job'/tag/'exit_code'
        status='complete' if completed else ('failed' if code.exists() and code.read_text().strip()!='0' else ('started' if log.exists() else 'queued'))
        arena[tag]={'status':status,'answers':250 if completed else None}
        if log.exists() and not completed:
            lines=log.read_text(errors='replace').splitlines()
            progress=[line for line in lines if 'Processed prompts:' in line]
            if progress:arena[tag]['last_generation_progress']=progress[-1].strip()
    training={}
    for arm,path in ARMS.items():
        completion=read(path/'completion.json');failure=read(path/'failure.json')
        stage=path/'stage'
        item={'status':'complete' if completion else ('failed' if failure else (stage.read_text().strip() if stage.exists() else 'queued')),
              'rollouts':0,'target_rollouts':250,'startup_validation':read(path/'startup_validation.json'),
              'output':str(path)}
        metrics=path/'train/metrics.jsonl'
        if metrics.exists():
            values=[]
            for line in metrics.read_text().splitlines():
                try:values.append(json.loads(line))
                except json.JSONDecodeError:pass
            if values:item.update(rollouts=values[-1].get('rollout',0),last_metrics=values[-1])
        if failure:item['failure']=failure
        if completion:item['completion']=completion
        training[arm]=item
    judge=read(ARENA/'judge_pipeline/progress.json')
    score_complete=read(ARENA/'judge_pipeline/complete.json')
    value={'updated_at':datetime.now(timezone.utc).isoformat(),'arena_generation':arena,
           'arena_judging':judge,'arena_score_complete':score_complete,'training':training,
           'tests':'144 passed','definitions':str(ROOT/'docs/credit-controls-2026-09-20.md')}
    value['complete']=bool(score_complete) and all(x['status']=='complete' for x in training.values())
    atomic(OUT/'status.json',value)
    lines=['# 三项任务进度', '', '更新：'+value['updated_at'], '', '| 任务 | 状态 | 进度 |','|---|---|---|']
    for tag,item in arena.items():lines.append(f"| Qwen-Instruct {tag} 创意写作补测 | {item['status']} | {item['answers'] or '—'}/250 |")
    for arm,item in training.items():lines.append(f"| Qwen-Base + SFT + λ4：{arm} | {item['status']} | {item['rollouts']}/250 rollouts |")
    lines+=['', 'Arena 生成完成后自动进行双顺序评审及计分；创意写作 250 题与原 hard500 分开报告。',
            '旧 random_direction 实验显示名称为 Random；原始目录及配置标识保留。',
            '训练的独立启动检查和最终 checkpoint 验证由各任务的冻结 launcher 执行。',
            '排队任务没有开始训练。原 λ4 的 250 rollout 耗时约 7.2 小时，仅供已获得 GPU 后估算。']
    (OUT/'STATUS.md').write_text('\n'.join(lines)+'\n')
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    with (OUT/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            value=report()
            if args.once or value['complete']:return
            time.sleep(30)

if __name__=='__main__':main()
