#!/usr/bin/env python3
"""Missing-only Arena recovery, preserving every original request and valid verdict."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/data/VPO-RM/runs/formal-arena750-20260920')
CREATIVE = Path('/data/VPO-RM/runs/qwen-instruct-creative250-20260920')
OUT = PROJECT / 'runs/arena-missing-recovery-20260920'
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(CAMPAIGN / 'source'))
import judge_worker as worker
import resilient_judge as transport
from scripts import judge_arena_hard as judge

read, sha, immutable = worker.read, worker.sha, worker.immutable
VALID = ('gpt4o', 'gpt41')
VALUES = {'A>>B': (3., 3), 'A>B': (1., 1), 'A=B': (.5, 1),
          'B>A': (0., 1), 'B>>A': (0., 3)}


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def checked_record(path, expected=None):
    path = Path(path)
    if expected:
        assert sha(path) == expected, path
    rec = read(path)
    assert rec['status'] == 'valid' and rec['finish_reason'] == 'stop', path
    assert judge.parse_score(rec['answer']) == rec['score'] and judge.valid_usage(rec['usage'])
    return dict(path=str(path), sha256=sha(path), score=rec['score'], judge=rec['request']['model'])


def prepare():
    OUT.mkdir(exist_ok=True)
    mp = OUT / 'manifest.json'
    if mp.exists():
        manifest = read(mp)
        assert manifest['runner_sha256'] == sha(__file__)
        for path, digest in manifest['inputs_sha256'].items():
            assert sha(path) == digest, path
        return manifest
    plan = read(CAMPAIGN / 'plan.json')
    questions = rows(CAMPAIGN / 'data/all.jsonl')
    records = {(r['model'], r['uid'], r['order']): r for r in rows(CAMPAIGN / 'results/game_provenance.jsonl')}
    old = CREATIVE / 'combined750_after_recheck/per_question.jsonl'
    for row in rows(old):
        if row['model'] == 'instruct':
            continue
        tag = 'qi42-' + row['model']
        for order, source in enumerate(row['sources']):
            records[tag, row['uid'], order] = dict(model=tag, uid=row['uid'], order=order,
                score=row['scores'][order], **source)
    for tag in ('grpo', 'lam4'):
        key = 'qi42-' + tag
        for q in questions:
            for order in (0, 1):
                token = key, q['uid'], order
                if token in records:
                    continue
                assert q['category'] == 'creative_writing'
                folder = CREATIVE / 'judge_pipeline/attempts' / tag / f"{q['uid']}-{order}"
                candidates = [p for p in folder.glob('*/record.json') if read(p).get('status') == 'valid']
                assert len(candidates) <= 1
                if candidates:
                    p = candidates[0]
                    records[token] = dict(model=key, uid=q['uid'], order=order,
                        **checked_record(p, read(p.parent / 'record_hash.json')['sha256']))
    tags = list(plan['models']) + ['qi42-grpo', 'qi42-lam4']
    cases, provenance_cache = [], {}
    for tag in tags:
        for q in questions:
            uid = q['uid']
            for order in (0, 1):
                if (tag, uid, order) in records:
                    continue
                if tag.startswith('qi42-'):
                    source = CREATIVE / 'judge_pipeline/attempts' / tag[5:] / f'{uid}-{order}/01/record.json'
                    expected_hash = read(source.parent / 'record_hash.json')['sha256']
                elif q['category'] == 'hard_prompt' and not tag.startswith('qi-'):
                    item = plan['models'][tag]
                    name = item['old_hard_provenance']
                    if name not in provenance_cache:
                        provenance_cache[name] = {(r['tag'], r['uid'], r['order']): r for r in rows(name)}
                    previous = provenance_cache[name][item['old_tag'], uid, order]
                    if 'record_path' in previous:
                        source, expected_hash = Path(previous['record_path']), previous['record_sha256']
                    else:
                        source = Path(previous['attempts'][0]['path'])
                        expected_hash = previous['attempts'][0]['sha256']
                else:
                    source = CAMPAIGN / 'attempts' / tag / f'{uid}-{order}/01/record.json'
                    expected_hash = read(source.parent / 'record_hash.json')['sha256']
                assert sha(source) == expected_hash
                rec = read(source)
                assert rec['status'] != 'valid'
                assert (rec.get('item', [None, rec.get('uid'), rec.get('order')]))[1:] == [uid, order]
                req = rec['request']
                assert req['temperature'] == 0 and req['max_tokens'] == 16000 and req['model'] == 'gpt-4o'
                assert rec['request_sha256'] == judge.digest(req)
                cases.append(dict(item=[tag, uid, order], category=q['category'], request=req,
                    original_record=str(source), original_record_sha256=expected_hash))
    assert len(records) + len(cases) == len(tags) * 1500
    assert len(cases) == 160, len(cases)
    judge.atomic_text(OUT / 'baseline_records.jsonl', ''.join(json.dumps(r) + '\n' for r in records.values()))
    manifest = dict(created_at=judge.now(), runner_sha256=sha(__file__), models=tags, cases=cases,
        authorization='用户：嗯嗯，那你那些缺失的数据补测一下。包括seed42主表和seed43重复评测的缺判。',
        policy='Preserve exact original requests. Per fresh authorized round: up to5 GPT-4o attempts, then1 GPT-4.1. Stop at first valid or uncertain delivery. No valid judgment is repeated.',
        inputs_sha256={str(p): sha(p) for p in [CAMPAIGN/'plan.json', CAMPAIGN/'data/all.jsonl',
            CAMPAIGN/'results/game_provenance.jsonl', old, OUT/'baseline_records.jsonl',
            CAMPAIGN/'judge_worker.py', CAMPAIGN/'resilient_judge.py', CAMPAIGN/'source/scripts/judge_arena_hard.py']})
    immutable(mp, manifest)
    immutable(OUT / 'billing.json', read(CAMPAIGN / 'billing.json'))
    print(json.dumps(dict(expected_missing=len(cases), by_model=dict(Counter(c['item'][0] for c in cases)))), flush=True)
    return manifest


class FreshClient:
    def __init__(self, key):
        self.key = key

    def request(self, method, url, **kwargs):
        import httpx
        assert url.startswith('https://api.linkapi.ai/v1/')
        with httpx.Client(timeout=httpx.Timeout(180, connect=30, write=30, pool=15),
                headers={'Authorization': 'Bearer ' + self.key}, trust_env=True,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)) as client:
            return getattr(client, method.lower())(url, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

    def close(self):
        pass


def recovered():
    result = {}
    for path in sorted(OUT.glob('round*/result.json')):
        for game in read(path)['games']:
            if game['resolution'] in VALID:
                key = tuple(game['item'])
                assert key not in result
                result[key] = game
    return result


def run(number, workers):
    manifest = prepare()
    output = OUT / f'round{number:02d}'
    output.mkdir(exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'result.json').exists():
            print('Round already finished; no requests sent.', flush=True)
            return
        done = recovered()
        cases = [c for c in manifest['cases'] if tuple(c['item']) not in done]
        mp = output / 'manifest.json'
        if mp.exists():
            assert read(mp)['cases'] == cases
        else:
            immutable(mp, dict(created_at=judge.now(), cases=cases, parent_sha256=sha(OUT/'manifest.json')))
        worker.ROOT = output
        worker.legacy.ROOT = OUT
        key = os.environ.get('LINKAPI_KEY') or Path('/root/.linkapi_key').read_text().strip()
        relay = object.__new__(judge.Relay)
        relay.client = transport.RetryingClient(FreshClient(key), transport.AuditLog(output/'connect_attempts.jsonl'), run_id=str(uuid.uuid4()))
        results = {}

        def save(status):
            counts = Counter(g['resolution'] for g in results.values())
            value = dict(status=status, updated_at=judge.now(), expected_games=len(cases),
                finished_games=len(results), valid_games=counts['gpt4o']+counts['gpt41'], counts=dict(counts), games=list(results.values()))
            judge.atomic_json(output/'progress.json', value)
            return value

        try:
            save('checking_billing')
            worker.legacy.billing_check(relay, 960)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, len(cases), workers * 4):
                    worker.legacy.billing_check(relay, 960)
                    batch = cases[start:start+workers*4]
                    futures = {pool.submit(worker.resolve, tuple(c['item']), c['request'], relay): tuple(c['item']) for c in batch}
                    for future in as_completed(futures):
                        item = futures[future]
                        results[item] = future.result()
                        value = save('judging')
                        print(json.dumps(dict(finished=len(results), expected=len(cases), valid=value['valid_games'],
                            item=list(item), resolution=results[item]['resolution'])), flush=True)
            value = save('complete' if all(g['resolution'] in VALID for g in results.values()) else 'incomplete')
            judge.atomic_json(output/'result.json', value)
            worker.legacy.billing_check(relay, 960, dispatch=False)
        finally:
            relay.close()


def aggregate():
    manifest = prepare()
    records = {(r['model'], r['uid'], r['order']): r for r in rows(OUT/'baseline_records.jsonl')}
    cases = {tuple(c['item']): c for c in manifest['cases']}
    for item, game in recovered().items():
        assert item not in records
        selected = game['selected']
        rec = checked_record(selected['path'], selected['sha256'])
        request = read(selected['path'])['request']
        assert {k:v for k,v in request.items() if k!='model'} == {k:v for k,v in cases[item]['request'].items() if k!='model'}
        records[item] = dict(model=item[0], uid=item[1], order=item[2], **rec)
    questions = {q['uid']: q for q in rows(CAMPAIGN/'data/all.jsonl')}
    groups = dict(qb=[k for k in manifest['models'] if k.startswith('qb-')],
        lb=[k for k in manifest['models'] if k.startswith('lb-')],
        li=[k for k in manifest['models'] if k.startswith('li-')],
        qi42=['qi-instruct', 'qi42-grpo', 'qi42-lam4'], qi43=['qi-instruct', 'qi-grpo', 'qi-lam4'])

    def pooled(tag, uids):
        total = weight = 0
        for uid in uids:
            for order in (0, 1):
                value, w = VALUES[records[tag, uid, order]['score']]
                total += value if order == 1 else w-value
                weight += w
        return dict(prompts=len(uids), weighted_sum=total, expanded_weight=weight,
                    raw_wr_pct=100*total/weight if weight else None)

    scored = {}
    for family, tags in groups.items():
        common = [u for u in questions if all((t,u,o) in records for t in tags for o in (0,1))]
        scored[family] = dict(common_valid_questions=len(common), common_uids=common,
            models={t:dict(combined=pooled(t,common), categories={c:pooled(t,[u for u in common if questions[u]['category']==c])
                for c in ('hard_prompt','creative_writing')}) for t in tags})
    missing = [dict(model=t, uid=u, order=o, category=questions[u]['category'])
        for t in manifest['models'] for u in questions for o in (0,1) if (t,u,o) not in records]
    result = dict(status='complete750' if not missing else 'incomplete', updated_at=judge.now(),
        recovered_games=len(recovered()), originally_missing=160, remaining_games=missing, groups=scored)
    dest = OUT/'results'; dest.mkdir(exist_ok=True)
    judge.atomic_json(dest/'summary.json',result)
    judge.atomic_text(dest/'game_provenance.jsonl',''.join(json.dumps(r)+'\n' for r in records.values()))
    original=read(CAMPAIGN/'original_tables.json'); report=[]
    for name in ('main','ablation'):
        values=[]
        for row in original[name]:
            tag=row['model']; group=tag[:2]
            if group=='qi':
                group='qi42'; tag=tag if tag=='qi-instruct' else tag.replace('qi-','qi42-')
            cells=row['values'].copy();cells[-1]=f"{scored[group]['models'][tag]['combined']['raw_wr_pct']:.2f}"
            assert cells[:-1]==row['values'][:-1]
            values.append(cells)
        with (dest/f'{name}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(original['headers']);writer.writerows(values)
        report += [name,'','| '+' | '.join(original['headers'])+' |','| '+' | '.join(['---']*7)+' |']
        report += ['| '+' | '.join(v)+' |' for v in values]+['']
    report += ['Main tables use generation seed42. Qwen-Instruct seed43 is reported separately in summary.json.',
        'Raw pooled two-order Arena verdicts, decisive weight3, tie0.5; GPT-4o-mini reference, GPT-4o / GPT-4.1 judges.',
        'Original shuffle row actually denotes random_direction; non-Arena cells preserved verbatim.',
        'Coverage: '+json.dumps({g:v['common_valid_questions'] for g,v in scored.items()}),
        f'Missing judgments: {len(missing)}. Do not impute missing verdicts.']
    judge.atomic_text(dest/'tables.md','\n'.join(report)+'\n')
    print(json.dumps(dict(status=result['status'],recovered=result['recovered_games'],remaining=len(missing),
        groups={g:dict(n=v['common_valid_questions'],scores={t:m['combined']['raw_wr_pct'] for t,m in v['models'].items()}) for g,v in scored.items()})),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','aggregate'])
    parser.add_argument('--round',type=int,default=1);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();assert args.round>=1 and 1<=args.workers<=8
    if args.command=='prepare':prepare()
    elif args.command=='run':run(args.round,args.workers)
    else:aggregate()
