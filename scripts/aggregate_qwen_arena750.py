#!/usr/bin/env python3
"""Read-only recomputation of Qwen-Instruct's combined Arena categories."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HARD=Path('/data/VPO-RM/runs/qwen-instruct-evals-20260919/arena')
CREATIVE=Path('/data/VPO-RM/runs/qwen-instruct-creative250-20260920')
OUT=CREATIVE/'combined750'
TAGS=('instruct','grpo','lam4')
VALID=('gpt4o','gpt41')
VALUES={'A>>B':[1,1,1],'A>B':[1],'A=B':[.5],'B>A':[0],'B>>A':[0,0,0]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonl(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def expand(record,order):
    v=VALUES[record['score']]
    return v if order==1 else [1-x for x in v]

def main(supplement=None, output=None):
    output=Path(output) if output else OUT
    if supplement and output == OUT:
        raise ValueError('Supplemented results require a separate output directory')
    questions={}
    records={}
    unavailable=[]
    provenance=HARD/'judge_pipeline/scores/mixed_gpt4o_gpt41/game_provenance.jsonl'
    for root,category in [(HARD,'hard_prompt'),(CREATIVE,'creative_writing')]:
        qs=jsonl(root/'question.jsonl')
        assert len(qs)==(500 if category=='hard_prompt' else 250)
        for q in qs:
            assert q['category']==category and q['uid'] not in questions
            questions[q['uid']]=q
        if category=='hard_prompt':
            lookup={(r['tag'],r['uid'],r['order']):r for r in jsonl(provenance)}
        for q in qs:
            for tag in TAGS:
                for order in (0,1):
                    key=(tag,q['uid'],order)
                    if category=='hard_prompt':
                        row=lookup[key]
                        selected=[a for a in row['attempts'] if a['state']=='valid'] if row['resolution'] in VALID else []
                        if not selected:
                            unavailable.append(dict(model=tag,uid=q['uid'],order=order,category=category,reason=row['resolution']));continue
                        assert len(selected)==1
                        item=selected[0];path=Path(item['path']);assert sha(path)==item['sha256']
                    else:
                        folder=root/'judge_pipeline/attempts'/tag/f"{q['uid']}-{order}"
                        selected=[]
                        for path in sorted(folder.glob('*/record.json')):
                            rec=read(path)
                            assert sha(path)==read(path.parent/'record_hash.json')['sha256']
                            if rec.get('status')=='valid':selected.append(path)
                        if not selected:
                            unavailable.append(dict(model=tag,uid=q['uid'],order=order,category=category,reason='unresolved_HTTP503'));continue
                        assert len(selected)==1;path=selected[0]
                    record=read(path)
                    assert record['item']==list(key) and record['status']=='valid' and record['score'] in VALUES
                    assert record['finish_reason']=='stop'
                    records[key]=dict(record=record,path=str(path),sha256=sha(path))
    if supplement:
        retry=read(supplement)
        for case in retry['cases']:
            if case['status']!='complete':continue
            key=(case['model'],case['uid'],case['order'])
            assert key not in records and questions[key[1]]['category']==case['category']
            selected=case['selected'];path=Path(selected['path'])
            assert sha(path)==selected['sha256']
            record=read(path)
            assert record['item']==list(key) and record['status']=='valid' and record['score'] in VALUES
            assert record['finish_reason']=='stop'
            root=HARD if case['category']=='hard_prompt' else CREATIVE
            original=read(root/'judge_pipeline/attempts'/key[0]/f'{key[1]}-{key[2]}'/'01/record.json')['request']
            assert record['request']['model'] in ('gpt-4o','gpt-4.1')
            assert {k:v for k,v in record['request'].items() if k!='model'}=={k:v for k,v in original.items() if k!='model'}
            records[key]=dict(record=record,path=str(path),sha256=sha(path))
        unavailable=[x for x in unavailable if (x['model'],x['uid'],x['order']) not in records]
    common=[uid for uid in questions if all((tag,uid,order) in records for tag in TAGS for order in (0,1))]
    by_category={c:[u for u in common if questions[u]['category']==c] for c in ('hard_prompt','creative_writing')}
    if not supplement:
        assert len(common)==747 and len(by_category['hard_prompt'])==499 and len(by_category['creative_writing'])==248
    else:
        assert 747<=len(common)<=750
    rows=[];per_question=[];bounds={}
    for tag in TAGS:
        row=dict(model=tag,common_valid_prompts=len(common),games=2*len(common))
        all_values=[]
        for category,uids in by_category.items():
            values=[]
            for uid in uids:
                pair=[records[tag,uid,o] for o in (0,1)]
                scores=expand(pair[0]['record'],0)+expand(pair[1]['record'],1)
                values.extend(scores)
                per_question.append(dict(model=tag,uid=uid,category=category,scores=[x['record']['score'] for x in pair],
                    expanded_scores=scores,sources=[dict(path=x['path'],sha256=x['sha256']) for x in pair]))
            row[category+'_raw_wr_pct']=100*sum(values)/len(values)
            if category=='hard_prompt':
                expected=read(HARD/'judge_pipeline/scores/mixed_gpt4o_gpt41/common_valid.json')['models'][tag]['raw']['weighted_direct_mean']*100
            else:
                expected=next(x for x in read(CREATIVE/'judge_pipeline/provisional_common248/summary.json')['models'] if x['model']==tag)['raw_wr_pct']
            if len(uids)==(499 if category=='hard_prompt' else 248):
                assert abs(row[category+'_raw_wr_pct']-expected)<1e-10
            all_values.extend(values)
        row.update(combined_raw_wr_pct=100*sum(all_values)/len(all_values),weighted_score_sum=sum(all_values),expanded_weight=len(all_values))
        rows.append(row)
        known=[];missing=0
        for uid in questions:
            for order in (0,1):
                item=records.get((tag,uid,order))
                if item:known.extend(expand(item['record'],order))
                else:missing+=1
        bounds[tag]=dict(missing_games=missing,known_games=1500-missing,
            full750_min_raw_wr_pct=100*sum(known)/(len(known)+3*missing),
            full750_max_raw_wr_pct=100*(sum(known)+3*missing)/(len(known)+3*missing))
    summary=dict(status='complete_common750' if len(common)==750 else f'provisional_common{len(common)}_of750',created_at=datetime.now(timezone.utc).isoformat(),
        requested_questions=750,common_valid_questions=len(common),category_counts={c:len(v) for c,v in by_category.items()},
        reference='gpt-4o-mini-2024-07-18',judge_policy='GPT-4o, existing GPT-4.1 fallbacks retained',
        metric='Pooled Arena raw weighted direct mean: decisive weight3, ties0.5. Recomputed from individual verdicts.',
        models=rows,unavailable_games=unavailable,full750_possible_bounds=bounds,
        delta_vpo_minus_grpo_pp=rows[2]['combined_raw_wr_pct']-rows[1]['combined_raw_wr_pct'],
        caveats=['Custom combined-category aggregate, not an official Arena-Hard leaderboard score.',
            f'{750-len(common)} common questions are incomplete. No missing verdict is assigned a win/loss/tie.',
            'Category scores are not directly averaged; their decisive-outcome weights differ.',
            'This is the raw metric used in the supplied paper table, not the style-adjusted metric.'],
        inputs_sha256={str(p):sha(p) for p in [HARD/'question.jsonl',CREATIVE/'question.jsonl',provenance,
            HARD/'judge_pipeline/campaign.json',CREATIVE/'judge_pipeline/campaign.json']})
    if supplement:summary['inputs_sha256'][str(Path(supplement).resolve())]=sha(supplement)
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (output/f'common{len(common)}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'per_question.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in per_question))
    print(json.dumps(dict(models=rows,delta_vpo_minus_grpo_pp=summary['delta_vpo_minus_grpo_pp'],full750_possible_bounds=bounds),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    main(args.supplement,args.output)
