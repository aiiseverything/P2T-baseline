#!/usr/bin/env python3
"""User-requested recheck of the three missing Arena games; original answers stay fixed."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.judge_arena_hard import atomic_json, parse_score, valid_usage, digest

OUT=Path('/data/VPO-RM/runs/qwen-instruct-arena750-recheck-20260920')
HARD=Path('/data/VPO-RM/runs/qwen-instruct-evals-20260919/arena')
CREATIVE=Path('/data/VPO-RM/runs/qwen-instruct-creative250-20260920')
CASES=[('hard_prompt','lam4','7bb0b31e023f4a6e',0,HARD),
       ('creative_writing','grpo','ba03f38ff6104771',1,CREATIVE),
       ('creative_writing','lam4','2251874a066b483e',1,CREATIVE)]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def immutable(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def one(case,client,key):
    category,tag,uid,order,suite=case
    source=suite/'judge_pipeline/attempts'/tag/f'{uid}-{order}'/'01/record.json'
    original=read(source)
    assert original['item']==[tag,uid,order]
    request=original['request']
    assert request['temperature']==0 and request['max_tokens']==16000 and request['model']=='gpt-4o'
    folder=OUT/category/tag/f'{uid}-{order}'
    attempts=[]
    for number,model in enumerate(['gpt-4o','gpt-4.1'],1):
        req={**request,'model':model}
        ip=folder/f'{number:02d}/intent.json';rp=folder/f'{number:02d}/record.json'
        if ip.exists():
            intent=read(ip);assert intent['request']==req
            if not rp.exists():return dict(category=category,model=tag,uid=uid,order=order,status='interrupted_inflight',attempts=attempts)
            record=read(rp);assert sha(rp)==read(rp.parent/'record_hash.json')['sha256']
        else:
            intent=dict(item=[tag,uid,order],category=category,request=req,request_sha256=digest(req),
                original_record=str(source),original_record_sha256=sha(source),attempt=number,
                local_request_id=str(uuid.uuid4()),started_at=now(),
                authorization='User explicitly requested re-evaluating the three missing questions; retry the missing game with original answers and judge settings.')
            immutable(ip,intent);record=dict(intent)
            try:
                response=client.post('https://api.linkapi.ai/v1/chat/completions',json=req)
                record['http_status']=response.status_code
                if response.status_code>=400:
                    try:
                        body=response.json();err=body.get('error',body)
                        if not isinstance(err,dict):err={'message':str(err)}
                        for field in ['type','code','message']:
                            value=str(err.get(field,''))[:2000].replace(key,'[REDACTED]')
                            value=re.sub(r'(?i)(?:sk-|Bearer\s+)[A-Za-z0-9_.-]{12,}','[REDACTED]',value)
                            record['provider_error_'+field]=value
                    except Exception:record['provider_error_type']='unparseable_error_body'
                    record['status']='http_error'
                else:
                    body=response.json();choices=body.get('choices',[])
                    if len(choices)!=1:raise ValueError('Expected exactly one judge response')
                    choice=choices[0]
                    answer=choice.get('message',{}).get('content')
                    score=parse_score(answer)
                    record.update(answer=answer,score=score,finish_reason=choice.get('finish_reason'),
                        usage=body.get('usage'),response_id=body.get('id'),response_model=body.get('model'),
                        provider_request_id=response.headers.get('x-request-id'))
                    valid=(score is not None and record['finish_reason']=='stop' and valid_usage(record['usage'])
                        and re.fullmatch(re.escape(model)+r'(?:-\d{4}-\d{2}-\d{2})?',record['response_model'] or '') is not None)
                    record['status']='valid' if valid else 'invalid'
            except Exception as e:
                record.update(status='ambiguous',error_type=type(e).__name__)
            record['finished_at']=now();immutable(rp,record);immutable(rp.parent/'record_hash.json',dict(sha256=sha(rp)))
        attempts.append(dict(path=str(rp),sha256=sha(rp),judge=model,status=record['status'],http_status=record.get('http_status')))
        print(json.dumps(dict(category=category,model=tag,uid=uid,order=order,judge=model,status=record['status'],http_status=record.get('http_status'),score=record.get('score'))),flush=True)
        if record['status']=='valid':
            return dict(category=category,model=tag,uid=uid,order=order,status='complete',selected=attempts[-1],attempts=attempts)
        if record['status']=='ambiguous':break
    return dict(category=category,model=tag,uid=uid,order=order,status='unresolved',attempts=attempts)

def main():
    import httpx
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'recheck.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(schema='user_requested_arena_missing3_recheck_v1',
            authorization='这样，你把3道题再补测一下',source_sha256=sha(__file__),
            policy='One fresh GPT-4o attempt, then one GPT-4.1 fallback per missing game. Stop at first valid verdict. Original prompts, answers, temperature0 and max_tokens16000 unchanged.',
            cases=[dict(category=c,model=t,uid=u,order=o,suite=str(s)) for c,t,u,o,s in CASES])
        mp=OUT/'manifest.json'
        if mp.exists():assert read(mp)==manifest
        else:immutable(mp,manifest)
        key=os.environ.get('LINKAPI_KEY') or Path('/root/.linkapi_key').read_text().strip()
        with httpx.Client(timeout=600,headers={'Authorization':'Bearer '+key},trust_env=True,
            limits=httpx.Limits(max_connections=3,max_keepalive_connections=3)) as client:
            rows=[]
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(one,case,client,key) for case in CASES]
                for future in as_completed(futures):
                    rows.append(future.result());atomic_json(OUT/'progress.json',dict(updated_at=now(),cases=rows))
            result=dict(status='complete' if all(r['status']=='complete' for r in rows) else 'incomplete',
                completed=sum(r['status']=='complete' for r in rows),expected=3,cases=rows,finished_at=now(),manifest_sha256=sha(mp))
            atomic_json(OUT/'result.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
