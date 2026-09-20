#!/usr/bin/env python3
"""Prepare, but do not submit, the missing Qwen-Instruct Arena creative250 campaign."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/VPO-RM/runs/qwen-instruct-evals-20260919/arena')
SUITE = Path('/data/VPO-RM/runs/qwen-instruct-creative250-20260920')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')
def rows(p):
    return [json.loads(s) for s in p.open() if s.strip()]
def jsonl(p, values):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))

def main():
    if SUITE.exists():
        raise FileExistsError(SUITE)
    source = SUITE / 'source'
    shutil.copytree(OLD / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['eval_arena_hard.py', 'judge_arena_hard.py']:
        shutil.copy2(ROOT / 'scripts' / name, source / 'scripts' / name)
    shutil.copytree(OLD / '.tiktoken-cache', SUITE / '.tiktoken-cache')
    allq = rows(ROOT / 'third_party/arena_hard/data/arena-hard-v2.0/question.jsonl')
    oldq = rows(OLD / 'question.jsonl')
    known = {q['uid'] for q in oldq}
    questions = [q for q in allq if q['uid'] not in known]
    assert len(allq) == 750 and len(known) == 500 and len(questions) == 250
    assert all(q['category'] == 'creative_writing' for q in questions)
    assert {q['uid'] for q in questions}.isdisjoint(known)
    jsonl(SUITE / 'question.jsonl', questions)
    refsource = ROOT / 'runs/arena-hard-v2-gpt4o-judge-20260917/reference/official-gpt-4o-mini-2024-07-18-750.jsonl'
    ref = {r['uid']: r for r in rows(refsource)}
    selected = [ref[q['uid']] for q in questions]
    for q, r in zip(questions, selected):
        assert r['messages'][0]['content'] == q['prompt']
        assert r['model'] == 'gpt-4o-mini-2024-07-18'
    jsonl(SUITE / 'model_answer/gpt-4o-mini-2024-07-18.jsonl', selected)
    old = json.loads((OLD / 'generation_manifest.json').read_text())
    manifest = {k: old[k] for k in ('tags','base_model','base_model_identity','models','gpu_versions','generation','base_fingerprint')}
    manifest.update(schema='qwen_instruct_creative250_v1', subset='creative_writing', n_questions=250,
        original_campaign=str(OLD), original_generation_manifest_sha256=sha(OLD/'generation_manifest.json'),
        source_root=str(source), questions=str(SUITE/'question.jsonl'), reference_source=str(refsource),
        reference_source_sha256=sha(refsource), old_question_sha256=sha(OLD/'question.jsonl'))
    worker = '''#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, importlib.metadata
SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE / 'source'))
from scripts import eval_arena_hard as arena
from scripts.eval_artifacts import file_hash, fingerprint
from scripts.judge_arena_hard import atomic_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True,choices=['instruct','grpo','lam4'])
    tag=parser.parse_args().tag
    manifest=json.loads((SUITE/'generation_manifest.json').read_text())
    before=file_hash(SUITE/'generation_manifest.json')
    for name, digest in manifest['files_sha256'].items():
        assert file_hash(SUITE/name)==digest, name
    item=manifest['models'][tag]
    assert fingerprint(manifest['base_model'],full_weights=False)==manifest['base_fingerprint']
    assert item['adapter']=='none' if tag=='instruct' else Path(item['adapter']).name=='checkpoint-250'
    if tag!='instruct':
        assert fingerprint(item['adapter'])==item['fingerprint']
        run=json.loads((Path(item['adapter'])/'run_manifest.json').read_text())
        assert run['resolved_config']['init_adapter']=='' and run['step']==250
    import torch
    assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
    actual={name:importlib.metadata.version(name) for name in manifest['gpu_versions']}
    assert actual==manifest['gpu_versions'], actual
    args=argparse.Namespace(model=manifest['base_model'],dataset=str(SUITE/'question.jsonl'),
        output=str(SUITE),adapters=[tag+'='+item['adapter']],category='creative_writing',
        max_tokens=4096,max_model_len=16384,max_num_seqs=32,seed=42,policy_head_dtype='float32')
    summary=arena.generate_all(args)[tag]
    assert summary['n_answers']==250
    assert file_hash(SUITE/'generation_manifest.json')==before
    atomic_json(SUITE/'completion'/f'{tag}.json', dict(status='complete',tag=tag,**summary,
        answer_sha256=file_hash(summary['answer_path']),manifest_sha256=file_hash(summary['manifest_path']),
        generation_manifest_sha256=before))
if __name__=='__main__':main()
'''
    (SUITE / 'generation_worker.py').write_text(worker)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
umask 077
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:?tag required}"
SHARED=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
mkdir -p "$SUITE/job/$TAG" "$SUITE/cache/$TAG" "$SUITE/tmp/$TAG"
exec > >(tee -a "$SUITE/job/$TAG/job.log") 2>&1
trap 'rc=$?; printf "%s\\n" "$rc" > "$SUITE/job/$TAG/exit_code"' EXIT
export TMPDIR="$SUITE/tmp/$TAG" TMP="$SUITE/tmp/$TAG" TEMP="$SUITE/tmp/$TAG"
export XDG_CACHE_HOME="$SUITE/cache/$TAG" HF_HOME="$SUITE/cache/$TAG/hf"
export VLLM_CACHE_ROOT="$SUITE/cache/$TAG/vllm" TORCHINDUCTOR_CACHE_DIR="$SUITE/cache/$TAG/inductor"
export TRITON_CACHE_DIR="$SUITE/cache/$TAG/triton" CUDA_CACHE_PATH="$SUITE/cache/$TAG/cuda"
export PYTHONPATH="$SUITE/source:$SHARED/runs/arena-hard-v2-canonical-20260917/.arena-extra:$SHARED/.vllm-extra"
export TIKTOKEN_CACHE_DIR="$SUITE/.tiktoken-cache" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EVAL_TOPP=1.0 EVAL_PP=0.0
cd "$SUITE/source"
python3 "$SUITE/generation_worker.py" --tag "$TAG"
'''
    (SUITE / 'run_generation.sh').write_text(shell)
    manifest['files_sha256'] = {str(p.relative_to(SUITE)):sha(p) for p in source.rglob('*') if p.is_file()}
    for name in ('question.jsonl','model_answer/gpt-4o-mini-2024-07-18.jsonl','generation_worker.py','run_generation.sh'):
        manifest['files_sha256'][name]=sha(SUITE/name)
    save(SUITE/'generation_manifest.json',manifest)
    # Reuse the audited failure policy and score math; select creative-writing prompt.
    pipe=SUITE/'judge_pipeline';pipe.mkdir()
    shutil.copytree(source,pipe/'source')
    for name in ('resilient_judge.py','arena_exclusion_policy.py'):
        shutil.copy2(OLD/'judge_pipeline'/name,pipe/name)
    s=(OLD/'judge_pipeline/pipeline.py').read_text()
    s=s.replace("QUESTION_SHA='39485afe9e6df0b392b5806d6d584a8ee01584d4bdda2f1bdeea47ed9432dc90'",f"QUESTION_SHA='{sha(SUITE/'question.jsonl')}'")
    s=s.replace("REFERENCE_SHA='595c165539eeabda138d864826a679eae90a3e5bfc0db046eb71fb66b1e26086'",f"REFERENCE_SHA='{sha(SUITE/'model_answer/gpt-4o-mini-2024-07-18.jsonl')}'")
    s=s.replace("judge_model='gpt-4o')", "judge_model='gpt-4o',category='creative_writing')")
    s=s.replace("judge.indexed(questions,'questions')","judge.indexed(questions,'questions',250)")
    s=s.replace("record.get('n_answers')==500","record.get('n_answers')==250")
    s=s.replace('expected_games=3000','expected_games=1500')
    s=s.replace("subset=subset,baseline=BASELINE,judge_policy=", "subset=subset,category='creative_writing',baseline=BASELINE,judge_policy=")
    (pipe/'pipeline.py').write_text(s)
    sh=(OLD/'judge_pipeline/run_controller.sh').read_text().replace(str(OLD/'judge_pipeline'),str(pipe))
    (pipe/'run_controller.sh').write_text(sh)
    save(pipe/'freeze.json',dict(generation_manifest_sha256=sha(SUITE/'generation_manifest.json'),
        files_sha256={str(p.relative_to(pipe)):sha(p) for p in pipe.rglob('*') if p.is_file()}))
    template=json.loads((OLD.parent/'gen_alpaca_arena/submit-lam4-attempt2.json').read_text())['command']
    for tag in manifest['tags']:
        c=template[:template.index('--')]
        c[c.index('--name')+1]='qwi-creative250-'+tag+'-0920'
        c+=['--','bash',str(SUITE/'run_generation.sh'),tag]
        save(SUITE/'job'/tag/'submit_command.json',c)
    (SUITE/'README.md').write_text('Qwen3-14B-Instruct direct-RL final checkpoints: native instruct, GRPO, VPO lambda4.\n'
        'Exactly the 250 creative_writing UIDs absent from the previous hard500 campaign.\n'
        'Seed42, temperature1, max4096 tokens, FP32 policy head; no SFT initialization.\n'
        'Reference: published GPT-4o-mini-2024-07-18. Official creative-writing judge prompt, GPT-4o with the existing five-attempt/GPT-4.1 fallback policy.\n'
        'Report creative250 separately; do not replace hard500 or call either an official leaderboard score.\n')
    print(SUITE)

if __name__=='__main__': main()
