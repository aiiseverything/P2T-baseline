"""Freeze the reviewed evaluation stack before any GPU/API dispatch."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.eval_artifacts import atomic_text,file_hash,fingerprint
from scripts.eval_policy import resolve_policy_head


def main():
    source=SUITE/'source'
    if source.exists():
        raise RuntimeError('Frozen source already exists; audit changes explicitly rather than overwrite')
    source.mkdir()
    skip=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    for name in ['scripts','vpo_rm','configs','tests']:
        shutil.copytree(ROOT/name,source/name,symlinks=True,ignore=skip)
    shutil.copytree(ROOT/'third_party/arena_hard',source/'third_party/arena_hard',symlinks=True,
                   ignore=shutil.ignore_patterns('.git','data','__pycache__','*.pyc'))
    for name in ['models','datasets','.vllm-extra']:
        (source/name).symlink_to(ROOT/name,target_is_directory=True)
    previous=json.loads((ROOT/'runs/ifeval-final-canonical-20260917/experiment.json').read_text())
    manifest={
      'created_at':datetime.now().astimezone().isoformat(),
      'benchmark':'Arena-Hard v2 hard_prompt 500, GPT-4.1 two answer orders',
      'upstream_revision':'196f6b826783b3da7310e361a805fa36f0be83f3',
      'base_model':previous['base_model'],'training_suite':previous['training_suite'],
      'base_identity':fingerprint(previous['base_model'],full_weights=False),
      'models':previous['models'],'gpu_versions':previous['gpu_versions'],
      'generation':json.loads((SUITE/'inputs.json').read_text())['generation'],
      'judge':{'model':'gpt-4.1','temperature':0.0,'max_tokens':16000,'games_per_question':2,
        'baseline':'o3-mini-2025-01-31','endpoint':'https://api.linkapi.ai/v1',
        'requested_calls':6000,'pilot_calls':60,'pilot_questions_per_model':5},
      'files_sha256':{}}
    for tag,item in manifest['models'].items():
        assert resolve_policy_head(item['adapter'],'float32')==item['policy'],tag
        for path,expected in item['files_sha256'].items():
            assert file_hash(path)==expected,(tag,path)
    paths=[p for p in source.rglob('*') if p.is_file() and not p.is_symlink()]
    paths.extend(p for p in (source/'.vllm-extra').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in (SUITE/'.tiktoken-cache').rglob('*') if p.is_file())
    paths.extend(SUITE/name for name in ['prepare_inputs.py','freeze_suite.py','run_evaluation.py','run_evaluation.sh',
      'question.jsonl','inputs.json','pilot_uids.json','model_answer/o3-mini-2025-01-31.jsonl'])
    for path in sorted(paths):
        if '__pycache__' not in path.parts:
            manifest['files_sha256'][str(path.relative_to(SUITE))]=file_hash(path)
    atomic_text(SUITE/'experiment.json',json.dumps(manifest,indent=2))
    atomic_text(SUITE/'status.json',json.dumps({'state':'frozen','created_at':manifest['created_at']},indent=2))
    print(f"Frozen {len(manifest['files_sha256'])} files and six model identities")

if __name__=='__main__':main()
