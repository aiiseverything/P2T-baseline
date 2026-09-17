"""Bind six-process execution and offline pandas to the previously pinned suite."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

SUITE=Path(__file__).resolve().parent
ROOT=SUITE.parents[1]
sys.path.insert(0,str(ROOT))
from scripts.eval_artifacts import atomic_text,file_hash


def main():
    manifest=json.loads((SUITE/'experiment.json').read_text())
    if manifest.get('execution',{}).get('gpu_count')==6:
        raise RuntimeError('Parallel manifest already finalized')
    for relative,expected in manifest['files_sha256'].items():
        assert file_hash(SUITE/relative)==expected,relative
    assert (SUITE/'job/attempt-1-single-gpu/exit_code').read_text().strip()=='1'
    assert list((SUITE/'model_answer').glob('*.jsonl'))==[SUITE/'model_answer/o3-mini-2025-01-31.jsonl']
    dep_check=json.loads((SUITE/'offline_dependency_check.json').read_text())
    assert dep_check['status']=='passed' and dep_check['verified_reference_style_records']==500
    test=ROOT/'tests/test_arena_parallel.py'
    shutil.copy2(test,SUITE/'source/tests'/test.name)
    paths=[SUITE/name for name in ['run_parallel_evaluation.py','run_parallel_evaluation.sh',
        'finalize_parallel_manifest.py','offline_dependency_check.json','run_full_judging.sh','source/tests/test_arena_parallel.py']]
    paths.extend(p for p in (SUITE/'.arena-extra').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for path in paths:
        manifest['files_sha256'][str(path.relative_to(SUITE))]=file_hash(path)
    manifest['execution']={'gpu_count':6,'processes':6,'one_policy_per_gpu':True,
        'policies':['base','sft-init','grpo','lam2','lam4','lam8'],
        'cpu_count':48,'memory_mib':384000,'per_worker_omp_threads':8,
        'entrypoint':'run_parallel_evaluation.sh','original_single_gpu_manifest_sha256':file_hash(SUITE/'job/attempt-1-single-gpu/experiment.json')}
    manifest['offline_gpu_dependencies']={k:dep_check['versions'][k] for k in ['pandas','python-dateutil','pytz','tzdata','six']}
    manifest['parallel_finalized_at']=datetime.now().astimezone().isoformat()
    atomic_text(SUITE/'experiment.json',json.dumps(manifest,indent=2))
    state=json.loads((SUITE/'status.json').read_text())
    state.update(state='frozen_parallel',gpu_count=6,updated_at=manifest['parallel_finalized_at'])
    atomic_text(SUITE/'status.json',json.dumps(state,indent=2))
    print(f"Pinned six-policy parallel execution, {len(manifest['files_sha256'])} code/input/dependency files")

if __name__=='__main__':main()
