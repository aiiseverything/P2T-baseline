import hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
suite=Path(__file__).resolve().parent
manifest=json.loads((suite/'experiment.json').read_text());source=Path(manifest['source_snapshot'])
sys.path.insert(0,str(source))
from scripts.watch_final_alpaca import generation_command,validate_generation
from scripts.watch_final_humaneval import final_checkpoint_ready
arm=sys.argv[1];assert arm in manifest['arms']
for name,expected in manifest['source_sha256'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256((suite/'references.jsonl').read_bytes()).hexdigest()==manifest['references_sha256']
checkpoint=final_checkpoint_ready(Path(manifest['training_suite'])/arm)
if checkpoint is None:raise RuntimeError('Final checkpoint is incomplete; refusing intermediate adapters')
assert hashlib.sha256((checkpoint/'run_manifest.json').read_bytes()).hexdigest()==manifest['checkpoint_run_manifest_sha256']
h=hashlib.sha256()
with (checkpoint/'adapter_model.safetensors').open('rb') as stream:
    for chunk in iter(lambda: stream.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==manifest['checkpoint_adapter_sha256']
versions={p:importlib.metadata.version(p) for p in ['torch','transformers','vllm']}
assert versions=={'torch':'2.13.0+cu129','transformers':'5.16.1','vllm':'0.28.1rc1.dev199+g7c5dc571c.cu129'},versions
(suite/arm/'runtime.json').write_text(json.dumps(versions,indent=2))
command=generation_command(source,suite.parents[1]/'models/Qwen3-14B-Base',checkpoint,suite/'references.jsonl',suite/'generations',arm)
(suite/arm/'command.json').write_text(json.dumps(command,indent=2))
subprocess.run(command,cwd=source,check=True)
rows=validate_generation(suite/'generations'/arm,suite/'references.jsonl',checkpoint)
(suite/arm/'generation_complete.json').write_text(json.dumps({'responses':len(rows),'checkpoint':str(checkpoint)},indent=2))
