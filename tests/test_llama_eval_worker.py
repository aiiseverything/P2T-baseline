"""Offline coverage and provenance gates for the immutable Llama eval worker."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from scripts.eval_artifacts import file_hash, fingerprint, digest

ROOT = Path(__file__).resolve().parents[1]
TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
STOPS = [128001, 128008, 128009]


def worker():
    path = ROOT / 'scripts/llama_eval_worker.py'
    assert path.is_file(), 'The immutable Llama evaluation worker is not implemented'
    spec = importlib.util.spec_from_file_location('llama_worker_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


@pytest.fixture
def manifest(tmp_path):
    models = {}
    for tag in TAGS:
        adapter = 'none' if tag == 'base' else str(tmp_path / tag / ('sft' if tag == 'sft-init' else 'step-250'))
        files = {}
        if adapter != 'none':
            p = Path(adapter) / 'adapter_config.json';write(p, {'base_model_name_or_path': str(tmp_path/'actor')})
            w = p.with_name('adapter_model.safetensors');w.write_bytes(b'offline adapter fixture')
            files = {str(p):file_hash(p),str(w):file_hash(w)}
        models[tag] = {'adapter':adapter, 'files_sha256':files,
                       'policy':{'policy_head_dtype':'float32','metadata':[],'resolver_sha256':'a'*64}}
    value = {'benchmark':'reward256', 'models':models,'dataset':str(tmp_path/'dataset.jsonl'),
        'base_model':str(tmp_path/'actor'),'reward_model':str(tmp_path/'reward'),
        'tokenizer':str(tmp_path/'sft-tokenizer'),'seeds':[42],'scoring_seed':42,
        'gpu_versions':{'torch':'test'},'scoring_versions':{},'files_sha256':{},
        'prompt_token_ids_sha256':'b'*64, 'expected_stop_token_ids':STOPS, 'actor_pad_token_id':128004}
    for key in ('base_model','reward_model','tokenizer'):
        p=Path(value[key]);p.mkdir();write(p/'config.json',{'kind':key})
        value[key+'_fingerprint']=fingerprint(p,full_weights=False)
    Path(value['dataset']).write_text('dataset')
    value['dataset_sha256']=file_hash(value['dataset'])
    return value


def adapter_fingerprint(manifest, tag):
    path=manifest['models'][tag]['adapter']
    return None if path=='none' else fingerprint(path)


def reward_fixture(tmp_path, manifest, tag='base'):
    output=tmp_path/'reward-output';output.mkdir()
    step=0 if tag in ('base','sft-init') else 250
    prompts=['prompt '+str(i) for i in range(256)]
    write(output/'eval_prompts.json',prompts)
    rows=[{'run':tag,'step':step,'temp':1.0,'prompt':i,'score':1.25,'response_tokens':2} for i in range(256)]
    generations=[{**{k:v for k,v in r.items() if k not in ('score','response_tokens')},'token_ids':[1,2]} for r in rows]
    jsonl(output/'eval.jsonl',rows);jsonl(output/'generations.jsonl',generations)
    summary={tag:{str(step):{'1.0':{'mean':1.25,'ci95':[1.25,1.25],'mean_response_tokens':2,
        'at_token_cap_rate':0,'reward_metric':'raw_skywork_scalar_no_length_or_kl_penalty','n':256}}}}
    write(output/'summary.json',summary)
    cfg={'policy_head_dtype':'float32','policies':{f'{tag}/{step}':manifest['models'][tag]['policy']},
        'top_p':1.0,'top_k':-1,'samples_per_prompt':1,'presence_penalty':0.0,
        'reward_input_protocol':'canonical_chat_v1','reward_metric':'raw_skywork_scalar_no_length_or_kl_penalty',
        'model':manifest['base_model_fingerprint'],'rm':manifest['reward_model_fingerprint'],
        'dataset':fingerprint(manifest['dataset']),'adapters':{f'{tag}/{step}':adapter_fingerprint(manifest,tag)},
        'tokenizer':{'source':manifest['tokenizer'],'fingerprint':manifest['tokenizer_fingerprint']},
        'prompt_token_ids_sha256':manifest['prompt_token_ids_sha256'],'prompts':prompts,
        'args':{'seed':42,'num_prompts':256,'temps':[1.0],'max_tokens':2048,'min_tokens':0,
                'max_num_seqs':32,'rm_microbatch':1,'rm_device':'cuda:0',
                'model':manifest['base_model'],'rm':manifest['reward_model'],'tokenizer':manifest['tokenizer'],
                'dataset_path':manifest['dataset']}}
    cache={'config':cfg,'outputs':{p.name:file_hash(p) for p in output.iterdir()}}
    write(output/'manifest.json',cache)
    return output


def ifeval_fixture(tmp_path,manifest,tag='base'):
    manifest.update(benchmark='ifeval5',seeds=[42,43,44,45,46])
    data=[{'key':i,'prompt':f'line\u2028{i}','instruction_id_list':['one']*(2 if i<293 else 1),'kwargs':[]} for i in range(541)]
    jsonl(Path(manifest['dataset']),data);manifest['dataset_sha256']=file_hash(manifest['dataset'])
    output=tmp_path/'ifeval-output';directory=output/'seed-42'/tag
    gens=[{'key':r['key'],'prompt':r['prompt'],'responses':['yes'],'response_tokens':[2],
           'finish_reason':['stop'],'stop_reason':[128009],'last_token_id':[128009]} for r in data]
    details=[{'key':r['key'],'sample':0,'strict_list':[True]*len(r['instruction_id_list']),
              'loose_list':[True]*len(r['instruction_id_list']),'strict_all':True,'loose_all':True} for r in data]
    result={'tag':tag,'adapter':manifest['models'][tag]['adapter'],'seed':42,
        'recipe':{'temp':1.0,'n':1,'top_p':1.0,'top_k':-1},'max_tokens':2048,
        'details':details,'prompt_strict':1.0,'prompt_loose':1.0,'inst_strict':1.0,'inst_loose':1.0,
        'response_length_mean':2.0,'response_length_p95':2}
    write(directory/'results_t1.0_n1.json',result);jsonl(directory/'generations_t1.0_n1.jsonl',gens)
    cfg={'policy':manifest['models'][tag]['policy'],
        'scoring':{'protocol':'official_seeded_v1','seed':42,'langdetect_seed':42},
        'recipe':result['recipe'],'seed':42,'max_tokens':2048,
        'engine':{'seed':42,'dtype':'bfloat16','max_model_len':4096},'stop_token_ids':STOPS,
        'dataset':fingerprint(manifest['dataset']),'model':manifest['base_model_fingerprint'],
        'adapter':adapter_fingerprint(manifest,tag),
        'tokenizer':{'source':manifest['tokenizer'],'fingerprint':manifest['tokenizer_fingerprint']},
        'prompt_token_ids_sha256':manifest['prompt_token_ids_sha256']}
    write(directory/'manifest_t1.0_n1.json',{'config':cfg,'outputs':{p.name:file_hash(p) for p in directory.iterdir()}})
    return output,directory


@pytest.mark.parametrize('benchmark',['reward256','ifeval5'])
def test_command_uses_frozen_source_explicit_sft_and_matched_sampling(tmp_path,manifest,benchmark):
    manifest['benchmark']=benchmark;manifest['seeds']=[42] if benchmark=='reward256' else [42,43,44,45,46]
    command=worker().generation_command(tmp_path,manifest,'base')
    assert command[1]==str(tmp_path/'source/scripts'/('eval_checkpoints.py' if benchmark=='reward256' else 'eval_ifeval.py'))
    assert command[command.index('--tokenizer')+1]==manifest['tokenizer']
    assert command[command.index('--policy-head-dtype')+1]=='float32'
    if benchmark=='reward256':
        assert command[command.index('--rm-microbatch')+1]=='1'
        assert command[command.index('--rm-device')+1]=='cuda:0'
        assert command[command.index('--run')+1]=='base=none'
    else:
        assert command[command.index('--seeds')+1:command.index('--seeds')+6]==['42','43','44','45','46']


@pytest.mark.parametrize('tag',['base','sft-init','grpo'])
def test_reward_validator_accepts_exact_256_and_null_base_adapter(tmp_path,manifest,tag):
    out=reward_fixture(tmp_path,manifest,tag)
    result=worker().validate_reward_result(out,tag,manifest)
    assert result['n_scores']==256 and result['mean']==1.25


@pytest.mark.parametrize('damage',['coverage','infinite','token_count','head','tokenizer','microbatch','prompt_hash','extra_output','summary'])
def test_reward_bad_outputs_fail_even_with_updated_checksums(tmp_path,manifest,damage):
    out=reward_fixture(tmp_path,manifest);cache=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(x) for x in (out/'eval.jsonl').read_text().splitlines()]
    if damage=='coverage':rows[1]['prompt']=0
    elif damage=='infinite':rows[0]['score']=float('nan')
    elif damage=='token_count':rows[0]['response_tokens']=3
    elif damage=='head':cache['config']['policy_head_dtype']='native'
    elif damage=='tokenizer':cache['config']['tokenizer']['source']='wrong'
    elif damage=='microbatch':cache['config']['args']['rm_microbatch']=8
    elif damage=='prompt_hash':cache['config']['prompt_token_ids_sha256']='c'*64
    elif damage=='extra_output':cache['outputs']['extra.json']='a'*64
    elif damage=='summary':write(out/'summary.json',{'base':{'0':{'1.0':{'mean':9}}}})
    jsonl(out/'eval.jsonl',rows)
    cache['outputs'].update({p.name:file_hash(p) for p in out.iterdir() if p.name!='manifest.json'})
    write(out/'manifest.json',cache)
    with pytest.raises((ValueError,KeyError)):worker().validate_reward_result(out,'base',manifest)


@pytest.mark.parametrize('tag',['base','sft-init','grpo'])
def test_ifeval_validator_keeps_old_return_fields_and_supports_base(tmp_path,manifest,tag):
    out,_=ifeval_fixture(tmp_path,manifest,tag)
    result=worker().validate_seed_result(out,tag,42,manifest)
    assert result['prompts']==541 and result['instructions']==834 and result['prompt_strict']==1
    assert result['seed']==42 and result['scoring_seed']==42 and result['mean_tokens']==2


@pytest.mark.parametrize('damage',['tokenizer','prompt_hash','stops','base_adapter','flags','metric','coverage'])
def test_ifeval_rejects_protocol_and_coverage_drift(tmp_path,manifest,damage):
    out,d=ifeval_fixture(tmp_path,manifest);c=json.loads((d/'manifest_t1.0_n1.json').read_text())
    result=json.loads((d/'results_t1.0_n1.json').read_text())
    if damage=='tokenizer':c['config']['tokenizer']['source']='wrong'
    elif damage=='prompt_hash':c['config']['prompt_token_ids_sha256']='c'*64
    elif damage=='stops':c['config']['stop_token_ids']=[128009]
    elif damage=='base_adapter':c['config']['adapter']={'path':'wrong','files':[]}
    elif damage=='flags':result['details'][0]['strict_list']=[True]
    elif damage=='metric':result['prompt_strict']=.5
    elif damage=='coverage':result['details'].pop()
    write(d/'results_t1.0_n1.json',result)
    c['outputs']={p.name:file_hash(p) for p in d.iterdir() if not p.name.startswith('manifest')}
    write(d/'manifest_t1.0_n1.json',c)
    with pytest.raises(ValueError):worker().validate_seed_result(out,'base',42,manifest)


def test_wrong_frozen_source_is_recorded_as_failure_without_execution(tmp_path,manifest):
    write(tmp_path/'experiment.json',manifest)
    with pytest.raises(ValueError):worker().main(['--suite',str(tmp_path),'--tag','base'])
    state=json.loads((tmp_path/'job/base/status.json').read_text())
    assert state['state']=='failed' and not (tmp_path/'job/base/completion.json').exists()


def test_child_environment_overrides_unmatched_sampling():
    env=worker().child_environment({'EVAL_TOPP':'.9','EVAL_PP':'1','CUDA_VISIBLE_DEVICES':'0'})
    assert env['EVAL_TOPP']=='1.0' and env['EVAL_PP']=='0.0' and env['CUDA_VISIBLE_DEVICES']=='0'


def bound_inputs(tmp_path,manifest):
    suite=tmp_path/'bound-suite'
    names=['scripts/llama_eval_worker.py','scripts/eval_checkpoints.py','scripts/eval_artifacts.py',
           'scripts/eval_policy.py','vpo_rm/trainer.py','vpo_rm/token_policy.py','vpo_rm/integration.py',
           'vpo_rm/alignment.py','vpo_rm/model_identity.py','vpo_rm/reward.py','vpo_rm/reward_inputs.py']
    for name in names:
        p=suite/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# frozen fixture\n')
        manifest['files_sha256']['source/'+name]=file_hash(p)
    return suite


@pytest.mark.parametrize('damage',['source','adapter','base','reward','tokenizer','dataset'])
def test_changed_frozen_inputs_block_before_runtime(tmp_path,manifest,damage):
    suite=bound_inputs(tmp_path,manifest)
    worker().verify_inputs(suite,manifest,'grpo')
    path={'source':suite/'source/scripts/eval_checkpoints.py',
          'adapter':Path(manifest['models']['grpo']['adapter'])/'adapter_model.safetensors',
          'base':Path(manifest['base_model'])/'config.json',
          'reward':Path(manifest['reward_model'])/'config.json',
          'tokenizer':Path(manifest['tokenizer'])/'config.json','dataset':Path(manifest['dataset'])}[damage]
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):worker().verify_inputs(suite,manifest,'grpo')


def test_empty_runtime_bindings_fail_closed(manifest):
    manifest['gpu_versions']={}
    with pytest.raises(ValueError,match='runtime|Runtime'):
        worker().validate_runtime_versions(manifest)


def test_runtime_version_drift_is_checked_without_gpu(manifest,monkeypatch):
    m=worker();manifest['gpu_versions']={k:'expected' for k in ('torch','transformers','vllm','peft','pyarrow','pandas')}
    monkeypatch.setattr(m.importlib.metadata,'version',lambda name:'changed' if name=='vllm' else 'expected')
    with pytest.raises(ValueError,match='drift'):m.validate_runtime_versions(manifest)


class Tokenizer:
    bos_token_id=128000;pad_token_id=128004;eos_token_id=128009
    def __init__(self,ids):self.ids=ids
    def get_vocab(self):return {'<|end_of_text|>':128001,'<|eom_id|>':128008,'<|eot_id|>':128009}
    def apply_chat_template(self,*args,**kwargs):return '<|begin_of_text|>rendered'
    def __call__(self,text,add_special_tokens):
        assert add_special_tokens is False
        return {'input_ids':self.ids}


@pytest.mark.parametrize('ids,message',[([128000,128000,1],'BOS'),([1,2],'BOS'),([128000]+[1]*2048,'context')])
def test_prompt_protocol_rejects_duplicate_bos_and_context_overflow(manifest,ids,message):
    manifest['prompt_token_ids_sha256']=digest([ids])
    with pytest.raises(ValueError,match=message):worker().validate_prompt_protocol(Tokenizer(ids),['test'],manifest)


def test_prompt_budget_inclusive_boundary_and_ids_digest(manifest):
    ids=[128000]+[1]*2047
    manifest['prompt_token_ids_sha256']=digest([ids])
    assert worker().validate_prompt_protocol(Tokenizer(ids),['test'],manifest)==[ids]
    manifest['prompt_token_ids_sha256']='0'*64
    with pytest.raises(ValueError,match='IDs'):worker().validate_prompt_protocol(Tokenizer(ids),['test'],manifest)
