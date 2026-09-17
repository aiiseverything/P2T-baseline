from pathlib import Path
import json

import pytest


def test_final_alpaca_commands_use_instruction_benchmark_and_bounded_judge():
    from scripts.watch_final_alpaca import generation_command, judge_command
    source = Path('/frozen/source')
    command = generation_command(source, '/model', '/arm/checkpoint-250', '/refs', '/generations', 'grpo')
    assert command[1] == '/frozen/source/scripts/eval_alpaca.py'
    assert command[command.index('--adapters') + 1] == 'grpo=/arm/checkpoint-250'
    assert command[command.index('--recipes') + 1] == '1.0:1:1.0:-1'
    assert command[command.index('--max-tokens') + 1] == '2048'
    assert command[command.index('--policy-head-dtype') + 1] == 'native'
    command = judge_command('/judge-python', source, '/generations', '/refs', '/template', 'grpo')
    assert command[1] == '/frozen/source/scripts/judge_alpaca.py'
    assert command[command.index('--budget-cny') + 1] == '5'
    assert command[command.index('--workers') + 1] == '4'
    assert command[command.index('--tags') + 1] == 'grpo'


def test_final_alpaca_rejects_partial_coverage_and_wrong_checkpoint(tmp_path):
    from scripts.eval_artifacts import commit_cache, fingerprint
    from scripts.eval_policy import resolve_policy_head
    from scripts.watch_final_alpaca import validate_generation
    refs = tmp_path/'refs.jsonl'
    refs.write_text(''.join(json.dumps({'instruction':str(i),'reference_output':'reference'})+'\n' for i in range(805)))
    tag = tmp_path/'grpo'; tag.mkdir()
    gen = tag/'generations_t1.0_n1.jsonl'
    rows = [{'instruction':str(i),'response':'answer','sample_idx':0} for i in range(805)]
    checkpoint = tmp_path/'checkpoint-250'; checkpoint.mkdir()
    (checkpoint/'adapter_model.safetensors').write_bytes(b'original weights')
    config = {'adapter':fingerprint(checkpoint), 'policy':resolve_policy_head(checkpoint),
              'recipe':{'temp':1.0,'n':1,'top_p':1.0,'top_k':-1},
              'max_tokens':2048,'seed':42,'dataset':fingerprint(refs)}
    manifest = tag/'manifest_t1.0_n1.json'
    gen.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    commit_cache(manifest,config,[gen])
    assert len(validate_generation(tag,refs,tmp_path/'checkpoint-250'))==805
    with pytest.raises(ValueError,match='protocol/checkpoint'):
        validate_generation(tag,refs,tmp_path/'checkpoint-200')
    gen.write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    commit_cache(manifest,config,[gen])
    with pytest.raises(ValueError,match='coverage'):
        validate_generation(tag,refs,tmp_path/'checkpoint-250')


def test_final_alpaca_rejects_checkpoint_mutated_after_generation(tmp_path):
    from scripts.eval_artifacts import commit_cache, fingerprint
    from scripts.eval_policy import resolve_policy_head
    from scripts.watch_final_alpaca import validate_generation
    refs = tmp_path/'refs.jsonl'
    refs.write_text(''.join(json.dumps({'instruction':str(i),'reference_output':'reference'})+'\n' for i in range(805)))
    tag = tmp_path/'grpo'; tag.mkdir()
    gen = tag/'generations_t1.0_n1.jsonl'
    gen.write_text(''.join(json.dumps({'instruction':str(i),'response':'answer','sample_idx':0})+'\n' for i in range(805)))
    checkpoint = tmp_path/'checkpoint-250'; checkpoint.mkdir()
    (checkpoint/'adapter_model.safetensors').write_bytes(b'original weights')
    config = {'adapter':fingerprint(checkpoint), 'policy':resolve_policy_head(checkpoint),
              'recipe':{'temp':1.0,'n':1,'top_p':1.0,'top_k':-1},
              'max_tokens':2048,'seed':42,'dataset':fingerprint(refs)}
    commit_cache(tag/'manifest_t1.0_n1.json',config,[gen])
    (checkpoint/'adapter_model.safetensors').write_bytes(b'replaced weights')
    with pytest.raises(ValueError,match='protocol/checkpoint'):
        validate_generation(tag,refs,checkpoint)
