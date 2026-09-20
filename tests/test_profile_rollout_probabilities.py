import copy
import io
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import profile_vllm_full as profile


def payload():
    return {'ok': True, 'adapter_id': 7, 'logprobs_mode': 'processed_logprobs',
            'prompt_token_ids': [[11, 12], [21]],
            'rows': [[31, 99], [32], [41, 42, 99], [43, 99]],
            'finish_reasons': ['stop', 'length', 'stop', 'stop'],
            'selected_logprobs': [[-.1, -.2], [-.3], [-.4, -.5, -.6], [-.7, -.8]],
            'engine_generation_sec': 1.25, 'sampling': {'temperature': 1.}}


def pack(result):
    batch = {'input_ids': torch.tensor([[11, 12], [0, 21]]),
             'attention_mask': torch.tensor([[1, 1], [0, 1]])}
    return profile.pack_vllm_rollout(result, batch, ['first', 'second'], group_size=2,
                                   max_response_tokens=4, pad_token_id=0, adapter_id=7)


def test_profile_enables_rollout_importance_correction(monkeypatch):
    captured = {}
    def config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(resolved=lambda: SimpleNamespace(**kwargs))
    monkeypatch.setattr(profile, 'TrainerConfig', config)
    profile.build_trainer_config(profile.parse_args(['--output-dir', 'unused']), 'unused')
    assert captured['rollout_importance_correction'] is True


@pytest.mark.parametrize('mode', ['native', 'float32'])
def test_profile_binds_head_precision_to_actor_and_generation_server(monkeypatch, mode):
    captured = {}
    monkeypatch.setattr(profile, 'TrainerConfig', lambda **kwargs: (
        captured.update(kwargs) or SimpleNamespace(resolved=lambda: SimpleNamespace(**kwargs))))
    args = profile.parse_args(['--output-dir', 'unused', '--policy-head-dtype', mode])
    profile.build_trainer_config(args, 'unused')
    command = profile.vllm_server_command(args, '/tmp/example.sock')
    assert captured['policy_head_dtype'] == mode
    assert command[command.index('--policy-head-dtype') + 1] == mode
    assert profile.parse_args(['--output-dir', 'unused']).policy_head_dtype == 'float32'


def test_profile_rejects_unknown_policy_head_precision():
    with pytest.raises(SystemExit):
        profile.parse_args(['--output-dir', 'unused', '--policy-head-dtype', 'float16'])


def test_pack_preserves_all_group_rows_and_pads_only_invalid_probabilities():
    result = payload()
    original = copy.deepcopy(result)
    rollout, summary = pack(result)
    assert len(rollout) == 8
    ids, mask, positions, responses, response_mask, rendered, reasons, logprobs = rollout
    assert responses.tolist() == [[31, 99, 0], [32, 0, 0], [41, 42, 99], [43, 99, 0]]
    assert ids[:, :2].tolist() == [[11, 12], [11, 12], [0, 21], [0, 21]]
    assert mask[:, :2].tolist() == [[1, 1], [1, 1], [0, 1], [0, 1]]
    assert positions.tolist() == [[2, 3, 4]] * 4
    assert rendered == ['first', 'first', 'second', 'second']
    assert reasons == original['finish_reasons']
    assert logprobs.dtype == torch.float32 and logprobs.device == responses.device
    assert logprobs.shape == responses.shape
    assert logprobs[response_mask].tolist() == pytest.approx([-.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8])
    assert torch.equal(logprobs[~response_mask], torch.zeros_like(logprobs[~response_mask]))
    assert result == original
    assert not {'rows', 'selected_logprobs', 'prompt_token_ids'}.intersection(summary)
    assert summary['selected_logprobs_count'] == 8
    assert summary['mean_selected_logprob'] == pytest.approx(-.45)
    assert summary['engine_generation_sec'] == 1.25 and summary['sampling'] == {'temperature': 1.}


@pytest.mark.parametrize('change,match', [
    (lambda r: r.update(logprobs_mode='raw_logprobs'), 'processed'),
    (lambda r: r.update(adapter_id=8), 'adapter'),
    (lambda r: r.update(prompt_token_ids=[[11, 12], [0, 21]]), 'prompt'),
    (lambda r: r.update(prompt_token_ids=[[21], [11, 12]]), 'prompt'),
    (lambda r: r.pop('selected_logprobs'), 'logprob'),
    (lambda r: r['selected_logprobs'].pop(), 'logprob'),
    (lambda r: r['selected_logprobs'][0].pop(), 'logprob'),
    (lambda r: r['selected_logprobs'][0].__setitem__(0, float('nan')), 'logprob'),
    (lambda r: r['selected_logprobs'][0].__setitem__(0, float('-inf')), 'logprob'),
    (lambda r: r['selected_logprobs'][0].__setitem__(0, .01), 'logprob'),
    (lambda r: r['selected_logprobs'][0].__setitem__(0, -1e100), 'logprob'),
])
def test_pack_rejects_unbound_or_invalid_generation_probabilities(change, match):
    result = payload()
    change(result)
    with pytest.raises((ValueError, RuntimeError), match=match):
        pack(result)


def test_each_generation_request_keeps_its_own_probabilities():
    first, _ = pack(payload())
    next_result = payload()
    next_result['selected_logprobs'] = [[value - 1 for value in row]
                                      for row in next_result['selected_logprobs']]
    second, _ = pack(next_result)
    assert first[7][first[4]].tolist() == pytest.approx([-.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8])
    assert second[7][second[4]].tolist() == pytest.approx([-1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8])


def test_main_requests_and_returns_probabilities_for_calibration_and_every_training_rpc(tmp_path, monkeypatch):
    args = profile.parse_args(['--output-dir', str(tmp_path / 'train'), '--max-rollouts', '1'])
    cfg = SimpleNamespace(method='grpo', prompts_per_rollout=8, group_size=2, microbatch_responses=1,
                          max_response_tokens=4, temperature=1., min_response_tokens=0,
                          top_p=1., top_k=0, length_reward_mode='soft',
                          policy_head_dtype='float32', rollout_importance_correction=True)
    monkeypatch.setattr(profile, 'parse_args', lambda: args)
    monkeypatch.setattr(profile, 'build_trainer_config', lambda *args: cfg)
    monkeypatch.setattr(profile, 'asdict', vars)
    monkeypatch.setattr(profile, 'write_profile_calibration_manifest', lambda *args: None)
    monkeypatch.setattr(profile, 'load_prompt_dataset', lambda *a, **kw: ([str(i) for i in range(8)], [], {}))
    for name, value in [('device_count', lambda: 3), ('manual_seed_all', lambda seed: None),
                        ('get_device_name', lambda i: 'fake'), ('synchronize', lambda device: None),
                        ('reset_peak_memory_stats', lambda device: None), ('max_memory_allocated', lambda device: 0)]:
        monkeypatch.setattr(profile.torch.cuda, name, value)
    actor = SimpleNamespace(eval=lambda: None, save_pretrained=lambda path: path.mkdir())
    trainer = SimpleNamespace(actor=actor, actor_device=torch.device('cpu'), reward_device=torch.device('cpu'),
        actor_tokenizer=SimpleNamespace(pad_token_id=0), cfg=cfg, rollout_index=1,
        filtered_prompt_count=0, filter_prompts=lambda values: values, sampling_manifest=lambda: {},
        _render_chat_prompt=lambda tokenizer, prompt: prompt, save_checkpoint=lambda step: None)
    def encode(prompts):
        assert len(prompts) in (1, 2)
        return {'input_ids': torch.tensor([[11, 12], [0, 21]])[:len(prompts)],
                'attention_mask': torch.tensor([[1, 1], [0, 1]])[:len(prompts)]}, list(prompts)
    trainer._encode_prompts = encode
    observed = []
    trainer.prepare_length_reward = lambda prompts: observed.append(trainer.rollout(prompts[:2]))
    def train(prompts):
        observed.extend([trainer.rollout(prompts[:2]), trainer.rollout(prompts[2:4])])
        return {'loss': .1}
    trainer.train_rollout = train
    monkeypatch.setattr(profile.VPOTrainer, 'from_pretrained', lambda config: trainer)
    server = SimpleNamespace(pid=123456, returncode=None, poll=lambda: None)
    def start(args, path, env):
        path.touch()
        return server
    monkeypatch.setattr(profile, 'start_vllm_server', start)
    monkeypatch.setattr(profile, 'shutdown_vllm_server', lambda server, request: setattr(server, 'returncode', 0))
    requests = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, timeout): pass
        def connect(self, path): pass
        def sendall(self, raw): requests.append(json.loads(raw))
        def makefile(self, mode):
            request = requests[-1]
            result = payload()
            result.update(adapter_id=request['adapter_id'], server_pid=server.pid)
            result['selected_logprobs'] = [[value - len(requests) for value in row]
                                           for row in result['selected_logprobs']]
            return io.StringIO(json.dumps(result) + '\n')
    monkeypatch.setattr(profile.socket, 'socket', lambda *args: Connection())
    profile.main()
    assert len(observed) == 3 and all(len(rollout) == 8 for rollout in observed)
    assert all(request['return_logprobs'] is True for request in requests[:3])
    assert all(request.get('prompt_token_ids') == [[11, 12], [21]] for request in requests[:3])
    assert requests[3]['prompt_token_ids'] == [[11, 12]]
    assert [float(rollout[7][0, 0]) for rollout in observed] == pytest.approx([-1.1, -2.1, -3.1])
    record = json.loads((tmp_path / 'train/profile_metrics.jsonl').read_text())
    assert len(record['generation_requests']) == 2
    assert [item['mean_selected_logprob'] for item in record['generation_requests']] == pytest.approx([-2.45, -3.45])
    assert all('selected_logprobs' not in item for item in record['generation_requests'])
    manifest = json.loads((tmp_path / 'train/profile_manifest.json').read_text())
    assert manifest['vllm_engine']['policy_head_dtype'] == 'float32'
    assert manifest['rollout_importance_correction'] is True
