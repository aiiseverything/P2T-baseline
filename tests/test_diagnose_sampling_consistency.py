import json

import pytest

from scripts.diagnose_sampling_consistency import comparison, engine_kwargs, load_cases


def test_cases_require_original_aligned_probabilities(tmp_path):
    path = tmp_path / 'cases.json'
    row = {'row': 0, 'actor_prefix_token_ids': [4], 'probe_response_token_ids': [5, 6],
           'hf_logprobs': [-1., -2.], 'vllm_logprobs': [-1.1, -2.1]}
    path.write_text(json.dumps({'cases': [row]}))
    assert load_cases(path)['cases'][0]['row'] == 0
    row['vllm_logprobs'].pop()
    path.write_text(json.dumps({'cases': [row]}))
    with pytest.raises(ValueError, match='aligned'):
        load_cases(path)


def test_engine_configs_preserve_native_except_requested_head():
    native = engine_kwargs('/model', fp32_head=False)
    precise = engine_kwargs('/model', fp32_head=True)
    assert native['enable_trace_replay'] is True
    assert native['logprobs_mode'] == 'processed_logprobs'
    assert native['lora_dtype'] == 'bfloat16'
    assert precise.pop('hf_overrides') == {'head_dtype': 'float32'}
    assert native == precise


def test_comparison_records_errors_without_a_pass_gate():
    row = comparison([-1., -2.], [-2., -4.])
    assert row['max_abs_error'] == 2.
    assert row['mean_abs_error'] == 1.5
    assert 'status' not in row
    with pytest.raises(ValueError, match='aligned'):
        comparison([1.], [])


def test_runtime_mismatch_fails_before_model_worker(tmp_path, monkeypatch):
    from scripts import diagnose_sampling_consistency as diagnostic
    from scripts.corrected_rl_launcher import RUNTIME
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps({'cases': [{'row': 0, 'actor_prefix_token_ids': [4],
        'probe_response_token_ids': [5], 'hf_logprobs': [-1.], 'vllm_logprobs': [-1.]}]}))
    runtime = {'versions': dict(RUNTIME, torch='incorrect'), 'gpus': [{}]}
    monkeypatch.setattr(diagnostic, 'runtime_report', lambda: runtime)
    monkeypatch.setattr(diagnostic, 'hf_worker', lambda *args: pytest.fail('Model must not load'))
    with pytest.raises(ValueError, match='runtime differs'):
        diagnostic.main(['--cases', str(path), '--output-dir', str(tmp_path), '--worker', 'hf'])
    assert json.loads((tmp_path / 'hf.json').read_text())['status'] == 'error'
