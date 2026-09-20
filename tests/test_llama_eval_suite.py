"""Experiment preparation must preserve the requested comparison and resources."""
import importlib
import json
from pathlib import Path

import pytest


def suite_module():
    return importlib.import_module('scripts.llama_eval_suite')


@pytest.mark.parametrize('benchmark', ['reward256', 'ifeval5'])
def test_six_distinct_single_gpu_jobs_mount_data_and_share_frozen_worker(tmp_path, benchmark):
    module = suite_module()
    suite = tmp_path / benchmark
    commands = [module.submission_command(suite, tag) for tag in module.TAGS]
    assert len(commands) == len({c[c.index('--name') + 1] for c in commands}) == 6
    for tag, command in zip(module.TAGS, commands):
        assert command[command.index('--gpu') + 1] == '1'
        assert command[command.index('--file-store-nfs-path') + 1] == '10.68.62.222:/data:/data'
        assert command[-3:] == ['bash', str(suite / 'run_worker.sh'), tag]
        assert command[command.index('--memory') + 1] == '128000'
    with pytest.raises(ValueError):
        module.submission_command(suite, 'obsolete')


def test_model_inputs_require_completed_training_and_match_export_bytes(tmp_path):
    module = suite_module()
    model = tmp_path / 'actor'; model.mkdir()
    sft = tmp_path / 'sft'; sft.mkdir()
    (sft / 'adapter_model.safetensors').write_bytes(b'sft')
    (sft / 'adapter_config.json').write_text('{}')
    root = tmp_path / 'train'; root.mkdir()
    manifest = {'common_config': {'model': str(model), 'init_adapter': str(sft)}}
    for arm in ('grpo', 'lam2', 'lam4', 'lam8'):
        path = root / arm / 'train/vllm-adapters/step-250'; path.mkdir(parents=True)
        (path / 'adapter_model.safetensors').write_bytes(arm.encode())
        (path / 'adapter_config.json').write_text('{}')
        (path.parent.parent / 'profile_manifest.json').write_text(json.dumps({'policy_head_dtype': 'float32'}))
        (root / arm / 'completion.json').write_text(json.dumps({'status': 'complete',
            'rollouts': 250, 'optimizer_steps': 250, 'skipped_rollouts': 0,
            'adapter_sha256': module.file_hash(path / 'adapter_model.safetensors')}))
    models = module.model_inputs(root, manifest, expected_sft_sha=module.file_hash(sft / 'adapter_model.safetensors'))
    assert models['base']['adapter'] == 'none'
    assert models['base']['policy']['policy_head_dtype'] == 'float32'
    assert models['sft-init']['adapter'] == str(sft)
    assert models['lam8']['adapter'].endswith('/step-250')
    (root / 'lam8/train/vllm-adapters/step-250/adapter_model.safetensors').write_bytes(b'drift')
    with pytest.raises(ValueError, match='export|checkpoint'):
        module.model_inputs(root, manifest, expected_sft_sha=module.file_hash(sft / 'adapter_model.safetensors'))


def test_ifeval_aggregation_includes_all_30_rows_and_mean_of_four_metrics():
    module = suite_module()
    rows = [{'model': tag, 'seed': seed, 'prompt_strict': .1, 'prompt_loose': .2,
             'inst_strict': .3, 'inst_loose': .4, 'mean_tokens': 20, 'capped': 0}
            for tag in module.TAGS for seed in range(42, 47)]
    individual, means = module.aggregate_ifeval(rows)
    assert len(individual) == 30 and len(means) == 6
    assert all(row['four_metric_mean_pct'] == 25 for row in individual)
    assert all(row['four_metric_mean_mean_pct'] == 25 and row['four_metric_mean_sd_pp'] == 0 for row in means)
    with pytest.raises(ValueError, match='30|coverage'):
        module.aggregate_ifeval(rows[:-1])
    with pytest.raises(ValueError, match='30|coverage'):
        module.aggregate_ifeval(rows[:-1] + [rows[0]])


def test_watch_detects_worker_crash_before_status_exists(tmp_path):
    module = suite_module()
    job = tmp_path / 'job/base'; job.mkdir(parents=True)
    (job / 'exit_code').write_text('1\n')
    assert module.worker_states(tmp_path)[0] == 'failed'
    (job / 'status.json').write_text('{"state":"complete"}')
    assert module.worker_states(tmp_path)[0] == 'failed'
    (job / 'exit_code').write_text('0\n')
    assert module.worker_states(tmp_path)[0] == 'complete'


def test_accepted_training_rejects_changed_completion_even_with_same_status(tmp_path):
    module = suite_module()
    bound = tmp_path / 'grpo/completion.json'; bound.parent.mkdir()
    bound.write_text('{"status":"complete"}')
    audit = {'status': 'passed', 'inputs_sha256': {str(bound): module.file_hash(bound)}, 'arms': {}}
    bound.write_text('{"status":"complete", "adapter_sha256":"changed"}')
    with pytest.raises(ValueError, match='acceptance|audit'):
        module.validate_training_acceptance(tmp_path, audit)
