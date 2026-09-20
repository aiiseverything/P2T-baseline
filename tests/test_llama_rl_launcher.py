"""Llama launch authorization must stay bound to audited inputs and real gates."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher():
    assert importlib.util.find_spec('scripts.llama_rl_launcher') is not None, 'Llama launcher is missing'
    return importlib.import_module('scripts.llama_rl_launcher')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def gpu_protocol(cpu, module):
    def phase(dtype, atol, invariance):
        return {'status': 'passed', 'score_protocol': 'raw_scalar_logit_no_sigmoid',
                'score_atol': atol, 'score_rtol': 0, 'singleton_scores': [1., 2.],
                'max_score_difference': atol / 2, 'padding_sides': ['left', 'right'],
                'max_batch_score_difference': atol / 2 if invariance else .15625,
                'terminal_eot_pooling': True, 'nonzero_input_gradient': True,
                'mapped_gradient_norm_min': .1, 'padding_gradient_max_abs': 0,
                'frozen_parameters': True, 'parameter_dtype': dtype,
                'attention_implementation': 'sdpa', 'batch_invariance_required': invariance,
                'same_input_paths': ['native', 'wrapper_no_grad', 'wrapper_input_gradients']}
    return {**cpu, 'mode': 'gpu', 'runtime': {'torch': module.RUNTIME['torch'],
        'transformers': module.RUNTIME['transformers'], 'gpu_name': 'NVIDIA H200'},
        'gpu': {'status': 'passed', 'audit_protocol': 'llama_reward_precision_audit_v2',
                'production_microbatch_responses': 1,
                'production_bf16': phase('torch.bfloat16', .125, False),
                'padding_fp32': {**phase('torch.float32', .001, True),
                                 'tf32_disabled': True, 'float32_matmul_precision': 'highest'}}}


def calibration(module, root):
    config = module.common_config(root)
    record = {'sigma0': .5, 'reward_format': 'canonical_chat_v1', 'mode': 'soft',
              'source': 'initial_policy', 'calibration_prompt_count': 128, 'group_size': 8,
              'seed': 42, 'model': config['model'], 'reward_model': config['rm'],
              'init_adapter': config['init_adapter'],
              'sampling': {'temperature': 1, 'top_p': 1, 'top_k': 0, 'min_tokens': 0,
                           'presence_penalty': 0, 'stop_token_ids': [128001, 128008, 128009],
                           'policy_head_dtype': 'float32',
                           'rollout_correction': 'detached_token_is_pg_and_kl_v1'},
              'responses': [{}] * 1024, 'prompt_sha256': [str(i) for i in range(128)]}
    record.update({key: config[key] for key in ('short_response_threshold', 'long_response_threshold',
        'short_penalty_strength', 'long_penalty_strength', 'advantage_std_floor_fraction', 'max_response_tokens')})
    return record


def test_common_settings_change_only_audited_model_and_data_paths(tmp_path):
    module = launcher()
    baseline = json.loads(module.BASELINE_MANIFEST.read_text())['common_config']
    config = module.common_config(tmp_path / 'code')
    paths = {'model', 'rm', 'init_adapter', 'dataset_path'}
    assert {k: v for k, v in config.items() if k not in paths} == {
        k: v for k, v in baseline.items() if k not in paths}
    assert config['model'] == str(tmp_path / 'models/Llama-3.1-8B-Instruct')
    assert config['rm'] == str(tmp_path / 'models/Skywork-Reward-Llama-3.1-8B-v0.2')
    assert config['init_adapter'] == str(tmp_path / 'models/sft-llama31-8b-instruct-clean2k5e2-20260917')
    assert 'length_reward_sigma0' not in config


@pytest.mark.parametrize('arm', ['grpo', 'lam2', 'lam4', 'lam8'])
def test_training_commands_preserve_formal_method_and_shared_sigma(tmp_path, arm):
    module = launcher()
    from scripts.profile_vllm_full import parse_args, build_trainer_config
    manifest = {'source_snapshot': str(tmp_path / 'snapshot'),
                'common_config': module.common_config(tmp_path / 'code'), 'arms': module.ARMS}
    command = module.training_command(manifest, arm, tmp_path / arm, sigma0=.123)
    args = parse_args(command[2:]); config = build_trainer_config(args, tmp_path / arm)
    assert config.model_name.endswith('/models/Llama-3.1-8B-Instruct')
    assert config.reward_model_name.endswith('/models/Skywork-Reward-Llama-3.1-8B-v0.2')
    assert config.rollout_iterations == 250 and config.length_reward_sigma0 == .123
    assert config.policy_head_dtype == 'float32' and config.rollout_importance_correction
    assert config.microbatch_responses == 1
    assert config.method == ('grpo' if arm == 'grpo' else 'vpo_rm')
    assert config.credit_lambda == (1 if arm == 'grpo' else int(arm[3:]))
    assert config.freeze_stop_tokens == config.freeze_structural == (arm != 'grpo')


@pytest.mark.parametrize('role,change', [('actor', {'architectures': ['LlamaForSequenceClassification']}),
    ('reward', {'architectures': ['LlamaForCausalLM']}), ('actor', {'model_type': 'qwen3'})])
def test_model_contract_rejects_wrong_model_classes(tmp_path, role, change):
    module = launcher(); config = module.common_config(tmp_path / 'code')
    for key, architecture in [('model', 'LlamaForCausalLM'), ('rm', 'LlamaForSequenceClassification')]:
        write(Path(config[key]) / 'config.json', {'model_type': 'llama', 'architectures': [architecture],
              'vocab_size': 128256, 'tie_word_embeddings': False})
    adapter = {'base_model_name_or_path': config['model'], 'r': 64, 'lora_alpha': 128,
        'lora_dropout': 0, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'bias': 'none', 'task_type': 'CAUSAL_LM'}
    write(Path(config['init_adapter']) / 'adapter_config.json', adapter)
    module.validate_model_contract(config)
    path = Path(config['model' if role == 'actor' else 'rm']) / 'config.json'
    write(path, {**json.loads(path.read_text()), **change})
    with pytest.raises(ValueError, match='actor|reward|Llama'):
        module.validate_model_contract(config)


def test_identity_rejects_changed_protected_sft_before_loading_models(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'file_hash', lambda path: 'changed')
    with pytest.raises(ValueError, match='SFT'):
        module.validation_identity(tmp_path / 'code')


@pytest.mark.parametrize('field,value', [('reward_format', 'old'), ('model', '/wrong-actor'),
    ('reward_model', '/wrong-rm'), ('init_adapter', '/wrong-sft'), ('sigma0', float('nan'))])
def test_calibration_rejects_other_models_protocols_and_bad_scales(tmp_path, field, value):
    module = launcher(); record = calibration(module, tmp_path / 'code')
    module.validate_calibration(record, module.common_config(tmp_path / 'code'))
    record[field] = value
    with pytest.raises(ValueError, match='calibration'):
        module.validate_calibration(record, module.common_config(tmp_path / 'code'))


def test_calibration_rejects_qwen_stop_tokens(tmp_path):
    module = launcher(); record = calibration(module, tmp_path / 'code')
    record['sampling']['stop_token_ids'] = [151643, 151645]
    with pytest.raises(ValueError, match='stop'):
        module.validate_calibration(record, module.common_config(tmp_path / 'code'))


def test_runtime_rejects_wrong_versions_gpu_count_or_missing_spawn():
    module = launcher()
    good = {'software': dict(module.RUNTIME), 'gpus': [{'name': 'H200', 'memory_bytes': 141 * 2**30}] * 3,
            'multiprocessing_method': 'spawn'}
    module.validate_runtime(good)
    for key, value in [('software', {'torch': 'old'}), ('gpus', good['gpus'][:2]),
                       ('multiprocessing_method', 'fork')]:
        with pytest.raises(ValueError, match='runtime|GPU|spawn'):
            module.validate_runtime({**good, key: value})


def test_submission_plan_captures_exact_four_jobs_and_nfs_mount(tmp_path):
    module = launcher()
    for arm in module.ARMS:
        command = module.submission_command(tmp_path, arm)
        assert command[:2] == ['rjob', 'submit']
        assert command[command.index('--gpu') + 1] == '3'
        assert command[command.index('--cpu') + 1] == '48'
        assert command[command.index('--memory') + 1] == '600000'
        assert command[command.index('--file-store-nfs-path') + 1] == '10.68.62.222:/data:/data'
        assert command[-3:] == ['bash', str(tmp_path / 'run_arm.sh'), arm]
    assert set(module.ARMS) == {'grpo', 'lam2', 'lam4', 'lam8'}


def test_gate_commands_select_llama_and_do_not_reuse_sigma(tmp_path):
    module = launcher()
    manifest = {'source_snapshot': str(tmp_path / 'source'), 'project_root': str(tmp_path / 'code'),
                'runtime_image': module.RUNTIME_IMAGE, 'common_config': module.common_config(tmp_path / 'code')}
    protocol, preflight = module.gate_commands(tmp_path, manifest)
    assert '--gpu' in protocol and protocol[1].endswith('/scripts/check_llama_protocol.py')
    assert protocol[protocol.index('--actor-eos') + 1] == '128009'
    assert preflight[preflight.index('--experiment-family') + 1] == 'llama'
    assert preflight[preflight.index('--experiment-profile') + 1] == 'instruct'
    index = preflight.index('--arms')
    assert preflight[index + 1:index + 5] == ['grpo', 'lam2', 'lam4', 'lam8']
    assert preflight[preflight.index('--capacity-arm') + 1] == 'lam8'
    assert preflight[preflight.index('--quality-rule') + 1] == 'strict'
    assert '--sigma0' not in preflight
    assert manifest['common_config']['model'] in protocol


def test_non_grpo_arm_cannot_start_or_run_gate_without_shared_evidence(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('Must not execute before gate'))
    with pytest.raises(FileNotFoundError, match='gate'):
        module.shared_gate(tmp_path, 'lam2', {'validation_identity': {}})


def test_source_verification_rejects_mutated_frozen_file(tmp_path):
    module = launcher(); file = tmp_path / 'source/scripts/code.py'
    file.parent.mkdir(parents=True); file.write_text('original')
    manifest = {'source_snapshot': str(tmp_path / 'source'),
                'source_sha256': {'scripts/code.py': module.file_hash(file)}}
    module.verify_source_snapshot(manifest)
    file.write_text('altered')
    with pytest.raises(ValueError, match='source'):
        module.verify_source_snapshot(manifest)


def test_gpu_report_requires_all_arms_and_pinned_runtime(tmp_path):
    module = launcher()
    identity = {'experiment_family': 'llama', 'reward_input_protocol': 'canonical_chat_v1'}
    good = {'status': 'passed', **identity, 'arms': {arm: {'status': 'passed', 'rollouts': 2,
            'runtime': {'versions': dict(module.RUNTIME)}} for arm in module.ARMS}}
    report = write(tmp_path / 'gpu.json', good)
    module.validate_report(report, identity, gpu=True)
    bad = copy.deepcopy(good); bad['arms']['lam8']['runtime']['versions']['torch'] = 'old'
    write(report, bad)
    with pytest.raises(ValueError, match='runtime'):
        module.validate_report(report, identity, gpu=True)


def _prepare_inputs(tmp_path, monkeypatch, profile):
    module = launcher()
    monkeypatch.setattr(module, 'ACTIVE_PROFILE', profile)
    project = tmp_path / 'project/code'; project.mkdir(parents=True)
    for name in ('vpo_rm', 'scripts', 'tests', 'configs'):
        (project / name).mkdir()
    (project / 'vpo_rm/example.py').write_text('x = 1\n')
    (project / 'scripts/llama_rl_launcher.py').write_text('# frozen launcher\n')
    (project / 'pyproject.toml').write_text('[project]\nname="test"\n')
    dependencies = tmp_path / 'extra'; dependencies.mkdir()
    monkeypatch.setattr(module, 'ROOT', project)
    monkeypatch.setattr(module, 'EXTRA_PACKAGES', dependencies)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    config = module.common_config(project)
    source = {'vpo_rm/example.py': module.file_hash(project / 'vpo_rm/example.py')}
    identity = {'experiment_family': 'llama', 'experiment_profile': profile,
                'reward_input_protocol': 'canonical_chat_v1',
                'source_sha256': source, 'config': config,
                'initialization_weights_sha256': 'protected', 'dataset_sha256': 'data'}
    monkeypatch.setattr(module, 'validation_identity', lambda _: identity)
    monkeypatch.setattr(module, 'audit_prompt_split', lambda _: dict(module.PROMPT_AUDIT))
    paths = {'actor': config['model'], 'reward': config['rm'], 'init_adapter': config['init_adapter']}
    protocol_identity = {}
    for role, path in paths.items():
        file = write(Path(path) / 'config.json', {'role': role})
        protocol_identity[role] = {'config.json': module.file_hash(file)}
    report = {'schema': 'llama_protocol_v1', 'status': 'passed', 'mode': 'cpu',
              'paths': paths, 'identity': protocol_identity, 'source_sha256': source,
              'protocol': {'actor_response_eos_id': module.PROFILES[profile]['actor_response_eos_id']}}
    cpu = write(tmp_path / 'cpu-protocol.json', report)
    return module, project, identity, report, cpu


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    return _prepare_inputs(tmp_path, monkeypatch, 'instruct')


@pytest.fixture
def prepared_base_inputs(tmp_path, monkeypatch):
    return _prepare_inputs(tmp_path, monkeypatch, 'base')


@pytest.fixture
def base_profile(monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'ACTIVE_PROFILE', 'base')
    return module


def test_prepare_only_freezes_inputs_and_writes_four_submission_commands(prepared_inputs, tmp_path, monkeypatch):
    module, project, identity, report, cpu = prepared_inputs
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('Prepare must not execute'))
    suite = tmp_path / 'suite'
    manifest = module.prepare_suite(suite, cpu, project_root=project)
    assert manifest['validation_identity'] == identity
    assert manifest['calibration']['mode'] == 'fresh_shared_gpu_gate'
    assert not (suite / 'gpu-validation.json').exists()
    assert not (suite / 'shared-calibration.json').exists()
    assert (suite / 'source/.vllm-extra').resolve() == module.EXTRA_PACKAGES
    assert 'VLLM_WORKER_MULTIPROC_METHOD=spawn' in (suite / 'run_arm.sh').read_text()
    for arm in module.ARMS:
        command = json.loads((suite / arm / 'submit_command.json').read_text())
        assert command == module.submission_command(suite, arm)
        assert not (suite / arm / 'train').exists()
    with pytest.raises(FileExistsError):
        module.prepare_suite(suite, cpu, project_root=project)


@pytest.mark.parametrize('change', ['failed', 'wrong_model', 'stale_source', 'wrong_mode', 'mutated_input'])
def test_prepare_rejects_stale_or_wrong_cpu_protocol_before_output(prepared_inputs, tmp_path, change):
    module, project, identity, report, cpu = prepared_inputs
    if change == 'failed': report['status'] = 'failed'
    elif change == 'wrong_model': report['paths']['actor'] = '/wrong'
    elif change == 'wrong_mode': report['mode'] = 'gpu'
    elif change == 'stale_source': report['source_sha256']['vpo_rm/example.py'] = 'stale'
    else: (Path(report['paths']['reward']) / 'config.json').write_text('mutated')
    write(cpu, report)
    suite = tmp_path / 'suite'
    with pytest.raises(ValueError, match='protocol|source|identity|input'):
        module.prepare_suite(suite, cpu, project_root=project)
    assert not suite.exists()


def test_shared_gate_runs_protocol_then_four_arm_preflight_before_publish(prepared_inputs, tmp_path, monkeypatch):
    module, project, identity, protocol, cpu = prepared_inputs
    suite = tmp_path / 'suite'
    manifest = module.prepare_suite(suite, cpu, project_root=project)
    monkeypatch.setattr(module, 'ROOT', suite / 'source')
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[1].endswith('check_llama_protocol.py'):
            write(Path(command[command.index('--output') + 1]), gpu_protocol(protocol, module))
        else:
            out = Path(command[command.index('--output-dir') + 1])
            measured = write(out / 'grpo/length_reward_calibration.json', calibration(module, project))
            arms = {arm: {'status': 'passed', 'rollouts': 2, 'runtime': {'versions': dict(module.RUNTIME)}}
                    for arm in module.ARMS}
            arms['lam8']['capacity'] = {'status': 'passed'}
            write(out / 'gpu-validation.json', {'status': 'passed', **identity, 'arms': arms,
                  'runtime_image': module.RUNTIME_IMAGE,
                  'calibration': {'path': str(measured), 'sha256': module.file_hash(measured), 'sigma0': .5}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', execute)
    record = module.shared_gate(suite, 'grpo', manifest)
    assert record['sigma0'] == .5 and len(calls) == 2
    gate = json.loads((suite / 'shared-gate.json').read_text())
    assert gate['status'] == 'passed'
    assert module.shared_gate(suite, 'lam4', manifest)['sigma0'] == .5
    assert len(calls) == 2
    (suite / 'shared-calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='hash|changed|calibration'):
        module.shared_gate(suite, 'lam8', manifest)


def test_failed_gpu_protocol_never_starts_preflight_or_publishes_gate(prepared_inputs, tmp_path, monkeypatch):
    module, project, _, protocol, cpu = prepared_inputs
    suite = tmp_path / 'suite'; manifest = module.prepare_suite(suite, cpu, project_root=project)
    monkeypatch.setattr(module, 'ROOT', suite / 'source')
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        write(Path(command[command.index('--output') + 1]), {**protocol, 'mode': 'gpu', 'status': 'failed'})
    monkeypatch.setattr(module.subprocess, 'run', execute)
    with pytest.raises(ValueError, match='protocol'):
        module.shared_gate(suite, 'grpo', manifest)
    assert len(calls) == 1 and not (suite / 'shared-gate.json').exists()
    assert json.loads((suite / 'gate-failure.json').read_text())['status'] == 'failed'


def test_completion_rejects_missing_final_checkpoint(tmp_path):
    module = launcher()
    rows = [{'rollout': step, 'loss': 0.1, 'grad_norm': .2, 'optimizer_steps': 1}
            for step in range(1, 251)]
    write(tmp_path / 'profile_summary.json', {'rollouts': rows})
    with pytest.raises((ValueError, FileNotFoundError), match='checkpoint|manifest'):
        module.validate_completion(tmp_path, {'common_config': {'max_rollouts': 250}})


@pytest.mark.parametrize('wrong', [None, 'stale_file', 'wrong_adapter', 'failed'])
def test_full_asset_evidence_is_bound_to_original_files(tmp_path, wrong):
    module = launcher(); config = module.common_config(tmp_path / 'code')
    records = {}
    for key in ('model', 'rm'):
        root = Path(config[key]); path = write(root / 'config.json', {'model': key})
        marker = write(root / 'download.json', {'verified': True}); stat = path.stat()
        records[root.name] = {'manifest': str(marker), 'manifest_sha256': module.file_hash(marker),
            'files': [{'file': path.name, 'sha256': module.file_hash(path), 'bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}]}
    init = write(Path(config['init_adapter']) / 'adapter_model.safetensors', {'adapter': True})
    report = {'status': 'passed', 'models': records,
              'sft_hashes': {'adapter_model.safetensors': module.file_hash(init)}}
    evidence = write(tmp_path / 'assets.json', report)
    if wrong == 'stale_file': (Path(config['rm']) / 'config.json').write_text('changed')
    elif wrong == 'wrong_adapter': init.write_text('changed')
    elif wrong == 'failed': write(evidence, {**report, 'status': 'failed'})
    if wrong:
        with pytest.raises(ValueError, match='asset|SFT'):
            module.validate_asset_evidence(evidence, config)
    else:
        assert module.validate_asset_evidence(evidence, config)['sha256'] == module.file_hash(evidence)


@pytest.mark.parametrize('mutation', [None, 'unchanged_default', 'changed_reference', 'nan_default'])
def test_final_adapter_validation_checks_real_tensors(tmp_path, mutation):
    import torch
    from safetensors.torch import save_file
    module = launcher(); init = tmp_path / 'init'; checkpoint = tmp_path / 'checkpoint'
    init.mkdir(); (checkpoint / 'ref').mkdir(parents=True)
    original = {'lora_A': torch.ones(2, 3), 'lora_B': torch.zeros(3, 2)}
    default = {key: value.clone() for key, value in original.items()}; default['lora_B'][0, 0] = .5
    reference = {key: value.clone() for key, value in original.items()}
    if mutation == 'unchanged_default': default = original
    elif mutation == 'changed_reference': reference['lora_A'][0, 0] = 2
    elif mutation == 'nan_default': default['lora_B'][0, 0] = float('nan')
    save_file(original, str(init / 'adapter_model.safetensors'))
    save_file(default, str(checkpoint / 'adapter_model.safetensors'))
    save_file(reference, str(checkpoint / 'ref/adapter_model.safetensors'))
    if mutation:
        with pytest.raises(ValueError, match='adapter|reference'):
            module.validate_checkpoint_adapters(checkpoint, init)
    else:
        assert module.validate_checkpoint_adapters(checkpoint, init)['changed_tensors'] == 1


@pytest.mark.parametrize('key,value', [('gpu', None), ('status', 'failed'), ('frozen_parameters', False),
    ('terminal_eot_pooling', False), ('nonzero_input_gradient', False), ('padding_gradient_max_abs', .1),
    ('mapped_gradient_norm_min', 0), ('mapped_gradient_norm_min', float('inf')),
    ('max_score_difference', .2), ('max_score_difference', float('nan')), ('score_atol', 100),
    ('padding_sides', ['right']), ('runtime', {'torch': 'old'})])
def test_gpu_protocol_requires_actual_numerical_evidence(prepared_inputs, key, value):
    module, project, _, cpu, _ = prepared_inputs
    record = gpu_protocol(cpu, module)
    config = module.common_config(project)
    module.validate_protocol_report(record, config, mode='gpu')
    if key in ('gpu', 'runtime'): record[key] = value
    else: record['gpu']['production_bf16'][key] = value
    with pytest.raises(ValueError, match='GPU|gpu|runtime|score|gradient|protocol'):
        module.validate_protocol_report(record, config, mode='gpu')


@pytest.mark.parametrize('key,value', [
    ('status', 'failed'), ('parameter_dtype', 'torch.bfloat16'),
    ('batch_invariance_required', False), ('tf32_disabled', False),
    ('float32_matmul_precision', 'high'), ('attention_implementation', 'eager'),
    ('same_input_paths', ['native', 'wrapper_no_grad']),
    ('score_atol', .125), ('max_score_difference', .002),
    ('max_batch_score_difference', .002), ('max_batch_score_difference', float('nan')),
])
def test_gpu_gate_requires_actual_full_precision_padding_control(prepared_inputs, key, value):
    module, project, _, cpu, _ = prepared_inputs
    record = gpu_protocol(cpu, module)
    config = module.common_config(project)
    module.validate_protocol_report(record, config, mode='gpu')
    record['gpu']['padding_fp32'][key] = value
    with pytest.raises(ValueError, match='GPU|protocol|precision|padding|score'):
        module.validate_protocol_report(record, config, mode='gpu')


def test_gpu_gate_rejects_missing_precision_phase_or_changed_microbatch(prepared_inputs):
    module, project, _, cpu, _ = prepared_inputs
    config = module.common_config(project)
    for phase in ('production_bf16', 'padding_fp32', 'production_microbatch_responses'):
        record = gpu_protocol(cpu, module)
        record['gpu'].pop(phase)
        with pytest.raises(ValueError, match='GPU|protocol|precision|microbatch'):
            module.validate_protocol_report(record, config, mode='gpu')


@pytest.mark.parametrize('updates,skipped', [(0, False), (2, False), (1, True)])
def test_final_metrics_reject_missing_or_extra_optimizer_updates(tmp_path, updates, skipped):
    module = launcher()
    rows = [{'rollout': i, 'loss': .1, 'grad_norm': .2, 'optimizer_steps': 1, 'skipped_rollout': False}
            for i in range(1, 251)]
    rows[-1].update(optimizer_steps=updates, skipped_rollout=skipped)
    write(tmp_path / 'profile_summary.json', {'rollouts': rows})
    with pytest.raises(ValueError, match='optimizer'):
        module.validate_completion(tmp_path, {'common_config': {'max_rollouts': 250}})


@pytest.mark.parametrize('case', ['legitimate_skip', 'updated_without_gradient',
                                  'skip_with_nan_gradient', 'skip_with_nan_loss'])
def test_completion_handles_gradient_absence_only_for_skipped_rollouts(tmp_path, case):
    """A real all-invalid skip has no gradient metric; an update must have one."""
    import torch
    from safetensors.torch import save_file
    module = launcher()
    rows = [{'rollout': i, 'loss': .1, 'grad_norm': .2, 'optimizer_steps': 1,
             'skipped_rollout': False} for i in range(1, 251)]
    # Match the trainer's all-invalid-group branch, which does not run backward.
    rows[-1] = {'rollout': 250, 'skipped_rollout': True, 'optimizer_steps': 0,
                'response_tokens': 0, 'reward_count': 0, 'loss': 0.,
                'resampled_groups': 8, 'skipped_groups': 8,
                'input_prompt_groups': 8, 'kept_prompt_groups': 0,
                'generated_response_tokens': 262144}
    if case == 'updated_without_gradient':
        rows[-1].update(skipped_rollout=False, optimizer_steps=1)
    elif case == 'skip_with_nan_gradient':
        rows[-1]['grad_norm'] = float('nan')
    elif case == 'skip_with_nan_loss':
        rows[-1]['loss'] = float('nan')
    write(tmp_path / 'profile_summary.json', {'rollouts': rows})
    checkpoint = tmp_path / 'checkpoint-250'
    initial = tmp_path / 'initial'
    initial.mkdir(); (checkpoint / 'ref').mkdir(parents=True)
    save_file({'lora_A': torch.ones(2, 3)}, str(initial / 'adapter_model.safetensors'))
    save_file({'lora_A': torch.full((2, 3), 2.)}, str(checkpoint / 'adapter_model.safetensors'))
    save_file({'lora_A': torch.ones(2, 3)}, str(checkpoint / 'ref/adapter_model.safetensors'))
    write(checkpoint / 'run_manifest.json', {'step': 250})
    torch.save({'step': 250}, checkpoint / 'trainer_state.pt')
    manifest = {'common_config': {'max_rollouts': 250, 'init_adapter': str(initial)}}
    if case == 'legitimate_skip':
        result = module.validate_completion(tmp_path, manifest)
        assert result['rollouts'] == 250
        assert result['optimizer_steps'] == 249 and result['skipped_rollouts'] == 1
    else:
        with pytest.raises(ValueError, match='finite'):
            module.validate_completion(tmp_path, manifest)


def test_base_profile_uses_pretrained_actor_native_eos_sft_and_two_arms(base_profile, tmp_path, monkeypatch):
    module = base_profile
    config = module.common_config(tmp_path / 'code')
    assert config['model'] == str(tmp_path / 'models/Llama-3.1-8B')
    assert config['init_adapter'] == str(tmp_path / 'models/sft-llama31-8b-base-clean2k5e2-20260919')
    assert config['rm'] == str(tmp_path / 'models/Skywork-Reward-Llama-3.1-8B-v0.2')
    monkeypatch.setattr(module, 'ACTIVE_PROFILE', 'instruct')
    instruct = module.common_config(tmp_path / 'code')
    monkeypatch.setattr(module, 'ACTIVE_PROFILE', 'base')
    assert {k: v for k, v in config.items() if k not in ('model', 'init_adapter')} == {
        k: v for k, v in instruct.items() if k not in ('model', 'init_adapter')}
    assert list(module.profile_arms()) == ['grpo', 'lam4']
    assert module.profile_arms()['lam4'] == module.ARMS['lam4']
    assert module.profile()['actor_response_eos_id'] == 128001
    assert module.profile()['quality_rule'] == 'final_word'
    assert module.profile()['capacity_arm'] == 'lam4'
    assert module.profile()['protected_sft_sha256'] != module.PROTECTED_SFT_SHA256
    for arm in ('grpo', 'lam4'):
        command = module.submission_command(tmp_path, arm)
        assert command[command.index('--name') + 1].startswith(f'llama-base-rl-{arm}-')
        assert command[command.index('--gpu') + 1] == '3'
        assert command[-3:] == ['bash', str(tmp_path / 'run_arm.sh'), arm]
    with pytest.raises(ValueError, match='arm'):
        module.submission_command(tmp_path, 'lam8')
    with pytest.raises(ValueError, match='profile'):
        module.set_profile('chat')


def test_base_profile_gate_commands_audit_native_eos_two_arms_and_final_word_rule(base_profile, tmp_path):
    module = base_profile
    manifest = {'source_snapshot': str(tmp_path / 'source'), 'project_root': str(tmp_path / 'code'),
                'runtime_image': module.RUNTIME_IMAGE, 'common_config': module.common_config(tmp_path / 'code')}
    protocol, preflight = module.gate_commands(tmp_path, manifest)
    assert protocol[protocol.index('--actor-eos') + 1] == '128001'
    assert protocol[protocol.index('--init-adapter') + 1].endswith('sft-llama31-8b-base-clean2k5e2-20260919')
    assert preflight[preflight.index('--experiment-profile') + 1] == 'base'
    index = preflight.index('--arms')
    assert preflight[index + 1:index + 4] == ['grpo', 'lam4', '--capacity-arm']
    assert preflight[preflight.index('--capacity-arm') + 1] == 'lam4'
    assert preflight[preflight.index('--quality-rule') + 1] == 'final_word'
    assert '--sigma0' not in preflight


def test_base_profile_gpu_report_requires_exactly_grpo_and_lam4_with_final_word_quality(base_profile, tmp_path):
    module = base_profile
    identity = {'experiment_family': 'llama', 'reward_input_protocol': 'canonical_chat_v1'}

    def arm(rule='final_word'):
        return {'status': 'passed', 'rollouts': 2, 'runtime': {'versions': dict(module.RUNTIME)},
                'quality': {'before': {'rule': rule}, 'after': {'rule': rule}}}
    good = {'status': 'passed', **identity, 'arms': {'grpo': arm(), 'lam4': arm()}}
    report = write(tmp_path / 'gpu.json', good)
    module.validate_report(report, identity, gpu=True)
    four = copy.deepcopy(good); four['arms'].update(lam2=arm(), lam8=arm())
    write(report, four)
    with pytest.raises(ValueError, match='arm'):
        module.validate_report(report, identity, gpu=True)
    strict = copy.deepcopy(good); strict['arms']['lam4']['quality']['after']['rule'] = 'strict'
    write(report, strict)
    with pytest.raises(ValueError, match='quality'):
        module.validate_report(report, identity, gpu=True)


def test_protocol_report_must_audit_the_profile_actor_eos(prepared_inputs):
    module, project, _, cpu, _ = prepared_inputs
    config = module.common_config(project)
    module.validate_protocol_report(cpu, config, mode='cpu')
    wrong = copy.deepcopy(cpu); wrong['protocol']['actor_response_eos_id'] = 128001
    with pytest.raises(ValueError, match='EOS'):
        module.validate_protocol_report(wrong, config, mode='cpu')
    wrong.pop('protocol')
    with pytest.raises(ValueError, match='EOS'):
        module.validate_protocol_report(wrong, config, mode='cpu')


def test_base_profile_sft_evidence_requires_native_eos_completion(base_profile, tmp_path):
    module = base_profile
    config = module.common_config(tmp_path / 'code')
    root = tmp_path / 'runs/llama31-base-sft-20260919'
    protected = module.PROFILES['base']['protected_sft_sha256']
    write(root / 'summary.json', {'status': 'complete', 'optimizer_steps': 157,
                                  'adapter': config['init_adapter'], 'expected_terminator': 128001})
    reload = write(root / 'job/final_reload.json', {'status': 'passed', 'response_eos_id': 128001,
        'tokenizer': {'response_eos_id': 128001}, 'hashes': {'adapter_model.safetensors': protected}})
    write(root / 'job/completion.json', {'rjob_final_state': 'Succeeded', 'exit_code': 0,
        'experiment_complete': True, 'final_reload': 'passed', 'adapter_model_sha256': protected})
    write(root / 'job/actor_integrity.json', {'status': 'passed'})
    write(Path(config['rm']) / 'DOWNLOAD_VERIFIED.json', {'status': 'complete_verified'})
    write(Path(config['model']) / 'DOWNLOAD_MANIFEST.json', {'verified': True})
    evidence = module.sft_evidence(tmp_path / 'code', config)
    assert set(evidence) == {'summary', 'reload', 'completion', 'actor_integrity', 'actor_download', 'reward_download'}
    record = json.loads(reload.read_text()); record['response_eos_id'] = 128009; write(reload, record)
    with pytest.raises(ValueError, match='SFT'):
        module.sft_evidence(tmp_path / 'code', config)


def test_prepare_records_base_profile_and_run_arm_reselects_it_from_the_manifest(prepared_base_inputs, tmp_path, monkeypatch):
    module, project, identity, report, cpu = prepared_base_inputs
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('Prepare must not execute'))
    suite = tmp_path / 'suite'
    manifest = module.prepare_suite(suite, cpu, project_root=project)
    assert manifest['experiment_profile'] == 'base' and manifest['actor_response_eos_id'] == 128001
    assert list(manifest['arms']) == ['grpo', 'lam4']
    assert 'case "$ARM" in grpo|lam4) ;;' in (suite / 'run_arm.sh').read_text()
    assert sorted(p.name for p in suite.iterdir() if p.is_dir() and p.name != 'source') == ['grpo', 'lam4']
    names = json.loads((suite / 'submission-plan.json').read_text())['names']
    assert set(names) == {'grpo', 'lam4'} and all(name.startswith('llama-base-rl-') for name in names.values())
    # A launcher process that starts with the default profile must switch to the suite's profile.
    monkeypatch.setattr(module, 'ROOT', suite / 'source')
    monkeypatch.setattr(module, 'ACTIVE_PROFILE', 'instruct')
    assert module.validate_prepared_suite(suite)['experiment_profile'] == 'base'
    assert module.ACTIVE_PROFILE == 'base'


def test_base_shared_gate_uses_native_eos_protocol_two_arms_and_lam4_capacity(prepared_base_inputs, tmp_path, monkeypatch):
    module, project, identity, protocol, cpu = prepared_base_inputs
    suite = tmp_path / 'suite'
    manifest = module.prepare_suite(suite, cpu, project_root=project)
    monkeypatch.setattr(module, 'ROOT', suite / 'source')
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if command[1].endswith('check_llama_protocol.py'):
            assert command[command.index('--actor-eos') + 1] == '128001'
            write(Path(command[command.index('--output') + 1]), gpu_protocol(protocol, module))
        else:
            index = command.index('--arms')
            assert command[index + 1:index + 3] == ['grpo', 'lam4']
            assert command[command.index('--capacity-arm') + 1] == 'lam4'
            assert command[command.index('--quality-rule') + 1] == 'final_word'
            out = Path(command[command.index('--output-dir') + 1])
            measured = write(out / 'grpo/length_reward_calibration.json', calibration(module, project))
            arms = {arm: {'status': 'passed', 'rollouts': 2, 'runtime': {'versions': dict(module.RUNTIME)},
                          'quality': {'before': {'rule': 'final_word'}, 'after': {'rule': 'final_word'}}}
                    for arm in ('grpo', 'lam4')}
            arms['lam4']['capacity'] = {'status': 'passed'}
            write(out / 'gpu-validation.json', {'status': 'passed', **identity, 'arms': arms,
                  'runtime_image': module.RUNTIME_IMAGE,
                  'calibration': {'path': str(measured), 'sha256': module.file_hash(measured), 'sigma0': .5}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', execute)
    record = module.shared_gate(suite, 'grpo', manifest)
    assert record['sigma0'] == .5 and len(calls) == 2
    assert json.loads((suite / 'shared-gate.json').read_text())['status'] == 'passed'
    assert module.shared_gate(suite, 'lam4', manifest)['sigma0'] == .5 and len(calls) == 2
    with pytest.raises(ValueError, match='arm'):
        module.run_arm(suite, 'lam8')
