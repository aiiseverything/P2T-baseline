import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_corrected_arm_commands_preserve_soft_protocol_and_restart_sft(tmp_path):
    from scripts.corrected_rl_launcher import ARMS, common_config, training_command
    from scripts.profile_vllm_full import parse_args, build_trainer_config
    manifest = {'source_snapshot': str(tmp_path / 'source'), 'common_config': common_config(tmp_path), 'arms': ARMS}
    for arm, expected in [('grpo', 1), ('lam2', 2), ('lam4', 4), ('lam8', 8)]:
        command = training_command(manifest, arm, tmp_path / arm / 'train', sigma0=.25)
        args = parse_args(command[2:])
        config = build_trainer_config(args, tmp_path / arm / 'train')
        assert config.rollout_iterations == 250
        assert config.credit_lambda == expected
        assert config.length_reward_mode == 'soft' and config.length_reward_sigma0 == .25
        assert config.init_adapter == str(tmp_path / 'models/sft-native-eos-clean2k5e2')
        assert (config.learning_rate, config.beta, config.temperature) == (5e-5, .03, 1)
        assert (config.short_response_threshold, config.long_response_threshold) == (8, 1024)
        assert (config.short_penalty_strength, config.long_penalty_strength) == (.5, 2)
        assert config.advantage_std_floor_fraction == .5
        assert (config.min_response_tokens, config.max_response_tokens) == (0, 2048)
        assert config.freeze_structural == (arm != 'grpo')
        assert args.keep_adapters_every == args.checkpoint_interval == 250


def test_failed_or_mismatched_validation_cannot_authorize_a_suite(tmp_path):
    from scripts.corrected_rl_launcher import validate_report
    identity = {'source_sha256': {'vpo_rm/reward_inputs.py': 'source'}, 'config': {'seed': 42},
                'initialization_weights_sha256': 'init', 'dataset_sha256': 'data',
                'reward_input_protocol': 'canonical_chat_v1'}
    path = tmp_path / 'cpu.json'
    path.write_text(json.dumps({'status': 'failed', **identity}))
    with pytest.raises(ValueError, match='validation'):
        validate_report(path, identity)
    path.write_text(json.dumps({'status': 'passed', **identity, 'source_sha256': {'vpo_rm/reward_inputs.py': 'old'}}))
    with pytest.raises(ValueError, match='source_sha256'):
        validate_report(path, identity)
    path.write_text(json.dumps({'status': 'passed', **identity}))
    assert validate_report(path, identity)['status'] == 'passed'


def test_gpu_gate_requires_all_four_successful_two_rollout_arms(tmp_path):
    from scripts.corrected_rl_launcher import validate_report
    path = tmp_path / 'gpu.json'
    path.write_text(json.dumps({'status': 'passed', 'arms': {'grpo': {'status': 'passed', 'rollouts': 2}}}))
    with pytest.raises(ValueError, match='four|arm'):
        validate_report(path, {}, gpu=True)


@pytest.mark.parametrize('bad_versions', [{'torch': 'old'}, {}])
def test_gpu_gate_rejects_unverified_runtime_versions(tmp_path, bad_versions):
    from scripts.corrected_rl_launcher import ARMS, RUNTIME, validate_report
    report = {'status': 'passed', 'arms': {arm: {
        'status': 'passed', 'rollouts': 2, 'runtime': {'versions': dict(RUNTIME)}} for arm in ARMS}}
    path = tmp_path / 'gpu.json'
    path.write_text(json.dumps(report))
    validate_report(path, {}, gpu=True)
    report['arms']['lam8']['runtime']['versions'] = bad_versions
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='runtime'):
        validate_report(path, {}, gpu=True)


def test_prepare_rejects_a_different_preflight_container_image(tmp_path, monkeypatch):
    from scripts import corrected_rl_launcher as launcher
    monkeypatch.setattr(launcher, 'validation_identity', lambda root: {})
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100 * 2**30))
    cpu, gpu = tmp_path / 'cpu.json', tmp_path / 'gpu.json'
    cpu.write_text(json.dumps({'status': 'passed'}))
    gpu.write_text(json.dumps({'status': 'passed', 'runtime_image': 'validated:image',
        'arms': {arm: {'status': 'passed', 'rollouts': 2,
                      'runtime': {'versions': launcher.RUNTIME}} for arm in launcher.ARMS}}))
    template = tmp_path / 'submit.json'
    template.write_text(json.dumps(['rjob', 'submit', '--name', 'old', '--gpu', '3',
        '--cpu', '48', '--memory', '600000', '--image', 'different:image', '--', 'bash', 'old.sh']))
    suite = tmp_path / 'new-suite'
    with pytest.raises(ValueError, match='image'):
        launcher.prepare_suite(suite, cpu, gpu, project_root=tmp_path, submit_template=template)
    assert not suite.exists()


def test_prepare_rejects_missing_validation_before_creating_output(tmp_path, monkeypatch):
    from scripts import corrected_rl_launcher as launcher
    monkeypatch.setattr(launcher, 'validation_identity', lambda root: {})
    suite = tmp_path / 'new-suite'
    with pytest.raises(FileNotFoundError):
        launcher.prepare_suite(suite, tmp_path / 'missing-cpu.json', tmp_path / 'missing-gpu.json', project_root=tmp_path)
    assert not suite.exists()


def test_preflight_calibration_requires_canonical_protocol_and_exact_inputs(tmp_path):
    from scripts.corrected_rl_launcher import common_config, validate_calibration
    config = common_config(tmp_path)
    record = {'sigma0': .25, 'reward_format': 'old-prefix', 'calibration_prompt_count': 128}
    with pytest.raises(ValueError, match='calibration'):
        validate_calibration(record, config)


def test_initial_rollout_gate_rejects_nan_and_inconsistent_termination(tmp_path):
    from scripts.corrected_rl_launcher import check_initial_rollouts
    manifest = {'common_config': {'max_response_tokens': 4}, 'startup_max_unmapped_content_fraction': .25}
    rows = [{'rollout': i, 'loss': .1, 'grad_norm': .2, 'rm_mapped_tokens': 2,
             'rm_unmapped_content_fraction': 0, 'skipped_rollout': False,
             'response_tokens': 3, 'reward_count': 1} for i in (1, 2)]
    (tmp_path / 'profile_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (tmp_path / 'profile_manifest.json').write_text(json.dumps({'reward_input_protocol': 'canonical_chat_v1',
                                                             'sampling': {'stop_token_ids': [99]}}))
    for step in (1, 2):
        (tmp_path / f'rollout-{step}-tokens.json').write_text(json.dumps([[1, 2, 99]]))
        (tmp_path / f'rollout-{step}-rewards.json').write_text(json.dumps([{'length': 3, 'finish_reason': 'stop'}]))
    assert check_initial_rollouts(tmp_path, manifest)['status'] == 'passed'
    rows[1]['loss'] = float('nan')
    (tmp_path / 'profile_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError, match='finite'):
        check_initial_rollouts(tmp_path, manifest)
    rows[1]['loss'] = .1
    (tmp_path / 'profile_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (tmp_path / 'rollout-2-tokens.json').write_text(json.dumps([[1, 99, 2]]))
    with pytest.raises(ValueError, match='termination'):
        check_initial_rollouts(tmp_path, manifest)


@pytest.mark.parametrize('reuse_calibration', [False, True])
def test_prepare_freezes_sources_and_only_writes_submission_plans(tmp_path, monkeypatch, reuse_calibration):
    from scripts import corrected_rl_launcher as launcher
    project = tmp_path / 'project'; project.mkdir()
    for name in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs', 'models', 'datasets'):
        (project / name).mkdir()
    (project / 'vpo_rm/reward_inputs.py').write_text('PROTOCOL = "canonical_chat_v1"\n')
    (project / 'pyproject.toml').write_text('[project]\nname="test"\n')
    identity = {'source_sha256': {'vpo_rm/reward_inputs.py': launcher.file_hash(project / 'vpo_rm/reward_inputs.py')},
                'config': {}, 'initialization_weights_sha256': 'init', 'dataset_sha256': 'data',
                'reward_input_protocol': 'canonical_chat_v1'}
    monkeypatch.setattr(launcher, 'validation_identity', lambda root: identity)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100 * 2**30))
    cpu, gpu = tmp_path / 'cpu.json', tmp_path / 'gpu.json'
    cpu.write_text(json.dumps({'status': 'passed', **identity}))
    gpu_report = {'status': 'passed', **identity, 'runtime_image': 'validated:image',
        'arms': {arm: {'status': 'passed', 'rollouts': 2,
                      'runtime': {'versions': launcher.RUNTIME}} for arm in launcher.ARMS}}
    if reuse_calibration:
        config = launcher.common_config(project)
        record = {'sigma0': .125, 'reward_format': 'canonical_chat_v1', 'mode': 'soft',
                  'source': 'initial_policy', 'calibration_prompt_count': 128, 'group_size': 8,
                  'seed': 42, 'model': config['model'], 'reward_model': config['rm'],
                  'init_adapter': config['init_adapter'],
                  'sampling': {'temperature': 1, 'top_p': 1, 'top_k': 0, 'min_tokens': 0, 'presence_penalty': 0,
                               'policy_head_dtype': 'float32', 'rollout_correction': 'detached_token_is_pg_and_kl_v1'},
                  'responses': [{}] * 1024, 'prompt_sha256': [str(i) for i in range(128)]}
        record.update({key: config[key] for key in ('short_response_threshold', 'long_response_threshold',
                      'short_penalty_strength', 'long_penalty_strength', 'advantage_std_floor_fraction', 'max_response_tokens')})
        calibration = tmp_path / 'new-canonical-calibration.json'
        calibration.write_text(json.dumps(record))
        gpu_report['calibration'] = {'path': str(calibration), 'sha256': launcher.file_hash(calibration)}
    gpu.write_text(json.dumps(gpu_report))
    template = tmp_path / 'submit.json'
    template.write_text(json.dumps(['rjob', 'submit', '--name', 'old', '--gpu', '3',
                                  '--cpu', '48', '--memory', '600000', '--image', 'validated:image',
                                  '--', 'bash', 'old.sh']))
    def no_external_execution(*args, **kwargs):
        pytest.fail('Preparing a suite must not execute any external command')
    monkeypatch.setattr(launcher.subprocess, 'Popen', no_external_execution)
    suite = tmp_path / 'fresh-suite'
    manifest = launcher.prepare_suite(suite, cpu, gpu, project_root=project, submit_template=template)
    assert manifest['calibration']['mode'] == ('verified_gpu_preflight' if reuse_calibration else 'fresh_flock_pilot')
    if reuse_calibration:
        saved = json.loads((suite / 'shared-calibration.json').read_text())
        assert saved['sigma0'] == .125
        assert saved['_verification']['source_calibration_sha256'] == gpu_report['calibration']['sha256']
    assert manifest['common_config']['max_rollouts'] == 250
    assert (suite / 'source/models').resolve() == project / 'models'
    plan = json.loads((suite / 'submission-plan.json').read_text())
    assert plan['status'] == 'prepared_not_submitted' and set(plan['names']) == set(launcher.ARMS)
    for arm in launcher.ARMS:
        command = json.loads((suite / arm / 'submit_command.json').read_text())
        assert command[-3:] == ['bash', str(suite / 'run_arm.sh'), arm]
        assert not (suite / arm / 'train').exists()
    with pytest.raises(FileExistsError):
        launcher.prepare_suite(suite, cpu, gpu, project_root=project, submit_template=template)


def test_training_supervisor_rejects_success_exit_without_startup_evidence(tmp_path):
    from scripts.corrected_rl_launcher import run_training_process
    manifest = {'common_config': {'max_response_tokens': 4}}
    with pytest.raises(RuntimeError, match='without two validated'):
        run_training_process([sys.executable, '-c', 'pass'], tmp_path, tmp_path / 'train',
                             manifest, status_dir=tmp_path)
    assert json.loads((tmp_path / 'startup_or_training_failure.json').read_text())['status'] == 'failed'


def test_runtime_manifest_cannot_change_hyperparameters_after_validation(tmp_path):
    from scripts.corrected_rl_launcher import ARMS, common_config, validate_prepared_manifest
    config = common_config(tmp_path)
    identity = {'config': {k: v for k, v in config.items() if k != 'max_rollouts'},
                'reward_input_protocol': 'canonical_chat_v1'}
    manifest = {'project_root': str(tmp_path), 'common_config': config, 'arms': ARMS,
                'validation_identity': identity, 'reward_input_protocol': 'canonical_chat_v1'}
    validate_prepared_manifest(manifest, identity)
    manifest['common_config']['learning_rate'] = 1.
    with pytest.raises(ValueError, match='config'):
        validate_prepared_manifest(manifest, identity)


def test_identity_rejects_changed_protected_sft_weights(tmp_path, monkeypatch):
    from scripts import corrected_rl_launcher as launcher
    monkeypatch.setattr(launcher, 'file_hash', lambda path: 'changed-weights')
    with pytest.raises(ValueError, match='SFT'):
        launcher.validation_identity(tmp_path)


@pytest.mark.parametrize('changed_input', [
    'models/Qwen3-14B-Base/model.safetensors',
    'models/Skywork-Reward-V2-Qwen3-8B/model.safetensors',
    'models/Skywork-Reward-V2-Qwen3-8B/tokenizer_config.json',
    'models/sft-native-eos-clean2k5e2/adapter_config.json',
    'datasets/alpacaeval/eval_gpt4turbo_reference.jsonl',
    'datasets/ifeval/ifeval_input_data.jsonl',
    'datasets/gsm8k/test.jsonl',
])
def test_validation_rejects_inputs_mutated_in_place(tmp_path, monkeypatch, changed_input):
    from scripts import corrected_rl_launcher as launcher
    from scripts import profile_vllm_full as profile
    from vpo_rm.data import DEFAULT_BENCHMARK_PATHS, ROOT as DATA_ROOT
    config = launcher.common_config(tmp_path)
    inputs = [Path(config['model']) / 'model.safetensors',
              Path(config['rm']) / 'model.safetensors',
              Path(config['rm']) / 'tokenizer_config.json',
              Path(config['init_adapter']) / 'adapter_model.safetensors',
              Path(config['init_adapter']) / 'adapter_config.json',
              Path(config['dataset_path']),
              *(tmp_path / path.relative_to(DATA_ROOT) for path in DEFAULT_BENCHMARK_PATHS.values())]
    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    monkeypatch.setattr(launcher, 'PROTECTED_SFT_SHA256',
                        launcher.file_hash(Path(config['init_adapter']) / 'adapter_model.safetensors'))
    monkeypatch.setattr(profile, 'profile_source_manifest', lambda: {'source_sha256': {},
                        'reward_input_protocol': 'canonical_chat_v1'})
    identity = launcher.validation_identity(tmp_path)
    report = tmp_path / 'validation.json'
    report.write_text(json.dumps({'status': 'passed', **identity}))
    launcher.validate_report(report, identity)
    (tmp_path / changed_input).write_text('mutated-input')
    with pytest.raises(ValueError, match='fingerprints|benchmark_exclusion_sha256'):
        launcher.validate_report(report, launcher.validation_identity(tmp_path))


@pytest.mark.parametrize('changed_key,changed_value', [
    ('policy_head_dtype', 'native'), ('policy_head_dtype', None),
    ('rollout_correction', 'none'), ('rollout_correction', None),
])
def test_calibration_rejects_pre_precision_or_uncorrected_policy(tmp_path, changed_key, changed_value):
    from scripts.corrected_rl_launcher import common_config, validate_calibration
    config = common_config(tmp_path)
    record = {'sigma0': .125, 'reward_format': 'canonical_chat_v1', 'mode': 'soft',
              'source': 'initial_policy', 'calibration_prompt_count': 128, 'group_size': 8,
              'seed': 42, 'model': config['model'], 'reward_model': config['rm'],
              'init_adapter': config['init_adapter'],
              'sampling': {'temperature': 1, 'top_p': 1, 'top_k': 0, 'min_tokens': 0,
                           'presence_penalty': 0, 'policy_head_dtype': 'float32',
                           'rollout_correction': 'detached_token_is_pg_and_kl_v1'},
              'responses': [{}] * 1024, 'prompt_sha256': [str(i) for i in range(128)]}
    record.update({key: config[key] for key in ('short_response_threshold', 'long_response_threshold',
                  'short_penalty_strength', 'long_penalty_strength', 'advantage_std_floor_fraction',
                  'max_response_tokens')})
    validate_calibration(record, config)
    record['sampling'][changed_key] = changed_value
    with pytest.raises(ValueError, match='sampling|precision|correction'):
        validate_calibration(record, config)


def corrected_startup_fixture(tmp_path):
    manifest = {'common_config': {'max_response_tokens': 4, 'policy_head_dtype': 'float32'}}
    profile = {'reward_input_protocol': 'canonical_chat_v1',
               'config': {'policy_head_dtype': 'float32', 'rollout_importance_correction': True},
               'sampling': {'stop_token_ids': [99], 'policy_head_dtype': 'float32',
                            'rollout_correction': 'detached_token_is_pg_and_kl_v1'}}
    # Deliberately broad finite IS/error values prove there is no extra numeric gate.
    probability = {'rollout_is_min': 1e-12, 'rollout_is_mean': 1e8, 'rollout_is_max': 1e12,
                   'rollout_is_p01': 1e-10, 'rollout_is_p50': 1., 'rollout_is_p99': 1e10,
                   'rollout_is_ess_ratio': 1e-9, 'rollout_logp_abs_error_mean': 2.,
                   'rollout_logp_abs_error_p99': 10., 'rollout_logp_abs_error_max': 20.,
                   'rollout_direct_ratio_clip_fraction': .99,
                   'initial_hf_logp_max_abs_error': .01, 'initial_hf_ratio_clip_fraction': 0.}
    rows = [{'rollout': step, 'loss': .1, 'grad_norm': .2, 'rm_mapped_tokens': 2,
             'rm_unmapped_content_fraction': 0, 'response_tokens': 3, 'reward_count': 1,
             **probability} for step in (1, 2)]
    (tmp_path / 'profile_manifest.json').write_text(json.dumps(profile))
    (tmp_path / 'profile_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    for step in (1, 2):
        (tmp_path / f'rollout-{step}-tokens.json').write_text(json.dumps([[1, 2, 99]]))
        (tmp_path / f'rollout-{step}-rewards.json').write_text(json.dumps([{'length': 3, 'finish_reason': 'stop'}]))
    return manifest, profile, rows


@pytest.mark.parametrize('key,value', [
    ('rollout_is_min', 0.), ('rollout_is_mean', float('nan')),
    ('rollout_is_max', float('inf')), ('rollout_is_ess_ratio', -1.),
    ('rollout_logp_abs_error_mean', float('nan')), ('rollout_logp_abs_error_p99', None),
    ('initial_hf_logp_max_abs_error', None), ('initial_hf_ratio_clip_fraction', .01),
])
def test_startup_rejects_invalid_correction_metrics_without_new_distribution_threshold(tmp_path, key, value):
    from scripts.corrected_rl_launcher import check_initial_rollouts
    manifest, _, rows = corrected_startup_fixture(tmp_path)
    assert check_initial_rollouts(tmp_path, manifest)['status'] == 'passed'
    rows[1][key] = value
    (tmp_path / 'profile_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='importance|probability|initial HF'):
        check_initial_rollouts(tmp_path, manifest)


@pytest.mark.parametrize('section,key,value', [
    ('config', 'policy_head_dtype', 'native'),
    ('config', 'rollout_importance_correction', False),
    ('sampling', 'policy_head_dtype', 'native'),
    ('sampling', 'rollout_correction', 'none'),
])
def test_startup_rejects_pre_precision_or_uncorrected_policy(tmp_path, section, key, value):
    from scripts.corrected_rl_launcher import check_initial_rollouts
    manifest, profile, _ = corrected_startup_fixture(tmp_path)
    profile[section][key] = value
    (tmp_path / 'profile_manifest.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='precision|correction'):
        check_initial_rollouts(tmp_path, manifest)
