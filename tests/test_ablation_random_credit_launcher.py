"""The random-credit ablation must differ from canonical lambda4 in exactly one variable."""
import copy
import json
from pathlib import Path

import pytest
import torch


def launcher():
    from scripts import ablation_random_credit_launcher as module
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def manifest_for(tmp_path, source):
    module = launcher()
    from scripts.corrected_rl_launcher import common_config
    return {'source_snapshot': str(tmp_path / 'source'), 'common_config': common_config(tmp_path),
            'arms': {source: module.arm_spec(source)}}


def canonical_resolved_for(tmp_path):
    """What the canonical lambda4 arm resolved to, computed from the same shared launcher."""
    from scripts.corrected_rl_launcher import ARMS, common_config, training_command
    module = launcher()
    manifest = {'source_snapshot': str(tmp_path / 'source'), 'common_config': common_config(tmp_path), 'arms': ARMS}
    command = training_command(manifest, 'lam4', tmp_path / 'canonical/train', sigma0=3.0323)
    resolved = module.resolved_from_command(command, tmp_path / 'canonical/train')
    resolved.pop('credit_source')
    return resolved


@pytest.mark.parametrize('source', ['random_direction', 'random_band', 'shuffle', 'norm_product'])
def test_training_command_changes_only_the_credit_source(tmp_path, source):
    module = launcher()
    manifest = manifest_for(tmp_path, source)
    command = module.training_command(manifest, source, tmp_path / source / 'train', sigma0=3.0323)
    assert command[-2:] == ['--credit-source', source]
    assert command[command.index('--credit-lambda') + 1] == '4'
    assert '--freeze-stop-tokens' in command and '--freeze-structural' in command
    assert command[command.index('--length-reward-sigma0') + 1] == '3.0323'
    check = module.verify_single_variable(command, tmp_path / source / 'train', canonical_resolved_for(tmp_path), source)
    assert check['status'] == 'passed' and check['differences'] == ['credit_source', 'output_dir']
    assert check['resolved_config']['credit_source'] == source
    assert check['resolved_config']['method'] == 'vpo_rm' and check['resolved_config']['credit_lambda'] == 4
    with pytest.raises(ValueError, match='sigma0'):
        module.training_command(manifest, source, tmp_path / 'x', sigma0=None)


def test_single_variable_check_rejects_any_other_configuration_drift(tmp_path):
    module = launcher()
    source = 'random_direction'
    manifest = manifest_for(tmp_path, source)
    command = module.training_command(manifest, source, tmp_path / 'train', sigma0=3.0323)
    canonical = canonical_resolved_for(tmp_path)
    for index, value in ((command.index('--learning-rate') + 1, '1e-4'), (command.index('--beta') + 1, '0.01'),
                         (command.index('--credit-lambda') + 1, '8')):
        drifted = list(command); drifted[index] = value
        with pytest.raises(ValueError, match='only'):
            module.verify_single_variable(drifted, tmp_path / 'train', canonical, source)
    no_freeze = [item for item in command if item != '--freeze-structural']
    with pytest.raises(ValueError, match='only'):
        module.verify_single_variable(no_freeze, tmp_path / 'train', canonical, source)
    rm_gradient = list(command); rm_gradient[-1] = 'rm_gradient'
    with pytest.raises(ValueError, match='only|random'):
        module.verify_single_variable(rm_gradient, tmp_path / 'train', canonical, source)
    with pytest.raises(ValueError, match='random'):
        module.verify_single_variable(command, tmp_path / 'train', canonical, 'random_band')


def test_arm_spec_and_sources_are_bound_to_canonical_lambda4():
    module = launcher()
    from scripts.corrected_rl_launcher import ARMS
    for source in ('random_direction', 'random_band'):
        spec = module.arm_spec(source)
        assert {k: v for k, v in spec.items() if k != 'credit_source'} == ARMS['lam4']
        assert spec['credit_source'] == source
    with pytest.raises(ValueError, match='source'):
        module.arm_spec('uniform')


def test_submission_command_reuses_the_canonical_envelope_with_an_ablation_name(tmp_path):
    module = launcher()
    if not (module.canonical_suite() / 'lam4/submit_command.json').is_file():
        pytest.skip('Canonical suite is not available')
    manifest = {'arms': {'random_direction': module.arm_spec('random_direction')}}
    command = module.submission_command(tmp_path, 'random_direction', manifest)
    assert command[:2] == ['rjob', 'submit']
    assert command[command.index('--name') + 1].startswith('rl-abl-randdir-')
    assert command[command.index('--gpu') + 1] == '3' and command[command.index('--cpu') + 1] == '48'
    assert command[command.index('--memory') + 1] == '600000'
    assert command[command.index('--image') + 1] == module.RUNTIME_IMAGE
    assert command[-6:] == ['--', '/usr/bin/env', 'OMP_NUM_THREADS=8', 'bash', str(tmp_path / 'run_arm.sh'), 'random_direction']


def test_canonical_bindings_verify_hashes_when_the_suite_is_present(tmp_path):
    module = launcher()
    suite = module.canonical_suite()
    if not (suite / 'experiment.json').is_file():
        pytest.skip('Canonical suite is not available')
    manifest = module.canonical_manifest(suite)
    assert manifest['arms']['lam4']['credit_lambda'] == 4
    from scripts.corrected_rl_launcher import common_config
    path, calibration = module.canonical_calibration(manifest, common_config(module.ROOT), suite)
    assert calibration['sigma0'] == pytest.approx(3.0323000897825447)
    _, resolved = module.canonical_resolved_config(suite)
    # A frozen source snapshot must resolve the canonical suite through the project root, never itself.
    assert module.canonical_suite(tmp_path / 'project') == (tmp_path / 'project').resolve() / module.CANONICAL_RELATIVE
    assert resolved['method'] == 'vpo_rm' and resolved['credit_lambda'] == 4 and 'credit_source' not in resolved


@pytest.mark.parametrize('case', ['complete', 'unchanged_adapter', 'changed_reference', 'wrong_source', 'missing_rollout'])
def test_completion_validation_checks_rollouts_manifest_and_adapters(tmp_path, case):
    from safetensors.torch import save_file
    module = launcher()
    rows = [{'rollout': i, 'loss': .1, 'grad_norm': .2, 'optimizer_steps': 1, 'skipped_rollout': False}
            for i in range(1, 251)]
    if case == 'missing_rollout':
        rows = rows[:-1]
    write(tmp_path / 'train/profile_summary.json', {'rollouts': rows})
    checkpoint = tmp_path / 'train/checkpoint-250'
    (checkpoint / 'ref').mkdir(parents=True)
    init = tmp_path / 'init'; init.mkdir()
    original = {'lora_A': torch.ones(2, 3), 'lora_B': torch.zeros(3, 2)}
    trained = {k: v.clone() for k, v in original.items()}; trained['lora_B'][0, 0] = .5
    reference = {k: v.clone() for k, v in original.items()}
    if case == 'unchanged_adapter':
        trained = {k: v.clone() for k, v in original.items()}
    if case == 'changed_reference':
        reference['lora_A'][0, 0] = 2.
    save_file(original, str(init / 'adapter_model.safetensors'))
    save_file(trained, str(checkpoint / 'adapter_model.safetensors'))
    save_file(reference, str(checkpoint / 'ref/adapter_model.safetensors'))
    torch.save({'step': 250}, checkpoint / 'trainer_state.pt')
    write(checkpoint / 'run_manifest.json', {'step': 250, 'resolved_config': {
        'credit_source': 'rm_gradient' if case == 'wrong_source' else 'random_direction'}})
    manifest = {'common_config': {'init_adapter': str(init)}}
    if case == 'complete':
        result = module.validate_completion(tmp_path / 'train', manifest)
        assert result['rollouts'] == 250 and result['changed_tensors'] == 1 and result['reference_unchanged']
    else:
        with pytest.raises(ValueError, match='rollout|adapter|reference|ablation'):
            module.validate_completion(tmp_path / 'train', manifest)


def test_run_arm_resolves_canonical_suite_from_the_manifest_project_root(tmp_path, monkeypatch):
    """Inside a frozen snapshot, ROOT is the snapshot; canonical evidence lives in the project root."""
    module = launcher()
    snapshot = tmp_path / 'suite/source'
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(module, 'ROOT', snapshot)
    assert module.canonical_suite() == snapshot.resolve() / module.CANONICAL_RELATIVE
    assert module.canonical_suite(tmp_path / 'project') == (tmp_path / 'project').resolve() / module.CANONICAL_RELATIVE
    with pytest.raises(FileNotFoundError):
        module.canonical_manifest(module.canonical_suite())
