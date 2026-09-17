import importlib.util
import json
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent


def validator():
    spec = importlib.util.spec_from_file_location('reused_validator', SUITE / 'run_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reused_answers_are_original_verified_bytes():
    if not (SUITE / 'experiment.json').is_file():
        pytest.skip('Requires local archived Arena experiment and original 3000 answers; not published with source')
    experiment = json.loads((SUITE / 'experiment.json').read_text())
    artifact_names = [name for name in experiment['files_sha256']
                      if not name.startswith('source/') and Path(name).suffix not in ('.py', '.sh')]
    original = Path(experiment['generation_reuse']['source_suite'])
    missing = [name for name in artifact_names if not (SUITE / name).is_file()]
    missing.extend(str(original / name) for name in experiment['generation_reuse']['original_files_sha256']
                   if not (original / name).is_file())
    if missing:
        pytest.skip('Requires local archived generation artifacts: ' + ', '.join(missing[:3]))
    result = validator().verify_reused_generation(SUITE)
    assert result['answers'] == 3000
    assert result['new_rollouts'] == 0
    assert result['baseline_model'] == 'gpt-4o-mini-2024-07-18'


def test_modified_copy_is_rejected(tmp_path):
    module = validator()
    source = tmp_path / 'source.jsonl'
    target = tmp_path / 'copy.jsonl'
    source.write_text('original\n')
    target.write_bytes(source.read_bytes())
    module.verify_copy(source, target, module.file_hash(source))
    target.write_text('replaced\n')
    with pytest.raises(ValueError, match='copy'):
        module.verify_copy(source, target, module.file_hash(source))


def test_replaced_source_cannot_legitimize_replaced_copy(tmp_path):
    module = validator()
    source = tmp_path / 'source.jsonl'
    target = tmp_path / 'copy.jsonl'
    source.write_text('original\n')
    expected = module.file_hash(source)
    source.write_text('replacement\n')
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='source'):
        module.verify_copy(source, target, expected)


def test_validator_has_no_generation_entrypoint():
    import subprocess
    result = subprocess.run(['/root/miniconda3/envs/sml/bin/python', str(SUITE / 'run_evaluation.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert '--validate-only' in result.stderr
