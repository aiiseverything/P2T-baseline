"""Repeated IFEval runs vary generation randomness, preserving checker identity."""
import json
from collections import Counter

import pytest

from test_ifeval_policy import checkpoint, cli_stack  # noqa: F401


def read_json(path):
    return json.loads(path.read_text())


def record_scoring_seeds(monkeypatch):
    from scripts import eval_ifeval
    original = eval_ifeval.score_one_sample
    observed = []

    def score(*args, scoring_seed=42):
        observed.append(scoring_seed)
        return original(*args, scoring_seed=scoring_seed)

    monkeypatch.setattr(eval_ifeval, 'score_one_sample', score)
    return observed


def test_five_models_five_seeds_have_25_single_sample_results_one_engine(
        tmp_path, cli_stack, monkeypatch):
    tags = ['sft-init', 'grpo', 'lam2', 'lam4', 'lam8']
    adapters = [f'{tag}={checkpoint(tmp_path / tag, "float32")}' for tag in tags]
    scoring_seeds = record_scoring_seeds(monkeypatch)
    cli_stack.run('--adapters', *adapters, '--seeds', '42', '43', '44', '45', '46',
                  '--scoring-seed', '42', '--policy-head-dtype', 'float32')

    assert len(cli_stack.constructors) == 1
    assert cli_stack.constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    assert cli_stack.constructors[0]['seed'] == 42
    assert len(cli_stack.generations) == 25
    assert Counter(g['params'].seed for g in cli_stack.generations) == {
        42: 5, 43: 5, 44: 5, 45: 5, 46: 5}
    assert Counter((g['lora_request'][0], g['params'].seed)
                   for g in cli_stack.generations) == {
        (f'lora-{tag}', seed): 1 for tag in tags for seed in range(42, 47)}
    assert all(g['params'].n == 1 for g in cli_stack.generations)
    assert scoring_seeds == [42] * 25
    assert len(list(cli_stack.output.rglob('results_t1.0_n1.json'))) == 25
    for seed in range(42, 47):
        for tag in tags:
            directory = cli_stack.output / f'seed-{seed}' / tag
            result = read_json(directory / 'results_t1.0_n1.json')
            config = read_json(directory / 'manifest_t1.0_n1.json')['config']
            assert result['seed'] == config['seed'] == seed
            assert config['engine']['seed'] == 42
            assert result['scoring_seed'] == 42
            assert result['tag'] == tag
            assert len(result['per_sample']) == 1
            assert [row['key'] for row in result['details']] == [0, 1]
            assert all(row['sample'] == 0 for row in result['details'])
            assert config['scoring'] == {
                'protocol': 'official_seeded_v1', 'seed': 42, 'langdetect_seed': 42}
            generations = [json.loads(line) for line in
                           (directory / 'generations_t1.0_n1.jsonl').open()]
            assert [row['prompt'] for row in generations] == ['question 0', 'question 1']
            assert all(len(row['responses']) == 1 for row in generations)
            assert not (cli_stack.output / tag).exists()


def test_single_seed_retains_legacy_directory_and_default_checker_seed(cli_stack, monkeypatch):
    observed = record_scoring_seeds(monkeypatch)
    cli_stack.run('--seed', '17', '--adapters', 'base=none')
    directory = cli_stack.output / 'base'
    result = read_json(directory / 'results_t1.0_n1.json')
    config = read_json(directory / 'manifest_t1.0_n1.json')['config']
    assert result['seed'] == config['seed'] == 17
    assert config['scoring'] == {
        'protocol': 'official_seeded_v1', 'seed': 17, 'langdetect_seed': 17}
    assert observed == [17]
    assert not (cli_stack.output / 'seed-17').exists()


def test_explicit_scoring_seed_is_independent_of_single_generation_seed(cli_stack, monkeypatch):
    observed = record_scoring_seeds(monkeypatch)
    cli_stack.run('--seed', '43', '--scoring-seed', '42')
    config = read_json(cli_stack.output / 'model/manifest_t1.0_n1.json')['config']
    assert config['seed'] == 43
    assert config['scoring'] == {
        'protocol': 'official_seeded_v1', 'seed': 42, 'langdetect_seed': 42}
    assert cli_stack.generations[0]['params'].seed == 43
    assert observed == [42]


def test_repeated_seed_caches_do_not_regenerate_and_new_seed_has_own_cache(cli_stack):
    cli_stack.run('--seeds', '42', '43', '--scoring-seed', '42')
    first = {p: p.read_bytes() for p in cli_stack.output.rglob('*.json*')}
    cli_stack.run('--seeds', '42', '43', '--scoring-seed', '42')
    assert len(cli_stack.constructors) == 1
    assert len(cli_stack.generations) == 2
    cli_stack.run('--seeds', '42', '43', '44', '--scoring-seed', '42')
    assert len(cli_stack.constructors) == 2
    assert [g['params'].seed for g in cli_stack.generations] == [42, 43, 44]
    assert all(path.read_bytes() == contents for path, contents in first.items())
    assert (cli_stack.output / 'seed-44/model/results_t1.0_n1.json').is_file()


def test_changing_checker_seed_rejects_repeat_cache_before_engine(cli_stack):
    cli_stack.run('--seeds', '42', '43', '--scoring-seed', '42')
    before = {p: p.read_bytes() for p in cli_stack.output.rglob('*.json*')}
    with pytest.raises(ValueError, match='different'):
        cli_stack.run('--seeds', '42', '43', '--scoring-seed', '43')
    assert len(cli_stack.constructors) == 1
    assert len(cli_stack.generations) == 2
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_repeat_default_checker_and_engine_seed_are_first_generation_seed(cli_stack, monkeypatch):
    observed = record_scoring_seeds(monkeypatch)
    cli_stack.run('--seeds', '17', '19')
    assert len(cli_stack.constructors) == 1
    assert cli_stack.constructors[0]['seed'] == 17
    assert [g['params'].seed for g in cli_stack.generations] == [17, 19]
    assert observed == [17, 17]
    for seed in (17, 19):
        config = read_json(cli_stack.output /
                           f'seed-{seed}/model/manifest_t1.0_n1.json')['config']
        assert config['seed'] == seed and config['engine']['seed'] == 17
        assert config['scoring'] == {
            'protocol': 'official_seeded_v1', 'seed': 17, 'langdetect_seed': 17}


@pytest.mark.parametrize('arguments,message', [
    (['--seeds', '42', '42'], 'distinct'),
    (['--seed', '42', '--seeds', '43'], 'not allowed'),
])
def test_invalid_seed_specification_fails_before_engine(cli_stack, capsys, arguments, message):
    with pytest.raises(SystemExit) as error:
        cli_stack.run(*arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert cli_stack.constructors == []
    assert cli_stack.generations == []
