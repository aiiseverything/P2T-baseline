"""IFEval binds saved Llama tokenizers and exact prompt tokens to each result."""
import json

import pytest

from test_ifeval_policy import checkpoint, cli_stack  # noqa: F401


def saved_tokenizer(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tokenizer_config.json').write_text('{"chat_template": "native chat"}')
    return root


def config_at(stack, tag='model'):
    return json.loads((stack.output / tag / 'manifest_t1.0_n1.json').read_text())['config']


def test_explicit_tokenizer_binds_engine_manifest_and_exact_prompt_tokens(tmp_path, cli_stack):
    from scripts.eval_artifacts import digest, fingerprint

    source = saved_tokenizer(tmp_path / 'saved-sft')
    cli_stack.run('--tokenizer', str(source))
    assert cli_stack.tokenizer_loads == [{
        'source': str(source), 'padding_side': 'left', 'trust_remote_code': True}]
    assert cli_stack.constructors[0]['tokenizer'] == str(source)
    ids = [[0, 1, 1], [0, 1, 2]]
    assert cli_stack.generations[0]['prompts'] == [{'prompt_token_ids': row} for row in ids]
    assert all(row['add_special_tokens'] is False for row in cli_stack.tokenizations)
    assert all(row.count(0) == 1 for row in ids)
    config = config_at(cli_stack)
    assert config['tokenizer'] == {
        'source': str(source), 'fingerprint': fingerprint(source, full_weights=False)}
    assert config['prompt_token_ids_sha256'] == digest(ids)


def test_unspecified_tokenizer_prefers_saved_adapter_artifacts(tmp_path, cli_stack):
    adapter = saved_tokenizer(checkpoint(tmp_path, 'float32'))
    cli_stack.run('--adapters', f'sft={adapter}')
    assert cli_stack.tokenizer_loads[0]['source'] == str(adapter)
    assert cli_stack.constructors[0]['tokenizer'] == str(adapter)
    assert config_at(cli_stack, 'sft')['tokenizer']['source'] == str(adapter)


def test_unspecified_tokenizer_keeps_base_fallback(cli_stack):
    cli_stack.run()
    assert cli_stack.constructors[0]['tokenizer'] == str(cli_stack.model)
    assert config_at(cli_stack)['tokenizer']['source'] == str(cli_stack.model)


def test_multiple_implicit_tokenizer_sources_require_explicit_shared_source(tmp_path, cli_stack):
    adapter = saved_tokenizer(checkpoint(tmp_path, 'float32'))
    with pytest.raises(ValueError, match='--tokenizer'):
        cli_stack.run('--adapters', 'base=none', f'sft={adapter}',
                      '--policy-head-dtype', 'float32')
    assert cli_stack.constructors == []
    assert cli_stack.generations == []


@pytest.mark.parametrize('change', ['artifact', 'source'])
def test_tokenizer_changes_reject_existing_cache_before_engine(tmp_path, cli_stack, change):
    source = saved_tokenizer(tmp_path / 'saved-sft')
    cli_stack.run('--tokenizer', str(source))
    before = {path: path.read_bytes() for path in cli_stack.output.rglob('*.json*')}
    if change == 'artifact':
        (source / 'tokenizer_config.json').write_text('{"chat_template": "changed"}')
    else:
        source = saved_tokenizer(tmp_path / 'other-tokenizer')
    with pytest.raises(ValueError, match='different'):
        cli_stack.run('--tokenizer', str(source))
    assert len(cli_stack.constructors) == len(cli_stack.generations) == 1
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('echo', ['extra_bos', 'missing'])
def test_bad_engine_prompt_echo_fails_before_scoring_or_cache(tmp_path, cli_stack, monkeypatch, echo):
    from scripts import eval_ifeval

    source = saved_tokenizer(tmp_path / 'saved-sft')
    cli_stack.controls.echo = echo
    monkeypatch.setattr(eval_ifeval, 'score_one_sample',
                        lambda *args, **kwargs: pytest.fail('invalid prompt reached checker'))
    with pytest.raises(ValueError, match='prompt token'):
        cli_stack.run('--tokenizer', str(source))
    assert not list(cli_stack.output.rglob('results_*.json'))
    assert not list(cli_stack.output.rglob('manifest_*.json'))


def test_saved_sft_protocol_mismatch_fails_before_engine(tmp_path, cli_stack):
    source = saved_tokenizer(tmp_path / 'saved-sft')
    (source / 'sft_manifest.json').write_text(json.dumps({'token_protocol': {'bos_token_id': 99}}))
    with pytest.raises(ValueError, match='SFT token protocol'):
        cli_stack.run('--tokenizer', str(source))
    assert cli_stack.constructors == []


def test_six_models_five_seeds_preserve_generation_and_official_scoring(tmp_path, cli_stack, monkeypatch):
    from test_ifeval_repeats import record_scoring_seeds

    source = saved_tokenizer(tmp_path / 'saved-sft')
    tags = ['base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8']
    adapters = ['base=none'] + [f'{tag}={checkpoint(tmp_path / tag, "float32")}' for tag in tags[1:]]
    scoring_seeds = record_scoring_seeds(monkeypatch)
    cli_stack.run('--tokenizer', str(source), '--adapters', *adapters,
                  '--seeds', '42', '43', '44', '45', '46', '--scoring-seed', '42',
                  '--max-tokens', '2048', '--policy-head-dtype', 'float32')
    assert len(cli_stack.constructors) == 1
    assert cli_stack.constructors[0]['tokenizer'] == str(source)
    assert cli_stack.constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    assert len(cli_stack.generations) == 30
    assert scoring_seeds == [42] * 30
    for seed in range(42, 47):
        for tag in tags:
            directory = cli_stack.output / f'seed-{seed}' / tag
            config = json.loads((directory / 'manifest_t1.0_n1.json').read_text())['config']
            result = json.loads((directory / 'results_t1.0_n1.json').read_text())
            assert config['seed'] == result['seed'] == seed
            assert config['engine']['seed'] == result['scoring_seed'] == 42
            assert config['max_tokens'] == result['max_tokens'] == 2048
            assert (config['adapter'] is None) == (tag == 'base')
            assert config['recipe'] == {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}
            assert all(result[key] == 1.0 for key in (
                'prompt_strict', 'prompt_loose', 'inst_strict', 'inst_loose'))
