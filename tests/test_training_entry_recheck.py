"""Fail costly training before model allocation when inputs cannot be valid."""
from pathlib import Path
from unittest.mock import patch

import pytest

from vpo_rm.trainer import TrainerConfig, VPOTrainer, check_fresh_output


@pytest.mark.parametrize('field,value', [
    ('learning_rate', float('nan')), ('learning_rate', 0.),
    ('weight_decay', -1.), ('beta', float('inf')), ('beta', -1.),
    ('tau', 0.), ('credit_lambda', .5), ('max_grad_norm', float('nan')),
    ('clip_eps', -0.1), ('rollout_iterations', 0), ('max_prompt_tokens', 0),
    ('generation_microbatch_responses', 0), ('token_chunk_size', 0),
    ('vocab_chunk_size', 0), ('group_size', 2.5), ('validation_size', -1),
])
def test_invalid_training_config_rejected_before_loading_model(field, value, tmp_path):
    config = TrainerConfig(output_dir=str(tmp_path), **{field: value})
    with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('model loading must not start')) as load:
        with pytest.raises(ValueError, match=field):
            VPOTrainer.from_pretrained(config)
        load.assert_not_called()


@pytest.mark.parametrize('marker', ['config.json', 'model.safetensors',
                                   'pytorch_model.bin.index.json', 'tokenizer.json'])
def test_model_or_tokenizer_directory_is_never_training_output(tmp_path, marker):
    (tmp_path / marker).write_text('{}')
    with pytest.raises(FileExistsError):
        check_fresh_output(tmp_path)


def test_prepared_data_split_is_allowed_in_new_training_directory(tmp_path):
    (tmp_path / 'data_split.json').write_text('{}')
    check_fresh_output(tmp_path)
