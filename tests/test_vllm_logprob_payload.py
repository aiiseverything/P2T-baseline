from types import SimpleNamespace

import pytest

from vpo_rm.integration import generation_payload, vllm_sampling_kwargs


class Tokenizer:
    eos_token_id = pad_token_id = 0

    def get_vocab(self):
        return {'<|endoftext|>': 0, 'x': 1, 'y': 2}


def test_optional_probability_request_preserves_sampling_and_requests_chosen_logprob():
    baseline = vllm_sampling_kwargs(Tokenizer(), 4, {'max_tokens': 16})
    actual = vllm_sampling_kwargs(Tokenizer(), 4, {'max_tokens': 16, 'return_logprobs': True})
    assert actual.pop('logprobs') == 1
    assert actual == baseline


def test_probability_payload_keeps_exact_sampled_ids_and_engine_prompt_ids():
    output = SimpleNamespace(token_ids=[2, 0], finish_reason='stop', stop_reason=0,
        logprobs=[{1: SimpleNamespace(logprob=-.1), 2: SimpleNamespace(logprob=-2.)},
                  {0: SimpleNamespace(logprob=-.4)}])
    generated = [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])]
    payload = generation_payload(generated, (0,), include_logprobs=True)
    assert payload['selected_logprobs'] == [[-2., -.4]]
    assert payload['prompt_token_ids'] == [[1, 2]]


@pytest.mark.parametrize('logprobs', [None, [], [{1: SimpleNamespace(logprob=-1.)}],
                                    [{0: SimpleNamespace(logprob=float('nan'))}]])
def test_probability_payload_rejects_missing_or_nonfinite_selected_values(logprobs):
    output = SimpleNamespace(token_ids=[0], finish_reason='stop', stop_reason=0, logprobs=logprobs)
    generated = [SimpleNamespace(prompt_token_ids=[1], outputs=[output])]
    with pytest.raises(ValueError, match='logprob'):
        generation_payload(generated, (0,), include_logprobs=True)
