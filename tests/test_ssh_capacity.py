"""CPU checks for the separate, synthetic A6000 capacity acceptance script."""
import types
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from scripts import check_ssh_capacity as capacity
from vpo_rm.alignment import check_response_tokens, shared_output_mask
from vpo_rm.token_policy import get_stop_token_ids
from vpo_rm.trainer import VPOTrainer


def tokenizers():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    words = ["<eos>", "<unk>", "<user>", "<assistant>", "<rm>",
             "Capacity", "check", "prompt", "Repeat", "these", "words:",
             "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "omega"]
    result = []
    for rm in [False, True]:
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                     pad_token="<eos>", unk_token="<unk>",
                                     additional_special_tokens=["<user>", "<assistant>", "<rm>"],
                                     padding_side="right" if rm else "left")
        prefix = "<user> <rm> " if rm else "<user> "
        tok.chat_template = "{{ '" + prefix + "' + messages[0]['content'] + ' <assistant>' }}"
        if rm:
            tok.chat_template += ("{% if messages|length > 1 %}"
                                  "{{ ' ' + messages[1]['content'] + ' <eos>' }}{% endif %}")
        result.append(tok)
    return result


def fixture_trainer(*, group_size=8, prompt_limit=32, response_limit=16):
    actor_tokenizer, reward_tokenizer = tokenizers()
    trainer = object.__new__(VPOTrainer)
    trainer.actor_tokenizer, trainer.reward_tokenizer = actor_tokenizer, reward_tokenizer
    trainer.actor = torch.nn.Linear(1, 1)
    trainer.actor_device = torch.device("cpu")
    trainer.cfg = types.SimpleNamespace(group_size=group_size, max_prompt_tokens=prompt_limit,
                                        max_response_tokens=response_limit)
    trainer.output_mask = shared_output_mask(actor_tokenizer, 32)
    trainer.stop_token_ids = get_stop_token_ids(actor_tokenizer)
    return trainer


def test_capacity_cli_and_config_preserve_exact_models_and_main_batch(tmp_path):
    args = capacity.parse_args(["--output-dir", str(tmp_path / "capacity")])
    cfg = capacity.build_config(args.output_dir)
    assert cfg.model_name.endswith("/models/Qwen3-14B-Base")
    assert cfg.reward_model_name.endswith("/models/Skywork-Reward-V2-Qwen3-8B")
    assert cfg.init_adapter.endswith("/models/sft-native-eos-clean2k5e2")
    assert cfg.method == "vpo_rm" and cfg.credit_lambda == 4
    assert cfg.freeze_stop_tokens and cfg.freeze_structural
    assert (cfg.prompts_per_rollout, cfg.group_size, cfg.max_prompt_tokens,
            cfg.max_response_tokens) == (8, 8, 2048, 2048)
    assert (cfg.microbatch_responses, cfg.credit_microbatch_responses,
            cfg.optimizer_minibatch_responses) == (1, 1, 64)
    assert cfg.allocated_gpu_count == 2 and cfg.rollout_iterations == 2
    assert cfg.length_reward_mode == "soft" and cfg.length_reward_sigma0 == 1.
    assert cfg.min_response_tokens == 0 and cfg.checkpoint_interval > 2


def test_synthetic_prompts_reach_joint_limit_through_both_real_templates():
    actor, reward = tokenizers()
    prompts = capacity.make_synthetic_prompts(actor, reward, count=8, max_prompt_tokens=32)
    assert len(prompts) == len(set(prompts)) == 8
    for prompt in prompts:
        actor_length, reward_length = capacity.prompt_lengths(actor, reward, prompt)
        assert actor_length == 30 and reward_length == 32
        assert max(capacity.prompt_lengths(actor, reward, prompt + " alpha")) > 32


def test_too_small_prompt_limit_is_rejected_instead_of_truncating():
    actor, reward = tokenizers()
    with pytest.raises(ValueError, match="prompt.*limit|limit.*prompt"):
        capacity.make_synthetic_prompts(actor, reward, count=8, max_prompt_tokens=2)


def test_capacity_counts_closed_reward_template_and_actual_response_tokens():
    trainer = fixture_trainer(group_size=2, prompt_limit=32, response_limit=8)
    trainer.reward_tokenizer.chat_template = (
        "{{ '<user> <rm> ' + messages[0]['content'] + ' <assistant>' }}"
        "{% if messages|length > 1 %}{{ ' ' + messages[1]['content'] + ' <eos> <rm>' }}{% endif %}")
    assert capacity.prompt_lengths(trainer.actor_tokenizer, trainer.reward_tokenizer, 'alpha') == (3, 6)
    _, shape = capacity.build_synthetic_rollout(trainer, ['alpha'])
    # Four prefix tokens, seven ordinary response tokens, two template suffix
    # tokens. Actor's native EOS is removed before canonical RM serialization.
    assert shape['reward_prompt_lengths'] == [6]
    assert shape['reward_sequence_lengths'] == [13, 13]
    assert shape['reward_max_sequence_length'] == 13


def test_full_64_by_2048_response_shape_eos_mask_and_alignment():
    trainer = fixture_trainer(response_limit=2048)
    prompts = capacity.make_synthetic_prompts(trainer.actor_tokenizer, trainer.reward_tokenizer,
                                              count=8, max_prompt_tokens=32)
    rollout, shape = capacity.build_synthetic_rollout(trainer, prompts)
    ids, mask, positions, responses, valid, rendered, reasons = rollout
    assert responses.shape == (64, 2048)
    assert valid.shape == responses.shape and valid.all()
    assert reasons == ["stop"] * 64
    assert len(rendered) == 64 and ids.shape == (64, 30 + 2048)
    assert (responses[:, -1] == trainer.actor_tokenizer.eos_token_id).all()
    for stop in trainer.stop_token_ids:
        assert not (responses[:, :-1] == stop).any()
    assert trainer.output_mask[responses].all()
    assert torch.unique(responses[:, :-1], dim=0).shape[0] == 8
    for token in responses[:, :-1].unique().tolist():
        assert trainer.actor_tokenizer.decode([token]).strip().isalpha()
    check_response_tokens(ids, mask, positions, responses, valid)
    for row in range(64):
        expected = VPOTrainer._render_chat_prompt(trainer.actor_tokenizer, prompts[row // 8], tokenize=True)
        assert ids[row, :30].tolist() == expected
    assert shape["response_shape"] == [64, 2048]
    assert shape["actor_prompt_lengths"] == [30] * 8
    assert shape["reward_prompt_lengths"] == [32] * 8
    assert shape["actor_padded_prompt_width"] == 30
    assert shape["reward_sequence_lengths"] == [32 + 2047] * 64
    assert shape["reward_max_sequence_length"] == 32 + 2047
    assert shape["native_eos_id"] == trainer.actor_tokenizer.eos_token_id


def test_synthetic_rollout_rejects_overlong_prompt_before_encode_truncation():
    trainer = fixture_trainer()
    with pytest.raises(ValueError, match="prompt.*limit|limit.*prompt"):
        capacity.build_synthetic_rollout(trainer, ["alpha " * 64] * 8)


def test_output_directory_must_be_new_and_never_overwrites(tmp_path):
    fresh = tmp_path / "new"
    assert capacity.create_output_dir(fresh) == fresh
    assert fresh.is_dir()
    marker = fresh / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        capacity.create_output_dir(fresh)
    assert marker.read_text() == "unchanged"


def test_report_marks_sigma_as_test_only_and_excludes_quality_claims():
    report = capacity.initial_report(capacity.build_config("unused"))
    assert report["synthetic_capacity_test_only"] is True
    assert report["sigma0_testonly"] == 1.
    assert report["valid_for_reward_calibration"] is False
    assert report["valid_for_quality_comparison"] is False
    assert report["allocated_gpu_count"] == 2
    assert report["vllm_tested"] is False
    assert report["steps"] == 2 and report["completed_steps"] == 0
    assert report["step_metrics"] == []


def test_synthetic_rollout_executes_two_real_tiny_vpo_updates_without_sampling_or_checkpoint(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification
    from vpo_rm.reward import LastTokenReward

    torch.manual_seed(91)
    model_config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                              num_hidden_layers=1, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=4,
                              max_position_embeddings=64, num_labels=1,
                              pad_token_id=0, eos_token_id=0, attention_dropout=0.)
    actor = Qwen3ForCausalLM(model_config)
    rm = Qwen3ForSequenceClassification(model_config)
    actor_tokenizer, reward_tokenizer = tokenizers()
    config = replace(capacity.build_config(tmp_path / "tiny"), actor_device="cpu",
                     reward_device="cpu", allocated_gpu_count=0, lora=False,
                     init_adapter="", gradient_checkpointing=False,
                     max_prompt_tokens=32, max_response_tokens=16,
                     short_response_threshold=2, long_response_threshold=8,
                     token_chunk_size=8, vocab_chunk_size=16)
    trainer = VPOTrainer(actor, actor_tokenizer, LastTokenReward(rm.model, rm.score),
                         reward_tokenizer, config)
    prompts = capacity.make_synthetic_prompts(actor_tokenizer, reward_tokenizer,
                                              count=8, max_prompt_tokens=32)
    rollout, _ = capacity.build_synthetic_rollout(trainer, prompts)
    before = actor.get_output_embeddings().weight.detach().clone()
    report = capacity.initial_report(config)
    with patch.object(actor, "generate", side_effect=AssertionError("must use synthetic responses")), \
            patch.object(trainer, "save_checkpoint", side_effect=AssertionError("no capacity checkpoints")), \
            patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as optimizer_step:
        capacity.run_capacity_steps(trainer, prompts, rollout, report)
    assert report["steps"] == report["completed_steps"] == optimizer_step.call_count == 2
    assert len(report["step_metrics"]) == 2
    for index, metrics in enumerate(report["step_metrics"], start=1):
        assert metrics["rollout"] == index
        assert metrics["optimizer_steps"] == 1 and metrics["reward_count"] == 64
        assert metrics["response_tokens"] == 64 * 16
        assert metrics["resampled_groups"] == metrics["skipped_groups"] == 0
        assert metrics["degenerate_responses"] == metrics["truncated_responses"] == 0
    assert all(state["step"].item() == 2 for state in trainer.optimizer.state.values())
    assert not torch.equal(before, actor.get_output_embeddings().weight)
    assert all(parameter.grad is None for parameter in rm.parameters())
    assert not list((tmp_path / "tiny").glob("checkpoint-*"))


@pytest.mark.parametrize("device_count", [0, 1, 3, 8])
def test_requires_exactly_two_visible_gpus_before_loading(device_count):
    with pytest.raises(RuntimeError, match="two|2"):
        capacity.check_gpu_count(device_count)


def test_llama_capacity_counts_native_bos_once_and_builds_full_responses():
    from pathlib import Path
    from transformers import AutoTokenizer
    from vpo_rm.token_policy import configure_model_padding
    root = Path('/data/VPO-RM/models')
    actor_path, reward_path = root / 'Llama-3.1-8B-Instruct', root / 'Skywork-Reward-Llama-3.1-8B-v0.2'
    if not all((path / 'tokenizer.json').exists() for path in (actor_path, reward_path)):
        pytest.skip('Local Llama tokenizers unavailable')
    actor = AutoTokenizer.from_pretrained(actor_path, local_files_only=True, padding_side='left')
    reward = AutoTokenizer.from_pretrained(reward_path, local_files_only=True)
    configure_model_padding(actor); configure_model_padding(reward)
    assert capacity.prompt_lengths(actor, reward, 'Say hello.') == (38, 39)
    trainer = object.__new__(VPOTrainer)
    trainer.actor_tokenizer, trainer.reward_tokenizer = actor, reward
    trainer.actor_device = torch.device('cpu')
    trainer.actor = torch.nn.Linear(1, 1)
    trainer.cfg = types.SimpleNamespace(group_size=2, max_prompt_tokens=64, max_response_tokens=8)
    trainer.output_mask = shared_output_mask(actor, 128256)
    trainer.stop_token_ids = get_stop_token_ids(actor)
    prompts = capacity.make_synthetic_prompts(actor, reward, count=1, max_prompt_tokens=64)
    rollout, shape = capacity.build_synthetic_rollout(trainer, prompts)
    assert shape['actor_prompt_lengths'] == [63] and shape['reward_prompt_lengths'] == [64]
    assert shape['response_shape'] == [2, 8]
    assert (rollout[0] == 128000).sum(-1).tolist() == [1, 1]
