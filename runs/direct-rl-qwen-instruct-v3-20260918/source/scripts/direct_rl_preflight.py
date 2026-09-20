#!/usr/bin/env python3
"""Run fresh-LoRA GRPO/lam4 GPU gates using the frozen profile implementation.

This script never submits/deletes jobs. Expose exactly three GPUs and invoke
the copy inside the source snapshot. Project-root is the family suite containing
experiment.json. Formal jobs restart independently with the measured sigma0.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import profile_vllm_full as profile

ARMS = ('grpo', 'lam4')
# Maximum token error is diagnostic; batch distribution/importance gates decide.
LOGPROB_LIMITS = {'mean_abs_error': .02, 'p99_abs_error': .15}
IMPORTANCE_LIMITS = {'min_ess_ratio': .99, 'max_ratio_clip_fraction': .01,
                     'min_loss_weighted_importance_ess_fraction': .99}
PROBABILITY_PROTOCOL = 'full_batch_v3'


def experiment_launcher():
    return importlib.import_module('scripts.direct_rl_launcher')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def profile_arguments(common, arm, output, sigma0=None):
    if arm not in ARMS:
        raise ValueError('Unknown preflight arm')
    if common.get('init_adapter') != '':
        raise ValueError('Direct RL requires an explicitly empty init_adapter; SFT is forbidden')
    config = dict(common, max_rollouts=2, output_dir=str(output),
                  method='grpo' if arm == 'grpo' else 'vpo_rm',
                  credit_lambda=1 if arm == 'grpo' else int(arm[3:]))
    config.pop('length_reward_sigma0', None)
    if sigma0 is not None:
        config['length_reward_sigma0'] = sigma0
    arguments = [value for key, item in config.items()
                 for value in ('--' + key.replace('_', '-'), str(item))]
    if arm != 'grpo':
        arguments += ['--freeze-stop-tokens', '--freeze-structural']
    return arguments


def tensor_state_digest(state):
    import torch
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_states(actor):
    """Require exactly one trainable LoRA adapter and a fully frozen backbone."""
    from peft import get_peft_model_state_dict
    if set(actor.peft_config) != {'default'} or set(actor.active_adapters) != {'default'}:
        raise ValueError('Exactly one default adapter must exist and be active')
    named = list(actor.named_parameters())
    default = [p for name, p in named if '.lora_A.default.' in name or '.lora_B.default.' in name]
    base = {name: p for name, p in named
            if '.lora_A.default.' not in name and '.lora_B.default.' not in name}
    if not default or not all(p.requires_grad for p in default):
        raise ValueError('Every default adapter parameter must be trainable')
    if not base or any(p.requires_grad or p.grad is not None for p in base.values()):
        raise ValueError('Every base/reference parameter must remain frozen without gradients')
    return (get_peft_model_state_dict(actor, adapter_name='default', save_embedding_layers=False),
            base)


def verify_initial_adapters(actor, initial):
    import torch
    if initial != '':
        raise ValueError('Direct RL initialization must not load an adapter')
    default, base = adapter_states(actor)
    config = actor.peft_config['default']
    expected = {'r': 64, 'lora_alpha': 128, 'lora_dropout': 0,
                'bias': 'none', 'init_lora_weights': True, 'use_dora': False, 'use_rslora': False}
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if (any(getattr(config, key, None) != value for key, value in expected.items())
            or set(config.target_modules) != targets or config.modules_to_save):
        raise ValueError('Fresh LoRA configuration differs from canonical initialization')
    a = {key: value for key, value in default.items() if '.lora_A.' in key}
    b = {key: value for key, value in default.items() if '.lora_B.' in key}
    if (not a or not b or len(a) + len(b) != len(default)
            or {key.replace('.lora_A.', '.') for key in a}
            != {key.replace('.lora_B.', '.') for key in b}):
        raise ValueError('Fresh LoRA must contain paired A/B matrices only')
    for key, value in default.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite initial adapter tensor: {key}')
        nonzero = bool(torch.count_nonzero(value))
        if key in a and not nonzero:
            raise ValueError(f'Initial A must be random and nonzero: {key}')
        if key in b and nonzero:
            raise ValueError(f'Initial B must be exactly zero: {key}')
    return {'status': 'passed', 'tensor_count': len(default),
            'initialization': 'fresh_lora_zero_contribution',
            'reference_mode': 'disabled_adapter_base',
            'a_tensor_count': len(a), 'b_tensor_count': len(b),
            'default_sha256': tensor_state_digest(default),
            'base_sha256': tensor_state_digest(base)}


def verify_adapter_update(actor, initial, *, previous_default_sha256=None):
    import torch
    default, base = adapter_states(actor)
    if any(not torch.isfinite(value).all() for value in default.values()):
        raise ValueError('Updated adapter tensors must be finite')
    current = tensor_state_digest(default)
    frozen = tensor_state_digest(base)
    if frozen != initial['base_sha256']:
        raise ValueError('Frozen base/reference weights changed during update')
    previous = previous_default_sha256 or initial['default_sha256']
    if current == previous:
        raise ValueError('Default adapter has not changed after optimizer update')
    if not any(bool(torch.count_nonzero(value)) for key, value in default.items() if '.lora_B.' in key):
        raise ValueError('LoRA B remains zero after the update; no learned policy contribution')
    return {'status': 'passed', 'default_sha256': current, 'base_sha256': frozen,
            'reference_unchanged': True, 'nonzero_b': True}


def verify_initial_reference(trainer):
    """Compare actual startup logits to disabled-adapter logits on a fixed prompt."""
    import torch
    if (trainer.cfg.init_adapter != '' or trainer.cfg.kl_reference != 'init'
            or trainer.reference_adapter != 'base' or trainer.reference_actor is not None):
        raise ValueError('Fresh policy reference must be the disabled-adapter base')
    batch, _ = trainer._encode_prompts(['What is 2 + 2?'])
    was_training = trainer.actor.training
    trainer.actor.eval()
    try:
        with torch.no_grad():
            current = trainer.actor(**batch, use_cache=False).logits[:, -1].float().cpu()
            with trainer.actor.disable_adapter():
                reference = trainer.actor(**batch, use_cache=False).logits[:, -1].float().cpu()
        if (not torch.isfinite(current).all() or not torch.isfinite(reference).all()
                or not torch.allclose(current, reference, rtol=0, atol=1e-6)):
            raise ValueError('Initial policy logits differ from the frozen base/reference')
        return {'status': 'passed', 'reference_mode': 'disabled_adapter_base',
                'max_abs_error': float((current - reference).abs().max()), 'atol': 1e-6,
                'prompt_token_ids': batch['input_ids'].cpu().tolist(),
                'reference_logits_sha256': tensor_state_digest({'logits': reference})}
    finally:
        trainer.actor.train(was_training)


def validate_arm_pair(grpo, lam4):
    for report in (grpo, lam4):
        if report.get('status') != 'passed' or report.get('rollouts') != 2:
            raise ValueError('Both direct-RL arms must pass two real rollouts')
    if grpo.get('seed') != lam4.get('seed') or type(grpo.get('seed')) is not int:
        raise ValueError('Direct-RL arm initialization seeds differ')
    for key in ('default_sha256', 'base_sha256'):
        if (not grpo['initial_adapters'].get(key)
                or grpo['initial_adapters'][key] != lam4['initial_adapters'].get(key)):
            raise ValueError(f'Direct-RL initial weights differ: {key}')
    if not grpo.get('prompt_sha256') or grpo['prompt_sha256'] != lam4.get('prompt_sha256'):
        raise ValueError('Direct-RL arms did not use identical training prompts')
    return {'status': 'passed', 'seed': grpo['seed'], 'same_initial_weights': True,
            'same_frozen_base': True, 'same_training_prompts': True,
            'formal_rng_policy': 'restart_both_formal_processes_with_explicit_shared_sigma0'}


def quality_diagnostic(texts, baseline_score=None):
    from scripts.preflight_quality import evaluate_quality
    result = evaluate_quality(texts)
    return {**result, 'baseline_score': baseline_score, 'gating': False,
            'used_for_checkpoint_selection': False, 'algorithm_evaluation': False,
            'interpretation': 'Initialization and update diagnostic; base models need not follow instructions'}


class ProbabilityGateError(ValueError):
    def __init__(self, message, report):
        self.report = dict(report, status='failed')
        super().__init__(message + ': ' + json.dumps(self.report, allow_nan=False))


def logprob_error_report(hf, engine, *, gating=True, response_lengths=None):
    """Global token and loss-weighted gates, with optional ungated diagnostics."""
    import torch
    from vpo_rm.core import rollout_importance_weights
    left, right = torch.as_tensor(hf, dtype=torch.float64), torch.as_tensor(engine, dtype=torch.float64)
    if left.shape != right.shape or not left.numel():
        raise ProbabilityGateError('HF/vLLM logprob shapes must match and be nonempty', {})
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ProbabilityGateError('HF/vLLM logprobs must be finite', {})
    delta = (left - right).flatten()
    if not torch.isfinite(delta).all():
        raise ProbabilityGateError('HF/vLLM logprob differences must be finite', {})
    lengths = [delta.numel()] if response_lengths is None else response_lengths
    if (not isinstance(lengths, (list, tuple)) or not lengths
            or any(type(length) is not int or length <= 0 for length in lengths)
            or sum(lengths) != delta.numel()):
        raise ProbabilityGateError('Response lengths must cover all paired tokens exactly', {})
    error = delta.abs()
    # Use the exact training FP32 cast/subtract/exp path, including its input
    # validity checks. Stable fallback statistics still explain a rejected batch.
    fp32_error = None
    try:
        rho = rollout_importance_weights(left.reshape(1, -1), right.reshape(1, -1),
                                        torch.ones_like(left.reshape(1, -1), dtype=torch.bool))
        rho = rho.flatten().double()
        scaled_rho = rho / rho.max()
    except ValueError as error_value:
        fp32_error = str(error_value)
        scaled_rho = (delta - delta.max()).exp()
    ess = scaled_rho.sum().square() / (delta.numel() * scaled_rho.square().sum())
    # The objective averages tokens within each response, then responses:
    # mu_rt = 1 / (R * T_r), ESS_fraction = E_mu[rho]^2 / E_mu[rho^2].
    response_rho = scaled_rho.split(lengths)
    loss_mean_rho = torch.stack([row.mean() for row in response_rho]).mean()
    loss_second_moment = torch.stack([row.square().mean() for row in response_rho]).mean()
    loss_ess = loss_mean_rho.square() / loss_second_moment
    clipped = (delta < math.log(.8)) | (delta > math.log(1.2))
    tail = error > .3
    report = {'status': 'passed' if gating else 'diagnostic', 'gating': gating,
              'protocol': PROBABILITY_PROTOCOL,
              'tokens': left.numel(), 'limits': dict(LOGPROB_LIMITS),
              'importance_limits': dict(IMPORTANCE_LIMITS), 'rho_definition': 'exp(hf - vllm)',
              'importance_weight_dtype': 'float32', 'fp32_importance_weights_valid': fp32_error is None,
              'loss_weighting': 'equal_response_then_equal_valid_token',
              'loss_weighted_responses': len(lengths),
              'loss_weighted_importance_ess_fraction': loss_ess.item(),
              'mean_abs_error': error.mean().item(), 'p99_abs_error': error.quantile(.99).item(),
              'max_abs_error': error.max().item(), 'ess_ratio': ess.item(),
              'ratio_clip_count': int(clipped.sum()), 'ratio_clip_fraction': clipped.double().mean().item(),
              'min_log_rho': delta.min().item(), 'max_log_rho': delta.max().item(),
              'tail_abs_gt_0_3_count': int(tail.sum()), 'tail_abs_gt_0_3_fraction': tail.double().mean().item(),
              'warnings': ['Some token logprob errors exceed 0.3'] if tail.any() else []}
    if any(not math.isfinite(value) for value in report.values() if isinstance(value, float)):
        raise ProbabilityGateError('HF/vLLM derived probability statistics must be finite', {})
    violations = [key for key, limit in LOGPROB_LIMITS.items() if report[key] > limit]
    if report['ess_ratio'] < IMPORTANCE_LIMITS['min_ess_ratio']:
        violations.append('ess_ratio')
    if report['loss_weighted_importance_ess_fraction'] < IMPORTANCE_LIMITS['min_loss_weighted_importance_ess_fraction']:
        violations.append('loss_weighted_importance_ess_fraction')
    if report['ratio_clip_fraction'] > IMPORTANCE_LIMITS['max_ratio_clip_fraction']:
        violations.append('ratio_clip_fraction')
    if fp32_error is not None:
        raise ProbabilityGateError(fp32_error,
                                  dict(report, violations=violations + ['fp32_importance_weights']))
    if gating and violations:
        raise ProbabilityGateError('HF/vLLM logprob tolerance exceeded', dict(report, violations=violations))
    return report


def runtime_report(image):
    import torch
    from scripts.corrected_rl_launcher import RUNTIME
    return {'image': image, 'python': platform.python_version(), 'executable': sys.executable,
            'versions': {name: importlib.metadata.version(name) for name in RUNTIME},
            'cuda': torch.version.cuda, 'job_id': os.environ.get('JOB_ID'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpus': [{'name': torch.cuda.get_device_name(i),
                      'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]}


class WorkerChecks:
    """Observe profile RPCs and trainer hooks without changing production files."""
    def __init__(self, args):
        self.args = args
        self.trainer = None
        self.checked_adapter_ids = set()
        self.last_rollout_request = None
        self.report = {'status': 'started', 'arm': args.worker_arm, 'rollouts': 0,
                       'updates': [], 'probability_checks': [], 'prompt_sha256': [],
                       'quality': {}, 'logprob_limits': LOGPROB_LIMITS,
                       'importance_limits': IMPORTANCE_LIMITS,
                       'probability_protocol': PROBABILITY_PROTOCOL}

    def save(self):
        write_json(Path(self.args.output_dir) / 'preflight-arm.json', self.report)

    def quality(self, connection, request, stage):
        from scripts.preflight_quality import CONTROLS
        tokenizer = self.trainer.actor_tokenizer
        batch, rendered = self.trainer._encode_prompts([item['prompt'] for item in CONTROLS])
        payload = {'adapter': request['adapter'], 'adapter_id': request['adapter_id'],
                   'prompts': rendered, 'prompt_token_ids': profile.unpadded_prompt_token_ids(batch),
                   'probe': True, 'max_tokens': 128}
        connection.sendall((json.dumps(payload) + '\n').encode())
        with connection.makefile('r') as reader:
            result = json.loads(reader.readline())
        if not result.get('ok') or len(result['rows']) != len(CONTROLS):
            raise ValueError('Quality generation returned incomplete rows')
        texts = [tokenizer.decode(row, skip_special_tokens=True,
                 clean_up_tokenization_spaces=False) for row in result['rows']]
        baseline = self.report['quality'].get('before', {}).get('score')
        checked = quality_diagnostic(texts, baseline_score=baseline if stage == 'after' else None)
        self.report['quality'][stage] = checked
        self.save()

    def check_probabilities(self, request, result):
        import torch
        from vpo_rm.integration import actor_response_logits, sampling_logits, selected_logp_from_logits
        from vpo_rm.token_policy import tokenize_rendered_prompts
        adapter_id = request.get('adapter_id')
        if adapter_id in self.checked_adapter_ids:
            return
        entry = {'adapter_id': adapter_id, 'adapter_path': request.get('adapter'),
                 'status': 'started', 'protocol': PROBABILITY_PROTOCOL,
                 'coverage': 'all_generated_responses_and_tokens', 'pairs': []}
        self.report['probability_checks'].append(entry)
        trainer = self.trainer
        was_training = trainer.actor.training
        try:
            if (type(adapter_id) is not int or type(result.get('adapter_id')) is not int
                    or result['adapter_id'] != adapter_id or result.get('ok') is not True):
                raise ValueError('Generation result is not bound to the requested adapter')
            if result.get('logprobs_mode') != 'processed_logprobs':
                raise ValueError('Preflight needs vLLM processed logprobs')
            if not isinstance(request.get('prompt_token_ids'), list):
                raise ValueError('Preflight requires explicit prompt token IDs')
            prompts = tokenize_rendered_prompts(trainer.actor_tokenizer, request['prompts'],
                                                request['prompt_token_ids'])
            expected_prefixes = [prompt['prompt_token_ids'] for prompt in prompts]
            actual_prefixes = result.get('prompt_token_ids')
            if (not isinstance(actual_prefixes, list) or any(not isinstance(row, list)
                    or any(type(token) is not int for token in row) for row in actual_prefixes)
                    or expected_prefixes != actual_prefixes):
                raise ValueError('HF and vLLM prompt tokenization differs')
            n, maximum = request.get('group_size', 8), request.get('max_tokens')
            if type(n) is not int or n < 1 or type(maximum) is not int or maximum < 1:
                raise ValueError('Generation group size and maximum length must be positive integers')
            rows, probabilities = result.get('rows'), result.get('selected_logprobs')
            if (not isinstance(rows, list) or len(rows) != len(prompts) * n
                    or not isinstance(probabilities, list) or len(probabilities) != len(rows)):
                raise ValueError('Generation response/logprob count differs from prompt groups')
            # Validate IDs on CPU once instead of synchronizing a GPU scalar for
            # every generated token in a potentially large calibration batch.
            support = trainer.output_mask.detach().cpu().tolist()
            for index, (response, expected) in enumerate(zip(rows, probabilities)):
                if (not isinstance(response, list) or not 0 < len(response) <= maximum
                        or not isinstance(expected, list) or len(expected) != len(response)):
                    raise ValueError(f'Invalid response/logprob length at row {index}')
                if any(type(token) is not int or not 0 <= token < len(support)
                       or not support[token] for token in response):
                    raise ValueError(f'Invalid response token at row {index}')
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in expected):
                    raise ValueError(f'Nonfinite/invalid selected logprob at row {index}')
            if 'tokens' in result and (type(result['tokens']) is not int
                                      or result['tokens'] != sum(map(len, rows))):
                raise ValueError('Generation token count differs from full response coverage')
            entry.update(responses=len(rows), groups=len(prompts), responses_per_group=n,
                         response_microbatch=1, sampling={key: request[key] for key in
                         ('max_tokens', 'group_size', 'min_tokens', 'temperature', 'top_p', 'top_k',
                          'presence_penalty') if key in request})
            left, right, response_statistics = [], [], []
            trainer.actor.eval()
            with torch.no_grad():
                for index, (response, expected) in enumerate(zip(rows, probabilities)):
                    prefix = expected_prefixes[index // n]
                    tokens = torch.tensor([response], device=trainer.actor_device)
                    ids = torch.tensor([prefix + response], device=trainer.actor_device)
                    valid = torch.ones_like(tokens, dtype=torch.bool)
                    positions = torch.arange(len(prefix), ids.shape[1], device=ids.device)[None, :]
                    logits = actor_response_logits(trainer.actor, ids, torch.ones_like(ids), positions,
                                                   valid, output_mask=trainer.output_mask)
                    logits = sampling_logits(logits, min_response_tokens=request.get('min_tokens', 8),
                                             stop_token_ids=trainer.stop_token_ids, inplace=True)
                    logp = selected_logp_from_logits(logits, tokens, valid,
                                                    policy_temperature=request.get('temperature', 1.))
                    if logp.shape != tokens.shape or not torch.isfinite(logp).all():
                        raise ValueError(f'Nonfinite/misaligned HF logprobs at row {index}')
                    actual = logp[0].cpu().tolist()
                    entry['pairs'].append({'row': index, 'group': index // n,
                        'prompt_token_ids': prefix, 'token_ids': response, 'hf': actual, 'vllm': expected})
                    response_statistics.append(dict(logprob_error_report(actual, expected, gating=False),
                                                     row=index, group=index // n))
                    left.extend(actual); right.extend(expected)
                    del logits, logp, ids, tokens, valid, positions
            group_statistics = []
            for group in range(len(prompts)):
                pairs = entry['pairs'][group * n:(group + 1) * n]
                group_statistics.append(dict(logprob_error_report(
                    [value for pair in pairs for value in pair['hf']],
                    [value for pair in pairs for value in pair['vllm']], gating=False,
                    response_lengths=[len(pair['hf']) for pair in pairs]), group=group))
            entry.update(response_statistics=response_statistics, group_statistics=group_statistics,
                balanced_statistics={'gating': False,
                    'response_equal_mean_abs_error': sum(row['mean_abs_error'] for row in response_statistics) / len(rows),
                    'group_equal_mean_abs_error': sum(row['mean_abs_error'] for row in group_statistics) / len(prompts),
                    'worst_group_mean_abs_error': max(row['mean_abs_error'] for row in group_statistics)})
            entry.update(logprob_error_report(left, right, response_lengths=[len(row) for row in rows]))
            self.checked_adapter_ids.add(adapter_id)
            self.save()
        except Exception as error:
            entry.update(getattr(error, 'report', {}))
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
            self.report['status'] = 'failed'
            self.save()
            raise
        finally:
            trainer.actor.train(was_training)

    def socket_factory(self, *args, **kwargs):
        checks = self

        class ObservedSocket:
            def __init__(self):
                self.connection = socket.socket(*args, **kwargs)
                self.request = None
            def __enter__(self): return self
            def __exit__(self, *exc): self.connection.close()
            def __getattr__(self, name): return getattr(self.connection, name)
            def sendall(self, data):
                request = json.loads(data)
                if 'prompts' in request:
                    if 'before' not in checks.report['quality']:
                        checks.quality(self.connection, request, 'before')
                    if request.get('probe'):
                        checks.quality(self.connection, request, 'after')
                        if checks.last_rollout_request is None:
                            raise ValueError('Final adapter probe requires a preceding full 8x8 rollout')
                        # Reuse the real rollout's contexts and full sampling
                        # protocol, but probe the newly saved adapter version.
                        final_adapter = {key: request[key] for key in ('adapter', 'adapter_id')}
                        request = copy.deepcopy(checks.last_rollout_request)
                        request.update(final_adapter, probe=False)
                    elif (len(request['prompts']) == 8 and request.get('group_size') == 8
                          and request.get('max_tokens') == 2048):
                        checks.last_rollout_request = copy.deepcopy(request)
                    request['return_logprobs'] = True
                self.request = request
                self.connection.sendall((json.dumps(request) + '\n').encode())
            def makefile(self, *args, **kwargs):
                stream = self.connection.makefile(*args, **kwargs)
                owner = self
                class Reader:
                    def __enter__(self): return self
                    def __exit__(self, *exc): stream.close()
                    def readline(self):
                        line = stream.readline()
                        if line and 'prompts' in owner.request:
                            result = json.loads(line)
                            checks.check_probabilities(owner.request, result)
                            # The production loss needs each selected token's
                            # actual sampler probability, including retry rows.
                            return json.dumps(result) + '\n'
                        return line
                return Reader()
        return ObservedSocket()


def run_capacity(checks):
    import torch
    from scripts import check_ssh_capacity as capacity
    trainer = checks.trainer
    output = Path(checks.args.output_dir).parent / 'capacity'
    output.mkdir(exist_ok=False)
    trainer.cfg = replace(trainer.cfg, output_dir=str(output), rollout_importance_correction=False)
    trainer.log_path = output / 'metrics.jsonl'
    report = capacity.initial_report(trainer.cfg)
    report.update(sigma0_testonly=None, calibration_sigma0=trainer.cfg.length_reward_sigma0,
                  source='trained_lam4_actor_after_two_real_rollouts',
                  credit_microbatch_responses=trainer.cfg.credit_microbatch_responses,
                  sampling_correction_testonly='disabled_synthetic_tokens_have_no_sampler',
                  allocated_gpu_count=3, active_gpu_count=2)
    checks.capacity_active = True
    try:
        for index in (0, 1):
            torch.cuda.reset_peak_memory_stats(index)
        prompts = capacity.make_synthetic_prompts(trainer.actor_tokenizer, trainer.reward_tokenizer,
                                                  count=8, max_prompt_tokens=2048)
        rollout, shape = capacity.build_synthetic_rollout(trainer, prompts)
        report['shape'] = shape
        capacity.run_capacity_steps(trainer, prompts, rollout, report)
        report.update(status='passed', reference=verify_adapter_update(trainer.actor,
                                                                      checks.report['initial_adapters']))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        checks.capacity_active = False
        report['gpus'] = capacity._memory_report(torch)
        write_json(output / 'capacity-report.json', report)
    return report


def run_worker(args):
    import torch
    launcher = experiment_launcher()
    RUNTIME, common_config, validation_identity = (launcher.RUNTIME, launcher.common_config,
                                                  launcher.validation_identity)
    if torch.cuda.device_count() != 3:
        raise RuntimeError('Expose exactly three GPUs for Actor, RM and vLLM')
    runtime = runtime_report(args.runtime_image)
    if runtime['versions'] != RUNTIME:
        raise RuntimeError('Preflight runtime versions differ from the formal training runtime: '
                           + json.dumps({'actual': runtime['versions'], 'expected': RUNTIME}))
    if any(gpu['memory_bytes'] < 120 * 2**30 for gpu in runtime['gpus']):
        raise RuntimeError('Each of the three preflight GPUs must have at least 120 GiB')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'Use a fresh preflight arm output: {output}')
    identity = validation_identity(args.project_root)
    arguments = profile_arguments(common_config(args.project_root), args.worker_arm, output, args.sigma0)
    checks = WorkerChecks(args)
    checks.report.update(identity, runtime=runtime, source_root=str(ROOT),
                         seed=common_config(args.project_root)['seed'])
    original_trainer = profile.VPOTrainer

    class CheckedTrainer(original_trainer):
        @classmethod
        def from_pretrained(cls, config):
            trainer = super().from_pretrained(config)
            checks.trainer = trainer
            head = trainer.actor.get_output_embeddings()
            checks.report['policy_head'] = {'dtype': str(head.weight.dtype),
                                            'trainable': head.weight.requires_grad}
            if config.policy_head_dtype == 'float32' and (
                    head.weight.dtype != torch.float32 or head.weight.requires_grad):
                raise ValueError('The validated policy head must be frozen FP32')
            checks.report['initial_adapters'] = verify_initial_adapters(trainer.actor, config.init_adapter)
            checks.report['initial_reference'] = verify_initial_reference(trainer)
            checks.save()
            return trainer
        def train_rollout(self, prompts):
            if getattr(checks, 'capacity_active', False):
                result = super().train_rollout(prompts)
                updates = checks.report.setdefault('capacity_updates', [])
                previous = (updates or checks.report['updates'])[-1]['default_sha256']
                updates.append(verify_adapter_update(self.actor, checks.report['initial_adapters'],
                               previous_default_sha256=previous))
                checks.save()
                return result
            checks.report['prompt_sha256'].append([hashlib.sha256(p.encode()).hexdigest() for p in prompts])
            result = super().train_rollout(prompts)
            if result.get('skipped_rollout') or result['optimizer_steps'] != 1 or result['reward_count'] != 64:
                raise ValueError('Preflight must execute one full 64-response optimizer step per rollout')
            if any(not math.isfinite(value) for value in result.values() if isinstance(value, float)):
                raise ValueError('Preflight training metrics are not finite')
            previous = (checks.report['updates'][-1]['default_sha256'] if checks.report['updates']
                        else checks.report['initial_adapters']['default_sha256'])
            checks.report['updates'].append(verify_adapter_update(self.actor, checks.report['initial_adapters'],
                                            previous_default_sha256=previous))
            checks.report['rollouts'] += 1
            checks.save()
            return result

    # This module-local dependency injection leaves the frozen source files
    # untouched and exercises the exact production profile entry point.
    original_socket, original_argv = profile.socket, sys.argv
    profile.VPOTrainer = CheckedTrainer
    profile.socket = SimpleNamespace(AF_UNIX=socket.AF_UNIX, SOCK_STREAM=socket.SOCK_STREAM,
                                      socket=checks.socket_factory)
    sys.argv = ['profile_vllm_full.py', *arguments]
    try:
        profile.main()
        if checks.report['rollouts'] != 2 or checks.checked_adapter_ids != {1, 2, 3}:
            raise ValueError('Preflight did not verify two updates and all three adapter versions')
        checks.report['startup_validation'] = launcher.check_initial_rollouts(
            output, {'common_config': common_config(args.project_root)})
        if checks.report['startup_validation'] is None:
            raise ValueError('Preflight lacks the formal first-two-rollout validation')
        if args.worker_arm == 'lam4':
            checks.report['capacity'] = run_capacity(checks)
        if validation_identity(args.project_root) != identity:
            raise ValueError('Source or input identity changed during preflight')
        checks.report['status'] = 'passed'
    except Exception as error:
        checks.report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        profile.VPOTrainer, profile.socket, sys.argv = original_trainer, original_socket, original_argv
        checks.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT,
                        help='Family suite directory containing the direct-RL experiment.json')
    parser.add_argument('--runtime-image', required=True, help='Actual container image supplied by the job launcher')
    parser.add_argument('--worker-arm', choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument('--sigma0', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.project_root = args.project_root.resolve()
    if args.worker_arm:
        return run_worker(args)
    launcher = experiment_launcher()
    common_config, validation_identity = launcher.common_config, launcher.validation_identity
    if args.dry_run:
        print(json.dumps({'source_root': str(ROOT), 'arms': {arm: profile_arguments(
            common_config(args.project_root), arm, args.output_dir / arm) for arm in ARMS},
            'calibration': 'GRPO canonical initial policy, 128 prompts; share measured sigma0',
            'capacity': 'lam4 after real updates: two 64x2048 steps with formal credit microbatch; synthetic sampling correction explicitly disabled'}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = validation_identity(args.project_root)
    result = {**identity, 'status': 'started', 'source_root': str(ROOT), 'arms': {},
              'runtime_image': args.runtime_image,
              'preflight_source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('scripts/direct_rl_preflight.py', 'scripts/preflight_quality.py',
                              'scripts/check_ssh_capacity.py', 'scripts/direct_rl_launcher.py')}}
    destination = args.output_dir / 'gpu-validation.json'
    sigma0 = None
    try:
        for arm in ARMS:
            command = [sys.executable, str(Path(__file__).resolve()), '--output-dir',
                       str(args.output_dir / arm), '--project-root', str(args.project_root),
                       '--runtime-image', args.runtime_image, '--worker-arm', arm]
            if sigma0 is not None:
                command += ['--sigma0', str(sigma0)]
            with (args.output_dir / f'{arm}.log').open('w') as log:
                subprocess.run(command, cwd=args.project_root, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((args.output_dir / arm / 'preflight-arm.json').read_text())
            result['arms'][arm] = report
            if report['status'] != 'passed':
                raise ValueError(f'Arm {arm} failed its preflight')
            if arm == 'grpo':
                calibration_path = args.output_dir / arm / 'length_reward_calibration.json'
                calibration = json.loads(calibration_path.read_text())
                from scripts.corrected_rl_launcher import validate_calibration
                validate_calibration(calibration, common_config(args.project_root))
                sigma0 = calibration['sigma0']
                if not math.isfinite(sigma0) or sigma0 <= 0:
                    raise ValueError('Canonical calibration sigma0 must be finite and positive')
                result['calibration'] = {'path': str(calibration_path),
                    'sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(), 'sigma0': sigma0}
            else:
                result['arm_consistency'] = validate_arm_pair(result['arms']['grpo'], report)
            write_json(destination, result)
        if validation_identity(args.project_root) != identity:
            raise ValueError('Source or input identity changed during preflight')
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination, result)


if __name__ == '__main__':
    main()
