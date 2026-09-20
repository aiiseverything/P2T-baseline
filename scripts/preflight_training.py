#!/usr/bin/env python3
"""Run four two-rollout GPU gates using the frozen profile implementation.

This script never submits/deletes jobs. Expose exactly three GPUs and invoke
the copy inside the source snapshot. Models/datasets are read from project-root.
"""
from __future__ import annotations

import argparse
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

ARMS = ('grpo', 'lam2', 'lam4', 'lam8')
LOGPROB_LIMITS = {'mean_abs_error': .02, 'p99_abs_error': .10, 'max_abs_error': .30}


def experiment_launcher(family, profile=None):
    """Select the experiment's immutable config/identity; Llama profiles are explicit."""
    modules = {'qwen': 'scripts.corrected_rl_launcher', 'llama': 'scripts.llama_rl_launcher'}
    if family not in modules:
        raise ValueError(f'Unknown experiment family: {family}')
    launcher = importlib.import_module(modules[family])
    if profile is not None:
        if not hasattr(launcher, 'set_profile'):
            raise ValueError(f'Experiment family {family} has no actor profiles')
        launcher.set_profile(profile)
    return launcher


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def profile_arguments(common, arm, output, sigma0=None):
    if arm not in ARMS:
        raise ValueError('Unknown preflight arm')
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
    from peft import get_peft_model_state_dict
    named = list(actor.named_parameters())
    default = [p for name, p in named if '.default.' in name]
    reference = [p for name, p in named if '.ref.' in name]
    if not default or not all(p.requires_grad for p in default):
        raise ValueError('Every default adapter parameter must be trainable')
    if not reference or any(p.requires_grad for p in reference):
        raise ValueError('Every ref adapter parameter must be frozen')
    if set(actor.active_adapters) != {'default'}:
        raise ValueError('The default adapter must be active')
    return (get_peft_model_state_dict(actor, adapter_name='default', save_embedding_layers=False),
            get_peft_model_state_dict(actor, adapter_name='ref', save_embedding_layers=False))


def verify_initial_adapters(actor, initial):
    import torch
    from safetensors.torch import load_file
    default, reference = adapter_states(actor)
    disk = load_file(str(Path(initial) / 'adapter_model.safetensors'))
    if set(default) != set(reference) or set(default) != set(disk):
        raise ValueError('SFT, default and reference adapter tensor keys differ')
    for key in default:
        value = default[key].detach().cpu()
        if not torch.equal(value, reference[key].detach().cpu()):
            raise ValueError(f'Initial default/reference mismatch: {key}')
        if not torch.equal(value, disk[key].to(value.dtype)):
            raise ValueError(f'Loaded SFT weights differ from disk: {key}')
    return {'status': 'passed', 'tensor_count': len(default),
            'default_sha256': tensor_state_digest(default),
            'reference_sha256': tensor_state_digest(reference)}


def verify_adapter_update(actor, initial, *, previous_default_sha256=None):
    default, reference = adapter_states(actor)
    current = tensor_state_digest(default)
    frozen = tensor_state_digest(reference)
    if frozen != initial['reference_sha256']:
        raise ValueError('Frozen reference adapter changed during update')
    previous = previous_default_sha256 or initial['default_sha256']
    if current == previous:
        raise ValueError('Default adapter has not changed after optimizer update')
    return {'status': 'passed', 'default_sha256': current, 'reference_sha256': frozen}


def logprob_error_report(hf, engine):
    import torch
    left, right = torch.as_tensor(hf).double().flatten(), torch.as_tensor(engine).double().flatten()
    if left.shape != right.shape or not left.numel():
        raise ValueError('HF/vLLM logprob shapes must match and be nonempty')
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('HF/vLLM logprobs must be finite')
    error = (left - right).abs()
    report = {'status': 'passed', 'tokens': left.numel(), 'limits': dict(LOGPROB_LIMITS),
              'mean_abs_error': error.mean().item(), 'p99_abs_error': error.quantile(.99).item(),
              'max_abs_error': error.max().item()}
    if any(report[key] > limit for key, limit in LOGPROB_LIMITS.items()):
        raise ValueError('HF/vLLM logprob tolerance exceeded: ' + json.dumps(report))
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
        self.report = {'status': 'started', 'arm': args.worker_arm, 'rollouts': 0,
                       'updates': [], 'probability_checks': [], 'prompt_sha256': [],
                       'quality': {}, 'logprob_limits': LOGPROB_LIMITS}

    def save(self):
        write_json(Path(self.args.output_dir) / 'preflight-arm.json', self.report)

    def quality(self, connection, request, stage):
        from scripts.preflight_quality import CONTROLS, evaluate_quality
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
        checked = evaluate_quality(texts, baseline_score=baseline if stage == 'after' else None,
                                   rule=getattr(self.args, 'quality_rule', 'strict'))
        self.report['quality'][stage] = checked
        self.save()
        if not checked['passed']:
            raise ValueError(f'Independent quality gate failed at {stage}')

    def check_probabilities(self, request, result):
        import torch
        from vpo_rm.integration import actor_response_logits, sampling_logits, selected_logp_from_logits
        from vpo_rm.token_policy import tokenize_rendered_prompts
        adapter_id = request['adapter_id']
        if adapter_id in self.checked_adapter_ids:
            return
        if result.get('logprobs_mode') != 'processed_logprobs':
            raise ValueError('Preflight needs vLLM processed logprobs')
        trainer, rows = self.trainer, result['rows']
        prompts = tokenize_rendered_prompts(trainer.actor_tokenizer, request['prompts'],
                                            request.get('prompt_token_ids'))
        expected_prefixes = [prompt['prompt_token_ids'] for prompt in prompts]
        if expected_prefixes != result.get('prompt_token_ids'):
            raise ValueError('HF and vLLM prompt tokenization differs')
        n = int(request.get('group_size', 8))
        indices = sorted({0, len(rows) // 2})
        pairs, left, right = [], [], []
        with torch.no_grad():
            for index in indices:
                prompt_index = index // n
                prefix = expected_prefixes[prompt_index]
                response = rows[index][:128]
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
                actual = logp[0].cpu().tolist()
                expected = result['selected_logprobs'][index][:len(response)]
                pairs.append({'row': index, 'token_ids': response, 'hf': actual, 'vllm': expected})
                left.extend(actual); right.extend(expected)
        entry = {'adapter_id': adapter_id, 'adapter_path': request['adapter'], 'pairs': pairs}
        self.report['probability_checks'].append(entry)
        self.save()
        entry.update(logprob_error_report(left, right))
        self.checked_adapter_ids.add(adapter_id)
        self.save()

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
                        # A stochastic final probe exposes the actual softmax
                        # probabilities of the newly saved second update.
                        request.update(probe=False, group_size=1, temperature=1., min_tokens=0)
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
                  source='trained_lam8_actor_after_two_real_rollouts',
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
    launcher = experiment_launcher(args.experiment_family, getattr(args, 'experiment_profile', None))
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
    checks = WorkerChecks(args)
    checks.report.update(identity, runtime=runtime, source_root=str(ROOT),
                         experiment_family=args.experiment_family)
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
    sys.argv = ['profile_vllm_full.py', *profile_arguments(common_config(args.project_root),
                args.worker_arm, output, args.sigma0)]
    try:
        profile.main()
        if checks.report['rollouts'] != 2 or checks.checked_adapter_ids != {1, 2, 3}:
            raise ValueError('Preflight did not verify two updates and all three adapter versions')
        checks.report['startup_validation'] = launcher.check_initial_rollouts(
            output, {'common_config': common_config(args.project_root)})
        if checks.report['startup_validation'] is None:
            raise ValueError('Preflight lacks the formal first-two-rollout validation')
        if args.worker_arm == getattr(args, 'capacity_arm', 'lam8'):
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
                        help='Read-only original model/dataset root; code always loads from this script snapshot')
    parser.add_argument('--runtime-image', required=True, help='Actual container image supplied by the job launcher')
    parser.add_argument('--experiment-family', choices=('qwen', 'llama'), default='qwen')
    parser.add_argument('--experiment-profile', default=None,
                        help='Llama actor profile recorded in the suite manifest (instruct or base)')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS),
                        help='Gate arms in order; grpo must come first because it measures sigma0')
    parser.add_argument('--capacity-arm', choices=ARMS, default='lam8',
                        help='VPO arm that runs the long-sequence capacity check after its rollouts')
    parser.add_argument('--quality-rule', choices=('strict', 'final_word'), default='strict',
                        help='Canary acceptance rule frozen by the experiment profile before GPU execution')
    parser.add_argument('--worker-arm', choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument('--sigma0', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.project_root = args.project_root.resolve()
    if args.worker_arm:
        return run_worker(args)
    if len(set(args.arms)) != len(args.arms) or args.arms[0] != 'grpo':
        raise ValueError('Gate arms must be distinct and start with grpo, which measures sigma0')
    if args.capacity_arm not in args.arms or args.capacity_arm == 'grpo':
        raise ValueError('The capacity arm must be one of the gated VPO arms')
    launcher = experiment_launcher(args.experiment_family, args.experiment_profile)
    common_config, validation_identity = launcher.common_config, launcher.validation_identity
    if args.dry_run:
        print(json.dumps({'source_root': str(ROOT), 'experiment_profile': args.experiment_profile,
            'arms': {arm: profile_arguments(
            common_config(args.project_root), arm, args.output_dir / arm) for arm in args.arms},
            'calibration': 'GRPO canonical initial policy, 128 prompts; share measured sigma0',
            'capacity': f'{args.capacity_arm} after real updates: two 64x2048 steps with formal credit microbatch; synthetic sampling correction explicitly disabled'}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = validation_identity(args.project_root)
    result = {**identity, 'status': 'started', 'source_root': str(ROOT), 'arms': {},
              'runtime_image': args.runtime_image, 'experiment_family': args.experiment_family,
              'gate_arms': list(args.arms), 'capacity_arm': args.capacity_arm,
              'quality_rule': args.quality_rule,
              'preflight_source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('scripts/preflight_training.py', 'scripts/preflight_quality.py',
                              'scripts/check_ssh_capacity.py', 'scripts/corrected_rl_launcher.py')}}
    destination = args.output_dir / 'gpu-validation.json'
    sigma0 = None
    try:
        for arm in args.arms:
            command = [sys.executable, str(Path(__file__).resolve()), '--output-dir',
                       str(args.output_dir / arm), '--project-root', str(args.project_root),
                       '--runtime-image', args.runtime_image, '--worker-arm', arm,
                       '--experiment-family', args.experiment_family,
                       '--capacity-arm', args.capacity_arm, '--quality-rule', args.quality_rule]
            if args.experiment_profile is not None:
                command += ['--experiment-profile', args.experiment_profile]
            if sigma0 is not None:
                command += ['--sigma0', str(sigma0)]
            with (args.output_dir / f'{arm}.log').open('w') as log:
                # The SFT checkpoint records a repository-relative base-model
                # path. Source imports remain anchored to this absolute script.
                subprocess.run(command, cwd=args.project_root, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((args.output_dir / arm / 'preflight-arm.json').read_text())
            result['arms'][arm] = report
            if report['status'] != 'passed':
                raise ValueError(f'Arm {arm} failed its preflight')
            if arm == 'grpo':
                calibration_path = args.output_dir / arm / 'length_reward_calibration.json'
                calibration = json.loads(calibration_path.read_text())
                if (calibration['source'] != 'initial_policy' or calibration['calibration_prompt_count'] != 128
                        or calibration['reward_format'] != identity['reward_input_protocol']):
                    raise ValueError('Preflight requires fresh canonical 128-prompt calibration')
                sigma0 = calibration['sigma0']
                if not math.isfinite(sigma0) or sigma0 <= 0:
                    raise ValueError('Canonical calibration sigma0 must be finite and positive')
                result['calibration'] = {'path': str(calibration_path),
                    'sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(), 'sigma0': sigma0}
            elif report['prompt_sha256'] != result['arms']['grpo']['prompt_sha256']:
                raise ValueError('The gated arms did not use identical training prompts')
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
