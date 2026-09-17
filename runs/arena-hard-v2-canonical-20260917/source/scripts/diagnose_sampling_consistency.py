#!/usr/bin/env python3
"""Read-only fixed-trace HF/vLLM numerical diagnosis; never trains or gates quality."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path):
    data = json.loads(Path(path).read_text())
    if not data.get('cases'):
        raise ValueError('Fixed cases must be nonempty')
    for row in data['cases']:
        prefix, response = row['actor_prefix_token_ids'], row['probe_response_token_ids']
        if not prefix or not response or any(type(t) is not int or t < 0 for t in prefix + response):
            raise ValueError('Fixed cases require nonempty nonnegative token IDs')
        if any(len(row[k]) != len(response) for k in ('hf_logprobs', 'vllm_logprobs')):
            raise ValueError('Fixed probabilities must be aligned to response tokens')
        if any(not math.isfinite(x) for k in ('hf_logprobs', 'vllm_logprobs') for x in row[k]):
            raise ValueError('Fixed probabilities must be finite')
    return data


def comparison(left, right):
    if not left or len(left) != len(right):
        raise ValueError('Comparisons require nonempty aligned arrays')
    errors = sorted(abs(a - b) for a, b in zip(left, right))
    pos = .99 * (len(errors) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return {'tokens': len(errors), 'mean_abs_error': sum(errors) / len(errors),
            'p99_abs_error': errors[lo] + (errors[hi] - errors[lo]) * (pos - lo),
            'max_abs_error': errors[-1],
            'signed_mean': sum(a - b for a, b in zip(left, right)) / len(errors)}


def engine_kwargs(model, *, fp32_head):
    return {'model': model, 'dtype': 'bfloat16', 'trust_remote_code': True,
            'enable_lora': True, 'max_lora_rank': 64, 'max_loras': 2, 'max_cpu_loras': 2,
            'lora_dtype': 'bfloat16', 'seed': 0, 'generation_config': 'vllm',
            'logprobs_mode': 'processed_logprobs', 'enable_trace_replay': True,
            'max_model_len': 4096, 'max_num_seqs': 32, 'gpu_memory_utilization': .45,
            'tensor_parallel_size': 1,
            **({'hf_overrides': {'head_dtype': 'float32'}} if fp32_head else {})}


def runtime_report():
    import torch
    from scripts.corrected_rl_launcher import RUNTIME
    return {'python': sys.version, 'executable': sys.executable,
            'versions': {key: importlib.metadata.version(key) for key in RUNTIME},
            'cuda': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpus': [{'name': torch.cuda.get_device_name(i),
                      'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]}


def loaded_sources(objects):
    paths = {str(Path(__file__).resolve())}
    for obj in objects:
        path = inspect.getsourcefile(obj)
        if path:
            paths.add(str(Path(path).resolve()))
    return {path: sha256(path) for path in sorted(paths)}


def adapter_dtypes(actor):
    counts = {}
    for name, param in actor.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            key = str(param.dtype)
            counts[key] = counts.get(key, 0) + param.numel()
    return counts


def model_metadata(model):
    """Runs in the vLLM worker, observing its loaded tensors and implementation."""
    import hashlib
    import inspect
    from pathlib import Path
    rows, sources = [], {}
    for name, module in model.named_modules():
        item = {}
        for key in ('lora_a_stacked', 'lora_b_stacked'):
            tensors = getattr(module, key, None)
            if tensors is not None:
                item[key] = [{'dtype': str(t.dtype), 'shape': list(t.shape)} for t in tensors]
        if hasattr(module, 'head_dtype'):
            item['head_dtype'] = str(module.head_dtype)
        if item:
            rows.append({'module': name, **item})
            path = inspect.getsourcefile(type(module))
            if path:
                sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    path = inspect.getsourcefile(type(model))
    if path:
        sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'model_class': str(type(model)), 'modules': rows, 'loaded_sources': sources}


def hf_worker(args, data, report):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vpo_rm.alignment import shared_output_mask
    from vpo_rm.integration import selected_logp_from_logits

    actor = AutoModelForCausalLM.from_pretrained(data['inputs']['actor_model'],
                torch_dtype=torch.bfloat16, trust_remote_code=True).to('cuda:0')
    actor = PeftModel.from_pretrained(actor, data['inputs']['init_adapter'], is_trainable=False).eval()
    actor.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(data['inputs']['actor_model'], trust_remote_code=True)
    head = actor.get_output_embeddings()
    support = shared_output_mask(tokenizer, head.weight.shape[0], torch.device('cuda:0'))
    report['loaded_sources'] = loaded_sources([PeftModel, actor.get_base_model().__class__,
                                               selected_logp_from_logits])
    report['attention_implementation'] = actor.config._attn_implementation
    report['head_weight_dtype'] = str(head.weight.dtype)
    report['initial_adapter_dtypes'] = adapter_dtypes(actor)
    report['cases'] = []
    parameters = {name: p for name, p in actor.named_parameters()
                  if '.lora_A.' in name or '.lora_B.' in name}
    originals = {name: p.detach().cpu().clone() for name, p in parameters.items()}

    def evaluate(case, cached, autocast):
        prefix, response = case['actor_prefix_token_ids'], case['probe_response_token_ids']
        captured = {}
        handle = head.register_forward_pre_hook(lambda module, inputs: captured.update(hidden=inputs[0].detach()))
        native, precise, output_dtypes = [], [], set()

        def reduce_output(output, targets):
            logits = output.logits
            output_dtypes.add(str(logits.dtype))
            hidden = captured.pop('hidden')
            with torch.autocast('cuda', enabled=False):
                fp32 = torch.mm(hidden.reshape(-1, hidden.shape[-1]), head.weight.t(),
                                out_dtype=torch.float32).reshape(*hidden.shape[:-1], -1)
            targets = torch.tensor([targets], device='cuda:0')
            valid = torch.ones_like(targets, dtype=torch.bool)
            for values, z in ((native, logits), (precise, fp32)):
                logp = selected_logp_from_logits(z.masked_fill(~support, -torch.inf), targets, valid)
                values.extend(logp[0].cpu().tolist())

        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                if not cached:
                    ids = torch.tensor([prefix + response], device='cuda:0')
                    keep = torch.arange(len(prefix) - 1, len(prefix) + len(response) - 1, device='cuda:0')
                    output = actor(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   position_ids=torch.arange(ids.shape[1], device='cuda:0')[None, :],
                                   use_cache=False, logits_to_keep=keep, return_dict=True)
                    reduce_output(output, response)
                else:
                    cache = None
                    for i, token in enumerate(response):
                        ids = torch.tensor([prefix if i == 0 else [response[i - 1]]], device='cuda:0')
                        output = actor(input_ids=ids,
                            attention_mask=torch.ones((1, len(prefix) + i), dtype=torch.long, device='cuda:0'),
                            past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
                        cache = output.past_key_values
                        reduce_output(output, [token])
            return {'native_head': native, 'fp32_head': precise,
                    'native_logits_dtypes': sorted(output_dtypes)}
        finally:
            handle.remove()

    try:
        for mode in ('fp32_lora', 'bf16_lora', 'autocast_bf16'):
            for name, param in parameters.items():
                param.data = originals[name].to(device='cuda:0',
                    dtype=torch.bfloat16 if mode == 'bf16_lora' else originals[name].dtype)
            rows = []
            for case in data['cases']:
                modes = {'full': evaluate(case, cached=False, autocast=mode == 'autocast_bf16')}
                if mode != 'autocast_bf16':
                    modes['cached'] = evaluate(case, cached=True, autocast=False)
                rows.append({'row': case['row'], 'adapter_mode': mode,
                             'adapter_dtypes': adapter_dtypes(actor), **modes})
                report['cases'] = report['cases'] + [rows[-1]]
                write_json(Path(args.output_dir) / 'hf.json', report)
    finally:
        for name, param in parameters.items():
            param.data = originals[name].to(device='cuda:0')
        report['adapter_restored_exactly'] = all(torch.equal(p.detach().cpu(), originals[n])
                                                for n, p in parameters.items())


def vllm_worker(args, data, report):
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vpo_rm.integration import checked_sampling_params, vllm_sampling_kwargs

    model = data['inputs']['actor_model']
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocabulary = AutoConfig.from_pretrained(model, trust_remote_code=True).vocab_size
    kwargs = engine_kwargs(model, fp32_head=args.worker == 'vllm-fp32')
    report['engine_kwargs'] = kwargs
    report['loaded_sources'] = loaded_sources([LLM, SamplingParams, vllm_sampling_kwargs])
    llm = LLM(**kwargs)
    request = LoRARequest('fixed-original-sft', 1, data['inputs']['init_adapter'])
    report['batches'] = []
    for size in (len(data['cases']), 32):
        cases = [data['cases'][i % len(data['cases'])] for i in range(size)]
        prompts = [{'prompt_token_ids': case['actor_prefix_token_ids']} for case in cases]
        params = []
        for case in cases:
            trace = case['probe_response_token_ids']
            values = vllm_sampling_kwargs(tokenizer, vocabulary, {
                'temperature': 1., 'min_tokens': 0, 'max_tokens': len(trace),
                'group_size': 1, 'return_logprobs': True})
            params.append(checked_sampling_params(SamplingParams, **values, trace_decode_token_ids=trace))
        results = llm.generate(prompts, params, lora_request=request)
        if len(results) != len(cases):
            raise ValueError('vLLM omitted fixed trace requests')
        rows = []
        for index, (case, result) in enumerate(zip(cases, results)):
            output = result.outputs[0]
            if list(result.prompt_token_ids) != case['actor_prefix_token_ids']:
                raise ValueError('vLLM changed the fixed prompt tokens')
            if list(output.token_ids) != case['probe_response_token_ids']:
                raise ValueError('vLLM did not replay the complete fixed trace')
            values = [float(mapping[token].logprob) for mapping, token in zip(output.logprobs, output.token_ids)]
            if len(values) != len(output.token_ids) or not all(math.isfinite(x) for x in values):
                raise ValueError('vLLM trace logprobs incomplete or nonfinite')
            rows.append({'row': case['row'], 'replica': index, 'logprobs': values,
                         'token_ids': list(output.token_ids), 'finish_reason': output.finish_reason})
        report['batches'].append({'batch_size': size, 'rows': rows})
        write_json(Path(args.output_dir) / f'{args.worker}.json', report)
    report['loaded_model_metadata'] = llm.apply_model(model_metadata)


def make_comparisons(data, hf, native, precise):
    rows = []
    original = {case['row']: case for case in data['cases']}
    for hrow in hf['cases']:
        source = original[hrow['row']]
        for forward in ('full', 'cached'):
            if forward not in hrow:
                continue
            for head in ('native_head', 'fp32_head'):
                values = hrow[forward][head]
                rows.append({'case': hrow['row'], 'adapter_mode': hrow['adapter_mode'],
                    'forward': forward, 'hf_head': head, 'against': 'original_vllm',
                    **comparison(values, source['vllm_logprobs'])})
                rows.append({'case': hrow['row'], 'adapter_mode': hrow['adapter_mode'],
                    'forward': forward, 'hf_head': head, 'against': 'original_hf',
                    **comparison(values, source['hf_logprobs'])})
                for engine in (native, precise):
                    for batch in engine['batches']:
                        candidates = [r for r in batch['rows'] if r['row'] == hrow['row']]
                        for candidate in candidates:
                            rows.append({'case': hrow['row'], 'adapter_mode': hrow['adapter_mode'],
                                'forward': forward, 'hf_head': head, 'against': engine['worker'],
                                'batch_size': batch['batch_size'], 'replica': candidate['replica'],
                                **comparison(values, candidate['logprobs'])})
        if 'cached' in hrow:
            for head in ('native_head', 'fp32_head'):
                rows.append({'case': hrow['row'], 'adapter_mode': hrow['adapter_mode'],
                    'hf_head': head, 'against': 'hf_full_vs_cached',
                    **comparison(hrow['full'][head], hrow['cached'][head])})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--actor-gpu', default='0')
    parser.add_argument('--vllm-gpu', default='1')
    parser.add_argument('--worker', choices=('hf', 'vllm-native', 'vllm-fp32'))
    args = parser.parse_args(argv)
    if args.actor_gpu == args.vllm_gpu and args.worker is None:
        parser.error('HF and vLLM require distinct GPU selectors')
    args.cases, args.output_dir = args.cases.resolve(), args.output_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    data = load_cases(args.cases)
    if args.worker:
        report = {'status': 'started', 'worker': args.worker, 'cases_sha256': sha256(args.cases),
                  'runtime': runtime_report(), 'script_sha256': sha256(__file__)}
        try:
            from scripts.corrected_rl_launcher import RUNTIME
            if report['runtime']['versions'] != RUNTIME:
                raise ValueError('Diagnostic runtime differs from the failed preflight runtime')
            if len(report['runtime']['gpus']) != 1:
                raise ValueError('Diagnostic worker requires exactly one visible GPU')
            (hf_worker if args.worker == 'hf' else vllm_worker)(args, data, report)
            report['status'] = 'completed'
        except Exception:
            report['status'] = 'error'
            report['error'] = traceback.format_exc()
            raise
        finally:
            write_json(args.output_dir / f'{args.worker}.json', report)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'fixed-cases.json', data)
    workers, logs = [], []
    try:
        for worker, gpu in [('hf', args.actor_gpu), ('vllm-native', args.vllm_gpu)]:
            log = (args.output_dir / f'{worker}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--cases', str(args.cases), '--output-dir', str(args.output_dir), '--worker', worker],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            workers.append(proc)
        native_code = workers[1].wait()
        if native_code:
            raise RuntimeError(f'Native vLLM worker failed: exit {native_code}')
        with (args.output_dir / 'vllm-fp32.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--cases', str(args.cases), '--output-dir', str(args.output_dir), '--worker', 'vllm-fp32'],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.vllm_gpu), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or workers[0].wait():
            raise RuntimeError('A diagnostic worker failed; inspect its JSON/log')
        reports = [json.loads((args.output_dir / f'{worker}.json').read_text())
                   for worker in ('hf', 'vllm-native', 'vllm-fp32')]
        write_json(args.output_dir / 'results.json', {
            'status': 'completed_observation_only', 'cases_sha256': sha256(args.cases),
            'script_sha256': sha256(__file__), 'comparisons': make_comparisons(data, *reports),
            'workers': reports})
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
