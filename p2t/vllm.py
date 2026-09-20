"""vLLM generation protocol: client helpers plus the server subprocess handle.

Port of the parent project's ``vpo_rm/integration.py`` generation helpers and
``scripts/vllm_generate_server.py``.  Generation runs in a separate process that
owns its own GPUs and stays resident, so the actor and reward model keep their
cards and no CUDA graph is rebuilt per rollout.

The protocol guarantees that matter for P2T:

* The sampled tokens are generated under exactly the support the trainer
  updates over -- Qwen3 reserves 271 embedding rows past the tokenizer's
  vocabulary, and a rare draw on one of them would otherwise produce a token
  with no legal probability at update time.
* Chosen-token log probabilities come back with the text, so the importance
  ratio is anchored on the sampler rather than on a re-forward of the actor.
* The LoRA adapter of the current step is hot-loaded, so rollouts are on-policy
  without reloading the base model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from .tokens import get_stop_token_ids, load_actor_tokenizer, shared_output_mask


def unpadded_prompt_token_ids(batch):
    """Serialize exactly the attended HF prompt IDs for generation RPCs."""
    return [ids[mask.bool()].tolist()
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]


def vllm_support_kwargs(tokenizer, vocab_size: int) -> dict:
    """Exact compact output ban, staying under vLLM's 1024-entry bias limit."""
    banned = (~shared_output_mask(tokenizer, vocab_size)).nonzero().flatten().tolist()
    if len(banned) > 1024:
        raise ValueError("This vLLM protocol supports at most 1024 suppressed vocabulary IDs")
    return {"logit_bias": {token_id: -math.inf for token_id in banned}}


def vllm_sampling_kwargs(tokenizer, vocab_size: int, request: dict) -> dict:
    """One explicit on-policy protocol shared by the server and the trainer."""
    probe = bool(request.get("probe", False))
    temperature = float(request.get("temperature", 1.0))
    if not math.isfinite(temperature) or not .01 <= temperature <= 2.0:
        raise ValueError("training temperature must be in [0.01, 2] to avoid vLLM clamping")
    if request.get("top_p", 1.0) != 1.0 or request.get("top_k", 0) != 0:
        raise ValueError("training supports top_p=1 and top_k=0 only")
    if float(request.get("presence_penalty", 0.0)) != 0:
        raise ValueError("presence_penalty is unsupported by the training probability protocol")
    maximum = int(request["max_tokens"])
    minimum = 0 if probe else int(request.get("min_tokens", 0))
    if maximum < 1 or not 0 <= minimum <= maximum:
        raise ValueError("generation requires 0 <= min_tokens <= max_tokens")
    return {"temperature": 0.0 if probe else temperature,
            "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
            "max_tokens": maximum, "min_tokens": minimum,
            "n": 1 if probe else int(request.get("group_size", 8)),
            "stop_token_ids": list(get_stop_token_ids(tokenizer)),
            **({"logprobs": 1} if request.get("return_logprobs", False) else {}),
            **vllm_support_kwargs(tokenizer, vocab_size)}


def checked_sampling_params(sampling_params_class, **kwargs):
    """Fail closed if the backend changes exact support or clamps temperature."""
    params = sampling_params_class(**kwargs)
    expected = kwargs.get("logit_bias") or {}
    actual = getattr(params, "logit_bias", None) or {}
    for token_id, value in expected.items():
        if value == -math.inf and actual.get(token_id) != -math.inf:
            raise RuntimeError("vLLM modified the exact output support; refusing approximate sampling")
    if "temperature" in kwargs and getattr(params, "temperature", kwargs["temperature"]) != kwargs["temperature"]:
        raise RuntimeError("vLLM modified sampling temperature; probability protocol would differ")
    return params


def sampling_summary(kwargs):
    """JSON-safe sampling provenance; never serialize nonstandard Infinity."""
    summary = {k: v for k, v in kwargs.items() if k not in {"logit_bias", "allowed_token_ids"}}
    suppressed = sorted((kwargs.get("logit_bias") or {}).keys())
    summary["suppressed_token_count"] = len(suppressed)
    summary["suppressed_token_ids_sha256"] = hashlib.sha256(json.dumps(suppressed).encode()).hexdigest()
    return summary


def generation_payload(generated, stop_token_ids, *, include_logprobs=False) -> dict:
    """Keep engine termination metadata; an emitted final stop is complete at cap."""
    rows, reasons, engine_reasons, stop_reasons = [], [], [], []
    selected_logprobs, prompt_token_ids = [], []
    for result in generated:
        if include_logprobs:
            prompt_token_ids.append(list(result.prompt_token_ids))
        for output in result.outputs:
            ids = list(output.token_ids)
            reason = output.finish_reason
            if not ids or reason not in {"stop", "length"}:
                raise ValueError("Generation must return nonempty responses with stop/length metadata")
            rows.append(ids)
            engine_reasons.append(reason)
            reasons.append("stop" if ids[-1] in stop_token_ids else reason)
            stop_reasons.append(output.stop_reason)
            if include_logprobs:
                probabilities = getattr(output, "logprobs", None)
                if probabilities is None or len(probabilities) != len(ids):
                    raise ValueError("Requested chosen-token logprobs are missing or misaligned")
                values = []
                for token, options in zip(ids, probabilities):
                    selected = options.get(token)
                    if selected is None or not math.isfinite(selected.logprob):
                        raise ValueError("Requested chosen-token logprob is missing or nonfinite")
                    values.append(float(selected.logprob))
                selected_logprobs.append(values)
    return {"rows": rows, "finish_reasons": reasons,
            "engine_finish_reasons": engine_reasons, "stop_reasons": stop_reasons,
            "tokens": sum(map(len, rows)),
            **({"selected_logprobs": selected_logprobs, "prompt_token_ids": prompt_token_ids}
               if include_logprobs else {})}


def tokenize_rendered_prompts(tokenizer, prompts, prompt_token_ids=None) -> list[dict]:
    """Bind explicit token inputs to rendered chats without a second BOS."""
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError("Generation prompts must be a nonempty list of rendered strings")
    encoded = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    if any(not row for row in encoded):
        raise ValueError("Generation prompt tokens must be nonempty")
    if prompt_token_ids is not None and prompt_token_ids != encoded:
        raise ValueError("Explicit prompt token IDs differ from the rendered chat protocol")
    return [{"prompt_token_ids": row} for row in encoded]


def pack_rollout(result, batch, rendered, *, group_size, max_response_tokens,
                 pad_token_id, adapter_id):
    """Bind one RPC's chosen-token probabilities to its exact HF rollout rows."""
    if result.get("logprobs_mode") != "processed_logprobs":
        raise RuntimeError("vLLM rollout requires processed logprobs")
    if type(result.get("adapter_id")) is not int or result["adapter_id"] != adapter_id:
        raise RuntimeError("vLLM rollout adapter identity differs from the request")
    expected_prompts = unpadded_prompt_token_ids(batch)
    if len(rendered) != len(expected_prompts) or result.get("prompt_token_ids") != expected_prompts:
        raise RuntimeError("vLLM prompt token IDs differ from unpadded HF prompts")
    rows, finish_reasons = result.get("rows", []), result.get("finish_reasons", [])
    expected = len(rendered) * group_size
    if (not expected or len(rows) != expected or len(finish_reasons) != expected
            or any(not row for row in rows)):
        raise RuntimeError("vLLM returned missing or empty responses")
    lengths = [len(row) for row in rows]
    if max(lengths) > max_response_tokens:
        raise RuntimeError("vLLM exceeded response token limit")
    probabilities = result.get("selected_logprobs")
    if (not isinstance(probabilities, list) or len(probabilities) != expected
            or any(not isinstance(values, list) or len(values) != length
                   for values, length in zip(probabilities, lengths))):
        raise RuntimeError("vLLM selected logprobs must cover every response token")
    flat = [value for values in probabilities for value in values]
    if any(type(value) not in (float, int) or not math.isfinite(value) or value > 0 for value in flat):
        raise RuntimeError("vLLM selected logprobs must be finite and nonpositive")
    device, width = batch["input_ids"].device, max(lengths)
    prompt_width = batch["input_ids"].shape[1]
    responses = torch.full((expected, width), pad_token_id, dtype=torch.long, device=device)
    rollout_logprobs = torch.zeros((expected, width), dtype=torch.float32, device=device)
    for i, (row, values) in enumerate(zip(rows, probabilities)):
        responses[i, :len(row)] = torch.tensor(row, device=device)
        rollout_logprobs[i, :len(row)] = torch.tensor(values, dtype=torch.float32, device=device)
    rmask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    expanded_input = batch["input_ids"].repeat_interleave(group_size, dim=0)
    prompt_mask = batch["attention_mask"].repeat_interleave(group_size, dim=0)
    input_ids = torch.cat([expanded_input, responses], dim=1)
    full_mask = torch.cat([prompt_mask, rmask.to(prompt_mask.dtype)], dim=1)
    positions = torch.arange(prompt_width, input_ids.shape[1], device=device).expand(expected, -1)
    summary = {key: value for key, value in result.items()
               if key not in {"rows", "selected_logprobs", "prompt_token_ids"}}
    summary.update(scope="last_generation_request", response_lengths=lengths,
                   mean_response_tokens=sum(lengths) / expected,
                   padded_response_width=width, prompt_width=prompt_width,
                   truncation_rate=sum(reason == "length" for reason in finish_reasons) / expected)
    rollout = (input_ids, full_mask, positions, responses, rmask,
               [text for text in rendered for _ in range(group_size)], finish_reasons,
               rollout_logprobs)
    return rollout, summary


class GenerationServer:
    """Spawns and talks to ``p2t.vllm_server`` over a Unix domain socket."""

    def __init__(self, *, model, tokenizer_source, socket_path, gpus, max_num_seqs,
                 seed, gpu_memory_utilization, tensor_parallel_size,
                 policy_head_dtype="float32", log_path=None):
        self.socket_path = Path(socket_path)
        self.log_path = Path(log_path) if log_path else None
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
               "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
               "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        env.pop("PRESENCE_PENALTY", None)
        command = [sys.executable, "-m", "p2t.vllm_server",
                   "--model", str(model), "--tokenizer", str(tokenizer_source),
                   "--socket", str(self.socket_path),
                   "--max-num-seqs", str(max_num_seqs), "--seed", str(seed),
                   "--gpu-memory-utilization", str(gpu_memory_utilization),
                   "--tensor-parallel-size", str(tensor_parallel_size),
                   "--policy-head-dtype", policy_head_dtype]
        handle = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        self._connection = None
        self._reader = None

    def wait_until_ready(self, timeout: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}")
            if self.socket_path.exists():
                try:
                    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    probe.connect(str(self.socket_path))
                    probe.close()
                    return
                except OSError:
                    time.sleep(0.5)
            else:
                time.sleep(1.0)
        raise TimeoutError("vLLM server did not become ready")

    def request(self, payload: dict) -> dict:
        if self._connection is None:
            self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._connection.connect(str(self.socket_path))
            self._reader = self._connection.makefile("r")
        self._connection.sendall((json.dumps(payload) + "\n").encode())
        line = self._reader.readline()
        if not line:
            raise RuntimeError("vLLM server closed the connection mid-request")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(f"vLLM request failed: {result.get('error')}")
        return result

    def close(self) -> None:
        try:
            if self._connection is not None:
                self.request({"shutdown": True})
        except Exception:
            pass
        finally:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
