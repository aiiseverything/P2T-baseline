#!/usr/bin/env python3
"""Stage 0 SFT initialization: light LoRA SFT on UltraFeedback chosen answers.

Produces the shared starting checkpoint for the GRPO and VPO-RM arms (p9g):
one epoch, lr 1e-4, same LoRA geometry as the RL trainer (r=64, alpha=128,
same target modules) so the adapter drops straight into the RL runs.  The
2000 SHA256-ordered validation prompts are excluded so the frozen 256-prompt
offline eval stays untouched by initialization.

Loss is masked to response tokens; sequences are capped at 4096 tokens
(2048 prompt + 2048 response, matching the experiment spec).  Length-sorted
batches with shuffled batch order keep padding low without correlation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.sft_response_tokens import (
        build_response_eos_spec,
        response_labels,
        rewrite_response_eos,
        supervised_token_sha256,
    )
except ModuleNotFoundError:  # direct invocation: python scripts/sft_init.py
    from sft_response_tokens import (
        build_response_eos_spec,
        response_labels,
        rewrite_response_eos,
        supervised_token_sha256,
    )

ROOT = Path(__file__).resolve().parents[1]


def load_chosen_pairs(parquet_path: str, validation_size: int,
                      exclude_benchmarks: bool = True, benchmark_paths=None):
    """(prompt, chosen_text) pairs for the TRAIN side of the SHA256 split."""
    import pyarrow.parquet as pq
    from vpo_rm.trainer import split_prompts
    from vpo_rm.data import exclude_benchmark_prompts, normalize_prompt
    table = pq.read_table(parquet_path).to_pylist()

    def as_text(chosen):
        if isinstance(chosen, str):
            return chosen
        if isinstance(chosen, list):  # H4 message list [user, assistant]
            # v1 bug: joining ALL messages made the target "question\nanswer",
            # teaching 94.7% question echo. Supervise the assistant turn only.
            texts = [str(m.get("content", "")) for m in chosen
                     if isinstance(m, dict) and m.get("role") == "assistant"]
            if texts:
                return texts[-1]
            raise ValueError("message list has no assistant turn")
        raise ValueError(f"unsupported chosen field type {type(chosen)}")

    pairs = [(str(r["prompt"]), as_text(r["chosen"])) for r in table]
    prompts = [p for p, _ in pairs]
    exclusion = None
    if exclude_benchmarks:
        prompts, exclusion = exclude_benchmark_prompts(prompts, benchmark_paths)
        first_pair = {}
        for prompt, answer in pairs:
            key = normalize_prompt(prompt)
            if key and key not in first_pair:
                first_pair[key] = (prompt, answer)
        pairs = [first_pair[normalize_prompt(prompt)] for prompt in prompts]
    train, valid, split = split_prompts(prompts, validation_size)
    keep = set(train)
    out = [(p, c) for p, c in pairs if p in keep]
    if exclusion is not None:
        split["benchmark_exclusion"] = exclusion
    return out, split


class SFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len, response_eos_spec):
        self.examples = []
        skipped_boundary = skipped_len = replaced_eos = 0
        for prompt, answer in pairs:
            user = [{"role": "user", "content": prompt}]
            full = user + [{"role": "assistant", "content": answer}]
            prefix = tokenizer.apply_chat_template(
                user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = tokenizer.apply_chat_template(
                full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if not text.startswith(prefix):
                skipped_boundary += 1
                continue
            pre_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if full_ids[:len(pre_ids)] != pre_ids:
                skipped_boundary += 1
                continue
            rewritten_ids = rewrite_response_eos(full_ids, len(pre_ids), response_eos_spec)
            if len(full_ids) > max_len or len(full_ids) == len(pre_ids):
                skipped_len += 1
                continue
            replaced_eos += int(rewritten_ids != full_ids)
            labels = response_labels(rewritten_ids, len(pre_ids))
            self.examples.append((rewritten_ids, labels))
        self.stats = {"kept": len(self.examples), "skipped_boundary": skipped_boundary,
                      "skipped_length": skipped_len, "total": len(pairs),
                      "response_eos_mode": response_eos_spec.mode,
                      "response_eos_id": response_eos_spec.actual_eos_id,
                      "response_eos_replaced": replaced_eos,
                      "target_token_sha256": supervised_token_sha256(
                          labels for _, labels in self.examples)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return ids, labels


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def build_microbatch_schedule(num_batches: int, epochs: float, seed: int) -> list[int]:
    """Return shuffled batch indices for exactly ceil(num_batches * epochs) batches.

    Every full epoch visits each length bucket once. Fractional epochs consume a
    deterministic prefix of a newly shuffled epoch rather than being rounded up
    to a complete epoch.
    """
    if num_batches < 1:
        raise ValueError("num_batches must be positive")
    if not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("epochs must be finite and positive")
    remaining = math.ceil(num_batches * epochs)
    rng = random.Random(seed)
    schedule = []
    while remaining:
        epoch_order = list(range(num_batches))
        rng.shuffle(epoch_order)
        take = min(remaining, num_batches)
        schedule.extend(epoch_order[:take])
        remaining -= take
    return schedule


def accumulation_windows(schedule, grad_accum: int) -> list[list[int]]:
    """Group microbatches into optimizer updates, retaining the partial tail."""
    if grad_accum < 1:
        raise ValueError("grad_accum must be positive")
    schedule = list(schedule)
    return [schedule[i:i + grad_accum] for i in range(0, len(schedule), grad_accum)]


def response_token_weights(labels, terminator_id: int, eos_weight: float) -> list[float]:
    """Per-label weights for a response, including its final terminator."""
    if not math.isfinite(eos_weight) or eos_weight <= 0:
        raise ValueError("eos_weight must be finite and positive")
    weights = [0.0 if token_id == -100 else 1.0 for token_id in labels]
    if eos_weight != 1.0:
        positions = [i for i, token_id in enumerate(labels) if token_id == terminator_id]
        if not positions:
            raise ValueError("supervised response has no configured terminator")
        weights[positions[-1]] = eos_weight
    return weights


def effective_batch_denominator(ds, batches, batch_indices, terminator_id, eos_weight):
    """Weighted supervised-token denominator for one optimizer update."""
    return sum(
        sum(response_token_weights(ds.examples[i][1], terminator_id, eos_weight))
        for batch_idx in batch_indices for i in batches[batch_idx]
    )


def step_is_due(step: int, total_steps: int, every: int) -> bool:
    """Periodic event schedule with an unconditional final-step event."""
    return step == total_steps or (every > 0 and step % every == 0)


def default_sft_output(model: str) -> str:
    """Choose a fresh output name that records native-EOS initialization."""
    model_name = Path(model.rstrip("/")).name.lower()
    if not model_name:
        raise ValueError("model path must have a basename")
    return f"models/sft-native-eos-{model_name}"


def prepare_sft_output(output) -> Path:
    """Create an output directory, refusing to append to a prior SFT run."""
    output = Path(output)
    markers = (
        output / "sft_metrics.jsonl",
        output / "sft_manifest.json",
        output / "adapter_config.json",
        output / "adapter_model.safetensors",
        output / "adapter_model.bin",
    )
    if any(marker.exists() for marker in markers):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def monitor_stop_token_ids(tokenizer) -> tuple[int, ...]:
    """Use the shared registered-token policy for free-running monitors."""
    from vpo_rm.token_policy import get_stop_token_ids
    stop_ids = get_stop_token_ids(tokenizer)
    if not stop_ids:
        raise ValueError("tokenizer has no registered EOS/chat-end stop token")
    return stop_ids


# --- free-running monitor: the v1 pathologies (echo, script salad,
# Confidence loops, non-termination) are invisible to teacher-forced loss ---
import re as _re
_SALAD = _re.compile(
    "[฀-๿ऀ-ॿಀ-೿؀-ۿ"
    "Ѐ-ӿ぀-ヿ一-鿿]")
_CONF = "Confidence:"


def _monitor_pass(model, tokenizer, prompts, device, max_new=384):
    model.eval()
    outs, lens = [], []
    stop_ids = monitor_stop_token_ids(tokenizer)
    with torch.no_grad():
        for p in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
            gen = model.generate(ids, do_sample=True, temperature=1.0,
                                 top_p=1.0, max_new_tokens=max_new,
                                 eos_token_id=stop_ids,
                                 pad_token_id=tokenizer.pad_token_id)
            resp = gen[0, ids.shape[1]:]
            outs.append(tokenizer.decode(resp, skip_special_tokens=True))
            lens.append(resp.shape[0])
    model.train()
    n = len(prompts)
    norm = lambda s: " ".join(s.lower().split())
    echo = conf = salad = trunc = 0
    for p, o, l in zip(prompts, outs, lens):
        pn, on = norm(p), norm(o)
        echo += int(len(pn) > 30 and (on[:len(pn)] == pn or pn in on[:2 * len(pn) + 100]))
        conf += int(o.count(_CONF) >= 2)
        salad += int(bool(_SALAD.search(o)))
        trunc += int(l >= max_new)
    return {"echo": echo / n, "conf_loop": conf / n, "script_salad": salad / n,
            "truncated": trunc / n, "mean_chars": sum(map(len, outs)) / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--dataset-path",
                   default="datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    p.add_argument("--output", default=None,
                   help="adapter output (default: models/sft-native-eos-<model-name>)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-examples", type=int, default=0,
                   help="Random (seeded) subsample of the train pairs; 0 = all")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="debug: cap examples (0 = all)")
    p.add_argument("--monitor-every", type=int, default=100,
                   help="free-running probe cadence in steps (0 = off)")
    p.add_argument("--monitor-prompts", default="datasets/sft_v2/monitor_prompts.json")
    p.add_argument("--eos-weight", type=float, default=1.0,
                   help="loss weight for each example's final response token "
                        "(the configured terminator); 1.0 = standard CE")
    p.add_argument("--response-eos", choices=["chat_template", "native"],
                   default="native",
                   help="response terminator to supervise; native replaces only "
                        "the final assistant <|im_end|> with tokenizer EOS")
    p.add_argument("--allow-benchmark-overlap", action="store_true",
                   help="historical reproduction only: do not exclude final benchmarks")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output is None:
        args.output = default_sft_output(args.model)
    if args.micro_batch < 1:
        p.error("--micro-batch must be positive")
    if args.grad_accum < 1:
        p.error("--grad-accum must be positive")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        p.error("--epochs must be finite and positive")
    if not math.isfinite(args.eos_weight) or args.eos_weight <= 0:
        p.error("--eos-weight must be finite and positive")
    out = prepare_sft_output(args.output)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    response_eos_spec = build_response_eos_spec(tokenizer, args.response_eos)
    print(f"response EOS: mode={response_eos_spec.mode} "
          f"id={response_eos_spec.actual_eos_id}", flush=True)
    pairs, split = load_chosen_pairs(
        args.dataset_path, validation_size=2000,
        exclude_benchmarks=not args.allow_benchmark_overlap)
    if args.max_examples and len(pairs) > args.max_examples:
        pairs = random.Random(args.seed).sample(pairs, args.max_examples)
    if args.limit:
        pairs = pairs[:args.limit]
    ds = SFTDataset(pairs, tokenizer, args.max_len, response_eos_spec)
    print(f"dataset: {ds.stats}", flush=True)
    if not ds:
        raise RuntimeError("no usable examples after filtering")

    # Length-sorted buckets, shuffled batch order: low padding, no length bias
    # in the batch sequence.
    order = sorted(range(len(ds)), key=lambda i: len(ds.examples[i][0]))
    batches = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
    schedule = build_microbatch_schedule(len(batches), args.epochs, args.seed)
    update_windows = accumulation_windows(schedule, args.grad_accum)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    opt = torch.optim.AdamW((p_ for p_ in model.parameters() if p_.requires_grad),
                            lr=args.learning_rate, weight_decay=0.0)

    steps_total = len(update_windows)
    warmup = max(1, int(steps_total * args.warmup_ratio))

    def lr_at(step):
        if step < warmup:
            return args.learning_rate * (step + 1) / warmup
        t = (step - warmup) / max(1, steps_total - warmup)
        return args.learning_rate * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * t)))

    metrics_path = out / "sft_metrics.jsonl"
    model.train()
    monitor_prompts = []
    if args.monitor_every and Path(args.monitor_prompts).exists():
        monitor_prompts = json.loads(Path(args.monitor_prompts).read_text())[:16]
        print(f"monitor: {len(monitor_prompts)} free-running probes every "
              f"{args.monitor_every} steps", flush=True)
    t0 = time.monotonic()
    tokens_seen = 0
    for step_index, window in enumerate(update_windows):
        step = step_index + 1
        denominator = effective_batch_denominator(
            ds, batches, window, response_eos_spec.actual_eos_id, args.eos_weight)
        if denominator <= 0:
            raise RuntimeError("optimizer update has no supervised response tokens")
        opt.zero_grad(set_to_none=True)
        update_numerator = 0.0
        content_sum = 0.0
        content_count = 0
        for batch_idx in window:
            ids, labels, attn = collate([ds[i] for i in batches[batch_idx]],
                                        tokenizer.pad_token_id)
            ids, labels, attn = ids.to(args.device), labels.to(args.device), attn.to(args.device)
            tokens_seen += int(attn.sum())
            if args.eos_weight == 1.0:
                mean_loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
                valid_count = int((labels[:, 1:] != -100).sum())
                numerator = mean_loss * valid_count
                update_numerator += float(numerator.detach())
            else:
                # weighted CE: all response tokens weight 1, the single final
                # response terminator weight eos_weight.
                # NOTE: 'fwd' not 'out' — 'out' is the output-dir Path below.
                fwd = model(input_ids=ids, attention_mask=attn)
                logits = fwd.logits[:, :-1].float()
                targets = labels[:, 1:]
                valid = targets != -100
                ce = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), targets.masked_fill(~valid, 0),
                    reduction="none")
                w = valid.float()
                # Weight the configured response terminator, not an arbitrary
                # stop token in response content or the trailing newline.
                is_stop = targets.eq(response_eos_spec.actual_eos_id)
                pos = torch.arange(targets.shape[1],
                                   device=targets.device).expand_as(targets)
                last_stop = torch.where(
                    is_stop & valid, pos,
                    torch.full_like(targets, -1)).amax(-1)
                rows = torch.arange(w.shape[0], device=w.device)
                sel = last_stop >= 0
                if not bool(sel.all()):
                    raise RuntimeError("supervised response has no configured terminator")
                w[rows[sel], last_stop[sel]] = args.eos_weight
                numerator = (ce * w).sum()
                content_mask = valid & ~(
                    pos.eq(last_stop[:, None]) & sel[:, None])
                update_numerator += float(numerator.detach())
                content_sum += float(ce[content_mask].sum().detach())
                content_count += int(content_mask.sum())
                del fwd, logits, ce, w, is_stop, pos, last_stop
            (numerator / denominator).backward()
        lr = lr_at(step_index)
        for g in opt.param_groups:
            g["lr"] = lr
        gn = torch.nn.utils.clip_grad_norm_(
            (p_ for p_ in model.parameters() if p_.requires_grad), 1.0)
        opt.step()
        if step_is_due(step, steps_total, 20):
            rec = {"step": step, "of": steps_total,
                   "loss": update_numerator / denominator,
                   "lr": lr, "grad_norm": float(gn),
                   "tokens_per_s": tokens_seen / max(time.monotonic() - t0, 1e-9),
                   "input_tokens_seen": tokens_seen,
                   "supervised_weight": denominator,
                   "microbatches": len(window)}
            if args.eos_weight != 1.0:
                rec["content_ce"] = content_sum / max(content_count, 1)
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if monitor_prompts and step_is_due(step, steps_total, args.monitor_every):
            mrec = {"event": "monitor", "step": step,
                    **_monitor_pass(model, tokenizer, monitor_prompts, args.device)}
            with metrics_path.open("a") as f:
                f.write(json.dumps(mrec) + "\n")
            print(json.dumps(mrec), flush=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    payload = hashlib.sha256("\n".join(f"{p}|{c[:64]}" for p, c in pairs).encode()).hexdigest()
    (out / "sft_manifest.json").write_text(json.dumps({
        "config": vars(args), "dataset_stats": ds.stats, "data_sha256": payload,
        "split": split,
        "training_schedule": {
            "microbatches": len(schedule), "optimizer_steps": steps_total,
            "final_update_microbatches": len(update_windows[-1]),
            "loss_normalization": "global_weighted_supervised_token_mean_per_update",
        },
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False))
    print(f"saved adapter to {out}", flush=True)


if __name__ == "__main__":
    main()
