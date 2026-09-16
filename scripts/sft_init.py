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

ROOT = Path(__file__).resolve().parents[1]


def load_chosen_pairs(parquet_path: str, validation_size: int):
    """(prompt, chosen_text) pairs for the TRAIN side of the SHA256 split."""
    import pyarrow.parquet as pq
    from vpo_rm.trainer import split_prompts
    table = pq.read_table(parquet_path).to_pylist()
    prompts = [str(r["prompt"]) for r in table]

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
    train, valid, split = split_prompts(prompts, validation_size)
    keep = set(train)
    out = [(p, c) for p, c in pairs if p in keep]
    return out, split


class SFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.examples = []
        skipped_boundary = skipped_len = 0
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
            if len(full_ids) > max_len or len(full_ids) == len(pre_ids):
                skipped_len += 1
                continue
            labels = [-100] * len(pre_ids) + full_ids[len(pre_ids):]
            self.examples.append((full_ids, labels))
        self.stats = {"kept": len(self.examples), "skipped_boundary": skipped_boundary,
                      "skipped_length": skipped_len, "total": len(pairs)}

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
    stop_ids = [i for i in (tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if isinstance(i, int) and i >= 0]
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
    p.add_argument("--output", default="models/sft-init-qwen3-14b-base")
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
                        "(the closing im_end); 1.0 = standard CE")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stop_ids = {i for i in (tokenizer.eos_token_id,
                             tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if isinstance(i, int) and i >= 0}
    print(f"loss EOS-weighting targets stop token ids: {sorted(stop_ids)}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs, split = load_chosen_pairs(args.dataset_path, validation_size=2000)
    if args.max_examples and len(pairs) > args.max_examples:
        pairs = random.Random(args.seed).sample(pairs, args.max_examples)
    if args.limit:
        pairs = pairs[:args.limit]
    ds = SFTDataset(pairs, tokenizer, args.max_len)
    print(f"dataset: {ds.stats}", flush=True)
    if not ds:
        raise RuntimeError("no usable examples after filtering")

    # Length-sorted buckets, shuffled batch order: low padding, no length bias
    # in the batch sequence.
    order = sorted(range(len(ds)), key=lambda i: len(ds.examples[i][0]))
    batches = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
    rng = random.Random(args.seed)
    rng.shuffle(batches)

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

    steps_total = math.ceil(len(batches) * args.epochs / args.grad_accum)
    warmup = max(1, int(steps_total * args.warmup_ratio))

    def lr_at(step):
        if step < warmup:
            return args.learning_rate * (step + 1) / warmup
        t = (step - warmup) / max(1, steps_total - warmup)
        return args.learning_rate * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * t)))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "sft_metrics.jsonl"
    model.train()
    monitor_prompts = []
    if args.monitor_every and Path(args.monitor_prompts).exists():
        monitor_prompts = json.loads(Path(args.monitor_prompts).read_text())[:16]
        print(f"monitor: {len(monitor_prompts)} free-running probes every "
              f"{args.monitor_every} steps", flush=True)
    t0 = time.monotonic()
    opt.zero_grad(set_to_none=True)
    micro = 0
    for epoch in range(math.ceil(args.epochs)):
        for batch_idx in range(len(batches)):
            ids, labels, attn = collate([ds[i] for i in batches[batch_idx]],
                                        tokenizer.pad_token_id)
            ids, labels, attn = ids.to(args.device), labels.to(args.device), attn.to(args.device)
            if args.eos_weight == 1.0:
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
                content_ce = None
            else:
                # weighted CE: all response tokens weight 1, the single final
                # response token (closing im_end) weight eos_weight.
                # NOTE: 'fwd' not 'out' — 'out' is the output-dir Path below.
                fwd = model(input_ids=ids, attention_mask=attn)
                logits = fwd.logits[:, :-1].float()
                targets = labels[:, 1:]
                valid = targets != -100
                ce = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), targets.masked_fill(~valid, 0),
                    reduction="none")
                w = valid.float()
                # weight the actual STOP TOKEN (<|im_end|> / <|endoftext|>),
                # NOT the last target — the template's trailing newline made
                # "last" land on '\n' in the first eosw attempt (2026-09-16)
                is_stop = torch.zeros_like(targets, dtype=torch.bool)
                for sid in stop_ids:
                    is_stop |= targets.eq(sid)
                pos = torch.arange(targets.shape[1],
                                   device=targets.device).expand_as(targets)
                last_stop = torch.where(
                    is_stop & valid, pos,
                    torch.full_like(targets, -1)).amax(-1)
                rows = torch.arange(w.shape[0], device=w.device)
                sel = last_stop >= 0
                w[rows[sel], last_stop[sel]] = args.eos_weight
                loss = (ce * w).sum() / w.sum()
                content_ce = float(ce[valid & (w == 1)].mean())
                del fwd, logits, ce, w, is_stop, pos, last_stop
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                step = micro // args.grad_accum
                lr = lr_at(step)
                for g in opt.param_groups:
                    g["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(
                    (p_ for p_ in model.parameters() if p_.requires_grad), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 20 == 0 or step == steps_total:
                    rec = {"step": step, "of": steps_total, "loss": float(loss.detach()),
                           "lr": lr, "grad_norm": float(gn),
                           "tokens_per_s": micro * args.micro_batch / (time.monotonic() - t0) *
                           sum(len(ds.examples[i][0]) for i in batches[batch_idx]) /
                           (args.micro_batch * max(1, micro))}
                    if content_ce is not None:
                        rec["content_ce"] = content_ce
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                if monitor_prompts and (step % args.monitor_every == 0 or step == steps_total):
                    mrec = {"event": "monitor", "step": step,
                            **_monitor_pass(model, tokenizer, monitor_prompts, args.device)}
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(mrec) + "\n")
                    print(json.dumps(mrec), flush=True)
                if step >= steps_total:
                    break
        else:
            continue
        break

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    payload = hashlib.sha256("\n".join(f"{p}|{c[:64]}" for p, c in pairs).encode()).hexdigest()
    (out / "sft_manifest.json").write_text(json.dumps({
        "config": vars(args), "dataset_stats": ds.stats, "data_sha256": payload,
        "split": split, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False))
    print(f"saved adapter to {out}", flush=True)


if __name__ == "__main__":
    main()
