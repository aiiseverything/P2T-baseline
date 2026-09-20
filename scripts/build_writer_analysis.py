#!/usr/bin/env python3
"""Build inspectable tables directly from saved experiment records (no API/GPU)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from writer_handoff_catalog import ROOT, DATA, OUT, PACKAGE, FAMILIES, SFT, ROOTS

TABLES = PACKAGE / "tables"
VALIDATION = []


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def check(name, actual, expected, tolerance=0):
    okay = abs(actual - expected) <= tolerance if isinstance(actual, (int, float)) else actual == expected
    VALIDATION.append(dict(check=name, passed=bool(okay), actual=actual, expected=expected, tolerance=tolerance))
    if not okay:
        raise ValueError(f"{name}: {actual!r} != {expected!r}")


def source(path):
    return str(Path(path).resolve())


def rl_tables(decode=False, audit_credit=False):
    import numpy as np
    if decode:
        from tokenizers import Tokenizer
    if audit_credit:
        import torch
        torch.set_num_threads(1)
    series, registry, response_rows, hist_rows, credit_rows, probability_rows = [], [], [], [], [], []
    all_prompts, prompt_sequences = {}, {}
    decoded_dir = PACKAGE / "readable_responses"
    if decode:
        decoded_dir.mkdir(exist_ok=True)
    for family, spec in FAMILIES.items():
        for arm in spec["arms"]:
            train = spec["rl"] / arm / "train"
            config = read(train / "profile_manifest.json")["config"]
            metrics = [r for r in jsonl(train / "metrics.jsonl") if "rollout" in r]
            check(f"{family}/{arm}/rollout_ids", [r["rollout"] for r in metrics], list(range(1, 251)))
            tokenizer = Tokenizer.from_file(str(Path(config["model_name"]) / "tokenizer.json")) if decode else None
            decoded = gzip.open(decoded_dir / f"{family}__{arm}.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) if decode else None
            prompt_order, response_count, credit_count = [], 0, 0
            for metric in metrics:
                roll = int(metric["rollout"])
                base = dict(family=family, arm=arm, rollout=roll)
                series.append({**base, **metric, "source": source(train / "metrics.jsonl")})
                tokens = read(train / f"rollout-{roll}-tokens.json")
                rewards = read(train / f"rollout-{roll}-rewards.json")
                prompts = read(train / f"rollout-{roll}-prompts.json")
                if isinstance(prompts, dict):
                    prompts = prompts["prompts"]
                group_size = config["group_size"]
                check(f"{family}/{arm}/{roll}/responses", len(tokens), len(rewards))
                check(f"{family}/{arm}/{roll}/groups", len(tokens), len(prompts) * group_size)
                check(f"{family}/{arm}/{roll}/reward_mean", statistics.mean(r["reward"] for r in rewards), metric["reward_mean"], 2e-5)
                # Saved GPU reductions are float32, including the division for
                # retained batches that have fewer than 64 responses.
                length_tolerance = max(2e-5, 2 * abs(float(np.spacing(np.float32(metric["mean_response_tokens"])))))
                check(f"{family}/{arm}/{roll}/length_mean", statistics.mean(map(len, tokens)), metric["mean_response_tokens"], length_tolerance)
                hashes = []
                for prompt in prompts:
                    digest = hashlib.sha256(prompt.encode()).hexdigest()
                    all_prompts[digest] = prompt; hashes.append(digest)
                prompt_order.extend(hashes)
                credit_path = train / f"rollout-{roll}-credit.pt"
                credit = torch.load(credit_path, map_location="cpu", weights_only=True) if audit_credit and credit_path.exists() else None
                if credit is not None:
                    credit_count += 1
                    check(f"{family}/{arm}/{roll}/credit_rows", credit["w"].shape[0], len(tokens))
                for i, (ids, reward) in enumerate(zip(tokens, rewards)):
                    row = dict(**base, response_index=i, group_index=i // group_size, prompt_sha256=hashes[i // group_size],
                               token_count=len(ids), **reward, token_source=source(train / f"rollout-{roll}-tokens.json"))
                    response_rows.append(row)
                    if decoded is not None:
                        decoded.write(json.dumps(dict(family=family, arm=arm, rollout=roll, response_index=i,
                            prompt_sha256=hashes[i // group_size], response=tokenizer.decode(ids, skip_special_tokens=False),
                            reward=reward["reward"], raw_reward=reward.get("raw_reward"), advantage=reward.get("advantage"),
                            token_count=len(ids), finish_reason=reward.get("finish_reason")), ensure_ascii=False) + "\n")
                    if credit is not None:
                        w = credit["w"][i, :len(ids)].float().numpy()
                        direction = credit["d"][i, :len(ids)].float().numpy()
                        if len(w) != len(ids) or not np.isfinite(w).all() or not np.isfinite(direction).all():
                            raise ValueError(f"Invalid credit {family}/{arm}/{roll}/{i}")
                        lam = config["credit_lambda"]
                        if w.min() < 1 / lam - 2e-3 or w.max() > lam + 2e-3 or abs(float(w.mean()) - 1) > 2e-3:
                            raise ValueError(f"Credit budget/band failure {family}/{arm}/{roll}/{i}")
                        credit_rows.append(dict(**base, response_index=i, token_count=len(w),
                            w_min=float(w.min()), w_max=float(w.max()), w_mean=float(w.mean()), w_std=float(w.std()),
                            ess_ratio=float(w.sum() ** 2 / (len(w) * (w*w).sum())), tau=float(credit["tau"][i]),
                            source=source(credit_path)))
                response_count += len(tokens)
            if decoded is not None:
                decoded.close()
            if (train / "credit_stats.jsonl").exists():
                for r in jsonl(train / "credit_stats.jsonl"):
                    for i, n in enumerate(r["w_hist"]):
                        hist_rows.append(dict(family=family, arm=arm, rollout=r["rollout"],
                            bin_left=r["bin_edges"][i], bin_right=r["bin_edges"][i+1], token_count=n,
                            source=source(train / "credit_stats.jsonl")))
            if audit_credit:
                for roll in [1,2,50,100,150,200,250]:
                    p = train / f"rollout-{roll}-probabilities.pt"
                    if not p.exists():
                        continue
                    data = torch.load(p, map_location="cpu", weights_only=True)
                    probability_rows.append(dict(family=family, arm=arm, rollout=roll,
                        tensors=json.dumps({k: {"shape": list(v.shape), "dtype":str(v.dtype)} for k,v in data.items() if hasattr(v,"shape")}),
                        source=source(p)))
            order_sha = hashlib.sha256(json.dumps(prompt_order).encode()).hexdigest()
            prompt_sequences[(family, arm)] = order_sha
            last25 = metrics[-25:]
            registry.append(dict(family=family, label=spec["label"], role=spec["role"], arm=arm,
                model=config["model_name"], reward_model=config["reward_model_name"], init_adapter=config.get("init_adapter", ""),
                initialization=spec["init"], training_seed=config["seed"], n_rollouts=len(metrics), n_responses=response_count,
                planned_rollouts=config["rollout_iterations"], group_size=group_size,
                optimizer_steps=sum(m.get("optimizer_steps",0) for m in metrics), skipped_rollouts=sum(bool(m.get("skipped_rollout")) for m in metrics),
                input_prompt_groups=sum(m.get("input_prompt_groups",0) for m in metrics),
                kept_prompt_groups=sum(m.get("kept_prompt_groups",0) for m in metrics),
                skipped_groups=sum(m.get("skipped_groups",0) for m in metrics),
                resampled_groups=sum(m.get("resampled_groups",0) for m in metrics),
                credit_lambda=config["credit_lambda"], credit_source=config.get("credit_source", "rm_gradient" if arm != "grpo" else "uniform"),
                credit_tensor_files=credit_count if audit_credit else len(list(train.glob("rollout-*-credit.pt"))),
                sigma0=config["length_reward_sigma0"], mean_last25_reward=statistics.mean(m["reward_mean"] for m in last25),
                mean_last25_raw_reward=statistics.mean(m.get("raw_reward_mean",m["reward_mean"]) for m in last25),
                mean_last25_tokens=statistics.mean(m["mean_response_tokens"] for m in last25),
                final_gpu_hours=metrics[-1].get("gpu_hours"), prompt_order_sha256=order_sha,
                source=source(train), completion_record_exists=(train.parent / "completion.json").exists(),
                profile_summary_exists=(train / "profile_summary.json").exists(),
                checkpoint_250_manifest_exists=(train / "checkpoint-250/run_manifest.json").exists()))
            print(f"RL {family}/{arm}: {len(metrics)} rollouts, {response_count} responses", flush=True)
    for family, spec in FAMILIES.items():
        orders = {prompt_sequences[(family, arm)] for arm in spec["arms"]}
        # Degenerate-response filtering can change the retained prompts by arm.
        # Report the observation rather than assume that retained sets match.
        for row in registry:
            if row['family'] == family:
                row['same_retained_prompt_order_within_family'] = len(orders) == 1
    write_csv("training_runs.csv", registry)
    write_csv("training_metrics.csv", series)
    write_csv("training_responses.csv", response_rows)
    write_csv("credit_histograms.csv", hist_rows)
    write_csv("credit_response_statistics.csv", credit_rows)
    write_csv("probability_tensor_samples.csv", probability_rows)
    (PACKAGE / "training_prompts.json").write_text(json.dumps(all_prompts, ensure_ascii=False, indent=2) + "\n")
    return registry


def benchmark_tables():
    alpaca, preference_rows, rewards, reward_summary, ifeval, constraints = [], [], [], [], [], []
    for family, spec in FAMILIES.items():
        alpaca_dirs = spec["alpaca"] if isinstance(spec["alpaca"], dict) else {p.name:p for p in spec["alpaca"].iterdir() if p.is_dir()}
        for tag, directory in alpaca_dirs.items():
            p = directory / "results_judged.json"
            if not p.is_file():
                continue
            r = read(p)
            if "annotations" not in r:
                generations=directory/'generations_t1.0_n1.jsonl'
                check(f"{family}/{tag}/legacy_alpaca_generation_count",len(jsonl(generations)),r['n_judged'])
                alpaca.append(dict(family=family,model=tag,n=r['n_judged'],
                    weighted_win_rate_pct=r['weighted_win_rate']*100,raw_win_rate_pct=r['win_rate']*100,
                    judge=r.get('judge_model','gpt-4.1'),source=source(p),annotations='',
                    verification='Historical summary plus generations; individual annotations not retained'))
                continue
            annotations_path = Path(r["annotations"])
            annotations = read(annotations_path)["rows"]
            valid = [(k,v) for k,v in annotations.items() if v.get("preference") is not None]
            check(f"{family}/{tag}/alpaca_annotations", len(valid), r["n_judged"])
            weighted = statistics.mean(float(v["preference"]) for _,v in valid)
            # This project's saved raw WR uses strict p > .5, with no half-win.
            raw = statistics.mean(float(float(v["preference"]) > .5) for _,v in valid)
            check(f"{family}/{tag}/alpaca_weighted", weighted, r["weighted_win_rate"], 1e-12)
            check(f"{family}/{tag}/alpaca_raw", raw, r["win_rate"], 1e-12)
            alpaca.append(dict(family=family, model=tag, n=len(valid), weighted_win_rate_pct=weighted*100,
                raw_win_rate_pct=raw*100, judge=r.get("judge_model", "gpt-4.1"), source=source(p), annotations=source(annotations_path),
                verification='Recomputed from all retained per-prompt annotations'))
            for key,v in valid:
                preference_rows.append(dict(family=family,model=tag,prompt_sha256=hashlib.sha256(v["instruction"].encode()).hexdigest(),
                    annotation_key=key,preference=v["preference"],candidate_chars=v.get("chars"),source=source(annotations_path)))
        for p in sorted((spec["reward"] / "results").rglob("eval.jsonl")):
            rows = jsonl(p)
            if not rows:
                continue
            tag = str(rows[0]["run"])
            seed = 42
            for r in rows:
                rewards.append(dict(family=family,model=tag,seed=seed,**r,source=source(p)))
            check(f"{family}/{tag}/reward256_n", len(rows), 256)
            score = statistics.mean(float(r["score"]) for r in rows)
            saved = read(p.parent / "summary.json")[tag]
            saved = next(iter(next(iter(saved.values())).values()))
            check(f"{family}/{tag}/reward256_mean", score, saved["mean"], 1e-5)
            reward_summary.append(dict(family=family,model=tag,seed=seed,n=len(rows),mean=score,
                mean_response_tokens=statistics.mean(r["response_tokens"] for r in rows),
                ci95_low=saved["ci95"][0],ci95_high=saved["ci95"][1],source=source(p)))
        for p in sorted((spec["ifeval"] / "results").rglob("results_t1.0_n1.json")):
            append_ifeval(ifeval,constraints,family,p,"primary_5_seeds")
    # A single fresh-protocol base evaluation is separate from the five-seed table.
    append_ifeval(ifeval,constraints,"qwen_base_sft",ROOT / "runs/ifeval-final-canonical-20260917/results/base/results_t1.0_n1.json","base_single_seed")
    for p in sorted((DATA / "runs/qwen-instruct-ifeval10-20260919/results").rglob("results_t1.0_n1.json")):
        append_ifeval(ifeval,constraints,"qwen_instruct_direct",p,"additional_10_seeds")
    for p in sorted((DATA / "runs/randdir-reward256-seeds-20260919/results").rglob("eval.jsonl")):
        seed = int(p.parent.name.split("-")[-1]); rows=jsonl(p)
        check(f"random_credit/reward256/seed{seed}",len(rows),256)
        for r in rows:
            rewards.append(dict(family="random_credit",model="randdir",seed=seed,**r,source=source(p)))
        saved=read(p.parent / "summary.json"); saved=next(iter(next(iter(next(iter(saved.values())).values())).values()))
        avg=statistics.mean(float(r["score"]) for r in rows)
        check(f"random_credit/reward256/seed{seed}/mean",avg,saved["mean"],1e-5)
        reward_summary.append(dict(family="random_credit",model="randdir",seed=seed,n=len(rows),mean=avg,
            mean_response_tokens=statistics.mean(r["response_tokens"] for r in rows),ci95_low=saved["ci95"][0],ci95_high=saved["ci95"][1],source=source(p)))
    aggregates=[]
    grouped=defaultdict(list)
    for r in ifeval:
        grouped[(r['family'],r['model'],r['cohort'])].append(r)
    for (family,tag,cohort),rows in grouped.items():
        record=dict(family=family,model=tag,cohort=cohort,n_seeds=len(rows),seeds=','.join(str(r['seed']) for r in rows))
        for metric in ['prompt_strict','prompt_loose','inst_strict','inst_loose','four_metric_mean']:
            values=[r[metric+'_pct'] for r in rows]
            record[metric+'_mean_pct']=statistics.mean(values)
            record[metric+'_sd_pp']=statistics.stdev(values) if len(values)>1 else ''
        aggregates.append(record)
    write_csv("alpaca_results.csv",alpaca);write_csv("alpaca_per_prompt.csv",preference_rows)
    write_csv("reward256_per_prompt.csv",rewards);write_csv("reward256_per_seed.csv",reward_summary)
    write_csv("ifeval_per_seed.csv",ifeval);write_csv("ifeval_aggregates.csv",aggregates);write_csv("ifeval_per_constraint.csv",constraints)
    print(f"Benchmarks: {len(alpaca)} Alpaca models, {len(reward_summary)} reward evaluations, {len(ifeval)} IFEval evaluations",flush=True)


def append_ifeval(rows,constraints,family,path,cohort):
    r=read(path);details=r['details']
    check(f"{family}/{r['tag']}/{r['seed']}/ifeval_prompts",len(details),541)
    check(f"{family}/{r['tag']}/{r['seed']}/ifeval_instructions",sum(len(d['strict_list']) for d in details),834)
    values=dict(prompt_strict=statistics.mean(d['strict_all'] for d in details),
                prompt_loose=statistics.mean(d['loose_all'] for d in details),
                inst_strict=statistics.mean(v for d in details for v in d['strict_list']),
                inst_loose=statistics.mean(v for d in details for v in d['loose_list']))
    for k,v in values.items():check(f"{family}/{r['tag']}/{r['seed']}/{k}",v,r[k],1e-12)
    values['four_metric_mean']=statistics.mean(values.values())
    rows.append(dict(family=family,model=r['tag'],cohort=cohort,seed=r['seed'],n_prompts=len(details),n_instructions=834,
        mean_tokens=r.get('response_length_mean'),**{k+'_pct':v*100 for k,v in values.items()},source=source(path)))
    for k,v in r.get('per_constraint',{}).items():
        constraints.append(dict(family=family,model=r['tag'],cohort=cohort,seed=r['seed'],constraint=k,
            strict_pct=v['strict']*100,loose_pct=v['loose']*100,n=v['count'],source=source(path)))


def arena_tables():
    rows=[]
    # Current mixed-judge campaigns retain both candidate-specific and common subsets.
    for family,spec in FAMILIES.items():
        root=spec['arena']
        if root is None:continue
        for p in sorted((root/'scores').rglob('*.csv')):
            if p.name not in ['common_valid.csv','per_model_valid.csv','results.csv','common_subset_results.csv']:continue
            if p.parent.name not in ['pure_gpt4o','mixed_gpt4o_gpt41']:continue
            common=p.name in ['common_valid.csv','common_subset_results.csv']
            for r in csv.DictReader(p.open()):
                def number(*keys):
                    for k in keys:
                        if k in r and r[k]!='':return float(r[k])
                    return ''
                rows.append(dict(family=family,model=r['model'],judge_variant=p.parent.name,
                    reference='gpt-4o-mini-2024-07-18',subset='common' if common else 'per_model',
                    n_prompts=int(r['prompts']),raw_weighted_direct_pct=number('raw_wr_pct','raw_weighted_direct_pct'),
                    raw_ci90_low_pct=number('raw_ci90_low_pct'),raw_ci90_high_pct=number('raw_ci90_high_pct'),
                    controlled_pct=number('controlled_wr_pct','length_markdown_controlled_pct'),
                    controlled_ci90_low_pct=number('controlled_ci90_low_pct'),controlled_ci90_high_pct=number('controlled_ci90_high_pct'),
                    source=source(p)))
    p=ROOT/'runs/arena-hard-v2-canonical-20260917/scores/results.csv'
    for r in csv.DictReader(p.open()):
        rows.append(dict(family='qwen_base_sft',model=r['model'],judge_variant='gpt-4.1',reference='o3-mini-2025-01-31',
            subset='full_500',n_prompts=int(r['prompts']),raw_weighted_direct_pct=float(r['raw_weighted_direct_pct']),
            raw_ci90_low_pct=float(r['raw_ci90_low_pct']),raw_ci90_high_pct=float(r['raw_ci90_high_pct']),
            controlled_pct=float(r['length_markdown_controlled_pct']),
            controlled_ci90_low_pct=float(r['controlled_ci90_low_pct']),controlled_ci90_high_pct=float(r['controlled_ci90_high_pct']),source=source(p)))
    write_csv('arena_results.csv',rows)
    print(f"Arena: {len(rows)} protocol/subset/model rows (not pooled)",flush=True)


def sft_tables():
    training,probes=[] ,[]
    for family,(root,model) in SFT.items():
        path=model/'sft_metrics.jsonl'
        for r in jsonl(path):
            if 'loss' in r:training.append(dict(family=family,**r,source=source(path)))
        for p in sorted((root/'tvt').glob('*/generations_t1.0_n1.jsonl')):
            group=p.parent.name;mode,split=group.rsplit('-',1)
            records=jsonl(p)
            for i,r in enumerate(records):
                # Saved generation layouts vary; inspect explicit token counts or IDs.
                gens=r.get('generations',[r])
                for g in gens:
                    n=g.get('num_tokens',g.get('response_tokens',g.get('tokens')))
                    if isinstance(n,list):n=len(n)
                    if n is None and 'token_ids' in g:n=len(g['token_ids'])
                    if n is None:raise ValueError(f"Unknown SFT generation schema {p}: {list(g)}")
                    probes.append(dict(family=family,mode=mode,split=split,prompt_index=i,tokens=n,
                        finish_reason=g.get('finish_reason'),source=source(p)))
    write_csv('sft_training_metrics.csv',training);write_csv('sft_probe_lengths.csv',probes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decode',action='store_true')
    parser.add_argument('--audit-credit',action='store_true')
    parser.add_argument('--benchmarks-only',action='store_true')
    args=parser.parse_args()
    TABLES.mkdir(parents=True,exist_ok=True)
    if not args.benchmarks_only:rl_tables(args.decode,args.audit_credit)
    benchmark_tables();arena_tables();sft_tables()
    (OUT/'audit/analysis_validation.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        checks=VALIDATION,passed=sum(v['passed'] for v in VALIDATION),failed=sum(not v['passed'] for v in VALIDATION)),indent=2)+'\n')
    print(f"Verified {len(VALIDATION)} checks; tables saved under {TABLES}",flush=True)


if __name__=='__main__':main()
