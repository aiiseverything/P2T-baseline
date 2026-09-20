#!/usr/bin/env python3
"""Find paper case studies: tokens that the VPO allocator singles out.

Scans the canonical VPO lambda-4 rollout artifacts (w / d / tau per token,
per-response rewards and prompts). A response enters a candidate pool when it
is the best (or worst) of its 8-response group AND carries at least one content
token pinned near the lambda=4 band top (w >= 3.2), i.e. the allocator
concentrated credit on identifiable tokens of a clearly good (A>0) or clearly
bad (A<0) response.
"""
import json
import math
from pathlib import Path

import torch

TRAIN = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/rl-fp32-is-canonical-20260917/lam4/train')
SIGMA0 = 3.0323000897825447          # shared calibration of the canonical suite
STD_FLOOR = 0.5 * SIGMA0             # soft-mode advantage denominator floor
W_TOP = 3.2                          # 0.8 * lambda
MIN_LEN = 30


def advantage(rewards):
    out = []
    for g in range(0, len(rewards), 8):
        grp = rewards[g:g + 8]
        mean = sum(grp) / len(grp)
        var = sum((x - mean) ** 2 for x in grp) / len(grp)
        std = max(math.sqrt(var), STD_FLOOR)
        out.extend((x - mean) / std for x in grp)
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        '/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/models/Qwen3-14B-Base', local_files_only=True)

    good, bad = [], []
    for n in range(1, 251):
        cp, tp = TRAIN / f'rollout-{n}-credit.pt', TRAIN / f'rollout-{n}-tokens.json'
        rp, pp = TRAIN / f'rollout-{n}-rewards.json', TRAIN / f'rollout-{n}-prompts.json'
        if not (cp.exists() and tp.exists() and rp.exists() and pp.exists()):
            continue
        credit = torch.load(cp, weights_only=True)
        tokens = json.loads(tp.read_text())
        rewards_raw = json.loads(rp.read_text())
        rewards = [r['reward'] if isinstance(r, dict) else r for r in rewards_raw]
        prompts = json.loads(pp.read_text())
        if isinstance(prompts, dict):
            prompts = prompts.get('prompts', [])
        adv = advantage(rewards)
        w_all = credit['w'].float()
        d_all = credit['d'].float()
        for i in range(len(tokens)):
            ids = tokens[i]
            L = min(len(ids), w_all.shape[1])
            if L < MIN_LEN or i >= len(adv):
                continue
            w = w_all[i, :L]
            rank = sum(1 for j in range(i - i % 8, i - i % 8 + 8) if rewards[j] > rewards[i])
            a = adv[i]
            if not (a >= 1.2 and rank == 0) and not (a <= -1.2 and rank == 7):
                continue
            wmax = float(w.max())
            if wmax < W_TOP:
                continue
            top = int(w.argmax())
            prompt = prompts[i // 8] if i // 8 < len(prompts) else ''
            if not isinstance(prompt, str) or not prompt.isascii() or not (20 <= len(prompt) <= 400):
                continue
            lo = max(0, top - 12)
            ctx = tok.decode(ids[lo:top])
            tok_text = tok.decode([ids[top]]) if ids[top] != 151643 else '<eos>'
            after = tok.decode(ids[top + 1:top + 12])
            rec = {'rollout': n, 'row': i, 'prompt': prompt[:280], 'A': round(a, 2),
                   'reward': round(rewards[i], 2), 'raw_reward': round(rewards_raw[i].get('raw_reward', rewards[i]), 2) if isinstance(rewards_raw[i], dict) else round(rewards[i], 2),
                   'length': rewards_raw[i].get('length') if isinstance(rewards_raw[i], dict) else L,
                   'rank': rank, 'w_max': round(wmax, 2),
                   'w_mean': round(float(w[:L].mean()), 3), 'tau': round(float(credit['tau'][i]), 2),
                   'L': L, 'top_pos': top, 'top_token': tok_text,
                   'context_before': ctx[-160:], 'context_after': after[:120],
                   'd_at_top': round(float(d_all[i, top]), 4),
                   'response_head': tok.decode(ids[:40])[:160]}
            (good if a > 0 else bad).append(rec)
    good.sort(key=lambda r: -r['w_max'])
    bad.sort(key=lambda r: -r['w_max'])
    # 每 prompt 只留最佳一条,保证多样性
    def dedupe(pool):
        seen, out = set(), []
        for r in pool:
            k = r['prompt'][:80]
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out
    good, bad = dedupe(good), dedupe(bad)
    out = {'good_pool': good[:20], 'bad_pool': bad[:20],
           'n_good_total': len(good), 'n_bad_total': len(bad)}
    Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/paper-2x2-rl-20260919/credit_case_candidates.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"good pool: {len(good)} (total {len(out['good_pool']) and out['n_good_total']}), "
          f"bad pool: {len(bad)} (total {out['n_bad_total']})")
    for r in good[:8]:
        print('GOOD', f"r{r['rollout']} A={r['A']} w_max={r['w_max']} tok={r['top_token']!r} | {r['prompt'][:60]}")
    for r in bad[:8]:
        print('BAD ', f"r{r['rollout']} A={r['A']} w_max={r['w_max']} tok={r['top_token']!r} | {r['prompt'][:60]}")


if __name__ == '__main__':
    main()
