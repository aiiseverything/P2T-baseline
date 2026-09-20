#!/usr/bin/env python3
"""Paper figures for the {Qwen, Llama} x {base, instruct} RL matrix.

Exports eight main GRPO/lambda=4 runs; Qwen-Base ablations are separate:
  fig1 reward curves, fig2 response-token length, fig3 KL-to-init,
  fig4 response entropy, fig5 credit ESS / weight spread (VPO arms).
One trailing-25 line + light raw scatter per arm; one panel per matrix cell.
"""
import csv
import json
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/paper-2x2-rl-20260919')
CELLS = [
    ('Qwen-base', '/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/rl-fp32-is-canonical-20260917'),
    ('Qwen-instruct', '/data/VPO-RM/runs/direct-rl-qwen-instruct-formal-20260918/qwen'),
    ('Llama-base', '/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/llama31-base-rl-20260919'),
    ('Llama-instruct', '/data/VPO-RM/runs/direct-rl-no-sft-20260918/llama'),
]
ARMS = ['grpo', 'lam4']
LABEL = {'grpo': 'GRPO', 'lam2': 'VPO λ=2', 'lam4': 'VPO λ=4', 'lam8': 'VPO λ=8', 'random_direction': 'Random direction, λ=4'}
COLOR = {'lam4': '#cc785c', 'grpo': '#447a9c', 'lam2': '#63957b', 'lam8': '#b68b39', 'random_direction': '#8d7c9f'}
TITLES = {'Qwen-base':'Qwen3-14B Base → SFT → RL', 'Qwen-instruct':'Qwen3-14B Instruct → RL',
          'Llama-base':'Llama-3.1-8B Base → SFT → RL', 'Llama-instruct':'Llama-3.1-8B Instruct → RL'}
FIELDS = ['reward_mean', 'raw_reward_mean', 'mean_response_tokens', 'kl_to_init', 'response_entropy',
          'credit_ess_ratio', 'credit_w_std', 'credit_w_max']
WINDOW = 25
SURF, INK, INK2, GRID = '#fcfcfb', '#1f1f1e', '#5f5e56', '#e8e7e0'


def trailing(values, w=WINDOW):
    return [sum(values[max(0, i - w + 1):i + 1]) / (i + 1 - max(0, i - w + 1)) for i in range(len(values))]


def load_all():
    data = {}
    for cell, root in CELLS:
        for arm in ARMS:
            p = Path(root) / arm / 'train/metrics.jsonl'
            if not p.exists():
                raise FileNotFoundError(p)
            config = json.loads((p.parent / 'profile_manifest.json').read_text())['config']
            assert bool(config['init_adapter']) == cell.endswith('-base'), (cell, config['init_adapter'])
            rows = [json.loads(l) for l in open(p) if l.strip()]
            rows = [r for r in rows if 'rollout' in r]
            rows.sort(key=lambda r: r['rollout'])
            data[(cell, arm)] = rows
    return data


def export_csv(data):
    with open(OUT / 'rl_metrics_2x2.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'arm', 'rollout'] + [k for k in FIELDS])
        for (cell, arm), rows in sorted(data.items()):
            for r in rows:
                w.writerow([cell, arm, r['rollout']] + [r.get(k) for k in FIELDS])


def panel(ax, data, cell, field, ylabel, scatter=True):
    for arm in ARMS:
        if (cell, arm) not in data:
            continue
        rows = data[(cell, arm)]
        x = [r['rollout'] for r in rows]
        y = [r[field] for r in rows if field in r]
        if len(y) != len(x):
            x = [r['rollout'] for r in rows if field in r]
        if scatter:
            ax.scatter(x, y, s=6, color=COLOR[arm], alpha=0.20, linewidths=0, zorder=2)
        ax.plot(x, trailing(y), color=COLOR[arm], linewidth=2, zorder=4, solid_capstyle='round')
    ax.set_title(TITLES[cell], fontsize=10.5, color=INK, loc='left', pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel('rollout', fontsize=8.5, color=INK2)


def figure(data, field, ylabel, title, name, note):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=160, sharex=True)
    fig.patch.set_facecolor(SURF)
    order = ['Qwen-base', 'Qwen-instruct', 'Llama-base', 'Llama-instruct']
    for ax, cell in zip(axes.flat, order):
        panel(ax, data, cell, field, ylabel)
        handles = [plt.Line2D([0], [0], color=COLOR[a], lw=2, label=LABEL[a])
                   for a in ARMS if (cell, a) in data]
        ax.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=False,
                  handlelength=1.4, labelcolor=INK2)
    fig.suptitle(title, fontsize=13, color=INK, x=0.02, ha='left')
    fig.text(0.02, 0.955, note, fontsize=8, color=INK2, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / name, facecolor=SURF, bbox_inches='tight')
    fig.savefig((OUT / name).with_suffix('.pdf'), facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42, 'ps.fonttype':42})
    data = load_all()
    export_csv(data)
    n = sum(len(v) for v in data.values())
    print(f'rl_metrics_2x2.csv: {len(data)} arms, {n} rollouts')
    (OUT / 'experiment_sources.json').write_text(json.dumps({
        'main_design':'Base → SFT → RL; Instruct → direct RL', 'main_arms':ARMS,
        'main_sources':dict(CELLS), 'ablation_family':'Qwen3-14B-Base + SFT only',
        'excluded_from_main':'llama31-rl-canonical-20260918-v3: additional SFT on Instruct',
        'reward_mean':'RM reward minus soft length penalty', 'raw_reward_mean':'Raw RM scalar',
    }, indent=2, ensure_ascii=False)+'\n')
    figure(data, 'reward_mean', 'training reward (length-adjusted)',
           'RL reward curves — {Qwen, Llama} × {base, instruct}',
           'fig1_reward_2x2.png',
           'lines: 25-rollout trailing mean · points: per-rollout mean · absolute scales differ across RM families (Skywork-Qwen3-8B vs Skywork-Llama-3.1-8B)')
    figure(data, 'raw_reward_mean', 'raw RM scalar',
           'Raw reward model scores during RL', 'fig1b_raw_reward_2x2.png',
           'lines: 25-rollout trailing mean · no length penalty in this scalar · compare within each RM family')
    figure(data, 'mean_response_tokens', 'mean response tokens',
           'Response length under RL — {Qwen, Llama} × {base, instruct}',
           'fig2_length_2x2.png',
           'lines: 25-rollout trailing mean · soft length reward active in all four suites (8/1024/2048 window, shared σ₀ per suite)')
    figure(data, 'kl_to_init', 'KL(policy ‖ init) — k3 estimate',
           'Drift from initialization — {Qwen, Llama} × {base, instruct}',
           'fig3_kl_2x2.png',
           'lines: 25-rollout trailing mean · reference: SFT init (base lines) or the posttrained checkpoint (instruct lines)')
    figure(data, 'response_entropy', 'mean token entropy (nats)',
           'Response entropy under RL — {Qwen, Llama} × {base, instruct}',
           'fig4_entropy_2x2.png',
           'lines: 25-rollout trailing mean · uniform sampling temperature 1.0')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=160)
    fig.patch.set_facecolor(SURF)
    for ax, field, yl in zip(axes, ('credit_ess_ratio', 'credit_w_std'),
                             ('credit ESS (1=uniform)', 'weight std w')):
        for cell, root in CELLS:
            if (cell, 'lam4') not in data:
                continue
            rows = data[(cell, 'lam4')]
            x = [r['rollout'] for r in rows if field in r]
            y = [r[field] for r in rows if field in r]
            shade = {'Qwen-base': '#2a78d6', 'Qwen-instruct': '#eb6834',
                     'Llama-base': '#1baf7a', 'Llama-instruct': '#eda100'}[cell]
            ax.plot(x, trailing(y), color=shade, linewidth=2)
        ax.set_title(f'VPO λ=4: {yl}', fontsize=10.5, color=INK, loc='left', pad=6)
        ax.set_ylabel(yl, fontsize=8.5, color=INK2)
        ax.set_xlabel('rollout', fontsize=8.5, color=INK2)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=cell) for cell, c in
               {'Qwen-base': '#2a78d6', 'Qwen-instruct': '#eb6834',
                'Llama-base': '#1baf7a', 'Llama-instruct': '#eda100'}.items()]
    axes[1].legend(handles=handles, loc='center right', fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.suptitle('Allocator behaviour (VPO λ=4) across the 2×2 matrix', fontsize=13, color=INK, x=0.02, ha='left')
    fig.text(0.02, 0.93, 'lines: 25-rollout trailing mean · ESS 1.0 = GRPO-equivalent uniform weights', fontsize=8, color=INK2, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / 'fig5_allocator_2x2.png', facecolor=SURF, bbox_inches='tight')
    fig.savefig(OUT / 'fig5_allocator_2x2.pdf', facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print('fig5_allocator_2x2.png')
    ablation = {}
    for arm in ['grpo','lam2','lam4','lam8','random_direction']:
        root = Path(CELLS[0][1]) if arm != 'random_direction' else OUT.parent/'rl-ablation-random-credit-20260919'
        ablation[arm] = [r for r in map(json.loads,(root/arm/'train/metrics.jsonl').read_text().splitlines()) if 'rollout' in r]
    with (OUT/'rl_metrics_qwen_base_ablation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','rollout']+FIELDS)
        for arm,rows in ablation.items():
            for r in rows:writer.writerow([arm,r['rollout']]+[r.get(k) for k in FIELDS])
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),dpi=160)
    for ax,field,title in zip(axes,['reward_mean','mean_response_tokens','credit_ess_ratio'],['Length-adjusted reward','Response tokens','Credit ESS ratio']):
        for arm,rows in ablation.items():
            values=[(r['rollout'],r[field]) for r in rows if field in r]
            if values:ax.plot([v[0] for v in values],trailing([v[1] for v in values]),color=COLOR[arm],label=LABEL[arm],lw=1.8)
        ax.set(title=title,xlabel='Rollout');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    handles=[plt.Line2D([0],[0],color=COLOR[a],label=LABEL[a],lw=2) for a in ablation]
    fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('Qwen3-14B Base + SFT: λ and random-direction ablations',x=.02,ha='left',fontsize=12)
    fig.tight_layout(rect=[0,.10,1,.95])
    for ext in ['png','pdf']:fig.savefig(OUT/f'fig7_qwen_base_ablations.{ext}',bbox_inches='tight',facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    main()
