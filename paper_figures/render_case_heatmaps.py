#!/usr/bin/env python3
"""Token-level credit heatmap figure for the ten selected case studies.

Each case shows a window of the response around its highest-weight token;
every token is a box whose fill depth encodes its credit weight w (single-hue
sequential ramp, w=1 pale, w=4 = lambda-band top darkest). Good-response cases
(A>0) and bad-response cases (A<0, penalty) are stacked with a metadata header.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrow

import torch
from transformers import AutoTokenizer

TRAIN = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/rl-fp32-is-canonical-20260917/lam4/train')
BASE = '/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/models/Qwen3-14B-Base'
OUT = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/paper-2x2-rl-20260919')
PICK = [('good', 2, ' accountable'), ('good', 2, ' might'), ('good', 4, ' adventure'),
        ('good', 1, ' accurately'), ('good', 7, ' dawn'), ('good', 4, ' analogy'),
        ('bad', 1, 'bv'), ('bad', 3, ' water'), ('bad', 2, ' Elim'), ('bad', 5, 'ara')]
WINDOW_BEFORE, WINDOW_AFTER = 55, 30
SURF, INK, INK2 = '#fcfcfb', '#1f1f1e', '#5f5e56'
RAMP = LinearSegmentedColormap.from_list('credit', ['#f2f6fc', '#c9dcf4', '#8fb8e8', '#5b8fd4', '#2a78d6', '#1a4f96', '#123a70'])
VMIN, VMAX = 1.0, 4.0
NUM2 = {'good': '①②③④⑤⑥', 'bad': '⑦⑧⑨⑩'}


def luminance(rgb):
    r, g, b = rgb[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def render_case(ax, kind, meta, tokens_text, weights, top_idx):
    header = (f"{NUM2[kind][meta['case_no']-1] if kind=='good' else NUM2['bad'][meta['case_no']-7]} "
              f"{'GOOD response — credit' if kind=='good' else 'BAD response — penalty'}   "
              f"rollout {meta['rollout']}   A = {meta['A']:+.2f}   group rank {meta['rank']+1}/8   "
              f"reward {meta['reward']:.1f}   w* = {meta['w_max']:.1f} on “{meta['top_token'].strip() or '<eos>'}”")
    ax.text(0, 1.06, header, fontsize=8.6, color=INK, va='bottom', transform=ax.transAxes,
            family='monospace' if False else 'sans-serif', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # 逐 token 排版:等宽字体按字符数计宽,自动换行
    char_w = 0.0062
    line_h = 0.205
    max_chars = 158
    x, y, line = 0.0, 0.92, 0
    for j, (txt, wv) in enumerate(zip(tokens_text, weights)):
        clean = txt.replace('\n', '\\n')
        width = max(len(clean), 1) * char_w + 0.0035
        if x + width > 1.0 and line < 3:
            x, line = 0.0, line + 1
            y -= line_h
        if line > 3:
            ax.text(x, y, '…', fontsize=7.5, color=INK2, va='top')
            break
        t = min(max((wv - VMIN) / (VMAX - VMIN), 0), 1)
        fill = RAMP(t)
        ink = 'white' if luminance(to_rgb(fill)) < 0.55 else INK
        ax.text(x, y, clean if clean else '·', fontsize=7.3, color=ink, va='top', ha='left',
                bbox=dict(boxstyle='square,pad=0.12', facecolor=fill, edgecolor='none'), zorder=2)
        if j == top_idx:
            ax.add_patch(FancyArrow(x + width / 2, y + line_h * 0.78, 0, -line_h * 0.16,
                                    width=0.0018, head_width=0.014, head_length=0.02,
                                    color='#b3261e', zorder=5))
        x += width


def main():
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    c = json.loads((OUT / 'credit_case_candidates.json').read_text())
    cases = []
    for no, (kind, roll, token) in enumerate(PICK, 1):
        r = next(x for x in c[f'{kind}_pool'] if x['rollout'] == roll and x['top_token'] == token)
        n, i = r['rollout'], r['row']
        ids = json.loads((TRAIN / f'rollout-{n}-tokens.json').read_text())[i]
        w = torch.load(TRAIN / f'rollout-{n}-credit.pt', weights_only=True)['w'][i, :r['L']].float()
        top = int(w.argmax())
        lo, hi = max(0, top - WINDOW_BEFORE), min(r['L'], top + WINDOW_AFTER)
        texts = [('<eos>' if t == 151643 else tok.decode([t])) for t in ids[lo:hi]]
        cases.append((kind, {**r, 'case_no': no}, texts, w[lo:hi].tolist(), top - lo))

    heights = [1.25] * len(cases)
    fig, axes = plt.subplots(len(cases), 1, figsize=(11.5, len(cases) * 1.5), dpi=170,
                             gridspec_kw=dict(height_ratios=heights, hspace=0.42))
    fig.patch.set_facecolor(SURF)
    for ax, (kind, meta, texts, ws, ti) in zip(axes, cases):
        ax.set_facecolor(SURF)
        render_case(ax, kind, meta, texts, ws, ti)
    fig.suptitle('Where the VPO allocator places credit — token-level weights (λ=4 band, canonical Qwen-base run)',
                 fontsize=12.5, color=INK, x=0.012, ha='left', y=0.995)
    cax = fig.add_axes([0.78, 0.997, 0.16, 0.012])
    import numpy as np
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, VMAX), cmap=RAMP),
                      cax=cax, orientation='horizontal')
    cb.set_label('credit weight w  (1 = GRPO-uniform, 4 = band top)', fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    cb.outline.set_visible(False)
    fig.savefig(OUT / 'fig6_credit_case_heatmaps.png', facecolor=SURF, bbox_inches='tight')
    print('fig6_credit_case_heatmaps.png')


if __name__ == '__main__':
    main()
