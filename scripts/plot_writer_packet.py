#!/usr/bin/env python3
"""Rebuild writer sketches from CSV tables; works offline with --packet PATH."""
import argparse
from pathlib import Path
import os
import json
import shutil
import html
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);args=parser.parse_args()
P=args.packet.resolve();G=P/'sketches';G.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'audit/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.dpi':150,'axes.titlelocation':'left'})
F={'qwen_base_sft':'Qwen Base + SFT','qwen_instruct_direct':'Qwen Instruct, direct RL',
   'llama_base_sft':'Llama Base + SFT','llama_instruct_direct':'Llama Instruct, direct RL'}
C={'grpo':'#447a9c','lam4':'#cc785c','lam2':'#63957b','lam8':'#b68b39','randdir':'#8d7c9f','random_direction':'#8d7c9f'}
CAP=[]

def read(n):return pd.read_csv(P/'tables'/n,low_memory=False)
def save(fig,name,caption):
    fig.tight_layout(rect=[0,0,1,.94])
    for ext in ['png','pdf']:fig.savefig(G/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig);CAP.append(dict(name=name,caption=caption));print(name,flush=True)
def axes4(title):
    fig,axes=plt.subplots(2,2,figsize=(10,6));fig.suptitle(title,x=.02,ha='left',fontsize=12)
    for ax,(f,label) in zip(axes.flat,F.items()):ax.set_title(label);ax.grid(axis='y',alpha=.15)
    return fig,axes
def main_rows(d):return d[d.family.isin(F)&d.arm.isin(['grpo','lam4'])]

# Storage and selection are distinct quantities, in decimal GB.
d=pd.read_csv(P/'inventory/storage_by_category.csv').groupby('category').logical_bytes.sum().sort_values()
fig,ax=plt.subplots(figsize=(10,6));ax.barh(d.index.str.replace('_',' '),d/1e9,color='#cc785c');ax.set(xlabel='Logical GB (all project files)',title='Storage inventory: most bytes are model/training state')
save(fig,'01_storage','Both approved roots; path-based logical sizes, not deduplicated physical allocation. Model/adapter/optimizer files are excluded from this packet.')
s=json.loads((P/'inventory/selection_summary.json').read_text())['sections']
fig,ax=plt.subplots(figsize=(10,5));ax.barh([k.replace('_',' ') for k in s],[v['bytes']/1e9 for v in s.values()],color='#cc785c');ax.set(xlabel='GB of selected source files',title='Writer packet: selected evidence by category')
save(fig,'02_packet_contents','Selected raw evidence only; derived tables, readable responses and sketches add a small amount. Exact final bytes are in packet_receipt.json outside the ZIP.')

t=read('training_metrics.csv');resp=read('training_responses.csv')
fig,axs=axes4('Raw rollout responses: token-length distributions')
for ax,f in zip(axs.flat,F):
    for arm in ['grpo','lam4']:
        r=resp[(resp.family==f)&(resp.arm==arm)]
        ax.hist(r.token_count,bins=np.arange(0,2113,64),density=True,histtype='step',color=C[arm],lw=1.6,label=f'{arm}, n={len(r):,}')
    ax.set(xlabel='Response tokens',ylabel='Density');ax.legend(fontsize=7,frameon=False)
save(fig,'03_response_distributions','All retained responses from the eight main runs, all 250 rollouts. Distribution plots are descriptive; no independent-training-seed uncertainty is implied.')

fig,axs=axes4('Token allocator: spread within each response')
credit=read('credit_response_statistics.csv')
for ax,f in zip(axs.flat,F):
    d=credit[(credit.family==f)&(credit.arm=='lam4')]
    q=d.groupby('rollout').ess_ratio.quantile([.1,.5,.9]).unstack()
    ax.fill_between(q.index,q[.1],q[.9],color=C['lam4'],alpha=.2);ax.plot(q.index,q[.5],color=C['lam4'])
    ax.set(xlabel='Rollout',ylabel='Credit ESS / token count',ylim=(0,1.02))
save(fig,'04_token_credit','VPO lambda=4; per-response median and 10–90% spread, computed from saved float16 weights. Shading is a response distribution, not a confidence interval.')

fig,axs=axes4('Saved probability diagnostics: rollout/HF log-probability mismatch')
for ax,f in zip(axs.flat,F):
    for arm in ['grpo','lam4']:
        d=t[(t.family==f)&(t.arm==arm)].sort_values('rollout')
        ax.plot(d.rollout,d.rollout_logp_abs_error_mean.rolling(25,min_periods=1).mean(),color=C[arm],label=arm)
    ax.set(xlabel='Rollout',ylabel='Mean absolute log-probability difference');ax.legend(frameon=False)
save(fig,'05_probability_diagnostics','Full saved aggregate diagnostics; raw probability tensors are sampled at fixed rollouts 1, 2, 50, 100, 150, 200, 250. Lines use a trailing 25-rollout average.')

def bars(d,value,title,name,caption,err=None,lo=None,hi=None):
    fig,axs=axes4(title)
    for ax,f in zip(axs.flat,F):
        v=d[(d.family==f)&d.model.isin(['grpo','lam4'])].set_index('model').reindex(['grpo','lam4'])
        x=np.arange(2);y=v[value].to_numpy(float)
        error=v[err].to_numpy(float) if err else np.array([y-v[lo].to_numpy(float),v[hi].to_numpy(float)-y]) if lo else None
        ax.bar(x,y,color=[C['grpo'],C['lam4']],width=.55,yerr=error,capsize=3)
        ax.set_xticks(x,['GRPO','VPO λ=4']);ax.set_ylabel(value.replace('_',' '))
        for i,z in enumerate(y):
            if np.isfinite(z):ax.annotate(f'{z:.2f}',(i,z),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.margins(y=.2)
    save(fig,name,caption)

alp=read('alpaca_results.csv')
bars(alp,'weighted_win_rate_pct','AlpacaEval: saved project judge preference','06_alpaca',
     '805 prompts per model; project GPT-4.1 weighted preference, not official length-controlled AlpacaEval. No training-seed error bars. Full lambda/random and initialization results remain in CSV.')
ife=read('ifeval_aggregates.csv');ife=ife[ife.cohort=='primary_5_seeds']
bars(ife,'prompt_strict_mean_pct','IFEval: prompt-level strict accuracy','07_ifeval_prompt',
     '541 prompts and 834 constraints per evaluation. Mean ± sample SD across generation seeds 42–46; trained policies are fixed. This is one of the four official metrics.',err='prompt_strict_sd_pp')
bars(ife,'inst_strict_mean_pct','IFEval: instruction-level strict accuracy','08_ifeval_instruction',
     'Mean ± sample SD across five generation seeds. All four official metrics and the separately labeled custom average are retained in CSV; extra Qwen seeds 47–56 are a separate cohort.',err='inst_strict_sd_pp')
rew=read('reward256_per_seed.csv');rew=rew[rew.seed==42]
bars(rew,'mean','Held-out reward256: raw reward model score','09_reward256',
     '256 fixed prompts, generation seed 42. Saved 95% intervals are within each model evaluation; different reward-model scales are not pooled.',lo='ci95_low',hi='ci95_high')
arena=read('arena_results.csv');arena=arena[(arena.judge_variant=='mixed_gpt4o_gpt41')&(arena.subset=='common')]
bars(arena,'raw_weighted_direct_pct','Arena: mixed-judge protocol, common valid subset','10_arena',
     'GPT-4o retries with GPT-4.1 fallback; GPT-4o-mini reference. Common valid prompts within each campaign, not necessarily identical across campaigns. Saved 90% intervals use expanded game rows, not prompt-cluster bootstrap.',lo='raw_ci90_low_pct',hi='raw_ci90_high_pct')

fig,axs=plt.subplots(1,3,figsize=(12,3.6));fig.suptitle('Qwen-Base only: evaluation of λ and random-direction ablations',x=.02,ha='left',fontsize=12)
for ax,d,col,title in zip(axs,[alp,ife,rew],['weighted_win_rate_pct','prompt_strict_mean_pct','mean'],['Alpaca preference (%)','IFEval prompt strict (%)','Reward256 score']):
    d=d[d.family.isin(['qwen_base_sft','random_credit'])].set_index('model').reindex(['grpo','lam2','lam4','lam8','randdir'])
    ax.bar(np.arange(5),d[col],color=[C[x] for x in d.index]);ax.set_xticks(np.arange(5),['GRPO','λ2','λ4','λ8','Random'],rotation=20);ax.set_title(title)
save(fig,'11_qwen_ablation_eval','One-factor random-direction control shares Qwen-Base initialization and the lambda=4 band. IFEval uses five generation seeds; reward256 shows seed 42 for every arm. Random-only extra reward seeds are separately retained.')

fig,axs=plt.subplots(2,2,figsize=(10,6));fig.suptitle('SFT initialization evidence for Base actors',x=.02,ha='left',fontsize=12)
st=read('sft_training_metrics.csv');pr=read('sft_probe_lengths.csv')
for j,f in enumerate(['qwen_base','llama_base']):
    d=st[st.family==f];axs[0,j].plot(d.step,d.loss,marker='o',color='#cc785c');axs[0,j].set(title=f.replace('_',' ').title(),xlabel='Logged optimizer step',ylabel='Saved SFT loss')
    d=pr[pr.family==f].groupby(['mode','split']).tokens.mean();axs[1,j].bar(np.arange(len(d)),d,color='#cc785c');axs[1,j].set_xticks(np.arange(len(d)),[' / '.join(x) for x in d.index],rotation=20);axs[1,j].set_ylabel('Probe mean response tokens')
save(fig,'12_sft','25 train and 25 test prompts per condition. Qwen control is an older SFT/EOS variant, not the pretrained base. Llama pretrained is the actual base checkpoint. Loss curves contain sparse logged steps only; they do not imply identical optimizer-update counts.')

fig,ax=plt.subplots(figsize=(9,4.8));runs=read('training_runs.csv');d=runs[runs.family.isin(F)&runs.arm.isin(['grpo','lam4'])]
labels=[F[r.family]+' / '+r.arm for r in d.itertuples()]
ax.barh(labels,d.final_gpu_hours,color=[C[a] for a in d.arm]);ax.set(xlabel='Logged cumulative GPU-hours',title='Training cost from saved runtime metrics')
save(fig,'13_training_cost','Descriptive GPU-hour totals from each run. Hardware allocation and generated response lengths affect cost; this is not an isolated allocator microbenchmark.')

fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
table=ax.table(cellText=[['Qwen3-14B','Base → SFT → RL','Instruct → RL','λ2 / λ4 / λ8 + random'],['Llama-3.1-8B','Base → SFT → RL','Instruct → RL','No paper λ sweep']],colLabels=['Actor','Base protocol','Instruct protocol','Ablation scope'],loc='center',cellLoc='left',colWidths=[.18,.24,.22,.36]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
ax.set_title('Paper design: GRPO vs VPO λ=4 in all four main cells',fontsize=12,pad=12)
save(fig,'14_experiment_design','Eight main runs plus three additional Qwen-Base arms = eleven unique paper runs. Four historical Llama-Instruct+SFT runs are explicitly excluded from the paper matrix.')

inventory=pd.read_csv(P/'inventory/source_manifest.csv')
old=inventory[inventory.section=='08_historical_and_diagnostic'].copy()
old['suite']=old.source.map(lambda s:s.split('/runs/',1)[-1].split('/')[0] if '/runs/' in s else 'Other historical assets')
sizes=old.groupby('suite').bytes.sum().nlargest(12).sort_values()
fig,ax=plt.subplots(figsize=(11,5));ax.barh(sizes.index,sizes/1e6,color='#a39487');ax.set(xlabel='Selected MB',title='Historical and diagnostic evidence: largest source groups')
save(fig,'15_historical_inventory','Historical files remain available for provenance. These groups are not pooled into the current eleven paper runs; directory counts do not equal independent experiment counts.')

prompts=json.loads((P/'training_prompts.json').read_text())
fig,axs=axes4('Input prompts: retained prompt text length')
for ax,f in zip(axs.flat,F):
    keys=resp[resp.family==f].prompt_sha256.unique()
    lengths=[len(prompts[k]) for k in keys]
    ax.hist(lengths,bins=30,color='#cc785c',alpha=.8);ax.set(xlabel='Characters in retained prompt text',ylabel='Unique prompts')
save(fig,'16_input_prompts','Unique prompt hashes across retained responses in each family. Length is measured in characters of saved model-formatted prompt text, not tokens. Tokenizers and original input datasets are included for further inspection.')

# Locate raw evidence through source_manifest.csv; no original mount is needed.
(G/'captions.json').write_text(json.dumps(CAP,indent=2)+'\n')
page=['<!doctype html><meta charset="utf-8"><title>VPO writer sketches</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#302a25}img{width:100%;border:1px solid #eee}section{margin:40px 0}a{color:#ad5b40}</style><h1>VPO writer sketches</h1><p>Descriptive sketches from saved data. Read the caption before using a figure in the paper. Editable plotting code and CSV tables are included.</p>']
for r in CAP:page.append(f'<section><h2>{html.escape(r["name"])}</h2><a href="{r["name"]}.pdf">Vector PDF</a><img src="{r["name"]}.png"><p>{html.escape(r["caption"])}</p></section>')
(G/'index.html').write_text('\n'.join(page))
