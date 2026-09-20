#!/usr/bin/env python3
"""Preserve the supplied paper tables and compare them to explicitly named saved metrics."""
from pathlib import Path
import csv
import json
from writer_handoff_catalog import PACKAGE as P

D=P/'table_review';D.mkdir(exist_ok=True)
def read(n,historical=False):
    return list(csv.DictReader((P/('historical/tables' if historical else 'tables')/n).open()))
def write(name,rows):
    with (D/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

actors={'qwen_base_sft':('Qwen3-14B-Base','Skywork-Qwen3-8B'),
        'qwen_instruct_direct':('Qwen3-14B-Instruct','Skywork-Qwen3-8B'),
        'llama_base_sft':('Llama-3.1-8B-Base','Skywork-Llama-3.1-8B-v0.2'),
        'llama_instruct_direct':('Llama-3.1-8B-Instruct','Skywork-Llama-3.1-8B-v0.2')}
submitted={
 'qwen_base_sft':[('base',-2.537,6.65,38.07,10.76),('sft-init',3.749,7.00,58.52,3.63),('grpo',10.870,37.39,59.47,27.25),('lam4',13.353,49.57,58.28,36.00)],
 'qwen_instruct_direct':[('instruct',11.426,63.4,88.03,71.37),('grpo',13.751,72.8,80.56,75.74),('lam4',14.433,73.7,82.00,72.03)],
 'llama_base_sft':[('base',-28.06,0,16.4,.02),('sft',-8.08,4.02,42.23,1.28),('grpo',4.55,9.73,37.65,3.26),('lam4',11.8,12.58,42.18,3.82)],
 'llama_instruct_direct':[('instruct',-.262,11.47,71.06,2.81),('grpo',19.608,21.61,65.43,7.86),('lam4',24.726,27.08,75.82,8.10)]}
ablation=[('base',-2.537,6.65,38.07,10.76),('sft-init',3.749,7,58.52,3.63),('grpo',10.870,37.39,59.47,27.25),('lam2',12.885,41.49,61.17,31.30),('lam4',13.353,49.57,58.28,36),('lam8',12.911,45.96,53.46,34.56),('randdir',10.73,32.62,51.70,26.73)]

tables={n:read(n) for n in ['alpaca_results.csv','reward256_per_seed.csv','ifeval_aggregates.csv','ifeval_per_seed.csv','arena_results.csv']}
for n in ['alpaca_results.csv','reward256_per_seed.csv','ifeval_aggregates.csv','ifeval_per_seed.csv']:
    for r in read(n,True):
        # This is the unadapted Instruct baseline, not the additional-SFT model.
        if r['model']=='base':
            tables[n].append(dict(r,family='llama_instruct_direct',model='instruct'))

def one(n,f,t,**conditions):
    candidates=[r for r in tables[n] if r['family']==f and r['model']==t and all(r[k]==v for k,v in conditions.items())]
    assert len(candidates)==1,(n,f,t,conditions,len(candidates))
    return candidates[0]

def resolved(f,t):
    a=one('alpaca_results.csv',f,t);w=one('reward256_per_seed.csv',f,t,seed='42')
    i=one('ifeval_aggregates.csv',f,t,cohort='base_single_seed' if (f,t)==('qwen_base_sft','base') else 'primary_5_seeds')
    ar=one('arena_results.csv',f,t,judge_variant='mixed_gpt4o_gpt41',subset='common')
    actor,rm=actors['qwen_base_sft' if f=='random_credit' else f]
    return dict(Actor=actor,RM=rm,method=t,RM_Reward_seed42=float(w['mean']),
                Alpaca_weighted_pct=float(a['weighted_win_rate_pct']),Alpaca_raw_strict_win_pct=float(a['raw_win_rate_pct']),
                IFEval_custom_four_metric_mean_pct=float(i['four_metric_mean_mean_pct']),
                IFEval_custom_four_metric_sd_pp=i['four_metric_mean_sd_pp'],IFEval_n_generation_seeds=i['n_seeds'],IFEval_generation_seeds=i['seeds'],
                IFEval_prompt_strict_pct=float(i['prompt_strict_mean_pct']),IFEval_prompt_loose_pct=float(i['prompt_loose_mean_pct']),
                IFEval_instruction_strict_pct=float(i['inst_strict_mean_pct']),IFEval_instruction_loose_pct=float(i['inst_loose_mean_pct']),
                Arena_mixed_common_weighted_pct=float(ar['raw_weighted_direct_pct']),Arena_n_prompts=ar['n_prompts'],
                baseline_note='Unadapted Instruct baseline reused from earlier evaluation campaign; adapter=None' if (f,t)==('llama_instruct_direct','instruct') else '',
                RM_source=w['source'],Alpaca_source=a['source'],Arena_source=ar['source'])

original_main=[];original_ablation=[];main=[];abl=[];diff=[]
def append(group,f,data,original,corrected):
    for t,r,a,i,h in data:
        actual_family='random_credit' if t=='randdir' else f
        row=resolved(actual_family,t);corrected.append(row)
        original.append(dict(Actor=row['Actor'],RM=row['RM'],method='vpo-lambda(4)-shuffle' if t=='randdir' else t,
                             RM_Reward=r,AlpacaEval=a,IFEval=i,Arena_Hard=h))
        for metric,given,col in [('RM-Reward',r,'RM_Reward_seed42'),('AlpacaEval',a,'Alpaca_weighted_pct'),('IF-Eval',i,'IFEval_custom_four_metric_mean_pct'),('Arena-Hard',h,'Arena_mixed_common_weighted_pct')]:
            expected=row[col];note=''
            if metric=='AlpacaEval':
                raw=row['Alpaca_raw_strict_win_pct'];note=f'Raw strict win={raw:.8f}; weighted preference={expected:.8f}'
            if metric=='IF-Eval':
                values=[x for x in tables['ifeval_per_seed.csv'] if x['family']==actual_family and x['model']==t]
                matches=[str(x['seed']) for x in values if abs(float(x['four_metric_mean_pct'])-given)<.006]
                note=f'Unified mean uses seeds {row["IFEval_generation_seeds"]}; matching single-seed values (rounding only): '+(','.join(matches) if matches else 'none')
            if metric=='RM-Reward' and t=='randdir':note='User 10.73 matches seed 46 (10.72855759); unified seed 42 is 11.05656242'
            if metric=='Arena-Hard':note=f'Mixed judge, campaign-common subset n={row["Arena_n_prompts"]}; separate random campaign has a different prompt set'
            diff.append(dict(table=group,family=actual_family,method=t,metric=metric,submitted=given,unified_saved_value=expected,
                             difference=expected-given,review_note=note))

for f,data in submitted.items():append('main',f,data,original_main,main)
append('ablation','qwen_base_sft',ablation,original_ablation,abl)
write('user_supplied_main.csv',original_main);write('user_supplied_ablation.csv',original_ablation)
write('reconciled_main.csv',main);write('reconciled_qwen_base_ablation.csv',abl);write('cell_by_cell_comparison.csv',diff)

def markdown_table(rows):
    lines=['| Actor | 方法 | RM seed42 | Alpaca weighted % | IFEval 四指标均值 % | Arena % |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["Actor"]} | {r["method"]} | {r["RM_Reward_seed42"]:.3f} | {r["Alpaca_weighted_pct"]:.2f} | {r["IFEval_custom_four_metric_mean_pct"]:.2f} | {r["Arena_mixed_common_weighted_pct"]:.2f} |')
    return '\n'.join(lines)

text='''# 用户结果表核对：先读这一页

两张用户原表已原样保留其数值在 `user_supplied_main.csv` / `user_supplied_ablation.csv`。它们混用了评测指标和生成 seed，不能直接标记为已核验的最终论文表。

`reconciled_main.csv` / `reconciled_qwen_base_ablation.csv` 是明确口径的核对版：RM-Reward 用 seed 42；Alpaca 用 weighted preference；IFEval 用四个官方指标的算术平均，再对固定生成 seeds 42–46 求均值；Arena 用 mixed GPT-4o/GPT-4.1、GPT-4o-mini reference 和每个 campaign 的共同有效题目。IFEval 的四指标平均是自定义汇总，完整 CSV 同时保留四个官方指标和 seed SD。

**例外与复用必须写入表注：**Qwen-Base 原始 base 的 IFEval 只有 seed 42（n=1），不能写成五 seed 均值。Llama-Instruct 原始 baseline 来自早期评测 campaign 中名为 `base` 的无 adapter 模型，已核对 manifest 是原始 Instruct 且 adapter=None；这不意味着使用额外 SFT。其 RL 行来自直接 RL campaign。

明确差异：

- Qwen GRPO/λ2/λ4/λ8 的原 Alpaca 数值大多来自 raw strict-win；Llama-Base、random 和部分初始化行用了 weighted preference。同一列应统一。
- IFEval 原表中 Qwen-SFT 58.52 匹配 seed 43、Qwen-GRPO 59.47 匹配 seed 45，而 Qwen λ2/4/8 匹配 seed 42；另外一些行是五 seed 均值。不能把这些写成同一评测口径。
- 当前 Qwen-Instruct λ4 的 RM seed42 原始逐题均值为 14.47116089，不是 14.433。
- Random 的 10.73 匹配 seed46；与其他行统一 seed42 时为 11.05656242。
- Qwen λ2 的 Arena mixed/common 495题结果为 31.38998682（31.39）；原表 31.30 未与该列其余行采用的保存口径对齐。
- 实际 ablation 为 `credit_source=random_direction`：用独立标准正态噪声代替 RM 梯度方向，再经同一 allocator。并非将真实 credit 权重打乱位置，建议写 Random-direction credit (λ=4)，不用 shuffle。

Arena random 来自单独 campaign（500题）；其他 Qwen-Base common 子集为495题。该列保持原协议并给出 n，不能据此宣称是相同题目上的配对显著性比较。未重新运行任何付费 judge。

下方结果是可追溯的统一口径版本，未按有利方向选择 seed。`cell_by_cell_comparison.csv` 逐格保留差异；源文件路径可通过 `../inventory/source_manifest.csv` 定位到包内证据。

## 主实验核对版

'''+markdown_table(main)+'\n\n## Qwen-Base 消融核对版\n\n'+markdown_table(abl)+'''

## 与已有草图的关系

此前 sketches/07、08 分别画的是 IFEval prompt strict 和 instruction strict，标签已明确；它们与本页“四指标平均”不是同一个指标。本页核对表与早先 tables/main_results.csv 都保留精确字段名，不能只按“IFEval”这一泛称交叉抄数。
'''
(D/'README_zh.md').write_text(text)
(P/'OPEN_FIRST_TABLE_REVIEW_zh.md').write_text('# 最新结果表核对\n\n请先读 [用户表与统一口径核对版](table_review/README_zh.md)。用户原表、统一口径表及逐格差异均已保留；两张原表尚不能视为同一指标/seed口径的最终论文表。\n')
print('Wrote 14 main rows, 7 ablation rows, supplied originals and 84 cell comparisons.')
