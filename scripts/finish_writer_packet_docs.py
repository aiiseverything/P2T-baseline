#!/usr/bin/env python3
"""Arrange finished tables and add a short, explicit writer reading guide."""
from pathlib import Path
import csv
import json
import shutil
from collections import Counter
from writer_handoff_catalog import ROOT, OUT, PACKAGE as P, FAMILIES, paper_role


def rows(name):return list(csv.DictReader((P/'tables'/name).open()))
def write(path,values):
    if not values:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


# Keep historical tables discoverable without making them look like main evidence.
hist=P/'historical/tables';hist.mkdir(parents=True,exist_ok=True)
for path in list((P/'tables').glob('*.csv')):
    with path.open() as f:
        r=csv.DictReader(f);fields=r.fieldnames
        if 'family' not in fields:continue
        main=[];old=[]
        for row in r:
            (old if row['family'] in ['llama_instruct_sft','llama_instruct'] else main).append(row)
    if old:
        write(hist/path.name,old);write(path,main)
for path in (P/'readable_responses').glob('llama_instruct_sft__*'):
    dst=P/'historical/readable_responses';dst.mkdir(exist_ok=True);shutil.move(path,dst/path.name)

runs=rows('training_runs.csv');alp=rows('alpaca_results.csv');ife=rows('ifeval_aggregates.csv');rew=rows('reward256_per_seed.csv');arena=rows('arena_results.csv')
out=[]
for run in runs:
    family=run['family'];tag='randdir' if family=='random_credit' else run['arm']
    a=next(r for r in alp if r['family']==family and r['model']==tag)
    i=next(r for r in ife if r['family']==family and r['model']==tag and r['cohort']=='primary_5_seeds')
    w=next(r for r in rew if r['family']==family and r['model']==tag and r['seed']=='42')
    ar=next(r for r in arena if r['family']==family and r['model']==tag and r['judge_variant']=='mixed_gpt4o_gpt41' and r['subset']=='common')
    row=dict(family=family,arm=run['arm'],role=paper_role(family,run['arm']),
             alpaca_weighted_pct=a['weighted_win_rate_pct'],ifeval_prompt_strict_pct=i['prompt_strict_mean_pct'],
             ifeval_prompt_strict_sd_pp=i['prompt_strict_sd_pp'],ifeval_inst_strict_pct=i['inst_strict_mean_pct'],
             ifeval_prompt_loose_pct=i['prompt_loose_mean_pct'],ifeval_inst_loose_pct=i['inst_loose_mean_pct'],
             reward256_mean=w['mean'],arena_mixed_common_weighted_pct=ar['raw_weighted_direct_pct'],arena_n=ar['n_prompts'],
             alpaca_source=a['source'],reward_source=w['source'],arena_source=ar['source'])
    out.append(row)
write(P/'tables/main_results.csv',[r for r in out if r['role']=='Main 2x2'])
write(P/'tables/qwen_base_ablation_results.csv',[r for r in out if r['family'] in ['qwen_base_sft','random_credit']])

code=P/'analysis_code';code.mkdir(exist_ok=True)
shutil.copy2(ROOT/'scripts/plot_writer_packet.py',code/'plot_writer_packet.py')
(code/'find_source.py').write_text('''#!/usr/bin/env python3
"""python analysis_code/find_source.py ORIGINAL_ABSOLUTE_PATH"""
import csv,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
for row in csv.DictReader((p/'inventory/source_manifest.csv').open()):
    if row['source']==sys.argv[1]:
        print(p/row['destination']);break
else:raise SystemExit('Source not in packet (e.g. excluded weights or historical token dumps).')
''')
(code/'requirements.txt').write_text('numpy\npandas\nmatplotlib\n')

src=ROOT/'runs/paper-2x2-rl-20260919';fig=P/'paper_figures';fig.mkdir(exist_ok=True)
for path in src.iterdir():
    if path.is_file() and (path.name.startswith(('fig','rl_metrics','experiment_sources','scope_correction'))):shutil.copy2(path,fig/path.name)
shutil.copy2(OUT/'audit/analysis_validation.json',P/'inventory/analysis_validation.json')

TABLE_DESCRIPTIONS={
 'main_results.csv':('8 main runs','Start here: four model/initialization cells, GRPO versus lambda=4. All reported metrics explicitly named.'),
 'qwen_base_ablation_results.csv':('5 Qwen-Base arms','GRPO, lambda=2/4/8 and random direction. GRPO/lambda=4 are reused controls, not new runs.'),
 'training_runs.csv':('11 paper runs','Initialization, seed, source, retained counts, last-25 descriptive means and measured GPU-hours.'),
 'training_metrics.csv':('2750 rollouts','Original saved per-rollout metrics; reward_mean includes length penalties; raw_reward_mean does not.'),
 'training_responses.csv':('175808 retained responses','Token counts, raw reward, length penalties, normalized advantages and prompt/source join keys.'),
 'credit_response_statistics.csv':('one response with a saved allocator tensor','Weight range/mean/std and effective sample size; derived from saved float16 dumps.'),
 'credit_histograms.csv':('run × rollout × weight bin','Full saved aggregate weight histograms, not a sample of hand-picked tokens.'),
 'probability_tensor_samples.csv':('77 fixed rollout samples','Tensor names, shapes and dtypes; raw tensors are included for fixed rollouts 1,2,50,100,150,200,250.'),
 'alpaca_results.csv':('model evaluation','Project GPT-4.1 preference evaluation. weighted WR is mean preference; raw WR is fraction preference > 0.5.'),
 'alpaca_per_prompt.csv':('retained judged prompt','Preference and prompt SHA. Historical Qwen Base/SFT annotations were not retained; their summaries/generations remain available.'),
 'reward256_per_prompt.csv':('prompt × model × generation seed','Raw held-out RM score and response token count. No length/KL penalties in score.'),
 'reward256_per_seed.csv':('model × generation seed','256-prompt mean and saved 95% interval. Random-only additional seeds remain separate.'),
 'ifeval_per_seed.csv':('model × generation seed × cohort','Four official prompt/instruction strict/loose metrics, plus separately named custom four-metric average.'),
 'ifeval_aggregates.csv':('model × cohort','Five primary generation seeds (42–46), one base seed, or ten additional Qwen seeds (47–56), kept separate.'),
 'ifeval_per_constraint.csv':('constraint × evaluation','Accuracy and count by instruction/constraint type.'),
 'arena_results.csv':('model × judge/reference/subset','Pure GPT-4o and mixed fallback protocols, own/common valid subsets, and older GPT-4.1/o3-mini reference, never pooled.'),
 'sft_training_metrics.csv':('logged optimizer step','Sparse SFT training logs for Qwen Base and Llama Base.'),
 'sft_probe_lengths.csv':('probe response','25 train and 25 test prompts per condition. Qwen control = old SFT EOS variant; Llama pretrained = actual Base.'),
}
lines=['# Data dictionary and interpretation','', '| Table | Unit | Meaning |','|---|---|---|']
for n,(grain,desc) in TABLE_DESCRIPTIONS.items():lines.append(f'| [{n}](tables/{n}) | {grain} | {desc} |')
lines+=['','## Join keys and units','',
 '- `family, arm, rollout, response_index` identifies a training response. `group_index` indexes the retained prompt group; group size is 8. `prompt_sha256` resolves in `training_prompts.json`.',
 '- `token_source` / `source` are the original absolute provenance paths. Use `inventory/source_manifest.csv` or `python analysis_code/find_source.py ORIGINAL_PATH` to find the copied file. Original absolute paths need not exist on the writer’s machine.',
 '- Original token ID arrays and reward arrays are aligned by row. `credit.pt` contains padded `w`, `d` and per-response `tau`; slice each row to the corresponding token-ID length. Padding is not evidence. Tokenizer JSON files are included; no model weights are required to decode IDs.',
 '- A token’s policy-gradient coefficient is proportional to response advantage `A` times positive allocator weight `w`. Darker case-study color encodes `w`, not a separate ground-truth token reward. Negative `A` makes larger `w` a larger penalty. The paper case figure uses a shared nonlinear display mapping; the original weights are retained.',
 '- `reward_mean` is shaped training reward (raw RM score minus soft length penalties); `raw_reward_mean` is raw RM reward. `length_penalty_mean` and the per-response penalties let the writer separate the two. KL is reported separately.',
 '- `ess_ratio = (sum(w))² / (n × sum(w²))`; 1 means uniform weights. Credit `.pt` dumps use float16, so budget equality is approximate in the dump. `tau`, `d`, and the allocator equations are documented in the frozen source.',
 '- Fields ending in `_pct` are percentages (0–100); `_sd_pp` is sample standard deviation in percentage points. Raw RM reward is in the corresponding RM’s own scalar units. Qwen and Llama RM scales must not be pooled.',
 '- `optimizer_steps` in raw metrics is per rollout; `training_runs.csv` sums it. `elapsed_sec` is rollout wall time; `gpu_hours` is the saved cumulative allocated GPU-time metric. Retained counts can differ because degenerate prompt groups are filtered.',
 '', '## Limits for paper claims','',
 'All eleven paper RL runs use training seed 42. Evaluation generation seeds measure sampling variability of fixed trained policies, not independent training replications. Smoothing is descriptive (trailing 25); its variation is not a confidence interval.',
 'Alpaca is the project GPT-4.1 weighted-preference protocol, not official LC-AlpacaEval. IFEval’s arithmetic four-metric mean is a custom summary, not a fifth official metric. The compact main table uses prompt-level strict accuracy and retains the other official metrics.',
 'Arena scores require matching judge, reference and subset. Main sketches use mixed GPT-4o/GPT-4.1 fallback and GPT-4o-mini reference, with a common valid subset within each campaign. Campaigns can have different common sets. Controlled scores are fitted jointly to the campaign candidate set. Saved 90% intervals resample expanded order/decisive game rows, not prompt clusters. Arena values here are transcribed from saved scoring outputs, not newly recomputed judge verdicts.',
 'The random arm has its own Arena campaign; comparisons across campaigns should use the same question intersection and a joint controlled fit before drawing statistical conclusions. Its standalone headline is not a matched paired comparison.',
 'Qwen Base/SFT old Alpaca individual annotations are missing, so those initialization baselines cannot be independently re-scored from saved preference labels. No evidence was invented to fill this gap.',
 'The SFT loss traces are sparsely logged. Llama SFT completion records report 157 optimizer updates. The frozen Qwen SFT loop steps only on full accumulation batches: with 1250 microbatches and accumulation 8, 156 actual updates are inferred from code; its nominal target was 157. Do not claim identical actual update counts without this qualification.',
 'Historical Llama-Instruct + additional SFT and p9/p10/p11 runs use different experimental histories. They are catalogued for completeness, not pooled into the confirmed paper design. Frozen per-experiment sources take precedence over current workspace code.',
 '', '## Rebuild sketches offline','',
 'Install the three packages in `analysis_code/requirements.txt`, then run from the extracted packet:',
 '```sh\npython analysis_code/plot_writer_packet.py --packet .\n```',
 'The new sketches use only CSVs/inventory; no GPU, model weights or API calls. Original training/evaluation scripts are provenance, not a promise of weight-free training reproduction.', '']
(P/'DATA_DICTIONARY.md').write_text('\n'.join(lines))

summary=json.loads((P/'inventory/selection_summary.json').read_text())
sections={'01_rl_metrics_and_protocol':'训练曲线、配置与协议','02_rl_raw_rollouts':'逐回答 token、prompt、reward','03_token_credit':'全部论文相关 token credit','04_probability_audit_samples':'固定步数概率张量样本','05_benchmark_evidence':'Benchmark 回答、分数与裁判记录','06_sft_evidence':'Base 模型 SFT 与探针','07_existing_paper_figures':'论文图、CSV 与 case study','08_historical_and_diagnostic':'历史实验与诊断','09_code_and_method':'代码、方法与 benchmark 上游资产','10_input_data_and_tokenizers':'输入数据与 tokenizer'}
guide=['# VPO 论文写手资料包','', '**论文设置已按用户确认：Base → SFT → RL；Instruct → 直接 RL。只有 Qwen-Base 做 λ=2/4/8 和 random 消融。**','',
 '建议阅读顺序：','',
 '1. [主实验结果表](tables/main_results.csv)：8 条主实验，四个模型条件 × GRPO/λ4。',
 '2. [Qwen-Base 消融表](tables/qwen_base_ablation_results.csv)：GRPO、λ2、λ4、λ8、random；共 11 条不同的论文相关 RL 运行。',
 '3. [分类草图画廊](sketches/index.html)及 [修正后的训练主图](paper_figures/fig1_reward_2x2.png)。所有分类草图同时提供 PNG 和矢量 PDF。',
 '4. [数据字典与口径](DATA_DICTIONARY.md)。需要重选 case 或进一步统计时，再查原始 evidence。',
 '', '## 范围和重要数据','',
 '11 条训练共 2750 个 rollout，175,808 条保留回答；逐回答 token、prompt、reward 和 1750 份 VPO/random credit 张量全部保留。概率张量固定抽样 77 份。历史 Llama-Instruct+SFT 的 4 条运行仅在 historical/和历史 evidence 中登记，不属于论文主实验或 λ 消融。',
 '训练数据和评测结果是核心；原始回答/逐题判分支撑错误分析；token credit 支撑案例图；配置、冻结代码和数据指纹支撑方法说明。模型、LoRA 权重、优化器状态、运行缓存不在包内。',
 '', '| 源数据类别 | GB |','|---|---:|']
for k in sorted(summary['sections']):guide.append(f'| {sections[k]} | {summary["sections"][k]["bytes"]/1e9:.3f} |')
guide += ['',f'筛选原始证据合计 {summary["selected_bytes"]/1e9:.3f} GB；最终包大小另见包外 `packet_receipt.json`，包含汇总表、可读回答和草图。GB 均按 10⁹ bytes。',
 '', '## 可以直接读出的结果','',
 '下表为 GRPO → VPO λ4；仅作描述，不能视为显著性结论。IFEval 是五个生成 seed 的 prompt strict 均值；Arena 是各 campaign 的共同有效子集和 mixed judge 协议。',
 '', '| 模型条件 | Alpaca 加权偏好 % | IFEval prompt strict % | Reward256 | Arena % |','|---|---:|---:|---:|---:|']
for family in ['qwen_base_sft','qwen_instruct_direct','llama_base_sft','llama_instruct_direct']:
    rr={r['arm']:r for r in out if r['family']==family};vals=[]
    for col in ['alpaca_weighted_pct','ifeval_prompt_strict_pct','reward256_mean','arena_mixed_common_weighted_pct']:
        vals.append(f'{float(rr["grpo"][col]):.2f} → {float(rr["lam4"][col]):.2f}')
    guide.append('| '+FAMILIES[family]['label']+' | '+' | '.join(vals)+' |')
guide += ['', 'Alpaca 和 held-out RM 在这四组中都提高，但 Qwen-Base 的 IFEval prompt strict、Qwen-Instruct 的 IFEval prompt strict/Arena 没有同步提高。论文应呈现这些差异，不写成全指标一致胜出。',
 '', '## 追溯方式','',
 '- `inventory/all_experiment_locations.csv`：两个项目根目录下所有 runs/analysis 一级位置及其体积（包含历史、诊断和独立图文件，不把每个文件夹都冒充独立实验）。',
 '- `inventory/all_project_files.csv.gz`：170,767 个原始文件的完整存储清点快照。新增交付文件不计入原项目体积。',
 '- `inventory/source_manifest.csv`：每个复制源文件、包内位置、字节数、SHA256 和保留理由。',
 '- `readable_responses/`：论文 11 条运行的 UTF-8 JSONL gzip 可读回答，保留特殊 token 文本和 response ID；`training_prompts.json` 用 prompt SHA 连接。',
 '- `historical/`：历史 Llama-Instruct+SFT 的派生表和可读回答；原始历史评测等位于 evidence/08_historical_and_diagnostic。',
 '- `paper_figures/`：修正后的主图、Qwen 消融图和 token case 图。Case 图的深色是 credit weight，正负更新方向由 response advantage 决定。',
 '', '存储清点约 305.5 GB，其中模型、LoRA、优化器约 282.7 GB。当前目录是写作/分析交付，不包含可重新加载的训练权重。',
 '', '完整数据格式、可复现范围及已知缺口见 DATA_DICTIONARY.md。', '']
(P/'README_zh.md').write_text('\n'.join(guide))
(P/'README.md').write_text('# VPO writer evidence packet\n\nStart with [the reading guide](README_zh.md), [main results](tables/main_results.csv), [Qwen-Base ablations](tables/qwen_base_ablation_results.csv), [sketch gallery](sketches/index.html), and [English data dictionary](DATA_DICTIONARY.md).\n\nConfirmed design: Base → SFT → RL; Instruct → direct RL. Eight main runs, three additional Qwen-Base ablation arms. Historical Llama-Instruct+SFT runs are kept separate. No model, adapter or optimizer weights are included.\n')
print('Reading guides, paper tables, historical separation and portable plotting code ready.')
