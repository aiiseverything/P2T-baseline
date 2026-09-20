# 实验状态与结果快照：2026-09-17

统计时间：**2026-09-17 21:55 HKT**。本文从本地实验产物核对结果；模型、数据集和原始运行产物不随 GitHub 仓库发布，`runs/` 中仅归档选定的控制源码及说明。下面的结果路径供持有完整实验目录的读者复核。进行中的任务以其状态文件为准。

## 当前训练基线

主实验使用 **Qwen3-14B-Base + Skywork-Reward-V2-Qwen3-8B**。SFT 初始化为 `models/sft-native-eos-clean2k5e2`，四组 RL 共用这份初始化与 KL reference，最终模型来自：

```text
runs/rl-fp32-is-canonical-20260917/{grpo,lam2,lam4,lam8}/train/checkpoint-250
```

四组均完成 250 次 rollout，分别为 GRPO 和 VPO λ=2/4/8。共同配置包括 UltraFeedback prompts、学习率 `5e-5`、`beta=0.03`、temperature 1、生成上限 2048 token、每次 rollout 后一轮 policy update、训练 seed 42。VPO 三组的 stop/structural token 使用固定 credit；GRPO 不使用 VPO 的 token credit 重分配。完整参数与源文件哈希见 `runs/rl-fp32-is-canonical-20260917/experiment.json`，不可用旧的同名方法实验替代这组 checkpoint。

当前长度奖励为 soft 模式：少于 8 token 时线性扣分；8–1024 token 不扣长度分；1024–2048 token 线性增加长答扣分。短答、长答最大扣分分别为 `0.5σ₀` 和 `2σ₀`，四组共享校准尺度；`min_response_tokens=0`，允许自然结束。组内优势的标准差下限为 `0.5σ₀`。公式和边界处理见 [长度奖励说明](length-reward-soft-window.md)。该说明的最初实现状态以本文的实际训练状态为准。

## 已完成的评测

### AlpacaEval 2.0 数据集

每个模型 805 题，reference 为数据集中的 GPT-4 Turbo 回答，judge 为 **GPT-4.1**。采用官方 weighted annotator 的提示模板、单 token 判断和基于 logprob 的偏好加权，具体实现见 [`scripts/judge_alpaca.py`](../scripts/judge_alpaca.py)。这里的加权胜率不是 length-controlled win rate；judge 模型也不同于官方 leaderboard，不能直接把下表作为官方榜单分数。

| 模型 | 加权胜率 | 原始胜率 |
|---|---:|---:|
| Base | 6.65% | 6.71% |
| SFT-init | 7.00% | 6.83% |
| GRPO | 38.16% | 37.39% |
| VPO λ=2 | 42.35% | 41.49% |
| VPO λ=4 | 49.91% | 49.57% |
| VPO λ=8 | 46.48% | 45.96% |

结果来源均为对应目录下的 `results_judged.json`：

- Base：`runs/alpacaeval-evals/base/`
- SFT-init：`runs/alpacaeval-native-eos/sft-native-eos-clean2k5e2/`
- GRPO：`runs/alpacaeval-final-canonical-20260917/generations/grpo/`
- λ=2/4/8：`runs/alpacaeval-lam{2,4,8}-canonical-20260917/generations/lam{2,4,8}/`

四组当前 RL 的生成配置为 temperature 1、top-p 1、每题一个回答、seed 42、2048-token 上限、FP32 policy head。Base/SFT 是复用的既有回答，保留其原实验来源；不能将它们描述为与当前四组一同重新生成的实验。

### IFEval：25 次评测及 Base 参照

每次评测有 541 条 prompt、834 个指令检查。五个训练后模型各生成五次，生成 seeds 为 42–46；评分器及其语言检测 seed 固定为 42。temperature 1、top-p 1、2048-token 上限、FP32 policy head。

PS/PL 是 prompt-level strict/loose accuracy，IS/IL 是 instruction-level strict/loose accuracy，均为 IFEval 自带指标。**Total 是本项目自定义的 `(PS + PL + IS + IL) / 4`，不是 IFEval 官方第五个指标。** 下表前四列为五次均值，Total 为逐次计算后的均值 ± 样本标准差；单位均为百分比，标准差单位为百分点。

| 模型 | PS | PL | IS | IL | Total |
|---|---:|---:|---:|---:|---:|
| SFT-init | 51.42 | 56.08 | 62.64 | 66.79 | 59.23 ± 0.55 |
| GRPO | 47.62 | 60.26 | 60.84 | 70.89 | 59.90 ± 0.42 |
| VPO λ=2 | 50.17 | 59.82 | 62.35 | 70.43 | 60.69 ± 0.63 |
| VPO λ=4 | 44.44 | 58.52 | 57.48 | 70.02 | 57.62 ± 0.63 |
| VPO λ=8 | 37.41 | 53.90 | 51.99 | 66.07 | 52.34 ± 0.78 |

| 模型 | seed 42 Total | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---:|---:|---:|---:|---:|
| SFT-init | 59.93 | 58.52 | 59.42 | 59.42 | 58.87 |
| GRPO | 59.76 | 60.10 | 60.52 | 59.47 | 59.64 |
| VPO λ=2 | 61.17 | 60.58 | 59.86 | 60.39 | 61.45 |
| VPO λ=4 | 58.28 | 58.21 | 57.37 | 57.44 | 56.78 |
| VPO λ=8 | 53.46 | 51.56 | 52.57 | 52.47 | 51.64 |

来源：`runs/ifeval-five-seeds-canonical-20260917/{experiment.json,summary.json,model_summary.csv,results_25.csv}`。Base 另有相同当前生成协议下的一次 seed-42 结果，PS/PL/IS/IL 为 **29.02/33.83/42.21/47.24%**，自定义 Total **38.07%**；只有一次，不能报告五次标准差。来源：`runs/ifeval-final-canonical-20260917/results/base/results_t1.0_n1.json`。

### Arena-Hard v2：o3-mini reference

使用 500 条 `hard_prompt`，每题交换回答顺序判两次；六模型共 3000 个回答、6000 个有效判分。judge 为 GPT-4.1，reference 为官方发布的 `o3-mini-2025-01-31` 回答。生成配置为 temperature 1、top-p 1、seed 42、4096-token 上限、FP32 policy head。

下面报告长度及 Markdown 控制后的胜率：按固定上游算法进行 100 次 bootstrap 拟合，取中位数，并给出 90% 区间。它与前面的 Alpaca 加权胜率不是同一种统计量，也使用不同 reference，不能直接比较两张表的百分比。

| 模型 | 控制后胜率 | 90% 区间 |
|---|---:|---:|
| Base | 1.01% | 0.84–1.20% |
| SFT-init | 1.60% | 1.34–1.93% |
| GRPO | 3.02% | 2.60–3.47% |
| VPO λ=2 | 3.18% | 2.78–3.61% |
| VPO λ=4 | 3.33% | 2.85–3.79% |
| VPO λ=8 | 3.67% | 3.25–4.14% |

来源：`runs/arena-hard-v2-canonical-20260917/scores/results.json`；完整性检查见同目录实验根下的 `evaluation_complete.json`，验证 6000 个有效判分且无丢弃样本。该实验没有发布 Elo 分数。

### 固定 256 条评测 prompt 的 RM reward

六模型均在同一固定 256 条评测 prompt 上生成一个回答，temperature 1、top-p 1、seed 42、2048-token 上限、FP32 actor head，再用冻结的 Skywork RM 和 `canonical_chat_v1` 格式打分。这里是 **原始 RM scalar**，不含长度或 KL 惩罚。

| 模型 | 平均 reward | 平均回答 token |
|---|---:|---:|
| Base | -2.537 | 622.2 |
| SFT-init | 3.749 | 253.5 |
| GRPO | 10.870 | 545.9 |
| VPO λ=2 | 12.885 | 570.8 |
| VPO λ=4 | 13.353 | 507.0 |
| VPO λ=8 | 12.911 | 591.3 |

来源：`runs/reward256-canonical-20260917/{experiment.json,summary.json}`，原始汇总还包含 95% bootstrap 区间和截断计数。RM reward 增长本身不能证明通用能力提升；例如 λ=8 的 IFEval 明显低于 λ=2，应分 benchmark 解释结果。

## 最近修复与复现边界

- **RM 输入格式**：完整 user/assistant chat 按 RM 自身模板序列化；不再用 actor token 序列直接拼接 actor native EOS 代替 RM 的结束格式。只有 token ID 和文本字节跨度同时匹配的位置才映射 RM 梯度，无法匹配的位置保留单位 credit。
- **采样概率与精度**：保留 vLLM 实际 sampled-token logprob，在 HF old-policy 裁剪目标外使用 detached `exp(HF_old_logp - rollout_logp)` 修正，并记录误差。训练及相应评测使用 FP32 policy head；单独加载 LoRA 不会自动恢复这一配置。
- **长度及退化输出**：使用上述 soft length reward、组内标准差下限，以及无有效完成组的重采样/跳过规则。旧长度机制、旧 RM 格式下的实验不能混入当前 canonical 结果。
- **评测身份与断点恢复**：绑定模型、数据、源文件和生成参数的指纹；保持原始生成与判分记录，已有效的判分不重复请求。新的 Arena 网络恢复记录连接阶段、请求摘要和重试链，对不确定是否发送成功的错误停止自动重试。

详细工程核查见 [全项目复核与重跑记录](superpowers/plans/2026-09-17-full-recheck-relaunch.md)；更早修复见 [2026-09-16 审查说明](project-audit-fixes-2026-09-16.md)。数值门禁覆盖范围及保留的失败记录也在复核记录中；通过检查不等于对所有数值尾部作无误差保证。

## 进行中的任务与存储

**GPT-4o-mini reference Arena-Hard**：`runs/arena-hard-v2-gpt4omini-20260917/` 复用同一批 3000 个候选回答，仅更换为官方 `gpt-4o-mini-2024-07-18` reference 后重新判分。仍是 GPT-4.1 judge、两个回答顺序；已进入 32 并发正式判分，21:55 的只读快照约 **411/6000** 个有效判分。当前使用端口 **17891** 的代理，并做有界连接重试；评测尚未完成，本文不发布其最终胜率。以 `resilient_status.json`、逐游戏记录及最终完整性检查为准。

**Llama 8B + 8B 下载**：目标为 `/data/VPO-RM/models/Llama-3.1-8B-Instruct/` 和 `/data/VPO-RM/models/Skywork-Reward-Llama-3.1-8B-v0.2/`。截至本次快照，两边各四个最终 `.safetensors` 文件均已出现，后台仍在最终处理/校验；actor `DOWNLOAD_MANIFEST.json` 中 `verified=false`，RM 未发现完成校验标记。**尚不能标记为验证完成或已可用于正式实验。** 监督状态见 `/data/VPO-RM/.download-logs/background_status.json`；这套 Llama 权重尚未用于上面的任何结果。

GitHub 保存源代码、配置、测试和文档；共享项目目录保留现有模型、数据和实验产物。`/data/VPO-RM` 用于新增模型和存储维护记录，`/data` 是 NFS 挂载，rjob 的访问与迁移路径尚须单独验证。已授权清理两批重构前旧权重，合计原占用约 **179.95 GiB**；当前 canonical checkpoint、SFT 原件及初始化副本保留。尚未将全部现有权重搬到 `/data`。盘点与删除审计见 `/data/VPO-RM/STORAGE_INVENTORY.md` 和 `.maintenance/`；容量随其他写入变化，部署前应重新查询。

SSH/A6000 部署与资产准备请使用 [部署交接文档](ssh-a6000-handoff.md)。

## GPT-4o裁判全量续跑（2026-09-18 00:03 HKT）

用户已明确允许忽略裁判格式失败并在成绩中排除对应case。`runs/arena-hard-v2-gpt4o-judge-20260917` 已恢复32并发全量判分，复用3000条原回答和59个有效判分；不改裁判提示词。任一顺序失败时，该模型该题两个顺序一起退出计分，同时提供六模型共同有效题目上的成绩。新增状态见`active_judging.json`、`exclusions_progress.json`和`continuation-with-exclusions/controller_state.json`；自动CPU计分已接续。最终胜率尚未产出。详细规则见[GPT-4o失败排除记录](arena-gpt4o-exclusions-2026-09-17.md)。


## Llama SFT完成（2026-09-18 00:14 HKT）

Llama-3.1-8B-Instruct的2500条、2 epoch、157步LoRA SFT已完成，最终GPU重载通过。生成检查因fork/OpenMP卡住后改用spawn独立补跑，任务`llama31-sft-check-spawn-0918-72366407`已Succeeded；100条原始/SFT比较回答全部正常结束。SFT的25条train/test平均长度为207.12/292.36 tokens，均无2048截断。权重在`/data/VPO-RM/models/sft-llama31-8b-instruct-clean2k5e2-20260917`；详情见[训练与完成记录](llama-sft-alignment-2026-09-17.md)。这些是新Llama权重的小样本生成检查，本文之前的六模型benchmark表仍对应Qwen体系。原先“Llama下载尚未验收、NFS待验证”的段落是早期快照；本次actor已通过实际训练/重载，NFS挂载已通过远程校验并用于GPU任务。


2026-09-18 00:17 HKT更新：GPT-4o全量判分在1148/6000完成后因只读账单查询失败暂停，已保存全部记录并恢复32并发，从剩余4852条继续。仅免费账单读取新增最多3次有界重试；已请求的有效/格式失败game均不重判。自动计分watcher也已接续。详见[恢复记录](arena-gpt4o-exclusions-2026-09-17.md)。


## Llama-3.1-8B base SFT 完成（2026-09-19 01:45 HKT）

用 pretrained `Llama-3.1-8B`（ModelScope 镜像，13 个文件 SHA256 对齐上游 revision d04e592b）做了与 Qwen native-EOS SFT 同数据、同超参的 LoRA SFT：2500 条冻结样本、2 epoch、157 步，rjob `llama31-base-sft-0919-2373829` Succeeded。base 没有 chat template，注入了与 Instruct/RM 字节一致的 Llama 3.1 模板；监督终止符是 `<|end_of_text|>`（128001）而非 `<|eot_id|>`，因为 base 权重里 `<|eot_id|>` 等特殊 token 的 embedding 为零且 lm_head 行与全部 reserved token 共用一行，LoRA 无法学会输出它。25+25 探针中 SFT 回答 50/50 以 128001 正常结束、无触顶（平均 199/263 token），base 对照有 13/50 触顶。权重在 `/data/VPO-RM/models/sft-llama31-8b-base-clean2k5e2-20260919`，套件与证据在 `/data/VPO-RM/runs/llama31-base-sft-20260919/`，详情见 `/data/VPO-RM/code/docs/llama-base-sft-2026-09-19.md`。这份权重尚未用于任何 RL 或 benchmark；做 RL 前需先把 Llama 协议检查中写死的 actor EOS 128009 泛化到 128001。

## Llama-3.1-8B base 的 RL 启动（2026-09-19 02:22 HKT）

从 `sft-llama31-8b-base-clean2k5e2-20260919` 初始化的 GRPO 与 VPO λ=4 已进入 RL 流水线：套件 `runs/llama31-base-rl-20260919/`（共享存储），GRPO rjob `llama-base-rl-grpo-2cc8f0341d-28415773` 先跑共享 GPU gate，λ4 由协调进程在 gate 通过后提交。为此 Llama launcher 增加了 `base` profile（actor EOS 128001、只跑 grpo/lam4、容量检查在 lam4、金丝雀 `final_word` 规则），协议检查器支持 `--actor-eos`。训练设置与 Llama-Instruct v3 完全一致。结果尚未产出；详见 `/data/VPO-RM/code/docs/llama-base-sft-2026-09-19.md`。

## 随机 credit 消融启动（2026-09-19 05:02 HKT）

与 canonical VPO λ=4 单变量对照：token 权重不再来自 RM 输入梯度，而是把方向分换成标准正态噪声后走同一个 λ 区间带分配器（`credit_source=random_direction`）。其余全部继承 canonical（SFT 初始化、σ₀=3.0323、采样与优化设置），套件 `runs/rl-ablation-random-credit-20260919/`，rjob `rl-abl-randdir-c3fe46666d-46879224`。详见 `docs/ablation-random-credit-2026-09-19.md`。
