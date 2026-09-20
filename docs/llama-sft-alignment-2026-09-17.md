# Llama 3.1 SFT 与 Qwen 初始化的对齐

本次使用 `/data/VPO-RM/models/Llama-3.1-8B-Instruct` 做 LoRA SFT，输出写入 `/data/VPO-RM/models/sft-llama31-8b-instruct-clean2k5e2-20260917`。当前实验目录为 `/data/VPO-RM/runs/llama31-sft-aligned-20260917-attempt2`。启动前需要该目录的输入、协议、存储与 GPU smoke 检查通过；本文本身不代表训练已完成。

## 保持的训练条件

| 项目 | 本次设置 |
|---|---|
| 训练样本 | Qwen native-EOS SFT 实际使用的同一批有序 2500 条 clean UltraFeedback prompt/response |
| Epoch、seed | 2、42 |
| Batch | microbatch 4，梯度累积 8，有效 batch 32 |
| 学习率 | 1e-4，warmup ratio .03，cosine 衰减到 .3 倍学习率 |
| 优化器 | AdamW，weight decay 0，梯度裁剪 1 |
| LoRA | r64、alpha128、dropout0、bias none；q/k/v/o/gate/up/down projection |
| 精度 | BF16 base/head、FP32 LoRA；loss 内部 FP32，与历史 SFT 一致 |
| 长度 | 总序列上限4096，超长整例丢弃；不另加2048 prompt截断 |
| Loss | 仅 assistant response 与末尾结束符；prompt/header/padding 为 -100；EOS权重1 |

原始 clean parquet SHA256 为 `ec5821bb8ccfa8ecf3a25e9efa2207ffcc2b59c52b18a45ab234b5c6b635c48c`。冻结有序完整 pairs 的 JSON 摘要为 `124c50dab599207c64dd8f740cdb123b642ca6f715d6550932b94fc37d0f3459`。新 `--selection-file` 会逐对检查完整 prompt 与 response 仍属于当前清理后的 train pool；不会跳过 benchmark 排除或重新随机抽样。

不能用相同 seed 重新抽样来声称同一数据：当前 pool 修正去重及 benchmark 排除后，重抽2500条只与旧集合重合118条。冻结原样本与现有 AlpacaEval、IFEval、GSM8K、HumanEval、Arena-Hard 及16条monitor的规范化 prompt 完全相等交集为0；这不排除语义近似或预训练污染。

## Llama 原生协议

保留下载版本中的原生 chat template，包括自动 system 内容与固定 `26 Jul 2024` 日期。模板已经插入 BOS，render 后 tokenize 必须使用 `add_special_tokens=False`。

| 用途 | Token / ID |
|---|---|
| BOS | `<\|begin_of_text\|>` /128000 |
| 正常 assistant 回答结束 | `<\|eot_id\|>` /128009 |
| 文档结束 | `<\|end_of_text\|>` /128001 |
| 工具交接 | `<\|eom_id\|>` /128008 |
| 右侧 padding | `<\|finetune_right_pad_id\|>` /128004 |

Llama 末尾的 EOT 已等于 tokenizer EOS，保留原样，没有 Qwen 的尾随换行。生成停止集使用已注册的 `[128001,128008,128009]`。训练仍只监督普通回答结束 EOT；停止集不意味着将三者都追加到回答上。

该2500条数据全部通过真实tokenizer边界核验：0丢弃，最长总序列2429，每epoch input tokens1,143,596、supervised tokens674,472；target token SHA256为 `df0dd89df29386fae501d45e83f5f3302ef92cc1b53e90d32703df97fcb40346`。两条prompt超过2048，但总长在4096以内。

依据：[Meta Llama 3.1 prompt format](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)、[Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating)、本地 pinned tokenizer/config 的实际输出。

## 需要保留的比较边界

Llama 起点是 Instruct，Qwen 是 Base；这不是只改变模型架构的受控比较。Skywork Llama RM 是独立 sequence-classification 模型，本次 SFT 不加载它；以后 RL 必须用 RM 自己的模板与 pad 配置，不能只复用 actor token IDs。[Skywork model card](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2)

历史 Qwen SFT 有已修复的实现问题。本次保留修复：1250微批对应157次更新，最后2微批执行更新；按每次更新的全部监督tokens归一化；每个epoch重新shuffle；学习率以0为起始索引。历史实际只更新156次、尾批梯度丢弃、等权平均微批loss，并复用epoch顺序；首步LR为5e-5，而修复后为2.5e-5。不能声称与历史运行逐步完全相同。

## 执行与验收

代码工作区 `/data/VPO-RM/code` 使用独立 `llama-sft-20260917` 分支。`/data` 是 NFS，已通过0-GPU rjob的远程文件与nonce往返检查，使用 `--use-file-store true --file-store-nfs-path 10.68.62.222:/data:/data` 挂载。

正式单卡任务先检查冻结输入和协议，再以独立adapter进行32条、1次更新的smoke及真实模型重载；通过后从原始Llama权重新建正式adapter训练。完成后重载验收，并在与Qwen历史完全相同的25条train和25条test prompt上分别生成 pretrained/SFT 回答，记录长度、停止原因和截断。该生成检查不是新的AlpacaEval评分。

## 运行记录

首个GPU任务 `llama31-sft-aligned-0917-22557974` 停在依赖安装阶段，pip尝试下载镜像中已有的NCCL依赖，已主动停止；没有进入smoke或正式训练。attempt2固定使用镜像Torch2.13.0+cu129、Transformers5.16.1与共享`.vllm-extra`中的PEFT0.20.0、PyArrow21.0.0，不再执行pip安装。

0-GPU依赖任务 `ll-sft-deps2-0917-151452-26f0-9287108` 已Succeeded：2500条协议预检通过，17项测试通过、7项因环境条件跳过；train/test最长prompt为1024/1846。预检发现Transformers5的`apply_chat_template(tokenize=True)`默认返回字典形式，测试与长度检查已显式设置`return_dict=False`；训练本身使用render后显式encode，不受此差异影响。

正式任务 `llama31-sft-aligned-v2-0917-28151841` 于2026-09-17 23:18 HKT提交：1 GPU、16 CPU、200000 MiB。CPU预检成功不代表GPU训练成功；最终需同时确认rjob Succeeded、`job/stage=complete`、`summary.json`通过。

2026-09-17 23:30 HKT：smoke真实单步训练与GPU重载验收通过，正式2500条训练已开始。smoke loss1.149398543、grad_norm1.386117339均有限；448个FP32 LoRA张量均有限，224个B矩阵全部发生更新。重新加载后head/logits为BF16，loss为FP32，forward loss1.597037911且logits全有限；详见attempt2的`job/smoke_reload.json`。首次16GB权重从NFS读取约9分钟；后续重载使用同节点缓存后明显加快。此状态仍不代表157次正式更新或后续生成验收完成。

2026-09-17 23:32 HKT：正式训练已记录step20/157，loss1.025197993、grad_norm0.411288828、吞吐3899.5 input tokens/s，292784累计input tokens，数值有限。rjob仍为Running；后续最终重载和100条生成检查尚未执行。启动验收记录位于attempt2的`job/launch_validation.json`。


## 最终完成（2026-09-18 00:14 HKT 验证）

全部2500条、2 epoch、157次更新已完成，输入token总计2,287,192；最终权重448个LoRA张量均有限，224个B矩阵全部更新。最终真实8B GPU重载及forward检查通过。末次小窗口loss0.976279、grad_norm0.702515；末窗口仅8条样本，不能用单点loss直接衡量通用效果。

原任务训练和重载完成后，生成检查卡在vLLM EngineCore初始化。父进程已启动OpenMP线程，默认fork子进程停在futex且GPU闲置；相同CPU运算的独立最小复现中fork超过10秒未完成、spawn约1.2秒完成，支持fork/OpenMP死锁的诊断。已提交独立生成检查任务`llama31-sft-check-spawn-0918-72366407`，随后停止旧任务并释放启动门禁；只设置`VLLM_WORKER_MULTIPROC_METHOD=spawn`，保持原冻结生成代码、权重、OMP/MKL8及采样设置。没有重训。

新检查任务已Succeeded，`postcheck-recovery/stage=complete`、`exit_code=0`，`summary.json`验收通过；原任务保留Stopped这一真实终态，其`job/stage`仍是历史generation_checks。由于采用独立任务补全生成检查，上文原先要求单个训练rjob直接Succeeded的验收路径由“训练与重载验收通过 + 新生成检查任务Succeeded + 完整summary”替代，不能把旧任务改记为Succeeded。

| 固定prompt集合 | 原始Llama Instruct平均tokens | SFT平均tokens | SFT中位数 | SFT达到2048上限 |
|---|---:|---:|---:|---:|
| train 25条 | 341.64 | 207.12 | 129 | 0/25 |
| test 25条 | 347.40 | 292.36 | 196 | 0/25 |

两模型各50条，共100条生成均正常stop，最后token均为原生EOT128009。参数temperature1、top-p1、seed42、max_tokens2048、native BF16 head。这是固定小样本生成检查，尚不是该Llama SFT的AlpacaEval、IFEval或Arena-Hard分数。最终证据见attempt2的`summary.json`、`job/final_reload.json`、`postcheck-recovery/completion_verified.json`与实际rjob终态记录。
