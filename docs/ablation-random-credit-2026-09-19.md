# 消融：随机 token credit 对照 RM 梯度 credit（2026-09-19）

目的：证明 VPO 的收益来自"按 RM 输入梯度分配 token credit"，而不只是"非均匀加权"本身。对照臂与 canonical 的 VPO λ=4（`runs/rl-fp32-is-canonical-20260917/lam4`）完全相同，唯一区别是 token 权重在同一个 λ 区间带内随机产生。

## 设置

Qwen3-14B-Base + Skywork-Reward-V2-Qwen3-8B，LoRA 从受保护的 `models/sft-native-eos-clean2k5e2`（SHA256 `21c0c7b9…`）初始化并作为 KL 锚点，UltraFeedback 训练 prompt 与顺序相同，250 rollout、8×8、lr 5e-5、β 0.03、temperature 1、2048 token、soft 长度奖励、σ₀ 直接继承 canonical 的共享校准 3.0323000897825447（校准取决于初始策略与 RM，二者不变），FP32 policy head、采样概率修正、`freeze_stop_tokens`/`freeze_structural`、λ=4、credit microbatch 1。

套件：`runs/rl-ablation-random-credit-20260919/`（共享存储）。`experiment.json` 的 `single_variable_check` 把实际解析出的训练配置与 canonical λ4 的 `checkpoint-250/run_manifest.json` 逐字段比对，差异必须恰为 `['credit_source', 'output_dir']`；`run-arm` 在任务内再比对一次。rjob `rl-abl-randdir-c3fe46666d-46879224`，3×H200，05:02 HKT 提交。

## 随机 credit 的定义（`credit_source=random_direction`）

`vpo_rm/core.py: random_credit`。把 RM 梯度方向 `d_t` 换成 i.i.d. 标准正态噪声（独立的 `torch.Generator`，seed 由训练 seed 派生，不改变采样器和 LoRA 初始化的随机流），然后走完全不变的 `allocate`：逐响应标准化、自适应 τ、`[1/λ, λ]` 区间带、stop/结构/未映射 token 固定权重 1、每条回答权重均值 1。因此序列 advantage 的符号与均值保持，token 权重的分布形状与集中度约束与真实方法同源，去掉的只有 RM 信息。RM 在该臂只做前向打分（不反传输入梯度，也不算全词表 old logits），canonical 字节映射仍用于把未映射位置固定为 1。

另一个可选模式 `random_band`（在区间带上均匀采样权重再投影到均值 1）也已实现并测试，本次未运行；它的噪声比真实方法更强（ESS 更低）。

## 代码改动

- `vpo_rm/core.py`：`random_credit`、`CREDIT_SOURCES`；`allocate` 未改动。
- `vpo_rm/trainer.py`：`TrainerConfig.credit_source`（随机模式要求 `method=vpo_rm`）、独立噪声 generator、`_reward_batch` 在随机模式下只做前向、`train_rollout` 的随机 credit 分支（复用 GRPO 的 old_logp/ref_logp 路径）。
- `scripts/profile_vllm_full.py`：`--credit-source`。
- `scripts/ablation_random_credit_launcher.py`：`record-cpu-validation` / `prepare` / `run-arm`，继承 canonical 配置、σ₀ 与 GPU 数值验证记录，单变量比对，首两轮 gate，完成校验（250 轮、最终 checkpoint、adapter 变化且冻结 ref 与 SFT 一致）。
- 测试：`tests/test_core.py`、`tests/test_trainer.py`、`tests/test_soft_length_trainer.py`（tiny 模型端到端）、`tests/test_ablation_random_credit_launcher.py`；全套 837 passed / 3 skipped（`.maintenance/ablation-random-credit-20260919/full_suite.log`）。

## 结果

2026-09-20 更新：该 Random-direction 臂已完成，250 轮曲线及评测汇总见 [当前项目状态](project-status-2026-09-20.md) 和 [论文图表包](../paper_figures/README.md)。早期表中的 `vpo-lambda(4)-shuffle` 实际指该随机方向臂，应展示为 Random，不能当作真正的 Shuffle。后续 Shuffle / Norm-product 的独立定义见 [credit 控制说明](credit-controls-2026-09-20.md)。

原始产物在 `runs/rl-ablation-random-credit-20260919/random_direction/train/`，`credit_stats.jsonl` 与 `rollout-N-credit.pt` 记录随机权重（`d` 为噪声本身）。评测时与 canonical GRPO / λ4 使用相同协议。
