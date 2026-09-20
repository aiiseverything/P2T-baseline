# Llama-3.1-8B base SFT（native EOS），2026-09-19

目的：给 pretrained `meta-llama/Llama-3.1-8B` 一个与 Qwen3-14B-Base canonical 线（`models/sft-native-eos-clean2k5e2`）以及 Llama-3.1-8B-Instruct SFT（`sft-llama31-8b-instruct-clean2k5e2-20260917`）同协议的 LoRA SFT 初始化。套件：`/data/VPO-RM/runs/llama31-base-sft-20260919/`；产物：`/data/VPO-RM/models/sft-llama31-8b-base-clean2k5e2-20260919`（657 MB，adapter SHA256 见 `job/completion.json`）。rjob `llama31-base-sft-0919-2373829`，1×H200，01:23 提交、01:45 HKT Succeeded，exit code 0。

## 训练条件（与 Instruct SFT 相同）

同一批冻结有序 2500 条 clean UltraFeedback pairs（`full_pairs_sha256 124c50da…`）、2 epoch、seed 42、micro 4 × accum 8、lr 1e-4（warmup .03，cosine 到 .3×）、LoRA r64/α128/dropout 0/七个 projection、BF16 base + FP32 LoRA、总长 4096、只监督 assistant 回答与终止符。157 次更新，2,287,192 输入 token，最后一步 loss 0.952（尾批 2 个 microbatch），步 100/140 的 loss 0.754/0.799。

## Llama base 协议

| 项目 | base 自带 | 本次 |
|---|---|---|
| chat template | 无 | Instruct/RM 的 Llama 3.1 原生模板，字节一致（SHA256 `e10ca381…`），从 `inputs/llama31_chat_template.jinja` 注入并随 adapter 保存 |
| BOS | 128000，由 tokenizer post-processor 添加 | 由模板提供一次；渲染文本以 `add_special_tokens=False` 分词（否则双 BOS） |
| tokenizer EOS | `<|end_of_text|>` 128001 | 不变 |
| 监督终止符 | — | `<|end_of_text|>` 128001（`--response-eos native`），2500 条全部替换模板末尾的 `<|eot_id|>`；prompt 侧的 `<|eot_id|>` 保留 |
| padding | 无 | `<|finetune_right_pad_id|>` 128004 |
| 生成停止集 | generation_config 只有 128001 | 注册集 `[128001, 128008, 128009]` |

**为什么不用 Instruct 的 EOT 128009**：base 权重中 `<|eot_id|>`、`<|eom_id|>`、两个 header token、pad 及全部 reserved token 的 embedding 行为 0，且共用同一条 lm_head 行（范数 0.4527，两两差 6e-5，余弦 1.0）；`<|end_of_text|>` 有独立训练过的行（0.7749）。LoRA 只作用于 attention/MLP 投影，无法让 `<|eot_id|>` 的 logit 区别于约 250 个共享该行的 token，以 EOT 结尾的 SFT 学不会停。Qwen3-14B-Base 的 `<|im_end|>` 输出行同样坍缩，这就是项目 native-EOS SFT 的机制。证据：`job/special_token_weight_evidence.json`。

分词核对：2500 条在 base tokenizer 上与 Instruct 版逐 token 一致，仅末位不同（target SHA256 `a3ddb843…` vs `df0dd89d…`），每 epoch 1,143,596 输入 / 674,472 监督 token。GPU preflight（Transformers 5.16.1 / tokenizers 0.23.1 / torch 2.13.0+cu129 / peft 0.20.0）与 CPU preflight（Transformers 4.57.6）得到相同统计。

## 代码改动（`/data/VPO-RM/code`，llama-sft-20260917 worktree，未提交）

- `scripts/sft_response_tokens.py`：`install_chat_template(tokenizer, template, expected_sha256)`，已有不同模板时拒绝覆盖。
- `scripts/sft_init.py`：`--chat-template-file` / `--chat-template-sha256`；manifest 记录模板来源与 sha。
- `scripts/verify_sft_adapter.py`：`--chat-template-file` / `--response-eos-id`（base 用 128001，Instruct 仍默认 128009）。
- `scripts/eval_alpaca.py`：`--tokenizer`，base 无模板时用保存的 SFT tokenizer 渲染，并记录 tokenizer 来源指纹。
- 新增回归测试：`tests/test_llama_sft_protocol.py`、`tests/test_verify_sft_adapter.py`、`tests/test_alpaca_prompt_tokens.py`。全套 1168 passed / 4 skipped（`job/local_full_suite.log`）。

## 结果

- 最终重载验证通过：448 个 LoRA 张量全部 FP32 有限，224 个 B 矩阵全部更新，max|w| 0.0222。
- 自由生成监控（16 条，384 token 上限，temperature 1）：step 100/157 均无回显、无 Confidence 循环、无脚本沙拉；6/16 触及 384 上限。
- 25+25 train/test 探针（vLLM，temperature 1，2048 上限，base 与 SFT 均用保存的 SFT tokenizer 渲染）：

| 臂 | train 平均 token | train 触顶 | test 平均 token | test 触顶 | 末 token |
|---|---:|---:|---:|---:|---|
| base + 模板，无 adapter | 988.4 | 8/25 | 812.9 | 5/25 | 17/20 次为 128001，其余触顶 |
| SFT | 199.2 | 0/25 | 262.8 | 0/25 | 25/25 为 128001 |

50 条 SFT 回答中无模板泄漏（未出现 header/eot 字面、Cutting Knowledge Date 等）。对照：Qwen native-EOS SFT 探针 149/257，Llama-Instruct SFT 207/292。

## 后续做 RL 前必须处理

`scripts/check_llama_protocol.py` 与 Llama RL launcher 目前写死 actor EOS 为 128009；base 初始化的 actor EOS 是 128001，需要先泛化并重新过 GPU 协议 gate。RM 侧不受影响：canonical RM 输入去掉 actor 的停止符后用 RM 自身模板补 EOT。

## RL 启动（2026-09-19 02:22 HKT）

套件：`/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/llama31-base-rl-20260919/`（源码快照与训练输出放共享存储，模型与数据从 `/data` 读取；experiment SHA256 `da0130b2…`）。臂：GRPO 与 VPO λ=4，两个 3×H200 rjob：`llama-base-rl-grpo-2cc8f0341d`、`llama-base-rl-lam4-2cc8f0341d`。GRPO 分配先跑共享 GPU gate（RM 精度审计 → grpo/lam4 各两轮预检 → 128 prompt σ₀ 校准 → lam4 容量检查），`operations/lam4_coordinator.py` 在 `shared-gate.json` 通过后才提交 λ4。所有训练设置与 `llama31-rl-canonical-20260918-v3` 一致，只有 actor、初始化 adapter 与相应协议不同。

为 base actor 做的 launcher 改动（`/data/VPO-RM/code`，未提交）：

- `scripts/llama_rl_launcher.py`：显式 `instruct`/`base` profile（actor、SFT adapter 与其受保护哈希、actor EOS、臂集合、容量臂、任务名前缀、SFT 证据目录、资产核验记录、质量规则），写入 manifest 并在 `run-arm` 时按 manifest 重新选择；GPU 报告校验改为按 profile 的臂集合，并核对质量规则；prompt 审计改用保存的 SFT tokenizer。
- `scripts/check_llama_protocol.py`：`--actor-eos {128001,128009}`，actor 终止符与 SFT manifest（`response_eos_replaced`）一致性检查，新增 `actor_stop` 用例。
- `scripts/preflight_training.py`：`--experiment-profile`、`--arms`、`--capacity-arm`、`--quality-rule`。
- `scripts/preflight_quality.py`：`final_word` 规则（句末词等于答案也算正确）。base SFT 在 CPU 上贪心复现 8 道金丝雀：严格规则 5/8（2+2 答 2；"…is A."、"…is purple." 因格式判错），`final_word` 7/8；证据 `runs/llama31-base-sft-20260919/job/quality_canaries_cpu.json`。规则在 GPU 执行前冻结在 profile 里。
- `vpo_rm/token_policy.py`：Transformers 4 无法用 AutoTokenizer 解析 Transformers 5 导出的 `TokenizersBackend` 类名时，受控回退到 `PreTrainedTokenizerFast` 读同一份保存的 backend 与模板，协议检查照常执行。

CPU 侧证据：`/data/VPO-RM/.maintenance/llama-base-rl-20260919/{assets_verified.json,cpu-protocol.json,prepare.log,full_suite_final.log}`。RM 对 128001 终止符的 canonical 输入处理与 EOT 完全一致（去掉 actor 停止符后由 RM 模板补 EOT，映射位置 `[37,38,39,-1]`）。
