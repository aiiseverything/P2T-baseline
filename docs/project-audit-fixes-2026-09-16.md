# 2026-09-16 项目审查修复说明

本次修复覆盖审查报告的 A01–A16，以及同一审查中确认的采样协议、脚本参数和统计问题。历史模型、训练记录、评测结果保留原样。审查证据位于 `runs/project-audit-20260916/`；修复的回归记录、独立验证、历史重算和逐项结案表位于 `runs/project-fixes-20260916/`。

## 新实验的行为

- SFT 默认监督 native EOS。梯度按每个实际 optimizer batch 的有效监督 token 加权平均，包括不足完整累积窗口的最后一批；EOS 加权也进入分母。每轮重新打乱长度桶，非整数 epoch 只消耗对应比例的 microbatch。此目标定义不同于旧的“每个 microbatch 等权平均”。
- VPO 结构 token 从实际 tokenizer 解码产生。冻结位置权重固定为 1；其余位置独立标准化、分配预算，最终同时校验 `1/λ ≤ w ≤ λ` 和平均权重为 1。padding 权重为 0。
- 原生 HF 和 vLLM 使用相同的注册输出词表、停止 token、temperature 和最短生成长度约束。正式训练支持 `top_p=1`、`top_k=0`、presence penalty 为 0、dropout 为 0；不支持的采样变换会显式报错。temperature 在 FP32 分块概率计算中处理，避免对整块 BF16 logits 先除温度产生额外舍入误差。
- 回答在第一个停止符处结束，停止符本身参与损失，后续 EOS padding 不参与。截断惩罚依据生成结束原因，正常在长度上限结束的回答不再误罚。
- KL 默认使用冻结的初始参考策略。新建 LoRA 使用禁用 adapter 的基座；SFT 初始化使用冻结的 reference adapter；全参数训练保留一份冻结初始模型，因而需要额外内存。`rollout` 参考仅作为显式旧模式保留，不能把它解释为长期初始策略约束。
- `policy_epochs_per_rollout` 和 `optimizer_minibatch_responses` 现在实际控制 optimizer 更新次数；微批按当前实际 optimizer batch 的响应数归一。非有限 loss/梯度会阻止相应 optimizer 更新。
- 正式 SFT/RL 训练先排除与 AlpacaEval、IFEval、GSM8K 的规范化 prompt 精确交集，再划分数据；记录排除清单和 hash。缺少本地 benchmark 文件会报错。历史离线 RM 评测仍使用旧划分，以保留历史可比性。

## 评测与启动脚本

评测缓存必须同时匹配完整 recipe、seed、长度、模型、adapter、数据和代码指纹，并验证输出文件 hash。没有 manifest 的旧缓存不再自动复用；请使用新的输出目录。adapter 权重、配置、tokenizer 和数据使用 SHA256；体积较大的基座权重使用文件路径、大小、inode、mtime/ctime 标识。替换或修改同一路径文件会导致缓存失效，基座标识不是整份权重的密码学内容校验。

AlpacaEval 保存所有 sample，并保留 `sample_idx`。judge 可通过 `--generation-file` 选择多采样输出，对完整且各题采样数相同的候选进行平均。逐条判分保存为 annotations，可在中断后续跑。每批最多提交 `workers` 个请求，验证费用后再提交下一批；账单接口异常时停止派发。该机制限制新增请求，已发出的请求和账单延迟仍可能使费用超过阈值。

GSM8K 用十进制数值精确比较，不再接受相近但错误的数字。RM 评测中，不同 run 的同一步 checkpoint 使用不同 LoRA ID。所有评测器从实际 tokenizer 获取停止 ID，并核验 adapter 基座兼容性。训练链把 Actor 和 RM 路径传到评测阶段；`EVAL_RUNS` 支持分号分隔的多个 `label=path`。

14B 正式训练脚本默认使用已有的 `models/sft-native-eos-clean2k5e2`；其他基座必须显式提供兼容初始化。这个现有 adapter 是历史模型，仍包含旧 SFT 的尾批问题，本次没有覆写或重新训练它。新 SFT 输出使用独立名称。新实验的完整配置以生成的 manifest 为准。`run_sft_native_eos.sh` 保留为历史单变量实验入口，依赖其显式指定的原始 `SOURCE_SNAPSHOT` 和数据 hash；修复后的新 SFT 应使用当前 `sft_init.py` / `run_sft_v2.sh`，不能沿用“只改变结束词”的实验描述。

## 历史结果的处理

- GSM8K：另存 42 份修正版，纠正 11 条误判、影响 9 份结果。26 份从保存的生成文本重新提取数字，16 份只能使用已保存的 pred/gold，因此无法恢复当时 float 转换前丢失的精度。见 `corrected-gsm8k-summary.json`。
- 结构 token 分析：使用修正后的 tokenizer 分类另存示例分析；历史实际训练中的错误冻结、λ 越界、无效 KL 和误罚无法通过事后改统计消除，需要重训才能评价修复后的方法。
- AlpacaEval：历史 RL 训练划分有 7 条交集，已保守地为 20 个 RL 评测目录排除这 7 题，生成 798 题候选和参考集。旧 judge 只保存汇总分，无法恢复删题后的精确分数；另存可严格推导的分数区间，未冒充重新判分的点估计，也未自动发出付费重评请求。当前 native-EOS SFT 的 2500 条训练样本与这三套 benchmark 的精确交集为 0。
- 旧模型和旧结果不能仅因代码已修复而标成“已修复实验”。比较新旧结果时应注明代码、数据排除、SFT 损失归一化等变化。

## 边界与复现

当前 checkpoint 保存状态用于检查和留档，**没有实现训练断点恢复**；manifest 明确记录 `resume_supported=false`。不要把加载 adapter 作为优化器、调度器和 RNG 的完整恢复。

多语种脚本过滤是既有 SFT 数据策略，本次未把它当成程序 bug 更改。它会删除部分合法的非拉丁文本；清洗后重新划分也可能改变验证集。比较 clean/raw 实验时必须核对 split hash，不能假定训练集或验证集完全相同。

本地完整测试使用已有的 PyTorch/Transformers 环境；PEFT 可从 `.vllm-extra/peft` 单独加入路径，避免把为 Python 3.12 安装的 `.vllm-extra/pyarrow` 放到 Python 3.10 的包搜索优先位置。绘图测试可设置 `PLOT_TEST_PYTHON` 为已有 matplotlib 环境。生产 vLLM 参数由单卡 H200 短任务单独验证；小模型 CPU 测试不代替 14B 多卡长训的性能与质量评估。
