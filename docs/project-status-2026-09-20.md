# 项目更新与论文材料：2026-09-20

本页汇总截至 2026-09-20 的代码、已保存实验结果及论文材料。它取代 README 中的“当前状态”入口；[2026-09-17 文档](project-status-2026-09-17.md) 保留为历史快照。下述结果来自已保存文件，本次 GitHub 更新没有重新训练或请求裁判。

## 实验范围与代码

主实验是 Qwen3-14B / Llama-3.1-8B 与 Base / Instruct 的四组组合。Base 先 SFT 再 RL，Instruct 使用官方 posttrained 权重直接 RL；各自对照 GRPO 与 VPO λ=4。Qwen-Base 另有 λ=2/8 与 Random-direction 消融。

本次合并共享目录、Llama/direct-RL 工作目录和 Arena 排除/恢复工作目录的更新，包括：

- Llama 原生 EOS、保存的 SFT tokenizer、显式 prompt token IDs、RM 输入预算与物理 microbatch 一致性检查。
- Llama Base / Instruct 和 direct-RL 启动、预检、评测、模型身份验证与存储控制代码。
- Arena 的 hard500 / creative250、GPT-4o 自定义裁判身份、失败题目排除、保留旧判分的恢复流程，以及 seed42 / seed43 汇总。
- Random-direction、Random-band、真正的 Shuffle 和 Norm-product credit 控制及相应测试。
- 论文写作材料的整理、分析、选例、绘图、上传与文件校验代码。

历史 Qwen 正式运行有专用冻结源码与直接启动入口，见[源码归档说明](experiment-source-archives.md)；其原始运行记录明确说明当时跳过了 GPU preflight，本次源码发布不补称该门禁已通过。

## Arena-Hard 后 250 题与主表

本次发布的表来自 `runs/arena-table-update-seed42-20260920` 在 **2026-09-20 14:55 HKT** 保存的快照：

- [两张表的 Markdown](results/2026-09-20/tables.md)、[主表 CSV](results/2026-09-20/main.csv)、[消融 CSV](results/2026-09-20/ablation.csv)。
- [Qwen-Instruct seed42 / seed43 独立结果](results/2026-09-20/qwen_instruct_seeds.csv)。
- [来源和哈希](results/2026-09-20/provenance.json)、[补测验证](results/2026-09-20/arena_verification.json)、[尚缺判分清单](results/2026-09-20/remaining_games.csv)。

两轮补测恢复 **152 / 160** 个缺失判定；累计有效 **28,492 / 28,500**。仍缺 8 次判定：6 次 HTTP 503、1 次 content_filter、1 次达到 16,000-token 上限。缺失判定没有填成输、赢或平局。

主表统一使用生成 seed42。每个家族使用对应模型共同有效的题集：

| 比较组 | hard_prompt | creative_writing | 共同有效题目 |
|---|---:|---:|---:|
| Qwen-Base，含七模型消融 | 499 | 249 | 748 / 750 |
| Llama-Base | 500 | 249 | 749 / 750 |
| Llama-Instruct | 500 | 249 | 749 / 750 |
| Qwen-Instruct seed42 / seed43 | 500 | 248 | 748 / 750 |

这里的 Arena 列是两个回答顺序的原始判分合并统计，强胜负权重 3、平局值 0.5；reference 为 GPT-4o-mini，裁判记录包含 GPT-4o 及恢复时的 GPT-4.1。它不是 2026-09-17 的 o3-mini reference 长度/Markdown 控制分数，也不是官方 leaderboard 分数。不同家族的共同有效题集略有差异；不能称为所有模型统一完整 750 题。

| Qwen-Instruct | seed42 Arena | seed43 Arena |
|---|---:|---:|
| Base（复用同一 seed42 baseline） | 66.75 | 66.75 |
| GRPO | 73.60 | 75.50 |
| VPO λ=4 | 73.16 | 74.86 |

其他 benchmark 列逐字保留用户原表，不代表本次重新统一过它们的 seed 或统计口径。CSV 中历史标签 `vpo-lambda(4)-shuffle` 实际对应 **Random-direction**，正式展示应改称 Random。真正的 Shuffle 和 Norm-product 是另外两个控制实验，定义见 [credit-controls](credit-controls-2026-09-20.md)；本页不把 Random 的结果移给它们。

## 完整论文图表与可编辑 PPT

[GitHub 图表目录](../paper_figures/README.md) 与 [Google Drive paper_figures](https://drive.google.com/drive/folders/12a-jhmrgIYsUiyxrb_JoewxHHbQlLXz_) 保存完整训练 reward、raw RM reward、length、KL、entropy、allocator ESS / 权重标准差、消融图和 heatmap，附 PNG / PDF、heatmap SVG、CSV / JSON、原脚本及可移植重画脚本。

原 `runs/paper-2x2-rl-20260919` 的 31 个顶层文件完整保留。图表包共 53 个文件、6,573,031 字节，Google Drive 的 MD5 与本地逐文件一致。GitHub 另加一页目录导航。七张训练 PNG 和四案例 heatmap PNG 从随附数据重画后，与原文件逐字节一致。

[可编辑 heatmap PPT](../paper_figures/fig6_credit_case_heatmaps_editable.pptx) 共五页：完整四案例图加四张放大页。文字、76 个 token 的背景色块和图例均为原生 PowerPoint 对象；每页备注保留对应案例数据。已用 LibreOffice Impress 实际打开、导出 PDF 并检查全部五页，预览随包提供。修改正文或字体后需相应对齐色块。

上级 Drive 的旧 ZIP 是先前快照；新增图表和 PPT 请从上述目录获取。模型、全量训练 dump、全量裁判记录与本地工具缓存保留在原实验存储中。

## 验证

合并后共收集 1,566 项测试。完整运行通过 1,564 项，另外两项实际 Unix socket 测试因共享目录绝对路径过长而触发 `AF_UNIX path too long`；将其临时目录改为允许的 `/data/VPO-RM/.github-update-20260920/pytest-tmp` 后，两项均通过，未修改被测代码。测试在 CPU 上运行。

从 Git 暂存内容导出的干净副本另通过 27 项检查，覆盖 credit 数学控制、Arena creative / 生成和冻结判分恢复。53 个图表包文件与 Drive 上传版本一致，144 个归档源码哈希及三份原表文件哈希均通过核对；所有 128 个新增或修改的 Python 文件通过语法解析。

上述保存的实验成绩来自原实验产物，不由单元测试代替实验验证。测试日志、输入备份和本次合并审计保留在 `runs/github-update-20260920/`，不纳入源码发布。
