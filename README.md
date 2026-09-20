# VPO-RM

用冻结奖励模型的输入梯度，为 GRPO 的序列优势分配 token credit。当前实验覆盖 **Qwen3-14B / Llama-3.1-8B × Base / Instruct**：Base 从 SFT 初始化做 RL，Instruct 直接做 RL；对照 GRPO 与 VPO λ=4，并在 Qwen-Base 上比较 λ=2/4/8 和 credit 消融。

- **迁移到普通 SSH / 8 卡 RTX A6000：** [部署与实验交接](docs/ssh-a6000-handoff.md)
- **短答 8 token、长答 1024–2048 token 的新奖励机制：** [长度奖励配置](docs/length-reward-soft-window.md)
- **既有项目修复与历史结果边界：** [审查修复说明](docs/project-audit-fixes-2026-09-16.md)
- **当前实验、Arena 补测与结果口径：** [2026-09-20 项目状态](docs/project-status-2026-09-20.md)
- **论文图表、绘图数据与可编辑 PPT：** [paper_figures](paper_figures/README.md) · [Google Drive](https://drive.google.com/drive/folders/12a-jhmrgIYsUiyxrb_JoewxHHbQlLXz_)
- **Random、Shuffle 与 Norm-product 的区别：** [credit 消融定义](docs/credit-controls-2026-09-20.md)
- 历史实验快照：[2026-09-17 项目状态](docs/project-status-2026-09-17.md)
- **评测控制代码与复现边界：** [实验源码归档](docs/experiment-source-archives.md)
- 数学说明：[数学原理](数学原理.md)；工程说明：[工程实现](工程实现.md)。旧文档中的环境和命令，以新的部署交接文档及代码为准。

GitHub 包含代码、测试、配置、文档，以及轻量论文图表和绘图输入。`runs/` 中仅按清单归档实验控制源码及说明；模型、数据集、自训 LoRA、全量原始回答和运行日志仍保留在实验存储中。四个 heatmap 案例的原文与逐 token 权重随图表包发布。

Arena-Hard 官方实现以固定版本子模块提供，克隆后先初始化，再安装测试依赖：

```bash
git submodule update --init --recursive
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q
```
