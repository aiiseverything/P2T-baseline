# VPO-RM

用冻结奖励模型的输入梯度，为 GRPO 的序列优势分配 token credit。当前主实验使用 **Qwen3-14B-Base + Skywork-Reward-V2-Qwen3-8B**，从 native-EOS SFT LoRA 初始化，对照 GRPO 与 VPO 的 λ=2/4/8。

- **迁移到普通 SSH / 8 卡 RTX A6000：** [部署与实验交接](docs/ssh-a6000-handoff.md)
- **短答 8 token、长答 1024–2048 token 的新奖励机制：** [长度奖励配置](docs/length-reward-soft-window.md)
- **既有项目修复与历史结果边界：** [审查修复说明](docs/project-audit-fixes-2026-09-16.md)
- **当前训练与评测结果：** [2026-09-17 项目状态](docs/project-status-2026-09-17.md)
- **评测控制代码与复现边界：** [实验源码归档](docs/experiment-source-archives.md)
- 数学说明：[数学原理](数学原理.md)；工程说明：[工程实现](工程实现.md)。旧文档中的环境和命令，以新的部署交接文档及代码为准。

GitHub 包含代码、测试、配置和文档。`runs/` 中仅按清单归档实验控制源码及说明；模型、数据集、自训 LoRA、原始生成回答和运行日志仍保留在实验存储中。

Arena-Hard 官方实现以固定版本子模块提供，克隆后先初始化，再安装测试依赖：

```bash
git submodule update --init --recursive
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q
```
