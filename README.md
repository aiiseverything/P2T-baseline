# VPO-RM

用冻结奖励模型的输入梯度，为 GRPO 的序列优势分配 token credit。当前主实验使用 **Qwen3-14B-Base + Skywork-Reward-V2-Qwen3-8B**，从 native-EOS SFT LoRA 初始化，对照 GRPO 与 VPO 的 λ=2/4/8。

- **迁移到普通 SSH / 8 卡 RTX A6000：** [部署与实验交接](docs/ssh-a6000-handoff.md)
- **短答 8 token、长答 1024–2048 token 的新奖励机制：** [长度奖励配置](docs/length-reward-soft-window.md)
- **既有项目修复与历史结果边界：** [审查修复说明](docs/project-audit-fixes-2026-09-16.md)
- 数学说明：[数学原理](数学原理.md)；工程说明：[工程实现](工程实现.md)。旧文档中的环境和命令，以新的部署交接文档及代码为准。

GitHub 包含代码、测试、配置和文档。模型、数据集、自训 LoRA、实验结果分别放在 `models/`、`datasets/`、`runs/`，通过交接文档准备，不包含在仓库中。
