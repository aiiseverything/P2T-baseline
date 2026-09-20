# 实验控制源码归档

2026-09-20 更新：主目录 `scripts/`、`vpo_rm/`、`tests/` 已合并 Llama / direct-RL、GPT-4o 判分恢复、creative250 和 credit 消融的工作目录改动；保留原 Arena 子模块版本。新增归档包括 GPT-4o 排除/恢复测试所需的冻结控制源码，以及 `runs/direct-rl-qwen-instruct-v3-20260918/source/` 中 Qwen 官方 posttrained actor 的专用 launcher、preflight、storage 和测试。实际正式启动入口保留在 `runs/direct-rl-qwen-instruct-formal-20260918/`，其 README 明确记录当时跳过 GPU preflight 的运行方式。

Qwen 专用冻结 launcher 与通用 `scripts/direct_rl_launcher.py` 不应互换；冻结版本保留了原始模型身份、哈希和运行约束。归档文件只用于源码审计，恢复旧任务还需要原 manifest、校准、模型及未归档依赖。最新结果见 [2026-09-20 快照](project-status-2026-09-20.md)，完整论文图表与可移植画图入口见 [paper_figures](../paper_figures/README.md)。

常规训练与评测入口位于 `scripts/` 和 `vpo_rm/`。本次提交另外保留选定 `runs/` 目录中的实验控制脚本、冻结源码、测试及说明，路径不变，便于核对实际运行版本。完整文件清单及 SHA256 见 [归档清单](experiment-source-manifest.json)。

| 目录 | 归档内容 |
|---|---|
| `runs/arena-hard-v2-canonical-20260917/` | 六模型并行生成、o3-mini reference 判分、成本限制、恢复与独立评分校验 |
| `runs/arena-hard-v2-gpt4omini-20260917/` | 复用回答、更换 reference、17891 代理连接恢复及传输审计 |
| `runs/ifeval-final-canonical-20260917/` | 六模型 IFEval 及评分随机性检查 |
| `runs/ifeval-five-seeds-canonical-20260917/` | 五模型各五次评测与四指标均值汇总 |
| `runs/reward256-canonical-20260917/` | 固定 256 条 prompt 的生成与原始 RM reward 评测 |
| `runs/alpacaeval-final-canonical-20260917/` | 当前 GRPO checkpoint 的 Alpaca 生成控制 |
| `runs/rl-fp32-is-canonical-20260917/run_arm.sh` | 当前四组 RL 的单组启动入口 |
| `runs/maintenance/` | 两个 Llama 权重的下载、上游哈希验证及断点续传源码 |

`runs/` 仍默认受 `.gitignore` 保护，这些文件通过明确清单纳入版本管理。原始回答、判分记录、动态状态、日志、密钥、模型、LoRA、缓存和训练数据不包含在提交中。日期目录中的控制器需要原实验的 manifest、配置和数据才能恢复；仅克隆代码不能接着执行现有付费任务。冻结源码也保留原路径和环境约束，不应当作通用部署入口。

## 测试

初始化固定版本的 Arena-Hard 子模块并安装测试依赖：

```bash
git submodule update --init --recursive
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q
```

`tests/` 下的 Arena 生命周期测试使用这里归档的控制源码，并在临时目录生成测试输入。它们不调用付费 API，也不加载大型模型。需要本地模型或可选训练依赖的测试会报告跳过原因。

2026-09-17 发布验证：本地完整项目 `804 passed, 3 skipped`；从 Git 暂存区导出且不含本地模型、数据的副本 `785 passed, 22 skipped`。两轮均无失败。下面三份恢复测试在干净副本中为 `99 passed, 14 skipped`，跳过项需要原始恢复快照。

GPT-4o-mini 套件还保存独立的恢复测试，可单独运行：

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  runs/arena-hard-v2-gpt4omini-20260917/test_resilient_judge.py \
  runs/arena-hard-v2-gpt4omini-20260917/test_resilient_continuation.py \
  runs/arena-hard-v2-gpt4omini-20260917/test_verify_resilient_scoring.py
```

部分归档集成测试需要未公开的原始实验产物；完整恢复测试还要求恢复前的 `22 valid + 2 unresolved` 快照。这些条件缺失时会明确跳过，不能把跳过解释为完整实验验证通过。运行中的正式评测仍须由其独立校验器验证全部 6000 条判分，并重算 42 项数值后才写完成标记。

## 下载脚本

`runs/maintenance/llama-download/source/` 保留实际后台下载程序的源码副本。它们将模型和临时文件写入 `/data/VPO-RM`，固定 Hugging Face revision，并核对 ModelScope 镜像与上游文件身份。部署时将三个脚本复制到 `/data/VPO-RM/.download-logs/`；运行环境需要 `huggingface_hub`、`modelscope`、`safetensors` 等下载依赖。现有部署使用 `sml` Python 环境。

`resume_llama_downloads_20260917.py` 会检查已有进程、备份旧状态，再通过确认的 17891 代理恢复分片。该端口是本次机器配置，其他环境需要相应配置。不要在已有监督进程运行时重复启动。
