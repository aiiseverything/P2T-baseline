# HumanEval：仅评测最终 checkpoint

本次目标是 `runs/rl-native-eos-20260916-143153` 中 GRPO、λ=2/4/8 的最终 `checkpoint-250`。不评测中间 checkpoint，也不混入新 soft 长度奖励训练的结果。

## 协议

- OpenAI HumanEval 原始 164 题，固定 upstream commit `6d43fb980f9fee3c892a914eda09951f772ad10d`。
- 数据文件 `HumanEval.jsonl.gz` SHA256：`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`。
- 原生函数补全：直接输入题目 `prompt`，不套聊天模板，不把测试或参考答案交给模型。
- 每题生成一次，temperature=0、top_p=1、top_k=-1、min_tokens=0、max_tokens=2048、seed=42。
- 保存原始 completion，保留缩进，不去 Markdown、不修复代码、不用额外自定义文本 stop。使用模型的 EOS 停止协议。
- 指标是这套确定性 greedy 协议的 pass@1；不声称是多次随机采样估计的 pass@k，也不等同于 HumanEval+。

生成示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_humaneval.py \
  --model models/Qwen3-14B-Base \
  --adapter runs/rl-native-eos-20260916-143153/grpo/train/checkpoint-250 \
  --dataset datasets/humaneval/HumanEval.jsonl.gz \
  --output runs/humaneval-grpo-final/generation --tag grpo
```

## 代码执行与验证

`third_party/human_eval/execution.py` 是上述固定提交的官方执行器，保留原许可证及文件 SHA。每道题在独立的 Linux user/mount/network/PID namespace 和最小只读 chroot 中执行；只提供系统 Python、标准库、官方执行器和临时目录，项目、模型、home 均不挂载进去。seccomp 禁止创建进程、网络 socket、exec 和再次挂载等操作；另设内存、CPU、文件大小上限与外部超时。

这套评分器需要 Linux `unshare`、`mount`、`chroot`、系统 `/usr/bin/python3`、x86_64 libseccomp，且环境允许创建相关 namespace。隔离创建失败会报错退出，没有在宿主机直接执行模型代码的回退路径。容器内不允许 namespace 时，回到具备该能力的 CPU 主机评分。

```bash
python scripts/score_humaneval.py \
  --dataset datasets/humaneval/HumanEval.jsonl.gz \
  --samples runs/humaneval-grpo-final/generation/samples.jsonl \
  --output runs/humaneval-grpo-final/results.json \
  --sandbox-root runs/humaneval-grpo-final/sandbox --workers 4
```

2026-09-17 已完成真实 CPU 验证：官方参考解 **164/164** 通过；错误答案判失败，死循环超时，隔离内无法访问项目/home、创建 socket 或 fork。全项目回归 **337 passed**。这验证评测管线，不代表任何训练 checkpoint 已取得该分数。

## 自动衔接

`scripts/watch_final_humaneval.py` 在登录机轻量轮询，不占 GPU 等待训练。只有同时满足以下条件才提交对应单卡生成 rjob：训练 `stage=complete`、`exit_code=0`、`completion.json` 记录250轮、profile summary含完整1–250轮、最终 adapter与manifest存在且step=250。单有checkpoint文件并不足以触发。

生成进程成功退出后，watcher 验证164条和文件 SHA，再调用隔离评分器。每组独立保存任务 ID、日志、生成配置、原始 token、逐题通过情况与总 pass@1；提交状态会先持久化，避免中断后盲目重复提交。

本次 watcher 与产物位于 `runs/humaneval-final-native-20260917/`，入口是其固定 `source/`；查看 `status.json`、`watcher.log` 和 `<arm>/results.json`。轮询上限48小时，超过时保存 `watcher_timeout.json`。`submitting/scoring` 中断状态、任务失败或超时需要检查日志后处理，不自动覆盖已有结果。

官方来源：[HumanEval](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d)。
