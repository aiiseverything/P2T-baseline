# 8×RTX A6000 / 普通 SSH Linux 服务器交接

更新：2026-09-16。目标是把当前 **14B Base + 8B RM + 已训练 native-EOS SFT LoRA** 的实验迁移到普通 SSH 服务器。无需 `rjob`、集群挂载、内网镜像或内网 pip 源。推荐先使用 4 卡跑一个实验，通过验收后再考虑两个实验各用 4 卡。

**验证边界：** 2026-09-16 在原开发环境完成全套 CPU 回归：333 passed、0 skipped、1 条已有 PEFT 配置提示；覆盖分批 credit 路径、GPU 分配和资产准备工具。公开依赖已在 Python 3.12 / Linux x86_64 目标上完成解析，本地资产检查通过。尚未在你的 A6000 机器实际训练。下文给出的 GPU 生成、最坏长度容量和两轮完整训练验收必须在目标机执行；通过前不把“48GB 能运行”当成已验证结论。

## 1. 接手者先知道这些

| 项目 | 当前约定 |
|---|---|
| GitHub | `https://github.com/kilgrims/VPO-RM`，分支 `niuniu` |
| Actor / vLLM 基座 | `Qwen/Qwen3-14B-Base`，不能替换成 Instruct、AWQ、GPTQ 或其他大小 |
| 奖励模型 | `Skywork/Skywork-Reward-V2-Qwen3-8B`，冻结参数；VPO 仍计算输入梯度 |
| 初始化 | `models/sft-native-eos-clean2k5e2`，自训 LoRA，2500 条 clean UltraFeedback、2 epochs、seed 42 |
| 结束词 | SFT target 使用 native EOS `151643`；采样支持 tokenizer 注册的停止 ID |
| RL 数据 | UltraFeedback Binarized `train_prefs`；去重、排除三个 benchmark 的精确 prompt 交集、留出 2000 validation、按模板长度过滤 |
| RL 更新 | 每轮 8 prompts × 8 responses = 64；optimizer minibatch 64，物理训练微批 1；每 rollout 一次更新，250 rollouts |
| 优化参数 | LR `5e-5`，β `0.03`，KL 参考冻结 SFT 初始化，τ `1`，temperature `1`，top_p `1`，训练 top_k `0` |
| 四组 | GRPO；VPO λ=2、4、8。VPO stop/structural credit 冻结为 1 |
| 新长度奖励 | `soft`：生成 min_tokens=0；L<8 轻罚；8≤L≤1024 不罚；1024–2048 线性长罚。L 含 EOS、不含 prompt/padding |
| 断点恢复 | **没有完整 resume**；加载已有 adapter 不恢复 optimizer/RNG，也不等同于续训 |

本次迁移默认用新 `soft` 机制。旧 H200 在跑的实验使用短罚 anchor=600、min_tokens=64、slope=0.00506，**不是同一个奖励实验**。需要复现旧配置时显式选择 `--length-reward-mode legacy`，结果分目录保存。

已有 SFT adapter 属于历史产物，包含当时尾批更新少一次的实现行为。迁移直接复制该产物。用现在修复后的 SFT 脚本重新训练会得到新模型，不能把它当成同一初始化替代。

## 2. 项目结构与 GPU 分工

```text
scripts/run_ssh_rl.py             普通 SSH 入口，选择 arm、物理 GPU、输出目录
scripts/profile_vllm_full.py      Actor/RM/vLLM 编排、LoRA 热加载、指标与checkpoint
scripts/vllm_generate_server.py  独立生成进程，经本机 Unix socket 通信
vpo_rm/trainer.py                rollout → RM → 组内优势 → Actor 更新
vpo_rm/core.py                   GRPO loss、VPO credit、λ权重约束
vpo_rm/integration.py            Actor logits、RM输入梯度、概率/输出词表协议
vpo_rm/length_reward.py          短/长软罚与固定奖励尺度校准
vpo_rm/rollout_selection.py      整组无正常完成回答时重采一次，否则跳过
scripts/eval_{alpaca,ifeval,gsm8k}.py  生成及本地规则评测
scripts/judge_alpaca.py          可选付费外部judge，和GPU生成分开执行
```

推荐单个实验分配 `CUDA_VISIBLE_DEVICES=0,1,2,3`：进程内 `cuda:0` 是 Actor/optimizer，`cuda:1` 是 RM，后两张卡独立交给 vLLM TP=2。第二个实验可分配物理卡 `4,5,6,7`。不要用 `torchrun --nproc_per_node=8` 启动现有 trainer：它不是 DDP/FSDP trainer，8 卡显存也不会自动合成一张大卡。

原始 H200 默认不适合直接照搬：vLLM 固定 45% 显存，在 48GB 卡上不足以加载约 27.5GiB 的 BF16 Actor 权重；VPO 原先完整缓存 `[64,2048,151936]` logits，单份 BF16 就约 37.1GiB。新入口将 vLLM 预算设为 85%、TP=2、生成并发 4，并设 `credit_microbatch_responses=1`，避免整批词表 logits 常驻。RM embedding 和小的组缓存仍然占显存；这些调整不改变 8×8 的逻辑批次和损失定义。

硬件与系统建议：

- RTX A6000 是 Ampere、48GB、compute capability 8.6。不是 RTX 6000 Ada 或 RTX PRO 6000。[NVIDIA 规格](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)、[能力表](https://developer.nvidia.com/cuda/gpus)
- Linux x86_64，推荐 Ubuntu 22.04/24.04，Python 3.12。当前锁文件按 manylinux / glibc 2.35 环境解析。
- 对固定 CUDA 12.9 wheel，保守要求 NVIDIA Linux 驱动 **≥575.57.08**。`nvidia-smi` 的 CUDA 字段是驱动支持上限，不是当前 Python 的 runtime。不要只按旧驱动的 CUDA minor compatibility 表判断 Triton/PTX 能否运行。[CUDA 12.9 U1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)、[兼容限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- 建议主机内存 ≥128GiB 跑单实验；两个并发实验建议 ≥256GiB，并观察实际峰值。这是部署预算建议，尚未实测该服务器。
- 模型与必需数据约 43GiB；加环境、下载缓存和输出，单实验建议预留 ≥200GiB，四组长训建议 300–400GiB。历史四组保存策略的输出峰值估计约 100.5GiB，临时打包和重复下载另算。
- `nvidia-smi topo -m` 检查 vLLM 两卡拓扑；优先有 NVLink 的配对。是否配有 NVLink 以实机为准，PCIe 配对可能更慢。

## 3. 新机拉取代码、建立独立环境

以下命令在目标机普通 SSH shell 执行。只在自己的工作目录创建环境，不依赖 `/root` 或旧集群路径。

如果 `$HOME` 所在盘空间不足，改在大容量数据盘下 clone，并相应设置 `VPO_ROOT`；模型和输出默认位于该 checkout 内。

系统需要 `git`、`curl`、`rsync`、`tar`、`tmux` 及 C/C++ 编译工具，供下载、后台运行和 Triton/JIT 使用。Ubuntu 缺少时由管理员安装 `git curl rsync tmux build-essential ca-certificates`。Python 的 CUDA runtime 由锁文件提供，无需为了这些预编译 wheel 另装一套系统 CUDA toolkit。

```bash
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone --branch niuniu --single-branch https://github.com/kilgrims/VPO-RM.git
cd VPO-RM
export VPO_ROOT="$PWD"
git rev-parse HEAD
nvidia-smi
nvidia-smi topo -m
free -h
df -h .
ldd --version | head -n 1
```

若仓库访问需要认证，使用你自己的 GitHub SSH/PAT；不要把凭据写进仓库。已经 clone 的目录先 `git pull --ff-only origin niuniu`，并记录实际 commit。

每个新的 SSH / tmux 会话先 `cd` 到实际 checkout，然后 `export VPO_ROOT="$PWD"`；环境装好后再 `source .venv/bin/activate`。普通 shell 变量不会自动跨会话持久化。

环境锁文件在 [`requirements/ssh-a6000-cu129.txt`](../requirements/ssh-a6000-cu129.txt)。其中五个关键版本对应原 H200 运行记录：Torch `2.13.0+cu129`、Transformers `5.16.1`、vLLM `0.28.1rc1.dev199+g7c5dc571c.cu129`、PEFT `0.20.0`、PyArrow `21.0.0`。vLLM 使用固定 commit 的公开 wheel，不依赖私有 registry；其它依赖也已解析为固定版本。**解析成功不等于 A6000 kernel 已验证。**

```bash
# 已有 uv 可以跳过安装。官方安装入口见下方链接。
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip sync --python .venv/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  requirements/ssh-a6000-cu129.txt
# 依赖已由锁文件安装，此处只安装项目本身。
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
mkdir -p runs/ssh-setup
uv pip freeze --python .venv/bin/python > runs/ssh-setup/pip-freeze.txt
git rev-parse HEAD > runs/ssh-setup/git-commit.txt
```

[uv 官方安装](https://docs.astral.sh/uv/getting-started/installation/)、[Python 管理](https://docs.astral.sh/uv/guides/install-python/)、[vLLM GPU 安装与固定 wheel](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。不要复制旧机的 `.venv`、`.vllm-extra`、Python 包目录或 CUDA 编译缓存；它们可能绑定另一套 Python/硬件。不要在这套环境里另行 `pip install -U torch/vllm/transformers`。

固定 wheel 不可访问时，先处理网络或从联网机器转存 wheel/缓存。不要悄悄换 `latest` 后继续标为相同运行环境。驱动升级由机器管理员处理；Python 环境脚本不会修改主机驱动。

## 4. 需要哪些模型、数据和私有训练产物

完整清单、文件大小和校验值见 [`configs/ssh-a6000-assets.json`](../configs/ssh-a6000-assets.json)。下面的路径均相对 `VPO_ROOT`。

| 路径 | 约体积 | 获取方式 / 用途 |
|---|---:|---|
| `models/Qwen3-14B-Base/` | 27.5GiB 权重 | 官方 HF 基座；Actor/vLLM 共用本地文件，不重复下载 |
| `models/Skywork-Reward-V2-Qwen3-8B/` | 14.1GiB 权重 | 官方 HF sequence-classification RM |
| `models/sft-native-eos-clean2k5e2/` | 0.97GiB 目录 | **从旧机复制**，自训 SFT LoRA，HF 不提供 |
| `datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet` | 215MiB | 官方 train_prefs，61,135 行；不是重新随机抽取的另一份 UF |
| `datasets/alpacaeval/eval_gpt4turbo_reference.jsonl` | 1.79MiB | **从旧机复制**，805 条、`gpt4_1106_preview` 参考回答 |
| `datasets/ifeval/ifeval_input_data.jsonl` | 202KiB | 官方 IFEval，541 条 |
| `datasets/gsm8k/test.jsonl` | 732KiB | OpenAI GSM8K，1319 条 |
| `datasets/sft_v2/sft_clean.parquet` | 64.4MiB | 可选，仅重新做 SFT 时需要；原 adapter 下 RL 不需要 |

**即便只做 RL，也必须准备三个 benchmark 文件**，因为代码在划分训练数据前使用它们排除 prompt 交集。默认数据应得到 59,057 个过滤后训练 prompts；`data_split.json` 中训练 hash 应为 `ea85088a7d3777bad6f47d17452a06c20c2e00cf411860167cb2dc5d21a9b739`。数量/hash 不符先排查数据与 tokenizer，不能直接与旧实验合并统计。

### 4.1 推荐：复制小型自训产物，公开模型在新机下载

先在**旧机项目根目录**打包必须传输的产物，保留相对路径：

```bash
mkdir -p runs/ssh-handoff
tar --exclude='*/.cache/*' -cf runs/ssh-handoff/private-assets.tar \
  models/sft-native-eos-clean2k5e2 \
  datasets/alpacaeval/eval_gpt4turbo_reference.jsonl
sha256sum runs/ssh-handoff/private-assets.tar > runs/ssh-handoff/private-assets.tar.sha256
```

目标机使用自己的 SSH host alias，`SOURCE_VPO` 填旧机项目绝对路径。本会话旧机路径为 `/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM`。

```bash
export SOURCE_HOST=old-server-alias
export SOURCE_VPO=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$VPO_ROOT"
mkdir -p runs/ssh-handoff
rsync -avP "$SOURCE_HOST:$SOURCE_VPO/runs/ssh-handoff/private-assets.tar" runs/ssh-handoff/
rsync -avP "$SOURCE_HOST:$SOURCE_VPO/runs/ssh-handoff/private-assets.tar.sha256" runs/ssh-handoff/
sha256sum -c runs/ssh-handoff/private-assets.tar.sha256
tar -xf runs/ssh-handoff/private-assets.tar
python scripts/prepare_ssh_assets.py --download
python scripts/prepare_ssh_assets.py --check > runs/ssh-setup/assets-check.json
```

如果两台服务器不能互相 SSH，先用你的工作站中转同一个 tar 和 SHA 文件；不需要开放额外服务。`--download` 只下载缺失的公开资产，绝不会替你重训或生成一个假的 SFT adapter；传输资产未到位时会以非零状态退出，并列出缺项。下载到不匹配的已有文件时工具会拒绝自动覆盖，先确认旧文件身份。

核心 LoRA 权重校验值：

```text
models/sft-native-eos-clean2k5e2/adapter_model.safetensors
SHA256 21c0c7b9e75c640b03a3fddfc6bb1e7a478e02c6724111800d605187364761ad
```

复制整个 adapter 目录，包括 config/tokenizer/chat template/SFT manifest，别只拿一个 `.safetensors`；别改 `adapter_config.json` 来“兼容”其他基座。

### 4.2 新机下载很慢：连同模型和数据一起 rsync

在目标机执行，先将上述 `SOURCE_HOST`、`SOURCE_VPO` 设置好：

```bash
cd "$VPO_ROOT"
rsync -aLP --relative --exclude='*/.cache/*' \
  "$SOURCE_HOST:$SOURCE_VPO/./models/Qwen3-14B-Base" \
  "$SOURCE_HOST:$SOURCE_VPO/./models/Skywork-Reward-V2-Qwen3-8B" \
  "$SOURCE_HOST:$SOURCE_VPO/./models/sft-native-eos-clean2k5e2" \
  "$SOURCE_HOST:$SOURCE_VPO/./datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet" \
  "$SOURCE_HOST:$SOURCE_VPO/./datasets/alpacaeval/eval_gpt4turbo_reference.jsonl" \
  "$SOURCE_HOST:$SOURCE_VPO/./datasets/ifeval/ifeval_input_data.jsonl" \
  "$SOURCE_HOST:$SOURCE_VPO/./datasets/gsm8k/test.jsonl" ./
python scripts/prepare_ssh_assets.py --check
```

`-L` 解引用 HF cache symlink，避免复制到新机后成为悬空链接；`/./` 保留其后的 `models/`、`datasets/` 路径。不要加 `--delete`。RM 旧目录中有未完成的下载缓存，排除 `.cache` 可以省去无用传输。

资产检查会核对模型配置/tokenizer/index的小文件 SHA、所有权重分片存在且非空，以及 LoRA/数据的 SHA。2026-09-16 已从 manifest 固定的 HF revision 下载核对这 9 个模型小文件，全部 SHA 匹配。**它没有逐字节哈希几十 GiB 的全部 Base/RM 权重**。严格位级复现可从旧机传完整模型后另做逐文件 `sha256sum`；manifest 对公开 revision 与本地来源证明的边界有明确记录。

公开来源：[Qwen Base](https://huggingface.co/Qwen/Qwen3-14B-Base)、[Skywork RM](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)、[UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)、[IFEval](https://huggingface.co/datasets/google/IFEval)、[GSM8K](https://github.com/openai/grade-school-math)。Alpaca 本地 JSONL 是三字段转换件，必须保持这份参考集身份。

## 5. 目标机验收：按顺序执行

所有命令从项目根目录、激活 `.venv` 后执行。以下 `runs/` 目录必须是新的；重复测试请换名称。

### A. CPU / 配置检查

```bash
python scripts/prepare_ssh_assets.py --check
python -m pytest -q
python scripts/run_ssh_rl.py --arm lam4 --gpus 0,1,2,3 \
  --output-dir runs/ssh-preview --dry-run
```

完整测试中某些旧测试依赖可选的本地 `Qwen3-8B-Base` tokenizer，新机没有该模型时可跳过对应测试；无需为此另下载一个 8B Actor。其余失败先定位原因。dry-run 不加载模型、不启动训练。

### B. 两卡 vLLM、BF16、词表屏蔽与真实 LoRA 生成

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_ssh_gpu.py \
  --tensor-parallel-size 2 --output-dir runs/ssh-gpu-gate
```

期望输出 `SSH_GPU_SMOKE_PASSED`，目录里有 `runtime.json`、`kernel.json`、`generations.json`、`result.json`。检查 GPU 型号、版本与 native EOS。kernel 检查故意把禁止 token logits 设为 1000，再验证严格变为 `-inf`，并分别验证 min_tokens=0/8；不是仅凭“几次采样没抽到非法 token”判断协议正确。

### C. Actor/RM 最坏长度容量检查

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/check_ssh_capacity.py \
  --output-dir runs/ssh-capacity-gate
```

这一步构造接近 2048-token 的真实模板 prompt 与 64 条 2048-token response，连续运行两轮真实 RM 梯度、VPO credit 和 Actor 更新，让第二轮覆盖持久 Adam 状态的显存开销，写 `capacity-report.json`。它是合成容量测试，不是质量实验；其中固定的测试尺度不能用于正式奖励校准。它不会保存可用于正式实验的训练模型。检查实际 prompt 长度、response 形状、成功状态、两卡 peak allocated/reserved，并留有余量；两轮通过不等于无限长训练的显存保证。

### D. 独立校准与两轮真实训练

```bash
python scripts/run_ssh_rl.py --arm lam4 --gpus 0,1,2,3 \
  --max-rollouts 2 --output-dir runs/ssh-calibration-pilot
```

默认先从同一初始 SFT 校准 128 train prompts ×8 responses；它有真实生成成本，并非瞬间完成。之后跑两轮，验证 vLLM 常驻进程、step-0→step-1 adapter 热加载、真实 optimizer 更新、最终 checkpoint，以及显存/日志没有 NaN/OOM。成功后使用校准 JSON 中的 σ₀，但**不要拿 pilot 训练后的 adapter 作为四组新实验的初始化**。

需要极短的连通性试跑时可显式 `--calibration-prompts 8`，但 8-prompt 校准不得冒充正式 128-prompt 校准。正式校准仍需单独完成。

## 6. 四组正式实验与后台运行

先从独立 pilot 提取固定尺度。四组正式实验均显式传相同 σ₀，均从原 SFT 开始，避免只有第一组消耗额外校准生成随机流。

```bash
export SIGMA0="$(python -c 'import json; d=json.load(open("runs/ssh-calibration-pilot/length_reward_calibration.json")); assert d["calibration_prompt_count"]==128; print(d["sigma0"])')"
```

先单组确认运行，再在内存和卡空闲的前提下开第二组。例如各占 4 卡：

```bash
# shell / tmux 会话 A
python scripts/run_ssh_rl.py --arm grpo --gpus 0,1,2,3 \
  --sigma0 "$SIGMA0" --output-dir runs/ssh-soft-grpo

# 另一个 shell / tmux 会话 B：独立执行，不要抢占 A 的 GPU
python scripts/run_ssh_rl.py --arm lam2 --gpus 4,5,6,7 \
  --sigma0 "$SIGMA0" --output-dir runs/ssh-soft-lam2
```

两组完成释放卡后，再跑 `--arm lam4` 和 `--arm lam8`，分别使用新的输出目录。也可以顺序运行全部四组。TP/并发设置各组保持一致；不同 GPU、TP 拆分与调度会产生数值/采样差异，不保证重放 H200 的逐 token 轨迹。

后台推荐 `tmux`。示例把命令和退出码完整留档：

```bash
tmux new -s vpo-grpo
# 在 tmux 内重新定位、激活环境并读取同一份校准。
cd "$HOME/work/VPO-RM"
export VPO_ROOT="$PWD"
source .venv/bin/activate
export SIGMA0="$(python -c 'import json; d=json.load(open("runs/ssh-calibration-pilot/length_reward_calibration.json")); assert d["calibration_prompt_count"]==128; print(d["sigma0"])')"
: "${SIGMA0:?请先完成128-prompt校准}"
mkdir -p runs/logs
set -o pipefail
python scripts/run_ssh_rl.py --arm grpo --gpus 0,1,2,3 \
  --sigma0 "$SIGMA0" --output-dir runs/ssh-soft-grpo 2>&1 | tee runs/logs/ssh-soft-grpo.log
training_status=${PIPESTATUS[0]}
printf 'training_exit=%s\n' "$training_status" | tee -a runs/logs/ssh-soft-grpo.log
# Ctrl-b 然后 d 脱离会话；重新连接用 tmux attach -t vpo-grpo。
```

如果不用 tmux，也可 `nohup python ... > runs/logs/grpo.log 2>&1 &` 后保存 `$!`，但它同样不提供训练断点恢复。不要在已有输出目录上重跑；进程退出后检查最终日志，单凭 SSH 断开或日志暂时没更新不能判断已失败。

默认每 50 步保留 adapter，每 100 步及最终保存 checkpoint。profile 还保留最新 adapter 和 step-0；`--keep-adapters-every=0` 会保留所有 adapter，容易占满磁盘。正常结束应存在最终 `checkpoint-250/` 与 `profile_summary.json`。

查看进度和输出：

```bash
nvidia-smi
tail -n 2 runs/ssh-soft-grpo/profile_metrics.jsonl
df -h .
du -sh runs/ssh-soft-*
```

关注 `optimizer_steps`、`raw_reward_mean`、`reward_mean`、`short_penalty_mean`、`long_penalty_mean`、`truncated_responses`、`newline_degenerate_responses`、`resampled_groups`、`skipped_groups`、`peak_allocated_gib`。整组无正常完成回答、重采后仍无效的 rollout 会记录 `skipped_rollout=true`、optimizer_steps=0；这是有意跳过，不是假装完成一次更新。四组因此不一定获得完全相同的有效更新数，分析时需报告。

## 7. 在新机评测

使用 Python 入口。历史 `run_*.sh` / `submit_*.sh` 中仍有集群安装逻辑或 rjob 命令，新机不要直接照搬。以下命令只使用指定单卡；先停止占用该卡的训练，或选空闲卡。评测器默认 vLLM 显存预算 80%，仍需实机留出空间。

### AlpacaEval 805 条生成

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_alpaca.py \
  --model models/Qwen3-14B-Base --output runs/ssh-alpaca \
  --dataset datasets/alpacaeval/eval_gpt4turbo_reference.jsonl \
  --max-tokens 2048 --recipes 1.0:1:1.0:-1 --seed 42 \
  --adapters sft=models/sft-native-eos-clean2k5e2 \
             grpo=runs/ssh-soft-grpo/checkpoint-250
```

这是生成步骤，不会自动购买 judge API。保存 `generations_t1.0_n1.jsonl`，其中有 response_tokens、finish_reason、stop_reason、last_token_id，可直接检查输出长度和截断率。评测 recipe 的 `top_k=-1` 是关闭截断；训练入口的 `top_k=0` 是训练协议写法，不要自行替换其它采样参数。

### IFEval 与 GSM8K

```bash
export NLTK_DATA="$VPO_ROOT/.nltk-data"
python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab

CUDA_VISIBLE_DEVICES=0 python scripts/eval_ifeval.py \
  --model models/Qwen3-14B-Base --output runs/ssh-ifeval \
  --max-tokens 1280 --recipes 1.0:1:1.0:-1 \
  --adapters sft=models/sft-native-eos-clean2k5e2

CUDA_VISIBLE_DEVICES=1 python scripts/eval_gsm8k.py \
  --model models/Qwen3-14B-Base --output runs/ssh-gsm8k \
  --max-tokens 1024 --recipes 1.0:1:1.0:-1 \
  --adapters sft=models/sft-native-eos-clean2k5e2
```

换 RL adapter 时修改 `--adapters tag=path`。当前 IFEval/GSM8K 上限沿用各自评测脚本默认，不等同于 RL 的 2048 cap；比较时保持 recipe、seed、长度、数据和 judge 一致。IFEval 的官方实现已放在 `third_party/ifeval/`；不需要重复克隆另一份实现。

### Alpaca judge 单独交接

本仓库 `judge_alpaca.py` 使用 LinkAPI 上的 `gpt-4.1`，参考回答为 GPT-4 Turbo；这是项目内部比较协议，不等同于官方 AlpacaEval2 leaderboard judge，也不会自动输出官方 length-controlled win rate。GPU 生成文件可以复制回已有 judge 环境判分。不要把 keys、账单信息或历史私有凭据放进 GitHub / 模型交接包。

若在新机判分，需要显式准备相同 judge 模板和个人 API 凭据，并先做小规模预算受限验证。下面固定的官方模板已与旧机模板逐字节核对一致，1398 bytes；模板路径使用 `--template`，不依赖旧机 `/root/.venvs/...`。

```bash
curl -fL \
  https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/15fd513071d389b79dab27fd464800c6fe10c15a/src/alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt \
  -o datasets/alpacaeval/judge_template.txt
echo '784227e6dc2832fc08c43d2c8ea3a308e7523780187a1aaad2f85e30bac85f62  datasets/alpacaeval/judge_template.txt' \
  | sha256sum -c -

# 以下会调用付费API；先自行准备该路径的私有key文件，或设置LINKAPI_KEY。
python scripts/judge_alpaca.py --gens-root runs/ssh-alpaca --tags sft \
  --template datasets/alpacaeval/judge_template.txt \
  --key-file "$HOME/.config/vpo-rm/linkapi.key" \
  --limit 25 --workers 2 --budget-cny 5
```

预算参数控制新请求派发，不是绝对预付费封顶；在途请求和账单延迟可能超出阈值。没有可用 judge 账户时，完成生成后将结果交给原有判分流程。本交接没有执行任何付费判分。

## 8. 常见问题与边界

| 现象 | 检查 / 处理 |
|---|---|
| `No matching distribution` / CUDA wheel import失败 | Python必须3.12、x86_64；确认使用完整锁文件和官方 cu129 index，检查glibc/驱动 |
| `vLLM modified the exact output support` | 当前版本屏蔽协议不匹配；不要删断言、换成 `-100` 或无限长 allowed_token_ids |
| vLLM 启动就 OOM | 检查卡上残留进程、TP2与4可见卡映射、预算85%、生成并发4；别沿用H20045% |
| Actor credit 阶段 OOM | 确认 manifest `credit_microbatch_responses=1`；只改 optimizer microbatch 无法解决旧的整批logits缓存 |
| Actor/RM在2048容量测试 OOM | 保存capacity报告与nvidia-smi；不能把缩短长度或减少8×8后所得实验标成同配置。需要进一步内存适配或明确另设实验 |
| NCCL / TP通信失败 | 检查可见卡、拓扑、驱动、共享内存；Docker检查IPC/shm；不要盲目永久关闭P2P |
| 找不到benchmark文件 | RL数据去重/排除也依赖三个benchmark，重新跑assets checker |
| 找不到SFT adapter | 从旧机复制整个自训目录，不能用公开基座替代初始化 |
| 输出目录已存在 | 用新名字；当前trainer没有完整resume，不覆盖旧metrics/checkpoint |
| 生成慢但无报错 | 先看GPU利用率和日志；A6000显存带宽/算力不同，不用H200每轮耗时预测完成时间 |
| 看不到length reward效果 | 核对 `soft` mode、σ₀、min_tokens=0及每条reward组件；旧实验快照不会随GitHub更新自动改变 |

Docker可作为另一种封装，但不是必须。需要管理员先配置 NVIDIA Container Toolkit；再把同一锁文件装进固定基础镜像，记录镜像 digest。不要使用旧私有 registry 或浮动 `latest` 代替本交接环境。Docker已经通过 `--gpus device=4,5,6,7`限制设备时，容器内应按它自己的可见编号或UUID选择卡，避免二次错误映射。[官方Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、[vLLM Docker说明](https://docs.vllm.ai/en/stable/deployment/docker/)

## 9. 交接成功的判据

1. 新机能从 `niuniu` 拉到这份文档、锁文件和全部辅助脚本；记录实际 commit。
2. `prepare_ssh_assets.py --check` 的必需资产全部 `ok`，native-EOS adapter SHA匹配。
3. 锁环境通过包依赖检查、CPU回归；GPU生成gate和2048容量gate成功，记录实际A6000型号和显存。
4. 至少两轮真实训练成功，adapter热加载、RM/VPO/optimizer/checkpoint链路完整。
5. 四个正式arm使用同一SFT、同一128-prompt校准尺度和同一协议，独立输出目录；结果按实际有效更新数和采样配置解读。
6. 至少完成SFT基线的805条Alpaca生成，能读取token长度和结束原因；付费judge另行明确执行。

在目标机跑完这些之前，本仓库提供的是**可执行的迁移交接及CPU验证过的适配**，并未声称完成了目标机部署或质量验证。
