# RL 训练与评测讲义

> 撰写于 2026-09-23（当时 `he20250b` 跑到 rollout 164/250）。写这份东西的动机：项目里
> 大部分代码是助手在若干轮里写的，作者本人需要一个「知识粒度足够细」的对照物来学。
>
> **取材**：`数学原理.md`、`工程实现.md`、`实验流程图.md`、`docs/length-reward-soft-window.md`、
> `docs/credit-controls-2026-09-20.md`、四份 `*_REPRO_NOTES.md`、`niuniu-ref/docs/evaluation.md`、
> `niuniu-ref/LOCAL-DEVIATION-L20.md`，以及对 `vpo_rm/`、`p2t/`、`red/`、`he20/`、`scripts/`、
> `niuniu-ref/scripts/` 的逐行阅读。**每条机制都给出 `file:line`**，可以顺藤摸瓜。
>
> **可信度标注**：默认都是读过代码/配置的。**我自己复核过**的关键断言标 ✅；从实测数据
> 算出来的标 📊；仅由文档或推断得到、我未能独立确认的标 ⚠️，请勿当作结论使用。
>
> **阅读顺序建议**：§0 全景观 → §1（科学内核：奖励怎么变成 token 信号）→ §2（工程躯干：
> 一个 rollout 的一生）→ §3（响应控制）→ §4（每张卡）→ §5（评测）。§6 是给未来的自己
> 留的核查清单。

---

## 目录

- [0. 全局图景](#0-全局图景)
- [1. 从标量奖励到 token 级学习信号](#1-从标量奖励到-token-级学习信号)
- [2. 一个 rollout 的一生（11 个相位）](#2-一个-rollout-的一生11-个相位)
- [3. 响应控制](#3-响应控制怎么控制-response)
- [4. 每张卡在干什么](#4-每张卡在干什么)
- [5. 评测原理](#5-评测原理)
- [6. 核查清单与已知问题](#6-核查清单与已知问题)
- [附录 A：关键文件索引](#附录-a关键文件索引)

---

## 0. 全局图景

**一句话总纲**：这个项目研究的其实是**一条标量奖励如何变成每个 token 的学习信号**
（credit assignment）。其余所有工程——分离的生成引擎、tokenizer 位置映射、长度整形、
近百项指标、置信区间——都是为了让这件事**可比较、可复现、失效可发现**。

三段流水线（`实验流程图.md`）：

| 阶段 | 做什么 | 产物 |
|---|---|---|
| **Stage 0 SFT** | UltraFeedback `train_prefs` 抽 10k 对 → Qwen3-14B-Base + LoRA r64/α128，lr 1e-4 × 2 epoch | `models/sft-p2t`（980 MB adapter）。**双职责**：所有 arm 的共同初值 + 冻结副本 = KL 锚点 |
| **Stage 1 RL** | 250 rollout ×（8 prompt × 8 回答）= 2000 prompt、16000 回答、250 次优化器更新 | 每个 arm 一套 metrics / 曲线 / checkpoint |
| **Stage 2 评测** | 四指标离线评测（独立 job） | **主张（claim）级**数字 |

**代码地图**

- `vpo_rm/` —— 母项目：VPO 方法本体 + 老的三卡训练器（HF 自回归生成，无 vLLM）。
- `p2t/`、`red/`、`he20/` —— 它的移植，每个都是**独立 package，运行时不互相 import**。
- `scripts/` + `niuniu-ref/` —— 评测（`niuniu-ref` 是另一台机器上开发后拉过来的 worktree，
  四指标工作流在那里）。

**四个 arm 的关系是「共享骨架 + 一个可插拔的 credit 插件」**：`he20/trainer.py` 与
`p2t/trainer.py` 有 446 行差异，26 个共享函数里 **18 个 AST 完全相同**，8 个不同的是
`__init__` / `resolved` / `main` / `train_rollout` / `_raw_rewards` / `save_checkpoint` /
`_dump_credit` / 诊断。✅（`HE20_REPRO_NOTES.md:261-272`）

**共享协议锚点**是 `configs/formal250.json`：250 rollout、每题 8 答 × 8 prompt = 64 条/步、
lr 5e-5、β=0.03、clip 0.2、`sigma0=3.0323000897825447`、短/长阈 8/1024、惩罚强度 0.5/2.0、
`advantage_std_floor_fraction=0.5`、`min_response_tokens=0`、`policy_head_dtype=float32`、
4 卡 device plan。`configs/he20250b.json` 与它只差 arm 专属的 `entropy_top_ratio`/`entropy_top_rule`，
以及命名——`tests/he20/test_alignment.py:81` 逐参数把守着这一点。✅

---

## 1. 从标量奖励到 token 级学习信号

### 1.1 共同地基：组内标准化（GRPO 的 A_i）

同一个 prompt 采 8 条回答构成一个「组」。四个 arm **全都**先做同一件事
（`vpo_rm/core.py:27-58`，各 arm 镜像同构）：

```
A_i = (R_i − mean_group) / max(std_pop(group), floor)

std_pop  = population std（不是样本 std）
floor    = advantage_std_floor_fraction × sigma0 = 0.5 × 3.0323000897825447 = 1.51615...
```

三个设计细节，每个都有理由：

1. **组内用 float64 计算**（`core.py:46-47`）：一个有限的 FP32 组，其均值/方差会在 FP32
   里溢出或相互抵消。
2. **分母是 `max(std, floor)`，不是 `std + eps`**（两个模式互斥，不叠加）。没有 floor 时，
   一组奖励几乎相同的回答（例如全部都落在长度窗口内）标准差异常地小，除以 ~1e-6 会把
   「一个 token 的长度差」放大成满强度的 ±1 advantage。
   `tests/test_group_advantages_floor.py::test_one_token_length_difference_is_not_normalized_to_unit_advantage`
   把这算术钉死了：L=1024/1025 时真实差距是 σ0/512，advantage 必须算出 ±1/512 而不是 ±1。
3. **σ0 是一个跨 arm 共享、冻结的数**：`calibrate_reward_scale` 在**初始策略**上、对 128 个
   prompt × 8 条回答，取「组内奖励总体标准差」的中位数（排除截断/退化行）
   （`vpo_rm/length_reward.py:74-113`）。冻结的原因见 §3：**长度窗口本身就是用 σ0 作单位
   表达的**。若某个 arm 自校准，它的长度惩罚和 advantage floor 就落在另一个绝对尺度上，
   「arm 之间的差异」可能是「长度窗口的差异」。所以每个 config 都硬写同一个 σ0，
   `p2t/trainer.py:443-447` 还会在自校准样本不足 128 时大声警告。

> **理解这个项目最重要的一把钥匙**：**组内标准化会把所有「对组内 8 条一视同仁的项」完全
> 消掉**。给一组里每条回答的奖励都 +1，梯度一点不变。所以任何偏置（RM 的系统性尺度偏差、
> 常数奖励、对全组同向的长度成本）**只有在该组内产生差异时才产生梯度**。

📊 这个 floor 不是理论问题：`p2t250` 的 `group_sigma_min` 在 **233/250 步**上等于 floor
（1.51615），`he20250b` 同样触底。

### 1.2 四个 arm 各自插在哪一层

| arm | 它替换掉的东西 | 用 RM 梯度？ |
|---|---|---|
| **VPO-RM** | token advantage：`Ã_t = A·w_t`（**乘法**，预算守恒 `mean_t w_t = 1`） | 是（输入梯度） |
| **P2T** | token advantage：`Ã_t = Â + α·R^P2T_t`（**加法**，破坏预算） | 是（同一个输入梯度） |
| **RED** | token advantage **并且**换掉整个 surrogate（RLOO/REINFORCE，无 clip，KL 折进奖励） | 否，只用 RM 的前向标量头 |
| **HE20** | **不换 credit**：group advantage 原样广播，它限制的是「哪些 token 进 loss」 | 否，只要前向分数 |

四者最终都交出同一个形状 `[B, T]` 的 `credit.advantage`（B=64，T=响应宽度），**detach 后**
送进 policy loss。这就是「唯一变量」的具体形态。

### 1.3 VPO-RM：用奖励模型的**输入梯度**造方向分，再按预算分配

**核心想法**：RM 参数冻结了，但它对**输入 embedding** 的 Jacobian 仍然可用。用
straight-through（ST）把这个 Jacobian 接到「概率」上：前向喂实际采样到的 token 的
embedding，反向按概率求导（`数学原理.md` §2）。

**第一步：把 RM 的梯度变成「每 token 的方向分」**（`数学原理.md` §3-4，代码
`vpo_rm/core.py:343-438`）

```
f_t = ∇_{e_t} R              # RM 对该 token 输入 embedding 的梯度
v_t = W f_t                  # 投到词表方向（W = RM 的 embedding 矩阵）
C_t = v_t − ⟨p_t, v_t⟩       # 词表中心化，满足 Σ_b p_t(b)·C_t(b) = 0
k_t = u_t − p_t              # log-softmax 对 logits 的 score 向量
g_t = p_t ⊙ C_t / σ          # 标准化过的奖励梯度
d_t = ⟨g_t, k_t⟩
```

工程展开成一次词表遍历就能算的形式：

```
d_t = [ p_t(a_t)·C_t(a_t) − Σ_b p_t(b)²·C_t(b) ] / σ_i
```

**它在测量什么**：沿着「提高被采样 token 概率」的方向走一小步，**标准化奖励预计会变化
多少**。所以 `d_t` 的正负和大小是「这个 token 对最终奖励的因果贡献」的一阶代理。
中心化恒等式 `Σ p C = 0` 让「提高整体概率」这件事本身不贡献信号。

**第二步：把方向分配成权重**（`vpo_rm/core.py:62-181`）

```
u_t  = A · z(d_t)          # 逐回答标准化（先按最大幅值预缩放避免溢出）
q*_t = softmax(u_t / τ)    # 解一个 KL 正则化分配问题 q* = argmax { A Σ q d − τ KL(q‖q0) }
w_t  = T · q*_t            # 归一化到 mean_t w_t = 1
Ã_t  = A · w_t             # Σ_t Ã_t / T = A：平均 advantage 不变
```

预算守恒的后果：**token advantage 保持序列 advantage 的符号与均值**（`数学原理.md` §6 证明
`mean_t Ã_t = A`）。A>0 时权重偏向 `d_t` 大的 token；A<0 时偏向 `d_t` 小的——因为降低这些
token 的概率预计收益更高。

**两个必须知道的历史坑**：

- **自适应 τ + λ 带**：`tau失配与修复方案.md` 记录了 p9c 的失败——未调温的 softmax 在
  110 个 rollout 上测得 **ESS 0.998**，也就是 VPO 实际上只是 GRPO + <2% 的扰动，机制根本
  没启动。修复是**逐回答标准化 d + 二分搜索最小的 τ 使所有权重落进 `[1/λ, λ]`**
  （`core.py:148-175`）。注意：**λ 是权重夹子，不是温度**；`credit_tau_adaptive_mean` 与
  `credit_lambda_binding` 报告它绑定的频率。
- **冻结位置权重恒为 1**：stop token、结构 token、以及在 RM 里没有精确对应位置的 token
  没有任何归因，给它们**恰好 1**（= GRPO 的每 token 系数），而不是发明信号
  （`core.py:145`）。相反的做法（把它们重新归一化掉）会同时改变 Eq.3 的分母，项目拒绝了。

### 1.4 P2T：同一个梯度，但用一阶 Taylor 归因 + 加法

论文：*Unlocking Token Rewards via Training-Free Reward Attribution*（Wu 等，CVPR），
本地文本 `paper/paper.txt`。

**归因**（`p2t/attribution.py:45-101`，论文 Eq.2）：以 RM 的 **pad token**
（`<|endoftext|>`，id 151643）为零点做位置级一阶展开

```
I_i = ∇R · (e_i − e_null)
```

只需**一次 RM 前向 + 一次输入反向**就同时得到所有 `I_i`——这就是论文「training-free」的
含义（不需要 Actor 的概率，也不需要全词表运算）。

**变成 token 奖励**（Eq.3，`p2t/reward.py:104-136`）：

```
share_t        = softmax_t(I)          # 逐回答归一化（先减 max，防 float32 溢出）
token_reward_t = R + ω·R·share_t       # ω = 0.6
Ã_t            = Â + α·R^P2T_t         # Eq.5，α = 0.1
```

**三个会让人卡住的地方**：

1. **必须喂「原始」奖励 R**，不是长度整形后的：因为 `R<0` 时 Eq.3 会奖励**归因最低**的那个
   token，语义反转（`*_REPRO_NOTES` §4 给了三条理由，这是承重的那条）。
2. **论文的 token 奖励求和不是 R**：把 Eq.3 在 N 个 token 上展开得到 `(N+ω)·R`，而不是论文
   声称的 R。凸组合会求和到 R，但论文没写凸组合。代码按论文实现并把矛盾钉进测试——实用
   后果是 `Ã` 里多了个**逐回答常数** `α·R·(1+ω/T)`。
3. **实测后果**（📊 `reports/p2t250/summary.json`，250 步）：`p2t_flat_response_fraction = 0.677`、
   **`p2t_varying_bonus_over_advantage = 4.7e-5`** —— token 之间**真正变化**的那部分是更新
   尺度的四万分之一。也就是说 **P2T 在这套配置下实际上跑成了「GRPO + 一个常数偏移」**，
   正是它的姊妹方法 p9c 当年踩的同一个坑（未调温的 softmax）。项目**刻意不加温度**去「修好」
   它：`paper-strict-fidelity` 原则——「修好的 P2T 不再是论文的方法」。

### 1.5 RED：完全不看梯度，只读 RM 的标量头

论文：*RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward
Redistribution*（Li 等，EMNLP 2025，pp. 4993-5022），本地文本 `paper/red.txt`。
算法层换成 RLOO（Ahmadian 等，arXiv 2402.14740，`paper/rloo.txt`）。

**核心想法**：RM 对**前缀**也能打分。把它的标量头在 canonical RM 行的**每个位置**都读一遍，
相邻**边界**的分数相减：

```
r̃_t = R_φ(x, y_≤t) − R_φ(x, y_≤t−1)
```

望远镜求和给出 `Σ_t r̃_t = R_φ(x,y) − R_φ(x,∅)`：整条回答的分数被**精确分解**到 token 上，
而 `R_φ(x,∅)` 是只依赖 prompt 的常数项。

**工程上比 VPO/P2T 便宜一半**：只要一次前向，不要输入反向。📊 reward 峰值显存
**14.4 GiB vs 26.8 GiB**；`phase_reward_model_prefix_sec` 中位数 0.61 s/1k tok，
P2T 是 1.17，HE20 是 0.48。

**一个不显然的推广**：论文伪代码差分**相邻位置**，这只在 policy 与 RM 共享 tokenizer 时
成立。这里 RM 池化在一个**永远不被映射**的特殊 token 上，所以代码差分**边界**
（`red/reward.py:83-122`）：`left_0 = 首个映射位置 − 1`（即 `R_φ(x,∅)`），
`left_t = right_{t−1}`，最后一个映射 token 的 `right` 取 pooled。于是每个 RM 没看过的 token
恰好贡献 0，望远镜恒等式精确成立（`tests/red/test_prefix_difference.py` 代数钉死）。

**算法层**：无 clip、无独立 KL 项（KL 已在 Eq.8 里从每个 token 奖励里减掉了），
`kl_metric` 仍然报告 `exp(d)−d−1` 作为 `kl_to_init` 以便跨 arm 比较曲线——但那**不是**奖励里
用的量。

**失败史（非常值得学习的一段）**：R3 规则 `A_{i,t} = r^final_{i,t} − b_i` 把一个**序列尺度**
的 baseline 从 **token 尺度**的奖励里减掉，留下一个巨大的逐回答常数 `−b_i`：📊 rollout 26 时
`b_i = −15.67` → 平均 token advantage **+15.66**、组内 spread 只有它的 8%、**100% 的 token
同号**、`red_advantage_flip_fraction` 恒为 **0.0000**；接着长度 451→1823、685 次截断、
奖励 +0.48→−9.17、崩成 2-token 回答、grad_norm 104，rollout 68 崩溃。同数据同 adapter 的
P2T 臂跑到 +11.04。R4 规则修了**符号通道**（把逐回答均值减掉），但**没修长度通道**。

### 1.6 HE20：不碰 credit，只限制「哪些 token 进 loss」

论文：*Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement
Learning for LLM Reasoning*（Wang、Yu 等，Qwen + 清华 LeapLab，NeurIPS 2025，
arXiv **2506.01939v2**），本地文本 `paper/high-entropy.txt`。作者参考实现是
`github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`（verl fork，commit `6cf90ceb…`）。

它的 credit **就是**把 A 广播到每个有效 token（`he20/reward.py:116-128`）：`direction` 全零、
`weight` 均匀、**`credit_ess_ratio` 恒等于 1**——而且代码明确写着这是「陈述事实，而不是假造一个
测得的数」（`he20/reward.py:20-27`）。

它真正做的是论文 Eq.(6) 的**熵 mask**（`he20/mask.py`）：

```
H_t   = −Σ_b p log p              # π_θ 的熵，只在对齐的有效词表上算，padded 位置不进排名
τ_ρ^B = quantile(H, 1−ρ)          # 池内的 (1−ρ) 分位数（论文的「阈值」规则：ties 全部保留）
I_t   = [H_t ≥ τ_ρ^B]             # 保留约 20% 的高熵 token
```

**三件必须讲清楚的事**：

1. **池是「整个 optimizer minibatch」**，即 64 条回答的全部有效 token（约 2–4 万），
   不是单条回答、也不是物理 micro-batch。论文自己说「batch」又说「(micro-)batch」，两者
   差 16 倍；代码选 minibatch 并在 `HE20_REPRO_NOTES.md` §2.2/§3 明确记录。若按物理
   micro-batch（=1 条回答）池化，就是「一条回答自己跟自己排名取前 20%」——那是**另一个方法**。
2. **必须一个 rollout 只做一次优化器更新**（`_require_single_optimizer_step`，
   `he20/trainer.py:568-599`）：mask 用的熵是 **π_θ 的**、在 old-log-prob 那一遍读一次；
   若有第二次更新，那一遍的熵描述的已是不存在的策略，代码**拒绝**而不是偷偷换成 π_old 的熵。
   `configs/he20250b.json` 由构造满足（`optimizer_minibatch_responses = 64 = 8×8`）。
   注意守卫是 `batch > minibatch` 而非 `!=`：丢了几个组的 rollout 仍然只走一步。
3. **mask 同时进分子和分母**（`he20/loss.py:42-62`）。只进分子 = 只是**重加权**
   （`v = p·(v/p)`），不是论文的限制；两者都进才是 Eq.6 的 `ΣI·surrogate / ΣI`。
   而且 **KL 不被 mask**（`he20/loss.py:26-30`）——Eq.6 只 mask surrogate，论文压根没有 KL 项，
   把 KL 也 mask 掉会悄悄收窄信赖域，比论文做的改动更大。

📊 实测：`entropy_top_kept_fraction = 0.20000000298` 每一步（阈值规则在这个分布上几乎不吃
到 ties）；kept 平均熵 1.79 vs 全体 0.574 —— mask 确实在取高熵少数。

### 1.7 一条贯穿的工程原则

**所有 credit 都是 detach + `@torch.no_grad()` + 整步缓存**（`vpo_rm/trainer.py:1000-1001`
明确写出这个契约）。原因：advantage 是这次更新的**常量**，而组统计描述的是**完整的 prompt 组**。
绝不能有梯度从 credit 路径倒流。

---

## 2. 一个 rollout 的一生（11 个相位）

设备标记：**A** = `actor_device`（`cuda:0`）、**R** = `reward_device`（`cuda:1`）、
**V** = vLLM 进程（`cuda:2,3`，只通过 socket 与文件系统接触）。

| # | 相位 | 设备 | 做什么 / 缓存什么 |
|---|---|---|---|
| 0 | prologue | — | `zero_grad`、清 A 的 cache、拒绝 σ0 缺失 |
| 1 | 选 prompt + 生成 + 筛选 | A + V | 缓存 8 元组 rollout |
| 2 | 落盘 | 磁盘 | `rollout-N-{tokens,prompts,rewards}.json` |
| 3 | **RM 前向** | **R** | 逐条算 `R_φ(x,y)`，缓存 `raw_rewards [B]` |
| 4 | 守卫 | — | `_require_single_optimizer_step`（在花任何前向成本**之前**） |
| 5 | 整形 + credit | **R** | 长度惩罚、退化 floor、组标准化；`credit.advantage/direction/weight [B,T]` |
| 6 | **old-logp 前向**（+熵） | **A** | `old_logp [B,T]`、`entropy [B,T]`，都是 θ_old |
| 7 | Eq.6 mask 构建 | A | `entropy_top [B,T]` |
| 8 | **ref-logp 前向** | **A** | `ref_logp [B,T]`（挂 `ref` adapter） |
| 9 | 采样器→训练器校正 | A | `importance [B,T]` |
| 10 | **更新** | **A** | 64 次 micro 前向/反向 → 一次 `step()` |
| 11 | metrics / checkpoint / autopush | A/R | 见 §2.5 |

`phase()` 闭包在每个相位边界**同步两个 CUDA 设备**再取时间，所以每个 `phase_*_sec` 都是真实
墙钟时间；📊 `elapsed_sec` 精确等于 7 个相位之和（100% 可归因，没有未计量的工作）。

### 2.1 生成：vLLM 是一个独立进程

- **进程模型**：`he20/vllm.py:195-232` 用 `subprocess.Popen` 起 `python -m he20.vllm_server`，
  经 **unix socket**（`<output_dir>/vllm.sock`）通信，tensor-parallel=2 吃掉两张卡。
  `max_model_len=4096` 正好 = prompt 2048 + response 2048。子进程的
  `CUDA_VISIBLE_DEVICES` 就是 config 里的 `vllm_gpus`。
- **LoRA 热插拔**：每个 rollout 把当前 adapter 存成 `vllm-adapters/step-{rollout_index}`，然后发
  `LoRARequest`。**路径每步都不同是刻意的**——复用同一个路径会让服务端命中缓存，悄悄把
  rollout 变成 off-policy（`he20/trainer.py:421-425`）。保留 2 份是因为「写新、切服务、删旧」
  之间需要缓冲；每份 **2.0 GiB**（可训练的 `default` + 嵌套的冻结 `ref/`）。
- **采样面**：`temperature=1.0, top_p=1.0, top_k=0, min_tokens=0, presence_penalty=0.0`。
  **267 个未注册的 embedding 行被 `logit_bias=-inf` 封禁**（词表 151,936 行、注册 151,669 行），
  actor 前向里也 `masked_fill(-inf)`，事后还会复验。
  presence penalty 在三处被拒绝：请求里硬编码 0、`vllm_sampling_kwargs` 拒绝非零、
  server 拒绝环境变量（并且启动器把该变量从子进程环境里 pop 掉）。
- **stop token 有两个**：`<|endoftext|>`(151643) 和 `<|im_end|>`(151645)。后者是给 SFT 出来的
  策略用的——没有它，回答会冲过 `<|im_end|>` 继续写。
- **chosen-token logprob 会被带回来**（`rollout_logprobs`），这是整个更新的锚点（见 §2.4）。

### 2.2 筛选：什么样的回答可以被训练

`rollout.py:139-196`：一条回答「可用」的条件是**自己停下来了**（`finish_reason == "stop"`）
且非空、非退化。一个组里**全员不可用** → 重采样**一次**；还不行 → 整组丢弃，这次 rollout
少几行继续。若什么都没剩，返回 `skipped_rollout`（`optimizer_steps: 0`），
**完全不更新——包括 KL 项、Adam 动量和 weight decay**。

📊 你这条 run 164 步里 `resampled_groups` 与 `skipped_groups` 都是 0。

### 2.3 三遍 actor 前向：为什么需要三遍

1. **old-log-prob pass**：产生 `old_logp`（clipped ratio 的锚点）**和** `entropy`
   （Eq.6 排名的总体）。花掉约 15% 的时间。
2. **mask build**：只有 HE20 有。
3. **ref-log-prob pass**：挂 `ref` adapter 算 `ref_logp`，给 KL 用。又是约 15%。
4. **training forward**：唯一带梯度的一遍，`new_logp` 按物理 micro-batch 重算。

有意思的是：因为输出头被强制成 FP32（§4），`old_logp` 与 training forward 在 θ_old 处
**逐位相同**（📊 `initial_hf_logp_max_abs_error = 0.0` 贯穿全程）。所以 old-logp 那一遍
**不是数值必需**，而是（a）验证这个恒等式、（b）在更新前用 π_θ 供熵、（c）让 ratio 锚在
采样器自己的 logprob 上。**整个 mask 的合法性是拿这一遍买来的**——这就是为什么「一步一更新」
守卫必须存在。

### 2.4 loss 与反向

```
log_ratio = new_logp − old_logp.detach()
# 在 log 空间 clip：exp(clamp(log r, max=log1p(+eps)))
#   ↑ 避免先 exp 再乘造成的 0*inf = NaN 反向（core.py:504-506）
objective = sign(a)·exp(selected_log_ratio + log|a| + log w)
loss      = −(objective / kept_counts.clamp_min(1)).mean(0).sum()
#   先逐回答 token 平均，再对回答平均；某条回答一个 token 都没保留 → 贡献恰好 0
+ β · KL(θ‖θ_ref)，   KL 估计器 exp(d) − d − 1，d = log p_ref − log p_new
```

- **重要性权重** `w = exp(old_logp − rollout_logprob)`（`he20/trainer.py:777`）：
  这是把 vLLM 采样器的概率接到 HF 训练器的桥。📊 `rollout_is_mean ≈ 0.9999`、ESS 0.999。
- **反向**：64 个物理 micro-batch 各自 `backward()`，梯度按 `(hi−lo)/(end−start)`
  （= 每 minibatch 均值）加权累积，然后 `clip_grad_norm_(1.0)`，**一次** `step()`
  （AdamW，lr 5e-5，wd 0.01）。动的参数**只有** LoRA 的 A/B 矩阵；backbone、FP32 输出头、
  `ref` adapter 全部冻结。
- 非有限的 loss 或梯度 → `zero_grad` + raise，这一步作废。

### 2.5 落盘

- **metrics**：每个 rollout 一行 JSON（98 个键），追加到 `reports/<run>/metrics.jsonl`。
- **逐 rollout dump**：`rollout-N-{prompts,tokens,rewards}.json` +
  `rollout-N-credit.pt`（含 `w/d/advantage/raw_reward/entropy_top_mask/protocol`）。
  HE20 的 `d` 全零、`w` 均匀；而 P2T 的 `"i"` 归因字段是**省略而不是填零**——因为填零会被
  读成「真的归因等于 0」。
- **checkpoint**（每 20 步 + 最后一次）：两个 adapter + tokenizer + `run_manifest.json`，
  里面明确写 `"resume_supported": false` 和一份 restart 配方（哪些能带走、哪些不能）。
- **autopush**（每 5 步 + 最后一次）：`git add -A` → 提交 → push 到 `p2t-origin main`。
  **`git add -A` 是已知的坑**：它会把工作树里不相干的文件扫进「he20 step N」的提交里。
  已知的一次实际破坏：第一次 `he20250` 的 step-5 提交删掉了 `plots/p11_latest.png` 和
  `reports/red250b/*`。✅ 重启后的 `he20250b` 从 step 5 到 160 的提交我核过，没有再出现
  越界路径（只多带了 `configs/he20250b.json`、checker 校准、以及 `reports/he20250/*` 那 5 个
  脏文件——后者是我预期内的，见 §6）。

---

## 3. 响应控制（怎么控制 response）

**一句话：不是 mask、不是强制停止，而是从奖励里扣分。**

`docs/length-reward-soft-window.md` + `he20/length_reward.py:31-57`：

```
L       = 生成的响应 token 数（含 EOS，不含 prompt）
P_short = 0.5 · σ0 · clip((8    − L)/(8),            0, 1)   # 上限 1.516 RM 单位
P_long  = 2.0 · σ0 · clip((L − 1024)/(2048 − 1024),  0, 1)   # 上限 6.065 RM 单位
reward_shaped = R_RM − P_short − P_long
```

用 σ0 作单位的意义：**长度整形对 RM 的绝对尺度不变**——换一个奖励模型，「太短/太长」的代价
仍然是同一个相对量级。

**几个必须理解的设计点**：

1. **只有非负成本，没有长度奖励**。没有任何东西因为回答长或短而**被付款**；惩罚只在组内
   **区分**长度。长惩罚（2σ0）是短惩罚（0.5σ0）的 4 倍，是对「不要冲爆上限」的故意偏置。
2. **唯一能对长度起作用的通道是「组内差异」**（回到 §1.1 的钥匙）：组内标准化的 mean 会被
   减掉，所以对所有回答一视同仁的长度成本**不产生任何梯度**。只有「同样 8 条回答里谁更长」
   才产生梯度。
3. **`min_response_tokens = 0`，而且非零值在 config 解析阶段被拒绝**（`he20/trainer.py:205-211`）：
   soft 协议下没有理由抑制 EOS，而且非零值会额外触发姊妹臂从不使用的 legacy 退化 floor。
   `min_response_tokens=0` 时 `sampling_logits` 直接短路——采样器和训练器都**不**做
   stop-token 抑制。
4. **退化守卫**：`empty = text.strip()==""`、`repeated = "\n"*32 in text`，只按**解码后的文本**
   判断（特殊 token 跳过）。组内每条退化回答的奖励被压到「该组最好的非退化奖励 − 1.0」；
   **整组都退化则 raise**。它作用在**整形后的序列奖励**上，从不作用在 RM 自己的分数上。
5. **截断**：`finish_reason == "length"` 的回答**不可用于训练**（所以全截断的组会被重采样/
   丢弃），会被计数，并且自动吃满长惩罚（L=2048 ⇒ `P_long` 取最大）。项目**没有** DAPO 的
   overlong buffer。📊 你这条 run 全程 `truncated_responses = 0`。
6. 一个反直觉的观察：你这条 run 的平均回复长度从 ~314 涨到 ~540 token。
   **这不是目标函数付钱买来的**——soft window 在 1024 以下没有任何项，所以 1024 以内的增长
   完全由 **RM 自己的偏好**驱动（长度惩罚在 1024 以下恒为 0）。

**credit 消融三条对照**（`docs/credit-controls-2026-09-20.md`，理解 credit 到底贡献了什么）：

- **Random**：随机方向。
- **Shuffle**：先算出 canonical 的输入梯度/方向/权重，再在**每条回答内部、跨所有有效位置**
  独立置换权重。**权重的多重集、总和、上下界、ESS 全部保持不变**，只有「哪个权重落在哪个
  token 上」被打断。
- **Norm product**：把 `d_t = ⟨g_t, k_t⟩` 换成 `‖g_t‖·‖k_t‖`——去掉梯度对齐（含余弦符号）。
  不是 `|⟨g,k⟩|`，也不是单独的 embedding 梯度范数。

---

## 4. 每张卡在干什么

| 角色 | config 键 | 值 | 物理卡（`HE20_GPUS=0,1,2,3`） | 实测峰值 |
|---|---|---|---|---|
| actor + 优化器 | `actor_device` | `cuda:0` | **0** | `actor_peak_gb` **39.55** |
| 冻结的 RM + 打分 | `reward_device` | `cuda:1` | **1** | `reward_peak_gb` **14.43** |
| vLLM 生成 | `vllm_gpus` | `["2","3"]`, tp=2, util 0.85 | **2, 3** | trainer 不记录 |

**语义差别（这是个坑）**：`actor_device` / `reward_device` 是 `CUDA_VISIBLE_DEVICES` 的
**索引**，而 `vllm_gpus` 是**物理 id**——因为它被原样交给子进程当它自己的 `CUDA_VISIBLE_DEVICES`。

**device-plan 门禁**（`he20/trainer.py:1221-1227`）：若 `CUDA_VISIBLE_DEVICES` 非空，它必须
**恰好**等于 `['0','1','2','3']`（= `2 + vllm_tensor_parallel_size`）。不一致就报错，理由是
「否则 trainer 跑在一组卡上、生成在另一组卡上，可能是别人的 job」。注意它**以『环境变量非空』
为条件**——所以不带 `HE20_GPUS` 启动（config 里写绝对 id）会**跳过**这个检查。

**39.55 GB 的组成**：28 GB bf16 backbone + **3.1 GB FP32 输出头**
（151,936 × 5,120 × 4B，因为要把采样器和训练器的概率变成同一个协议）+ LoRA 梯度 + 激活。
它是只增不减的高水位线，随回复变长从 34.7 涨到 39.6 GB。

**时间账**（📊 最近 20 步）：

| 相位 | 均值 | 占比 |
|---|---|---|
| `phase_actor_update_sec`（64 次前向+反向、clip、step） | 203.6 s | **50.5%** |
| `phase_actor_old_logp_sec` | 61.7 s | 15.3% |
| `phase_ref_logp_sec` | 61.5 s | 15.3% |
| `phase_generation_sec`（含 2 GiB adapter 写盘 + socket） | 59.4 s | 14.7% |
| `phase_reward_model_forward_sec` | 16.9 s | 4.2% |
| `phase_credit_sec` | 0.027 s | ~0 |
| `phase_logging_sec` | 0.003 s | ~0 |
| **`elapsed_sec`** | **403 s**（≈6.7 min） | 100% |

**三遍 actor 前向占了 81%**；两遍 log-prob（123 s）存在的唯一目的是产出 `old_logp`/`entropy`/
`ref_logp`。

> **一个跨 arm 的读法提醒**：相位名与 p2t 共享，**只有一个是不同的**——p2t 写
> `phase_reward_model_gradient_sec`（它的 RM 那一遍带输入反向），HE20 写
> `phase_reward_model_forward_sec`（母项目给纯前向的名字）。跨 arm 比较相位时必须做映射。

---

## 5. 评测原理

### 5.1 四个指标，各测什么

`niuniu-ref/scripts/run_benchmarks.py:17`：`METRICS = ('rm_reward', 'alpacaeval', 'ifeval', 'arena_hard')`

| 指标 | 测什么 | 怎么算 | 关键局限 |
|---|---|---|---|
| **rm_reward** | 冻结的 Skywork RM 有多喜欢这个策略的输出（256 个固定 prompt） | vLLM 生成 1 条/题、t=1.0 → RM 逐条打分 → 均值 + bootstrap 95% CI（2000 次重采样、seed 0） | **这就是训练奖励本身**，只是换了 held-out prompt。优化 RM 必然抬高它 → 它是**进度指标，不是独立证据** |
| **alpacaeval** | 裁判是否更喜欢候选回答而非固定的 GPT-4 Turbo 参考（805 题） | 裁判只回 1 个 token，取 `P("m")/(P("m")+P("M"))` 得**对数概率加权的胜率** | **不是 length-controlled win rate**！代码里没有逻辑回归、没有长度项，`summarize_benchmarks.py:190` 自己写着 `metric_note: "Logprob-weighted preference, not length-controlled win rate."` |
| **ifeval** | 可程序化验证的指令遵循（541 题 / 834 条指令检查） | **官方 Google 检查器**（vendored 在 `third_party/ifeval/`），四条官方口径：prompt/inst × strict/loose | 只测规则遵从，不测正确性/相关性/事实性；loose 模式（8 种变体任一通过即算过）会让真违规通过；语言/大小写检查器在异常时 **fail open** |
| **arena_hard** | 与 `gpt-4o-mini-2024-07-18` 参考的成对胜率（750 题 = 500 hard + 250 creative） | 两局换序、官方模板、regex 解析，再用**官方 Bradley-Terry 数学**做 style control（`third_party/arena_hard/utils/math_utils.py`，hash 校验后加载） | style 系数是**在候选集合上联合拟合**的——单独汇总一个模型会被扭曲；CI 是 **CI90** 不是 95%；LLM 裁判不是执行验证 |

**style control 的意义**（Arena）：更长、更多格式的回答可能在观感上就赢了；拟合出的 style
系数把这些混杂因素扣掉，再在「零风格差异」处读胜率。

**IFEval 的两个项目自造聚合**：`project_four_metric_mean_pct`（四个百分比的等权平均，
**按生成种子分别算再对 5 个种子平均**）与 `generation_seed_sd_pp`（这 5 个复合值的样本标准差，
单位百分点）。两者**都不是官方 IFEval 指标**，`docs/project-status-2026-09-17.md` 明确说了。
5 个生成种子是为了给**策略采样的随机性**定价，而检查器身份保持固定。

### 5.2 两类指标：训练奖励 vs held-out 主张

这是最该学会的区分。`工程实现.md` §2 说得对：**「训练 RM 分数作为训练指标，独立评价用于衡量
实际质量」**。

- **(a) 训练信号**：`shaped = raw_rewards − short_penalty − long_penalty`（+ 退化 floor）。
  RM 分数与整形分数**都**记进 metrics（`raw_reward_mean` 与 `shaped_reward_mean`）。
- **(b) 主张**：只有 `alpacaeval` / `ifeval` / `arena_hard` 是由**不是训练奖励模型**的裁判或
  检查器产生的（GPT-4.1、官方规则检查器、GPT-4o）。
- **`rm_reward` 卡在中间，是个陷阱**：它用的是**同一个 checkpoint 的同一个 RM**
  （`eval_checkpoints.py:272` 默认路径与训练相同，manifest 也确认）——它只是「训练奖励在
  held-out prompt 上重新测一遍」。

代码用四种机制把它和主张分开：行里带
`reward_metric: "raw_skywork_scalar_no_length_or_kl_penalty"` 与
`reward_input_protocol: "canonical_chat_v1"`；评测路径**完全不 import** 长度奖励模块；
每个指标写自己的 `<metric>/metric_summary.json`；prompt 集按构造 held-out。

### 5.3 无重叠规则

训练集 = `train_prefs` 去掉与 alpaceval / ifeval / gsm8k 规范化后**精确匹配**的 prompt
（本数据集去掉 8 条），然后按 `sha256(规范化 prompt)` **升序**排，**前 2000 条保留为验证集**。
评测用的 256 个 prompt 就是这 2000 条里的前 256 条。

排序而非洗牌的理由（代码注释）：**让切分独立于输入顺序，同一个语料永远给出同一个训练集**。

✅ 子代理用仓库自己的 loader 实测验证过：`frozen == valid[:256]` 为真，与训练集交集为 0，
与 `valid[256:]` 交集为 0。

**但要知道它的边界**：这是**精确匹配**去污，没有语义级或预训练污染控制；而且
**arena-hard 的题目根本不在训练侧的排除表里**——它们的「不在训练集」是来源论证，不是检查。

### 5.4 本机（L20）的偏离——哪些数字不是 pristine

这一节直接决定能不能把 `runs/eval-qwen3-14b-base/` 的数字当作可信结果：

1. **`gpu_memory_utilization` 0.45 → 0.80**（RM 指标）：46 GB 卡上 0.45×46 = 20.7 GiB <
   Qwen3-14B 的 ~29.6 GiB bf16 权重，EngineCore 根本起不来。这是**硬编码字面量、没有 CLI/env
   覆盖**（只有 `EVAL_PP`/`EVAL_TOPP` 从环境读），同文件夹的三个兄弟脚本都用 0.80。
   不改 prompts / seed / temperature。
2. **flashinfer 采样器被禁用**（`VLLM_USE_FLASHINFER_SAMPLER=0`）：它的 JIT 给 `nvcc` 传
   `--compress-mode=size`，本机 CUDA 12.2 拒绝 → 所有 generate 阶段死掉。这是性能开关，
   不改变指标定义。**注意：这个变量是操作者 shell 导出的，不在仓库代码里。**
3. **urllib 连不上网、curl 可以**：`prepare_evaluation_assets.py:51` 的 `urlopen` 报
   `Network is unreachable`，而 curl 能拿到同样的 URL。两个 Arena 资产是用 curl 拉的，
   SHA256 与字节数逐项匹配 manifest。
4. **`models/` 软链**：LoRA adapter 记录的是**相对**路径 `models/Qwen3-14B-Base`，评测脚本按
   **它自己的 checkout 根**解析 → 在 `niuniu-ref` 评测 `baseline` 的 adapter 必须有
   `niuniu-ref/models → baseline/models`。另外 `third_party/arena_hard` 在 `baseline/` 里是
   **空的**（子模块没检出），所以 **Arena 打分只能在 niuniu-ref 里跑**。
5. **Arena 裁判实际是 `gpt-4o`，不是 `LOCAL-DEVIATION-L20.md` 里写的 `gpt-4.1`**，而且覆盖是
   **732/750**（排除 18 题）。证据：`arena_hard/*/judgments/state/protocol.json` 与
   `metric_summary.json` 自己写着 `judge: "gpt-4o"`，被排除的 UID 与日志里的 `ALL_EXCLUSIONS`
   完全一致。**偏离文档的 §3 已过时，以 `metric_summary.json` 为准。**
6. 两个脚本有**未提交的本地修改**（`eval_checkpoints.py` 的内存预算、
   `summarize_benchmarks.py` 的排除开关），而且**没有任何启动脚本被提交**——`run.log` /
   `rest.log` 记的是手工敲的命令，`benchmark_config.json` 是唯一机器可查的运行记录。

### 5.5 什么会让一次评测失效（清单）

1. 把 `rm_reward` 当成质量证据（它是训练奖励重测）。
2. 把 AlpacaEval 说成「length-controlled win rate」（它不是；更长的回答会在这里占便宜）。
3. 用不同覆盖率或不同裁判的 Arena 数字互相比较（732/750 ≠ 750/750；裁判模型是协议身份的一部分）。
4. 一次只汇总一个模型来做 style control。
5. 跨 prompt 集 / 配置比较（有防护：`benchmark_config.json` 不匹配是硬错误，各汇总器拒绝部分
   覆盖——Alpaca 要求恰好 805 题且 0 解析失败、IFEval 要求 5 个种子文件齐全、RM 要求 n=256）。
6. 重跑或手改缓存（有 SHA256 与逐记录 digest 防护；**但模型身份就是 tag 字符串**——同一个 tag
   下换 adapter 是分不出来的）。
7. **thinking 模式悄悄打开**：`_render_chat_prompt` 请求 `enable_thinking=False`，但如果模板
   拒绝这个 kwarg 就会**回退到不带它的调用**。要验的是 `prompt_token_ids_sha256`，不是代码路径。
8. 裁判版本漂移对你不可见（只有记录下来的 `judge_model` 字符串是唯一线索；provider 端点并未
   按 revision 固定）。
9. 混淆两类成本：`--budget-cny` 是**每次裁判调用的派发护栏**（读余额接口），不是总额、也不是
   硬上限——计费有延迟、在途请求会超。

---

## 6. 核查清单与已知问题

### 6.1 我核验过的三件事

1. **文档漂移（本文已按代码纠正）** ✅
   - `HE20_REPRO_NOTES.md:218` 引用的 `tests/he20/test_conformance.py` **不存在**（实测目录里
     只有 `test_alignment.py` / `test_loss.py` / `test_mask.py` / `test_trainer.py`），真正的断言
     在 `test_loss.py`（未 mask 的 loss 与 P2T 逐位相同，`atol=0, rtol=0`）与 `test_alignment.py`。
   - `p2t/loss.py:7`、`p2t/reward.py:77` 引用的 `tests/test_conformance.py` 少了 arm 子目录
     （真名 `tests/p2t/test_conformance.py`）。
   - `工程实现.md` §3/§4 说 credit 在 GPU 1 上从旧 logits 构造，而 `vpo_rm/trainer.py:633`
     明确把 RM embedding 拷到 **actor 卡**上算（`_credit_cache_microbatch`）。**P2T 才是在
     reward 卡上算归因**——两个 arm 真的不同，而且没有测试能抓住这个回归（CPU 测试把两者
     都钉在 `cpu`）。
2. **阈值校准已进 git** ✅  commit `e10b770`（"he20 step 5"，09-22 21:36）把
   `configs/he20250b.json`、对 `he20/scripts/check_he20_health.py` 的阈值校准、以及
   `reports/he20250/*` 那 5 个脏文件一起提交了——正是 autopush `git add -A` 的已知行为。
   所以它**不再**是「工作树里的未提交修改」。
3. **`rm_unmapped_content_fraction` 是偶发尖峰，不是漂移** ✅📊
   160 个 rollout 里只有 **7 次**超过 1%（r77、r97、r101、r109、**r113 = 0.2101**、r114、r160），
   其余在 1e-4 量级。尖峰与**长回答**相关（尖峰时均 736 token vs 全局 574）。
   **机制尚未确定**——训练器里这个门禁**只在 rollout 0 跑一次**，真正的护栏是外部的
   `check_he20_health.py:290-294`（阈值 0.25）。
   → **run 跑完后值得单独查**：看 r113 那条 rollout 的 prompt/tokens，以及
   `rm_max_input_tokens` 是否接近 4096 预算。

### 6.2 未来重跑任何 arm 之前

- [ ] 确认 `he20/scripts/check_he20_health.py` 的两条趋势线阈值（length −180 / entropy −0.30）
      是否为当前值——它的 `rate()` 返回**原始半-半位移**，而它抄来的
      `scripts/check_run_health.py:47` 返回**每-rollout 速率**，两者共用同一个字面阈值会让这条
      线紧 5 倍（window=10 时）。这条 bug 曾以误报 SIGTERM 掉第一次 `he20250`。
- [ ] 本 trainer **没有 in-place resume**（`resume_supported: false`）。checkpoint 只能作为
      「新 run 的种子」，而那会**重新锚定 beta-KL** 并把不同的 `init_adapter` 写进每一行 metrics，
      破坏跨 arm 可比性。重启前先算清楚代价。
- [ ] `check_fresh_output` **只检查 `output_dir`**，不检查 `report_dir`。一个 output 新、report 旧的
      config 会往旧 metrics 里追加，并让已 armed 的 watcher 在第一次检查时 SIGTERM 掉新 run。
- [ ] `start_he20.sh` 的 pidfile 检查**不会**发现输出目录已被占用（旧 pid 已死就能过），
      真正的守卫是 trainer 的 `FileExistsError`。
- [ ] autopush 会 `git add -A`：提交前先 `git status` 看清工作树，否则不相干的文件会进
      「he20 step N」的提交并被推上去。

### 6.3 评测侧待办

- [ ] **HE20 还没有四指标评测**：`reports/he20250b/` 只有训练指标。目前这条 arm 只有训练奖励
      曲线，`runs/eval-qwen3-14b-base/` 里唯一完成的 tag 是 `p2t`（= `runs/p2t250/checkpoint-250`）。
- [ ] Arena 的 style control 要**把各 arm 一起汇总**（候选池依赖），并且记住现有那个数字是
      **gpt-4o 判的、732/750 覆盖**的。
- [ ] 要报「length-controlled win rate」的话，得先真的实现它——现在这条路径上没有。

---

## 附录 A：关键文件索引

| 想知道什么 | 去哪看 |
|---|---|
| VPO 的数学推导（代理梯度、分配） | `数学原理.md` |
| 两卡工程与实验设计（历史版） | `工程实现.md` |
| SFT→RL→评测三段流程图 | `实验流程图.md` |
| 长度窗口的公式与理由 | `docs/length-reward-soft-window.md` |
| credit 三条消融的定义 | `docs/credit-controls-2026-09-20.md` |
| τ 失配与 Plan B 修复 | `tau失配与修复方案.md` |
| σ 归一化的重复/抵消 | `sigma归一化核对.md` |
| 四个 arm 的保真记录与偏离 | `P2T_REPRO_NOTES.md` / `RED_REPRO_NOTES.md` / `HE20_REPRO_NOTES.md`（各 arm 目录下） |
| 训练骨架的规范实现 | `he20/trainer.py`（← 移植自 `p2t/trainer.py` ← `vpo_rm/trainer.py`） |
| 组内 advantage 与分配 | `vpo_rm/core.py:27-58`（advantage）、`:62-181`（allocate）、`:343-438`（credit）、`:474-524`（loss） |
| RM 的输入反向 | `vpo_rm/reward.py:52-69`；`p2t/rm.py:127-165`（P2T 版） |
| RED 的前缀分数 | `red/rm.py:70-105`（前缀打分）、`red/reward.py:83-122`（边界差分） |
| HE20 的 mask | `he20/mask.py`、`he20/loss.py:42-62` |
| 评测协议权威 | `niuniu-ref/docs/evaluation.md` |
| 本机 L20 的偏离 | `niuniu-ref/LOCAL-DEVIATION-L20.md`（**Arena 一节已过时，以 `metric_summary.json` 为准**） |
| 怎么跑四指标 | `niuniu-ref/scripts/run_benchmarks.py`（`--stage {generate,judge,score} --metric … --tag … --adapter …`） |
| 该 arm 期望的评测口径 | `p2t/README.md:82-142` |
