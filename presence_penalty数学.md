# Presence Penalty 的数学定义与理论代价

> 对应讨论:VPO p9g/p9h 坍缩分析、p9 系列归因矩阵。PP 在实验中的角色:p9h 换行刷屏坍缩的止血带(2026-09-11 引入),后经 p9k 定价确认为 reward 负资产(−1.2 ~ −1.8 分)。
> 代码位置:`scripts/vllm_generate_server.py` 的 `SamplingParams(presence_penalty=0.3)`,仅作用于 rollout 采样。

---

## 一句话总结

PP 只改采样分布、不改任何训练组件;数学后果是丢掉了重要性权重 $q/\pi$,使梯度成为**有偏估计**——系统性降权"爱重复的序列"、升权"不重复的序列",净效果等于在优化一个"反重复先验下加权"的目标,而非策略真实目标。实测这份偏差 + 诱发的更多截断,合计约 −1.2 ~ −1.8 分。

## 1. 定义

设策略在位置 $t$ 的原始 logits 为 $\ell_t(b)$(候选词 $b$),已生成序列的去重词集为

$$S_t = \{a_1, \dots, a_{t-1}\}$$

指示函数 $\mathbb{1}[b \in S_t]$ 表示"词 $b$ 在本条回答中出现过(至少一次)"。修改后的 logits:

$$\tilde\ell_t(b) = \ell_t(b) - \theta \cdot \mathbb{1}[b \in S_t], \qquad \theta = 0.3$$

实际采样分布:

$$q_t(b) = \frac{\exp \tilde\ell_t(b)}{\sum_{b'} \exp \tilde\ell_t(b')}$$

**要点**:
- 出现过一次的词,logit 罚固定值 0.3,**与出现次数无关**——按次数线性罚的是另一个参数 frequency_penalty(我们未使用);
- 只作用于**采样**这一步。策略网络参数、RM 打分、credit 计算($d_t$、$\lambda$ 带)、policy loss,全部不知道 PP 的存在;
- eval(`eval_checkpoints.py`)不设 PP——最终评测是纯策略分布口径。

## 2. 它为什么"亏分":丢掉的重要性权重

训练想优化的是策略自身分布下的期望回报:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[A(y)]$$

但样本现在来自被 PP 修改的分布 $q_\theta \ne \pi_\theta$。实际使用的梯度估计器(policy gradient 形式)是:

$$\hat g = A(y)\, \nabla \log \pi_\theta(y), \qquad y \sim q_\theta$$

对其取期望,通过换元 $y \sim \pi$ 重写:

$$\mathbb{E}_{q}[\hat g] = \mathbb{E}_{\pi}\Big[\underbrace{\frac{q_\theta(y)}{\pi_\theta(y)}}_{\text{被丢掉的重要性权重}} \, A(y)\, \nabla \log \pi_\theta(y)\Big] \;\ne\; \nabla J(\theta)$$

**偏差的形态**(展开到 token 级):

$$\frac{q(y)}{\pi(y)} = \prod_{t} \frac{q_t(a_t)}{\pi_t(a_t)}$$

- 每选中一个"重复词",该步的 $q_t/\pi_t < 1$ → 整条序列在梯度中被乘上一个小于 1 的因子;
- 爱重复的序列:$q < \pi$ → **系统性降权**;不重复的序列:$q > \pi$ → **升权**;
- 净效果:我们优化的实际上是 $\mathbb{E}_\pi\!\big[\tfrac{q}{\pi} A \nabla\log\pi\big]$,即一个**"反重复先验"加权的目标**,不是策略真实目标 $J$。

## 3. 实测代价的分解(p9 系列)

$$\underbrace{(\lambda 2{+}\mathrm{PP}) - \mathrm{GRPO}}_{-0.35} \;=\; \underbrace{(\lambda 2{+}\mathrm{PP}) - (\mathrm{GRPO}{+}\mathrm{PP})}_{+1.43\;(\text{credit 机制净增益})} \;+\; \underbrace{(\mathrm{GRPO}{+}\mathrm{PP}) - \mathrm{GRPO}}_{-1.79\;(\text{PP 代价})}$$

PP 代价(−1.79)的两个来源:
1. **上述系统偏差**(理论部分);
2. **被逼出来的截断**:PP 推高多样性与长度 → 截断率 8–13% vs 裸 GRPO 的 4.6%,被 floor 压分。

## 4. 与 λ 带的对照:为什么 PP 特殊

| | λ 带 | PP |
|---|---|---|
| 改什么 | 训练信号**怎么分配**(credit 权重) | 训练数据**从哪来**(采样分布) |
| 采样分布 | 仍是纯策略 $\pi_\theta$ | 变成 $q_\theta \ne \pi_\theta$ |
| 梯度性质 | **无偏**(标准 policy gradient,只是 advantage 形状变了) | **有偏**(丢重要性权重) |

这也是 p9l(VPO λ=2、去 PP)在理论上被看好的原因:去掉的是体系中**唯一的偏差源**;留下的 credit 增益(同 PP 背景下 +1.43)是无偏部分。

## 5. 一页总结

| 问题 | 答案 |
|---|---|
| PP 在哪 | vLLM SamplingParams,仅 rollout 采样时;策略/RM/credit/loss/eval 全不涉及 |
| 公式 | $\tilde\ell_t(b) = \ell_t(b) - 0.3\cdot\mathbb{1}[b\in S_t]$,再 softmax;按"出现过"罚,与次数无关 |
| 为什么亏分 | 样本来自 $q \ne \pi$ 但按 $\pi$ 的梯度形式估计 → 丢重要性权重 $q/\pi$ → 有偏;重复序列被降权、不重复被升权 |
| 偏差多大 | 实测 PP 代价 −1.79(含偏差 + 截断增加的混合账) |
| 为什么当初加 | p9h 换行刷屏坍缩的止血带;λ 带封顶集中度后,该病根已除,药成了白交的税 |
| 对策 | p9l:VPO λ=2 无 PP(验证去掉偏差源后的净增益) |
