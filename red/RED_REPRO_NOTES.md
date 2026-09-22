# Reproduction notes

Paper: *RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward
Redistribution*, Jiahui Li, Lin Li, Tai-Wei Chang, Kun Kuang, Long Chen, Jun Zhou,
Cheng Yang (EMNLP 2025, pages 4993–5022).

Second paper, for the RL algorithm this arm uses: *Back to Basics: Revisiting
REINFORCE Style Optimization for Learning from Human Feedback in LLMs*,
Arash Ahmadian et al. (2024, arXiv 2402.14740).

Local copies: `paper/` at the repository root (gitignored — this repository is
public). Extracted text: `paper/red.txt` and `paper/rloo.txt`. Note that the
repository's own `paper/extract_pdf.py` does **not** work on the RED PDF — it
emits font-licence binary rather than text, because this PDF is produced by
pdfTeX with a different stream layout than the P2T one. `pypdf` extracts it
cleanly; that is how `paper/red.txt` was produced.

This arm reproduces the paper's **RLOO variant** strictly: no scale
normalisation, no added temperature, no adaptation of its formulas. Where the
paper contradicts itself, or leaves a quantity undefined, the code follows the
equations and the discrepancy is recorded below rather than repaired.

The arm is a *negative* control as much as a positive one. See §2.6: RED is
potential-based reward shaping, and the paper's own argument is that the true
advantage function is unchanged, so any gain is an estimation/optimisation effect.
RLOO is the RL algorithm with the least machinery for such an effect to act on.
The paper's numbers agree — its RLOO-RED gains are near zero on TL;DR and one
order of magnitude below PPO-RED on Nectar.

## 1. Equation to code

| Paper | Meaning | Code |
|---|---|---|
| Eq. (3) | `r^RM_t = 0` for `t < T`, `R_phi(x,y)` at `t = T` | `reward.sequence_reward_at_eos` |
| Eq. (4) | `r^KL_t = KL(pi_theta ‖ pi_ref)`, written as a divergence | `reward.red_kl_reward` |
| Eq. (5) | `r^final_t = r^RM_t − beta·r^KL_t` | superseded by Eq. (8) |
| Eq. (6) | `r~_t = R_phi(x, y_<=t) − R_phi(x, y_<=t−1)` | `reward.prefix_token_rewards` + `reward.prefix_boundaries`, `rm.score_prefixes` |
| Eq. (7) | `r^_t = beta_c·r~_t + (1−beta_c)·r_t`, `beta_c = 1` | `reward.red_convex_combination` |
| Eq. (8) | `r^final_t = r^_t − beta·r^KL_t` | `reward.red_final_reward` |
| Figure 4 | pseudo-code: the shifted difference, the sparse term at the EOS slot, the signed KL | `rm.score_prefixes`, `reward.sequence_reward_at_eos`, `reward.red_kl_reward` |
| §3.3 (2) | potential-based shaping preserves the optimal policy | §2.6 |
| RLOO §2.3 | `1/k sum_i [R_i − 1/(k−1) sum_{j≠i} R_j] ∇log pi` | `reward.rloo_baseline` |
| RLOO Eq. (11) | `L^{k=2} = (R(y⁺)−R(y⁻))/2 · (−log pi(y⁺) + log pi(y⁻))` | `loss.rloo_policy_loss` |
| **R4** (this arm) | `A_{i,t} = A_seq_i + alpha·(r^final_{i,t} − mean_t r^final_{i,t})` | `reward.rloo_red_credit` |
| ~~R3~~ (retired) | `A_{i,t} = r^final_{i,t} − b_i` | was `reward.rloo_red_credit`; see §2.2 |

Constants as configured in `configs/red250b.json`: `beta_c = 1.0` (the paper's
default; Table 7 uses 1 everywhere except LLaMA3 on TL;DR, which uses 0.5),
`beta = 0.03`, `k = 8` (`group_size`), `red_alpha = 1.0`. `configs/red250.json`
is the first attempt's config and pins the retired R3, so the trainer refuses it
rather than reinterpreting it.

Two of those are deviations from the paper's own numbers and are stated here
rather than left to be inferred:

* **`beta = 0.03`, where RED's Table 7 gives KL coeff `0.02`** for its LLaMA
  Nectar and TL;DR settings. 0.03 is the project's shared value, and happens to be
  the value RLOO's own paper uses for TL;DR, but it is *not* this paper's number.
  Kept for cross-arm comparability; flipping it is one key.
* **`k = 8`, where the paper uses `K = 4`** (`RLOO sample K 4`, Table 7). The
  project's `group_size` is 8, and the arms must be comparable; see §7.

The prefix-score read is the whole method: one forward pass over the canonical
reward-model row, the existing scalar head applied to every position's hidden
state. Nothing is retrained and nothing is added to the model, which is the
paper's "minimal additional computational costs" claim. It is also one pass
cheaper than the sibling P2T arm, which needs a backward pass through the reward
model for its input gradients.

## 2. Where the paper contradicts itself, or is silent (reproduced, not repaired)

### 2.1 The redistributed rewards do not sum to the sequence reward

The abstract, the introduction and Figure 1's caption all state that the
fine-grained rewards sum to the original sequence score ("the sum of the
fine-grained rewards is equivalent to the original sparse reward"). They do not.
Telescoping Eq. (6) gives

```
sum_{t=0..T} r~_t = R_phi(x, y_<=T) - R_phi(x, y_<=-1) = R_phi(x, y) - R_phi(x, empty)
```

so the redistribution moves `R_phi(x, empty)` — the reward model's score for the
prompt with no response — out of the response and into the offset. The paper does
acknowledge this later: §3.3 names the term `r~_{-1}` and concedes it "introduces
the potential for bias"; §4.5's ablation (`PPO-RED-w.o/DI`) exists precisely to
measure it. But the headline claim in three places is not corrected, and Appendix
A.2 sidesteps it by *assuming* `R_phi(x, empty) = 0` ("let `R_phi(x, y_{-1}) =
R_phi(x, empty) = 0`") while simultaneously arguing that it need not be zero. The
identity above is what the code implements and what
`tests/red/test_rm.py::test_score_prefixes_redistributes_exactly_the_sequence_score`
pins.

The offset is a function of the prompt alone, so within a prompt group it is a
constant. That fact is load-bearing for §2.2.

### 2.2 The RLOO baseline's treatment of token-level rewards is undefined

This is the one place the arm had to make a decision the paper does not make, so
it is recorded in full.

The paper adopts both PPO and RLOO (§3.2) but gives training details for **PPO
only** — Appendix A.1 states "The training details of PPO are provided in the
Appendix A.1", and A.1 contains Algorithm 1 for PPO and nothing for RLOO.
Figure 4's pseudo-code ends at `# RL using the final reward ......`. There is no
code release; the only links are to SafeRLHF and to datasets. Table 7 supplies
`RLOO sample K = 4` and nothing else.

Meanwhile RLOO's own paper is unambiguous that its advantage is a *sequence-level
scalar*: Appendix Eq. (11) writes the k=2 objective as
`(R(y⁺)−R(y⁻))/2 · (−log pi(y⁺) + log pi(y⁻))`, where `log pi(y|x)` is the whole
sequence's log-probability, and §3.3 argues explicitly that modelling partial
completions is unnecessary in RLHF and empirically worse. RLOO has no per-token
advantage at all — even the KL enters through the sequence-level return
`R(x,y) = r_phi(x,y) − beta log(pi_theta(y|x)/pi_ref(y|x))` of its Eq. (3).

RED's reward, by contrast, is per-token. So the two must be combined by hand.
Three readings are possible; one is provably wrong:

**(R2, rejected) Replace RLOO's sequence return with RED's return.** RED's return
is `G^RED_i = R_i − R(x, empty)`. The dynamic-initialisation offset `R(x, empty)`
is the same constant for every sample in a prompt group, so the leave-one-out mean
differences it away:

```
b_i^RED = 1/(k−1) sum_{j≠i} (R_j − R(x,empty)) = b_i^RLOO − R(x, empty)
A_i^RED = G^RED_i − b_i^RED = R_i − b_i^RLOO = A_i^RLOO
```

The advantages are *identical*, so the gradient is identical to plain RLOO and
the arm would measure nothing. This is not a guess: it is pinned as an executable
fact by
`tests/red/test_rloo_advantage.py::test_replacing_the_return_with_reds_return_is_exactly_rloo`.

**(R1, rejected) Leave-one-out per token position.** Subtract the other samples'
reward *at the same token index*. Mechanically implementable with padding and a
mask, but it departs from RLOO's baseline definition, and token index `t` of one
response has no semantic relation to token index `t` of another, so the baseline
becomes noise. Rejected as unfaithful to RLOO without being more faithful to RED.

**(R3, chosen for `red250` and then retired — see the subsection below) Scalar
sequence baseline subtracted per token.**

```
b_i    = 1/(k−1) * sum_{j != i} R_j          (RLOO's own baseline, unchanged)
A_{i,t} = r^final_{i,t} - b_i
```

Its gradient decomposes exactly into RLOO's term and RED's redistribution:

```
sum_t A_{i,t} ∇log pi_t = sum_t r^final_{i,t} ∇log pi_t  −  b_i ∇log pi(y⁽ⁱ⁾|x)
                          └ RED's per-token credit ┘       └ RLOO's baseline term ┘
```

and it is self-consistent with the papers' own arithmetic: summing Eq. (8) over
tokens gives `sum_t r^final_{i,t} = R_phi(x,y) − R(x,empty) − beta·log(pi_theta/pi_ref)`,
which is RLOO's KL-shaped return from its Eq. (3) minus the §2.1 offset that §2.2
just showed cancels in the baseline.

**But the decomposition above is about the sum, not about what the gradient sees,
and the two differ** (this correction was forced by an adversarial review pass, and
the earlier version of this section overstated the agreement). The loss divides by
the number of valid tokens (§2.8), so the per-response scalar the gradient actually
sees is

```
(1/T_i) * sum_t r^final_{i,t}  -  b_i
```

The credit is therefore **length-normalised while the baseline is a full sequence
return**. RLOO's own estimator is `R_i − b_i` with no length scaling at all —
its Eq. (11) is a scalar times a sum of sequence log-probabilities, and its paper
merges `1/k` into the learning rate rather than normalising by length. So this arm
is *not* "RLOO exactly as written plus a redistribution"; it is RLOO's baseline
combined with a length-normalised per-token credit, which reweights responses by
`1/T_i` within a group.

A consequence worth stating: `b_i` is the leave-one-out mean of a *different*
random variable than the quantity the credit sums to. The DI offset of §2.1 is
prompt-constant and does cancel, but the length-window and degeneracy terms are
per-response, so the baseline is not an exactly unbiased control variate for the
estimated return. The residual is the same order as the length shaping itself.

Both properties are inherited from the project's reduction, which the sibling arms
use as well, and both are one-line changes if strict RLOO is wanted (drop the
`/ mask.sum(-1)` in `loss.rloo_policy_loss`, and pass the raw reward to
`reward.sequence_returns`). They are kept because this arm's purpose is to be
comparable with its siblings, and `test_r4_does_differ_from_plain_rloo` pins that
it is still not plain RLOO.

#### R3 is retired (2026-09-22)

The paragraph above identifies the length half of R3's asymmetry and reads it as a
reweighting. It misses the other half, which is a sign inversion, and the run
`red250` hit it at rollout 26 and never recovered.

`b_i` is a *sequence*-scale quantity subtracted from a *token*-scale reward. A
single token's share of the redistributed return is `O(R/T)`, so once the group's
rewards are negative the advantage is dominated by the constant `−b_i`, and that
constant is positive. Measured on the run's own credit dumps
(`runs/red250/rollout-N-credit.pt`):

| rollout | mean `b_i` | mean token advantage | within-response `std(A)` / `|mean A|` | tokens with `A > 0` |
|---|---|---|---|---|
| 20 (healthy) | +0.85 | −0.83 | 1.98 | 0.465 |
| 26 | −15.67 | **+15.66** | **0.08** | **1.000** |
| 40 | −14.48 | +14.48 | 0.09 | 0.998 |

`|mean A|` equals `|b_i|` to two decimals: RED's redistribution had stopped
contributing anything to the update. The estimator had degenerated into RLOO's
baseline applied T times per response, i.e. "raise the probability of every token
you sampled", which is precisely the entropy collapse the run then showed. Two
independent readings confirm it:

* `red_advantage_flip_fraction` — the arm's own measure of how far its gradient is
  from plain RLOO's — was **0.0000** from rollout 26 through rollout 50. The
  metric was reporting the failure for 40 rollouts before the process died.
* The length channel opened the other way at the same time: at rollout 40,
  `corr(len, sum_t r~) = −0.294` (longer responses scored worse) against
  `corr(len, mean_t A) = +0.281` (longer responses were rewarded more), because
  the loss divides the credit by `T_i` and R3 did not divide the baseline.

What followed: mean response length 451 → 1823 tokens, 685 truncations, raw reward
0.48 → −9.17, then a collapse to ~2-token responses, `grad_norm` 104, `kl_to_init`
6.15, and a hard crash at rollout 68 (§9). The sibling P2T arm, on the same data,
reward model, initial adapter and length window, ran 250 rollouts with 0 resamples,
10 truncations and a final reward of **+11.04** — so the difference is the credit
rule, not the environment.

**R4** replaces it:

```
A_seq_i          = returns_i − baseline_i        (RLOO's own leave-one-out advantage)
direction_{i,t}  = alpha · ( r^final_{i,t} − mean_{t'} r^final_{i,t'} )
A_{i,t}          = A_seq_i + direction_{i,t}
```

The sequence contrast carries the level, at full strength and unnormalised, and
RED's redistribution decides only *which tokens within a response* receive it. The
retired rule's own rejected alternative (R2, above) is what forces the shape: RLOO
applied to RED's reward collapses back to plain RLOO, so the redistribution must
enter per-token or not at all, and the only way to enter per-token without
re-introducing a sequence-scale constant is to centre it.

Properties, all pinned by tests in `tests/red/test_rloo_advantage.py`:

* `advantage == sequence_advantage + direction` — the invariant the sibling arms
  keep and R3 had inverted.
* the valid-token mean of `direction` is zero (to float32 rounding), so the
  valid-token mean of `A` is exactly `A_seq`. No response-level level leaks into
  the token term, so there is nothing for the length to scale.
* `alpha = 0` reproduces plain RLOO exactly, which is the ablation this arm's
  claim rests on.

`red_alpha = 1.0`, chosen a priori rather than tuned: the paper specifies no
advantage function at all (§2.3), so equal weight for the two terms is the only
defensible default. Measured on the dumps, the centred token term has
`std ≈ 1.4–2.0` against `|A_seq| ≈ 2.3–2.8`, so one puts them within a factor of
~1.5 — the order the sibling arm runs at (`p2t_bonus_over_advantage` ≈ 3.3 with
`alpha = 0.1`, its token term being reward-scale rather than difference-scale).
`red_bonus_over_advantage` is now reported so the ratio is visible rather than
assumed.

Replaying R4 over `red250`'s dumps (`A_seq` proxied by `raw − LOO mean`, since
those dumps predate the `sequence_advantage` field):

| rollout | tokens with `A > 0`: R3 → R4 | `|mean A|`: R3 → R4 | `corr(len, mean_t A)`: R3 → R4 |
|---|---|---|---|
| 26 | 1.000 → 0.505 | 15.66 → 1.79 | −0.065 → −0.049 |
| 40 | 0.998 → 0.503 | 14.48 → 2.92 | +0.282 → −0.133 |

The advantage scale no longer tracks the group's reward level, and the positive
length coupling is gone. The residual negative correlation at a healthy rollout is
the signal working: long responses were scored worse by the reward model, so they
now receive a negative sequence advantage instead of having it washed out by a
constant.

### 2.3 Algorithm 1's PPO ratio is written with the wrong denominator

Algorithm 1 step 13 clips `pi_theta(a|s)/pi_ref(a|s)`. Standard PPO — and RLOO's
own Eq. (5) — clips `pi_theta/pi_old`, where `pi_old` is the behaviour policy, not
the reference. Using `pi_ref` would fold the KL constraint into the clip on top of
the explicit `beta·r^KL` term subtracted in Eq. (8).

This arm does not use PPO, so the contradiction does not bite here; it is recorded
because the PPO variant would have to resolve it, and the pseudo-code does not.
The reading implemented elsewhere in the project (the P2T arm) is that this is a
typo.

This is not the only thing the PPO variant would have to resolve by hand — §5 now
lists the critic's architecture, the optimiser, the inner-epoch count, the target
values and the entropy coefficient as equally open. That list is why §2.2's
advantage rule is this arm's choice rather than a departure from a specification:
the paper's only specified training recipe does not specify enough to implement.

### 2.4 Appendix Eq. (11) weights the KL by the token probability

Appendix Eq. (11) writes the redistributed reward's KL term as

```
-beta * sum_t pi_theta(y_t | x, y_<t) * (log pi_theta(...) - log pi_ref(...))
```

— an *expected* per-step KL. Eq. (4), Eq. (5), Eq. (8) and Figure 4's code all use
the unweighted sampled log-ratio (`kl_divergence = log_probs - ref_log_probs`).
The code follows Eq. (8)/Figure 4, i.e. the unweighted form, since that is what the
main text and the pseudocode both specify and what a trainer can compute. Recorded
because the appendix is not equivalent to the main text here.

### 2.5 Eq. (4) writes a divergence; the code computes a signed log ratio

Eq. (4) defines `r^KL_t = KL(pi_theta(a_t|s_t) ‖ pi_ref(a_t|s_t))`, which is
non-negative, and Eq. (5) subtracts it — so taken literally, every token would be
penalised regardless of direction. Figure 4's pseudo-code is
`kl_divergence = log_probs - ref_log_probs` and `final_reward = reward_combine - beta * kl_divergence`,
i.e. the signed per-token log ratio, which is also what standard RLHF
implementations use. The code follows the pseudo-code, so
`test_kl_reward_is_the_signed_log_ratio_not_a_divergence` asserts the sign
survives rather than being clamped. The metric `kl_to_init` deliberately does
**not** use this quantity — see §4.

### 2.6 RED is potential-based shaping, so its benefit is an estimation effect

Substituting Eq. (6) into the standard shaping form `R'(s,a) = R(s,a) + gamma*Phi(s') - Phi(s)`
with `gamma = 1` and the potential `Phi(s_t) = R_phi(x, y_<=t-1)` reproduces the
redistributed reward exactly. §3.3 of the paper makes this argument itself, and
its conclusion is that `A'(s,a) = A(s,a)` — the advantage function is *unchanged*.
The stated benefit is therefore not a different objective but "faster
convergence": a variance/optimisation effect on the estimator.

That matters for the choice of RL algorithm. RLOO's whole design is to remove the
machinery such effects act on: no critic, no bootstrapping, no GAE, an unbiased
Monte-Carlo return with a parameter-free baseline (RLOO §3.1, §3.3). The paper's
own results are consistent with this reading:

| Task / model | Baseline | RED | Note |
|---|---|---|---|
| Nectar, LLaMA | RLOO −0.079 | RLOO-RED 0.253 | vs **PPO-RED 3.475** in the same table |
| TL;DR, LLaMA | RLOO 0.202 | RLOO-RED 0.205 | reward score moves 0.003 |
| TL;DR, GPT-4 head-to-head | RLOO-RED vs RLOO | 50.0 win / 2.0 tie / 48.0 lose | a coin flip |
| TL;DR, GPT-4 vs SFT | lose 39.0% | lose 44.5% | worse |
| SafeRLHF, LLaMA3, GPT-4 | 34.5% | 33.5% | worse |

The paper itself observes that "the improvement achieved by RLOO-based methods is
not as significant as that of PPO-based methods" (§4.3). Also note the coverage
gap: RLOO is only ever run on LLaMA and LLaMA3 — there is no Qwen2.5 row for any
RLOO variant — and this project's actor is Qwen3-14B-Base.

So this arm should be read as a control on *where* RED's benefit comes from: if
per-token redistribution helps under a critic (PPO) but not under an unbiased
Monte-Carlo estimator (RLOO), the gain was an advantage-estimation effect, not a
better reward. `red_advantage_flip_fraction` is reported each step precisely to
quantify how far RED's gradient sits from plain RLOO's.

### 2.7 Four departures from the project's conventions, and one alignment choice

The project (and the sibling P2T arm) uses a different convention in three places.
The paper wins in each, because these are *method*, not training parameters:

1. **KL placement.** Eq. (8) puts `−beta·r^KL_t` inside the reward, so the
   constraint reaches the policy through the advantage and the loss has **no
   separate KL term**. The project adds a `kl_from_logp` term to the loss instead.
   Applying both would double-count the KL.
2. **KL estimator.** The reward uses the signed log ratio (Figure 4). The
   project's loss term uses the second-order estimator `exp(d) − d − 1`. They are
   not interchangeable; the code uses the former for the reward and the latter
   only for the reported metric (§4).
3. **No clipping.** RLOO §3.2 removes it (the loss is clipped < 5% of the time per
   batch, and removing clipping slightly *helps*). A clipped GRPO surrogate would
   be a different algorithm wearing RED's rewards.

   `clip_eps` is still accepted from the shared config and is read in **three**
   places, none of which clamps the objective: the config validation (which
   requires `0 < clip_eps < 1`, so setting it to 0 or 1 to mean "no clipping"
   *raises*), the `rollout_direct_ratio_clip_fraction` metric, and the startup gate
   that checks the trainer's re-forward reproduces the sampler's probabilities.
   The manifest stamps `red_clipping` so the actual behaviour is unambiguous after
   the fact. Read the knob as "the band for the sampler-agreement check", not as a
   clip.

   The one importance-like factor in the loss is the sampler-to-trainer correction
   `exp(old_logp − rollout_logprob)`, which multiplies the objective. It is ~1 by
   construction — the startup gate asserts exactly that, since both sides run the
   same weights under an FP32 head — and it is a numerical fix for two kernels
   differing, not a policy ratio. It is nevertheless not in RLOO Eq. (11) and it
   does scale the gradient, so it is recorded in §2.8 rather than waved away.

And one place where the project's convention wins deliberately:

4. **Which sequence reward feeds the baseline.** RLOO's Eq. (3) return is
   `r_phi − beta·KL`; the code uses the project's **shaped** sequence reward
   (raw reward model score minus the soft length window minus the degeneracy
   floor) in that role, while the *redistribution* uses the raw prefix scores.
   This mirrors the sibling arm's two-`R` structure and is what keeps the length
   window acting on the advantage in an arm that does not standardise. Using the
   raw score for the baseline too is a one-line change in `trainer.py`.

### 2.8 Two reductions in the loss that are not RLOO's Eq. (11)

Both were surfaced by an adversarial review pass and are recorded rather than
silently kept. Both are inherited from the project's reduction so that the arms'
gradient scales are comparable, and both are one-line changes if strict RLOO is
wanted.

1. **Per-response length normalisation.** The loss divides each response's summed
   objective by its valid-token count:

   ```
   loss = -(1/B) * sum_i (1/T_i) * sum_t A_{i,t} * log pi(y_t | s_t)
   ```

   RLOO Eq. (11) has no such factor: it is a scalar times a sum of sequence
   log-probabilities, and its paper merges `1/k` into the learning rate rather than
   normalising by length. The consequence is a **length-dependent reweighting of
   responses within a group** — measured as exactly `1/T_i` against RLOO's
   per-response gradient — which is a real change to the estimator, not a scale-free
   convention. Combined with §2.2 this is the second reason the arm is not "RLOO
   with a redistributed reward": the credit is length-normalised while the baseline
   is not.

2. **The sampler-to-trainer importance factor.** The objective is multiplied by
   `exp(old_logp − rollout_logprob)`, which is an importance ratio, is not in Eq.
   (11), and does scale the gradient. It exists because the sampler (vLLM) and the
   trainer (HF) run different kernels; the project's arms all carry it, and the
   startup gate asserts the two forwards agree so that it stays ≈ 1.

Why keep both: the arm's purpose is comparison with GRPO, VPO-RM and P2T, and all
three length-normalise and carry the same importance factor. Removing them here
would make RED's gradient scale incommensurable with the others for no gain in
fidelity — the paper whose equations are being reproduced is RED's, and RED says
nothing about the RL estimator's reduction. It says nothing about RLOO either,
which is why §2.2 had to be decided at all.

## 3. The boundary construction: the one place the code generalises the pseudo-code

Figure 4 differences *adjacent positions of the reward-model row*:
`reward_token[1:] = reward_model_outputs[1:] - reward_model_outputs[:-1]`. That is
exactly Eq. (6), and it works unchanged when the policy and the reward model share
a tokenizer — which is the paper's setting, where the reward model is initialised
from the same SFT model.

This project's actor and reward model are different checkpoints, so the
actor-to-reward-model token mapping is byte-exact but *incomplete*: special tokens
are never mapped, and BPE merge/split rewrites leave holes. Two facts make the
naive port wrong:

* The reward model pools at the **last valid position of the canonical chat**,
  which is a special token and therefore *always* unmapped. Differencing adjacent
  actor tokens would drop that trailing segment.
* A hole in the middle would let a token borrow a neighbour's difference.

`reward.prefix_boundaries` therefore differences between *boundaries* rather than
between adjacent actor tokens:

```
left_0  = (first mapped position) - 1      -- this is R_phi(x, empty), the §2.1 offset
left_t  = right_{t-1}                      -- the chain
right_t = mapped_t, except right at the last mapped token = pooled
```

Every token the reward model never saw contributes exactly zero; the span it
occupies is carried by the following mapped token, which is what preserves the
identity. The identity is asserted two ways: algebraically on synthetic layouts
with holes and trailing specials (`tests/red/test_prefix_difference.py`) and
end-to-end through a real reward model row (`tests/red/test_rm.py`).

This is a **faithful generalisation forced by tokenizer misalignment**, not a
change of method: with a fully-mapped row it reduces to the pseudo-code exactly.
It is recorded here because it is the one place where the code cannot be a literal
transcription.

## 4. Diagnostics the paper does not define

The paper defines no monitoring statistics, and RED's redistributed rewards are
signed differences, so unlike the sibling arm's attribution softmax they do not
form a distribution over tokens. Two project-level additions, stated as such:

* `credit_share` (and therefore `red_share_max_mean`, the flat/one-hot fractions
  and `credit_ess_ratio`) is the **positive** part of the reward normalised to sum
  to one over the response. Rows with no positive credit at all fall back to a
  uniform share rather than dividing by zero. This keeps the credit-concentration
  diagnostics readable and comparable with the sibling arm's,
  whose `credit_ess_ratio` has the same direction (1 flat, 1/T one-hot).
* `group_sigma_*` is the population std of the shaped reward within each prompt
  group — the same spread the sibling arms *divide by*. RED reports it and does
  not divide, which is exactly why reporting it matters for cross-arm comparison
  (`tests/red/test_conformance.py::test_group_sigma_is_the_projects_group_scale`).

`kl_to_init` keeps the project's `exp(d) − d − 1` estimator so the drift curve is
comparable with the sibling arms', even though the reward's own KL uses the signed
log ratio (§2.5). `red_advantage_flip_fraction` is the fraction of tokens whose
update direction differs from the baseline-only direction — i.e. how far this
arm's gradient is from plain RLOO's, which is the quantity §2.6 turns on.

## 5. Deliberately not implemented

* **PPO.** The paper's headline configuration, with GAE, a critic and clipping.
  This arm is RLOO by choice; §2.3 records what PPO would additionally have to
  resolve. A full re-read of the paper for the R4 work (2026-09-22) turned up
  considerably more than §2.3 lists, and it is recorded here because it is the
  reason §2.2's advantage rule had to be chosen by this arm rather than copied:
  PPO is the paper's *only* specified training recipe, and it does not specify
  enough to implement. Open, in the paper's own text:

  - **The critic.** Algorithm 1's `Require:` line says "Initial critic model V_φ"
    and nothing else — not the architecture, not the backbone, not what it is
    initialised from. §5 of the paper says only that PPO needs "policy model,
    reward model, critic model, and reference model", implying a separate model.
    The one value-head description in the paper (§2.2, Figure 2) is about the
    *reward* model, not the critic.
  - **The optimiser.** Actor and critic learning rates, weight decays, warmups and
    schedules are given separately (Table 7: actor 1e-5, critic 5e-6, critic weight
    decay 0.0), which implies two optimisers; the paper never names one, and
    "Adam" appears nowhere in the text.
  - **The PPO ratio's denominator.** Algorithm 1 step 13 writes
    `π_θ(a|s)/π_ref(a|s)` in both the ratio and the clip. `π_old` is never defined
    or mentioned anywhere in the paper, and no step stores the behaviour policy's
    log-probabilities. Implemented literally this is not PPO: it is a trust region
    around the SFT model layered on top of the KL penalty in Eq. (8), which already
    anchors to the same reference.
  - **Entropy bonus: absent.** Zero occurrences of "entropy" in 30 pages, and no
    row for a coefficient in Table 7.
  - **PPO inner epochs and minibatch size: absent.** Table 7's "total epochs 3" is
    the outer loop; there is no inner-epoch count and no minibatch size.
  - **Advantage normalisation: absent, not merely unspecified.** No whitening, no
    reward standardisation, no running baseline is mentioned anywhere.
  - **The target `V′` is never defined.** Algorithm 1 step 12 says only "compute
    target values {V′} for each y_i with V_φ", and step 15 is an unweighted MSE
    against it. Whether `V′` is the λ-return, the Monte-Carlo return, or GAE is not
    stated, and there is no value-loss coefficient, no value clipping and no reward
    clipping.
  - **Eq. (4)'s KL vs Figure 4's.** Eq. (4) defines the full per-token divergence
    `KL(π_θ(a_t|s_t) ‖ π_ref(a_t|s_t))`; Figure 4 computes the single-sample
    `log_probs − ref_log_probs`. §2.5 already records which one this arm follows;
    the paper never reconciles them, and the choice changes the reward's magnitude.
  - **Two more internal contradictions.** Figure 4 zeroes `reward_token[0]` while
    Eq. (6) at `t=0` gives `R_φ(x,y_≤0) − R_φ(x,∅)`; and Table 7's
    `reward shaping α` is 1 while §4.4 says "α is a scaling factor set to -1 in our
    experiments".

  The relevance to §2.2 is direct: "the paper is silent about the RLOO baseline" is
  not one loose end among many, it is the *whole* estimator. Cutting the PPO route
  would have meant making all of the decisions above by hand, which is a strictly
  worse fidelity position than choosing the one rule §2.2 needs and stamping it.
* **The harmfulness/helpfulness two-model setup.** The paper applies
  redistribution separately to a reward model and a cost model and combines them
  (`r~agg = 1/2 (r~_t + alpha·c~_t)`, `alpha = −1`). This arm runs the single
  reward model the project's other arms use.
* **`beta_c != 1`.** Implemented and configurable, but not swept. The paper's own
  ablation shows the trend is monotone in `beta_c` with 1 best on Nectar, and
  Table 7 uses 1 everywhere except one cell.
* **The random-noise robustness experiment** of §4.5. It is a property of the
  method, not of a training run.

## 6. Numerical details

* Prefix scores are read in the reward model's dtype and returned as float32
  (`rm.prefix_scores` returns `.float()`), so the difference and every downstream
  quantity are float32. The telescoping identity therefore holds to float32
  rounding, not exactly — measured relative error ~2.6e-7 on a synthetic row, and
  the tests assert with `rtol=1e-5`.
* The per-row hidden state is materialised to read the head at every position, so
  the reward-model microbatch is one; `microbatch_responses` is validated to be 1.
* `prefix_token_rewards` casts to float32 rather than promoting to float64. This
  matches the sibling arm, which computes its token reward in float32 for the same
  reason: the reward model's own output is already lower precision.

## 7. What is shared with the project, and how that is checked

`red/` never imports `p2t` or `vpo_rm` at runtime — it is a standalone package in
the same sense the P2T arm is, and for the same reason (a baseline should be
auditable on its own). A test asserts the two deliberate algorithmic divergences
as *absences*: no `group_advantages`, and no `clip_eps`/`old_logp` parameter in
the loss.

`tests/red/test_conformance.py` imports `vpo_rm` from the same checkout and
asserts the pieces that must be identical really are, using the project's tolerance
convention (`atol=0, rtol=0` for pure ports):

| Shared piece | Compared against |
|---|---|
| byte-exact actor-to-RM mapping (`canonical_chat_v1`) | `vpo_rm.reward_inputs.build_reward_input` |
| soft length window | `vpo_rm.length_reward.soft_length_penalties` |
| degeneracy flags | `vpo_rm.length_reward.response_degeneracy` |
| reward-scale calibration | `vpo_rm.length_reward.calibrate_reward_scale` |
| group spread diagnostic | `vpo_rm.core.group_advantages` (scale output) |

Mirrored training protocol, from `configs/red250b.json` (identical to the sibling
arm's `formal250.json` except for the credit assignment): actor `Qwen3-14B-Base`,
reward model `Skywork-Reward-V2-Qwen3-8B`, `init_adapter models/sft-p2t`, group
size 8, 8 prompts per rollout, 250 rollouts, actor lr 5e-5, weight decay 0.01,
grad-norm clip 1.0, LoRA r 64 alpha 128, `beta` 0.03, temperature 1.0, max
response 2048, the project's soft length window (8 / 1024, strengths 0.5 / 2.0),
and the shared `sigma0 = 3.0323000897825447`. The two `k` values that differ from
the paper are deliberate: `k = 8` because the project's `group_size` is 8 and the
arms must be comparable, against the paper's `K = 4`, and the actor itself is a
larger, newer model than any the paper ran RLOO on.

Because RED does not standardise the advantage, the project's `sigma0`-based
`advantage_std_floor_fraction` has nothing to floor, and the raw reward scale
reaches the gradient directly. The sibling arm mixes a standardised and a raw term,
so this is not a new kind of scale difference, but the gradient norms are worth
eyeballing against the siblings' for the first few steps.

## 8. Review record

Three adversarial reviews were run against this package: paper fidelity (against
`paper/red.txt` and `paper/rloo.txt`), numerical correctness, and integration
hygiene (fork residuals, shared tooling, isolation). Every finding is listed with
its disposition; nothing in the table was waived silently.

| Finding | Disposition |
|---|---|
| **Cross-device crash on the first rollout.** The credit block mixed devices: `rloo_baseline(returns, group_ids)` with `returns` on the actor device and `group_ids` on the reward device, and `_red_diagnostics` was handed a reward-device mask while `credit` lives on the actor device. Both raise on a real 2-GPU box; the CPU suite cannot see either, because it pins `actor_device == reward_device == "cpu"`. Inherited from the sibling arm, which builds credit on the reward device and moves the three `[B,T]` fields afterwards — RED cannot, because Eq. (8) needs the reference log-probs. | **Fixed.** The credit block now moves the group ids and the mask to the actor device explicitly, and `rloo_baseline`/`rloo_red_credit` validate their own device arguments so a future mixing fails with a readable message instead of an indexing error. Two guard tests use `meta` tensors, the only way to produce a genuine device mismatch on a CPU box. This class of bug is *not* fully coverable by the CPU suite: watch the first GPU rollout. |
| **`clip_eps` was documented as unread and is read three times**, one of which aborts the run (the startup agreement gate). Setting it to 0 or 1 to mean "no clipping" raises during validation. | **Fixed in the docs, kept in the code.** The trainer docstring, the config comment and §2.7.3 now state all three reads and that none of them clamps the objective. The gate is worth keeping — it is what catches a sampler/trainer protocol mismatch before an update is applied. |
| **"No importance ratio" was false about the code as run.** The objective is multiplied by `exp(old_logp − rollout_logprob)`, which is an importance factor, is not in RLOO Eq. (11), and does scale the gradient. | **Fixed and recorded** in §2.7.3 and §2.8.2. |
| **The loss length-normalises, which RLOO Eq. (11) does not.** The per-response `1/T_i` reweights responses within a group and is absent from the equation map. | **Recorded** in §2.8.1, together with the correction to §2.2: the decomposition there is about the *sum*, while the gradient sees a length-normalised credit against an un-normalised baseline. Kept for cross-arm comparability; the one-line change to remove it is noted. |
| **§2.2 overstated how intact RLOO's estimator remains** ("RLOO exactly as written", "baseline, intact"). | **Corrected.** §2.2 now states that the arm is RLOO's baseline combined with a length-normalised per-token credit, and that `b_i` is the leave-one-out mean of a different random variable than the credit sums to (the DI offset cancels because it is prompt-constant; the length and degeneracy terms are per-response). |
| **RED Table 7's own KL coefficient is 0.02, and the notes did not say so** — they justified `beta = 0.03` only by the project's and RLOO's values. | **Fixed** in §1, which now names both deviations from the paper's numbers (`beta` and `k`) in one place. |
| **`kl_metric` was imported and never called**, while its docstring claimed it was the reporting path; the metric was computed inline. | **Fixed.** The trainer now calls it, and its reduction was changed to the token-weighted mean so the reported `kl_to_init` is identical to the sibling arm's inline computation. |
| **Unused imports** (`nullcontext`, `replace`, the latter a leftover of the sibling arm's device-move helper). | **Fixed.** |
| **`plot_red.py` and `watch_red.sh` were listed in `README.md` but did not exist.** | **Fixed.** Both written; they key on RED's own metrics so no panel or summary field silently goes blank. |
| **`check_fresh_output`'s marker set is partly inert** — `metrics.jsonl` lives in `report_dir` and `run_manifest.json` inside `checkpoint-N/`, so neither can ever appear in `output_dir`; `report_dir` is never checked, so relaunching with a fresh `output_dir` appends to the old metrics with duplicated rollout numbers; and `vllm-adapters/` is created in `__init__` before the model loads, so a launch that dies during loading leaves a marker that blocks every retry. | **Recorded, not fixed.** All three are inherited from the sibling arm, and changing them would make this arm's startup contract differ from the one its siblings are validated against. |
| **The push settings point at the sibling arm's branch and remote** (`p2t-origin` / `p2t-baseline`), and the checkout is on `p2t-baseline`. | **Decided: autopush is enabled** (`push_every: 5`). The repository owner approved it explicitly, with the destination known: commits land on the local `p2t-baseline` branch (the one this checkout is already on) and are pushed to `p2t-origin/main`, the same place the sibling run publishes to. Two consequences the owner accepted: the two arms' commits interleave on one branch, and a push can lose a race with the sibling run's push — the pusher is best-effort and non-fatal, so a lost race only leaves the commit local. `tests/red/test_trainer.py` pins the destination so a silent change of remote becomes a test failure rather than a surprise push. |
| The KL reward uses `pi_old` (Figure 4's rollout log-probs) while Eq. (4) writes `pi_theta`. | **Recorded** as immaterial *as configured*: the batch equals `optimizer_minibatch_responses`, so there is one optimizer step per rollout and `pi_old ≡ pi_theta` at the update. Material if minibatching is ever changed. |
| **An entirely unmappable response aborted the run.** `score_prefixes` masked the redistribution with `response_mask & positions.ge(0)`. An immediate-stop response is a single special token, and specials are never mapped, so that intersection emptied the row — and `_binary_mask` requires at least one valid token per row. Reachable and reachable *by design*: the trainer flags such a response degenerate and floors its shaped reward, so it trains on it deliberately, and the rollout-0 gate cannot see it because `rm_unmapped_content_fraction` excludes special tokens. | **Fixed.** The mask is now the actor response mask alone. The intersection was redundant as well as harmful: `prefix_boundaries` gives unmapped positions `right == left`, so their difference is already exactly zero. Two regression tests use the immediate-stop row directly, and one checks it does not take its batch neighbours down with it. |
| `left.clamp_min(0)` would drop the dynamic-initialisation term if the first mapped token sat at reward-model position 0 (the first boundary should be `-1`, which is not a valid index). | **Recorded, not reachable.** The canonical chat template always precedes the response, so a response's bytes never start at offset 0. Left as is: clamping keeps the gather in range, and the alternative is an out-of-range index. |
| `prefix_boundaries` uses `cummax` over `mapped`, which means "the last mapped position at or before `t`" only while `mapped` is non-decreasing. It is exported, so a caller could pass a non-monotone row. | **Recorded.** In the only call path the precondition is enforced first and does hold: `rm.score_prefixes` calls `check_response_tokens` (strictly increasing over valid positions) before `prefix_boundaries`. With a non-monotone row the *sum* still telescopes but individual tokens are misattributed, i.e. it fails silently — hence recorded rather than dismissed. |
| Tokens the reward model never saw receive a **non-zero advantage** (their redistributed reward is zero, but the sequence advantage and the KL term are not), which reads as contradicting the "contributes exactly zero" language in `mapping.py` and `prefix_boundaries`. | **Recorded, faithful.** Under R4 the advantage is `A_seq_i + alpha*(0 − mean_t r^final)`, so an unmapped token shares its response's sequence advantage instead of the retired rule's constant. Only the *redistribution* is zero there. `rloo_red_credit`'s docstring says so explicitly, since the shorter phrasing invited the wrong reading. |
| Device guards live in `rloo_baseline` and `rloo_red_credit` only; `credit_share`, `red_final_reward`, `sequence_returns`, `sequence_reward_at_eos` and `red_kl_reward` would still fail with a raw torch device error rather than a named diagnostic. | **Recorded.** The two guarded functions are the ones that touch two different roles' tensors; the others take tensors from one side. Adding five more guards would be noise for no additional protection. |
| `rloo_baseline`'s docstring calls the leave-one-out mean "exact"; it is float32-exact. Grouped in float32 and differenced as `sum − r`, cancellation grows with the reward magnitude (measured deviation vs float64 ≤ 3e-7 on `randn(8)*3`). | **Recorded.** The reward model's own output is already lower precision, and the sibling arm standardises in the same way. |

### Verified solid (independent re-derivation, not just inspection)

Worth recording so a future reader does not re-open settled ground: the telescoping
identity on every adversarial layout tried (leading/trailing unmapped runs,
interleaved holes, a single mapped token, no mapped tokens, non-contiguous values,
`pooled` coinciding with a mapped position); `left[t] == right[t-1]` for every
consecutive mapped pair; `weight[mask].mean() == 1` exactly on all-positive,
all-negative, all-zero, mixed-sign, denormal and 1e6-skewed rows; the loss
numerically identical to `−(1/B) Σ_b (1/|y_b|) Σ_t A·log pi` and to
`… A·w·log pi` with the importance factor; the credit ordering forced by Eq. (8);
and `kl_metric` bit-identical to the sibling arm's inline `kl_to_init`.

### A repository-level finding, outside this package

While reviewing, `git add -A` in the autopush helper (`autopush.py:95`, inherited
verbatim) was found to stage the **entire working tree**, not just the run's
artifacts. Consequences observed in this repository:

* The whole `red/` package was committed into the sibling arm's history while
  still under construction (`9c772f9`, "p2t step 170"), and pushed.
* A copy of the RED paper PDF placed in `p2t/` was committed (`4544859`, "p2t step
  165") and **pushed to `p2t-origin/main`**, i.e. into a public repository — the
  exact thing this repository's own policy forbids ("the paper and its extracted
  text stay local: this repository is public").

The staging behaviour is worth narrowing to the run's own paths before the next
arm runs. Removing the PDF from history is a force-push against a published
branch and is therefore a decision for the repository owner, not for this file.

## 9. Run record

### `red250` — the R3 run, collapsed and crashed (2026-09-21 20:01 → 2026-09-22 04:00)

67 rollouts completed, then a fatal `RuntimeError` during rollout 68. The artifacts
are kept on disk as the evidence for §2.2: `runs/red250/` (68 rollout artifacts,
checkpoints at 40 and 60, `train.log` with the traceback), `reports/red250/`
(`metrics.jsonl`, `health.log`, `summary.json`, `reward.png`). The config that
produced it, `configs/red250.json`, pins R3 and is refused by the trainer rather
than reused.

**Timeline**, from `reports/red250/metrics.jsonl`:

| rollouts | mean length | raw reward | what changed |
|---|---|---|---|
| 1–19 | 178 → 519 | +2 … +7 | ordinary: token-level credit near zero, `flip_fraction` 0.05–0.08 |
| 20–26 | 582 → 1753 | +2 → −6.8 | length runs to the 2048 cap, truncations 0 → 43/64, `flip_fraction` → **0.0000** |
| 26–50 | 345 → 1710 | −4 … −10.8 | the all-positive-advantage regime of §2.2; `kl_to_init` 0.09 |
| 50–67 | 345 → 2.1 | −6.5 … −10.6 | entropy collapse, `grad_norm` 3.9 → 104, `kl_to_init` → 6.15, degenerate rows appear |
| 68 | — | — | fatal `RuntimeError` in `rollout._pack` |

**The crash** is a bug in this package, not in the paper, and it is fixed. When a
rollout contains a prompt group whose responses are all degenerate, the selector
resamples that group; the retry is a *separately generated* batch, so its response
block has its own padded width. `_pack` sliced a piece's response block by the
merged width while building a `keep` mask at that same merged width, so a piece
whose own block was narrower produced

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

Rollout 68 was the first in the run where the resample path met two different block
widths — the collapsed policy's ~2-token responses are a 2-column block, the retry
was 3. `_pack` now clamps every slice to the piece's own block and zero-pads the
tail, which is outside every row's mask and so already excluded by the loss.
`red/rollout.py` and `p2t/rollout.py` are byte-identical apart from one docstring
word, so **the sibling P2T arm carried the same latent bug**; it never fired there
only because `p2t250` resampled no group in 250 rollouts. Both arms now have a
regression test (`tests/red/test_rollout.py`, `tests/p2t/test_rollout.py`) that
constructs the narrow-piece merge and fails against the old `_pack`.

**Why nothing was recovered from it.** `checkpoint_interval` was 20 with
`keep_checkpoints: 2`, so the surviving checkpoints were 40 and 60 — both already
inside the bad regime (rollout 40: reward −8.6, length 1059; rollout 60: length
7.2, `kl_to_init` 1.5). The last healthy state was around rollout 19–20 and was
never written. `configs/red250b.json` keeps the sibling's 20/2 cadence anyway,
because `tests/red/test_trainer.py` pins every shared training parameter against
`formal250.json`; the protection against a repeat is the health gate below, which
also caps how much recoverable progress a stop can cost.

**The health check now fails fast where it used to take notes.** `red250`'s
`health.log` shows the checker reporting `PROBLEM: entropy falling`, `PROBLEM: KL
to init is 5.19` and `PROBLEM: response length falling` from rollout 62 onward,
but `red_advantage_flip_fraction` — the earliest and most specific signature, flat
at 0.0000 from rollout 26 — was a *note*, so nothing stopped. It is now a hard
problem when it sits below 0.01 over a window, alongside two new checks:
`red_positive_advantage_fraction` outside `(0.05, 0.95)` and `red_bonus_over_advantage`
outside `(0.02, 50)`. `red/scripts/check_red_health.py --stop-on-problem --pidfile …`
SIGTERMs the run when a problem is found, and `watch_red.sh` takes
`stop-on-problem` as a third argument to enable it. It is off by default: killing a
run is the launcher's call.

**The sibling arm is the control.** P2T ran the same data, the same reward model,
the same `models/sft-p2t` initial adapter and the same length window for 250
rollouts: 0 resampled groups, 10 truncations, minimum mean length 150, raw reward
0.55 → **+11.04**. The environment was not the problem.

### `red250b` — the R4 run, stopped deliberately at step 23

Config `configs/red250b.json`, the R4 credit rule, four cards (`tp=2`) instead of
`red250`'s three (`tp=1`). Nothing else differs from `red250` in the credit path,
so this run is the test of whether the R3 fix was sufficient.

**It was not.** The R3 channel is closed — `red_positive_advantage_fraction` sits
at 0.28–0.54 rather than collapsing onto 1.0, `red_advantage_flip_fraction` stays
at 0.06–0.21 rather than 0.0000, `red_bonus_over_advantage` at 0.19–0.65. But the
run reproduces `red250`'s **length** trajectory almost step for step, and the two
runs are independent evidence for the same conclusion because they differ in the
credit rule:

| step | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|
| `red250b` mean tokens | 649 | 623 | 679 | 876 | 987 | **1116** |
| `red250` mean tokens | 451 | 519 | 582 | 875 | 925 | **1048** |
| `red250b` truncated /64 | 4 | 4 | 8 | 8 | 9 | **22** |
| `red250` truncated /64 | 0 | 0 | 5 | 6 | 11 | **20** |
| `red250b` raw reward | 4.26 | 7.64 | 3.42 | 6.87 | 3.10 | **−0.16** |
| `red250` raw reward | 4.35 | 7.03 | 2.32 | 6.16 | 1.29 | **−0.38** |

(`red250b` entropy ran 1.74 → 2.68 over the same steps, against `red250`'s
1.35 → 2.47.) **Stopped at step 23** by decision, with the four cards released and
`runs/red250b/` + `reports/red250b/` kept as the record. The run did not reach its
endpoint, so this is the onset of the collapse rather than its full course — but
the onset is what §2.2's second channel predicts, and the full course is already
on record from `red250`.

**The reading.** The two failure modes are independent. R3's all-positive
advantage was a bug in *this arm's* estimator and R4 removed it; the length
explosion is a property of RED's **prefix differencing at this reward scale**,
untouched by that fix. The mechanism is the one in §2.2's tail analysis: the
telescoping chain lands the response's final verdict on the last token, and the
loss divides by `T`, so lengthening a response dilutes its punishment while every
earlier token keeps whatever positive prefix drift it collected. A run that never
truncated would eventually meet the verdict; this one truncates at the 2048 cap,
so the negative is collected once, late, and divided by ~2000.

### The health gate does not catch this mode (must be fixed before a re-run)

`reports/red250b/health.log` reported **zero** problems across steps 13–23 while
the run was visibly collapsing. Every trip-wire added in §2.2's retirement work is
calibrated to the *R3* signature — flip fraction → 0, positive-advantage fraction
→ 1, bonus/advantage in a band — and the shared checks in
`scripts/check_run_health.py` look for length **falling**, entropy **falling**,
truncation above 50%, and `kl_to_init` above 5. The length channel moves all of
those the *other* way: length rising, entropy rising, truncation rising but still
far below 50%. So `stop-on-problem` was armed and correctly did nothing.

A re-run needs a trip-wire for the length channel specifically — truncation rate
rising across a window, or mean response length doubling across one — and it
should be added on the run that follows, not after watching another 40 steps of a
predicted trajectory. The same gap applies to the sibling arms if a reward scale
can drive them into length growth.
