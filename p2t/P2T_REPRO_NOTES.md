# Reproduction notes

Paper: *Unlocking Token Rewards via Training-Free Reward Attribution*,
Sitong Wu, Haoru Tan, Bin Xia, Xichen Zhang, Jingyao Li, Shaofeng Zhang,
Xiaojuan Qi, Bei Yu, Jiaya Jia (CVPR).
Local copy: `paper/` at the repository root (gitignored — this repository is public).

This arm reproduces the paper's method **strictly**: no scale normalisation, no
added temperature, no adaptation of its formulas. Where the paper contradicts
itself the code follows the equations and the discrepancy is recorded below.

## 1. Equation to code

| Paper | Meaning | Code |
|---|---|---|
| Eq. (1) | `I_i = R(E) − R(E_{i←∅})` | the definition |
| Eq. (2) | `I_i ≈ ∇_{e_i}R(E)ᵀ (e_i − e_∅)` | `attribution.null_token_attribution` |
| Eq. (3) | `R^P2T_i = R + ω·R·exp(I_i)/Σ_j exp(I_j)`, ω=0.6 | `reward.p2t_token_reward` |
| Eq. (4) | `Â_n = (R_n − mean)/std` | `reward.group_advantages` |
| Eq. (5) | `Ã_{n,i} = Â_n + α·R^P2T_{n,i}`, α=0.1 | `reward.p2t_token_advantage` |
| §3.3.2 | GRPO clipping and KL unchanged | `loss.py` |

Null token: the reward model's **pad** token, `<|endoftext|>`. Paper Table 3(a)
ablates this choice on MATH-500 pass@1 — pad 53.8, mean-of-vocabulary 52.5,
zero embedding 51.4, EOS 50.3 — and pad is what we use.

α = 0.1 is the paper's short-CoT value ("such as the Qwen2.5 series"); α = 1.0 is
for long-CoT models. The actor here is `Qwen3-14B-Base` generating with
`enable_thinking=False`, i.e. the short-CoT regime. The constant is exposed as
`alpha` in the config and recorded in every run manifest.

Approximation: the vanilla first-order Taylor expansion of Eq. (2). The paper's
Table 3(g) also reports an integrated-gradients variant (53.8 vs 52.6) but does
not define its integration path, so it is not implemented. `P2T_APPROXIMATION`
stamps every artifact with `taylor_first_order_eq2`.

## 2. Where the paper contradicts itself (reproduced, not repaired)

### 2.1 The token rewards do not sum to the sequence reward

Section 3.3.1 states that the normalisation "guarantees that the sum of the
token-level rewards exactly equals the original sequence reward". Expanding
Eq. (3) gives

```
Σ_i R^P2T_i = Σ_i (R + ω·R·p_i) = (1 + ω)·R        with Σ_i p_i = 1
```

and the paper's own `Σ_i` runs over the N response tokens, so `(N + ω)·R`. A
convex combination `R·(1−ω)/N + ω·R·p_i` would sum to R; the paper does not write
that. We implement Eq. (3) verbatim and pin the actual value in
`tests/p2t/test_reward.py::test_eq3_matches_hand_computed_values_and_the_paper_sum_is_not_R`.

Practical consequence: Eq. (5) adds a **per-response constant** `α·R·(1 + ω/T)`
to every token advantage. With a reward model scored in the tens and α = 0.1,
that is a large shift on top of an O(1) group advantage. Every step logs
`p2t_bonus_over_advantage` so the magnitude is visible rather than assumed.

### 2.2 Eq. (3)'s softmax has no temperature

`I` is an inner product of gradient and embedding difference, with no intrinsic
scale. If `|I|` is large the softmax is one-hot and one token absorbs the bonus;
if small it is nearly uniform and the token term is a constant. Nothing in the
paper bounds it.

This is the same failure the parent project hit at p9c (`tau失配与修复方案.md`:
raw-utility softmax at τ=1 measured ESS 0.998, mechanism inert). We do not add a
temperature — the user asked for a strict reproduction — but the run logs the
diagnostics that reveal it:

* `credit_ess_ratio` — `ESS/T = 1/(T·Σp²)`; `→ 1` is a **flat** softmax (inert),
  `→ 1/T` is one-hot. This is the same normalisation and the same reading as the
  VPO-RM arms' quantity of the same name: near one means the weighting is doing
  nothing.
* `p2t_flat_response_fraction` — share of responses with `max p ≤ 1/T + 1e-3`
* `p2t_onehot_response_fraction` — share with `max p ≥ 0.9`
* `p2t_varying_bonus_over_advantage` — `mean|α·ω·R·(p − 1/T)| / mean|Â|`. This is
  the part of the token bonus that actually varies across tokens; the plain
  `p2t_bonus_over_advantage` includes Eq. (3)'s per-response constant
  `α·R·(1 + ω/T)` and so barely moves between the flat and one-hot regimes.
* `p2t_sign_flip_fraction` — share of tokens whose `Ã` has the opposite sign to
  their response's `Â`. Large values mean the constant term, not the outcome,
  is deciding the update direction.
* `p2t_zero_attribution_share_mass` — share of the softmax mass landing on tokens
  with `I = 0` (unmapped tokens and any mapped token with a zero gradient). High
  values mean the bonus is being spent where the reward model gave no signal.

Because the paper's α = 0.1 presumes an O(1) reward and Skywork scores are an
order of magnitude larger, expect Eq. (3)'s constant term to dominate. The
diagnostics above are what make that visible; nothing rescales it.

### 2.3 The paper's R is a process reward; ours is a sequence reward

The paper attributes a *process reward model's* step score. Here the reward model
is Skywork-Reward-V2-Qwen3-8B, which emits one scalar for the whole response.
P2T is defined for "any differentiable coarse-grained reward model", so applying
it is legitimate, but the setting is not identical to the paper's and is not
claimed to be. This is "P2T with this project's reward source".

Because the reward is sequence-level, Skywork's raw scores (roughly ±30 on this
data, per `docs/results/2026-09-20/main.csv`) are far from the O(1) scale that
α = 0.1 presumes. That gap is *measured* (σ0, and the reward distribution in
`metrics.jsonl`) and reported, never corrected.

## 3. Structural consequence of a chat-template reward model

Skywork pools its score at the **last valid position** of the canonical chat,
which in Qwen's template is `<|im_end|>`/EOS — a special token. `mapping.py`
follows the parent project's byte-exact mapping, which assigns `-1` to any actor
token without an exact reward-model counterpart, and the gather step turns those
into a **zero** gradient.

So the attribution at the exact position the reward model reads is structurally
zero, and the attribution vector is effectively truncated there. This is a
property of applying Eq. (2) to a chat-template reward model, not a defect in the
implementation. It is logged as `p2t_unmapped_share_mean` and left alone.

A zero gradient is also why unmapped positions keep `I_i = 0` and still
participate in Eq. (3)'s normaliser: the paper's own reading of `I ≈ 0` is a
negligible marginal effect, so re-normalising them away would change the formula.

## 4. Which R enters Eq. (3)

`raw_rewards` — the reward model's pristine scalar — not the length-shaped
reward, and never the degeneracy floor.

1. The paper's R has no shaping term.
2. The soft length window can flip R's sign. Under Eq. (3) a negative R *rewards*
   the lowest-attribution token, so a shaped R would invert the method's meaning
   for reasons that belong to this project's reward pipeline, not to the paper.
3. `guard_degenerate_rewards` floors flagged responses to `group minimum − penalty`,
   which is a **cross-response coupled** value. Feeding that into Eq. (3) would
   make one response's token rewards depend on its group's other members, which
   has no counterpart in the paper.

The cost is that the two terms of Eq. (5) live on different scales. That is what
α = 0.1 exists to temper, and `p2t_bonus_over_advantage` reports it.

## 5. Deliberately not implemented

The paper has no allocator, so none of the parent project's VPO machinery is
reachable from this package: there is no `tau`, no `credit_lambda` weight band,
no `freeze_stop_tokens` and no `freeze_structural`. `TrainerConfig` simply has no
such fields — `tests/p2t/test_trainer.py::test_config_has_no_vpo_allocator_knobs`
asserts their absence, so a P2T run cannot silently inherit them.

## 6. Numerical details

* **Max-subtraction in Eq. (3) is required, not a liberty.** `exp` overflows
  float32 past `I > 88.7`, and `I` is an unnormalised inner product of
  bf16-derived gradients. `exp(I−m)/Σexp(I−m) ≡ exp(I)/Σexp(I)` per response;
  `test_max_subtraction_is_an_identity_against_a_float64_reference` checks the
  implementation against a float64 reference that never subtracts.
* **Sign.** Eq. (2) is `e_i − e_∅`. The paper's introductory prose says "the
  difference between the null token embedding and the token's original
  embedding", which is the negation of its own equation; Eq. (1) settles it, and
  `test_taylor_attribution_is_exact_for_an_affine_reward_model` verifies the
  sign against a re-forwarded reward.
* **Attribution is computed on the reward device** — the RM embedding matrix and
  its gradients are already there — and only the three `[B, T]` credit fields
  cross to the actor device.

## 7. What is shared with the project, and how that is checked

`tests/p2t/test_conformance.py` imports the parent project from the same
checkout and asserts equality on everything that must not drift:

* `group_advantages` vs `vpo_rm.core.group_advantages`, both eps and std-floor modes
* `grpo_policy_loss` vs `vpo_rm.core.grpo_policy_loss`, value **and** gradients,
  with and without rollout importance weights
* `build_reward_input` vs `vpo_rm.reward_inputs.build_reward_input` on a real
  tokenizer, comparing the RM token IDs and every mapped position
* `soft_length_penalties`, `response_degeneracy`, `calibrate_reward_scale` vs
  `vpo_rm.length_reward`

The training protocol is likewise mirrored: 8 prompts × 8 responses, physical
microbatch 1, lr 5e-5, β 0.03, KL to the frozen initialisation, clip 0.2, AdamW
with wd 0.01 and grad-norm 1.0, temperature 1.0 with top_p 1.0 / top_k 0, soft
length window with `advantage_std_floor_fraction` 0.5, and the shared 128-prompt
σ0 calibration. The FP32 output head (`policy_head_dtype: float32`), the prompt
length filter, the rollout importance correction, the resample-then-drop policy
for wholly-bad prompt groups, and the response-termination check are all ported
for the same reason.

## 8. Review record

An independent review of the first complete draft found one fatal defect and a
set of comparability gaps. All were addressed; the list is kept here because
several of them are the kind of thing a reader would otherwise have to
rediscover.

| Finding | Disposition |
|---|---|
| The RM gradient was checked but never gathered onto actor positions, so `score_responses` returned `[B, L_rm, D]` with a per-chunk width and crashed on `torch.cat` | Fixed in `rm.py`; `tests/p2t/test_rm.py` now covers unequal canonical widths, an autograd oracle for the gathered rows, and microbatch invariance |
| No prompt length filter, so an over-long prompt would kill a run mid-training instead of being dropped | `filter_prompts` ported and applied before training |
| Shipped configs carried a `_comment` key the loader rejected | `load_config` ignores keys beginning with `_` |
| The ESS convention was documented backwards in three places, including the reward-curve axis label | Corrected; `test_share_ess_convention_matches_the_project` pins flat → 1 |
| `p2t_bonus_over_advantage` is dominated by Eq. (3)'s per-response constant and so cannot detect an inert softmax | `p2t_varying_bonus_over_advantage`, `p2t_sign_flip_fraction`, `p2t_zero_attribution_share_mass` added |
| `degenerate_responses` was structurally always zero | Now reports the flagged set actually floored |
| `p2t_unmapped_share_mean` measured token counts, not share mass | `p2t_zero_attribution_share_mass` added |
| FP32 output head missing on the HF side while vLLM used one | `policy_precision.py` ported and installed; sampler-vs-trainer log-prob agreement now logged |
| Response termination metadata never validated | `validate_response_termination` ported |
| No resample/drop for wholly-bad prompt groups | `select_training_rollout` ported |
| Adapter path reused every step, which can let vLLM serve a cached adapter and silently go off-policy | Distinct `step-N` paths with pruning |

Two further defects were found by the new tests rather than by review: the
selection packer was indexing rows with group indices, and the group-order
convention after a resample was undocumented. Both are fixed and pinned.

Residual, deliberate: `select_training_rollout` places first-pass survivors
before resampled groups, so a step's prompt order is not always the corpus order.
Group order inside a rollout changes no group-relative quantity.
