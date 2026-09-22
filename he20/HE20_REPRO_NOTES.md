# Reproduction notes — the high-entropy minority token arm (`he20`)

Paper: *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
Reinforcement Learning for LLM Reasoning* — Shenzhi Wang, Le Yu, Chang Gao, Chujie
Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang,
Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang,
Junyang Lin (Qwen Team, Alibaba Inc.; LeapLab, Tsinghua University).  NeurIPS 2025,
arXiv **2506.01939v2** (first version 2025-06-02, last revised 2025-11-13).

Local copies, kept out of git (`paper/` is `.gitignore`d at line 20 of the repo's
`.gitignore`): `paper/80-20-rule-high-entropy-minority-tokens.pdf` and the extracted
`paper/high-entropy.txt` (25 pages, 88 262 characters, extracted with `pypdf` — the
repo's own `paper/extract_pdf.py` is written for a different PDF and returns font
licence garbage for this one).  The authors' reference implementation is
`github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`, a `verl` fork at commit
`6cf90ceb079bdc0721b51c23e0107410651ccd82`; the four files that matter were read into
`/home/ml-user/workdir/ref-he20/` (outside every repository) and are cited below by
their paths inside that fork.

This arm reproduces the paper's **method** strictly — Eq. (1)'s entropy, Eq. (6)'s
mask, and the normaliser change that comes with it.  Where the paper is silent or
self-contradictory the code follows what it says and the discrepancy is recorded
below rather than repaired.  Where the paper's *setting* differs from this project's
it is recorded in §5; those are deliberate departures, not oversights.

## 1. Equation to code

| Paper | Meaning | Code |
|---|---|---|
| Eq. (1) | `H_t := -Σ_j p_t,j log p_t,j`, `p_t = π_θ(·|q,o_<t) = Softmax(z_t/T)` | `policy.response_entropy` |
| Eq. (3) | group-relative advantage `Â_t^i = (R_i - mean{R}) / std{R}` | `reward.group_advantages` |
| Eq. (4) | the base algorithm's clipped surrogate | `loss.grpo_policy_loss` (the project's, unchanged) |
| Eq. (6) | `I[H_t^i ≥ τ_ρ^B]` on the surrogate **and** on the normaliser | `mask.entropy_top_mask`, `loss.grpo_policy_loss(..., entropy_top_mask=...)` |
| §5.2 | `ρ = 20%` | `configs/he20250.json:entropy_top_ratio` |
| §5.1 | `max_response_length = 20480`, overlong buffer 4096, penalty 1.0 | **not reproduced** — §5 |
| §5.1 | `train_prompt_bsz = 512`, `mini = 32`, lr `1e-6`, no warmup | **not reproduced** — §5 |
| §5.1 | `use_kl_loss = False`, `kl_coef = 0` | **not reproduced** — §5 |

The arm's whole method is one flag in the authors' own run scripts: `diff` between
`recipe/rlvr_with_high_entropy_tokens_only/run_dapo_qwen3_14b.sh` and
`…/run_only_top20_high_entropy_tokens_dapo_qwen3_14b.sh` shows exactly two lines, an
`exp_name` and `actor_rollout_ref.actor.entropy_top_ratio=0.2`.  Everything else
about the two runs is identical, which is why this arm's loss is the project's GRPO
surrogate with a mask and nothing else.

## 2. Where the paper is silent, ambiguous, or contradicts its own code

### 2.1 The mask is a threshold in the paper and a top-k in the reference code

Eq. (6) and its gloss define the mask by a **threshold**:

> "`τ_ρ^B` is the corresponding entropy threshold within the batch B such that only
> tokens with `H_t^i ≥ τ_ρ^B`, comprising the top-ρ fraction of all tokens in the
> batch, are used to compute the gradient." (p.7)

so tokens tied at `τ` are **all** kept and the retained fraction can exceed `ρ`.  The
reference implementation
(`verl/trainer/ppo/core_algos.py:53`, `get_global_entropy_top_mask`) instead sorts:

```python
top_k = max(1, int(len(response_entropy) * top_ratio + 0.9999))   # ceil(n*rho)
_, topk_idx = torch.topk(response_entropy, k=top_k)
```

keeping exactly `ceil(ρn)` and breaking ties by index.

**The code here follows the paper**: `rule="threshold"` is the default and is what
`configs/he20250.json` uses; `τ_ρ^B` is the `(1-ρ)` quantile of the pooled response
entropies.  The reference's rule is reachable as `rule="topk"`.  The two agree
whenever the entropies are distinct — which they almost always are, being floats
from a softmax — and disagree on ties, where the paper keeps more.
`tests/he20/test_mask.py` pins both: agreement on distinct values against a
transcription of the authors' function, and disagreement on an all-equal population.

### 2.2 "batch" or "(micro-)batch": the paper uses both, 16× apart

§5.2 says "the top 20% highest-entropy tokens **within each batch**".  The paragraph
introducing Eq. (6) says "ensuring that within each **(micro-)batch** B, only tokens
`o_t^i` whose entropy `H_t^i ≥ τ_ρ^B` are involved in the policy gradient loss".  The
paper's own configuration has a training batch of 512 and a mini-batch of 32, so the
two readings differ by a factor of 16 in pool size and by which tokens survive.  The
paper never reconciles them.

The reference implementation resolves it silently in favour of its micro-batch,
because it computes the entropy inside `_forward_micro_batch` — one forward pass that
yields both the entropy and the log-probabilities — and pools within that call.

**This arm pools over the optimizer minibatch** (see §3).  That is the reading that
matches this project, where the physical micro-batch is a single response.

### 2.3 There are two different "20%"s and the paper never reconciles them

§3's decoding-temperature experiment uses an **absolute** threshold
`h_threshold = 0.672`, described as "the 80th percentile among the sampled 10⁶
tokens" (p.5), and Eq. (5) uses it to perturb individual tokens during a controlled
experiment.  The training method uses a **batch-relative quantile** `τ_ρ^B`.  These
are different objects: one is a fixed number, the other moves with every batch.  Only
the second is the training method, and only the second is implemented here.

### 2.4 When the entropy is computed is never stated

Eq. (6) writes `H_t^i` with no policy subscript.  §3.2 does say the entropy is the
*training* policy's even off-policy — "the training policy is π_θ … The entropy is
still calculated using π_θ, as defined in Equation (1), to measure the uncertainty of
the training policy in the given sequence" (p.3) — but nothing says whether it is
computed once from the rollout's logits or re-derived at each of the 16 gradient
steps.  This matters, because `π_θ` moves across those steps and the mask moves with
it.

The reference implementation answers it: it computes the entropy in the **same
forward pass that produces the loss's log-probabilities**, i.e. from the current
`π_θ`, per micro-batch.  This arm matches that (§3), and the one-step guard there is
what keeps the match exact.

### 2.5 The `max(1, …)` guard is the code's, not the paper's

The reference clamps the kept count to at least one token; the paper says nothing
about populations smaller than `1/ρ`.  This arm needs the same guard for a different
reason: with the paper's threshold rule the population can, in principle, be ranked
so that no token clears `τ` — it cannot, in fact, because `τ` is a quantile of the
same values and therefore never exceeds their maximum, and
`test_the_paper_rule_keeps_at_least_the_maximum` asserts exactly that.  So no clamp is
needed under the paper's rule; the equivalent protection under `rule="topk"` is the
`ceil` (which is ≥ 1 for any non-empty population).

## 3. The one place this arm must choose: the selection population

This is the only design decision the paper leaves open that a reproduction cannot
avoid, so it is stated here and encoded in the code rather than left implicit.

**The population is the whole optimizer minibatch** — the responses whose gradients
are averaged into one `optimizer.step()` — not the physical micro-batch.  The project
runs `microbatch_responses=1`, so pooling over a micro-batch would rank a single
response against itself and keep its own top 20%, which is a different method from
the paper's batch-level mask.  §2.2 records that the paper's own words support either
reading.

Two consequences are implemented explicitly:

* The entropy used is the one the trainer already computes during the old-log-prob
  pass (`rollout_logp_microbatch(..., entropy_out=entropy)`), which is `π_θ`'s
  entropy because it is computed from the same weights the loss forward will use.
* The trainer therefore **requires exactly one optimizer step per rollout** whenever
  the mask is enabled, and raises a `ValueError` otherwise.  With more than one step
  the later steps' masks would need `π_θ`'s entropy recomputed after the earlier
  steps had already moved the weights; using the pre-step buffer there would be a
  silent approximation of the paper's rule rather than the rule.
  `configs/he20250.json` satisfies the guard by construction
  (`optimizer_minibatch_responses = group_size × prompts_per_rollout = 64`).

## 4. Diagnostics the paper does not define

The paper reports benchmark accuracy.  This project compares arms on a metrics row,
so the arm emits the shared vocabulary its siblings emit plus the mask's own
statistics: `entropy_top_kept_fraction` (the realised kept fraction, which the
threshold rule can push above `ρ` on ties), `entropy_top_threshold`,
`entropy_top_mean_kept_entropy` and `entropy_top_mean_all_entropy` (so the run shows
the kept tokens really are the high-entropy ones), and `entropy_top_ratio` /
`entropy_top_rule` so the row says which setting produced it.

`credit_ess_ratio` is **exactly 1** in this arm, by construction: there is no
token-level credit to concentrate, so the diagnostic measures nothing here.  It is
reported because the health checker and the cross-arm plots read the key, and the
value states the truth rather than being faked.

## 5. Deliberately not implemented

* **DAPO.**  The paper's base is DAPO (its Eq. (4), with clip-higher, dynamic
  sampling, a token-level loss and overlong reward shaping).  This arm's base is the
  project's plain GRPO, so that the mask is the only difference from the sibling
  arms it is compared against.  The consequence is stated plainly: the paper's
  reported gains are DAPO-with-mask against DAPO, and this arm's are mask against
  *this project's* GRPO.  Nothing here can reproduce the paper's AIME numbers, and
  the notes do not claim it.
* **The paper's setting.**  Its task is RLVR on mathematics with a verifiable
  answer (`is_equivalent(a, o)`), responses up to **20 480** tokens, Qwen3-8B/14B/32B,
  and the dataset DAPO-Math-17K.  This arm runs UltraFeedback with the frozen
  Skywork reward model and a 2048-token cap, which is the project's protocol.  The
  paper's mechanism analysis — "forking tokens" as reasoning decision points — is
  drawn entirely from math CoT, so its transfer to preference data and a scalar RM
  is an open question this arm measures rather than assumes.
* **`use_kl_loss = False`, `kl_coef = 0`.**  DAPO removes the KL term; this arm keeps
  the project's `beta` KL, because every sibling arm has it and dropping it would
  change the comparison in a second way at once.
* **The absolute threshold `0.672`** of §3 — see §2.3.
* **The paper's hyperparameters** (lr `1e-6`, batch 512, mini-batch 32, 16 gradient
  steps per batch, no warmup or schedule, `max_response 20480`).  This arm uses the
  project's, so that a difference between arms is not a difference of optimisers.
* **The KL term is not masked.**  Eq. (6) masks the surrogate; the KL is this
  project's separate constraint and the paper has no KL at all, so it says nothing
  about its scope.  The alternative reading — that "the policy is updated using only
  the gradients of the top 20%" should restrict the KL as well — would silently
  narrow the trust region to the masked tokens, which is a larger change than the one
  the paper makes.  Recorded, not adopted.

## 6. Numerical details

* Entropy is computed in FP32 blocks from the model-precision logits by
  `policy.response_entropy`, over the *supported* vocabulary (the same `output_mask`
  the sampler used), which is the distribution Eq. (1) means.
* The threshold is `torch.quantile(values, 1 - ρ)` with the default linear
  interpolation between order statistics.  Because interpolation can land strictly
  between two values, the realised kept fraction is `ρ` up to one token; ties at the
  threshold move it further, in the direction §2.1 records.
* `torch.quantile` requires a float input and is applied to the pooled valid tokens
  only; padded positions never enter the ranking.
* The loss reduces in float32, as the sibling arms' does; `test_a_mask_restricts_the_numerator_and_the_denominator`
  states its hand-computed expectations in that dtype.

## 7. What is shared with the project, and how that is checked

`he20/policy.py`, `rollout.py`, `mapping.py`, `tokens.py`, `tensors.py`,
`length_reward.py`, `data.py`, `vllm.py`, `vllm_server.py`, `policy_precision.py` and
`autopush.py` are ports of the sibling arms' modules, which are themselves ports of
`vpo_rm/`.  Their behaviour is **identical** to the siblings' apart from the arm's
own name in three runtime strings (the vLLM server module path, the LoRA adapter
name and the autopush commit prefix); the AST of each file, with docstrings stripped,
matches `red/`'s.  `tests/he20/test_conformance.py` asserts the parts that must agree
numerically:

* `loss.grpo_policy_loss` equals `vpo_rm.core.grpo_policy_loss` and `p2t`'s mirror
  at `atol=0, rtol=0`, gradients included, when the mask is off — which is also the
  paper's baseline claim;
* `reward.group_advantages` equals the parent project's implementation on random
  inputs, in both of its scale modes;
* the soft length window and the degeneracy floor match the project's.

Nothing in this package imports `p2t`, `red` or `vpo_rm` at runtime; the conformance
tests import them deliberately, and skip if they are absent.

### Comparability with VPO and P2T, checked rather than asserted

This arm is a comparison point, so "the arms differ only in the method" is a
contract with a test, not a claim.  `tests/he20/test_alignment.py` walks the chain
the configs themselves describe — VPO is the parent's
`corrected_rl_launcher.common_config`, `configs/formal250.json` (P2T) is that
protocol plus its credit rule, `configs/he20250.json` is P2T's plus the mask — and
asserts:

* **every training parameter the parent protocol defines matches**, through a
  named key mapping (18 parameters: rollouts, response cap, learning rate, beta,
  temperature, minibatch and microbatch, both seeds, the response-length window and
  its strengths, the advantage scale floor, calibration and degeneracy settings,
  the head dtype);
* the keys that deliberately differ are **named with their reason**, and the parent
  must still define them, so a rename upstream surfaces as a failure instead of
  silently dropping out of the comparison.  Those are `checkpoint_interval` and
  `keep_adapters_every` (delivery cadence), `vllm_gpu_memory_utilization` and
  `vllm_tensor_parallel_size` (this box's 46 GiB L20 cards against the H200s the
  protocol was written for), and the VPO-only knobs (`tau`, `kl_reference`,
  `length_reward_mode`, `length_penalty_slope`);
* **the initialisation is the same policy.**  The parent names
  `models/sft-native-eos-clean2k5e2`; this checkout has `models/sft-p2t`, and the
  test compares the adapter's weight hash against the one
  `configs/ssh-a6000-assets.json` pins for the canonical asset — `21c0c7b9…`,
  identical.  Two names for the same bytes, not two different starting policies;
  had they differed, no amount of parameter alignment would have made the arms
  comparable.
* the actor and reward model are the same pair by name.

Beyond the parameters, the *logic* was compared too: `he20/trainer.py` is a port of
`p2t/trainer.py`, and 18 of their 26 shared functions are AST-identical with
docstrings stripped — including rollout validation, prompt filtering, the
initial-adapter identity check, checkpoint pruning, the fresh-output guard and
`load_config`.  The eight that differ are `__init__`, `resolved`, `main`,
`train_rollout`, `_raw_rewards`, `save_checkpoint`, `_dump_credit` and the
diagnostics, and in `train_rollout` (282 lines against 330) every added line is
either the mask — the guard, the selection, the `entropy_top_mask=` argument — or
one of this arm's renamed seams; the deleted lines are P2T's attribution and
`p2t_credit` call and nothing else.  The rollout selection, the length window, the
KL term, the importance weights, the optimizer loop, the gradient accumulation and
the startup gates are untouched.

**One convention had to be corrected to make the comparison honest.**  `group_sigma_*`
is a shared diagnostic, and the three arms did not agree on what it means: VPO
(`vpo_rm/trainer.py:892`) and P2T both standardise by
`max(std, advantage_std_floor_fraction * sigma0)` and report *that* value, while RED
reports the raw population spread.  The floor is not a technicality — in `p2t250`
the reported `group_sigma_min` equals the floor on **233 of 250 steps** — so the two
quantities are not interchangeable.  This arm now reports the floored scale, because
that is both what its siblings report and what it actually divides by, and reports
the unfloored spread beside it as `raw_group_sigma_*` so that the project's
"every group has near-zero reward spread" watchdog still has a quantity that can
reach zero to fire on.  `test_the_shared_spread_key_reports_the_floored_scale_its_siblings_report`
pins both, using constant rewards where the raw spread is exactly zero and the
reported one is exactly the floor.

The one further reporting difference is the phase key: P2T writes
`phase_reward_model_gradient_sec` because its reward-model pass carries a gradient,
this arm writes `phase_reward_model_forward_sec` (the parent's own name for a
forward-only pass, `vpo_rm/trainer.py:855`) because its does not.  The names differ
because the work differs; a cross-arm phase comparison has to map them.

## 8. Review record

| Finding | Disposition |
|---|---|
| The paper's threshold rule and the reference implementation's top-k disagree on ties. | **Recorded**, §2.1. The paper's rule is implemented; the reference's is reachable and both behaviours are tested, including the disagreement. |
| The paper says "batch" and "(micro-)batch" for the selection population, 16× apart. | **Recorded**, §2.2, and resolved explicitly in §3 with the trainer's one-step guard rather than by a silent default. |
| §3's absolute `h_threshold = 0.672` and the training mask's relative `τ_ρ^B` are two different "20%"s. | **Recorded**, §2.3. Only the relative one is implemented. |
| The paper never says when the entropy is computed relative to the update. | **Recorded**, §2.4; resolved from the reference implementation and made exact by the one-step guard. |
| `credit_ess_ratio` is degenerate (= 1) in this arm. | **Recorded**, §4. Reported because the shared vocabulary requires it, with the value's meaning stated rather than papered over. |
| The KL term's scope under the mask is undefined. | **Recorded**, §5. The surrogate is masked and the KL is not, with the alternative reading named. |
| The `entropy_top_mask` goes into the loss *mask*, not the numerator alone. | **Pinned by test**: `tests/he20/test_loss.py::test_a_mask_restricts_the_numerator_and_the_denominator`. Restricting only the numerator would be Eq. (6) with its normaliser unchanged, i.e. reweighted rather than restricted. |
| A response can keep no token, which would divide by zero under a naive masked mean. | **Handled**: the numerator is zeroed there and the count clamped, so the response contributes nothing — what Eq. (6)'s indicator sum does — and `test_a_response_that_keeps_no_token_contributes_nothing` pins it. |
| The loss first rejected an all-False mask, which is wrong once it is called on micro-batch *slices* of a batch-level mask: a slice may legitimately keep nothing. | **Fixed.** The check was removed from the loss and left where the whole population is visible, in `mask.entropy_top_mask`; `test_a_slice_that_keeps_nothing_contributes_nothing_rather_than_raising` records both halves, including that the mask's own "kept nothing" branch is unreachable through the public API because `tau` never exceeds the maximum it is a quantile of. |
| `group_sigma_*` did not mean the same quantity as in VPO and P2T (raw spread here, floored scale there), on a key the health checker and the cross-arm plots read. | **Fixed**, and recorded in §7 with the measurement that makes it non-theoretical (the floor binds on 233 of p2t250's 250 steps). The arm now reports the floored scale, matching its siblings and its own divisor, with `raw_group_sigma_*` beside it for the zero-spread watchdog; both are pinned by a test. |
| The aligned `group_sigma_*` can never reach zero, so the watchdog's near-zero-spread check would have become unfirable. | **Fixed** with the same change: the checker tests `raw_group_sigma_mean` when present and falls back to `group_sigma_mean` for the sibling arms' rows. |

### Verified solid (independent re-derivation, not just inspection)

The unmasked loss's bit-identity with `p2t`'s and `vpo_rm`'s, gradients included
(`atol=0, rtol=0`); the threshold rule's agreement with a transcription of the
authors' `get_global_entropy_top_mask` on distinct values and its divergence on ties;
that padded positions never enter the ranking even when they hold the largest value;
that a single-token population survives every ratio; and that only kept tokens
receive gradient.

## 9. Run record

*(no run yet)*
