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

Approximation: the vanilla first-order Taylor expansion of Eq. (2), the only
estimator the paper actually defines. See §2.4 for why that is nonetheless a
deviation from the paper's own headline configuration. `P2T_APPROXIMATION`
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
* `p2t_sign_flip_fraction` — share of tokens where `sign(Â + bonus) ≠ sign(Â)`,
  i.e. the bonus is large enough and opposed enough to reverse the response's
  direction. It is token-weighted, so a short reversed response is diluted by
  long ones; read it as a whole-run proportion, not a per-response verdict.
* `p2t_zero_attribution_share_mass` — share of the softmax mass landing on tokens
  with `I = 0`. That mixes two populations: unmapped tokens (specials, BPE
  rewrites, and the pooling position itself) and mapped tokens whose gradient
  happens to be zero. High values mean the bonus is being spent where the reward
  model gave no signal. It reads 1.0 by construction when the whole attribution
  vector is zero, which is the fully inert case.

### Measured: the token term is inert at this reward scale

An independent audit sampled eight responses from the real `Qwen3-14B-Base`
under the rollout contract and scored them with the real Skywork reward model
through the shipped mapping. The result is the failure mode this section
predicts, quantified:

| quantity | measured |
|---|---|
| attribution spread `std(I)` | 0.007 - 0.029 |
| Eq. (3) softmax `ESS/T` | **0.9992 - 0.99995** |
| `p2t_flat_response_fraction` | **1.0** |
| the token-*varying* part of `A~`, relative to the update scale | **2.2e-6** |

At `T ~ 821` the attribution range spans only about `exp(0.5)`, so no token can
take more than ~2.4x its uniform share. **This is not the null token's doing**:
`<|endoftext|>`, `<|im_end|>`, the RM's own `<|vision_pad|>`, `<|fim_pad|>`, a
zero embedding and the vocabulary mean all give `ESS/T` between 0.99980 and
0.99989. The spread is set by `grad R . e_i`; the untempered `exp` in Eq. (3)
is what discards it.

The practical reading is that this arm, as the paper specifies it, is GRPO with
the advantage shifted by roughly `alpha*R`. That is a property of
(paper formula + sequence-level reward + long responses), and it is reported as
a result rather than repaired -- the same call the p9c post-mortem made for the
VPO allocator's own untuned softmax.

`p2t_bonus_over_advantage` cannot see this: it printed 1.06387 identically to
six digits in the flat, measured and one-hot regimes. `credit_ess_ratio` and
`p2t_flat_response_fraction` are the metrics that can.

A reading caveat on `p2t_varying_bonus_over_advantage`: for a one-hot share it
equals `2αω|R|(1 − 1/T)/T`, so it shrinks with response length in *both* regimes.
Only the flat-versus-peaked contrast carries information, not the absolute value.

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

### 2.4 The paper's headline number uses an estimator it never defines

Table 3(g) reports two attribution estimators: "Vanilla in Eq.(2)" at 52.6 and
"Integrated in Eq.(4)" at 53.8. The value 53.8 is also what every *other* ablation
row reports for the default configuration — pad null token in 3(a), ReasonFlux-PRM
in 3(d), P2T token reward in 3(f). So the paper's default, and therefore its main
tables, appear to use the **integrated-gradient** estimator, while plain Eq. (2)
is 1.2 points worse.

The paper never writes that estimator down. Its own cross-reference is broken:
"Integrated in Eq.(4)" points at the GRPO group-advantage equation, and no
integration path, baseline or step count appears anywhere in the text. §3.3.1 only
says the score comes from "the aforementioned vanilla gradient-based approximation
or the integrated gradient approximation".

We therefore implement Eq. (2) as written. This is the one place where a strict
reproduction of the paper's *equations* cannot also reproduce the paper's *numbers*,
and it is recorded rather than guessed at: inventing an integration path would be a
deviation from the paper dressed up as fidelity. Every artifact is stamped
`taylor_first_order_eq2` so no result is ever mistaken for the integrated variant.

### 2.5 ω = 0.6 is the stated default but is absent from the ω ablation

§4.1 states "For the hyperparameter ω in Eq. (3), we set ω = 0.6 by default", and
we use 0.6. But Table 3(b) sweeps ω over {0.25, 0.5, 0.75, 1.0} → {53.5, 53.8,
53.4, 53.2}: the default value is not in its own ablation grid, and the best grid
point is 0.5. The table also labels the row "ω in Eq.(5)" although ω appears in
Eq. (3), matching the mis-numbering in 2.4.

We follow the text (0.6), not the grid, because the text is the explicit statement
of the default. `omega` is a config field and is recorded in every run manifest,
so an ω = 0.5 arm is a config change rather than a code change.

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

## 9. Two defects found by the first launch attempt

Neither is about P2T's mathematics; both would have stopped the run regardless of
the credit-assignment scheme, and the unit suite was fully green with both present.

### 9.1 The generation server died before step 1

`runs/p2t-pilot/vllm_server.log` records

```
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid argument'
Worker proc VllmWorker-1 died unexpectedly
EngineCore failed to start.
```

and the trainer exited on `vLLM server exited with code 1`.

The L20 is compute capability 8.9 with PCIe-bridge peer-to-peer and no NVLink,
where vLLM's custom all-reduce is not safe. The code already intended to disable
it, but did so by exporting `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` — and the same log
answers `Unknown vLLM environment variable detected` while the resolved engine
config still reads `disable_custom_all_reduce=False`. That environment variable
does not exist in the installed vLLM; `EngineArgs` carries a
`disable_custom_all_reduce` field instead. The guard was a dead string.

The setting now travels as `--disable-custom-all-reduce` on the server's command
line into `LLM(...)`, the server prints `custom_all_reduce_disabled=` at startup,
and it raises if the backend reports the feature still enabled. Two standalone
probes had reached `vllm_server_ready` on the same two cards, so the fault is
intermittent rather than deterministic — which is exactly why it is disabled
outright rather than retried.

### 9.2 No step could ever have pushed

Every shipped config sets `push_branch: "main"`, because the push remote is a
dedicated P2T repository whose default branch is `main`. `_push_locked` compared
that name against `git rev-parse --abbrev-ref HEAD`, which is `p2t-baseline` in
this checkout, and returned `refused_wrong_branch` before staging anything. The
"push every 5 steps" requirement was inert.

`push_branch` now means the branch **on the remote** and `push_local_branch` the
branch this checkout must be on; the push uses an explicit
`p2t-baseline:main` refspec. The guard against committing from an unexpected
branch is unchanged, and still tested.

### 9.3 Every phase timing was mislabelled by one stage

`phase(name, start)` records the elapsed time *between* `start` and the call, so
each call has to come **after** the work it names. The first call sat before
generation, which shifted every label by one position: `generation_sec` reported
roughly zero, generation's real cost was logged as `reward_model_gradient_sec`,
the reward model's under `credit_sec`, and so on down the chain. The measured
time budget is backfilled from these keys, so the numbers were describing the
wrong stages. All seven calls now follow their work, and
`test_every_phase_timing_is_reported_and_nonnegative` pins both the key set and
the constraint that the phases sum to no more than the step.

### 9.4 Verified

The launch that follows these three fixes cleared the pilot's failure point:

```
custom_all_reduce_disabled=True
disable_custom_all_reduce=True          # the pilot's log read False here
vllm_server_ready
```

with no `Cuda error`, no `died unexpectedly` and no `EngineCore failed to start`.
Device placement came up as planned: actor ~32 GiB on cuda:0, reward model
~15 GiB on cuda:1, generation ~38 GiB on each of cuda:2/3 at
`gpu_memory_utilization=0.85`. Engine init took 136 s of which 66 s was
compilation, because the compile cache had to be rebuilt (see below).

### 9.5 `ADAPTER_LOAD_FAILED` in the old log was a false alarm

`logs/.adapter_check.log` ends in `ADAPTER_LOAD_FAILED`, which looks like the SFT
initialisation is unusable. It is not. That check compared the adapter file's key
set against the in-memory PEFT parameter names, and those two namings differ by
design: a saved adapter holds `...lora_A.weight` while a mounted adapter holds
`...lora_A.default.weight`. A raw set difference therefore reports *all* 560
tensors as simultaneously "missing from model" and "not in adapter", which is
exactly the symptom in that log, printed next to `all shapes match: True`.

Inspected directly, `models/sft-p2t/adapter_model.safetensors` is healthy:

| property | value |
|---|---|
| tensors | 560 (= 40 layers x 7 modules x 2 matrices) |
| target modules | q/k/v/o/gate/up/down, 80 tensors each |
| layers covered | 0 - 39, all 40 |
| `lora_B` magnitude | nonzero (mean abs 6.9e-4) -- trained, not fresh init |

`adapter_config.json` declares r=64, alpha=128, dropout=0, and
`base_model_name_or_path: models/Qwen3-14B-Base`, so the trainer's
`_verify_init_adapter` binding check passes and records the weight hash.

One real gap: there is no `sft_manifest.json` beside the adapter. The tokenizer
protocol assertion in `load_actor_tokenizer` and the full
`scripts/verify_sft_adapter.py` both key off that file, so neither runs for this
adapter. The adapter directory also carries no tokenizer export, so the actor
tokenizer resolves to the base model's -- correct here, but unverified against
whatever the SFT stage actually used.

### 9.6 `plot_reward.py` crashed on its own metrics file

The trainer writes non-step events to `metrics.jsonl` -- `prompt_filter` is
emitted before training begins and has no `"rollout"` key. The plotter mapped
`row["rollout"]` over every row and raised `KeyError`. Because `watch_run.sh`
redirects the plotter into `health.log`, a live run's curves silently failed to
redraw on every 300 s tick while the run itself was perfectly healthy.
`load_metrics` now keeps only rollout rows and says so explicitly when none exist
yet.

## 10. Environment casualty: the runtime was rebuilt before this run

Between the failed pilot and this run, everything under `/root` was lost. The
virtual environment survived only in part: its 11 GiB of `site-packages` sits
under the repository, but `.venv/bin/python` is a symlink into
`/root/.local/share/uv/python/…`, and both that interpreter and the `uv` binary
were gone. Every interpreter invocation failed with "No such file or directory"
even though the path existed, which is the signature of a dangling symlink.

Repair, following `setup_env.sh` rather than improvising: reinstall `uv`,
`uv python install 3.12`. That recreated `cpython-3.12.14` at the version-generic
path the symlink targets, so the existing packages were reused as-is — no
3.1 GiB wheel reinstall. Confirmed afterwards: torch 2.13.0+cu129 with 7 visible
devices, vLLM 0.28.1rc1.dev199, transformers 5.16.1, peft 0.20.0, and the full
suite green at 93 passed.

Two casualties are **not** repairable from inside the repository:

* `/root/.cache/vllm` — the torch.compile cache. Costs about 65 s of extra
  engine init on a cold start; correctness is unaffected.
* `/root/.p2t-git-credentials` — the push token. `credential.helper` still points
  at it, so `git push` has no credentials. The remote is reachable and **empty**
  (`git ls-remote` exits 0 with no refs), so `main` does not exist yet and the
  refspec would create it. Until the token is restored, every autopush attempt
  logs `push_failed` in `reports/<run>/git_push.log`; the commits still land
  locally and nothing is lost but the upload. `_run` now sets
  `GIT_TERMINAL_PROMPT=0`, which matters specifically because the run is detached
  and has no terminal: without it git could block waiting for a username that
  nobody can type.

## 11. Measured cost (10-step smoke, Qwen3-14B-Base, no SFT adapter)

Backfills the plan's estimated budget with what the hardware actually did. Three
rollouts of 8 prompts x 8 responses = 64 responses, mean 627 response tokens.

| stage | plan estimate | measured (rollout 1) |
|---|---|---|
| vLLM generation, 64 responses | 1.5 - 3 min | **1.4 min** (83.7 s) |
| RM forward + input backward, mb=1 | 2 - 3 min | **0.6 min** (38.3 s) |
| Eq. (2)-(5) credit | n/a | 0.2 s |
| actor old logp | 1 - 1.5 min | **1.1 min** (66.7 s) |
| reference logp (LoRA disabled) | 1 - 1.5 min | **0.9 min** (52.4 s) |
| actor update, fwd+bwd x64 | 2.5 - 4 min | **3.7 min** (220.3 s) |
| **total per step** | **8 - 13 min** | **7.7 - 9.9 min** (462 - 595 s) |

The estimate held. The reward model was about 3x faster than assumed; the actor
update dominates, as expected at microbatch 1 with gradient checkpointing.

Peak memory, against 45.0 GiB usable per L20:

| device | holds | measured peak |
|---|---|---|
| cuda:0 | actor bf16 + FP32 head + LoRA + AdamW + activations | **38.6 GiB** |
| cuda:1 | frozen RM + input-gradient graph | **28.6 GiB** |
| cuda:2,3 | vLLM TP=2 at `gpu_memory_utilization=0.85` | **38.0 GiB each** |

Two notes for the formal run, which differs from the smoke in ways that move these
numbers:

* It mounts the SFT adapter **twice** -- `default` (trainable) and `ref` (frozen
  KL reference) -- where the smoke had one freshly-initialised adapter and used
  the base model as its reference. That is roughly +0.5 GiB of bf16 weights on
  cuda:0 on top of a measured 38.6 GiB peak. Headroom is thin but real; it is
  worth watching rather than assuming.
* Its `sigma0` is the shared 3.0323 calibrated with the sibling arms, not the
  smoke's self-calibrated 6.7349. The soft length window is expressed in units of
  `sigma0`, so the length penalties in the formal run are roughly half the
  smoke's in absolute reward terms.

Disk: each step writes a 1.0 MiB credit dump plus about 0.25 MiB of prompt/token
JSON, so 250 steps is about 320 MiB, and `keep_adapters=2` caps the vLLM adapter
directory at about 2.0 GiB. Against 9.2 TiB free this is not a constraint.

## 12. Checkpoints, and what a restart does and does not preserve

The formal run is ~30 hours, so it checkpoints every 20 rollouts and keeps the
two most recent (`checkpoint_interval: 20`, `keep_checkpoints: 2`). The interval
does not divide 250, which is fine: `train()` saves a final checkpoint
unconditionally after the loop, so a completed run ends holding `checkpoint-240`
and `checkpoint-250`. Each checkpoint is about 2.0 GiB (LoRA `default` + frozen
`ref`, plus the tokenizer), so the policy costs 4 GiB steady-state rather than
the 26 GiB that keeping all thirteen would.

Before this, `checkpoint_interval` was 0 — nothing was written until step 250, so
a crash at hour 25 would have left only the per-step vLLM adapter snapshots under
`vllm-adapters/` (themselves pruned to the newest two).

**There is no in-place resume.** `check_fresh_output` refuses to relaunch into a
used run directory, deliberately: two attempts interleaved in one `metrics.jsonl`
under duplicated rollout numbers would be unanalysable. What a checkpoint
supports is *relaunching as a new run seeded from those weights*: point a new
config's `init_adapter` at the checkpoint directory and give it a fresh
`output_dir`/`report_dir`. Every checkpoint's `run_manifest.json` now records
that recipe inline, together with what does not carry over:

* **AdamW moment estimates** are not saved, so the first steps after a restart
  take a larger effective step than they otherwise would.
* **The KL reference identity changes.** `beta`-KL is measured against
  `init_adapter`, so a restart re-anchors the reference to the checkpoint's
  weights instead of the original SFT initialisation. The restarted segment is
  therefore not simply a continuation of the same objective.
* **The prompt cursor resets.** Prompts are walked as
  `(rollout_index * prompts_per_rollout) % len(prompts)`, so a restart begins
  again at the start of the corpus and re-trains on prompts the first attempt
  already saw.

None of this is fatal for a baseline arm, but a restarted run is a different
experiment from an uninterrupted one and should be reported as such rather than
spliced into one curve.

### Memory, and the two ways to read it

Actor peak on cuda:0 measured 36.4 GiB via `torch.cuda.max_memory_allocated`
while `nvidia-smi` showed 39.3 GiB for the same process. The ~3 GiB gap is the
CUDA context plus memory the caching allocator holds but is not using;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation but does
not return the reservation. Judge headroom by the logged `actor_peak_gb`, not by
`nvidia-smi`, or the run looks about 3 GiB closer to the limit than it is.

Activation checkpointing is already enabled (`gradient_checkpointing: true`) and
the physical microbatch is already 1, so those levers are spent. If
`actor_peak_gb` ever approaches ~42 GiB the remaining lever is
`token_chunk_size` (128 -> 64), which trades throughput in
`selected_logp_from_logits` and `response_entropy` for a smaller transient.
