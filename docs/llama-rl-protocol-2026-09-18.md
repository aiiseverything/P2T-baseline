# Llama 3.1 RL protocol audit and experiment

This experiment adapts the corrected Qwen suite `runs/rl-fp32-is-canonical-20260917` to Meta Llama-3.1-8B-Instruct and Skywork-Reward-Llama-3.1-8B-v0.2. Four arms use the same completed Llama SFT adapter: GRPO and VPO lambda 2, 4, 8. The original Qwen results and their frozen sources remain separate.

## Verified input artifacts

- Actor: `/data/VPO-RM/models/Llama-3.1-8B-Instruct`, pinned upstream revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Reward model: `/data/VPO-RM/models/Skywork-Reward-Llama-3.1-8B-v0.2`, pinned revision `d4117fbfd81b72f41b96341238baa1e3e90a4ce1`.
- Initial adapter: `/data/VPO-RM/models/sft-llama31-8b-instruct-clean2k5e2-20260917`; weight SHA256 `52a68bd14ecea9bf660e0bb04a78cb68e6d2c263eedcfb9c40c13acba01453cc`.
- The SFT run used 2,500 clean UltraFeedback examples, two epochs, and 157 optimizer updates. Its saved adapter was reloaded on GPU, and the subsequent generation check completed 100/100 responses with native EOT. The generation recovery job used spawn after a post-training fork/OpenMP deadlock; it did not retrain the adapter.
- On 2026-09-18, all actor and reward download-manifest files, including every weight shard, were rehashed against verified upstream hashes. The adapter and its tokenizer were also checked against the SFT reload record. Evidence: `/data/VPO-RM/.maintenance/llama-rl-20260918/assets_verified.json`.

## Confirmed protocol defects and corrections

The previous generic RL prompt path rendered a native Llama BOS and then tokenized with `add_special_tokens=True`, introducing a second BOS. A representative native prompt has 38 tokens; that path produced 39. The HF and vLLM paths could agree with each other while both differed from the SFT format. Rendered chats now use explicit token IDs without additional special tokens. Generation requests bind those IDs to the rendered text, and the returned vLLM prefixes must match exactly. Overlong prompts are rejected instead of silently truncated.

The reward scalar input already contained the correct single BOS, but the text used for byte attribution had its BOS removed. This made full-stream reconstruction fail, leaving all answer-token positions unmapped. Keeping the same native rendered text and explicit tokenization corrects that mismatch. Llama's template also trims whitespace at both ends; attribution accepts only unchanged whole-token byte spans in the surviving body. Removed specials, ambiguous spans, and boundary merges retain fixed credit. For `Hello world!` followed by EOT, the expected positions are `[37, 38, 39, -1]`.

The old loader overwrote both tokenizers' padding with actor EOS. Llama now preserves its dedicated pad 128004 independently for actor and RM; the actor's output distribution suppresses that pad. Existing Qwen callers retain their shared actor-EOS padding convention. Stop tokens are the registered Llama IDs 128001, 128008, 128009. Backend-added special tokens are included when tokenizer exports omit them from `all_special_ids`, so decoding and attribution agree.

The actor uses the saved SFT tokenizer and template. Its automatic system text includes `Cutting Knowledge Date: December 2023` and fixed `Today Date: 26 Jul 2024`; the template SHA256 is `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`. An ordinary assistant response ends with native EOT 128009. Actor and RM vocabularies, backend segmentation, and normalization must match before token attribution is enabled.

The reward model is `LlamaForSequenceClassification`, with a finite BF16 `score.weight` of shape `[1, 4096]`. The intended reward is its raw scalar logit, pooled at the last attended token, including the final assistant EOT. No sigmoid or undocumented normalization is applied. The GPU gate compares the wrapper with direct HF scoring for singleton and left/right-padded inputs, verifies finite nonzero answer-input gradients and zero padding gradients, and verifies that RM parameters remain frozen.

## Controlled comparison

The launcher derives configuration from the frozen corrected Qwen manifest. It keeps 250 rollouts, 8 prompts x 8 responses, one policy epoch, 64-response optimizer batches, learning rate 5e-5, KL beta 0.03 to a frozen copy of the SFT initialization, temperature 1, training seed 42, and generation seed 0. Prompt and response limits are 2048 each. Actor/RM/vLLM occupy separate GPUs.

The native BF16 backbone is retained. As in the corrected Qwen RL experiment, the frozen policy projection and trainable LoRA tensors use FP32. Thus RL policy-head precision intentionally differs from the native BF16 SFT forward pass. The pinned vLLM implementation supports an FP32 Llama output projection; actual HF/vLLM token probabilities must still pass numerical checks on GPU. The detached, unnormalized token importance correction applies to both policy and KL terms.

Length reward retains the approved thresholds 8/1024/2048 and short/long strengths 0.5/2. A fresh initial-policy calibration on 128 distinct prompts x 8 answers determines the Llama reward scale. All four arms share this scale; the Qwen value is never reused. GRPO uses uniform credit. VPO uses tau 1, lambda 2/4/8, and fixed stop/structural credit.

The dataset SHA256 is `0f951ca4502001d31f3e4c70716ae51d20e4ce4f847d12b6a6695a40d4d353a8`. Deduplication and benchmark exclusions leave 61,097 unique prompts, with 2,000 held out. Native Llama filtering keeps 59,061 training prompts. The first 2,000 filtered prompts, covering all 250 formal rollouts, match the corrected Qwen run exactly; their ordered JSON SHA256 is `9c685ae9a4a652b8215ae50988614dd8723a6293222022e5381ab274f5f4f283`.

## Launch and validation

Development is in `/data/VPO-RM/code`. CPU regression tests and the actual-tokenizer protocol checker precede source freezing. The GRPO rjob first runs the real RM GPU audit and four two-rollout training gates, including adapter initialization/update/reference checks, sampled probability comparisons, response mapping, basic quality controls, and a long-sequence capacity check. It then reloads the protected SFT for its formal run. The other three rjobs can start only after the shared GPU evidence and calibration are complete and hash-bound.

Each rjob requests three GPUs with at least 120 GiB each, 48 CPUs, 600000 MiB RAM, the existing pinned-package GPU image, the shared project mount, and explicit NFS `/data`. `VLLM_WORKER_MULTIPROC_METHOD=spawn` is mandatory. No GPU-time package installation is planned. Initial available space was about 122 GiB; the launcher requires at least 60 GiB free. Protected SFT files are read-only inputs to this workflow.

CPU validation does not establish GPU numerical correctness. Until real GPU reports and formal completion artifacts exist, those stages remain pending. Jobs save stage, startup, error, calibration, source/input identity, runtime, and checkpoint evidence. The final checkpoint is step 250.

The existing IFEval/GSM8K raw-string generation paths also need explicit native Llama prompt-token handling before they are used for Llama evaluation. They are outside this RL launch change; previous Qwen evaluation records are unchanged.

## GPU launch status, 2026-09-18 01:44 HKT

The formal training gate has not passed. No Llama calibration, optimizer update,
or lambda2/4/8 formal job has started.

- Initial suite `llama31-rl-canonical-20260918`, job
  `llama-rl-grpo-afa68a64b0-22233133`, failed before model loading because
  Transformers 5 returns a dictionary from `apply_chat_template` by default.
  Explicit `return_dict=False` fixed the checker; frozen failed evidence is retained.
- The repaired actual-Transformers-5 CPU regression passed 950 tests with 4 skips.
  Its sources remained unchanged during verification. The fresh CPU protocol audit
  also passed. Evidence is in `/data/VPO-RM/.maintenance/llama-rl-20260918`.
- Suite `llama31-rl-canonical-20260918-v2`, job
  `llama-rl-grpo-de516854ce-37925285`, loaded the real 8B reward model on H200 and
  failed at `right padded raw reward differs`. The checker did not preserve the
  individual failing scores; numerical magnitude and cause are not yet established.
  Source inspection and actual-tokenizer CPU checks show identical final-EOT pooling,
  position IDs and causal masks for native/wrapper paths. BF16 projection shapes,
  gradient mode and padded/singleton SDPA dispatch still need real-weight comparison.
  No tolerance has been loosened and no production reward arithmetic has changed.

A separate one-GPU diagnostic is being prepared under the maintenance directory.
Remaining formal arms stay unsubmitted until the real GPU and training gates pass.

## Padding diagnosis and revised validation, 2026-09-18 02:07 HKT

The isolated H200 diagnostic completed 84 forward cases and 288 comparisons.
For identical BF16 inputs, native HF, the project wrapper and input-gradient
scoring agree exactly. The ASCII singleton score is -3.5625; the corresponding
right-padded batch score is -3.40625 in all three paths. The 0.15625 difference is
already present in native HF's BF16 backbone computation. A head-only FP32 cast
does not remove it. Evidence and original failed attempts remain under
`/data/VPO-RM/.maintenance/llama-rl-20260918/rm-padding-diagnostic-v2`.

The GPU gate now separates production BF16 same-input path equivalence from a
validation-only full-FP32 padding control. BF16 same-input tolerance stays 0.125;
the full-FP32 model must satisfy 0.001 absolute padding/singleton tolerance with
TF32 disabled. Both retain EOT, padding-gradient, mapped-gradient and frozen-weight
checks. The profile checks physical RM microbatch size 1 both after configuration
resolution and after actual trainer loading, before calibration or training.
Production reward math and all experiment hyperparameters are unchanged.

CPU regressions and independent review precede a fresh source freeze. The new
full-FP32 control has not yet been measured on the real GPU; its success must not
be inferred from the earlier BF16/head-only diagnostic. No formal RL updates or
lambda2/4/8 allocations have started.

## Real GPU protocol result, 2026-09-18 02:23 HKT

Suite `llama31-rl-canonical-20260918-v3` passed the new two-phase RM audit on H200.
BF16 same-input native/wrapper/no-gradient/input-gradient maximum difference is
exactly 0; its separately recorded cross-layout difference is 0.15625. The entire
FP32 RM, with TF32 disabled and SDPA retained, has a maximum required/padding
difference of 0.000031948089599609375, below 0.001. Both phases have zero padding
gradients, nonzero mapped-response gradients and unchanged frozen parameters.
The root independently validated the report against frozen source hashes and the
CPU artifact identities. Evidence: `gpu-protocol.json` in the v3 suite.

GRPO job `llama-rl-grpo-3e1b3385dd-41227307` is now running the shared calibration
and training preflight. The remaining three formal arms remain unsubmitted until
the full shared gate passes. Completed CPU regression: 972 passed, 4 skipped;
120 source files matched the frozen snapshot and remained unchanged during tests.

## Calibration and GRPO preflight, 2026-09-18 02:40 HKT

The v3 initial-policy calibration completed 128 distinct prompts and 1,024
responses, with 1,020 valid responses. All 128 groups qualified for scale
estimation. An independent recomputation of population standard deviations and
their median exactly reproduced `sigma0=6.166627762618061`; evidence is in
`audit/calibration-root-review.json`. Initial generated length averaged 278.09
tokens (median 198.5); four responses reached the 2,048-token cap.

The GRPO two-rollout preflight passed. Both optimizer updates changed the
trainable adapter while preserving the frozen reference tensor digest. Initial,
first-update, and second-update HF/vLLM probability checks passed their original
limits. Eight objective quality checks passed before and after the two updates.
The actual head is frozen FP32, SFT tokenizer IDs match returned vLLM prefixes,
and initial default/reference adapters both match all 448 SFT tensors.

The installed vLLM tokenizer loader logs unsuccessful AutoConfig probes for the
adapter-only tokenizer directory, then intentionally suppresses that missing
model-config exception and loads the explicitly requested SFT tokenizer. This
did not cause a base-tokenizer fallback or a failed generation; the actual
quality and token-ID/probability checks above passed.

The shared gate is still pending lambda2/4/8 preflights and lambda8 capacity.
The three remaining formal jobs are not submitted yet. An independent read-only
review found no blocking discrepancy between the frozen four-arm configurations,
commands, shared calibration design, or fixed stop/structural credit behavior.

## Observed mapping fallbacks and lambda2 preflight

Read-only CPU reconstruction with the actual TF5 tokenizer exactly reproduced
the saved GRPO and lambda2 mapping metrics. GRPO rollout2's 382 unmapped content
tokens all belonged to one response containing an incomplete UTF-8 prefix; the
other 63 responses mapped all content. Lambda2 rollout1's 1,705 unmapped content
tokens comprised one 1,683-token invalid-UTF-8 response plus 22 ordinary
resegmentation/trim tokens. Lambda2 rollout2 had just two unmapped BPE tokens.
These are conservative whole-answer byte-identity fallbacks or unchanged boundary
rules; canonical RM token IDs still match the native full-chat template. VPO
keeps weight 1 for unmapped positions and retains their sequence advantage.

Lambda2 passed two optimizer updates, unchanged-reference checks, and all three
probability probes. Credit mean/max were 1/2 in both updates. Objective quality
went from 8/8 to 7/8: the sole failed check answered `9 is larger than 3.` instead
of the requested number-only `9`. This satisfied the existing preflight threshold;
it is recorded as an instruction-format miss, not an arithmetic error. Shared
validation is continuing with lambda4 and lambda8; no threshold was changed.

## Full shared gate and formal submissions, 2026-09-18 02:58 HKT

All four two-rollout preflights passed, followed by two complete lambda8 capacity
updates with 64x2048 response tokens, 64x4095 actor inputs, and all canonical RM
inputs at 4095 tokens. The frozen reference stayed unchanged. Actor/RM PyTorch
allocated peaks were 24.05/31.46 GiB; these exclude the separate vLLM GPU.
The capacity-only importance-correction setting did not enter formal GRPO:
its actual profile has correction enabled, 250 rollouts, the original protected
SFT initialization, and the shared sigma0 of 6.166627762618061.

The shared gate SHA256 is
`69ad443c5c2a1ccd8309302f178ee668bfa797cab0383859c259b50a0054c9a8`.
Root revalidated frozen inputs, source, evidence hashes, calibration, and actual
capacity shapes/updates on the GPU pod before submitting the remaining arms.
An independent capacity review also found no blocker. See
`audit/shared-gate-root-review.json` and `all-submissions.json` in the v3 suite.

| Arm | Formal rjob | Scheduler state at 02:58 HKT |
|---|---|---|
| GRPO | `llama-rl-grpo-3e1b3385dd-41227307` | Running, formal stage |
| VPO lambda2 | `llama-rl-lam2-3e1b3385dd-65152256` | Inqueue / STARTING |
| VPO lambda4 | `llama-rl-lam4-3e1b3385dd-67138494` | Inqueue / STARTING |
| VPO lambda8 | `llama-rl-lam8-3e1b3385dd-69128330` | Inqueue / STARTING |

Each submission was recorded before invoking the scheduler and invoked once;
job registration was then independently queried. `/data` had approximately
100 GiB available after shared preflight. Formal initial-update validation and
all four final 250-rollout checkpoints remain pending. Submission alone is not
evidence of successful training or completion of the full experiment goal.

## Formal runtime checkpoint, 2026-09-18 03:10 HKT

All four scheduler jobs are now Running on H200 nodes: GRPO0487, lambda20484,
lambda40480, lambda80494. GRPO has passed its formal first-two-rollout startup
validation and completed nine rollouts. The other arms are still loading their
models; they do not yet have formal update evidence. Lambda8's earlier queue
delay was a scheduler resource shortage, and it has since been allocated.

A read-only monitoring process records fresh scheduler queries and persisted
metrics under the v3 suite's `audit/live-monitor/`. It performs no job or training
mutations. Its snapshots are observations, not substitutes for final checkpoints
and completion validation. The source is maintained separately at
`/data/VPO-RM/.maintenance/llama-rl-20260918/monitor_v3.py`; frozen training sources
remain unchanged. The monitor session is 72961 at this checkpoint.

## Monitoring correction, 2026-09-18 03:21 HKT

Local NFS reads of growing log files lagged the producer pods: the same lambda2
metrics inode appeared as 124 bytes locally while its producer read 7,318 bytes.
The profile log also showed different lengths. Forced attribute refresh did not
remove this discrepancy, so a local growing-file snapshot is not treated as
current progress evidence. This affected monitoring counts, not the running
training code or its producer-side startup checks.

The helper now reads `profile_metrics.jsonl` and status artifacts directly on
each producer pod, using IPs from fresh scheduler queries. It records observation
origin and its own source SHA256. Only the monitor helpers were stopped/restarted;
no training job or frozen source was changed. Earlier observations and helper
versions are retained. See `audit/live-monitor/monitor-correction.json`; current
monitor session is 97795.

The first corrected observation confirms GRPO19/lambda2-6/lambda4-5 formal
rollouts, all three startup validations passed, and no nonfinite metric rows.
Lambda8 is running but has not yet published a formal rollout. All three VPO
actual profile manifests were independently checked for the protected SFT,
shared sigma0, 250 rollouts, seeds, FP32 head, importance correction and lambda.

## All formal starts verified, 2026-09-18 03:25 HKT

All four producer pods now report passed formal first-two-rollout validations.
Root directly read their actual profiles, startup reports, runtime records and
artifact hashes on the producer pods, confirming the same SFT, shared sigma0,
250-rollout configuration, seeds, native protocol source hashes, FP32 head and
importance correction. The only arm differences are method/lambda, corresponding
freeze switches, and output paths. Evidence:
`audit/all-formal-startups-root-review.json`.

An independent CPU-only, single-thread comparison also verified all four actual
saved formal step-0 adapters against the protected SFT: each had 448 FP32 tensors
and 167,772,160 elements, with zero key, shape, dtype or value differences. It
loaded no base/RM and did not initialize CUDA. Evidence:
`audit/formal-step0-sft-comparison.json`. Original numeric stat values were not
retained; the report explicitly records only the observed unchanged-stat result.

The last observed formal rollout counts at 03:23:43 HKT were GRPO21/lambda2-7/
lambda4-6/lambda8-2, all Running. These are interim counts. The full experiment
goal still requires all four final 250-rollout checkpoints and completion/reference
integrity checks; none is claimed complete at this checkpoint.

## First ten formal rollouts, 2026-09-18 03:43 HKT

Producer-local snapshots confirm identical prompt batches across all four arms
for rollouts 1–10, with 640 responses and ten optimizer updates per arm. There
are no skipped updates, recorded empty/degenerate responses, or nonfinite
selected metrics. Root independently checked prompt hashes, response counts,
reward/length/cap arithmetic, update counts and loss/gradient finiteness against
the retained snapshots. See `audit/first10-health/README.md`, `summary.json` and
`root-arithmetic-verification.json` in the v3 suite.

| Arm | Mean response tokens | Capped at 2048 | Token-weighted entropy |
|---|---:|---:|---:|
| GRPO | 393.56 | 1.56% | 0.7145 |
| VPO lambda2 | 415.15 | 2.03% | 0.6652 |
| VPO lambda4 | 438.27 | 4.06% | 0.7321 |
| VPO lambda8 | 409.14 | 2.50% | 0.6122 |

Lambda4's higher cap rate and lambda8's lower pooled entropy are watchpoints;
ten updates over changing prompt batches do not establish collapse or a held-out
quality ranking. VPO mean weights remain within 1.2e-7 of one and observed
maxima equal the configured lambda. Exact minimum weights are not logged, so
histogram support is not presented as an exact minimum verification. All four
formal runs remain in progress; final checkpoint/reference checks are pending.

## Lambda8 repetition watchpoint, 2026-09-18 03:51 HKT

A bounded review inspected every capped lambda8 answer from formal rollouts
11–14: 24 of 256 responses (10/5/7/2 per rollout, 9.375%). These contain repeated
answer blocks, unsolicited Q&A continuations or unrelated enumeration/code,
rather than sustained useful long-form answers. Some begin with an appropriate
short answer before continuing. This is observed local generation degradation;
it does not establish global collapse from four selected batches.

Root compared exactly the same prompt batches on the other arms: GRPO had
3/256 capped responses, lambda2 5/256 and lambda4 11/256. All capped responses
across these four-arm batches have negative recorded advantages. Lambda8's 24
capped responses have mean raw RM reward -16.578125 and shaped reward
-28.911380767822266; each receives a 12.333255767822266 length penalty, and each
scores below its own prompt's uncapped median before and after that penalty.
These samples do not show repetition being rewarded above same-prompt peers.

Direct producer-side token checks found no native stop IDs in any of the 24
capped sequences. All other 232 answers end at native EOT128009, with no earlier
native stop. Actual sampling has stop IDs128001/128008/128009 and min_tokens0.
This evidence does not implicate ignoring an emitted stop token. The existing
`degenerate_responses` metric only flags whitespace and consecutive literal
newlines; its zero value must not be interpreted as absence of general repeated
text. No training code, reward rule, configuration or job was changed.

Evidence is retained under `audit/lam8-rollouts11-14-review/`: producer snapshots,
decoded capped responses, per-response review, `stop-metadata-root-review.json`
and `same-prompt-arms-root-review.json`. Further monitoring should examine whether
the cap/entropy trend persists in later batches while preserving the experiment.

The next nonoverlapping window, lambda8 rollouts15–20, contains 9/384 capped
responses (2.34375%), versus 24/256 (9.375%) in rollouts11–14. Token-weighted
entropy is 0.5233 versus 0.3722, respectively, and every capped response still
has a negative advantage. These windows use different prompts; this observation
does not establish recovery or a causal effect of the penalty. No additional
text review is claimed for rollouts15–20. Producer-local records and calculations
are retained in `audit/lam8-rollouts11-14-review/followup-through20.json`.

Lambda8 rollouts21–30 subsequently contain 2/640 capped responses (0.3125%),
mean length401.6297 tokens and token-weighted entropy0.5690. All ten optimizer
updates completed without skips; both capped responses have negative advantage.
These producer-local records are retained in
`audit/lam8-rollouts11-14-review/followup-21-30.json`. This later window does not
show the earlier cap spike continuing. Prompt changes and the lack of a new
text-level audit still preclude a general quality-recovery claim.

## First fifty formal rollouts, 2026-09-18 04:37 HKT

All four arms completed the same first50 prompt batches: 3,200 responses and
50 optimizer updates each, with no skips or recorded nonfinite selected metrics.
Actual RM input lengths stayed within the configured budget. Root independently
recomputed prompt equality, update counts, pooled reward/length/cap arithmetic
and token-weighted entropy from the retained producer-side snapshots.

| Arm | Raw RM mean, 1–50 | Mean tokens | Capped /3200 | KL at50 |
|---|---:|---:|---:|---:|
| GRPO | 5.8643 | 364.21 | 22 (0.69%) | 0.05615 |
| VPO lambda2 | 6.8208 | 388.74 | 25 (0.78%) | 0.07480 |
| VPO lambda4 | 6.8513 | 405.54 | 66 (2.06%) | 0.07817 |
| VPO lambda8 | 7.1619 | 433.09 | 64 (2.00%) | 0.12450 |

Rollouts41–50 have only5/4/3/5 capped responses out of640 per arm. Lambda8's
higher KL remains a watchpoint; these changing on-policy training batches do
not establish held-out quality or collapse. Empty/newline degeneration flags
are not general repetition detectors. The evidence and independent arithmetic
are in `audit/first50-health/README.md`, `summary.json` and
`root-arithmetic-verification.json`.

A separate producer-side retention check found only step0 and the latest LoRA
per arm, with no intermediate full checkpoints or failure artifacts; `/data`
had about90.4GiB free. See `audit/first50-health/retention-root-review.json`.
All four runs remain active; final250 checkpoints and reference-integrity checks
are still pending.

## First one hundred formal rollouts, 2026-09-18 05:43 HKT

Producer-side snapshots cover exactly rollouts1–100 for every arm. The 100 prompt
batches match across all four arms; each has 6,400 responses and 100 optimizer
updates, with no skipped rollouts or groups. Root independently checked the raw
JSONL hashes, all recorded profile/reward floats, response counts and lengths,
reward arithmetic, token-weighted entropy/KL and the logged VPO mean/max weights.
All 72 independently recomputed all100/last10 summary fields agree. Actual RM
input maxima are2610/2803/3524/3373, all within4096.

| Arm | Raw RM mean, 1–100 | Mean tokens | Capped /6400 | KL at100 |
|---|---:|---:|---:|---:|
| GRPO | 10.1113 | 409.86 | 33 (0.516%) | 0.10750 |
| VPO lambda2 | 11.2910 | 433.07 | 35 (0.547%) | 0.09457 |
| VPO lambda4 | 12.5757 | 447.53 | 86 (1.344%) | 0.15160 |
| VPO lambda8 | 12.9766 | 465.81 | 83 (1.297%) | 0.19470 |

In the same rollouts91–100, the arms have3/3/3/2 capped responses per640,
token-weighted entropy0.8067/0.9125/0.8483/0.8485, and mean response lengths
510.16/481.55/491.63/570.42. Lambda8's higher KL and length remain watchpoints;
these observations do not establish held-out quality or exclude general text
repetition. Exact minimum credit weights were not logged and are not claimed
verified. Evidence: `audit/first100-health/README.md`, `summary.json`, original
JSONL/source snapshots, and `root-arithmetic-verification.json`.

Lambda2 rollout99 has4,109 unmapped content tokens (19.2396%). A bounded CPU
review reproduced the exact cause: responses0/3 each reach2048 tokens and end
inside a UTF-8 character, triggering the existing whole-answer byte-mapping
fallback (4,096 content tokens); responses57/58 contribute2/11 tokens from local
BPE merges. All4,171 fallback positions, including62 native stops, have saved
weight exactly1 and direction exactly0. Root independently checked their actual
saved tensors and source/artifact hashes. The two capped answers retain negative
sequence advantages; these positions are not dropped from training.

All64 reconstructed canonical RM ID sequences match the native template, with
one BOS and terminal EOT; the maximum input2326 matches the recorded maximum.
Producer TF5.16.1/tokenizers0.23.1 CPU reconstruction agrees with the controller's
tokenizers0.23.2 results for every canonical-ID and mapping-position hash. The
training did not save RM-forward input-ID tensors, so this is deterministic
reconstruction from identity-checked inputs/code plus actual saved-credit/count
evidence, not a repeated RM forward. No new protocol or training implementation
error was found. The known cost is loss of fine-grained VPO modulation on19.24%
of this batch's content tokens; sequence-level training signal remains. Scope is
only this batch. Evidence: `audit/lam2-rollout99-mapping-review/README.md`,
`mapping-review.json`, `crosscheck-validation.json` and
`root-credit-verification.json`. Training configuration was not changed.

The producer-side retention snapshot at05:34 found only step0 and the current
adapter per arm, no intermediate full checkpoints and about90.08GiB free.
Evidence: `audit/retention-20260918-0533.json` (actual read time is recorded inside).
All runs remain active. Final acceptance still requires all250 rollouts, final
checkpoint artifacts, finite/changed trained weights, unchanged frozen reference
keys/shapes/dtypes/values, optimizer-state inspection and protected-SFT integrity.

## Final-verifier review, 2026-09-18 07:10 HKT

Review found a conditional reporting defect: a legitimate whole-rollout skip
does not run backward and omits `grad_norm`, but the frozen final verifier
requires that field unconditionally. The development launcher now permits the
missing gradient metric only for skips; updated rounds still require it, and
supplied gradient/loss metrics must be finite. The regression failed before the
fix;115 focused launcher/trainer/lifecycle tests then passed with3 warnings.
Independent review found no blocking issue. This changes final validation only,
not training math or any running frozen source.

Producer reads at07:10 showed215/187/177/178 completed rounds and zero skips or
missing gradient fields. All292 frozen source hashes still match the suite.
Evidence is in `audit/completion-validator-skip-fix.json`, its `.patch`, and
`audit/skip-policy-observation-20260918.json`. The remaining final artifact checks
and conditional skip handling are detailed in
`audit/final-verification-requirements.md`. If the frozen validator later rejects
a real skip, retain its failure record and independently audit the completed
training; do not rewrite history or rerun training merely for this reporting bug.
No final checkpoint has been accepted at this checkpoint.

## Reward curve snapshot, 2026-09-18 07:34 HKT

Producer-side snapshots cover GRPO234/lambda2-206/lambda4-198/lambda8-197
completed rollouts. All four retained prompt sequences match on common rounds
1–197; every captured rollout has64 responses. The figure presents raw RM score
and training reward after configured penalties on shared linear axes, showing
all per-round observations and trailing means over up to10 rounds. Runs start
from the protected SFT initialization. The unequal-progress tail after197 is
shaded and should not be treated as a matched prompt comparison.

Common rounds173–197 have raw RM means20.3803/22.7088/24.8269/24.5902 and shaped
reward means20.0487/22.4333/24.7336/24.4356, in GRPO/lambda2/4/8 order. These are
training objective scores, not held-out quality or wall-clock efficiency.

Figure, PDF, CSV, summary, collection/render scripts and hash-bound source
snapshots are in the formal suite's
`audit/reward-curves-20260918-0733/`; the directory name labels the collection
task, while each snapshot records its actual producer read timestamp.

## Training completed and independently verified, 2026-09-18 08:50 HKT

GRPO and VPO lambda2/4/8 all report scheduler Succeeded, exit code0, 250
rollouts, 250 optimizer updates and zero skips. Each retained its final
`train/checkpoint-250`, step0 and step250 adapters, logs and completion metadata.
Independent checks cover all16,000 responses per arm, intended prompt order,
token/stop/padding semantics, finite reward/profile values, reward arithmetic,
configuration/tokenizer identity and all292 unchanged frozen source hashes.

All448 trained FP32 adapter tensors changed in every arm; frozen references
remain exactly equal to the protected SFT. Final optimizer states have448
unique IDs at step250, finite moments, nonnegative squared moments and matching
shape/dtype counts. Adapter configs and config/RNG structures passed review;
input hashes remained unchanged across verification. Optimizer ID-to-parameter
name binding and resumable-training behavior were not independently proved.

Root's aggregate acceptance report is
`/data/VPO-RM/runs/llama31-rl-canonical-20260918-v3/audit/FINAL_TRAINING_REPORT.md`.
Machine-readable evidence is in `audit/final-root-review.json`, the four
`final-*-root-metadata.json` files and `final-*-tensors/` reports. Fresh terminal
scheduler evidence is `audit/final-scheduler.log` and
`audit/live-monitor/final.json`; earlier live snapshots remain historical.
These results establish completed training and artifact integrity, not
held-out benchmark quality. No frozen training source or protected SFT was
changed during final verification.
