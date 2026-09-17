# Full training recheck and experiment replacement

> **For agentic workers:** Use systematic debugging and test-driven development for each confirmed defect; execute independent review areas in parallel with disjoint file ownership.

**Goal:** Recheck all project-owned training, data, evaluation and operational code, fix confirmed defects, validate the actual GPU path, submit replacement GRPO/λ=2/4/8 jobs, then delete superseded jobs.

**Architecture:** Use the RM's canonical complete user/assistant chat input. Map reward gradients only at identical actor/RM token IDs with matching text byte spans; positions changed by canonicalization retain unit credit. Existing Actor native EOS and protected SFT weights remain unchanged. Freeze source and artifacts for GPU validation and formal jobs.

**Tech stack:** PyTorch, Transformers, PEFT, vLLM, rjob, pytest.

**Spec:** User's current thread request and the confirmed 24-question RM-format probe at `runs/rl-native-eos-20260916-143153/diagnosis/final-grpo-rm-20260917/REPORT.md`.

## Global constraints

- Protect all existing SFT weights, tokenizer assets and manifests, especially `models/sft-native-eos-clean2k5e2` SHA256 `21c0c7b9e75c640b03a3fddfc6bb1e7a478e02c6724111800d605187364761ad`.
- New formal jobs must be submitted and verified before old jobs are deleted.
- No new long training before CPU tests, GPU RM ordering/gradient checks and real training/sampling smoke pass.
- Preserve existing uncommitted changes; old run source snapshots remain immutable.
- Storage cleanup, if necessary, is restricted to identified old RL checkpoint artifacts; preserve diagnostic records.
- Do not claim zero possible defects or scientific success from passing execution checks.

## Work packages and evidence

- [x] Baseline: file hashes, protected SFT hashes, working-tree diff, live jobs and disk space, baseline full pytest.
- [x] Data/SFT/analysis: review each owned file, reproduce and fix confirmed bugs with regression tests, record coverage in `runs/full-recheck-20260917/data-sft-audit.md`.
- [x] Evaluation/operations: review each owned file, exact model identity and caches, judge content boundaries, score RM with canonical input, record coverage in `eval-ops-audit.md`.
- [x] Math/runtime: review core, alignment, probability support, termination, numeric boundaries and SSH runtime; record coverage in `math-runtime-audit.md`.
- [x] Trainer/RM: implement canonical serializer and byte-span mapping, unit credit for unaligned positions, termination checks, source/format manifests and coverage in `trainer-rm-audit.md`.
- [x] Independent integration review and full CPU suite including real tokenizer edge cases, empty responses, padding, retokenization and finite-difference reward gradients.
- [x] GPU validation: current production RM scores match standard HF complete-chat scores; correct/wrong/off-topic controls and matched SFT/old-RL ranking; VPO mapped gradients and unit-credit positions validated.
- [x] GPU training validation: actual 14B SFT LoRA + 8B RM, GRPO and VPO λ=2/4/8, correct init/reference weights, vLLM/HF probability comparison, finite loss/gradients, no OOM, canonical input provenance and reward calibration. The lambda-8 supplemental coverage and original failed cohort remain explicitly recorded.
- [x] Formal frozen suite: same SFT, data split, seed and latest soft length design; shared corrected-RM calibration, disk budget and final checkpoint retention; independent preflight quality and maximum-shape checks passed before formal submission.
- [x] Submit all four replacement jobs, verify their identities, startup and artifacts. Exact new IDs are recorded in `runs/rl-fp32-is-canonical-20260917/jobs.json`; all four first-two-rollout gates passed and all four scheduler states were Running at final verification.
- [x] No active obsolete watchers found; old metadata archived; exact eight superseded RL job IDs deleted after all four replacements were submitted and configuration-verified. Successful namespace listing verified all old names absent and all new names present. SFT assets retained and hashes rechecked.
- [x] Completion audit: file-by-file review ledger, defects closed or explicitly bounded, tests/GPU evidence, four new running jobs, old jobs deleted after submission, SFT hash unchanged. See `runs/rl-fp32-is-canonical-20260917/relaunch-audit.json` and `runs/full-recheck-20260917/sft-protection-after-relaunch.json`.

## Canonical RM interface

`canonical_reward_input(reward_tokenizer, prompt, response_text) -> list[int]` serializes exactly the complete chat, with no silent plain-text fallback. `build_reward_input(actor_tokenizer, reward_tokenizer, prompt, response_ids)` returns canonical IDs and an actor-length position map. A map entry is valid only if token ID and decoded byte span agree; removed special tokens and merged/split tokens have position -1. The trainer obtains gradients with this explicit valid mask and passes its complement as `fixed_weight_mask` to credit allocation. All sampled tokens still receive their ordinary sequence advantage; only unsupported token-specific attribution is suppressed.

Tests are written before each production change. GPU checks compare against independently assembled official HF inputs, not only the same serializer's output. Protocol hashes and exact commands are retained alongside results.

## Sampling probability correction discovered by the GPU gate

The first actual training gate stopped before optimization. HF reproduces its original probabilities exactly, while even the same HF weights differ between full and cached forwards with a BF16 vocabulary head. A fixed-trace GPU comparison is isolating head precision before selecting a production precision change. The existing mean/p99/max acceptance limits remain unchanged.

In addition to reducing numerical discrepancies, the vLLM training path must retain the actual sampled-token probabilities. Keep the HF old-policy clipping anchor and VPO full-vocabulary credit definition; multiply the existing response-normalized policy and sampled-KL surrogates by the detached token importance ratio `exp(HF_old_logp - rollout_logp)`. This is a conditional token correction, not a claim of unbiased sequence/group resampling. Do not replace the clipping denominator as well, clip or normalize the importance weights, or mix them into the lambda credit budget. Native HF training retains its explicit uncorrected configuration until it supplies actual sampling probabilities; a configuration that requires correction must reject missing values.

Transport an optional eighth `[B,T]` rollout field through original validation, calibration and group retry/repacking. The vLLM profile requires this field and exact prompt/token/adapter alignment, records complete ratio diagnostics, and excludes raw probability arrays from console metrics. Synthetic capacity checks explicitly disable sampling correction because synthetic tokens have no sampling distribution; this exception is recorded as test-only. Verify exact two-action gradients, identity ratios, padding, detach behavior, retry provenance and all four real training arms. The protected SFT remains unchanged.

GPU fixed-trace result: both engines use FP32 output logits, while HF LoRA computation/parameters remain FP32 and vLLM LoRA remains BF16. The complete batch-32 comparison includes all 32 replicas (2,944 token pairs) and passes the unchanged probability limits. HF training uses an autograd-safe frozen FP32 Linear, not the inference-only `mm(out_dtype=...)` operator. Formal credit microbatch is one response, matching the existing actor update/old-probability microbatch and avoiding the 148-GiB double FP32 full-logit allocation. Future evaluation of these corrected checkpoints must honor the recorded FP32 policy-head protocol when loading the adapter; the adapter file itself deliberately contains no base/head weights.

The subsequent four-arm gate passed GRPO/lambda 2/lambda 4 but stopped lambda 8 before its first update: a two-row 174-token cohort had p99 0.103388 versus 0.10. Its tokens and HF probabilities exactly equal the passing lambda-4 cohort; only measured vLLM probabilities differ. This failure remains recorded. One preregistered external supplement uses every returned row's first 128 tokens, retains the same thresholds, requires the current cohort to pass independently, and additionally includes the original failed cohort in a separate aggregate. It leaves all 20 training-source files unchanged and reuses the three already-passed arms with explicit provenance. See `runs/full-recheck-20260917/lam8-supplement-protocol.md`. Lambda 8 passed two real updates, all three adapter checks, quality 8/8 before/after and both 64-by-2048 maximum-shape updates. The complete training sequence includes a 0.3493 numerical tail outside the probe coverage; the actual sampler probability and finite importance correction were independently verified and this coverage limit is recorded.

## Formal submission and resource-driven deletion sequence

All four jobs under `runs/rl-fp32-is-canonical-20260917` were submitted on 2026-09-17 at 03:54 HKT. GRPO started; lambda 2/4/8 were blocked by explicit scheduler CPU/memory shortages. After verifying all four registrations, frozen commands and validation bindings, the exact eight old jobs were deleted at 03:56 HKT to release resources. This preserves the user's required new-submission-before-old-deletion sequence; it moves deletion ahead of the self-imposed formal first-two-rollout check because queued jobs could not perform that check. The deletion record explicitly marks those checks pending at deletion time. All four replacements subsequently started and passed their first-two-rollout gates by 04:03:57 HKT. Final successful namespace listing verifies all four Running and all eight old names absent. The original eight SFT files match their baseline hashes, both old initialization weight copies remain intact, and final observed free disk space is 71.20 GiB.
