# Project audit fixes implementation plan

> For agentic workers: use subagent-driven development with disjoint ownership and a final integrated review.

**Goal:** Fix all confirmed defects in the 2026-09-16 project audit, including the 16 primary findings and secondary correctness defects, with regression coverage.

**Architecture:** Keep the current trainer and evaluation entry points. Centralize tokenizer policy, benchmark exclusions, and evaluation provenance where several existing callers must agree. Preserve historical models/results; write corrected historical analyses to a separate directory.

**Tech stack:** Python, PyTorch, Transformers/PEFT, vLLM, pytest, Bash.

**Spec:** `runs/project-audit-20260916/REPORT.md` and `findings.json`; user explicitly authorized fixing all reported bugs.

## Constraints

- Preserve pre-existing native-EOS changes; snapshot at `runs/project-fixes-20260916/before/`.
- Work in the current shared checkout with disjoint file ownership; no broad unrelated refactor, no old result overwrite.
- Reuse `/root/miniconda3/envs/sml/bin/python`; CPU tests first; do not install heavy login-node dependencies.
- Before each implementation, add and run a regression that demonstrates the audited failure. Retain failing-run evidence and verify the fixed behavior.
- Every A01–A16 finding must have a concrete implementation and validation entry; secondary confirmed defects are included below. Design choices must be documented rather than falsely presented as proven bugs.

## 1. Token policy and credit allocation (math worker)

Own `vpo_rm/core.py`, new `vpo_rm/token_policy.py`, `scripts/analyze_rm_artifacts.py`, `tests/test_core.py`, new token-policy tests.

Interfaces: `get_stop_token_ids(tokenizer) -> tuple[int,...]`, `get_structural_token_ids(tokenizer) -> tuple[int,...]`. Add optional `stop_token_ids` / `structural_token_ids` parameters to `allocate` / `compute_credit` (default stop IDs preserve direct Qwen callers, but production callers pass tokenizer-derived values; structural freezing must not silently use unverified IDs).

- [x] Regression: actual local tokenizer maps known bad IDs to content; allocate with legal group advantage and 2048 tokens must satisfy both lambda bounds.
- [x] Solve allocation directly over unfrozen positions with frozen weights=1, adaptively bracket a feasible temperature, then bisect and validate final weights. Handle all frozen, all equal, zero advantage and lambda=1 padding.
- [x] Avoid overflow in clipped positive-advantage ratio; finite or fail-fast behavior for truly unrepresentable gradients.
- [x] Use the same token policy in RM artifact analysis, recording decoded categories.

## 2. Trainer and generation (trainer worker)

Own `vpo_rm/trainer.py`, `vpo_rm/integration.py`, `scripts/train_skywork.py`, `scripts/profile_vllm_full.py`, `scripts/vllm_generate_server.py`, `scripts/vllm_generate_once.py`, trainer/integration regression tests.

Consume token-policy helper above and the benchmark helper from task 3. Stop/finish metadata must flow from both samplers to reward guard. Defaults must anchor KL to a real initial reference; support new LoRA with base as reference and SFT-initialized LoRA with frozen ref adapter. Implement policy epochs and optimizer minibatches, rather than silently ignoring them.

- [x] Regression: mixed EOS/im_end and padding, unsupported sampled ID, beta effective after drift, requested update counts, length-cap EOS vs truncation, seed-before-LoRA, small-file split metadata.
- [x] Apply identical output support to generation/log probabilities; fail before optimizer.step on nonfinite values. Preserve valid stopped lengths and finish metadata.
- [x] Make configured updates real, with correct partial minibatch and microbatch weighting, frozen caches and reference.
- [x] Fix KL diagnostic sign, factory seeding, small custom split isolation; no hidden train=valid fallback.
- [x] Match/report sampling constraints in probability calculations or use an explicitly supported on-policy protocol; capture presence/minimum/temperature settings in manifests. Bring old one-shot sampler into agreement.

## 3. SFT and benchmark isolation (data worker)

Own `scripts/sft_init.py`, `scripts/sft_response_tokens.py`, `scripts/clean_sft_data.py`, new `vpo_rm/data.py`, SFT/data regression tests. Do not edit trainer.py; coordinate helper API.

Interface: `exclude_benchmark_prompts(prompts, benchmark_paths=None) -> (filtered_prompts, metadata)` using NFC/whitespace normalization; default local benchmark files, explicit missing-file failure for formal training. Preserve stable split selection for offline historical evaluation by opting training callers into exclusions.

- [x] Regression: 1250 microbatches/8 produces 157 updates including properly normalized tail; final metrics/monitor run; fractional epochs handled explicitly.
- [x] Correct accumulation using a documented global supervised-token mean over each effective batch; include weighted EOS counts when enabled. Correct throughput and epoch shuffling.
- [x] Remove exact benchmark overlaps before new SFT/RL training split, record counts/keys/hash. Keep historical dataset/artifact files untouched.
- [x] Require numeric confidence annotations in cleaning; preserve normal prose/labels and handle formatting accurately.

## 4. Evaluation, launchers and historical corrections (root)

Own all eval/judge scripts, new evaluation provenance helper, shell launchers, configs/docs and their regression tests.

- [x] Regression: changed recipe/model/data/adapter cannot hit old cache; incomplete cache rejected; n>1 outputs all persist with sample_idx.
- [x] Record complete generation/scoring config and file fingerprints; atomically write outputs and validate sample identities/counts. Unique LoRA IDs per engine across runs. Derive stopping IDs from tokenizer.
- [x] GSM8K numeric exact scoring, including large integers without float aliasing; regenerate corrected metrics from saved results to a new directory.
- [x] Judge bounded submission, budget checks before further dispatch, dynamic usage dates and strict errors; checkpoint per-row annotations for resumability, preserve existing summaries and validate all candidates. No paid judging for tests.
- [x] Pass model/RM throughout train-eval chain; support semicolon multiple runs; forward CALIB_INIT; make script failures propagate; correct default native EOS paths/modes without silently using incompatible 8B adapters.
- [x] Fix plot temperature labels, document unsupported resume explicitly and ensure source/sampling provenance includes behavior-defining files.
- [x] Annotate historical affected runs and produce reproducible corrected analyses, keeping originals intact. No automatic full retraining of all historic experiments.

## 5. Completion gates

- [x] Run focused regression suites and full pytest; validate each new regression exercises the real fixed behavior.
- [x] Parse all Python, bash -n all shell; review final diff for regressions and lost pre-existing changes.
- [x] Run realistic small-model integration (CPU and/or short rjob where necessary) for native/vLLM boundaries; do not claim GPU coverage from mocks.
- [x] Cross-review independent components; inspect each A01–A16 evidence entry and all secondary confirmed defects.
- [x] Write `runs/project-fixes-20260916/REPORT.md` with per-finding fix/test evidence and limits. Mark goal complete only when remaining required fixes/checks are zero.

## Final evidence

151 tests passed, no skips; 51 Python files parse and 32 shell files pass bash -n; git diff --check passes. Actual tiny-Qwen SFT runs validate both EOS-weighted paths and tail accumulation. H200 job `vpo-fixes-exact-0916-65479500` succeeded with actual GPU support-mask checks and 14B native-LoRA 2x2 sampling (all four native-EOS stops). Final details and historical limitations: `runs/project-fixes-20260916/REPORT.md` and `final-verification.json`.
