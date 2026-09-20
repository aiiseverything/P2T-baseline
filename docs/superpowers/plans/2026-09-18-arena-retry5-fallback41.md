# Arena failed-judgment retries and GPT-4.1 fallback

> Implementation follows the existing isolated Arena worktree and tests first.

**Goal:** Execute the user's authorized retry/fallback policy for exactly the590
final failed games whose original GPT-4o attempt counter is0, without regenerating
candidate/reference responses or modifying the previous completed evaluation.

**Design:** Retain each original attempt as attempt1. Make up to4 additional
identical GPT-4o requests, stopping at the first valid verdict. Only after five
completed invalid GPT-4o judgments, issue one GPT-4.1 fallback with the identical
request except its `model` field. Save immutable per-attempt records in a new
campaign and score both the recovered pure-GPT-4o variant and the explicitly
labeled mixed-judge variant using the unchanged official aggregation math.

**Authorization:** The user's current instruction explicitly authorizes this
execution. No additional approval step is necessary. Existing five-attempt
failure `lam8/707e6bd0d8994e71-0` is outside the590 selected targets and is retained.

**Workspace:** `/data/VPO-RM/arena-exclusions-code`, an existing isolated git
worktree. Preserve its preexisting edits. No commits, pushes or model training.
Original suite is the shared-storage `runs/arena-hard-v2-gpt4o-judge-20260917`.
New campaign is `/data/VPO-RM/runs/arena-hard-gpt4o-retry5-gpt41-20260918`.

## Constraints and interfaces

- Unit: model tag, question UID and answer order. A/B order reversal is not a
  retry. Targets are the exact590 original failure records with attempt0, whose
  file hashes must match the old exclusion manifest; successful partners and
  the other5,410 source games remain untouched.
- All original source identities, prompts, reference/candidate text, parser,
  temperature0 and max_tokens16000 remain bound and unchanged. GPT-4.1 changes
  only request.model. First-valid selection never depends on who wins.
- Valid means completed stop response, accepted verdict, valid usage and intact
  identity. Missing verdict, truncation and completed content_filter are invalid
  outputs. Ambiguous transport, malformed records or inflight attempts block
  dispatch and are never silently counted as a completed invalid judgment.
- At most32 concurrent targets, serial attempts within each target. Durable
  inflight records precede every paid request. One controller lock, no repeated
  dispatch on restart; completed attempt files cannot be overwritten.
- Reuse the existing relay and17891 proxy. Keep the prior account-wide billing
  baseline and144 CNY cumulative dispatch guard; save updated billing in the new
  campaign only. Bounded free billing retries are distinct from judge retries.
  Proven pre-request connection establishment failures may use the existing
  audited transport retry; ambiguous request delivery blocks for inspection.
- Runner module: `scripts/arena_retry_fallback.py`. Preparation freezes target
  hashes and policy. `load_resolution(campaign, require_complete=True)` returns
  `(manifest, resolutions)` after validating all saved attempt chains. Each
  resolution contains `tag`, `uid`, `order`, `resolution` (`gpt4o`, `gpt41`,
  `failed`), `selected_record` (or null) and bound attempt evidence. It never
  creates a relay, accesses credentials or makes network requests.
- Scorer module: `scripts/score_arena_retry_fallback.py`, calling this read-only
  interface and reusing `score_arena_with_exclusions` pure scoring functions.
  Both variants retain only complete valid pairs and report each model's own
  denominator plus the six-model common subset. Every chosen verdict retains
  its actual judge provenance. No mixed result is called pure-GPT-4o or official
  full500 leaderboard output.

## Tasks

- [ ] Runner: implement immutable preparation, exact selection, serial5-attempt
  policy, one fallback, first-valid/resume, durable request state, bounded32-way
  execution, billing and completion reports. Write failing tests for early valid
  stop, fifth-attempt success, fallback gating/model-only mutation, still-invalid
  fallback, content_filter, no duplicate resume and corrupt/ambiguous blocking.
- [ ] Scorer: test and implement pure-GPT-4o and mixed-judge overlays without
  changing old records. Test unchanged partners, pair exclusions, fallback
  provenance, target/request mismatches and first-valid selection. Reuse pinned
  decisive/tie weighting and bootstrap/style formulas.
- [ ] Root: review both changes, run focused regression and read-only full590
  preparation, freeze execution source and verify hashes, check proxy/models and
  current free billing state. Start the authorized controller with32 workers;
  verify real request progress and no preexisting record changes.
- [ ] Follow live execution through retry/fallback and both score outputs, verify
  actual counts/hashes and publish a clear summary. Preserve all failures and
  any transport/billing interruption. Never label mere launch as completion.

The ongoing four Llama training jobs continue independently; their original
250-rollout/final-checkpoint goal remains active and unchanged.
