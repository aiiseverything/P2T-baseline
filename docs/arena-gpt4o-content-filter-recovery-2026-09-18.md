# GPT-4o Arena-Hard content-filter continuation

Prepared 2026-09-18 (Hong Kong time). No paid requests or background processes were started during preparation.

The original runner stopped after the paid response for `lam4/52c9cea50d8a4236/1` completed with `finish_reason=content_filter`, `status=invalid`, explicit `answer=null` and `score=null`, valid usage (2495 prompt + 1196 completion = 3691 total tokens), and a completion timestamp. This is a completed unusable judge response, rather than an ambiguous network request.

Policy `arena_judge_output_exclusions_v2` admits exactly that shape as `judge_content_filter`. Missing fields, non-null answers/scores, null answers under stop/length, unknown finish reasons, bad usage, inflight requests, and ambiguous transport remain blocked. Both answer orders for the affected model/question are excluded from scores. The surviving opposite-order verdict is not independently retained. Existing accepted format/truncation exclusions and all scoring math remain unchanged.

The original v1 policy and source stay frozen. New source and metadata live at:

`/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/arena-hard-v2-gpt4o-judge-20260917/continuation-with-exclusions-v2`

Only four production files differ from v1: the policy branch/version; the result protocol version; and optional validated `--continuation-dir` support for the billing wrapper and score watcher. The underlying paid-request runner, frozen judge request, relay, reference answers, model answers, and strict scorer are unchanged. Existing cumulative billing baseline and 144 CNY dispatch guard continue with 32 workers and no new model rollouts. Paid requests are not retried. Three bounded read-only billing attempts remain unchanged.

Verification:

- Three new content-filter tests first failed against v1; the two new continuation-directory tests first failed before implementation.
- 200 policy, runner, scorer, billing, watcher, legacy-judge and legacy-scoring tests passed in 30.53 seconds.
- Read-only production preflight classified 4161 valid, 507 judge-failed, 0 blocked, 1332 missing games. Exactly one record changed classification from v1, with no record edits.
- All 4668 existing game records, 18 archived attempts, and 111 original frozen files retained their SHA256 values.
- Saved cumulative billing was 36.060022 CNY. This is the last saved billing reading, not a fresh account query; the normal dispatch guard rechecks before each batch.

Review `changes-from-v1.patch`, `offline_preflight.json`, `preexisting_records.json`, both source manifests, and the frozen source. Run `verify_prepared.py` again immediately before launch. It makes no network calls, validates the production records with the original frozen validator, and verifies old and new source hashes. The prepared shell runner also repeats this check before any paid dispatch.

The durable exit marker, judge/watcher launch records, watcher state, score directory and completion report belong to v2. The existing suite-level `exclusions_progress.json` and `exclusions_judging_complete.json` continue to describe the active exclusion-policy run, including its v2 policy ID. Old v1 files, including its stopped controller, are preserved. Do not restart the v1 runner/watcher.

After independent root review, launch the prepared judge shell with a detached `subprocess.Popen`, save its PID and the SHA256 of `judging_source_manifest.json` to v2 `judge_launch.json`, then launch the frozen v2 watcher with `--suite <suite> --continuation-dir <suite>/continuation-with-exclusions-v2`. Both processes must use `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=<v2>/source:<suite>/source`. The shell already configures port 17891 proxy, 32 workers and the unchanged 144 CNY cumulative guard. `launch_after_review.py` in the prepared directory encodes these commands, is not automatically invoked, and refuses an existing launch or exit marker. Preparation did not execute it.
