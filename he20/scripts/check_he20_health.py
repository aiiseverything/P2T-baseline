"""Health check for an he20 run, reading only ``metrics.jsonl``.

    python he20/scripts/check_he20_health.py --report reports/he20250 [--window 10]
    python he20/scripts/check_he20_health.py --report reports/he20250 \
        --stop-on-problem --pidfile runs/he20250/train.pid

Written for this arm rather than reusing a sibling's checker because the failure
modes differ.  The RED checker's trip-wires are calibrated to that arm's
credit-assignment collapse -- the redistribution ceasing to flip any update's
direction, every token pushed the same way, the bonus term detaching from the
sequence advantage -- and none of them can happen here: he20 has no token-level
credit to collapse, its advantage is the project's plain group-relative scalar
broadcast onto every valid token (``he20/reward.py``).  What can happen here is
that Eq. (6)'s **mask** stops being the mask: the run keeps a fraction of the
population unrelated to ``entropy_top_ratio``, keeps a set that is not the
high-entropy one, or loses the threshold it claims to have selected at.

Exit code 1 iff a problem was found, so a watcher can branch on it.  The problems
below are all hard: a mask that is not doing what its own row says it is doing, a
row that does not carry this arm's protocol stamp at all, and the shared
sameness checks every arm is held to (metrics the sibling arms also report, finite
values, sampler/trainer agreement, and the collapse signatures -- falling length
and entropy, a policy that has drifted far from its initialization).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
from pathlib import Path

# The protocol stamp and the mask's vocabulary, mirrored from ``he20/reward.py``
# and ``he20/mask.py`` rather than imported: this script is invoked standalone
# (``python he20/scripts/check_he20_health.py``), which puts ``he20/scripts`` on
# sys.path and not the repository root, and it is kept dependency-free for the same
# reason the paper extraction script is.
# ``tests/he20/test_check_he20_health.py`` asserts the mirror, so drift between the
# two fails the suite instead of silently mis-certifying a run.
HE20_PROTOCOL = "entropy_top_mask_eq6"
ENTROPY_TOP_RULES = ("threshold", "topk")
# Eq. (6)'s threshold is a *batch* statistic; the trainer's single-step guard makes
# its optimizer minibatch the whole rollout batch.  The same ratio taken over one
# response would be a different method, so the population is stamped and checked
# rather than assumed.
MASK_POPULATION = "optimizer_minibatch"

# Keys that must be finite on every row, or the run is not trustworthy.  These are
# the names every arm reports -- the he20 trainer mirrors its siblings' vocabulary,
# so a row that stops carrying one has drifted away from the comparison.
FINITE = ("loss", "grad_norm", "raw_reward_mean", "response_entropy", "kl_to_init",
          "mean_response_tokens", "advantage_abs_mean", "group_sigma_mean",
          "credit_ess_ratio")
# The mask's own numbers.  Checked only on a row that carries this arm's protocol
# stamp, so another arm's artifact draws the one clear message below instead of a
# page of missing-key noise.
HE20_FINITE = ("entropy_top_kept_fraction", "entropy_top_mean_kept_entropy",
               "entropy_top_mean_all_entropy", "he20_unmapped_share_mean")
# Keys whose absence means the row was not written by this arm's trainer at all.
# ``entropy_top_threshold`` is on the list although it is None whenever the mask is
# off: the key has to be present either way, because "no threshold" and "a threshold
# nobody recorded" are different statements and only the first is the unmasked path.
HE20_REQUIRED = ("he20_protocol", "he20_mask_population", "entropy_top_ratio",
                 "entropy_top_rule", "entropy_top_kept_fraction",
                 "entropy_top_threshold", "entropy_top_mean_kept_entropy",
                 "entropy_top_mean_all_entropy")
# Keys the sibling arms also report; their absence means the arms' metric
# vocabularies have drifted apart and cross-arm plots would silently break.
SHARED = ("raw_reward_mean", "mean_response_tokens", "response_entropy", "kl_to_init",
          "group_sigma_mean", "credit_ess_ratio", "reward_count", "optimizer_steps",
          "truncated_responses")
# The trainer stamps ``he20_mask_ratio_is_effective`` into the checkpoint manifest;
# the metric rows it writes do not carry the field.  This checker reads
# ``metrics.jsonl`` and nothing else, so it derives the statement instead --
# ``TrainerConfig.mask_enabled`` is exactly "ratio is not None and ratio < 1" -- and
# checks the stamped field only when a row happens to carry it.
MASK_EFFECTIVE_FIELD = "he20_mask_ratio_is_effective"

# Eq. (6)'s threshold rule keeps every token tied at tau, so a tie mass can only
# push the realised kept fraction *above* rho; the only downward deviation either
# rule can produce is rounding to a whole token, because top-k keeps ceil(rho*n) and
# the interpolated quantile lands above the rho*n-th order statistic.  The band is
# asymmetric for that reason: a shortfall beyond one token of the population means
# the mask is not the configured one, while an overshoot is a failure only when it
# is gross enough that the "high-entropy minority" is a majority.
TOKEN_SLACK = 1
TIE_ALLOWANCE = 3.0


def rate(rows, field, window):
    """Half-vs-half mean shift over the last ``window`` rows.

    A window's own scatter is the bar a shift has to clear: comparing the first
    half against the second and requiring the gap to exceed the window's standard
    deviation tests the trend against noise instead of against a fixed slope.
    """
    values = [row.get(field) for row in rows[-window:] if isinstance(row.get(field), (int, float))]
    if len(values) < max(2, window // 2):
        return None
    half = len(values) // 2
    early = sum(values[:half]) / half
    late = sum(values[half:]) / (len(values) - half)
    spread = (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5
    return (late - early) if abs(late - early) > spread else 0.0


def number(value):
    """True for a real finite number; ``bool`` is not one here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stop_run(pidfile, problems):
    """Ask the training process to stop, after a problem it cannot recover from.

    Only ever reached through ``--stop-on-problem``.  Killing a run is
    destructive enough that the launcher has to ask for it explicitly, and the
    reason is printed so the health log records why a run was stopped next to the
    run itself.  This stops the process; it does not remove anything, so the most
    recent checkpoint stays on disk and the failure stays inspectable.
    """
    path = Path(pidfile)
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        print(f"  stop-on-problem: no usable pid file at {path}; leaving the run alone")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"  stop-on-problem: pid {pid} is already gone")
    except PermissionError:
        print(f"  stop-on-problem: not permitted to signal pid {pid}; leaving the run alone")
    else:
        print(f"  stop-on-problem: sent SIGTERM to pid {pid} for: {problems[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="he20 run health check")
    parser.add_argument("--report", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stop-on-problem", action="store_true",
                        help="SIGTERM the run named by --pidfile when a problem is found")
    parser.add_argument("--pidfile",
                        help="the run's train.pid, required with --stop-on-problem")
    args = parser.parse_args()
    if args.stop_on_problem and not args.pidfile:
        parser.error("--stop-on-problem needs --pidfile so it cannot signal the wrong process")

    path = Path(args.report) / "metrics.jsonl"
    if not path.is_file():
        print(f"PROBLEM: no metrics at {path}")
        return 1
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print("PROBLEM: metrics.jsonl has a malformed line")
            return 1
        if "rollout" in row:
            rows.append(row)
    if not rows:
        print("PROBLEM: no completed rollouts yet")
        return 1

    problems, notes = [], []
    last = rows[-1]

    if last.get("optimizer_steps", 0) == 0:
        problems.append("the last rollout took no optimizer step")
    for key in SHARED:
        if key not in last:
            problems.append(f"shared metric {key} is missing; cross-arm comparison would break")
    for key in FINITE:
        value = last.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"{key} is not a finite number ({value!r})")

    # Sampler-vs-trainer agreement: judged on the typical token, because the tail
    # of that difference is bf16 numerics across two kernels by nature.
    if last.get("rollout_logp_abs_error_p50", 0) > 0.05:
        problems.append(f"systematic sampler/trainer log-prob disagreement "
                        f"(p50 {last['rollout_logp_abs_error_p50']:.4f})")
    if last.get("rollout_is_ess_ratio", 1) < 0.9:
        problems.append(f"importance weights are degenerate "
                        f"(ESS ratio {last['rollout_is_ess_ratio']:.4f})")
    if last.get("initial_hf_logp_max_abs_error", 0) > 1e-4:
        problems.append("the trainer's first re-forward disagrees with the sampler")

    # he20-specific: everything below reads the mask's own vocabulary, so it is
    # gated on the protocol stamp.  A metrics row from another arm (or from an he20
    # build predating the stamp) must draw exactly one message rather than being
    # read for keys it was never going to carry -- and must never be certified,
    # because the stamp is what says which method produced the numbers.
    ratio = last.get("entropy_top_ratio")
    kept = last.get("entropy_top_kept_fraction")
    threshold = last.get("entropy_top_threshold")
    kept_entropy = last.get("entropy_top_mean_kept_entropy")
    all_entropy = last.get("entropy_top_mean_all_entropy")
    # The trainer's mask_enabled, mirrored: null and 1.0 both mean "no mask", and at
    # rho = 1 the paper's own method reduces to the unmasked baseline it is measured
    # against -- so the disabled path is a real configuration, not a broken one.
    mask_on = number(ratio) and ratio < 1.0
    # One token of the population is the widest the realised fraction can fall short
    # by, so the allowance is converted from the row's own population size -- the
    # same denominator kept_fraction uses.  Without it the allowance falls back to a
    # fixed 0.1%, far below any real failure (a mask keeping half the tokens is off
    # by 0.3 at rho = 0.2).
    population = last.get("response_tokens")
    slack = TOKEN_SLACK / population if number(population) and population > 0 else 1e-3

    he20_row = "he20_protocol" in last
    if not he20_row:
        problems.append(
            f"no {HE20_PROTOCOL!r} protocol stamp on the last row: this metrics row was "
            f"not written by an he20 trainer (another arm's report, or a build predating "
            f"the stamp), so the mask diagnostics below cannot be read from it and it "
            f"must not be certified as an he20 run")
    else:
        for key in HE20_REQUIRED:
            if key not in last:
                problems.append(f"he20 metric {key} is missing; the row does not carry this "
                                f"arm's vocabulary")
        for key in HE20_FINITE:
            value = last.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{key} is not a finite number ({value!r})")
        if last.get("he20_protocol") != HE20_PROTOCOL:
            problems.append(f"unexpected protocol {last.get('he20_protocol')!r}; the mask is "
                            f"only certified for {HE20_PROTOCOL!r}")
        if last.get("he20_mask_population") != MASK_POPULATION:
            problems.append(f"unexpected mask population {last.get('he20_mask_population')!r}; "
                            f"Eq. (6)'s threshold is a batch statistic, and the same ratio "
                            f"taken over anything narrower is a different method")
        if "entropy_top_rule" in last and last.get("entropy_top_rule") not in ENTROPY_TOP_RULES:
            problems.append(f"unknown mask rule {last.get('entropy_top_rule')!r}; this checker "
                            f"knows {ENTROPY_TOP_RULES}")
        stamped = last.get(MASK_EFFECTIVE_FIELD)
        if stamped is not None and bool(stamped) != mask_on:
            problems.append(f"{MASK_EFFECTIVE_FIELD} is {stamped!r} while "
                            f"entropy_top_ratio={ratio!r} says the mask is "
                            f"{'on' if mask_on else 'off'}")

        if mask_on:
            # The threshold the selection was actually taken at.  Non-finite (or
            # absent) while the mask ranks tokens means the mask is not the one this
            # row says it is.
            if not number(threshold):
                problems.append(f"entropy_top_threshold is {threshold!r} while the mask is "
                                f"enabled (entropy_top_ratio={ratio!r}); the selection has no "
                                f"finite threshold")
            if number(kept):
                if kept < ratio - slack:
                    problems.append(
                        f"the mask kept {kept:.4f} of the population where entropy_top_ratio "
                        f"is {ratio}: no rule keeps fewer than rho of the tokens (top-k rounds "
                        f"up to ceil(rho*n) and the interpolated quantile lands above the "
                        f"rho*n-th order statistic), so the mask is not the configured one")
                elif kept > min(1.0, ratio * TIE_ALLOWANCE + slack):
                    problems.append(
                        f"the mask kept {kept:.4f} of the population against entropy_top_ratio "
                        f"{ratio}: ties at tau can only lift the realised fraction by the size "
                        f"of the tie mass, and more than {TIE_ALLOWANCE:g}x rho is past that -- "
                        f"the high-entropy minority has become a majority")
            # What says the selection is the *high-entropy* one rather than a set of
            # the right size.  Only meaningful while the mask is on: with rho = 1 or
            # null every token is "kept" and the two means coincide by construction.
            if number(kept_entropy) and number(all_entropy) and kept_entropy <= all_entropy:
                problems.append(
                    f"the mask kept tokens no more entropic than the population (kept "
                    f"{kept_entropy:.6f} <= all {all_entropy:.6f}): it is not selecting the "
                    f"high-entropy minority Eq. (6) is defined on")
        else:
            # rho = null or 1.0 is the paper's own unmasked baseline, so a row on
            # this path has to look like one: every token kept, and no threshold.
            if number(kept) and kept < 1.0 - slack:
                problems.append(f"entropy_top_ratio={ratio!r} says every token is kept, but "
                                f"{kept:.4f} of the population was")
            if number(threshold):
                problems.append(f"entropy_top_threshold is {threshold!r} while the mask is off "
                                f"(entropy_top_ratio={ratio!r}); a row reporting a threshold "
                                f"reads as a masked run")
        # The trainer's own startup gate: a content-mapping failure above this bound
        # aborts rollout 0, so a later row past it is the same failure arriving after
        # the gate has stopped looking.
        unmapped = last.get("rm_unmapped_content_fraction")
        if number(unmapped) and unmapped > 0.25:
            problems.append(f"{unmapped:.3f} of the response's content tokens map to no "
                            f"reward-model position, above the 0.25 the trainer's startup "
                            f"gate bounds (max_unmapped_content_fraction); the sequence "
                            f"score the advantage is built on came from text the reward "
                            f"model never saw")

    # The two thresholds here are this arm's own, NOT the sibling checker's.  This
    # arm's ``rate`` returns the raw half-vs-half *shift* (line 105) where the shared
    # checker it was copied from returns a per-rollout *rate*
    # (``scripts/check_run_health.py:47``: ``shift / (len(values) - half)``).  At
    # window=10 the same literal means 5x more here, and the string both checkers
    # print says "tokens/rollout" for a quantity that is not one.  Inheriting the
    # literals -20 / -0.02 therefore made this arm's two trend wires fire on 22 and
    # 24 of p2t250's prefixes -- every one of them a false positive on the arm that
    # finished all 250 rollouts at raw reward +11.0 (its healthy maxima were a
    # length shift of -156.5 and an entropy shift of -0.217) -- and they are what
    # SIGTERM'd he20250 at rollout 12 on a shift that cleared its window's own spread
    # of 83.3 by 0.9 tokens while kl_to_init was 0.0256 and response_entropy was
    # 1.018, above p2t250's 0.742.  The calibrated pair sits just past every value
    # that healthy run produced and still fires on red250's genuine collapse (-238.7
    # length, -0.564 entropy).  Local edit, kept visible and uncommitted, the same
    # discipline as niuniu-ref/LOCAL-DEVIATION-L20.md.
    length = rate(rows, "mean_response_tokens", args.window)
    if length is not None and length < -180:
        problems.append(f"response length falling {length:.0f} tokens/rollout")
    entropy = rate(rows, "response_entropy", args.window)
    if entropy is not None and entropy < -0.30:
        problems.append(f"entropy falling {entropy:.4f} nats/rollout (collapse?)")
    if number(last.get("kl_to_init")) and last["kl_to_init"] > 5:
        problems.append(f"KL to init is {last['kl_to_init']:.2f}; the policy has drifted far")
    count = last.get("reward_count", 0)
    if count and last.get("truncated_responses", 0) > 0.5 * count:
        problems.append(f"{last['truncated_responses']}/{count} responses were truncated")
    # The *unfloored* spread.  This arm reports the floored scale under the shared
    # `group_sigma_*` name (so the key means the same thing as its siblings', which
    # apply the `advantage_std_floor_fraction * sigma0` floor too), and a floored
    # value can never reach zero -- testing that name here would make this check
    # unfirable.  Rows from before the unfloored pair existed fall back to the old
    # name, which is what the sibling arms still report.
    spread = last.get("raw_group_sigma_mean", last.get("group_sigma_mean"))
    if number(spread) and spread <= 1e-6:
        problems.append("every prompt group has near-zero reward spread")

    # Notes, not problems, and only on a row that is this arm's: both of them read
    # he20's vocabulary, so on another arm's artifact they would describe the wrong
    # run.  An unmasked run is a legitimate configuration -- it is the baseline the
    # paper measures its own method against -- but it is not Eq. (6), and a reader
    # comparing artifacts should not have to work that out from a kept fraction of 1.
    if he20_row:
        if not mask_on and (ratio is None or number(ratio)):
            notes.append(f"entropy_top_ratio={ratio!r}: the mask is off, so this run is the "
                         f"unmasked baseline, not Eq. (6) itself")
        elif number(kept) and kept > ratio * 1.5:
            notes.append(f"the realised kept fraction {kept:.4f} sits well above rho {ratio}; "
                         f"ties at tau have pulled the selection wider than the ratio")
        # The arm's weight is uniform by construction, which makes the shared ESS
        # exactly 1.  Anything else means the credit is no longer the plain broadcast
        # advantage.
        if number(last.get("credit_ess_ratio")) and abs(last["credit_ess_ratio"] - 1) > 1e-3:
            notes.append(f"credit_ess_ratio is {last['credit_ess_ratio']:.6f} where this arm's "
                         f"uniform weight makes it exactly 1 (he20/reward.py)")

    print(f"rollout {last.get('rollout')}: raw_reward {last.get('raw_reward_mean')}, "
          f"loss {last.get('loss')}, grad_norm {last.get('grad_norm')}, "
          f"tokens {last.get('mean_response_tokens')}, "
          f"mask kept {kept} of rho {ratio} (threshold {threshold}), "
          f"entropy kept {kept_entropy} vs all {all_entropy}, "
          f"protocol {last.get('he20_protocol')}")
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if not problems:
        print("  ok")
    if problems and args.stop_on_problem:
        stop_run(args.pidfile, problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
