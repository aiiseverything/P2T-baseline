# RED baseline (RLOO variant)

A standalone reproduction of **RED: Unleashing Token-Level Rewards from Holistic
Feedback via Reward Redistribution** (Jiahui Li et al., EMNLP 2025) as the fourth
arm of the VPO-RM project, next to GRPO, VPO-RM λ=4 and P2T. The RL algorithm is
**RLOO** (Ahmadian et al. 2024), the variant the paper reports alongside PPO.

The paper reads the reward model's scalar head at every position of the canonical
reward-model row and differences consecutive prefix scores. That turns one
sequence-level score into a per-token reward without retraining the reward model
and without a backward pass through it:

```
Eq. (6)   r~_t     = R_phi(x, y_<=t) - R_phi(x, y_<=t-1)
Eq. (7)   r^_t     = beta_c * r~_t + (1 - beta_c) * r_t        beta_c = 1
Eq. (8)   r^final  = r^_t - beta * r^KL_t
R4        A_{i,t}  = A_seq_i + alpha * (r^final_{i,t} - mean_t r^final_{i,t})
          A_seq_i  = RLOO's leave-one-out sequence advantage
```

Eq. (6)-(8) are the paper's. The advantage rule is not: the paper defines no
advantage function, so it is this arm's choice, stamped into every metric row and
manifest as `red_advantage_rule` and `red_alpha`. **R3** — subtracting RLOO's
sequence-level `b_i` off every token — was the first choice and is retired: on a
group whose rewards are all negative it made every token's advantage positive, the
update stopped being contrastive, and `runs/red250` collapsed and then crashed.
`RED_REPRO_NOTES.md` §2.2 has the measurements; §9 has the run record.

It is the natural sibling of the P2T arm — both are training-free reward
redistribution — but they differ in the shape of the extra work:

| | extra cost | token reward from | the sum telescopes to |
|---|---|---|---|
| **RED** | one reward-model **forward** | prefix-score differences | `R_phi(x,y) - R_phi(x, empty)` |
| **P2T** | one forward + one **backward** (input gradients) | `I_i = R(E) - R(E_{i←∅})` | not closed in the paper's own algebra |

RED's redistribution is potential-based reward shaping with `gamma = 1` and the
potential `Phi(s_t) = R_phi(x, y_<=t-1)`, which is why the paper can argue the
optimal policy is unchanged. That also means its benefit is an
estimation/optimisation effect, and the paper's own RLOO numbers are consequently
weak. See `RED_REPRO_NOTES.md` §2.6 before reading anything into this arm.

## What lives here

```
red/
  rm.py             the frozen reward model + `prefix_scores` (Eq. 6's primitive)
  reward.py         Eq. (6)-(8), the RLOO baseline, and the R4 advantage
  loss.py           REINFORCE with no ratio and no clipping, plus the KL metric
  mapping.py        byte-exact actor-token -> RM-position map (shared protocol)
  policy.py         actor log-probs, entropy, reference adapter handling
  length_reward.py  the project's soft length window, calibration, degeneracy guard
  data.py           UltraFeedback dedup, benchmark isolation, the fixed split
  vllm.py           generation protocol and the resident-server handle
  vllm_server.py    the generation process itself
  trainer.py        the loop, metrics, checkpoints
  autopush.py       periodic commit-and-push, best effort
  scripts/          run_red.py, start_red.sh, check_red_health.py, plot_red.py
configs/red250b.json   the run to launch; red250.json is kept as the failed run's record
tests/red/          unit tests, two autograd-free oracles, conformance checks
```

Nothing in this package imports `p2t` or `vpo_rm`. The correspondences that must
be exact — the canonical-chat token mapping, the length window, the degeneracy
flags and the reward-scale calibration — are enforced by
`tests/red/test_conformance.py`, which imports the parent project from the same
checkout and asserts agreement.

## Two departures from the project's protocol, and why

Both are forced by the paper and both are recorded rather than hidden:

1. **The KL lives inside the reward** (Eq. 8), so the trainer applies no separate
   KL loss term. The project's arms add one; doing both would double-count it.
2. **No clipping and no importance ratio.** RLOO's paper removes both. `clip_eps`
   is still read from the shared config, but only as the band in the startup gate
   that checks the trainer's re-forward reproduces the sampler's probabilities —
   never to clip the objective. The run manifest stamps `red_clipping`.

## Running

```bash
source .venv/bin/activate
python -m pytest tests/red -q                        # CPU, no model needed
bash red/scripts/start_red.sh red250b                # detached, survives a closed terminal
# monitor it, and stop it if it starts to collapse (see RED_REPRO_NOTES.md §9)
setsid nohup bash red/scripts/watch_red.sh red250b 30 stop-on-problem &
python red/scripts/check_red_health.py --report reports/red250b
```

Three cards are enough (actor, reward model, generation); four let generation use
tensor parallelism. See `RED_REPRO_NOTES.md` §7 for the device layout and the
memory budget, and change `vllm_gpus` / `vllm_tensor_parallel_size` together —
`tests/red/test_trainer.py` and the trainer's own startup check will reject a
mismatch.

## Evaluation

Evaluate this arm with the **`niuniu` four-metric workflow**, which is a separate
branch of the VPO-RM repository at `../niuniu-ref` (pinned revision `c8abb02`).
Its `docs/evaluation.md` is the authority; do not restate the protocol from
memory, and do not reach for the `runs/*-canonical-20260917` suites checked in
here — those are a *different* protocol, not an earlier version of this one.
**Why**, concretely: their Arena-Hard run uses 500 hard prompts only, an
`o3-mini-2025-01-31` reference and a GPT-4.1 judge, so three of that metric's four
inputs differ from the workflow's 750 questions / `gpt-4o-mini-2024-07-18`
reference / GPT-4o judge, and their repeated-seed counts differ as well.
`docs/evaluation.md` §6 warns against presenting new-protocol results as a
reproduction of those centres. No RED arm has ever been evaluated, so there is no
old-protocol RED number to confuse this with.

The four metrics and their reported fields: RM-Reward `mean` (raw Skywork scalar,
no length or KL penalty), AlpacaEval `weighted_win_rate_pct` (GPT-4.1 judge
against GPT-4 Turbo references; not length-controlled), IFEval
`project_four_metric_mean_pct` (541 prompts / 834 instructions, generation seeds
42–46), Arena-Hard `arena_hard_style_pct` (750 questions, GPT-4o judge).

```bash
cd ../niuniu-ref
source ../baseline/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1            # two free cards; the RM takes the second
export VLLM_USE_FLASHINFER_SAMPLER=0       # required on this box, see LOCAL-DEVIATION-L20.md
python scripts/run_benchmarks.py --config configs/evaluation/qwen3-14b-base.json \
  --stage generate --metric all \
  --model "$PWD/models/Qwen3-14B-Base" --tokenizer "$PWD/models/Qwen3-14B-Base" \
  --rm "$PWD/models/Skywork-Reward-V2-Qwen3-8B" \
  --tag red --adapter "$PWD/runs/red250b/checkpoint-250" \
  --output "$PWD/runs/eval-qwen3-14b-base"
# then --stage judge (paid; --budget-cny is per judge invocation), then --stage score
```

`--adapter` needs the final checkpoint, so this waits on `red250b` reaching step
250. **Record which rule the checkpoint ran under**: `run_manifest.json` in the
checkpoint directory stamps `red_advantage_rule` and `red_alpha`, and the credit
rule changed materially between `red250` (R3, retired) and `red250b` (R4) — a
number without that stamp does not say which estimator produced it. See
`RED_REPRO_NOTES.md` §2.2.

Arena-Hard's style control is fitted across the candidate set, so summarise the
arms together once they all have generations: `summarize_benchmarks.py --metric
arena_hard --output runs/eval-qwen3-14b-base --tags red base sft grpo vpo p2t`.
