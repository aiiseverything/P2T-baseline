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
R3        A_{i,t}  = r^final_{i,t} - b_i        b_i = leave-one-out mean of R_j
```

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
  reward.py         Eq. (6)-(8), the RLOO baseline, and the R3 advantage
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
configs/red250.json
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
bash red/scripts/start_red.sh red250                 # detached, survives a closed terminal
python red/scripts/check_red_health.py --report reports/red250
```

Three cards are enough (actor, reward model, generation); four let generation use
tensor parallelism. See `RED_REPRO_NOTES.md` §7 for the device layout and the
memory budget, and change `vllm_gpus` / `vllm_tensor_parallel_size` together —
`tests/red/test_trainer.py` and the trainer's own startup check will reject a
mismatch.
