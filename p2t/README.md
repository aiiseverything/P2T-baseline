# P2T baseline

A standalone reproduction of **Unlocking Token Rewards via Training-Free Reward
Attribution** (Sitong Wu et al., CVPR; "P2T") as the third arm of the VPO-RM
project, next to GRPO and VPO-RM λ=4.

The paper turns a coarse reward-model score into per-token rewards using one
forward and one backward pass of the reward model, then feeds the result to
standard GRPO. It is the natural control for VPO-RM: both methods read the same
reward-model input gradients, but

| | credit assignment | effect on the sequence advantage |
|---|---|---|
| **VPO-RM** | multiplicative: re-allocate the update budget across tokens | mean preserved exactly, `1/T Σ Ã_t = A` |
| **P2T** | additive: add an attribution-driven token reward | mean shifts by `alpha·R·(1 + ω/T)` |

## What lives here

```
p2t/
  attribution.py    Eq. (1)-(2): I_i = <grad_{e_i} R, e_i - e_null>
  reward.py         Eq. (3)-(5): R^P2T, the group advantage, the token advantage
  rm.py             frozen reward model: scoring and input gradients
  mapping.py        byte-exact actor-token -> RM-position map (Eq. 2 needs it)
  policy.py         actor log-probs, entropy, reference adapter handling
  loss.py           clipped GRPO surrogate and the KL term
  length_reward.py  the project's soft length window, calibration, degeneracy guard
  data.py           UltraFeedback dedup, benchmark isolation, the fixed split
  vllm.py           generation protocol and the resident-server handle
  vllm_server.py    the generation process itself
  trainer.py        the loop, metrics, checkpoints
  autopush.py       periodic commit-and-push, best effort
configs/            smoke10.json (10 rollouts, no SFT) and formal250.json
scripts/            run_train.py, start_detached.sh, status.sh, plot_reward.py
tests/p2t/          unit tests, an autograd oracle, and conformance checks
```

Nothing in this package imports or modifies `vpo_rm/`. The correspondence is
enforced by `tests/p2t/test_conformance.py`, which imports the parent project
from the same checkout and asserts that the shared pieces agree exactly.

## Running

```bash
bash setup_env.sh                                   # pinned cu129 runtime, once
source .venv/bin/activate
python -m pytest tests/p2t -q                       # CPU, no model needed

bash scripts/start_detached.sh smoke10              # survives a closed terminal
bash scripts/status.sh smoke10
python scripts/plot_reward.py --run reports/smoke10
```

`start_detached.sh` uses `setsid nohup` because `tmux` and `screen` are not
installed on the target host; the run has no controlling terminal and is not a
child of the shell that started it.

## Where things run

| device | holds |
|---|---|
| `cuda:0` | actor (bf16) + LoRA + optimizer |
| `cuda:1` | frozen reward model, scored one response at a time |
| `cuda:2,3` | vLLM, tensor parallel 2, `gpu_memory_utilization=0.85` |

P2T needs no policy logits to build credit, so the `[B, T, V]` response-logits
tensor that VPO-RM requires never exists here. That is most of why it fits on a
48 GB card with the physical microbatch left at one.

## Reading the curves

`p2t_share_ess_mean` and `p2t_flat_response_fraction` decide whether the method
is doing anything at all. Eq. (3) has no temperature, so if the attribution
softmax is flat the token term becomes a per-response constant and the arm
quietly degenerates into "GRPO with a shifted advantage". `p2t_bonus_over_advantage`
reports how large the token term is relative to the sequence advantage.

See [P2T_REPRO_NOTES.md](P2T_REPRO_NOTES.md) for the equation-to-code map, the
three places where the paper contradicts itself, and what was deliberately left
unfixed.
