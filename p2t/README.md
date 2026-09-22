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

`credit_ess_ratio` and `p2t_flat_response_fraction` decide whether the method
is doing anything at all. Eq. (3) has no temperature, so if the attribution
softmax is flat the token term becomes a per-response constant and the arm
quietly degenerates into "GRPO with a shifted advantage". `p2t_bonus_over_advantage`
reports how large the token term is relative to the sequence advantage.

See [P2T_REPRO_NOTES.md](P2T_REPRO_NOTES.md) for the equation-to-code map, the
three places where the paper contradicts itself, and what was deliberately left
unfixed.

## Evaluation

The four paper metrics for this arm are produced by the **`niuniu` evaluation
workflow**, which is a separate branch of the VPO-RM repository. It is *not* the
`runs/*-canonical-20260917` suites checked in here, and the two must not be mixed
— see "Why not the old suites" below.

- **Where**: branch `niuniu`, pinned revision `c8abb02` ("Support two-GPU credit
  ablations and publish four-metric evaluation workflow"), checked out at
  `../niuniu-ref`. Its `docs/evaluation.md` is the authority for the protocol;
  do not restate it from memory.
- **What it produces**, one `metric_summary.json` per metric:

  | metric | reported field | protocol |
  |---|---|---|
  | RM-Reward | `mean` | frozen 256 UltraFeedback prompts, seed 42, T=1, top_p=1, 2048 tokens; raw Skywork scalar, **no** length or KL penalty |
  | AlpacaEval | `weighted_win_rate_pct` | 805 prompts, GPT-4.1 judge against GPT-4 Turbo references; **not** official length-controlled |
  | IFEval | `project_four_metric_mean_pct` | 541 prompts / 834 instructions, generation seeds 42–46, official checker at seed 42 |
  | Arena-Hard | `arena_hard_style_pct` | 750 questions (500 hard + 250 creative), GPT-4o judge against the `gpt-4o-mini-2024-07-18` reference |

**Why not the old suites.** They are a different protocol, not an earlier version
of this one: their Arena-Hard run uses **500 hard prompts only**, an
`o3-mini-2025-01-31` reference and a **GPT-4.1** judge, i.e. three of the four
inputs differ from the workflow above, and their repeated-seed counts differ too.
Their centres therefore are not comparable with the numbers this workflow
produces, and `docs/evaluation.md` §6 warns against presenting new-protocol
results as a reproduction of them. This arm has **never** been evaluated, so it
has no old-protocol number to be confused with in the first place.

**Run it** (`../niuniu-ref/LOCAL-DEVIATION-L20.md` records the two environment
adaptations this host needs, including why `VLLM_USE_FLASHINFER_SAMPLER=0` is
required — without it the engine cannot start on this box's CUDA 12.2):

```bash
cd ../niuniu-ref
source ../baseline/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1            # two free cards; the RM takes the second
export VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/run_benchmarks.py --config configs/evaluation/qwen3-14b-base.json \
  --stage generate --metric all \
  --model "$PWD/models/Qwen3-14B-Base" --tokenizer "$PWD/models/Qwen3-14B-Base" \
  --rm "$PWD/models/Skywork-Reward-V2-Qwen3-8B" \
  --tag p2t --adapter "$PWD/runs/p2t250/checkpoint-250" \
  --output "$PWD/runs/eval-qwen3-14b-base"
# then --stage judge (paid; --budget-cny is per judge invocation, not a total),
# then --stage score, which is what writes the metric_summary.json files
```

The run records what it actually did: each metric writes a `manifest_*.json`
holding the sampling recipe, seed, dataset path and that dataset's SHA256, and
the output root holds the resolved `benchmark_config.json`. Check those rather
than trusting a label.

**Arena-Hard's style control is fitted across the candidate set**, so a single
model summarised alone is distorted. Once the sibling arms have generations too,
summarise them together:

```bash
python scripts/summarize_benchmarks.py --config configs/evaluation/qwen3-14b-base.json \
  --metric arena_hard --output runs/eval-qwen3-14b-base --tags p2t base sft grpo vpo
```
