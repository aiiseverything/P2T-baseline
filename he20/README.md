# he20 baseline — high-entropy minority tokens

A reproduction of **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive
Effective Reinforcement Learning for LLM Reasoning** (Shenzhi Wang, Le Yu, Chang
Gao, Chujie Zheng et al., Qwen Team, Alibaba + LeapLab, Tsinghua; NeurIPS 2025,
arXiv 2506.01939) as an arm of the VPO-RM project, next to GRPO, VPO-RM λ and P2T.

The paper's finding is that only a small minority of chain-of-thought tokens carry
high entropy — the "forking" decision points — and that they drive nearly all of
what RLVR achieves.  Its method is correspondingly small: **leave the algorithm
alone and compute the policy-gradient loss only over the highest-entropy tokens of
a batch.**

```
Eq. (1)  H_t             = -sum_j p_t,j log p_t,j        p_t = softmax(z_t / T)
Eq. (6)  A^final_{i,t}   = I[H_t^i >= tau_rho^B] * min(r_t^i A_t^i, clip(r_t^i, 1-eps, 1+eps) A_t^i)
         normaliser      = sum_i sum_t I[H_t^i >= tau_rho^B]      (not sum_i |o_i|)
         tau_rho^B       = the entropy threshold that keeps the top rho fraction
```

Two things in Eq. (6) matter and both are implemented: the indicator multiplies the
surrogate, **and** the token-count normaliser is restricted to the same tokens — a
masked mean over the kept tokens rather than a reweighted mean over all of them.
`rho = 20%`.

## What is and is not reproduced

The **method** is reproduced as written, including the parts the authors' own code
does differently (the paper defines the mask by a threshold, their reference
implementation sorts; `he20/mask.py` follows the paper and exposes the other rule).

The **base algorithm is this project's**, not the paper's.  The paper's base is
DAPO; here it is the project's plain GRPO, so that the mask is the only difference
from the arms this one is compared against.  The paper's setting — math RLVR with a
verifiable answer, 20 480-token responses, DAPO-Math-17K — is likewise not
reproduced: this arm runs UltraFeedback with the frozen Skywork reward model and a
2048-token cap.  Its reported gains are DAPO-with-mask against DAPO; this arm's are
mask against *this project's* GRPO, and `HE20_REPRO_NOTES.md` §5 records the
difference rather than blurring it.

## Comparability with VPO and P2T

This arm exists as a comparison point, so its comparability is enforced by tests
rather than asserted.  `tests/he20/test_alignment.py` checks the chain end to end:
VPO's parameters are the parent's `corrected_rl_launcher.common_config`, P2T's
config is that protocol plus its credit rule, and this arm's config is P2T's plus
the mask.  Every training parameter the parent protocol defines must match, the
init adapter must be the canonical SFT checkpoint **by weight hash** (the parent
calls it `models/sft-native-eos-clean2k5e2`, this checkout has the same bytes at
`models/sft-p2t`), and the keys that deliberately differ are named with their
reason.  The trainer is a port of `p2t/trainer.py` — 18 of its 26 shared functions
are AST-identical to P2T's, and the differences are the mask and the arm's own
seams, nothing else.

One convention worth knowing when reading the metrics: `group_sigma_*` here is the
**floored** group scale (`max(std, advantage_std_floor_fraction * sigma0)`), which
is what VPO and P2T report and what this arm standardises by.  RED reports the
unfloored spread.  The unfloored value is reported beside it as
`raw_group_sigma_*` so the "every group has near-zero spread" watchdog keeps
something to fire on.

## What lives here

```
he20/
  mask.py           Eq. (6): the threshold and the selection population
  loss.py           the project's GRPO surrogate plus the optional mask
  reward.py         the credit: group advantage broadcast, no token-level term
  rm.py             the frozen reward model, pooled score only
  policy.py         actor log-probs, entropy, reference adapter handling
  mapping.py        byte-exact actor-token -> RM-position map (shared protocol)
  length_reward.py  the project's soft length window and the degeneracy floor
  rollout.py        rollout validation and selection (resampled groups)
  data.py           UltraFeedback prompts: dedup, isolation, the fixed split
  tokens.py         tokenizer-derived token sets
  tensors.py        small tensor helpers
  policy_precision.py  FP32 policy output projection
  vllm.py           generation protocol and the resident-server handle
  vllm_server.py    the generation process itself
  trainer.py        the loop, the mask wiring, metrics, checkpoints
  autopush.py       periodic commit-and-push, best effort
  scripts/          run_he20.py, start_he20.sh, watch_he20.sh,
                    check_he20_health.py, plot_he20.py
  HE20_REPRO_NOTES.md
configs/he20250.json    the run to launch; he20-smoke2.json is its two-step copy
tests/he20/             the mask's semantics, the loss, the alignment contract
```

Nothing in this package imports `p2t`, `red` or `vpo_rm` at runtime; the
conformance tests import them deliberately and skip if they are absent.

## One design decision the paper leaves open

The paper says the selection happens "within each batch" and also "within each
(micro-)batch", which its own configuration puts 16× apart.  This arm selects over
the **optimizer minibatch** — the responses whose gradients are averaged into one
`optimizer.step()` — and the trainer **refuses any configuration that would take
more than one optimizer step per rollout** while the mask is on.  The reason is
fidelity rather than caution: the entropy must be `pi_theta`'s (Eq. (1)), the mask
is built once from the old-log-prob pass's buffer, and with more than one step that
buffer would be stale for every step but the first.  Ranking it anyway would be a
silent approximation of Eq. (6).  `HE20_REPRO_NOTES.md` §3 has the full argument.

## Running

```bash
source .venv/bin/activate
python -m pytest tests/he20 -q                       # CPU, no model needed
HE20_GPUS=0,1,2,3 bash he20/scripts/start_he20.sh he20250
# monitor it, and stop it if the mask stops doing what it says
setsid nohup bash he20/scripts/watch_he20.sh he20250 30 stop-on-problem &
python he20/scripts/check_he20_health.py --report reports/he20250
```

Four cards: actor on 0, reward model on 1, vLLM tensor-parallel across 2 and 3.
`HE20_GPUS` must name them in that order or the trainer's device-plan check will
reject the launch.  See `HE20_REPRO_NOTES.md` §7 for the shared protocol and §8 for
the review record.

## Evaluation

Evaluate this arm with the **`niuniu` four-metric workflow** at
`../niuniu-ref` (branch `niuniu`, revision `c8abb02`), whose `docs/evaluation.md`
is the authority.  The `runs/*-canonical-20260917` suites checked in here are a
*different* protocol, not an earlier version of it — their Arena-Hard uses 500 hard
prompts, an `o3-mini-2025-01-31` reference and a GPT-4.1 judge, so three of that
metric's four inputs differ from the workflow's.  Same command as the sibling arms,
with `--tag he20` and this arm's final checkpoint:

```bash
cd ../niuniu-ref && source ../baseline/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/run_benchmarks.py --config configs/evaluation/qwen3-14b-base.json \
  --stage generate --metric all \
  --model "$PWD/models/Qwen3-14B-Base" --tokenizer "$PWD/models/Qwen3-14B-Base" \
  --rm "$PWD/models/Skywork-Reward-V2-Qwen3-8B" \
  --tag he20 --adapter "$PWD/runs/he20250/checkpoint-250" \
  --output "$PWD/runs/eval-qwen3-14b-base"
# then --stage judge (paid), then --stage score, which writes metric_summary.json
```
