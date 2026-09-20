# Qwen direct formal RL — 2026-09-18

The user explicitly requested immediate formal submission without further smoke tests. The previous coordinator (PID 2772172) was stopped and its Qwen smoke job `direct-qwen-grpo-c7529cfe-42775631` was stopped before submitting these replacements. No GPU preflight pass is claimed.

Both jobs were submitted at 16:24 Hong Kong time and confirmed Starting at 16:25:11:

- GRPO: `direct-qwen-grpo-72dbc535-85786297`, 3 GPUs.
- VPO lambda4: `direct-qwen-lam4-72dbc535-86881747`, 3 GPUs.

The direct entry uses the unchanged frozen v3 training implementation, with no pilot, shared-gate dependency, probability preflight, or repeated calibration. Both arms use the previously completed and independently verified fresh-Qwen calibration, sigma0=1.2118192354369524, SHA256 bcb1efbf99d2883381133079467a5f93d828316c43e5e70e4b020abeeea5a459.

Actor: Qwen/Qwen3-14B (official posttrained model), fresh LoRA, empty SFT adapter, seed42, generation seed0. All training/reward/sampling settings match the frozen manifest. Each arm runs250 rollouts and retains only checkpoint250. Trainer-internal finite-value checks and final artifact validation remain enabled. Outputs are in `/data/VPO-RM/runs/direct-rl-qwen-instruct-formal-20260918/qwen`.

Only the entry orchestration changed. The two existing Llama jobs remain running. All four requested experiments now have independent three-GPU submissions.
