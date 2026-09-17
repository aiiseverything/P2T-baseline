# Reward on 256 held-out UltraFeedback prompts

Six independent one-H200 rjobs: Base, native-EOS SFT initialization, and final
corrected GRPO / VPO lambda 2, 4, 8 (`checkpoint-250`).

All models use the exact same 256 prompts, one response per prompt,
temperature 1, top_p 1, unrestricted top_k, seed 42, minimum length 0,
maximum 2048 generated tokens, 4096 context, and float32 actor output head.
Both native stop IDs are active. Skywork-Reward-V2-Qwen3-8B scores complete
canonical user/assistant conversations using its BF16 scalar head.
The reported reward is the raw RM score, without length penalties or KL.

`data_model_audit.json` records prompt split/overlap and model checks.
`experiment.json` freezes the common protocol, inputs, source and runtime.
`job/<model>/` contains submitted commands, runtime, logs and status.
`results/<model>/` contains all 256 rewards, response token IDs and summary.

The summary's `at_token_cap_rate` counts responses whose stored token length
reaches 2048, including any response that ends naturally exactly at the cap;
it is not a finish-reason-based truncation statistic. Bootstrap intervals
describe variation across prompts for this one generation seed.

Existing weights are read in place; compilation and dataset caches live in
each job's local `/tmp`. The protected SFT checkpoint is never modified.

## Completed results

All six rjobs succeeded by 2026-09-17 17:19:53 Asia/Hong_Kong.
All 1,536 scores and their token arrays passed coverage and integrity checks.

| Model | Mean raw reward | Bootstrap 95% CI | Mean tokens | At 2048 tokens |
|---|---:|---|---:|---:|
| base | -2.536978 | [-3.610601, -1.450800] | 622.21 | 17/256 |
| sft-init | 3.748572 | [3.043757, 4.485385] | 253.48 | 0/256 |
| grpo | 10.870316 | [9.962479, 11.816185] | 545.90 | 1/256 |
| lam2 | 12.885454 | [11.863375, 13.945114] | 570.84 | 0/256 |
| lam4 | 13.353210 | [12.249264, 14.445801] | 507.02 | 0/256 |
| lam8 | 12.910884 | [11.873471, 13.889579] | 591.29 | 0/256 |

See `summary.csv`, `summary.json`, and `completion_verification.json`.
The scorer is also the reward model used for RL training, so this is held-out
reward performance, not an independent judge-based quality metric.
