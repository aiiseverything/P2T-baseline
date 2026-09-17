# Corrected GRPO final-checkpoint AlpacaEval

Training source: /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/rl-fp32-is-canonical-20260917
Checkpoint: /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/rl-fp32-is-canonical-20260917/grpo/train/checkpoint-250
Protocol: 805 shared prompts, one answer each at temperature 1, maximum 2048 tokens, FP32 output head; internal GPT-4.1 pairwise judge against GPT-4 Turbo references. This is an internal raw/weighted win rate, not official length-controlled AlpacaEval 2 leaderboard.

Only GRPO is requested in this evaluation suite. Generation and paid judging statuses are tracked in status.json. experiment.json binds inputs and frozen source hashes.

Completed 2026-09-17 11:18 HKT. Generation job `alpaca-can-grpo-0917-75998537` succeeded. All 805 generated rows use the corrected GRPO checkpoint-250 and FP32 output head; every response ended normally, mean 560.44 response tokens, 0/805 at the 2048-token limit. The generation manifest hash and checkpoint identity passed validation before judging.

The internal GPT-4.1 judge returned 805 valid judgments and zero parse failures: raw win rate **301/805 = 37.3913%**, weighted win rate **38.1575%**. Independent recomputation from the 805 saved preferences exactly matches `generations/grpo/results_judged.json`, whose output hash matches its cache manifest. Judge cost was ¥1.036. The official AlpacaEval 2 length-controlled leaderboard score was not computed.

Historical comparison: the superseded native-EOS GRPO suite had 0.1252% internal weighted win rate and the protected SFT had 7.0014% in older evaluation artifacts. Training and judge implementation changed between runs, so these historical figures do not isolate the effect of any single fix. This evaluation suite contains no lambda=2/4/8 jobs or scores.
