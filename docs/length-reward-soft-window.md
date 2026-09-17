# Soft response-length reward window

The `soft` mode keeps generation free to stop naturally while discouraging two observed failures: very short outputs and long padding that consumes the 2,048-token generation limit. The existing `legacy` mode remains the default for old commands and run reproduction. The corrected GRPO and VPO λ=2/4/8 runs now use this configuration; see the [current experiment status](project-status-2026-09-17.md) for their protocols and evaluation results. Those results also include reward-input and sampling corrections, so they do not isolate the effect of the length reward alone.

Let `L` be the number of **generated response tokens** in the response mask. It includes a generated EOS token when present, and excludes prompt tokens. Let `M` be `max_response_tokens`, `σ₀` the initial calibration scale, and `clip(x,0,1)` clamp to `[0,1]`. The soft reward deductions are

```text
P_short = short_penalty_strength × σ₀ × clip((short_response_threshold − L) / short_response_threshold, 0, 1)
P_long  = long_penalty_strength  × σ₀ × clip((L − long_response_threshold) / (M − long_response_threshold), 0, 1)
reward_shaped = reward_RM − P_short − P_long
```

Defaults are `short_response_threshold=8`, `long_response_threshold=1024`, `short_penalty_strength=0.5`, `long_penalty_strength=2.0`, and `M=2048`. The degenerate-output guard also examines runs of at least 32 consecutive literal newline characters in the decoded response. Group advantages use the shaped and guarded rewards, with denominator `max(population_std(group_rewards), advantage_std_floor_fraction × σ₀)`; the floor fraction defaults to `0.5`. This prevents a nearly constant reward group from amplifying tiny differences. The response-length terms are reward deductions, not token masks or a forced stopping point.

If a prompt group has **no normally completed, nondegenerate response**—including a mix of truncations and degenerate outputs—the trainer resamples that group once. If the retry is also unusable, it skips the group without an optimizer update for those responses. In a group with at least one valid completion, a genuinely truncated response receives only its soft length cost; truncation alone does not trigger the hard degenerate floor. A response that stops at the cap receives the same length cost as one that reaches the cap with a `length` finish reason: the cost depends on `L`, not on the finish-reason label.

In `soft` mode, `min_response_tokens` resolves to **0**. Explicit nonzero values are rejected so EOS is available from the first generation step. In `legacy` mode the omitted minimum resolves to **8**, preserving old behavior. The legacy `--length-penalty-slope` and `--length-penalty-anchor` retain their old meaning only in `legacy`; `soft` rejects a positive legacy slope. The long threshold must be below `max_response_tokens`.

When `--length-reward-sigma0` is omitted in `soft` mode, the trainer samples exactly `--length-calibration-prompts 128` training prompts using the configured seed, generating `group_size` responses for each at the initial actor policy (normally the specified SFT LoRA). It computes population reward standard deviations for groups with at least two completed, nondegenerate responses, then takes their median; an invalid or nonpositive result fails rather than silently choosing a scale. Calibration uses the same rollout backend as training and consumes its random stream; the trainer does not reset that stream afterward. The automatic result is recorded in the output directory's `length_reward_calibration.json`. Confirm the reward-model chat format and generation settings before treating that scale as comparable across arms. To share an already established scale across GRPO/VPO arms, pass the same positive `--length-reward-sigma0` to both; this skips sampling.

Example three-GPU profile command, run inside the GPU image with its vLLM-enabled Python and a fresh output directory:

```bash
python scripts/profile_vllm_full.py \
  --model models/Qwen3-14B-Base \
  --rm models/Skywork-Reward-V2-Qwen3-8B \
  --init-adapter models/sft-native-eos-clean2k5e2 \
  --dataset-path datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet \
  --output-dir runs/my-soft-vpo \
  --method vpo_rm --length-reward-mode soft \
  --max-response-tokens 2048 --long-response-threshold 1024
```

Use a different fresh output directory and `--method grpo` for the comparison arm; pass the first arm's recorded `σ₀` as `--length-reward-sigma0` to both arms when a shared, fixed scale is required. The `scripts/train_skywork.py` and `python -m vpo_rm.trainer` entrances accept the same length arguments and `--init-adapter`. The profile records all generation requests for each rollout, because calibration and retries can make a single last-request summary misleading. Final adopted prompt/token artifacts are written by the trainer after selection.

The local Qwen3-tokenized data motivates a soft start at 1,024: 239/59,057 (0.405%) of the actual UltraFeedback training chosen answers would exceed 1,024 when counting a normal EOS, whereas 11/805 (1.37%) GPT-4 Turbo AlpacaEval reference bodies exceed 1,024 and two exceed 1,536. These observed answers are evidence about coverage, **not** a quality-optimal length target. Longer requests such as a play act or a detailed tutorial remain possible until the 2,048-token cap. See the CPU-only [UF/SFT analysis](../runs/rl-native-eos-20260916-143153/diagnosis/length-design-data.md) and [Alpaca reference analysis](../runs/rl-native-eos-20260916-143153/diagnosis/alpaca-reference-lengths.json).
