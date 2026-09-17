# Six-model reward evaluation on 256 held-out prompts

User request: launch six independent rjobs for Base, protected native-EOS SFT,
and corrected GRPO / VPO lambda 2, 4, 8 final checkpoints.

Protocol: the existing frozen first 256 validation prompts; verify equality to
the corrected training split and absence from training before submission.
One sample per prompt, temperature 1, top_p 1, top_k unrestricted, seed 42,
2048 response tokens, 4096 context, float32 actor head for every policy.
Skywork raw scalar reward uses canonical_chat_v1 and the native BF16 RM head;
no length penalty or KL subtraction. Retain per-prompt rewards and token IDs,
report mean / bootstrap 95% CI / mean length / fraction at the token cap.

Implementation and verification:

1. Extend `scripts/eval_checkpoints.py` narrowly: `--run LABEL=none` for Base,
   a direct adapter directory for SFT/final checkpoint, and `--rm-device cuda:0`
   for a single H200. Preserve legacy run-directory discovery and cuda:1 default.
   Test Base dispatch without LoRA, checkpoint step identity, single-GPU CLI,
   missing adapter weights, finite complete scores, and existing precision/cache tests.
2. Recompute the split using the actual dataset and compare all four training
   manifests. Check prompt context lengths and SFT training overlap. Hash all five
   adapters and bind data, source, policies, runtime versions and common sampling.
3. Freeze a small source snapshot under `runs/reward256-canonical-20260917`;
   reference existing weights in place. Use one H200 per job, vLLM memory fraction
   0.45 and RM microbatch 4. Put compilation/HF caches in private local `/tmp`.
   Verify input hashes and H200/runtime versions before model allocation.
4. Submit six recorded commands sequentially, with durable submission logs and
   registry verification to prevent duplicates. Check job startup logs, raw-score
   output completeness, and observed GPU memory before reporting their status.

No training changes, checkpoint deletion, weight copying, or remote judge calls.
The protected SFT directory is read-only input.
