# Llama reward and IFEval evaluation

User request: evaluate Base, SFT-init, GRPO and VPO lambda 2/4/8 after completed
Llama training. Submit six parallel reward rjobs on the same 256 held-out prompts
as Qwen, using the Llama RM and one generation seed. Evaluate all six policies
on IFEval with five generation seeds. Preserve protected SFT and all checkpoints.

This is a bounded adaptation of existing evaluation flows. Use the existing
isolated `/data/VPO-RM/code` worktree and new `/data` output directories.

- [x] Audit Qwen's actual frozen experiments, rather than current CLI defaults.
  Reward: seed42, temperature/top_p1, unrestricted top_k, max2048/min0,
  one response per each of the frozen256 validation prompts, raw RM score with
  no length or KL penalty. IFEval: canonical541 prompts/834 instructions,
  generation seeds42–46, checker/langdetect seed42, same sampling and max2048.
- [x] Correct both evaluation entry points to use the saved Llama tokenizer and
  explicit prompt IDs without an added BOS; check vLLM's returned prompt IDs.
  Bind tokenizer provenance, native three-stop policy, dedicated pad128004 and
  FP32 actor head. Preserve Qwen behavior with regression tests.
- [x] Use canonical complete RM chats and native BF16 RM scores. Llama uses
  RM microbatch1, as verified during training; Qwen used microbatch4. Record
  this model-specific difference without suggesting raw scales match across RMs.
- [x] Freeze code, canonical data, six input identities and commands. Reuse
  final step250 exports only after comparing their hashes with accepted final
  checkpoints. Base means the unmodified Llama-3.1-8B-Instruct checkpoint.
- [x] Test and independently review worker commands, six-model coverage,
  Base-without-LoRA handling, all30 IFEval model/seed results and result validation.
- [x] Submit six single-H200 reward jobs and six single-H200 IFEval jobs, each
  IFEval job evaluating five seeds in one engine. Record scheduler IDs and
  verify real runtime, data/stop/prompt protocol, source identities and progress.
- [ ] Save automatic post-run reward summaries and the30 IFEval rows with the
  four official metrics plus the project-defined four-metric arithmetic mean.

Actor: `meta-llama/Llama-3.1-8B-Instruct`.
Reward model: `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`.

Expected output parent: `/data/VPO-RM/runs/llama31-evals-20260918`.
Submission checkpoint, 09:20 HKT: all12 jobs submitted and scheduler-registered.
All currently Starting; real runtime/generation checks remain pending. Evidence
is in the campaign `jobs.json`, `submission-complete.json`, per-job registration
logs and `README.md`. A detached watcher will validate and summarize completed
results. Launch must not be reported as completed evaluation.

Runtime checkpoint, 09:23 HKT: all12 actual H200 workers passed input/source,
software, tokenizer, complete prompt ID and context-budget gates and entered
the evaluation subprocess. Root checked all12 runtime/command artifacts in
`startup-root-review.json`. Models are loading/initializing; final results remain
pending. The detached summarizer is running (PID recorded in `watch-launch.json`).

No changes to frozen training sources, no new model downloads, no checkpoint
copies or deletions, and no external judge/API charges are needed.
