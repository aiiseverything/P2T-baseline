# Llama AlpacaEval execution plan

The user authorized one generation seed for all six Llama policies, parallel rjobs, and the same reference answers and judge as the Qwen experiment.

- [x] Recover the actual Qwen canonical protocol from commands, manifests and annotations.
- [x] Select the accepted Llama Base, protected SFT, and final step-250 GRPO/VPO adapters; reuse existing weights.
- [x] Fix and test the existing Alpaca entry point's missing Llama PAD configuration; verify explicit prompt IDs and returned IDs.
- [x] Freeze a separate suite at `/data/VPO-RM/runs/llama31-alpacaeval-20260918`, including all executable source, exact references/template and resource/model fingerprints.
- [x] Validate all 805 prompts with the real tokenizer, compare Base/SFT tokenization, and check tests before submission.
- [x] Submit exactly six independent 1-H200 jobs, one per model, with recorded commands and submission IDs.
- [x] Check actual runtime preflight and generation progress. Require complete, correctly paired 805-row outputs before paid judging.
- [x] Run the unchanged Qwen judge once over all six tags, with 16 request workers and a 30-CNY process-level dispatch guard; preserve the original internal API retry behavior.
- [x] Validate all 4,830 annotations, result hashes and recomputed win rates, then write summaries and report results.

Generation uses seed42, temperature1, top_p1, top_k-1, n1, max_tokens2048, context4096, BF16 backbone and FP32 actor head. The six heads use one consistent protocol, matching Qwen canonical RL; historical Qwen Base/SFT used older native-head runs and must not be represented as fully identical generation implementations.

Reference file SHA256: `348901ca132cd78b77682a21cfcd1942b19738be5bafc0add8a6cbfec5992d1f`, reference generator `gpt4_1106_preview`. Judge template SHA256: `784227e6dc2832fc08c43d2c8ea3a308e7523780187a1aaad2f85e30bac85f62`. Judge source SHA256: `9fa7f4cbf110b21bb520805a9e913cc2ccbf8da7b539ca184963429231ef0dce`.

Judge is GPT-4.1 via the existing LinkAPI endpoint, max_tokens1/temperature1/logprobs/top_logprobs5, one deterministic MD5-selected answer order per instruction. Weighted win rate averages candidate preference; hard win rate counts preference >0.5. This is the project's internal AlpacaEval protocol, not an official LC leaderboard score. Runtime retries remain at most five POST attempts per request, while the watcher never automatically restarts a failed paid invocation. The dispatch budget is not a prepaid hard cap.

Credentials stay on the networked host. Generated outputs and manifests go to `/data`; original weights and older experiment artifacts remain untouched. No GPU training or new checkpoint export is needed.
