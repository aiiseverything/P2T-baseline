# Arena-Hard v2 — six canonical policies

Status: COMPLETE at 2026-09-17 16:24:14 HKT. All six policies have 500 prompts and 1,000 valid ordered games each (3,000 candidate answers, 6,000 valid games, zero dropped questions). The independent supplemental verifier passed all 6,000 request/parse checks and all 42 official numerical comparisons (maximum absolute difference 6.94e-18). Total recorded API attempts: 6,015, including 15 archived predecessor attempts.

The comparison evaluates Base, protected native-EOS SFT-init, and the corrected final GRPO / VPO λ2 / λ4 / λ8 adapters from `runs/rl-fp32-is-canonical-20260917`, all at checkpoint 250. Exact model identities and frozen code hashes are recorded in `experiment.json`.

- Upstream: https://github.com/lmarena/arena-hard-auto at `196f6b826783b3da7310e361a805fa36f0be83f3`.
- Dataset: all 500 `hard_prompt` records (253 coding, 247 math); the separate 250 creative-writing prompts are outside this run.
- Reference: the upstream `o3-mini-2025-01-31` answers for these exact 500 IDs. Reference prompts and style metadata are verified against the official question file and helpers.
- Generation: common Qwen3-14B-Base, explicit FP32 output head, existing chat template and token support, both stop IDs 151643/151645, one response, temperature 1, top_p 1, top_k -1, seed 42, maximum 4,096 output tokens and 16,384 total context tokens. No input truncation. The longest rendered prompt is 8,431 Qwen tokens.
- Judge: GPT-4.1 through the existing configured relay, official system/user prompts, temperature 0, max_tokens 16,000, two answer orders for each response. This requires 6,000 valid judgments in total; the fixed five-question-per-model pilot uses 12 concurrent requests and is reused in the full run, which uses 32 concurrent requests. Structurally invalid completed outputs may require additional identical-request attempts under the predeclared five-attempt policy.
- Five outcomes: strong/slight preference in either direction or tie. Official raw aggregation weights strong decisions three times, ties one-half, and reverses game zero to the candidate's perspective.
- Style control: official combined GPT-4o token-length and Markdown-header/list/bold controls, with the pinned upstream Bradley–Terry optimizer. Qwen generation lengths are reported separately from GPT-4o style lengths.
- Uncertainty: official 100-replicate bootstrap with 5th/95th percentiles (90% intervals), seed 42. The official method resamples expanded judgment rows, not paired prompts. These are not paired significance tests of VPO against GRPO.
- SFT's actual 2,500-example sample was reconstructed and its historical sample SHA verified. Its prompts and each RL arm's 2,000 actual prompt records have zero normalized exact matches with these 500 questions. This does not establish absence of semantic overlap or base-model pretraining contamination.

`prepare_inputs.py` records input verification and the predetermined pilot. `run_evaluation.py --preflight` verifies code, inputs and model identities; `--validate-only` verifies generated outputs. `model_answer/` contains all answers in official format. Per-answer generation metadata includes Qwen length, finish reason and stop token. The judging state records each paid attempt before dispatch and allows completed games to be reused without new requests.

Scores compare each candidate against the reference under this run's common settings. They are not direct VPO-versus-GRPO match results, and this benchmark uses an LLM judge rather than executing all proposed programs or proving mathematical answers.

## Verified generation

| Model | Answers | Mean Qwen tokens | At 4096-token limit | Empty |
|---|---:|---:|---:|---:|
| base | 500 | 1206.78 | 32 | 0 |
| sft-init | 500 | 424.31 | 0 | 0 |
| grpo | 500 | 850.97 | 1 | 0 |
| lam2 | 500 | 794.39 | 0 | 0 |
| lam4 | 500 | 708.95 | 0 | 0 |
| lam8 | 500 | 819.62 | 2 | 0 |

All 3,000 style metadata objects were independently recomputed using host tiktoken 0.10.0 and match GPU tiktoken 0.14.0 exactly.

The durable host controller was launched at 2026-09-17 13:33 HKT. `continuation_state.json` tracks its current phase, and `job/continuation/` retains command logs. It reuses valid judgments, resolves only eligible structural failures within the fixed retry bound, scores full coverage, and writes `evaluation_complete.json` only after the independent full verification passes.

## Explicit transport recovery

`manual_transport_recovery.json` declares the single reviewed recovery of Base / `34fd667185674f47` / order 1 before its retry. The original RemoteProtocolError record has no returned answer, verdict, usage or provider ID and remains byte-for-byte in the attempt archive. The exact request was retried once and returned a valid judgment; possible duplicate remote billing remains included in account usage. The separate λ2 truncated output was retried under the original uniform structural-invalid policy.

`resume_manual_recovery.py` reuses the original controller and paid caches, with an explicitly logged final-verifier override to `verify_manual_recovery.py`. The latter accepts only this declared two-attempt transport chain; all other chains retain the original strict checks, and the original full 6,000-game coverage, official parsing, and 42 score comparisons remain required. Original controller, verifier, frozen judge, retry helper, model identities, and cost decision remain unchanged. Original failed controller state and host/cost snapshots are in `job/transport_incident/`.

## Final verified scores

Judge: GPT-4.1; reference: official `o3-mini-2025-01-31`; all 500 hard prompts. Raw below uses the official mean of 100 bootstrap means; the interim snapshots used the direct weighted mean. Controlled below uses the official joint six-model length+Markdown bootstrap median. Intervals are the official 90% intervals.

| Model | Raw (%) | Raw 90% interval | Length+Markdown controlled (%) | Controlled 90% interval |
|---|---:|---:|---:|---:|
| base | 2.57 | 2.17–3.03 | 1.01 | 0.84–1.20 |
| sft-init | 0.68 | 0.44–0.93 | 1.60 | 1.34–1.93 |
| grpo | 5.08 | 4.42–5.85 | 3.02 | 2.60–3.47 |
| lam2 | 5.05 | 4.42–5.80 | 3.18 | 2.78–3.61 |
| lam4 | 5.41 | 4.79–6.00 | 3.33 | 2.85–3.79 |
| lam8 | 6.36 | 5.68–7.24 | 3.67 | 3.25–4.14 |

VPO λ8 has the highest controlled point estimate, 0.65 percentage points above GRPO. No paired VPO-versus-GRPO significance test was performed; the official intervals resample expanded game rows. These results do not establish a statistically significant pairwise advantage. The reference effort is inferred to be medium from upstream naming/defaults; published answer records do not retain per-request effort.

Post-completion account usage delta: approximately CNY 38.01, including retries; the last saved run billing was CNY 38.01. This is an account-wide delta, not an itemized invoice. See `job/final_billing_observation.json`.

Initially unresolved games: 12 structurally invalid outputs and 1 transport failure. All were resolved under the declared policies, preserving all prior records. Final files: `scores/results.json`, `scores/results.csv`, and `evaluation_complete.json`. The verifier uses the original official computations plus the single explicitly documented transport-chain exception; the original strict verifier alone is not claimed to accept that exception.
