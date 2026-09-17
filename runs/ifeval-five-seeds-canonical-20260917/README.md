# Five-model IFEval repeated over five generation seeds

Requested models: protected native-EOS SFT initialization, corrected final
GRPO and VPO lambda 2, 4, 8 (`checkpoint-250`). All 25 runs are generated anew.

Five independent one-H200 rjobs each load one model and run generation seeds
42, 43, 44, 45, 46. Each seed evaluates all 541 prompts / 834 instructions with
one answer per prompt, temperature 1, top_p 1, unrestricted top_k, maximum
2048 output tokens, 4096 context and an FP32 actor head. The engine is loaded
once with seed 42; each generation call explicitly uses its own sampling seed.

The official seeded checker and langdetect always use seed 42. Changing the
generation seed must not change the grading rules. The unchanged vendored
checker and dataset are copied from the preceding canonical IFEval suite.
`dataset_note.json` preserves that suite's declared key 2785 prompt correction
to three placeholders, matching its fixed checker. Its two symbol-to-letter
checker quirks (keys 1122 and 1129) are also unchanged in primary scores.

Each independent run writes
`results/seed-<seed>/<model>/{results,generations,manifest}_t1.0_n1.*`.
The four main metrics are prompt strict, prompt loose, instruction strict,
and instruction loose. Final reporting includes all 25 individual rows and
each model's five-seed mean and sample standard deviation (ddof=1).
These are repeated generations of fixed trained models, not five training
seeds, and scores are not pass@5 or best-of-five.

The experiment manifest binds model inputs, dataset, runtime and source.
Weights are read in place. Job caches use local `/tmp`; no model is copied
or trained and the protected SFT is not modified.

## Completion

All 25 evaluations completed successfully by 2026-09-17 18:08:10 Asia/Hong_Kong.
See [REPORT.md](REPORT.md), [results_25.csv](results_25.csv), and [model_summary.csv](model_summary.csv).
All 13,525 generated answers passed the completion checks.

## Project-defined four-metric mean

The user-requested overall summary is the equal-weight arithmetic mean of the
four official IFEval metrics, computed separately for each generation seed.
This is a project-defined composite, not an official fifth IFEval metric.
The 25-run CSV and report append this column; model summaries report the five
composite values' mean, sample SD (ddof=1), minimum and maximum. The composite
SD is calculated from those five values, not by averaging component SDs.
