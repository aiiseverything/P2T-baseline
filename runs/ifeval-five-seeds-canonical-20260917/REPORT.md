# IFEval: all 25 repeated evaluations

All five rjobs succeeded. Each fixed model generated five fresh sets of 541 answers with seeds 42–46. The official checker and langdetect remained fixed at seed 42. All scores below are percentages.

The final column is a project-defined composite: (Prompt strict + Prompt loose + Instruction strict + Instruction loose) / 4. Each component has weight 25%. It is not an official fifth IFEval metric.

## Five-seed mean ± sample standard deviation

| Model | Prompt strict | Prompt loose | Instruction strict | Instruction loose | Four-metric mean (custom) |
|---|---:|---:|---:|---:|---:|
| SFT-init | 51.42 ± 1.00 | 56.08 ± 0.73 | 62.64 ± 0.50 | 66.79 ± 0.28 | 59.23 ± 0.55 |
| GRPO | 47.62 ± 0.43 | 60.26 ± 0.74 | 60.84 ± 0.43 | 70.89 ± 0.70 | 59.90 ± 0.42 |
| VPO λ=2 | 50.17 ± 1.12 | 59.82 ± 0.38 | 62.35 ± 0.92 | 70.43 ± 0.46 | 60.69 ± 0.63 |
| VPO λ=4 | 44.44 ± 1.20 | 58.52 ± 0.31 | 57.48 ± 0.94 | 70.02 ± 0.59 | 57.62 ± 0.63 |
| VPO λ=8 | 37.41 ± 1.22 | 53.90 ± 0.73 | 51.99 ± 0.92 | 66.07 ± 0.81 | 52.34 ± 0.78 |

Standard deviations are over five generation seeds, with ddof=1, in percentage points. For the composite, average the four metrics within each seed first, then calculate SD over the five composite values. Component standard deviations are not averaged. These repetitions do not reflect independently retrained models.

## Individual results

| Model | Generation seed | Prompt strict | Prompt loose | Instruction strict | Instruction loose | Four-metric mean (custom) |
|---|---:|---:|---:|---:|---:|---:|
| SFT-init | 42 | 52.50 | 57.12 | 63.07 | 67.03 | 59.93 |
| SFT-init | 43 | 50.46 | 55.08 | 61.99 | 66.55 | 58.52 |
| SFT-init | 44 | 51.76 | 56.19 | 63.07 | 66.67 | 59.42 |
| SFT-init | 45 | 52.13 | 56.19 | 62.83 | 66.55 | 59.42 |
| SFT-init | 46 | 50.28 | 55.82 | 62.23 | 67.15 | 58.87 |
| GRPO | 42 | 46.95 | 60.44 | 60.43 | 71.22 | 59.76 |
| GRPO | 43 | 48.06 | 60.07 | 61.15 | 71.10 | 60.10 |
| GRPO | 44 | 47.87 | 61.37 | 61.15 | 71.70 | 60.52 |
| GRPO | 45 | 47.50 | 59.33 | 61.15 | 69.90 | 59.47 |
| GRPO | 46 | 47.69 | 60.07 | 60.31 | 70.50 | 59.64 |
| VPO λ=2 | 42 | 50.83 | 60.26 | 62.59 | 70.98 | 61.17 |
| VPO λ=2 | 43 | 49.72 | 59.52 | 62.59 | 70.50 | 60.58 |
| VPO λ=2 | 44 | 48.61 | 59.89 | 60.91 | 70.02 | 59.86 |
| VPO λ=2 | 45 | 50.09 | 59.33 | 62.23 | 69.90 | 60.39 |
| VPO λ=2 | 46 | 51.57 | 60.07 | 63.43 | 70.74 | 61.45 |
| VPO λ=4 | 42 | 46.03 | 58.78 | 58.63 | 69.66 | 58.28 |
| VPO λ=4 | 43 | 45.29 | 58.78 | 58.27 | 70.50 | 58.21 |
| VPO λ=4 | 44 | 43.25 | 58.41 | 57.07 | 70.74 | 57.37 |
| VPO λ=4 | 45 | 44.18 | 58.60 | 57.07 | 69.90 | 57.44 |
| VPO λ=4 | 46 | 43.44 | 58.04 | 56.35 | 69.30 | 56.78 |
| VPO λ=8 | 42 | 39.37 | 54.34 | 53.36 | 66.79 | 53.46 |
| VPO λ=8 | 43 | 36.04 | 53.42 | 51.20 | 65.59 | 51.56 |
| VPO λ=8 | 44 | 37.15 | 54.90 | 51.32 | 66.91 | 52.57 |
| VPO λ=8 | 45 | 37.52 | 53.79 | 52.52 | 66.07 | 52.47 |
| VPO λ=8 | 46 | 36.97 | 53.05 | 51.56 | 64.99 | 51.64 |

## Verification and artifacts

- 25 distinct model/seed pairs; all 13,525 answers and 20,850 instruction flags per strict/loose mode verified.
- Every model has five distinct response sets; 540/541 SFT prompts and 541/541 prompts for each RL model change text across seeds.
- Independent CPU re-scoring of all five seed-42 runs matches every saved strict/loose flag and metric.
- Main scores preserve the prior vendored dataset/checker protocol, including the declared key 2785 placeholder alignment and two unchanged symbol-checker quirks documented in README.md and dataset_note.json.
- Machine-readable files: results_25.csv, model_summary.csv, summary.json, completion.json.
- Each run is an n=1 full evaluation. No pass@5 or best-of-five aggregation is used.

