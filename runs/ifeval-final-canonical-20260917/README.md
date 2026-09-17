# IFEval: Base, SFT init, and four final RL policies

This suite evaluates Base, protected native-EOS SFT init, and the final checkpoint-250 for corrected GRPO and VPO lambda=2/4/8. It uses one H200 and one shared vLLM engine sequentially, with a single response per prompt, temperature 1, top_p 1, top_k -1, seed 42, max 2048 response tokens, and an explicit FP32 output head for every policy. Base and SFT use the same weights as before; the FP32 head is an explicit evaluation setting for a uniform comparison.

The 541 prompts contain 834 verifiable constraints. All prompts fit in the 4096-token context (maximum rendered prompt length 373, leaving room for 2048 generated tokens). The selected data is the existing vendored `third_party/ifeval/input_data.jsonl`, copied into this suite. It corrects key 2785 to request three placeholders, consistent with its checker. The older `datasets/ifeval/ifeval_input_data.jsonl` asks for one placeholder despite checking for three. The full difference is recorded in dataset_note.json, before generation.

Primary scores use the unchanged vendored Google strict/loose checkers, with Python random seed 42 reset separately before strict and loose and before each sample/model, and langdetect seed 42. The scoring function restores the caller's random state. The official implementation replaces the explicitly requested '#' and '!' symbols in keys 1122 and 1129 with random alphabetic characters. Fixing random seeds prevents different models or strict/loose passes from receiving different random checker parameters. It does not fix the semantic defect; a separately labelled symbol_sensitivity.json recomputes only these two symbol constraints literally from the SAME responses and reports the score impact. Primary results are never overwritten by this supplementary analysis.

Upstream checker source reviewed: https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/instructions.py (LetterFrequencyChecker).

Generation and scoring source, local pure-Python scoring dependencies, NLTK data, dataset, and adapter identities are bound in experiment.json. No package installation or data download occurs in the GPU job. Model weights and the protected SFT are read-only. Status is tracked in status.json; complete results and independent arithmetic checks are written to summary.json and completion.json.

## Completed results

Job `ifeval-can-six-0917-6293219` succeeded with exit code 0. All six policies have exactly 541 generated responses and 834 strict/loose constraint flags. The driver independently recomputed all four metrics and validated output hashes, generation parameters, policy identity and dataset identity; a second host-side `--validate-only` passed after completion.

| Model | Prompt strict | Prompt loose | Instruction strict | Instruction loose | Mean tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| base | 29.02% | 33.83% | 42.21% | 47.24% | 657.05 | 54 |
| sft-init | 52.31% | 57.12% | 62.95% | 66.91% | 271.46 | 0 |
| grpo | 48.80% | 60.44% | 61.99% | 71.46% | 459.29 | 3 |
| lam2 | 50.09% | 57.30% | 63.43% | 69.30% | 429.66 | 3 |
| lam4 | 46.58% | 59.33% | 59.35% | 69.78% | 407.67 | 2 |
| lam8 | 36.04% | 53.79% | 50.96% | 65.95% | 505.96 | 6 |

Symbol sensitivity completed using the same saved responses. Literal checks change only SFT: prompt strict/loose each decrease by 1/541 (0.184843 percentage points), and instruction strict/loose each decrease by 1/834 (0.119904 percentage points). All other policies are unchanged, and all four metric rankings are unchanged. See symbol_sensitivity.json for the per-question evidence.

Machine-readable summaries: summary.json and summary.csv. Primary per-policy results remain under results/<tag>/results_t1.0_n1.json.
