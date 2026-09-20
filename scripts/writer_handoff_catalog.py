"""Explicit experiment identities and selection rules for the writer handoff."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/VPO-RM")
OUT = ROOT / "runs/paper-writer-handoff-20260919"
PACKAGE = OUT / "writer_packet"
ROOTS = {"shared": ROOT, "data": DATA}

FAMILIES = {
    "qwen_base_sft": dict(label="Qwen3-14B-Base + SFT", role="Main 2x2", init="2500-example SFT", rm="Skywork-Qwen3-8B",
        rl=ROOT / "runs/rl-fp32-is-canonical-20260917", arms=["grpo", "lam2", "lam4", "lam8"],
        reward=ROOT / "runs/reward256-canonical-20260917",
        ifeval=ROOT / "runs/ifeval-five-seeds-canonical-20260917",
        alpaca={"base": ROOT / "runs/alpacaeval-evals/base", "sft-init": ROOT / "runs/alpacaeval-native-eos/sft-native-eos-clean2k5e2",
                "grpo": ROOT / "runs/alpacaeval-final-canonical-20260917/generations/grpo",
                **{f"lam{n}": ROOT / f"runs/alpacaeval-lam{n}-canonical-20260917/generations/lam{n}" for n in [2,4,8]}},
        arena=DATA / "runs/arena-hard-gpt4o-retry5-gpt41-20260918"),
    "qwen_instruct_direct": dict(label="Qwen3-14B Instruct, direct RL", role="Main 2x2", init="No additional SFT", rm="Skywork-Qwen3-8B",
        rl=DATA / "runs/direct-rl-qwen-instruct-formal-20260918/qwen", arms=["grpo", "lam4"],
        reward=DATA / "runs/qwen-instruct-evals-20260919/reward256",
        ifeval=DATA / "runs/qwen-instruct-evals-20260919/ifeval5",
        alpaca=DATA / "runs/qwen-instruct-evals-20260919/alpaca/generations",
        arena=DATA / "runs/qwen-instruct-evals-20260919/arena/judge_pipeline"),
    "llama_base_sft": dict(label="Llama-3.1-8B Base + SFT", role="Main 2x2", init="2500-example SFT", rm="Skywork-Llama-3.1-8B",
        rl=ROOT / "runs/llama31-base-rl-20260919", arms=["grpo", "lam4"],
        reward=DATA / "runs/llama-base-evals-20260919/reward256",
        ifeval=DATA / "runs/llama-base-evals-20260919/ifeval5",
        alpaca=DATA / "runs/llama-base-evals-20260919/alpaca/generations",
        arena=DATA / "runs/llama-base-evals-20260919/arena/judge_pipeline"),
    "llama_instruct_sft": dict(label="Llama-3.1-8B Instruct + SFT", role="Historical additional-SFT runs; outside paper experiment matrix", init="Additional 2500-example SFT", rm="Skywork-Llama-3.1-8B",
        rl=DATA / "runs/llama31-rl-canonical-20260918-v3", arms=["grpo", "lam2", "lam4", "lam8"],
        reward=DATA / "runs/llama31-evals-20260918/reward256",
        ifeval=DATA / "runs/llama31-evals-20260918/ifeval5",
        alpaca=DATA / "runs/llama31-alpacaeval-20260918/generations", arena=None),
    "llama_instruct_direct": dict(label="Llama-3.1-8B Instruct, direct RL", role="Main 2x2 (corrected per user clarification)", init="No additional SFT", rm="Skywork-Llama-3.1-8B",
        rl=DATA / "runs/direct-rl-no-sft-20260918/llama", arms=["grpo", "lam4"],
        reward=DATA / "runs/direct-llama-reward-arena-20260918/reward256",
        ifeval=DATA / "runs/direct-llama-evals-20260918/ifeval5",
        alpaca=DATA / "runs/direct-llama-evals-20260918/alpacaeval/generations",
        arena=DATA / "runs/direct-llama-reward-arena-20260918/arena_hard/judge_pipeline"),
    "random_credit": dict(label="Qwen3-14B random-credit control", role="Credit ablation", init="Same SFT as Qwen base", rm="Skywork-Qwen3-8B",
        rl=ROOT / "runs/rl-ablation-random-credit-20260919", arms=["random_direction"],
        reward=DATA / "runs/randdir-evals-20260919/reward256",
        ifeval=DATA / "runs/randdir-evals-20260919/ifeval5",
        alpaca=DATA / "runs/randdir-evals-20260919/alpaca/generations",
        arena=DATA / "runs/randdir-evals-20260919/arena/judge_pipeline"),
}

SFT = {
    "qwen_base": (ROOT / "runs/sft-native-eos-clean2k5e2", ROOT / "models/sft-native-eos-clean2k5e2"),
    "llama_base": (DATA / "runs/llama31-base-sft-20260919", DATA / "models/sft-llama31-8b-base-clean2k5e2-20260919"),
    "llama_instruct": (DATA / "runs/llama31-sft-aligned-20260917-attempt2", DATA / "models/sft-llama31-8b-instruct-clean2k5e2-20260917"),
}

BENCHMARK_SUITES = {
    "shared": ["alpacaeval-final-canonical-20260917", "alpacaeval-lam2-canonical-20260917", "alpacaeval-lam4-canonical-20260917", "alpacaeval-lam8-canonical-20260917", "alpacaeval-native-eos", "ifeval-final-canonical-20260917", "ifeval-five-seeds-canonical-20260917", "reward256-canonical-20260917", "arena-hard-v2-canonical-20260917", "arena-hard-v2-gpt4omini-20260917", "arena-hard-v2-gpt4o-judge-20260917"],
    "data": ["llama31-evals-20260918", "llama31-alpacaeval-20260918", "llama-base-evals-20260919", "qwen-instruct-evals-20260919", "qwen-instruct-ifeval10-20260919", "randdir-evals-20260919", "randdir-reward256-seeds-20260919", "direct-llama-evals-20260918", "direct-llama-reward-arena-20260918", "arena-hard-gpt4o-retry5-gpt41-20260918"],
}
BENCHMARK_ROOTS = {ROOTS[label] / "runs" / name for label,names in BENCHMARK_SUITES.items() for name in names}
RL_ROOTS = {spec["rl"] for spec in FAMILIES.values()}
SFT_ROOTS = {p for pair in SFT.values() for p in pair}

EXCLUDE_CATEGORIES = {"adapter_weights", "base_and_reward_model_weights", "optimizer_training_state", "runtime_cache_and_dependencies"}
CODE_SUFFIXES = {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}
RECORD_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".png", ".svg", ".pdf", ".sha256"}
PROBABILITY_ROLLOUTS = {1, 2, 50, 100, 150, 200, 250}


def paper_role(family, arm=None):
    if family == 'llama_instruct_sft':
        return 'Historical; outside paper matrix'
    if family == 'random_credit' or (family == 'qwen_base_sft' and arm in ['lam2','lam8']):
        return 'Qwen-Base ablation'
    return 'Main 2x2'


def contains(path, roots):
    return any(path.is_relative_to(root) for root in roots)


def select_file(row):
    """Return package section and rationale, or an explicit exclusion reason."""
    path = ROOTS[row["root"]] / row["relative_path"]
    rel = Path(row["relative_path"])
    parts, name, cat = rel.parts, rel.name, row["category"]
    if cat in EXCLUDE_CATEGORIES:
        return None, "Weights, optimizer state, runtime dependencies or cache"
    if any(p in [".claude", ".agents", ".codex", ".maintenance", ".download-logs"] for p in parts):
        return None, "Operational/user configuration; not writer evidence"
    if name.startswith(".") or any(word in name.lower() for word in ["oauth", "credential", "billing", "balance", "usage-query"]):
        return None, "Operational or account-related material"
    if any(p.startswith(('llama31-rl-canonical-20260918','llama31-sft-aligned-20260917','sft-llama31-8b-instruct-clean2k5e2-20260917')) for p in parts):
        return None, 'User explicitly excludes Llama-Instruct + SFT campaigns, including failed/preflight versions'
    old = FAMILIES['llama_instruct_sft']
    historical_roots = {old['rl'],old['reward'],old['ifeval'],old['alpaca'],*SFT['llama_instruct']}
    if contains(path, historical_roots):
        # Original, unadapted Instruct baseline evaluations were stored in an
        # older campaign. Retain only those baseline records, not its SFT/RL arms.
        if 'base' in parts and ('results' in parts or 'generations' in parts) and path.suffix in RECORD_SUFFIXES:
            return '05_benchmark_evidence', 'Reused original Instruct baseline; adapter=None, not an SFT experiment'
        return None, 'User explicitly excludes Llama-Instruct + SFT runs from the writer packet'
    if any(p.startswith('pytest-') or p.startswith('test_') for p in parts[:-1]):
        return None, "Synthetic test fixtures, including deliberately corrupted inputs; keep validation reports only"
    source_part = any(p == "source" or p.endswith("-source") or p in ["source_snapshot", "source-snapshot"] for p in parts)
    if source_part and path.suffix not in CODE_SUFFIXES and name not in ["requirements.txt", "LICENSE", "input_data.jsonl"]:
        return None, "Bundled source assets duplicated elsewhere; retain code and data identities"
    if "checkpoint-" in str(rel) or any(p.startswith("vllm-lora") for p in parts):
        if name not in ["run_manifest.json", "adapter_config.json"]:
            return None, "Checkpoint copy; original identity manifests are sufficient"
    if contains(path, RL_ROOTS):
        if any(p in ["gpu-preflight", "preflight", "audit", "analysis"] for p in parts):
            if cat in {"token_credit_tensors", "token_probability_tensors", "rollout_response_token_ids"}:
                return None, "Preflight/duplicated tensors; formal rollouts retained"
        if cat == "token_probability_tensors":
            rollout = int(name.split("-")[1])
            if rollout not in PROBABILITY_ROLLOUTS:
                return None, "Probability tensor audit sample: retain fixed rollouts 1,2,50,100,150,200,250"
            return "04_probability_audit_samples", "Fixed non-outcome-based sample; full aggregate diagnostics retained"
        if cat == "token_credit_tensors":
            return "03_token_credit", "All saved formal token weights/directions/temperatures"
        if cat in ["rollout_response_token_ids", "rollout_per_response_rewards", "rollout_prompt_text"]:
            return "02_rl_raw_rollouts", "Original per-response tokens, prompts, and rewards for all formal rollouts"
        if path.suffix in RECORD_SUFFIXES | CODE_SUFFIXES or name in ["exit_code", "stage"]:
            return "01_rl_metrics_and_protocol", "Training curves, protocols, completed-run evidence and frozen code"
    if contains(path, BENCHMARK_ROOTS):
        if any(p.startswith("interim-") or p in ["cache", "history", "original_generation"] for p in parts):
            if name not in ["README.md", "experiment.json", "evaluation_complete.json"]:
                return None, "Intermediate/duplicated snapshots; final and per-attempt records retained"
        if path.suffix in RECORD_SUFFIXES | CODE_SUFFIXES or name in ["exit_code", "stage"]:
            return "05_benchmark_evidence", "Prompts, responses, scores, judge attempts/exclusions, protocols and source"
    if contains(path, SFT_ROOTS):
        if path.suffix in RECORD_SUFFIXES | CODE_SUFFIXES and "tokenizer" not in name and "inputs" not in parts:
            return "06_sft_evidence", "SFT training records and train/test generation probes; no adapters"
    if rel.parts[:2] == ("runs", "paper-2x2-rl-20260919"):
        return "07_existing_paper_figures", "Existing paper plot CSV, figures, scripts, and exact case-study tokens"
    # Historical results are deliberately separate from current training/evaluation.
    if parts[0] in ["runs", "analysis"]:
        if any(p in ["source", "source_snapshot", "source-snapshot"] for p in parts):
            return None, "Historical source duplicate; current source/protocol records retained"
        if cat in ["rollout_response_token_ids", "rollout_prompt_text", "token_credit_tensors", "token_probability_tensors"]:
            return None, "Historical raw token dumps: inventoried, not in the writer packet"
        if path.suffix in RECORD_SUFFIXES | CODE_SUFFIXES and int(row["bytes"]) <= 30_000_000:
            return "08_historical_and_diagnostic", "Legacy/SFT-variant/early benchmark context; never pooled with canonical results"
        if name == "archived-profile-manifests.tar.gz":
            return "08_historical_and_diagnostic", "Archived historical configuration identities"
    if parts[0] in ["scripts", "vpo_rm", "configs", "docs", "requirements"] and path.suffix in CODE_SUFFIXES | {".json", ".txt"}:
        return "09_code_and_method", "Current source and method documentation; frozen experiment sources take precedence"
    if parts[0] == 'code' and len(parts)>1 and parts[1] in ['scripts','vpo_rm','configs','docs'] and path.suffix in CODE_SUFFIXES | {'.json','.txt'}:
        return "09_code_and_method", "Data-root source and method documentation; frozen experiment sources take precedence"
    if parts[0] == 'third_party' and len(parts)>1 and parts[1] == 'arena_hard' and path.suffix in CODE_SUFFIXES | {'.json','.jsonl'}:
        return "09_code_and_method", "Pinned Arena scoring source and published benchmark assets"
    if len(parts) == 1 and path.suffix in [".md", ".toml"]:
        return "09_code_and_method", "Project/method documentation"
    if rel.as_posix() in ["datasets/ultrafeedback_binarized/data/train_sft-00000-of-00001.parquet", "datasets/sft_v2/sft_clean.parquet"]:
        return "10_input_data_and_tokenizers", "Canonical source data once; omit duplicate corpus variants"
    if parts[0] == "datasets" and path.suffix in [".json", ".jsonl", ".md", ".csv"] and int(row["bytes"]) <= 5_000_000:
        return "10_input_data_and_tokenizers", "Benchmark questions, small data manifests and prompt-set provenance"
    if parts[0] == "models" and len(parts) == 3 and parts[1] in ["Qwen3-14B-Base", "Qwen3-14B", "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Skywork-Reward-V2-Qwen3-8B", "Skywork-Reward-Llama-3.1-8B-v0.2"]:
        if name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json", "chat_template.jinja"]:
            return "10_input_data_and_tokenizers", "Decode saved token IDs and inspect actor/RM protocol without model weights"
    return None, "Not needed for experiment writing; indexed in the full storage inventory"
