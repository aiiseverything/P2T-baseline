from pathlib import Path
import pytest
from scripts import eval_arena_hard as arena, judge_arena_hard as judge

ROOT = Path(__file__).resolve().parents[1]


def test_creative_subset_and_prompt_are_separate_from_hard500(tmp_path):
    rows = arena.read_jsonl(ROOT / "third_party/arena_hard/data/arena-hard-v2.0/question.jsonl")
    creative = [q for q in rows if q["category"] == "creative_writing"]
    hard = [q for q in rows if q["category"] == "hard_prompt"]
    assert len(creative) == 250 and len(hard) == 500
    assert {q["uid"] for q in creative}.isdisjoint({q["uid"] for q in hard})
    arena.validate_questions(creative, 250, "creative_writing")
    with pytest.raises(ValueError):
        arena.validate_questions(creative)
    upstream=ROOT / "third_party/arena_hard"
    protocol=judge.load_protocol(upstream, "gpt-4o-mini-2024-07-18", judge_model="gpt-4o", category="creative_writing")
    old=judge.load_protocol(upstream, "gpt-4o-mini-2024-07-18", judge_model="gpt-4o")
    assert protocol["system_prompt"] != old["system_prompt"]
    assert protocol["official_baseline_model"] == "gemini-2.0-flash-001"
    def answers(model):
        return [dict(uid=q["uid"],model=model,messages=[dict(role="user",content=q["prompt"]),
            dict(role="assistant",content=dict(answer="example"))]) for q in creative]
    run=judge.JudgeRun(tmp_path,creative,answers(protocol["baseline"]),{"instruct":answers("instruct")},protocol)
    request=run.request("instruct",creative[0]["uid"],0)
    assert request["messages"][0]["content"] == protocol["system_prompt"]
    with pytest.raises(ValueError):
        judge.JudgeRun(tmp_path,creative,answers(protocol["baseline"]),{"instruct":answers("instruct")},old)
