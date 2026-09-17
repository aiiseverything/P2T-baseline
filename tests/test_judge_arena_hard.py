"""Arena judge protocol and paid-request lifecycle, entirely offline."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import judge_arena_hard as judge


@pytest.fixture
def inputs():
    questions = [{"uid": f"u{i}", "category": "hard_prompt", "prompt": f"Question {i} {{ANSWER_A}}"}
                 for i in range(500)]
    def answers(model):
        return [{"uid": q["uid"], "model": model, "messages": [
            {"role": "user", "content": q["prompt"]},
            {"role": "assistant", "content": {"answer": model + " answer " + q["uid"]}},
        ]} for q in questions]
    return questions, answers(judge.BASELINE_MODEL), {"candidate": answers("candidate")}


@pytest.fixture
def protocol():
    return judge.load_protocol(Path(__file__).resolve().parents[1] / "third_party/arena_hard")


class FakeRelay:
    def __init__(self, text="Reasoning. [[B>A]]", error=None, usage_values=None):
        self.text, self.error = text, error
        self.calls = []
        self.usage_values = iter(usage_values or [10.0] * 100)
    def usage(self, start_date):
        return next(self.usage_values)
    def judge_call(self, request):
        self.calls.append(copy.deepcopy(request))
        if self.error:
            raise self.error
        return {"answer": self.text, "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                               "total_tokens": 120},
                "finish_reason": "stop", "response_id": "fake-response"}


def make_run(tmp_path, inputs, protocol):
    return judge.JudgeRun(tmp_path, *inputs, protocol)


CUSTOM_BASELINE = "gpt-4o-mini-2024-07-18"


def test_custom_baseline_protocol_keeps_official_source_and_default_unchanged(protocol):
    custom = judge.load_protocol(judge.ROOT / "third_party/arena_hard", baseline_model=CUSTOM_BASELINE)
    assert custom["baseline"] == CUSTOM_BASELINE
    assert custom["protocol"] == "arena_hard_v2_gpt41_two_order_custom_baseline_v1"
    assert custom["official_baseline_model"] == judge.BASELINE_MODEL
    assert custom["uses_official_baseline"] is False
    for key in ("source_sha256", "system_prompt", "prompt_template", "temperature", "max_tokens", "judge"):
        assert custom[key] == protocol[key]
    assert protocol["baseline"] == judge.BASELINE_MODEL
    assert protocol["protocol"] == "arena_hard_v2_gpt41_two_order_v1"
    assert "uses_official_baseline" not in protocol


def test_custom_baseline_still_rejects_modified_official_settings(tmp_path):
    source = judge.ROOT / "third_party/arena_hard"
    for name in ("config/arena-hard-v2.0.yaml", "utils/judge_utils.py", "gen_judgment.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    settings = tmp_path / "utils/judge_utils.py"
    settings.write_text(settings.read_text().replace(judge.BASELINE_MODEL, CUSTOM_BASELINE))
    with pytest.raises(ValueError, match="pinned official"):
        judge.load_protocol(tmp_path, baseline_model=CUSTOM_BASELINE)


def test_custom_baseline_identity_export_and_resume_never_reuses_o3_cache(tmp_path, inputs, protocol):
    custom = judge.load_protocol(judge.ROOT / "third_party/arena_hard", baseline_model=CUSTOM_BASELINE)
    with pytest.raises(ValueError, match="model identity"):
        make_run(tmp_path, inputs, custom)
    altered = copy.deepcopy(inputs)
    for row in altered[1]:
        row["model"] = CUSTOM_BASELINE
        row["messages"][-1]["content"]["answer"] = "new baseline response " + row["uid"]
    old = tmp_path / "old"
    make_run(old, inputs, protocol).run(lambda: FakeRelay(), ["u0"], budget_cny=5)
    with pytest.raises(ValueError, match="identity"):
        make_run(old, altered, custom).run(lambda: pytest.fail("must not dispatch"), ["u0"], budget_cny=5)
    fresh = tmp_path / "new"
    relay = FakeRelay()
    make_run(fresh, altered, custom).run(lambda: relay, ["u0"], budget_cny=5)
    row = judge.load_jsonl(fresh / "candidate.jsonl")[0]
    assert row["baseline"] == CUSTOM_BASELINE
    assert len(relay.calls) == 2
    assert "new baseline response u0" in row["games"][0]["prompt"][1]["content"]
    make_run(fresh, altered, custom).run(lambda: pytest.fail("no duplicate calls"), ["u0"], budget_cny=5)


def test_custom_baseline_cli_dry_run_requires_explicit_model_and_does_no_writes(tmp_path, inputs, monkeypatch, capsys):
    questions, baseline, answers = copy.deepcopy(inputs)
    for row in baseline:
        row['model'] = CUSTOM_BASELINE
    for filename, rows in [('questions.jsonl', questions), ('reference.jsonl', baseline),
                            ('candidate.jsonl', answers['candidate'])]:
        (tmp_path / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    output = tmp_path / 'untouched'
    argv = ['judge_arena_hard.py', '--questions', str(tmp_path / 'questions.jsonl'),
            '--baseline', str(tmp_path / 'reference.jsonl'), '--answers-dir', str(tmp_path),
            '--output-dir', str(output), '--tags', 'candidate', '--budget-cny', '5', '--dry-run']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='model identity'):
        judge.main()
    monkeypatch.setattr(sys, 'argv', [*argv, '--baseline-model', CUSTOM_BASELINE])
    judge.main()
    assert json.loads(capsys.readouterr().out)['status'] == 'inputs_validated'
    assert not output.exists()


@pytest.mark.parametrize("text,expected", [
    ("[[A>>B]]", "A>>B"), ("[[A>B]]", "A>B"), ("[[A=B]]", "A=B"),
    ("[[B>A]]", "B>A"), ("[[B>>A]]", "B>>A"),
    ("Earlier [[A>B]] then [[b>>a]]", "B>>A"),
    ("[[A>B]] later [B>A]", "A>B"), ("[B>A]", "B>A"),
    ("m", None), ("[[A>>>B]]", None), ("[[A>B]] then [[A==B]]", None),
    ("No verdict", None),
])
def test_five_scores_and_official_last_regex_priority(text, expected, protocol):
    assert judge.parse_score(text, protocol["regex_patterns"]) == expected


def test_exact_official_prompts_and_answer_orders(inputs, protocol):
    q, baseline, models = inputs
    for order in [0, 1]:
        request = judge.make_request(q[0], baseline[0], models["candidate"][0], order, protocol)
        first, second = ((baseline[0], models["candidate"][0]) if order == 0
                         else (models["candidate"][0], baseline[0]))
        assert request == {
            "model": "gpt-4.1", "temperature": 0.0, "max_tokens": 16000,
            "messages": [{"role": "system", "content": protocol["system_prompt"]},
                         {"role": "user", "content": protocol["prompt_template"].format(
                             QUESTION=q[0]["prompt"], ANSWER_A=first["messages"][-1]["content"]["answer"],
                             ANSWER_B=second["messages"][-1]["content"]["answer"])}],
        }
        assert "{ANSWER_A}" in request["messages"][1]["content"]
        assert "logprobs" not in request


@pytest.mark.parametrize("damage", ["missing", "duplicate", "prompt", "baseline", "category"])
def test_bad_coverage_rejected_before_dispatch(tmp_path, inputs, protocol, damage):
    questions, baseline, answers = copy.deepcopy(inputs)
    if damage == "missing": answers["candidate"].pop()
    elif damage == "duplicate": baseline[-1] = baseline[0]
    elif damage == "prompt": answers["candidate"][0]["messages"][0]["content"] = "wrong"
    elif damage == "baseline": baseline[0]["model"] = "wrong"
    else: questions[0]["category"] = "creative_writing"
    with pytest.raises(ValueError):
        judge.JudgeRun(tmp_path, questions, baseline, answers, protocol)


def test_per_game_resume_and_official_rows_only_after_both_valid(tmp_path, inputs, protocol):
    run = make_run(tmp_path, inputs, protocol)
    relay = FakeRelay()
    first = run.run(lambda: relay, ["u0"], workers=1, budget_cny=5, max_requests=1)
    assert first["complete"] is False and len(relay.calls) == 1
    assert (tmp_path / "candidate.jsonl").read_text() == ""
    second = make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0"], workers=1, budget_cny=5)
    assert second["complete"] is True and len(relay.calls) == 2
    rows = [json.loads(x) for x in (tmp_path / "candidate.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and set(rows[0]) == {"uid", "category", "judge", "model", "baseline", "games"}
    assert rows[0]["baseline"] == judge.BASELINE_MODEL
    assert len(rows[0]["games"]) == 2
    assert all(set(g) == {"score", "judgment", "prompt"} for g in rows[0]["games"])
    assert rows[0]["games"][0]["prompt"] != rows[0]["games"][1]["prompt"]
    third = make_run(tmp_path, inputs, protocol).run(
        lambda: pytest.fail("completed resume must not call API or usage"), ["u0"], budget_cny=5)
    assert third["complete"] is True
    state = json.loads(run.game_path("candidate", "u0", 0).read_text())
    assert state["usage"]["total_tokens"] == 120 and state["status"] == "valid"
    assert state["request"]["messages"] == rows[0]["games"][0]["prompt"]


def test_pilot_records_reused_when_uid_subset_expands(tmp_path, inputs, protocol):
    relay = FakeRelay()
    make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0"], budget_cny=5)
    make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0", "u1"], budget_cny=5)
    assert len(relay.calls) == 4
    assert len((tmp_path / "candidate.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("kind", ["invalid", "ambiguous", "inflight"])
def test_failed_or_ambiguous_requests_never_automatically_repeat(tmp_path, inputs, protocol, kind):
    run = make_run(tmp_path, inputs, protocol)
    relay = FakeRelay(text="[[A==B]]" if kind == "invalid" else "[[A=B]]",
                      error=TimeoutError("secret API key must never be persisted") if kind == "ambiguous" else None)
    if kind == "inflight":
        run.prepare()
        run.save_inflight("candidate", "u0", 0)
    else:
        with pytest.raises(RuntimeError):
            run.run(lambda: relay, ["u0"], workers=1, budget_cny=5)
    before = len(relay.calls)
    with pytest.raises(RuntimeError, match="blocked"):
        make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0"], budget_cny=5)
    assert len(relay.calls) == before
    assert "secret API key" not in run.game_path("candidate", "u0", 0).read_text()


def test_changed_input_or_protocol_cannot_reuse_paid_records(tmp_path, inputs, protocol):
    make_run(tmp_path, inputs, protocol).run(lambda: FakeRelay(), ["u0"], budget_cny=5)
    altered = copy.deepcopy(inputs)
    altered[2]["candidate"][0]["messages"][-1]["content"]["answer"] += " altered"
    with pytest.raises(ValueError, match="identity"):
        make_run(tmp_path, altered, protocol).run(lambda: pytest.fail("no dispatch"), ["u0"], budget_cny=5)


def test_budget_is_persistent_across_resume_and_blocks_next_batch(tmp_path, inputs, protocol):
    relay = FakeRelay(usage_values=[10, 10, 12, 12])
    run = make_run(tmp_path, inputs, protocol)
    run.run(lambda: relay, ["u0"], workers=1, budget_cny=5, max_requests=1)
    with pytest.raises(RuntimeError, match="budget"):
        make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0"], budget_cny=1)
    assert len(relay.calls) == 1


def test_usage_failure_sends_no_paid_request(tmp_path, inputs, protocol):
    relay = FakeRelay(usage_values=[float("nan")])
    with pytest.raises(RuntimeError, match="usage"):
        make_run(tmp_path, inputs, protocol).run(lambda: relay, ["u0"], budget_cny=5)
    assert relay.calls == []


def test_per_request_record_exists_before_transport(tmp_path, inputs, protocol):
    run = make_run(tmp_path, inputs, protocol)
    class CheckedRelay(FakeRelay):
        def judge_call(self, request):
            matching = [json.loads(p.read_text()) for p in (tmp_path / "state/games/candidate").glob("*.json")]
            assert any(r["status"] == "inflight" and r["request"] == request for r in matching)
            return super().judge_call(request)
    relay = CheckedRelay()
    run.run(lambda: relay, ["u0"], workers=2, budget_cny=5)
    assert len(relay.calls) == 2


def test_truncated_judgment_is_invalid_even_with_verdict(tmp_path, inputs, protocol):
    class Truncated(FakeRelay):
        def judge_call(self, request):
            result = super().judge_call(request)
            result["finish_reason"] = "length"
            return result
    with pytest.raises(RuntimeError, match="blocked"):
        make_run(tmp_path, inputs, protocol).run(lambda: Truncated(), ["u0"], workers=1, budget_cny=5)


def test_jsonl_unicode_line_separators_are_not_record_boundaries(tmp_path):
    path = tmp_path / "unicode.jsonl"
    rows = [{"prompt": "before\u2028middle\u2029after\u0085end"}, {"prompt": "next"}]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    assert judge.load_jsonl(path) == rows


def test_tag_must_bind_actual_model_name(tmp_path, inputs, protocol):
    altered = copy.deepcopy(inputs)
    for row in altered[2]["candidate"]:
        row["model"] = "another_policy"
    with pytest.raises(ValueError, match="model identity"):
        make_run(tmp_path, altered, protocol)


def test_explicit_one_game_retry_archives_prior_attempt_without_repeating_other_game(tmp_path, inputs, protocol):
    run = make_run(tmp_path, inputs, protocol)
    bad = FakeRelay(text="invalid verdict")
    with pytest.raises(RuntimeError):
        run.run(lambda: bad, ["u0"], workers=1, budget_cny=5)
    previous = run.game_path("candidate", "u0", 0).read_bytes()
    good = FakeRelay()
    make_run(tmp_path, inputs, protocol).run(lambda: good, ["u0"], workers=1, budget_cny=5,
                                           retry_games=[("candidate", "u0", 0)])
    assert len(good.calls) == 2
    archives = list((tmp_path / "state/attempts/candidate/u0-0").glob("*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == previous
    current = json.loads(run.game_path("candidate", "u0", 0).read_text())
    assert current["attempt"] == 1 and current["status"] == "valid"
    assert current["supersedes_local_request_id"] == json.loads(previous)["local_request_id"]
    with pytest.raises(ValueError, match="retry"):
        make_run(tmp_path, inputs, protocol).run(lambda: pytest.fail("no network"), ["u0"], budget_cny=5,
                                               retry_games=[("candidate", "u0", 0)])


def test_failed_explicit_retry_cannot_repeat_again(tmp_path, inputs, protocol):
    bad = FakeRelay(text="invalid")
    with pytest.raises(RuntimeError):
        make_run(tmp_path, inputs, protocol).run(lambda: bad, ["u0"], workers=1, budget_cny=5)
    with pytest.raises(RuntimeError):
        make_run(tmp_path, inputs, protocol).run(lambda: bad, ["u0"], workers=1, budget_cny=5,
                                               retry_games=[("candidate", "u0", 0)])
    with pytest.raises(ValueError, match="retry"):
        make_run(tmp_path, inputs, protocol).run(lambda: pytest.fail("no network"), ["u0"], budget_cny=5,
                                               retry_games=[("candidate", "u0", 0)])
    assert len(bad.calls) == 2


def test_retry_preserves_successful_other_order_without_paid_duplicate(tmp_path, inputs, protocol):
    class OnceBad(FakeRelay):
        def judge_call(self, request):
            result = super().judge_call(request)
            if len(self.calls) == 2:
                result["answer"] = "invalid"
            return result
    run = make_run(tmp_path, inputs, protocol)
    relay = OnceBad()
    with pytest.raises(RuntimeError):
        run.run(lambda: relay, ["u0"], workers=1, budget_cny=5)
    first_bytes = run.game_path("candidate", "u0", 0).read_bytes()
    good = FakeRelay()
    make_run(tmp_path, inputs, protocol).run(lambda: good, ["u0"], workers=1, budget_cny=5,
                                           retry_games=[("candidate", "u0", 1)])
    assert len(good.calls) == 1
    assert run.game_path("candidate", "u0", 0).read_bytes() == first_bytes


def test_exclusive_process_lock_prevents_double_dispatch(tmp_path, inputs, protocol):
    with judge.exclusive_lock(tmp_path):
        with pytest.raises(RuntimeError, match="holds"):
            make_run(tmp_path, inputs, protocol).run(lambda: pytest.fail("must not dispatch"),
                                                   ["u0"], budget_cny=5)


def test_real_relay_transport_has_one_post_and_saves_response_provenance(monkeypatch, protocol, inputs):
    calls = []
    class Client:
        def __init__(self, **_): pass
        def post(self, url, json):
            calls.append((url, json))
            return SimpleNamespace(
                raise_for_status=lambda: None, headers={"x-request-id": "provider-request"},
                json=lambda: {"id": "chat-response", "model": "gpt-4.1-2025-04-14",
                              "choices": [{"message": {"content": "[[A=B]]"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=Client, Limits=lambda **_: None))
    relay = judge.Relay("fake-not-a-real-key")
    request = judge.make_request(inputs[0][0], inputs[1][0], inputs[2]["candidate"][0], 0, protocol)
    response = relay.judge_call(request)
    assert len(calls) == 1 and calls[0][0] == judge.BASE_URL + "/chat/completions"
    assert response["response_id"] == "chat-response"
    assert response["provider_request_id"] == "provider-request"
    assert response["response_model"] == "gpt-4.1-2025-04-14"
    def failure(*_, **__):
        calls.append("failed")
        raise TimeoutError("transport interruption")
    relay.client.post = failure
    with pytest.raises(TimeoutError):
        relay.judge_call(request)
    assert calls.count("failed") == 1
