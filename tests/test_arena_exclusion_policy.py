import copy

import pytest

from scripts import arena_exclusion_policy as policy
from scripts import judge_arena_hard as judge


def record(answer="[[A>B]]", finish="stop", status="valid"):
    return {"status": status, "answer": answer, "finish_reason": finish,
            "score": judge.parse_score(answer), "finished_at": "2026-09-17T16:00:00Z",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def test_only_completed_judge_format_failures_can_be_excluded():
    assert policy.classify_record(None, judge) == "missing"
    assert policy.classify_record(record(), judge) == "valid"
    missing = record("Here is how to revert a Git commit.", status="invalid")
    assert policy.classify_record(missing, judge) == "judge_failed"
    assert policy.judge_failure_reason(missing, judge) == "missing_verdict"
    truncated = record("[[A>B]]", finish="length", status="invalid")
    assert policy.classify_record(truncated, judge) == "judge_failed"
    assert policy.judge_failure_reason(truncated, judge) == "truncated_judge_response"


@pytest.mark.parametrize("change", [
    {"status": "ambiguous"}, {"status": "inflight"},
    {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 2}},
    {"answer": None}, {"score": "A>B"}, {"finished_at": None},
    {"finish_reason": "unknown"},
])
def test_transport_or_corrupt_failures_are_never_silently_excluded(change):
    row = record("No verdict", status="invalid")
    row.update(copy.deepcopy(change))
    assert policy.classify_record(row, judge) == "blocked"


def test_invalid_status_cannot_discard_a_valid_decision():
    assert policy.classify_record(record(status="invalid"), judge) == "blocked"
    row = record("No verdict")
    assert policy.classify_record(row, judge) == "blocked"


def test_policy_must_bind_requested_judge_and_reference():
    requested = {"policy_id": "arena_judge_output_exclusions_v2", "judge": "gpt-4o",
                 "baseline": "gpt-4o-mini-2024-07-18"}
    assert policy.validate_policy(requested) == requested
    for key in requested:
        wrong = dict(requested, **{key: "different"})
        with pytest.raises(ValueError):
            policy.validate_policy(wrong)


def test_completed_filtered_null_response_is_excluded_with_distinct_reason():
    row = record('No verdict', finish='content_filter', status='invalid')
    row.update(answer=None, score=None)
    assert policy.classify_record(row, judge) == 'judge_failed'
    assert policy.judge_failure_reason(row, judge) == 'judge_content_filter'


@pytest.mark.parametrize('change', [
    {'status': 'valid'}, {'status': 'ambiguous'}, {'status': 'inflight'},
    {'answer': '[[A>B]]'}, {'answer': ''}, {'score': 'A>B'},
    {'finished_at': None}, {'usage': None},
    {'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 14}},
    {'finish_reason': 'stop'}, {'finish_reason': 'unknown'},
])
def test_content_filter_exception_cannot_admit_inconsistent_or_ambiguous_records(change):
    row = record('No verdict', finish='content_filter', status='invalid')
    row.update(answer=None, score=None)
    row.update(change)
    assert policy.classify_record(row, judge) == 'blocked'


@pytest.mark.parametrize('key', ['answer', 'score'])
def test_filtered_record_must_explicitly_preserve_null_answer_and_score(key):
    row = record('No verdict', finish='content_filter', status='invalid')
    row.update(answer=None, score=None)
    del row[key]
    assert policy.classify_record(row, judge) == 'blocked'
