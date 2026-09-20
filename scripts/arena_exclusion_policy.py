"""Identify only completed, unscoreable judge outputs for explicit exclusion."""

POLICY_ID = "arena_judge_output_exclusions_v2"


def validate_policy(policy):
    expected = {"policy_id": POLICY_ID, "judge": "gpt-4o",
                "baseline": "gpt-4o-mini-2024-07-18"}
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError("Exclusion policy must identify the authorized judge and reference")
    return policy


def classify_record(record, judge):
    """Missing calls remain pending; transport/corruption never become exclusions."""
    if record is None:
        return "missing"
    if not isinstance(record, dict) or record.get("status") not in ("valid", "invalid"):
        return "blocked"
    if (not isinstance(record.get("finished_at"), str) or not record["finished_at"]
            or not judge.valid_usage(record.get("usage"))):
        return "blocked"
    if record.get("finish_reason") == "content_filter":
        # A completed provider-filtered response is terminal, not a transport
        # retry. Accept only the observed, explicit null-answer/null-score shape.
        return ("judge_failed" if record["status"] == "invalid"
                and "answer" in record and record["answer"] is None
                and "score" in record and record["score"] is None else "blocked")
    if (not isinstance(record.get("answer"), str)
            or record.get("finish_reason") not in ("stop", "length")):
        return "blocked"
    score = judge.parse_score(record["answer"])
    if record.get("score") != score:
        return "blocked"
    valid = score is not None and record["finish_reason"] == "stop"
    if record["status"] == "valid":
        return "valid" if valid else "blocked"
    return "blocked" if valid else "judge_failed"


def judge_failure_reason(record, judge):
    if classify_record(record, judge) != "judge_failed":
        raise ValueError("Record is not an excludable judge-output failure")
    if record["finish_reason"] == "content_filter":
        return "judge_content_filter"
    return ("truncated_judge_response" if record["finish_reason"] == "length"
            else "missing_verdict")
