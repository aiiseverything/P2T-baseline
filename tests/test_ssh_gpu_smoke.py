import pytest


def test_smoke_accepts_native_stop_and_true_length_with_exact_support():
    from scripts.smoke_ssh_gpu import validate_rows
    validate_rows({"rows": [[1, 0], [1, 2, 1]], "finish_reasons": ["stop", "length"]},
                  {0, 1, 2}, {0}, 3)


@pytest.mark.parametrize("payload", [
    {"rows": [[1, 9]], "finish_reasons": ["stop"]},
    {"rows": [[1, 2]], "finish_reasons": ["stop"]},
    {"rows": [[1, 2]], "finish_reasons": ["length"]},
    {"rows": [[]], "finish_reasons": ["stop"]},
    {"rows": [[1, 0]], "finish_reasons": []},
])
def test_smoke_rejects_invalid_support_or_termination(payload):
    from scripts.smoke_ssh_gpu import validate_rows
    with pytest.raises(ValueError):
        validate_rows(payload, {0, 1, 2}, {0}, 3)
