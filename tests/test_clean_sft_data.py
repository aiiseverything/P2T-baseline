import pytest

from scripts.clean_sft_data import strip_confidence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer\nConfidence: 95%", "Answer"),
        ("Answer\nConfidence level: 0.95", "Answer"),
        ("Answer\n[Confidence: 90%]", "Answer"),
        ("Answer\n*Confidence: 90%*", "Answer"),
        ("Answer\n**Confidence: 90%**", "Answer"),
        ("Answer\n**Confidence:** 100%", "Answer"),
        ("Answer\nConfidence: 95%.", "Answer"),
        ("Answer\nConfidence: 0.95.", "Answer"),
        ("Answer\nConfidence: 95.", "Answer"),
        ("First\n**Confidence:** 100%\nSecond", "First\nSecond"),
    ],
)
def test_strip_confidence_removes_numeric_annotations_in_supported_formats(text, expected):
    assert strip_confidence(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Confidence:",
        "Public trust and confidence: Electoral reforms matter.",
        "The answer reflects confidence: high after checking.",
        "**Confidence:** unknown",
        "Confidence: 90th percentile is a threshold label.",
        "The chart reports Confidence: 95% for the proposed model.",
    ],
)
def test_strip_confidence_preserves_non_numeric_prose_and_labels(text):
    assert strip_confidence(text) == text
