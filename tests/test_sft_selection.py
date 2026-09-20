import json

import pytest

from scripts import sft_init


def selection(tmp_path, rows):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(rows))
    return path


def test_frozen_selection_retains_original_order_and_entire_answer(tmp_path):
    path = selection(tmp_path, [
        {"prompt": "second", "response": "b" * 64 + "real tail"},
        {"prompt": "first", "response": "a"},
    ])
    pool = [("first", "a"), ("second", "b" * 64 + "real tail"), ("third", "c")]
    assert sft_init.load_pair_selection(path, pool) == [pool[1], pool[0]]


@pytest.mark.parametrize("row", [
    {"prompt": "held-out", "response": "b"},
    {"prompt": "train", "response": "a" * 64 + "wrong tail"},
])
def test_selection_cannot_bypass_clean_training_membership(tmp_path, row):
    path = selection(tmp_path, [row])
    with pytest.raises(ValueError, match="training pair"):
        sft_init.load_pair_selection(path, [("train", "a" * 64 + "correct tail")])


@pytest.mark.parametrize("rows", [
    [],
    [{"prompt": "q", "response": "a"}, {"prompt": "q", "response": "a"}],
    [{"prompt": "q", "response": "a"}, {"prompt": " q ", "response": "b"}],
    [{"prompt": "q", "response": 42}],
    [{"prompt": "q"}],
])
def test_selection_rejects_empty_malformed_or_duplicate_prompts(tmp_path, rows):
    path = selection(tmp_path, rows)
    with pytest.raises(ValueError):
        sft_init.load_pair_selection(path, [("q", "a"), (" q ", "b")])
