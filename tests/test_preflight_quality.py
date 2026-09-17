import pytest
from scripts.preflight_quality import CONTROLS, evaluate_quality


def test_objective_answers_accept_common_short_sentence_format():
    answers=['The answer is 4.', '15', 'Paris.', 'H₂O', '9', 'There are seven days in a week.', 'A', 'Purple.']
    result=evaluate_quality(answers)
    assert result['passed'] and result['score']==8 and len(result['rows'])==len(CONTROLS)==8


def test_rm_style_off_topic_salad_never_passes_quality_gate():
    result=evaluate_quality(['Hello there! Let us talk about geometry.']*8)
    assert not result['passed'] and result['score']==0


def test_comparison_checks_regression_and_empty_outputs():
    answers=['4','15','Paris','H2O','9','seven','A','purple']
    answers[0]='5'; answers[1]='16'
    assert evaluate_quality(answers)['passed']
    assert not evaluate_quality(answers,baseline_score=8)['passed']
    answers[0]=''
    result=evaluate_quality(answers,baseline_score=6)
    assert not result['passed'] and result['empty_count']==1


def test_wrong_answer_with_correct_word_quoted_is_not_accepted():
    answers=['It is not 4; the answer is 5.','15','Paris','H2O','9','seven','A','purple']
    assert evaluate_quality(answers)['score']==7
    with pytest.raises(ValueError): evaluate_quality(answers[:-1])
