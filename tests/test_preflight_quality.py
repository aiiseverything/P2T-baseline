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


def test_final_word_rule_accepts_verbose_correct_sentences_only():
    verbose = ['2', '15', 'The capital of France is Paris.', 'The chemical formula of water is H2O.', '9',
               'There are 7 days in one week.',
               'The letter immediately preceding B in the English alphabet is A.',
               'The secondary color made by mixing red and blue is purple.']
    strict = evaluate_quality(verbose)
    assert strict['score'] == 5 and not strict['passed']
    assert strict['rule'] == 'strict' and strict['protocol'] == 'eight_objective_canaries_v1'
    loose = evaluate_quality(verbose, rule='final_word')
    assert loose['score'] == 7 and loose['passed'] and loose['minimum_score'] == 6
    assert loose['rule'] == 'final_word' and loose['protocol'] == 'eight_objective_canaries_v2_final_word'
    assert [row['id'] for row in loose['rows'] if not row['correct']] == ['addition']
    wrong = list(verbose); wrong[2] = 'The capital of France is Paris, not Lyon.'
    assert evaluate_quality(wrong, rule='final_word')['score'] == 6
    assert not evaluate_quality(['Hello there! Let us talk about geometry.'] * 8, rule='final_word')['passed']
    assert evaluate_quality(verbose, baseline_score=8, rule='final_word')['passed']
    assert not evaluate_quality(wrong, baseline_score=8, rule='final_word')['passed']
    with pytest.raises(ValueError, match='rule'):
        evaluate_quality(verbose, rule='lenient')
