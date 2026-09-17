"""Small objective canaries for immediate corruption, independent of RM scores.

This is a launch check, not an evaluation benchmark or evidence that long RL
improves quality. Prompts and acceptance rules are frozen before GPU execution.
"""
import re
import unicodedata

CONTROLS = [
    {'id': 'addition', 'prompt': 'What is 2 + 2? Reply with only the number.', 'answer': r'4|four'},
    {'id': 'multiplication', 'prompt': 'What is 5 multiplied by 3? Reply with only the number.', 'answer': r'15|fifteen'},
    {'id': 'capital', 'prompt': 'What is the capital of France? Reply with only the city name.', 'answer': r'(?:the capital of france is )?paris'},
    {'id': 'water', 'prompt': 'What is the chemical formula of water? Reply with only the formula.', 'answer': r'(?:the chemical formula (?:of|for) water is )?h2o'},
    {'id': 'comparison', 'prompt': 'Which number is larger: 9 or 3? Reply with only the larger number.', 'answer': r'(?:the larger number is )?(?:9|nine)(?: is larger)?'},
    {'id': 'week', 'prompt': 'How many days are in one week? Reply with only the number.', 'answer': r'(?:there are )?(?:7|seven)(?: days(?: in (?:a|one) week)?)?'},
    {'id': 'alphabet', 'prompt': 'What letter immediately precedes B in the English alphabet? Reply with only the letter.', 'answer': r'(?:the letter is )?a'},
    {'id': 'color', 'prompt': 'In the traditional paint color wheel, what secondary color is made by mixing red and blue? Reply with only the color name.', 'answer': r'purple|violet'},
]


def evaluate_quality(responses, baseline_score=None):
    if len(responses) != len(CONTROLS) or any(not isinstance(x, str) for x in responses):
        raise ValueError('Expected exactly eight textual quality-control responses')
    if baseline_score is not None and (type(baseline_score) is not int or not 6 <= baseline_score <= 8):
        raise ValueError('baseline_score must be a passing initial score between 6 and 8')
    rows = []
    for control, text in zip(CONTROLS, responses):
        clean = unicodedata.normalize('NFKC', text).casefold().strip()
        clean = re.sub(r'[*`$]', '', clean).strip(' .!\n\t')
        clean = re.sub(r'^(?:the answer is|answer:|it is)\s+', '', clean)
        empty = not text.strip()
        repeated = bool(re.search(r'(.{4,}?)\1{3,}', text, re.DOTALL)) or '\n' * 32 in text
        correct = re.fullmatch(control['answer'], clean) is not None
        rows.append({'id': control['id'], 'prompt': control['prompt'], 'response': text,
                     'correct': correct, 'empty': empty, 'repeated': repeated})
    score = sum(row['correct'] for row in rows)
    minimum = 6 if baseline_score is None else baseline_score - 1
    empty_count = sum(row['empty'] for row in rows)
    repeated_count = sum(row['repeated'] for row in rows)
    return {'passed': score >= minimum and empty_count == 0 and repeated_count == 0,
            'score': score, 'minimum_score': minimum, 'baseline_score': baseline_score,
            'empty_count': empty_count, 'repeated_count': repeated_count, 'rows': rows,
            'protocol': 'eight_objective_canaries_v1'}
