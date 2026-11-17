"""Stop-token handling must precede strict answer parsing."""
from scripts.diagnose_gsm8k_stop_tokens import summarize


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return ''.join({1: 'Final answer: 80\n', 2: 'Final answer: 9\n',
                        10: '', 11: '', 12: ''}[t] for t in tokens)


def test_chat_eos_prevents_post_answer_tokens_from_changing_grade():
    tokens = [1, 11, 2, 12]
    legacy = summarize(tokens, Tokenizer(), [10], 4, '80')
    corrected = summarize(tokens, Tokenizer(), [10, 11], 4, '80')
    assert legacy['truncated'] and not legacy['correct']
    assert corrected['correct'] and not corrected['truncated']
    assert corrected['generated_tokens'] == 2


def test_no_eos_keeps_budget_exhaustion_and_wrong_answer_rejection():
    missing = summarize([1, 12, 12, 12], Tokenizer(), [10, 11], 4, '80')
    wrong = summarize([2, 11, 12, 12], Tokenizer(), [10, 11], 4, '80')
    assert missing['truncated'] and not missing['correct']
    assert not wrong['truncated'] and not wrong['correct']


def test_original_eos_still_stops_and_counts_at_budget_boundary():
    result = summarize([1, 10], Tokenizer(), [10, 11], 2, '80')
    assert result['correct'] and not result['truncated']
