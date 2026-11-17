"""Prevent evaluation leakage and silent truncation in the calibration control."""
import copy

import pytest

from scripts.prepare_gsm8k_reasoning_control import prepare_bundle
from sparsifier.data.gsm8k import digest
from test_gsm8k_calibration import dataset, prepare


class Tokenizer:
    eos_token_id = 99
    chat_template = 'test template'

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def test_reasoning_uses_only_frozen_calibration_parents_and_retains_eos():
    calibration, evaluation = prepare()
    bundle = prepare_bundle(dataset(),calibration,evaluation,Tokenizer(),'train-source')
    expected = [r for r in calibration['rows'] if r['split'] in ('geometry_fit','pruning_calibration')]
    assert [r['id'] for r in bundle['rows']] == [r['id'] for r in expected]
    assert not {r['id'] for r in bundle['rows']} & {r['id'] for r in evaluation['rows']}
    assert not {r['id'] for r in bundle['rows']} & {
        r['id'] for r in calibration['rows'] if r['split']=='validation'}
    for row in bundle['rows']:
        assert row['input_ids'][-1] == Tokenizer.eos_token_id
        assert row['prompt_readout_position'] == row['response_start']-1
        assert row['response_tokens'] == len(row['input_ids'])-row['response_start']
        assert row['response_sha256'] == digest('Reasoning\nFinal answer: 3')
    assert bundle['response_token_count'] == sum(r['response_tokens'] for r in bundle['rows'])
    assert digest({k:v for k,v in bundle.items() if k!='manifest_sha256'}) == bundle['manifest_sha256']


def test_source_mismatch_and_untrusted_parents_are_rejected():
    calibration, evaluation = prepare()
    with pytest.raises(ValueError,match='source hash'):
        prepare_bundle(dataset(),calibration,evaluation,Tokenizer(),'other-source')
    data = copy.deepcopy(dataset())
    data[calibration['rows'][0]['source_index']]['answer'] = '#### 7'
    with pytest.raises(ValueError,match='parent content'):
        prepare_bundle(data,calibration,evaluation,Tokenizer(),'train-source')


def test_calibration_never_silently_truncates_a_solution_or_omits_eos():
    calibration, evaluation = prepare()
    class LongTokenizer(Tokenizer):
        def encode(self,text,**kwargs):
            return [1]*2048
    with pytest.raises(ValueError,match='silently truncate'):
        prepare_bundle(dataset(),calibration,evaluation,LongTokenizer(),'train-source')
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError,match='EOS'):
        prepare_bundle(dataset(),calibration,evaluation,tokenizer,'train-source')
