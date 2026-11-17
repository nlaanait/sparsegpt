"""Verify corrected stopping with the real parser, plus provenance guards."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.run_gsm8k_eos_control import corrected_behavior, install, stopping_policy
from sparsifier.data.gsm8k import digest, write_immutable
import sparsifier.eval.gsm8k_pruning as evaluator


class Tokenizer:
    eos_token_id = 248046
    pad_token_id = 248044

    def apply_chat_template(self, *args, **kwargs):
        return 'prompt'

    def encode(self, *args, **kwargs):
        return [3, 4]

    def decode(self, tokens, **kwargs):
        return ''.join({1:'Final answer: 80\n', 2:'Final answer: 9\n',
                        248046:'', 248044:''}[t] for t in tokens)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=248044)

    def generate(self, input_ids, **kwargs):
        ids = self.generation_config.eos_token_id
        tokens = [1, 248046] if isinstance(ids, list) and 248046 in ids else [1, 248046, 2, 2]
        return torch.cat([input_ids, torch.tensor([tokens])], dim=1)


def test_corrected_eos_gets_strict_credit_and_persists_raw_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, 'answer_likelihoods', lambda *args: [0.])
    model = Model()
    original = model.generation_config
    row = {'id':'one', 'question':'q', 'family':'one', 'answer':'80'}
    result = corrected_behavior(evaluator.behavior, model, Tokenizer(), [row],
        tmp_path, 'hash', raw_output=tmp_path/'raw', max_new_tokens=4)
    answer = json.loads((tmp_path/'answers.jsonl').read_text())
    assert result['accuracy'] == 1 and not answer['truncated']
    raw = json.loads(next((tmp_path/'raw').glob('*.json')).read_text())
    assert raw['batch_ids'] == ['one'] and raw['generated_token_ids'] == [[1,248046]]
    assert model.generation_config is original


def test_restore_model_settings_on_generation_failure(tmp_path):
    model = Model()
    original = model.generation_config
    def fail(*args, **kwargs):
        raise RuntimeError('generation failed')
    with pytest.raises(RuntimeError, match='generation failed'):
        corrected_behavior(fail, model, Tokenizer(), [], tmp_path, 'hash', raw_output=tmp_path/'raw')
    assert model.generation_config is original


def test_policy_tag_is_hashed_and_historical_dense_is_rejected(tmp_path):
    runner = SimpleNamespace(write_immutable=write_immutable, behavior=lambda *a: None,
                             dense_answers=lambda *a: 'ok')
    install(runner, tmp_path)
    config = {'arm':'dense'}
    old_hash = digest(config)
    runner.write_immutable(tmp_path/'config.json', config)
    assert config['stopping_policy'] == stopping_policy() and digest(config) != old_hash
    assert runner.dense_answers(tmp_path) == 'ok'
    (tmp_path/'config.json').write_text(json.dumps({'arm':'dense'}))
    with pytest.raises(ValueError, match='matching corrected EOS dense'):
        runner.dense_answers(tmp_path)


def test_changed_pinned_eos_fails_before_generation(tmp_path):
    model = Model()
    model.generation_config.eos_token_id = 5
    with pytest.raises(ValueError, match='EOS changed'):
        corrected_behavior(None, model, Tokenizer(), [], tmp_path, 'hash', raw_output=tmp_path/'raw')
