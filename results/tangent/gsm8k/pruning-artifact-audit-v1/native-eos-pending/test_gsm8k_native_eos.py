"""Native EOS regression using actual Qwen generation and stale-cache rejection."""
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from sparsifier.data.gsm8k import digest
from sparsifier.eval import gsm8k_pruning as evaluator
from sparsifier.eval.gsm8k_batching import generation_eos, policy, reference_backend


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [5] if text.startswith(' ') else [3, 4]

    def decode(self, tokens, **kwargs):
        return ''.join('Final answer: 80\n' if t == 5 else '' for t in tokens)


@pytest.mark.parametrize('budget', [2, 4])
def test_actual_qwen_stops_at_tokenizer_eos_without_mutating_model(budget, tmp_path, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling, name, getattr(modeling, name))
    reference_backend()
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=['linear_attention', 'full_attention'],
        eos_token_id=1, pad_token_id=0,
        rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':.5})
    config._attn_implementation = 'sdpa'
    torch.manual_seed(704)
    model = modeling.Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    class AnswerThenChatEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            token = 5 if input_ids.shape[1] == 2 else 2
            scores.fill_(-torch.inf)
            scores[:, token] = 0.
            return scores
    original_generate = model.generate
    def generate(**kwargs):
        return original_generate(**kwargs, logits_processor=LogitsProcessorList([AnswerThenChatEOS()]))
    monkeypatch.setattr(model, 'generate', generate)
    with torch.inference_mode():
        old = generate(input_ids=torch.tensor([[3,4]]), attention_mask=torch.ones(1,2,dtype=torch.long),
                       max_new_tokens=4, do_sample=False, eos_token_id=[1], pad_token_id=0)
    assert old[0, 2:].tolist() == [5, 2, 2, 2]
    previous = model.generation_config
    result = evaluator.behavior(model, Tokenizer(),
        [{'id':'one','family':'test','question':'q','answer':'80'}],tmp_path,'new-config',max_new_tokens=budget)
    answer = json.loads((tmp_path/'answers.jsonl').read_text())
    assert result['accuracy'] == 1. and result['truncated'] == 0
    assert answer['generated_token_ids'] == [5,2] and answer['eos_token_ids'] == [1,2]
    assert answer['parsed_answer'] == '80' and answer['generated_tokens'] == 2
    assert model.generation_config is previous and previous.eos_token_id == 1


@pytest.mark.parametrize('model_eos,tokenizer_eos,expected', [
    (1,2,[1,2]), ([1,7],2,[1,2,7]), ([2,2],2,[2]), (None,2,[2]), (1,None,[1])])
def test_stopping_union_preserves_all_declared_ids(model_eos, tokenizer_eos, expected):
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos))
    assert generation_eos(model, SimpleNamespace(eos_token_id=tokenizer_eos)) == expected


@pytest.mark.parametrize('model_eos,tokenizer_eos', [(None,None), (True,2), ([-1],2)])
def test_invalid_stopping_configuration_is_rejected(model_eos,tokenizer_eos):
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos))
    with pytest.raises(ValueError):generation_eos(model,SimpleNamespace(eos_token_id=tokenizer_eos))


def test_execution_metadata_binds_behavior_parser_and_prompt():
    actual=policy();here=Path(evaluator.__file__)
    assert actual['stopping_policy']=='model_and_tokenizer_eos_v1'
    for key,path in [('behavior_source_sha256',here),('prompt_source_sha256',here.with_name('gsm8k.py')),
                     ('parser_source_sha256',here.parents[1]/'data/gsm8k.py')]:
        assert actual[key] == sha256(path.read_bytes()).hexdigest()


def test_dense_reference_rejects_cached_model_only_stopping(tmp_path):
    from sparsifier.eval.gsm8k import MODEL,REVISION
    from sparsifier.eval.gsm8k_decision import dense_answers
    historical_policy=dict(policy())
    for key in ['stopping_policy','behavior_source_sha256','prompt_source_sha256','parser_source_sha256']:
        historical_policy.pop(key)
    manifest={'manifest_sha256':'same-manifest','rows':[{'id':'one'}]}
    config={'arm':'dense','manifest_sha256':'same-manifest','evaluation_ids':['one'],
            'backend':{},'packages':{},'model':MODEL,'revision':REVISION,'max_new_tokens':2048,
            'batching':historical_policy}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'report.json').write_text(json.dumps({'complete':True,'config_sha256':digest(config)}))
    with pytest.raises(ValueError,match='protocol/runtime mismatch'):
        dense_answers(tmp_path,manifest,{}, {})


@pytest.mark.parametrize('module,manifest_name', [
    ('gsm8k_decision','decision64-light128-repcal-seed0-v1.json'),
    ('gsm8k_decision_full','decision128-full512-repcal-seed0-batch8-v1.json')])
def test_driver_rejects_stale_complete_report_before_loading_model(module,manifest_name,tmp_path,monkeypatch):
    import importlib
    import sys
    from sparsifier.eval import memory_guard
    runner=importlib.import_module('sparsifier.eval.'+module)
    data=Path(__file__).resolve().parents[1]/'results/tangent/gsm8k/data'
    monkeypatch.setattr(runner,'reference_backend',lambda:{})
    monkeypatch.setattr(runner,'package_versions',lambda:{})
    class BeforeModelLoading(Exception):pass
    def stop(*args):raise BeforeModelLoading
    monkeypatch.setattr(memory_guard,'configure_cuda',stop)
    monkeypatch.setattr(sys,'argv',[module,'--manifest',str(data/manifest_name),
        '--calibration-manifest',str(data/'representative-calibration-seed0-v1.json'),
        '--output',str(tmp_path)])
    with pytest.raises(BeforeModelLoading):runner.main()
    config_path=tmp_path/'config.json';config=json.loads(config_path.read_text())
    for key in ('stopping_policy','behavior_source_sha256','prompt_source_sha256','parser_source_sha256'):
        config['batching'].pop(key)
    config_path.write_text(json.dumps(config))
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps({'complete':True,'config_sha256':digest(config)}))
    before=[p.read_bytes() for p in (config_path,report_path)]
    with pytest.raises(ValueError,match='immutable artifact differs'):runner.main()
    assert [p.read_bytes() for p in (config_path,report_path)] == before


def legacy_screen_fixture(tmp_path, *, legacy_config=False, legacy_answers=False):
    from sparsifier.eval.gsm8k import MODEL,REVISION
    from sparsifier.data.gsm8k import write_immutable
    data=[{'id':f'gsm8k/train/{i}','family':'test','split':'validation'} for i in range(512)]
    manifest={'source_split':'train','rows':data};manifest['manifest_sha256']=digest(manifest)
    manifest_path=tmp_path/'manifest.json';write_immutable(manifest_path,manifest)
    execution=dict(policy());old=dict(execution)
    for key in ('stopping_policy','behavior_source_sha256','prompt_source_sha256','parser_source_sha256'):
        old.pop(key)
    reference=tmp_path/'reference';reference.mkdir()
    config={'model':MODEL,'revision':REVISION,'arm':'dense','manifest_sha256':manifest['manifest_sha256'],
            'evaluation_ids':[r['id'] for r in data],'max_new_tokens':2048,
            'batching':old if legacy_config else execution}
    write_immutable(reference/'config.json',config)
    write_immutable(reference/'report.json',{'complete':True,'config_sha256':digest(config),'behavior':{'count':512}})
    answers=[{'id':r['id'],'config_sha256':digest(config),'correct':True,
              'batching':old if legacy_answers else execution} for r in data]
    (reference/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in answers))
    return manifest_path,reference,data,execution,old


@pytest.mark.parametrize('legacy_config,legacy_answers',[(True,False),(False,True)])
def test_legacy_screen_reference_rejects_model_only_EOS_cache(tmp_path,legacy_config,legacy_answers):
    from sparsifier.eval.gsm8k_screened import prepare_screen
    manifest,reference,*_=legacy_screen_fixture(tmp_path,legacy_config=legacy_config,legacy_answers=legacy_answers)
    with pytest.raises(ValueError,match='protocol|execution policy'):
        prepare_screen(manifest,reference,tmp_path,2048)


def test_legacy_screen_accepts_current_reference(tmp_path):
    from sparsifier.eval.gsm8k_screened import prepare_screen
    manifest,reference,rows,*_=legacy_screen_fixture(tmp_path)
    actual,baseline,_=prepare_screen(manifest,reference,tmp_path,2048)
    assert actual==rows and len(baseline)==512


@pytest.mark.parametrize('legacy_config,legacy_answers',[(True,False),(False,True)])
def test_legacy_screen_candidate_rejects_cached_false_zero_before_return(tmp_path,monkeypatch,legacy_config,legacy_answers):
    import sys
    from sparsifier.eval import gsm8k_screened as screened,gsm8k_original
    from sparsifier.eval.gsm8k import MODEL,REVISION
    from sparsifier.data.gsm8k import write_immutable
    manifest,reference,rows,execution,old=legacy_screen_fixture(tmp_path)
    source=tmp_path/'source';source.mkdir();source_cfg={'arm':'local'}
    write_immutable(source/'config.json',source_cfg)
    write_immutable(source/'report.json',{'complete':True,'config_sha256':digest(source_cfg),'stages':[]})
    out=tmp_path/'candidate';out.mkdir();config={'model':MODEL,'revision':REVISION,
        'manifest_sha256':json.loads(manifest.read_text())['manifest_sha256'],
        'evaluation_ids':[r['id'] for r in rows],'max_new_tokens':2048,
        'source_config_sha256':digest(source_cfg),'candidate_chain':[],
        'source_sha256':sha256(Path(gsm8k_original.__file__).read_bytes()).hexdigest(),
        'batching':old if legacy_config else execution}
    write_immutable(out/'config.json',config)
    (out/'answers.jsonl').write_text(''.join(json.dumps({'id':r['id'],'config_sha256':digest(config),
        'correct':False,'batching':old if legacy_answers else execution})+'\n' for r in rows[:128]))
    before=[p.read_bytes() for p in (out/'config.json',out/'answers.jsonl')]
    monkeypatch.setattr(sys,'argv',['screened','--manifest',str(manifest),'--source',str(source),
        '--output',str(out),'--screen-reference',str(reference)])
    monkeypatch.setattr(gsm8k_original,'main',lambda:pytest.fail('stale screen must fail before model loading'))
    with pytest.raises(ValueError,match='provenance|execution policy'):screened.main()
    assert [p.read_bytes() for p in (out/'config.json',out/'answers.jsonl')]==before
    assert not (out/'screen.json').exists()


def test_legacy_screen_behavior_rejects_old_rows_even_with_matching_outer_hash(tmp_path):
    from sparsifier.eval.gsm8k_screened import screened_behavior
    _,_,rows,execution,old=legacy_screen_fixture(tmp_path)
    baseline={r['id']:{'correct':True,'batching':execution} for r in rows}
    (tmp_path/'answers.jsonl').write_text(''.join(json.dumps({'id':r['id'],'config_sha256':'same',
        'correct':False,'batching':old})+'\n' for r in rows[:128]))
    with pytest.raises(ValueError,match='execution policy'):
        screened_behavior(lambda *a:pytest.fail('must reject before generation'),rows,baseline,'dense',
            None,None,tmp_path,'same',max_new_tokens=2048)
