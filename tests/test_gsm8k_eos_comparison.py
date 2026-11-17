"""Reject checkpoint, batching, or result changes when isolating EOS effects."""
import json

import pytest

from scripts.compare_gsm8k_eos_evaluations import compare
from sparsifier.data.gsm8k import digest


@pytest.fixture
def pair(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir();new.mkdir()
    common={'evaluation_ids':['one'], 'source_config_sha256':'same-weights',
            'batching':{'batch_size':32}, 'manifest_sha256':'same-questions'}
    a=dict(common,dense_config_sha256='old-dense')
    b=dict(common,dense_config_sha256='new-dense',stopping_policy={'version':'gsm8k-model-and-tokenizer-eos-v1'})
    (old/'config.json').write_text(json.dumps(a));(new/'config.json').write_text(json.dumps(b))
    row={'id':'one','expected_answer':'8','answer_nll':1.25,
         'prompt_sha256':'same-rendered-prompt','input_tokens':2,
         'batch_ids':['one'],'batch_size':1,'batching':common['batching']}
    left=dict(row,config_sha256=digest(a),text='Final answer: 8\n\n\n',truncated=True,correct=False)
    right=dict(row,config_sha256=digest(b),text='Final answer: 8\n',truncated=False,correct=True)
    (old/'answers.jsonl').write_text(json.dumps(left)+'\n');(new/'answers.jsonl').write_text(json.dumps(right)+'\n')
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'config_sha256':digest(b),'count':1,
        'audited_records_sha256':digest([right])}))
    return old,new,audit


def test_matched_eos_rescue_has_unchanged_prefix_and_likelihood(pair):
    result=compare(*pair,count=1)
    assert result['old_correct']==0 and result['corrected_correct']==1
    assert result['decoded_prefix_matches']==1 and result['max_answer_nll_abs_difference']==0.
    assert result['gained_correct_ids']==['one'] and result['old_strict_marker_correct_ignoring_truncation']==1


@pytest.mark.parametrize('field,value',[('source_config_sha256','other-weights'),('batching',{'batch_size':8})])
def test_checkpoint_or_batch_change_cannot_be_called_an_eos_effect(pair,field,value):
    old,new,audit=pair;path=new/'config.json';config=json.loads(path.read_text());config[field]=value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='EOS is not isolated'):compare(old,new,audit,count=1)


def test_result_not_covered_by_raw_audit_is_rejected(pair):
    old,new,audit=pair;path=new/'answers.jsonl';row=json.loads(path.read_text());row['text']='Final answer: 9\n'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):compare(old,new,audit,count=1)


@pytest.mark.parametrize('field,value',[('prompt_sha256','different-prompt'),('batch_ids',['one','other'])])
def test_changed_recorded_inputs_are_rejected_even_with_matching_config_and_audit(pair,field,value):
    old,new,audit=pair;path=new/'answers.jsonl';row=json.loads(path.read_text());row[field]=value
    path.write_text(json.dumps(row)+'\n')
    verified=json.loads(audit.read_text());verified['audited_records_sha256']=digest([row])
    audit.write_text(json.dumps(verified))
    with pytest.raises(ValueError,match='recorded generation inputs differ'):
        compare(old,new,audit,count=1)
