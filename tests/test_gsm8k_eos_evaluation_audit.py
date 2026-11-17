"""Independent raw-token audit must catch corrupted text, gold, and provenance."""
import json

import pytest

from scripts.audit_gsm8k_eos_evaluation import audit
from sparsifier.data.gsm8k import digest


class Tokenizer:
    eos_token_id = 248046

    def decode(self, tokens, **kwargs):
        return ''.join({1:'Final answer: 80\n', 2:'Final answer: 9\n', 3:'\n',
                        248044:'', 248046:''}[t] for t in tokens)


@pytest.fixture
def saved(tmp_path):
    ids = [str(i) for i in range(128)]
    manifest = {'manifest_sha256':'manifest', 'rows':[{'id':i,'answer':'80'} for i in ids]}
    config = {'arm':'dense','evaluation_ids':ids,'manifest_sha256':'manifest',
              'max_new_tokens':4,'batching':{'batch_size':32},
              'stopping_policy':{'version':'gsm8k-model-and-tokenizer-eos-v1',
                                 'eos_token_ids':[248044,248046]}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'report.json').write_text(json.dumps({'config_sha256':digest(config),
        'behavior':{'count':128,'accuracy':1.,'truncated':0}}))
    raw_dir=tmp_path/'raw-generations';raw_dir.mkdir();answers=[]
    for start in range(0,128,32):
        batch=ids[start:start+32]
        raw={'config_sha256':digest(config),'batch_ids':batch,'eos_token_ids':[248044,248046],
             'max_new_tokens':4,'generated_token_ids':[[1,248046,2,248044] for i in batch]}
        (raw_dir/(digest(batch)+'.json')).write_text(json.dumps(raw))
        answers.extend({'id':i,'config_sha256':digest(config),'batch_ids':batch,'text':'Final answer: 80\n',
                        'generated_tokens':2,'truncated':False,'parsed_answer':'80',
                        'expected_answer':'80','correct':True} for i in batch)
    (tmp_path/'answers.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in answers))
    return tmp_path,manifest


def test_exact_chat_eos_rescore_with_post_eos_wrong_answer(saved):
    folder,manifest=saved
    result=audit(folder,Tokenizer(),manifest)
    assert result['common128']['correct']==128 and result['raw_token_rescore_mismatches']==0
    assert result['terminal_token_counts']=={'248046':128}


def test_changed_raw_answer_is_detected_even_when_saved_grade_is_unchanged(saved):
    folder,manifest=saved
    path=next((folder/'raw-generations').glob('*.json'));raw=json.loads(path.read_text())
    raw['generated_token_ids'][0][0]=2;path.write_text(json.dumps(raw))
    with pytest.raises(AssertionError):audit(folder,Tokenizer(),manifest)


def test_changed_saved_gold_is_detected_against_manifest(saved):
    folder,manifest=saved
    path=folder/'answers.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    rows[0]['expected_answer']='9';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):audit(folder,Tokenizer(),manifest)


def test_wrong_raw_configuration_is_detected(saved):
    folder,manifest=saved
    path=next((folder/'raw-generations').glob('*.json'));raw=json.loads(path.read_text())
    raw['config_sha256']='other';path.write_text(json.dumps(raw))
    with pytest.raises(AssertionError):audit(folder,Tokenizer(),manifest)


def test_remaining_truncation_censoring_is_reported_separately(saved):
    folder,manifest=saved
    for path in (folder/'raw-generations').glob('*.json'):
        raw=json.loads(path.read_text())
        raw['generated_token_ids']=[[1,3,3,3] for _ in raw['batch_ids']]
        path.write_text(json.dumps(raw))
    path=folder/'answers.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    for row in rows:
        row.update(text='Final answer: 80\n\n\n\n',generated_tokens=4,truncated=True,
                   parsed_answer=None,correct=False)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    path=folder/'report.json';report=json.loads(path.read_text())
    report['behavior'].update(accuracy=0.,truncated=128);path.write_text(json.dumps(report))
    result=audit(folder,Tokenizer(),manifest)
    assert result['common128']['correct']==0
    assert result['common128']['strict_marker_correct_ignoring_truncation']==128
    assert result['common128']['truncated_correct_with_complete_marker_line']==128


def test_live_prefix_is_explicitly_labeled_and_not_a_full_screen(saved):
    folder,manifest=saved
    (folder/'report.json').unlink()
    path=folder/'answers.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:32])+'\n'+ '{"id":')
    with pytest.raises(ValueError,match='completed evaluation report required'):
        audit(folder,Tokenizer(),manifest)
    result=audit(folder,Tokenizer(),manifest,allow_partial=True)
    assert result['common128']['count']==32 and result['common128']['correct']==32
    assert not result['common128_complete'] and result['report_sha256'] is None
    assert result['scope']=='live_prefix_not_final_evaluation'
