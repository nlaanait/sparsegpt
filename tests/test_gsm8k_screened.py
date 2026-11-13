import json
import pytest
from sparsifier.eval.gsm8k import read_completed
from sparsifier.eval.gsm8k_screened import decision, screened_behavior, ScreenRejected


def rows():
    return [{'id':f'id-{i}', 'family':f'id-{i}'} for i in range(512)]


def record(row, correct=True):
    return {'id':row['id'], 'config_sha256':'candidate', 'correct':correct,
            'answer_nll':1., 'truncated':False}


def test_threshold_uses_matched_problem_counts():
    data=rows(); baseline={r['id']:record(r) for r in data}
    candidate={r['id']:record(r, i>=19) for i,r in enumerate(data[:128])}
    assert decision(data,baseline,candidate,'dense','candidate')['passed']
    candidate[data[19]['id']]['correct']=False
    assert not decision(data,baseline,candidate,'dense','candidate')['passed']


def test_promoted_screen_reuses_answers_and_continues_same_call(tmp_path):
    data=rows(); baseline={r['id']:record(r) for r in data}; calls=[]
    # Simulate an interrupted evaluation with some saved first-stage answers.
    with (tmp_path/'answers.jsonl').open('w') as out:
        for row in data[:50]: out.write(json.dumps(record(row))+'\n')
    def fake_behavior(model, tokenizer, chosen, output, config_hash, *, max_new_tokens):
        saved=read_completed(output/'answers.jsonl',config_hash)
        with (output/'answers.jsonl').open('a') as out:
            for row in chosen:
                if row['id'] not in saved:
                    calls.append(row['id']); out.write(json.dumps(record(row))+'\n')
        return {'count':len(chosen)}
    result=screened_behavior(fake_behavior,data,baseline,'dense',None,None,tmp_path,'candidate',max_new_tokens=2048)
    assert result['count']==512
    assert len(calls)==462 and len(set(calls))==462
    assert len(read_completed(tmp_path/'answers.jsonl','candidate'))==512
    assert json.loads((tmp_path/'screen.json').read_text())['passed']


def test_rejected_screen_keeps_existing_answers_and_never_evaluates_remaining(tmp_path):
    data=rows(); baseline={r['id']:record(r) for r in data}
    # Already-completed answers beyond the screen must not be discarded.
    with (tmp_path/'answers.jsonl').open('w') as out:
        for row in data[:150]: out.write(json.dumps(record(row,False))+'\n')
    def fail_if_called(*args,**kwargs): pytest.fail('no more answers should be generated')
    with pytest.raises(ScreenRejected):
        screened_behavior(fail_if_called,data,baseline,'dense',None,None,tmp_path,'candidate',max_new_tokens=2048)
    assert len(read_completed(tmp_path/'answers.jsonl','candidate'))==150
    assert not json.loads((tmp_path/'screen.json').read_text())['passed']
