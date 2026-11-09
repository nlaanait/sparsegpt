import copy
import pytest
from sparsifier.eval.gsm8k_groups import matched_rows, summarize_matrix


def fixture():
    outcomes = [{'match_id':0,'kind':kind,'compensation':mode,'entries':[entry],
                 'ambient_cost_sum':1.} for kind,entry in [('high_geometry',1),('low_geometry',2)]
                for mode in ('none','obs')]
    return {'outcomes':outcomes},{'group_pairs':1,'group_size':1}


def test_incomplete_groups_do_not_yield_selected_subset_result():
    r,c = fixture()
    assert len(matched_rows(r,c)) == 4
    r['outcomes'].pop()
    assert matched_rows(r,c) is None


def test_rejects_membership_and_ambient_mismatches():
    r,c = fixture(); r['outcomes'][1]['entries'] = [3]
    with pytest.raises(ValueError,match='membership'): matched_rows(r,c)
    r,c = fixture(); r['outcomes'][0]['ambient_cost_sum'] = 2.
    with pytest.raises(ValueError,match='tolerance'): matched_rows(r,c)
    r,c = fixture(); r['outcomes'].append(copy.deepcopy(r['outcomes'][0]))
    with pytest.raises(ValueError,match='duplicate'): matched_rows(r,c)


def test_paired_outcome_sign_and_dimension():
    assert summarize_matrix([[2,2],[2,2]],['a','b'])['mean'] == 2
    with pytest.raises(ValueError,match='matrix'): summarize_matrix([[1,2]],['a'])
