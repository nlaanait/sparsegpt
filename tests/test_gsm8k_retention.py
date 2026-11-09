import pytest
from sparsifier.eval.gsm8k_retention import retention, frontier


def rows(n):
    return {str(i):{'correct':True,'family':str(i),'expected_answer':'1'} for i in range(n)}


def test_small_perfect_sample_does_not_certify_retention():
    d = rows(32)
    r = retention(d,d,margin=.02,unit='problem')
    assert r['point_estimate_within_tolerance']
    assert not r['retention_established_under_assumptions']
    assert r['drop_upper_bound'] > .1
    d = rows(1000)
    assert retention(d,d,margin=.02,unit='problem')['retention_established_under_assumptions']


def test_family_dependence_and_losses_recoveries():
    d = rows(1000); a = rows(1000)
    for v in d.values(): v['family'] = 'one'
    for v in a.values(): v['family'] = 'one'
    a['0']['correct'] = False; d['1']['correct'] = False
    r = retention(a,d,margin=.02)
    assert r['dense_correct_lost'] == r['dense_wrong_recovered'] == 1
    assert r['accuracy_drop'] == 0 and r['drop_upper_bound'] == 1
    with pytest.raises(ValueError): retention({},d,margin=.02)


def test_frontier_uses_actual_global_sparsity_and_measured_points():
    def point(name,s,ok):
        return {'run':name,'whole_model_removed_fraction':s,'retention':{
            'point_estimate_within_tolerance':ok,'retention_established_under_assumptions':False}}
    result = frontier([point('unknown',None,True),point('low',.1,True),
                       point('middle',.3,False),point('high',.4,True)])
    assert result['largest_observed_within_tolerance'] == 'high'
    assert result['largest_statistically_supported'] is None
