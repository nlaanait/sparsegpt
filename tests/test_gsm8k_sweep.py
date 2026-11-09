import pytest
from sparsifier.eval.gsm8k_sweep import build_plan, smoke_gate


def test_matched_sweep_preserves_scope_and_rejects_leakage(tmp_path):
    cal = {'manifest_sha256':'a','rows':[{'source_index':2}]}
    val = {'manifest_sha256':'b','rows':[{'source_index':1001}]}
    plan = build_plan(cal,val)
    assert len(plan['tasks']) == 16
    names = {r['name'] for r in plan['tasks']}
    assert len(names) == 16
    for t in plan['tasks']:
        if t['phase'] == 'original_validation': assert t['depends_on'][0] in names
    assert plan['allowed_accuracy_drop'] is None
    assert not smoke_gate(tmp_path,'local')['ready']
    with pytest.raises(ValueError,match='leakage'): build_plan(cal,cal)


def test_submission_serializes_resources_and_preserves_dependencies(tmp_path):
    from scripts.submit_gsm8k_retention_sweep import command
    cmd = command(tmp_path,{'name':'qwen-9b-gsm8k-example'},[62,64])
    assert '--shared' not in cmd
    assert cmd[cmd.index('--gpus')+1] == '1'
    assert cmd[cmd.index('--gpu-memory')+1] == '64G'
    assert cmd[cmd.index('--memory')+1] == '80G'
    assert cmd[cmd.index('--depends-on-all')+1] == '62,64'
    assert str(tmp_path/'scripts/run_gsm8k_guarded.sh') in cmd


def test_smoke_gate_requires_complete_six_site_integration(tmp_path):
    import json
    from sparsifier.data.gsm8k import digest
    from sparsifier.eval.gsm8k_sweep import MODULES
    folder = tmp_path/'qwen-9b-gsm8k-sequential-task-smoke-v1'; folder.mkdir()
    config = {'arm':'sparsegpt_task'}
    (folder/'config.json').write_text(json.dumps(config))
    report = {'config_sha256':digest(config),'complete':False}
    (folder/'report.json').write_text(json.dumps(report))
    assert not smoke_gate(tmp_path,'sparsegpt_task')['ready']
    report.update(complete=True,stages=[{'site':[b,m]} for b in (15,16) for m in MODULES.split(',')],
                  behavior={'count':4},whole_model_removed_fraction=.01)
    (folder/'report.json').write_text(json.dumps(report))
    assert smoke_gate(tmp_path,'sparsegpt_task')['ready']
    report['stages'].pop()
    (folder/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='scope'): smoke_gate(tmp_path,'sparsegpt_task')


@pytest.mark.parametrize('normalization', [None, 'rank-v1', 'curvature-trace-v2'])
def test_worker_uses_corrected_default_and_rejects_legacy_plan(tmp_path, monkeypatch, normalization):
    import json
    import sys
    from sparsifier.data.gsm8k import digest
    from sparsifier.eval import gsm8k_sweep_worker as worker
    task = {'name':'local-test','phase':'prune','arm':'local','eligible_sparsity':.5}
    plan = {'tasks':[task], 'calibration_manifest_sha256':'cal',
            'validation_manifest_sha256':'val', 'modules':['mlp.down_proj'],
            'alpha':1., 'seed':0, 'max_new_tokens':2048}
    if normalization is not None:
        plan['local_normalization'] = normalization
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(worker, 'load_manifest', lambda path: {
        'manifest_sha256':'cal' if path.name == 'controlled-v1.json' else 'val'})
    monkeypatch.setattr(worker, 'smoke_gate', lambda *args: {'ready':True})
    invoked = []
    monkeypatch.setattr(worker.runpy, 'run_module', lambda module, **kwargs: invoked.append((module, sys.argv.copy())))
    monkeypatch.setattr(sys, 'argv', ['worker','--root',str(tmp_path),'--task','local-test',
                                    '--plan','plan.json','--plan-sha256',digest(plan)])
    if normalization == 'rank-v1':
        with pytest.raises(ValueError, match='no longer supported'):
            worker.main()
        assert not invoked
        return
    worker.main()
    module, argv = invoked[0]
    assert module == 'sparsifier.eval.gsm8k_sequential'
    assert '--local-normalization' not in argv


def test_local_smoke_gate_rejects_legacy_geometry(tmp_path):
    import json
    folder = tmp_path/'qwen-9b-gsm8k-sequential-local-smoke-v2'
    folder.mkdir()
    (folder/'config.json').write_text(json.dumps({'arm':'local'}))
    (folder/'report.json').write_text(json.dumps({'complete':True}))
    gate = smoke_gate(tmp_path, 'local')
    assert not gate['ready']
    assert 'legacy local smoke' in gate['reason']
