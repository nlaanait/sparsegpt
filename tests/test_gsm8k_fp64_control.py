"""The precision control must retain scaling, checkpoint, and provenance rules."""
import pytest
import torch

from scripts import run_gsm8k_fp64_control as control
from scripts.audit_gsm8k_solver_precision import reference_run
from sparsifier.prune.rescale import local_row_scales


def test_candidate_matches_reference_with_production_fp32_unscale_boundary():
    torch.manual_seed(712)
    w, x, z = torch.randn(7,16), torch.randn(9,16), torch.randn(4,16)
    pd = torch.rand(4,7)
    scale, _ = local_row_scales(x,z,pd,1.)
    expected, _ = reference_run(w*scale,x,.7)
    actual, d = control.fp64_candidate(w,x,arm='local',readout_inputs=z,pdiag=pd,device='cpu')
    torch.testing.assert_close(actual,expected.float()/scale,rtol=0,atol=0)
    assert actual.dtype == torch.float32 and actual.device.type == 'cpu'
    assert (actual==0).sum() == int(w.numel()*.7)
    assert d['normalization']=='curvature-trace-v2'
    assert d['added_curvature_trace_ratio']==pytest.approx(1.)
    assert d['selection_damping']['factorization_attempts']==1


def test_rejects_unmatched_arm():
    with pytest.raises(ValueError,match='unchanged local arm'):
        control.fp64_candidate(torch.ones(2,3),torch.ones(4,3),arm='sparsegpt_task',device='cpu')


def test_adapter_hashes_control_before_config_digest_and_restores_aliases(monkeypatch,tmp_path):
    from sparsifier.eval import gsm8k_sequential as runner
    from sparsifier.eval import gsm8k_batching
    from sparsifier.data.gsm8k import digest
    monkeypatch.setattr(gsm8k_batching,'policy',lambda:{'batch_size':32})
    config={'purpose':'development_sequential_MLP_retention','arm':'local','sparsity':.7}
    saved=[]
    def write(path,c):
        assert c is config
        assert len(c['solver_precision_control']['reference_sha256'])==64
        saved.append(digest(c))
    monkeypatch.setattr(runner,'write_immutable',write)
    original=runner.prune_candidate
    def run():
        assert runner.prune_candidate is control.fp64_candidate
        runner.write_immutable(tmp_path/'config.json',config)
        assert saved==[digest(config)]
        raise RuntimeError('test interrupted worker')
    monkeypatch.setattr(runner,'main',run)
    with pytest.raises(RuntimeError,match='interrupted worker'):
        control.main()
    assert runner.prune_candidate is original and runner.write_immutable is write
