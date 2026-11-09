import pytest
import torch
from sparsifier.eval.gsm8k_sequential import sites_for, verify_stage


def test_sites_follow_forward_order_and_reject_bad_scope():
    assert sites_for([0,2], ['mlp.down_proj','mlp.up_proj','mlp.gate_proj'], 3) == [
        (0,'mlp.gate_proj'),(0,'mlp.up_proj'),(0,'mlp.down_proj'),
        (2,'mlp.gate_proj'),(2,'mlp.up_proj'),(2,'mlp.down_proj')]
    for blocks in ([2,0], [0,0], [-1], [3], []):
        with pytest.raises(ValueError): sites_for(blocks,['mlp.down_proj'],3)
    with pytest.raises(ValueError): sites_for([0],['lm_head'],3)


def test_resume_requires_same_upstream_model_and_site():
    stored = {'config_sha256':'config','upstream':'previous-weights','site':[1,'mlp.down_proj'],
              'weight':torch.ones(3,4)}
    verify_stage(stored,'config','previous-weights',(1,'mlp.down_proj'))
    for config,upstream,site in [('other','previous-weights',(1,'mlp.down_proj')),
        ('config','dense',(1,'mlp.down_proj')),('config','previous-weights',(2,'mlp.down_proj'))]:
        with pytest.raises(ValueError,match='provenance'):
            verify_stage(stored,config,upstream,site)
    stored['weight'][0,0] = float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        verify_stage(stored,'config','previous-weights',(1,'mlp.down_proj'))


def test_calibration_refresh_observes_upstream_pruning():
    import numpy as np
    from sparsifier.eval.gsm8k_sequential import collect
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upstream = torch.nn.Linear(2,2,bias=False)
            self.site = torch.nn.Linear(2,2,bias=False)
            with torch.no_grad():
                self.upstream.weight.copy_(torch.eye(2))
                self.site.weight.copy_(torch.eye(2))
        def forward(self,input_ids,**kwargs):
            x = torch.stack([input_ids.float(),torch.ones_like(input_ids).float()],-1)
            return self.site(self.upstream(x))
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs): return 'prompt'
        def encode(self,*args,**kwargs): return list(range(1,34))
    model = Model()
    rows = [{'question':'q'} for _ in range(130)]
    c4 = np.array([list(range(1,34))])
    before = collect(model,Tokenizer(),model.site,rows,c4,0)
    with torch.no_grad(): model.upstream.weight[0].zero_()
    after = collect(model,Tokenizer(),model.site,rows,c4,0)
    assert before[2].shape == after[2].shape == (4129,2)
    assert np.all(before[0][:,0] != 0)
    assert np.all(after[0][:,0] == 0)
    assert torch.all(after[2][:,0] == 0)  # Both math and C4 refreshed.
    assert torch.equal(before[2][:,1],after[2][:,1])


def test_stage_validation_rejects_legacy_local_weights():
    from sparsifier.prune.rescale import LOCAL_PRUNING_VERSION
    stored = {'config_sha256':'config','upstream':'previous','site':[1,'mlp.down_proj'],
              'weight':torch.ones(2,3),
              'diagnostics':{'solver':'row_diagonal_congruence_approximation'}}
    with pytest.raises(ValueError, match='legacy local candidate'):
        verify_stage(stored, 'config', 'previous', (1,'mlp.down_proj'))
    stored['diagnostics']['normalization'] = LOCAL_PRUNING_VERSION
    verify_stage(stored, 'config', 'previous', (1,'mlp.down_proj'))
