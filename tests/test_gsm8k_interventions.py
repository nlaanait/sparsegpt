import numpy as np
import torch

from sparsifier.eval.gsm8k_interventions import compensated_rows, matched_design, pool_scores


def test_joint_compensation_satisfies_both_constraints_and_free_stationarity():
    torch.manual_seed(9)
    x = torch.randn(20, 5, dtype=torch.double)
    h = x.T @ x + torch.eye(5, dtype=torch.double)*.1
    inv = torch.linalg.inv(h)
    w = torch.randn(3, 5, dtype=torch.double)
    updates = compensated_rows(w, inv, [1, 3, 7])
    assert updates[0][1] == updates[0][3] == updates[1][2] == 0
    for row, cols in ((0,[1,3]), (1,[2])):
        delta = w[row]-updates[row]
        free = [j for j in range(5) if j not in cols]
        torch.testing.assert_close((h@delta)[free], torch.zeros(len(free),dtype=torch.double),atol=1e-10,rtol=0)


def test_discordant_design_has_disjoint_groups_and_matched_ambient_cost():
    rng = np.random.default_rng(10)
    ambient = np.exp(np.linspace(-3,3,8192))
    geometry = ambient*rng.uniform(.01,1,8192)
    design = matched_design(np.arange(8192),ambient,geometry,singles=128,group_pairs=8,group_size=8)
    all_ids = [entry for item in design for entry in item['entries']]
    assert len(all_ids) == len(set(all_ids))
    for i in range(8):
        high,low = [r for r in design if r.get('match_id') == i]
        ratio = high['ambient_cost_sum']/low['ambient_cost_sum']
        assert 1/1.1 <= ratio <= 1.1
        assert high['geometric_cost_sum'] > low['geometric_cost_sum']


def test_compensated_geometric_scores_match_measured_projected_row_error():
    torch.manual_seed(4)
    w, x, readout = torch.randn(3,4), torch.randn(20,4), torch.randn(5,4)
    inverse = torch.linalg.inv(x.T@x + torch.eye(4))
    pdiag = np.tile([.2,.3,.5], (5,1))
    d,labels = torch.randn(6,4), torch.tensor([0,0,0,1,1,1])
    gx,gg = torch.randn(5,4),torch.randn(5,3)
    entries = np.array([1,6,11])
    scores = pool_scores(w,inverse,x,readout,pdiag,d,labels,gx,gg,entries)
    for k, entry in enumerate(entries):
        row,col = divmod(int(entry),4)
        update = compensated_rows(w,inverse,[entry])[row]
        expected = ((readout@(w[row]-update)).square()*torch.from_numpy(pdiag[:,row])).mean()
        np.testing.assert_allclose(scores['geometric_compensated'][k],float(expected),rtol=1e-5)
