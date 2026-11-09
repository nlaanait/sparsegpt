import numpy as np
from sparsifier.eval.gsm8k_controls import whole_chart_diagonals


def test_whole_chart_control_assigns_heldout_without_refitting():
    rng = np.random.default_rng(5)
    fit = rng.normal(size=(80,6))
    query = rng.normal(size=(5,6))
    together, info = whole_chart_diagonals(fit,query,charts=3,rank=2)
    first,_ = whole_chart_diagonals(fit,query[:1],charts=3,rank=2)
    np.testing.assert_allclose(first[0],together[0],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(together.sum(1),2,rtol=1e-5)
    assert sum(info['chart_counts']) == 5
