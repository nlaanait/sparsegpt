import numpy as np
from sparsifier.eval.gsm8k_prediction import split_ids, predict, crossed_interval


def test_split_is_identity_based_and_order_independent():
    ids = [str(i) for i in range(128)]
    assert split_ids(ids) == split_ids(ids[::-1])
    assert len(split_ids(ids)) == 89


def test_incremental_predictor_recovers_independent_signal():
    rng = np.random.default_rng(7)
    x = np.exp(rng.normal(size=(128, 2)))
    y = 2*np.log10(x[:, 1])
    baseline = predict(x[:89, :1], y[:89], x[89:, :1])
    augmented = predict(x[:89], y[:89], x[89:])
    assert np.mean((baseline-y[89:])**2) > .1
    assert np.max(abs(augmented-y[89:])) < 1e-12


def test_crossed_interval_preserves_constant_difference():
    assert crossed_interval(np.ones((10, 4)), ['a','a','b','b'], draws=100) == [1., 1.]
