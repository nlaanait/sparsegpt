import numpy as np
import pytest
from sparsifier.prune.geometry_features import BlockDictionary
from sparsifier.prune.gram_dictionary import GramBlockDictionary


@pytest.mark.parametrize('active',[1,2,4])
def test_gram_coding_matches_residual_reference(active):
    rng = np.random.default_rng(9)
    original = BlockDictionary(4,3,active)
    fast = GramBlockDictionary(4,3,active)
    original.dictionary = rng.normal(size=(64,12))
    original.mean = rng.normal(size=(64,1))
    fast.dictionary = original.dictionary.copy(); fast.mean = original.mean.copy()
    h = rng.normal(size=(64,48))
    for steps in (0,1,35):
        expected,actual = original.encode(h,steps),fast.encode(h,steps)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
        np.testing.assert_array_equal(actual != 0,expected != 0)


def test_fitting_and_zero_dictionary_remain_finite():
    rng = np.random.default_rng(71); h = rng.normal(size=(48,64))
    original = BlockDictionary(4,3,2,seed=7).fit(h,iterations=3)
    fast = GramBlockDictionary(4,3,2,seed=7).fit(h,iterations=3)
    np.testing.assert_allclose(fast.dictionary,original.dictionary,rtol=1e-9,atol=1e-10)
    fast.dictionary[:] = 0
    assert np.array_equal(fast.encode(h),np.zeros((12,64)))
