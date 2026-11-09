"""Algebraically equivalent block coding using small sufficient statistics.

Used by sequential GSM8K fits; existing experiment estimators are unchanged.
Floating-point reassociation can change near-tied support choices, so this
implementation is tracked separately in experiment source provenance.
"""
import numpy as np
from sparsifier.prune.geometry_features import BlockDictionary, top_blocks


class GramBlockDictionary(BlockDictionary):
    def encode(self, h, steps=35):
        x = h-self.mean
        d = self.dictionary
        gram = d.T@d
        rhs = d.T@x
        z = np.zeros((self.groups*self.width,h.shape[1]))
        # Keep the original spectral-norm step, including its zero guard.
        step = 1.0/max(np.linalg.norm(d,2)**2,1e-12)
        for _ in range(steps):
            z = top_blocks(z+step*(rhs-gram@z),self.groups,self.width,self.active)
        return z
