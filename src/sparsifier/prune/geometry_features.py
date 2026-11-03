"""Small NumPy block-dictionary estimator; not the paper's BSF implementation.

Used only for calibration, never inference. For scale use a GPU-trained BSF.
"""
import numpy as np

def top_blocks(z, groups, width, active):
    """Keep whole blocks, including signed coordinates within retained blocks."""
    if not 1 <= active <= groups:
        raise ValueError('active must lie in [1, groups]')
    blocks = z.reshape(groups, width, -1)
    norms = np.linalg.norm(blocks, axis=1)
    selected = np.argpartition(norms, groups-active, axis=0)[groups-active:]
    mask = np.zeros_like(norms, dtype=bool)
    np.put_along_axis(mask, selected, True, axis=0)
    return (blocks * mask[:, None, :]).reshape(z.shape)


class BlockDictionary:
    """Alternating dictionary learning with iterative block hard thresholding.

    Nonconvex; neither global optimality nor semantic concept recovery is assured.
    Training mean is stored. Dictionary is calibration-only, never deployed.
    """
    def __init__(self, groups=12, width=3, active=2, seed=0):
        if groups < 1 or width < 1 or not 1 <= active <= groups:
            raise ValueError('invalid block dimensions or activity')
        self.groups, self.width, self.active, self.seed = groups, width, active, seed

    def encode(self, h, steps=35):
        x = h - self.mean
        z = np.zeros((self.groups*self.width, h.shape[1]))
        # Step 2a: iterative sparse coding. Faster: amortized BSF encoder on GPU,
        # warm starts, cached D.T@D, or batched group proximal optimization.
        step = 1.0 / max(np.linalg.norm(self.dictionary, 2)**2, 1e-12)
        for _ in range(steps):
            z = top_blocks(z + step*self.dictionary.T@(x-self.dictionary@z),
                           self.groups, self.width, self.active)
        return z

    def fit(self, h, iterations=20):
        if h.ndim != 2 or h.shape[1] < 2 or not np.isfinite(h).all():
            raise ValueError('H must be a finite matrix with at least two samples')
        rng = np.random.default_rng(self.seed)
        self.mean = h.mean(axis=1, keepdims=True)
        x = h-self.mean
        self.dictionary = rng.normal(size=(h.shape[0], self.groups*self.width))
        self._normalize()
        for _ in range(iterations):
            z = self.encode(h)
            # Step 2b: ridge dictionary update. Faster: minibatch sufficient
            # statistics and Cholesky solves; better recovery: restarts, dead-block
            # resampling, validation early stopping, or the paper's trained BSF.
            gram = z@z.T
            ridge = 1e-5*max(np.trace(gram)/len(gram), 1e-12)
            self.dictionary = np.linalg.solve(gram+ridge*np.eye(len(gram)), z@x.T).T
            self._normalize()
        return self

    def _normalize(self):
        for g in range(self.groups):
            sl = slice(g*self.width, (g+1)*self.width)
            block = self.dictionary[:, sl]
            block /= max(np.linalg.norm(block, 2), 1e-12)

    def tangents(self, h, neighbors=16, anchors=10, rank=1, seed=1):
        """Local PCA in each active block, mapped back to ambient coordinates.

        Pooling concept-wise tangents approximates an additive manifold model.
        It does not model interactions, support transitions, or joint curvature.
        """
        if not 1 <= rank <= self.width or neighbors < 2 or anchors < 1:
            raise ValueError('invalid tangent estimation settings')
        z = self.encode(h).reshape(self.groups, self.width, -1)
        rng = np.random.default_rng(seed)
        probes = []
        # Step 3: local charts. Faster: approximate nearest-neighbor indices per
        # block, randomized PCA, and batched GPU SVD. Better: adaptive rank/radius,
        # sample-density weighting, and joint-chart estimates for coactive blocks.
        for g in range(self.groups):
            codes = z[g][:, np.linalg.norm(z[g], axis=0) > 1e-8]
            if codes.shape[1] < max(3, rank+1):
                continue
            basis = self.dictionary[:, g*self.width:(g+1)*self.width]
            ambient = basis@codes
            for anchor in rng.choice(codes.shape[1], min(anchors, codes.shape[1]), replace=False):
                dist = np.sum((ambient-ambient[:, anchor:anchor+1])**2, axis=0)
                ix = np.argsort(dist)[:min(neighbors, len(dist))]
                local = codes[:, ix]-codes[:, ix].mean(axis=1, keepdims=True)
                u, s, _ = np.linalg.svd(local, full_matrices=False)
                directions = basis@u[:, :min(rank, int(np.sum(s > 1e-10))) ]
                if directions.shape[1]:
                    # SVD avoids inventing tangent directions for degenerate blocks.
                    q, sv, _ = np.linalg.svd(directions, full_matrices=False)
                    probes.append(q[:, sv > 1e-10])
        return np.concatenate(probes, axis=1) if probes else np.empty((h.shape[0], 0))



def synthetic(seed=7):
    """Dense embeddings of additive circles; held-out analytic tangent oracle."""
    rng = np.random.default_rng(seed)
    d, m, groups, width = 48, 24, 12, 3
    dictionary = np.concatenate([np.linalg.qr(rng.normal(size=(d,width)))[0]
                                 for _ in range(groups)], axis=1)
    def sample(n):
        z = np.zeros((groups*width,n))
        tangents = []
        for j in range(n):
            for g in rng.choice(groups, 2, replace=False):
                angle = rng.uniform(-np.pi,np.pi)
                z[g*width:g*width+width,j] = [np.cos(angle),np.sin(angle),0.3*np.sin(2*angle)]
                v = dictionary[:,g*width:(g+1)*width]@np.array([-np.sin(angle),np.cos(angle),0.6*np.cos(2*angle)])
                tangents.append(v/np.linalg.norm(v))
        return dictionary@z, np.stack(tangents,axis=1)
    train, _ = sample(480)
    test, tangent_test = sample(240)
    return rng.normal(size=(m,d))/np.sqrt(d), train, test, tangent_test

