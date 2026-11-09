# Tangent-aware pruning: from layer reconstruction to downstream geometry

**Technical article draft — revised 2026-11-08.** Objectives, algorithms, and implementation scope. Empirical benefits remain research hypotheses. Detailed derivations and earlier alternatives are retained in [the revised research note](tangent-aware-pruning-revised.md); experiment progress belongs in [the GSM8K status document](GSM8K-EXPERIMENT-STATUS.md).

**Revision focus:** Sec. 6 now distinguishes forward tangent transport from metric pullback, derives shared-weight/token coupling, identifies exactly what the VJP diagonal estimator preserves, and qualifies two-sided tangent propagation. The existing implementation is described as an activation-diagonal surrogate; the full shared-weight estimator is a proposed correction, not an implemented claim.

## Abstract

Layerwise pruning compresses a model by reconstructing each linear layer on calibration activations. SparseGPT makes this practical through approximate sparse reconstruction with compensating weight updates. We develop tangent-aware extensions by changing which activation errors the reconstruction should emphasize. Input-side methods preserve responses to meaningful activation differences. Output-side methods emphasize sample-dependent tangent and presence directions, with a scalable approximation that rescales weights before the SparseGPT solve. Downstream methods pull these geometric metrics backward through the network and couple them to local activations. This progression separates the geometry being protected from the approximation used to choose a sparse matrix. The resulting methods preserve ambient reconstruction as a baseline objective; their value must be established through held-out behavioral retention at measured compression, beyond improvements in the geometric objective itself.

## 1. Introduction

A pruning method makes two choices: which errors count, and how to find a sparse weight matrix that controls those errors. Standard layer reconstruction measures Euclidean error over observed activations. This is computationally attractive, but treats every output direction equally. A small error in a direction used by downstream computation may matter more than a larger error in another direction.

Activation geometry offers a candidate refinement. Around examples expressing a factor, local changes may concentrate along a low-dimensional tangent space. Protecting those directions could preserve useful variation more efficiently than treating all output errors equally. This requires reliable estimates of local variation and evidence that the protected variation matters to behavior. Low-dimensional structure alone supplies neither guarantee.

We first formulate SparseGPT, its OBS foundation, and the related Wanda baseline within one reconstruction problem. We then generalize the objective along the input and output axes before introducing downstream tangent pullbacks. The central question is whether these geometric terms permit more weight removal at a fixed behavioral-retention criterion.

## 2. Layer reconstruction: SparseGPT, OBS, and Wanda

### 2.1 The baseline problem

Let a target linear layer have weights \(W\in\mathbb R^{d_o\times d_i}\), calibration inputs \(\phi_n\), and sparse candidate \(\hat W\). Write \(A=W-\hat W\) for the pruning residual and let \(\mathcal S\) specify the sparsity constraint. The baseline is

\[
\min_{\hat W\in\mathcal S}\mathcal L_{\rm amb}(A),
\qquad
\mathcal L_{\rm amb}(A)=\mathbb E_n\|A\phi_n\|_2^2
=\operatorname{tr}(AH_0A^\top),
\qquad H_0=\mathbb E_n[\phi_n\phi_n^\top].
\]

Each output row shares the same input second moment: writing row \(p\) of \(A\) as \(a_p^\top\), its contribution is \(a_p^\top H_0a_p\). The Hessian of this expression with respect to \(a_p\) is \(2H_0\). We use the term *curvature* for \(H_0\), absorbing this common factor of two into the objective convention. Correlations between input features appear in the off-diagonal entries and determine how one weight can compensate for another.

**Why damping is needed.** The empirical matrix \(H_0\) is positive semidefinite, but need not be invertible. Its rank cannot exceed the number of calibration vectors; redundant features or insufficient excitation can also produce zero or very small eigenvalues. An inverse-based compensation rule can therefore be undefined or numerically unstable. In particular, a nearly unobserved input direction can admit a large weight change with little measured calibration error, even though that change may behave poorly on new inputs.

SparseGPT adds diagonal damping before factorization:

\[
\mu=\frac{1}{d_i}\operatorname{tr}(H_0),
\qquad \delta=\rho\mu,
\qquad H=H_0+\delta I,
\]

where \(\rho\) is the dimensionless `percdamp` parameter (normally 0.01 in these experiments). In an eigenbasis of \(H_0\), damping replaces each eigenvalue \(\sigma_j\) by \(\sigma_j+\delta\); the inverse then weights that direction by \(1/(\sigma_j+\delta)\). For \(\delta>0\), this bounds the inverse sensitivity and makes the matrix positive definite. It does not recover information missing from calibration data. Our implementation doubles \(\rho\) and retries when Cholesky factorization fails, for up to 20 attempts, so the effective damping can exceed the requested value.

**The objective changes as well.** Using \(H\) instead of \(H_0\) is equivalent to optimizing

\[
\mathcal L_\delta(A)
=\operatorname{tr}(AHA^\top)
=\mathbb E_n\|A\phi_n\|_2^2+\delta\|A\|_F^2.
\]

The added term penalizes deviation from the dense weights, including compensating changes to surviving entries. It is a ridge penalty on the pruning residual \(W-\hat W\), rather than weight decay on \(\hat W\) toward zero. Damping therefore affects both the cost assigned to a deletion and how its error is redistributed. An exact constrained solve using \(H\) is exact for this regularized quadratic; it need not minimize the undamped reconstruction error, and it does not make discrete mask selection globally optimal.

**Curvature scaling convention.** The code accumulates a positive scalar multiple of the input second moment, inherited from SparseGPT's sequence-based normalization. This is compatible with the equations above when damping uses the same scale. If the accumulated curvature is \(\tilde H_0=cH_0\), then

\[
\tilde\delta=\rho\frac{\operatorname{tr}(\tilde H_0)}{d_i}=c\delta,
\qquad \tilde H=cH,
\qquad \tilde H^{-1}=c^{-1}H^{-1}.
\]

Consequently, the OBS compensation ratios in Sec. 2.2 are unchanged, while all deletion costs in that comparison pool are multiplied by \(c\); their ranking is unchanged. A fixed absolute damping value would break this equivalence. This statement assumes a uniform rescaling of the same processed curvature and exact arithmetic. Coordinates with zero observed input energy are handled separately in the implementation by zeroing their weights and replacing their curvature diagonal before inversion; their treatment is not evidence of safe removal outside the calibration distribution.

### 2.2 SparseGPT and the OBS compensation principle

[SparseGPT](https://proceedings.mlr.press/v202/frantar23a.html) solves an approximate sparse reconstruction problem using shared curvature and a columnwise sequence of pruning and compensation. Its second-order foundation is [Optimal Brain Surgeon](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html), which predates SparseGPT.

For a current weight row \(w\), deleting coordinate \(q\) while allowing the other coordinates to compensate solves

\[
\min_{\Delta w}\Delta w^\top H\Delta w
\quad\text{subject to}\quad \Delta w_q=-w_q.
\]

Its solution and cost are

\[
\Delta w=-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}e_q,
\qquad s_q^{\rm OBS}=\frac{w_q^2}{[H^{-1}]_{qq}}.
\]

Let \(C\) be upper triangular with \(H^{-1}=C^\top C\). SparseGPT uses the score and update

\[
s_{pq}=\frac{W_{pq}^2}{C_{qq}^2},
\qquad
W_{p,j}\leftarrow W_{p,j}-\frac{W_{pq}}{C_{qq}}C_{qj},\quad j\ge q.
\]

The update corresponds to restricted quadratic compensation over columns still available in the elimination order. It does not imply that the complete discrete mask is globally optimal. Blockwise mask selection and processing order are additional algorithmic choices; scores within a selected block may be computed before later compensation changes its weights.

### 2.3 Wanda as a related diagonal approximation

[Wanda](https://arxiv.org/abs/2306.11695) ranks weights within each output row by \(|W_{pq}|\|X_{q,:}\|_2\), without compensating updates. For one isolated deletion, the mean ambient reconstruction cost is

\[
I_{pq}^{\rm amb}=W_{pq}^2(H_0)_{qq}.
\]

Squared Wanda scores are proportional to this cost. Replacing the input curvature by its diagonal gives the same ranking in the OBS expression. This connects the criteria mathematically; Wanda is a related method rather than a historical derivative of SparseGPT.

The distinction matters below: predicting an isolated deletion's energy is easier than predicting its cost after other weights compensate.

## 3. Motivation: which activation variations should pruning preserve?

### 3.1 The local manifold assumption

Fix an activation site with ambient space \(\mathcal H=\mathbb R^d\). The local manifold hypothesis models factor-conditioned activations as samples near a smooth embedded submanifold \(\mathcal M\subset\mathcal H\) of dimension \(r\ll d\). This hypothesis is local to a regime of factor activity; it does not require the full activation distribution to lie on a single manifold. Changes in active support may instead require a collection of manifolds or a stratified description. Discrete token substitutions provide samples for this hypothesis, not a differentiable parameterization of the input space.

Let \(\iota:\mathcal M\hookrightarrow\mathcal H\) be the inclusion and \((\mathcal U,\psi)\) a chart, with \(\psi:\mathcal U\to\Omega\subset\mathbb R^r\). The associated local parameterization is \(f=\iota\circ\psi^{-1}\). At \(h=f(u)\), its differential identifies the tangent space with an ambient linear subspace:

\[
d\iota_h(T_h\mathcal M)=\operatorname{im}Df(u),
\qquad \operatorname{rank}Df(u)=r.
\]

We use \(T_h\mathcal M\) for this embedded tangent space when the identification is unambiguous. The Euclidean inner product on \(\mathcal H\) induces the Riemannian metric \(g=\iota^*\langle\cdot,\cdot\rangle\), whose coordinate matrix is \(g_u=Df(u)^\top Df(u)\). The ambient orthogonal projector onto the tangent space is

\[
P_h=Df(u)\,g_u^{-1}\,Df(u)^\top=Q_hQ_h^\top,
\]

where \(Q_h\) is any Euclidean-orthonormal tangent frame. The projector is independent of chart coordinates and of the chosen orthonormal frame, although its diagonal entries depend on the ambient coordinate basis. Curvature permits \(T_h\mathcal M\) to vary with \(h\); replacing these spaces by a fixed global span is an additional approximation.

### 3.2 Estimating directions and using them in pruning

The manifold and its charts are unknown. Our estimators approximate tangent frames from centered activation neighborhoods using local PCA, optionally within separately estimated factor components. These neighborhood models constitute an empirical atlas only in an approximate sense: the implementation does not recover transition maps or establish smooth compatibility on chart overlaps. Tangent estimation depends on neighborhood scale, sampling density, curvature, noise, and separation of co-active factors; stability and held-out approximation error must be measured.

At sample \(n\), let \(Q_n\in\mathbb R^{d\times r_n}\) be an estimated orthonormal frame and \(P_n=Q_nQ_n^\top\). The splitting \(\mathcal H=T_{h_n}\mathcal M\oplus N_{h_n}\mathcal M\) decomposes a pruning error \(e_n\) into tangential and normal components. The penalty \(\|P_ne_n\|_2^2\) emphasizes its tangential component in the induced metric. It does not constrain the normal component or imply preservation of intrinsic distances, curvature, or downstream behavior. Frame estimates should be distinguished from the exact tangent spaces in this interpretation.

For co-active factors, let \(E_{i,n}\subset\mathcal H\) be their estimated tangent subspaces after mapping them into the same activation space. Protect the sum \(E_n=\sum_{i\in S_n}E_{i,n}\) using an orthonormal frame for that sum. Its dimension can be smaller than the sum of factor dimensions. Identifying \(E_n\) with the tangent of a joint factor manifold additionally requires a compatible local composition model; a product-manifold or direct-sum decomposition is not assumed automatically.

Tangential variation also need not encode factor presence. Given a chart anchor \(\bar h_n\) and an explicit concept-absent reference \(h_n^{(0)}\), the ambient offset \(a_n=\bar h_n-h_n^{(0)}\) can have a component \((I-P_n)a_n\) normal to the protected tangent sum. When nonzero, its normalized direction augments the protected subspace to account for attenuation toward that reference. This is an extrinsic, reference-dependent construction, not a canonical intrinsic tangent vector. Denote the projector onto the augmented span by \(P_n^{\rm geo}\). Without an absent reference, retain the tangent-only interpretation.

These projectors define selected directions in activation space, not an established semantic decomposition. Local charts should be compared with fixed factor spans, a global union, and shuffled assignments on held-out data. A geometry objective that ignores the complementary space can allow arbitrarily large unmeasured errors there. We therefore add geometric terms to ambient reconstruction.

The site must also be explicit: a projection output, post-MLP residual, and normalized residual are different spaces. Geometry measured at one site applies to another only through a specified intervening map.

## 4. Input-side geometry: augment the shared curvature

The simplest extension protects responses to meaningful changes in the layer input. For aligned paired inputs, define \(t_k=\phi(x_k')-\phi(x_k)\) and \(H_\Delta=\mathbb E_k[t_kt_k^\top]\). Then

\[
\mathcal L_{\rm in}(A)
=\mathbb E_n\|A\phi_n\|_2^2
+\gamma_{\rm in}\mathbb E_k\|At_k\|_2^2
=\operatorname{tr}\bigl(A(H_0+\gamma_{\rm in}H_\Delta)A^\top\bigr).
\]

This changes the shared input curvature, so the existing SparseGPT solve applies directly after augmentation and damping. Explicit input tangent probes can replace paired differences. Finite differences capture the chosen paired intervention; they approximate a tangent only at appropriate step sizes and alignments.

Set the added-to-ambient trace ratio through

\[
\gamma_{\rm in}=\alpha_{\rm in}
\frac{\operatorname{tr}(H_0)}{\operatorname{tr}(H_\Delta)}.
\]

Thus \(\alpha_{\rm in}\) is a ratio to ambient curvature; its fraction of the augmented trace is \(\alpha_{\rm in}/(1+\alpha_{\rm in})\). Normalize heterogeneous intervention families explicitly and disable or reject zero-energy terms.

For a linear layer, \(At_k\) is exactly the error in its local response difference. For a gate or upstream projection, it is a preactivation error; behavior after nonlinearities requires further analysis or a downstream pullback.

## 5. Local output geometry: sample-dependent reconstruction metrics

### 5.1 Objective and direct deletion scores

For a layer writing into the geometric site, use

\[
\mathcal L_{\rm out}(A)
=\mathbb E_n\|A\phi_n\|_2^2
+\gamma\mathbb E_n\|P_n^{\rm geo}A\phi_n\|_2^2.
\]

This replaces the output identity metric by \(M_n=I+\gamma P_n^{\rm geo}\). An isolated deletion has geometric energy

\[
I_{pq}^{\rm geo}
=W_{pq}^2\mathbb E_n[(P_n^{\rm geo})_{pp}\phi_{qn}^2].
\]

Define the activation-conditioned coupling

\[
G_{pq}=\frac{\mathbb E_n[(P_n^{\rm geo})_{pp}\phi_{qn}^2]}
{\mathbb E_n[\phi_{qn}^2]}.
\]

It combines output-coordinate participation with input-feature activity on the same examples. An average projector computed before this pairing loses that dependence. These direct energies motivate mask scores but do not include compensation.

### 5.2 From the coupled objective to a scalable solve

The full output metric couples weight rows. Dropping its off-diagonal entries gives a separate curvature for each output row:

\[
H^{(p)}=H_0+\gamma\mathbb E_n[(P_n^{\rm geo})_{pp}\phi_n\phi_n^\top].
\]

Approximate each row curvature by a diagonal congruence of one shared curvature \(H'\):

\[
H^{(p)}\approx\Lambda_pH'\Lambda_p,
\qquad \lambda_{pq}^2=\frac{H^{(p)}_{qq}}{H'_{qq}}.
\]

For \(H'=H_0\), this gives \(\lambda_{pq}^2=1+\gamma G_{pq}\). The corresponding algorithm is

```text
Estimate sample-paired geometric moments and shared ambient curvature.
Match row-curvature diagonals to obtain positive scales Λ.
Run SparseGPT on W_scaled = W ⊙ Λ with the shared curvature.
Return W_pruned = W_scaled_pruned ⊘ Λ.
```

For a different shared curvature, the entire row diagonal must be divided by
the shared diagonal: \(\lambda_{pq}^2=((H_0)_{qq}/H'_{qq})(1+\gamma G_{pq})\).
This includes the ambient term even when \(\gamma=0\).

The GSM8K local runners normalize geometric weight-curvature strength directly.
With ambient input-energy sums
\(S_q=\sum_n\phi_{nq}^2\), paired readout sums
\(K_{pq}=\sum_{n\in R}(P_n^{\rm geo})_{pp}\phi_{nq}^2\), and
\(r_n=\operatorname{tr}P_n^{\rm geo}\), the implementation uses
\(\gamma=\alpha d_o\sum_q S_q/\sum_{pq}K_{pq}\), matching the added
weight-curvature trace to alpha before damping. Both sums in this
normalization exclude unsupported input coordinates, which retain unit scale;
excluded geometric energy is reported. Geometry is zero outside readouts,
and both curvatures use the ambient-token averaging denominator. A supported
projector with zero geometric input energy disables the added trace term;
an atlas with no supported geometry is rejected. This is the only local
normalization; no selector flag is required. The former rank-based formula
\(\gamma=\alpha d_oN_a/\sum_{n\in R}r_n\) failed to control weight-curvature
strength and has been removed. Its results are excluded from corrected local
comparisons. New candidate diagnostics record both trace
ratios, scale distributions, ranks, and geometry coverage.

The scale changes both selection and compensation:

\[
s_{pq}=\lambda_{pq}^2\frac{W_{pq}^2}{C_{qq}^2},
\qquad
\Delta W_{pj}=-W_{pq}\frac{\lambda_{pq}}{\lambda_{pj}}\frac{C_{qj}}{C_{qq}}.
\]

Multiplying scores alone omits the matching update and remains a heuristic ablation. Unsupported input coordinates keep unit scale and are reported.

There are two distinct approximations: removing output coupling and replacing row curvatures by shared congruences. Damping adds another qualification. Standard damping in scaled coordinates induces \(\delta\Lambda_p^2\) in original coordinates, so it is not generally equivalent to adding \(\delta I\) to each intended row curvature. The implemented solve must be evaluated as that damped surrogate.

### 5.3 When reweighting can change a mask

A fixed output metric produces constant row factors in this approximation. They can redistribute deletions across rows under blockwise unstructured selection, but cancel in row-local Wanda or N:M comparisons. Sample-conditioned geometry can vary within a row through its coupling with input activity. The comparison group and observed mask changes must therefore be reported.

Coordinate leverages are basis-dependent and can vary substantially even for random subspaces. Random-looking leverage distributions or random-control-like mask-flip rates do not establish usefulness or failure. The relevant test is held-out impact prediction and pruning behavior against matched controls.

### 5.4 References beyond row-diagonal approximation

For a fixed mask, the full sample-dependent objective is a convex quadratic. Matrix-free products use \(\mathbb E_n[M_nA\phi_n\phi_n^\top]\); solving its free-coordinate normal equations provides a reference refit, subject to numerical convergence. Small problems can explicitly construct the quadratic to measure approximation error.

A complementary construction uses a fixed global union projector \(\Pi=QQ^\top\). Starting from ambient sparse weights \(W_{\rm sp}\), restore

\[
\hat W=W_{\rm sp}+\Pi(W-W_{\rm sp}).
\]

This cancels error in that fixed union but adds a dense low-rank correction. It is sparse-plus-low-rank compression, with its full storage counted, and does not reproduce sample-dependent chart selection.
