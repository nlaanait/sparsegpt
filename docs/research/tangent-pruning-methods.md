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

## 6. Downstream geometry: pull back an error metric, not tangent vectors

### 6.1 Specify the intervention and the sequence-level map

Tangent vectors propagate **forward** by a Jacobian: a local tangent frame U has downstream image span(JU), possibly with reduced rank. The construction below instead propagates **covectors backward** and pulls back a quadratic metric. J-transpose V is generally neither an orthonormal tangent frame nor the inverse image of a downstream manifold. Orthogonalizing it would discard the downstream amplification that the metric is meant to retain.

Index independent calibration sequences by s, target token positions by t, and downstream anchor positions by u. Stack the target outputs into y_s, and write the anchor state as h_{k,s}=F_{k←ℓ,s}(y_s). This is an intervention map: other incoming branches, attention masks, position information, and reference inputs are held fixed. The map must include every path from the selected target output to the anchor; the output of a gate projection, an MLP down projection, and a residual state are different intervention sites.

At a frozen reference forward, define

\[
J_{k,s}=D_yF_{k\leftarrow\ell,s}(y_s),\qquad
D_{k,s}=\operatorname{blockdiag}_u(\omega_{k,s,u}P^{\rm geo}_{k,s,u}),
\qquad \omega_{k,s,u}\ge0.
\]

The token-block-diagonal choice for D is an **anchor-metric modeling choice**, not an assertion that the network is token-local. If geometry describes a joint sequence manifold, use a joint sequence metric instead. Weights specify whether anchors are summed, averaged, or reweighted; padding must receive zero weight. For geometric frames V with orthonormal columns, P=VV-transpose. More generally any declared positive-semidefinite D can be used.

For the shared matrix residual A=W−W-hat, define the linear operator

\[
T_s(A)=\operatorname{stack}_t(A\phi_{s,t}).
\]

Replacing W by W-hat perturbs the target output by −T_s(A), assuming its input is held fixed during this layer's reconstruction. The downstream change is

\[
\delta h_{k,s}=-J_{k,s}T_s(A)+O(\|T_s(A)\|^2).
\]

This expansion requires a differentiable intervening map with controlled local derivatives. At a support boundary, hard routing change, or sufficiently large perturbation it need not predict the finite change accurately. The frozen geometric penalty is

\[
B_s=\sum_k\beta_kJ_{k,s}^{\top}D_{k,s}J_{k,s},\quad \beta_k\ge0,
\qquad
\mathcal L_{\rm down}(A)=\mathcal L_{\rm amb}(A)
+\gamma\mathbb E_s[T_s(A)^\top B_sT_s(A)].
\]

B is PSD but usually not a projector. Its nullspace includes errors invisible to the selected downstream measurements, including errors mapped into unprotected normal directions. This is why ambient reconstruction remains in the objective. It is a frozen geometric/Gauss–Newton-style surrogate, not the exact Hessian of task loss and not a guarantee of GSM8K retention. A finite-change geometric discrepancy with fixed D agrees to quadratic order under the stated smoothness assumptions.

### 6.2 Shared weights require token cross terms

Partition B_s into blocks B_{s,tt′} of size d_o by d_o. The exact geometric quadratic for the frozen model is

\[
\mathbb E_s\sum_{t,t'}(A\phi_{s,t})^\top B_{s,tt'}(A\phi_{s,t'}).
\]

The same A acts at every token. Consequently, retaining a full downstream Jacobian but dropping cross-token entries of B still changes the weight objective. For one deletion A=W_{pq}e_pe_q-transpose, the exact geometric energy is

\[
I^{\rm down}_{pq}=W_{pq}^2\mathbb E_s\sum_{t,t'}
\phi_{s,t,q}(B_{s,tt'})_{pp}\phi_{s,t',q}.
\]

It is generally **not** W_{pq} squared times the sum of b_{s,t,p} phi_{s,t,q} squared. Cross-token contributions may reinforce or cancel; removing them is neither a general upper nor lower bound. For a scalar output shared over two tokens with phi_1=phi_2=1, B=[[1,−1],[−1,1]] gives zero exact energy for A=a but diagonalization gives 2a². B=[[1,1],[1,1]] gives 4a² exactly versus 2a² after diagonalization. Both B are PSD.

The full weight-space curvature operator is E_s[T_s* B_s T_s], where T_s*(g)=sum_t g_t phi_{s,t}-transpose. This operator, plus ambient curvature and explicitly specified damping, is the appropriate fixed-mask reference. It also makes clear why tokenwise diagonalization and weight-space diagonalization are different approximations.

### 6.3 VJPs: what is unbiased, and what is discarded

Factor D_{k,s}=L_{k,s}L_{k,s}-transpose. Draw zero-mean r with E[rr-transpose]=I and form

\[
v_{k,s}=L_{k,s}r,\qquad g_{k,s}=J_{k,s}^{\top}v_{k,s},\qquad
\mathbb E_r[g_{k,s}g_{k,s}^{\top}]=J_{k,s}^{\top}D_{k,s}J_{k,s}.
\]

For token-local frames, L has blocks sqrt(omega_u)V_u. Independent coefficients at different positions encode the specified block-diagonal anchor metric; reusing coefficients across positions introduces a different metric with cross-position terms. Multiple anchors can be estimated separately with beta_k weights, or combined using independent zero-mean seeds scaled by sqrt(beta_k). Shared seeds without a specified covariance introduce unwanted cross-anchor terms.

Compute each VJP from the scalar inner product of stopgrad(v) with h_k. Freeze bases, chart assignments, anchor weights, and reference states for the quadratic solve. Parameters need no accumulated gradient buffers, but the computational graph from target output to anchor must remain intact. In a joint multi-target forward, detaching an intermediate target can sever paths required by an earlier target; a separate forward per target avoids that error.

Averaging g-coordinate squares estimates diag(B), not the complete weight-space curvature. Enumerating all columns of L gives this diagonal exactly up to numerical precision. To estimate the **exact shared-weight curvature diagonal**, first form

\[
Z_{k,s}=T_s^*(g_{k,s})=\sum_t g_{k,s,t}\phi_{s,t}^{\top},
\qquad
K^{\rm full}_{pq}=\mathbb E_s\sum_k\beta_k\mathbb E_r[(Z_{k,s})_{pq}^2].
\]

Summing over tokens **before squaring** retains token interference. Summing squared token contributions instead gives the activation-diagonal surrogate in Sec. 6.4. The full weight diagonal still omits off-diagonal compensation interactions and does not by itself define an exact SparseGPT solve. Exact matrix-free products can instead apply T, J, D, J-transpose, and T* in sequence.

Unbiasedness holds conditional on the frozen forward and geometry, before normalization. With finitely many probes, trace-normalizing an estimated K is a nonlinear operation: the resulting normalized metric/scales are not generally unbiased. Report probe-seed variability and convergence, especially after mask selection.

### 6.4 The implemented scalable surrogate and its normalization

The current downstream accumulator keeps

\[
b_{s,t,p}=(B_s)_{(t,p),(t,p)},\qquad
K_{pq}=\mathbb E_s\sum_t b_{s,t,p}\phi_{s,t,q}^2,
\qquad
H^{(p)}=H_0+\gamma\mathbb E_s\sum_t b_{s,t,p}\phi_{s,t}\phi_{s,t}^{\top}.
\]

Here H_0=E_s sum_t phi_{s,t} phi_{s,t}-transpose uses the same sequence-sum convention. A token-average convention is also valid if declared consistently. First, activation-space diagonalization removes output-coordinate and token correlations; second, the congruence approximation of Sec. 5 replaces the resulting row curvatures by scaled copies of a shared H_0; third, SparseGPT approximates the sparse solve. None of these steps follows from the unbiased VJP identity alone.

For the diagonal surrogate define

\[
\gamma=\alpha\frac{d_o\operatorname{tr}(H_0)}{\sum_{p,q}K_{pq}},
\qquad \lambda_{pq}^2=1+\gamma K_{pq}/(H_0)_{qq}.
\]

This matches the added **surrogate weight-curvature trace**, not projector rank or the full downstream curvature trace. Local ambient and backward samples must remain paired within K, but their budgets can differ. Concretely, with ambient sums S_q and geometric sums G_{pq}, the code uses gamma_sum=alpha d_o sum_q S_q / sum_{pq} G_{pq} and lambda²=1+gamma_sum G_{pq}/S_q. If their averaging denominators are N_a and N_g, the corresponding mean-convention coefficient is gamma_mean=gamma_sum N_g/N_a. Record the convention; do not interpret a raw-sum gamma as budget-independent.

The code excludes unsupported ambient coordinates from normalization and assigns them unit scale, and disables a zero-trace added term. Report excluded geometric energy: this choice does not protect a geometry-only input coordinate. Preserve relative anchor weights before global normalization; separately trace-normalizing each anchor specifies a different objective. Scaled-coordinate damping has the original-coordinate penalty described in Sec. 5.2.

### 6.5 Optional two-sided tangent response: a distinct approximation

For a single-position input perturbation with frame U and covariance R=UU-transpose, the frozen response penalty is

\[
\eta\operatorname{tr}(BA R A^{\top})
=\eta\sum_k\beta_k\|L_k^{\top}J_k A U\|_F^2.
\]

For sequences, input tangent probes must specify their joint token structure: for a probe xi=stack_t xi_t use T_xi(A)=stack_t A xi_t and E_xi[T_xi(A)-transpose B T_xi(A)]. Independent token probes and correlated whole-sequence probes are different choices. Under the activation-diagonal approximation, token-local covariances give the accumulated row-diagonal term eta E_s sum_t b_{s,t,p}(R_{s,t})_{qq}. This is the implemented optional input-basis coupling.

This term controls the contribution from changing W while **freezing the downstream Jacobian**. It is not the full change in the downstream tangent-transfer map. With input frame U fixed, y=W phi, perturbation ΔW, and transfer J(y)WU, its first variation is

\[
\delta[J(y)WU]
=J(y)\Delta W U
+\bigl(D_yJ(y)[\Delta W\phi]\bigr)WU.
\]

The second term involves downstream second derivatives. Changes to U, downstream geometry/assignments, or upstream inputs add further terms. Thus even infinitesimal weight changes need not justify dropping this term: it is also first order in ΔW. The approximation needs a nearly constant downstream Jacobian or an empirical bound on the omitted contribution. A meaningful finite-difference check compares the complete transfer change against the frozen-J prediction.

Factor-local geometry must remain sample- and factor-dependent in every construction above. Push each co-active factor's tangent through the specified map before forming its image span; distinct regions may have different frames or ranks. If measuring downstream sensitivity, pull back each declared downstream metric and sum with explicit weights instead. A global union or a pooled covariance erases some of this conditioning and is a baseline, not an equivalent representation of a union of local factor manifolds.


## 7. Algorithms and implementation scope

The techniques form the following progression. Each row changes the objective or its representation; none implies a demonstrated empirical improvement.

| Technique | Additional information | Computational change |
|---|---|---|
| Ambient SparseGPT | Local input second moment | Shared curvature and compensating updates |
| Input-side augmentation | Paired differences or input tangent probes | Add shared input curvature |
| Local output geometry | Sample-conditioned tangent/presence projectors | Approximate row curvatures; rescale weights and compensation |
| Downstream pullbacks | Downstream geometry and intervening VJPs | Backward calibration, then the same approximate row solve |
| Two-sided tangent coupling | Local input tangents plus downstream metric | Add paired geometric curvature moments |
| Full-metric refit | Retained output/token correlations | Coupled fixed-mask quadratic solve; small reference problems |

The implemented downstream procedure is:

1. Select target linears and downstream activation anchors. Fit chart geometry on a separate fitting split and freeze the assignment rule for calibration.
2. Collect the matched ambient input budget. Run backward probes on declared calibration sequences, preserving pairing between each local activation and its pulled-back sensitivity.
3. Accumulate geometric curvature diagonals, set their trace ratios, and construct positive scales. Quantify unsupported coordinates and probe variability.
4. Prune scaled weights with SparseGPT and divide by the scales. Persist candidates and upstream checkpoint hashes.
5. Before the next sequential target, refresh activations and, when specified by the experiment, refit geometry on the current partially pruned model. Evaluate on held-out examples.

Backward calibration requires activation memory and backward-compatible kernels. Stopping forwards at bounded anchors limits memory and defines the scope of the metric. It does not make a bounded-anchor experiment equivalent to a full-network metric. The implementation supports frozen models without changing parameter gradient buffers; hard-mask selection remains outside differentiation.

Bias correction can be represented by an appended constant input for ambient reconstruction. With sample-dependent output metrics, exact weighted bias correction requires the corresponding coupled normal equations; a row-diagonal solver remains an approximation. Input differences cancel constant offsets automatically.

| Component | Repository implementation |
|---|---|
| Ambient SparseGPT; Wanda | `src/sparsifier/prune/sparsegpt.py`; `wanda.py` |
| Shared input augmentation | `src/sparsifier/prune/input_side.py` |
| Matched rescaling and compensation | `src/sparsifier/prune/rescale.py` |
| Chart bases, VJPs, downstream calibration and solve | `src/sparsifier/prune/downstream.py` |
| Small coupled quadratic/refit reference | `src/sparsifier/prune/sample_metric.py` |
| Sequential GSM8K downstream arm | `src/sparsifier/eval/gsm8k_downstream.py` |

## 8. Evaluation and limitations

Three questions should be separated. First, do fitted tangents describe held-out variation better than fixed spans at matched fitting and storage budgets? Second, do geometric scores predict compensated and downstream intervention effects beyond ambient scores? Third, does geometric pruning extend the measured compression frontier while retaining behavior?

Use disjoint fitting, calibration, selection, and final evaluation data. Compare against task-calibrated SparseGPT with matched ambient exposure, Wanda where applicable, fixed-factor/global geometry, shuffled assignments, matched-rank random directions, and a gradient-based metric. A Fisher-style metric can use the same row-weight machinery but represents task sensitivity rather than activation geometry. Extra backward compute must be counted.

Measure approximation error on tractable coupled quadratics (including cross-token cancellation and reinforcement), compare activation-diagonal and shared-weight-diagonal VJP estimates, test backward-probe convergence and the omitted Jacobian-variation term, and measure finite-perturbation prediction error, mask changes, runtime, and memory. Leverage distributions and gradient enrichment are diagnostics rather than sufficient decision rules. Variable-rank random controls must match the sample-dependent ranks and energy weighting of the tested statistic.

The practical outcome is the greatest measured whole-model removal fraction satisfying a predeclared dense-relative behavioral tolerance with appropriate uncertainty. Eligible-matrix sparsity, whole-model sparsity, and sparse-plus-low-rank storage are different quantities. Better calibration tangent reconstruction alone is expected under the objective and does not establish task retention.

CPU checks establish selected numerical contracts, including pullback diagonals against explicit Jacobians, paired curvature accumulation, and matched compensation. They do not establish language-model utility. The GSM8K downstream arm tests next-block residual tangents with refreshed sequential calibration, alongside the existing local method; its experimental configuration and results are tracked separately. Claims of semantic geometry, manifold preservation, or improved accuracy retention require independent held-out evidence.

## References

- Frantar and Alistarh. [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html). ICML, 2023.
- Hassibi and Stork. [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html). NeurIPS 5, 1992 proceedings.
- Sun et al. [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695). Wanda.
- [Revised tangent-aware pruning note](tangent-aware-pruning-revised.md): extended derivations and alternatives.
- [Review of the revised note](tangent-aware-pruning-revised-luna6-review.md): qualifications on damping, normalization, and empirical claims.
