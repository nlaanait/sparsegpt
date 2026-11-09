# Review: Tangent-aware pruning (revised)

**Reviewed:** 2026-11-05<br>
**Document:** [tangent-aware-pruning-revised.md](tangent-aware-pruning-revised.md)<br>
**Scope:** Read-only mathematical and experimental-design critique. This note records issues and proposed clarifications; it does not edit or replace the reviewed document.

## Overall assessment

The revision substantially improves the framing: the empirical benefits are hypotheses, the geometric loss is kept alongside ambient reconstruction, and local activation reconstruction is distinguished from downstream task behavior. Several derivations and experiment decisions still need qualification, especially where the note calls an approximation exact or treats a diagnostic null as a decision rule.

## Findings

### 1. The (R d_{\mathrm{in}}) constraint count needs a feasibility caveat

At Sec. 3.2, the note says that the lexicographic limit spends (R d_{\mathrm{in}}) constraints, where (R) is the span rank of all geometric directions, and uses this to argue that exact calibration preservation is cheap.

That count describes a sufficient, stronger condition: requiring the fixed-union residual to vanish, \(\Pi A = 0\), ensures zero sample-conditioned geometric error for every calibration example. It is not generally the number of independent constraints in the input-conditioned objective \(\sum_n \|Q_n^\top A\phi_n\|^2\); that objective's constraint rank depends on the full sample operator

\[
 A \longmapsto \{Q_n^\top A\phi_n\}_{n=1}^N.
\]

The fixed-union construction is a useful sufficient bound if the allowed surviving entries can realize \(\Pi\hat W=\Pi W\). State that feasibility assumption (for example, adequate rank in the surviving output rows for each input column), and distinguish this upper-bound construction from the actual rank of the sample-conditioned constraints. This keeps the budget claim without implying that every local-projector constraint system has rank \(R d_{\mathrm{in}}\).

### 2. The random-subspace null for \(\kappa\) can differ from one

At Sec. 4.8, the denominator uses \(\bar r^{\mathrm{geo}}/d\). If projector rank varies across samples and gradient norm correlates with rank, the expected random-projector ratio of summed energies is instead

\[
\kappa_0 =
\frac{\mathbb{E}_n[r_n^{\mathrm{geo}}\|g_n\|_2^2]}
     {d\,\mathbb{E}_n[\|g_n\|_2^2]}.
\]

Use this matched-rank null, or define an enrichment statistic normalized per sample before averaging. Also present \(\kappa\) as a screening diagnostic, not a sufficient test of whether geometry weighting can improve task loss: pruning errors may be structured and correlated with gradients, and aggregate gradient energy alone does not capture those effects.

### 3. Stage-0 nulls do not imply the stated decisions

At Sec. 6, D1 compares coordinate leverages with the Beta distribution for a randomly oriented subspace, then says that being indistinguishable from this null is grounds to skip entrywise output-side scores. That null tests coordinate orientation; it does not establish that the real geometric score is uninformative for held-out impact. A real projector can have a random-looking leverage distribution and still carry predictive signal through its alignment with feature activity.

D2 compares mask flips with matched-rank random tangents and interprets a near-null flip rate as evidence that the output metric cannot matter. A flip rate measures mask sensitivity, not whether the resulting mask helps or harms. A rate similar to a random-tangent rate means the masks respond about as much as the random control does; it does not show that the real metric is ineffective.

Keep these as descriptive diagnostics. Test usefulness by comparing held-out score prediction and pruning outcomes for real directions against matched-rank random directions, with the same sparsity and compensation settings. A near-zero flip rate relative to the vanilla mask is evidence of little mask change; a random-control rate alone is not.

### 4. SparseGPT exactness and damping are mixed together

Sec. 4.1 defines the empirical ambient loss with \(H_0=XX^\top/N\), but gives the OBS cost using \(H^{-1}\), while the notation defines \(H=H_0+\lambda I\). The cost is exact for the damped quadratic, not necessarily for the raw empirical reconstruction loss. State which curvature each exactness claim refers to and distinguish the damped surrogate from the undamped data loss.

There is a related issue in Variant B. The row scaling is computed from undamped moments, but the pseudocode calls SparseGPT with its usual damping. In general,

\[
\Lambda_p(H'+\delta I)\Lambda_p
\ne \Lambda_p H'\Lambda_p + \delta I.
\]

Thus standard post-scaling damping need not match the row curvature defined in the note. Qualify the exactness claims, including the static sanity case, or specify a damping rule that preserves the stated row Hessian. The congruence approximation itself is already acknowledged; this issue is separate from its off-diagonal approximation error.

### 5. Variant A's guarantee depends on which layer output is measured

Sec. 3.3 correctly says that \(A\,\delta\phi\) is the difference error for a linear layer's local output. Sec. 4.4 then extends the variant to weights such as \(W_{\mathrm{up}}\), \(W_{\mathrm{gate}}\), and attention projections.

For these weights, \(A\,\delta\phi\) generally measures a preactivation or projection error. It does not, by itself, equal the effect at the block output or residual stream after nonlinearities and downstream projections. Clarify whether Variant A targets local layer-output reconstruction in each case. If the intended target is block-output concept behavior, form paired output differences there or pull the metric back through the downstream Jacobian, with the approximation stated.

### 6. The weighted-bias equivalence needs a solver qualification

Sec. 3.4 says that an appended constant-one input column handles a weighted bias automatically when \(M_n\ne I\). For sample-varying output metrics, the full objective couples output rows. A standard SparseGPT pass with one shared input Hessian does not automatically represent that coupled problem.

The constant-column equivalence is exact for \(M_n=I\). For non-identity, sample-varying \(M_n\), restrict the exactness claim to a solver that includes the full output coupling, or explicitly call it an approximation under the diagonal-output treatment used in Variant B. The centered-covariance equivalence should likewise remain limited to the unweighted case unless the weighted normal equations are derived.

### 7. \(\alpha_{\mathrm{in}}\) is a ratio, not a share of total curvature

The code sets

\[
\gamma_{\mathrm{in}} = \alpha_{\mathrm{in}}
\frac{\operatorname{tr}(H)}{\operatorname{tr}(H_\Delta)}.
\]

This makes the added trace equal to \(\alpha_{\mathrm{in}}\operatorname{tr}(H)\). Thus \(\alpha_{\mathrm{in}}\) is the added-to-base curvature ratio. Its fraction of total curvature is \(\alpha_{\mathrm{in}}/(1+\alpha_{\mathrm{in}})\). Change “share of total curvature” to “ratio to the ambient curvature,” or reparameterize if the intended parameter is a fraction of the total. Under the stated isotropy normalization, the output-side \(\alpha\) is also a geometric-to-ambient ratio, not a fraction of their sum.

### 8. Wanda's score is proportional to the stated mean loss

The ambient loss at Sec. 4.1 is \(\|AX\|_F^2/N\). For a single-entry knockout, this gives \(W_{pq}^2(H_0)_{qq}\). Squared Wanda is \(W_{pq}^2\|X_{q,:}\|_2^2 = N W_{pq}^2(H_0)_{qq}\). The rankings are identical and the scores differ only by a global factor \(N\), but “exactly the cost” should be “proportional to the mean per-sample cost,” or the loss convention should be changed consistently.

### 9. The toy result is not reproducible from the note

Sec. 3.1 reports an ambient error “157 times the layer's output energy” for a random toy, but does not define the random construction, seed, fitting or mask procedure, or the normalization meant by “output energy.” Add enough detail to reproduce the result, ideally a small script or pseudocode. Otherwise, present it as an illustrative example without a precise numeric ratio.

### 10. H3's “better front” needs an operational decision rule

H3 compares several objectives at once: ambient error, tangent and presence error, and downstream metrics. “Traces a better front” is directionally clear but leaves the support criterion open. Specify what counts as evidence—for example, a held-out Pareto-dominance rule over the \(\alpha\) sweep, with uncertainty intervals and matched sparsity, pattern, and compensation budget. This makes a null or disconfirming result easier to interpret.

## Suggested review order

1. Correct or qualify the \(\kappa\) null and Stage-0 decision rules.
2. Clarify damped versus undamped curvature and the exactness of Variant B.
3. Scope the input-side and weighted-bias guarantees to the quantities and solver actually used.
4. Refine the constraint-budget explanation, parameter normalization, and Wanda wording.
5. Make the toy result reproducible and give H3 an explicit outcome criterion.
