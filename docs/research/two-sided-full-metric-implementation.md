# Two-sided coupling and replay-backed full-metric refitting

The entrywise and whole-neuron GSM8K drivers now support local input tangents
combined with downstream VJPs. Full-metric refitting uses frozen factors on
disk and matrix-free preconditioned CG at production dimensions. The explicit
Hessian implementation retains its 2,048-coordinate reference limit; the CG
path does not construct that matrix.

## Two-sided moments

For independent sequence \(s\), output probe \(p\), and local input frame
\(U_{s,t}\), backward calibration provides \(g_{s,p,t}\). The ordinary term
uses \(Z_{s,p}=\sum_t g_{s,p,t}x_{s,t}^{\mathsf T}\). The input-tangent term
has two explicitly different sequence covariances:

- `independent_tokens`: independent input coefficients at each token. Its
  factors are \(g_{s,p,t}u_{s,t,r}^{\mathsf T}\), summed as separate PSD outer
  products over tokens and tangent columns.
- `shared_directions`: a shared coefficient per tangent column across tokens.
  Its factors are \(\sum_t g_{s,p,t}u_{s,t,r}^{\mathsf T}\), preserving token
  cancellation and reinforcement before squaring.

The default is `independent_tokens`. This covariance is invariant to local
orthogonal changes of input frame. For `shared_directions`, the driver aligns
each fitted local frame to a fixed geometry-fit dictionary frame by orthogonal
Procrustes rotation. This preserves each local projector while declaring a
specific cross-token column alignment; the two sequence covariances are not
interchangeable. Library callers can supply their own aligned frames.

Input charts are fitted on `geometry_fit` readouts only, then frozen and
assigned to the selected pruning-calibration tokens. Anchor charts retain
their own fit-only provenance. Neither held-out examples nor targets are used
to fit either atlas. Atlas hashes, fit IDs, sequence structure, and source
versions are recorded.

Let \(K_b\) and \(K_i\) be the ordinary and input-tangent diagonals, each
divided by the ambient token count. With supported ambient trace
\(T_0=d_o\operatorname{tr}H_0\), use separate coefficients

\[
\gamma=\alpha T_0/\operatorname{tr}_{S}K_b,\qquad
\eta=\alpha_{\rm in}T_0/\operatorname{tr}_{S}K_i.
\]

Whole-neuron pruning rejects material geometry outside ambient support;
entrywise calibration reports its excluded mass. Unsupported coordinates are
excluded consistently from normalization, scaling, and the replayed geometric
operator. The row approximation matches
\(\operatorname{diag}H_0+\gamma K_b+\eta K_i\). Hard-mask selection still uses
the existing SparseGPT/group-OBS machinery. These are frozen-J response
penalties: they do not include downstream Jacobian variation or second
derivatives of the network.

## Scalable fixed-mask solve

`curvature_replay.py` stores content-addressed low-rank factors instead of a
dense parameter Hessian or a cache of dense weight-shaped contractions.

- Ordinary/shared-direction factors store \(G\) and \(X\), representing
  \(Z=G^{\mathsf T}X\). A Hessian action first computes the scalar
  \(\langle Z,A\rangle\), then adds that scalar times \(G^{\mathsf T}X\).
  Token chunks are combined before this scalar is used, preserving all
  cross-token terms.
- Independent-token input factors store \(G\) and \(U\), applying the sum of
  token-local PSD actions without materializing their dense outer products.
- Local output metrics store batches of input vectors and output frames.

Tensor blobs deduplicate repeated local inputs and frames. The manifest binds
factor order, coefficients, term identities, tensor hashes, and calibration
provenance. It is sealed before solving; changed manifests or tensor contents
are rejected. `DiskFactorReplay.open` can additionally require an expected
manifest hash. Replay does not depend on the current model weights, backward
graph, or RNG state.

The cache budget bounds the LRU, not total process memory. Solver vectors,
ambient curvature, current factor, and chunk workspaces are additional memory.
Large blobs are memory mapped; chunks bound GPU transfers. The disk budget is
per calibration stage, and replay directories remain available for audit.

For error \(E=W-\widehat W\), the full operator is

\[
\mathcal H(E)=EH_0+\gamma\mathcal G_b(E)+\eta\mathcal G_i(E)+D\odot E.
\]

Deleted entries remain fixed; CG solves only retained entries. Its diagonal
preconditioner includes both geometric terms and the same induced original-
coordinate ridge \(D=\delta\Lambda^2\). Entrywise refitting obtains the
effective delta from SparseGPT's successful factorization, including damping
retries, and converts it to ambient-token-mean units. Whole-neuron refitting
uses its existing selection damping owner. Explicit and CG library APIs
support both column masks and weight-shaped masks.

CG periodically recomputes the true residual. Production drivers reject
nonconverged/nonfinite candidates before checkpointing or applying weights.
The recorded full-metric objective and stationarity describe this coupled
operator; surrogate diagnostics still describe the mask-selection surrogate.
When full refitting is requested from activation-diagonal calibration, its
diagonal is recalibrated from the retained shared-weight factors, and this
effective metric is recorded.

## Experiment options

Both drivers accept `--input-alpha` and `--input-structure`. Entrywise uses
`--full-metric-refit`; whole-neuron uses `--compensation full_metric_refit`.
CG controls are `--full-metric-rtol` and `--full-metric-max-iterations`.
Whole-neuron additionally accepts `--full-metric-solver auto|cg|direct`;
`auto` selects CG above the explicit reference limit. Both expose replay
cache/disk budgets and `--replay-chunk-size`.

For example, submit a new candidate-only block-15 run through gflow:

```bash
gbatch --gpus 1 --gpu-memory 64G --memory 80G --time 08:00:00 \
  --project qwen-gsm8k --name qwen-9b-two-sided-full-b15-s30-v1 --auto-close -- \
  bash scripts/run_gsm8k_guarded.sh qwen-9b-two-sided-full-b15-s30-v1 \
  results/tangent/gsm8k/two-sided-full-b15-v1/memory.jsonl \
  .pixi/envs/gb10/bin/python -u -m sparsifier.eval.gsm8k_whole_neuron \
  --manifest results/tangent/gsm8k/data/representative-calibration-seed0-v1.json \
  --pilot-report results/tangent/gsm8k/qwen-9b-gsm8k-dense-pilot-v2/report.json \
  --c4-capture results/tangent/gsm8k/capture-v1 \
  --output results/tangent/gsm8k/two-sided-full-b15-v1 \
  --blocks 15 --metric downstream_weight_diagonal --sparsity .3 \
  --alpha .5 --input-alpha .5 --compensation full_metric_refit \
  --full-metric-solver cg --skip-behavior
```

The example is a new method configuration and must use a new output directory.
Batch-eight capture/evaluation, token budgets, seed rules, existing frozen
checkpoint loaders, and staged transaction checks remain in place.
The default atlas-fitting population is now 512 original training questions,
with 128 separate pruning-calibration questions. Controlled templates are
explicit diagnostics; see [representative calibration](representative-gsm8k-calibration.md).

## Verification

Tests compare both sequence covariances to explicit Jacobian contractions;
check replayed actions, diagonals, direct scores, and refits against dense
coupled Hessians; exercise both experiment drivers; verify damping retries,
support projection, replay integrity, cache limits, and rejection on CG
failure. A test above 2,048 weight coordinates forbids explicit Hessian
construction. CUDA variants are scheduled through gflow rather than run beside
existing GPU experiments. Large-model throughput and iteration counts still
require measurements on the chosen calibration and pruning configuration.

The affected CPU regression suite passes **172 tests**. Three CUDA cases are
queued through gflow as **job 206**, behind the existing exclusive experiment.
They check both two-sided sequence structures and the matrix-free refit above
the explicit reference limit. This is an implementation check, not a new
pruning-accuracy experiment.
