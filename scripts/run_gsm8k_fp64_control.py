"""Matched sequential local-70 control with FP64 ordered OBS (gflow only).

Keep the original 8192-token collection, FP32 geometry/scaling, model BF16
cast, stage chaining, and scoring. Vary curvature accumulation, factorization,
and ordered compensation precision together. This is not exact geometric OBS.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import sys

import torch

from scripts.audit_gsm8k_solver_precision import reference_run
from sparsifier.prune.rescale import local_row_scales
from sparsifier.prune.whole_neuron_io import file_hash


@torch.no_grad()
def fp64_candidate(weight, inputs, *, arm, sparsity=.7, alpha=1.,
                   deltas=None, labels=None, readout_inputs=None, pdiag=None,
                   device="cuda", progress=None):
    if arm != 'local' or pdiag is None or deltas is not None or labels is not None:
        raise ValueError('FP64 control requires the unchanged local arm')
    w = weight.to(device=device, dtype=torch.float32)
    x = inputs.to(device)
    scale, diagnostics = local_row_scales(x, readout_inputs, pdiag, alpha)
    initial = w * scale
    candidate, numerical = reference_run(initial, x, sparsity, blocksize=128,
        progress=None if progress is None else lambda row: progress(
            {'event':'fp64_ordered_block', **row}))
    # Restore the original FP32 output/division boundary before the runner's
    # unchanged BF16 checkpoint cast; only the internal solver uses doubles.
    result = candidate.to(device=device, dtype=torch.float32) / scale
    diagnostics.update(solver='row_diagonal_congruence_approximation',
        precision='float64_curvature_factor_and_ordered_OBS_FP32_geometry',
        damping=.01, blocksize=128, column_blocks=(w.shape[1]+127)//128,
        selection_damping={'requested_percdamp':.01,'effective_percdamp':.01,
            'damping':numerical['damping'],'factorization_attempts':1},
        inverse_probe_residual=numerical['inverse_probe_residual'],
        achieved_zero_fraction=float((result==0).double().mean()))
    return result.cpu().clone(), diagnostics


def main():
    from sparsifier.eval import gsm8k_sequential as runner
    from sparsifier.eval.gsm8k_batching import policy
    if policy()['batch_size'] != 32:
        raise ValueError('FP64 control requires the matched batch32 policy')
    original_prune, original_write = runner.prune_candidate, runner.write_immutable
    control = {'version':'gsm8k-local-70-fp64-ordered-v1',
        'varied':'curvature_accumulation_factorization_and_ordered_compensation_precision',
        'calibration':'unchanged_4096_C4_plus_4096_task_prompt_inputs',
        'geometry':'unchanged_FP32_readouts_atlas_trace_normalization_and_scaling',
        'output_boundary':'unchanged_FP32_unscale_then_BF16_checkpoint',
        'source_sha256':file_hash(Path(__file__)),
        'reference_sha256':file_hash(Path(__file__).with_name('audit_gsm8k_solver_precision.py')),
        'torch':torch.__version__}
    @wraps(original_write)
    def write_config(path, config):
        if (Path(path).name != 'config.json'
                or config.get('purpose') != 'development_sequential_MLP_retention'
                or config.get('arm') != 'local' or config.get('sparsity') != .7):
            raise ValueError('unexpected FP64 control configuration')
        config['solver_precision_control'] = control
        return original_write(path, config)
    runner.prune_candidate, runner.write_immutable = fp64_candidate, write_config
    try:
        runner.main()
    finally:
        runner.prune_candidate, runner.write_immutable = original_prune, original_write


if __name__ == '__main__':
    main()
