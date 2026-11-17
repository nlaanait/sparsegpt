"""Full recorded-layer FP32/FP64 ordered-SparseGPT comparison (gflow only).

Uses the same inputs, initial scaled weights, mask rule, relative ridge, and
block size. FP64 remains ordered OBS, not an exact coupled geometric solve.
No language model is loaded and no experiment checkpoint is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from sparsifier.data.gsm8k import digest,write_immutable
from sparsifier.eval.gsm8k import atomic_json
from sparsifier.eval.gsm8k_geometry import load_capture
from sparsifier.eval.gsm8k_pruning import calibration
from sparsifier.eval.memory_guard import configure_cuda
from sparsifier.prune.factor_atlas import FactorAtlas,projector_diagonals
from sparsifier.prune.rescale import local_row_scales
from sparsifier.prune.sparsegpt import SparseGPT
from sparsifier.prune.whole_neuron_io import file_hash


def inverse_probe_residual(matrix,upper):
    """Probe A(U.T U)v=v in double precision without forming a second inverse."""
    generator = torch.Generator(device='cpu').manual_seed(707)
    v = (2*torch.randint(0,2,(matrix.shape[0],4),generator=generator)-1).to(matrix.device,dtype=torch.double)
    a,u = matrix.double(),upper.double()
    residual = a @ (u.T @ (u @ v))-v
    relative = residual.norm(dim=0)/v.norm(dim=0)
    return {'max':float(relative.max()),'mean':float(relative.mean()),'probes':4}


class AuditedSparseGPT(SparseGPT):
    def _invert_hessian(self,hessian,percdamp,*,progress=None):
        upper = super()._invert_hessian(hessian,percdamp,progress=progress)
        actual = hessian.clone()
        actual.diagonal().add_(self.damping_diagnostics['damping'])
        actual = (actual+actual.T)/2
        self.inverse_residual = inverse_probe_residual(actual,upper)
        return upper


@torch.no_grad()
def source_run(weight,inputs,sparsity=.7,blocksize=128):
    layer = torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,
                            device=weight.device,dtype=torch.float32)
    layer.weight.copy_(weight.float())
    solver = AuditedSparseGPT(layer);solver.add_batch(inputs)
    solver.fasterprune(sparsity,blocksize=blocksize,percdamp=.01,mask_weight=torch.ones_like(weight))
    result = layer.weight.detach().cpu().clone()
    diagnostics = {'inverse_probe_residual':solver.inverse_residual,'damping':solver.damping_diagnostics}
    solver.free()
    return result,diagnostics


@torch.no_grad()
def reference_run(weight,inputs,sparsity=.7,blocksize=128,progress=None):
    """Separate double-precision implementation of the ordered block algorithm."""
    x,w = inputs.double(),weight.double().clone()
    h = 2*(x.T @ x)
    dead = h.diag()==0
    h[dead,dead] = 1;w[:,dead] = 0
    ridge = .01*h.diag().mean()
    h.diagonal().add_(ridge);h = (h+h.T)/2
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    upper = torch.linalg.cholesky((inverse+inverse.T)/2,upper=True)
    del inverse
    residual = inverse_probe_residual(h,upper)
    del h,x
    for start in range(0,w.shape[1],blocksize):
        stop = min(start+blocksize,w.shape[1])
        current = w[:,start:stop].clone()
        u = upper[start:stop,start:stop]
        scores = current.square()/u.diag().square()[None,:]
        mask = torch.zeros_like(scores,dtype=torch.bool)
        count = int(scores.numel()*sparsity)
        if count:mask.flatten()[torch.topk(scores.flatten(),count,largest=False).indices] = True
        errors = torch.zeros_like(current);committed = torch.empty_like(current)
        for j in range(stop-start):
            original = current[:,j].clone()
            chosen = original.masked_fill(mask[:,j],0)
            committed[:,j] = chosen
            error = (original-chosen)/u[j,j]
            current[:,j:] -= error[:,None]*u[j,j:][None,:]
            errors[:,j] = error
        w[:,start:stop] = committed
        w[:,stop:] -= errors @ upper[start:stop,stop:]
        if progress is not None and (start//blocksize)%8==0:
            progress({'columns_done':stop,'columns_total':w.shape[1]})
    return w.cpu(),{'inverse_probe_residual':residual,'damping':float(ridge)}


@torch.no_grad()
def reconstruction(weight,candidate,inputs,ridge):
    x = inputs.double();delta = weight.to(x).double()-candidate.to(x).double()
    squared = 0.
    for start in range(0,weight.shape[0],256):
        squared += float((delta[start:start+256] @ x.T).square().sum())
    norm = float(delta.square().sum())
    return {'mean_squared_output_norm':squared/len(x),
            'common_scaled_surrogate_half_quadratic':.5*(squared/len(x)+ridge*norm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('scheduler-provided CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(4)
    capture,index,_,x,y = load_capture(args.capture)
    ambient,_,use = calibration(args.capture,index,x)
    weight = torch.from_numpy(np.load(args.capture/'weight.npy'))
    frames,_,_ = FactorAtlas.load(args.atlas).frames(y[use])
    pd = projector_diagonals(frames)
    args.output.mkdir(parents=True,exist_ok=True)
    config = {'purpose':'recorded_full_layer_precision_audit_not_model_wide_retention',
        'capture_sha256':digest(capture),'atlas_sha256':file_hash(args.atlas),
        'weight_sha256':file_hash(args.capture/'weight.npy'),
        'ambient_sha256':hashlib.sha256(memoryview(ambient.numpy()).cast('B')).hexdigest(),
        'source_sha256':file_hash(Path(__file__)),'sparsity':.7,'blocksize':128,'percdamp':.01,
        'weight_shape':list(weight.shape),'ambient_shape':list(ambient.shape),
        'modes':['ambient','local_alpha1'],'scales':'fixed_FP32_for_both_solver_precisions',
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'torch':torch.__version__,
        'sources':{p:file_hash(Path('src/sparsifier')/p) for p in (
            'prune/sparsegpt.py','prune/rescale.py','eval/gsm8k_pruning.py')}}
    write_immutable(args.output/'config.json',config)
    path = args.output/'report.json'
    if path.exists() and __import__('json').loads(path.read_text()).get('complete'):return
    memory = configure_cuda(16.)
    device = 'cuda';inputs = ambient.to(device)
    report = {'config_sha256':digest(config),'complete':False,'memory':memory,'modes':[]}
    for mode in config['modes']:
        if mode=='ambient':scale = torch.ones_like(weight)
        else:
            scale,_ = local_row_scales(inputs,torch.from_numpy(x[use]),pd,1.)
            scale = scale.cpu()
        # Match the actual FP32 coordinate transformation in both solvers.
        initial = (weight*scale).to(device)
        source,sd = source_run(initial,inputs)
        reference,rd = reference_run(initial,inputs,progress=lambda row:print({'mode':mode,**row},flush=True))
        common_ridge = rd['damping']/(2*len(inputs))
        source_metric = reconstruction(initial,source,inputs,common_ridge)
        reference_metric = reconstruction(initial,reference,inputs,common_ridge)
        a,b = source/scale,reference/scale.double()
        a16,b16 = a.to(torch.bfloat16).double(),b.to(torch.bfloat16).double()
        row = {'mode':mode,'source':sd,'reference':rd,
            'mask_changed_fraction':float(((source==0)!=(reference==0)).double().mean()),
            'source_zero_fraction':float((source==0).double().mean()),
            'reference_zero_fraction':float((reference==0).double().mean()),
            'relative_candidate_difference':float((a.double()-b).norm()/b.norm()),
            'relative_bf16_candidate_difference':float((a16-b16).norm()/b16.norm()),
            'source_objective':source_metric,'reference_objective':reference_metric,
            'scaled_surrogate_ratio':source_metric['common_scaled_surrogate_half_quadratic']/reference_metric['common_scaled_surrogate_half_quadratic']}
        report['modes'].append(row);atomic_json(path,report);print(row,flush=True)
        del initial,source,reference,a,b,a16,b16,scale
        torch.cuda.empty_cache()
    report['complete'] = True;atomic_json(path,report)


if __name__ == '__main__':main()
