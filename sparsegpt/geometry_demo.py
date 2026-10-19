"""Compare activation-only and tangent-aware calibration with ONE SparseGPT solver.

No model downloads. Accept a real layer NPZ or run the synthetic manifold demo.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .geometry import GeometrySparseGPT
from .geometry_features import BlockDictionary, synthetic
from .sparsegpt import SparseGPT


def validate_matrix(value, name, rows=None):
    if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f'{name} must have {rows} rows')


def resolve_device(device):
    """Map 'cpu'/'cuda' to a torch device; require CUDA only when requested."""
    if isinstance(device, torch.device):
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        return device
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        return torch.device('cuda')
    if device == 'cpu':
        return torch.device('cpu')
    raise ValueError("device must be 'cpu', 'cuda', or a torch.device")


@torch.no_grad()
def prune(w, h, t, alpha, sparsity, device='cpu', blocksize=128, percdamp=0.01):
    # Step 1: fresh identical teacher weights for each arm, no cross-arm leakage.
    # For large models prune sequential layers with the OPT/BLOOM hooks instead.
    # CPU and CUDA share this path: the layer, Hessian, and SparseGPT solver all
    # live on `device`. Faster: keep calibration activations already on-device.
    device = resolve_device(device)
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False, device=device)
    layer.weight.copy_(torch.as_tensor(w, dtype=layer.weight.dtype, device=device))
    pruner = SparseGPT(layer) if t is None else GeometrySparseGPT(layer)
    for start in range(0, h.shape[1], 128):
        batch = torch.as_tensor(
            h[:, start:start + 128].T,
            device=device,
            dtype=torch.float32,
        )
        pruner.add_batch(batch, None)
    if t is not None:
        # NumPy probes are chunk-copied onto the layer device inside add_geometry.
        pruner.add_geometry(t, alpha)
    # Step 5: upstream sparse solver, shared damping, order, mask selection.
    # Its thresholding may retain slightly fewer weights than the nominal budget;
    # report actual sparsity instead of claiming exact matching by requested ratio.
    with contextlib.redirect_stdout(io.StringIO()):
        pruner.fasterprune(sparsity, blocksize=blocksize, percdamp=percdamp)
    result = layer.weight.detach().cpu().numpy().copy()
    pruner.free()
    return result


def relative_mse(s, w, x):
    return float(np.linalg.norm((s-w)@x)**2/max(np.linalg.norm(w@x)**2, 1e-30))


def evaluate(s, w, h, t):
    # Step 6: independent samples/probes; geometry estimates never see test data
    # during weight selection. Better: task metrics and downstream Jacobian action.
    # Gram distortion can be computed with small output_dim² matrices; no P²
    # Gram materialization. This includes cross-chart probe pairs, not geodesics.
    u,v = (w@t).astype(np.float64),(s@t).astype(np.float64)
    uu,vv,uv=u@u.T,v@v.T,u@v.T
    norm=np.sum(uu*uu)
    error=max(0.,float(np.sum(vv*vv)+norm-2*np.sum(uv*uv)))
    nnz=int(np.count_nonzero(s))
    csr_bytes=nnz*8+(s.shape[0]+1)*4
    return dict(output_relative_mse=relative_mse(s,w,h),
                tangent_relative_mse=relative_mse(s,w,t),
                probe_gram_relative_mse=error/max(float(norm),1e-30),
                actual_sparsity=1-nnz/s.size,nnz=nnz,
                estimated_csr_bytes=csr_bytes,dense_float32_bytes=s.size*4,
                estimated_storage_ratio=s.size*4/csr_bytes)


def save_csr(path,s):
    # Step 7: export true sparse storage, not a dense tensor filled with zeros.
    # Faster deployment: hardware-compatible blocks/N:M, low-bit retained values,
    # sparse kernels and end-to-end latency profiling. NPZ headers are extra bytes.
    rows,cols=np.nonzero(s)
    np.savez(path,data=s[rows,cols].astype(np.float32),indices=cols.astype(np.int32),
             indptr=np.r_[0,np.cumsum(np.bincount(rows,minlength=s.shape[0]))].astype(np.int32),
             shape=np.array(s.shape,dtype=np.int64))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,help='NPZ: W,H_train,H_test; optional T_train,T_test')
    p.add_argument('--output',type=Path,default=Path('geometry_results'))
    p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--requested-sparsity',type=float,nargs='+',default=[.5,.75,.9,.95],
                   dest='requested_sparsity',
                   help='Target unstructured sparsities to evaluate (0 < s < 1)')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--blocksize',type=int,default=128)
    p.add_argument('--percdamp',type=float,default=.01)
    p.add_argument('--groups',type=int,default=12)
    p.add_argument('--width',type=int,default=3)
    p.add_argument('--active',type=int,default=2)
    args=p.parse_args()
    if not np.isfinite(args.alpha) or args.alpha<0 or any(not 0<s<1 for s in args.requested_sparsity):
        p.error('Require finite alpha>=0 and 0<requested-sparsity<1 (upstream threshold endpoints excluded)')
    if args.blocksize<1 or not np.isfinite(args.percdamp) or args.percdamp<=0:
        p.error('Require positive blocksize and finite positive percdamp')
    try:
        resolve_device(args.device)
    except RuntimeError as exc:
        p.error(str(exc))
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    supplied_t=test_t=None
    if args.input:
        with np.load(args.input,allow_pickle=False) as data:
            w,h,test=(np.asarray(data[k],dtype=np.float32) for k in ('W','H_train','H_test'))
            supplied_t=data['T_train'] if 'T_train' in data else None
            test_t=data['T_test'] if 'T_test' in data else None
        test_source='supplied' if test_t is not None else 'learned held-out charts'
    else:
        w,h,test,test_t=synthetic(args.seed)
        w,h,test=(x.astype(np.float32) for x in (w,h,test))
        test_source='analytic held-out oracle'
    validate_matrix(w,'W')
    validate_matrix(h,'H_train',w.shape[1]); validate_matrix(test,'H_test',w.shape[1])
    # Step 2/3: calibration-only learned block dictionary and local tangent charts.
    # Faster: reuse a trained BSF, batched GPU fits or supply T_train directly.
    model=None
    if supplied_t is None or test_t is None:
        model=BlockDictionary(args.groups,args.width,args.active,seed=args.seed).fit(h)
    t=model.tangents(h) if supplied_t is None else supplied_t
    test_t=model.tangents(test,seed=args.seed+1) if test_t is None else test_t
    validate_matrix(t,'T_train',w.shape[1]); validate_matrix(test_t,'T_test',w.shape[1])
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for sparsity in args.requested_sparsity:
        for name,probes in [('activation_only',None),('geometry_aware',t)]:
            s=prune(w,h,probes,args.alpha,sparsity,args.device,args.blocksize,args.percdamp)
            record=dict(method=name,requested_sparsity=sparsity,**evaluate(s,w,test,test_t))
            path=args.output/f'{name}_{sparsity:g}.npz'
            save_csr(path,s)
            record['artifact_bytes']=path.stat().st_size
            records.append(record)
    report=dict(seed=args.seed,alpha=args.alpha,device=args.device,
                solver='upstream SparseGPT.fasterprune',shape=list(w.shape),
                percdamp=args.percdamp,blocksize=args.blocksize,tangent_probe_count=t.shape[1],
                tangent_evaluation=test_source,results=records)
    if model is not None:
        prediction=model.mean+model.dictionary@model.encode(test)
        report['dictionary_heldout_relative_mse']=float(np.linalg.norm(test-prediction)**2/np.linalg.norm(test)**2)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
