"""Bounded matrix-only diagnostic; no model, checkpoint, or queue changes."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--check-cuda-tests', action='store_true')
    args = parser.parse_args()
    import torch
    from sparsifier.eval.memory_guard import host_memory
    torch.set_num_threads(8)
    torch.manual_seed(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.check_cuda_tests:
        import pytest
        result = pytest.main(['-q', 'tests/test_pullback_contractions.py', '-k', 'cuda'])
        if result:
            raise SystemExit(result)
    if args.pipeline:
        from sparsifier.prune.downstream import tangent_pullback_diagonal, WEIGHT_CONTRACTION
        torch.manual_seed(0)
        y = torch.randn(1,128,4096,device='cuda',dtype=torch.bfloat16).requires_grad_(True)
        x = torch.randn(1,128,12288,device='cuda',dtype=torch.bfloat16)
        basis = torch.ones(1,1,device='cuda')
        def run(backend):
            return tangent_pullback_diagonal(y.sin().sum().reshape(1,1),basis,{'site':y},
                weight_inputs={'site':x},num_probes=8,generator=torch.Generator().manual_seed(7),
                contraction_backend=backend)['site']
        reference=run('cpu')
        del_value=run('auto');del del_value
        rows=[]
        for repeat in range(3):
            for backend in (('cpu','auto') if repeat%2==0 else ('auto','cpu')):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start=time.perf_counter();value=run(backend);torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                row={'repeat':repeat,'backend':backend,'seconds':elapsed,
                     'max_absolute_difference':float((value-reference).abs().max()),
                     'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3,'host':host_memory()}
                rows.append(row);print(json.dumps(row),flush=True);del value
        medians={backend:statistics.median(r['seconds'] for r in rows if r['backend']==backend)
                 for backend in ('cpu','auto')}
        report={'rows':rows,'median_seconds':medians,'speedup':medians['cpu']/medians['auto'],
                'cuda_cap_gib':2,'background_job':215,'probes':8,'tokens':128,'outputs':4096,'inputs':12288,
                'policy':WEIGHT_CONTRACTION,'purpose':'synthetic_VJP_pipeline_not_full_model'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'event':'summary',**medians,'speedup':report['speedup']}),flush=True)
        return
    g = torch.randn(128, 4096, dtype=torch.bfloat16).double()
    x = torch.randn(128, 12288, dtype=torch.bfloat16).double()
    records = []
    reference = None
    for backend, chunk in [('cpu',4096), ('cuda',512), ('cuda',2048), ('cuda',4096)]:
        gg, xx = g.to(backend), x.to(backend)
        out = torch.zeros(4096, 12288, dtype=torch.float64, device=backend)
        def work():
            out.zero_()
            for begin in range(0,4096,chunk):
                z = gg[:,begin:begin+chunk].T@xx
                z.square_()
                out[begin:begin+chunk].add_(z)
            if backend=='cuda':
                torch.cuda.synchronize()
        work()
        times=[]
        for _ in range(5):
            started=time.perf_counter();work();times.append(time.perf_counter()-started)
        if reference is None:
            reference=out.cpu().clone()
        actual=out.cpu()
        difference=float((actual-reference).abs().max())
        record={'backend':backend,'chunk_rows':chunk,'median_seconds':statistics.median(times),
            'seconds':times,'max_absolute_difference':difference,'host':host_memory(),
            'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3}
        records.append(record)
        print(json.dumps(record),flush=True)
        del out, actual, gg, xx
        torch.cuda.empty_cache()
    (args.output/'report.json').write_text(json.dumps({'records':records,'cuda_cap_gib':2,
        'background_job':215,'tokens':128,'outputs':4096,'inputs':12288,
        'dtype':'float64','purpose':'single_probe_contraction_not_end_to_end'},indent=2)+'\n')


if __name__=='__main__':
    main()
