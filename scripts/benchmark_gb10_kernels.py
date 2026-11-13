"""Benchmark Qwen3.5-9B linear-attention kernels on the GB10 (run through gflow)."""
import inspect
import json
import statistics
import torch
import transformers
import causal_conv1d
import importlib.metadata
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen


def reference(name):
    dispatcher = getattr(qwen, name)
    wrapper = inspect.getclosurevars(dispatcher.forward).nonlocals['func']
    return inspect.getclosurevars(wrapper).nonlocals['torch_function']


def timed(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def compare(name, fast, slow):
    with torch.inference_mode():
        a, b = fast(), slow()
        if isinstance(a, tuple): a, b = a[0], b[0]
        max_abs = (a.float() - b.float()).abs().max().item()
        fast_ms, slow_ms = timed(fast), timed(slow)
    print(json.dumps({'case': name, 'fast_ms': fast_ms, 'torch_ms': slow_ms,
                      'speedup': slow_ms / fast_ms, 'max_absolute_difference': max_abs}), flush=True)


def main():
    torch.manual_seed(0)
    print(json.dumps({'device':torch.cuda.get_device_name(0), 'torch':torch.__version__,
        'transformers':transformers.__version__, 'fla':importlib.metadata.version('flash-linear-attention'),
        'causal_conv1d':causal_conv1d.__version__, 'dtype':'bfloat16',
        'note':'kernel forward only, batch 1, not end-to-end; warm timings; no backward'}), flush=True)
    conv_ref = reference('causal_conv1d_fn')
    delta_ref = reference('torch_chunk_gated_delta_rule')
    recurrent_ref = reference('torch_recurrent_gated_delta_rule')
    device='cuda'; dtype=torch.bfloat16
    for length in (256, 2048):
        x=torch.randn(1,8192,length,device=device,dtype=dtype)
        w=torch.randn(8192,4,device=device,dtype=dtype)
        compare(f'causal_conv_prefill_{length}', lambda: causal_conv1d.causal_conv1d_fn(x,w,activation='silu'),
                lambda: conv_ref(x,w,activation='silu'))
        del x,w
        q=torch.randn(1,length,32,128,device=device,dtype=dtype)
        k=torch.randn_like(q); v=torch.randn_like(q)
        g=-torch.rand(1,length,32,device=device,dtype=torch.float32)
        beta=torch.rand(1,length,32,device=device,dtype=dtype)
        options=dict(g=g,beta=beta,use_qk_l2norm_in_kernel=True,output_final_state=False)
        compare(f'gated_delta_prefill_{length}',lambda:qwen.torch_chunk_gated_delta_rule(q,k,v,**options),
                lambda:delta_ref(q,k,v,**options))
        del q,k,v,g,beta
    q=torch.randn(1,1,32,128,device=device,dtype=dtype); k=torch.randn_like(q); v=torch.randn_like(q)
    g=-torch.rand(1,1,32,device=device); beta=torch.rand(1,1,32,device=device,dtype=dtype)
    state=torch.randn(1,32,128,128,device=device,dtype=torch.float32)
    options=dict(g=g,beta=beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
    compare('gated_delta_decode_1',lambda:qwen.torch_recurrent_gated_delta_rule(q,k,v,**options),
            lambda:recurrent_ref(q,k,v,**options))

if __name__ == '__main__': main()
