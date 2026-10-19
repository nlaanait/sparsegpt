"""CUDA tests for Qwen3.5 gated-delta linear attention.

The model calls ``torch_chunk_gated_delta_rule`` (prefill) and
``torch_recurrent_gated_delta_rule`` (single-token decode). Both names are
wrapped by ``use_kernel_func_from_hub_with_fallback``, which tries an HF hub
kernel, then the installed ``fla`` op, then the pure PyTorch body.

On import this module unwraps that decorator and prints one line to stderr:

    linear-attention backend: ...

Resolution on this machine: the ``kernels`` package is not installed, so
``Atlas-Inference/gdn`` (the SM 12.1 ``Qwen3_5GatedDeltaNet`` hub layer) is
not selected. ``chunk_gated_delta_rule`` itself is mapped to
``kernels-community/fla`` when ``kernels`` is present. The CUDA function
under test is therefore ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``
(Triton). The reference is the undecorated PyTorch function recovered from
``__wrapped__``. If that resolution ever falls through to PyTorch, collection
fails instead of comparing the reference with itself.

Head layouts are read from ``sparse_opt`` configs (weights are not loaded):

- ``qwen3.5-0.8b``: linear heads 16/16, head dim 128/128
- ``qwen3.5-9b``: linear heads 16/32, head dim 128/128 (grouped value heads)
- ``qwen3.5-27b``: 64 layers, hidden 5120, full-attention head dim 256;
  linear heads 16/48, head dim 128/128

FLA allows ``num_v_heads > num_k_heads`` when the value-head count is a
multiple of the key-head count. The PyTorch reference does not; Qwen repeats
key/query heads before calling it. Grouped-value cases call the kernel on the
unexpanded layout and the reference on that repeated layout.

Qwen calls both kernels with ``g=`` and ``beta=`` as keywords and with
``use_qk_l2norm_in_kernel=True``. The fused FLA signature is
``(q, k, v, g, gk, gv, beta)``, so a positional beta lands in ``gk``. The
kernel then reads that ``[B, T, H]`` tensor as ``[B, T, H, K]`` and walks off
the end of the allocation. On GB10 that is NaNs in bf16 and a wrong fp32
result. Keyword arguments match the reference. Unnormalized head-dim 128
inputs are ill-conditioned in both implementations; Qwen does not use that path.

Sequence length 2048 is the eval length and is run once at batch 1 with the
27B linear-attention head layout. Full-model activations are not allocated.
"""

from __future__ import annotations

import inspect
import json
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = {
    "0.8b": REPO_ROOT / "sparse_opt" / "qwen3.5-0.8b" / "config.json",
    "9b": REPO_ROOT / "sparse_opt" / "qwen3.5-9b" / "config.json",
    "27b": REPO_ROOT / "sparse_opt" / "qwen3.5-27b-c4-2-4-edfb7658" / "config.json",
}

# rtol, atol after casting both sides to fp32 on CPU.
# bf16/fp16 are looser than fp32. atol is a floor for unit-scale outputs and
# is multiplied by max(1, peak |reference|), capped at 1e3, so ~1e2 activations
# are not judged against a 1e-2 absolute bar. The cap keeps an exploded
# unnormalized comparison from widening the bar without limit.
_TOLERANCE = {
    torch.float32: (1e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
    torch.float16: (2e-2, 1e-2),
}
_ATOL_SCALE_CAP = 1e3

_CALL_TIMEOUT_S = 60


def _unwrap(fn):
    seen = set()
    while hasattr(fn, "__wrapped__") and id(fn.__wrapped__) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn


def _called_implementation(wrapped):
    """Return the function the hub/fla wrapper actually calls, plus the flag."""
    current = wrapped
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "__closure__", None):
            try:
                nonlocals = inspect.getclosurevars(current).nonlocals
            except TypeError:
                nonlocals = {}
            if "implementation" in nonlocals:
                return nonlocals["implementation"], bool(nonlocals.get("is_new_implementation"))
        current = getattr(current, "__wrapped__", None)
    raise RuntimeError(
        "Could not find the wrapped gated-delta implementation. "
        "Refusing to compare torch_chunk_gated_delta_rule with itself."
    )


def _backend_of(impl, ref, is_new: bool) -> str:
    if not is_new or impl is ref:
        return "torch"
    module = getattr(impl, "__module__", "") or ""
    if module.startswith("fla"):
        return "fla"
    if "modeling_qwen3_5" in module or module.startswith("transformers."):
        return "torch"
    return "hub"


def _load_layouts() -> dict[str, dict[str, int]]:
    layouts = {}
    for name, path in CONFIG_DIR.items():
        with path.open() as handle:
            cfg = json.load(handle)
        layouts[name] = {
            "num_k_heads": int(cfg["linear_num_key_heads"]),
            "num_v_heads": int(cfg["linear_num_value_heads"]),
            "k_head_dim": int(cfg["linear_key_head_dim"]),
            "v_head_dim": int(cfg["linear_value_head_dim"]),
            "hidden_size": int(cfg["hidden_size"]),
            "num_hidden_layers": int(cfg["num_hidden_layers"]),
            "head_dim": int(cfg["head_dim"]),
        }
    return layouts


def _resolve_kernels():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. These tests require the gated-delta CUDA "
            "kernel and will not compare the PyTorch reference with itself."
        )
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Qwen3.5 gated-delta reference from transformers."
        ) from exc

    chunk_impl, chunk_new = _called_implementation(torch_chunk_gated_delta_rule)
    recurrent_impl, recurrent_new = _called_implementation(torch_recurrent_gated_delta_rule)
    chunk_ref = _unwrap(torch_chunk_gated_delta_rule)
    recurrent_ref = _unwrap(torch_recurrent_gated_delta_rule)
    chunk_backend = _backend_of(chunk_impl, chunk_ref, chunk_new)
    recurrent_backend = _backend_of(recurrent_impl, recurrent_ref, recurrent_new)
    if chunk_backend == "torch" or recurrent_backend == "torch":
        raise RuntimeError(
            "CUDA gated-delta backend is unavailable "
            f"(chunk={chunk_backend}, recurrent={recurrent_backend}: "
            f"{getattr(chunk_impl, '__module__', None)}."
            f"{getattr(chunk_impl, '__name__', None)}). "
            "The PyTorch reference would be compared with itself. "
            "Install fla, or the HF hub kernel (the `kernels` package; "
            "Atlas-Inference/gdn is the SM 12.1 Qwen3_5GatedDeltaNet layer)."
        )
    return {
        "chunk_impl": chunk_impl,
        "chunk_ref": chunk_ref,
        "chunk_backend": chunk_backend,
        "recurrent_impl": recurrent_impl,
        "recurrent_ref": recurrent_ref,
        "recurrent_backend": recurrent_backend,
    }


LAYOUTS = _load_layouts()
_KERNELS = _resolve_kernels()
CUDA_CHUNK = _KERNELS["chunk_impl"]
REF_CHUNK = _KERNELS["chunk_ref"]
CUDA_RECURRENT = _KERNELS["recurrent_impl"]
REF_RECURRENT = _KERNELS["recurrent_ref"]
CHUNK_BACKEND = _KERNELS["chunk_backend"]
RECURRENT_BACKEND = _KERNELS["recurrent_backend"]

BACKEND_NOTE = (
    "linear-attention backend: "
    f"chunk={CHUNK_BACKEND} ({CUDA_CHUNK.__module__}.{CUDA_CHUNK.__name__}), "
    f"fused_recurrent={RECURRENT_BACKEND} "
    f"({CUDA_RECURRENT.__module__}.{CUDA_RECURRENT.__name__}). "
    "Reference is the undecorated PyTorch body via __wrapped__. "
    "The `kernels` package is not installed, so Atlas-Inference/gdn is not used."
)
print(BACKEND_NOTE, file=sys.stderr, flush=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@contextmanager
def _alarm(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("signal.SIGALRM is required; pytest-timeout is not installed.")

    def _handler(signum, frame):
        raise TimeoutError(f"CUDA call exceeded {seconds}s and was treated as a hang")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "nv_err_no_memory" in text


def _expand_gqa(query, key, num_v_heads: int):
    num_k_heads = query.shape[2]
    if num_k_heads == num_v_heads:
        return query, key
    if num_v_heads % num_k_heads != 0:
        raise AssertionError(
            f"num_v_heads={num_v_heads} is not a multiple of num_k_heads={num_k_heads}"
        )
    repeats = num_v_heads // num_k_heads
    return (
        query.repeat_interleave(repeats, dim=2),
        key.repeat_interleave(repeats, dim=2),
    )


def _layout(name: str) -> dict[str, int]:
    return LAYOUTS[name]


def _make_inputs(
    *,
    batch: int,
    seq: int,
    layout_name: str,
    dtype: torch.dtype,
    pattern: str,
    beta_mode: str,
    gate_mode: str,
    init_mode: str,
    seed: int,
):
    layout = _layout(layout_name)
    num_k = layout["num_k_heads"]
    num_v = layout["num_v_heads"]
    k_dim = layout["k_head_dim"]
    v_dim = layout["v_head_dim"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def _activation(shape):
        if pattern == "zeros":
            return torch.zeros(*shape, device="cuda", dtype=dtype)
        values = torch.randn(*shape, device="cuda", dtype=dtype)
        if pattern == "large":
            values = values * 100
        elif pattern == "sparse50":
            drop = torch.rand(shape, device="cuda") < 0.5
            values = values.masked_fill(drop, 0)
        elif pattern == "sparse90":
            drop = torch.rand(shape, device="cuda") < 0.9
            values = values.masked_fill(drop, 0)
        elif pattern != "normal":
            raise AssertionError(f"unknown pattern {pattern}")
        return values

    query = _activation((batch, seq, num_k, k_dim))
    key = _activation((batch, seq, num_k, k_dim))
    value = _activation((batch, seq, num_v, v_dim))

    if gate_mode == "zero":
        gate = torch.zeros(batch, seq, num_v, device="cuda", dtype=dtype)
    elif gate_mode == "very_negative":
        gate = torch.full((batch, seq, num_v), -20.0, device="cuda", dtype=dtype)
    elif gate_mode == "uniform_neg":
        gate = -torch.rand(batch, seq, num_v, device="cuda", dtype=dtype)
    else:
        raise AssertionError(f"unknown gate mode {gate_mode}")
    if torch.any(gate > 0):
        raise AssertionError("gate g must stay <= 0 (log-space decay)")

    if beta_mode == "zero":
        beta = torch.zeros(batch, seq, num_v, device="cuda", dtype=dtype)
    elif beta_mode == "one":
        beta = torch.ones(batch, seq, num_v, device="cuda", dtype=dtype)
    elif beta_mode == "uniform":
        beta = torch.rand(batch, seq, num_v, device="cuda", dtype=dtype)
    else:
        raise AssertionError(f"unknown beta mode {beta_mode}")

    if init_mode == "none":
        initial_state = None
    elif init_mode == "zeros":
        initial_state = torch.zeros(
            batch, num_v, k_dim, v_dim, device="cuda", dtype=dtype
        )
    elif init_mode == "random":
        initial_state = 0.1 * torch.randn(
            batch, num_v, k_dim, v_dim, device="cuda", dtype=dtype
        )
    else:
        raise AssertionError(f"unknown init mode {init_mode}")
    return query, key, value, gate, beta, initial_state


def _invoke(fn, query, key, value, gate, beta, initial_state, *, output_final_state, use_qk_l2norm, expand_gqa, label):
    if expand_gqa:
        query, key = _expand_gqa(query, key, value.shape[2])
    query = query.detach().clone()
    key = key.detach().clone()
    value = value.detach().clone()
    gate = gate.detach().clone()
    beta = beta.detach().clone()
    if initial_state is not None:
        initial_state = initial_state.detach().clone()
    try:
        with torch.no_grad():
            # Keyword g/beta. FLA's fused kernel is (q, k, v, g, gk, gv, beta);
            # a positional beta is read as gk and walks off the buffer.
            output, final_state = fn(
                query,
                key,
                value,
                g=gate,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
            )
            torch.cuda.synchronize()
    except TimeoutError:
        raise
    except RuntimeError as exc:
        if _is_oom(exc):
            torch.cuda.empty_cache()
            pytest.skip(f"CUDA OOM on {label}; not retrying. {exc}")
        raise
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        pytest.skip(f"CUDA OOM on {label}; not retrying. {exc}")
    return output, final_state


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, label: str) -> None:
    rtol, atol = _TOLERANCE[dtype]
    peak = float(expected.detach().abs().max().clamp(min=1))
    scale = min(peak, _ATOL_SCALE_CAP)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol * scale, msg=label)


def _assert_finite_match(
    cuda_fn,
    ref_fn,
    *,
    batch,
    seq,
    layout_name,
    dtype,
    pattern,
    beta_mode,
    gate_mode,
    use_qk_l2norm,
    output_final_state,
    init_mode,
    label,
    seed,
):
    query, key, value, gate, beta, initial_state = _make_inputs(
        batch=batch,
        seq=seq,
        layout_name=layout_name,
        dtype=dtype,
        pattern=pattern,
        beta_mode=beta_mode,
        gate_mode=gate_mode,
        init_mode=init_mode,
        seed=seed,
    )
    layout = _layout(layout_name)
    output, final_state = _invoke(
        cuda_fn,
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        expand_gqa=False,
        label=label,
    )
    expected_out = (batch, seq, layout["num_v_heads"], layout["v_head_dim"])
    assert tuple(output.shape) == expected_out, (tuple(output.shape), expected_out, label)
    output_cpu = output.detach().float().cpu()
    assert torch.isfinite(output_cpu).all(), f"non-finite CUDA output on finite inputs: {label}"
    if output_final_state:
        assert final_state is not None, label
        expected_state = (
            batch,
            layout["num_v_heads"],
            layout["k_head_dim"],
            layout["v_head_dim"],
        )
        assert tuple(final_state.shape) == expected_state, (
            tuple(final_state.shape),
            expected_state,
            label,
        )
        state_cpu = final_state.detach().float().cpu()
        assert torch.isfinite(state_cpu).all(), f"non-finite CUDA state on finite inputs: {label}"
    else:
        assert final_state is None, label
        state_cpu = None

    ref_output, ref_state = _invoke(
        ref_fn,
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        expand_gqa=True,
        label=f"{label} reference",
    )
    ref_output_cpu = ref_output.detach().float().cpu()
    _assert_close(output_cpu, ref_output_cpu, dtype, label)
    if output_final_state:
        _assert_close(
            state_cpu,
            ref_state.detach().float().cpu(),
            dtype,
            f"{label} final state",
        )
    else:
        assert ref_state is None


@pytest.fixture(scope="session", autouse=True)
def _warmup_kernels():
    """Compile the Triton specializations outside the per-test alarm."""
    print(BACKEND_NOTE, flush=True)
    with _alarm(180):
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            _invoke(
                CUDA_CHUNK,
                *_make_inputs(
                    batch=1,
                    seq=8,
                    layout_name="0.8b",
                    dtype=dtype,
                    pattern="normal",
                    beta_mode="uniform",
                    gate_mode="uniform_neg",
                    init_mode="none",
                    seed=0,
                )[:5],
                None,
                output_final_state=False,
                use_qk_l2norm=True,
                expand_gqa=False,
                label=f"warmup-{dtype}",
            )
        for layout_name in ("0.8b", "9b", "27b"):
            query, key, value, gate, beta, initial_state = _make_inputs(
                batch=1,
                seq=8,
                layout_name=layout_name,
                dtype=torch.bfloat16,
                pattern="normal",
                beta_mode="uniform",
                gate_mode="uniform_neg",
                init_mode="random",
                seed=1,
            )
            _invoke(
                CUDA_CHUNK,
                query,
                key,
                value,
                gate,
                beta,
                initial_state,
                output_final_state=True,
                use_qk_l2norm=True,
                expand_gqa=False,
                label=f"warmup-{layout_name}",
            )
            _invoke(
                CUDA_CHUNK,
                query,
                key,
                value,
                gate,
                beta,
                initial_state,
                output_final_state=True,
                use_qk_l2norm=False,
                expand_gqa=False,
                label=f"warmup-{layout_name}-nol2",
            )
        for dtype in (torch.float32, torch.bfloat16):
            _invoke(
                CUDA_RECURRENT,
                *_make_inputs(
                    batch=1,
                    seq=1,
                    layout_name="0.8b",
                    dtype=dtype,
                    pattern="normal",
                    beta_mode="uniform",
                    gate_mode="uniform_neg",
                    init_mode="none",
                    seed=2,
                )[:5],
                None,
                output_final_state=False,
                use_qk_l2norm=True,
                expand_gqa=False,
                label=f"warmup-fused-{dtype}",
            )
    yield


@pytest.fixture(autouse=True)
def _per_test_timeout():
    try:
        with _alarm(_CALL_TIMEOUT_S):
            yield
    except TimeoutError as exc:
        pytest.fail(str(exc))


def test_cuda_backend_is_distinct_from_the_reference():
    """Recorded backend is fla (Triton), not the PyTorch reference."""
    assert CUDA_CHUNK is not REF_CHUNK
    assert CUDA_RECURRENT is not REF_RECURRENT
    assert CHUNK_BACKEND in {"fla", "hub"}
    assert RECURRENT_BACKEND in {"fla", "hub"}
    assert CHUNK_BACKEND == "fla"
    assert "gated_delta_rule" in CUDA_CHUNK.__module__


_SHAPE_CASES = [
    ("tiny", 1, 8, "0.8b"),
    ("chunk_boundary", 1, 1, "0.8b"),
    ("chunk_boundary", 1, 63, "0.8b"),
    ("chunk_boundary", 1, 64, "0.8b"),
    ("chunk_boundary", 1, 65, "0.8b"),
    ("chunk_boundary", 1, 128, "0.8b"),
    ("gqa_9b", 1, 8, "9b"),
    ("gqa_27b", 1, 8, "27b"),
    ("gqa_27b_chunk", 1, 64, "27b"),
    ("gqa_27b_past_chunk", 1, 65, "27b"),
    ("eval_27b", 1, 2048, "27b"),
]


@pytest.mark.parametrize(
    ("case_name", "batch", "seq", "layout_name"),
    _SHAPE_CASES,
    ids=[f"{name}_B{batch}_T{seq}_{layout}" for name, batch, seq, layout in _SHAPE_CASES],
)
def test_chunk_shapes(case_name, batch, seq, layout_name):
    """bf16, l2-normalized, standard normal. Eval length is batch 1, not 1234."""
    _assert_finite_match(
        CUDA_CHUNK,
        REF_CHUNK,
        batch=batch,
        seq=seq,
        layout_name=layout_name,
        dtype=torch.bfloat16,
        pattern="normal",
        beta_mode="uniform",
        gate_mode="uniform_neg",
        use_qk_l2norm=True,
        output_final_state=True,
        init_mode="none",
        label=f"{case_name} B={batch} T={seq} {layout_name}",
        seed=1000 + seq + (0 if layout_name == "0.8b" else 17),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("pattern", ["normal", "zeros", "sparse50", "sparse90", "large"])
def test_chunk_dtypes_and_inputs(dtype, pattern):
    """0.8B heads, one full chunk, l2-normalized. fp16 is accepted by fla."""
    _assert_finite_match(
        CUDA_CHUNK,
        REF_CHUNK,
        batch=1,
        seq=64,
        layout_name="0.8b",
        dtype=dtype,
        pattern=pattern,
        beta_mode="uniform",
        gate_mode="uniform_neg",
        use_qk_l2norm=True,
        output_final_state=True,
        init_mode="none",
        label=f"dtype={dtype} pattern={pattern}",
        seed=2000 + {"normal": 1, "zeros": 2, "sparse50": 3, "sparse90": 4, "large": 5}[pattern],
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta_mode", ["zero", "one", "uniform"])
@pytest.mark.parametrize("gate_mode", ["zero", "very_negative"])
def test_chunk_beta_and_gate(dtype, beta_mode, gate_mode):
    """beta in {0, 1, uniform} and g in {0, -20}. g stays in log space."""
    _assert_finite_match(
        CUDA_CHUNK,
        REF_CHUNK,
        batch=1,
        seq=8,
        layout_name="0.8b",
        dtype=dtype,
        pattern="normal",
        beta_mode=beta_mode,
        gate_mode=gate_mode,
        use_qk_l2norm=True,
        output_final_state=True,
        init_mode="none",
        label=f"beta={beta_mode} g={gate_mode} {dtype}",
        seed=3000,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_qk_l2norm", [True, False])
@pytest.mark.parametrize("output_final_state", [True, False])
@pytest.mark.parametrize("init_mode", ["none", "zeros", "random"])
def test_chunk_flags(dtype, use_qk_l2norm, output_final_state, init_mode):
    """T=65 crosses the chunk size of 64. Initial state is none, zeros, or small random."""
    if not use_qk_l2norm:
        pytest.xfail(
            "head-dim 128 without L2 norm is ill-conditioned: both the kernel "
            "and the reference grow to ~1e26 and disagree. Qwen always enables "
            "use_qk_l2norm_in_kernel."
        )
    _assert_finite_match(
        CUDA_CHUNK,
        REF_CHUNK,
        batch=1,
        seq=65,
        layout_name="0.8b",
        dtype=dtype,
        pattern="normal",
        beta_mode="uniform",
        gate_mode="uniform_neg",
        use_qk_l2norm=use_qk_l2norm,
        output_final_state=output_final_state,
        init_mode=init_mode,
        label=(
            f"l2={use_qk_l2norm} final_state={output_final_state} "
            f"init={init_mode} {dtype}"
        ),
        seed=4000 + int(use_qk_l2norm) + 2 * int(output_final_state),
    )


@pytest.mark.parametrize(
    ("kind", "dtype"),
    [("nan_query", torch.bfloat16), ("inf_value", torch.bfloat16)],
)
def test_nonfinite_inputs_do_not_hang(kind, dtype):
    """NaN in query or +inf in value must raise or return within the timeout.

    Agreement with the reference is not required.
    """
    query, key, value, gate, beta, initial_state = _make_inputs(
        batch=1,
        seq=8,
        layout_name="0.8b",
        dtype=dtype,
        pattern="normal",
        beta_mode="uniform",
        gate_mode="uniform_neg",
        init_mode="none",
        seed=5,
    )
    if kind == "nan_query":
        query.view(-1)[0] = torch.nan
    elif kind == "inf_value":
        value.view(-1)[0] = float("inf")
    else:
        raise AssertionError(kind)
    try:
        output, _final_state = _invoke(
            CUDA_CHUNK,
            query,
            key,
            value,
            gate,
            beta,
            initial_state,
            output_final_state=False,
            use_qk_l2norm=True,
            expand_gqa=False,
            label=kind,
        )
    except TimeoutError:
        raise
    except Exception:
        return
    assert output is not None
    assert tuple(output.shape) == (1, 8, LAYOUTS["0.8b"]["num_v_heads"], LAYOUTS["0.8b"]["v_head_dim"])


_FUSED_CASES = [
    ("decode_0.8b", 1, "0.8b", torch.bfloat16),
    ("decode_0.8b", 1, "0.8b", torch.float32),
    ("short_0.8b", 8, "0.8b", torch.float32),
    ("decode_27b_gqa", 1, "27b", torch.bfloat16),
]


@pytest.mark.parametrize(
    ("case_name", "seq", "layout_name", "dtype"),
    _FUSED_CASES,
    ids=[f"{name}_T{seq}_{layout}_{dtype}" for name, seq, layout, dtype in _FUSED_CASES],
)
def test_fused_recurrent_short_sequences(case_name, seq, layout_name, dtype):
    """Cheap fused recurrent calls: decode length and a short prefill, real head dims."""
    _assert_finite_match(
        CUDA_RECURRENT,
        REF_RECURRENT,
        batch=1,
        seq=seq,
        layout_name=layout_name,
        dtype=dtype,
        pattern="normal",
        beta_mode="uniform",
        gate_mode="uniform_neg",
        use_qk_l2norm=True,
        output_final_state=True,
        init_mode="none",
        label=f"fused {case_name} T={seq} {layout_name} {dtype}",
        seed=5000 + seq,
    )
