"""Independent candidate for problem 14 (fmha_fp8, L20D/SM103).

Pure CUDA/PTX implementation in fmha_fp8.cu next to this file; this module only
compiles/loads it and binds the ABI. No library numerics, no Torch fallback.

Contract notes (from problem/reference.py + metadata.json):
  * out: bf16 [Tq, H, Dv], caller-owned, returned identically.
  * lse: fp32 [Tq, H], natural-log lse divided by ln(2) (log2 units).
  * return (out, lse) iff return_lse else out (same objects).
  * sinks: fp32 [H]; total=logaddexp(base, sink); out *= exp(base-total).
  * workspace_buffer is scratch (zero-restored between calls by the harness).
"""
import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "fmha_fp8.cu"

_FP8 = torch.float8_e4m3fn


def _build_library():
    source = _SRC.read_bytes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the fmha_fp8 candidate")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if (major, minor) < (8, 9):
        raise RuntimeError(f"fp8 mma.sync requires sm_89+; got {arch}")
    if (major, minor) == (10, 3):
        arch_args = ["-gencode", "arch=compute_103a,code=sm_103a"]
    else:
        arch_args = ["-arch=" + arch, "-DFMHA_NO_TC"]
    # Cache key covers every compiled source (the .cu plus all headers it
    # includes) so a header-only edit can never reuse a stale .so.
    hasher = hashlib.sha256(source)
    for hdr in sorted(_HERE.glob("*.cuh")):
        hasher.update(hdr.name.encode())
        hasher.update(hdr.read_bytes())
    key = hashlib.sha256(hasher.digest() + arch.encode()).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / f"fmha_fp8_p14_{key}"
    root.mkdir(parents=True, exist_ok=True)
    so = root / "fmha_fp8.so"
    if not so.exists():
        cmd = [
            "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-shared",
            "-Xcompiler", "-fPIC", *arch_args, "-lineinfo",
            "-Xptxas", "-v", str(_SRC), "-o", str(so),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
        (root / "build.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError("nvcc failed:\n" + proc.stdout + proc.stderr[-4000:])
    return ctypes.CDLL(str(so))


_PREPARE_ARGTYPES = (
    [ctypes.c_void_p] * 10
    + [ctypes.c_longlong] * 7
    + [ctypes.c_int] * 13
    + [ctypes.c_float] * 2
    + [ctypes.c_longlong, ctypes.c_int, ctypes.c_longlong]
)


class Model(torch.nn.Module):
    def __init__(self, **prepared):
        super().__init__()
        lib = _build_library()
        prep = lib.fmha_fp8_prepare
        prep.argtypes = _PREPARE_ARGTYPES
        prep.restype = ctypes.c_int
        run = lib.fmha_fp8_run
        run.argtypes = [ctypes.c_int, ctypes.c_void_p]
        run.restype = ctypes.c_int
        self._prepare = prep
        self._run = run
        self._cache = None
        self._fast = None   # (query, kv_cache, out, lse, rl, scalars..., handle, dev, refs)
        self._raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)

    def _stream(self, dev):
        if self._raw_stream is not None:
            return self._raw_stream(dev)
        return torch.cuda.current_stream(dev).cuda_stream

    def forward(
        self,
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        window_left=-1,
        out=None,
        out_dtype=None,
        o_sf_scale=None,
        o_sf_vec_size=None,
        kv_layout="HND",
        enable_pdl=None,
        sinks=None,
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        causal=True,
        lse=None,
        return_lse=False,
        multi_ctas_kv_counter_buffer=None,
        use_fp16_softmax=None,
        uses_spcompress=None,
        backend="trtllm-gen",
    ):
        f = self._fast
        if (
            f is not None
            and query is f[0]
            and kv_cache is f[1]
            and out is f[2]
            and lse is f[3]
            and return_lse is f[4]
            and causal is f[5]
            and window_left == f[6]
            and max_q_len == f[7]
            and max_kv_len == f[8]
            and batch_size == f[9]
            and bmm1_scale == f[10]
            and bmm2_scale == f[11]
            and sinks is f[14]
            and workspace_buffer is f[15]
            and block_tables is f[16]
            and seq_lens is f[17]
            and cum_seq_lens_q is f[18]
        ):
            code = self._run(f[12], self._stream(f[13]))
            if code:
                raise RuntimeError(f"fmha_fp8 run failed with code {code}")
            return (out, lse) if return_lse else out
        key = (
            id(query), id(kv_cache), id(out), id(lse), id(workspace_buffer),
            id(block_tables), id(seq_lens), id(cum_seq_lens_q), id(sinks),
            batch_size, max_q_len, max_kv_len, window_left, causal, return_lse,
            bmm1_scale, bmm2_scale,
        )
        ent = self._cache
        if ent is not None and ent[0] == key:
            code = self._run(ent[1], self._stream(ent[2]))
        else:
            handle = self._prepare_call(
                query, kv_cache, workspace_buffer, block_tables, seq_lens,
                max_q_len, max_kv_len, bmm1_scale, bmm2_scale, batch_size,
                cum_seq_lens_q, window_left, out, sinks, causal, lse,
            )
            dev = query.device.index
            if dev is None:
                dev = torch.cuda.current_device()
            # Strong refs keep ids in the key unambiguous while cached.
            self._cache = (key, handle, dev,
                           (query, kv_cache, out, lse, workspace_buffer,
                            block_tables, seq_lens, cum_seq_lens_q, sinks))
            self._fast = (query, kv_cache, out, lse, return_lse, causal,
                          window_left, max_q_len, max_kv_len, batch_size,
                          bmm1_scale, bmm2_scale, handle, dev, sinks,
                          workspace_buffer, block_tables, seq_lens,
                          cum_seq_lens_q)
            code = self._run(handle, self._stream(dev))
        if code:
            raise RuntimeError(f"fmha_fp8 run failed with code {code}")
        return (out, lse) if return_lse else out

    # Engine hot path: model(...) -> forward directly, bypassing
    # nn.Module._call_impl (hook/autograd machinery unused under
    # inference_mode; ~1-2us per call matters against the ~9-10us measured
    # launch floor). Same signature, same return-identity semantics.
    __call__ = forward

    def _prepare_call(self, query, kv_cache, workspace_buffer, block_tables,
                      seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale,
                      batch_size, cum_seq_lens_q, window_left, out, sinks,
                      causal, lse):
        k, v = kv_cache
        if out is None:
            raise ValueError("out tensor is required by this candidate")
        if query.dim() != 3:
            raise ValueError("only THD 3-D query is inside the frozen shape_range")
        tq, heads, dqk = query.shape
        kv_heads, page, dvc = k.shape[1], k.shape[2], v.shape[3]
        if dvc != dqk or (dqk not in (64, 128, 256)):
            raise ValueError(f"unsupported head dims q={dqk} v={dvc}")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if query.dtype is not _FP8 or k.dtype is not _FP8 or v.dtype is not _FP8:
            raise ValueError("q/k/v must be float8_e4m3fn")
        if out.dtype is not torch.bfloat16 or not out.is_contiguous():
            raise ValueError("out must be contiguous bf16")
        if not query.is_contiguous():
            raise ValueError("query must be contiguous")
        has_lse = lse is not None
        if has_lse and (lse.dtype is not torch.float32 or not lse.is_contiguous()):
            raise ValueError("lse must be contiguous fp32")
        ks = k.stride()
        vs = v.stride()
        if workspace_buffer is not None:
            ws_ptr = workspace_buffer.data_ptr()
            ws_floats = workspace_buffer.numel() * workspace_buffer.element_size() // 4
        else:
            ws_ptr = 0
            ws_floats = 0
        handle = self._prepare(
            query.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            block_tables.data_ptr(),
            seq_lens.data_ptr(),
            cum_seq_lens_q.data_ptr(),
            out.data_ptr(),
            lse.data_ptr() if has_lse else 0,
            ws_ptr,
            sinks.data_ptr() if sinks is not None else 0,
            query.stride(0),
            ks[0], ks[1], ks[2],
            vs[0], vs[1], vs[2],
            block_tables.stride(0),
            page,
            heads,
            kv_heads,
            batch_size,
            tq,
            dqk,
            int(max_q_len),
            int(max_kv_len),
            int(window_left),
            1 if causal else 0,
            1 if has_lse else 0,
            1 if sinks is not None else 0,
            float(bmm1_scale),
            float(bmm2_scale),
            ws_floats,
            0,  # force_impl: 0 = auto dispatch (tcgen05 for D=128 dense, mma.sync else)
            k.shape[0],  # kv_pages: physical page count (TMA map bounds)
        )
        if handle < 0:
            raise RuntimeError(f"fmha_fp8 prepare failed with code {handle}")
        return handle
