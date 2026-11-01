"""bf16_nvfp4_gemm candidate — independent CUDA C++ W4A16 GEMM.

out = alpha * (A bf16 (m,k) @ dequant(B nvfp4 (n,k), S e4m3 (n,k/16))^T), fp32
accumulation, bf16 output written into the caller-owned `out` buffer, which is
returned unchanged (identity alias per the frozen ABI).

All numerical work runs in candidate/kernels.cu (tcgen05 + TMA tensor-core
kernels with exact fp4*e4m3 -> bf16 dequant fused into the GEMM pipeline;
mma.sync fallback). No library GEMM, no Torch numerical fallback.

Launch binding: HELD REJECTED ARM (probe72): CUDA-graph replay
(BF16NVFP4_GRAPH=1, default OFF) captured 11/11 incl. cluster launches and
was numerically exact, but measured enqueue WORSENED +1.1-1.6us (torch
replay python/pybind + key construction > saved param marshalling) and
big-grid medians/mins skewed negative -> enqueue ledger closed on its 4th
vector (memo adopted / mm_ptr slower / PDL useless / graphs slower). Kept
inert for the record; do not enable without a new mechanism model.
"""
import os
import threading

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT = None
_EXT_LOCK = threading.Lock()


def _get_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load

            sources = [os.path.join(_DIR, "kernels.cu")]
            cap = torch.cuda.get_device_capability()
            # tcgen05 + cvt.e2m1x2 need the arch-accelerated target on cc>=10.
            suffix = "a" if cap[0] >= 10 else ""
            gencode = (f"-gencode=arch=compute_{cap[0]}{cap[1]}{suffix},"
                       f"code=sm_{cap[0]}{cap[1]}{suffix}")
            _EXT = load(
                name="bf16_nvfp4_gemm_cand",
                sources=sources,
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas", "-v", gencode],
                extra_cflags=["-O3"],
                extra_ldflags=["-lcuda"],
                verbose=True,
            )
    return _EXT


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Kernel-selection override for experiments: 0 = auto (production path).
        self._cfg = int(os.environ.get("BF16NVFP4_CFG", "0"))
        # Graph-replay launch binding (probe72): off by default until adopted.
        self._use_graph = os.environ.get("BF16NVFP4_GRAPH", "0") == "1"
        self._graphs = {}          # key -> CUDAGraph | None (None = fallback)
        self._seen = {}            # key -> call count (capture on 2nd sighting)
        self._graph_dead = False   # env cannot capture; stop trying
        self._f = None
        self._mm = None
        self._validated = None

    def _full_check(self, a, b, b_descale, alpha, out):
        ok = (a.dtype is torch.bfloat16 and out.dtype is torch.bfloat16
              and a.dim() == 2 and b.dim() == 2 and b_descale.dim() == 2 and out.dim() == 2
              and b.element_size() == 1 and b_descale.element_size() == 1
              and a.is_contiguous() and b.is_contiguous()
              and b_descale.is_contiguous() and out.is_contiguous()
              and (alpha is None or (alpha.dtype is torch.float32 and alpha.numel() >= 1)))
        return ok

    def forward(self, a, b, b_descale, alpha=None, *, backend=None, out_dtype=None,
                out=None, block_size=16, enable_pdl=True):
        if self._mm is None:
            self._mm = _get_ext().mm
        if out is None:
            out = torch.empty((a.shape[0], b.shape[0]),
                              dtype=out_dtype if out_dtype is not None else torch.bfloat16,
                              device=a.device)
        if self._use_graph:
            ap = alpha.data_ptr() if isinstance(alpha, torch.Tensor) else alpha
            key = (a.data_ptr(), b.data_ptr(), b_descale.data_ptr(), ap, out.data_ptr(),
                   a.shape[0], a.shape[1], b.shape[0], self._cfg)
            try:
                g = self._graphs[key]
                if g is not None:
                    g.replay()
                    return out
            except KeyError:
                n = self._seen.get(key, 0) + 1
                self._seen[key] = n
                if n >= 2 and not self._graph_dead:
                    try:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            self._mm(a, b, b_descale, alpha, out, self._cfg)
                        self._graphs[key] = g
                        g.replay()
                        return out
                    except Exception:
                        # Permanent per-key fallback; give up globally after
                        # repeated failures (environment lacks capture support).
                        self._graphs[key] = None
                        if sum(1 for v in self._graphs.values() if v is None) >= 4:
                            self._graph_dead = True
        self._mm(a, b, b_descale, alpha, out, self._cfg)
        return out
