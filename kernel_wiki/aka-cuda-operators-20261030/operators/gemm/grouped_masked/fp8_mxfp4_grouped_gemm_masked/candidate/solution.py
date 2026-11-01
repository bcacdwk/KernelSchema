"""fp8_mxfp4_grouped_gemm_masked — independent candidate for L20D (SM103).

Pure CUDA C++/PTX implementation (candidate/kernel.cu): tcgen05 block-scaled
MMA (kind::mxf8f6f4, UE8M0 scale factors staged in TMEM via UTCCP), TMA loads
with hardware fp4->8-bit container unpack (16U4_ALIGN16B), persistent masked
grouped scheduler with swap-AB dynamic UMMA-N shrink, STSM/TMA-store epilogue,
and float32->ue8m0 scale conversion FUSED into the measured kernel (no separate
repack pass). No GEMM/attention library, DSL or Torch numerical fallback; this
module only compiles, loads and binds the ABI as permitted.

Mechanism references (studied read-only; implementation written independently):
DeepGEMM e1f418c2a4f2 (frozen production revision: sm100_fp8_fp4_gemm_1d1d
pipeline structure, SF UTCCP transpose layout, descriptor arithmetic, masked
scheduler semantics); PTX ISA tcgen05/mbarrier/bulk-tensor families; CUTLASS
cute SM100 descriptor bit layouts (InstrDescriptorBlockScaled / SmemDescriptor).
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch

_KERNEL_SRC = Path(__file__).resolve().parent / "kernel.cu"
# Probe-validated toolchain path for tcgen05 on this image (see experiments/probe2).
# FMGGM_EXTRA_FLAGS: same-job A/B of compile-time kernel variants (e.g.
# "-DFMGGM_SF_PF=0"); included in the build tag so each variant caches its own .so.
_BUILD_FLAGS = ["-gencode", "arch=compute_103a,code=sm_103a"] + \
    os.environ.get("FMGGM_EXTRA_FLAGS", "").split()


def _build_once():
    src_bytes = _KERNEL_SRC.read_bytes()
    tag = (hashlib.sha256(src_bytes).hexdigest()[:16] + "-"
           + hashlib.sha256(" ".join(_BUILD_FLAGS).encode()).hexdigest()[:8])
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", tempfile.gettempdir())) / f"fmggm-{tag}"
    root.mkdir(parents=True, exist_ok=True)
    so = root / "fmggm.so"
    done = root / "build.done"
    if not done.exists():
        tmp = root / f"fmggm.{os.getpid()}.so"
        cmd = ["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", *_BUILD_FLAGS,
               str(_KERNEL_SRC), "-o", str(tmp), "-lcuda",
               "-Xcompiler", "-fPIC", "-shared", "--ptxas-options=-v"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        (root / "build.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"fmggm build failed rc={proc.returncode}:\n{(proc.stdout + proc.stderr)[-6000:]}")
        os.replace(tmp, so)
        done.write_text("ok\n")
    return so


class _Ext:
    _fn = None

    @classmethod
    def get(cls):
        if cls._fn is None:
            lib = ctypes.CDLL(str(_build_once()))
            fn = lib.fmggm_launch
            fn.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_longlong] * 5
                           + [ctypes.c_void_p])
            fn.restype = ctypes.c_int
            cls._fn = fn
        return cls._fn


def _raw_stream(device_index):
    getter = getattr(torch.cuda, "_cuda_getCurrentRawStream", None)
    if getter is not None:
        return getter(device_index)
    return torch.cuda.current_stream(device_index).cuda_stream


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._fn = _Ext.get()
        self._key = None
        self._call_args = None
        self._device_index = None

    @staticmethod
    def _validate(aq, asf, bq, bsf, d, masked_m):
        if aq.dtype != torch.float8_e4m3fn:
            raise TypeError(f"a payload must be float8_e4m3fn, got {aq.dtype}")
        if bq.dtype not in (torch.int8, torch.uint8):
            raise TypeError(f"b payload must be packed fp4 int8/uint8, got {bq.dtype}")
        if asf.dtype != torch.float32 or bsf.dtype != torch.float32:
            raise TypeError("scale factors must be float32")
        if d.dtype != torch.bfloat16:
            raise TypeError(f"d must be bfloat16, got {d.dtype}")
        if masked_m.dtype != torch.int32:
            raise TypeError(f"masked_m must be int32, got {masked_m.dtype}")
        for name, t in (("a", aq), ("a_sf", asf), ("b", bq), ("b_sf", bsf),
                        ("d", d), ("masked_m", masked_m)):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if not t.is_cuda:
                raise ValueError(f"{name} must be on CUDA")
        e, m, k = aq.shape
        n = bq.shape[1]
        if asf.shape != (e, m, k // 32) or bsf.shape != (e, n, k // 32):
            raise ValueError(f"scale shapes {tuple(asf.shape)}/{tuple(bsf.shape)} "
                             f"do not match payloads for gran_k=32")
        if bq.shape != (e, n, k // 2) or d.shape != (e, m, n):
            raise ValueError("operand shape mismatch")
        if masked_m.shape != (e,):
            raise ValueError("masked_m must have one entry per group")
        if k % 128 != 0:
            raise ValueError(f"k={k} must be a multiple of 128 (frozen shape_range guarantees this)")

    def forward(self, a, b, d, masked_m, expected_m, recipe=None, recipe_a=None,
                recipe_b=None, compiled_dims="nk", disable_ue8m0_cast=False):
        aq, asf = a
        bq, bsf = b
        key = (aq.data_ptr(), asf.data_ptr(), bq.data_ptr(), bsf.data_ptr(),
               d.data_ptr(), masked_m.data_ptr(), aq.shape, bq.shape, d.shape,
               expected_m)
        if key != self._key:
            self._validate(aq, asf, bq, bsf, d, masked_m)
            e, m, k = aq.shape
            n = bq.shape[1]
            self._call_args = (aq.data_ptr(), asf.data_ptr(), bq.data_ptr(),
                               bsf.data_ptr(), d.data_ptr(), masked_m.data_ptr(),
                               e, m, n, k, int(expected_m))
            self._device_index = d.device.index
            self._key = key
        rc = self._fn(*self._call_args, _raw_stream(self._device_index))
        if rc != 0:
            raise RuntimeError(f"fmggm_launch failed rc={rc}")
        return None
