"""Autonomous candidate for the frozen nvfp4_fused_moe contract (L20D, sm_103).

Python side only allocates, inspects public metadata, compiles/loads the CUDA
object and binds its ABI. All numerical work lives in candidate/nvfp4_moe.cu
(own CUDA C++/PTX: NVFP4 decode, grouped bf16 tensor-core GEMMs with FP32
accumulation, clamp/activation, NVFP4 block-16 QDQ, per-route BF16
materialization and the routing-weighted FP32 reduction).

The caller-owned `output` tensor is written in place and returned inside a
list, matching the reference's identity/alias contract exactly.
"""
import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "nvfp4_moe.cu"

_ABI_FIELDS = [
    "ids", "scores", "hid_q", "hid_sf", "w1_q", "w1_sf", "w2_q", "w2_sf",
    "alpha", "beta", "limit", "out",
    "counts", "fill", "offs", "toffs", "ptok", "pslot", "pexp", "route_of",
    "texp", "tm0", "hid_bf", "fc1", "act", "rout",
    "T", "K", "E", "H", "I", "Rmax", "max_tiles", "act_type", "has_limit",
    "bm", "cfg", "stream", "cfg2", "part", "sk", "hidq_g", "hidsf_g",
    "cnt", "cnt_n", "acc32", "small_thresh", "zflag", "zflag_host",
]


class _ABI(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in _ABI_FIELDS]


def _gencode():
    """Architecture-specific code generation for the actual device.

    `-arch=sm_103a` alone silently degrades to `.target sm_103` with this
    toolchain (measured on job dv_6a6795c6bfc8), which drops every
    architecture-specific instruction, so the explicit -gencode form is used.
    """
    major, minor = torch.cuda.get_device_capability()
    base = f"{major}{minor}"
    return ["-gencode", f"arch=compute_{base}a,code=sm_{base}a"]


def _arch_tag():
    """Filesystem-safe label for the target architecture (no '=' or ',')."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}a"


def _build_dir():
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.environ.get("CUDA_CACHE_PATH")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "nvfp4-moe-candidate")
    root = Path(root) / "nvfp4_fused_moe"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build(verbose=False):
    """Compile (or reuse) the shared object; returns a loaded CDLL."""
    source = _SRC.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    root = _build_dir()
    name = f"nvfp4_moe_{digest}_{_arch_tag()}.so"
    library = root / name
    if not library.exists():
        tmp = root / f".tmp-{os.getpid()}-{name}"
        cmd = ([os.environ.get("CUDA_NVCC", "/usr/local/cuda/bin/nvcc"),
                "-O3", "-std=c++17", "--ptxas-options=-v",
                *_gencode(), "-shared", "-Xcompiler", "-fPIC",
                str(_SRC), "-o", str(tmp)])
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "2")
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=900, env=env)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        (root / f"build-{digest}.log").write_text(text)
        artifact = os.environ.get("BATCH_ARTIFACT_DIR")
        if artifact:
            try:
                Path(artifact).mkdir(parents=True, exist_ok=True)
                (Path(artifact) / f"nvcc-{digest}.log").write_text(text)
            except OSError:
                pass
        if proc.returncode != 0:
            head, tail = text[:4000], text[-4000:]
            raise RuntimeError("nvcc build failed (cmd: " + " ".join(cmd) + ")\n"
                               "--- head ---\n" + head + "\n--- tail ---\n" + tail)
        os.replace(tmp, library)
    handle = ctypes.CDLL(str(library))
    handle.moe_init.argtypes = [ctypes.c_int64]
    handle.moe_init.restype = ctypes.c_int
    handle.moe_run.argtypes = [ctypes.c_void_p]   # raw int address per call
    handle.moe_run.restype = ctypes.c_int          # (byref also converts)
    handle.moe_zero_probe.argtypes = [ctypes.c_int64] * 16
    handle.moe_zero_probe.restype = ctypes.c_int
    return handle


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _build()
    return _LIB


def _ptr(value):
    """Plain int for the ctypes struct field (0 for an absent optional tensor)."""
    return 0 if value is None else value.data_ptr()


# Host fast-path bindings. torch.cuda.current_stream() costs ~3.2us per call
# (Stream object construction) against 0.04us for the raw C getter that
# torch.compile's own wrappers use (measured on L20D, job dv_cfce00bc6b51:
# forward host 18.5us of which ~14us is python-side); the official eager
# protocol exposes the whole host enqueue inside the measured window on
# small shapes. Same semantics: current stream of the resolved device.
_raw_stream = torch._C._cuda_getCurrentRawStream
_env_get = os.environ.get

# J1: identity-verified single-entry fast path, read once at import.
#   0 -> the 37-item id-tuple key and its dict lookup (the adopted bytes)
#   1 -> verify the ABI-relevant arguments by identity against one cached tuple,
#        write abi.stream only when it actually changed, and store the
#        diagnostics-only _last_abi with object.__setattr__ instead of going
#        through nn.Module.__setattr__.
# Host-side only: no device byte changes, so both variants share one .so and
# the outputs are identical by construction. The 21 arguments that the lite
# path does not verify (routing_bias, the three bias/scale tensors and their
# gate/output scalars, n_group, topk_group, local_expert_offset,
# local_num_experts, routed_scaling_factor, routing_method_type, do_finalize,
# enable_pdl, per_token_scale, tune_max_num_tokens, gemm1_lora_delta,
# valid_hidden_size, valid_intermediate_size, num_fused_shared_experts,
# hidden_states_scale_layout) have no ABI field and are read nowhere else in
# this file -- each appears only in the signature, the key and (for eight of
# them) the keepalive tuple -- so no computed value can depend on them. If a
# future change makes the device side depend on one, it must be added to the
# verified set below.
_HOSTLITE = int(_env_get("NVFP4_MOE_HOSTLITE", "1") or 1)

# The fifteen arguments the cached call verifies, in the order forward()
# publishes them in _hot[0]: the eleven objects whose data pointers enter the
# ABI, then the four scalars that do. This tuple is the single source of truth
# for both the python path and nvfp4_fast.c, which reads _hot[0] positionally
# against the names it is handed by bind(). `_env_snap` is published at index 15
# and is written exactly once, in __init__, before any entry can exist, so
# neither path needs to re-read it.
_HOT_KEYS = ("topk_ids", "hidden_states", "hidden_states_scale",
             "gemm1_weights", "gemm1_weights_scale", "gemm2_weights",
             "gemm2_weights_scale", "gemm1_alpha", "gemm1_beta",
             "gemm1_clamp_limit", "output", "activation_type", "num_experts",
             "top_k", "intermediate_size")

# J5: run the cached call from C behind a descriptor assigned to Model.__call__.
#   0 -> the J1 python path (unchanged)
#   1 -> nvfp4_fast.Call, when that extension built and bound successfully
# What it removes is the ~0.75 us CPython spends binding 36 keyword arguments to
# 36 named parameters and the ~0.2 us of ctypes marshalling around moe_run: a C
# callee gets the kwargs dict by reference and looks up only the fifteen names it
# verifies. Measured on these bytes (job dv_91fe853ef863, experiments/run_j5.sh, two
# rounds, alternating processes, four digests bit-exact): host per call
# 4.76 -> 3.46 us on fixed_0 and 4.57 -> 3.44 on fixed_8 while dev_run, events
# around a bare moe_run, stayed at 29.2 -> 29.5 and 14.3 -> 13.7, and end to end
# 1.0163x / 1.0187x / 1.0061x / 1.0310x on fixed_0 / 1 / 2 / 8. What is left of
# the 3.45 us is 2.95-3.03 us inside moe_run, so the host path is now ~85%
# cudaGraphLaunch and there is nothing further to remove short of graph nodes. Nothing about the check changes -- same objects, same identity test,
# same fallback to forward() on anything unexpected -- and the whole thing is
# optional: no Python.h, no host compiler, a failed compile or a failed import
# all leave _FASTMOD None and the candidate on the J1 path.
_HOSTCFAST = int(_env_get("NVFP4_MOE_CFAST", "1") or 1)

# CFAST2: inside the C cached call, invoke moe_run through its raw C address
# (published as _hot[7]) instead of the ctypes function object (~0.2 us of
# marshalling per call, J1's measurement) and skip the per-call _last_abi
# setattr (~0.10 us) -- publish already stores that same ABI object on the
# instance, and republishes whenever the ABI is rebuilt. Unlike HOSTFAST's
# getenv caching (officially neutral: the engine sanitizes its environ,
# dv_421566e63cb5), these savings are environ-independent. Read once at
# import, passed to bind(); =0 restores the exact J5 behavior for A/B.
# Host-side only: no device bytes, digests identical either way.
_CFAST2 = int(_env_get("NVFP4_MOE_CFAST2", "1") or 1)


def _build_fast():
    """Compile and import nvfp4_fast.c, or return None. Cached next to the .so.

    Every failure mode records why in _FASTERR and leaves the candidate on the
    J1 python path: a silently absent extension would otherwise make an A/B of
    it read as "the axis is closed".
    """
    global _FASTERR
    src = _HERE / "nvfp4_fast.c"
    if not src.is_file():
        _FASTERR = "no nvfp4_fast.c"
        return None
    import shutil
    import sysconfig
    cc = None
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            cc = cand
            break
    inc = sysconfig.get_paths().get("include") or ""
    if cc is None:
        _FASTERR = "no host compiler (CC/cc/gcc/clang)"
        return None
    if not os.path.isfile(os.path.join(inc, "Python.h")):
        _FASTERR = f"no Python.h under {inc}"
        return None
    try:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        root = _build_dir()
        out = root / f"nvfp4_fast_{digest}.so"
        if not out.exists():
            tmp = root / f".tmpfast-{os.getpid()}-{out.name}"
            cmd = [cc, "-O2", "-shared", "-fPIC", "-I", inc, str(src),
                   "-o", str(tmp)]
            proc = subprocess.run(cmd, text=True, capture_output=True,
                                  timeout=300)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            artifact = os.environ.get("BATCH_ARTIFACT_DIR")
            if artifact:
                try:
                    Path(artifact).mkdir(parents=True, exist_ok=True)
                    (Path(artifact) / f"cc-{digest}.log").write_text(text)
                except OSError:
                    pass
            if proc.returncode != 0:
                _FASTERR = f"cc exit {proc.returncode}: {text[-400:]}"
                return None
            os.replace(tmp, out)
        import importlib.util
        spec = importlib.util.spec_from_file_location("nvfp4_fast", str(out))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.bind(_HOT_KEYS, _raw_stream, _CFAST2)
        stats = mod.stats()
        if not stats[0]:
            _FASTERR = "extension imported but bind() reported unbound"
            return None
        return mod
    except Exception as exc:
        _FASTERR = repr(exc)
        return None


_FASTERR = ""
_FASTMOD = _build_fast() if (_HOSTCFAST and _HOSTLITE) else None
if _FASTMOD is None and not _FASTERR:
    _FASTERR = ("disabled by env" if not (_HOSTCFAST and _HOSTLITE)
                else "unknown")


# Tile ladder: (BM, BN, BK, WM, WN, flops-per-shared-byte, threads).
# Rungs 0-2 use a 32x64 warp tile (MT=2, NT=8) which needs ~10.7 FLOP per byte
# of ldmatrix traffic; rung 3 trades that ratio for block-level parallelism,
# which is what tiny batches need to reach HBM bandwidth.
# Must match launch_gemm() in candidate/nvfp4_moe.cu.
CFGS = [(128, 128, 64), (64, 256, 64), (32, 256, 64), (16, 32, 256),
        (128, 128, 64),   # rung 4: tcgen05, 128 tokens/CTA, GEMM1 N=128
        (128, 256, 64),   # rung 5: tcgen05, 128 tokens/CTA, GEMM1 N=256
        (256, 128, 64),   # rung 6: tcgen05, 2x128 tokens share one decoded tile
        (128, 128, 256),  # rung 7: tcgen05 block-scaled NVFP4, operands packed
        (128, 128, 256),  # rung 8: rung 7 with a 3-buffer copy pipeline
        (128, 128, 256),  # rung 9: rung 7 with a 4-buffer copy pipeline
        (128, 64, 256),   # rung 10: N=64, 2 buffers -> 2 CTAs/SM
        (128, 64, 256),   # rung 11: N=64, 3 buffers
        (128, 128, 256),  # rung 12: rung 8 with the operand tiles staged by TMA
        (128, 64, 256),   # rung 13: rung 10 with TMA-staged 64-row weight boxes
        (128, 64, 256),   # rung 14: rung 13 + TMA gather4 activation operand
        (128, 128, 256),  # rung 15: rung 12 + TMA gather4 activation operand
        (128, 64, 256),   # rung 16: rung 13 with a 3-buffer payload ring and a
                          #          2-slot SF ring + SF register prefetch
        (128, 64, 256),   # rung 17: rung 13 + SF register prefetch (SFS=2)
        (128, 128, 256),  # rung 18: rung 12 + SF register prefetch
        (128, 128, 256),  # rung 19: GEMM2 = rung 12 + SF register prefetch
        (128, 64, 256),   # rung 20: rung 17 + split-K x4 (fp32 partials)
        (128, 64, 256),   # rung 21: rung 17 + split-K x2
        (128, 128, 256),  # rung 22: reserved
        (128, 128, 256),  # rung 23: GEMM2 flat-stream across item boundaries
        (128, 64, 256),   # rung 24: rung 17 + TMA gather4 activation operand
        (128, 64, 256),   # rung 25: BN64, 3 payload buffers, single SF slot
        (128, 64, 256),   # rung 26: warp-specialized GEMM1 (12 warps, 1 CTA/SM)
        (128, 64, 256),   # rung 27: rung 17 + PRE-GATHERED A (k_gather_a +
                          #          one 2-D TMA box per stage instead of the
                          #          1024-request cp.async gather)
        (128, 64, 256),   # rung 28: rung 21 (SK2) + pre-gathered A
        (128, 64, 256),   # rung 29: rung 20 (SK4) + pre-gathered A
        (128, 64, 256),   # rung 30: rung 27 + three-buffer payload ring
        (128, 64, 256),   # rung 31: rung 29 (SK4) + fused split-K activation
        (128, 64, 256),   # rung 32: rung 28 (SK2) + fused split-K activation
        (128, 64, 256),   # rung 33: rung 27 + single-warp mainloop
        (128, 64, 256),   # rung 34: rung 28 + single-warp mainloop
        (128, 64, 256),   # rung 35: rung 29 + single-warp mainloop
        (128, 128, 256),  # rung 36: rung 19 GEMM2 + single-warp mainloop
        (128, 64, 256),   # rung 37: rung 29 with an eight-way K split
        (128, 128, 256),  # rung 38: rung 19 GEMM2 with SFS=2 + SFPRE2
        (128, 64, 256),   # rung 39: rung 27 + SFPRE2
        (128, 64, 256),   # rung 40: rung 17 + SFPRE2 (cp.async A)
        (128, 64, 256),   # rung 41: rung 27 + branchless quantize guard
        (128, 64, 256),   # rung 42: rung 28 + k_act_packed<BRLESS>
        (128, 64, 256),   # rung 43: rung 29 + k_act_packed<BRLESS>
        (128, 128, 256),  # rung 44: rung 19 GEMM2 + SF via cp.async (SFCP)
        (128, 64, 256),   # rung 45: rung 41 GEMM1 + SF via cp.async (SFCP)
        (128, 128, 256),  # rung 46: rung 44 GEMM2 + REDF fused routing
                          #          reduction (fp32 atomics into [T,H] acc,
                          #          k_convert final rounding; no rout buffer)
        (128, 128, 256),  # rung 47: rung 19 GEMM2 + REDF
        (128, 128, 256),  # rung 48: GEMM1 BN=128 + pre-gathered A + SFPRE2
                          #          + BRLESS, 3-stage ring, 1 CTA/SM
        (128, 128, 256),  # rung 49: rung 48 with a 2-stage ring
        (128, 64, 256),   # rung 50: GEMM1 rung 41 tiling + PIPE mainloop
                          #          (MMA issue decoupled from prior-stage
                          #          retire via ping-pong commit barriers)
        (128, 64, 256),   # rung 51: rung 50 with a 3-stage payload ring
        (128, 128, 256),  # rung 52: GEMM2 rung 38 tiling + PIPE mainloop
        (128, 128, 256),  # rung 53: rung 52 with a 2-stage payload ring
        (128, 64, 256),   # rung 54: rung 42 (SK2) + PIPE mainloop
        (128, 64, 256),   # rung 55: rung 43 (SK4) + PIPE mainloop
        (128, 64, 256),   # rung 56: WS3 warp-specialized GEMM1 (224 thr)
        (128, 128, 256),  # rung 57: WS3 warp-specialized GEMM2
        (128, 64, 256),   # rung 58: rung 42 (SK2) with the WS3 role split
        (128, 64, 256),   # rung 59: rung 43 (SK4) with the WS3 role split
        (128, 64, 256),   # rung 60: reserved (keeps CFGS indices == rung ids)
        (128, 64, 256),   # rung 61: WS3 GEMM2 BN=64 + TMEM accumulator
                          #          ping-pong, 3-stage ring
        (128, 64, 256),   # rung 62: rung 61 with a 4-stage ring
        (128, 64, 256),   # rung 63: WS3 GEMM1 fused + TMA gather4 A operand
        (128, 64, 256),   # rung 64: rung 63 tiling with SK2
        (128, 64, 256),   # rung 65: DIAG WS3 GEMM1 fused, ring S=4/SFS=2
        (128, 128, 256),  # rung 66: DIAG WS3 GEMM2, ring S=4/SFS=2
        (128, 128, 256),  # rung 67: DIAG WS3 GEMM2, ring S=5/SFS=3
        (128, 128, 256),  # rung 68: DIAG WS3 GEMM2, ring S=6/SFS=3
        (128, 64, 256),   # rung 69: DIAG WS4 GEMM1 fused 352thr/8-epi
        (128, 64, 256),   # rung 70: DIAG WS4 GEMM1 fused + PP acc
        (128, 64, 256),   # rung 71: DIAG WS3 GEMM2 BN=64 S=2/SFS=2 -- TALLOC
                          #          drops to 128 TMEM columns, so THREE CTAs
                          #          are resident per SM (three-wave grid)
        (128, 64, 256),   # rung 72: DIAG WS3 GEMM2 BN=64 S=3/SFS=2 -- the
                          #          two-CTA control for rung 71
        (128, 128, 256),  # rung 73: DIAG WS3 GEMM2 = rung 57 with the payload
                          #          ring decoupled from the scale ring (DEEP)
        (128, 64, 256),   # rung 74: WS3 GEMM1 fused = rung 56 with DEEP (J6,
                          #          promoted for the fused band; 56 is the
                          #          control and NVFP4_MOE_CFG=56 restores it)
        (128, 64, 256),   # rung 75: WS3 GEMM1 SK2 = rung 58 with DEEP (J8,
                          #          promoted for the split-K band; 58 is the
                          #          control and NVFP4_MOE_CFG=58 restores it)
        (128, 64, 256),   # rung 76: rung 43 (mono SK4) with GEMM1's A operand
                          #          staged by CLIPPED TMA gather4:
                          #          ceil(rows_valid/4) four-row boxes straight
                          #          out of hid_q instead of one 128-row box over
                          #          the pre-gathered buffer, so the
                          #          k_gather_a/k_gather_t node is not launched
                          #          and hidq_g/hidsf_g are not allocated. One
                          #          graph node fewer with the SAME split-K grid.
                          #          PROMOTED by J13 for both leaves that
                          #          returned 43 (dv_180714de901d, four rounds,
                          #          bit-exact): fixed_0 1.1222x, domain_0
                          #          1.1056x, domain_1 1.0319x. Rung 43 is the
                          #          control and NVFP4_MOE_CFG=43 restores the
                          #          pre-gather form byte-for-byte.
        (128, 64, 256),   # rung 77: DIAG rung 43 with UNCLIPPED TMA gather4 --
                          #          rung 76's control. The k_gather node is
                          #          gone but all 32 four-row boxes and all 128
                          #          A-scale rows are still staged per stage, so
                          #          43 -> 77 prices deleting the node and
                          #          77 -> 76 prices the clip.
        (128, 64, 256),   # rung 78: rung 75 (WS3 split-K GEMM1, SK=2, DEEP)
                          #          with the CLIPPED gather4 A operand -- rung
                          #          64's A path plus DEEP plus the clip.
                          #          PROMOTED by J18 for the R <= 1536 leaf
                          #          (dv_a7ad9c85aac9, three rounds, bit-exact):
                          #          domain_9 1.1108x, fixed_1 1.0875x, domain_5
                          #          1.0343x, fixed_2 1.0241x, domain_7 1.0167x,
                          #          every clip round below every cur round. Rung
                          #          75 is the control and NVFP4_MOE_CFG=75
                          #          restores it byte-for-byte.
        (128, 64, 256)]   # rung 79: DIAG rung 75 with UNCLIPPED gather4 and
                          #          DEEP -- rung 78's same-job control, so
                          #          75 -> 79 prices deleting the pre-gather node
                          #          inside WS3+DEEP and 79 -> 78 prices the clip
                          #          inside it. Rung 64 is the other unclipped
                          #          WS3 rung but lacks DEEP, so it cannot serve
                          #          as the control without combining across jobs,
                          #          which J15 ruled out.
# Rung 80 (arm-b TSTORE port, join 20261023): WS3 GEMM2 = rung 57 with the
# rout drain issued as per-warp-quadrant S2G bulk-tensor stores. Bit-exact
# by construction; override-only (NVFP4_MOE_CFG2=80) pending the paired
# A/B on this tree. Arm-b measured WIN on its own lineage: fixed_6 0.943 /
# fixed_5 0.962 / fixed_4 0.977 / fixed_7 0.970 / domain_6 0.847.
CFGS += [(128, 128, 256),   # rung 80: WS3 GEMM2 57 + S2G bulk-tensor rout drain
         #                            (TSTORE, arm-b port; promoted default in-band)
         (128, 128, 256),  # rung 81: rung 80 + J6 DEEP wait split (DEEPxTSTORE
                           #          cross-arm combo probe; bit-exact, diag-only)
         (128, 64, 256),   # rung 82: rung 74 + TMA L2 prefetch x2 stages (PFTMA
                           #          probe; pure hint, bit-exact, diag-only)
         (128, 128, 256),  # rung 83: rung 80 + PFTMA x2 (bit-exact, diag-only)
         (128, 128, 256),  # rung 84: reserved (index spacer)
         (128, 128, 256),  # rung 85: reserved (index spacer)
         (128, 128, 256),  # rung 86: reserved (index spacer)
         (128, 128, 256),  # rung 87: reserved (index spacer)
         (128, 64, 256),   # rung 88: rung 78 + FUSE (WS3 last-arriver split-K
                           #          activation reduce; deletes the
                           #          k_act_packed node; bit-exact with 78 by
                           #          construction; REJECTED 1.02-1.27x on all
                           #          8 SK-band shapes, dv_8072cd899abd;
                           #          override-only control)
         (128, 128, 256),  # rung 89: rung 81 + NIPRE=6 (cross-item operand
                           #          L2 prefetch; targets the nk=1 I=256
                           #          GEMM2 band; bit-exact hint; override
                           #          NVFP4_MOE_CFG2=89)
         (128, 64, 256),   # rung 90: rung 74 + NIPRE=6 (fused-band GEMM1;
                           #          override NVFP4_MOE_CFG=90)
         (128, 64, 256),   # rung 91: MEGA barrier-free single-kernel decode
                           #          path (whole pipeline in one node for
                           #          tiny routed work; NVFP4_MOE_MEGA=1 +
                           #          R<=64 shape gate; see k_mega_decode)
         (128, 64, 256),   # rung 92: MEGA DIRECT variant (k_mega_direct,
                           #          no staging smem): selected when the
                           #          capture-time route statistic is EMPTY
                           #          at any R <= 8192 -- the whole forward
                           #          collapses to output zeroing; non-empty
                           #          replays fall back to the (slow but
                           #          exact) direct path
         (128, 128, 256),  # rung 93: rung 81 + NIPRE=1 (depth-1 cross-item
                           #          prefetch; REJECTED 1.013-1.039x at all
                           #          depths -- per-instruction cost, not L2
                           #          capacity; dv_b29a13ded597; override)
         (128, 128, 256),  # rung 94: rung 81 + NIPRE=2 (rejected, same job)
         (128, 64, 256),   # rung 95: rung 78 with SK=4 -- REJECTED
                           #          1.007-1.099x on all six SK-band shapes
                           #          (dv_e99560cbff52); override control
         (128, 64, 256),   # rung 96: rung 78 + SELFPREP (routing map
                           #          recomputed in-grid; deletes the prep
                           #          NODE for R<=256). Override CFG=96 or
                           #          NVFP4_MOE_SELFPREP=1.
         (128, 64, 256),   # rung 97: rung 74 + NPAIR fused pair-MMA (one
                           #          m128n128k64 per kset replaces the
                           #          up/gate n64 pair). PROMOTED fused-band
                           #          default (dv_8c9f8253b80e); CFG=74
                           #          restores the control.
         (128, 64, 256),   # rung 98: rung 75 + NPAIR (SK=2, AMODE 2,
                           #          SFCH off) — override-only probe
         (128, 64, 256),   # rung 99: rung 78 + NPAIR (SK=2, AMODE 3
                           #          clipped gather4, SFCH on) — PROMOTED
                           #          split-K band default (dv_70eff7810dcb)
         (128, 64, 256),   # rung 100: GEMM2 BN=64 + PP (2-CTA-legal
                           #          ping-pong, TCOLS 192) — REJECTED
                           #          (dv_2b258065c744), override-only
         (128, 64, 256),   # rung 101: rung 97 + MBH(256ns) mbarrier wait
                           #          hints — override-only probe
         (128, 128, 256),  # rung 102: rung 81 + MBH(256ns) — GEMM2 probe
                           #          REJECTED (dv_e6be9e45976d)
         (128, 64, 256),   # rung 103: rung 97 + NIPRE=6 — REJECTED
                           #          (dv_d734f8143966), override-only
         (128, 64, 256),   # rung 104: rung 97 + SFS=3 — REJECTED on valid
                           #          re-probe (dv_bd7a05081c86: +5-9% high-
                           #          fill = un-DEEP cost), override-only
         (128, 128, 256)]  # rung 105: GEMM2 EPI=8 (8-warp drain, no
                           #          TSTORE/EARLYDONE) — CFG2=105 probe
# Rung 3 uses BK=256 so that each thread keeps 64 B of weight payload in
# flight; with BK=64 it was 16 B and the tiny-batch shapes were latency bound
# at ~0.27 TB/s (fixed_0 gemm1 47 us for 12.6 MB).
CFGS_THREADS = [256, 256, 128, 64]
CFGS_FLOP_PER_BYTE = [10.7, 10.7, 10.7, 5.3]


def _rung7_ok(H, I):
    """Shape gate for the packed-NVFP4 tcgen05 rung.

    Its k stage holds 256 FP4 values for both operands and its N tile is 128
    columns, so both K dims (H for GEMM1, I for GEMM2) must be multiples of 256
    and both feature dims multiples of 128. All four legal geometry tuples in
    shape_range.json satisfy this; the gate exists so that a shape which does
    not falls back to the ladder instead of launching an illegal tile.
    """
    return H % 256 == 0 and I % 256 == 0 and H % 128 == 0 and I % 128 == 0


_RESIDENT = None


def _resident_ctas():
    """Resident CTA capacity of the persistent GEMM grid (two waves of the
    automatic grid cap = 2 CTAs per SM on every fam-2 rung)."""
    global _RESIDENT
    if _RESIDENT is None:
        try:
            _RESIDENT = 2 * torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        except Exception:
            _RESIDENT = 296
    return _RESIDENT


def _pick_cfg(H, I, E, R):
    """Choose a tile rung from shapes only.

    Scores each rung by tensor-core efficiency divided by the M-tile padding
    waste for the expected routes per expert, discounted when the rung would
    not produce enough blocks to fill the GPU. No data, case or seed identity
    is inspected: the same shapes always select the same rung.
    """
    n1, n2 = 2 * I, H
    used = max(1, min(E, R))
    per_expert = max(1.0, R / used)
    # Rung 7 keeps both operands packed as NVFP4 and feeds the block-scaled
    # tensor core directly, so the fp4->bf16 decode (1830 of 2663 cycles per
    # k-tile in the ablation, job dv_a2ac5b2e9d68) disappears, operand bytes in
    # shared memory halve, and the hidden-state decode kernel is not launched at
    # all. Measured on L20D (job dv_e0afbf95bb7c, experiments/profile_kernels.py,
    # CUDA-graph path): geomean 292.9 us over the ten fixed shapes against
    # 561 us for rung 5 and 579 us for the previous default, and it also won the
    # tiny-batch shapes (fixed_0 96.0 vs 100.1, fixed_1 97.3 vs 147.4, fixed_8
    # 58.3 vs 58.1) because two launches and the decode buffer are gone.
    # dev_check: 19/19 configs, 0 mismatched elements, max rel_l2 2.26e-05.
    #
    # Rung 8 is rung 7 with a three-buffer copy pipeline instead of two, and it
    # measured faster on every shape tried (job dv_cba49fff62cc, same process and
    # tensors, automatic grid cap in both): fixed_0 90.5 vs 96.1 us, fixed_3
    # 524.3 vs 577.7, fixed_6 1176.7 vs 1304.5 -- one CTA per SM here, so a
    # longer ring is the only way to keep more HBM latency in flight. Four
    # buffers (rung 9, 216 KiB of the 227 KiB limit) measured *worse* than two
    # (fixed_6 1439.1 us), so three is the measured optimum, not an extrapolated
    # one. dev_check: 19/19 for both rung 8 and rung 9.
    # Rung 12 is rung 8 with the operand tiles staged by TMA instead of 3072
    # (GEMM1) or 2048 (GEMM2) sixteen-byte cp.async requests per stage: the frozen
    # 32-row block shuffle is a 4x8 transpose inside each block, which a
    # multi-dimensional box expresses because TMA writes dim1 fastest, and the box
    # carries the SW128 swizzle so the shared-memory image -- and therefore every
    # result -- is bit-identical (experiments/tma_box_probe.py proved the image
    # byte-for-byte against this kernel's own staging loop, job dv_816ecb04c15a).
    #
    # Measured back to back in one process on the same tensors (job dv_5322b982f946,
    # experiments/profile_kernels.py with PROFILE_CFGS=8,12, CUDA-graph path,
    # automatic grid cap in both): e2e geomean over the ten fixed shapes
    # 244.29 -> 233.56 us = 1.046x, with no shape slower (fixed_0 1.021, fixed_1
    # 1.018, fixed_2 1.085, fixed_3 1.074, fixed_4 1.047, fixed_5 1.047, fixed_6
    # 1.049, fixed_7 1.052, fixed_8 1.002, fixed_9 1.068). The gain is entirely
    # GEMM2 (geomean 1.189x, fixed_2 105.7 -> 72.7 us), whose A operand is already
    # contiguous in routed-row order and so also becomes one box; GEMM1 is
    # unchanged (1.0002x) because its A operand is gathered per token and still
    # costs 1024 requests per stage. dev_check 19/19 with rel_l2 identical to rung
    # 8 on every case, and tma_status 15 (all three maps encoded) on every fixed
    # shape. launch_gemm falls back to rung 8 by itself when a geometry cannot be
    # described, so this is safe for the whole shape_range.
    # Promoted on three agreeing measurements: the official fixed suite at 1.026x
    # (job dv_f816981d7c06, 250.1 -> 243.8 us, no case slower), a same-process
    # fixed-shape A/B at 1.046x (job dv_5322b982f946) and a same-process A/B over
    # six domain shapes at 1.052x graph / 1.045x eager (job dv_4f0b77a4f77b), where
    # the gain is GEMM2 at 1.2317x and GEMM1 is flat at 1.0014x because its A
    # operand is gathered per token and still costs 1024 cp.async requests per
    # stage. dev_check is 19/19 with rel_l2 identical to rung 8 on every case,
    # which is what a byte-identical shared-memory image implies.
    #
    # One official domain run at this rung (job dv_5fda0d7864e3) reported 0.860x.
    # It is recorded as an anomalous measurement, not as a verdict: its own
    # baseline is reproducible (the in-process cfg8 column matches the previous
    # official rung-8 numbers within 1-4%) while its rung-12 numbers sit 25-52%
    # above the in-process rung-12 numbers for the same shapes and the same code,
    # twelve minutes after the fixed job at the same commit showed a gain. See
    # PROGRESS.md for the cross-reference table.
    # Promoted policy v2 (measured on L20D, jobs dv_d2b570024b75 and
    # dv_e4ba60ba0b9e, experiments/bench.py --runs in-process A/B, official
    # eager protocol, fixed-suite geomean): rung 17 (BN=64, two payload
    # buffers, two-slot SF ring with SF REGISTER PREFETCH, two CTAs per SM)
    # for GEMM1 + rung 19 (rung 12 + SF prefetch) for GEMM2, with the routing
    # small-path threshold at 8192: geomean 203.8 us vs 213.2 for the
    # batch-1 policy and 238.1/233.4 for V0 in the same processes = 1.168x.
    # Rung 17 wins EVERY fixed shape including the tiny ones (fixed_0 72.1 vs
    # 73.6, fixed_1 71.4 vs 72.2): the SF prefetch removes the per-stage
    # scale-load latency that made BN=64 lose on latency-bound shapes in the
    # B2 sweep, so no route-count bands are needed anymore. Rejected by
    # measurement: rung 18 (BN=128 + prefetch, geomean 221.3, fixed_6 GEMM1
    # 596 vs 480 -- two CTAs per SM beat the -25% operand traffic), rung 16
    # (three payload buffers, 205.8 -- no gain over two once SF is prefetched),
    # GEMM2 at BN=64, and gather4 rungs at throughput sizes. dev_check on
    # rungs 13/16/17/18/19: 6 shapes, 0 mismatched elements, outputs
    # bit-identical to rung 13 (job dv_d2b570024b75).
    if _rung7_ok(H, I):
        # The former single-wave band (rung 12 when tiles_est*(I//128) fits
        # one resident wave but tiles_est*(I//64) does not) is REMOVED on the
        # AGA bytes: the band only ever fires at R < 256, exactly the range
        # where the split-K pre-gathered rungs now dominate -- domain_9
        # (R=40, the band's original evidence case) measures 82.5 us at rung
        # 12 vs 75.9 at rung 29 (+8.7%, job T2 dv_6c27fd41182a). The split-K
        # bands below therefore decide every fam-2 shape.
        # Split-K bands for latency-bound GEMM1, RE-TUNED after the split-K
        # grid fix (230c9d2 -- slices used to serialize in-CTA, which is what
        # the original "SK loses at R>1536" verdict actually measured).
        # Paired same-job stage A/Bs (20261020T122740Z + 20261020T123523Z,
        # official eager protocol):
        #   R <= 256 -> SK-depth split by the BASE item count (v2.5):
        #     items_est = min(E,R) * (I//64) is the rung-17/21 item space
        #     (m-tiles x BN=64 n-tiles), the same host-side bound the
        #     single-wave band uses. SK4 only while items_est <= 1536 (~5
        #     resident waves): deeper slicing wins when the machine is
        #     starved (fixed_0 96 items: 53.8 vs 58.0 SK2; domain_0 192:
        #     61.3 vs 62.4; domain_1 96: 56.2 vs 58.3; fixed_1 1536: g1
        #     18.5 vs 20.5, e2e tied) -- above that the slice overhead
        #     dominates (domain_7 3072 items: SK2 341.2 vs SK4 352.3, g1
        #     227.6 vs 237.7). v2.4's flat R<=256->SK2 cost fixed_1 the
        #     2-us g1 edge (official case 1: 59.2->63.3, within that
        #     case's noise history but against the stage evidence).
        #   R <= 1536 -> SK2 (rung 21): fixed_2 254.9 vs SK4 274.0;
        #     domain_5 163.1 vs SK4 175.5.
        #   1536 < R <= 8192 with I <= 256 and H >= 4096 -> SK2: narrow N
        #     (n_tiles = I/64 = 4) keeps the item count near one wave while
        #     the nk=16..28 chain stays latency-bound, and the [2][R][2I]
        #     fp32 partials stay small (fixed_4 R=8192 H=7168 I=256: 361.7
        #     vs 383.6 fused; g1 195.3->164.3 against act 2.4->17.4). The
        #     same geometry at R=32768 already loses (domain_6 600.1 vs
        #     551.3: 134 MB of partial traffic), and every I>=768 shape at
        #     R>1536 loses (fixed_9 160.2 vs 137.9, fixed_3 406.7 vs 394.1,
        #     fixed_5 365.4 vs 289.1, fixed_7 304.6 vs 257.6, fixed_6,
        #     domain_2 1114 vs 761, domain_3) -> fused rung 17.
        # Every legal H has nk = H/256 in {8, 16, 28}, divisible by 4, so
        # slice counts are exact; rung_ok re-checks divisibility anyway.
        # Pre-gathered-A upgrade (jobs dv_c4c57ce1742d sweep + dv_9e15a2881d44
        # stage attribution, same-process graph-mode medians): rungs 27/28/29
        # (= 17/21/20 with k_gather_a staging hid_q/hid_sf into route order
        # once, then ONE 2-D TMA A-box per stage instead of 1024+ 16-byte
        # cp.async gathers) are bit-identical on every compared shape and won
        # or tied nearly everywhere: fixed_0 1.224x (g1 stage 25.0->13.3 us),
        # domain_7 1.216, domain_5 1.153, fixed_1 1.149/1.161, fixed_2 1.142,
        # fixed_3 1.143, domain_3 1.119, domain_0 1.177, fixed_9 1.060-1.068,
        # domain_8 1.033, fixed_5 1.026, fixed_7 1.020, fixed_6 1.007,
        # domain_2 1.004, empties 0.979-1.002 (grid-stride gather, 592-CTA
        # cap), domain_6 0.993. The band structure below is unchanged; each
        # leaf now selects the AGA counterpart. Rung 26 (warp-specialized)
        # measured 0.34x-0.93x on ten shapes (job dv_9e15a2881d44): closed.
        # C12: BRLESS flip -- the branchless quantize guard (rungs 41/42/43 =
        # 27/28/29 with rcp(fmaxf(sf, FLT_MIN)) replacing the per-16-block
        # `sf > 0` branch) is bit-exact on every compared shape (job Y
        # cmp41/42/43 incl. domain_3 situ and domain_4 empty: bit_exact=True,
        # max_abs_diff=0.0) and won all seven same-run sweep medians
        # (fixed_0 +2.0%, fixed_6 +0.7%, domain_9 +1.3%, rest +0.2..0.7%):
        # the guard was the hot FSETP+BRA+BSSY/BSYNC cluster in job X's SASS
        # sampling. Band structure unchanged; leaves select the BRLESS rung.
        # WS3 promotion (jobs dv_d48a78b22bfd + dv_c04069a2255d, paired
        # in-process A/B, official eager protocol, bit-exact outputs on every
        # compared shape): the 7-warp role-split rungs replace their
        # monolithic counterparts at unchanged band structure -- 59 (=43 SK4)
        # domain_9 1.486x, fixed_0 0.989x; 58 (=42 SK2) fixed_2 1.374x,
        # domain_7 1.453x; 56 (=41 fused) fixed_7 1.123x, domain_3 1.066x,
        # fixed_5 1.069x, fixed_6/fixed_3/fixed_9/domain_2/domain_8
        # 1.063-1.087x (dv_d8cc9b623cad). NCU root cause (dv_cea324bddbfe):
        # the monolithic loop spent 78% of cycles with no eligible warp
        # (barrier stall 38.8%), tensor pipe at 20%; WS3 keeps the tensor fed
        # from a dedicated MMA warp while two producer warps stage payload
        # (TMA) and scales (16B LDG + 4B cp.async) behind full/empty
        # mbarriers -- the production cake/alphamoe choreography.
        # Tiny-route boundary (sweep dv_ee4b86a25938 on synthetic in-range
        # shapes + official evals dv_a611e801e6c4/dv_496c03fc436f): below
        # ~64 routes the WS3 SK2 form (58) cannot amortize its role-split
        # startup against mono SK4 (43), which also brings 2x more items to
        # the grid. Deep-K geometries (H>=7168) keep 43 through R=256 (58
        # measured 0.897x @R=16, 0.920x @R=24, 0.922x @R=64, 0.960x @R=256:
        # 7-stage SK4 slices pipeline fine inside the monolith); qwen-like
        # geometries flip at R=64 (58: tie @R=16 -- but official domain_0
        # regressed 0.959x there -- 0.966x @R=32, 1.013x @R=64, 1.162x
        # @R=128, 1.216-1.235x above). 58 covers R<=1536 (R~1024: fixed_2
        # 1.021x, domain_5 1.018x vs 56); 56 above (deepseek fixed_4 1.051x
        # vs 58, domain_6 1.022x vs 17). Rungs 43/59/17 remain
        # override-only fallbacks.
        #
        # J6 (job dv_e33b3b932743, two rounds, alternating processes, every
        # digest bit-exact): the fused leaf selects rung 74 rather than 56.
        # 74 IS 56 -- same BN, same STAGES=3/SFS=2, same shared memory, same
        # TMEM, same staged images, same MMA sequence -- with one difference:
        # warp 1 stages the payload behind a wait on s_empty[g-STAGES] and the
        # weight scales behind the existing wait on s_empty[g-SFS], instead of
        # both behind the later of the two. With SFS=2 < STAGES=3 that coupling
        # meant the TMA issue waited on MMA(g-2) for a payload buffer that
        # MMA(g-3) had already released, so a three-deep ring ran two deep and
        # the third buffer was dead shared memory: 128 KB in flight per SM
        # instead of the 192 KB allocated. Decoupling them measured -5.6% /
        # -5.2% / -4.3% on the gemm1 stage slot and +2.3% / +1.8% / +2.2% end to
        # end on fixed_5 / fixed_6 / fixed_9 (geomean of the three 1.0209x).
        # GEMM2's copy of the same change (rung 73) measured neutral
        # (1.0007x) and is left alone, so cfg2 stays 57. Rung 56 remains as the
        # control and NVFP4_MOE_CFG=56 restores it exactly.
        #
        # J8 (job dv_..., run_j8.sh): the split-K leaf selects rung 75 rather
        # than 58 for the same reason. 75 IS 58 -- WS3 split-K GEMM1 at SK=2,
        # same BN, same STAGES=3/SFS=2, same smem, TMEM, grid, AMODE=2
        # pre-gathered A, same fp32 partials and the same k_act_packed<true>
        # reduce, so the slices still sum in the same order -- with the DEEP
        # wait split, and it is the other leaf that reaches the coupled wait.
        # Against a forced rung 58, two rounds, alternating processes, both
        # digests bit-exact: the gemm1 slot falls 13.34 -> 12.90 on fixed_1 and
        # 104.70 -> 99.01 (-5.4%) on fixed_2 with every other slot unchanged,
        # end to end 1.0025x on fixed_1 (0.44 us of a 34 us shape, inside its
        # noise) and 1.0263x on fixed_2. Never worse on either shape that
        # selects it, so the leaf moves; rung 58 stays as the control and
        # NVFP4_MOE_CFG=58 restores it byte-for-byte.
        # J13 (job dv_180714de901d, run_j13.sh, four rounds, alternating
        # processes, every digest bit-exact): both leaves that returned 43 now
        # return 76, which IS 43 -- same BN=64, STAGES=2, SFS=2, SFPRE=1, SK=4,
        # same pre-gathered-free A operand source, same fp32 partials in the same
        # slice order, same k_act_packed<true> -- with GEMM1's A staged by
        # ceil(rows_valid/4) four-row tile::gather4 boxes out of hid_q instead of
        # one 128-row box over a buffer k_gather_t built first. That deletes a
        # graph node WITHOUT deleting grid work, which is the condition every
        # previous attempt failed: J2 and J12 traded nodes against parallelism
        # and lost, and the pre-join C57 last-arriver reduce moved the work into
        # one CTA per tile group and lost at 0.836x.
        #
        #   fixed_0   30.6 -> 27.2 (1.1222x)   domain_0  37.3 -> 33.7 (1.1056x)
        #   domain_1  33.2 -> 32.1 (1.0319x)
        #
        # The slots say where it comes from: the gather slot falls to the
        # empty-event floor (domain_0 6.30 -> 2.43, domain_1 6.43 -> 2.53 us), so
        # the node really is gone, and gemm1 falls too (domain_0 18.66 -> 14.78)
        # because an 8-route tile now stages TWO boxes and four scale rows per
        # stage instead of 32 and 128. stage_sum -7.0 / -3.1 us, end to end -3.4 /
        # -1.0. Rung 77 -- the same body UNCLIPPED, so the node is gone but all 32
        # boxes are still issued -- is the control that prices the two halves
        # apart: 0.783x on fixed_0 and 0.834x on fixed_1, against the pre-join
        # chain's 0.797x for the same comparison. Deleting the node alone is a
        # 22% LOSS; the clip is what pays.
        #
        # The gate carries the boundary that was MEASURED, not the one inferred
        # from the current range: <= 4 routes per expert. Under the frozen
        # shape_range (local_experts in {128,256}, topk 8) min(E,R) <= 32 already
        # implies R <= 32, so the guard cannot bite; it becomes load-bearing if E
        # ever drops below 64, where a tile's rows_valid approaches BM, the clip
        # saves nothing and gather4's 32 boxes cost more issue slots than the one
        # 2-D box they replace (the pre-join chain measured exactly that: -11% at
        # 32 routes per expert). Rung 43 stays as the control and
        # NVFP4_MOE_CFG=43 restores it byte-for-byte.
        #
        # NOT extended to the R <= 1536 leaf, on evidence rather than caution:
        # fixed_1 (R=256, 2 routes/expert, promoted rung 75) measured 1.0019x
        # against it -- the clip exactly cancels gather4's per-stage cost there --
        # and domain_9 (R=40, promoted 75) measured 1.1444x, but that column also
        # changes GEMM1 family and SK 2 -> 4, so it is a dispatch result and not
        # an A-operand result. results/j13-clipped-gather4.md.
        # J18 (job dv_a7ad9c85aac9, run_j18.sh, three columns, three rounds,
        # alternating processes, every digest bit-exact on all five shapes and all
        # three columns): the R <= 1536 leaf selects rung 78 rather than 75. 78 IS
        # 75 -- same BN=64, STAGES=3, SFS=2, BRLESS, SK=2, EPI=4, DEEP, same shared
        # memory, TMEM, grid, fp32 partials, slice order and k_act_packed<true> --
        # with GEMM1's A operand staged by ceil(rows_valid/4) four-row
        # tile::gather4 boxes out of hid_q instead of one 128-row box over a buffer
        # k_gather_t built first. It is bit-exact with 75 by construction and the
        # digests confirm it; what changes is that the pre-gather node is not
        # launched at all and a tile holding 8 valid routes stages TWO boxes and
        # eight A-scale rows per stage instead of 32 and 128.
        #
        #   shape     routes/expert   cur (75)    rung 78    speedup
        #   domain_9      1.00         60.48       54.32     1.1108x
        #   fixed_1       2.00         35.22       32.75     1.0875x
        #   domain_5      8.00        109.18      105.56     1.0343x
        #   fixed_2       8.00        177.18      173.22     1.0241x
        #   domain_7      1.94        216.21      212.13     1.0167x
        #
        # Every clip round is below every cur round on all five, and the gather slot
        # falls to the ~2.6 us empty-event floor on all five, so the node is gone
        # rather than faster. Extrapolated as a constant that is +1.57% of the
        # domain geomean and +1.08% of the fixed.
        #
        # This REFUTES the prior J14 drew. J14 measured the clip inside the MONO
        # family at 1.1192x / 1.0269x / 1.0036x / 0.9927x for 1.00 / 2.00 / 1.94 /
        # 8.00 routes per expert and concluded the clip goes negative where tiles
        # are full, so rung 76 could not be extended here. Rung 78 wins at 8.00
        # routes per expert (+2.4% and +3.4%) because what matters is routes per
        # TILE, not per expert: fixed_2's 8 routes per expert still fill only 8 of a
        # 128-row tile, so nbox is 2 of 32 and the clip saves 94% of the A staging
        # either way. What differs between the families is the COST of gather4's
        # extra issue slots -- in WS3 a dedicated producer warp issues them and has
        # nothing else to do, while in the monolith they come out of the same warp
        # that stages B and waits on the ring. Rung 79 (75 with UNCLIPPED gather4
        # and DEEP held constant) prices that cost in the same job: 0.8321x /
        # 0.7818x / 0.8096x / 0.8129x / 0.7500x, and clip/g4deep is 1.26-1.39x on
        # every shape. So inside WS3 the node is expensive to lose and the clip pays
        # for it everywhere; inside the monolith the node was cheap to lose and the
        # clip only paid below ~2 routes per tile.
        #
        # The H >= 7168 leaf moves too. No sampled shape reaches its R <= 1536
        # branch (domain_1 is R=24 and takes the rung-76 leaf above it), so this is
        # the one promoted path the five measurements do not cover; nothing in the
        # mechanism distinguishes it -- the box count is ceil(rows_valid/4) and the
        # A bytes per stage are BM x 128 B, both independent of H -- and the
        # measured shapes span H=2048 and H=4096 with I=768 and I=1536. Rung 75
        # stays as the control and NVFP4_MOE_CFG=75 restores it byte-for-byte.
        if H >= 7168:
            if R <= 256 and R <= 4 * min(E, R):
                # Ported from ARM A (join 20261025, their C33, dv_97795616d3f7):
                # the WS3 AMODE3 family (78 + SFCH/SFPF) beats the mono leaf on
                # EVERY sparse geometry tuple they measured -- at H=7168: R=24
                # 0.8967, R=256 0.8825, both bit-exact; their SK1-fused variant
                # loses at R=24 (1.0206), so the uniform sparse choice is the
                # SK2 clipped leaf, not a fused one. 76 stays restorable via
                # NVFP4_MOE_CFG=76.
                # NPAIR-SK promoted (dv_70eff7810dcb): rung 99 = rung 78 +
                # the fused pair-MMA; this leaf's sampled shape domain_1
                # measured 1.000/1.001 (neutral, band wins at f2/d5/d7).
                # NVFP4_MOE_CFG=78 restores the control byte-for-byte.
                return 99
            if R <= 1536:
                return 99   # NPAIR-SK (dv_70eff7810dcb); CFG=78 restores
            # NPAIR promoted (dv_8c9f8253b80e): rung 97 = rung 74 + the fused
            # pair-MMA (one m128n128k64 per kset replaces the up/gate n64
            # pair; bit-exact, 8/8 real-pipeline shapes win both seeds, this
            # leaf's sampled shape fixed_4 1.008/1.008). NVFP4_MOE_CFG=74
            # restores the control byte-for-byte.
            return 97
        if min(E, R) <= 32 and R <= 4 * min(E, R):
            # ARM A C32 (join 20261025): 76 -> the WS3 clipped leaf, bit-exact
            # wins across the sparse leaf (their case 0 0.9339, x_t2h 0.9333,
            # x_t4 0.8745 at H=2048; y_t2 0.9572 at H=4096). The leaf was tuned
            # within the mono family before the ws3 AMODE3 rungs existed.
            # NPAIR-SK promoted (dv_70eff7810dcb): 99 = 78 + fused pair-MMA;
            # sampled leaf shapes fixed_0 1.000/1.070 (gemm1 12.4->11.3) and
            # domain_0 1.000/1.000 — neutral-or-better, bit-exact.
            return 99
        if min(E, R) <= 32:
            return 43
        if R <= 1536:
            # NPAIR-SK promoted (dv_70eff7810dcb,
            # results/npair-sk-ab-dv_70eff7810dcb.md): fused pair-MMA 99
            # replaces 78 — fixed_2 1.013/1.012 (gemm1 90.3->87.3), domain_5
            # 1.019/1.020 (53.5->51.4), domain_7 1.020/1.010 (115.1->111.6),
            # fixed_1/domain_1 neutral, all bit-exact; domain_9 keeps its
            # zero-capture shortcut via the gate extension (99 in the list).
            return 99
        # NPAIR promoted (dv_8c9f8253b80e, results/npair-ab-dv_8c9f8253b80e.md):
        # fused pair-MMA rung 97 replaces 74 for the fused band — e2e
        # fixed_3 1.010/1.015, fixed_5 1.026x2, fixed_6 1.032x2, fixed_7
        # 1.016/1.022, domain_2 1.027x2, domain_3 1.044x2, domain_8
        # 1.017x2, every arm bit-exact; gemm1 slots -1.8..-41.1us. The
        # zero-capture gate below includes 97 so shortcut shapes keep
        # their fast path. NVFP4_MOE_CFG=74 restores the control.
        return 97
    # Measured on L20D (jobs dv_9a8bd6ef5e14 and dv_a2ac5b2e9d68): the tcgen05
    # rung (128 tokens per CTA, N=256 MMAs, clamp/activation/QDQ fused into the
    # GEMM1 epilogue) beats the whole warp-mma ladder once a batch fills the
    # machine -- fixed_3 1342 vs 1822 us, fixed_7 836 vs 1187, fixed_9 421 vs
    # 584, fixed_6 2407 vs 2744 -- and loses badly on tiny batches, where
    # 128-token tiles leave almost no CTAs resident (fixed_0 211 vs 90 us).
    # The switch is a shape threshold only: never data, case or seed identity.
    if R >= 512 and I % 256 == 0 and H % 256 == 0 and I % 64 == 0:
        # Two 128-token sub-tiles sharing one decoded weight tile measured
        # better only once an expert holds >= 256 routes (fixed_6 2291 vs
        # 2407 us); below that the extra M padding and the halved CTA count
        # dominate (fixed_3 1707 vs 1342, fixed_4 1814 vs 1195, fixed_9 554
        # vs 421), so the switch is a routes-per-expert threshold.
        if per_expert >= 256 and I % 128 == 0:
            return 6
        return 5
    best, best_score = 3, -1.0
    for cfg, (bm, bn, bk) in enumerate(CFGS[:4]):
        if n1 % bn or n2 % bn or H % bk or I % bk:
            continue
        tiles = -(-per_expert // bm)              # ceil
        waste = tiles * bm / per_expert
        blocks = (n1 // bn) * used * tiles
        parallelism = min(1.0, blocks / 1200.0)
        score = CFGS_FLOP_PER_BYTE[cfg] / waste * (0.3 + 0.7 * parallelism)
        if score > best_score:
            best, best_score = cfg, score
    return best


def _c(tensor):
    """Contiguous view/copy; never mutates the caller's tensor."""
    return tensor if tensor.is_contiguous() else tensor.contiguous()


class Model(torch.nn.Module):
    """Routed NVFP4 MoE: FC1 -> clamp/activation -> NVFP4 QDQ -> FC2 -> reduce."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scratch = {}
        self._tables_ready = set()
        self._last_abi = None
        self._last_meta = {}
        # Identity-keyed fast path: key -> (abi, lib, keepalive). See forward().
        self._fast = {}
        # J1: single identity-verified entry consulted ahead of the key build.
        # None -> the key/dict path below handles the call.
        self._hot = None
        # Resolved device index for the raw current-stream getter; the slow
        # path overwrites it from the actual input device before any fast hit.
        self._dev_idx = 0
        # Mirrors the launcher's persistent-grid cap so that diagnostics can
        # report it without a host round-trip inside the measured region.
        # -1 = automatic (two waves of resident CTAs), matching the launcher's
        # own default; 0 would mean "uncapped" and misreport the diagnostics.
        self._grid_cap = int(os.environ.get("NVFP4_MOE_GRID_CAP", "-1") or -1)
        # Rung-override env is snapshotted at construction: every harness
        # (official and experimental) sets it before Model(), and three
        # os.environ.get calls cost ~1.3us inside the measured fast path
        # (local microbench). Env changes mid-lifetime need a fresh Model.
        self._env_snap = (os.environ.get("NVFP4_MOE_CFG"),
                          os.environ.get("NVFP4_MOE_CFG2"),
                          os.environ.get("NVFP4_MOE_REDF"),
                          os.environ.get("NVFP4_MOE_G2GATE"),
                          os.environ.get("NVFP4_MOE_MEGA"),
                          os.environ.get("NVFP4_MOE_SELFPREP"),
                          os.environ.get("NVFP4_MOE_PREP2"),
                          os.environ.get("NVFP4_MOE_SMALL_THRESH"),
                          os.environ.get("NVFP4_MOE_ZERO"))

    # -- workspace management ------------------------------------------------
    def _workspace(self, key, sizes, device):
        ws = self._scratch.get(key)
        if ws is None:
            int32 = torch.int32
            # The packed-FP4 rungs (>= 7) consume the hidden states exactly as
            # the contract delivers them, and every tcgen05 rung (>= 4) fuses
            # clamp+activation+QDQ into the GEMM1 epilogue, so `hid_bf` and
            # `fc1` are never written or read on those paths. Allocating them
            # anyway costs T*H*2 and R*2I*4 bytes -- 33 MB and 402 MB on the
            # largest fixed shape -- and dead device memory of that size is what
            # an allocator has to work around when one process measures several
            # shapes back to back. The cache key includes the rung, so a shape
            # that falls back to a warp rung still gets both buffers.
            cfg = sizes["cfg"]
            cfg2 = sizes.get("cfg2", cfg)
            # counts and fill are ONE allocation so the big routing path can
            # zero both with a single memset node (two ~2-3 us nodes -> one).
            # The fused split-K rungs (31/32) append their per-(tile,n) arrival
            # counters to the SAME allocation so the one memset covers them
            # too; k_prepare_small zeroes them on the small path.
            cnt_n = (sizes["tiles"] * (sizes["I"] // 64)
                     if cfg in (31, 32, 88)
                     else (sizes["T"] if cfg in (91, 92) else 0))
            counts_fill = torch.zeros(2 * sizes["E"] + cnt_n, dtype=int32,
                                      device=device)
            ws = {
                "counts": counts_fill[: sizes["E"]],
                "fill": counts_fill[sizes["E"] : 2 * sizes["E"]],
                "cnt": (counts_fill[2 * sizes["E"] :] if cnt_n else None),
                "cnt_n": cnt_n,
                "offs": torch.empty(sizes["E"] + 1, dtype=int32, device=device),
                "toffs": torch.empty(sizes["E"] + 1, dtype=int32, device=device),
                "ptok": torch.empty(sizes["R"], dtype=int32, device=device),
                "pslot": torch.empty(sizes["R"], dtype=int32, device=device),
                "pexp": torch.empty(sizes["R"], dtype=int32, device=device),
                "route_of": torch.empty(sizes["R"], dtype=int32, device=device),
                "texp": torch.empty(sizes["tiles"], dtype=int32, device=device),
                "tm0": torch.empty(sizes["tiles"], dtype=int32, device=device),
                "hid_bf": (None if cfg >= 7 else
                           torch.empty(sizes["T"] * sizes["H"], dtype=torch.bfloat16,
                                       device=device)),
                "fc1": (None if cfg >= 4 else
                        torch.empty(sizes["R"] * 2 * sizes["I"], dtype=torch.float32,
                                    device=device)),
                "act": torch.empty(sizes["R"] * sizes["I"], dtype=torch.bfloat16,
                                   device=device),
                "rout": torch.empty(sizes["R"] * sizes["H"], dtype=torch.bfloat16,
                                    device=device),
                # REDF rungs (46/47) only: [T,H] fp32 routing accumulator.
                # Zeroed on a side-stream branch each call; the GEMM2
                # epilogue reduces into it and k_convert writes the output.
                # MEGA (91/92): [R,H] fp32 per-ROUTE accumulator, zeroed ONCE
                # at allocation and self-cleaned by every in-kernel finalize.
                "acc32": ((torch.zeros if cfg in (91, 92) else torch.empty)(
                              (sizes["R"] if cfg in (91, 92) else sizes["T"])
                              * sizes["H"], dtype=torch.float32,
                              device=device)
                          if (cfg2 in (46, 47) or cfg in (91, 92)) else None),
                # Split-K rungs (20/21/28/29) only: raw fp32 GEMM1 partials,
                # one [sk][R][2I] slab; k_act_packed reduces it in slice order.
                "part": (torch.empty(sizes["sk"] * sizes["R"] * 2 * sizes["I"],
                                     dtype=torch.float32, device=device)
                         if sizes["sk"] > 1 else None),
                # Pre-gathered-A rungs (27/28/29) only: one allocation holding
                # hidq_g [R,H/2] then hidsf_g [R,H/16] (views below); k_gather_a
                # writes it once per call so GEMM1 stages its A operand with a
                # single 2-D TMA box per stage.
                "gath": (torch.empty(sizes["R"] * sizes["H"] * 9 // 16,
                                     dtype=torch.uint8, device=device)
                         if (27 <= sizes["cfg"] <= 35
                             or sizes["cfg"] in (37, 39, 41, 42, 43, 45,
                                                 48, 49, 50, 51, 54, 55, 56,
                                                 58, 59, 65, 69, 70, 74,
                                                 75, 82, 90, 97, 98,
                                                 101, 103, 104))
                         else None),
            }
            if ws["gath"] is not None:
                nq = sizes["R"] * sizes["H"] // 2
                ws["hidq_g"] = ws["gath"][:nq]
                ws["hidsf_g"] = ws["gath"][nq:]
            else:
                ws["hidq_g"] = None
                ws["hidsf_g"] = None
            self._scratch[key] = ws
        return ws

    def ensure_tables(self, device=None):
        """Fill the format conversion tables once per device (idempotent)."""
        device = torch.device("cuda") if device is None else device
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index in self._tables_ready:
            return
        stream = torch.cuda.current_stream(index).cuda_stream
        code = _lib().moe_init(ctypes.c_int64(stream))
        if code:
            raise RuntimeError(f"moe_init failed: {code}")
        torch.cuda.synchronize(index)
        self._tables_ready.add(index)

    # The official timing protocol exposes host enqueue inside the measured
    # window; nn.Module.__call__'s hook/tracing machinery is pure per-call
    # overhead here (no hooks are registered by the harness or this module),
    # so route calls straight to forward. Semantics of the call are unchanged:
    # same signature, same [output] return, same mutations.
    def forward(self, topk_ids, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar,
                num_experts, top_k, n_group, topk_group, intermediate_size,
                local_expert_offset, local_num_experts, routed_scaling_factor,
                routing_method_type=0, do_finalize=True, enable_pdl=None,
                activation_type=3, per_token_scale=None, output=None,
                tune_max_num_tokens=8192, gemm1_lora_delta=None, valid_hidden_size=None,
                valid_intermediate_size=None, num_fused_shared_experts=None,
                hidden_states_scale_layout=None):
        # Host fast path. The official eager protocol restores inputs in place
        # and calls forward with the SAME tensor objects every iteration, and
        # its start event executes on an idle device while the host is still
        # preparing the launch -- so every microsecond of python ahead of
        # moe_run lands directly on the measured timeline (~20 us for the full
        # preparation below, i.e. ~30% of a small-shape measurement). The
        # identity-keyed cache holds a prebuilt ABI plus strong references to
        # every keyed object (so no id can be recycled behind a stale entry)
        # and cuts the per-call host work to the key build, a stream refresh
        # and one ctypes call. Any argument, shape, scalar or rung-override
        # change misses and rebuilds through the fully validated slow path.
        #
        # J1: the same guarantee, cheaper. The key below costs ~1.6us of the
        # measured window (16 id() calls, a 37-item tuple, its hash and the
        # dict probe); this verifies the ABI-relevant subset by object identity
        # against one cached tuple (~0.4us), which is a strictly stronger test
        # for the arguments it covers -- objects rather than recycled ids --
        # and skips the 21 arguments that cannot reach a computed value (see
        # _HOSTLITE). A mismatch falls through to the key path, so a caller
        # that changes anything at all still gets the fully validated rebuild.
        hot = self._hot
        if hot is not None:
            v = hot[0]
            if (topk_ids is v[0]
                    and hidden_states is v[1]
                    and hidden_states_scale is v[2]
                    and gemm1_weights is v[3]
                    and gemm1_weights_scale is v[4]
                    and gemm2_weights is v[5]
                    and gemm2_weights_scale is v[6]
                    and gemm1_alpha is v[7]
                    and gemm1_beta is v[8]
                    and gemm1_clamp_limit is v[9]
                    and output is v[10]
                    and activation_type == v[11]
                    and num_experts == v[12]
                    and top_k == v[13]
                    and intermediate_size == v[14]
                    and self._env_snap is v[15]):
                mrun, aptr, outl, abi, st = hot[1], hot[2], hot[3], hot[4], hot[5]
                s = _raw_stream(st[1])
                if s != st[0]:
                    # Only a real change pays the ctypes struct store; the
                    # harness keeps one stream for the whole timed loop.
                    abi.stream = s
                    st[0] = s
                # Diagnostics only, but nn.Module.__setattr__ costs ~1.0us of
                # the measured window (three container probes plus two
                # isinstance checks) against ~0.1us for the direct form.
                object.__setattr__(self, "_last_abi", abi)
                code = mrun(aptr)
                if code:
                    raise RuntimeError(f"moe_run failed with CUDA error {code}")
                return outl
        key = (id(topk_ids), id(routing_bias), id(hidden_states),
               id(hidden_states_scale), id(gemm1_weights), id(gemm1_weights_scale),
               id(gemm1_bias), id(gemm1_alpha), id(gemm1_beta),
               id(gemm1_clamp_limit), id(gemm2_weights), id(gemm2_weights_scale),
               id(gemm2_bias), id(output1_scale_scalar),
               id(output1_scale_gate_scalar), id(output2_scale_scalar),
               num_experts, top_k, n_group, topk_group, intermediate_size,
               local_expert_offset, local_num_experts, routed_scaling_factor,
               routing_method_type, do_finalize, enable_pdl, activation_type,
               id(per_token_scale), id(output), tune_max_num_tokens,
               id(gemm1_lora_delta), valid_hidden_size, valid_intermediate_size,
               num_fused_shared_experts, hidden_states_scale_layout,
               self._env_snap)
        hit = self._fast.get(key)
        if hit is not None:
            mrun, aptr, outl, abi = hit[0], hit[1], hit[2], hit[3]
            abi.stream = _raw_stream(self._dev_idx)
            self._last_abi = abi
            code = mrun(aptr)
            if code:
                raise RuntimeError(f"moe_run failed with CUDA error {code}")
            return outl
        lib = _lib()
        ids, scores = topk_ids
        if ids.dim() != 2:
            raise ValueError("topk_ids[0] must be [tokens, top_k]")
        device = hidden_states.device
        self._dev_idx = (device.index if device.index is not None
                         else torch.cuda.current_device())
        T, K = int(ids.shape[0]), int(ids.shape[1])
        E = int(gemm1_weights.shape[0])
        N1 = int(gemm1_weights.shape[1])          # prepared payload rows = 2*I
        I = N1 // 2
        # hidden_states is the packed E2M1 payload: two values per byte.
        H = int(hidden_states.shape[1]) * 2
        if tuple(gemm1_weights.shape) != (E, N1, H // 2):
            raise ValueError(f"unexpected gemm1_weights shape {tuple(gemm1_weights.shape)}")
        if tuple(gemm2_weights.shape) != (E, H, I // 2):
            raise ValueError(f"unexpected gemm2_weights shape {tuple(gemm2_weights.shape)}")
        if tuple(hidden_states_scale.shape) != (T, H // 16):
            raise ValueError(f"unexpected hidden_states_scale shape "
                             f"{tuple(hidden_states_scale.shape)}")
        if output is None:
            raise ValueError("the frozen contract always supplies an output tensor")
        if E > 256:
            raise ValueError(f"expert count outside the supported scan width: E={E}")
        if (H % 128) or (I % 64) or (N1 % 128):
            raise ValueError(f"shape outside the declared prepared-weight domain: "
                             f"H={H} I={I}")
        if H != int(gemm2_weights.shape[1]) or I != int(gemm2_weights.shape[2]) * 2:
            raise ValueError("inconsistent weight shapes")
        pre_c = (ids, scores, hidden_states, hidden_states_scale, gemm1_weights,
                 gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                 gemm1_alpha, gemm1_beta, gemm1_clamp_limit)
        ids = _c(ids)
        scores = _c(scores)
        hidden_states = _c(hidden_states)
        hidden_states_scale = _c(hidden_states_scale)
        gemm1_weights = _c(gemm1_weights)
        gemm1_weights_scale = _c(gemm1_weights_scale)
        gemm2_weights = _c(gemm2_weights)
        gemm2_weights_scale = _c(gemm2_weights_scale)
        gemm1_alpha = _c(gemm1_alpha)
        gemm1_beta = _c(gemm1_beta)
        limit = None if gemm1_clamp_limit is None else _c(gemm1_clamp_limit)
        # The fast path may only be populated when _c() returned the caller's
        # objects unchanged: a contiguous() COPY would freeze stale bytes into
        # the cached ABI while the caller keeps mutating the originals.
        no_copies = (ids is pre_c[0] and scores is pre_c[1]
                     and hidden_states is pre_c[2]
                     and hidden_states_scale is pre_c[3]
                     and gemm1_weights is pre_c[4]
                     and gemm1_weights_scale is pre_c[5]
                     and gemm2_weights is pre_c[6]
                     and gemm2_weights_scale is pre_c[7]
                     and gemm1_alpha is pre_c[8] and gemm1_beta is pre_c[9]
                     and limit is pre_c[10])

        self.ensure_tables(device)

        R = T * K
        cfg = _pick_cfg(H, I, E, R)
        override = int(os.environ.get("NVFP4_MOE_CFG", "-1"))
        if override >= 0:
            cfg = override
        # GEMM2 may use a different rung, but both GEMMs share one route tile
        # map, so they must agree on BM (128 for every tcgen05 rung). GEMM2
        # pairing (measured in job dv_fa956693910c): the flat-stream rung 23
        # wins big on nk==1 geometries (I=256, deepseek: fixed_4 stage
        # 177.3 -> 134.4 us, e2e 454.4 -> 421.0) but loses on I>=768 where
        # the boundary epilogue inside the stream costs more than the
        # per-item prime it saves (fixed_6 stage 204 -> 225), so it is paired
        # exactly when I == 256. Rungs 13/14/16/18/20/21 are GEMM1-specific.
        # Capture-time routing statistic (dispatch-only; shared by the G2GATE
        # ambiguity band and the MEGA empty gate; computed at most once per
        # slow-path call, and only when a gate asks). Counts the CURRENT ids
        # exactly as k_count does (valid = 0 <= id < E). No numerics leave
        # the device graph and nothing is cached across calls: the statistic
        # only selects among kernel variants whose outputs agree within the
        # frozen tolerance (GEMM2 leaves) or are exactly the zero contract
        # (empty routing), and every replay still recomputes from its own
        # current ids on-device.
        _route_hist = None
        def route_hist():
            nonlocal _route_hist
            if _route_hist is None:
                h = {}
                for row in ids.tolist():
                    for v in row:
                        if 0 <= v < E:
                            h[v] = h.get(v, 0) + 1
                _route_hist = h
            return _route_hist

        cfg2 = int(os.environ.get("NVFP4_MOE_CFG2", "-1"))
        if cfg2 < 0:
            if ((17 <= cfg <= 35 and cfg not in (22, 23))
                    or cfg in (37, 39, 40, 41, 42, 43, 45, 48, 49, 50, 51,
                               54, 55, 56, 58, 59, 63, 64, 65, 69,
                               70, 74, 75, 76, 77, 78, 79, 82, 88, 90,
                               95, 96, 97, 98, 99, 101, 103, 104)):
                # C16: GEMM2 default 19 -> 44 (SF staged by cp.async, SFCP):
                # bit-identical (cmp44) and won all six job-AF sweep shapes
                # (fixed_6 -2.1%, fixed_3 -3.4%, fixed_2 -1.6%, domain_2
                # -2.6%, domain_7 -1.0%, domain_3 -0.7%). SFCP on the GEMM1
                # side (rung 45) loses (+4..13%: STAGES=2 leaves the cp.async
                # wait exposed) and stays an override-only rung.
                # REDF (rung 46 = rung 44 + fused routing reduction) is
                # env-gated until the paired A/B promotes it: it replaces the
                # [R,H] rout buffer + scattered-read k_finalize with fp32
                # atomics into [T,H] + a k_convert final rounding.
                redf = os.environ.get("NVFP4_MOE_REDF", "0") not in ("0", "", None)
                # GEMM2 WS3 (57) replaces 44 AND flat 23 except on tiny work
                # spaces, where the role-split startup is not amortized:
                # ~128 items 0.915x, ~640 items 0.985x, ~2048 items +1.2%,
                # >= ~4096 items +3-6% (jobs dv_c04069a2255d +
                # dv_17f07be8b7ae); at I=256 it beat flat-23 on domain_6
                # (1.008x, dv_616bb3921e2f) but LOST on domain_1 (R=24:
                # 0.895x combined, dv_496c03fc436f) because each item there
                # holds a single k-set stage (K=I=256), so the gate must
                # count STAGES, not items: items x (I//256) >= 3072 keeps
                # the calibrated I=768 boundary (1024 items) and the
                # measured I=256 wins (domain_6/fixed_4: 14336) while
                # returning domain_1 (1344) to flat 23. Below the gate
                # I=256 keeps flat 23 and the rest keeps monolithic 44.
                hi_units = min(E, R) * (H // 128) * max(1, I // 256)
                # G2GATE: the 3072 boundary is calibrated in REAL routed tile
                # items, but min(E, R) is only the spread-routing UPPER bound
                # on m-tiles; concentrated routing overshoots it 16x. fixed_1
                # (T=32, hotspot -> 8 real m-tiles vs bound 128) runs GEMM2
                # deep in the measured 0.915x lose-regime: the mono leaf wins
                # 0.8845x e2e, bit-exact (dv_5ab45d03329d). No shape-only
                # proxy separates that from balanced routing of the same
                # size (domain_7: T=31, R=248 -> ~128 real m-tiles, WS3 wins
                # in band), so when -- and only when -- the provable bounds
                # straddle the threshold (lo: every route valid and piled on
                # the fewest experts, m-tiles >= ceil(R/bm); hi: every route
                # valid and spread), count the real route tiles from the
                # current ids, mirroring k_count's validity filter [0, E)
                # and k_scan's per-expert ceil(cnt/bm). Dispatch statistic
                # only: every GEMM2 leaf is bit-identical, the launch grid
                # stays bound-sized and the tile map is recomputed on-device
                # every replay, so ANY replay content still computes the
                # correct output; the count runs once per graph capture on
                # the slow path, never inside the measured replay fast path.
                # PROMOTED default ON (dv_e0b98925d4f1, all columns
                # torch.equal bit-exact): flips measured fixed_1 0.9442 /
                # fixed_8 0.9579 (empty) / domain_9 0.9542 (boundary);
                # in-band-stays and out-of-band nulls 0.9934-1.0007.
                # GEMM1 axis at fixed_1's 96 real items re-verified in the
                # same job: mono SK4 43 = 1.119x, fused-reduce 31 = 1.475x
                # -- the SK-band leaf 78 stays; results/g2gate-promote.md.
                if (hi_units >= 3072
                        and os.environ.get("NVFP4_MOE_G2GATE", "1")
                        not in ("0", "", None)):
                    bm_g = CFGS[cfg][0]
                    per_unit = (H // 128) * max(1, I // 256)
                    if ((R + bm_g - 1) // bm_g) * per_unit >= 3072:
                        ws_g2 = True      # densest legal routing amortizes
                    else:                 # WS3; no readback needed
                        real = 0
                        for c in route_hist().values():
                            real += (c + bm_g - 1) // bm_g
                        ws_g2 = real * per_unit >= 3072
                else:
                    ws_g2 = hi_units >= 3072
                # TSTORE promotion on the unified tree (dv_5776eb5e1161:
                # bit-exact WIN fixed_4 0.976 / fixed_5 0.959 / fixed_6
                # 0.941 / fixed_7 0.976 / domain_3 0.960 / domain_6
                # 0.842, zero losses): the S2G bulk-tensor rout drain
                # replaces WS3 57 in-band. Generic shape gate T*K >= 32
                # mirrors the drain tensor map's 32-row box against the
                # rout slab (Rmax = T*K); below it -- unreachable for
                # ws_g2 on every legal geometry except micro-T corners
                # like T=2,K=8 (ws_g2 exactly 3072, Rmax=16) -- keep the
                # bit-identical 57.
                # DEEPxTSTORE promotion (rung 81 = 80 + the J6 DEEP wait
                # split; wait-topology only, bit-exact by construction,
                # identical smem/occupancy): confirmed under TWO
                # independent pre-registrations -- dv_afd94dc7d4fd
                # (6/6 direction-consistent, mean 0.9907) and
                # dv_f6174956a378 (8/8, mean 0.9898, ALTERNATED variant
                # order, 5x7 samples, adds domain_2/domain_5) =>
                # PROMOTE-PROCEED per the fresh rule (mean <= 0.993,
                # >= 6/8 <= 1.0, max <= 1.010, bit-exact). Their J6
                # measured DEEP neutral on GEMM2 pre-TSTORE (1.0007x);
                # the drain offload changed the cadence and re-exposed
                # the w1 payload-vs-SF wait coupling exactly as it had
                # on GEMM1 (-4.3..-5.6% stage there).
                tstore_ok = T * K >= 32
                # CPGATE-v2: rung 81 is the ONLY GEMM2 WS3 form again; the
                # CPHOIST boundary hoist is gated at RUNTIME inside the kernel
                # on the shape-only average-fill proxy Rmax >= 16*E (see the
                # f4_body_ws hoist condition). The separate rung-84 binary was
                # withdrawn: the standalone (EPITB,!CPHOIST) instantiation
                # deterministically corrupted GEMM2 output on low-fill shapes
                # (dv_f64f43b5f600 / dv_dfc3c4a7872a triage: 653/1024 rout
                # rows non-finite, upstream clean, sanitizer 0 errors) while
                # the hoist-off path INSIDE the CPHOIST binary is exercised
                # cleanly by every CTA's first item across five jobs.
                cfg2 = (81 if tstore_ok else 57) if (ws_g2 and not redf) else (
                    23 if I == 256 else (46 if redf else 44))
            elif cfg in (13, 14, 16, 18):
                cfg2 = 12
            else:
                cfg2 = cfg

        def rung_ok(c):
            bm_, bn_, bk_ = CFGS[c]
            if c == 96:
                # SELFPREP scope: chist holds 4 chunks of 64 routes, so
                # R <= 256 is a CORRECTNESS bound; SK2 divisibility as 78.
                return (R <= 256 and _rung7_ok(H, I)
                        and (H // 256) % 2 == 0)
            if c in (91, 92):
                # MEGA decode paths (correctness scope): K=8 finalize route
                # arrays; 32-wide I chunks (two QDQ blocks); 4-h thread
                # windows over H; x-window smem (H + H/8 + 96) floats in
                # the 48 KB default budget. Rung 91 additionally stages
                # 48*H weight bytes per unit, which must fit the 227 KB
                # opt-in budget (H=2048: 105 KB, H=4096: 211 KB, H>=7168:
                # excluded). The PERFORMANCE scope (R<=64 + weight bytes,
                # or capture-time-empty) lives in the dispatch gate;
                # forcing NVFP4_MOE_CFG bypasses only that, never this.
                ok = (K == 8 and I % 32 == 0 and H % 256 == 0
                      and (H + H // 8 + 96) * 4 + 64 <= 48 * 1024)
                if c == 91:
                    ok = ok and (48 * H + (H + H // 8 + 96) * 4 + 64
                                 <= 227 * 1024)
                return ok
            if c in (20, 21, 28, 29, 31, 32, 34, 35, 37, 42, 43, 54, 55,
                     58, 59, 64, 75, 76, 77, 78, 79, 88, 95, 98, 99):
                # Split-K additionally requires nk = H/256 stages to divide by
                # the slice count (nk is 8, 16 or 28; rung 37's eight-way
                # split therefore excludes H=7168, where it falls back).
                return (_rung7_ok(H, I)
                        and (H // 256) % (8 if c == 37 else
                                          (4 if c in (20, 29, 31, 35, 43, 55,
                                                        59, 76, 77)
                                           else 2)) == 0)
            if c >= 7:
                # rungs 7-15 differ only in copy-pipeline depth, N tile, copy
                # mechanism (cp.async vs TMA vs TMA gather4) and, for 13/14,
                # the BN=64 two-buffer two-CTA tiling; the N=64 rungs
                # additionally need both feature dims divisible by 64, which
                # _rung7_ok's 128-divisibility already implies. The TMA rungs
                # also need the tensor maps to be encodable -- a row of K/2
                # payload bytes that is 16-byte aligned and at least 128 bytes
                # long -- and launch_gemm falls back to the matching cp.async
                # rung by itself when they are not, so selecting them is safe
                # for every shape that passes this gate.
                return _rung7_ok(H, I)
            if c >= 4:
                bn_g1 = 256 if c == 5 else 128
                return (I % bn_g1 == 0 and H % 256 == 0
                        and H % bk_ == 0 and I % bk_ == 0)
            return ((2 * I) % bn_ == 0 and H % bn_ == 0
                    and H % bk_ == 0 and I % bk_ == 0)

        def fam(c):
            """Intermediate-representation family.

            0 = warp-mma rungs (fc1 then bf16 act), 1 = tcgen05 bf16 rungs
            (bf16 act written by the fused epilogue), 2 = tcgen05 block-scaled
            FP4 (the activation stays packed as payload + E4M3 scale). The two
            GEMMs must agree on the family because they share that buffer; 0 and
            1 both consume bf16 and may be mixed.
            """
            return 2 if c >= 7 else (1 if c >= 4 else 0)

        if not rung_ok(cfg):
            cfg = _pick_cfg(H, I, E, R)
        if not rung_ok(cfg2):
            cfg2 = _pick_cfg(H, I, E, R)
        if fam(cfg) != fam(cfg2) and 2 in (fam(cfg), fam(cfg2)):
            cfg = cfg2 = 7 if rung_ok(7) else _pick_cfg(H, I, E, R)
        elif (cfg >= 4) != (cfg2 >= 4):
            partner = 0 if rung_ok(0) else min(cfg, cfg2)
            if cfg >= 4:
                cfg2 = partner
            else:
                cfg = partner
            if (cfg >= 4) != (cfg2 >= 4):
                cfg = cfg2 = _pick_cfg(H, I, E, R)
        # MEGA (rung 91): barrier-free single-kernel decode path for tiny
        # routed work (R = T*K <= 64). The 6-node graph chain's launch
        # cadence dominates these shapes (case0 28.7 us vs ~19 MB of weight
        # bytes = ~5.4 us of stream); one kernel with per-route fp32 atomics
        # + per-token arrival tickets removes every boundary. Both barrier-
        # based consolidations measured WORSE on this device: conditional
        # graph nodes +9 us/replay (dv_48fec882fc3b) and cooperative
        # grid.sync 1.4-4.0 us/barrier (dv_f75228c35259). Knob default OFF
        # until the paired A/B prices it; explicit rung overrides win.
        # MEGA: barrier-free single-kernel decode paths. PROMOTED for the
        # EMPTY-capture scope (rung 92, direct variant, MEGA default "1"):
        # when the capture-time route histogram is empty at R <= 8192, the
        # whole forward collapses from ~6 graph nodes to 1 (pass-0 output
        # zeroing + census), bit-exact vs the promoted tree (dv_a94224386668:
        # fixed_8 0.6207, domain_4 0.5271, synth_T2_empty 0.7103; every
        # replay still recomputes from current ids -- a non-empty replay
        # takes the slow-but-exact direct path, so correctness never
        # depends on the statistic). MEGA="2" additionally enables the
        # experimental STAGED compute scope (rung 91, R<=64 + <=48MB
        # weights) which still LOSES to the tensor pipeline (fixed_0
        # 1.4455, domain_0 1.8955; NCU: L1TEX-scoreboard-bound SIMT decode
        # chains) and stays override/experiment-only. Explicit rung
        # overrides win over both. results/mega-promote.md.
        mega_env = os.environ.get("NVFP4_MOE_MEGA", "1")
        if (mega_env not in ("0", "", None)
                and override < 0
                and int(os.environ.get("NVFP4_MOE_CFG2", "-1")) < 0
                and K == 8):
            if (R <= 8192 and R * H <= 32_000_000 and rung_ok(92)
                    and not route_hist()):
                cfg = cfg2 = 92
            elif (mega_env == "2" and R <= 64
                    and min(E, R) * 3 * I * H <= 96_000_000
                    and rung_ok(91)):
                cfg = cfg2 = 91
        # SELFPREP (rung 96): when the default GEMM1 leaf is the SK-band 78
        # and the route count is inside the in-grid-prep scope, recompute
        # the routing map inside the GEMM1 grid and drop the prep node.
        # Knob default OFF pending the paired A/B; explicit overrides win.
        if (os.environ.get("NVFP4_MOE_SELFPREP", "0") not in ("0", "", None)
                and override < 0
                and int(os.environ.get("NVFP4_MOE_CFG2", "-1")) < 0
                and cfg == 78 and rung_ok(96)):
            cfg = 96
        # Routing-aware small/big-path threshold (results/threshold-pursue.md,
        # corrected-protocol job 20261027T015803Z): the big path's four
        # parallel prep nodes beat the single-block k_prepare_small by
        # 0.95-2.2% at NON-EMPTY R=4096, while the single block keeps its
        # ~4% edge only for EMPTY routing outside the mega scope (R*H > 32M).
        # Explicit env override wins; the readback (route_hist, cached, same
        # helper as MEGA/G2GATE) runs only inside the (3000,6500] band, at
        # publish time, never in the measured replay. Dispatch statistic
        # only: both prep paths are bit-identical, so an in-place data
        # change behind a captured graph can only shift performance, never
        # correctness.
        st_env = _env_get("NVFP4_MOE_SMALL_THRESH")
        if st_env:
            small_thresh = int(st_env)
        elif 3000 < R <= 6500 and route_hist():
            small_thresh = 3000
        else:
            small_thresh = 6500
        # ZERO-CAPTURE (NVFP4_MOE_ZERO): the declared zero distributions
        # (fixed_9, domain_6/9) make the exact answer the zero contract for
        # ANY routing: x=0 => fc1=0 => act=0 => rout=0 => out=0 (no biases
        # exist in the frozen math). The capture-time probe (one-time device
        # scan at publish) accepts only all-zero payloads with finite
        # scales/scores/params AND finite weight scales; acceptance adds a
        # leading k_zero_verify node that RE-DERIVES the property from the
        # current bytes on EVERY replay (writing the zero out and flagging
        # the chain to early-exit) -- a dirty replay runs the full exact
        # chain and overwrites out. Same contract class as MEGA-92's empty
        # capture; correctness never depends on the statistic. Scoped to the
        # promoted GEMM families whose kernels all carry the ZGUARD.
        # PROMOTED default ON (dv_fc972229e3ca, 3 interleaved rounds:
        # fixed_9 0.2558, domain_9 0.3158, domain_6 0.1839; nulls
        # digest-equal; escape 3/3 exact).
        # ZERO2 PROMOTED as the default mode (dv_d2fd1c858e02 + r1
        # dv_f232cec991d9): two-graph replay — G_clean=[k_zero_verify] only,
        # G_full=the mode-1 chain; moe_run launches G_clean, bounded
        # host-spin on the pinned verdict mirror (zflag_host), and launches
        # G_full ONLY on dirty/timeout (correct under every interleaving;
        # escape tests bit-exact vs the full-chain arm with non-zero dirty
        # outputs). Clean-path wins: f9 1.464x (bimodal band collapsed),
        # d6 1.755x, d9 ~1.02x; controls flat. ZERO=1 restores the mode-1
        # single-graph behavior exactly; ZERO=0 disables the shortcut.
        zero_ok = False
        if (os.environ.get("NVFP4_MOE_ZERO", "2") not in ("0", "", None)
                and override < 0
                and int(os.environ.get("NVFP4_MOE_CFG2", "-1")) < 0
                and cfg in (74, 78, 97, 99) and cfg2 in (23, 44, 57, 81)
                and K == 8):
            try:
                zero_ok = lib.moe_zero_probe(
                    _ptr(hidden_states), _ptr(hidden_states_scale),
                    _ptr(scores), _ptr(gemm1_alpha), _ptr(gemm1_beta),
                    _ptr(limit), 0 if gemm1_clamp_limit is None else 1,
                    _ptr(gemm1_weights_scale), E * 2 * I * (H // 16),
                    _ptr(gemm2_weights_scale), E * H * (I // 16),
                    T, K, E, H, _raw_stream(self._dev_idx)) == 1
            except Exception:
                zero_ok = False
        zflag_t = (torch.zeros(4, dtype=torch.int32, device=device)
                   if zero_ok else None)
        bm = CFGS[cfg][0]
        sk = (8 if cfg == 37 else
              (4 if cfg in (20, 29, 31, 35, 43, 55, 59, 76, 77, 95)
               else (2 if cfg in (21, 28, 32, 34, 42, 54, 58, 64, 75, 78, 79,
                                  88, 96, 98, 99)
                    else 1)))
        # Only experts that received at least one route can produce a tile, so
        # the host-side upper bound is much tighter than ceil(R/bm) + E for
        # small batches; this is what keeps the launched grid from being
        # dominated by empty blocks.
        tiles = (R + bm - 1) // bm + min(E, R)
        ws = self._workspace((T, K, E, H, I, tiles, cfg, cfg2),
                             {"T": T, "K": K, "E": E, "H": H, "I": I, "R": R,
                              "tiles": tiles, "cfg": cfg, "cfg2": cfg2,
                              "sk": sk}, device)

        values = {
            "ids": _ptr(ids), "scores": _ptr(scores),
            "hid_q": _ptr(hidden_states), "hid_sf": _ptr(hidden_states_scale),
            "w1_q": _ptr(gemm1_weights), "w1_sf": _ptr(gemm1_weights_scale),
            "w2_q": _ptr(gemm2_weights), "w2_sf": _ptr(gemm2_weights_scale),
            "alpha": _ptr(gemm1_alpha), "beta": _ptr(gemm1_beta),
            "limit": _ptr(limit), "out": _ptr(output),
            "T": T, "K": K, "E": E, "H": H, "I": I, "Rmax": R,
            "max_tiles": tiles, "act_type": int(activation_type),
            "has_limit": 0 if gemm1_clamp_limit is None else 1,
            "bm": bm, "cfg": cfg, "cfg2": cfg2, "sk": sk,
            "stream": _raw_stream(self._dev_idx),
        }
        for name in ("counts", "fill", "offs", "toffs", "ptok", "pslot", "pexp",
                     "route_of", "texp", "tm0", "hid_bf", "fc1", "act", "rout",
                     "part", "hidq_g", "hidsf_g", "cnt", "acc32"):
            # A buffer the selected rung does not use is not allocated; the ABI
            # field is then a null pointer, which no launched kernel dereferences.
            values[name] = 0 if ws[name] is None else _ptr(ws[name])
        values["cnt_n"] = int(ws["cnt_n"])
        values["small_thresh"] = small_thresh
        values["zflag"] = _ptr(zflag_t)
        # ZERO2 (NVFP4_MOE_ZERO=2): pinned host mirror of zflag[0]. C-side
        # moe_run then replays a verify-ONLY clean graph and host-spins on
        # this word, launching the full chain graph only on a dirty/timeout
        # verdict (correct under every interleaving). The tensor must stay
        # alive as long as the published ABI -- the keep tuple below owns it.
        zhost_t = None
        # T*H >= 1<<20 shape gate (official dv_c6a32c55c0dc): mode 2's
        # measured span in the EAGER official protocol is the HOST round
        # trip (~16-20us), which only beats the mode-1 chain span when
        # the chain is expensive: f9 (T=512,H=2048=1M) 24.5->16.5 win,
        # d6 57.3->32.4 win, but d9 (T=5, T*H=20K) 16.35->19.90 LOSS —
        # its ultra-light chain is faster than any host round trip. Gate
        # at the proven win boundary on declared shape metadata only;
        # below it the certified mode-1 chain stays.
        if (zero_ok and os.environ.get("NVFP4_MOE_ZERO", "2") == "2"
                and T * H >= (1 << 20)):
            zhost_t = torch.zeros(1, dtype=torch.int32, device="cpu",
                                  pin_memory=True)
            zhost_t.fill_(-1)
        values["zflag_host"] = _ptr(zhost_t)
        abi = _ABI(*[int(values[name]) for name in _ABI_FIELDS])
        self._last_abi = abi          # diagnostics only (experiments/profile_kernels.py)
        self._last_meta = {"T": T, "K": K, "E": E, "H": H, "I": I, "cfg": cfg,
                           "cfg2": cfg2, "bm": bm, "tiles": tiles, "R": R,
                           "grid_cap": self._grid_cap,
                           "zero": int(zero_ok)}
        code = lib.moe_run(ctypes.addressof(abi))
        if code:
            raise RuntimeError(f"moe_run failed with CUDA error {code}")
        if no_copies:
            if len(self._fast) >= 16:
                self._fast.clear()
            aptr = ctypes.addressof(abi)
            keep = (topk_ids, routing_bias, hidden_states,
                    hidden_states_scale, gemm1_weights,
                    gemm1_weights_scale, gemm1_bias,
                    gemm1_alpha, gemm1_beta,
                    gemm1_clamp_limit, gemm2_weights,
                    gemm2_weights_scale, gemm2_bias,
                    output1_scale_scalar,
                    output1_scale_gate_scalar,
                    output2_scale_scalar, per_token_scale,
                    output, gemm1_lora_delta, ws, zflag_t, zhost_t)
            self._fast[key] = (lib.moe_run, aptr, [output], abi, lib, keep)
            # J1: publish the identity-verified entry. Every object named here
            # is the caller's own -- that is exactly what no_copies asserts --
            # so an identity match on a later call proves each pointer in `abi`
            # is still the caller's. `keep` (index 6) owns the strong
            # references, including the workspace that backs every non-caller
            # pointer, and `abi` itself, which the int address does not.
            if _HOSTLITE:
                # CFAST2: index 7 publishes the raw C address of moe_run so the
                # extension can call it without ctypes marshalling. 0 on the
                # (impossible-in-practice) parse failure, which the C side
                # treats as a bad publish and falls back to forward().
                fnaddr = ctypes.cast(lib.moe_run, ctypes.c_void_p).value or 0
                object.__setattr__(self, "_hot", (
                    (topk_ids, hidden_states, hidden_states_scale,
                     gemm1_weights, gemm1_weights_scale, gemm2_weights,
                     gemm2_weights_scale, gemm1_alpha, gemm1_beta,
                     gemm1_clamp_limit, output, activation_type, num_experts,
                     top_k, intermediate_size, self._env_snap),
                    lib.moe_run, aptr, [output], abi,
                    [values["stream"], self._dev_idx], keep, fnaddr))
                return [output]
        else:
            # A copy was made, so this call's ABI must not be reused: it would
            # freeze a snapshot the caller keeps mutating. Drop any entry from
            # an earlier call so an identity match cannot resurrect it.
            object.__setattr__(self, "_hot", None)
        return [output]

    # Direct binding: the *args/**kwargs re-pack hop costs ~0.4us per call
    # and the official eager protocol exposes host time inside the events.
    __call__ = forward

    def _graph_active(self):
        """True when the replayed CUDA-graph path is in use (diagnostics)."""
        lib = _lib()
        try:
            lib.moe_graph_active.restype = ctypes.c_int64
        except AttributeError:
            return False
        return bool(lib.moe_graph_active())

    def graph_status(self):
        """[begin_capture, pipeline, end_capture, instantiate, pending_hits]."""
        lib = _lib()
        out = (ctypes.c_int64 * 5)()
        lib.moe_graph_status(out)
        return list(out)

    def set_graph(self, enable):
        lib = _lib()
        lib.moe_set_graph.argtypes = [ctypes.c_int64]
        lib.moe_set_graph.restype = None
        lib.moe_set_graph(ctypes.c_int64(1 if enable else 0))

    def set_grid_cap(self, cap):
        """Cap the persistent GEMM grid (experiments only; 0 = uncapped).

        The GEMM kernels walk their item space with a grid stride, so any grid
        shape computes the same result; the cap only limits how many CTAs are
        launched when the host-side tile bound is much larger than the work.
        """
        lib = _lib()
        lib.moe_set_grid_cap.argtypes = [ctypes.c_int64]
        lib.moe_set_grid_cap.restype = ctypes.c_int64
        self._grid_cap = int(lib.moe_set_grid_cap(ctypes.c_int64(int(cap))))
        return self._grid_cap

    def grid_cap(self):
        lib = _lib()
        try:
            lib.moe_grid_cap.restype = ctypes.c_int64
        except AttributeError:
            return 0
        return int(lib.moe_grid_cap())

    def tma_status(self):
        """Bitmask for the TMA rung: 1 = driver entry point resolved,
        2 = GEMM1 weight map cached, 4 = GEMM2 weight map, 8 = GEMM2 intermediate
        map. Zeroes in bits 1-3 after a rung-12 forward mean launch_gemm fell back
        to rung 8, so an A/B of the two rungs cannot silently measure the same
        code twice."""
        lib = _lib()
        try:
            lib.moe_tma_status.restype = ctypes.c_int64
        except AttributeError:
            return -1
        return int(lib.moe_tma_status())

    STAGES = ["memset", "count", "scan", "scatter", "tiles", "decode_hidden",
              "gemm1", "act", "gemm2", "finalize"]

    def profile_stages(self, reps=5):
        """Per-stage device nanoseconds for the most recent forward (diagnostic)."""
        lib = _lib()
        lib.moe_profile.argtypes = [ctypes.POINTER(_ABI),
                                    ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
        lib.moe_profile.restype = ctypes.c_int
        out = (ctypes.c_int64 * len(self.STAGES))()
        code = lib.moe_profile(ctypes.byref(self._last_abi), out, reps)
        if code:
            raise RuntimeError(f"moe_profile failed: {code}")
        return list(out)


if _FASTMOD is not None:
    # J5: replace the python __call__ (which is `forward`, already bound
    # directly to skip nn.Module._call_impl) with a C descriptor. `instance(...)`
    # reaches slot_tp_call, which binds an attribute only if it implements
    # tp_descr_get -- a plain C function assigned here would be called WITHOUT
    # the instance -- so nvfp4_fast.Call is a descriptor whose __get__ returns a
    # small bound object whose tp_call runs the cached call. That skips the
    # frame and the 36-name binding entirely and reads _hot straight off the
    # instance, so no state lives in the extension and nothing can outlive the
    # model. Anything it does not recognise falls through to forward(), whose
    # own J1 path re-checks the same identity condition, so the two layers
    # cannot disagree about what is cached.
    Model.__call__ = _FASTMOD.Call
