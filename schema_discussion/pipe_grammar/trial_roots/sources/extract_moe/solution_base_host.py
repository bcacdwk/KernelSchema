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
    "cnt", "cnt_n", "acc32",
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
    handle.moe_run.argtypes = [ctypes.POINTER(_ABI)]
    handle.moe_run.restype = ctypes.c_int
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
        (128, 64, 256)]   # rung 70: DIAG WS4 GEMM1 fused + PP acc
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
        if H >= 7168:
            if R <= 256:
                return 43
            if R <= 1536:
                return 58
            return 56
        if min(E, R) <= 32:
            return 43
        if R <= 1536:
            return 58
        return 56
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
        # Resolved device index for the raw current-stream getter; the slow
        # path overwrites it from the actual input device before any fast hit.
        self._dev_idx = 0
        # Mirrors the launcher's persistent-grid cap so that diagnostics can
        # report it without a host round-trip inside the measured region.
        # -1 = automatic (two waves of resident CTAs), matching the launcher's
        # own default; 0 would mean "uncapped" and misreport the diagnostics.
        self._grid_cap = int(os.environ.get("NVFP4_MOE_GRID_CAP", "-1") or -1)

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
                     if cfg in (31, 32) else 0)
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
                "acc32": (torch.empty(sizes["T"] * sizes["H"],
                                      dtype=torch.float32, device=device)
                          if cfg2 in (46, 47) else None),
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
                                                 58, 59, 65, 69, 70))
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
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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
               _env_get("NVFP4_MOE_CFG"), _env_get("NVFP4_MOE_CFG2"),
               _env_get("NVFP4_MOE_REDF"))
        hit = self._fast.get(key)
        if hit is not None:
            abi, lib, _keep = hit
            abi.stream = _raw_stream(self._dev_idx)
            self._last_abi = abi
            code = lib.moe_run(ctypes.byref(abi))
            if code:
                raise RuntimeError(f"moe_run failed with CUDA error {code}")
            return [output]
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
        cfg2 = int(os.environ.get("NVFP4_MOE_CFG2", "-1"))
        if cfg2 < 0:
            if ((17 <= cfg <= 35 and cfg not in (22, 23))
                    or cfg in (37, 39, 40, 41, 42, 43, 45, 48, 49, 50, 51,
                               54, 55, 56, 58, 59, 63, 64, 65, 69,
                               70)):
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
                ws_g2 = (min(E, R) * (H // 128) * max(1, I // 256)) >= 3072
                cfg2 = 57 if (ws_g2 and not redf) else (
                    23 if I == 256 else (46 if redf else 44))
            elif cfg in (13, 14, 16, 18):
                cfg2 = 12
            else:
                cfg2 = cfg

        def rung_ok(c):
            bm_, bn_, bk_ = CFGS[c]
            if c in (20, 21, 28, 29, 31, 32, 34, 35, 37, 42, 43, 54, 55,
                     58, 59, 64):
                # Split-K additionally requires nk = H/256 stages to divide by
                # the slice count (nk is 8, 16 or 28; rung 37's eight-way
                # split therefore excludes H=7168, where it falls back).
                return (_rung7_ok(H, I)
                        and (H // 256) % (8 if c == 37 else
                                          (4 if c in (20, 29, 31, 35, 43, 55,
                                                        59)
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
        bm = CFGS[cfg][0]
        sk = (8 if cfg == 37 else
              (4 if cfg in (20, 29, 31, 35, 43, 55, 59)
               else (2 if cfg in (21, 28, 32, 34, 42, 54, 58, 64) else 1)))
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
        abi = _ABI(*[int(values[name]) for name in _ABI_FIELDS])
        self._last_abi = abi          # diagnostics only (experiments/profile_kernels.py)
        self._last_meta = {"T": T, "K": K, "E": E, "H": H, "I": I, "cfg": cfg,
                           "cfg2": cfg2, "bm": bm, "tiles": tiles, "R": R,
                           "grid_cap": self._grid_cap}
        code = lib.moe_run(ctypes.byref(abi))
        if code:
            raise RuntimeError(f"moe_run failed with CUDA error {code}")
        if no_copies:
            if len(self._fast) >= 16:
                self._fast.clear()
            self._fast[key] = (abi, lib, (topk_ids, routing_bias, hidden_states,
                                          hidden_states_scale, gemm1_weights,
                                          gemm1_weights_scale, gemm1_bias,
                                          gemm1_alpha, gemm1_beta,
                                          gemm1_clamp_limit, gemm2_weights,
                                          gemm2_weights_scale, gemm2_bias,
                                          output1_scale_scalar,
                                          output1_scale_gate_scalar,
                                          output2_scale_scalar, per_token_scale,
                                          output, gemm1_lora_delta, ws))
        return [output]

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
