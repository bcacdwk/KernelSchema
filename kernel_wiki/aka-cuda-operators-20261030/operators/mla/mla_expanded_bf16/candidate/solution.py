"""Autonomous candidate for mla_expanded_bf16 (D07) on L20D/SM103.

Three independently written CUDA kernels, all raw CUDA C++/PTX, no libraries:
  * kernel.cu     — FlashAttention-2 style mma.sync m16n8k16 bf16 pipeline
                    (templated configs; tiny-total_q many-batch regime).
  * kernel_tc.cu  — tcgen05 (Blackwell UMMA) FA4-style warp-specialized
                    kernel (sm_103a), BM=128, direct-TMA raw K/V loads.
  * kernel_tc2.cu — dual-subtile tcgen05 kernel (sm_103a): BM=256 as
                    2x128-row subtiles, BN=64, 3 KV stages, direct-TMA.
                    Halves KV bytes per MMA flop vs BM=128 (job7
                    dv_b57a30f277f6: fixed geomean 173.7->147.3us, domain
                    110.4->94.2us; ncu c5 tensor-pipe 42%->58%).
  Campaign deltas (all A/B-measured vs immutable v0 under the official
  eager+restore protocol, same-job pairs):
  - direct-TMA KV loads over rank-4 maps of the raw [total_kv, kv_heads,
    192/128] tensors (8-token SW128 boxes; no prep kernel, no workspace).
  - lagged running max (2^8 threshold): alpha == 1.0f exactly => TMEM
    O-correction skipped.
  - L2 prefetch kept ON in tc1 (removal regressed; job4). tc2: no prefetch.

This module only compiles/loads them and binds the exact public ABI:
  * out/lse are caller-owned; written in place and returned with identity
    preserved. Return (out, lse) iff return_lse, else out.
  * lse is log-base-2; sinks apply as logaddexp2 plus output rescale, exactly
    matching problem/reference.py.
  * bmm1_scale folds into scores; bmm2_scale scales the output.
  * All other input storage is read-only; workspace_buffer untouched.
  * No host synchronization: sequence metadata is consumed on device.

Kernel choice: MLA_FORCE=sync|tc forces a path (experiments); default auto
uses the evidence-tuned rule in _pick_engine().
"""
import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

_LIBS = {}


def _find_nvcc():
    for cand in ("/usr/local/cuda/bin/nvcc", shutil.which("nvcc")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("nvcc not found; cannot build the mla_expanded_bf16 kernels")


def _build(name, src_name, arch_flags):
    """Compile one kernel .cu into a cached .so (flock-serialized, atomic)."""
    if name in _LIBS:
        return _LIBS[name]
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    nvcc = _find_nvcc()
    src = HERE / src_name
    src_bytes = src.read_bytes()
    key = hashlib.sha256(src_bytes + arch.encode() + nvcc.encode()
                         + " ".join(arch_flags).encode()).hexdigest()[:16]
    build_dir = HERE / f".build_{name}_{arch}_{key}"
    so_path = build_dir / "kernel.so"
    if not so_path.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        with open(build_dir / "build.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if not so_path.exists():
                    tmp_path = build_dir / f"kernel.{os.getpid()}.tmp.so"
                    cmd = [nvcc, "-O3", "-std=c++17", "-shared",
                           "-Xcompiler", "-fPIC", *arch_flags,
                           "--ptxas-options=-v", "-DNDEBUG",
                           str(src), "-o", str(tmp_path)]
                    env = dict(os.environ)
                    env.setdefault("HOME", str(build_dir))
                    env.setdefault("TMPDIR", str(build_dir))
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=270, env=env, cwd=str(build_dir))
                    (build_dir / "build.log").write_text(
                        f"cmd: {cmd}\nexit: {proc.returncode}\n"
                        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
                    if proc.returncode != 0:
                        tmp_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"nvcc build failed for {name} (see build.log):\n"
                            + (proc.stdout + proc.stderr)[-3000:])
                    tmp_path.replace(so_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
    lib = ctypes.CDLL(str(so_path))
    if name in ("sync", "td"):
        fn = lib.mla_forward
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
        ]
    else:
        # v4.1 tc ABI: q,k,v,out,lse,sinks,cum_q,cum_kv + kws,vws,
        # total_q,total_kv,batch,heads,kv_heads,kvblks,
        # window,causal,has_lse,has_sink, bmm1,bmm2, max_kv, stream, dbg, mode,
        # pf_mode, lag_max
        fn = lib.mla_tc_forward
        fn.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int] * 10
                       + [ctypes.c_float, ctypes.c_float, ctypes.c_int]
                       + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 6)
        ws = lib.mla_tc_ws
        ws.argtypes = ([ctypes.c_int] * 3
                       + [ctypes.POINTER(ctypes.c_longlong)] * 3)
        ws.restype = None
    fn.restype = ctypes.c_int
    _LIBS[name] = (lib, fn)
    return _LIBS[name]


def _build_all():
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    tca = [f"-gencode=arch=compute_{major}{minor}a,code={arch}a", "-lcuda"]
    # Forced-engine runs (experiments) compile ONLY the forced engine -
    # nvcc time dominates slim job budgets on the 2-core pods. Auto mode
    # builds the dispatch closure: tc2 (preferred), tc (fallback if the tc2
    # build fails), sync (tiny-shape regime + last-resort fallback).
    forced = _FORCE if _FORCE in ("sync", "tc", "tc2", "tc2t", "tc2d",
                                   "tc2dt", "tc3", "tc4", "tc4t", "td",
                                   "tc2p", "tc2pt", "tc2s", "tc2st",
                                   "tc2c", "tc2ct", "tc7", "tc7e",
                                   "tc7nr", "tc7enr", "tc7enp",
                                   "tc7eu", "tc7ep", "tc7f", "tc7fm",
                                   "tc7eux", "tc7eus", "tc7l", "tc9", "tc9r",
                                   "tc9u", "tc9g", "tc9s", "tc9b", "tc9pm", "tc9t", "tc9q", "tc9qp", "tc9t4", "tc9e", "tc9et4", "tc9t5", "tc9l", "tc9lt5",
                                   "tc2pb", "tc2b", "tc2pe", "tc2pet",
                                   "tc2pf", "tc2pft", "tc2e", "tc2et",
                                   "tc2x", "tc2px", "tc2p14",
                                   "tc2p64", "tc2pm", "tc2pmt", "tc2em",
                                   "tc2p64x", "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5",
                                   "tc2emt", "tc2l2q", "tc2l2n") else None
    if forced in ("sync", "td"):
        _build(forced, "kernel.cu" if forced == "sync" else "kernel_td.cu",
               ["-arch=" + arch])
        return
    if forced is not None:
        srcname = {"tc": "kernel_tc.cu", "tc2": "kernel_tc2.cu",
                   "tc2t": "kernel_tc2.cu", "tc2d": "kernel_tc2.cu",
                   "tc2dt": "kernel_tc2.cu", "tc3": "kernel_tc3.cu",
                   "tc4": "kernel_tc4.cu", "tc4t": "kernel_tc4.cu",
                   "tc2p": "kernel_tc2p.cu", "tc2pt": "kernel_tc2p.cu",
                   "tc2s": "kernel_tc2s.cu", "tc2st": "kernel_tc2s.cu",
                   "tc2c": "kernel_tc2c.cu", "tc2ct": "kernel_tc2c.cu",
                   "tc7": "kernel_tc7.cu", "tc7e": "kernel_tc7.cu",
                   "tc7nr": "kernel_tc7.cu", "tc7enr": "kernel_tc7.cu",
                   "tc7enp": "kernel_tc7.cu",
                   "tc7eu": "kernel_tc7.cu", "tc7ep": "kernel_tc7.cu",
                   "tc7f": "kernel_tc7.cu", "tc7fm": "kernel_tc7.cu",
                   "tc7eux": "kernel_tc7.cu",
                   "tc7eus": "kernel_tc7.cu",
                   "tc7l": "kernel_tc7.cu",
                   "tc9": "kernel_tc9.cu", "tc9r": "kernel_tc9.cu",
                   "tc9pm": "kernel_tc9.cu",
                   "tc9t": "kernel_tc9.cu",
                   "tc9q": "kernel_tc9.cu",
                   "tc9qp": "kernel_tc9.cu",
                   "tc9t4": "kernel_tc9.cu",
                   "tc9e": "kernel_tc9.cu",
                   "tc9et4": "kernel_tc9.cu",
                   "tc9t5": "kernel_tc9.cu",
                   "tc9l": "kernel_tc9.cu",
                   "tc9lt5": "kernel_tc9.cu",
                   # tc9u/tc9g/tc9s/tc9b: join-20261024 arm-b attribution set.
                   # tc9u = tc9r's PRE-adoption flags (neither -DTC9_GSORT nor
                   # -DTC9_SERP: the unbalanced static stride the join start
                   # shipped); tc9g adds GSORT alone, tc9s SERP alone, tc9b
                   # both (= the shipped tc9r). Force-only, never dispatched.
                   # They exist so the adoption prices against its own
                   # predecessor in ONE process with rotated rounds - the only
                   # measurement trusted below ~2% (arm-b: a child-per-engine
                   # A/B carries a per-process systematic of that size,
                   # dv_2565cd9dd9c1). Neither flag can change an output
                   # element: they decide WHICH cluster computes an item, not
                   # what the item is - pure functions of (blockIdx.x,
                   # gridDim.x) plus immutable cum_q/cum_kv, no counter,
                   # handshake or barrier in the item sequence. Arm-b measured
                   # the shipped combo in-process (dv_3ac96f20d630): f3 -7.2%,
                   # f8 -4.6%, x4096 -9.5%, xq156 -14.9%, xr16b2q1024 -11.0%,
                   # xr12b2q2048 -8.1%, bit-identical (cross-engine image
                   # max=0.00000 on 11 shapes, 44 rows bad=0), 0 spills at
                   # 157 registers.
                   "tc9u": "kernel_tc9.cu", "tc9g": "kernel_tc9.cu",
                   "tc9s": "kernel_tc9.cu", "tc9b": "kernel_tc9.cu",
                   "tc2pb": "kernel_tc2p_v1.cu", "tc2b": "kernel_tc2_v1.cu",
                   "tc2pe": "kernel_tc2p.cu", "tc2pet": "kernel_tc2p.cu",
                   "tc2pf": "kernel_tc2p.cu",
                   "tc2pft": "kernel_tc2p.cu",
                   "tc2e": "kernel_tc2.cu", "tc2et": "kernel_tc2.cu",
                   "tc2x": "kernel_tc2.cu", "tc2px": "kernel_tc2p.cu",
                   "tc2p14": "kernel_tc2p_v14.cu",
                   "tc2p64": "kernel_tc2p.cu",
                   "tc2p64x": "kernel_tc2p.cu",
                   "tc2p64t2": "kernel_tc2p.cu",
                   "tc2p64t3": "kernel_tc2p.cu",
                   "tc2p64lp": "kernel_tc2p.cu",
                   "tc2p64t3lp": "kernel_tc2p.cu",
                   "tc2p64t4": "kernel_tc2p.cu",
                   "tc2p64t5": "kernel_tc2p.cu",
                   "tc2pm": "kernel_tc2p.cu", "tc2pmt": "kernel_tc2p.cu",
                   "tc2em": "kernel_tc2.cu", "tc2emt": "kernel_tc2.cu",
                   "tc2l2q": "kernel_tc2.cu", "tc2l2n": "kernel_tc2.cu"}[forced]
        # Production "tc2" = kernel_tc2.cu with -DTC2_DUAL3 (352 threads;
        # warp10 = subtile-1 MMA issuer under opts bit64, warp9 pure TMA
        # loader; the 352-thread build measured neutral with bit64 off,
        # job24 d_o6). tc2t/tc2dt add -DTC2_TIMING (stage clock64
        # accounting; production builds have zero timing code - job21
        # measured 21% overhead from the runtime-gated version).
        flagmap = {"tc2t": ["-DTC2_TIMING"],
                   "tc2pt": ["-DTC2_TIMING"],
                   "tc2st": ["-DTC2_TIMING"],
                   "tc2ct": ["-DTC2_TIMING"],
                   "tc7": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                           "-DTC7_RELAX_ARRIVE"],
                   "tc7e": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16"],
                   "tc7nr": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY"],
                   "tc7enr": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_EX2F16"],
                   "tc7enp": ["-DTC7_UNBOUNDED", "-DTC7_RELAX_ARRIVE",
                              "-DTC7_EX2F16"],
                   "tc7eu": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK"],
                   "tc7ep": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_PREFIX"],
                   "tc7f": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC7_MASKMAX",
                            "-DTC7_EX2F16"],
                   "tc7fm": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC7_MASKMAX"],
                   "tc7eux": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK",
                              "-DTC7_XMAX1"],
                   "tc7eus": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK",
                              "-DTC7_SPIN"],
                   "tc7l": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK",
                            "-DTC7_LPTG"],
                   "tc9": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                           "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16"],
                   # -DTC9_MSKIP adoption (E25, run 20261027T002529Z-023749):
                   # skip fully-hidden 32-col blocks in the masked softmax
                   # (bit-exact by the -200 exact-underflow law; pc/fmaxf
                   # identities). Same-job interleaved: f7 0.9624 (10/10),
                   # f8 0.9526 (10/10), f3 0.9591 (10/10), f9 0.9990 flat;
                   # family geomean 0.9681; bit-equal out+lse on all shapes,
                   # zero tolerance/reference mismatch. Official: fixed
                   # -2.30% (certified f58ede7).
                   # tc2p64-twin history CORRECTED (E28/E29): E25's
                   # 1.05-1.43 "codegen regression" was the opts confound
                   # (unregistered probe names -> op=0 plain order), NOT a
                   # register cliff; the valid retest (E29, dv_1d51f66584c7)
                   # measured FLAT (core geos 0.9969-0.9977) - tc2p64 is
                   # feed-bound, stage-share != span-share. TC2P_MSKIP*
                   # stays inert; re-entry only if tc2p64 becomes
                   # softmax-paced.
                   # -DTC9_EARLY_PST adoption (E33, dv_1738fb8fb689): store
                   # each P half inside the exp loop instead of deferring
                   # both TMEM P stores past it (32 packed registers no
                   # longer live across the second half; final wait_st no
                   # longer pays for both). E4a rejected it PRE-MSKIP as
                   # family-mixed (f8 0.9792 but f7 1.0115); POST-MSKIP the
                   # combination is superadditive - wins ~one cluster
                   # cadence quantum (~1.05us/2115cyc) of span on ALL FIVE
                   # tc9r cases: f8 0.9756 (8/10 mode-pinned), f3 0.9804
                   # (7/10), f7 0.9867 (10/10), f9 0.9765, f4 0.9795 (6/6);
                   # family geomean 0.9797 median-of-rounds; bit-equal
                   # out+lse all 5, zero tolerance/reference mismatch.
                   # XMAX1 adds nothing on top (EPX ~= EP) and stays out.
                   # Evidence + statistic correction:
                   # results/e33_early_pst_decision.md.
                   # join-20261027 phase ADOPTION (E46): + TC9_PFL per-warp
                   # local PF/PL arrives (counts 9/8) - removes the per-tile
                   # 8-warp bar_softmax convergence from the paced path.
                   # Evidence: probe202 dv_5923287e45e9 (slot-41 barconv
                   # 82-168 cyc/tile) -> probe203 dv_9101631e8f6d (BE 8/8,
                   # refs, det; pf22 drops exactly the barconv; exp
                   # invariant; f3 0.9616 21/30, f7 0.9757, no regressions;
                   # 2/6 bars -> iterate branch) -> probe204
                   # dv_60cf513f49e2 45-round confirm (f3 0.9626 25/45
                   # replicated; f4 0.9896 43/45; f7 0.9971 23/45 = missed
                   # the win bar calibrated on probe203's mode draw,
                   # passes the non-regression bar applied to f2/f4/f8/f9;
                   # ambiguity resolution documented in the evidence log;
                   # official fixed compare #2 is the final judge).
                   "tc9r": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                            "-DTC9_PFL"],
                   # join-20261027 phase, probe187: tc9r + the DORMANT
                   # TC9_PREMETA port (in arm-b's tree since their E5-era
                   # port of the adopted tc2p mechanism - cum_q/cum_kv smem
                   # mirrors, warp-9 entry-issued cold reads overlapping the
                   # init/fence/alloc chain, CTA-wide publication before the
                   # init barrier, batch>63 fallback to the global arrays,
                   # flag-off reduction by const-alias construction). Never
                   # wired or priced by arm-b (absent from their flags and
                   # name lists); all five decode_item call sites and
                   # build_perm consume the cq/ck aliases, so the coverage
                   # is complete. tc2p-family track record: d3 -2.1%,
                   # d5 -1.6%, f1 -1.1% (E5). Probe-only forced name;
                   # _pick_engine never returns it.
                   "tc9pm": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                            "-DTC9_PREMETA"],
                   # probe194 census build: production tc9r flags + the
                   # E34 slot decomposition (TC9_TIMING + TC9_TIMING2);
                   # never in production builds, probe-only forced name.
                   "tc9t": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                            "-DTC9_TIMING", "-DTC9_TIMING2"],
                   # E42 (probe197): TC9_QPF = loader-issued fire-and-forget
                   # TMA L2 prefetch of the NEXT item's Q box (decode_item is
                   # pure; prefetch writes nothing). Attacks the slot-32
                   # refill bubble (f9 4155 cyc/item = 34% chain, f8 2800,
                   # f3 2303; probe194). tc9qp adds the census slots.
                   # Probe-only forced names; _pick_engine never returns them.
                   "tc9q": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_QPF"],
                   "tc9qp": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                             "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                             "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                             "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_QPF",
                             "-DTC9_TIMING", "-DTC9_TIMING2"],
                   # E43 (probe198): slot-19 exp substage decomposition
                   # (TC9_TIMING4 slots 36-38; tc2p mirror TC2_TIMING4
                   # slots 55-57). Probe-only census names.
                   "tc9t4": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                             "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                             "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                             "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                             "-DTC9_TIMING", "-DTC9_TIMING2",
                             "-DTC9_TIMING4"],
                   # E44 (probe199): TC9_PR16 = tc2p-exact per-cc packed
                   # window (pr[16] scoped inside the cc loop; registers
                   # die at the store) - bit-identical reshape attacking
                   # probe198's R1 loop-body residual (+105..+235 cyc/tile
                   # vs tc2p on identical element shares). tc9e =
                   # production+PR16; tc9et4 adds the census slots.
                   "tc9e": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_PR16"],
                   "tc9et4": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                              "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                              "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                              "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_PR16",
                              "-DTC9_TIMING", "-DTC9_TIMING2",
                              "-DTC9_TIMING4"],
                   # E45 (probe202): PF-path split (41 barconv / 42 arrive)
                   "tc9t5": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                             "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                             "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                             "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                             "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                             "-DTC9_TIMING", "-DTC9_TIMING2",
                             "-DTC9_TIMING5"],
                   # E46 (probe203): TC9_PFL = per-warp local PF/PL arrives
                   # (counts 9/8), removing the per-tile 8-warp bar_softmax
                   # convergence (probe202 slot 41: 82-168 cyc/tile).
                   # Protocol-only; relay + issuer parity unchanged.
                   "tc9l": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                            "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                            "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_PFL"],
                   "tc9lt5": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                              "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                              "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                              "-DTC9_MSKIP", "-DTC9_EARLY_PST", "-DTC9_PFL",
                              "-DTC9_TIMING", "-DTC9_TIMING2",
                              "-DTC9_TIMING5"],
                   "tc9g": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT"],
                   "tc9s": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_SERP"],
                   "tc9b": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP"],
                   "tc9u": ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                            "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                            "-DTC9_RANKMAJOR"],
                   "tc2d": ["-DTC2_DUAL3"],
                   "tc2b": ["-DTC2_DUAL3"],
                   "tc2dt": ["-DTC2_DUAL3", "-DTC2_TIMING"],
                   "tc2": ["-DTC2_DUAL3"],
                   "tc2pe": ["-DTC2_EX2BF16"],
                   "tc2pet": ["-DTC2_EX2BF16", "-DTC2_TIMING"],
                   "tc2pf": ["-DTC2_EX2F16"],
                   "tc2pft": ["-DTC2_EX2F16", "-DTC2_TIMING"],
                   "tc2e": ["-DTC2_DUAL3", "-DTC2_EX2F16"],
                   "tc2et": ["-DTC2_DUAL3", "-DTC2_EX2F16", "-DTC2_TIMING"],
                   "tc2x": ["-DTC2_DUAL3"],
                   "tc2px": ["-DTC2_EX2F16"],
                   "tc2p14": ["-DTC2_EX2F16"],
                   # tc2p64 = PRODUCTION since dv_9510050b757c: the batch==1 LPTG
                   # identity shortcut is enumerated in. At batch==1 with opts bit1 the
                   # loader's counting sort is provably the identity (491 domain shapes,
                   # 126 batch-1, 0 divergences; 44 of 126 fail with bit1 off, so the
                   # guard's bit1 test is necessary), letting it skip the 257-entry
                   # histogram zero, the tile_T walk, the serially dependent descending
                   # prefix and the scatter. Same-job interleaved A/B, 7 reps: f0
                   # +2.048us 7/7, f2 +4.096 7/7, f4 +4.128 7/7; the three batch>1
                   # controls f1/f6/f7 tie at delta 0.000, so the gain needs no build-cost
                   # correction. Implied fixed geomean +1.21%. Cross-engine bit-identical
                   # on 51 shapes (bad_out=0, max=0.00000), 152 regs both arms.
                   # PREMETA adoption (E5 rerun dv_9b43a2383c28): see the
                   # _build_all comment below for the numbers.
                   # -DTC2P_QSPLIT adoption (E38b dv_7350ce285b05 + E39
                   # dv_12cf51cf8ca4): per-subtile half-box Q arrival at
                   # item boundaries (QF2/QF3 mirror the QF0/QF1 parity
                   # scheme; each issuer warp's MMA reads only its own
                   # 49KB half). E38a census: the 98KB cold Q TMA arrival
                   # wait = 81% of f0's / 61% of f1's per-item chain (9-33%
                   # of span). Wins replicate at high item density: f1
                   # 0.9779 + 0.9828 across two jobs (12/12 rounds);
                   # flat no-regress f0/f2/f6/d5 (density law: low
                   # items/CTA hide the wait in prologue/epilogue slack);
                   # bit-equal everywhere; 80/80 deterministic peaked
                   # stress launches (f6p+f1p, stacked-flag worst case).
                   # QEARLY (early half-0 refill) measured FLAT beyond
                   # QSPLIT and stays inert - half-1's free point IS the
                   # item end, the family's limit. ITEMS_OFF is flag-gated
                   # 232 with static_assert(NBARS==28); the plain tc2p
                   # (box8) build stays UNTOUCHED (validated on box64
                   # only). results/e38b_qsplit_decision.md,
                   # results/e39_qearly_decision.md.
                   "tc2p64": ["-DTC2_EX2F16", "-DTC2P_BOX64", "-DTC2P_TID",
                              "-DTC2P_PREMETA", "-DTC2P_QSPLIT"],
                   # probe194 census build: production tc2p64 flags + the
                   # E37/E38a slot decomposition; probe-only forced name.
                   "tc2p64t2": ["-DTC2_EX2F16", "-DTC2P_BOX64", "-DTC2P_TID",
                                "-DTC2P_PREMETA", "-DTC2P_QSPLIT",
                                "-DTC2_TIMING", "-DTC2_TIMING2"],
                   # probe195 prologue-timeline build: + TC2_TIMING3
                   # (E40 slots 38-46, worst-CTA entry-relative marks).
                   "tc2p64t3": ["-DTC2_EX2F16", "-DTC2P_BOX64", "-DTC2P_TID",
                                "-DTC2P_PREMETA", "-DTC2P_QSPLIT",
                                "-DTC2_TIMING", "-DTC2_TIMING2",
                                "-DTC2_TIMING3"],
                   # E41 (probe196): warp-parallel LPTG table build
                   # (bit-identical enumeration; slot-47 ord checksum in
                   # timing builds proves order identity). Probe-only.
                   "tc2p64lp": ["-DTC2_EX2F16", "-DTC2P_BOX64", "-DTC2P_TID",
                                "-DTC2P_PREMETA", "-DTC2P_QSPLIT",
                                "-DTC2P_LPTGP"],
                   "tc2p64t3lp": ["-DTC2_EX2F16", "-DTC2P_BOX64",
                                  "-DTC2P_TID", "-DTC2P_PREMETA",
                                  "-DTC2P_QSPLIT", "-DTC2_TIMING",
                                  "-DTC2_TIMING2", "-DTC2_TIMING3",
                                  "-DTC2P_LPTGP"],
                   "tc2p64t4": ["-DTC2_EX2F16", "-DTC2P_BOX64",
                                "-DTC2P_TID", "-DTC2P_PREMETA",
                                "-DTC2P_QSPLIT", "-DTC2_TIMING",
                                "-DTC2_TIMING2", "-DTC2_TIMING4"],
                   # E45 (probe202): loop-top SF wait as slot 58
                   "tc2p64t5": ["-DTC2_EX2F16", "-DTC2P_BOX64",
                                "-DTC2P_TID", "-DTC2P_PREMETA",
                                "-DTC2P_QSPLIT", "-DTC2_TIMING",
                                "-DTC2_TIMING2", "-DTC2_TIMING5"],
                   # tc2p64x = the PRE-TID parent, i.e. what tc2p64 was at the join start.
                   # Kept for two reasons: it is the attribution control if this shortcut
                   # ever needs re-pricing in a later tree, and it is the engine
                   # check_production_unchanged reduces to prove the guarded scaffolding
                   # still leaves the pre-TID path byte-identical to
                   # baselines/join20261025_kernel_tc2p.cu.
                   "tc2p64x": ["-DTC2_EX2F16", "-DTC2P_BOX64"],
                   "tc2pm": ["-DTC2_EX2F16", "-DTC2_MAXTREE"],
                   "tc2pmt": ["-DTC2_EX2F16", "-DTC2_MAXTREE",
                              "-DTC2_TIMING"],
                   "tc2em": ["-DTC2_DUAL3", "-DTC2_EX2F16", "-DTC2_MAXTREE"],
                   "tc2emt": ["-DTC2_DUAL3", "-DTC2_EX2F16", "-DTC2_MAXTREE",
                              "-DTC2_TIMING"],
                   "tc2l2q": ["-DTC2_DUAL3", "-DTC2_EX2F16", "-DTC2_PROMO256"],
                   "tc2l2n": ["-DTC2_DUAL3", "-DTC2_EX2F16",
                              "-DTC2_PROMO_NONE"]}
        flags = tca + flagmap.get(forced, [])
        _build(forced, srcname, flags)  # loud failure if a forced engine dies
        return
    try:
        # -DTC2_EX2F16 (v14): softmax exp phase via ex2.approx.f16x2
        # (halves MUFU ops; probe27/28 same-job: f1 -4..-7%, f0 -9.1/-9.6%,
        # d3 -5..-8%, f9 -9.3%, d7 -13.0%; tensor-bound shapes neutral;
        # numerics 16/16 clean incl. official failing seeds; sm_exp
        # 1236->736 tc2p, 2192->1367 tc2).
        _build("tc2", "kernel_tc2.cu",
               tca + ["-DTC2_DUAL3", "-DTC2_EX2F16"])
    except Exception as exc:
        import sys
        print(f"WARNING: tc2 kernel build failed: {exc}", file=sys.stderr)
        _LIBS["tc2"] = None
    if _LIBS.get("tc2") is not None:
        # Persistent-CTA tc2 (probe16 winner on waves>=2 shapes). Optional:
        # any build failure keeps the classic tc2 dispatch intact.
        try:
            # -DTC2P_PREMETA (E5 rerun dv_9b43a2383c28, adopt verdict):
            # cold-chain prologue trim - warp-9 lanes issue the cum_q/cum_kv
            # reads at kernel entry (overlapping init/fences/TMEM alloc),
            # mirror them into ctrl smem for the TID check/table walk/decode,
            # and tid0 prefetches the counter line. Interleaved same-process
            # A/B vs production tc2p64, official-proxy cold protocol: median
            # 0.9893 over {f0,f1,f2,f6,d3,d5}, worst shape 0.9995, 46/60
            # rounds, bit-equality (out+lse) on every shape and fp32-
            # reference zero-mismatch on f0/d3/d5 all arms. Per shape:
            # d3 0.9790 (10/10, batch-32 table walk), d5 0.9835, f6 0.9888,
            # f2 0.9897 (mode-suspect), f1 0.9954, f0 0.9995. First E5 run
            # (dv_a2bc04e06461) caught the 33-vs-34-entry mirror bug on d3
            # via the bit gate; fixed before this verdict (af41c2a record).
            # tc2p (box8) shares the loader, so it carries the same flag;
            # its d4-class shapes are covered by the domain A/B.
            _build("tc2p", "kernel_tc2p.cu",
                   tca + ["-DTC2_EX2F16", "-DTC2P_PREMETA"])
        except Exception as exc:
            import sys
            print(f"WARNING: tc2p kernel build failed; multi-wave shapes "
                  f"stay on classic tc2: {exc}", file=sys.stderr)
            _LIBS["tc2p"] = None
        if _LIBS.get("tc2p") is not None:
            try:
                # -DTC2P_TID enumerated into production on the strength of
                # dv_9510050b757c (see the flagmap entry above for the numbers).
                # Must stay flag-for-flag equal to flagmap["tc2p64"], otherwise the
                # forced-engine isolate stops being the same build as production and
                # every A/B priced against it silently changes meaning.
                _build("tc2p64", "kernel_tc2p.cu",
                       tca + ["-DTC2_EX2F16", "-DTC2P_BOX64", "-DTC2P_TID",
                              "-DTC2P_PREMETA", "-DTC2P_QSPLIT"])
            except Exception as exc:
                import sys
                print(f"WARNING: tc2p64 kernel build failed; small-kv "
                      f"multi-wave shapes stay on box8 tc2p: {exc}",
                      file=sys.stderr)
                _LIBS["tc2p64"] = None
        # tc7: 2-CTA cluster BN128 kernel (arm-b join import, JOIN
        # 20261022). Promoted arm-b build config. Optional: any failure
        # keeps the full tc2/tc2p dispatch closure intact.
        # v18: production cluster build = tc7eu (E2 in the UNMASKED exp
        # loop only; probe48/49: full E2 zeroed early-causal-token rows -
        # tc7's hardware redmax covers masked columns -> f16 exp underflow
        # -> dtot=0; unmasked-only clean 6/6 and keeps 90%+ of the E2
        # gain: f5 0.956, f2 0.972, f9 0.978 vs plain tc7).
        try:
            _build("tc7eu", "kernel_tc7.cu",
                   tca + ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                          "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK"])
        except Exception as exc:
            import sys
            print(f"WARNING: tc7eu cluster kernel build failed; tensor-bound "
                  f"shapes stay on tc2/tc2p: {exc}", file=sys.stderr)
            _LIBS["tc7eu"] = None
        # v21: tc7l = tc7eu + TC7_LPTG (static global descending-T dispatch
        # order; probe57 dv_a4574d929f00: bit-equal 8/8 vs tc7eu, cold
        # f9 0.891, f8 1.005 null). Only used by the shallow-wave envelope
        # gate below; any build failure falls back to tc7eu.
        if _LIBS.get("tc7eu") is not None:
            try:
                _build("tc7l", "kernel_tc7.cu",
                       tca + ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC7_EX2F16_UNMASK",
                              "-DTC7_LPTG"])
            except Exception as exc:
                import sys
                print(f"WARNING: tc7l cluster kernel build failed; "
                      f"shallow-wave envelope shapes stay on tc7eu: {exc}",
                      file=sys.stderr)
                _LIBS["tc7l"] = None
        # v23: tc9r = arm-b join-20261023 persistent-cluster engine
        # (kernel_tc9.cu). probe73 (dv_7d90ed7ff7cb): numerics 12/12 on our
        # E2-law suite; cold f8 0.9562 vs tc2p+LPTG; the v23 gate was the
        # (148,176] window. join 20261024: kernel_tc9.cu re-adopted WHOLESALE
        # from arm-b (launcher tail byte-identical, host ABI unchanged) with
        # TC9_GSORT + TC9_SERP rebalancing the STATIC item assignment, and the
        # window generalised to _quant_prefers_cluster + the shallow branch.
        # The plain stride cost 1.281x the mean load over the ten fixed cases
        # and 1.713x on f9 (arm-b experiments/item_balance.py reproduces both
        # from the kernel constants) against 1.089x for a dynamic counter this
        # kernel deliberately does not have (a counter needs a leader-fetch/
        # peer-publish handshake per item). Serpentine zigzags the stride so
        # pass k+1 fills the clusters that drew the LIGHTEST item in pass k;
        # gsort orders pair ids by globally descending T, which is what makes
        # the zigzag safe when batch > 1 (the list is `batch` CONCATENATED
        # ramps; zigzag alone costs f6 1.310->1.438, f8 1.358->1.697 in sim).
        # Both self-gate inside the kernel on the cases where the sort can
        # differ from the identity (serp: causal; gsort: causal && batch>1),
        # because building the table costs ~1.5us of cold cum_q/cum_kv latency
        # (arm-b dv_3ac96f20d630). Optional: build failure keeps the join-start
        # gates exactly (tc7l/tc7eu/tc2p).
        # C5 (probe116 dv_cd646eb8dd23, results/tc9_kvsplit_design.md):
        # TC9_KVSPLIT splits the KV ring's stage release - K freed at
        # QK(j-3) COMMITMENT (SF0, the guarantee the V/PV0 release already
        # rides), V kept at PV retirement - so the K TMA and its peer-relay
        # leg run ~1 tile ahead with NO new SMEM (row 14's wall is depth,
        # not triggers). Same-job interleaved A/B: f8 0.9571, f3 0.9646,
        # x1024 0.9171, S3r 0.9990, f9 1.0048, d9 quantum tie; BE 12/12
        # bit-identical to the certified build, bounded canary 0 timeouts
        # (incl. d9 49932 tiles), peaked stress 0/360, 156 regs 0 spills.
        # Timing-twin attribution did NOT reproduce cross-job (baseline bf
        # halved in the new environment draw); adopted on the pre-registered
        # e2e verdict with the gap on record. Controls tc9/tc9u/g/s/b keep
        # their exact pre-KVSPLIT flags (frozen attribution set).
        # C6 (probe119 dv_dc0784afc054): TC9_DRAINLATE defers the epilogue's
        # tid0 bulk_wait_read0 (32KB staging read drain) from right-after the
        # TMA store commit to the NEXT epilogue's 256-thread rendezvous + a
        # final drain after the item loop - removing it from every non-final
        # item boundary's warp-serial path. Mechanism confirmed in-job:
        # epi_us -0.83/-1.22/-1.62us (f9/f8/f3, timing twins), span_avg
        # -0.58/-0.97/-1.45, e2e f3 0.9592 (f9/f8/S3r/x1024 ties: transfer
        # is span_max-criticality dependent); BE 12/12 bit-identical,
        # stress 0/300 peaked, regs unchanged. MID verdict -> nominated per
        # the pre-registered branch (mech confirmed + gates clean).
        if _LIBS.get("tc7eu") is not None:
            try:
                _build("tc9r", "kernel_tc9.cu",
                       tca + ["-DTC7_UNBOUNDED", "-DTC7_PF_RELAY",
                              "-DTC7_RELAX_ARRIVE", "-DTC8_EX2F16",
                              "-DTC9_RANKMAJOR", "-DTC9_GSORT", "-DTC9_SERP",
                              "-DTC9_KVSPLIT", "-DTC9_DRAINLATE",
                              "-DTC9_MSKIP", "-DTC9_EARLY_PST",
                              "-DTC9_PFL"])
            except Exception as exc:
                import sys
                print(f"WARNING: tc9r persistent-cluster build failed; "
                      f"quantisation-favourable shapes stay on tc2/tc2p: {exc}",
                      file=sys.stderr)
                _LIBS["tc9r"] = None
    if _LIBS.get("tc2") is None:
        try:
            _build("tc", "kernel_tc.cu", tca)
        except Exception as exc:
            import sys
            print(f"WARNING: tcgen05 kernel build failed; falling back to "
                  f"mma.sync engine: {exc}", file=sys.stderr)
            _LIBS["tc"] = None
    _build("sync", "kernel.cu", ["-arch=" + arch])
    try:
        _build("td", "kernel_td.cu", ["-arch=" + arch])
    except Exception as exc:
        import sys
        print(f"WARNING: td kernel build failed; decode-class shapes fall "
              f"back to the general engines: {exc}", file=sys.stderr)
        _LIBS["td"] = None


# Cached device index (the harness never migrates devices mid-run; the
# exception fallback stays correct otherwise). Saves a
# torch.cuda.current_device() (~0.2-0.4us of lazy-init/dispatch machinery)
# from EVERY memoized call - fully exposed CPU time under the official
# eager protocol.
_DEV_IDX = [None]
# Prebuilt c_void_p per raw stream pointer (streams are stable across
# calls; bounded dict, miss path constructs fresh).
_SP_CACHE = {}


def _current_stream_ptr():
    try:
        d = _DEV_IDX[0]
        if d is None:
            d = torch.cuda.current_device()
            _DEV_IDX[0] = d
        return torch._C._cuda_getCurrentRawStream(d)
    except Exception:
        return torch.cuda.current_stream().cuda_stream


# Grow-only scratch for the tc engine's padded canonical workspaces (candidate
# memory, never returned or observable; inputs stay read-only). One-time
# cudaMalloc per size class; steady-state calls reuse the cached tensor.
_WS = {"t": None}
_TC_CACHE = {}
_SYNC_CACHE = {}
_GRAPHS_OK = os.environ.get("MLA_NO_GRAPH", "").strip() not in ("1", "true")
# forward()-level fast path: id-keyed (pinned refs -> ids cannot be recycled)
# -> captured graph replay, skipping validation/dispatch/arg-tuple building
# (~2-4us/call of python in the official end-to-end timing window).
_FAST = {}
_LAST_GRAPH = [None]
_FAST_OK = os.environ.get("MLA_NOFAST", "").strip() not in ("1", "true")


def _align256(n):
    return (n + 255) & ~255


def _ensure_ws(nbytes):
    t = _WS["t"]
    if t is None or t.numel() < nbytes:
        t = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        _WS["t"] = t
    return t


def _capture_graph(fn, args, stream_idx):
    """Capture engine launch(es) into a CUDA graph (stream arg swapped for
    the capture stream).

    Legality: replay re-executes the full numerical work each call (no cached
    outputs); all pointers are part of the cache key, so a graph is only
    replayed for the exact same tensor set (the harness restores inputs
    in-place between calls). Capture cost is a one-off absorbed by warmup."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        a = list(args)
        a[stream_idx] = torch.cuda.current_stream().cuda_stream
        rc = fn(*a)
    if rc != 0:
        raise RuntimeError(f"capture launch rc {rc}")
    _LAST_GRAPH[0] = g
    return g


# tc_launch args layout (29 entries): 10 ptrs | 10 ints | bmm1,bmm2,max_kv |
# stream_ptr(23), dbg_ptr | dbg_mode,pf,lm,direct. Graph capture patches the
# stream slot by absolute index; the build-time assert below guards drift.
_TC_STREAM_IDX = 23

# ---------------------------------------------------------------------------
# v5 dispatch memo: single-slot identity check + driver-API graph launch.
#
# Why: the official eager protocol device-syncs (RNG restore) before each
# timed call, so the whole python dispatch path is inside the measured
# window. Measured per-call CPU (job dv_bf43cd5564ad): fk-tuple + dict +
# torch CUDAGraph.replay() = ~7.8us; a pre-built ctypes launch = ~5.3us.
# This memo does: 8 identity checks + 7 scalar checks (~0.7us) then
# cuGraphLaunch(exec, current_stream) via libcuda ctypes (~2-2.5us total),
# with the same "replay re-executes full numerics" legality argument as
# _capture_graph: every pointer/shape/scalar that enters the kernel args is
# either identity-checked (tensors are pinned so ids cannot recycle) or
# value-checked before the launch is reused.
_DRV = None           # dict of libcuda callables, or False if unavailable
_MEMO = []            # single slot: see _memo_set()
_GEXEC = {}           # fk -> exec handle / False, survives memo thrash
# MLA_NODRV: 1 = skip the driver-capture fast path (raw ctypes launch only);
# MLA_NOMEMO: 1 = disable the memo level entirely (fk/_FAST path only).
# Used to bisect dispatch-mechanics costs in same-job A/B probes.
_NODRV = os.environ.get("MLA_NODRV", "").strip() in ("1", "true")
_MEMO_OK = os.environ.get("MLA_NOMEMO", "").strip() not in ("1", "true")


def _drv():
    global _DRV
    if _DRV is not None:
        return _DRV or None
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        vp, ui, ull = ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong

        def bind(name, restype, *argtypes):
            f = getattr(lib, name)
            f.restype = restype
            f.argtypes = list(argtypes)
            return f

        d = {
            "create": bind("cuStreamCreate", ctypes.c_int, vp, ui),
            "begin": bind("cuStreamBeginCapture_v2", ctypes.c_int, vp, ctypes.c_int),
            "end": bind("cuStreamEndCapture", ctypes.c_int, vp, vp),
            "inst": bind("cuGraphInstantiateWithFlags", ctypes.c_int, vp, vp, ull),
            "launch": bind("cuGraphLaunch", ctypes.c_int, vp, vp),
            "destroy_graph": bind("cuGraphDestroy", ctypes.c_int, vp),
        }
        s = vp()
        if d["create"](ctypes.byref(s), 1) != 0:   # NON_BLOCKING side stream
            raise RuntimeError("cuStreamCreate failed")
        d["stream"] = s
        _DRV = d
    except Exception:
        _DRV = False
    return _DRV or None


def _drv_capture(fn, args, stream_idx):
    """Capture fn(*args) into a CUgraphExec via the driver API. Returns the
    exec handle (int) or None. Thread-local capture mode: no other thread's
    CUDA call can invalidate it; the window contains exactly one launch."""
    d = _drv()
    if d is None:
        return None
    cs = d["stream"]
    a = list(args)
    a[stream_idx] = cs.value
    g = ctypes.c_void_p()
    ex = ctypes.c_void_p()
    if d["begin"](cs, 1) != 0:
        return None
    rc = fn(*a)
    rc2 = d["end"](cs, ctypes.byref(g))
    if rc != 0 or rc2 != 0 or not g:
        if g:
            d["destroy_graph"](g)
        return None
    rc3 = d["inst"](ctypes.byref(ex), g, 0)
    d["destroy_graph"](g)
    if rc3 != 0 or not ex:
        return None
    return ex.value


# Last (fn, args, stream_idx) used by a slow-path launch; consumed by the
# memo registration at the end of forward(). Only set for launches that are
# memo-safe (direct/no-workspace engines).
_LAST_ARGS = [None]



def tc_launch(query, key, value, out, lse, sinks, cum_q, cum_kv, batch,
              window_left, causal, has_lse, has_sink, bmm1, bmm2, max_kv,
              stream_ptr, dbg_ptr=0, dbg_mode=0, pf_mode=None, lag_max=None,
              direct=None, engine="tc", box_tok=None, opts=None):
    """Launch the tc engine (v4.1 ABI). Single source of truth: used by
    Model.forward and by the experiment drivers (tc_dbg2/tc_timing).

    Fast paths (the official end-to-end protocol serializes per-call CPU
    time; tc was measured at ~+14us/call vs sync's ~+3, runs
    20261019T215111Z/215314Z):
      1. cached 26-arg ctypes tuple per (pointers, shapes, flags, stream);
      2. CUDA graph replay per key from the 2nd call on (~3us/call).
    Entries pin the workspace tensor identity: a grow-only realloc
    invalidates every entry (stale base pointers) and forces a rebuild."""
    global _GRAPHS_OK
    lib, fn = _LIBS[engine]
    pf = _PF_MODE if pf_mode is None else int(pf_mode)
    lm = _LAG_MAX if lag_max is None else int(lag_max)
    dr = 1 if engine in ("tc2", "tc2t", "tc2d", "tc2dt", "tc3", "tc4",
                         "tc4t", "tc2p", "tc2pt", "tc2s", "tc2st",
                         "tc2c", "tc2ct", "tc7", "tc7e",
                         "tc7nr", "tc7enr", "tc7enp",
                         "tc7eu", "tc7ep", "tc7f", "tc7fm",
                         "tc7eux", "tc7eus", "tc7l", "tc9", "tc9r",
                         "tc9u", "tc9g", "tc9s", "tc9b", "tc9pm", "tc9t", "tc9q", "tc9qp", "tc9t4", "tc9e", "tc9et4", "tc9t5", "tc9l", "tc9lt5",
                         "tc2pb", "tc2b", "tc2pe", "tc2pet", "tc2pf",
                         "tc2pft", "tc2e", "tc2et", "tc2x", "tc2px",
                         "tc2p14", "tc2p64", "tc2pm", "tc2pmt", "tc2em",
                         "tc2p64x", "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5",
                         "tc2emt", "tc2l2q", "tc2l2n") else (
        _TC_DIRECT if direct is None else int(direct))
    bx = _BOX_TOK if box_tok is None else int(box_tok)
    op = _OPTS if opts is None else int(opts)
    if engine == "tc3":
        bx = 8
        op = _T3OPT if opts is None else int(opts)
    elif engine in ("tc7", "tc7e", "tc7nr", "tc7enr", "tc7enp",
                    "tc7eu", "tc7ep", "tc7f", "tc7fm", "tc7eux",
                    "tc7eus", "tc7l", "tc9", "tc9r",
                    "tc9u", "tc9g", "tc9s", "tc9b", "tc9pm", "tc9t", "tc9q", "tc9qp", "tc9t4", "tc9e", "tc9et4", "tc9t5", "tc9l", "tc9lt5"):
        # tc7 (arm-b join import, JOIN 20261022): host encodes fixed box
        # shapes and ignores box_tok/opts entirely (arm-b promoted policy:
        # bx=64, op=0). Cluster kernel; adoption gated on probe44 same-job
        # re-measurement per campaign law.
        bx = 64
        op = 0
    elif engine in ("tc2c", "tc2ct"):
        # Cluster split-KV: tc2s-family opts layout (bit1 LPT order, bit3
        # whint, bit16 no-split diagnostic), but bit0 (OC fast-arrive)
        # stays OFF at every mk: split part1 windows are causal + Ts0<Ts1
        # asymmetric at T~16 - d7's exact bit0-race trigger geometry
        # (probe23/24/33, results/bit0_race_notes.md). Relax only after a
        # tc2c-specific stress gate.
        bx = 8
        if opts is None and _OPTS_ENV is None:
            op = 2
    elif engine in ("tc2p", "tc2pt", "tc2s", "tc2st", "tc2pb",
                    "tc2pe", "tc2pet", "tc2pf", "tc2pft", "tc2px",
                    "tc2p14", "tc2p64", "tc2pm", "tc2pmt", "tc2p64x", "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5",
                    # probe-only names (experiments/e4b_tc2p_flags.py builds
                    # flag variants of tc2p64 under distinct _LIBS keys);
                    # _pick_engine never returns them. Listed here so the
                    # variants get the SAME opts policy branch as production
                    # tc2p64 instead of the op=0 fall-through - without it
                    # an A/B silently compares LPTG-on vs LPTG-off.
                    "tc2p64_e4bM", "tc2p64_e4bR", "tc2p64_e4bRM",
                    "tc2p64_e5PM", "tc2p64_e5RU", "tc2p64_e5PMRU",
                    # E29 (tc2p MSKIP retest after the E25/E27 opts-confound
                    # retraction): same hazard, same guard. The generic
                    # fall-through forces op=0 even when opts is passed
                    # explicitly, so tuple registration is the ONLY way a
                    # probe name gets the production LPTG policy.
                    "tc2p64_e29e", "tc2p64_e29m", "tc2p64_e29em",
                    # E38b (QSPLIT half-box Q arrival A/B): same law.
                    # E39 (QEARLY early half-0 refill, stacked on QSPLIT).
                    "tc2p64_e38b", "tc2p64_e39q",
                    # join-20261027 census (probe194): production flags +
                    # TC2_TIMING+TIMING2 slot builds; same fall-through law.
                    "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5"):
        # Persistent/split tc2 family: always dual-issuer + TMA-Q + box8
        # (host ignores box_tok); effective bits: 2=LPT always.
        # v26: bit0 (OC fast-arrive) DISABLED AT ALL mk. The v13-era
        # containment ("tc2p's item-loop structure closes the window",
        # bit0_race_notes.md) is OVERTURNED by probe88 dv_5377313bc95e:
        # tc2p+bit0 nondeterministically corrupts O at 5-15% per launch
        # under the contract-legal PEAKED distribution at mk>=1024
        # (f1-shape: g24 6/40, g48 4/40, plain 6/40, bit20 1/40; lse
        # always clean; single-(tile,head) localization; det_repro
        # False). The old clean history was normal-distribution sampled
        # (value-gated: peaked's one-hot softmax amplifies a dropped/dup
        # O contribution to O(1); normal divides it under tolerance -
        # probe89 C2: 0/100 at N=100). bit0-OFF: 220/220 clean across
        # orders (probe89 C1). Price of disabling: 1.001x mean over 9
        # bit0-on cases = noise (probe89 C4) - the historical "~1%" does
        # not exist in the current build. Cluster engines clean under
        # peaked (tc7eu 0/20, tc9r 0/40, tc7l 0/40) - race confined to
        # the tc2p di-bit0 protocol. Addendum 21/22.
        # join-20261024 C3: archived-v26 hunk RE-APPLIED on the post-join
        # tree (start bytes carry the race; so does arm-b's tree).
        # probe100 (dv_db8159e5ac73) extends the race record to the BOX64
        # build: f1-geometry (g24+ragged, mk=2049) on tc2p64+bit0 peaked
        # 2/40 bad (lse clean, value-gated) while probe93's mk=1024
        # boundary shapes were 0/60+0/40 - box64 "immunity" was a geometry
        # artifact, not a mechanism difference. C2 routes every in-domain
        # causal mk>1024 shape onto tc2p64, so the kill covers box64 too.
        # probe100 also re-confirms the rebalanced tc9r clean under
        # peaked: f9p/f3p/S3rp 0/120 launches (build_perm barrier +
        # non-identity perm included).
        bx = 8
        if opts is None and _OPTS_ENV is None:
            op = 2
            # LPTG family (probe54/60): ws <= gate -> bit20 global order;
            # ws > gate AND causal -> bit21 group-major order with gsz =
            # largest divisor of heads under the gate (f1 0.976, fallback
            # f5/d6/d9 0.89-0.92 vs plain); non-causal stays plain (d4
            # uniform-T loses under any tile-major order).
            heads_q = int(query.shape[1])
            if heads_q * int(max_kv) * 640 <= _LPTG_WS_MAX:
                op |= 1 << 20
            elif causal and heads_q >= 2:
                g = _lptg_group(heads_q, max_kv)
                if 1 <= g < heads_q:
                    op |= (1 << 21) | ((g - 1) << 22)
    elif engine not in ("tc2", "tc2t", "tc2d", "tc2dt", "tc4", "tc4t",
                        "tc2b", "tc2e", "tc2et", "tc2x", "tc2em",
                        "tc2emt", "tc2l2q", "tc2l2n"):
        # tc1 loader: canonical 8-token boxes by default; box_tok=128 selects
        # the one-big-box-per-operand path (campaign tc1b experiment: n128 QK
        # MMAs are 23% more pipe-efficient, feed has ~3x slack).
        bx = _BOX_TOK if box_tok is None else int(box_tok)
        if bx not in (8, 128):
            bx = 8
        op = 0
    elif (engine in ("tc2", "tc2b", "tc2e", "tc2et", "tc2x", "tc2em",
                     "tc2emt", "tc2l2q", "tc2l2n")
          and opts is None and _OPTS_ENV is None):
        # Shape-derived policy. bit7 TMA-Q always (job24/25: f0 +4.5%,
        # d8 +9.7%, f1/f3 +2.3-2.7%, never worse).
        # bit6 dual-issuer ALWAYS (probe17 dv_44e5adcc5803 same-job:
        # d8 -8.0%, d1 -5.2%, d7 -2.2% vs merged; probe24 dv_...: op198
        # keeps most of it: d7 -3.6%, d8 -1.3%).
        # bit0 OC fast-arrive DISABLED AT ALL mk (v26). History: probe23/24
        # proved di+bit0 NONDETERMINISTICALLY CORRUPTS O on d7-class shapes
        # (mk<1024, Ts0<T asymmetric short items; 4/14 dirty incl. the v12
        # official domain failure dv_7b27405fa2b9) -> gated to mk>=1024 on
        # a normal-distribution-only clean history. probe88/89 then proved
        # the tc2p sibling races at ALL mk under peaked (addendum 21). For
        # classic tc2 the short-T boundary holds empirically (probe89 C3a:
        # op199 peaked 0/40 at mk=1024), but: officials have ZERO classic-
        # tc2-bit0 exposure (d1/d7/d8 are mk<1024 = op198; f9 moved to
        # tc9r on the join-20261024 shallow branch), the benefit is
        # unmeasured in the current build (tc2p's measured 1.001x), and
        # the mk>=1024 clean history is value-gated like tc2p's was.
        # Conservative kill; relaxation requires a classic-tc2-specific
        # peaked stress gate (tc2c precedent).
        op = op | 128 | 64
        # v16 box64 KV feed for classic tc2 (probe38 dv_09920a6a514b):
        # BIT-EXACT vs box8 on 5 shapes (torch.equal out+lse - identical
        # smem layout and MMA descriptors; only TMA request granularity
        # changes, 16 -> 2 requests/tile). Official-like cold-mode A/B:
        # f9 -1.45%, d1 -1.1%, d7 -0.06%, d8 +0.27%; warm-isolated f9
        # -1.26% (kernel-side feed/issue win, not cold-specific).
        if box_tok is None:
            bx = 64
    lp = lse.data_ptr() if lse is not None else 0
    sp = sinks.data_ptr() if sinks is not None else 0
    ck = (engine, query.data_ptr(), key.data_ptr(), value.data_ptr(),
          out.data_ptr(),
          lp, sp, cum_q.data_ptr(), cum_kv.data_ptr(),
          query.shape[0], key.shape[0], batch, query.shape[1], key.shape[1],
          window_left, causal, has_lse, has_sink, bmm1, bmm2, max_kv,
          stream_ptr, dbg_ptr, dbg_mode, pf, lm, dr, bx, op)
    hit = _TC_CACHE.get(ck)
    if hit is not None and hit[0] is _WS["t"]:
        if dr and dbg_mode == 0 and dbg_ptr == 0:
            _LAST_ARGS[0] = (fn, hit[1], _TC_STREAM_IDX)
        if hit[2] is not None:
            hit[2].replay()
            return 0
        if _GRAPHS_OK and dbg_mode == 0:
            try:
                # stream_ptr sits at args index _TC_STREAM_IDX (fixed by the
                # ABI layout below); a wrong index silently captures an EMPTY
                # graph (job dv_a37437c5d3be) - guarded by the assert at
                # args-build time and the harness no-write check.
                hit[2] = _capture_graph(fn, hit[1], _TC_STREAM_IDX)
                hit[2].replay()
                return 0
            except Exception:
                _GRAPHS_OK = False   # capture unsupported here; stay plain
        return fn(*hit[1])
    heads, kv_heads = query.shape[1], key.shape[1]
    total_q, total_kv = query.shape[0], key.shape[0]
    if dr:
        ws = _WS["t"]          # no workspace in direct mode (may stay None)
        kws_p = vws_p = 0
        kvblks = 0
    else:
        o = [ctypes.c_longlong() for _ in range(3)]
        lib.mla_tc_ws(total_kv, batch, kv_heads, *[ctypes.byref(x) for x in o])
        kvblks, kby, vby = (x.value for x in o)
        ws = _ensure_ws(_align256(kby) + _align256(vby))
        kws_p = ws.data_ptr()
        vws_p = kws_p + _align256(kby)
    args = (query.data_ptr(), key.data_ptr(), value.data_ptr(), out.data_ptr(),
            lp, sp, cum_q.data_ptr(), cum_kv.data_ptr(),
            kws_p, vws_p,
            total_q, total_kv, batch, heads, kv_heads, kvblks,
            window_left, causal, has_lse, has_sink,
            float(bmm1), float(bmm2), max_kv, stream_ptr, dbg_ptr, dbg_mode,
            pf, lm, dr, bx, op)
    assert args[_TC_STREAM_IDX] == stream_ptr, \
        "graph-capture stream index must track the tc ABI layout"
    if len(_TC_CACHE) > 64:
        _TC_CACHE.clear()
    _TC_CACHE[ck] = [ws, args, None]
    if dr and dbg_mode == 0 and dbg_ptr == 0:
        # memo-safe: direct mode has no workspace dependency to invalidate
        _LAST_ARGS[0] = (fn, args, _TC_STREAM_IDX)
    return fn(*args)


_FORCE = os.environ.get("MLA_FORCE", "").strip().lower() or None


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name, default):
    """Same contract as _env_int for a non-integer knob.

    Added (join 20261024, arm-b) for the quantisation gate's penalty and work
    floor, which are not sensibly integers. A missing or unparseable override
    yields the default, so an experiment typo can never take the candidate
    down at import.
    """
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# tc engine knobs (experiments override via env; defaults = promoted policy).
# MLA_PF: 1 = V0 L2-prefetch policy (job4: prefetch ON is best on fixed and
#         domain; OFF regressed c1/c4/d4 -> TMA feed is latency-bound).
# MLA_LM: 1 = lagged running max, skip TMEM O correction (job4/job5:
#         consistent ~1-2% tc win; out rel-L2 vs V0 ~1e-3 << bf16 noise).
# MLA_TCD: 1 = direct-TMA raw K/V loads (no prep kernel/workspace; job5
#          dv_...: fixed tc-geomean 186.2->173.6us, small cases -8..-26%),
#          0 = V0 prep-workspace path.
_PF_MODE = _env_int("MLA_PF", 1)
_LAG_MAX = _env_int("MLA_LM", 1)
_TC_DIRECT = _env_int("MLA_TCD", 1)
# tc2 TMA box rows (8/16/32). job8 probe (dv_...): 3KB rank-4 boxes ingest
# ~15B/cyc/SM (L2-hot) vs ~116B/cyc for 48KB boxes -> tc2 dual tiles were
# feed-bound at ~2650 cyc. 16/32 write rows contiguously (SBO=1024 descs).
_BOX_TOK = _env_int("MLA_BOX", 8)
# tc2 opts bitmask: 1=OC fast-arrive (campaign: -2.3% geomean on big di-mode
# cases, outputs OK; DETERMINISTIC HANG on the merged single-issuer loop -
# job7 dv_64312c1e7e23 reproduced the V0-era probe - safe ONLY with bit6
# dual-issuer, which probe17 made unconditional for tc2),
# 2=LPT q-tile order (heavy causal tiles first), 4=skip OOB Q row loads,
# 8=pure-spin mbarrier waits (job20: zero effect), 64=dual-issuer
# (tc2d builds ONLY: warp8 drives subtile 0, warp10 subtile 1, warp9 stays
# a pure loader; BE barriers take one commit per issuer warp. Targets the
# 1083-cyc serial QK issue stream measured in job22. The 320-thread merged
# variant lost in job23 - loader folded into an issuer warp starves BF).
# 128=TMA-Q: two {64,128,3,1} boxes replace the 24x320 cp.async16 Q fill
# (atom-major SW128 layout; issuer QK descs switch SBO/atom-stride to match;
# job25 experiment). Retired bits: 16=runtime
# stage timing (moved to the -DTC2_TIMING "tc2t" build; job21: runtime gates
# cost the default path 21%), 32=issuer descriptor fast-add (job21 A/B: +7.8%
# fixed geomean -> now the only path).
_OPTS_ENV = os.environ.get("MLA_OPTS")
_OPTS = _env_int("MLA_OPTS", 6)
# tc2p box64 feed gate: shapes with max_kv <= this AND waves>=2 run the
# TC2P_BOX64 build (probe30 dv_...810623 same-job: f0(mk512) -5.1%,
# d5(mk1024) -4.0%; large-T/tensor-bound shapes stay box8: f5/d6 +0.2-0.4%,
# f1 ambiguous). One-request-per-stage feed cuts small-T stage latency.
_BOX64_MAX_KV = _env_int("MLA_BOX64KV", 1024)
# join 20261024 (arm-b adoption): CAUSAL box64 extension to mk<=16384.
# arm-b experiments/interleave_ab.py timed both binaries in ONE process in
# rotated order over 15 rounds and paired the rounds (dv_fddbd169af6e):
# box64 priced 0.9910-0.9957 on all seven shapes, faster in 8-15 of 15 rounds
# each, sign-test p=0.0078 under a null of no effect - f1 0.9918 (15/15),
# x2048h96 0.9910 (14/15), f4 0.9957 (12/15), f6 0.9942 (11/15), f0 0.9929
# (10/15), f2 0.9931 (10/15), f7 0.9929 (8/15); about half what forced pairs
# said, worth ~-0.32% of the fixed geomean over the causal mk>1024 cases that
# move. Mechanism: one full-tile TMA request per operand at ~116 B/cyc of
# ingest against ~15 B/cyc for 3KB boxes. CAUSAL ONLY: the three non-causal
# shapes above the old gate measured mixed (xn4096h12 1.0139 against,
# xn2048h96 0.9912 for, xnb2h12 a tie) - not the evidence quality to promote
# on; this tree already gates grouped-LPTG causally for the same reason.
# NOTE (ours): arm-b's tree had neither the v24 SR counter nor the v25 g-rule,
# so the f1 combination tc2p64+bit21-g24+SR is UNMEASURED - probe100 carries
# an f1 attribution arm. MLA_BOX64KVC=1024 restores the join-start gate.
_BOX64_CAUSAL_MAX_KV = _env_int("MLA_BOX64KVC", 16384)
# RETIRED v26 (probe88/89, addendum 21): the OC fast-arrive (opts bit0)
# max_kv gate. History: di+bit0 corrupts short-T items below 1024
# (probe23/24: d7 4/14 dirty incl. the v12 official failure) -> bit0 was
# gated to mk>=1024; probe88 then proved tc2p+bit0 races at ALL mk under
# the peaked distribution (5-15%/launch, value-gated on normal) -> bit0
# disabled in every production policy branch. Constant kept for historical
# probe scripts (probe78's formula echo); no policy reads it.
# join-20261024 C3: re-applied on the post-join tree; probe100 f1p 2/40
# (dv_db8159e5ac73) extends the race record to the tc2p64 build.
_DI_MIN_KV = 1024
# LPTG (opts bit20, probe54 dv_79ce314e8488): GLOBAL descending-T item
# enumeration for the tc2p family. Per-head-block LPT leaves the last heavy
# tile at 94-96% of the steal order (measured residual span_max-span_avg ~=
# 0.5*max_unit). Global sort: f7 0.872, d3 0.841, SYN-ragged 0.823, d5
# 0.937, f8 0.945, f0 0.964 cold - but tile-major order costs the per-head
# cross-tile KV L2 reuse: f1 1.033, d4 1.144 when the concurrent working
# set heads*max_kv*640B exceeds L2 (winners <=63MB, losers >=126MB of the
# 126.5MB L2; 90MB gate separates 8/8 with margin). Bit-equality 9/9 vs
# production opts (enumeration-only change).
_LPTG_WS_MAX = _env_int("MLA_LPTGWS", 90_000_000)

# Grouped LPTG (opts bit21, probe60 dv_28aab7be98ab): for ws > gate shapes
# the tile-major bit20 order busts L2, but GROUP-major order (gsz heads at
# a time over the same sorted table) keeps the residual win where one
# exists: f1 0.976 vs prod (causal, T-spread), fallback-class f5/d6/d9
# 0.89-0.92 vs plain tc2p; d4 (non-causal, uniform T -> no residual) loses
# 1.064 -> causal-only gate. Bit-equality 9/9 3-way (prod/bit20/bit21)
# incl. 2 seeds.
# v25 (probe86 dv_0caf1a3bffa0): the g-curve is U-SHAPED - g=1 (sorted
# head-major) LOSES (f1 1.037, SYN96 1.088, SYN128 1.053: minimal live
# set does NOT win), and the optimum sits at live set g*mk*640 ~= 30-40MB
# (~25-33% of the 126.5MB L2): f1 {g24 0.9683 < g16/g32 0.9715 < g8
# 0.9884 < g48 1.0 < g4 1.0124 < g1 1.0374}, SYN96 g24 0.9713, SYN128
# g16 0.9793. One shape-derived rule reproduces all three winners:
# g = largest divisor of heads with g*mk*640 <= _LPTG_G_OPT (40MB).
# The 90MB _LPTG_WS_MAX SELECTION gate (bit20 vs bit21) is unchanged -
# the 40-90MB band stays bit20 (measured wins; d5/f0-f7/d3 untouched).
# BE bit-equal all-g both seeds (reorder-only); numerics 0 mismatch.
_LPTG_G_OPT = _env_int("MLA_LPTGGOPT", 40_000_000)
_LPTG_G_CACHE = {}


# --- persistent-cluster wave-quantisation gate (join 20261024, arm-b) ------
# WAVE QUANTISATION. With n = (total_q//256 + batch)*heads pair-items, a
# persistent single-CTA engine runs ceil(n/148) rounds against a mean of
# n/148, and a persistent cluster engine runs ceil(n/74) against n/74. The
# ratio of those two penalties predicted every one of arm-b's 42 measurements:
# q_cl/q_cta <= 0.75 was always a win (x1024 0.732 at ratio 0.50, xq156 0.800,
# f8 0.917, x4096 0.951 and xq192h24 0.946 at 0.75) and >= 0.875 always a tie
# or a loss (xgqa8 1.100, f5 1.061, xq252 1.077, f2 1.159, x2048h96 1.176,
# xq288h16 1.172, f6 1.152, f9 1.186, f1 1.225). heads > 24 lost regardless of
# ratio - the same limit this tree's cluster envelope was measured to.
# The ratio is not enough on its own: every one of the 8 shapes where the bare
# ratio rule misfired is latency-bound rather than throughput-bound (all
# total_q <= 257 and w_tf <= 0.0005), while every win measured >= 20.5 us -
# quantisation cannot pay when one round is already inside the launch/prologue
# floor. dv_b4e2a6fa5e18 measured a point inside the gap (xgap768: 768 tokens,
# heads 12, w_tf 0.0023, n 48, wins 0.835), so the floors sit ON the measured
# boundary: w_tf 0.0015 is 3x above the largest loss (xgqa2, 0.0005) and 1.5x
# below the smallest win; n 48 is xgap768's own count and 2x the largest
# loss's (n 12-24).
_TC9R_HEADS_MAX = _env_int("MLA_TC9RH", 24)
# Per-item handicap the quantisation gain has to cover: measured tc9r/auto on
# the shapes where quantisation is neutral (ratio 1.00) is f7 1.000, f4 1.077,
# f0 1.110, f2 1.159, f6 1.152, xq252 1.077, xq288h16 1.172, f9 1.186, and f5
# implies 1.157. 1.25 is above that whole band, so the rule fires only where
# quantisation alone more than pays for the engine's per-item cost; anything
# below 1.09 would admit f5, which loses 6.1%. NOT 1.20 (what the band was
# fitted to): the band's ratio is exactly 5/6 and 1.20 is exactly 6/5, so
# q_cl*P/q_cta is 1.000000 there and the strict comparison is decided by
# floating-point noise (n=300 "no", n=360 "fire" at the same ratio) - a gate
# whose third region sits on its own zero-margin contour is not a mechanism.
# 1.25 closes it (1.0417) and costs nothing measured: every win has ratio
# <= 0.75, keeping a 6% margin.
_TC9R_PENALTY = _env_float("MLA_TC9RP", 1.25)
_TC9R_MIN_WTF = _env_float("MLA_TC9RW", 0.0015)
_TC9R_MIN_ITEMS = _env_int("MLA_TC9RN", 48)
# MLA_TC9SH=0 restores the join-start routing of the shallow cluster branch to
# tc7l. The rule it gates is measured in arm-b dv_45555815566f: the persistent
# cluster engine beat tc7l on all six shapes that branch can reach, 11/11
# interleaved rounds each, by 7.7-16.6%, against a same-binary noise floor of
# 0.5%.
_TC9_SHALLOW = _env_int("MLA_TC9SH", 1)


def _quant_prefers_cluster(n_items, heads, w_tf):
    """True when 74 persistent clusters quantise this shape better than 148 CTAs.

    A function rather than inline arithmetic because the audits have to replay
    the shipped decision: an audit that re-implements a gate reports the cost
    of a gate that no longer exists (arm-b experiments/policy_diff.py lifts
    this by name).
    """
    if (heads > _TC9R_HEADS_MAX or n_items < _TC9R_MIN_ITEMS
            or w_tf < _TC9R_MIN_WTF):
        return False
    q_cl = -(-n_items // 74) * 74.0 / n_items     # ceil(n/74) / (n/74)
    q_cta = -(-n_items // 148) * 148.0 / n_items  # ceil(n/148) / (n/148)
    return q_cl * _TC9R_PENALTY < q_cta


_TC9R_DEEPMK = _env_int("MLA_TC9RDMK", 8192)
# Quant-degenerate deep-window clause (join-20261027 phase: probe188
# dv_2fa0c88f856a + probe189 dv_da7147708f9b + probe190 dv_e43c9deeee8b
# + probe191 dv_5925d817694f). Knobs: kill MLA_TC9DEG=0; window
# MLA_TC9DEGMK/MKMAX; heads MLA_TC9DEGH (16 until the h24 degenerate
# classes n408/n432 are priced - probe192).
_TC9_DEGEN = _env_int("MLA_TC9DEG", 1)
_TC9_DEGEN_MIN_KV = _env_int("MLA_TC9DEGMK", 4096)
_TC9_DEGEN_MAX_KV = _env_int("MLA_TC9DEGMKMAX", 5120)
# heads 24 (probe193 dv_ad16e1487cfc, 30-round confirmation per the
# measurement law): the h24 degenerate window classes n408 ([4096]h24,
# 0.9655 30/30) and n432 ([4500]h24, 0.9584 30/30) are measured wins;
# n456/n480-h24 in-window are NON-degenerate (skipped by the test);
# h96/h128 stay out (heads cap). 16 -> 24 raised on probe193's
# pre-registered RAISE_HEADS_24 verdict (probe192's 15-round 0.9747 had
# missed the <=0.97 bar; the 30-round draw measured 0.9655).
_TC9_DEGEN_HEADS_MAX = _env_int("MLA_TC9DEGH", 24)
# join-20261027 phase (probe206/207/208): big-ws class move. Inside the
# envelope gate, ws = heads*mk*640 > 90e6 AND heads <= 12 -> tc9r instead
# of tc7eu. Evidence: f5 0.9661/0.9926 (60/60 rounds over two jobs;
# baseline mode draw documented), d9 0.9858/0.9857 (50/50), x12288h12
# 0.9338 x2, xw16384h12w4096 0.7795/0.7791 - 180/180 pooled paired wins,
# refs clean. RESTRICTION priced, not inherited: x16384h16 1.0042 (band;
# h16 splits on page_size - d6 0.9827 forgone), x16384h24 1.0395 CLOSED.
# PFL is the mechanism mover: probe189's f5 1.0057 was the PRE-PFL
# binary. Kill MLA_TC9BIGWS=0 restores the tc7eu locality law.
_TC9_BIGWS = _env_int("MLA_TC9BIGWS", 1)


def _deepmk_prefers_cluster(n_items, heads, w_tf, batch, total_q, max_kv_len):
    """True when a DEEP-WAVE shape's kv extent alone says cluster, at equal quantisation.

    _quant_prefers_cluster only ever picks tc9r when 74 clusters quantise the shape better
    than 148 CTAs, penalised by _TC9R_PENALTY. For f2, f4 and f6 that comparison is
    degenerate: q_cl == q_cta EXACTLY (f2 1.088/1.088, f4 and f6 1.121/1.121), so the
    cluster offers no quantisation advantage at all and the fitted 1.25 penalty decides all
    three by itself. dv_b3b3bf5bfbcd measured them instead of trusting the penalty:

        f4  tc9r 198.848 vs tc2p64 208.672  ->  tc9r faster by 8.224 us (4.0%), 7/7
        f2  tc9r  84.000 vs tc2p64  84.000  ->  tie, 6/7
        f6  tc9r  96.448 vs tc2p64  94.208  ->  tc2p64 faster by 2.208 us, 0/7

    all three correct under force, with f3/f8/f9 riding along as controls that reproduced
    their dv_e0ebd558e98d margins to +0.000, +0.000 and -1.952 us. So the branch is not
    wrong, it is UNMEASURED for this class, and measuring it splits the three rather than
    moving them together.

    The separator is max_kv_len, and it has a mechanism rather than being a fitted cut: the
    cluster's advantage is that its half of a pair M256N128K16 is FREE (64.02 cyc, the same
    as a single M128N128K16, so the second CTA's half of the tile costs nothing). That is an
    advantage PER MMA, so it compounds with the number of kv tiles in an item while the
    per-item prologue and epilogue amortise over them. f4's largest item spans 8192 kv
    columns = 64 tiles at BN=128 against 128 steps at tc2p64's BN=64; f2 and f6 sit at 4096
    and 4095 and do not get enough of that compounding to overcome the per-item cost. The
    measured ordering across the class is monotone in mk: 4095 -> -2.3%, 4096 -> 0.0%,
    8192 -> +4.0%.

    Deliberately a STRICT SUBSET of the cluster envelope above (heads <= 24, mk >= 1024,
    w_tf >= 0.0161, total_q >= 512*batch), re-stated here rather than inherited, so this
    branch can never reach a shape the envelope excluded for a measured reason. It also sits
    after the ws > 90MB -> tc7eu return, so the high-ws class (f5, domain d6/d9) keeps the
    routing the locality law gave it. Kill switch MLA_TC9RDMK=<big> restores the certified
    routing exactly; check_dispatch_change.py asserts that over 491 shapes and all 10 domain
    cases rather than taking it on faith.
    """
    if _TC9R_DEEPMK <= 0 or int(max_kv_len) < _TC9R_DEEPMK:
        return False
    return (heads <= _TC9R_HEADS_MAX and n_items >= _TC9R_MIN_ITEMS
            and w_tf >= 0.0161 and int(total_q) >= 512 * max(int(batch), 1))


def _degen_deep_prefers_cluster(n_items, heads, w_tf, batch, total_q,
                                max_kv_len, causal, window_left):
    """True at EXACT quantisation degeneracy inside the measured mk window.

    join-20261027 phase (probe188-191). Degenerate := ceil(n/74) ==
    2*ceil(n/148), i.e. q_cl == q_cta exactly - the class where arm-b's
    own deepmk comment records that "the fitted 1.25 penalty decides by
    itself" (dv_b3b3bf5bfbcd measured f2 at a TIE, 84.000/84.000, on the
    OLD tc9 binary). On the NEW binary (MSKIP + EARLY_PST on top of
    KVSPLIT/DRAINLATE) the tie broke to tc9r's side and the class split
    by mk window. Measured map (tc9r/tc2p64, refs clean everywhere):
      WINS inside mk[4096,5120):
        f2 [4096]h16 n272    0.9524 15/15 + 0.9756 15/15 + 0.9750 30/30
                             (three jobs, 60/60 rounds) - OFFICIAL case
        x5000 [5000]h12 n240 0.9170 15/15 + 0.9378 15/15 (two jobs)
        p4500h16 [4500]h16 n288 0.9796 12/15 (modest; inclusion recorded
                             with the probe191 pre-registry ambiguity note)
      LOSSES outside the window (excluded by CAP=5120 / floor 4096):
        p5200 n252 mk5200    1.0883 0/15   p5500 n264 mk5500 1.0621 0/15
        x6000 n288 mk6000    1.0636 0/12 + 1.0545 0/30 (two jobs; the
                             loss is mk/ws-specific, NOT n288-specific -
                             n288-h16 at mk4500 wins)
        x7000 n336 non-degen 1.0409 0/15   f6 n396 mk4095 tie (below floor)
      WINS outside the window (NOT captured - conservative; recorded for
      future boundary work): x8191 n384 0.9604 10/10 + 0.9506 20/20;
        g6144h16 n400 0.9490 15/15; b2deep n312 NON-degen 0.9412 12/12 +
        0.9289 20/20.
    Guards: causal + window<0 (all measured points causal; nc degenerate
    stays with quant/tc2p64 - x4096nc already rides tc9r via quant);
    heads <= _TC9_DEGEN_HEADS_MAX = 16: the h12/h16 classes above are
    measured, the h24 degenerate classes inside the window (n408
    mk[4096,4352), n432 mk[4352,4608)) are UNMEASURED (probe192 prices
    them; win -> raise the knob to 24). w_tf >= 0.0161 and tq >= 512*b
    reuse the deepmk/envelope fitted guards (d3's tq/b=158 exclusion
    measured 1.0438 loss - the guard holds it out). Non-degenerate
    shapes inside the window (h16 n304/n320, h12 n216-quant-owned) are
    skipped by the degeneracy test itself.
    Mechanism reading (conditional): at equal quantisation the choice
    reduces to per-item throughput, where the pair-MMA rate compounds
    with kv depth (arm-b's deepmk story) - the new binary moved that
    balance at mk>=4096, while the 5120 cap bounds where the compounding
    still beats tc2p64's QSPLIT-fed cadence (n252/n264 losses).
    Kill switch MLA_TC9DEG=0 restores the certified routing exactly.
    """
    if (_TC9_DEGEN <= 0 or not causal or int(window_left) >= 0
            or heads > _TC9_DEGEN_HEADS_MAX
            or n_items < _TC9R_MIN_ITEMS
            or w_tf < 0.0161
            or int(total_q) < 512 * max(int(batch), 1)
            or not (_TC9_DEGEN_MIN_KV <= int(max_kv_len)
                    < _TC9_DEGEN_MAX_KV)):
        return False
    n = int(n_items)
    return -(-n // 74) == 2 * (-(-n // 148))


# --- span-dominance gate (E1 dv_b2d688c4da55 + E1b dv_995b4540252d) --------
# The quantisation gate models THROUGHPUT (rounds x penalty); it systematically
# rejects shapes whose makespan is set by the DEEPEST SINGLE ITEM instead: a
# causal q-tile at the end of a deep sequence chains ceil(mk/64) BN=64 tiles on
# ONE SM under tc2p64 (~2000-2200 cyc each), while the cluster engine runs the
# same item as ceil(mk/128) BN=128 pair tiles (~1600 cyc, 8189 vs 5459
# flop/cyc/SM). arm-b only ever priced the SERIAL cluster there (tc7: f7 1.019,
# d3 1.171 - its ~17k cyc/item prologue over ~370-560 items), which is why the
# class stayed on tc2p64.
#
# In-process interleaved rotated A/B vs THIS tree's production routing (10
# rounds x 3 reps, official-proxy cold timing per restore-l2-evict-event-v2;
# cross-arm + fp32-reference zero-mismatch numerics on every checked shape):
#     f7   (15x1+4096, h12, page1)   tc9r/tc2p64 0.9091  10/10 rounds
#     f7p  (same + peaked)                         0.9321  10/10
#     x4096t15 (page64 twin of f7)                 0.9104  10/10
#     x2048t15 (15x1+2048, h12)                    0.7805  10/10
#     x1024t15 (15x1+1024, h12)                    0.8887  10/10
#     x4096t0  ([4096] alone, arm P forced tc2p64) 0.8301  10/10
# Losses the gate must NOT admit (all 0/10 rounds for C):
#     d3 family (31 odd 1..61 + 4096, h12; peaked/normal, lse/no-lse,
#                page128/64 variants)              1.0627-1.0848
#     f6   (257/1023/2049/4095, h12)               1.0221
#     x4096b4 / x2048b8 (uniform deep, work-dominated) 1.0179 / 1.0277
#     x8192b2 (forced P)                           1.0052 (tie)
# Mechanism separating them: tc9r's marginal per-item cost is ~3x tc2p64's
# (d3's +192 items vs f7 cost the C arm exactly the measured +22.5us:
# 192/74 clusters x ~17-19k cyc). Span saving has to cover items x delta:
#   span-dominance proxy (dispatch-visible aggregates only, no host sync):
#     R' = heavy_item_flops / mean_flops_per_SM
#        = (256 * mk * 640) / (0.5 * total_q * mk * heads * 640 / 148)
#        = 75776 / (total_q * heads)          [causal, mk cancels]
#   item-count proxy n_tc2 = (total_q//256 + batch) * heads  (already computed)
# Measured points under the proxies: winners R' = 1.54-6.09 with n_tc2 =
# 204-384; losers R' = 0.39-1.25 with n_tc2 = 396-816 (d3: R' 1.25 AND n_tc2
# 612 - it fails BOTH cuts, so the conjunction has a redundant margin around
# the single boundary pair f7(1.54, 384) / d3(1.25, 612)).
# Thresholds: R' >= 1.4 (12% above d3, 9% below f7) and n_tc2 <= 460 (17%
# above the largest winning n, 25% below the smallest losing n). mk >= 1024
# reuses the envelope's measured floor; heads <= _TC9R_HEADS_MAX and the
# w_tf/n floors reuse the quantisation gate's fitted latency-bound guards
# (every measured win >= 32.7us, above arm-b's 20.5us win floor). Causal-only
# (the R' cancellation assumes the causal half-triangle; non-causal shallow
# shapes stay on arm-b's measured n<=148 branch). window_left guard: the
# frozen domain fixes -1; a windowed shape's span model is unmeasured.
# LOWER bound n_tc2 > 148 (waves >= 2): every measured win on THIS branch has
# n in [240, 384] against tc2p64; the routing audit found the unconstrained
# form would also move ten single-wave shapes off classic tc2 (n in [48,148],
# e.g. [2047] h12 misses the envelope's w_tf floor by 0.06%) - a zone neither
# E1b nor arm-b priced against tc2 (arm-b's shallow branch measured tc9r vs
# tc7l, not vs tc2). Left on classic tc2 pending a dedicated probe;
# MLA_TC9RSPMIN overrides the floor.
# Kill switch MLA_TC9RSPMK=0 restores the certified routing exactly.
_TC9R_SPAN_MK = _env_int("MLA_TC9RSPMK", 1024)
_TC9R_SPAN_ITEMS_MIN = _env_int("MLA_TC9RSPMIN", 149)
_TC9R_SPAN_ITEMS_MAX = _env_int("MLA_TC9RSPN", 460)
_TC9R_SPAN_RP = _env_float("MLA_TC9RSPRP", 1.4)
_TC9R_SPAN_RP_NUM = 75776.0     # 256-row heavy item x 148 SMs / 0.5 causal


def _span_prefers_cluster(n_items, heads, w_tf, total_q, max_kv_len):
    """True when the deepest single item, not aggregate work, sets the makespan
    and the item count is low enough for the cluster's per-item cost to stay
    covered. See the block comment above for the full evidence table."""
    if (_TC9R_SPAN_MK <= 0 or _TC9R_SPAN_ITEMS_MAX <= 0
            or int(max_kv_len) < _TC9R_SPAN_MK
            or heads > _TC9R_HEADS_MAX
            or n_items < max(_TC9R_MIN_ITEMS, _TC9R_SPAN_ITEMS_MIN)
            or n_items > _TC9R_SPAN_ITEMS_MAX
            or w_tf < _TC9R_MIN_WTF):
        return False
    rp = _TC9R_SPAN_RP_NUM / (max(int(total_q), 1) * max(int(heads), 1))
    return rp >= _TC9R_SPAN_RP


def _lptg_group(heads, mk):
    key = (int(heads), int(mk))
    g = _LPTG_G_CACHE.get(key)
    if g is None:
        lim = _LPTG_G_OPT // (int(mk) * 640)
        g = min(int(heads), lim)
        while g >= 1 and int(heads) % g:
            g -= 1
        _LPTG_G_CACHE[key] = g
    return g
# RETIRED (probe22): tc2s split-KV dispatch gate knob - see _pick_engine
# comment; forced-engine runs (MLA_FORCE=tc2s) still work.
_SPLIT_MIN_T = _env_int("MLA_SPLITMIN", 10)
# tc3 bits: 1=always wait PV(i-1) done before OC arrive (reference-style
# serialization; REQUIRED for correctness - job17 dv_...: without it, peaked
# cases nondeterministically corrupt O while lse stays correct), 2=never
# scale O (WRONG, bisection only), 4=exact per-tile max (no lag).
_T3OPT = _env_int("MLA_T3OPT", 1)


def _pick_engine(total_q, total_kv, batch, heads, kv_heads, max_kv_len, causal,
                 max_q_len=None, window_left=-1):
    if _FORCE in ("sync", "tc", "tc2", "tc2t", "tc2d", "tc2dt", "tc3",
                  "tc4", "tc4t", "td", "tc2p", "tc2pt", "tc2s", "tc2st",
                  "tc2c", "tc2ct", "tc7", "tc7e",
                  "tc7nr", "tc7enr", "tc7enp",
                  "tc7eu", "tc7ep", "tc7f", "tc7fm",
                  "tc7eux", "tc7eus", "tc7l", "tc9", "tc9r",
                  "tc9u", "tc9g", "tc9s", "tc9b", "tc9pm", "tc9t", "tc9q", "tc9qp", "tc9t4", "tc9e", "tc9et4", "tc9t5", "tc9l", "tc9lt5",
                  "tc2pb", "tc2b", "tc2pe", "tc2pet", "tc2pf",
                  "tc2pft", "tc2e", "tc2et", "tc2x", "tc2px",
                  "tc2p14", "tc2p64", "tc2pm", "tc2pmt", "tc2em",
                  "tc2p64x", "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5",
                  "tc2emt", "tc2l2q", "tc2l2n"):
        if _FORCE in ("sync", "td"):
            return _FORCE
        if _LIBS.get(_FORCE) is None:
            # Loud failure for forced experimental engines: a silent sync
            # fallback masked the tc2 build failure for a whole job
            # (dv_42647ee6bbde).
            raise RuntimeError(f"MLA_FORCE={_FORCE} but that engine failed to build")
        return _FORCE
    # Decode-singleton regime (generic shape metadata): every sequence is
    # q==1 AND kv==1 -> attention degenerates to out = v * bmm2 (x sink
    # rescale), lse = score-derived; the warp-per-(batch,head) td kernel
    # computes it without any tiled-pipeline prologue. Measured (probe12,
    # dv_...720432): d2-class 32x1x16 e2e -36%; numerics 0 mismatch (d0/d2
    # bit-exact vs reference). q==1 with kv>1 stays on the general engines:
    # td's per-row shfl-reduce chain serializes badly there (probe12: 4.4x
    # slower at kv=64) and the frozen domain invariant query_lengths ==
    # kv_lengths makes q=1 ^ kv>1 unreachable in-domain.
    if (_LIBS.get("td") is not None and max_q_len is not None
            and int(max_q_len) == 1 and int(max_kv_len) == 1
            and int(window_left) < 0):
        return "td"
    # Generic cost-model dispatch on shape metadata only (no host sync, no
    # case recognition):  est = call_oh + work/rate + CTA-waves*wave_cost.
    proxy_work = total_q * max_kv_len * (0.5 if causal else 1.0)
    w_tf = heads * proxy_work * 640.0 / 1e12
    big = max_kv_len >= 8192          # sync engine uses C3 (1 CTA/SM) there
    n_sy = ((total_q // (128 if big else 64)) + batch) * heads
    waves_sy = -(-n_sy // (148 if big else 296))
    est_sy = 10.0 + w_tf * 1e6 / 300.0 + waves_sy * 4.5
    if _LIBS.get("tc2") is not None:
        # refit#5 on job7 medians (dv_b57a30f277f6; fit_dispatch3.py:
        # 20/20 oracle, min margin 2.4% at d8). tc2 = dual-subtile
        # BM=256/BN=64/3-stage direct-TMA kernel.
        n_tc2 = ((total_q // 256) + batch) * heads
        waves_tc2 = -(-n_tc2 // 148)
        est_tc2 = 10.0 + w_tf * 1e6 / 850.0 + waves_tc2 * 8.0
        if est_tc2 < est_sy:
            # tc7 cluster kernel (arm-b join import; probe44 dv_d6c7fde0edc9
            # same-job cold official-proxy A/B vs THIS tree's production):
            # wins the tensor-bound envelope - f4 0.898, f2 0.902, f9 0.901,
            # f5 0.931, f3 0.957, f6 0.973, f8 0.986, d6 0.966, d9 0.947;
            # numerics 12/12 zero-mismatch (winners x2 seeds, out+lse).
            # Loses: softmax-bound h96/h128 (no E2: f1 1.337, d4 1.208),
            # small-mk (f0 1.209), ragged-tiny-mean (f7 1.019, d3 1.171).
            # Gate = the measured winning envelope, shape-metadata only:
            #   heads <= 24        (softmax-bound classes stay on E2 tc2p)
            #   mk >= 1024         (small-kv stays on tc2p64/tc2)
            #   w_tf >= 0.0161     (arm-b floor: f9 = smallest measured win;
            #                       below it the constant terms mis-rank)
            #   total_q/batch >= 512 (ragged-tiny sequences keep tc2p)
            # v21 three-way envelope split (probe56/57/58; ws = heads*mk*640
            # = tile-major concurrent KV working set vs the 126.5MB L2):
            #   ws > 90MB: tile-major order breaks per-head KV L2 reuse
            #     (locality law: d4 +14.4%, tc7l f6 +10.7%) -> plain tc7eu
            #     (f5/d6/d9 class).
            #   ws <= 90MB && n_tc2 > 148: deep waves -> persistent tc2p with
            #     LPTG beats BOTH cluster forms (probe56: f6 0.802, f2 0.860,
            #     f3 0.922, f4 0.950; probe58: f8 0.9755 at n_tc2=156;
            #     probe57 X2: tc7l 1.05-1.36 vs tc2pL) -> fall through to the
            #     tc2p branch below (n_tc2 > 148 <=> waves_tc2 >= 2; op gets
            #     bit20 from the tc_launch ws gate).
            #   ws <= 90MB && n_tc2 <= 148: shallow static waves -> tc7l
            #     (probe57: f9 0.891; steal never cycles so tc2p is out and
            #     tc7eu carries the late-heavy-start residual). Build
            #     failure falls back to tc7eu.
            if (_LIBS.get("tc7eu") is not None and heads <= 24
                    and int(max_kv_len) >= 1024 and w_tf >= 0.0161
                    and total_q >= 512 * max(batch, 1)):
                if heads * int(max_kv_len) * 640 > _LPTG_WS_MAX:
                    # probe206/208: POST-PFL tc9r wins the h12 big-ws
                    # class (f5/d9/x12288h12/windowed, 180/180 paired
                    # rounds); h16 mixed (page-size split), h24 loses
                    # 1.0395 -> heads<=12 restriction. Kill switch
                    # MLA_TC9BIGWS=0.
                    if (_TC9_BIGWS and heads <= 12
                            and _LIBS.get("tc9r") is not None):
                        return "tc9r"
                    return "tc7eu"
                if n_tc2 <= 148:
                    # join 20261024 (arm-b): the PERSISTENT cluster engine
                    # takes this branch when there is more than one pass to
                    # amortise; tc7l keeps the single-pass shapes. probe57
                    # chose tc7l here on f9 0.891, but that was calibrated
                    # against a tc9r whose static stride cost 1.713x the mean
                    # load on exactly that shape; the rebalance (TC9_GSORT/
                    # TC9_SERP) removed it, and arm-b dv_45555815566f then
                    # measured all six shapes this branch can reach,
                    # interleaved in one process over 11 rotated rounds, with
                    # tc7l/auto = 0.995-1.003 as the same-binary noise floor:
                    #   f9        tc9r/auto 0.9230  (11/11 rounds)
                    #   xsh16c    0.9076 (11/11)   xsh12nc   0.8932 (11/11)
                    #   xsh12b2nc 0.8337 (11/11)   xsh16nc   0.8901 (11/11)
                    #   xsh24nc   0.8780 (11/11)
                    # Four of those six are NON-causal, where the rebalance is
                    # gated off inside the kernel - this is persistence
                    # beating a per-item cold start, not balance, and it does
                    # not depend on the causal gate. n_tc2 > 74 is the
                    # mechanism condition rather than a fitted one: at or
                    # below one cluster's worth of items every cluster gets a
                    # single item, persistence has nothing to amortise and the
                    # zigzag degenerates. Kill switch MLA_TC9SH=0 restores the
                    # join-start (tc7l) routing.
                    if (_TC9_SHALLOW and n_tc2 > 74
                            and _LIBS.get("tc9r") is not None):
                        return "tc9r"
                    if _LIBS.get("tc7l") is not None:
                        return "tc7l"
                    return "tc7eu"
            # Wave quantisation (see _quant_prefers_cluster), join 20261024
            # arm-b adoption; GENERALISES the v23 (148,176] window (probe73:
            # f8 0.9562; f3/f7 ties back then - f3 now fires the gate on the
            # rebalanced engine: q_cl(216)*1.25 = 1.285 < q_cta = 1.370, and
            # arm-b's certified f3 is -3.9% against this tree's 53.248).
            # Placed after the cluster envelope above, whose measured wins
            # keep priority, and before the persistent single-CTA branch -
            # deliberately OUTSIDE the envelope's own guards, because the
            # largest measured win is x1024 (0.732), whose w_tf of 0.0040 is
            # 4x below the envelope's 0.0161 floor and which the envelope
            # therefore never even considers. f8 (0.917) is the case that
            # reaches it from inside: in-envelope, ws <= 90MB, n_tc2 > 148,
            # so it falls through to here.
            # Span-dominance branch (E1/E1b, see _span_prefers_cluster):
            # deepest-item makespan shapes the throughput gates reject.
            # Placed with the quant/deepmk gate it generalises; the envelope
            # above (with its measured wins) keeps priority, and the
            # persistent single-CTA branch below is unchanged for everything
            # this does not fire on (f7-class moves; d3/f6/x4096b4 measured
            # losses stay on tc2p64).
            if (_LIBS.get("tc9r") is not None
                    and (_quant_prefers_cluster(n_tc2, heads, w_tf)
                         or _deepmk_prefers_cluster(n_tc2, heads, w_tf, batch,
                                                    total_q, max_kv_len)
                         or _degen_deep_prefers_cluster(n_tc2, heads, w_tf,
                                                        batch, total_q,
                                                        max_kv_len, causal,
                                                        window_left)
                         or (causal and int(window_left) < 0
                             and _span_prefers_cluster(n_tc2, heads, w_tf,
                                                       total_q,
                                                       max_kv_len)))):
                return "tc9r"
            # Persistent-CTA variant (kernel_tc2p.cu) when the shape runs
            # >= 2 waves of (q-tile, head) items: serial CTAs pay ~12k cyc
            # fixed cost per item (prologue ~7.4k + epilogue ~3.5-5k);
            # tc2p pays it once per SM. probe16 (dv_7361e784b559) same-job
            # A/B: f1 -13.7%, d4 -4.5%, f0 -3.7%, f2 -2.1%, d9 -1.6%,
            # f5 -1.1%, f3/f8/d6 within noise; EVERY waves==1 shape lost
            # (d8 +18.1%, d7 +8.4%, f9 +5.3%, d1 0.0%) -> waves>=2 gate.
            if _LIBS.get("tc2p") is not None and waves_tc2 >= 2:
                # join 20261024 (arm-b): causal shapes take the box64 build
                # up to mk<=16384 (see _BOX64_CAUSAL_MAX_KV); non-causal
                # keeps the old 1024 gate (mixed measurements).
                if _LIBS.get("tc2p64") is not None and int(max_kv_len) <= (
                        _BOX64_CAUSAL_MAX_KV if causal else _BOX64_MAX_KV):
                    return "tc2p64"
                return "tc2p"
            # tc2s split-KV RETIRED from dispatch (probe22 dv_9ec9fdb11993,
            # f9 3-arm decomposition): the split mechanism itself wins
            # (r_splitcost 0.927, span_max -14k cyc) but base machinery
            # costs +10.3% (prep node + grid-148 ramp + phase-0 fetch) and
            # the gmem merge epilogue runs 11.3k (part0 export) / 28.2k
            # (part1 spin+merge) cyc - coalescing the ws layout moved it
            # <5%. Net f9 ratio 1.02. A cluster/DSMEM merge (TMEM-local,
            # no L2 round-trip) is the only design that could make this
            # pay; kernel kept buildable via MLA_FORCE=tc2s for that
            # future attempt. Evidence: results/probe{18..22}_*.json.
            #
            # tc2c cluster/DSMEM split-KV: BUILT, MEASURED, RETIRED to
            # forced-only (probe34-37 + probe35 x2, 2026-10-21/22;
            # results/tc2c_evidence.md). The DSMEM mechanism works
            # (numerics 10/10 zero-mismatch incl. split/lse/sink/multi-
            # batch; remote read contention-free at 74 clusters), but the
            # f9 economics never close: v1 serial merge +10.2%, v2
            # bidirectional +4.9% same-job A/B vs tc2@199. Split saving
            # (~35.6k cyc on the Tmax item) is eaten by the merge chain
            # (export ~5k + 4-warp-bound fabric read 7-12k + ops + ACK
            # tail ~ +15-20k) plus cluster base costs (phase0 ~3.6k,
            # launch ramp ~2.3k, cadence 2115->2226/tile) plus tail skew.
            # Ceiling even if perfect: ~-0.6-0.8% fixed geomean; the unit
            # budget caps sp=4 on f9 and the single-unit span floor (T16
            # part + chain ~66k) blocks multi-wave from beating it.
            # AXIS CLOSED with quantitative evidence; MLA_FORCE=tc2c/tc2ct
            # stays wired for any future mk>=8192 waves-1 shape class
            # (where T/2 savings >> chain).
            return "tc2"
        return "sync"
    if _LIBS.get("tc") is not None:
        # refit#4 fallback (tc1 direct-TMA) if the tc2 build failed.
        n_tc = ((total_q // 128) + batch) * heads
        waves_tc = -(-n_tc // 148)
        est_tc = 14.0 + w_tf * 1e6 / 700.0 + waves_tc * 3.0
        return "tc" if est_tc < est_sy else "sync"
    return "sync"


class Model(torch.nn.Module):
    """Exact public ABI of the frozen contract; numerics in candidate/*.cu."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if torch.cuda.is_available():
            _build_all()

    # v3 call-level fast path: the official harness invokes model(**inputs)
    # with one stable tensor set per benchmark phase, and the eager protocol
    # device-syncs before each timed call so ALL python cost is measured.
    # cpu_ab (job dv_3d73aaf8e91b): memo+cuGraphLaunch dispatch is ~7.3us/call
    # of which ~2us is nn.Module.__call__ machinery + 27-kwarg forward
    # binding. This override runs the memo check directly on the kwargs dict
    # (no binding) and falls back to the normal module call on any miss.
    # Semantics on hit are identical to forward()'s memo path: identity +
    # scalar + shape guards, then graph launch of the full numerical work.
    def __call__(self, *args, **kwargs):
        if not args and _MEMO_OK and _MEMO:
            m = _MEMO[0]
            try:
                q = kwargs["query"]
                if (q is m[0] and kwargs["key"] is m[1]
                        and kwargs["value"] is m[2]
                        and kwargs["out"] is m[3]
                        and kwargs.get("lse") is m[4]
                        and kwargs.get("attention_sinks") is m[5]
                        and kwargs["cum_seq_lens_q"] is m[6]
                        and kwargs["cum_seq_lens_kv"] is m[7]
                        and kwargs["max_kv_len"] == m[8]
                        and kwargs["batch_size"] == m[9]
                        and kwargs["window_left"] == m[10]
                        and kwargs["is_causal"] == m[11]
                        and kwargs["return_lse"] == m[12]
                        and kwargs["bmm1_scale"] == m[13]
                        and kwargs["bmm2_scale"] == m[14]
                        and q.shape == m[15]
                        and kwargs["key"].shape == m[16]):
                    ex = m[21]
                    if ex is None and not _NODRV:
                        ex = _drv_capture(m[18], m[19], m[20])
                        m[21] = False if ex is None else ex
                        if ex is not None:
                            m[22] = ctypes.c_void_p(ex)
                    if ex is not None and ex is not False:
                        sp = _current_stream_ptr()
                        cv = _SP_CACHE.get(sp)
                        if cv is None:
                            cv = ctypes.c_void_p(sp)
                            if len(_SP_CACHE) < 16:
                                _SP_CACHE[sp] = cv
                        if _DRV["launch"](m[22], cv) == 0:
                            return m[17]
                        m[21] = False
                    a = list(m[19])
                    a[m[20]] = _current_stream_ptr()
                    rc = m[18](*a)
                    if rc != 0:
                        raise RuntimeError(
                            f"mla memo launch failure: cuda error {rc}")
                    return m[17]
            except KeyError:
                pass
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, query, key, value, workspace_buffer, seq_lens, max_q_len,
                max_kv_len, bmm1_scale, bmm2_scale, o_sf_scale, batch_size,
                window_left, cum_seq_lens_q, cum_seq_lens_kv, enable_pdl,
                is_causal, return_lse, attention_sinks=None,
                skip_softmax_threshold_scale_factor=None, out=None, lse=None,
                sage_attn_sfs=(None, None, None, None),
                num_elts_per_sage_attn_blk=(0, 0, 0, 0), backend='trtllm-gen',
                q_seq_lens_cpu=None, kv_seq_lens_cpu=None, use_fp16_softmax=None,
                uses_spcompress=None, skip_all_rows_active_check=True):
        global _GRAPHS_OK
        # v5 memo fast path: single-slot identity+scalar check, then driver
        # cuGraphLaunch (or raw prebuilt ctypes launch as the fallback).
        # Legality: tensors are pinned by the slot (ids cannot recycle) and
        # every scalar/shape that reaches the baked launch args is re-checked
        # per call; the graph re-executes the full numerical work each call.
        m = _MEMO[0] if (_MEMO_OK and _MEMO) else None
        if (m is not None
                and query is m[0] and key is m[1] and value is m[2]
                and out is m[3] and lse is m[4] and attention_sinks is m[5]
                and cum_seq_lens_q is m[6] and cum_seq_lens_kv is m[7]
                and max_kv_len == m[8] and batch_size == m[9]
                and window_left == m[10] and is_causal == m[11]
                and return_lse == m[12] and bmm1_scale == m[13]
                and bmm2_scale == m[14]
                and query.shape == m[15] and key.shape == m[16]):
            fn, args, si = m[18], m[19], m[20]
            ex = m[21]
            if ex is None and not _NODRV:
                ex = _drv_capture(fn, args, si)
                m[21] = False if ex is None else ex
                if ex is not None:
                    m[22] = ctypes.c_void_p(ex)   # prebuilt exec handle
            if ex is not None and ex is not False:
                if _DRV["launch"](m[22],
                                  ctypes.c_void_p(_current_stream_ptr())) == 0:
                    return m[17]
                m[21] = False   # failed launch: distrust the exec, go raw once
            a = list(args)
            a[si] = _current_stream_ptr()   # raw fallback honors current stream
            rc = fn(*a)
            if rc != 0:
                raise RuntimeError(f"mla memo launch failure: cuda error {rc}")
            return m[17]
        # Second level: id-keyed dict -> captured torch graph replay.
        fk = (id(query), id(key), id(value), id(out), id(lse),
              id(attention_sinks), id(cum_seq_lens_q), id(cum_seq_lens_kv),
              query.shape, key.shape, int(max_kv_len), int(batch_size),
              int(window_left), bool(is_causal), bool(return_lse),
              float(bmm1_scale), float(bmm2_scale))
        fhit = _FAST.get(fk) if _FAST_OK else None
        if fhit is not None and fhit[1] is not None:
            fhit[1]()
            return (out, lse) if return_lse else out
        if not _LIBS:
            _build_all()
        _LAST_GRAPH[0] = None
        _LAST_ARGS[0] = None
        engine = _pick_engine(query.shape[0], key.shape[0], int(batch_size),
                              query.shape[1], key.shape[1], int(max_kv_len),
                              bool(is_causal), max_q_len=int(max_q_len),
                              window_left=int(window_left))
        _, fn = _LIBS[engine]

        if (query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16
                or value.dtype != torch.bfloat16 or out.dtype != torch.bfloat16):
            raise TypeError("mla_expanded_bf16 candidate supports bf16 q/k/v/out only")
        if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
                and out.is_contiguous()):
            raise ValueError("mla_expanded_bf16 candidate requires contiguous q/k/v/out")
        if lse is not None and (lse.dtype != torch.float32 or not lse.is_contiguous()):
            raise ValueError("lse must be contiguous float32 when present")
        if attention_sinks is not None and (attention_sinks.dtype != torch.float32
                                            or not attention_sinks.is_contiguous()):
            raise ValueError("attention_sinks must be contiguous float32 when present")

        if engine in ("tc", "tc2", "tc2t", "tc2d", "tc2dt", "tc3",
                      "tc4", "tc4t", "tc2p", "tc2pt", "tc2s", "tc2st",
                      "tc2c", "tc2ct", "tc7", "tc7e",
                      "tc7nr", "tc7enr", "tc7enp",
                      "tc7eu", "tc7ep", "tc7f", "tc7fm",
                      "tc7eux", "tc7eus", "tc7l", "tc9", "tc9r",
                      "tc9u", "tc9g", "tc9s", "tc9b", "tc9pm", "tc9t", "tc9q", "tc9qp", "tc9t4", "tc9e", "tc9et4", "tc9t5", "tc9l", "tc9lt5",
                      "tc2pb", "tc2b", "tc2pe", "tc2pet", "tc2pf",
                      "tc2pft", "tc2e", "tc2et", "tc2x", "tc2px",
                      "tc2p14", "tc2p64", "tc2pm", "tc2pmt", "tc2em",
                      "tc2p64x", "tc2p64t2", "tc2p64t3", "tc2p64lp", "tc2p64t3lp", "tc2p64t4", "tc2p64t5",
                      "tc2emt", "tc2l2q", "tc2l2n"):
            rc = tc_launch(query, key, value, out, lse, attention_sinks,
                           cum_seq_lens_q, cum_seq_lens_kv, int(batch_size),
                           int(window_left), 1 if is_causal else 0,
                           1 if lse is not None else 0,
                           1 if attention_sinks is not None else 0,
                           float(bmm1_scale), float(bmm2_scale), int(max_kv_len),
                           _current_stream_ptr(), engine=engine)
        else:
            call_args = (
                query.data_ptr(), key.data_ptr(), value.data_ptr(), out.data_ptr(),
                lse.data_ptr() if lse is not None else None,
                attention_sinks.data_ptr() if attention_sinks is not None else None,
                cum_seq_lens_q.data_ptr(), cum_seq_lens_kv.data_ptr(),
                query.shape[0], int(batch_size), query.shape[1], key.shape[1],
                int(window_left), 1 if is_causal else 0,
                1 if lse is not None else 0,
                1 if attention_sinks is not None else 0,
                float(bmm1_scale), float(bmm2_scale), int(max_kv_len),
                _current_stream_ptr(),
            )
            # call_args is freshly built from THIS call's tensors, so the
            # memo source is valid on hit and miss paths alike.
            _LAST_ARGS[0] = (fn, call_args, len(call_args) - 1)
            # v4.4: same cached-args + graph-replay fast path as tc (helps the
            # sync-picked small shapes where the official end-to-end protocol
            # exposes per-call CPU time). Key includes the engine: sync and td
            # share the 20-arg ABI, so identical tensors could otherwise
            # replay the wrong engine's cached args/graph.
            skey = (engine, call_args)
            hit = _SYNC_CACHE.get(skey)
            if hit is not None:
                if hit[1] is not None:
                    hit[1].replay()
                    rc = 0
                elif _GRAPHS_OK:
                    try:
                        hit[1] = _capture_graph(fn, call_args, -1)
                        hit[1].replay()
                        rc = 0
                    except Exception:
                        _GRAPHS_OK = False
                        rc = fn(*call_args)
                else:
                    rc = fn(*call_args)
            else:
                if len(_SYNC_CACHE) > 64:
                    _SYNC_CACHE.clear()
                _SYNC_CACHE[skey] = [call_args, None]
                rc = fn(*call_args)
        if rc != 0:
            raise RuntimeError(f"mla_{engine}_forward launch failure: cuda error {rc}")
        # Register (call 1) / promote to graph replay (call 2, when the
        # engine captured one) the forward fast-path entry.
        if not _FAST_OK:
            pass
        elif fhit is None:
            if len(_FAST) > 32:
                _FAST.clear()
            _FAST[fk] = [(query, key, value, out, lse, attention_sinks,
                          cum_seq_lens_q, cum_seq_lens_kv), None]
            la = _LAST_ARGS[0] if _MEMO_OK else None
            if la is not None:
                # v5 memo slot: pins + scalar guards + prebuilt launch.
                slot = [query, key, value, out, lse, attention_sinks,
                        cum_seq_lens_q, cum_seq_lens_kv,
                        max_kv_len, batch_size, window_left, is_causal,
                        return_lse, bmm1_scale, bmm2_scale,
                        query.shape, key.shape,
                        (out, lse) if return_lse else out,
                        la[0], la[1], la[2], None, None]
                _MEMO[:] = [slot]
        elif _LAST_GRAPH[0] is not None:
            fhit[1] = _LAST_GRAPH[0].replay
        return (out, lse) if return_lse else out
