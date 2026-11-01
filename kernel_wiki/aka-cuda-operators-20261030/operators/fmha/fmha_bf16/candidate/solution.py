"""Independent FMHA BF16 candidate for L20D (SM103).

Numerical work: custom CUDA C++/PTX kernel in fmha_kernel.cu (warp-level
mma.sync bf16, cp.async paged-KV gather, flash-attention online softmax,
GQA/MQA M-packing, causal + left-window + sinks, base-2 LSE, split-KV
flash-decoding with combine). Python here only inspects public metadata,
plans launch geometry from tensor shapes (no device sync), compiles/loads
the shared object and binds the ABI. No library or torch numerical calls.
"""
import ctypes
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

LOG2E = 1.4426950408889634


def q_stride_key(query):
    return query.stride() if query.ndim == 3 else (query.stride(), query.ndim)

# Mirrors struct FmhaParams in fmha_kernel.cu exactly (checked at load time).
_FMT = "<12Q12q24i4f"
_NFIELDS = 52   # 12 ptr + 12 i64 + 24 i32 + 4 f32 (unused; the ABI check is fmha_params_bytes vs calcsize)


def _compile(name, arch_flag, tag, defs=()):
    src = Path(__file__).resolve().with_name(name)
    code = (src.read_bytes() + src.with_name("fmha_params.h").read_bytes()
            + " ".join(sorted(defs)).encode())
    digest = hashlib.sha256(code).hexdigest()[:16]
    base = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or tempfile.gettempdir()) / "fmha_bf16_build"
    base.mkdir(parents=True, exist_ok=True)
    lib = base / f"{name[:-3]}_{tag}_{digest}.so"
    if not lib.exists():
        work = base / f"build_{name[:-3]}_{tag}_{digest}_{os.getpid()}"
        work.mkdir(parents=True, exist_ok=True)
        target = work / "lib.so"
        nvcc = os.environ.get("CUDA_NVCC", "/usr/local/cuda/bin/nvcc")
        cmd = [nvcc, "-O3", "-std=c++17", "--use_fast_math", "-shared",
               "-Xcompiler", "-fPIC", "-Xptxas", "-v", arch_flag, *defs,
               str(src), "-o", str(target)]
        cenv = {k: v for k, v in os.environ.items() if k not in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')}
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=280, env=cenv)
        (work / "build.log").write_text("CMD: " + " ".join(cmd) + "\n\n" + proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError("nvcc build failed:\n" + (proc.stdout + proc.stderr)[-4000:])
        os.replace(target, lib)
    return ctypes.CDLL(str(lib))


def _loader_defs():
    """A/B knobs: TMA = bulk-tensor KV loads (tc, d128/d256 NT64);
    V3 = pages-in-smem loader (REJECTED x0.88-0.92); V2 = shfl variant
    (REJECTED x0.73-0.87). Kept for the record."""
    defs = []
    # A/B control for the mask-branch fix. FMHA_MASK_PRED=1 restores the old
    # single-loop form, in which ptxas if-converts the causal/window mask into
    # per-element predicates that still ISSUE (predicated off) on interior
    # stages: the c4 SASS census measured ISETP 15.9% + FSEL 15.4% + VIADD 7.9%
    # = 39% of executed instructions with only 60% of thread slots predicated on.
    if os.environ.get("FMHA_MASK_PRED", "0") == "1":
        defs.append("-DFMHA_MASK_PRED")
    # Row-major cp.async KV loader: one page-table lookup + 64-bit stride chain
    # per 128B row instead of per 16B unit (see tc_load_kv_rm). Matters on the
    # issue-bound d64 prefill path, which does not use TMA by default.
    if os.environ.get("FMHA_LOADER_ROW", "0") == "1":
        defs.append("-DFMHA_LOADER_ROW")
    # FMHA_CHAIN1 selects the single-accumulator row max/sum chains. DEFAULT ON:
    # 4 partial accumulators (FMHA_CHAIN1=0) measured x0.994 WORSE overall (job
    # dv_889aec8778a9, same-job arms: c6 572.2 serial vs 581.2 four-way, c5
    # 18.1 vs 18.7, c8 17.9 vs 18.4, against small wins on c1 860.1 vs 855.5 and
    # c2 53.9 vs 52.9). The dependency chain is only HC=32 long on the WS tiles,
    # so the extra live registers cost more than the latency they hide. The max
    # is bitwise identical either way; the sum differs by fp32 reassociation.
    if os.environ.get("FMHA_CHAIN1", "1") == "1":
        defs.append("-DFMHA_CHAIN1")
    # FMHA_EXP_FFMA2: packed f32x2 arg-FMA/SUB in the exp loops (Blackwell
    # FFMA2, mechanism per the mounted cudnn_fe_oss sm100 reference).
    # Bitwise identical (per-lane RN/FTZ semantics == scalar fast-math
    # lowering; accumulation orders untouched). Targets the issue-bound
    # softmax tail (probe dv_49a02c9c9fa9: exp_pack 437 cyc/stage of the
    # ~2923 period; NCU history: IPC 2.07, SM 44%).
    # ADOPTED DEFAULT ON - own same-job A/B (dv_7f1d77e885cc main8ws-only,
    # dv_9e891a7cd0fb full suite): bitwise identical on all 10 fixed cases,
    # c0 x0.972, c1 x0.978, c4 x0.984, c6 x0.984, c7 x0.988, c9 x0.981
    # (min-statistic; its median rode the mode lottery), decode nulls.
    # c8's x1.109 median in dv_9e891a7cd0fb is the known 16.4/18.4 bimodal
    # draw on identical dispatch (min 16.38 vs 16.13), not an effect.
    # FMHA_EXP_FFMA2=0 restores the scalar loops for A/B.
    if os.environ.get("FMHA_EXP_FFMA2", "1") != "0":
        defs.append("-DFMHA_EXP_FFMA2")
    # PACK2 phase-2 (post-FFMA2 issue-slot reduction; screened separately):
    # RMAX = 4-way tree for the CHAIN1 serial local row-max chain (bitwise) -
    # REJECTED x1.0016 (dv_8e24f2621097: c1 x0.995 but c6/c7 x1.004-1.011;
    # the local rmax chain is not the rmx-phase limiter, consistent with the
    # PAIRMAX/MAXROW exchange nulls). Default OFF, preserved for the record.
    # LSUM = fadd2-packed paired lsum accumulation in the main8ws + main exp
    # loops; MUL = fmul2-packed O rescale. Both TOLERANCE-level (lsum pairing
    # order ~1ulp; mul denormal-only divergence); checker-clean on every arm.
    # ADOPTED DEFAULT ON - own same-job A/B: lsum2 x0.9902 geomean
    # (dv_8e24f2621097), lsmul x0.9838 (dv_61e2575aabd6): c1 x0.9858,
    # c6 x0.9922, c9 x0.9799, c7/c4 x1.000 null, c0 x0.946-0.972 (MUL shifts
    # c0's median to its low mode; reproducible 2/3 rounds + partial in
    # screen-1 p2all). Set FMHA_PACK2_LSUM=0 / FMHA_PACK2_MUL=0 for A/B.
    if os.environ.get("FMHA_PACK2_RMAX", "0") != "0":
        defs.append("-DFMHA_PACK2_RMAX")
    if os.environ.get("FMHA_PACK2_LSUM", "1") != "0":
        defs.append("-DFMHA_PACK2_LSUM")
    if os.environ.get("FMHA_PACK2_MUL", "1") != "0":
        defs.append("-DFMHA_PACK2_MUL")
    # FMHA_E2E: emulated-exp2 interleaving in the main8ws PACK2_LSUM loop
    # (vendor MUFU/FMA-pipe balancing per the mounted cudnn_fe_oss sm100
    # reference, OK_FA_E2E_FREQ=16/RES=8; independent re-encoding + NaN
    # fixup). 1 = 50% of pairs emulated, 2 = 25%. Tolerance-level. Default
    # OFF pending screen verdict.
    _e2e = os.environ.get("FMHA_E2E", "0")
    if _e2e not in ("0", ""):
        defs.append("-DFMHA_E2E=" + _e2e)
    # FMHA_WS_WSYNC: __syncwarp between fence_before_sync and the wlead
    # sfree/pready mbar arrivals in main8ws. tmem_wait_ld/st are PER-LANE;
    # without the warp rendezvous lane 0 can publish the arrival while
    # another lane's TMEM access is in flight - benign in the base
    # choreography (ctrl re-waits hide the window), fatal under QKEARLY
    # (score-side nondeterministic corruption). Bitwise identical on all
    # paths. Default OFF pending the QKEARLY-revival screen verdict.
    if os.environ.get("FMHA_WS_WSYNC", "0") != "0":
        defs.append("-DFMHA_WS_WSYNC")
    # FMHA_DEC_PACK2: fmul2-packed m16 lazy O-rescale in fmha_kernel.cu
    # (d256: 64 -> 32 slots per thread per fire). Companion to the adopted
    # prefill FMHA_PACK2_MUL; tolerance-level (denormal-only divergence).
    # Default OFF pending the decode screen verdict.
    if os.environ.get("FMHA_DEC_PACK2", "0") != "0":
        defs.append("-DFMHA_DEC_PACK2")
    # FMHA_QK_LATE_ISSUE (diagnostic, default OFF): inside the QKEARLY ctrl
    # loop only - moves the QK(s+1) issue after PV(s), restoring the base
    # pipe order while keeping every other QKEARLY element (inline boxes,
    # V lead 1, wsync, single-arm V prologue). Bisects the QK-first
    # corruption: clean => the QK-before-PV pipe overlap is the disease;
    # corrupt => the inline box mechanism is.
    if os.environ.get("FMHA_QK_LATE_ISSUE", "0") != "0":
        defs.append("-DFMHA_QK_LATE_ISSUE")
    # FMHA_SCALAR_PO=1 restores the scalar part_o stores in the split epilogues
    # (A/B control for the float4 form, which must be bitwise identical).
    if os.environ.get("FMHA_SCALAR_PO", "0") == "1":
        defs.append("-DFMHA_SCALAR_PO")
    if os.environ.get("FMHA_DEC_D256W4", "0") != "0":
        defs.append("-DFMHA_DEC_D256W4")
    # FMHA_DEC_QKSHARE: d256 decode shares one QK+softmax pass between the
    # two dv-half warp groups instead of computing it twice (the ab22 phase
    # probe measured QK at 60.5% of the c2 stage). Producers publish
    # {pfrag, mn} through the QPRE-freed smem_q region behind one extra
    # mid-stage barrier; consumers replay the state recurrence bitwise.
    # Output must be bitwise identical to the unshared build.
    if os.environ.get("FMHA_DEC_QKSHARE", "0") != "0":
        defs.append("-DFMHA_DEC_QKSHARE")
    # FMHA_TC_MAXROW: the WS softmax warps read the sibling S half as well and
    # finish the row max locally, dropping the per-stage STS + 256-thread named
    # barrier + LDS exchange from the critical path. Bitwise identical (fmaxf is
    # associative and commutative, and the same 64 values are reduced).
    if os.environ.get("FMHA_TC_MAXROW", "0") != "0":
        defs.append("-DFMHA_TC_MAXROW")
    # FMHA_TC_PAIRMAX: ADOPTED DEFAULT ON (bundle with CS-NH4): pair-
    # scoped row-max exchange. Retest under the flipped c0 pacer
    # (dv_d8154267ade9): standalone on ctrl-paced c0 x0.9997 (the old
    # null, reconfirmed) and c4 x1.0000; combined with CS-NH4 (which
    # makes c0 softmax-paced) csnh4pm x0.9559 vs csnh4 x0.9863 same-job -
    # PAIRMAX is worth another ~3% on c0 exactly where the rmx exchange
    # (536 cyc/stage, probe dv_fc8d4bb9a57c) became critical. Bitwise
    # clean in all rounds. =0 restores the full-SW named barrier.
    # FMHA_TC_PAIRMAX (mechanism): pair-scoped row-max exchange. Each
    # row's max is shared by exactly NH warps (w, w+4, ...); sync only that
    # group (bar id 1+w%4, NH*32 threads) instead of the full 256/512-thread
    # named barrier, so each row group absorbs its own skew instead of the
    # slowest warp's (rmx_exch = 553 cyc/stage on c1, 536 on c0-NH4, probe
    # dv_fc8d4bb9a57c). Values/reduction order unchanged -> bitwise identical.
    if os.environ.get("FMHA_TC_PAIRMAX", "1") != "0":
        defs.append("-DFMHA_TC_PAIRMAX")
    # FMHA_PHASE=1: DIAGNOSTIC-only fmha_tc.cu build adding clock64() per-phase
    # cycle counters to fmha_tc_main8ws (experiments/phase_probe.py reads them).
    # Never on by default; the shipped token stream is unchanged without it.
    if os.environ.get("FMHA_PHASE", "0") != "0":
        defs.append("-DFMHA_PHASE")
    # FMHA_TC_PAGEPRE: DEFAULT ON since ab16 (dv_5100dd1b4a98): x0.9629
    # geomean, bitwise identical on all 10 cases x 3 check reps (c1 844.9 ->
    # 755.0, c6 575.0 -> 535.7, c7 193.0 -> 175.7, c9 112.2 -> 108.0). The WS
    # control warp resolves each stage's page-table entries ONE control
    # iteration before it issues the TMA boxes (probe dv_de0e856f253d: the
    # dependent __ldg inside tc_tma_kv made the TMA phase 1345 cyc/iter = 44%
    # of the control loop that paces c1/c7/c9). Same boxes/coords/order.
    # FMHA_TC_PAGEPRE=0 restores the blocking form for A/B.
    if os.environ.get("FMHA_TC_PAGEPRE", "1") != "0":
        defs.append("-DFMHA_TC_PAGEPRE")
    # FMHA_PGDEEP: deep (4-stage) cp.async page-ID ring for the CSPLIT
    # TMA-lead (mechanism M5 of the production reference, which pipelines
    # page-table loads 6 deep in smem via a dedicated producer warp).
    # Replaces the 1-deep PAGEPRE register hoist for the CS box path;
    # same page values, same boxes/coords/order -> bitwise class.
    if os.environ.get("FMHA_PGDEEP", "0") != "0":
        defs.append("-DFMHA_PGDEEP")
    # FMHA_IMMMAC: immediate-operand tcgen05.mma issue groups for the
    # CSPLIT MMA-lead (mechanism M1 of the production reference; SASS
    # census dv_7f06138c3a3a: the lane-0 helper loops compile to ~10 SASS
    # slots per UTCHMMA with per-instruction ELECT+BRA retry + R2UR
    # moves). Same instructions/descriptors/order -> bitwise class.
    # ADOPTED DEFAULT ON: c0 x0.9353 (31.74->29.68), c7 x0.9867 bitwise
    # (dv_e6c2289729d3); c6-TMALATE null x1.0016 (its mma_issue ~1252 ==
    # the 1280-cyc pipe service - issue-blocked on service, not slots),
    # kept for the CS shapes. =0 restores the helper-loop issue.
    if os.environ.get("FMHA_IMMMAC", "1") != "0":
        defs.append("-DFMHA_IMMMAC")
    # FMHA_LDRED: fused tcgen05.ld.red row-max S load for the NH1
    # row-split softmax (mechanism M6 of the production reference;
    # sm_103-family instruction). Tolerance class (max.f32 FTZ edge on
    # denormals vs fast-math fmaxf). Default OFF pending screen.
    # ADOPTED DEFAULT ON: c1 x0.9668, c7 x0.9733, c9 x0.9849, c6
    # x0.9983; combo with IMMMAC c1 x0.9649 / c7 x0.9477 / c9 x0.9852 /
    # c6 x0.9984, mismatched=0 and rel_l2 identical to base everywhere
    # (empirically bitwise) - dv_5510a3695015 / dv_351aa34e0b0b.
    # =0 restores the 2x tmem_ld32 + fmaxf tree.
    if os.environ.get("FMHA_LDRED", "1") != "0":
        defs.append("-DFMHA_LDRED")
    # FMHA_DH2_ORDERB: DH2 MMA-lead issue order B - per-half interleaved
    # PV0(s) -> QK0(s+1) -> PV1(s) -> QK1(s+1) instead of order A's
    # PV0,PV1,QK0,QK1. Probe (results/20261027-dh2-phase-probe.md): order
    # A's half-0 S refill queues behind pre1+PV1 (mma_pre_wait 500
    # cyc/iter); order B decouples the halves at unchanged per-half
    # order/commit coverage (bitwise class). Diagnostic A/B, default OFF.
    if os.environ.get("FMHA_DH2_ORDERB", "0") != "0":
        defs.append("-DFMHA_DH2_ORDERB")
    # FMHA_DH2_PEARLY: DH2 M2-lite early-P chunk (reference split-P p_full
    # mechanism): softmax publishes keys 0-31 (chunk c=0 stored + this
    # stage's lazy O rescale done) on a pre_early barrier; the MMA-lead
    # splits PV_h into two 2-UTCHMMA sub-steps so the pipe starts ~the
    # c=1-pack time earlier per half. The pv commit still lands after
    # BOTH pairs (the ORDER B V-gate lesson). Requires FMHA_IMMMAC;
    # mutually exclusive with FMHA_DH2_ORDERB (#error). Bitwise class.
    # Diagnostic A/B, default OFF pending screen.
    if os.environ.get("FMHA_DH2_PEARLY", "0") != "0":
        defs.append("-DFMHA_DH2_PEARLY")
    # FMHA_DH2_MMA2: DH2 v2 - a SECOND MMA-lead warp (one tcgen05 issuer per
    # half, 352 threads). tcgen05.commit tracks the issuing thread's prior
    # async ops, so pv_h/qk_h commits cover exactly warp h's chain: the
    # cross-half commit poisoning that killed ORDER B is structurally
    # impossible, and pre1/sfree1 waits leave half-0's chain entirely.
    # Probe target: mma_pre_wait 500 cyc/iter of the balanced ~1900 period.
    # Diagnostic A/B, default OFF.
    if os.environ.get("FMHA_DH2_MMA2", "0") != "0":
        defs.append("-DFMHA_DH2_MMA2")
    # FMHA_DH2_MC_SOLO: MC hang-bisection flag (cluster launch, solo boxes).
    if os.environ.get("FMHA_DH2_MC_SOLO", "0") != "0":
        defs.append("-DFMHA_DH2_MC_SOLO")
    # FMHA_IMM_D64: immediate-operand mma groups for the CS-NH1-d64 NT64
    # tile (4x QK d64nt64 + 4x PV nt64_dv64).
    # ADOPTED DEFAULT ON - own same-job A/B (dv run 20261027T235201Z, 3
    # reversed rounds x60): c4 x0.9105/0.9143/0.9110 (113.65->103.5us;
    # -16.5% vs the certified NH4 123.9), BITWISE out+lse, dsb2_repro
    # cmp 4/4 bitwise, c0 exact null (its NH4-NT128 groups were already
    # immediate). The flipped d64 tile was still issue-bound on the
    # helper path (~10 SASS slots/UTCHMMA dominate the small d64
    # instruction bodies). =0 restores the helper loops for A/B.
    if os.environ.get("FMHA_IMM_D64", "1") != "0":
        defs.append("-DFMHA_IMM_D64")
    # FMHA_PG_EARLY: move the PAGEPRE table load from the iteration end to
    # right after the iteration's box issues. The late position leaves only
    # the ~80-200 cyc PV wait between load issue and use, so an L1 miss still
    # stalls the ctrl warp's issue path (phase probe dv_...: tma 978 cyc/iter
    # even with PAGEPRE on). Same load, same values: bitwise identical.
    if os.environ.get("FMHA_PG_EARLY", "0") != "0":
        defs.append("-DFMHA_PG_EARLY")
    # FMHA_TC_TMA_LATE: reorder the WS control loop - arm+issue stage s+SB's
    # TMA boxes AFTER the PV(s)/QK(s+1) mma issues instead of before them.
    # Phase probe dv_fc8d4bb9a57c: the box-issue phase costs ~750-990 cyc/iter
    # sitting in program order ahead of every mma issue, starving the tensor
    # pipe (c1: ctrl mma_issue ~1350 = backpressure, softmax qkwait ~1474,
    # period 2923 vs ~2x768 pipe work). Same guards (K slot via sfree(s),
    # V slot via a new mbar_pv(s) wait that gates no mma), same PV-before-QK
    # commit invariant, same boxes/coords/order -> bitwise identical.
    if os.environ.get("FMHA_TC_TMA_LATE", "0") != "0":
        defs.append("-DFMHA_TC_TMA_LATE")
    # FMHA_TC_PRO2: fill both KV slots in the WS prologue and skip iteration
    # 0's step-(a) issue. ADOPTED DEFAULT ON (S11, same-job A/B on the v2
    # engine path): geomean x0.9930 over c1/c6/c7/c9 (c7 x0.9842, c6
    # x0.9922, c1 x0.9957, c9 x0.9999), bitwise identical x3 reps on
    # c1/c7/c9. The deeper cold-start prefetch was the whole of TMA_LATE's
    # replicated c6/c9 win without its c1 x1.040 late-issue loss.
    # FMHA_TC_PRO2=0 restores the single-slot prologue for A/B.
    if os.environ.get("FMHA_TC_PRO2", "1") != "0":
        defs.append("-DFMHA_TC_PRO2")
    # FMHA_CS_PRO2: give the CSPLIT dual-ctrl-warp kernel the same deep
    # prologue as the base WS kernel (fill BOTH KV slots before the loop; the
    # TMA-lead's iteration-0 step-(a) issue is skipped). The peer CSPLIT
    # kernel had no PRO2 (developed in parallel on their branch); this tree
    # has PRO2 default-on for the base kernel, so the combo is untested
    # head-to-head. Same boxes/coords/order as the peer-verified CSPLIT
    # prologue plus one earlier stage -> bitwise claim. S39 screens it.
    if os.environ.get("FMHA_CS_PRO2", "0") != "0":
        defs.append("-DFMHA_CS_PRO2")
    # FMHA_DEC_QK4: 4-way k-split QK accumulator chains for the d256 decode
    # tile (dec_probe: qk = 561 cyc/warp-stage = 39% of the c2 stage,
    # latency-bound; QKSHARE proved issue-count is not the limiter). fp32
    # reassociation at the merge - tolerance-level, not bitwise.
    if os.environ.get("FMHA_DEC_QK4", "0") != "0":
        defs.append("-DFMHA_DEC_QK4")
    if os.environ.get("FMHA_LOADER_V3") == "1":
        defs.append("-DFMHA_LOADER_V3")
    if os.environ.get("FMHA_LOADER_V2") == "1":
        defs.append("-DFMHA_LOADER_V2")
    if os.environ.get("FMHA_TMA", "1") != "0":
        defs.append("-DFMHA_TMA")
    # FMHA_TILE_FOLD (compliance, default ON; mechanism from arm-b fold-v10,
    # join 20261023 exchange, adopted after independent verification): at the
    # range-legal max corner (batch32 x q8192 x h32/kvh1 -> total_m 8,388,608;
    # m_tile 128) tiles_ub = 65,568 > the 65,535 gridDim.y cap, so EVERY tc
    # prefill launch on d64/d256 fails with invalid configuration (d128 rides
    # tc2's m_tile 256 + grid.x and is safe; decode grids are <= ~2*batch).
    # Fold tiles into grid.z when tiles_ub > 65535; kernels decode via
    # gridDim.z > splits_ub. fold==1 (every suite/domain shape) is a
    # bit-identical grid. fold>1 implies splits_ub==1 (base_blocks >= 65,536
    # >> any wave target), so the combine is immune. FMHA_TILE_FOLD=0
    # restores the legacy grid for A/B.
    if os.environ.get("FMHA_TILE_FOLD", "1") != "0":
        defs.append("-DFMHA_TILE_FOLD")
    if os.environ.get("FMHA_DEC_PIPE", "0") == "1":
        defs.append("-DFMHA_DEC_PIPE_ON")
    # FMHA_DEC_QPRE256 / FMHA_DEC_NOBAR: peer arm-b decode-stack mechanisms
    # (join 20261022; peer ships both default ON alongside PIPE). QPRE256:
    # d256 decode preloads Q fragments to registers (qf[16][4] = 64 regs),
    # removing 16 of ~40 per-warp-per-stage ldmatrix ops from the !QPRE
    # in-loop path. NOBAR: drops the redundant TMA-mode LEADING stage-loop
    # barrier (per-thread mbar_wait + the trailing barrier already prove slot
    # safety; peer NCU c2 dv_cda7a70b4949: barrier = 1.23 of 8.39 stall
    # cy/inst) and adds a one-time prologue barrier as the cross-thread
    # Q-staging completion proof for the !QPRE in-loop ldmatrix.
    # ADOPTED default ON: ab33 (dv_b9be3ba413b8) full stack same-job domain
    # c1 x0.8218, fixed c2 x0.9518; ab34 (dv_30da083b0350) qn-without-PIPE
    # matches/beats stack everywhere (c1 75.58 vs 75.74; c2 48.35 vs 48.64;
    # c5 17.89 vs 19.46 - PIPE hurts d128 short-split decode x1.088; third
    # independent PIPE rejection after V0 dv_164157 and the pre-join
    # closure). No accumulation-order change ships with this adoption.
    if os.environ.get("FMHA_DEC_QPRE256", "1") != "0":
        defs.append("-DFMHA_DEC_QPRE256")
    if os.environ.get("FMHA_DEC_NOBAR", "1") != "0":
        defs.append("-DFMHA_DEC_NOBAR")
    # FMHA_DEC_PGPRE: decode TMA loader prefetches each stage's page-table IDs
    # one iteration ahead (tid0 registers), the same fix as prefill PAGEPRE
    # (ab16 x0.9629): tma_kv_stage's dependent __ldg stalls tid0 ~300-500 cyc
    # per stage with the whole CTA waiting at the __syncthreads behind it.
    # Same pages/boxes/order -> bitwise identical.
    # MEASURED (ab21 dv_b74c9d4cce15): bitwise identical on all 10 cases x3
    # reps, but geomean x0.9974 (c2 55.8 vs 54.7 slightly WORSE; c3/c5/c8
    # ~0.5-2% better) - the decode stage period is not made of the tid0 table
    # read, unlike the WS prefill control loop. Default OFF.
    if os.environ.get("FMHA_DEC_PGPRE", "0") != "0":
        defs.append("-DFMHA_DEC_PGPRE")
    # FMHA_DEC_PHASE=1: DIAGNOSTIC-only fmha_kernel.cu build adding clock64()
    # per-phase cycle counters to m16_body (experiments/dec_probe.py reads
    # them). Never on by default.
    if os.environ.get("FMHA_DEC_PHASE", "0") != "0":
        defs.append("-DFMHA_DEC_PHASE")
    # 8-warp d128 decode blocks (NDV=2, DVH=64, MINB=2 -> 2 blocks/SM = 16
    # warps/SM): REJECTED by own A/B (job dv_..., same-job arms, fixed geomean
    # 118.8us vs 117.6us for the 4-warp build; c3 x1.122 vs x1.195, c2 x1.123
    # vs x1.143, c5 x1.121 vs x1.138 vs V0). Registers fit (115/128, no
    # spills), so the loss is real: duplicating QK across the two dv halves and
    # widening both stage barriers to 256 threads costs more than the extra
    # warps per scheduler buy. FMHA_DEC_W8=1 to re-test.
    if os.environ.get("FMHA_DEC_W8", "0") == "1":
        defs.append("-DFMHA_DEC_W8_ON")
    return tuple(defs)


def _build_ext():
    """Compile the host-side launcher as a CPython extension (optional).

    Returns (module, run, capture) or None. A compiled extension call costs
    0.02-0.04 us against 0.5-1.0 us for the equivalent ctypes call (hostbench,
    job dv_a1159241c2de), and the evaluator's measured region includes host
    launch cost because InputReplay's RNG restore synchronizes the stream.
    Any failure leaves the ctypes path in charge; numerics are untouched.
    """
    if os.environ.get("FMHA_EXT", "1") == "0":
        return None
    try:
        import sysconfig
        inc = Path(sysconfig.get_paths()["include"])
        if not (inc / "Python.h").exists():
            return None
        src = Path(__file__).resolve().with_name("fmha_ext.cu")
        if not src.exists():
            return None
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        base = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or tempfile.gettempdir()) / "fmha_bf16_build"
        base.mkdir(parents=True, exist_ok=True)
        d = base / f"ext_{digest}"
        d.mkdir(parents=True, exist_ok=True)
        so = d / "fmha_ext.so"
        if not so.exists():
            nvcc = os.environ.get("CUDA_NVCC", "/usr/local/cuda/bin/nvcc")
            tmp = d / "build_tmp.so"
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                arch = f"sm_{major}{minor}"
            else:
                arch = "sm_103"
            cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                   f"-I{inc}", "-arch=" + arch, str(src), "-o", str(tmp)]
            cenv = {k: v for k, v in os.environ.items()
                    if k not in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')}
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=280, env=cenv)
            (d / "build.log").write_text(" ".join(cmd) + "\n\n" + proc.stdout + proc.stderr)
            if proc.returncode:
                return None
            os.replace(str(tmp), str(so))
        if str(d) not in sys.path:
            sys.path.append(str(d))
        import fmha_ext
        return fmha_ext
    except Exception:  # noqa: BLE001 - the ctypes path stays authoritative
        return None


def _build_library():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
    else:
        arch = "sm_103"
    library = _compile("fmha_kernel.cu", "-arch=" + arch, arch, _loader_defs())
    library.fmha_run.argtypes = [ctypes.c_void_p] * 4
    library.fmha_run.restype = ctypes.c_int
    library.fmha_combine_only.argtypes = [ctypes.c_void_p] * 4
    library.fmha_combine_only.restype = ctypes.c_int
    library.fmha_params_bytes.argtypes = []
    library.fmha_params_bytes.restype = ctypes.c_int
    library.fmha_init.argtypes = []
    library.fmha_init.restype = ctypes.c_int
    library.fmha_is_capturing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.fmha_is_capturing.restype = ctypes.c_int
    library.fmha_side_stream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.fmha_side_stream.restype = ctypes.c_int
    library.fmha_capture_begin.argtypes = [ctypes.c_void_p]
    library.fmha_capture_begin.restype = ctypes.c_int
    library.fmha_capture_end.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    library.fmha_capture_end.restype = ctypes.c_int
    library.fmha_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.fmha_graph_launch.restype = ctypes.c_int
    if hasattr(library, "fmha_encode_kv_maps"):
        library.fmha_encode_kv_maps.argtypes = [ctypes.c_void_p] * 4 + \
            [ctypes.c_int] * 5 + [ctypes.c_longlong] * 7
        library.fmha_encode_kv_maps.restype = ctypes.c_int
    if library.fmha_params_bytes() != struct.calcsize(_FMT):
        raise RuntimeError("FmhaParams ABI mismatch between C++ and packer")
    return library


def _build_tc_library():
    """tcgen05 prefill library; optional (None on any failure -> mma.sync path)."""
    if os.environ.get("FMHA_DISABLE_TC") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (10, 0):
        return None
    try:
        # nvcc 13.1 quirk: -arch=sm_103a shorthand emitted compute_103 PTX for
        # -shared builds (tcgen05 rejected); explicit gencode keeps compute_103a.
        library = _compile("fmha_tc.cu",
                           f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
                           f"sm_{major}{minor}a", _loader_defs())
        library.fmha_tc_run.argtypes = [ctypes.c_void_p] * 4
        library.fmha_tc_run.restype = ctypes.c_int
        library.fmha_tc_params_bytes.argtypes = []
        library.fmha_tc_params_bytes.restype = ctypes.c_int
        library.fmha_tc_init.argtypes = []
        library.fmha_tc_init.restype = ctypes.c_int
        if hasattr(library, "fmha_encode_kv_maps"):
            library.fmha_encode_kv_maps.argtypes = [ctypes.c_void_p] * 4 + \
                [ctypes.c_int] * 5 + [ctypes.c_longlong] * 7
            library.fmha_encode_kv_maps.restype = ctypes.c_int
        if library.fmha_tc_params_bytes() != struct.calcsize(_FMT):
            return None
        if library.fmha_tc_init():
            print("fmha: tcgen05 init failed; using mma.sync path", file=__import__('sys').stderr)
            return None
        return library
    except Exception as exc:
        print(f"fmha: tcgen05 library unavailable ({exc}); using mma.sync path",
              file=__import__('sys').stderr)
        return None


def _build_tc2_library():
    """2-CTA cluster tcgen05 library; optional (None -> fall back)."""
    if os.environ.get("FMHA_TC2") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (10, 0):
        return None
    try:
        library = _compile("fmha_tc2.cu",
                           f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
                           f"sm_{major}{minor}a")
        library.fmha_tc2_run.argtypes = [ctypes.c_void_p] * 4
        library.fmha_tc2_run.restype = ctypes.c_int
        library.fmha_tc2_params_bytes.argtypes = []
        library.fmha_tc2_params_bytes.restype = ctypes.c_int
        library.fmha_tc2_init.argtypes = []
        library.fmha_tc2_init.restype = ctypes.c_int
        if library.fmha_tc2_params_bytes() != struct.calcsize(_FMT):
            return None
        if library.fmha_tc2_init():
            return None
        return library
    except Exception as exc:
        import sys as _sys
        print(f"fmha: tcgen05-2cta library unavailable ({exc})", file=_sys.stderr)
        return None


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._lib = _build_library()
        self._buf = bytearray(struct.calcsize(_FMT))
        self._cbuf = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
        self._cbuf_addr = ctypes.addressof(self._cbuf)
        self._ws = None
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        self._sm = sm
        self._devidx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        # Experiment knobs (unset in evaluation; planning stays shape-derived).
        self._target2 = int(os.environ.get("FMHA_TARGET2", 2 * sm))
        self._target4 = int(os.environ.get("FMHA_TARGET4", 4 * sm))
        self._target_tc = int(os.environ.get("FMHA_TARGET_TC", sm))
        # Non-causal tc prefill wave target. 2xSM ceil-fills a second wave, but
        # for a uniform (non-causal) grid that buys no makespan: the same work is
        # spread over twice the blocks, and it adds a combine launch plus
        # splits*m_tile*dv of part_o traffic. FMHA_TARGET_NC=SM disables it.
        # Non-causal split target. FMHA_TARGET_NC=148 REJECTED (ab22
        # dv_0d863d37054f): it flips c4 to splits=1 (NH4-gated-in), and c4
        # goes 128.2->156.8 (x1.22) - for the uniform non-causal d64 grid,
        # 480 short-split main<64,128,3> blocks (1.62 waves at 2 CTAs/SM,
        # fused combine) beat 216 long NH4 blocks (1.46 waves at 1 CTA/SM).
        # The splits are load-bearing for c4; do not re-litigate without a
        # 2-CTA/SM d64 NT128 kernel.
        self._target_nc = int(os.environ.get("FMHA_TARGET_NC", 2 * sm))
        # Wave target for d64 decode (m16, base_blocks >= 32). The d64 stage
        # is ~0.7x the d128 stage (dec_probe dv_fef6b0596561: 679 vs 952
        # cyc) and the fused-combine epilogue is paid per block, so the
        # optimal split count differs from d128's. ADOPTED 240: c8 x0.9477
        # (S12 dv_8395042f800e, full pass; 300->240 blocks = 1.01->0.81
        # waves), S11's ungated 240 had moved d128/c3 the wrong way
        # (x1.0173) -> dq==64 gate; controls c2 x1.0000 / c3 x1.0033.
        self._target2_d64 = int(os.environ.get("FMHA_TARGET2_D64", "240"))
        self._m16_max = int(os.environ.get("FMHA_M16_MAX", 64))
        self._chunk_floor = int(os.environ.get("FMHA_CHUNK_FLOOR", 512))
        # Uniform-work split floor in KEYS (0 = V0 behaviour). With per-tile
        # chunks, one launch mixes 2-stage and 14-stage blocks whenever the
        # batch's kv lengths differ (fixed c2 spans 4097..32769), and the wave
        # scheduler then idles: c2's main kernel moves 126 MB at 2.4 TB/s of the
        # 7.09 TB/s DRAM roof. A floor makes every block do ~the same stages.
        # Uniform-work split chunk floor in KEYS. 1280 since ab22
        # (dv_0d863d37054f): c2 54.8->52.3 (x0.954), geomean x0.9924, all
        # non-c2 cases bitwise identical; reproduced 3x (ab20 floor1280 c2
        # 51.4/53.5, ab21 pgfloor 53.0, ab22 52.3/52.0). 512 was WORSE
        # (x1.074 on c2: per-block overhead + combine grow faster than
        # parallelism), and the decode phase probe (ab22) shows c2 is
        # QK-mma-bound (60.5% of stage), not DRAM-bound, so fewer/longer
        # split blocks win on prologue+epilogue amortization.
        self._chunk_floor_keys = int(os.environ.get("FMHA_SPLIT_CHUNK_FLOOR", "1280"))
        # Hard cap on splits_ub (part_o/part_ml workspace and combine walk
        # grow with it). 64 is the historical value; 128 lets wavefill plans
        # for base_blocks<=2 decode representable (windowed combine_tile).
        self._splits_cap = int(os.environ.get("FMHA_SPLITS_CAP", "64"))
        # Wave-fill decode splits for base_blocks<=2 (d1-class: single tiny
        # batch, huge kv). Peer arm-b mechanism (join-20261022; their same-job
        # dv_46892e9f4d9d: domain d1 x1.065, others neutral by gate). Classic
        # 64 cap => 128 CTAs = 0.86 waves of the 1-CTA/SM decode main, 20 SMs
        # idle while each CTA walks a 4096-token chunk (own d1 probe
        # dv_3ec85ab2fbec: main = 89.6% of device time, plan splits=64).
        # ceil(SM/base) splits fills exactly one wave (base 2 -> 74 splits,
        # chunk 4096 -> 3584). Needs windowed combine_tile + 128 launcher
        # guard. ADOPTED on own same-job A/B (ab31 dv_575c486de8f7): domain
        # c1 x0.9346 over 3 rounds (peer independently measured x1.065);
        # fixed suite bitwise-identical (all 10, reps 2) and timing-neutral
        # (geomean x0.9990); domain non-gated c0/c3/c9 bitwise-identical.
        self._wavefill = os.environ.get("FMHA_WAVEFILL", "1") != "0"
        # FMHA_WAVEFILL_MAX: cap on the wave-fill split count for
        # base_blocks<=2 decode plans (historically hardcoded 128 on the
        # 148-SM part - 0.86 waves, deliberately under one wave). d1-class
        # (single-batch 262K-kv d256 decode) measures ~3.5 of ~7 TB/s =
        # latency-bound streams; splits == SM count fills the machine
        # exactly and shortens each block's serial chunk by 148/128.
        # >one wave is counterproductive (wave quantization x chain).
        self._wavefill_max = int(os.environ.get("FMHA_WAVEFILL_MAX", "128"))
        # Balanced per-tile decode splits (join-20261022 arm-b mechanism).
        # chunk_target > 0: device split_range gives each tile
        # min(splits_ub, ceil(L/chunk_target)) splits of ~chunk_target tokens
        # (per-tile chunk, uniform-work floor bypassed), so short tiles stop
        # being chopped into many small CTAs and unequal tiles finish
        # together. Peer same-job evidence (dv_de714deabd75, order-controlled):
        # c2 x1.035/x1.069 replicated, c3/c5/c8 neutral. DECODE ONLY
        # (m_tile 16). Default OFF pending own same-job A/B (ab30).
        self._bal_split = os.environ.get("FMHA_BAL_SPLIT", "0") != "0"
        self._tiles_est = os.environ.get("FMHA_TILES_EST", "0") != "0"
        self._tc_nt128 = os.environ.get("FMHA_TC_NT128") == "1"
        # d64 prefill stage width. 128 is V0's choice (4-warp main<64,128,3>);
        # 64 selects the warp-specialized 8+1 tile main8ws<64,64,4,2> (2 CTAs/SM,
        # 16 softmax warps/SM, TMEM 192 of 256 columns, 84608 B smem).
        self._d64_nt = int(os.environ.get("FMHA_D64_NT", "128"))
        self._d256_nt32 = os.environ.get("FMHA_D256_NT32") == "1"
        # d256 decode at NT=32 with 4 warps per block (needs the matching
        # FMHA_DEC_D256W4 compile flag, which changes NWARP and the launched
        # thread count). V0 rejected NT32 d256 at x1.17-1.41, but that arm ran
        # 8 warps -> NDV=4 -> the QK is computed 4x; 4 warps keep it at 2x and
        # the halved smem is what buys 2 CTAs/SM on a DRAM-bound kernel.
        self._dec_d256w4 = os.environ.get("FMHA_DEC_D256W4", "0") != "0"
        # Decode d256 A/B knobs: NT48 6-warp SB2 (106KB smem -> 2 CTA/SM =
        # 12 warps vs 8; NCU c2: 2.0 warps/sched, 65% no-eligible -> latency
        # bound) and m16 lazy O-rescale (_pad2 bit5; same algebra as the tc
        # lazy path: P <= 2^thresh, bf16 relative precision scale-invariant).
        self._m16_lazy = os.environ.get("FMHA_M16_LAZY", "1") != "0"
        # lazy O-rescale threshold: the rescale is skipped while the cumulative
        # (msafe-mbase) drift stays below thresh, so P <= 2^thresh exactly. The
        # algebra is unchanged (P and l share the base; bf16 relative precision
        # is scale-invariant; lse = mbase + log2(l) stays exact), it just runs
        # the O-fix TMEM ld/fma/st passes less often. Own A/B (job dv_164157,
        # same-job arms, fixed suite geomean): 2.0 -> 119.5us, 6.0 -> 118.4us,
        # 16.0 -> 118.4us, with 0 mismatches on the fixed decode cases and on
        # all 10 boundary cases (incl. peaked/tiny/zero and sink variants).
        # RETUNED 6.0 -> 16.0 on the PACK2 stack (screen dv_60f3d980618a,
        # 20261026T191138Z, 3x60, 3 reversed rounds): c9 x0.981 stable in
        # ALL rounds (median/mode-share effect - its bimodal 98.2/100.3us
        # distribution shifts to the low mode; min-statistic null), c0
        # x0.994 mixed, c1/c6/c7 x1.000 null; decode cases null (job
        # dv_5adea790f74e re-verified the same null class). lazy32 showed no
        # trend gain over 16. Checker-clean (tolerance gate) every round.
        # Overflow headroom is enormous (bf16 max 2^127, l <= 2^16 * 2^18).
        self._lazy_thresh = float(os.environ.get("FMHA_LAZY_THRESH", "16.0"))
        self._tc_w8 = 1 if os.environ.get("FMHA_TC_W8", "1") == "1" else 0
        # Phase-A QK(s+1) issue for the 8-warp TMA prefill kernels.
        # REJECTED by own A/B (job dv_061e8627cf04, same-job arms): c1 x0.911,
        # c6 x0.930, c7 x0.924, c9 x0.914 vs the V0 issue order, i.e. moving
        # QK(s+1) ahead of PV(s) costs ~8% - the tcgen05 commit is cumulative,
        # so the mbar_pv wait at stage s+1 then also covers QK(s+1), and tid0
        # stalls on the K(s+1) TMA barrier in the middle of the stage instead
        # of overlapping it with the other warps' exp/pack. Off by default.
        self._tc_early = os.environ.get("FMHA_TC_EARLY", "0") != "0"
        # Warp-specialized 8 softmax + 1 control warp prefill (_pad2 bit9, 288
        # threads, d128 NT64 TMA only): the control warp owns every TMA issue,
        # KV mbarrier wait and tcgen05.mma issue, so QK(s+1) is issued the
        # moment the softmax warps drain S(s) and runs *during* stage s's
        # exp2/pack instead of after it. This is the structural fix for the
        # phase-A experiment above failing: there the issuing thread was also a
        # softmax thread, so warp 0 stalled on the K(s+1) TMA barrier in the
        # middle of the stage.
        # Default ON (job25 dv_..., 3 arms on one binary): fixed geomean
        # 107.5 -> 103.4us = x1.040 over the promoted bytes with c1
        # 1093.0->998.0 (x1.095), c7 249.3->233.5 (x1.068), c9 166.1->157.4
        # (x1.056) and every non-eligible case inside +-0.6%; unlisted domain
        # 90.1 -> 85.0 (x1.060). 94 regs, 0 spills, still 2 CTAs/SM.
        # Correctness: fixed c1/c6/c7/c9 and boundary b0/b1/b2/b3/b4/b16/b18
        # mm=0 vs the frozen reference, with c7 and b0/b16/b18 repeated 3x (this
        # bug class is a skew race - one clean run proves nothing).
        # FMHA_TC_WS=0 restores main8.
        self._tc_ws = os.environ.get("FMHA_TC_WS", "1") != "0"
        # Same warp specialization for the d256 prefill (_pad2 bit10): TMEM
        # forces 1 CTA/SM there, i.e. 8 softmax warps per SM instead of 16, so an
        # idling tensor pipe costs even more - and it is the biggest single win
        # of cand15: fixed c6 708.4 -> 596.7us = x1.187, domain d8 34.0 -> 31.3
        # (x1.087), both mm=0. 128 regs, 0 spills (288 threads x 1 CTA allows
        # 227). Default ON; FMHA_TC_WS256=0 restores main8<256,...>.
        self._tc_ws256 = os.environ.get("FMHA_TC_WS256", "1") != "0"
        # CSPLIT (join-20261025 peer arm-a mechanism, ported with attribution;
        # dispatch bit26 - the peer's bit14 is FMHA_GRIDORDER in this tree):
        # dual control warps for the WS d128 NT64 tile - a TMA-lead (boxes/
        # arms/page prefetch) and an MMA-lead (waits + all tcgen05 issues +
        # commits), 320 threads, same smem/TMEM footprint. Turns the serial
        # single-ctrl chain (peer ex2: tma 985 + mma_issue 1346 + waits ~600
        # = the critical 2923 vs softmax 2776) into a max => softmax-bound
        # period. Peer evidence on THEIR base (no PRO2/TLSEL): bitwise clean
        # (cs1 dv_145c0b1fa0cc 14/14, cs3 dv_c61f78ba783b 17/17 incl. all 10
        # fixed cases); gated wins c1 x0.969-0.973, c9 x0.924-0.973, d2
        # x0.938, t2 x0.939-0.954, sd1 x0.966-0.979 cold, sw1 x0.982 cold,
        # sedge x0.968, t1 x0.990; loses c7 x1.084 in the partial-tail band
        # (excluded by the wave gate). Peer join fixed job vs THIS tree:
        # c9 102.4 vs 106.5us, c1 741.3 vs 747.6us.
        # ADOPTED default ON - S38 dv_611e25172e66 same-job A/B on THIS tree:
        # c9 x0.9601 (102.37 vs 106.62; 6-13x the prefill floor), c1 x0.9757
        # (739.33 vs 757.73), c4 x1.0000 (neutral - the peer join's c4 x1.015
        # traced to their missing tree-min, not CSPLIT), c7 x1.0004
        # (gate-excluded guard = path unchanged). Bitwise identical x3 reps
        # on c1/c9/c4 (out+lse, base deterministic); engine FULL 2-seed pass
        # c1/c4/c9 (csfull arm). FMHA_WS_CSPLIT=0 restores single-ctrl.
        self._tc_csplit = os.environ.get("FMHA_WS_CSPLIT", "1") != "0"
        # FMHA_CS_PARTIAL: lift the partial-tail-band exclusion in cs_wave_ok
        # (causal single-split grids of 1-2 waves). The exclusion rested on
        # peer evidence (c7 x1.084) from the NH2 column-split era, and
        # re-tested x1.0833 there (dv_9c3dc2616003). Under NH1 row-split it
        # INVERTS: the single ctrl warp is the co-pacer (probe dv_27d51ac1b73f:
        # c7 ctrl_total 2064 vs sm_total 2285 with tma_issue 633 in the mma
        # lead's path), and CSPLIT's dedicated TMA-lead removes it.
        # ADOPTED DEFAULT ON - own same-job A/B (dv_9076af5fc51f, 3x60, 3
        # reversed rounds): c7 x0.9277 IDENTICAL in all rounds (min to
        # x0.9258), c0/c3 null with pad2 unchanged, checker clean. The tail-
        # split cross-comparison predicted the flip (CS absorbed 5% of the
        # split cost under NH1: x1.036 vs x1.085). FMHA_CS_PARTIAL=0
        # restores the certified-band gate for A/B.
        self._cs_partial = os.environ.get("FMHA_CS_PARTIAL", "1") != "0"
        # FMHA_TAILSPLIT (dispatch bit30): split only the heavy tail tiles of
        # longest-block-bound causal d128 grids (see the planner block).
        # bit30 also switches unsplit (active==1) tiles to direct out/lse
        # writes in the tc epilogue with a matching combine skip - bitwise
        # identical to the combine-copy path by construction. Default OFF.
        self._tailsplit = os.environ.get("FMHA_TAILSPLIT", "0") != "0"
        # DTA (FMHA_TC_DTA, dispatch bit28): dual-group TMEM-alternating WS
        # prefill for d128 NT64. Two 256-thread softmax groups alternate the
        # two S buffers (P packs into the drained S stage), and the control
        # warp primes QK two stages ahead, so a group's softmax tail overlaps
        # the pipe work of BOTH tiles' PV/QK stream instead of serializing
        # against its own. Probe basis (dv_49a02c9c9fa9): shipped period =
        # qkwait ~1700 + tail ~1240 at ~52% pipe duty; DTA targets the
        # pipe-throughput floor (~1004 cyc/stage, 1 CTA/SM, 576 threads =
        # 2x256 softmax + dual control warps, TMEM 256/512, ~133KB smem). Mechanism per the public CUTLASS SM100
        # FMHA mainloop (two softmax groups + S/P-aliased stages + QK
        # lookahead); implementation independent. Same MMAs/values/order ->
        # bitwise identical outputs intended (PROVEN: bitwise out+lse on
        # c1/c7/c9 incl. the split path, dv_90ad968c6f58 + dv_e2b1e22e81b4,
        # multiple rounds). v1 gate: d128 only (d256 KV slots bust the
        # 1-CTA smem budget at SB=3, and SB=3 is structurally required by
        # the 2-ahead QK pipeline).
        # MEASURED AND REJECTED: v1 single-ctrl x1.56-1.69 (dv_90ad968c6f58,
        # ctrl warp serializes issue backpressure + refill); v2 dual-ctrl
        # x1.28-1.30 (dv_e2b1e22e81b4) - 16 softmax warps in ONE CTA contend
        # (barriers/TMEM ports/schedulers) past the 2-CTA equilibrium.
        # Kept compiled, default OFF (separate kernel; shipped paths
        # untouched: null controls bitwise + x1.000 in both jobs). See
        # results/20261026-dta-v1-screen.md and the phase synthesis note.
        self._tc_dta = os.environ.get("FMHA_TC_DTA", "0") != "0"
        # NH1 row-split softmax for the d128/d256 NT64 WS tiles (_pad2
        # bit29): 4 softmax warps own whole rows (32 rows x all NT columns,
        # sreg[64]) so the row max is lane-local - the 441-559 cyc/stage
        # cross-warp exchange (STS + 256-thread bar + LDS, phase probe
        # dv_a7e8cbe32362) disappears entirely, matching the vendor d128
        # softmax ownership. Trade: half the softmax warps (2x per-warp
        # ILP, same total issue slots), 2x per-warp rescale/epilogue chunk
        # latency (rare-fire under lazy16), tolerance-level lsum
        # reassociation (P values bitwise identical: same max, same
        # ffma2/ex2 path).
        # ADOPTED DEFAULT ON - own same-job A/B, 3x60, 3 reversed rounds:
        # v1 (dv_82baea629c9d, plain WS): c7 x0.9881 stable, c0/c3 null.
        # v2 (dv_95ab17a9289b, + CS/TL/d256 combos): c1 x0.9799 (identical
        # all 3 rounds), c9 x0.9785 (0.9785-0.9788), c7 x0.9773 med /
        # x0.977 min, c6 x0.9960 (all rounds); c0 x1.003 / c3 x1.000 null
        # with pad2 unchanged; checker clean everywhere (c9 out bitwise,
        # lse tolerance). FMHA_WS_NH1=0 restores the NH2 column-split.
        self._ws_nh1 = os.environ.get("FMHA_WS_NH1", "1") != "0"
        # QK-first issue order for the warp-specialized prefill (_pad2 bit11).
        # Rationale (measured, not assumed): with PV(s) issued before QK(s+1),
        # the control warp cannot queue QK(s+1) until P(s) is packed, so the
        # tensor pipe is EMPTY for the whole exp2/pack phase of every stage; two
        # CTAs/SM then lock-step through the shared pipe and c1 runs at 1240 of
        # the 2732 MACs/cycle/SM its N=64 tile can sustain (45%). Issuing
        # QK(s+1) right after the softmax warps drain S(s) fills that window.
        # Legal because P is parity double buffered and the O rescale waits
        # mbar_pv explicitly; results must be BITWISE identical (same MMAs, same
        # accumulation order), so any mismatch is a race, not a rounding change.
        # MEASURED AND REJECTED (job dv_28b824820a65, same-job arms, 6 bitwise
        # repeats per case): c1 862->1036 (x1.20 WORSE), c6 575->661, c7
        # 196->242, c9 128->151, geomean x1.071, AND bitwise mismatches on every
        # WS case - so V0's "PV(s) before QK(s+1)" invariant really is
        # load-bearing. Two independent reasons, either fatal:
        #  (1) speed - with SB=2 the K(s+1) bulk load is issued in the SAME
        #      iteration that consumes it, and the PV-first order is what gave
        #      that TMA the whole exp2/pack window to land. QK-first exposes the
        #      full TMA latency on the critical path of every stage.
        #  (2) correctness - the swap did not reproduce base bit-for-bit, i.e.
        #      some hazard in the reordered chain is real, not theoretical.
        # Kept compiled but off; do not enable without fixing (1) with a deeper
        # K pipeline (3 K slots: 32 + 3x16 + 2x16 = 112 KB -> still 2 CTAs/SM)
        # and explaining (2).
        self._tc_qkfirst = os.environ.get("FMHA_TC_QKFIRST", "0") != "0"
        # Deep-K QK-first (_pad2 bit12): issue QK(s+1) as soon as sfree(s)
        # proves S(s) is drained, and hand the K(s+SB) box to the TMA engine a
        # full iteration earlier so that wait is not on the critical path (the
        # flaw that made FMHA_TC_QKFIRST x1.071 slower). Targets the 36.2%
        # tensor-pipe duty measured on c1.
        self._tc_qkearly = os.environ.get("FMHA_TC_QKEARLY", "0") != "0"
        # Wave dephase for 2-CTA/SM warp-specialized prefill (_pad2 bits 16-23
        # = cycles>>4): the two resident CTAs lock-step through the shared
        # tensor pipe (both softmax while it idles, then queue PV+QK behind
        # each other; c1 period 2972 = softmax ~1440 + own ~768 + sibling
        # queue ~764). Delaying one wave-parity class of control warps by
        # ~half a pipe round overlaps one CTA's tensor phase with the other's
        # softmax. Pure issue delay -> bitwise identical. 0 = off.
        self._ws_dephase = int(os.environ.get("FMHA_WS_DEPHASE", "0"))
        # Grid-order knob for the tc prefill (_pad2 bit14, FMHA_GRIDORDER=2):
        # launcher swaps grid x/y so tiles walk blockIdx.x and kvh blockIdx.y
        # (kvh-major execution). Under the evaluator's per-iteration L2
        # eviction the big causal prefills re-stream every tile's KV prefix
        # (c1: ~5.0 GB of tile x kvh re-reads == its measured 755.8 us at
        # ~94% of the 7.09 TB/s roof; warm-L2 device time is 350 us). With
        # kvh-major order the concurrent window is one kv head's unique KV
        # (~8-13 MB on c1) instead of ~62 MB of mixed-kvh prefixes, so the
        # nested prefixes hit L2. Pure block->work permutation: bitwise
        # identical outputs. 0 = legacy kvh-on-x order.
        self._gridorder = int(os.environ.get("FMHA_GRIDORDER", "0"))
        # Multicast-pair prefill (FMHA_MC2, _pad2 bit15): adjacent m-tiles of
        # one batch/kvh form a 2-CTA cluster and share ONE TMA fetch per KV
        # stage (rank0 issues multicast boxes into both CTAs' smem and arms
        # both barriers). Under the v2 protocol's per-iteration L2 eviction
        # the big prefills are DRAM re-stream bound (c1 755.8us ~= ~3.6GB of
        # effective prefix re-reads at the ~4.6-5.2TB/s cold roof measured by
        # bw_probe dv_2406f2e05fae, vs ~350us warm-L2 device time); pairing
        # halves the fetches at the unchanged 2-CTA/SM mma rate. Gated to the
        # shapes where pairing is exact and simple: tc d128/d256 NT64,
        # splits==1 (combine sc layout is m_tile<=128 and pair ranges would
        # diverge per tile), window<0 (both tiles start at lo=0), tiles fit
        # the padded gridDim.y. Bitwise-identical outputs intended.
        self._mc2 = os.environ.get("FMHA_MC2", "0") != "0"
        # Selective TMA_LATE dispatch (FMHA_TC_TLSEL, _pad2 bit24): the
        # late-box-issue WS loop won d256 (c6 x0.9828 on PRO2 base, S12
        # dv_8395042f800e; x0.9761/x0.9849 pre-PRO2 S3+S10) and split-d128
        # (c9 x0.9853 S12; x0.9685/x0.9851 S3+S10), bitwise identical in
        # every check, while losing long-tile non-split d128 (c1 x1.040) -
        # hence the per-shape gate instead of a compile-default flip.
        # ADOPTED DEFAULT ON. NH4/c0 measured x1.0008 (S13 dv_2d4a8e207441,
        # neutral) and stays on the PRO2 base loop.
        self._tlsel = os.environ.get("FMHA_TC_TLSEL", "1") != "0"
        # W8 (main8) for d128 default ON: fixed c1 x1.006/c7 x1.005/c9 x1.000
        # (dv_d5782a1d88d4 _w8 arm), domain d7 x1.036/d2 x1.007
        # (dv_2b67f4e06ee0); 0 mismatches on both. Opt-out env.
        self._w8_d128 = os.environ.get("FMHA_TC_W8_D128", "1") != "0"
        # W8 d64 (main8<64,64,3>) REJECTED by A/B dv_d90992bd4ad1: c0 x1.086,
        # c4 x1.110 vs main<64,128,3> despite its 248-reg sreg[128] profile.
        # That arm kept NT=64 (twice the stages and barriers for the same work)
        # and, at 248 regs, only 1 CTA/SM = 8 warps/SM - the same warp count
        # main<64,128,3> already has. FMHA_TC_W8_D64=1 selects the untested
        # combination instead: main8<64,128,2,MINB=2>, 84736 B smem, 2 CTAs/SM
        # = 16 warps/SM, sreg[64] per thread, TMEM S128+P64+O64 = 256 columns.
        # Motivation (probe dv_740ece2abf33): the d64 tiles are the furthest
        # from the measured tcgen05 ceiling - c0 150 TFLOPS and c4 256 TFLOPS
        # against ~1600 TFLOPS reachable for the QK(N=128)/PV(N=64) mix, with
        # NCU showing 1.13-1.66 active warps/scheduler and 54-63% no-eligible.
        self._w8_d64 = os.environ.get("FMHA_TC_W8_D64", "0") != "0"
        if os.environ.get("FMHA_TILE_REV", "1") == "1":
            self._tc_w8 |= 2
        self._fuse = os.environ.get("FMHA_FUSE_COMBINE", "1") != "0"
        self._fuse_max = int(os.environ.get("FMHA_FUSE_MAX", "24576"))
        # PDL + tc-splits-fill default ON (A/B dv_57c79a00cbb8: pdl c5 x1.079
        # c2 x1.029 c9 x1.016; fill c9 x1.184; combined c9 x1.197, 0 mism).
        # TMA KV loads default ON (A/B dv_8553b28f183a: c1 x1.116 c6 x1.167
        # c7 x1.120 c9 x1.117, controls c0/c4 unchanged, 0 mismatches).
        self._tma = os.environ.get("FMHA_TMA", "1") != "0"
        # Bulk-tensor KV loads for the m16 (decode/MTP) path as well: one thread
        # issues 8-32 cp.async.bulk.tensor boxes per stage instead of NT*D/8
        # cp.async issues plus their page-table/stride chains. NCU c2 (V0):
        # l1tex 41.7%, 11.2M instructions for 1.46 TB/s of useful traffic.
        self._dec_tma = os.environ.get("FMHA_DEC_TMA", "1") != "0"
        # Bulk-tensor KV loads for the d64 NT128 tc prefill tile. Implemented
        # and boundary-validated (b6 d64 p16, b8 d64 p32 pass vs the frozen
        # reference) but REJECTED by measurement: own A/B job dv_a67... (job10,
        # same-job arms) fixed c0 63.1us with vs 62.5us without, c4 1.019 vs
        # 1.024 vs V0, i.e. ~1% slower - the d64 tile is latency bound at
        # 8 warps/SM, not loader bound. Off by default; FMHA_TC_D64_TMA=1 to
        # re-test. NOTE it needs the SB>=3 TMA prologue prefetch (stages
        # 0..SB-2), without which slot 1 is armed but never filled and the
        # kernel spins forever.
        self._tc_d64_tma = os.environ.get("FMHA_TC_D64_TMA", "0") != "0"
        # NH=4 warp-specialized d64 NT128 prefill (_pad2 bit13): 16 softmax
        # warps + 1 control warp (544 threads, 1 CTA/SM - TMEM S 128 + P 2x64
        # + O 64 = 320 -> 512 columns). The d64 tiles are the furthest from
        # the measured tcgen05 ceiling (c0 150 / c4 256 TFLOPS vs ~1600
        # reachable; probe dv_740ece2abf33: 1.13-1.66 active warps/scheduler,
        # 54-63% no-eligible on main<64,128,3>). NH=4 keeps HC=32 columns per
        # group (the proven d128-NT64 per-thread shape) where NH=2 at NT=128
        # would need sreg[64] and bust every 2-CTA register budget.
        # DEFAULT ON since ab19 (dv_a1727089fd16) with the nh4_ok host gate
        # below: x0.9813 geomean, c0 41.8->35.5 (x0.849), all other cases
        # bitwise-identical (gate sends splits>1 / deep grids to the exact
        # base path); range_check 10/10 under the UNGATED kernel (ab18
        # dv_2a9db7499dc1), so the gated subset is tolerance-covered.
        # Ungated NH4 loses on c4 (x1.143, 432 blocks, splits_ub=2) - the
        # gate is load-bearing. FMHA_TC_WSNH4=0 restores main<64,128,3>.
        self._tc_wsnh4 = os.environ.get("FMHA_TC_WSNH4", "1") != "0"
        # FMHA_CS_NH4 (pad2 bit26 on the NH4 d64 NT128 tile): CSPLIT-NH4
        # dual control warps - peer arm-a mechanism (join-20261027
        # exchange). ADOPTED DEFAULT ON: c0 x0.9394 stable 3/3 bitwise
        # (dv_71e9da562cb8, 33.79->31.74us); reconfirmed dv_e7f9c1fa1394 /
        # dv_d8154267ade9; c4/c3 nulls. CS_PRO2 CANCELS the win
        # (dv_e7f9c1fa1394: x0.9990) - keep the shallow CS prologue.
        # =0 restores the single-ctrl NH4 for A/B.
        self._cs_nh4 = os.environ.get("FMHA_CS_NH4", "1") != "0"
        # FMHA_CS_NH1_D64 (bit26+bit29 on d64 with n_tile flipped to 64):
        # CSPLIT + NH1 row-split softmax for the d64 tile. Probe (run
        # 20261027T231538Z): c0's NH4-NT128 tile runs at ~24% pipe duty -
        # softmax-bound (exp_pack 682 under 16-warp slot contention +
        # PAIRMAX rmx 446) vs 512-cyc service. NH1 row-split (no exchange,
        # LDRED, 4+4 warps, 2 CTA/SM at NT64) trades pipe rate (moot at
        # 24% duty) for a ~2x shorter softmax chain. Tolerance class
        # (row-split lsum reassociation + n_tile geometry change).
        # ADOPTED DEFAULT ON (gated): own same-job A/B (dv_9a9225d416eb,
        # 3 reversed rounds x60): c4 x0.9011/0.9012/0.9015 (123.9->111.6us
        # all-time best, fired), c0 x1.0000 EXACT (gate holds it on NH4),
        # c8 x1.0000 exact; dsb2_repro cmp c4 4/4 deterministic rel_l2
        # 6.4e-4 mm=0 vs frozen gate. Ungated screen (dv run
        # 20261027T232335Z): c0 x1.2108 regression -> the chunk>=2048-key
        # range gate is LOAD-BEARING. FMHA_CS_NH1_D64=0 restores NH4 for
        # all d64 (A/B).
        self._cs_nh1_d64 = os.environ.get("FMHA_CS_NH1_D64", "1") != "0"
        # FMHA_CS_NH1_D64NT128 (bit26+bit29 on the d64 NT128 tile): NH1
        # row-split ON the NT128 geometry (192 threads, 1 CTA/SM, sreg[128],
        # no exchange/PAIRMAX, HALF the stages of the NT64 flip). Targets
        # the SHORT-chunk class the NT64 gate rejects (c0: doubling stages
        # cost x1.21 there; NH4's exchange/contention cost is the other
        # horn). IMMMAC d64nt128 QK + nt128_dv64 PV groups already exist.
        # ADOPTED DEFAULT ON - own same-job A/B (dv run 20261028T001130Z,
        # 3 reversed rounds x60): c0 x0.9348/0.9364/0.9370 (29.7->27.8us,
        # all-time best), cmp rel_l2 2.7e-6 deterministic, c4/c3 exact
        # nulls. Long-chunk class STAYS on the NT64 flip: nt128only screen
        # (dv run 20261028T001504Z) c4 x1.20-1.22 (103.5->124-126) - the
        # chunk>=2048 gate splits the two geometries (NT64 long / NT128
        # short). FMHA_CS_NH1_D64NT128=0 restores NH4 for short-chunk d64.
        self._cs_nh1_d64nt128 = os.environ.get(
            "FMHA_CS_NH1_D64NT128", "1") != "0"
        # FMHA_WS_DH2 (dispatch bit7 + bit26 + bit29 with m_tile 256): DH2
        # dual-M-half pipeline for the d128 NT64 CSPLIT-NH1 tile (mechanism
        # M4a of the production reference, guidance L20D_GUIDANCE_20261027_48):
        # TWO independent 128-row attention halves share one K/V stream in
        # ONE CTA (320 threads, TMEM 512 = S0|S1|P0|P1|O0|O1, ~133.7KB smem,
        # 1 CTA/SM). The two softmax chains overlap the ctrl issue chain -
        # the structural play against the chain-bound duty equilibrium
        # (pipe 35-45% at ~99% peak-when-active); KV box work per MAC halves.
        # Host gate: splits_ub == 1 && >= one wave of 256-row tiles (bounds
        # ragged-tail waste).
        # ADOPTED DEFAULT ON - own same-job A/B (dv_0462e5676d4c, 3 reversed
        # rounds x 60): c1 x0.9254/0.9255/0.9254 (659.5->610.3us, all-time
        # best), c7 x0.986-0.989 (143.4->141.4us); gate nulls c9 x1.0000
        # exact / c0 noise-band with pad2 binary-identical. BITWISE out+lse
        # on both fired cases; dsb2_repro cmp 6/6 (dv_88ef56a2486b) + 4/4 on
        # the skew-isolated reorder (dv_0462e5676d4c, e247d63) incl. c1
        # ragged tiles mrows 252 and 4 (fully-masked half-1).
        # FMHA_WS_DH2=0 restores the m_tile-128 CS-NH1 dispatch for A/B.
        self._ws_dh2 = os.environ.get("FMHA_WS_DH2", "1") != "0"
        # FMHA_DH2_SPLITS: extend the DH2 plan flip to SPLIT shapes (c9
        # class): splits recomputed to fill ONE wave of 1-CTA/SM blocks;
        # fmha_combine handles m_tile 256 via 128-row half-tiles. Gates:
        # splits>1 originally, tiles256*kvh >= 8, total_m >= 160*batch
        # (average half-1 fill must justify the fat tile). Default OFF
        # pending the c9 screen.
        self._dh2_splits = os.environ.get("FMHA_DH2_SPLITS", "0") != "0"
        # FMHA_DH2_WAVE: DH2 one-wave gate as a fraction of self._sm (in
        # percent, default 100 = tiles256*kvh >= sm). The 74-147-block band
        # (256-granularity) keeps the same wave COUNT as the 2-CTA base
        # (both grids quantize identically), so the ragged-tail argument
        # for the gate may be weaker than assumed - screening shapes via
        # range_cmp --time before any default change.
        self._dh2_wave = int(os.environ.get("FMHA_DH2_WAVE", "100"))
        # FMHA_DH2_SB: DH2 KV ring depth (2 = shipped, 3/4 = deeper box
        # prefetch; dispatch bit8/bit25, smem 166.5/199.3KB, still 1
        # CTA/SM). S/P parity stays 2-deep. Probe: mma_kv_wait 110
        # cyc/iter + tma_pv_wait slack - depth targets DRAM latency
        # jitter on the c1/d7-class cold re-streams. Screen before adopt.
        self._dh2_sb = int(os.environ.get("FMHA_DH2_SB", "2"))
        # FMHA_DH2_MC2 (bit15 under DH2): cluster-multicast pairs of
        # adjacent 256-row DH2 tiles - ONE K/V TMA fetch serves 512 rows,
        # halving SM<-L2 KV traffic on top of DH2's per-tile halving. The
        # second level of the reuse hierarchy (base: 2 CTAs re-stream the
        # prefix; DH2: one stream per 256-row tile; MC-DH2: one L2 read per
        # 512 rows). Distinct from the pre-join MC2 closures (those were
        # DH1 128-row pairs at 2 CTA/SM; DH2's 1-CTA/SM 256-row regime is a
        # new traffic geometry). Default OFF pending screen.
        self._dh2_mc2 = os.environ.get("FMHA_DH2_MC2", "0") != "0"
        # FMHA_CS_D256 (bit26 under the bit10 d256 WS gate): CSPLIT dual
        # control warps for the d256 NH1 tile. Probe (run 20261027T222148Z):
        # c6's TMALATE single ctrl chain IS the period - tma_issue 744 +
        # mma_issue 1252 (service-blocked at the 1280-cyc pipe floor) +
        # waits 320 ~ 2316 cyc/stage vs softmax 2449 (qkwait 1886); pipe
        # duty ~52%. Splitting the box issue into its own warp turns the
        # serial sum into a max (~1500 expected; measured ~1810).
        # ADOPTED DEFAULT ON - own same-job A/B (run 20261027T222937Z,
        # 3 reversed rounds x60): c6 x0.8512/0.8512/0.8514
        # (508.9->433.2us, all-time best), BITWISE out+lse; dsb2_repro
        # cmp 4/4 bitwise; c2 null (decode dispatch, pad2 identical).
        # FMHA_CS_D256=0 restores the TMALATE single-ctrl d256 path.
        self._cs_d256 = os.environ.get("FMHA_CS_D256", "1") != "0"
        # Host launch path v2: (a) the tc library launches the split combine
        # itself through a registered function pointer, so a split tc forward is
        # ONE ctypes call instead of two; (b) the fast entry caches the device
        # index, avoiding a torch.device construction per call; (c) module-level
        # Model.__call__ = Model.forward skips nn.Module's two wrapper frames and
        # its torch._C._get_tracing_state() call. The evaluator times the host
        # path inside the measured region (~10us of a 24us decode case).
        self._hostv2 = os.environ.get("FMHA_HOSTV2", "1") != "0"
        self._pdl = os.environ.get("FMHA_PDL", "1") != "0"
        self._tc_fill = os.environ.get("FMHA_TC_SPLITS_FILL", "1") != "0"
        self._graph = os.environ.get("FMHA_GRAPH", "0") == "1"
        self._gcache = {}
        self._graph_ok = True
        self._cstat = ctypes.c_int(0)
        self._cnt = None
        self._tag = os.environ.get("FMHA_TAG", "")
        rc = self._lib.fmha_init()
        if rc:
            raise RuntimeError(f"fmha_init failed rc={rc}")
        self._tc = _build_tc_library()
        self._tc2 = _build_tc2_library()
        # One host call for main+combine when the tc library supports it.
        self._c_in_c = False
        if self._hostv2 and self._tc is not None \
                and hasattr(self._tc, "fmha_tc_set_combine"):
            try:
                self._tc.fmha_tc_set_combine.argtypes = [ctypes.c_void_p]
                self._tc.fmha_tc_set_combine.restype = ctypes.c_int
                addr = ctypes.cast(self._lib.fmha_combine_only,
                                   ctypes.c_void_p).value
                if addr and not self._tc.fmha_tc_set_combine(addr):
                    self._c_in_c = True
            except Exception:  # noqa: BLE001 - keep the two-call path
                self._c_in_c = False
        self._raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        if self._raw_stream is None:
            self._raw_stream = lambda idx: torch.cuda.current_stream().cuda_stream
        self._plan_cache = {}
        self._dispatch_cache = {}
        self._fast_cache = {}
        # Host-side launcher extension (see _build_ext): replaces the ctypes
        # marshalling on the hot path and can replay a captured graph. Off by
        # default until measured; None on any build/import failure, which leaves
        # the ctypes tiers below in charge.
        # Default ON (job dv_bed95451dc03, same-job arms, evaluator-faithful
        # timing): ext x0.9914 and ext+graph x0.9868 geomean vs the ctypes path,
        # bitwise identical on all 10 fixed cases (this is a launch-path change
        # only). FMHA_EXT=0 restores the ctypes tiers, which also remain the
        # automatic fallback whenever the extension cannot be built.
        self._ext = _build_ext()
        self._ext_run = self._ext.run if self._ext is not None else None
        # FMHA_EXT_GRAPH: tier-0 CUDA-graph replay of the ext launch chain.
        # DEFAULT FLIPPED OFF (was "1") - the all-direct policy measured a
        # reproducible win (dv_63b3bdcaacbf, 3x60, 3 rounds, BITWISE on all
        # cases; cross-job value replication with dv_9edbc5158f89/
        # dv_a8648c1d96ea): c3 x0.9225-0.9564 (22.67-23.50 vs 24.58), c5
        # x0.9444 all rounds (17.41 vs 18.43), c9 x0.9894 all rounds (95.23
        # vs 96.26), c4 x0.9919 all rounds (123.92 vs 124.93), c0 low-mode
        # pinned (32.80-32.94), c8 16.38-16.46 (== base's best draw; the
        # "graph wins c8" reading was a mode artifact), c7/c2 within mode
        # band. Under this engine (eager + restore + 512MB L2-evict per
        # call) cudaGraphLaunch is ~1-2us SLOWER than the direct ext chain
        # for most chains, and the call-4 capture occasionally lands in a
        # degraded state (c8's 18.4us attractor). FMHA_EXT_GRAPH=1 restores
        # replay for A/B; FMHA_GRAPH_SMART (below) retains the per-shape
        # gating experiment.
        self._ext_graph = self._ext is not None and \
            os.environ.get("FMHA_EXT_GRAPH", "0") != "0"
        # FMHA_GRAPH_SMART: capture graphs ONLY for the chains that measure
        # faster under replay, and capture at the FIRST direct call.
        # Evidence (dv_9edbc5158f89 + dv_a8648c1d96ea, stable across rounds):
        # - multi-launch decode (main+combine, UNfused: c8 4 splits, c2 37):
        #   graph wins ~1us (c8 16.48 vs 17.41 direct; c2 47.25 vs 48.13) -
        #   one cudaGraphLaunch replaces two launches incl. the PDL setup.
        # - FUSED single-kernel chains (c3, c8-class fuse16) and single-
        #   kernel prefill (c0 NH4, c7 CS, c5-fused...): the graph launch is
        #   ~1us SLOWER than direct (c3 24.58->23.55, c0 34.78->33.79,
        #   c5... c7 within mode noise) -> keep direct.
        # - c8's high mode (18.4) appears when the call-4 capture lands in a
        #   bad state; call-1 capture pins the low mode (16.48 both rounds).
        # Default OFF pending the confirmation screen.
        self._graph_smart = os.environ.get("FMHA_GRAPH_SMART", "0") != "0"
        self._graph_after = int(os.environ.get(
            "FMHA_EXT_GRAPH_AFTER", "1" if self._graph_smart else "4"))
        self._slot_cache = {}
        self._used_tc = 0

    def _workspace(self, floats, device):
        if floats == 0:
            return None
        if self._ws is None or self._ws.numel() < floats or self._ws.device != device:
            self._ws = torch.empty(floats, dtype=torch.float32, device=device)
        return self._ws

    def _plan(self, total_q, batch, group, kvh, dq, dv, max_kv_len, causal,
              window_left=-1, max_q_len=1):
        key = (total_q, batch, group, kvh, dq, dv, max_kv_len, causal,
               window_left, max_q_len)
        plan = self._plan_cache.get(key)
        if plan is None:
            total_m = total_q * group
            use_tc = (self._tc is not None and dq == dv and dq in (64, 128, 256)
                      and total_m > 64 * batch)
            use_tc2 = (self._tc2 is not None and dq == 128 and dv == 128
                       and total_m > 64 * batch
                       and (total_m // 256 + batch) * kvh >= self._sm)
            if use_tc2:
                use_tc = 2
                m_tile = 256
                n_tile = 128
            elif use_tc:
                use_tc = 1
                m_tile = 128
                n_tile = (self._d64_nt if dq == 64
                          else (128 if (dq == 128 and self._tc_nt128) else 64))
            else:
                m_tile = 16 if total_m <= self._m16_max * batch else 64
                if m_tile == 16:
                    # NT32 d256: REJECTED x1.17-1.41 (dv_642613337223).
                    # NT48 d256 6-warp 2-CTA: REJECTED (ab7 dv_5887b7e1f5b8:
                    # c2 x1.029 vs cand-base, d1 x1.166 despite 168 regs /
                    # 0 spills - 2-CTA decode contention, not warp starvation).
                    n_tile = 32 if (dv == 256
                                    and (self._d256_nt32 or self._dec_d256w4)) else 64
                else:
                    n_tile = 32 if (dq == 256 or dv == 256) else 64
            tiles_ub = total_m // m_tile + batch
            # tiles_ub is the grid bound: it adds `batch` slack tiles because the
            # per-batch row counts are not visible host-side (cu_seqlens_q is a
            # device tensor and reading it would sync inside the measured
            # region). For the SPLIT policy that slack is a lie - fixed c3 packs
            # 8 real tiles into 16, so base_blocks doubles and the split count
            # halves, leaving 160 of 296 CTA slots empty and the DRAM at 54% of
            # its 7.10 TB/s roof. FMHA_TILES_EST=1 plans splits from the tight
            # lower bound ceil(total_m/m_tile) instead (the grid still uses
            # tiles_ub, so coverage is unchanged).
            tiles_plan = tiles_ub
            if self._tiles_est:
                tiles_plan = max(1, (total_m + m_tile - 1) // m_tile)
            base_blocks = tiles_plan * kvh
            if m_tile == 16:
                # Decode A/B (dv_6d7e02104a6c, clean FUSE-template build):
                # base>=32: target 2xSM wins (c3 x1.06 splits 10->5, c8 x1.10);
                # base<32 (c2 base16): 4xSM best (2xSM x0.92, 6xSM x1.01).
                if base_blocks < 32:
                    target = self._target4
                else:
                    target = self._target2_d64 if dq == 64 else self._target2
            elif use_tc:
                # Own A/B (job dv_6c99ef-era sweep, arm ttc296 on V0 bytes):
                # non-causal c4 222.3 -> 201.0us (x1.106) when the wave target
                # is 2xSM and `want` rounds UP, while causal c0 (66.3->89.8) and
                # c9 (171->219) regress with the same setting. The difference is
                # structural: causal tiles have a 1:T work gradient and chunk
                # rounding to n_tile leaves light tiles unsplit, so extra splits
                # only add part_o traffic and a combine launch; non-causal tiles
                # are uniform per batch, so ceil-filling one 2xSM wave balances
                # unequal batch lengths (c4: kv 4096 vs 8192 -> 2:1 blocks).
                target = self._target_tc if causal else self._target_nc
            else:
                target = self._target2
            splits_ub = 1
            if base_blocks < target:
                cap = max(1, max_kv_len // (2 * n_tile))
                cap2 = max(1, max_kv_len // self._chunk_floor)
                if use_tc and self._tc_fill and causal:
                    # tc d128 fits 2 blocks/SM: round splits to fill one
                    # 2xSM wave (c9: ceil(148/72)=3 -> 216/296 slots = 73%;
                    # round(296/72)=4 -> 288 = 97%). base>=target gate keeps
                    # c0/c7 at splits=1. Causal only - see `target` above.
                    want = (2 * self._sm + base_blocks // 2) // base_blocks
                else:
                    want = (target + base_blocks - 1) // base_blocks
                # Wave-fill decode splits (see _wavefill knob): only fires for
                # base_blocks<=2 m_tile-16 non-tc plans; every other shape
                # keeps eff_cap = splits_cap (64) and is plan-identical.
                eff_cap = self._splits_cap
                if (self._wavefill and m_tile == 16 and base_blocks <= 2
                        and not use_tc):
                    fill = (self._sm + base_blocks - 1) // base_blocks
                    eff_cap = min(self._wavefill_max, max(eff_cap, fill))
                splits_ub = min(want, cap, cap2, eff_cap)
            # CS-NH1-d64 range gate (FMHA_CS_NH1_D64): flip the d64 tile to
            # NT64 NH1 row-split ONLY when each block's expected chunk is
            # long enough to amortize the doubled stage count (NT64 = 2x the
            # stages of NT128 with ~fixed per-stage overhead). Screen
            # (run 20261027T232335Z): c0 (avg causal chunk ~1024 keys = 16
            # NT64 stages) REGRESSED x1.21; c4 (chunks 2048-4096 = 32-64
            # stages) WON x0.9025. Threshold 32 stages = 2048 keys;
            # windowed shapes are excluded (tiny per-row ranges, cf. the
            # c3/est128 splits lesson); tail-split is dq==128-only so it
            # cannot interact. n_tile does not affect tiles_ub
            # (m_tile unchanged), so only the split caps need a recompute.
            # INVARIANT: when this gate passes, splits_ub is unchanged by
            # the recompute - chunk_est >= 2048 implies cap = max_kv//256
            # >= 8 x splits and cap2 = max_kv//512 >= 4 x eff/chunk... so
            # the pre-flip splits was want- or eff_cap-bound, and both are
            # n_tile-invariant. ws_floats/combine_z (computed earlier from
            # splits_ub) therefore stay consistent without a redo.
            if (self._cs_nh1_d64 and self._ws_nh1 and self._tc_ws
                    and self._tc_csplit and not self._tc_early
                    and use_tc == 1 and dq == 64 and dv == 64
                    and n_tile == 128 and int(window_left) < 0):
                eff_range = max_kv_len - (max_q_len // 2 if causal else 0)
                chunk_est = eff_range // max(1, splits_ub)
                if chunk_est // 64 >= 32:
                    n_tile = 64
                    cap = max(1, max_kv_len // (2 * n_tile))
                    cap2 = max(1, max_kv_len // self._chunk_floor)
                    want = ((target + base_blocks - 1) // base_blocks
                            if base_blocks < target else 1)
                    splits_ub = min(want, cap, cap2, self._splits_cap)
            # TAIL-SPLIT (FMHA_TAILSPLIT, dispatch bit30): causal single-split
            # tc-d128 grids whose makespan is LONGEST-BLOCK bound (the causal
            # stage gradient leaves the machine work-bound-short: c7 520
            # blocks = 1.76 waves, longest tile 128 stages ~ 158us vs the
            # 139us work bound == the measured wall). Split ONLY the heavy
            # tail tiles by raising chunk_floor to ~3/4 of the longest tile
            # range: light tiles keep active==1 and (via bit30) write out/lse
            # directly, so part_o/combine traffic stays at c9-class levels
            # (~16MB for c7) instead of the ~48-59MB a uniform splits=2 plan
            # would add. Fires only when longest x capacity > total work
            # (c7 yes; c1 work-bound x4 over; c6 work-bound at 1 CTA/SM;
            # c9 already split; decode/d64/d256 out of scope). batch==1 keeps
            # the host-side geometry estimate exact enough for the gate.
            tail_split = False
            if (self._tailsplit and splits_ub == 1 and use_tc == 1 and causal
                    and batch == 1 and dq == 128 and dv == 128
                    and m_tile == 128 and n_tile == 64
                    and self._tc_ws and not self._tc_early):
                toks_tile = max(1, m_tile // max(1, group))
                tiles_est = max(1, (total_q + toks_tile - 1) // toks_tile)
                lo = max(0, max_kv_len - total_q)
                cap_blocks = 2 * self._sm
                tot_st = 0
                for i in range(1, tiles_est + 1):
                    tot_st += (lo + i * toks_tile + n_tile - 1) // n_tile
                max_st = (lo + tiles_est * toks_tile + n_tile - 1) // n_tile
                if max_st * cap_blocks > tot_st * kvh:
                    max_l = lo + tiles_est * toks_tile
                    ts_floor = ((max_l * 3 // 4) + n_tile - 1) // n_tile * n_tile
                    if ts_floor > n_tile:
                        tail_split = True
                        splits_ub = 2
            spad = (splits_ub + 3) & ~3
            ws_floats = (tiles_ub * kvh * (splits_ub * m_tile * dv + m_tile * 2 * spad)
                         if splits_ub > 1 else 0)
            combine_z = 1
            if splits_ub > 1:
                nch_max = (m_tile * dv // 4 + 255) // 256  # float4 channels (device MAXI=8)
                z_par = (2 * self._sm + tiles_ub * kvh - 1) // (tiles_ub * kvh)
                combine_z = min(64, max((nch_max + 7) // 8, min(z_par, nch_max)))
            if use_tc == 2 and splits_ub > 1:
                # combine_tile sc layout supports m_tile<=128; use 1-CTA tc path
                use_tc = 1 if self._tc is not None else 0
                m_tile, n_tile = (128, 64) if use_tc else (
                    16 if total_m <= self._m16_max * batch else 64,
                    64 if (not use_tc and total_m <= self._m16_max * batch)
                    else (32 if (dq == 256 or dv == 256) else 64))
                tiles_ub = total_m // m_tile + batch
                base_blocks = tiles_ub * kvh
                splits_ub = 1
                if base_blocks < target:
                    cap = max(1, max_kv_len // (2 * n_tile))
                    cap2 = max(1, max_kv_len // self._chunk_floor)
                    splits_ub = min((target + base_blocks - 1) // base_blocks,
                                    cap, cap2, self._splits_cap)
                ws_floats = (tiles_ub * kvh * (splits_ub * m_tile * dv
                           + m_tile * 2 * ((splits_ub + 3) & ~3))
                           if splits_ub > 1 else 0)
                combine_z = 1
                if splits_ub > 1:
                    nch_max = (m_tile * dv // 4 + 255) // 256  # float4 channels (device MAXI=8)
                    z_par = (2 * self._sm + tiles_ub * kvh - 1) // (tiles_ub * kvh)
                    combine_z = min(64, max((nch_max + 7) // 8, min(z_par, nch_max)))
            # DH2 (M4a, FMHA_WS_DH2): flip the d128 NT64 single-split CS-NH1
            # plan to the dual-M-half dispatch when at least one wave of
            # 256-row tiles exists (m_tile 256: grid/tiles_ub halve, 1 CTA/SM;
            # split shapes keep the m_tile-128 CS path by default).
            # FMHA_DH2_SPLITS extends the flip to split shapes: splits are
            # recomputed to fill ONE wave of 1-CTA/SM blocks (the base plan
            # targets 2-CTA blocks), and fmha_combine reduces m_tile-256
            # part_o in 128-row half-tiles (grid.y = tiles x halves).
            # window_left < 0 gate: with a left window, rows whose rlo sits
            # above the tile's aligned kv0 run fully-masked LEADING stages,
            # and the count of those stages is tile-geometry dependent. The
            # lazy-rescale state (mbase -> msafe=0 on an all-masked stage
            # while mbase is still -inf) then takes dispatch-dependent
            # trajectories: range sweep shape 4 measured 1-2 bf16 ulp out
            # drift + 6.4% lse ulp drift vs the m_tile-128 path (well
            # inside the frozen .01+.05*|actual| gate, but not bitwise).
            # Windowed prefill keeps the certified 128-row dispatch - the
            # same conservative gate mc2_ok uses. No fixed/domain shape
            # loses: none of the fired suite shapes has a window.
            if (self._ws_dh2 and self._tc_ws and self._ws_nh1
                    and self._tc_csplit and not self._tc_early
                    and use_tc == 1 and dq == 128 and dv == 128
                    and n_tile == 64 and not tail_split
                    and int(window_left) < 0
                    and (total_m // 256 + batch) <= 65535):
                tiles256 = total_m // 256 + batch
                if (splits_ub == 1
                        and tiles256 * kvh * 100 >= self._sm * self._dh2_wave):
                    m_tile = 256
                    tiles_ub = tiles256
                    base_blocks = tiles_ub * kvh
                    ws_floats = 0
                    combine_z = 1
                elif (self._dh2_splits and splits_ub > 1
                        and tiles256 * kvh >= 8
                        and total_m >= 160 * batch):
                    base256 = tiles256 * kvh
                    cap = max(1, max_kv_len // (2 * n_tile))
                    cap2 = max(1, max_kv_len // self._chunk_floor)
                    want = (self._sm + base256 - 1) // base256
                    splits256 = min(want, cap, cap2, self._splits_cap)
                    if splits256 >= 2:
                        m_tile = 256
                        tiles_ub = tiles256
                        base_blocks = base256
                        splits_ub = splits256
                        spad = (splits_ub + 3) & ~3
                        ws_floats = (tiles_ub * kvh
                                     * (splits_ub * m_tile * dv
                                        + m_tile * 2 * spad))
                        nch_max = (m_tile * dv // 4 + 255) // 256
                        z_par = ((2 * self._sm + tiles_ub * kvh - 1)
                                 // (tiles_ub * kvh))
                        combine_z = min(64, max((nch_max + 7) // 8,
                                                min(z_par, nch_max)))
            # Gate 24K floats: A/B fused wins c3 (20480) x1.21, c8 (10240) x1.14;
            # loses c5 (32768) x0.87, c2 (151552) x0.91 -> exclude. (Those
            # A/B losses were on the register-pressured runtime-branch build;
            # FMHA_FUSE_MAX re-tests them on the clean FUSE-template build.)
            fuse16 = 1 if (self._fuse and m_tile == 16 and splits_ub > 1
                           and splits_ub * m_tile * dv <= self._fuse_max) else 0
            # Uniform-work split floor, applied only when it cannot starve the
            # machine. Per-tile chunks are sized from each tile's own length, so
            # heterogeneous kv_lengths produce blocks whose work differs by
            # max_kv/min_kv (fixed c2: 2-stage and 14-stage blocks in the same
            # launch, 4 waves, 126 MB at 2.4 of 7.10 TB/s). A floor in keys
            # equalises them, but it also lowers the block count, so gate it on
            # an upper-bound estimate of that count: the longest possible tile
            # range is max_kv_len, or window_left+1+max_q_len when a left window
            # clips it (fixed c3/c5/c8 are all windowed and must NOT be floored).
            floor_keys = self._chunk_floor_keys
            if tail_split:
                # Tail-split sets its own high floor (heavy-tiles-only) and
                # opts out of the starvation gate below: that gate sizes
                # tiles_ub*act against ONE wave of SMs for decode grids,
                # while a tail-split grid is already >= 2 waves of blocks
                # (tiles_ub*kvh >= 2*sm is exactly the band it fires in).
                floor_keys = max(self._chunk_floor_keys, ts_floor)
            elif floor_keys and splits_ub > 1:
                eff = max_kv_len if window_left < 0 else min(
                    max_kv_len, window_left + 1 + max(1, max_q_len))
                act = min(splits_ub,
                          max(1, (eff + floor_keys - 1) // floor_keys))
                if tiles_ub * act < self._sm:
                    floor_keys = 0
            # Balanced per-tile splits (decode only; see _bal_split knob).
            # chunk_target from the longest possible tile range so every
            # tile's split count scales with its own length on the device.
            chunk_target = 0
            if self._bal_split and splits_ub > 1 and m_tile == 16 and not use_tc:
                eff_kv = (max_kv_len if window_left < 0
                          else min(max_kv_len, window_left + 1))
                chunk_target = max(
                    n_tile,
                    ((eff_kv + splits_ub - 1) // splits_ub + n_tile - 1)
                    // n_tile * n_tile)
            plan = (m_tile, n_tile, tiles_ub, splits_ub, ws_floats,
                    use_tc, combine_z, fuse16, floor_keys, chunk_target,
                    tail_split)
            self._plan_cache[key] = plan
        return plan

    def _launch(self, use_tc, splits_ub, stream, buf_addr, kp=0, vp=0):
        # _used_tc records which launcher actually served this call so the
        # extension fast path registers the SAME one; the -200 fallback below is
        # ctypes-only logic and must not be baked into a slot.
        self._used_tc = 0
        if use_tc == 2 and self._tc2 is not None:
            rc = self._tc2.fmha_tc2_run(buf_addr, stream, 0, 0)
            if rc != -200:
                self._used_tc = 2
                if rc == 0 and splits_ub > 1:
                    rc = self._lib.fmha_combine_only(buf_addr, stream, 0, 0)
                return rc
            use_tc = 1
        if use_tc == 1 and self._tc is not None:
            rc = self._tc.fmha_tc_run(buf_addr, stream, kp, vp)
            if rc != -200:
                self._used_tc = 1
                if rc == 0 and splits_ub > 1 and not self._c_in_c:
                    rc = self._lib.fmha_combine_only(buf_addr, stream, 0, 0)
                return rc
        elif use_tc >= 1:
            # planned tc path unavailable (library build failed): plan is still
            # valid for the mma.sync kernels only when m_tile matches; the
            # packed m_tile=128/256 is unsupported there, so this is a hard
            # configuration error, not a silent fallback.
            raise RuntimeError("fmha: tcgen05 path planned but library unavailable")
        return self._lib.fmha_run(buf_addr, stream, kp, vp)

    def _fast_entry(self, ent, *pins):
        """Pre-resolved launch functions for the tier-1 hit path.
        ent = (addr, cbuf, buf, use_tc, splits_ub, ws, cnt); pins keep the
        caller tensors alive so tier-1 object ids cannot be recycled."""
        use_tc, splits_ub = ent[3], ent[4]
        if use_tc == 2 and self._tc2 is not None:
            fn = self._tc2.fmha_tc2_run
        elif use_tc == 1 and self._tc is not None:
            fn = self._tc.fmha_tc_run
        else:
            fn = self._lib.fmha_run
            use_tc = 0
        cfn = None
        if use_tc and splits_ub > 1 and not self._c_in_c:
            cfn = self._lib.fmha_combine_only
        # FMHA_GRAPH_SMART: >=2 kernel launches in this chain? (tc paths
        # chain combine via cfn or internally; the decode path launches it
        # inside fmha_run unless the combine is FUSED into the main kernel.)
        multi = splits_ub > 1 and (use_tc in (1, 2)
                                   or not (ent[10] if len(ent) > 10 else 0))
        return (ent[0], ent[3], ent[4], (ent,) + pins, fn, cfn,
                ent[7] if len(ent) > 7 else 0, ent[8] if len(ent) > 8 else 0,
                self._devidx, multi)

    def _try_capture(self, ent, stream, g):
        """Capture this key's launch chain into a CUDA graph on a private side
        stream (the current call already ran via direct launches above). Any
        failure permanently disables capture; direct launches continue."""
        try:
            self._lib.fmha_is_capturing(stream, ctypes.byref(self._cstat))
            if self._cstat.value != 0:
                return
            side = ctypes.c_void_p(0)
            if self._lib.fmha_side_stream(ctypes.byref(side)):
                self._graph_ok = False
                return
            if self._lib.fmha_capture_begin(side):
                self._graph_ok = False
                return
            rc = self._launch(ent[1], ent[2], side, ent[0], ent[6], ent[7])
            ex = ctypes.c_void_p(0)
            rc2 = self._lib.fmha_capture_end(side, ctypes.byref(ex))
            if rc or rc2 or not ex.value:
                self._graph_ok = False
                return
            g[1] = ex.value
        except Exception:  # noqa: BLE001 - never break the launch path
            self._graph_ok = False

    def forward(self, query, kv_cache, workspace_buffer, block_tables, seq_lens,
                max_q_len, max_kv_len, bmm1_scale, bmm2_scale, batch_size,
                cum_seq_lens_q, cum_seq_lens_kv, window_left=-1, out=None,
                out_dtype=None, o_sf_scale=None, o_sf_vec_size=None, kv_layout='HND',
                enable_pdl=None, sinks=None, kv_cache_sf=None,
                skip_softmax_threshold_scale_factor=None, uses_shared_paged_kv_idx=True,
                causal=True, lse=None, return_lse=False,
                multi_ctas_kv_counter_buffer=None, use_fp16_softmax=None,
                uses_spcompress=None, backend='trtllm-gen'):
        if out is None or kv_layout != 'HND' or o_sf_scale is not None \
                or o_sf_vec_size is not None or kv_cache_sf is not None \
                or multi_ctas_kv_counter_buffer is not None \
                or skip_softmax_threshold_scale_factor is not None \
                or use_fp16_softmax is not None or uses_spcompress is not None:
            return self._forward_slow(
                query, kv_cache, workspace_buffer, block_tables, seq_lens,
                max_q_len, max_kv_len, bmm1_scale, bmm2_scale, batch_size,
                cum_seq_lens_q, cum_seq_lens_kv, window_left, out, out_dtype,
                o_sf_scale, o_sf_vec_size, kv_layout, enable_pdl, sinks,
                kv_cache_sf, skip_softmax_threshold_scale_factor,
                uses_shared_paged_kv_idx, causal, lse, return_lse,
                multi_ctas_kv_counter_buffer, use_fp16_softmax, uses_spcompress,
                backend)
        if self._ext is not None:
            # Tier-0: one dict lookup on a scalar key, nine identity checks and
            # one compiled-extension call. The slot already holds the packed
            # params, launcher pointer and host tensormaps, and every tensor it
            # references is pinned by the entry below, so the ids cannot be
            # recycled while the entry lives. Identity checks (not pointer
            # equality of contents) keep this exact: a different out/lse/kv/
            # index tensor object, or a resized/rebased query or out, falls
            # through to the full-key tier that revalidates shapes and strides.
            ent = self._slot_cache.get(
                (id(query), query.data_ptr(), window_left, causal, return_lse,
                 max_kv_len, bmm1_scale, bmm2_scale))
            if ent is not None and ent[1] is out and ent[2] is kv_cache \
                    and ent[3] is lse and ent[4] is sinks \
                    and ent[5] is block_tables and ent[6] is seq_lens \
                    and ent[7] is cum_seq_lens_q and ent[8] is cum_seq_lens_kv:
                stream = self._raw_stream(self._devidx)
                rc = self._ext_run(ent[0], stream)
                if rc:
                    raise RuntimeError(f"fmha kernel launch failed rc={rc}")
                g = ent[10]
                if g is not None:
                    g[0] += 1
                    if g[0] == self._graph_after:
                        g[0] = -1 << 30       # capture at most once per slot
                        self._ext.capture(ent[0], stream)
                return ent[9]
        k, v = kv_cache
        # Tier-1 hot key: object ids + scalars + ptr canaries only (~2 C calls).
        # Entries PIN the referenced tensors, so ids cannot be recycled while
        # cached; any storage swap changes a canary ptr -> full-key path
        # revalidates. ALL tensor arguments that feed the packed params are
        # id-pinned (block_tables/seq_lens/cum_* included): swapping any of
        # them forces the full-key tier, which rechecks ptrs/shapes/strides.
        fkey = (id(query), id(kv_cache), id(out), id(lse), id(sinks),
                id(block_tables), id(seq_lens), id(cum_seq_lens_q),
                id(cum_seq_lens_kv),
                query.data_ptr(), k.data_ptr(), out.data_ptr(),
                return_lse, causal, max_kv_len, window_left,
                bmm1_scale, bmm2_scale)
        ent = self._fast_cache.get(fkey)
        if ent is not None:
            stream = self._raw_stream(ent[8] if len(ent) > 8 else query.device.index)
            g = self._gcache.get(fkey) if self._graph else None
            if g is not None and g[1]:
                # Replay path. A graph launch is illegal while the CALLER is
                # capturing (contract: candidate must stay capturable in
                # cuda_graph_replay mode) -> guard and fall back to the
                # direct chain, which is itself capture-safe.
                self._lib.fmha_is_capturing(stream, ctypes.byref(self._cstat))
                if self._cstat.value == 0:
                    rc = self._lib.fmha_graph_launch(g[1], stream)
                    if rc:
                        raise RuntimeError(f"fmha graph launch failed rc={rc}")
                    return (out, lse) if return_lse else out
            rc = ent[4](ent[0], stream, ent[6], ent[7])
            if rc == -200:
                rc = self._launch(ent[1], ent[2], stream, ent[0], ent[6], ent[7])
            elif rc == 0 and ent[5] is not None:
                rc = ent[5](ent[0], stream, 0, 0)
            if rc:
                raise RuntimeError(f"fmha kernel launch failed rc={rc}")
            if (self._graph and self._graph_ok
                    and (not self._graph_smart or ent[9])):
                if g is None:
                    self._gcache[fkey] = [1, 0]
                elif not g[1]:
                    g[0] += 1
                    if g[0] >= 3:
                        self._try_capture(ent, stream, g)
            return (out, lse) if return_lse else out
        key = (id(query), id(kv_cache), id(out), id(lse), id(sinks),
               query.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
               block_tables.data_ptr(), seq_lens.data_ptr(),
               cum_seq_lens_q.data_ptr(),
               lse.data_ptr() if lse is not None else 0,
               sinks.data_ptr() if sinks is not None else 0,
               query.shape, k.shape, v.shape, out.shape,
               seq_lens.shape, block_tables.shape, cum_seq_lens_q.shape,
               q_stride_key(query), k.stride(), v.stride(), out.stride(),
               block_tables.stride(0),
               lse.stride() if lse is not None else None,
               bmm1_scale, bmm2_scale, window_left, causal, return_lse,
               max_kv_len)
        ent = self._dispatch_cache.get(key)
        if ent is not None:
            self._fast_cache[fkey] = self._fast_entry(ent, query, kv_cache, out,
                                                      lse, sinks, block_tables,
                                                      seq_lens, cum_seq_lens_q,
                                                      cum_seq_lens_kv)
            rc = self._launch(ent[3], ent[4], self._raw_stream(self._devidx),
                              ent[0], ent[7], ent[8])
            if rc:
                raise RuntimeError(f"fmha kernel launch failed rc={rc}")
            return (out, lse) if return_lse else out
        q = query
        ret_out = out
        if q.ndim == 4:
            q = q.flatten(0, 1)
            out = out.flatten(0, 1)
        if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 \
                or v.dtype is not torch.bfloat16 or out.dtype is not torch.bfloat16 \
                or (out_dtype is not None and out_dtype is not out.dtype) \
                or (lse is not None and lse.dtype is not torch.float32) \
                or (sinks is not None and sinks.dtype is not torch.float32) \
                or q.stride(2) != 1 or out.stride(2) != 1 \
                or k.stride(3) != 1 or v.stride(3) != 1 \
                or (lse is not None and lse.stride(1) != 1) \
                or not block_tables.is_contiguous() or not seq_lens.is_contiguous() \
                or not cum_seq_lens_q.is_contiguous() \
                or seq_lens.dtype is not torch.int32 \
                or cum_seq_lens_q.dtype is not torch.int32 \
                or block_tables.dtype is not torch.int32:
            raise ValueError("unsupported operand dtypes/strides for the fast path")
        total_q, heads, dq = q.shape
        dv = v.shape[3]
        kvh = k.shape[1]
        page_size = k.shape[2]
        if page_size & (page_size - 1) or page_size < 1:
            raise ValueError("page_size must be a power of two")
        page_log = page_size.bit_length() - 1
        if heads % kvh or dq != k.shape[3] or out.shape != (total_q, heads, dv) \
                or (lse is not None and (lse.shape[0] != total_q or lse.shape[1] != heads)) \
                or (return_lse and lse is None):
            raise ValueError("inconsistent fmha contract shapes")
        batch = seq_lens.shape[0]
        group = heads // kvh
        max_kv = int(max_kv_len)
        (m_tile, n_tile, tiles_ub, splits_ub, ws_floats, use_tc, combine_z,
         fuse16, floor_keys, chunk_target, tail_split) = self._plan(
            total_q, batch, group, kvh, dq, dv,
            max_kv, bool(causal),
            int(window_left), int(max_q_len))
        # NH4 gate (ab18 dv_2a9db7499dc1): the 4-quarter WS d64 tile is
        # 1 CTA/SM, so it dominates the low-parallelism single-split regime
        # (c0 x0.881 at 204 grid-bound blocks; range r2/r3/r9 all faster) but
        # loses to the 2-CTA/SM main tile when the grid is deep or the split
        # path is involved (c4 x1.143 at 432 blocks, splits_ub=2). Maps stay
        # OFF for gated-out shapes so they take the exact cp.async main path
        # they take today, not the TMA main variant.
        nh4_ok = (self._tc_wsnh4 and dq == 64 and n_tile == 128
                  and splits_ub == 1 and tiles_ub * kvh <= 2 * self._sm)
        # CSPLIT-NH4 (peer arm-a mechanism, join-20261027 exchange; their
        # default ON, arm under test on {0} at join): dual-ctrl-warp split
        # for the d64 NT128 NH4 tile, attacking the measured worst serial
        # pre-mma phase in the suite (c0-NH4 tma_issue 1058 cyc/iter,
        # page16 -> 16 boxes/iter). No wave gate in the peer version: c0
        # (204 blocks = 1.38 waves at 1 CTA/SM) sits inside the
        # partial-tail band where d128-CS once lost; whether that taxonomy
        # transfers to the 1-CTA NH4 tile is exactly what the screen
        # measures (c0 AND c4). Bitwise class. Default OFF pending screen.
        cs_nh4_ok = (self._cs_nh4 and nh4_ok and not self._tc_early)
        # DH2 dual-M-half (FMHA_WS_DH2): the plan flip already encoded the
        # eligibility (m_tile 256 only ever comes from the DH2 conversion);
        # re-check the mode flags here so the pack and the plan cannot drift.
        dh2_ok = (self._ws_dh2 and self._tc_ws and self._ws_nh1
                  and self._tc_csplit and not self._tc_early
                  and use_tc == 1 and dq == 128 and dv == 128
                  and m_tile == 256 and n_tile == 64
                  and (splits_ub == 1 or self._dh2_splits))
        # Wave dephase applies only to 2-CTA/SM tc tiles: d64 NT128
        # (main/main8), d64 NT64 (WS SB4) and d128 NT64 (WS/main8/main).
        # NH4, WS NT128 and every d256 tile run 1 CTA/SM: there is no
        # sibling to overlap and the spin would be pure loss.
        dph = 0
        if (self._ws_dephase and use_tc == 1 and not nh4_ok and dq != 256
                and not dh2_ok and not (dq == 128 and n_tile == 128)):
            dph = min(self._ws_dephase, 4080) & ~15
        mc2_ok = (self._mc2 and use_tc == 1 and splits_ub == 1
                  and dq == dv and dq in (128, 256) and m_tile == 128
                  and n_tile == 64 and int(window_left) < 0
                  and tiles_ub <= 65534)
        dta_ok = (self._tc_dta and self._tc_ws and not self._tc_early
                  and not mc2_ok and use_tc == 1 and dq == 128 and dv == 128
                  and m_tile == 128 and n_tile == 64)
        # DH2-MC: multicast pairs over the m_tile-256 DH2 dispatch (bit15
        # shared with DH1-MC2 - m_tile separates the launcher branches).
        dh2mc_ok = (self._dh2_mc2 and dh2_ok and splits_ub == 1
                    and int(window_left) < 0 and tiles_ub <= 65534)
        cs_d256_ok = (self._cs_d256 and self._tc_ws and self._ws_nh1
                      and self._tc_ws256 and not self._tc_early
                      and not mc2_ok and not dta_ok and use_tc == 1
                      and dq == 256 and dv == 256
                      and m_tile == 128 and n_tile == 64)
        tl_ok = (self._tlsel and not mc2_ok and not dta_ok
                 and not cs_d256_ok and use_tc == 1
                 and self._tc_ws
                 and dq == dv and dq in (128, 256) and m_tile == 128
                 and n_tile == 64 and (dq == 256 or splits_ub > 1))
        # CSPLIT wave gate (peer arm-a cs2 dv_39cc7a645a52, ported): the
        # dual-ctrl-warp WS kernel wins when per-block stage counts are
        # uniformized (splits_ub > 1: c9 x0.924, t2 x0.939, sd1 x0.966,
        # sw1 x0.982 cold), when the grid is sub-wave so every block runs
        # concurrently with no partial tail (blocks < 2 CTA/SM x SM: sedge
        # x0.968, t1 x0.990), when it is non-causal (d2 x0.942), or when it
        # is >= 2 full waves so short-block tails amortize (c1 x0.973 at 776
        # blocks >= 592). The peer's partial-tail-band exclusion (c7 x1.084,
        # re-confirmed x1.0833 in the NH2 era, dv_9c3dc2616003) INVERTED
        # under NH1 row-split (c7 x0.9277 stable 3/3, dv_9076af5fc51f):
        # with 4 row-owning softmax warps the single ctrl warp became the
        # co-pacer (probe dv_27d51ac1b73f: c7 ctrl 2064 vs sm 2285 cyc with
        # tma_issue 633 in the mma lead's path), and the dedicated TMA-lead
        # is a pure win. The band exclusion is therefore LIFTED by default
        # (FMHA_CS_PARTIAL=1); =0 restores the certified-era gate for A/B.
        # Gated-out shapes take the exact certified dispatch. Shape terms
        # mirror the d128 NT64 WS dispatch branch that consumes the bit.
        cs_wave_ok = (splits_ub > 1 or not causal
                      or tiles_ub * kvh < 2 * self._sm
                      or tiles_ub * kvh >= 4 * self._sm
                      or self._cs_partial)
        cs_ok = (self._tc_csplit and self._tc_ws and not self._tc_early
                 and not mc2_ok and not dta_ok
                 and use_tc == 1 and m_tile == 128 and cs_wave_ok
                 and ((dq == 128 and dv == 128 and n_tile == 64)
                      or (dq == 64 and dv == 64 and self._ws_nh1
                          and ((n_tile == 64 and self._cs_nh1_d64)
                               or (n_tile == 128
                                   and self._cs_nh1_d64nt128)))))
        # NH1 row-split: d128/d256 NT64 WS shapes; combines with the TL/CS
        # bits (launcher instantiates NH1 x plain/TL/CS for d128, NH1 x
        # plain/TL for d256). v1 screen: c7 (plain WS) x0.9881 stable 3/3.
        nh1_ok = (self._ws_nh1 and self._tc_ws and not self._tc_early
                  and not mc2_ok and not dta_ok
                  and use_tc == 1 and dq == dv and m_tile == 128
                  and ((dq in (128, 256) and n_tile == 64)
                       or (dq == 64
                           and ((n_tile == 64 and self._cs_nh1_d64)
                                or (n_tile == 128
                                    and self._cs_nh1_d64nt128)))))
        if fuse16:
            need = tiles_ub * kvh
            if self._cnt is None or self._cnt.numel() < need or self._cnt.device != q.device:
                self._cnt = torch.zeros(max(need, 4096), dtype=torch.int32, device=q.device)
            cnt_ptr = self._cnt.data_ptr()
        else:
            cnt_ptr = 0
        ws = self._workspace(ws_floats, q.device)
        if splits_ub > 1:
            ws_ptr = ws.data_ptr()
            part_o = ws_ptr
            part_ml = ws_ptr + tiles_ub * kvh * splits_ub * m_tile * dv * 4
        else:
            ws = None
            part_o = 0
            part_ml = 0
        lse_ptr = lse.data_ptr() if lse is not None else 0
        sink_ptr = sinks.data_ptr() if sinks is not None else 0
        buf = bytearray(len(self._buf))
        struct.pack_into(
            _FMT, buf, 0,
            q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), lse_ptr,
            block_tables.data_ptr(), seq_lens.data_ptr(), cum_seq_lens_q.data_ptr(),
            sink_ptr, part_o, part_ml, cnt_ptr,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            lse.stride(0) if lse is not None else 0,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            sinks.stride(0) if sinks is not None else 0,
            batch, heads, kvh, group, total_q,
            dq, dv, page_log, page_size - 1, block_tables.stride(0),
            int(window_left), 1 if causal else 0,
            1 if lse is not None else 0, 1 if sinks is not None else 0,
            m_tile, n_tile, splits_ub, n_tile, tiles_ub, max_kv, combine_z,
            ((self._tc_w8 & 1)
             if (dq == 256 or (dq == 128 and self._w8_d128)
                 or (dq == 64 and self._w8_d64)) else 0)
            | (self._tc_w8 & 2)
            | (4 if fuse16 else 0)
            | (8 if self._pdl else 0)
            | (32 if self._m16_lazy else 0)
            | (128 if (self._tc_early or dh2_ok) else 0)
            | (256 if (dh2_ok and self._dh2_sb == 3) else 0)
            | ((1 << 25) if (dh2_ok and self._dh2_sb >= 4) else 0)
            | (512 if (self._tc_ws and not self._tc_early) else 0)
            | (1024 if (self._tc_ws256 and not self._tc_early) else 0)
            | (2048 if (self._tc_qkfirst and not self._tc_early) else 0)
            | (4096 if (self._tc_qkearly and not self._tc_early) else 0)
            | (8192 if (nh4_ok and not self._tc_early) else 0)
            | (16384 if self._gridorder == 2 else 0)
            | (32768 if (mc2_ok or dh2mc_ok) else 0)
            | ((1 << 24) if tl_ok else 0)
            | ((1 << 26) if (cs_ok or cs_nh4_ok or dh2_ok or cs_d256_ok) else 0)
            | ((1 << 28) if dta_ok else 0)
            | ((1 << 29) if (nh1_ok or dh2_ok) else 0)
            | ((1 << 30) if tail_split else 0)
            | ((dph >> 4) << 16),
            floor_keys, chunk_target,
            float(bmm1_scale) * LOG2E, float(bmm2_scale), LOG2E,
            self._lazy_thresh)
        cbuf = (ctypes.c_char * len(buf)).from_buffer(buf)
        # TMA tensormaps: encode once per key (kv ptrs/strides/geometry are part
        # of the key and pinned by the entry). Falls back to plain launches on
        # any encode failure or missing symbol (non-TMA builds).
        kp = vp = 0
        maps_buf = None
        # Tensormaps: encode once per key (kv ptrs/strides/geometry are part of
        # the key and pinned by the entry). tc prefill needs them for d128/d256
        # NT64; the m16 decode path uses them for every supported head dim.
        # Any encode failure or non-TMA build leaves kp/vp = 0 -> cp.async path.
        enc = getattr(self._lib, "fmha_encode_kv_maps", None)
        if enc is None:
            enc = getattr(self._tc, "fmha_encode_kv_maps", None) if self._tc else None
        # Decode TMA pays for itself once a block walks several stages; with a
        # couple of stages the extra prologue barrier costs more than the
        # cp.async chain it replaces (boundary b5: 14.5us TMA vs 12.0us cp.async).
        dec_long = max_kv >= 8 * n_tile
        want_maps = self._tma and enc is not None and dq == dv and dq in (64, 128, 256) and (
            (use_tc == 1 and m_tile == 128
             and ((dq == 64 and n_tile == 128 and (self._tc_d64_tma or nh4_ok))
                  or (dq >= 128 and n_tile == 64)
                  or (dq == 128 and n_tile == 128 and self._tc_ws)
                  or (dq == 64 and n_tile == 64 and self._tc_ws)))
            or (use_tc == 1 and dh2_ok and m_tile == 256 and n_tile == 64
                and dq == 128)
            or (use_tc == 0 and m_tile == 16 and self._dec_tma and dec_long
                and n_tile in (32, 64)))
        if want_maps:
            maps_buf = ctypes.create_string_buffer(384)
            base_addr = ctypes.addressof(maps_buf)
            off = (-base_addr) % 64
            kp = base_addr + off
            vp = kp + 128
            rc_enc = enc(
                kp, vp, k.data_ptr(), v.data_ptr(),
                dq, dv, page_size, kvh, n_tile, k.shape[0],
                k.stride(2), v.stride(2), k.stride(1), v.stride(1),
                k.stride(0), v.stride(0))
            if rc_enc:
                kp = vp = 0
                maps_buf = None
        # ent: (addr, cbuf, buf, use_tc, splits_ub, ws, cnt, pins) — keeps
        # buffers AND all keyed input tensors alive so cached data_ptrs/ids
        # cannot be recycled to a different tensor while an entry exists.
        self._dispatch_cache[key] = (ctypes.addressof(cbuf), cbuf, buf,
                                     use_tc, splits_ub, ws, self._cnt,
                                     kp, vp,
                                     (query, kv_cache, out, lse, sinks,
                                      block_tables, seq_lens,
                                      cum_seq_lens_q, cum_seq_lens_kv,
                                      maps_buf), fuse16)
        ent = self._dispatch_cache[key]
        self._fast_cache[fkey] = self._fast_entry(ent, query, kv_cache, out,
                                                  lse, sinks, block_tables,
                                                  seq_lens, cum_seq_lens_q,
                                                  cum_seq_lens_kv)
        stream = self._raw_stream(self._devidx)
        rc = self._launch(use_tc, splits_ub, stream, ctypes.addressof(cbuf),
                          kp, vp)
        if rc:
            raise RuntimeError(f"fmha kernel launch failed rc={rc}")
        if self._ext is not None:
            self._register_slot(ret_out, lse, return_lse, query, kv_cache, out,
                                sinks, block_tables, seq_lens, cum_seq_lens_q,
                                cum_seq_lens_kv, cbuf, kp, vp, splits_ub,
                                window_left, causal, max_kv_len, bmm1_scale,
                                bmm2_scale, ent, fuse16)
        return (ret_out, lse) if return_lse else ret_out

    def _register_slot(self, ret_out, lse, return_lse, query, kv_cache, out,
                       sinks, block_tables, seq_lens, cum_seq_lens_q,
                       cum_seq_lens_kv, cbuf, kp, vp, splits_ub, window_left,
                       causal, max_kv_len, bmm1_scale, bmm2_scale, ent,
                       fuse16=0):
        """Publish a tier-0 slot for the launcher that just served this key.

        The launcher choice mirrors _launch's outcome (self._used_tc), including
        whether the split combine still needs a second call from Python. The
        entry pins every tensor the slot's packed params point at, plus the
        params buffer and the host tensormaps, so the recorded addresses stay
        valid for as long as the slot is reachable.
        """
        try:
            used = self._used_tc
            comb = None
            if used == 2 and self._tc2 is not None:
                run = self._tc2.fmha_tc2_run
                if splits_ub > 1:
                    comb = self._lib.fmha_combine_only
            elif used == 1 and self._tc is not None:
                run = self._tc.fmha_tc_run
                if splits_ub > 1 and not self._c_in_c:
                    comb = self._lib.fmha_combine_only
            else:
                run = self._lib.fmha_run     # launches the combine itself
            run_addr = ctypes.cast(run, ctypes.c_void_p).value
            comb_addr = ctypes.cast(comb, ctypes.c_void_p).value if comb else 0
            if not run_addr:
                return
            slot = self._ext.register_slot(ctypes.addressof(cbuf), run_addr,
                                           comb_addr, kp, vp)
            if slot is None or slot < 0:
                return
            skey = (id(query), query.data_ptr(), window_left, causal, return_lse,
                    max_kv_len, bmm1_scale, bmm2_scale)
            # See FMHA_GRAPH_SMART: only multi-launch chains benefit from the
            # graph; single-kernel slots stay on the (faster) direct chain.
            multi = splits_ub > 1 and (used in (1, 2) or not fuse16)
            graph_slot = self._ext_graph and (
                multi or not self._graph_smart)
            self._slot_cache[skey] = (
                slot, ret_out, kv_cache, lse, sinks, block_tables, seq_lens,
                cum_seq_lens_q, cum_seq_lens_kv,
                (ret_out, lse) if return_lse else ret_out,
                [0] if graph_slot else None, ent)
        except Exception:  # noqa: BLE001 - the ctypes tiers remain correct
            return

    def _forward_slow(self, query, kv_cache, workspace_buffer, block_tables,  # noqa: C901
                      seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale,
                      batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left,
                      out, out_dtype, o_sf_scale, o_sf_vec_size, kv_layout,
                      enable_pdl, sinks, kv_cache_sf,
                      skip_softmax_threshold_scale_factor, uses_shared_paged_kv_idx,
                      causal, lse, return_lse, multi_ctas_kv_counter_buffer,
                      use_fp16_softmax, uses_spcompress, backend):
        """Full validation path for non-default ABI options; domain uses defaults."""
        if out is None:
            raise ValueError("out tensor is required by this contract")
        if kv_layout != 'HND':
            raise ValueError("only HND kv_layout is supported")
        for name, value in (('o_sf_scale', o_sf_scale), ('o_sf_vec_size', o_sf_vec_size),
                            ('kv_cache_sf', kv_cache_sf),
                            ('multi_ctas_kv_counter_buffer', multi_ctas_kv_counter_buffer),
                            ('skip_softmax_threshold_scale_factor',
                             skip_softmax_threshold_scale_factor),
                            ('use_fp16_softmax', use_fp16_softmax),
                            ('uses_spcompress', uses_spcompress)):
            if value is not None:
                raise ValueError(f"unsupported contract option {name}={value!r}")
        raise ValueError("kv_cache/option combination not supported on the fast path")


# nn.Module.__call__ adds two Python frames plus a torch._C._get_tracing_state()
# call before reaching forward; this operator registers no hooks, so bind the
# entry point directly. Same arguments, same return value, same mutations.
if os.environ.get("FMHA_HOSTV2", "1") != "0":
    Model.__call__ = Model.forward
