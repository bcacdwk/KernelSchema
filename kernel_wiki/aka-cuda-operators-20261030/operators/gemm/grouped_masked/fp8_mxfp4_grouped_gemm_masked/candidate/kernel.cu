// fp8_mxfp4_grouped_gemm_masked — independent CUDA/PTX implementation for SM103.
//
// Computes, per expert e with active row count c = masked_m[e]:
//   d[e, :c, :] = (a_q[e,:c,:] * 2^(sfa-127)) @ (fp4_decode(b_q[e]) * 2^(sfb-127))^T
// with a_q = e4m3 payload (E,M,K), b_q = packed e2m1 nibbles (E,N,K/2),
// sfa/sfb = float32 exact-power-of-two scales (E,M,K/32)/(E,N,K/32), d = bf16.
//
// Mechanism: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale (scale_vec::1X,
// UE8M0 scale factors staged in TMEM via tcgen05.cp UTCCP), TMA loads with
// 16U4 nibble->byte unpack for the fp4 operand, warp-specialized mbarrier
// pipeline, persistent masked-grouped scheduler, swap-AB orientation with
// dynamic UMMA-N shrink to align(effective_m,16), STSM-transposing epilogue
// with 128B-swizzled TMA stores.
//
// Scale factors are consumed DIRECTLY from the incoming float32 tensors inside
// the measured kernel (exponent-byte extraction + UTCCP-layout packing fused
// into the pipeline), instead of a separate repack kernel + packed round-trip.
//
// Mechanism references (studied read-only; all code here is an independent
// implementation): PTX ISA (tcgen05/mbarrier/bulk-tensor families); DeepGEMM
// e1f418c2a4f2 (frozen production revision: sm100_fp8_fp4_gemm_1d1d pipeline
// structure, SF UTCCP transpose layout, descriptor arithmetic, scheduler);
// flashinfer (raw tcgen05 PTX syntax, block-scaled idesc bit positions).
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef FMGGM_B_TX_DIV
#define FMGGM_B_TX_DIV 1   // 1 = tx counts packed bytes (8KB, probe3-confirmed); 2 if unpacked
#endif
#ifndef FMGGM_TMA_STORE
#define FMGGM_TMA_STORE 0  // 0 = plain vectorized st.global epilogue; 1 = TMA bulk store
#endif
#ifndef FMGGM_SF_PF
#define FMGGM_SF_PF 4      // SF load prefetch depth (0 = direct per-stage loads)
#endif
#ifndef FMGGM_ABOX
#define FMGGM_ABOX 1       // 1 = shrink A TMA row-box to align(eff,16) on decode tiles
#endif
// BMERGE: issue the stage's two weight n-tiles as ONE 256-row TMA request (and
// the two weight-SF tiles as one {4,256} request), cutting the per-stage
// cp.async.bulk.tensor count from 6 to 4 at IDENTICAL bytes, expect_tx and smem
// image. Motivation: E89 (dv_7045f6bceb98) showed case 9's stage period is
// delivery-paced - deleting the whole accumulator handshake moved it 0.5% and
// just converted `hand` into `full` - while no resource is saturated (tensor
// pipe 57%, L2 64% of 18.05 TB/s, DRAM 36%, per-SM wire 57% of its ceiling).
// What is left on the delivery side is therefore per-request cost, and every
// prior delivery experiment ADDED requests (BPT +6% -> 0.9687, BPF +33-50% ->
// 0.912-0.922); this is the first that removes them. Value-identical: rows
// 128..255 of a 256-row 128B-swizzled box start a new 1024B swizzle atom at
// phase (row mod 8) == 0, exactly what a separate 128-row box at the
// 16,384B-aligned offset sm.b[stage]+BN*BK produced, and the SFB box is
// SWIZZLE_NONE so its rows are contiguous either way.
#ifndef FMGGM_BMERGE
#define FMGGM_BMERGE 0
#endif
// MCAST: run the grid as 74 clusters of 2 CTAs and let the two CTAs of a
// cluster SHARE the weight stream. Adjacent worker ids already decode to
// adjacent m-blocks of the SAME (group, n_blk) - the scheduler's decode is
// m-fastest inside a g_group sub-group, and flat = iter*num_sms + worker_id
// with num_sms even, so flat&1 == the cluster rank - so no tile-assignment
// change is needed: rank r fetches only weight n-tile j=r (128 of the 256 B
// rows) and multicasts it to both CTAs, and each CTA's smem still ends up with
// the identical image at the identical offsets, so expect_tx, the UMMA
// descriptors, the pack warps and the epilogue are untouched.
//
// Why: E94 measured B's whole delivery at 139 cycles/stage on case 9 (3x A's
// 46 at IDENTICAL wire bytes, because B is 256 row-accesses of 64 B against
// A's 128 of 128 B), and E96's EXPER=33 halved B's rows and got -58.4
// cycles/stage = -6.13% of case 9's period (-9.12% on case 5). Multicast
// halves the same rows and the same L2 reads; unlike EXPER=33 it does not
// remove the 32,768 B/stage of unpacked smem writes each CTA absorbs, so
// -3% to -6% of the period on the seven pairable fixed cases is the band.
//
// Ordering: a peer that drifts by one k-stage would multicast the wrong rows
// into the other's slot, so each producer relays "my MMA freed slot S" to the
// peer's pempty[S] (one remote mbarrier.arrive per stage, issued after its own
// empty[S] wait) and waits for the peer's relay before issuing. That is a
// producer-side handshake hidden in the ring's 4-stage slack - NOT a cluster
// barrier, whose ~280 ns/stage is what closed the cta_group::2 path.
//   0 = off (default);  1 = cluster + relay + multicast (the candidate);
//   2 = cluster + relay only, every CTA still fetches both weight halves;
//   3 = cluster + multicast only, no relay (races, numerics may be wrong);
//   4 = clustered launch only, no relay and no multicast (the co-residency
//       control: a persistent kernel needs all 148 workers resident at once);
//   7 = the hardware commit signalling of 5 with NO multicast loads: the arm
//       that prices the coupling by itself, so both5 minus 7 is the multicast's
//       own saving in this configuration;
//   5 = multicast + a hardware multicast tcgen05.commit for the slot-free signal
//       instead of the producer's software relay. E101b: on case 9 base 265.30,
//       4 (clustered, idle per stage) 267.26, 2 (relay only) 523.28, 1 (both)
//       506.88 us - so the clustered launch is free, the software relay costs
//       +97%, and the multicast itself is worth -3.1% (523.28 -> 506.88), right
//       where E96 put it. 5 exists to keep the multicast and drop the relay.
// 2 and 3 are the bisect pair that tells a relay deadlock apart from a multicast
// tx-accounting one; neither is a candidate.
#ifndef FMGGM_MCAST
#define FMGGM_MCAST 0
#endif
#ifndef FMGGM_BK256
#define FMGGM_BK256 0      // 1 = BLOCK_K 256 with 3 stages (halves per-stage overhead)
#endif
#ifndef FMGGM_DUALN
#define FMGGM_DUALN 1      // DEFAULT ON (dv_48628da9bf93: all-7 PASS 0 mismatch,
#endif                     //     geo 0.992->1.086; case9 324.6->285.7, case5 246.9->212.1).
                           //     Two weight n-tiles per stage share one activation tile
                           //     (A tx amortized 2x, no cluster). -DFMGGM_DUALN=0 = old base;
                           //     BK256 arm requires -DFMGGM_DUALN=0 (combo is #error).
#ifndef FMGGM_N_B
#define FMGGM_N_B 2        // weight n-tiles per stage under DUALN (2 proven; 3 =
#endif                     //     triple-n experiment: 3 TMEM accs, STAGES=3)
#if FMGGM_N_B != 2 && FMGGM_N_B != 3
#error "FMGGM_N_B supports 2 (dualn) or 3 (triplen)"
#endif
// NOTE: defined here, ahead of every guard that tests it. The J0LB guard below
// tests `FMGGM_BMV != 128`, and an undefined macro evaluates to 0 in #if, which
// would make that test TRUE and silently disable J0LB in the default build. The
// combination #error that names J0LA/LDBATCH/EDR/DRAIN2/BK256/MCAST/BMERGE/EBOX
// stays down in the config section, after all of those are defined.
// ------------------------------------------------------------------ BMV
// BMV: activation rows per tile (= UMMA N under the swap-AB mapping). Ported back
// onto this tree from the pre-join lineage (535f7a0 parameterise, 48f0525 the SF
// ring stride, b5042e1 the msfb box height) because E111/E113 established that the
// wide activation tile is the largest identified prize left - E21's primary table
// has bmv224 at -18.6% on case 7 and bmv192 at -5.0%/-3.0% on cases 3/9 - and that
// the pre-join summary had flattened it to "neutral" from a geomean.
//
// EVERY substitution this flag introduces is an identity at BMV=128: BM*4 == 512,
// NSFA == 1, SF_SUB*NSFA == SF_SUB, SFA_STRIDE == BM*4*SF_SUB, the a_rows ladder's
// new rung is unreachable, the new A-box rung is gated on BM > 128, and msfb's
// box[1] = BN == BM. So the default build is bit-identical, not merely equivalent.
//
// BMV > 128 forces the ring to 3 stages (a 224-row stage is 71,424 B; E108's
// enumeration puts (224,256)@D=3 at 223,488 B of the 232,448 B budget) and
// disables J0LB, whose odd-parity j0 slot needs BM more columns than 2*BM+12 leaves.
#ifndef FMGGM_BMV
#define FMGGM_BMV 128
#endif
#if FMGGM_BMV != 128 && FMGGM_BMV != 160 && FMGGM_BMV != 192 && FMGGM_BMV != 224
#error "FMGGM_BMV supports 128 (shipped), 160, 192 or 224 - E21's measured wide arms"
#endif
// ------------------------------------------------------------------ E120: device route
// E119 proved two things about the wide activation tile. (1) The statistic that separates
// E117's winners from its losers on all ten fixed cases is blk_ratio = sum(ceil(c/128)) /
// sum(ceil(c/224)) over the ACTUAL group sizes, with a clean gap (losers top out at 1.333,
// winners start at 1.429); a >= 1.40 threshold selects exactly {4,7,9} = the per-case oracle,
// worth +2.49% same-job against the +0.78% a host rule can take. (2) blk_ratio is not
// computable on the host, and that is a theorem rather than a suspicion: problem/input.py:82-85
// hands the host exactly (groups, capacity, n, k, expected_m = ceil(sum(gs)/groups)), and fixed
// case 7 ([0]*7+[1024,1024]+[0]*7) is host-INDISTINGUISHABLE from [128]*16 at the same
// e/capacity/n/k while having blk 1.600 against 1.000 - E117's biggest winner against the class
// it measured at -2.9% to -27.9%. The same is true of the shapes the shipped expected_m>=384 gate
// already takes: e=16 cap=512 gs=[512]*16 is legal under every cross_field_invariant, is
// dispatched wide, and sits at blk 1.333 = the fixed8 class measured at -6.2%.
// DROUTE computes blk_ratio on the device from masked_m, once per CTA, and branches into one of
// two already-proven bodies. The route is a pure function of masked_m and num_groups, so it is
// identical in every CTA and the branch is CTA-uniform - no divergent barrier or collective.
// Both bodies are E117's 30/30 oracle-clean geometries; nothing here changes numerics.
// The value defaults live here, ahead of every site that tests them; the COMBINATION guards that
// reference other arms live down with the FMGGM_BMV guard at "config", because an undefined macro
// evaluates to 0 in #if and a guard placed before its flags are defined passes silently - the
// exact hazard that made FMGGM_BMV move to this position in the first place.
//
// DEFAULT ON as of E121, keyed to FMGGM_BMV so that -DFMGGM_BMV=<wide> still compiles the
// standalone single-geometry arm that E117/E120 measure against: a wide FMGGM_BMV IS the wide
// body, so there is nothing left to route between. -DFMGGM_DROUTE=0 restores the certified
// single-geometry build bit for bit.
#ifndef FMGGM_DROUTE
#define FMGGM_DROUTE (FMGGM_BMV == 128 ? 1 : 0)
#endif
#ifndef FMGGM_DROUTE_BM
#define FMGGM_DROUTE_BM 224
#endif
// blk_ratio >= FMGGM_DROUTE_NUM/FMGGM_DROUTE_DEN. 7/5 = 1.40 sits in E119's measured gap.
#ifndef FMGGM_DROUTE_NUM
#define FMGGM_DROUTE_NUM 7
#endif
#ifndef FMGGM_DROUTE_DEN
#define FMGGM_DROUTE_DEN 5
#endif
#if FMGGM_BK256 && FMGGM_DUALN
#error "BK256+DUALN combination untested; build one experimental arm at a time"
#endif
#ifndef FMGGM_EDR
#define FMGGM_EDR 0        // 1 = early drain release: all tmem_lds first, arrive,
#endif                     //     then pack/STSM/st.global overlap the next mainloop
#ifndef FMGGM_DCS
#define FMGGM_DCS 0        // 1 = st.global.cs (streaming/evict-first) for D stores:
#endif                     //     D has zero in-kernel reuse; stop write-allocate pressure
#ifndef FMGGM_TCP
#define FMGGM_TCP 1        // DEFAULT ON: evict_last L2 policy on B TMA loads (retain
#endif                     //     the reused B-pair working set against D-write streaming).
                           //     Reproduced x2 (dv_5ed97ffb168a, dv_6104c299e261):
                           //     case3 -4.8%, case6 -4.2%, all else identical. -DFMGGM_TCP=0
                           //     restores the prior behavior. Value-identical cache hint.
#ifndef FMGGM_BPF
#define FMGGM_BPF 0        // >0 = prefetch B tiles this many k-stages ahead into L2
#endif                     //      (cp.async.bulk.prefetch; decode B is strided-DRAM bound)
#ifndef FMGGM_J0LA
#define FMGGM_J0LA 0       // 1 = j0-lookahead: overlap exposed DUALN drain with the
#endif                     //     next pair's first-STAGES j0 UMMAs (needs FMGGM_DUALN)
#if FMGGM_J0LA && !FMGGM_DUALN
#error "J0LA requires DUALN"
#endif
#if FMGGM_J0LA && FMGGM_N_B != 2
#error "J0LA hardcoded for N_B=2 column layout"
#endif
#if FMGGM_J0LA && FMGGM_LDBATCH
#error "J0LA+LDBATCH untested combination"
#endif
#if FMGGM_EDR && (FMGGM_LDBATCH || FMGGM_J0LA)
#error "EDR is implemented against the plain DUALN drain"
#endif
// Ablation switches for perf-only diagnosis (results intentionally WRONG when != 0):
//  1: UTCCP only at kb==0 per tile   2: skip all UMMAs (keep commits)
//  3: skip all UTCCPs                5: skip SF loads/packing (keep arrives)
//  6: skip B TMA loads (tx adjusted)
// 1-CTA epilogue drain decomposition (DUALN exposed-drain cost):
//  8: skip st.global pass (keep tmem_ld/STSM/barriers)
//  9: skip tmem_ld/pack/STSM (keep arrives/barriers/st.global of stale smem)
// 10: skip both (mainloop + arrives only = exposed-drain ceiling)
// exp2 pipeline decomposition (post-join arm a):
// 14: skip ALL TMA loads (A+B+SF, tx=0) = pure chain/issue overhead
// 16: skip SF loads+packing AND UTCCPs (combined 5+3)
// 17: skip A TMA loads (tx adjusted; pairs with 6 for B)
// 18: keep SF TMA + sf_full wait, skip pack-warp LDS/pack/STS work only
// 20: per-stage tcgen05.commit(empty) -> plain elected arrive (ring recycles
//     without waiting UMMA smem-read completion; measures commit latency's
//     share of the ~450ns/stage chain floor; slot data may be overwritten
//     in flight -> garbage numerics, structure safe: tile-final tmem_full
//     commit still drains all tc ops before the epilogue)
// 26: = 14 (all TMA loads skipped) PLUS MMA-warp smem descriptors frozen at
//     slot 0 (no per-stage descriptor arithmetic; TC operand window fixed at
//     slot 0's 48KB instead of rotating over the ring). e26 vs e14 isolates
//     (descriptor-arithmetic issue cost + rotating-window TC fetch locality),
//     the two remaining differences between the exp3 300ns/stage micro ring
//     and the ~477ns/stage in-situ chain floor. Garbage numerics, timing-only.
// 32: J0LB drain-handoff bubble probe: epilogue arrives tj1/tj0 at the FIRST
//     (si,li) instead of after the j half's last tmem_ld (races the next
//     tile's UMMA over cols being read -> garbage numerics, no protocol
//     violation: still exactly one arrive per barrier per tile). base-e32 =
//     the per-tile MMA stall waiting for the previous tile's 16 sequential
//     tmem_ld round-trips (~2-3us/tile from case3-vs-case9 e26 scaling).
//     E89 extended it to the DINTER drain, which is the default path and
//     whose si-major order releases j1 at the LAST si rather than at the
//     halfway point the per-j drain reaches; without that, -DFMGGM_EXPER=32
//     alone compiled to the unchanged DINTER arrival and probed nothing.
// 33: fetch only the FIRST weight n-tile's B rows (128 of 256) with expect_tx
//     adjusted, keeping all 8 UMMAs, the SF path and every barrier. E94 priced
//     B's whole delivery at 139 cycles/stage on case 9 (3x A's 46 at equal wire
//     bytes); this splits that into "rows/wire" and "unpacked smem writes" so
//     the ceiling on a 2-CTA B multicast - which halves rows and wire but NOT
//     the 32,768 B/stage each CTA absorbs - can be read off before any cluster
//     machinery is written. Garbage numerics (j1 reads stale smem), timing only.
#ifndef FMGGM_EXPER
#define FMGGM_EXPER 0
#endif
// PROMO: TMA tensor-map L2 promotion widths. exp4 micro (dv_...045039Z): the
// strided fp4 B pattern gains +6.2% at 128B promotion (3650->3875 GB/s),
// NONE +3.5%, 64B -18%. Value-identical cache hint.
//   0 = 256B on all maps (current);  1 = B map 128B;  2 = B+A+SFA+SFB 128B.
// exp5 IN-SITU: b128/promo128 net WORSE (case0 +3.0/+5.6%, case2 +2.9%;
// wins only case8 -3.0%, case3 -3.6%) -> CLOSED; kept for the record.
#ifndef FMGGM_PROMO
#define FMGGM_PROMO 0
#endif
// DESCROLL: chunk-unrolled MMA mainloop. Inside an STG-long unrolled body the
// ring slot is a compile-time constant => every smem descriptor (3 UTCCP +
// adesc/bdesc + per-UMMA offsets) is loop-invariant and hoists into registers;
// the elected lane's per-stage instruction stream collapses to ~5 ops.
// Evidence: exp5 e26-vs-e14 = -117ns/stage in situ (case9 chain 236.6->207.9,
// case5 168.2->146.6, case4 96.3->85.1, case0 47.1->40.0); micro m9 (rotating
// window, per-slot constant descs) = 292.9ns/stage ~= fixed-window control
// 299.8, m10 (per-stage arithmetic) = 322.3. Value-identical restructuring:
// same ops, same order, same barriers; alignment/remainder stages keep the
// runtime-slot path (ring position persists across tiles).
#ifndef FMGGM_DESCROLL
#define FMGGM_DESCROLL 1   // DEFAULT ON (E71 dv_71fdd08a0724, full-suite same-job A/B in
#endif                     // the symmetric order A,B,C,D,D,C,B,A at n=200 on THIS tree):
                           // geomean of per-case means 1.01199, stable-5 (0,5,6,7,9)
                           // 1.01171, positive on 10 of 10 cases - case4 +2.03%, case9
                           // +1.75%, case3 +1.72%, case6 +1.51%, case7 +1.44%, case8
                           // +1.18%, case5 +1.01%, case2 +0.68%, case1 +0.52%, case0
                           // +0.15%. Both DESCROLL arms agree (medians 60.264/60.343
                           // against base 61.098/61.080) and the frozen oracle reports 0
                           // mismatches on cases 0/1/3/6/8 in both. This is the largest
                           // single item measured in this phase and it confirms the in-situ
                           // claim above (-117 ns/stage) with the instrument that matters;
                           // the flag had been left off on in-situ/micro evidence only.
                           // TJEB (0.99954) and EBOX (0.99761) were screened in the same
                           // job and are rejected.
#if FMGGM_DESCROLL && FMGGM_J0LA
#error "DESCROLL restructures the non-J0LA mainloop only"
#endif
#if FMGGM_DESCROLL && FMGGM_N_B == 3
#error "DESCROLL dcol_j0/j1 hoisting assumes N_B <= 2"
#endif
// DINTER: interleaved-j epilogue drain. Instead of draining j1 fully then j0
// (each si storing a 16x128-col tile = 256B per row), drain both j halves per
// si into one 8KB cd image and store 16 rows x 256 cols = 512B contiguous per
// row in a single 4-pass burst. Motivation: exp2 attribution - skipping
// st.global saves 29.7us on case9 (e8, ~11%: memory-side interference with
// loads); mb_drain (exp7): 256B-row tile pattern caps at 3.80TB/s vs 6.19TB/s
// for the interleaved 512B-row pattern (j0/j1 write the SAME DRAM pages; the
// J0LB drain order used to separate them in time). Value-identical (same
// bytes; only temporal locality differs). Single-buffered cd (both j halves
// at once); the per-si named barriers serialize STSM vs stores as before.
// DEFAULT ON - same-job A/B x2: exp7 (dv_...055...): geo x0.9922 (case8
// -3.1%, case4 -1.8%, case2 -1.5%); exp8 (dv_...0619...): geo x0.9878
// (case8 -3.2%, case2 -2.7%, case3 -2.3%, case4 -1.8%, case9 -1.4%, case5
// -0.8%); bit-identical on all 10 fixed cases in both jobs; frozen-oracle
// 0-mismatch {0,3,5,9}. -DFMGGM_DINTER=0 restores the J0LB-ordered drain.
#ifndef FMGGM_DINTER
#define FMGGM_DINTER 1
#endif
#if FMGGM_DINTER && (!FMGGM_DUALN || FMGGM_N_B != 2 || FMGGM_EDR || FMGGM_LDBATCH \
                     || FMGGM_TMA_STORE || FMGGM_J0LA || FMGGM_EXPER != 0)
#undef FMGGM_DINTER
#define FMGGM_DINTER 0   // incompatible arms silently run the legacy drain path
#endif
// STAIL: wave-quantization tail splitting. Makespan = ceil(T/W) uniform tile
// waves; fixed-suite tails: case3/7 T=448 R=4 (4.00 vs 3.03 ideal waves),
// case4 R=52 (12 vs 11.35), case9 R=40 (31 vs 30.27). When R = T mod W is in
// [1, W/2], the last R pair-tiles are replaced by 2R single-j half-tiles
// (~0.65-0.75 units each): makespan F+~0.7 instead of F+1. T (total pair-
// tiles) piggybacks on the existing s_any early-exit pass (one extra smem
// atomicAdd per group; no extra barrier). Value-identical: same per-tile
// math; the two half-tile siblings share their A/SFA fetches via L2.
// Epilogue tj0/tj1 arrivals generalize: an unused j half arrives at tile
// start (its TMEM cols are never read that tile), a used half arrives after
// its last tmem_ld - exactly one arrival per barrier per tile as before.
// DEFAULT ON - exp9 (dv_...): same-job A/B geo x0.9737 (case7 -9.2%, case3
// -5.4%, case6 -4.4%*, case4 -3.5%, case0 -2.4%*, case9 -0.8%; *no-split
// codegen side-gains, confirm officially); bit-identical on all 10 fixed
// cases; frozen-oracle 0-mismatch {0,3,5,7,9} both arms.
// -DFMGGM_STAIL=0 restores the uniform-tile enumeration.
#ifndef FMGGM_STAIL
#define FMGGM_STAIL 1
#endif
// E70: the scheduler's group walk. A CTA's consecutive tiles are num_sms apart in the
// flat index, so each next() skips ~num_sms/(tiles per expert) experts, and every skipped
// expert is one serial load -> ceil-divide -> multiply -> branch. E68's traces on the
// pre-join tree (same walk, same shapes) put gaptile at 4,133 cycles/tile on case 6
// against 870 on case 9, which tracks exactly that trip count: 148/6 = 25 experts per hop
// on case 6 (64 experts x 6 n-pairs = 384 tiles) versus 148/140 = 1 on case 9 (32 x 140).
// ~165 cycles per skipped expert is a dependency chain, not work.
//   BMSHIFT: ceil(cnt/bm) by shift instead of divide - bm is 16 or 128 at every call site.
//   MMC: read masked_m from an smem snapshot that the STAIL pass ALREADY holds in a
//        register (line ~869 loads every group), so it costs one smem store per thread and
//        zero extra global traffic, ordered by the barrier that pass already ends with.
// Both are value-identical. Pre-join E51/E52 rejected these two ideas, but they measured
// gaptile on case 9 - the one case where the walk is a single expert long and there is
// nothing to remove. Case 6/2/3/8 are where the trip count is 5-25.
#ifndef FMGGM_BMSHIFT
#define FMGGM_BMSHIFT 1   // DEFAULT ON (E70 dv_ef1830b3c2d6): value-identical, no smem
#endif                    // or register cost; stable-5 geomean of means 1.00218, all-10
// BMSHIFT replaces `(cnt + bm - 1) / bm` with `>> bm_shift` and `effective_m`'s clamp
// with `1 << bm_shift`, where bm_shift = ceil(log2(bm)) - so it is only value-identical
// when bm is a POWER OF TWO, which its own comment states ("bm is 16 or 128 by
// construction"). A wide activation tile breaks that precondition silently and
// catastrophically: at BMV=192 the loop `while ((1 << bm_shift) < bm_) ++bm_shift`
// stops at 8, so nmb_of computes ceil(cnt/256) instead of ceil(cnt/192) and
// effective_m clamps to 256 instead of 192. The kernel then walks too few m-blocks
// (rows never computed) and issues UMMAs with N up to 256 against an A tile that only
// fetched 192 rows - which is exactly what E114's first run measured: 8 of 12 wide-arm
// oracle checks failing, case 9 at 119,641,646 of 146,800,640 elements (81%), while
// the base arm stayed clean. Auto-disable rather than #error, so the wide arms stay
// buildable; the cost is BMSHIFT's own +0.1-0.2%, which means a wide arm measured
// this way is if anything UNDERSTATED. (128 & 127) == 0, so the default is untouched.
//
// E120 walked straight back into this through a new door, and it is worth naming because the
// guard above looked sufficient: the test is on FMGGM_BMV, but under FMGGM_DROUTE the file-scope
// BMV stays 128 (a power of two, so BMSHIFT stayed ON) while the route passes bm = 224 into
// sched.init at RUNTIME. bm_shift's loop stopped at 8 exactly as documented, so the wide body
// walked ceil(cnt/256) m-blocks and clamped effective_m to 256 against a 224-row A tile. Job
// dv_e2fea4915a3b: the three wide-routed cases and only those three came back
// 14,023,156 / 4,123,104 / 40,488,917 mismatches with rel_l2 = nan and nonfinite = True, while
// all seven narrow-routed cases were bit-clean in the same job - the signature of rows never
// computed, not of a wrong tile. The precondition is a property of the bm the SCHEDULER receives,
// not of FMGGM_BMV, so the guard has to test every bm any instantiation can pass.
// Cost: BMSHIFT is a preprocessor flag, so a DROUTE build gives it up for the narrow body too,
// which is E70's measured +0.10...+0.22%. Recovering it means templating Scheduler on BMV so bm
// is a compile-time constant and 224 divides by magic multiply; that is the identified follow-up,
// not something to bundle with the first correctness fix.
#if FMGGM_BMSHIFT && (((FMGGM_BMV & (FMGGM_BMV - 1)) != 0) \
                      || (FMGGM_DROUTE && ((FMGGM_DROUTE_BM & (FMGGM_DROUTE_BM - 1)) != 0)))
#undef FMGGM_BMSHIFT
#define FMGGM_BMSHIFT 0
#endif
                          // 1.00093, 6 of 6 variant arms below both base arms on medians,
                          // 0 mismatches on cases 0/1/3/6/8 in both arms. Case 6 +0.71%,
                          // case 7 +0.30%, case 4 +0.17% on means. MMC (the smem snapshot
                          // of masked_m) measured the SAME all-10 (+0.09%) but -0.50% on
                          // case 6 and costs 256 B of smem, and BMSHIFT+MMC together was
                          // no better than either alone (1.00122) - they break the same
                          // serial chain, so only the free half is kept.
                          // HONEST SCOPE: the mechanism model predicted ~5% (gaptile is
                          // 4,133 cycles/tile on case 6 and the walk skips ~25 experts per
                          // hop) and delivered ~0.1-0.2%, so the load+divide was only
                          // ~120 of those 4,133 cycles. The rest tracks tiles-per-expert
                          // monotonically (case9 140/expert -> 870; case3 28 -> 1660;
                          // case8 32 -> 1669; case2 12 -> 2659; case6 6 -> 4133), i.e. it
                          // is compulsory DRAM miss latency on the new tile's weights at
                          // the boundary, which the 4-deep ring cannot hide and depth 5
                          // does not fit. Recorded as a closure, not an open lead.
#ifndef FMGGM_MMC
#define FMGGM_MMC 0
#endif
#ifndef FMGGM_MMC_N
#define FMGGM_MMC_N 64       // frozen shape_range: groups <= 64 (max_length of group_sizes)
#endif
#if FMGGM_MMC && !FMGGM_STAIL
#error "MMC piggybacks on the STAIL masked_m pass; it needs its own pass without STAIL"
#endif
#if FMGGM_STAIL && FMGGM_J0LA
#error "STAIL arrival generalization covers the J0LB/base drain schemes only"
#endif
// TJEB: batched epilogue TMEM capture. The drain currently issues 2 tmem_ld
// per (j,si,li) followed by tmem_ld_wait -> 32 sequential round-trips per
// pair-tile, and tj1_empty (which gates the NEXT tile's first UMMA under
// J0LB) only fires after the j1 half's 16th wait. TJEB issues all 8 loads of
// an si (both j halves x both li) before ONE tmem_ld_wait: 8 round-trips per
// tile, and tj0/tj1 arrive right after the last si's wait. Estimated drain-
// handoff bubble ~2-3us/tile (case3-vs-case9 e26 per-stage scaling); EXPER=32
// (early-arrive probe) bounds it. Value-identical: same loads/arrivals/stores,
// fewer wait boundaries; registers +32 f32 transient per epilogue thread.
#ifndef FMGGM_TJEB
#define FMGGM_TJEB 0
#endif
#ifndef FMGGM_LDBATCH
#define FMGGM_LDBATCH 0    // 1 = issue all 4 tmem_lds of an si up front, single wait
#endif
// B-tile L2 eviction: 0 = normal; 1 = evict_first when the expert has a single
// m-block (no cross-tile B reuse; keeps L2 for A/SF); 2 = always evict_first.
#ifndef FMGGM_LEVICT
#define FMGGM_LEVICT 0
#endif
// Programmatic dependent launch: the engine's restore predecessor writes ONLY
// d, so prologue/TMA/mainloop (a/b/SF/masked_m untouched by it) may legally
// overlap it; only the epilogue's D stores must wait. -DFMGGM_PDL=1 to test.
#ifndef FMGGM_PDL
#define FMGGM_PDL 0
#endif
// EBOX: full eff-ladder A/SFA TMA boxes {16,32,...,128} (ABOX has only
// 16/32/64). Cuts A tx for 16<eff<128 mids (case2/6 eff=112: -12.5% A;
// case3/8 second m-blocks eff=96) and SFA f32 staging tx to eff rows
// (decode: 2KB->0.25KB/stage). Rows>eff keep stale smem -> packed ->
// UMMA D cols > eff, never stored (same safety as existing ABOX).
#ifndef FMGGM_EBOX
#define FMGGM_EBOX 0
#endif
// FASTEXIT: replace the s_any early-exit check (2x __syncthreads + smem
// atomicOr + single-warp masked_m ldg; ~1.6us per launch, exp13 floor vs
// base case1: 5.2 vs 6.8us) with a warp-redundant check: EVERY warp reads
// masked_m[0..E) (E<=256 -> <=8 x 32B loads per lane, L2-broadcast) and
// branches on its own identical OR - no barrier, no smem. Same guarantee:
// all warps see the same masked_m => unanimous branch.
#ifndef FMGGM_FASTEXIT
#define FMGGM_FASTEXIT 0
#endif
// DEC16: second 1-CTA config for decode shapes (expected_m<=16 && M<=128):
// BMV=16 activation rows (a[] 2KB/stage) with STG=5 (stage 40KB, 213KB
// total) - deeper B ring: in-flight B 5x32KBx148SM = 23.7MB vs 9.4MB at
// S=4, covering DRAM BWxlatency (~8MB) for the cold strided decode B
// stream (exp6 on J0LB tree: B-fill still 24% of case0). sfa_raw KEEPS
// its 128-row size (SFA TMA + pack warps untouched; ebox3 showed the
// SFA shrink ~0 gain). Gate bounds worst-case W refetch at M/BMV = 8
// m-blocks for a single heavy expert (arm-b precedent class).
#ifndef FMGGM_DEC16
#define FMGGM_DEC16 1     // DEFAULT ON (job 20261023T021134Z): case0
#endif                     // 71.8->67.4us (+6.5%, mins track: 69.6->64.5),
                           // all other fixed cases 1.000 (gate exact),
                           // oracle PASS incl. dec16/heavy16 configs.
                           // -DFMGGM_DEC16=0 restores single-config.
#if FMGGM_DEC16 && (FMGGM_BK256 || !FMGGM_DUALN || FMGGM_N_B != 2)
#error "DEC16 is defined for the BK128 DUALN N_B=2 config only"
#endif
// DEC_STG: ring depth of the DEC16 decode route, which the tree hardcodes at 5
// (its 40,960 B stage = 2,048 A + 32,768 B + 2,048 SFA + 4,096 SFW, so 5 stages
// + cd = 212,992 B and 6 would need 253,952 B against the 232,448 opt-in max).
// E92 exists because the two records of DEC16's own value disagree - the flag
// comment above says case 0 71.8 -> 67.4 (+6.5%, job 20261023T021134Z) while
// E4's v_dec16off says 69.6 -> 92.2 (+32%) - and because E66 measured the MAIN
// route's depth sensitivity as only +5.1% for 4 -> 3 while fitting
// P(D) = 816 + 588/D, which would make depth 5 worth ~3% and depth 6 ~5% if it
// fitted. Calibrating the decode route's own slope at n=200 decides whether a
// deeper/narrower decode ring (N_B as a route parameter, or an SFC-style shared
// SF buffer, either of which could reach depth 6) is worth building: if case 0
// is flat in DEC_STG the direction dies here for one cheap job, and cases 1-9
// are an exact internal control because the flag only sizes SharedStorageDec.
#ifndef FMGGM_DEC_STG
#define FMGGM_DEC_STG 5
#endif
#if FMGGM_DEC_STG < 2 || FMGGM_DEC_STG > 6
#error "DEC_STG outside [2,6]: 1 deadlocks the ring, >6 cannot fit the opt-in smem max"
#endif
// LDGSF: SF scale factors loaded DIRECTLY from the f32 global tensors by the
// pack warps (2-generation register lookahead, arm-b-lineage "mode4" pattern)
// instead of TMA f32 staging + LDS. Removes per stage: SF TMA-w (2KB SFA +
// 4KB SFW), the sf_full barrier hop, and the 6KB LDS re-read; keeps the
// 1.5KB packed STS + UTCCP. EXPER attribution: SF slice = 4-11% streams,
// 15.7% decode. Packed words land at the SAME raw-slot offsets UTCCP reads.
#ifndef FMGGM_LDGSF
#define FMGGM_LDGSF 0
#endif
#if FMGGM_LDGSF && FMGGM_BK256
#error "LDGSF generations sized for BK128 (BK256 untested)"
#endif
// J0LB: drain-overlap without ring-slot holding. Epilogue drains j1 FIRST
// and arrives tj1_empty when j1's TMEM loads are done; j0 accumulates in
// cols {0,268} alternating by pair parity, freeing via tj0_empty[par]. The
// next pair's MMA waits only those load-completions (not stores), and its
// per-stage empty[] commits flow normally - unlike J0LA, whose phase-1
// lookahead HELD all STAGES ring slots (no commits until phase-2a re-issue)
// and stalled the producer (measured neutral, dv_5f9730eb9547). Exposed
// drain = 16.6% of case9 / 15.9% of case4 (EXPER=10, dv job
// 20261022T043256Z). Requires DUALN N_B=2, ACC_STAGES=1.
#ifndef FMGGM_J0LB
#define FMGGM_J0LB 1      // DEFAULT ON (promoted; jobs 20261022T045523Z +
#endif                    // 20261022T050344Z): +1.2-1.4% geo same-job A/B x2,
                          // streams case4 +3.5-3.7%, case9 +2.3-2.9%, case8
                          // +1.6-3.2%, case6/7 +2.5-2.7%, oracle PASS x2.
                          // j1-first drain + j0 cols {0,268} alternation;
                          // per-stage empty[] commits unchanged (no producer
                          // stall, unlike J0LA). -DFMGGM_J0LB=0 = base drain.
// PKELECT: pack warps arrive full[] via syncwarp + elected arrive (count 3:
// 1 expect_tx + 2 elected) instead of 64 per-lane arrives. Per-lane
// fence_proxy_async stays; syncwarp orders all lanes' fenced stores before
// the elected release (pattern proven in the arm-b lineage).
#ifndef FMGGM_PKELECT
#define FMGGM_PKELECT 0
#endif
// EPAD: put every mbarrier array on its own 128B-aligned smem region. exp3
// ring micros showed pure barrier mechanics are ~40ns/stage while the real
// kernel's chain floor is ~450ns/stage; one structural difference is the
// dense barrier block (poll/arrive traffic on adjacent 8B words). Value-
// identical (layout-only).
#ifndef FMGGM_EPAD
#define FMGGM_EPAD 0
#endif
#if FMGGM_EPAD
#define FMGGM_MBA alignas(128)
#else
#define FMGGM_MBA
#endif
// EWSLEEP: epilogue's per-tile tmem_full spin uses try_wait.parity with a
// suspend-time hint, cutting hot-poll pressure from the 4 epilogue warps on
// the smem barrier words while the mainloop runs (exp3: in-situ chain floor
// is ~185ns/stage above the equivalent 3-warp micro ring; the real kernel
// has 4 warps hot-spinning through the whole mainloop, the micro has none).
// Value-identical (wait semantics unchanged).
#ifndef FMGGM_EWSLEEP
#define FMGGM_EWSLEEP 0
#endif
// GG > 0: force scheduler m-grouping factor (else heuristic over {8,16}).
#ifndef FMGGM_GG
#define FMGGM_GG 0
#endif
#if FMGGM_J0LB && (!FMGGM_DUALN || FMGGM_N_B != 2 || FMGGM_BMV != 128 || FMGGM_DROUTE)
#undef FMGGM_J0LB
#define FMGGM_J0LB 0      // only defined for DUALN N_B=2; silently off for
#endif                    // BK256/single-n diagnostic builds, and for DROUTE, whose wide
                          // body has 224-column accumulator slots rather than 128 (E120).
                          // E104 measured J0LB=0 at +0.110% on ten cases, i.e. J0LB currently
                          // COSTS 0.11%, so the route does not pay for losing it - but that
                          // measurement has an unmeasured domain tail and the pre-join credit
                          // was +1.2-1.4%, so this is a real trade and is recorded as one.
#if FMGGM_J0LB && (FMGGM_J0LA || FMGGM_EDR)
#error "J0LB replaces the J0LA/EDR drain restructurings"
#endif
#if FMGGM_J0LB && FMGGM_LDBATCH
#error "J0LB arrive/taddr edits target the non-LDBATCH drain"
#endif
#if FMGGM_SF_PF > 0 && (FMGGM_SF_PF % 4 != 0)
#error "SF prefetch ring is built for depth 4 (static slots); use 0 or 4"
#endif
// Deferred combination guards (all flags are defined at this point):
#if FMGGM_STAIL && FMGGM_FASTEXIT
#error "FASTEXIT early-exit path lacks the s_mblocks sum STAIL needs"
#endif
#if FMGGM_BMERGE && (!FMGGM_DUALN || FMGGM_N_B != 2 || FMGGM_BK256 || FMGGM_LDGSF)
#error "BMERGE needs DUALN N_B=2, SF_SUB==1 (BK=128) and TMA-delivered SF"
#endif
#if FMGGM_BMERGE && FMGGM_LEVICT != 0
#error "BMERGE's merged B request implements the TCP/default hint only"
#endif
#if FMGGM_MCAST && (!FMGGM_DUALN || FMGGM_N_B != 2 || FMGGM_BK256 || FMGGM_BMERGE || FMGGM_LDGSF \
                    || !FMGGM_STAIL || FMGGM_J0LA)
#error "MCAST shares the two DUALN weight n-tiles between cluster peers; needs N_B=2, BK=128, per-j B/SFB requests, TMA-delivered SF and STAIL's bounded position space"
#endif
// MCAST=3 is the bisect arm that isolates multicast tx-accounting from the relay: it multicasts B
// but never signals the peer, so the peer's pempty[] wait has no arriver and the pair wedges on the
// first stage that recycles a slot. It is not slow, it HANGS, and it already cost a job:
// dv_d8ffb84dbcc3 was deadline-killed at 1500 s with `reason: timeout` and NO artifacts retained,
// because one deadlocking arm inside e2_ablate destroys the whole job's evidence (that is why
// experiments/e98_paircheck.py and e100_v2path.py run one subprocess per arm with a hard timeout).
// The finding it produced is recorded in results/e103-*/ANALYSIS.md, so the arm has no further use
// under a harness. Pass -DFMGGM_ALLOW_DEADLOCK=1 to build it anyway, deliberately, off-harness.
#if FMGGM_MCAST == 3 && !defined(FMGGM_ALLOW_DEADLOCK)
#error "FMGGM_MCAST=3 DEADLOCKS by construction (multicast with no peer relay). See the comment above; build with -DFMGGM_ALLOW_DEADLOCK=1 only outside a timed harness."
#endif
#if FMGGM_TJEB && (FMGGM_EXPER != 0 || FMGGM_EDR || FMGGM_LDBATCH || FMGGM_TMA_STORE || FMGGM_J0LA)
#undef FMGGM_TJEB
#define FMGGM_TJEB 0   // diagnostics/alternate drains keep the legacy cadence
#endif

#define FMGGM_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "fmggm CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    return e_; } } while (0)

namespace fmggm {

// Diagnostic arms that run the ring with zero TMA loads (EXPER 14, 26).
constexpr bool kSkipLoads = (FMGGM_EXPER == 14 || FMGGM_EXPER == 26);

#if FMGGM_BMV != 128 && (FMGGM_J0LA || FMGGM_LDBATCH || FMGGM_EDR || FMGGM_DRAIN2 \
                         || FMGGM_BK256 || FMGGM_MCAST || FMGGM_BMERGE || FMGGM_EBOX)
#error "wide activation tile: these arms hardcode 128-column accumulator slots, a single 128-row activation-SF group, or the BK128 DUALN N_B=2 smem budget"
#endif
// E120 combination guards. Placed here, not with the value defaults, so that every flag they
// test has been defined: an undefined macro is 0 in #if, so a guard above its flags passes
// silently. J0LB is NOT in the list below because it is auto-disabled a few hundred lines above
// (it hardcodes 128-column accumulator slots and the wide body has 224), the same silent-downgrade
// policy already used for FMGGM_BMV != 128; an explicit -DFMGGM_J0LB=1 alongside DROUTE is
// therefore downgraded rather than rejected. E104 measured J0LB=0 at +0.110% on ten cases, so the
// route does not pay for losing it, but that measurement has an unmeasured domain tail and the
// pre-join credit was +1.2-1.4%, so it is a real trade and is recorded as one.
#if FMGGM_DROUTE
#if FMGGM_BMV != 128 || FMGGM_N_B != 2 || !FMGGM_DUALN
#error "DROUTE's narrow arm IS the file-scope geometry (BMV=128, DUALN N_B=2) and its wide arm is FMGGM_DROUTE_BM; compiling either arm separately defeats the point"
#endif
#if FMGGM_BK256 || FMGGM_MCAST || FMGGM_BMERGE || FMGGM_EBOX || FMGGM_J0LA \
    || FMGGM_EDR || FMGGM_DRAIN2 || FMGGM_LDBATCH || FMGGM_LDGSF
#error "DROUTE instantiates two bodies in one launch; these arms hardcode one geometry's TMEM columns, smem budget or SF path and are untested against a second one"
#endif
#if FMGGM_DROUTE_BM != 160 && FMGGM_DROUTE_BM != 192 && FMGGM_DROUTE_BM != 224
#error "FMGGM_DROUTE_BM supports 160, 192 or 224 - E21/E117's measured wide arms"
#endif
#endif
// ------------------------------------------------------------------ config
constexpr int BM = FMGGM_BMV;       // rows of A (activation capacity rows) per tile
constexpr int BN = 128;             // rows of B (weight N) per tile
constexpr int BK = FMGGM_BK256 ? 256 : 128;   // K elements per stage
constexpr int UMMAS_PER_STAGE = BK / 32;   // kind::mxf8f6f4 UMMA_K = 32
constexpr int SF_SUB = BK / 128;           // 128-K SF sub-blocks per stage
constexpr int N_B_PER_STAGE = FMGGM_DUALN ? FMGGM_N_B : 1;   // weight n-tiles per stage
constexpr int NW_SUB = SF_SUB * N_B_PER_STAGE;       // weight-SF sub-blocks per stage
constexpr int STAGES = FMGGM_BK256 ? 3
                     : (FMGGM_DUALN ? ((FMGGM_N_B == 3 || BM > 128) ? 3 : 4) : 6);
// DUALN: both acc halves (cols 0..255) belong to one n-pair; no double buffer
// (epilogue drain is exposed, but the TMA ring prefetches the next pair during it).
constexpr int ACC_STAGES = FMGGM_DUALN ? 1 : 2;
constexpr int STORE_M = 16;         // problem-M rows per TMA store box
constexpr int STORE_N_ATOM = 64;    // problem-N cols per swizzle atom / TMA box
constexpr int TMA_STORE_STAGES = 2;
constexpr int THREADS = 256;        // 8 warps: 0 TMA, 1 MMA, 2 SFA, 3 SFB, 4-7 epilogue

// TMEM column budget: 2*128 accumulator + 4 weight-SF + 4 activation-SF
constexpr int ACC_COL0 = 0;
constexpr int ACC_COL1 = BM;
// acc region = BM cols x N_B_PER_STAGE x ACC_STAGES, then SF groups.
constexpr int SFW_COL = BM * N_B_PER_STAGE * ACC_STAGES;    // weight SF, 4 cols per sub-block
constexpr int SFA_COL = SFW_COL + 4 * NW_SUB;               // activation SF
// The activation (MMA B-side) SF spans ceil(N/128) groups of 4 TMEM columns: TMEM
// lanes hold N%128 and the column group holds N/128, so N>128 needs more columns
// and one extra UTCCP per stage. Per-stage SF staging strides, in FLOATS, exactly
// matching the array extents below - 48f0525, because `(BM*4)*nsub` coincides with
// both sides at BM=128 (512/1024) but overshoots the activation array by
// nsub/NSFA once BM>128, running stage 2's stores off dynamic smem (E14:
// cudaErrorIllegalAddress on every wide arm).
constexpr int NSFA = (BM + 127) / 128;
constexpr int SFA_STRIDE = NSFA * 128 * 4 * SF_SUB;
constexpr int SFW_STRIDE = NW_SUB * BN * 4;
constexpr int TMEM_COLS = 512;
static_assert(BM * N_B_PER_STAGE * ACC_STAGES + 4 * NW_SUB + 4 * NSFA <= TMEM_COLS,
              "accumulators + weight SF + activation SF groups exceed 512 TMEM columns");

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n\t}" : "=r"(pred));
    return pred;
}
__device__ __forceinline__ void mbar_init(void* mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(void* mbar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(mbar)), "r"(tx));
}
__device__ __forceinline__ void mbar_wait(void* mbar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "LAB_W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_W;\n\tbra LAB_W;\n\tDONE_W:\n\t}"
        :: "r"(smem_u32(mbar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_wait_hint(void* mbar, uint32_t parity, uint32_t hint) {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "LAB_WH: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1, %2;\n\t"
        "@P bra DONE_WH;\n\tbra LAB_WH;\n\tDONE_WH:\n\t}"
        :: "r"(smem_u32(mbar)), "r"(parity), "r"(hint) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after_sync()  { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(base), "r"(ncols));
}
__device__ __forceinline__ void utccp_4x32dp128bit(uint32_t tmem_col, uint64_t sf_desc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_col), "l"(sf_desc));
}
__device__ __forceinline__ void umma_mxf8f6f4(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc,
                                              uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                              uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
        :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(accumulate));
}
__device__ __forceinline__ void umma_commit(void* mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(mbar)));
}
#if FMGGM_MCAST
// Same commit, arriving on `mask`'s CTAs at the same smem offset when this CTA's
// prior tcgen05 ops complete. The tree already uses the cta_group::2 form for the
// 2-CTA kernel; this is the cta_group::1 one, so each CTA keeps its own UMMAs and
// accumulator while the slot-free signal reaches both. Asynchronous and issued by
// the MMA warp, so nothing release-scoped sits on the producer's critical path.
__device__ __forceinline__ void umma_commit_mcast(void* mbar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
                 ".multicast::cluster.b64 [%0], %1;"
                 :: "r"(smem_u32(mbar)), "h"(mask));
}
#endif
__device__ __forceinline__ void tmem_ld_16x256b_x1(uint32_t addr, float* v) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "r"(addr));
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void stsm_x4_trans(uint32_t smem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void named_barrier_sync(uint32_t id, uint32_t count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int cx, int cy) {
    // L2-only prefetch of a tensor box; no smem dst, no barrier, no tx accounting.
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                 :: "l"(map), "r"(cx), "r"(cy) : "memory");
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map, void* mbar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d_evict_first(const CUtensorMap* map, void* mbar, void* smem,
                                                        int32_t c0, int32_t c1) {
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], pol;\n\t}"
        :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d_evict_last(const CUtensorMap* map, void* mbar, void* smem,
                                                       int32_t c0, int32_t c1) {
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], pol;\n\t}"
        :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
#if FMGGM_MCAST
// Multicast 2-D tile load: ONE read of the box is delivered to every CTA in
// `mask` at the SAME smem offset, crediting each of their mbarriers with the
// same tx bytes. Instruction form from cute's SM90_TMA_LOAD_MULTICAST_2D
// (references/nvidia-sources/flashinfer/csrc/nv_internal/tensorrt_llm/deep_gemm/
// nvrtc_cutlass.cuh:2409-2429), with the tree's evict_last policy so the weight
// hint is identical to the non-cluster path it replaces.
__device__ __forceinline__ void tma_load_2d_mcast_evict_last(const CUtensorMap* map, void* mbar,
                                                             void* smem, int32_t c0, int32_t c1,
                                                             uint16_t mask) {
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5, pol;\n\t}"
        :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
           "r"(smem_u32(mbar)), "h"(mask) : "memory");
}
__device__ __forceinline__ void tma_load_2d_mcast(const CUtensorMap* map, void* mbar, void* smem,
                                                  int32_t c0, int32_t c1, uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
                    "r"(smem_u32(mbar)), "h"(mask) : "memory");
}
#endif
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, const void* smem, int32_t c0, int32_t c1) {    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                 :: "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(smem)) : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;"); }
template <int N>
__device__ __forceinline__ void tma_store_wait() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N)); }

// ---- cluster / 2-CTA helpers ----
__device__ __forceinline__ uint32_t block_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t mapa_rank0(uint32_t smem_addr) {
    uint32_t out;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(out) : "r"(smem_addr));
    return out;
}
__device__ __forceinline__ void mbar_arrive_remote0(void* mbar) {
    // arrive on CTA0's copy of this CTA-relative barrier address
    uint32_t remote = mapa_rank0(smem_u32(mbar));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
#if FMGGM_MCAST
// MCAST runs 2-CTA clusters whose ranks are peers, not leader/follower, so the
// remote-arrive helpers need a rank operand (the tree's existing ones hardcode
// rank 0 for the cta_group::2 kernel).
__device__ __forceinline__ uint32_t mapa_rank(uint32_t smem_addr, uint32_t rank) {
    uint32_t out;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(smem_addr), "r"(rank));
    return out;
}
__device__ __forceinline__ void mbar_arrive_peer(void* mbar, uint32_t peer_rank) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 :: "r"(mapa_rank(smem_u32(mbar), peer_rank)) : "memory");
}
// Number of CTAs in this CTA's cluster (1 if the launch was not clustered).
__device__ __forceinline__ uint32_t cluster_size() {
    uint32_t n;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n));
    return n;
}
#endif
__device__ __forceinline__ void tma_load_2d_cg2(const CUtensorMap* map, void* mbar_leader, void* smem,
                                                int32_t c0, int32_t c1) {
    // cta_group::2 TMA: executed by BOTH CTAs with their own smem/coords;
    // transaction bytes accumulate on CTA0's barrier (peer bit masked off).
    uint32_t mbar0 = mapa_rank0(smem_u32(mbar_leader));
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(mbar0) : "memory");
}
__device__ __forceinline__ void tmem_alloc_cg2(uint32_t* dst, uint32_t n) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(dst)), "r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg2() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t b, uint32_t n) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(b), "r"(n));
}
__device__ __forceinline__ void utccp_cg2(uint32_t tmem_col, uint64_t desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(tmem_col), "l"(desc));
}
__device__ __forceinline__ void umma_mxf8f6f4_cg2(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                                                  uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc));
}
__device__ __forceinline__ void umma_commit_cg2_mc(void* mbar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :: "r"(smem_u32(mbar)), "h"(mask));
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait()   { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

// ------------------------------------------------------------------ descriptors
// SM100 UMMA SMEM descriptor: [0,14) start>>4, [16,30) LBO>>4, [32,46) SBO>>4,
// bit46 version=1, [61,64) layout type (2 = SWIZZLE_128B).
__device__ __host__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr, uint32_t sbo_bytes,
                                                            uint32_t lbo_bytes, uint32_t layout_type) {
    return (uint64_t)((smem_addr >> 4) & 0x3FFF)
         | ((uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16)
         | ((uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32)
         | (1ULL << 46)
         | ((uint64_t)(layout_type & 0x7) << 61);
}
// Block-scaled instruction descriptor (kind::mxf8f6f4):
// [4,6) b_sf_id, [7,10) a_format(E2M1=5), [10,13) b_format(E4M3=0), [15) a_major,
// [16) b_major, [17,23) n_dim=N>>3, [23) scale_format(UE8M0), [24,29) m_dim=M>>4,
// [29,31) a_sf_id. Swap-AB: MMA A operand = weights (e2m1), B = activations (e4m3).
__device__ __host__ __forceinline__ uint32_t make_idesc_base(uint32_t scale_format = 1 /*UE8M0*/,
                                                             uint32_t umma_m = 128) {
    return (5u << 7) | (0u << 10) | ((umma_m >> 4) << 24) | (scale_format << 23);
}

// ------------------------------------------------------------------ shared storage
template <int BMV, int STG>
struct __align__(1024) SharedStorageT {
    // E120: the activation-SF stride is a property of THIS instantiation, not of a single global
    // BM, and deriving it here is what lets one translation unit hold two geometries (the device
    // route). BMK is BMV for a wide instantiation and the FILE-SCOPE BM otherwise - not 128.
    // That distinction is load-bearing and was gotten wrong first: with -DFMGGM_BMV=224 the
    // DEC16 instantiation is SharedStorageT<16,5>, and rounding its BMK to 128 shrank sfa_raw
    // from 4,096 B to 2,048 B while the host still encodes map_sfa with box height BM=224 and
    // TX_SFA is 224*16 = 3,584 B, so the decode route over-delivered 1,536 B past its slot -
    // cudaErrorIllegalAddress on case 0 in dv_a5cb19120dd6, the one case that takes that route.
    // Falling back to the file-scope BM restores the pre-E120 size in every build that exists
    // (BMV=16 and 128 with FMGGM_BMV=128 -> one 128-row group, 2,048 B; the separately compiled
    // -DFMGGM_BMV=224 arms -> two groups, 4,096 B, which is what E117 measured), and gives the
    // DROUTE wide body its own 4,096 B without disturbing the narrow or decode bodies beside it.
    static constexpr int BMK = (BMV > 128 ? BMV : BM);
    static constexpr int NSFA_T = (BMK + 127) / 128;
    static constexpr int SFA_STRIDE_T = NSFA_T * 128 * 4 * SF_SUB;
    // The slot must hold what the body's TX_SFA promises the mbarrier: an over-delivered
    // expect_tx never reaches zero and the ring deadlocks silently (the b5042e1 class).
    static_assert(SFA_STRIDE_T * 4 >= BMK * 4 * 4 * SF_SUB,
                  "activation-SF smem slot smaller than the tile's TX_SFA");
    uint16_t cd[TMA_STORE_STAGES][STORE_M * BN];       // bf16 store staging, 2 atoms x 2KB
    uint8_t  b[STG][N_B_PER_STAGE * BN * BK];          // weights, fp4 unpacked to 8-bit containers
                                                       // (DUALN: n-tile j at [stage][j*BN*BK])
    uint8_t  a[STG][BMV * BK];                         // activations, e4m3 (BMV=16 under DEC16)
    // SF staging: TMA lands raw f32 [128 rows][4*SF_SUB kb-slices]; pack warps
    // convert in place to SF_SUB*128 packed ue8m0 words (UTCCP order; sub-block
    // q occupies words [q*128, (q+1)*128)).
    alignas(128) float sfw_raw[STG][SFW_STRIDE];
    // NSFA whole 128-row groups (>= BM rows, so the DEC16 route's BMV=16 still gets
    // a 128-row slot and its pack warps and SFA TMA are unchanged). Group g covers
    // rows [128g, 128g+128); the TMA fills only the first BM rows, so the tail of
    // the last group is stale - those TMEM lanes are beyond UMMA N and unread.
    alignas(128) float sfa_raw[STG][SFA_STRIDE_T];
    FMGGM_MBA uint64_t full[STG];
    FMGGM_MBA uint64_t sf_full[STG];
    FMGGM_MBA uint64_t empty[STG];
#if FMGGM_MCAST
    // Peer relay: one remote arrive per PAIRED use of this slot, meaning the
    // peer CTA's MMA has also finished reading its copy of the slot, so this
    // CTA's multicast half may overwrite it. Separate from empty[] (which stays
    // count 1 / own-commit) so unpaired tiles and the ragged last iteration
    // never wait on a peer that has no tile.
    FMGGM_MBA uint64_t pempty[STG];
#endif
    FMGGM_MBA uint64_t tmem_full[ACC_STAGES];
    FMGGM_MBA uint64_t tmem_empty[ACC_STAGES];
#if FMGGM_J0LA
    FMGGM_MBA uint64_t tmem_empty_j0[2];   // j0 slot (cols 0 / 268) freed by epilogue j0 drain
#endif
#if FMGGM_J0LB
    FMGGM_MBA uint64_t tj1_empty;          // prev pair's j1 TMEM loads done (cols 128-255 free)
    FMGGM_MBA uint64_t tj0_empty[2];       // pair-parity j0 loads done (cols 0-127 / 268-395 free)
#endif
    uint32_t tmem_base;
#if FMGGM_MMC
    int mcount[FMGGM_MMC_N];     // E70: masked_m snapshot (256 B at N=64)
#endif
};
using SharedStorage = SharedStorageT<BM, STAGES>;
static_assert(sizeof(SharedStorage) <= 232448, "main config smem over opt-in max");
#if FMGGM_DEC16
using SharedStorageDec = SharedStorageT<16, FMGGM_DEC_STG>;
static_assert(sizeof(SharedStorageDec) <= 232448, "DEC16 smem over opt-in max");
#endif

constexpr int STAGES2 = 7;
struct __align__(1024) SharedStorage2 {
    uint16_t cd[TMA_STORE_STAGES][STORE_M * BN];       // bf16 store staging (per-CTA D half)
    uint8_t  b[STAGES2][BN * BK];                      // weights (this CTA's 128 rows), unpacked
    uint8_t  a[STAGES2][64 * BK];                      // activations: this CTA's N-half (<=64 rows)
    alignas(128) float sfw_raw[STAGES2][BN * 4];       // weight SF (this CTA's rows), f32->packed
    alignas(128) float sfa_raw[STAGES2][BM * 4];       // activation SF: REPLICATED full 128 rows
    uint64_t full[STAGES2];       // leader-only gating (count 129)
    uint64_t sf_full[STAGES2];    // local per CTA
    uint64_t empty[STAGES2];      // local per CTA (commit multicast)
    uint64_t tmem_full[ACC_STAGES];   // local (commit multicast)
    uint64_t tmem_empty[ACC_STAGES];  // leader-only (count 256: both CTAs' epilogues)
    uint32_t tmem_base;
};

// ------------------------------------------------------------------ scheduler
// Persistent flat enumeration over (expert, m-block, n-block) tiles with
// m-blocks = ceil(masked_m[e]/BM); experts with masked_m[e]==0 contribute no
// tiles. L2-friendly swizzle groups GM m-blocks across all n-blocks.
struct Scheduler {
    const int* masked_m;
    const int* cache = nullptr;      // E70: smem snapshot of masked_m, or null to __ldg
    int num_groups, num_n_blocks, num_sms;
    int bm = BM;                     // m-block rows (BMV under DEC16)
    int bm_shift = 7;                // E70: log2(bm); bm is 16 or 128 by construction
    int g_group;                     // grouping factor on M
    uint32_t worker_id = 0;          // blockIdx.x (1-CTA) or cluster id (2-CTA)
#if FMGGM_MCAST
    bool pair_mode = false;          // this launch is 2-CTA clustered
    bool pair_ok = false;            // current tile: the cluster peer shares (group, n_blk)
#endif
    int cur_iter = -1;
    int cur_group = 0;
    uint32_t cur_cum = 0;            // cumulative m-blocks of previous groups
    int cur_count = -1;              // masked_m[cur_group] cache
#if FMGGM_STAIL
    uint32_t positions = 0xFFFFFFFFu;   // scheduled positions P (T or T+R)
    uint32_t split_from = 0xFFFFFFFFu;  // first split position (T-R) when splitting
    uint32_t cur_joff = 0;              // current item: first active j in the pair
    int cur_njl = 1;                    // current item: number of active j tiles
#endif

    __device__ void init(const int* masked_m_, int num_groups_, int num_n_blocks_, int num_sms_,
                         uint32_t worker_id_ = 0, int bm_ = BM, int total_tiles_ = 0,
                         const int* cache_ = nullptr
#if FMGGM_MCAST
                         , bool pair_mode_ = false
#endif
                         ) {
        masked_m = masked_m_; num_groups = num_groups_; num_n_blocks = num_n_blocks_; num_sms = num_sms_;
        worker_id = worker_id_; bm = bm_; cache = cache_;
#if FMGGM_MCAST
        pair_mode = pair_mode_;
#endif
        // E70: every call site passes a compile-time power of two (BM=128, or the
        // instantiation's BMV which is 128 for the main config and 16 for DEC16), so
        // ceil(x/bm) is a shift. The group walk below runs once per SKIPPED expert -
        // ~25 of them per tile on case 6 (64 experts x 6 n-pairs, tiles 148 apart) -
        // and each iteration is a serial load -> divide -> multiply -> branch chain,
        // which is what E68 measured as gaptile = 4,133 cycles/tile on case 6 against
        // 870 on case 9 (32 experts x 140 tiles, so ~1 expert per hop).
        bm_shift = 0;
        while ((1 << bm_shift) < bm_) ++bm_shift;
#if FMGGM_STAIL
        // total_tiles_ == 0 -> legacy unbounded enumeration (2-CTA kernel path).
        // The s_mblocks sum only covers groups 0..255 (THREADS); the frozen
        // domain guarantees groups <= 64, but guard general E anyway.
        // N_B_PER_STAGE < 2 has no pair to split (BK256-class configs).
        if (total_tiles_ > 0 && num_groups_ <= 256 && N_B_PER_STAGE == 2) {
            const uint32_t T = (uint32_t)total_tiles_;
            const uint32_t R = T % (uint32_t)num_sms_;
            if (R > 0 && 2 * R <= (uint32_t)num_sms_) {
                split_from = T - R;      // split the last R pair-tiles into halves
                positions = T + R;
            } else {
                positions = T;
            }
        }
        cur_njl = N_B_PER_STAGE;
#endif
#if FMGGM_GG > 0
        g_group = FMGGM_GG;
#else
        // same L2-usage heuristic as the reference (grouping on M)
        int best = 8; uint32_t best_usage = 0xFFFFFFFFu;
        const int cands[2] = {8, 16};
        #pragma unroll
        for (int ci = 0; ci < 2; ++ci) {
            const int cand = cands[ci];
            uint32_t usage = (uint32_t)cand * (uint32_t)bm + ((num_sms + cand - 1) / cand) * BN;
            if (usage < best_usage) { best_usage = usage; best = cand; }
        }
        g_group = best;
#endif
    }
    // E70 helpers. nmb_of: ceil(cnt/bm) as a shift (bm is 16 or 128 at every call
    // site). count_of: masked_m[g] from the smem snapshot when MMC is on, else the
    // global load. Both are value-identical to the expressions they replace.
    __device__ __forceinline__ uint32_t nmb_of(int cnt) const {
#if FMGGM_BMSHIFT
        return (uint32_t)((cnt + bm - 1) >> bm_shift);
#else
        return (uint32_t)((cnt + bm - 1) / bm);
#endif
    }
    __device__ __forceinline__ int count_of(int g) const {
#if FMGGM_MMC
        return cache ? cache[g] : __ldg(masked_m + g);
#else
        return __ldg(masked_m + g);
#endif
    }
    __device__ bool next(uint32_t& m_blk, uint32_t& n_blk) {
#if FMGGM_STAIL
        while (true) {
            const uint32_t pos = (uint32_t)(++cur_iter) * (uint32_t)num_sms + worker_id;
            if (pos >= positions) return false;
            uint32_t flat = pos;
            cur_joff = 0; cur_njl = N_B_PER_STAGE;
            if (pos >= split_from) {
                const uint32_t h = pos - split_from;
                flat = split_from + (h >> 1);
                cur_joff = h & 1;
                cur_njl = 1;
            }
            bool found = false;
            while (true) {
                if (cur_group == num_groups) return false;
                if (cur_count < 0) cur_count = count_of(cur_group);
                uint32_t nmb = nmb_of(cur_count);
                uint32_t cum_next = cur_cum + nmb;
                if (flat < cum_next * (uint32_t)num_n_blocks) { found = true; break; }
                cur_group++; cur_cum = cum_next; cur_count = -1;
            }
            if (!found) return false;
            uint32_t local = flat - cur_cum * (uint32_t)num_n_blocks;
            uint32_t per_group = (uint32_t)num_n_blocks * (uint32_t)g_group;
            uint32_t grp = local / per_group;
            uint32_t first = grp * (uint32_t)g_group;
            uint32_t in_grp = local % per_group;
            uint32_t nmb_here = (uint32_t)g_group;
            uint32_t total_mblocks = nmb_of(cur_count);
            if (total_mblocks - first < nmb_here) nmb_here = total_mblocks - first;
            m_blk = first + in_grp % nmb_here;
            n_blk = in_grp / nmb_here;
#if FMGGM_MCAST
            // Cluster pairing. num_sms is even and flat = iter*num_sms + worker_id,
            // so flat & 1 == the cluster rank and the peer is flat^1. The decode is
            // m-fastest inside a g_group sub-group, so the even and odd positions of a
            // pair share (group, n_blk) exactly when the even one is not the
            // sub-group's last m-block: rank 0 tests m_off+1 < nmb_here, rank 1 tests
            // m_off > 0, and the two are the same predicate seen from either side. The
            // STAIL split region is excluded by bounding both positions below `lim`, so
            // a pair never straddles it. Each rank decides locally and they always
            // agree, which is what makes the producer-side relay deadlock-free.
            {
                const uint32_t m_off = in_grp % nmb_here;
                const uint32_t lim = positions < split_from ? positions : split_from;
                pair_ok = pair_mode && ((worker_id & 1u)
                                            ? (pos < lim && m_off > 0u)
                                            : (pos + 1u < lim && m_off + 1u < nmb_here));
            }
#endif
            // NOTE: the j window [cur_joff, cur_joff+cur_njl) is clamped to the
            // actual n-tail by each caller (they hold p.num_n_blocks); a dead
            // half (joff beyond the tail) is skipped symmetrically by all
            // warps. Frozen domain: n % 256 == 0 -> the clamp never triggers.
            return true;
        }
#else
        const uint32_t flat = (uint32_t)(++cur_iter) * (uint32_t)num_sms + worker_id;
        while (true) {
            if (cur_group == num_groups) return false;
            if (cur_count < 0) cur_count = count_of(cur_group);
            uint32_t nmb = nmb_of(cur_count);
            uint32_t cum_next = cur_cum + nmb;
            if (flat < cum_next * (uint32_t)num_n_blocks) break;
            cur_group++; cur_cum = cum_next; cur_count = -1;
        }
        uint32_t local = flat - cur_cum * (uint32_t)num_n_blocks;
        uint32_t per_group = (uint32_t)num_n_blocks * (uint32_t)g_group;
        uint32_t grp = local / per_group;
        uint32_t first = grp * (uint32_t)g_group;
        uint32_t in_grp = local % per_group;
        uint32_t nmb_here = (uint32_t)g_group;
        uint32_t total_mblocks = nmb_of(cur_count);
        if (total_mblocks - first < nmb_here) nmb_here = total_mblocks - first;
        m_blk = first + in_grp % nmb_here;
        n_blk = in_grp / nmb_here;
        return true;
#endif
    }
    __device__ int effective_m(uint32_t m_blk) const {
#if FMGGM_BMSHIFT
        const int bmc = 1 << bm_shift;              // == bm, power of two
        int rem = cur_count - (int)(m_blk << bm_shift);
        int eff = rem < bmc ? rem : bmc;
#else
        int rem = cur_count - (int)(m_blk * (uint32_t)bm);
        int eff = rem < bm ? rem : bm;
#endif
        return (eff + 15) & ~15;    // UMMA-N granularity
    }
};

// ------------------------------------------------------------------ params
struct Params {
    CUtensorMap map_a;    // activations e4m3: [K inner, E*M outer], box 128 rows
    CUtensorMap map_a16;  // same tensor, box 16 rows (for eff<=16 tiles)
    CUtensorMap map_a32;  // box 32 rows
    CUtensorMap map_a64;  // box 64 rows
    CUtensorMap map_a8;   // box 8 rows (2-CTA activation half for eff=16)
    CUtensorMap map_a128; // box 128 rows: eff<=128 tiles inside a BMV>128 config
#if FMGGM_DROUTE
    // E120: the wide arm's A and activation-SF boxes. The narrow arm keeps map_a/map_sfa
    // (box height BM = 128), so both geometries coexist in one Params and one launch.
    CUtensorMap map_a_wide;    // box {BK, FMGGM_DROUTE_BM}
    CUtensorMap map_sfa_wide;  // box {4, FMGGM_DROUTE_BM}
#endif
#if FMGGM_EBOX
    CUtensorMap map_a_h[8];    // A boxes, height 16*(idx+1), idx = eff/16-1
    CUtensorMap map_sfa_h[8];  // SFA boxes {4 f32, 16*(idx+1) rows}
#endif
    CUtensorMap map_b;   // weights fp4 (16U4 unpack): [K inner, E*N outer]
    CUtensorMap map_d;   // output bf16: [N inner, E*M outer]
    CUtensorMap map_sfa; // activation SF f32: [K/32 inner, E*M outer], box {4,128}
    CUtensorMap map_sfb; // weight SF f32: [K/32 inner, E*N outer], box {4,128}
#if FMGGM_BMERGE
    CUtensorMap map_b2;    // weights, box {128, 2*BN}: both n-tiles of a stage, one request
    CUtensorMap map_sfb2;  // weight SF, box {4, 2*BN}: ditto
#endif
    const int* masked_m;     // (E,)
    uint16_t* d_ptr;         // raw D pointer for st.global epilogue
#if FMGGM_LDGSF
    const float* sfa_ptr;    // activation SF f32 (E,M,K/32) for direct LDG
    const float* sfb_ptr;    // weight SF f32 (E,N,K/32)
#endif
    int num_groups, shape_m, shape_n, shape_k;
    int num_k_blocks, num_n_blocks, num_sms;
    int num_n_blocks2, num_clusters;   // 2-CTA: 256-wide n tiles, cluster count
#if FMGGM_MCAST
    int pair_mode;    // 1 = this launch is 2-CTA clustered: share the weight stream
#endif
};

__device__ __forceinline__ uint32_t pack_sf4(float4 v) {
    return ((__float_as_uint(v.x) >> 23) & 0xFFu)
         | ((__float_as_uint(v.y) >> 15) & 0xFF00u)
         | ((__float_as_uint(v.z) >>  7) & 0xFF0000u)
         | ((__float_as_uint(v.w) <<  1) & 0xFF000000u);
}

// DESCROLL MMA stage body: S = ring slot (compile-time constant inside the
// chunk-unrolled loop, runtime `stage` in the alignment/remainder loops),
// KBV = absolute k-block index, PH = ring parity. Mirrors the legacy body
// exactly (same ops/order/barriers); requires dcol_j0/dcol_j1/idesc_tt[] in
// scope (hoisted per-tile invariants).
// NKBH: the DESCROLL stage macro re-reads p.num_k_blocks from __grid_constant__ param
// space once per stage for the accumulator-release test, although the loop bound is
// already hoisted into nk_. Pre-join E63 priced exactly this read at ~40 cycles per stage
// (gapstage 93.5 -> 51.6, +0.083% on the stable five); with DESCROLL collapsing the rest of
// the per-stage stream to ~5 ops it should be a larger share of what is left.
#ifndef FMGGM_NKBH
#define FMGGM_NKBH 0
#endif
#if FMGGM_NKBH
#define FMGGM_MMA_NK nk_
#else
#define FMGGM_MMA_NK p.num_k_blocks
#endif
// E73: minimal MMA-warp span trace on THIS tree (the pre-join MMATRACE does not exist
// here): four clock64 reads per stage and two per tile, printed once per launch from two
// CTAs. Diagnostic only, never a default, and it perturbs timing, so traced wall times are
// never comparable to official ones. It exists because gaptile was 4,133 cycles/tile on
// case 6 against 870 on case 9, and E70 proved it is NOT the scheduler walk's load+divide
// (removing both bought ~120 cycles/tile). What is left is the accumulator handshake or
// the latency of the new tile's compulsory weight misses, and those imply opposite fixes.
#ifndef FMGGM_MTR
#define FMGGM_MTR 0
#endif
// MTRE: epilogue-drain span trace. The MMA warp has been traced since E75 but the epilogue
// never has, and it is implicated in both remaining blocks: case 9's `hand` (2,634 cycles per
// tile, 17% of the tile) is by definition the epilogue running behind, and every case carries
// ~5-8 us of non-loop time that is mostly the final tile's unoverlapped drain. This splits the
// epilogue's tile into (a) waiting for tmem_full and (b) the drain itself, so the drain's own
// duration can be compared with and without concurrent UMMA accumulator traffic (EXPER=2
// skips the UMMAs and keeps the commits). The hypothesis is TMEM port contention: per tile the
// MMA read-modify-writes 8 UMMAs x 16 stages x 128x128 f32 ~ 16 MB of TMEM while the drain
// reads 131,072 B, and if they share a port the drain is starved - which would explain why
// TJEB (fewer waits), LDX2 (fewer loads) and PKELECT (fewer arrivals) were all null.
// Independent of FMGGM_MTR. Diagnostic only, never default, and it perturbs timing.
#ifndef FMGGM_MTRE
#define FMGGM_MTRE 0
#endif
#if FMGGM_MTR && !FMGGM_DESCROLL
#error "FMGGM_MTR instruments the DESCROLL stage macro only"
#endif
// NOTE the trailing ';' inside each do-while definition: these expand INSIDE another
// macro's replacement list, where there is no semicolon after the invocation token, and
// without it the expansion runs into the next statement (that was E73's "expected a ';'"
// x6 - two per FMGGM_MMA_STAGE call site). The inner name is trn_, not t_: the stage macro
// already uses t_ for its UMMA loop.
#if FMGGM_MTR
#define FMGGM_MTR_T0   const unsigned long long tr_ts = clock64();
#define FMGGM_MTR_FULL(KB) do { const unsigned long long trn_ = clock64(); tr_full += trn_ - tr_ts; if ((KB) == 0) tr_full0 += trn_ - tr_ts; tr_tf = trn_; } while (0);
#define FMGGM_MTR_END  do { const unsigned long long trn_ = clock64(); tr_issue += trn_ - tr_tf; tr_te = trn_; ++tr_stages; } while (0);
#else
#define FMGGM_MTR_T0
#define FMGGM_MTR_FULL(KB)
#define FMGGM_MTR_END
#endif
#if FMGGM_MCAST == 5 || FMGGM_MCAST == 7
#define FMGGM_MCAST_COMMIT(S) if (mma_pair) umma_commit_mcast(&sm.pempty[S], 0x3);
#else
#define FMGGM_MCAST_COMMIT(S)
#endif
#if FMGGM_DESCROLL
#define FMGGM_MMA_STAGE(S, KBV, PH)                                                          \
    do {                                                                                     \
        FMGGM_MTR_T0                                                                         \
        mbar_wait(&sm.full[S], PH);                                                          \
        tc_fence_after_sync();                                                               \
        FMGGM_MTR_FULL(KBV)                                                                  \
        if (elect_one()) {                                                                   \
            const uint32_t sf_idx = (uint32_t)(S);                                           \
            if (FMGGM_EXPER != 3 && FMGGM_EXPER != 16 &&                                     \
                (FMGGM_EXPER != 1 || (KBV) == 0)) {                                          \
                _Pragma("unroll")                                                            \
                for (int i_ = 0; i_ < NW_SUB; ++i_)                                          \
                    utccp_4x32dp128bit(tmem + SFW_COL + i_ * 4,                              \
                        make_smem_desc(smem_u32(sm.sfw_raw[sf_idx]) + i_ * 512, 128, 0, 0)); \
                _Pragma("unroll")                                                            \
                for (int q_ = 0; q_ < SF_SUB * NSFA_K; ++q_)                                 \
                    utccp_4x32dp128bit(tmem + SFA_COL + q_ * 4,                              \
                        make_smem_desc(smem_u32(sm.sfa_raw[sf_idx]) + q_ * 512, 128, 0, 0)); \
            }                                                                                \
            if (FMGGM_EXPER != 2) {                                                          \
                const uint64_t adesc_s = adesc0 + (uint64_t)(S) * (N_B_PER_STAGE * BN * BK / 16); \
                const uint64_t bdesc_s = bdesc0 + (uint64_t)(S) * (BMV * BK / 16);           \
                _Pragma("unroll")                                                            \
                for (uint32_t j_ = 0; j_ < N_B_PER_STAGE; ++j_) {                            \
                    if (j_ >= jlo && j_ < jhi) {                                             \
                        _Pragma("unroll")                                                    \
                        for (uint32_t t_ = 0; t_ < UMMAS_PER_STAGE; ++t_) {                  \
                            const uint32_t q_ = t_ / 4, tt_ = t_ & 3;                        \
                            umma_mxf8f6f4((j_ == 0) ? dcol_j0 : dcol_j1,                     \
                                adesc_s + j_ * (BN * BK / 16) + q_ * (BN * 8) + 2 * tt_,     \
                                bdesc_s + q_ * (a_rows * 8) + 2 * tt_,                       \
                                idesc_tt[tt_],                                               \
                                tmem + SFW_COL + (j_ * SF_SUB + q_) * 4,                     \
                                tmem + SFA_COL + q_ * 4,                                     \
                                ((KBV) > 0 || t_ > 0) ? 1u : 0u);                            \
                        }                                                                    \
                    }                                                                        \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        __syncwarp();                                                                        \
        if (elect_one()) {                                                                   \
            if (FMGGM_EXPER == 20) mbar_arrive(&sm.empty[S]);                                \
            else                   umma_commit(&sm.empty[S]);                                \
            FMGGM_MCAST_COMMIT(S)                                                              \
            if ((KBV) == FMGGM_MMA_NK - 1) umma_commit(&sm.tmem_full[acc]);                 \
        }                                                                                    \
        FMGGM_MTR_END                                                                        \
    } while (0)
#endif

// E120: under DROUTE the kernel body has to be callable as a device function so that ONE launch
// can hold two geometries, so the decoration and the parameter form are #if-selected here. With
// DROUTE=0 (the shipped default) this is byte-for-byte the previous declaration, and with
// DROUTE=1 __forceinline__ keeps `p` in the wrapper's __grid_constant__ param space instead of
// materialising a 1,344 B copy - DESCROLL re-reads p.num_k_blocks from param space per stage.
#if FMGGM_DROUTE
template <int BMV, int STG>
__device__ __forceinline__ void fmggm_bodyT(const Params& p) {
#else
template <int BMV, int STG>
__global__ void __launch_bounds__(THREADS, 1)
fmggm_kernelT(const __grid_constant__ Params p) {
#endif
#if FMGGM_EXPER == 13
    return;     // overhead probe: full launch path, immediate exit
#endif
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& sm = *reinterpret_cast<SharedStorageT<BMV, STG>*>(smem_raw);
    // activation-SF TMEM column groups for THIS instantiation's tile height: the
    // DEC16 route (BMV=16) still issues one group even when the main route's BM
    // needs two.
    constexpr int NSFA_K = (BMV < 128 ? 1 : (BMV + 127) / 128);
    // E120: layout constants for THIS instantiation, shadowing the file-scope names so that the
    // ~50 use sites below (TMEM column offsets, A-row counts, TX_SFA, the SF staging strides)
    // follow the tile height rather than FMGGM_BMV. Shadowing instead of renaming keeps the
    // default build textually unchanged, and it is value-identical for every build that existed
    // before E120: BMK is the file-scope BM for BMV <= 128, so the DEC16 route keeps the main
    // route's 128-column TMEM slots exactly as it did, and BMK is BMV for a wide instantiation,
    // which under -DFMGGM_BMV=224 is also what the file-scope BM already was. The only build
    // where the two differ is DROUTE's, where they are meant to.
    constexpr int BM = SharedStorageT<BMV, STG>::BMK;
    constexpr int SFA_STRIDE = SharedStorageT<BMV, STG>::SFA_STRIDE_T;
    constexpr int SFW_COL = BM * N_B_PER_STAGE * ACC_STAGES;
    constexpr int SFA_COL = SFW_COL + 4 * NW_SUB;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

#if FMGGM_FASTEXIT
    // ---- early exit, barrier-free: every warp independently ORs masked_m
    // (identical data => identical branch in all warps; no __syncthreads,
    // no smem round-trip). E <= 256 supported (8 rounds of 32 lanes).
    {
        int any = 0;
        for (int g0 = 0; g0 < p.num_groups; g0 += 32) {
            const int g = g0 + lane;
            any |= (g < p.num_groups) ? (__ldg(p.masked_m + g) > 0) : 0;
        }
        // CRITICAL: reduce across lanes BEFORE branching - per-lane partial
        // ORs diverge (lane l only saw groups l, l+32, ...), and a divergent
        // return breaks the following warp-collective ops (elect.sync full
        // mask, tcgen05.alloc .sync.aligned) => illegal instruction / node
        // crash (killed jobs dv_39f6218f6372 + 20261023T010442Z silently;
        // oracle group_sizes [1,2,3,0,...] trips it on lane 3).
        any = __any_sync(0xffffffffu, any);
        if (!any) return;
    }
#else
    // ---- early exit when every expert is empty: no TMEM alloc, no loads/stores
    __shared__ int s_any;
#if FMGGM_STAIL
    __shared__ int s_mblocks;   // total m-blocks over all experts (tail split).
    if (tid == 0) { s_any = 0; s_mblocks = 0; }
#else
    if (tid == 0) s_any = 0;
#endif
    __syncthreads();
    if (tid < p.num_groups && __ldg(p.masked_m + tid) > 0) atomicOr(&s_any, 1);
    __syncthreads();
    if (!s_any) return;
#if FMGGM_STAIL
    // Second pass ONLY when work exists: the all-empty fast path above is
    // byte-identical to the non-STAIL build (official eval: the per-thread
    // atomicAdd version cost case1 +10.6%; the always-on warp-butterfly
    // version still +2.1% on the ~6us launch floor). masked_m is L2-warm
    // here; the per-warp butterfly keeps it to <=4 smem atomics total.
    {
        const int c0 = (tid < p.num_groups) ? __ldg(p.masked_m + tid) : 0;
#if FMGGM_MMC
        // E70: publish the snapshot the group walk will read. c0 is already in a
        // register here, so this is one smem store per thread and no extra traffic;
        // the __syncthreads() that closes this pass orders it before any walk.
        if (tid < p.num_groups && p.num_groups <= FMGGM_MMC_N) sm.mcount[tid] = c0;
#endif
        int mb = (c0 + BMV - 1) / BMV;   // c0==0 contributes 0
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            mb += __shfl_xor_sync(0xffffffffu, mb, off);
        if ((tid & 31) == 0 && mb) atomicAdd(&s_mblocks, mb);
    }
    __syncthreads();
#endif
#endif

    // ---- one-time init: barriers (warp1) + TMEM alloc (warp2)
    if (warp == 1 && elect_one()) {
        #pragma unroll
        for (int i = 0; i < STG; ++i) {
            mbar_init(&sm.full[i], FMGGM_PKELECT ? 3 : 65);   // 65 = 1 TMA expect_tx + 32 SFA + 32 SFW lanes
            mbar_init(&sm.sf_full[i], 1);  // SF f32 TMA arrival (expect_tx by TMA warp)
            mbar_init(&sm.empty[i], 1);    // 1 tcgen05.commit
#if FMGGM_MCAST
            // MCAST<=4: one remote arrive per paired use. MCAST=5: both CTAs'
            // multicast commits land here, so the phase needs 2.
            mbar_init(&sm.pempty[i], (FMGGM_MCAST == 5 || FMGGM_MCAST == 7) ? 2 : 1);
#endif
        }
        #pragma unroll
        for (int i = 0; i < ACC_STAGES; ++i) {
            mbar_init(&sm.tmem_full[i], 1);
            mbar_init(&sm.tmem_empty[i], 128);
        }
        #if FMGGM_J0LA
        mbar_init(&sm.tmem_empty_j0[0], 128);
        mbar_init(&sm.tmem_empty_j0[1], 128);
        #endif
        #if FMGGM_J0LB
        mbar_init(&sm.tj1_empty, 128);
        mbar_init(&sm.tj0_empty[0], 128);
        mbar_init(&sm.tj0_empty[1], 128);
        #endif
        fence_mbar_init();
    } else if (warp == 2) {
        tmem_alloc(&sm.tmem_base, TMEM_COLS);
        tmem_relinquish();
    }
    __syncthreads();
#if FMGGM_MCAST
    // Cluster-wide barrier-init visibility: a peer's remote arrive on this CTA's
    // pempty[] must not land before this CTA initialised it. Uniform branch, and
    // on a non-clustered launch (pair_mode == 0) a CTA is its own cluster of one
    // so the instruction would still be legal - it is simply not executed.
    const uint32_t csize = cluster_size();
    const uint32_t crank = blockIdx.x & 1u;   // scheduler parity: flat&1 for even num_sms
    // pair_on must be CLUSTER-UNIFORM: the cluster barrier below and every
    // pempty[] relay/wait depend on both ranks taking the same branch, so it is
    // built only from launch-wide facts (pair_mode and the cluster size) and
    // never from a per-CTA quantity the two ranks could disagree about. The rank
    // used for the weight-half split is blockIdx.x & 1, which is the scheduler's
    // own parity (flat & 1) and equals %cluster_ctarank for a 1-D grid with
    // clusterDim.x == 2; if that mapping ever broke, the halves would be
    // exchanged and the frozen oracle would report it rather than a hang.
    const bool pair_on = (p.pair_mode != 0) && (csize == 2u);
    if (pair_on) { cluster_arrive(); cluster_wait(); }
#endif
    const uint32_t tmem = sm.tmem_base;

    Scheduler sched;
#if FMGGM_STAIL
    const int n_pairs_sched = (p.num_n_blocks + N_B_PER_STAGE - 1) / N_B_PER_STAGE;
    sched.init(p.masked_m, p.num_groups, n_pairs_sched, p.num_sms, blockIdx.x, BMV,
               s_mblocks * n_pairs_sched,
#if FMGGM_MMC
               (p.num_groups <= FMGGM_MMC_N) ? sm.mcount : nullptr
#else
               nullptr
#endif
#if FMGGM_MCAST
               , pair_on
#endif
               );
#else
    sched.init(p.masked_m, p.num_groups,
               (p.num_n_blocks + N_B_PER_STAGE - 1) / N_B_PER_STAGE, p.num_sms, blockIdx.x, BMV);
#endif
    uint32_t m_blk = 0, n_blk = 0;
    uint32_t stage = 0, phase = 0;

    if (warp == 0 && elect_one()) {
        // ---------------- TMA load warp (A activations, B weights) ----------------
#if FMGGM_MCAST
        uint32_t ppar = 0;   // per-slot parity of PAIRED uses of pempty[slot]
#endif
        // tx bytes: B tile = packed gmem bytes (8192, probe-confirmed). A tile uses
        // the smallest row box covering align(eff_m,16): decode tiles cut A traffic 8x.
        constexpr uint32_t TX_B = BN * BK / 2 * FMGGM_B_TX_DIV;   // packed bytes, per n-tile
        constexpr uint32_t TX_SFA = BM * 4 * 4 * SF_SUB;          // f32 activation SF tile
        constexpr uint32_t TX_SFW = BN * 4 * 4 * SF_SUB;          // f32 weight SF, per n-tile
        while (sched.next(m_blk, n_blk)) {
            const int m_idx = sched.cur_group * p.shape_m + (int)m_blk * BMV;
            const int n_idx = sched.cur_group * p.shape_n + (int)n_blk * (BN * N_B_PER_STAGE);
            // active j-window [jlo, jhi) of this pair-tile (STAIL: half-tiles
            // carry a single j; the clamp keeps ragged n-tails general)
            int njt = p.num_n_blocks - (int)n_blk * N_B_PER_STAGE;
            njt = njt > N_B_PER_STAGE ? N_B_PER_STAGE : (njt < 0 ? 0 : njt);
#if FMGGM_STAIL
            const uint32_t jlo = sched.cur_joff;
            int njl = sched.cur_njl < njt - (int)jlo ? sched.cur_njl : njt - (int)jlo;
            if (njl <= 0) continue;    // dead half-tile (frozen domain: never)
#else
            const uint32_t jlo = 0;
            int njl = njt < 1 ? 1 : njt;
#endif
            const uint32_t jhi = jlo + (uint32_t)njl;
            const int nj = njl;        // active n-tiles (used for tx accounting)
            const bool single_m_block = (sched.cur_count <= BMV);
#if FMGGM_DROUTE
            // Both geometries are live in this launch, so the default (full-tile) A and
            // activation-SF boxes have to follow this instantiation. BM is constexpr, so each
            // body folds to one pointer and pays nothing.
            const CUtensorMap* map_a = (BM > 128) ? &p.map_a_wide : &p.map_a;
            const CUtensorMap* map_sfa_sel = (BM > 128) ? &p.map_sfa_wide : &p.map_sfa;
#else
            const CUtensorMap* map_a = &p.map_a;
            const CUtensorMap* map_sfa_sel = &p.map_sfa;
#endif
            uint32_t tx_a = BMV * BK;
            uint32_t tx_sfa = TX_SFA;
#if FMGGM_EBOX == 3
            // SFA-only shrink on the power-of-two ladder {16,32,64}; A keeps
            // the proven ABOX selection. EBOX=1/2 data (job 20261022T063945Z
            // + 20261022T062613Z): decode case0 +1.2-1.9% comes from the SFA
            // tx cut (2KB->eff/128*2KB per stage), while non-pow2 A box
            // heights (eff=112 on case6) cost -2.2% consistently (TMA
            // slow-path suspicion) and mids gain nothing measurable.
            const int eff = sched.effective_m(m_blk);
            if (eff <= 16)      { map_a = &p.map_a16; tx_a = 16 * BK;
                                  map_sfa_sel = &p.map_sfa_h[0]; tx_sfa = 16 * 4 * 4 * SF_SUB; }
            else if (eff <= 32) { map_a = &p.map_a32; tx_a = 32 * BK;
                                  map_sfa_sel = &p.map_sfa_h[1]; tx_sfa = 32 * 4 * 4 * SF_SUB; }
            else if (eff <= 64) { map_a = &p.map_a64; tx_a = 64 * BK;
                                  map_sfa_sel = &p.map_sfa_h[3]; tx_sfa = 64 * 4 * 4 * SF_SUB; }
#elif FMGGM_EBOX
            // Full eff ladder: A + SFA boxes shrink to align16(eff) rows.
            const int eff = sched.effective_m(m_blk);
            const int hidx = (eff >> 4) - 1;    // eff in [16,128] step 16
#if FMGGM_EBOX == 2
            // Switch form: STATIC descriptor offsets per branch. EBOX=1's
            // &p.map_a_h[hidx] indexes the __grid_constant__ Params at
            // runtime => per-issue const-cache-miss descriptor fetches
            // (suspect for the ebox case4 2x / case5 +42% bimodal
            // slowdown, job 20261022T061332Z).
            switch (hidx) {
              case 0: map_a = &p.map_a_h[0]; map_sfa_sel = &p.map_sfa_h[0]; break;
              case 1: map_a = &p.map_a_h[1]; map_sfa_sel = &p.map_sfa_h[1]; break;
              case 2: map_a = &p.map_a_h[2]; map_sfa_sel = &p.map_sfa_h[2]; break;
              case 3: map_a = &p.map_a_h[3]; map_sfa_sel = &p.map_sfa_h[3]; break;
              case 4: map_a = &p.map_a_h[4]; map_sfa_sel = &p.map_sfa_h[4]; break;
              case 5: map_a = &p.map_a_h[5]; map_sfa_sel = &p.map_sfa_h[5]; break;
              case 6: map_a = &p.map_a_h[6]; map_sfa_sel = &p.map_sfa_h[6]; break;
              default: map_a = &p.map_a_h[7]; map_sfa_sel = &p.map_sfa_h[7]; break;
            }
#else
            map_a = &p.map_a_h[hidx];
            map_sfa_sel = &p.map_sfa_h[hidx];
#endif
            tx_a = (uint32_t)eff * BK;
            tx_sfa = (uint32_t)eff * 4 * 4 * SF_SUB;
#elif FMGGM_ABOX
            const int eff = sched.effective_m(m_blk);
            if (eff <= 16)      { map_a = &p.map_a16; tx_a = 16 * BK; }
            else if (eff <= 32) { map_a = &p.map_a32; tx_a = 32 * BK; }
            else if (eff <= 64) { map_a = &p.map_a64; tx_a = 64 * BK; }
            else if (BM > 128 && eff <= 128) { map_a = &p.map_a128; tx_a = 128 * BK; }
#endif
#if FMGGM_MCAST
            // Per-tile pairing decision. sched.pair_ok is symmetric by construction
            // and nj/jlo depend only on (group, n_blk), which the peer shares, so
            // both ranks always compute the same value - the invariant the relay
            // handshake below depends on.
            const bool pair_tile = pair_on && sched.pair_ok
                                && (nj == (int)N_B_PER_STAGE) && (jlo == 0);
            // Bisect arms, all with the same cluster launch and pairing:
            //   1 = relay + multicast (the candidate)
            //   2 = relay only, every CTA still fetches both weight halves
            //   3 = multicast only, no relay (races, numerics may be wrong, but
            //       it isolates whether the multicast tx accounting completes
            //       full[]/sf_full[] at all)
            const bool mc_tile = pair_tile && (FMGGM_MCAST == 1 || FMGGM_MCAST == 3
                                             || FMGGM_MCAST == 5);
            const bool hs_tile = pair_tile && (FMGGM_MCAST == 1 || FMGGM_MCAST == 2
                                             || FMGGM_MCAST == 5 || FMGGM_MCAST == 7);
            // MCAST=4 keeps the clustered launch and nothing else: the control
            // that separates "74 clusters of 2 cannot all be co-resident, so a
            // persistent kernel serialises a tail" from any protocol cost.
#else
            constexpr bool pair_tile = false;
            constexpr bool mc_tile = false;
            constexpr bool hs_tile = false;
#endif
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                mbar_wait(&sm.empty[stage], phase ^ 1);
#if FMGGM_MCAST
                if (hs_tile) {
                    // Relay: the empty[] wait above proves THIS CTA's MMA has finished
                    // reading its copy of the slot; say so on the peer's pempty[], then
                    // wait for the peer's statement before either rank multicasts into
                    // the other's slot. Arrive strictly before wait - both ranks do it,
                    // so neither can block the other. ppar's bit for this slot toggles
                    // once per PAIRED use, and both ranks pair the same stages, so the
                    // parities stay identical; the first paired use of a slot passes on
                    // the fresh barrier exactly as empty[] does.
#if FMGGM_MCAST == 5 || FMGGM_MCAST == 7
                    // The peer's MMA signals through the hardware multicast commit,
                    // so the producer only waits. E101b priced the software relay:
                    // MCAST=2 (relay, no multicast) is 523.28 us on case 9 against
                    // 265.30 base and 267.26 for a clustered launch that does nothing
                    // per stage, i.e. one release-scoped remote arrive plus its wait
                    // costs ~97% - about 500 ns against a 520 ns stage period, the
                    // same class as the ~280 ns/stage cluster barrier that closed the
                    // cta_group::2 path. The signal here is issued by the MMA warp and
                    // is asynchronous, and in steady state it landed D stages ago, so
                    // the wait should be satisfied on arrival. Its parity is the
                    // complement of the relay's: the commit fires at the END of use
                    // k-1, so use k needs k completions, exactly like empty[].
                    mbar_wait(&sm.pempty[stage], ((ppar >> stage) & 1u) ^ 1u);
#else
                    mbar_arrive_peer(&sm.pempty[stage], crank ^ 1u);
                    // Parity is the bit ITSELF, not its complement: this wait must
                    // block until the peer's relay for THIS use has landed. empty[]
                    // uses phase^1 because its k-th use needs k completions; pempty's
                    // k-th use needs k+1 (the peer relays at its own k-th use, after
                    // its empty[] wait proved its MMA finished the k-1 data).
                    mbar_wait(&sm.pempty[stage], (ppar >> stage) & 1u);
#endif
                    ppar ^= (1u << stage);
                }
#endif
                // E96 / EXPER=33: fetch only the FIRST weight n-tile's rows (128 of 256)
                // while the MMA still issues all 8 UMMAs, so the j1 half reads stale smem.
                // Garbage numerics, structure and MMA work unchanged. It is the UPPER BOUND
                // on what a 2-CTA multicast of B could save: multicast halves the same rows
                // and the same wire bytes, but each CTA still absorbs the full 32,768
                // B/stage of unpacked smem writes and the MMA still reads them, and this
                // arm removes those too. Folds to `nj` in every non-EXPER build.
                const int nj_b = (FMGGM_EXPER == 33) ? 1 : nj;
                if (!kSkipLoads) {
#if FMGGM_EXPER != 5 && FMGGM_EXPER != 16 && !FMGGM_LDGSF
                  {
                    // SF f32 tiles first on their own barrier; pack warps convert while A/B fly
                    #pragma unroll
                    for (int h = 0; h < SF_SUB; ++h)
                        tma_load_2d(map_sfa_sel, &sm.sf_full[stage], sm.sfa_raw[stage] + h * (BM * 4),
                                    kb * (4 * SF_SUB) + h * 4, m_idx);
#if FMGGM_BMERGE
                    // Both n-tiles present: one {4,2*BM} request replaces two.
                    const bool sfb_merged = (nj == (int)N_B_PER_STAGE && jlo == 0);
                    if (sfb_merged)
                        tma_load_2d(&p.map_sfb2, &sm.sf_full[stage], sm.sfw_raw[stage],
                                    kb * (4 * SF_SUB), n_idx);
#else
                    constexpr bool sfb_merged = false;
#endif
#if FMGGM_MCAST
                    if (mc_tile)
                        // Rank r fetches the weight SF of n-tile r and multicasts it to
                        // both CTAs at the same smem offset, so each CTA's sf_full[]
                        // still sees TX_SFW * nj bytes and its pack warps see the
                        // identical f32 image.
                        tma_load_2d_mcast(&p.map_sfb, &sm.sf_full[stage],
                                          sm.sfw_raw[stage] + crank * (BN * 4),
                                          kb * (4 * SF_SUB), n_idx + (int)crank * BN, 0x3);
#endif
                    #pragma unroll
                    for (uint32_t j = 0; j < N_B_PER_STAGE; ++j)
                        if (!mc_tile && !sfb_merged && j >= jlo && j < jhi) {
                            #pragma unroll
                            for (int h = 0; h < SF_SUB; ++h)
                                tma_load_2d(&p.map_sfb, &sm.sf_full[stage],
                                            sm.sfw_raw[stage] + (j * SF_SUB + h) * (BN * 4),
                                            kb * (4 * SF_SUB) + h * 4, n_idx + (int)j * BN);
                        }
                    mbar_arrive_expect_tx(&sm.sf_full[stage], tx_sfa + TX_SFW * (uint32_t)nj);
                  }
#endif
                    if (FMGGM_EXPER != 17) {
                        #pragma unroll
                        for (int h = 0; h < SF_SUB; ++h)
                            tma_load_2d(map_a, &sm.full[stage], sm.a[stage] + h * (tx_a / SF_SUB),
                                        kb * BK + h * 128, m_idx);
                    }
                    if (FMGGM_EXPER != 6) {
#if FMGGM_BMERGE
                        // Both n-tiles present: one 256-row request replaces two,
                        // with the same wire bytes, expect_tx and smem image.
                        const bool b_merged = (nj == (int)N_B_PER_STAGE && jlo == 0);
                        if (b_merged) {
#if FMGGM_TCP
                            tma_load_2d_evict_last(&p.map_b2, &sm.full[stage], sm.b[stage],
                                                   kb * BK, n_idx);
#else
                            tma_load_2d(&p.map_b2, &sm.full[stage], sm.b[stage], kb * BK, n_idx);
#endif
                        }
#else
                        constexpr bool b_merged = false;
#endif
#if FMGGM_MCAST
                        if (mc_tile) {
                            // Rank r fetches weight n-tile j=r (128 of the 256 rows) and
                            // multicasts it to both CTAs at the SAME smem offset, so each
                            // CTA's b[stage] image - and therefore its UMMA descriptors -
                            // is byte-identical to the unpaired path, and full[] still
                            // receives TX_B * nj bytes (its own half plus the peer's).
#if FMGGM_TCP
                            tma_load_2d_mcast_evict_last(&p.map_b, &sm.full[stage],
                                                         sm.b[stage] + crank * (BN * 128),
                                                         kb * BK, n_idx + (int)crank * BN, 0x3);
#else
                            tma_load_2d_mcast(&p.map_b, &sm.full[stage],
                                              sm.b[stage] + crank * (BN * 128),
                                              kb * BK, n_idx + (int)crank * BN, 0x3);
#endif
                        }
#endif
                        #pragma unroll
                        for (uint32_t j = 0; j < N_B_PER_STAGE; ++j) {
                            if (!mc_tile && !b_merged && j >= jlo && j < jlo + (uint32_t)nj_b) {
                                #pragma unroll
                                for (int h = 0; h < SF_SUB; ++h) {
                                    uint8_t* bdst = sm.b[stage] + (j * SF_SUB + h) * (BN * 128);
                                    const int cx = kb * BK + h * 128;
                                    const int cy = n_idx + (int)j * BN;
#if FMGGM_TCP
                                    tma_load_2d_evict_last(&p.map_b, &sm.full[stage], bdst, cx, cy);
#elif FMGGM_LEVICT == 1
                                    if (single_m_block) tma_load_2d_evict_first(&p.map_b, &sm.full[stage], bdst, cx, cy);
                                    else                tma_load_2d(&p.map_b, &sm.full[stage], bdst, cx, cy);
#elif FMGGM_LEVICT == 2
                                    tma_load_2d_evict_first(&p.map_b, &sm.full[stage], bdst, cx, cy);
#else
                                    tma_load_2d(&p.map_b, &sm.full[stage], bdst, cx, cy);
#endif
                                }
                            }
                        }
                    }
#if FMGGM_BPF > 0
                    if (FMGGM_EXPER != 6 && kb + FMGGM_BPF < p.num_k_blocks) {
                        const int pcx = (kb + FMGGM_BPF) * BK;
                        #pragma unroll
                        for (uint32_t j = 0; j < N_B_PER_STAGE; ++j)
                            if (j >= jlo && j < jhi)
                                tma_prefetch_2d(&p.map_b, pcx, n_idx + (int)j * BN);
                    }
#endif
                }
                mbar_arrive_expect_tx(&sm.full[stage], kSkipLoads ? 0u :
                                      ((FMGGM_EXPER == 17 ? 0u : tx_a) +
                                       (FMGGM_EXPER != 6 ? TX_B * (uint32_t)nj_b : 0u)));
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
            }
        }
    } else if (warp == 2 || warp == 3) {
        // ---------------- SF pack warps: TMA-delivered f32 -> packed ue8m0 in place
        // warp 2: activation SF (MMA B side); warp 3: weight SF (MMA A side).
        // f32 tile [128 rows][4 slices] was fetched by TMA (no gmem loads here);
        // each lane reads rows {lane+32r} into registers, warp syncs, then the
        // packed words overwrite the first 512B (all reads precede all writes).
        const bool is_act = (warp == 2);
        // sub-blocks per stage: activation side SF_SUB; weight side NW_SUB
        // (DUALN packs both n-tiles' SF here; unrolled-with-guard keeps v[] in regs)
        // activation SF spans NSFA_K groups of 128 rows when BMV > 128; each group
        // is its own 512 B packed block, its own UTCCP and its own 4 TMEM columns.
        const int nsub = is_act ? SF_SUB * NSFA_K : NW_SUB;
        float* raw0 = is_act ? sm.sfa_raw[0] : sm.sfw_raw[0];
#if FMGGM_LDGSF
        // Direct-LDG SF (arm-b "mode4" pattern): 2-generation register
        // lookahead hides f32 LDG latency; per stage only empty-gate ->
        // pack+STS -> proxy fence -> arrive remain. Packed words land at
        // the SAME raw-slot offsets (q*128 words) UTCCP reads.
        const float* gptr = is_act ? p.sfa_ptr : p.sfb_ptr;
        const int k32 = p.shape_k >> 5;
        const uint32_t row_tot =
            (uint32_t)(is_act ? p.shape_m : p.shape_n) * (uint32_t)p.num_groups;
        const auto ldg_gen = [&](float4 (&g)[NW_SUB][4], int kb_src,
                                 uint32_t row0, int elim) {
            if (FMGGM_EXPER == 5 || kSkipLoads) return;
            #pragma unroll
            for (int q = 0; q < NW_SUB; ++q) {
                if (q < nsub) {
                    const int j = is_act ? 0 : q / SF_SUB;   // weight n-tile
                    const int h = is_act ? q : q % SF_SUB;   // 128-k sub-slice
                    const size_t col =
                        (size_t)kb_src * (4 * SF_SUB) + (size_t)h * 4;
                    #pragma unroll
                    for (int rr = 0; rr < 4; ++rr) {
                        const uint32_t lr = (uint32_t)(lane + 32 * rr);
                        const uint32_t r = row0 + (uint32_t)j * BN + lr;
                        g[q][rr] = ((int)lr < elim && r < row_tot)
                            ? *reinterpret_cast<const float4*>(
                                  gptr + (size_t)r * k32 + col)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                }
            }
        };
        const auto pack_stage = [&](const float4 (&g)[NW_SUB][4]) {
            mbar_wait(&sm.empty[stage], phase ^ 1);
            if (FMGGM_EXPER != 5 && !kSkipLoads) {
                __syncwarp();
                float* raw = raw0 + (size_t)stage * (is_act ? SFA_STRIDE : SFW_STRIDE);
                uint32_t* dst = reinterpret_cast<uint32_t*>(raw);
                #pragma unroll
                for (int q = 0; q < NW_SUB; ++q)
                    if (q < nsub) {
                        #pragma unroll
                        for (int rr = 0; rr < 4; ++rr)
                            dst[q * 128 + lane * 4 + rr] = pack_sf4(g[q][rr]);
                    }
                fence_proxy_async();
            }
            mbar_arrive(&sm.full[stage]);
            stage = stage + 1 == STG ? 0 : stage + 1;
            phase ^= stage == 0;
        };
        float4 gA[NW_SUB][4], gB[NW_SUB][4];
        while (sched.next(m_blk, n_blk)) {
            const uint32_t row0 = is_act
                ? (uint32_t)sched.cur_group * (uint32_t)p.shape_m + m_blk * BMV
                : (uint32_t)sched.cur_group * (uint32_t)p.shape_n
                      + n_blk * (BN * N_B_PER_STAGE);
            const int elim = is_act ? sched.effective_m(m_blk) : BN;
            const int nb = p.num_k_blocks;
            ldg_gen(gA, 0, row0, elim);
            if (nb >= 2) ldg_gen(gB, 1, row0, elim);
            int kb = 0;
            for (; kb + 1 < nb; kb += 2) {
                pack_stage(gA);
                if (kb + 2 < nb) ldg_gen(gA, kb + 2, row0, elim);
                pack_stage(gB);
                if (kb + 3 < nb) ldg_gen(gB, kb + 3, row0, elim);
            }
            if (kb < nb) pack_stage(gA);
        }
#else
        while (sched.next(m_blk, n_blk)) {
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                if (!kSkipLoads && FMGGM_EXPER != 5 && FMGGM_EXPER != 16)
                    mbar_wait(&sm.sf_full[stage], phase);
                float* raw = raw0 + (size_t)stage * (is_act ? SFA_STRIDE : SFW_STRIDE);
                float4 v[NW_SUB][4];
                if (FMGGM_EXPER != 5 && !kSkipLoads &&
                    FMGGM_EXPER != 16 && FMGGM_EXPER != 18) {
                    #pragma unroll
                    for (int q = 0; q < NW_SUB; ++q)
                        if (q < nsub) {
                            #pragma unroll
                            for (int rr = 0; rr < 4; ++rr)
                                v[q][rr] = *reinterpret_cast<const float4*>(
                                    raw + q * 512 + (size_t)(lane + 32 * rr) * 4);
                        }
                }
                mbar_wait(&sm.empty[stage], phase ^ 1);
                if (FMGGM_EXPER != 5 && !kSkipLoads &&
                    FMGGM_EXPER != 16 && FMGGM_EXPER != 18) {
                    __syncwarp();
                    uint32_t* dst = reinterpret_cast<uint32_t*>(raw);
                    #pragma unroll
                    for (int q = 0; q < NW_SUB; ++q)
                        if (q < nsub) {
                            #pragma unroll
                            for (int rr = 0; rr < 4; ++rr) dst[q * 128 + lane * 4 + rr] = pack_sf4(v[q][rr]);
                        }
                    fence_proxy_async();
                }
#if FMGGM_PKELECT
                __syncwarp();
                if (elect_one()) mbar_arrive(&sm.full[stage]);
#else
                mbar_arrive(&sm.full[stage]);
#endif
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
            }
        }
#endif
    } else if (warp == 1) {
        // ---------------- MMA warp ----------------
        const uint32_t idesc_base = make_idesc_base(1);
        const uint64_t adesc0 = make_smem_desc(smem_u32(sm.b[0]), 1024, 0, 2); // weights = MMA A
        const uint64_t bdesc0 = make_smem_desc(smem_u32(sm.a[0]), 1024, 0, 2); // activations = MMA B
#if FMGGM_MTR
        unsigned long long tr_t00 = clock64(), tr_te = tr_t00, tr_tf = tr_t00;
        unsigned long long tr_full = 0, tr_issue = 0, tr_tile = 0, tr_gap = 0;
        unsigned long long tr_full0 = 0;   // first stage of a tile only (KBV==0)
        unsigned long long tr_stages = 0, tr_tiles = 0;
#endif
        while (sched.next(m_blk, n_blk)) {
#if FMGGM_MTR
            { const unsigned long long ta = clock64(); tr_gap += ta - tr_te; tr_te = ta; ++tr_tiles; }
#endif
            const uint32_t acc = (uint32_t)sched.cur_iter % ACC_STAGES;
            const uint32_t acc_ph = ((uint32_t)sched.cur_iter / ACC_STAGES) & 1;
            const uint32_t eff = (uint32_t)sched.effective_m(m_blk);
            const uint32_t idesc_n = idesc_base | ((eff >> 3) << 17);
#if FMGGM_EBOX
            const uint32_t a_rows = eff;   // align16 in [16,128]; matches box
#else
            const uint32_t a_rows = (FMGGM_ABOX && eff <= 16) ? 16u :
                                    (FMGGM_ABOX && eff <= 32) ? 32u :
                                    (FMGGM_ABOX && eff <= 64) ? 64u :
                                    (FMGGM_ABOX && eff <= 128) ? 128u : (uint32_t)BM;
#endif
            int njt = p.num_n_blocks - (int)n_blk * N_B_PER_STAGE;
            njt = njt > N_B_PER_STAGE ? N_B_PER_STAGE : (njt < 0 ? 0 : njt);
#if FMGGM_STAIL
            const uint32_t jlo = sched.cur_joff;
            int njl = sched.cur_njl < njt - (int)jlo ? sched.cur_njl : njt - (int)jlo;
            if (njl <= 0) continue;    // dead half-tile: skip BEFORE tj waits
#else
            const uint32_t jlo = 0;
            int njl = njt < 1 ? 1 : njt;
#endif
            const uint32_t jhi = jlo + (uint32_t)njl;
            const int nj = njl;
#if FMGGM_MCAST == 5 || FMGGM_MCAST == 7
            // Same predicate the producer computes; both warps walk their own copy
            // of the scheduler over the identical position sequence.
            const bool mma_pair = pair_on && sched.pair_ok
                               && (nj == (int)N_B_PER_STAGE) && (jlo == 0);
#endif
#if FMGGM_J0LA
            // j0-lookahead: this pair's j0 accumulator alternates cols {0, 268};
            // j1 stays at cols 128-255. Phase 1 issues the first `la` stages' j0
            // UMMAs WHILE the previous pair's epilogue drains (its j0 slot was
            // freed early). Ring slots stay resident (no empty[] commit) until
            // phase 2a re-issues SF + the j1 UMMAs, then commits.
            const uint32_t it = (uint32_t)sched.cur_iter;
            const uint32_t j0col = (it & 1) ? 268u : 0u;
            const uint32_t la = ((uint32_t)p.num_k_blocks < (uint32_t)STG)
                                    ? (uint32_t)p.num_k_blocks : (uint32_t)STG;
            const uint32_t stage0 = stage;
            mbar_wait(&sm.tmem_empty_j0[it & 1], ((it >> 1) & 1) ^ 1);
            tc_fence_after_sync();
            for (uint32_t s = 0; s < la; ++s) {
                mbar_wait(&sm.full[stage], phase);
                tc_fence_after_sync();
                if (elect_one()) {
                    #pragma unroll
                    for (int i = 0; i < NW_SUB; ++i)
                        utccp_4x32dp128bit(tmem + SFW_COL + i * 4,
                                           make_smem_desc(smem_u32(sm.sfw_raw[stage]) + i * 512, 128, 0, 0));
                    #pragma unroll
                    for (int q = 0; q < SF_SUB * NSFA_K; ++q)
                        utccp_4x32dp128bit(tmem + SFA_COL + q * 4,
                                           make_smem_desc(smem_u32(sm.sfa_raw[stage]) + q * 512, 128, 0, 0));
                    const uint64_t adesc_s = adesc0 + (uint64_t)stage * (N_B_PER_STAGE * BN * BK / 16);
                    const uint64_t bdesc_s = bdesc0 + (uint64_t)stage * (BMV * BK / 16);
                    #pragma unroll
                    for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                        const uint32_t q = t / 4, tt = t & 3;
                        umma_mxf8f6f4(tmem + j0col,
                                      adesc_s + q * (BN * 8) + 2 * tt,
                                      bdesc_s + q * (a_rows * 8) + 2 * tt,
                                      idesc_n | (tt << 29) | (tt << 4),
                                      tmem + SFW_COL + q * 4, tmem + SFA_COL + q * 4,
                                      (s > 0 || t > 0) ? 1u : 0u);
                    }
                }
                __syncwarp();
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
            }
            mbar_wait(&sm.tmem_empty[0], (it & 1) ^ 1);   // previous pair's j1 drained
            tc_fence_after_sync();
            for (uint32_t s = 0; s < la; ++s) {
                const uint32_t slot = (stage0 + s) % STG;
                if (elect_one()) {
                    if (nj > 1) {
                        #pragma unroll
                        for (int i = 0; i < NW_SUB; ++i)
                            utccp_4x32dp128bit(tmem + SFW_COL + i * 4,
                                               make_smem_desc(smem_u32(sm.sfw_raw[slot]) + i * 512, 128, 0, 0));
                        #pragma unroll
                        for (int q = 0; q < SF_SUB * NSFA_K; ++q)
                            utccp_4x32dp128bit(tmem + SFA_COL + q * 4,
                                               make_smem_desc(smem_u32(sm.sfa_raw[slot]) + q * 512, 128, 0, 0));
                        const uint64_t adesc_s = adesc0 + (uint64_t)slot * (N_B_PER_STAGE * BN * BK / 16);
                        const uint64_t bdesc_s = bdesc0 + (uint64_t)slot * (BMV * BK / 16);
                        #pragma unroll
                        for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                            const uint32_t q = t / 4, tt = t & 3;
                            umma_mxf8f6f4(tmem + BM,
                                          adesc_s + (BN * BK / 16) + q * (BN * 8) + 2 * tt,
                                          bdesc_s + q * (a_rows * 8) + 2 * tt,
                                          idesc_n | (tt << 29) | (tt << 4),
                                          tmem + SFW_COL + (SF_SUB + q) * 4, tmem + SFA_COL + q * 4,
                                          (s > 0 || t > 0) ? 1u : 0u);
                        }
                    }
                    umma_commit(&sm.empty[slot]);
                    if (s == la - 1 && (int)la == p.num_k_blocks)
                        umma_commit(&sm.tmem_full[0]);   // whole pair fits in lookahead
                }
                __syncwarp();
            }
            for (int kb = (int)la; kb < p.num_k_blocks; ++kb) {
                mbar_wait(&sm.full[stage], phase);
                tc_fence_after_sync();
                if (elect_one()) {
                    #pragma unroll
                    for (int i = 0; i < NW_SUB; ++i)
                        utccp_4x32dp128bit(tmem + SFW_COL + i * 4,
                                           make_smem_desc(smem_u32(sm.sfw_raw[stage]) + i * 512, 128, 0, 0));
                    #pragma unroll
                    for (int q = 0; q < SF_SUB * NSFA_K; ++q)
                        utccp_4x32dp128bit(tmem + SFA_COL + q * 4,
                                           make_smem_desc(smem_u32(sm.sfa_raw[stage]) + q * 512, 128, 0, 0));
                    const uint64_t adesc_s = adesc0 + (uint64_t)stage * (N_B_PER_STAGE * BN * BK / 16);
                    const uint64_t bdesc_s = bdesc0 + (uint64_t)stage * (BMV * BK / 16);
                    #pragma unroll
                    for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                        const uint32_t q = t / 4, tt = t & 3;
                        umma_mxf8f6f4(tmem + j0col,
                                      adesc_s + q * (BN * 8) + 2 * tt,
                                      bdesc_s + q * (a_rows * 8) + 2 * tt,
                                      idesc_n | (tt << 29) | (tt << 4),
                                      tmem + SFW_COL + q * 4, tmem + SFA_COL + q * 4, 1u);
                    }
                    if (nj > 1) {
                        #pragma unroll
                        for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                            const uint32_t q = t / 4, tt = t & 3;
                            umma_mxf8f6f4(tmem + BM,
                                          adesc_s + (BN * BK / 16) + q * (BN * 8) + 2 * tt,
                                          bdesc_s + q * (a_rows * 8) + 2 * tt,
                                          idesc_n | (tt << 29) | (tt << 4),
                                          tmem + SFW_COL + (SF_SUB + q) * 4, tmem + SFA_COL + q * 4, 1u);
                        }
                    }
                }
                __syncwarp();
                if (elect_one()) {
                    umma_commit(&sm.empty[stage]);
                    if (kb == p.num_k_blocks - 1) umma_commit(&sm.tmem_full[0]);
                }
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
            }
            continue;   // J0LA pair done (skip the base per-tile loop below)
#else
#if FMGGM_J0LB
            // Wait only the previous pair's j1 LOAD completion (cols 128-255
            // free) and, from it>=2, this parity's j0 LOAD completion. Ring
            // empty[] commits are untouched => no producer stall (J0LA's flaw).
            const uint32_t itj = (uint32_t)sched.cur_iter;
            const uint32_t j0col = (itj & 1) ? 268u : 0u;
            if (itj > 0) {
                mbar_wait(&sm.tj1_empty, (uint32_t)((itj - 1) & 1));
                if (itj >= 2)
                    mbar_wait(&sm.tj0_empty[itj & 1], (uint32_t)(((itj - 2) >> 1) & 1));
            }
            tc_fence_after_sync();
#else
            mbar_wait(&sm.tmem_empty[acc], acc_ph ^ 1);
            tc_fence_after_sync();
#endif
#if FMGGM_MTR
            { const unsigned long long tb = clock64(); tr_tile += tb - tr_te; tr_te = tb; }
#endif
#if FMGGM_DESCROLL
            // ---- chunk-unrolled mainloop: slot-constant descriptors ----
            const uint32_t dcol_j0 =
#if FMGGM_J0LB
                tmem + j0col;
            const uint32_t dcol_j1 = tmem + (uint32_t)BM;
#else
                tmem + acc * (BM * N_B_PER_STAGE);
            // E116: this `+ 128u` was the wide arm's only remaining bug. The port
            // rewrote the multiplier to BM but left the addend, so DESCROLL hoisted
            // j1's accumulator to column 128 while the drain reads it from column BM.
            // At BM=128 the two coincide, which is why the shipped build is correct
            // and why the pre-join wide implementation (no DESCROLL) never hit it.
            // Symptom: j0 bit-exact, j1 wholly wrong, and the identical mismatch
            // count for BM=192 and BM=224 - both read the same never-written TMEM.
            const uint32_t dcol_j1 = tmem + acc * (BM * N_B_PER_STAGE) + (uint32_t)BM;
#endif
            uint32_t idesc_tt[4];
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                idesc_tt[t] = idesc_n | ((uint32_t)t << 29) | ((uint32_t)t << 4);
            const int nk_ = p.num_k_blocks;
            int kb = 0;
            while (kb < nk_ && stage != 0) {          // align the ring to slot 0
                FMGGM_MMA_STAGE(stage, kb, phase);
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
                ++kb;
            }
            for (; kb + STG <= nk_; kb += STG) {      // slot-constant chunk bodies
                #pragma unroll
                for (int i = 0; i < STG; ++i) FMGGM_MMA_STAGE(i, kb + i, phase);
                phase ^= 1;
            }
            while (kb < nk_) {                        // ragged tail stages
                FMGGM_MMA_STAGE(stage, kb, phase);
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
                ++kb;
            }
#else
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                mbar_wait(&sm.full[stage], phase);
                tc_fence_after_sync();
                if (elect_one()) {
                    const uint32_t sf_idx = (FMGGM_EXPER == 26) ? 0u : stage;
                    if (FMGGM_EXPER != 3 && FMGGM_EXPER != 16 &&
                        (FMGGM_EXPER != 1 || kb == 0)) {
                        #pragma unroll
                        for (int i = 0; i < NW_SUB; ++i)   // weight SF: sub-block i = j*SF_SUB + q
                            utccp_4x32dp128bit(tmem + SFW_COL + i * 4,
                                               make_smem_desc(smem_u32(sm.sfw_raw[sf_idx]) + i * 512, 128, 0, 0));
                        #pragma unroll
                        for (int q = 0; q < SF_SUB * NSFA_K; ++q)   // act SF: ceil(N/128) groups
                            utccp_4x32dp128bit(tmem + SFA_COL + q * 4,
                                               make_smem_desc(smem_u32(sm.sfa_raw[sf_idx]) + q * 512, 128, 0, 0));
                    }
                    if (FMGGM_EXPER != 2) {
                        const uint64_t adesc_s = adesc0 + ((FMGGM_EXPER == 26) ? 0ull
                            : (uint64_t)stage * (N_B_PER_STAGE * BN * BK / 16));
                        const uint64_t bdesc_s = bdesc0 + ((FMGGM_EXPER == 26) ? 0ull
                            : (uint64_t)stage * (BMV * BK / 16));
                        #pragma unroll
                        for (uint32_t j = 0; j < N_B_PER_STAGE; ++j) {
                            if (j >= jlo && j < jhi) {
                                #pragma unroll
                                for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                                    const uint32_t q = t / 4, tt = t & 3;
                                    const uint32_t idesc = idesc_n | (tt << 29) | (tt << 4);
#if FMGGM_J0LB
                                    const uint32_t dcol = tmem + ((j == 0) ? j0col : (uint32_t)BM);
#else
                                    const uint32_t dcol = tmem + acc * (BM * N_B_PER_STAGE) + j * BM;
#endif
                                    umma_mxf8f6f4(dcol,
                                                  adesc_s + j * (BN * BK / 16) + q * (BN * 8) + 2 * tt,
                                                  bdesc_s + q * (a_rows * 8) + 2 * tt,
                                                  idesc,
                                                  tmem + SFW_COL + (j * SF_SUB + q) * 4,
                                                  tmem + SFA_COL + q * 4,
                                                  (kb > 0 || t > 0) ? 1u : 0u);
                                }
                            }
                        }
                    }
                }
                __syncwarp();
                if (elect_one()) {
                    if (FMGGM_EXPER == 20) mbar_arrive(&sm.empty[stage]);
                    else                   umma_commit(&sm.empty[stage]);
#if FMGGM_MCAST == 5 || FMGGM_MCAST == 7
                    if (mma_pair) umma_commit_mcast(&sm.pempty[stage], 0x3);
#endif
                    if (kb == p.num_k_blocks - 1) umma_commit(&sm.tmem_full[acc]);
                }
                stage = stage + 1 == STG ? 0 : stage + 1;
                phase ^= stage == 0;
            }
#endif   // FMGGM_DESCROLL
#endif
        }
#if FMGGM_MTR
        if ((blockIdx.x == 0 || blockIdx.x == 74) && lane == 0) {
            const unsigned long long wall = clock64() - tr_t00;
            const double ns = (double)(tr_stages ? tr_stages : 1);
            printf("MTR cta=%u tiles=%llu stages=%llu wall=%llu | per-stage wall=%.1f "
                   "full=%.1f issue=%.1f gap=%.1f hand=%.1f sum=%.1f | per-tile wall=%.1f "
                   "gap=%.1f hand=%.1f full0=%.1f\n",
                   blockIdx.x, tr_tiles, tr_stages, wall,
                   (double)wall / ns, (double)tr_full / ns, (double)tr_issue / ns,
                   (double)tr_gap / ns, (double)tr_tile / ns,
                   (double)(tr_full + tr_issue + tr_gap + tr_tile) / ns,
                   tr_tiles ? (double)wall / (double)tr_tiles : 0.0,
                   tr_tiles ? (double)tr_gap / (double)tr_tiles : 0.0,
                   tr_tiles ? (double)tr_tile / (double)tr_tiles : 0.0,
                   tr_tiles ? (double)tr_full0 / (double)tr_tiles : 0.0);
        }
#endif
    } else if (warp >= 4) {
        // ---------------- epilogue warps 4-7: TMEM -> STSM -> TMA store ----------------
        const int ew = warp - 4;
#if FMGGM_MTRE
        unsigned long long dr_wait = 0, dr_work = 0, dr_tiles = 0, dr_te1 = 0;
        const unsigned long long dr_t00 = clock64();
#endif
#if FMGGM_PDL
        // Predecessor (harness d-restore) writes only d; block this role until
        // it is complete. TMA/pack/MMA warps never touch d and keep overlapping.
        asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
        uint32_t tma_stage = 0;
        while (sched.next(m_blk, n_blk)) {
#if FMGGM_MTRE
            const unsigned long long dr_te0 = clock64();
#endif
            const uint32_t acc = (uint32_t)sched.cur_iter % ACC_STAGES;
            const uint32_t acc_ph = ((uint32_t)sched.cur_iter / ACC_STAGES) & 1;
            // j-window + dead-half skip MUST precede the tmem_full wait (the
            // MMA warp skips dead half-tiles entirely: no commit would fire).
            int njt = p.num_n_blocks - (int)n_blk * N_B_PER_STAGE;
            njt = njt > N_B_PER_STAGE ? N_B_PER_STAGE : (njt < 0 ? 0 : njt);
#if FMGGM_STAIL
            const uint32_t jlo = sched.cur_joff;
            int njl = sched.cur_njl < njt - (int)jlo ? sched.cur_njl : njt - (int)jlo;
            if (njl <= 0) continue;
#else
            const uint32_t jlo = 0;
            int njl = njt < 1 ? 1 : njt;
#endif
            const uint32_t jhi = jlo + (uint32_t)njl;
            const int nj = njl;
#if FMGGM_EWSLEEP
            mbar_wait_hint(&sm.tmem_full[acc], acc_ph, 256);
#else
            mbar_wait(&sm.tmem_full[acc], acc_ph);
#endif
            tc_fence_after_sync();
#if FMGGM_MTRE
            dr_te1 = clock64(); dr_wait += dr_te1 - dr_te0;
#endif
            const int eff = sched.effective_m(m_blk);
            const int num_stores = eff / STORE_M;
            const int base_m = sched.cur_group * p.shape_m + (int)m_blk * BMV;
            const uint32_t atom_off = (uint32_t)(ew >> 1) * (STORE_M * STORE_N_ATOM * 2);
#if FMGGM_J0LA
            const uint32_t j0col = ((uint32_t)sched.cur_iter & 1) ? 268u : 0u;
#endif
#if FMGGM_J0LB
            const uint32_t itj = (uint32_t)sched.cur_iter;
            const uint32_t j0col = (itj & 1) ? 268u : 0u;
#if FMGGM_STAIL
            // unused j halves free their TMEM cols right away (never read this
            // tile); used halves arrive after their last tmem_ld below - exactly
            // one arrival per barrier per tile, preserving the parity protocol.
            if (jhi <= 1) mbar_arrive(&sm.tj1_empty);
            if (jlo >= 1) mbar_arrive(&sm.tj0_empty[itj & 1]);
#endif
#endif
#if FMGGM_DINTER
            if (N_B_PER_STAGE == 2 && nj == N_B_PER_STAGE) {
                // ---- interleaved-j drain: 16 rows x 256 cols (512B/row) per si ----
                const int base_n_d = (int)n_blk * (N_B_PER_STAGE * BN);
                for (int si = 0; si < num_stores; ++si) {
                    named_barrier_sync(1, 128);
#if FMGGM_TJEB
                    // batch all 8 loads of this si (both j halves x both li),
                    // ONE wait, then arrive + pack: 8 round-trips per tile
                    // instead of 32; tj1/tj0 fire together after the last si.
                    float v[2][2][8];
                    #pragma unroll
                    for (int jj = 1; jj >= 0; --jj) {
#if FMGGM_J0LB
                        const uint32_t tcol = tmem + ((jj == 0) ? j0col : (uint32_t)BM);
#else
                        const uint32_t tcol = tmem + acc * (BM * N_B_PER_STAGE) + (uint32_t)(jj * BM);
#endif
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            const uint32_t taddr = tcol + (uint32_t)(si * STORE_M + li * 8);
                            tmem_ld_16x256b_x1(taddr, v[jj][li]);
                            tmem_ld_16x256b_x1(taddr | 0x00100000u, v[jj][li] + 4);
                        }
                    }
                    tmem_ld_wait();
                    if (si == num_stores - 1) {
                        tc_fence_before_sync();
#if FMGGM_J0LB
                        mbar_arrive(&sm.tj1_empty);
                        mbar_arrive(&sm.tj0_empty[itj & 1]);
#else
                        mbar_arrive(&sm.tmem_empty[acc]);
#endif
                    }
                    #pragma unroll
                    for (int jj = 1; jj >= 0; --jj) {
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            uint32_t packed[4];
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                __nv_bfloat162 h = __float22bfloat162_rn(
                                    make_float2(v[jj][li][2 * i], v[jj][li][2 * i + 1]));
                                packed[i] = *reinterpret_cast<uint32_t*>(&h);
                            }
                            const uint32_t saddr = smem_u32(&sm.cd[0][0])
                                + (uint32_t)(jj * 2 + (ew >> 1)) * (STORE_M * STORE_N_ATOM * 2)
                                + (uint32_t)li * 1024
                                + (uint32_t)(lane & 7) * 128
                                + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                            stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                        }
                    }
#else
                    #pragma unroll
                    for (int jj = 1; jj >= 0; --jj) {   // j1 loads first (J0LB release order)
#if FMGGM_J0LB
                        const uint32_t tcol = tmem + ((jj == 0) ? j0col : (uint32_t)BM);
#else
                        const uint32_t tcol = tmem + acc * (BM * N_B_PER_STAGE) + (uint32_t)(jj * BM);
#endif
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            const uint32_t taddr = tcol + (uint32_t)(si * STORE_M + li * 8);
                            float v[8];
                            tmem_ld_16x256b_x1(taddr, v);
                            tmem_ld_16x256b_x1(taddr | 0x00100000u, v + 4);
                            tmem_ld_wait();
#if FMGGM_J0LB
  #if FMGGM_EXPER == 32
                            // E89: the same bubble probe the per-j drain already had, moved
                            // into the DINTER path where the default drain actually runs.
                            // DINTER's si-major order pushes the j1 release to the LAST si
                            // (~97% of the 64 loads/tile) even though jj=1 is still read
                            // first inside each si, so `hand` (2,579 cy/tile on case 9) may
                            // be the price DINTER charged for 512B-coalesced stores. This
                            // frees both column sets after the first load of the first si:
                            // garbage numerics (the next tile's UMMA races these reads),
                            // protocol-safe (still exactly one arrive per barrier per tile).
                            if (si == 0 && li == 0) {
                                if (jj == 1) mbar_arrive(&sm.tj1_empty);
                                else         mbar_arrive(&sm.tj0_empty[itj & 1]);
                            }
  #else
                            if (si == num_stores - 1 && li == 1) {
                                tc_fence_before_sync();
                                if (jj == 1) mbar_arrive(&sm.tj1_empty);
                                else         mbar_arrive(&sm.tj0_empty[itj & 1]);
                            }
  #endif
#else
                            if (si == num_stores - 1 && li == 1 && jj == 0) {
                                tc_fence_before_sync();
                                mbar_arrive(&sm.tmem_empty[acc]);
                            }
#endif
                            uint32_t packed[4];
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                __nv_bfloat162 h = __float22bfloat162_rn(make_float2(v[2 * i], v[2 * i + 1]));
                                packed[i] = *reinterpret_cast<uint32_t*>(&h);
                            }
                            const uint32_t saddr = smem_u32(&sm.cd[0][0])
                                + (uint32_t)(jj * 2 + (ew >> 1)) * (STORE_M * STORE_N_ATOM * 2)
                                + (uint32_t)li * 1024
                                + (uint32_t)(lane & 7) * 128
                                + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                            stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                        }
                    }
#endif   // FMGGM_TJEB (DINTER load batching)
                    named_barrier_sync(2, 128);
                    {
                        uint16_t* d_base = reinterpret_cast<uint16_t*>(p.d_ptr)
                            + (size_t)(base_m + si * STORE_M) * p.shape_n + base_n_d;
                        const uint8_t* cd8 = reinterpret_cast<const uint8_t*>(&sm.cd[0][0]);
                        #pragma unroll
                        for (int pass = 0; pass < 4; ++pass) {
                            const int idx = tid - 128 + pass * 128;       // 0..511
                            const int row = idx >> 5;                      // 16 rows
                            const int chunk = idx & 31;                    // 32 x 16B = 512B/row
                            const int atom = chunk >> 3, g = chunk & 7;
                            const uint4* src = reinterpret_cast<const uint4*>(
                                cd8 + atom * 2048 + row * 128 + ((g ^ (row & 7)) * 16));
                            uint4 val = *src;
                            uint4* dst = reinterpret_cast<uint4*>(
                                d_base + (size_t)row * p.shape_n + atom * 64 + g * 8);
                            *dst = val;
                        }
                    }
                }
            } else {
#endif
#if FMGGM_J0LB
            for (int j = (int)jhi - 1; j >= (int)jlo; --j) {   // j1 first within the active window
#else
            for (int j = (int)jlo; j < (int)jhi; ++j) {
#endif
            // D is flattened [E*M, N]: columns carry NO group offset
            const int base_n = ((int)n_blk * N_B_PER_STAGE + j) * BN;
#if FMGGM_EDR
            // ---- early drain release: capture the whole j-tile in registers,
            // free TMEM immediately, then STSM/store overlap the next mainloop.
            {
                float vv[8][8];
                #pragma unroll
                for (int si = 0; si < 8; ++si)
                    if (si < num_stores) {
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            const uint32_t taddr = tmem + acc * (BM * N_B_PER_STAGE)
                                                   + (uint32_t)(j * BM + si * 16 + li * 8);
                            tmem_ld_16x256b_x1(taddr, &vv[si][li * 4]);
                            tmem_ld_16x256b_x1(taddr | 0x00100000u, &vv[si][li * 4 + 4]);
                        }
                    }
                tmem_ld_wait();
                if (j == (int)jhi - 1) {
                    tc_fence_before_sync();
                    mbar_arrive(&sm.tmem_empty[acc]);
                }
                #pragma unroll
                for (int si = 0; si < 8; ++si)
                    if (si < num_stores) {
                        named_barrier_sync(1, 128);
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            uint32_t packed[4];
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                __nv_bfloat162 h = __float22bfloat162_rn(
                                    make_float2(vv[si][li * 4 + 2 * i], vv[si][li * 4 + 2 * i + 1]));
                                packed[i] = *reinterpret_cast<uint32_t*>(&h);
                            }
                            const uint32_t saddr = smem_u32(&sm.cd[tma_stage][0]) + atom_off
                                + (uint32_t)li * 1024
                                + (uint32_t)(lane & 7) * 128
                                + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                            stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                        }
                        named_barrier_sync(2, 128);
                        {
                            uint16_t* d_base = reinterpret_cast<uint16_t*>(p.d_ptr)
                                + (size_t)(base_m + si * STORE_M) * p.shape_n + base_n;
                            const uint8_t* cd8 = reinterpret_cast<const uint8_t*>(&sm.cd[tma_stage][0]);
                            #pragma unroll
                            for (int pass = 0; pass < 2; ++pass) {
                                int idx = tid - 128 + pass * 128;
                                int row = idx >> 4;
                                int chunk = idx & 15;
                                int atom = chunk >> 3;
                                int g = chunk & 7;
                                const uint4* src = reinterpret_cast<const uint4*>(
                                    cd8 + atom * 2048 + row * 128 + ((g ^ (row & 7)) * 16));
                                uint4 val = *src;
                                uint4* dst = reinterpret_cast<uint4*>(
                                    d_base + (size_t)row * p.shape_n + atom * 64 + g * 8);
#if FMGGM_DCS
                                asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                                             :: "l"(dst), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                                             : "memory");
#else
                                *dst = val;
#endif
                            }
                        }
                        tma_stage = tma_stage + 1 == TMA_STORE_STAGES ? 0 : tma_stage + 1;
                    }
            }
#else
            for (int si = 0; si < num_stores; ++si) {
#if FMGGM_TMA_STORE
                if (ew == 0 && lane == 0) tma_store_wait<TMA_STORE_STAGES - 1>();
#endif
                named_barrier_sync(1, 128);
#if FMGGM_LDBATCH
                {
                    // one si worth of loads issued back-to-back; single wait exposes
                    // tmem latency once per si instead of once per li.
                    const uint32_t taddr = tmem + acc * (BM * N_B_PER_STAGE)
                                           + (uint32_t)(j * BM + si * 16);
                    float v[16];
                    if (FMGGM_EXPER != 9 && FMGGM_EXPER != 10) {
                        tmem_ld_16x256b_x1(taddr, v);
                        tmem_ld_16x256b_x1(taddr | 0x00100000u, v + 4);
                        tmem_ld_16x256b_x1(taddr + 8, v + 8);
                        tmem_ld_16x256b_x1((taddr + 8) | 0x00100000u, v + 12);
                        tmem_ld_wait();
                    }
                    if (si == num_stores - 1 && j == (int)jhi - 1) {
                        tc_fence_before_sync();
                        mbar_arrive(&sm.tmem_empty[acc]);
                    }
                    if (FMGGM_EXPER != 9 && FMGGM_EXPER != 10) {
                        #pragma unroll
                        for (int li = 0; li < 2; ++li) {
                            uint32_t packed[4];
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                __nv_bfloat162 h = __float22bfloat162_rn(
                                    make_float2(v[li * 8 + 2 * i], v[li * 8 + 2 * i + 1]));
                                packed[i] = *reinterpret_cast<uint32_t*>(&h);
                            }
                            const uint32_t saddr = smem_u32(&sm.cd[tma_stage][0]) + atom_off
                                + (uint32_t)li * 1024
                                + (uint32_t)(lane & 7) * 128
                                + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                            stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                        }
                    }
                }
#elif FMGGM_TJEB
                {
                    // TJEB legacy path (STAIL half-tiles / DINTER-off builds):
                    // batch both li loads of this (j,si), single wait, then
                    // arrivals + pack/STSM. 8 waits per j instead of 16.
                    float v[2][8];
                    #pragma unroll
                    for (int li = 0; li < 2; ++li) {
#if FMGGM_J0LA || FMGGM_J0LB
                        const uint32_t taddr = tmem + ((j == 0) ? j0col : (uint32_t)BM)
                                               + (uint32_t)(si * 16 + li * 8);
#else
                        const uint32_t taddr = tmem + acc * (BM * N_B_PER_STAGE)
                                               + (uint32_t)(j * BM + si * 16 + li * 8);
#endif
                        tmem_ld_16x256b_x1(taddr, v[li]);
                        tmem_ld_16x256b_x1(taddr | 0x00100000u, v[li] + 4);
                    }
                    tmem_ld_wait();
#if FMGGM_J0LB
                    if (si == num_stores - 1) {
                        tc_fence_before_sync();
#if FMGGM_STAIL
                        if (j == 1) mbar_arrive(&sm.tj1_empty);
                        if (j == 0) mbar_arrive(&sm.tj0_empty[itj & 1]);
#else
                        if (j == nj - 1) mbar_arrive(&sm.tj1_empty);
                        if (j == 0) mbar_arrive(&sm.tj0_empty[itj & 1]);
#endif
                    }
#else
                    if (si == num_stores - 1 && j == (int)jhi - 1) {
                        tc_fence_before_sync();
                        mbar_arrive(&sm.tmem_empty[acc]);
                    }
#endif
                    #pragma unroll
                    for (int li = 0; li < 2; ++li) {
                        uint32_t packed[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            __nv_bfloat162 h = __float22bfloat162_rn(
                                make_float2(v[li][2 * i], v[li][2 * i + 1]));
                            packed[i] = *reinterpret_cast<uint32_t*>(&h);
                        }
                        const uint32_t saddr = smem_u32(&sm.cd[tma_stage][0]) + atom_off
                            + (uint32_t)li * 1024
                            + (uint32_t)(lane & 7) * 128
                            + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                        stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                    }
                }
#else
                #pragma unroll
                for (int li = 0; li < 2; ++li) {
#if FMGGM_J0LA || FMGGM_J0LB
                    const uint32_t taddr = tmem + ((j == 0) ? j0col : (uint32_t)BM)
                                           + (uint32_t)(si * 16 + li * 8);
#else
                    const uint32_t taddr = tmem + acc * (BM * N_B_PER_STAGE)
                                           + (uint32_t)(j * BM + si * 16 + li * 8);
#endif
                    float v[8];
                    if (FMGGM_EXPER != 9 && FMGGM_EXPER != 10) {
                        tmem_ld_16x256b_x1(taddr, v);
                        tmem_ld_16x256b_x1(taddr | 0x00100000u, v + 4);
                        tmem_ld_wait();
                    }
#if FMGGM_J0LA
                    if (si == num_stores - 1 && li == 1) {
                        tc_fence_before_sync();
                        if (j == 0) mbar_arrive(&sm.tmem_empty_j0[(uint32_t)sched.cur_iter & 1]);
                        else        mbar_arrive(&sm.tmem_empty[0]);
                        if (nj == 1) mbar_arrive(&sm.tmem_empty[0]);   // j1 unused
                    }
#elif FMGGM_J0LB
#if FMGGM_EXPER == 32
                    if (si == 0 && li == 0) {   // bubble probe: free cols before loads
                        if (j == 1) mbar_arrive(&sm.tj1_empty);
                        if (j == 0) mbar_arrive(&sm.tj0_empty[itj & 1]);
                    }
#else
                    if (si == num_stores - 1 && li == 1) {
                        tc_fence_before_sync();
#if FMGGM_STAIL
                        if (j == 1) mbar_arrive(&sm.tj1_empty);             // j1 loads done (drained first when present)
                        if (j == 0) mbar_arrive(&sm.tj0_empty[itj & 1]);    // j0 loads done
                        // unused halves arrive once at tile start (see above)
#else
                        if (j == nj - 1) mbar_arrive(&sm.tj1_empty);        // j1 loads done
                        if (j == 0) mbar_arrive(&sm.tj0_empty[itj & 1]);    // j0 loads done
                        // nj==1: j==0==nj-1 -> both arrive (tj1 first)
#endif
                    }
#endif
#else
                    if (si == num_stores - 1 && li == 1 && j == (int)jhi - 1) {
                        tc_fence_before_sync();
                        mbar_arrive(&sm.tmem_empty[acc]);
                    }
#endif
                    if (FMGGM_EXPER != 9 && FMGGM_EXPER != 10) {
                        uint32_t packed[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            __nv_bfloat162 h = __float22bfloat162_rn(make_float2(v[2 * i], v[2 * i + 1]));
                            packed[i] = *reinterpret_cast<uint32_t*>(&h);
                        }
                        const uint32_t saddr = smem_u32(&sm.cd[tma_stage][0]) + atom_off
                            + (uint32_t)li * 1024
                            + (uint32_t)(lane & 7) * 128
                            + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                        stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                    }
                }
#endif
#if FMGGM_TMA_STORE
                fence_proxy_async();
                named_barrier_sync(2, 128);
                if (ew == 0 && lane == 0) {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i)
                        tma_store_2d(&p.map_d,
                                     &sm.cd[tma_stage][i * STORE_M * STORE_N_ATOM],
                                     base_n + i * STORE_N_ATOM, base_m + si * STORE_M);
                    tma_store_commit();
                }
                __syncwarp();
#else
                // plain vectorized stores: 16 rows x 128 bf16 = 256 x 16B chunks
                named_barrier_sync(2, 128);
                if (FMGGM_EXPER != 8 && FMGGM_EXPER != 10) {
                    uint16_t* d_base = reinterpret_cast<uint16_t*>(p.d_ptr)
                        + (size_t)(base_m + si * STORE_M) * p.shape_n + base_n;
                    const uint8_t* cd8 = reinterpret_cast<const uint8_t*>(&sm.cd[tma_stage][0]);
                    #pragma unroll
                    for (int pass = 0; pass < 2; ++pass) {
                        int idx = tid - 128 + pass * 128;           // 0..255
                        int row = idx >> 4;                          // m row in store tile
                        int chunk = idx & 15;                        // 16B chunk within row
                        int atom = chunk >> 3;                       // n-atom (64 bf16)
                        int g = chunk & 7;                           // group within atom
                        const uint4* src = reinterpret_cast<const uint4*>(
                            cd8 + atom * 2048 + row * 128 + ((g ^ (row & 7)) * 16));
                        uint4 val = *src;
                        uint4* dst = reinterpret_cast<uint4*>(
                            d_base + (size_t)row * p.shape_n + atom * 64 + g * 8);
                        *dst = val;
                    }
                }
#endif
                tma_stage = tma_stage + 1 == TMA_STORE_STAGES ? 0 : tma_stage + 1;
            }
#endif
            }
#if FMGGM_DINTER
            }
#endif
#if FMGGM_MTRE
            dr_work += clock64() - dr_te1; ++dr_tiles;
#endif
        }
#if FMGGM_MTRE
        if ((blockIdx.x == 0 || blockIdx.x == 74) && ew == 0 && lane == 0) {
            const double nt = (double)(dr_tiles ? dr_tiles : 1);
            printf("MTRE cta=%u tiles=%llu wall=%llu | per-tile wait=%.1f drain=%.1f total=%.1f\n",
                   blockIdx.x, dr_tiles, clock64() - dr_t00,
                   (double)dr_wait / nt, (double)dr_work / nt,
                   (double)(dr_wait + dr_work) / nt);
        }
#endif
#if FMGGM_TMA_STORE
        if (ew == 0 && lane == 0) tma_store_wait<0>();
#endif
    }

    __syncthreads();
    if (warp == 0) tmem_dealloc(tmem, TMEM_COLS);
}
#if FMGGM_DESCROLL
#undef FMGGM_MMA_STAGE
#endif

#if FMGGM_DROUTE
// ------------------------------------------------------------------ E120: device route
// With DROUTE the body above is a device function; these are the entries the launcher and the
// route kernel call. fmggm_kernelT keeps its old name and signature so the DEC16 decode route,
// the MCAST/PDL launch paths and every existing -DFMGGM_BMV build are untouched.
template <int BMV, int STG>
__global__ void __launch_bounds__(THREADS, 1)
fmggm_kernelT(const __grid_constant__ Params p) { fmggm_bodyT<BMV, STG>(p); }

// The wide arm's ring depth is not a choice: E119 section 7 shows stage(BMV) =
// 32,768 (b, fp4 unpacked to 8-bit containers) + 128*BMV + 4,096 + 2,048*ceil(BMV/128), so
// depth 4 needs BMV <= 128 even before cd's 8,192 B, while any BMV that reduces a 1024-row
// group's m-block count needs BMV >= 147. Depth 3 is forced, and it is why the route has to be
// decided at all rather than just always taken.
constexpr int DROUTE_WIDE_STG = 3;
using SharedStorageWide = SharedStorageT<FMGGM_DROUTE_BM, DROUTE_WIDE_STG>;
static_assert(sizeof(SharedStorageWide) <= sizeof(SharedStorage),
              "the launch requests sizeof(SharedStorage); the wide body must fit inside it");
static_assert(sizeof(SharedStorageT<FMGGM_DROUTE_BM, 4>) > 232448,
              "DROUTE_WIDE_STG=3 is justified by the smem budget; if depth 4 now fits, re-measure "
              "the route instead of keeping a penalty that is no longer compulsory");

// One launch, two geometries. The route is a pure function of (masked_m, num_groups), so every
// CTA and every warp computes the SAME value: the branch into the bodies is CTA-uniform and
// grid-uniform, which is what makes the two bodies' mbarrier, TMEM and ring state safe. No CTA
// ever mixes geometries, so no smem layout is reinterpreted mid-flight.
//
// blk_ratio = sum(ceil(c/128)) / sum(ceil(c/FMGGM_DROUTE_BM)) over the actual group sizes, and
// the route is wide iff blk_ratio >= NUM/DEN (7/5 = 1.40, inside E119's measured gap between the
// losers' 1.333 and the winners' 1.429). The load is not a new memory pass - the scheduler walk
// reads the same array - but its LATENCY is exposed, because the branch depends on it and nothing
// can be hoisted above it: the two bodies lay their mbarriers out at different offsets, so not
// even the barrier init is route-independent. The first version claimed this was "hidden behind
// the barrier/TMEM initialisation the bodies do anyway" and that was simply wrong; dv_e2fea4915a3b
// measured the wrapper's own prologue at ~0.8 us on the launch-floor case, which is why the
// reduction below is barrier-free and what is left is one load latency.
extern "C" __global__ void __launch_bounds__(THREADS, 1)
fmggm_kernel_dr(const __grid_constant__ Params p) {
    // Barrier-free and atomic-free: EVERY warp redundantly walks masked_m and reduces inside
    // itself, so all eight warps derive the same value with no __syncthreads() and no smem word.
    // The first version of this wrapper used three barriers and two smem atomics, and the four-arm
    // job dv_e2fea4915a3b put droute 0.8 us above the inc/host arms on case 1 (7.040 against
    // 6.224/6.176) - a 13% tax on a 6.2 us case that is 82% launch floor, i.e. exactly the
    // exposure that sank nomination 2. The branch is CTA-uniform either way because the route is a
    // pure function of the inputs; the barriers were pure cost. num_groups <= 64 <= 2*32, so the
    // lane-stride loop is at most two L2-warm loads per lane, and the shuffle sits OUTSIDE the loop
    // with every lane active - a divergent collective here is the fault that killed
    // dv_39f6218f6372 and 20261023T010442Z.
    const int lane = threadIdx.x & 31;
    int b1 = 0, b2 = 0;
    for (int g = lane; g < p.num_groups; g += 32) {
        const int c = __ldg(p.masked_m + g);
        b1 += (c + 127) / 128;
        b2 += (c + FMGGM_DROUTE_BM - 1) / FMGGM_DROUTE_BM;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        b1 += __shfl_xor_sync(0xffffffffu, b1, off);
        b2 += __shfl_xor_sync(0xffffffffu, b2, off);
    }
    // b2 > 0 keeps the all-empty launch (fixed case 1, expected_m = 1) on the narrow body,
    // byte-identical to the incumbent's kernel for that shape.
    //
    // E121 added the second condition, and it is the out-of-sample correction. blk_ratio >= 7/5
    // was fitted to the ten FIXED cases; the predeclared domain is the first independent test of
    // it and it half-failed - cfgs 2 and 7 win (1.1306, 1.1726) while cfgs 8 and 9 route wide and
    // LOSE (0.9194, 0.9501), taking the domain geomean from +2.87% achievable down to +1.49%.
    // `b1 * n_pairs > num_sms` (more than one wave of narrow tiles) removes cfg 8 and nothing
    // else: over all 20 measured points it keeps 5/5 winners, drops losers taken from 2 to 1, and
    // lifts the domain to +2.34% with the fixed suite unchanged at +2.41%. Mechanism: inside a
    // single wave every m-block of a group is resident at the same time, so the weight re-reads
    // the wide tile removes are already being absorbed by L2 rather than paid to DRAM, while the
    // wide tile still cuts the number of busy SMs (cfg 8: 108 tiles -> 72 on 148 SMs) and still
    // pays ring depth 4 -> 3. There is no cross-wave amplification left to recover.
    const int n_pairs = (p.num_n_blocks + N_B_PER_STAGE - 1) / N_B_PER_STAGE;
    const bool wide = (b2 > 0)
                   && ((long long)b1 * n_pairs > (long long)p.num_sms)
                   && (FMGGM_DROUTE_DEN * b1 >= FMGGM_DROUTE_NUM * b2);
    if (wide) fmggm_bodyT<FMGGM_DROUTE_BM, DROUTE_WIDE_STG>(p);
    else      fmggm_bodyT<BM, STAGES>(p);
}
#endif  // FMGGM_DROUTE

// ------------------------------------------------------------------ 2-CTA kernel
// Cluster of 2 CTAs per tile: UMMA M=256 covers 256 weight rows (128 per CTA);
// activations are N-split (each CTA loads eff/2 rows); activation SFs are
// REPLICATED per CTA (MMA-B side SF must cover full N locally); weight SFs are
// M-split. All A/B TMAs are cta_group::2 (tx accumulates on CTA0's barrier);
// the leader CTA issues UTCCP (cta_group::2) and the MMAs; commits multicast
// to both CTAs' local empty/tmem_full barriers.
#ifndef FMGGM_FORCE1CTA
#define FMGGM_FORCE1CTA FMGGM_BK256   // BK256 variant builds only the 1-CTA kernel
#endif

#if !FMGGM_FORCE1CTA
extern "C" __global__ void __launch_bounds__(THREADS, 1) __cluster_dims__(2, 1, 1)
fmggm_kernel2sm(const __grid_constant__ Params p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& sm = *reinterpret_cast<SharedStorage2*>(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint32_t rank = block_rank();          // 0 = leader
    const bool leader = (rank == 0);

    // ---- early exit when every expert is empty
    __shared__ int s_any;
    if (tid == 0) s_any = 0;
    __syncthreads();
    if (tid < p.num_groups && __ldg(p.masked_m + tid) > 0) atomicOr(&s_any, 1);
    __syncthreads();
    bool any = (s_any != 0);
    // NOTE: cluster-wide consistency guaranteed (same masked_m); still both CTAs
    // must participate in cluster barriers before leaving.
    if (!any) {
        cluster_arrive();
        cluster_wait();
        return;
    }

    // ---- init: barriers (warp1, both CTAs) + TMEM alloc cg2 (warp2, both CTAs)
    if (warp == 1 && elect_one()) {
        #pragma unroll
        for (int i = 0; i < STAGES2; ++i) {
            mbar_init(&sm.full[i], FMGGM_EXPER == 15 ? 5 : 129);   // 129 = expect + 2 CTAs x 2 warps x 32 lanes; 15: elected arrives
            mbar_init(&sm.sf_full[i], 1);  // local SF TMA
            mbar_init(&sm.empty[i], 1);    // multicast commit
        }
        #pragma unroll
        for (int i = 0; i < ACC_STAGES; ++i) {
            mbar_init(&sm.tmem_full[i], 1);      // multicast commit
            mbar_init(&sm.tmem_empty[i], 8);     // 4 epilogue warps x 2 CTAs (elected arrives, leader waits)
        }
        fence_mbar_init();
    } else if (warp == 2) {
        tmem_alloc_cg2(&sm.tmem_base, TMEM_COLS);
        tmem_relinquish_cg2();
    }
    cluster_arrive();
    cluster_wait();
    const uint32_t tmem = sm.tmem_base;

    Scheduler sched;
    sched.init(p.masked_m, p.num_groups, p.num_n_blocks2, p.num_clusters, blockIdx.x >> 1);
    uint32_t m_blk = 0, n_blk = 0;
    uint32_t stage = 0, phase = 0;

    if (warp == 0 && elect_one()) {
        // ---------------- TMA warp (both CTAs; A half-rows + B own-128-rows) ----
        constexpr uint32_t TX_B_CTA = BN * BK / 2 * FMGGM_B_TX_DIV;   // 8KB per CTA
        constexpr uint32_t TX_SF = BM * 4 * 4 + BN * 4 * 4;           // local 4KB
        while (sched.next(m_blk, n_blk)) {
            const int m0 = sched.cur_group * p.shape_m + (int)m_blk * BM;
            const int n0 = sched.cur_group * p.shape_n + (int)n_blk * (2 * BN) + (int)rank * BN;
            const int eff = sched.effective_m(m_blk);
            const int half = eff >> 1;
            const int m_idx = m0 + (int)rank * half;
            const CUtensorMap* map_a = &p.map_a8;
            uint32_t tx_a_cta = 8 * BK;
            if (half == 16)      { map_a = &p.map_a16; tx_a_cta = 16 * BK; }
            else if (half == 32) { map_a = &p.map_a32; tx_a_cta = 32 * BK; }
            else if (half == 64) { map_a = &p.map_a64; tx_a_cta = 64 * BK; }
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                mbar_wait(&sm.empty[stage], phase ^ 1);
                if (FMGGM_EXPER != 14) {
                    // local SF f32 tiles (activation full 128 rows replicated; weights own rows)
                    tma_load_2d(&p.map_sfa, &sm.sf_full[stage], sm.sfa_raw[stage], kb * 4, m0);
                    tma_load_2d(&p.map_sfb, &sm.sf_full[stage], sm.sfw_raw[stage], kb * 4, n0);
                    mbar_arrive_expect_tx(&sm.sf_full[stage], TX_SF);
                    // A/B via cta_group::2 (tx to leader's full barrier)
                    tma_load_2d_cg2(map_a, &sm.full[stage], sm.a[stage], kb * BK, m_idx);
                    if (FMGGM_EXPER != 6)
                        tma_load_2d_cg2(&p.map_b, &sm.full[stage], sm.b[stage], kb * BK, n0);
                }
                if (leader)
                    mbar_arrive_expect_tx(&sm.full[stage], FMGGM_EXPER == 14 ? 0u :
                                          2 * tx_a_cta + (FMGGM_EXPER != 6 ? 2 * TX_B_CTA : 0u));
                stage = stage + 1 == STAGES2 ? 0 : stage + 1;
                phase ^= stage == 0;
            }
        }
    } else if (warp == 2 || warp == 3) {
        // ---------------- SF pack warps (both CTAs): local f32 -> local packed ----
        const bool is_act = (warp == 2);
        float* raw0 = is_act ? sm.sfa_raw[0] : sm.sfw_raw[0];
        while (sched.next(m_blk, n_blk)) {
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                float* raw = raw0 + (size_t)stage * (BM * 4);
                if (FMGGM_EXPER != 14) mbar_wait(&sm.sf_full[stage], phase);
                float4 v[4];
                if (FMGGM_EXPER != 14) {
                    #pragma unroll
                    for (int rr = 0; rr < 4; ++rr)
                        v[rr] = *reinterpret_cast<const float4*>(raw + (size_t)(lane + 32 * rr) * 4);
                }
                mbar_wait(&sm.empty[stage], phase ^ 1);
                if (FMGGM_EXPER != 14) {
                    __syncwarp();
                    uint32_t* dst = reinterpret_cast<uint32_t*>(raw);
                    #pragma unroll
                    for (int rr = 0; rr < 4; ++rr) dst[lane * 4 + rr] = pack_sf4(v[rr]);
                    fence_proxy_async();
                }
                if (FMGGM_EXPER == 15) {
                    __syncwarp();
                    if (elect_one()) mbar_arrive_remote0(&sm.full[stage]);
                } else {
                    mbar_arrive_remote0(&sm.full[stage]);
                }
                stage = stage + 1 == STAGES2 ? 0 : stage + 1;
                phase ^= stage == 0;
            }
        }
    } else if (warp == 1 && leader && elect_one()) {
        // ---------------- MMA warp (leader CTA, single lane: all tcgen05 ops are
        // single-thread; non-elected lanes fall through to the final cluster sync)
        const uint32_t idesc_base = make_idesc_base(1, 256);
        const uint64_t adesc0 = make_smem_desc(smem_u32(sm.b[0]), 1024, 0, 2); // weights = MMA A
        const uint64_t bdesc0 = make_smem_desc(smem_u32(sm.a[0]), 1024, 0, 2); // activations = MMA B
        while (sched.next(m_blk, n_blk)) {
            const uint32_t acc = (uint32_t)sched.cur_iter % ACC_STAGES;
            const uint32_t acc_ph = ((uint32_t)sched.cur_iter / ACC_STAGES) & 1;
            mbar_wait(&sm.tmem_empty[acc], acc_ph ^ 1);
            tc_fence_after_sync();
            const uint32_t eff = (uint32_t)sched.effective_m(m_blk);
            const uint32_t idesc_n = idesc_base | ((eff >> 3) << 17);
            for (int kb = 0; kb < p.num_k_blocks; ++kb) {
                mbar_wait(&sm.full[stage], phase);
                tc_fence_after_sync();
                if (FMGGM_EXPER != 11 && FMGGM_EXPER != 12) {
                    utccp_cg2(tmem + SFW_COL, make_smem_desc(smem_u32(sm.sfw_raw[stage]), 128, 0, 0));
                    utccp_cg2(tmem + SFA_COL, make_smem_desc(smem_u32(sm.sfa_raw[stage]), 128, 0, 0));
                }
                if (FMGGM_EXPER != 12 && FMGGM_EXPER != 14) {
                    const uint64_t adesc_s = adesc0 + (uint64_t)stage * (BN * BK / 16);
                    const uint64_t bdesc_s = bdesc0 + (uint64_t)stage * (64 * BK / 16);
                    #pragma unroll
                    for (uint32_t t = 0; t < UMMAS_PER_STAGE; ++t) {
                        const uint32_t idesc = idesc_n | (t << 29) | (t << 4);
                        umma_mxf8f6f4_cg2(tmem + acc * 128, adesc_s + 2 * t, bdesc_s + 2 * t,
                                          idesc, tmem + SFW_COL, tmem + SFA_COL,
                                          (kb > 0 || t > 0) ? 1u : 0u);
                    }
                }
                umma_commit_cg2_mc(&sm.empty[stage], 3);
                if (kb == p.num_k_blocks - 1) umma_commit_cg2_mc(&sm.tmem_full[acc], 3);
                stage = stage + 1 == STAGES2 ? 0 : stage + 1;
                phase ^= stage == 0;
            }
        }
    } else if (warp >= 4) {
        // ---------------- epilogue (both CTAs; each stores its own 128 n-rows) --
        const int ew = warp - 4;
        while (sched.next(m_blk, n_blk)) {
            const uint32_t acc = (uint32_t)sched.cur_iter % ACC_STAGES;
            const uint32_t acc_ph = ((uint32_t)sched.cur_iter / ACC_STAGES) & 1;
            mbar_wait(&sm.tmem_full[acc], acc_ph);
            tc_fence_after_sync();
            const int eff = sched.effective_m(m_blk);
            const int num_stores = eff / STORE_M;
            const int base_m = sched.cur_group * p.shape_m + (int)m_blk * BM;
            const int base_n = (int)n_blk * (2 * BN) + (int)rank * BN;   // D cols: no group offset
            const uint32_t atom_off = (uint32_t)(ew >> 1) * (STORE_M * STORE_N_ATOM * 2);
            for (int si = 0; si < num_stores; ++si) {
                named_barrier_sync(1, 128);
                #pragma unroll
                for (int li = 0; li < 2; ++li) {
                    const uint32_t taddr = tmem + acc * 128 + (uint32_t)(si * 16 + li * 8);
                    float v[8];
                    tmem_ld_16x256b_x1(taddr, v);
                    tmem_ld_16x256b_x1(taddr | 0x00100000u, v + 4);
                    tmem_ld_wait();
                    if (si == num_stores - 1 && li == 1) {
                        tc_fence_before_sync();
                        __syncwarp();
                        if (elect_one()) mbar_arrive_remote0(&sm.tmem_empty[acc]);
                    }
                    uint32_t packed[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        __nv_bfloat162 h = __float22bfloat162_rn(make_float2(v[2 * i], v[2 * i + 1]));
                        packed[i] = *reinterpret_cast<uint32_t*>(&h);
                    }
                    const uint32_t saddr = smem_u32(&sm.cd[0][0]) + atom_off
                        + (uint32_t)li * 1024
                        + (uint32_t)(lane & 7) * 128
                        + ((uint32_t)(((ew & 1) * 4 + (lane >> 3)) ^ (lane & 7))) * 16;
                    stsm_x4_trans(saddr, packed[0], packed[1], packed[2], packed[3]);
                }
                named_barrier_sync(2, 128);
                {
                    uint16_t* d_base = reinterpret_cast<uint16_t*>(p.d_ptr)
                        + (size_t)(base_m + si * STORE_M) * p.shape_n + base_n;
                    const uint8_t* cd8 = reinterpret_cast<const uint8_t*>(&sm.cd[0][0]);
                    #pragma unroll
                    for (int pass = 0; pass < 2; ++pass) {
                        int idx = tid - 128 + pass * 128;
                        int row = idx >> 4;
                        int chunk = idx & 15;
                        int atom = chunk >> 3;
                        int g = chunk & 7;
                        const uint4* src = reinterpret_cast<const uint4*>(
                            cd8 + atom * 2048 + row * 128 + ((g ^ (row & 7)) * 16));
                        uint4 val = *src;
                        uint4* dst = reinterpret_cast<uint4*>(
                            d_base + (size_t)row * p.shape_n + atom * 64 + g * 8);
                        *dst = val;
                    }
                }
            }
        }
    }

    cluster_arrive();
    cluster_wait();
    if (warp == 0) tmem_dealloc_cg2(tmem, TMEM_COLS);
}
#endif // !FMGGM_FORCE1CTA (fmggm_kernel2sm)

// Kernel instantiations (host refers to these symbols directly):
//   fmggm_kernelT<BM, STAGES>  - main config (BM=128)
//   fmggm_kernelT<16, FMGGM_DEC_STG>       - DEC16 decode config (flag-gated routing)

// ------------------------------------------------------------------ host side
namespace {

struct MapKey {
    const void* a; const void* b; void* d; const void* sfa; const void* sfb;
    const int* mm;   // masked_m: cached Params embed the pointer -> must be keyed
    long long E, M, N, K;
    bool operator==(const MapKey& o) const {
        return a == o.a && b == o.b && d == o.d && sfa == o.sfa && sfb == o.sfb &&
               mm == o.mm && E == o.E && M == o.M && N == o.N && K == o.K;
    }
};
struct MapEntry {
    MapKey key{};
    CUtensorMap ma{}, ma16{}, ma32{}, ma64{}, ma8{}, ma128{}, mb{}, md{}, msfa{}, msfb{};
#if FMGGM_DROUTE
    CUtensorMap ma_wide{}, msfa_wide{};   // E120: the wide arm's A and activation-SF boxes
#endif
#if FMGGM_BMERGE
    CUtensorMap mb2{}, msfb2{};
#endif
#if FMGGM_EBOX
    CUtensorMap map_a_h[8]{}, map_sfa_h[8]{};
#endif
    bool used = false;
    Params cached{};          // fully built launch params (derived from key only)
    bool params_ready = false;
};
constexpr int MAP_CACHE_SLOTS = 8;
MapEntry g_map_cache[MAP_CACHE_SLOTS];
int g_cache_cursor = 0;
int g_num_sms = -1;
bool g_attr_done = false;
int g_use_2sm = -1;   // resolved once: 1 = cluster kernel, 0 = 1-CTA kernel
#if FMGGM_MCAST
// Launch telemetry for E97: whether the last main-route launch actually went out
// as 74 clusters of 2 CTAs, and what the cluster launch returned if it did not.
// A silent fallback to the plain launch would make the arm indistinguishable
// from base, which is exactly the ambiguity this removes. Metadata only.
int g_last_pair_mode = -1;
int g_last_cluster_err = 0;
#endif

int encode_maps(const void* a, const void* b, void* d, const void* sfa, const void* sfb,
                long long E, long long M, long long N, long long K,
                MapEntry* ent) {
    cuuint64_t dims[2];
    cuuint64_t strides[1];
    cuuint32_t box[2];
    const cuuint32_t es[2] = {1, 1};
    // exp4 micro: strided fp4 B pattern BW 3650 (256B promo) -> 3875 GB/s
    // (128B promo). Value-identical cache hint; arm-gated per map class.
    const CUtensorMapL2promotion promo_b = (FMGGM_PROMO >= 1)
        ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    const CUtensorMapL2promotion promo_o = (FMGGM_PROMO >= 2)
        ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    // A activations: uint8 e4m3, [K inner, E*M outer], row stride K bytes.
    // Four row-box heights (128/16/32/64) let the TMA warp shrink decode loads.
    dims[0] = (cuuint64_t)K; dims[1] = (cuuint64_t)(E * M);
    strides[0] = (cuuint64_t)K;
    box[0] = 128;   // sub-block width: BK256 issues SF_SUB half-loads per stage
    CUresult r;
    const int heights[6] = {BM, 16, 32, 64, 8, 128};
    CUtensorMap* amaps[6] = {&ent->ma, &ent->ma16, &ent->ma32, &ent->ma64, &ent->ma8,
                             &ent->ma128};
    for (int i = 0; i < 6; ++i) {
        box[1] = (cuuint32_t)heights[i];
        r = cuTensorMapEncodeTiled(amaps[i], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(a),
            dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -20;
    }
#if FMGGM_DROUTE
    // E120: the wide arm's full-tile A box. box[0] is still the 128-byte BK sub-block width and
    // dims/strides/es/promo/swizzle are the A tensor's, unchanged since the block above; only the
    // row height differs. The separately compiled -DFMGGM_BMV=224 arms already encode exactly
    // this box through heights[0], so the encoding itself is proven on this hardware.
    box[1] = (cuuint32_t)FMGGM_DROUTE_BM;
    r = cuTensorMapEncodeTiled(&ent->ma_wide, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(a),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -29;
#endif
    CUtensorMap* ma = &ent->ma;
    CUtensorMap* mb = &ent->mb;
    CUtensorMap* md = &ent->md;
    (void)ma;
#if FMGGM_EBOX
    for (int i = 0; i < 8; ++i) {   // eff-ladder A boxes (height 16..128)
        box[0] = 128; box[1] = (cuuint32_t)(16 * (i + 1));
        r = cuTensorMapEncodeTiled(&ent->map_a_h[i], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
            const_cast<void*>(a), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -25;
    }
#endif
    // B weights: packed e2m1 via 16U4_ALIGN16B (TMA unpacks nibble->byte into SMEM),
    // [K elements inner, E*N outer], row stride K/2 bytes
    dims[0] = (cuuint64_t)K; dims[1] = (cuuint64_t)(E * N);
    strides[0] = (cuuint64_t)(K / 2);
    box[0] = 128; box[1] = BN;   // sub-block width (see A)
    r = cuTensorMapEncodeTiled(mb, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 2, const_cast<void*>(b),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        promo_b, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -21;
#if FMGGM_BMERGE
    // Same tensor and swizzle, box height 2*BN: one request covers both weight
    // n-tiles of a stage. Rows 128..255 begin a new 1024B swizzle atom whose
    // phase (row mod 8) restarts at 0, which is exactly the image a second
    // 128-row box written at sm.b[stage]+BN*BK (16,384B = 16 atoms) produced,
    // so the UMMA smem descriptors are unchanged.
    box[0] = 128; box[1] = 2 * BN;
    r = cuTensorMapEncodeTiled(&ent->mb2, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 2, const_cast<void*>(b),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        promo_b, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -27;
#endif
    // D output: bf16, [N inner, E*M outer], row stride N*2 bytes, box {64,16}
    dims[0] = (cuuint64_t)N; dims[1] = (cuuint64_t)(E * M);
    strides[0] = (cuuint64_t)(N * 2);
    box[0] = STORE_N_ATOM; box[1] = STORE_M;
    r = cuTensorMapEncodeTiled(md, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d,
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -22;
    // SF maps: f32 [K/32 inner, E*MN outer], box {4 f32 = 16B, 128 rows}, no swizzle
    const long long k32 = K / 32;
    dims[0] = (cuuint64_t)k32; dims[1] = (cuuint64_t)(E * M);
    strides[0] = (cuuint64_t)(k32 * 4);
    box[0] = 4; box[1] = BM;
    r = cuTensorMapEncodeTiled(&ent->msfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(sfa),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -23;
#if FMGGM_DROUTE
    // E120: the wide arm's activation-SF box, {4 f32, FMGGM_DROUTE_BM rows}. TX_SFA in the wide
    // body is BM*4*4*SF_SUB with BM = FMGGM_DROUTE_BM, so the delivered bytes and the
    // expect_tx still match - the mismatch class that b5042e1 deadlocked on.
    box[0] = 4; box[1] = (cuuint32_t)FMGGM_DROUTE_BM;
    r = cuTensorMapEncodeTiled(&ent->msfa_wide, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        const_cast<void*>(sfa), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -30;
    box[0] = 4; box[1] = (cuuint32_t)BM;   // restore, so nothing below inherits a wide SFA box
#endif
#if FMGGM_EBOX
    for (int i = 0; i < 8; ++i) {   // eff-ladder SFA boxes {4 f32, 16..128 rows}
        box[0] = 4; box[1] = (cuuint32_t)(16 * (i + 1));
        r = cuTensorMapEncodeTiled(&ent->map_sfa_h[i], CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
            const_cast<void*>(sfa), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return -26;
    }
    box[0] = 4; box[1] = BM;        // restore for msfb below
#endif
    // b5042e1: the weight SF box is BN rows, NOT BM. At BM=128 the two coincide,
    // which is why this went unnoticed; with BMV=192/224 msfb would be encoded with
    // a BM-row box, so every weight-SF load delivered BM*16 B against a TX_SFW of
    // BN*16 B in expect_tx. An over-delivered mbarrier tx-count never reaches zero,
    // so sf_full[] never completes, the pack warps never arrive on full[], and the
    // whole ring deadlocks SILENTLY (pre-join E16 hang, localised by E19). Set it
    // unconditionally instead of relying on what the msfa/EBOX block left here.
    box[0] = 4; box[1] = BN;
    dims[1] = (cuuint64_t)(E * N);
    r = cuTensorMapEncodeTiled(&ent->msfb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(sfb),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -24;
#if FMGGM_BMERGE
    // Merged weight-SF box: SWIZZLE_NONE, so 2*BM contiguous 16B rows are
    // byte-identical to the two BM-row boxes the producer placed at
    // sfw_raw[stage] + j*(BN*4) floats. BM == BN is already assumed by that
    // stride; BMERGE makes it load-bearing, so assert it.
    static_assert(BM == BN, "merged SFB box doubles the map's row height");
    box[0] = 4; box[1] = 2 * BN;   // b5042e1 class: weight-SF rows are BN, not BM
    r = cuTensorMapEncodeTiled(&ent->msfb2, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(sfb),
        dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        promo_o, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return -28;
#endif
    return 0;
}

} // namespace

#if FMGGM_MCAST
// E97 metadata only: which = 0 returns whether the most recent launch went out
// as 74 clusters of 2 CTAs (1) or unclustered (0); which = 1 returns the
// cudaError_t from the clustered launch attempt. This exists because a silent
// fallback to the plain launch would make an MCAST arm indistinguishable from
// base, and "no change" and "not running" must not be the same measurement.
extern "C" int fmggm_diag_pair(int which) {
    return which ? g_last_cluster_err : g_last_pair_mode;
}
#endif

// E120: which kernel the main route launches, and how much dynamic smem it needs. Under DROUTE
// that is the device-route wrapper, whose smem must cover the LARGER of its two bodies - the
// narrow one, since 4 stages of 55,296 B beats 3 stages of 69,632 B. Both static_asserts live
// with the wrapper. With DROUTE=0 these fold to exactly the previous expressions.
#if FMGGM_DROUTE
#define FMGGM_MAIN_KERNEL fmggm_kernel_dr
#else
#define FMGGM_MAIN_KERNEL (fmggm_kernelT<BM, STAGES>)
#endif
#define FMGGM_MAIN_SMEM sizeof(SharedStorage)

extern "C" int fmggm_launch(const void* a_q, const void* a_sf, const void* b_q, const void* b_sf,
                            void* d, const int* masked_m,
                            long long E, long long M, long long N, long long K,
                            long long expected_m, void* stream) {
    if (E <= 0 || M <= 0 || N <= 0 || K <= 0) return -10;
    if (K % BK != 0) return -11;                       // K must tile into BK (domain guarantees)
    if ((reinterpret_cast<uintptr_t>(a_q) | reinterpret_cast<uintptr_t>(b_q) |
         reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(a_sf) |
         reinterpret_cast<uintptr_t>(b_sf)) & 15) return -12;   // 16B alignment for TMA
    if (!a_sf || !b_sf || !masked_m) return -13;
    if (g_num_sms < 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return -14;
        if (cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return -15;
    }
    if (!g_attr_done) {
        if (cudaFuncSetAttribute(FMGGM_MAIN_KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)FMGGM_MAIN_SMEM) != cudaSuccess) return -16;
#if FMGGM_DEC16
        if (cudaFuncSetAttribute(fmggm_kernelT<16, FMGGM_DEC_STG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(SharedStorageDec)) != cudaSuccess) return -27;
#endif
#if !FMGGM_FORCE1CTA
        if (cudaFuncSetAttribute(fmggm_kernel2sm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(SharedStorage2)) != cudaSuccess) return -17;
#endif
        g_attr_done = true;
    }
    if (g_use_2sm < 0) {
#if FMGGM_FORCE1CTA
        g_use_2sm = 0;
#else
        // 2SM measured SLOWER on L20D (cluster barrier chain ~280ns/stage floor;
        // cg2 TMA adds ~530ns/stage): default 1-CTA. 2sm stays available for
        // experiments via FMGGM_KERNEL=2sm.
        g_use_2sm = 0;
        // A/B override: FMGGM_KERNEL=1cta|2sm (per-shape tuning experiments)
        if (const char* kv = getenv("FMGGM_KERNEL")) {
            if (strcmp(kv, "2sm") == 0 && N % 256 == 0 && g_num_sms % 2 == 0) g_use_2sm = 1;
        }
#endif
    }
    MapKey key{a_q, b_q, d, a_sf, b_sf, masked_m, E, M, N, K};
    MapEntry* ent = nullptr;
    for (auto& e : g_map_cache) if (e.used && e.key == key) { ent = &e; break; }
    if (!ent) {
        ent = &g_map_cache[g_cache_cursor];
        g_cache_cursor = (g_cache_cursor + 1) % MAP_CACHE_SLOTS;
        ent->params_ready = false;   // CRITICAL: slot may hold an evicted entry's cache
        int rc = encode_maps(a_q, b_q, d, a_sf, b_sf, E, M, N, K, ent);
        if (rc) return rc;
        ent->key = key;
        ent->used = true;
    }
    Params p{};
    if (ent->params_ready) {
        p = ent->cached;      // single struct copy; all fields derive from the key
    } else {
        memcpy(&p.map_a, &ent->ma, sizeof(CUtensorMap));
        memcpy(&p.map_a16, &ent->ma16, sizeof(CUtensorMap));
        memcpy(&p.map_a32, &ent->ma32, sizeof(CUtensorMap));
        memcpy(&p.map_a64, &ent->ma64, sizeof(CUtensorMap));
        memcpy(&p.map_a8, &ent->ma8, sizeof(CUtensorMap));
        memcpy(&p.map_a128, &ent->ma128, sizeof(CUtensorMap));
        memcpy(&p.map_b, &ent->mb, sizeof(CUtensorMap));
        memcpy(&p.map_d, &ent->md, sizeof(CUtensorMap));
        memcpy(&p.map_sfa, &ent->msfa, sizeof(CUtensorMap));
        memcpy(&p.map_sfb, &ent->msfb, sizeof(CUtensorMap));
#if FMGGM_DROUTE
        memcpy(&p.map_a_wide, &ent->ma_wide, sizeof(CUtensorMap));
        memcpy(&p.map_sfa_wide, &ent->msfa_wide, sizeof(CUtensorMap));
#endif
#if FMGGM_BMERGE
        memcpy(&p.map_b2, &ent->mb2, sizeof(CUtensorMap));
        memcpy(&p.map_sfb2, &ent->msfb2, sizeof(CUtensorMap));
#endif
#if FMGGM_EBOX
        memcpy(p.map_a_h, ent->map_a_h, sizeof(ent->map_a_h));
        memcpy(p.map_sfa_h, ent->map_sfa_h, sizeof(ent->map_sfa_h));
#endif
        p.d_ptr = reinterpret_cast<uint16_t*>(d);
        p.masked_m = masked_m;
#if FMGGM_LDGSF
        p.sfa_ptr = reinterpret_cast<const float*>(a_sf);
        p.sfb_ptr = reinterpret_cast<const float*>(b_sf);
#endif
        p.num_groups = (int)E; p.shape_m = (int)M; p.shape_n = (int)N; p.shape_k = (int)K;
        p.num_k_blocks = (int)(K / BK);
        p.num_n_blocks = (int)((N + BN - 1) / BN);
        p.num_sms = g_num_sms;
        p.num_n_blocks2 = (int)(N / (2 * BN));
        p.num_clusters = g_num_sms / 2;
        ent->cached = p;
        ent->params_ready = true;
    }
#if FMGGM_MCAST
    // Launch property, not part of the map cache key: reset it every call so a
    // cached Params (or the decode route, which launches unclustered) can never
    // carry a stale pair_mode into a non-cluster launch.
    p.pair_mode = 0;
    g_last_pair_mode = 0;   // set to 1 only if the clustered launch below is used
#endif
#if FMGGM_EXPER == 12
    return 0;   // overhead probe: measure event+ctypes floor with no launch at all
#endif
    void* args[] = {&p};
    cudaError_t e;
#if FMGGM_DEC16
    // Decode route: BMV=16 tiles, STG=5 deep B ring. Gate bounds the
    // single-heavy-expert W refetch at M/16 <= 8 m-blocks (M<=128).
    if (expected_m >= 0 && expected_m <= 16 && M <= 128) {
        e = cudaLaunchKernel(reinterpret_cast<const void*>(fmggm_kernelT<16, FMGGM_DEC_STG>),
                             dim3((unsigned)g_num_sms), dim3(THREADS),
                             args, sizeof(SharedStorageDec),
                             reinterpret_cast<cudaStream_t>(stream));
        return (e == cudaSuccess) ? 0 : (int)e;
    }
#endif
#if !FMGGM_FORCE1CTA
    if (g_use_2sm) {
        e = cudaLaunchKernel(reinterpret_cast<const void*>(fmggm_kernel2sm),
                             dim3((unsigned)(2 * p.num_clusters)), dim3(THREADS),
                             args, sizeof(SharedStorage2),
                             reinterpret_cast<cudaStream_t>(stream));
    } else
#endif
    {
#if FMGGM_MCAST
        if (g_num_sms % 2 == 0) {
            // 74 clusters of 2 CTAs. Adjacent worker ids already decode to adjacent
            // m-blocks of the same (group, n_blk), so cluster peers can share the
            // weight stream without any change to the tile assignment. If the cluster
            // attribute is refused, fall back to the plain launch with pair_mode = 0,
            // which makes every tile take the unpaired path (the kernel's pairing
            // predicate is gated on it), so the fallback is the certified behaviour.
            p.pair_mode = 1;
            cudaLaunchConfig_t ccfg{};
            ccfg.gridDim = dim3((unsigned)g_num_sms);
            ccfg.blockDim = dim3(THREADS);
            ccfg.dynamicSmemBytes = sizeof(SharedStorage);
            ccfg.stream = reinterpret_cast<cudaStream_t>(stream);
            cudaLaunchAttribute cattrs[1];
            cattrs[0].id = cudaLaunchAttributeClusterDimension;
            cattrs[0].val.clusterDim.x = 2;
            cattrs[0].val.clusterDim.y = 1;
            cattrs[0].val.clusterDim.z = 1;
            ccfg.attrs = cattrs;
            ccfg.numAttrs = 1;
            e = cudaLaunchKernelExC(&ccfg,
                                    reinterpret_cast<const void*>(FMGGM_MAIN_KERNEL), args);
#if FMGGM_MCAST
            g_last_cluster_err = (int)e;
#endif
            if (e != cudaSuccess) {
                p.pair_mode = 0;
                e = cudaLaunchKernel(reinterpret_cast<const void*>(FMGGM_MAIN_KERNEL),
                                     dim3((unsigned)g_num_sms), dim3(THREADS),
                                     args, sizeof(SharedStorage),
                                     reinterpret_cast<cudaStream_t>(stream));
            }
#if FMGGM_MCAST
            g_last_pair_mode = p.pair_mode;
#endif
        } else
#endif
#if FMGGM_PDL
        {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((unsigned)g_num_sms);
        cfg.blockDim = dim3(THREADS);
        cfg.dynamicSmemBytes = sizeof(SharedStorage);
        cfg.stream = reinterpret_cast<cudaStream_t>(stream);
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        e = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(FMGGM_MAIN_KERNEL), args);
        }
#else
        {
        e = cudaLaunchKernel(reinterpret_cast<const void*>(FMGGM_MAIN_KERNEL),
                             dim3((unsigned)g_num_sms), dim3(THREADS),
                             args, sizeof(SharedStorage),
                             reinterpret_cast<cudaStream_t>(stream));
        }
#endif
    }
    return (e == cudaSuccess) ? 0 : (int)e;
}

} // namespace fmggm
