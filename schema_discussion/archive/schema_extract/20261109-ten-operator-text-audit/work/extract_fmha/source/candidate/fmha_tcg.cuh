// Phase D tensor-core kernel for fmha_mxfp8_qkv: tcgen05 block-scaled MMA.
//
// Every mechanism used here was validated against known answers on this SM103
// part by harness/bench/tcgen05_blockscale.cu (jobs pD1..pD4, all with zero
// mismatching elements over 16384/32768 outputs):
//
//   * `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
//     consumes E4M3 operands with E8M0 scale factors taken from TMEM, one scale
//     per 32-element k-block, which is exactly this operator's MX block size.
//   * idesc = m_dim<<24 | scale_format(E8M0)<<23 | a_sf_id<<29 | n_dim<<17 |
//     b_major<<16 | a_major<<15 | b_fmt<<10 | a_fmt<<7 | b_sf_id<<4, and the
//     sf_id fields (not the address) pick which byte of the addressed TMEM word
//     is the scale for that k-block.
//   * SFA for output row m is read at TMEM lane m, word (base + m/32 + 4*(kb/4));
//     SFB for output column n is read at lane (n%32) + 32*subpartition for ALL
//     four subpartitions, word (base + n/32 + 4*(kb/4)).  SFB therefore has to
//     be replicated across subpartitions; SFA must not be.
//   * a scale-factor address must be 2-word aligned (offset +1 faults).
//   * K-major operands live in 128-byte swizzled chunks, `desc += 2` per 32-byte
//     k-step, LBO=16 SBO=1024; MN-major operands (V for PV, so V never needs a
//     transpose) use the same [key][d] bytes with LBO cleared and advance
//     4096 bytes per k-step.
//
// Structure: BM=128 rows (queries x packed GQA heads) x BN=128 keys, 128 threads
// so TMEM lane == thread == one query row.  QK^T, the softmax and PV all keep the
// row in one thread, which removes every cross-lane reduction the mma.sync path
// needed.  Hardware block scaling replaces the rank-1 per-MX-block rescale and
// the BF16 V decode entirely, and the scales are powers of two so the products
// are bit-identical to reference.py's dequantized FP32 matmul.
#pragma once

#include <cuda.h>

#include "fmha_common.cuh"
#include "fmha_tcg_ptx.cuh"

namespace fmha_mxfp8 {

// Whether the tcgen05 kernels may stage K and V with `cp.async.bulk.tensor` at all.
// 0 compiles the cp.async staging everywhere and the host encodes no descriptors.
#ifndef FMHA_TCG_TMA
#define FMHA_TCG_TMA 1
#endif

// Whether the D=128 software-pipelined fused form may ALSO stage through bulk
// tensors.  It is the one form `FMHA_TCG_TMA` still excludes, on agt2/agt3b's
// measurement of +4.9% on s7, +4.1% on s0 and +3.2% on d5 - taken before
// FMHA_TCG_SF_BULK existed, i.e. when a TMA form still issued 32 sixteen-byte
// cp.async per stage for the scale atoms plus a commit_group/wait_group pair, which
// is the traffic agu2 then priced at -5.85%/-6.28% on the sequential form.  With
// the atoms on the tile's mbarrier a TMA-staged pipelined form has no cp.async at
// all, so the old number no longer describes it.  The blocker was that this loop
// reads ring slot `nxt`'s atoms before the MMA that owns that slot's barrier; that
// order has now been derived (both slots' barriers are proven above step 2, by
// every thread).  Shapes affected: s0, s8, d1, d4, d9.
// Measured a SECOND time by agu8 (dv_bd9b93c38a11, base best/ao9_full, one
// allocation, two rounds, 8/8 PASS in all four with zero mismatched elements) and
// rejected again: on min, d4 +5.19%, s0 +5.37%, s8 +2.85%, d9 +2.76%, d1 +1.33% -
// geomean +3.49% over the five shapes that dispatch to this form, with the three
// controls (s7, s1, s2, which compile nothing changed) at -0.70% as a group.  Both
// rounds agree in sign on s8, d9 and d4.
//
// That closes the question the switch was written to ask.  agu2 moved the scale
// atoms onto the tile's mbarrier, so a TMA-staged pipelined form now has no cp.async
// at all, and the form still loses by about the same margin agt2/agt3b measured when
// the atoms were cp.async (+4.9%/+4.1%/+3.2% then, +3.5% geomean now).  The atoms
// were therefore not why this form was excluded, and the original explanation stands
// on a second measurement: at one CTA per SM there is a single bulk-copy stream
// feeding an SM where 256 threads' worth of sixteen-byte cp.async used to issue in
// parallel with the compute.  The census is consistent - <128,0,2,true> needs only
// 176 registers against <128,0,2,false>'s 184, so the TMA variant is the cheaper
// program and still the slower kernel.
//
// The barrier-order derivation this needed stays: it is what makes the combination
// buildable and correct rather than a trap, and the pipelined group-maximum loop
// needed the same reorder for the atoms it does stage in bulk.
#ifndef FMHA_TMA_PIPE_D128
#define FMHA_TMA_PIPE_D128 0
#endif

// WHICH forms get it is not a list of dimensions, it is one property: the form must
// run two CTAs per SM, i.e. `TcgTmem<...>::ALLOC == 256`.  That is the single thing
// that separated the four forms bulk-tensor staging made 4-5% faster from the one
// it made 4-6% slower, and it also predicts the D=256 group-maximum pass - 256
// columns, two CTAs per SM - which the D=128-only scope left on cp.async.  The
// measurement behind the rule, and behind rejecting prefetch depth as the
// mechanism:
//
//   case  loop                       cp.async  TMA, 4 stages  TMA, 5 stages
//   s7    fused PIPE                  234.0     246.1 (+4.9%)  248.1 (+6.0%)
//   s0    fused PIPE                   41.7      43.1 (+4.1%)   43.3 (+3.8%)
//   d5    fused PIPE                   98.7     101.8 (+3.2%)  104.3 (+6.1%)
//   s1    fused sequential, 2 CTA/SM  220.8     209.5 (-5.1%)  209.9 (-4.8%)
//   s2    split nsplit=37              51.5      49.4 (-3.3%)   49.3 (-4.1%)
//   d9    split, window, KV=513        54.9      51.2 (-5.1%)   51.9 (-5.5%)
//
// (dv_8c071cc3142a and dv_c27893635494; one allocation per job, two alternating
// rounds, controls s5 and d1 inside 1%.)  Prefetch distance is not the mechanism -
// a fifth stage and a three-group fill bought nothing and cost 32 KB of the pool
// the SM shares between shared memory and L1 - and neither is spilling: the
// register census gives 196 registers for the cp.async form and 212 for the TMA
// one, both with 0 bytes of stack.  What separates this form from every winner is
// one CTA per SM, i.e. exactly one bulk-copy stream feeding an SM where 256
// threads' worth of 16-byte cp.async used to issue in parallel with the compute.
// So the one-CTA-per-SM forms keep cp.async and the two-CTA-per-SM forms take the
// bulk copy.  The superseded per-dimension switches were FMHA_TCG_TMA_D=128 and
// FMHA_TCG_TMA_PIPE=0, which selected the same four forms at D=128 and nothing at
// D=256; they are gone rather than left to drift out of sync with this rule.

// Whether the D=128 fused launch may use the key split at all (two 128-thread
// warpgroups per 128-key group, eight warps and all 512 TMEM columns in one CTA,
// which is what selects the software-pipelined loop and, with it, cp.async staging).
// agk2 re-measured pH36's two clauses now that the single-warpgroup form carries
// bulk-tensor staging: the "fewer CTAs than SMs" clause still wins big (s0 +30.0%,
// d4 +37.0% without it) and the "skewed causal" clause now loses big (s7 -11.9%,
// d5 -5.3% with it), so the dispatcher in fmha_mxfp8_qkv.cu keeps only the first.
// This switch is the master off for experiments; 1 is the shipped behaviour.
#ifndef FMHA_KEY_SPLIT_D128
#define FMHA_KEY_SPLIT_D128 1
#endif

// EXPERIMENT ONLY: take the D=128 key split even where the wave-packing argument
// above says not to.  0 is the shipped behaviour, which restricts the split to
// launches that cannot fill the machine on their own (`fused_ctas <= SMs`, worth
// +30% on s0 and +37% on d4 without it).  Setting it to 1 measures the
// O-in-TMEM sequential form that `ab1`'s TMEM attribution asks for on the shapes
// that currently take the two-CTA-per-SM three-readout form instead, i.e. s7
// (512 CTAs) and s1 (256).
// agu1 measured this at 1 on s7, s1 and d5 (the launches the wave-packing rule
// refuses) together with FMHA_PIPE_D128=0: +24.7%, +22.0% and +10.2%.  Kept at 0.
#ifndef FMHA_FORCE_KEY_SPLIT_D128
#define FMHA_FORCE_KEY_SPLIT_D128 0
#endif

// Whether the D=256 fused form issues QK one group ahead.  It does not, and it was
// built and measured rather than argued: agq4 (dv_467195c6a276, base 9430dd1, one
// allocation, two rounds, 7/7 PASS in all four rounds with zero mismatched elements)
// gives
//
//   case  what it is                        exposed waits   QK one group ahead
//   s4    D=256 fused, 96 CTAs, 32 groups        93.53            94.74   +1.29%
//   s6    D=256 fused, 256 CTAs, causal+W        91.28            92.34   +1.16%
//   d6    D=256 fused, 128 CTAs                  94.17            94.47   +0.31%
//   d8    D=256 fused, 64 CTAs                   72.20            72.24   +0.05%
//   s5    control, D=256 split                   80.29            80.37   +0.10%
//   d3    control, D=256 split                   64.61            64.11   -0.77%
//   s7    control, D=128 fused sequential       206.31           206.30   -0.00%
//
// So overlapping QK(g+1) with PV(g) buys nothing here and the third K slot costs
// about 1% on the two shapes with the most groups.  The mechanism the change was
// aimed at - `pH31`-era profiling's 0.34 of 4 instructions issued per cycle with
// 34.3% long_scoreboard on this configuration - is therefore not the two MMA waits
// serialising; whatever it is, it is not fixed by putting both chains on the pipe at
// once.  The schedule is retained behind this switch because it is correct (the
// arithmetic on O is untouched, so the delayed-maximum recurrence stays
// bit-identical, and all four rounds agreed with the reference) and because the
// split K/V ring depth it needs is now a general facility.  Do not re-enable it on
// the same reasoning; profile the D=256 fused form first.
#ifndef FMHA_TCG_QK_AHEAD
#define FMHA_TCG_QK_AHEAD 0
#endif

// One predicate so TcgLayout (which needs the deeper K ring) and TcgTmem (which
// needs the alternating scale-factor slots) cannot disagree about it.
constexpr bool tcg_qk_ahead(int d, int pass_id) {
  return (FMHA_TCG_QK_AHEAD != 0) && (d == 256) && (pass_id == PASS_FUSED);
}

// Whether the sequential loop stages its E8M0 scale atoms THREE key groups ahead of
// the tiles they belong to, so that the TMEM write for group g+1 can move above the
// loop's staging barrier and be ordered by it - which deletes the second of the
// loop's per-group CTA-wide barriers.
//
// That barrier exists only because a warp addresses TMEM lanes [32*(warp%4), +32) and
// no more, so writing the 128 lanes of a scale-factor operand takes four warps while
// `tcgen05.mma` is issued by one thread.  It cannot be deleted where the atoms are
// staged in the same iteration that consumes them, because the four warps read those
// bytes out of shared memory and only the staging barrier publishes them.  Staging the
// atoms early is what breaks that: `cp.async.wait_group 1` at iteration g-1 proves the
// commit made at iteration g-2, which carries atoms(g+1), and iteration g-1's barrier
// then publishes them to every thread, so at iteration g the atoms are readable BEFORE
// the barrier and the write can sit above it.  Three groups of depth, not two: with two,
// the cp.async that fills the slot would be issued in the same iteration that reads it.
//
// The price is two alternating TMEM slots per operand - the write for g+1 happens while
// QK(g) is still reading the slot for g - i.e. `TcgTmem::SF_ALT`, i.e. 16 more columns
// of scale factors.  `harness/analysis/tmem_map_check.py` carries the resulting maps as
// PROPOSED rows: D=256 lands at 464 of 512 with P_TILE still at 448, D=128 at 208 of 256
// with P_TILE still at 192, and the 64-column form does not fit at all (SFB_V [96,112)
// runs through O [96,224)), which is why QK64 is excluded by arithmetic rather than by
// choice.  The group split is excluded for a different hard reason: its map is 512 of
// 512 with the 16 spare columns being an alignment gap, so there is no room for a second
// SFB_K slot.
//
// PHASE 1 IS D=256 ONLY, and the reason is the atom stream rather than the prize:
// `FMHA_TCG_SF_BULK` is D=128-only, so at D=256 the atoms already ride their own
// cp.async stream while the tiles ride bulk tensor copies, and giving them a deeper ring
// is arithmetic.  At D=128 they share the tile's mbarrier, so getting them early needs a
// separate `bulk_copy_1d` with its own barrier - more moving parts, on a form where the
// barrier is 128 threads wide instead of 256 and therefore cheaper.  Calibrated by `bt3`,
// which measured the loop's FOURTH per-group barrier at -1.32% on s4 and -0.54% on s6,
// these same 256-thread D=256 forms.  Full derivation, lifetimes and failure modes:
// results/summaries/2026-10-21-design-sf-atoms-ahead.md.
#ifndef FMHA_SF_AHEAD_D256
#define FMHA_SF_AHEAD_D256 1
#endif

// One predicate so TcgLayout (which needs the deeper atom ring) and TcgTmem (which needs
// the alternating scale-factor slots) cannot disagree about it.  Mutually exclusive with
// tcg_qk_ahead, which reorders the same loop for a different reason and would otherwise
// fight this one for the atom ring's slot indices.
constexpr bool tcg_sf_ahead(int d, int pass_id, int ksplit, bool gsplit) {
  return (FMHA_SF_AHEAD_D256 != 0) && (d == 256) && (pass_id != PASS_GROUP_MAX) && !gsplit &&
         (ksplit == 2) && !tcg_qk_ahead(d, pass_id);
}

// EXPERIMENT qk6 - the D=256 fused mid-pipeline.  prof4's per-line stall attribution
// (results/2026-10-28-prof4-source-attribution.md) localized ~38% of the s4 stall
// samples on TWO zero-slack MMA-completion spins: the head QK wait (issue and wait
// are adjacent instructions) and the foot PV wait (commit and wait likewise).  The
// shipped form never has more than one MMA chain in flight.  Both single moves were
// measured and failed for identifiable reasons: agq4's QK_AHEAD (+1.2%) issued the
// next QK at the FOOT but kept the foot PV wait, so FIFO queued QK behind PV and the
// head wait GREW to ~PV+QK; qk5's deferred PV (NULL, bitwise identical) moved only
// the PV retirement, which relocated that spin into the QK wait's shadow while the
// binding QK exposure stayed.  qk6 is the composition neither arm measured:
//   * QK(grp+1) issues MID-iteration, right after the key-split combine's barrier
//     proves every warp retired the S(grp) readout - the GM loop's single-score-slot
//     reuse trick and the PIPE_GM issue pattern verbatim;
//   * PV(grp-1) retires at the same mid point (qk5's mechanism), which frees the V
//     ring slot for V(grp+1)'s issue right below it (the GSPLIT-documented rule) and
//     keeps the single P_TILE legal for the P(grp) store;
//   * K runs two groups ahead on a three-slot ring (the QK_AHEAD layout facility:
//     +32 KB, 183 -> 216 KB of 228, free on a one-CTA-per-SM form), proven at the
//     mid point so the mid QK issue reads a landed tile;
//   * SFB_V becomes a THREE-slot ring (416..440, inside the existing gap below
//     P_TILE 448) so the head atom write for grp+1 never touches a slot the
//     in-flight PV(grp-1) reads ((grp+1) % 3 != (grp-1) % 3);
//   * the head tile wait and the head barrier DELETE: the combine barrier orders
//     everything the head barrier ordered (tile fences, atom stores, readout
//     retirement) for the tid-0 QK issue below it.  Barrier count per group is
//     unchanged at two; the exposed-wait count goes from two zero-slack spins to
//     waits that each carry >= half an iteration of slack.
// Full derivation, phase formulas, edge cases and the decision rule:
// experiments/2026-10-28-qk6-d256-midpipe-design.md.  Numerics: identical per-group
// MMA sequences and PV FIFO order, so bitwise identity vs control is the
// discriminator (two-arm run required before any official A/B).
//
// DEFAULT 1 as of the screen verdict (dv_42703138824f,
// results/2026-10-28-qk6-screen.md): ALL gates passed - bitwise arm==ctl on all
// 8 shapes x 3 iterations (identity PROVEN, not just argued), census controls
// identical (D256-fused regs 180->184, no spill), zero mismatches everywhere
// except qk_s6's documented pre-existing corner (arm-identical), directional
// same-job medians ctl_d256 79.49->74.50us (0.9372) and qk_s6 50.27->48.16
// (0.9580).  The official fixed-suite A/B vs 2cb76c8 decides promotion.
#ifndef FMHA_D256_MIDPIPE
#define FMHA_D256_MIDPIPE 1
#endif

// Whether the D=128 fused single-warpgroup form issues its QK as two N=64 MMAs
// into a 64-column score tile instead of one N=128 MMA into a 128-column one.
//
// This is the only map that satisfies both of the measurements that bracket the
// D=128 fused form.  `ab1` (`dv_5d981d5c6167`) priced the TMEM readout at **-36.4%
// of s7 and -31.7% of s1**, and the shipped form reads three 128-float tiles per
// key group: the scores for the maximum, the same scores again for the
// probabilities (128 score floats cannot be live next to a 128-float register
// accumulator, so `SINGLE_READ` is off), and the PV partial on its way into that
// accumulator.  `agu1` (`dv_58dc70d197c2`) then measured the obvious fix - O in
// TMEM plus `SINGLE_READ`, one readout per group - at **+24.7% on s7 and +22.0% on
// s1**, because at BN=128 the map is S(128) + O(128) + SF(24) = 280 columns,
// `tcgen05.alloc` granularity is a power of two, and the form drops to one CTA per
// SM.  Two CTAs per SM is worth about +48% there against the -24% the readouts
// were worth.
//
// Halving only the score tile fits both: **S(0..63) + SF(64..95) + O(128..255) =
// 208 used of a 256-column allocation**, so O leaves the register file, the scores
// are read exactly once per group, and two CTAs per SM survive. `scores[128]`
// replaces `oacc[128]` rather than being added to it, which is why the register
// budget works at all.
//
// `bs3` (`dv_7590718fb5ca`) is what makes it a design rather than a wish: the
// block-scaled MMA accepts **N=64** (`G0` PASS bad=0 of 8192, `G1` exact
// accumulator total, `G4r` byte-packed with SFB replicated to four subpartitions
// PASS bad=0 of 8192, `G5r` PASS), and `G3b` decodes the same scale-factor rule as
// at N=128 - output column n reads lane (n%32)+32*sp, word n/32, YES for all four
// subpartitions - so an N=64 QK simply reads two of the four words the existing
// atom staging already writes.  `G6` confirms the word offsets +2 and +4 are legal
// while +1 faults, which is what lets the second half address SFB_K + 2.  pD9 had
// only established that M=64 and M=256 are `illegal instruction`; the shape freedom
// is entirely on the N axis.
//
// Costs: two QK MMAs per group instead of one at the same total FLOPs, one extra
// CTA-wide barrier per group (the tile is single-buffered, so every warp must have
// retired half h-1's `tcgen05.ld` before half h's MMA overwrites it) against
// `barrier` at 0.22 of 4.17 cycles per issued instruction today, and the PV MMA is
// untouched - K is still 128 keys over the whole probability tile, because the
// group is still 128 keys and the contract's increasing-128-key-group order and
// delayed maximum are unchanged.
// Measured by agu5 (dv_9f187e4620ff, base best/ao8_full, one allocation, two
// rounds, 8/8 PASS in all four, zero mismatched elements) and NOT promoted.  The
// map works exactly as bs3 predicted - <128,0,1,true> came in at **254 registers
// with zero bytes of stack and zero spills**, i.e. two CTAs per SM survive with 512
// registers to spare, and TMEM uses 208 of its 256 columns - and the form is
// numerically exact.  But the three official cases that dispatch to it split:
//
//   case  CTAs   waves   groups/CTA   min delta
//   d5    1024   3.46       ~16        -8.09%   WIN
//   s7     512   1.73       ~32        +4.40%   loss
//   s1     256   0.86       ~60        +7.20%   loss
//
// (with all five controls reading +1.29% as a group, i.e. an arm-level bias in that
// allocation, so bias-corrected these are -9.4%, +3.1% and +5.9%).  The ordering
// follows the wave count, not the group count, which says the extra per-group
// latency this form adds - one more mbar_wait and one more fence-plus-barrier pair,
// because both halves share one 64-column tile - is hidden when every SM has two
// CTAs and exposed when it has one.  At 0.86 and 1.73 waves a large fraction of SMs
// run a single CTA for part of the launch; at 3.46 waves essentially none do.
//
// ab2 (dv_060c1ba59ec6) then ablated THIS tree and retired two explanations.
// `norescale` is -0.02% on s7, -0.32% on s1 and +0.61% on d5, so the O-in-TMEM
// delayed-maximum round trip - the prime suspect, since it turns 128 register
// multiplies into a 128-float TMEM load/store pair fired by a warp vote - never
// happens often enough to matter, exactly as a 4*ln(2) threshold on N(0,1) scaled
// scores implies.  And `notmemld` is now +2.7% on s7 and +6.4% on s1, i.e. SLOWER,
// because with one tile read per group instead of three the ablation's own cost (32
// MOVs per replaced tcg_ld32) outweighs what it removes; the -36.4% that ab1
// measured on the shipped form does not transfer to a form that already reads once.
//
// So the readout saving is real and about a third of what it was, while the added
// per-group latency is not.  A wave-count gate (`fused_ctas >= 2 * slots`) would
// keep d5 and exclude s7/s1, but no FROZEN shape reaches two waves in this form, so
// it would buy about -0.8% of the domain geomean and nothing of the frozen one, and
// it would need a fifth template parameter to compile both variants and dispatch on
// geometry.  Recorded as a next experiment rather than shipped on one domain case.
// EXPERIMENT agu9: 1 for the paired A/B against best/ao9_full, where the decision is
// now per launch (wave count) rather than unconditional as it was for agu5.
#ifndef FMHA_QK64_D128
#define FMHA_QK64_D128 1
#endif

// EXPERIMENT bd1: how many full waves of fused CTAs a D=128 launch must have before
// the dispatcher gives it the 64-column score form.  2 is the shipped value and 0
// takes the gate out entirely.
//
// The gate's evidence is `agu5` (`dv_9f187e4620ff`, base `best/ao8_full`), which
// measured the form at **-9.4% on d5 (3.46 waves), +3.1% on s7 (1.73) and +5.9% on
// s1 (0.86)**, all bias-corrected against five controls at +1.29%, and attributed
// the ordering to waves rather than to groups per CTA: the latency the form adds per
// group - one more `mbar_wait` and one more fence-plus-barrier pair, because both
// 64-key halves share one single-buffered score tile - is hidden when every SM holds
// two CTAs and exposed when it holds one.  That measurement is now three promotions
// stale in a way that matters, because **one of the two things it priced has since
// been deleted from every form in the candidate**: `bb1`/`bb2` moved the probability
// tile into TMEM, which removed the `fence.proxy.async.shared::cta` that published it
// and the eight `sts128` behind it, and `bb3` gave the 64-column form the same
// treatment (its map is now S=[0,64) SF=[64,96) O=[96,224) P=[224,256)).  The
// fence-plus-barrier pair `agu5` charged to this form was the *score tile's*, not the
// probability tile's, so the arithmetic does not simply cancel - but the two effects
// were never separated, and the shapes at stake are the two largest in the frozen
// suite (s7 171.01 us, s1 175.38 us in `bb4`'s allocation), where -5% each would be
// -1% of the frozen geomean.
//
// **`bd1` (`dv_744e893e6b05`, base `best/bb4_full`, one allocation, two rounds, 6/6
// PASS in all four with `mismatch_zero=11 pass=17` identical) measured this at 0 and
// the answer is no, decisively: s7 +3.95% and s1 +10.01% bias-corrected, against a
// control bias of -0.090%** from four byte-identical controls (d5, s9, d6, d4 - the
// register census is *identical* in both arms, as it must be, since the gate is a
// host dispatch decision and not a code change).  Both rounds and both statistics
// agree: s7 +3.58/+4.14 median and +2.66/+3.37 min, s1 +9.67/+10.15 and
// +10.56/+9.92.  So `agu5`'s wave-count ordering survives the probability tile's
// move to TMEM, and the composition argument in the previous version of this comment
// - "+3.1% minus 1.5% is +1.6%, but this campaign has been wrong four times
// composing a deletion price with a relocation, so measure it" - was wrong in the
// *other* direction: the penalty grew rather than shrank, from +3.1%/+5.9% to
// +3.95%/+10.01%, i.e. from ~+10 us to ~+17 us on s1.  The reading is that the
// form's cost is the serialized MMA chain its single-buffered 64-column score tile
// forces (half h+1's QK cannot issue until every warp has retired half h's
// `tcgen05.ld`), which is a *latency* the probability tile's fence was never the
// bulk of, and which gets worse relative to a baseline that `bb1` just made faster.
// The switch stays, at its shipped value, because the question was worth asking once
// on the current tree and because a future change to the score tile's buffering
// should re-ask it.  **Do not set this to 0 or 1 on the reasoning above; it has now
// been measured twice, on two different trees, with the same sign.**
//
// EXPERIMENT qk1 re-asks it under its own stated condition, because the form itself
// changed rather than the gate: FMHA_QK64_DEFER_PV removes the exposed PV wait from
// the QK64 loop (below), which is one of the three waits bd1's "serialized MMA
// chain" was composed of, and the only one whose removal does not need a second
// score tile.  MIN_WAVES goes to 0 IN THE SAME ARM because the treated form is no
// longer the form bd1 priced; its value is the host-side half of qk1 and reverts
// with it.
#ifndef FMHA_QK64_MIN_WAVES
#define FMHA_QK64_MIN_WAVES 0
#endif

// EXPERIMENT qk1 - deferred PV retirement in the 64-column score form.
//
// WHAT IT DOES.  The QK64 loop today issues PV(g) at the foot of iteration g and
// spin-waits it there, between the P-store barrier and the iteration boundary, with
// nothing else in flight: the tensor pipe idles through the group's exp/readout
// half and the wait idles through nothing.  qk1 moves the retirement to the middle
// of iteration g+1 - exactly the PIPE loop's step-3 discipline ("its wait is
// deferred to step 3 of the next iteration, which is what lets the pipe stay busy
// across the boundary") - so PV(g) now executes under iteration g+1's staging wait,
// K-atom write, barrier A/A2, both QK half issues+waits and score readouts.  The
// form keeps everything the campaign already measured as good: 256 TMEM columns,
// two CTAs per SM, ONE score-tile readout per group (SINGLE_READ), O accumulated in
// TMEM across the whole span, P published through TMEM.
//
// WHY IT CAN BEAT BOTH INCUMBENTS.  pI14 priced the sequential form's two exposed
// waits at 19.5% of s7 with the tensor pipe only 12.1% busy - the pipe time is
// latency the warps spin on.  bd1 showed the OLD QK64 form (which kept its PV wait
// at the foot) losing +3.95%/+10.01% on s7/s1 despite reading one tile per group
// instead of three, and diagnosed the serialized MMA chain.  qk1 is that chain
// minus its PV link: exposed waits per group drop from {QK, PV} (sequential) or
// {QK-h0, QK-h1, PV} (old QK64) to {QK-h0, QK-h1} with the two halves' MMAs half
// the depth of the sequential form's full-tile QK.  Readout traffic stays at one
// tile per group.  Nothing else in either form's budget moves.
//
// WHY THE PIECES SIT WHERE THEY SIT (full ordering derivation; every claim below is
// a lifetime, and the loop's barriers are unchanged in count and position):
//   * P tile [224,256): written by iteration g's probability pass, read by PV(g)
//     issued at g's foot; the next P write is iteration g+1's probability pass,
//     which the mid-iteration retire precedes.  Same lifetime PIPE documents.
//   * O [96,224): the rare delayed-max rescale (alpha != 1) load/multiplies/stores
//     O; PV(g-1) accumulating into it must be complete first - the retire sits
//     ABOVE the rescale, mirroring PIPE step 3's position above its rescale.
//   * SFB_V is a SINGLE slot in the 256-column map (an alternating pair would need
//     40 scale columns and O would no longer be 32-aligned inside 256).  PV(g-1)
//     was the slot's last reader and PV(g) is its next one, so V(g)'s atom write
//     moves out of the loop-top SF block down to just below the retire, and the
//     P-store barrier (fence_before + syncthreads + tid0's fence_after) is what
//     orders it against the PV issue, exactly as the loop-top block is ordered
//     today by barrier A2.
//   * V ring, depth 2: TMA V(g+1) targets slot (g+1)&1, which IS PV(g-1)'s slot
//     (g-1 = g+1 mod 2), so its issue moves below the retire too.  Depth 2 then
//     suffices: the next writer of V(g)'s slot is V(g+2), issued at iteration g+1's
//     retire point, one wait later.  A depth-3 V ring was designed first and
//     rejected on arithmetic: it needs split K/V barriers anyway AND 117.5 KB of
//     dynamic smem, which is 235 KB at two CTAs/SM against the SM's 228 KB.
//   * K ring, depth 2, untouched: QK(g,h1) is waited inside iteration g, so the
//     loop-top K(g+1) issue into slot (g+1)&1 races nothing, exactly as today.
//   * sSfV[g&1] (atom smem): read mid-iteration g, next overwritten by V(g+2)'s
//     TMA issued mid-iteration g+1 - ordered by iteration g's P-store barrier.
//   * mbar_pv parity: one commit per iteration, one wait per iteration, offset by
//     one - the phase advances by exactly one between consecutive waits, the only
//     shape a parity wait may track (bm2's lesson).  The post-loop epilogue retires
//     the last PV before the O readout; the first iteration skips its retire, so
//     the counts stay equal.
//   * The split K/V stage barriers (mbar_stage for K+atoms, mbar_stageV for
//     V+atoms, armed via tcg_issue_stage's SPLITBAR accounting) each see one
//     expect_tx and one completion per use, waited once per use, per-slot parity.
//   * V(g)'s bytes reach the UMMA proxy via each thread's fence.proxy.async right
//     after the V wait, then barrier C, then tid0's fence_after before the issue -
//     the same fence -> barrier -> issue shape barrier A uses today.
//   * Prime: the prologue's hoisted slot-0 prime splits into K(cLo) and V(cLo)
//     issues; nothing is in flight yet, so ordering is trivial.  First-iteration
//     waits find both barriers armed.
//
// SCOPE.  DPV = FMHA_QK64_DEFER_PV && QK64 && PASS_FUSED && TMA && D==128: the
// one instantiation <128,PASS_FUSED,KSPLIT=1,TMA=1,QK64=1> - with MIN_WAVES=0 in
// this arm, frozen s7, s1 (treated, currently sequential) and domain d5 (treated,
// currently old QK64).  Every other launch compiles the identical body: DPV folds
// to false in all of them, the only cross-form text is tcg_issue_stage's SPLITBAR
// default (false at every existing call site) and TmaStage's extra pointer.  The
// cp.async fallback instantiation of the same form (maps.valid false) keeps the
// shipped foot-wait behavior.
//
// BIT-EXACTNESS.  Arithmetic is untouched: same MMAs, same descriptors, same
// accumulate order (PV(g) still accumulates onto PV(g-1)'s finished O - the retire
// precedes every reader/writer of O), same probabilities, same delayed-maximum
// recurrence.  Only async-operation ISSUE/WAIT positions move.
//
// RISK.  An ordering error here is a hang (300 s perf-timeout failure) or wrong
// numbers (correctness gate, two seeds x ten cases in the same job) - both are
// caught by the official gates, never silent.  1 = treated arm; 0 = shipped.
#ifndef FMHA_QK64_DEFER_PV
#define FMHA_QK64_DEFER_PV 1
#endif

// Whether the quantized probability tile is published to the PV MMA through TENSOR
// MEMORY instead of through shared memory.
//
// The prize is the one `ab1` (`dv_5d981d5c6167`) measured by simply deleting the
// stores: **-12.7% on s7 and -5.6% on s1**, the two largest frozen shapes, and
// that ablation removed only the eight `sts128` per thread per group - not the
// `fence.proxy.async.shared::cta` that publishes them to the UMMA proxy, and not
// the CTA-wide barrier that orders them before tid 0's MMA issue.  `ap18`
// (`dv_4c05696d15e1`) is the reason to expect the whole three to be worth more
// than the sum the ablation priced: d6's fused D=256 launch has **no pipe above
// 22.7%**, 2.00 warps per scheduler and 69.96% of cycles with nothing eligible,
// and s9's split launch runs IPC 1.61 at 11.49% occupancy.  Nothing in this
// candidate is DRAM-bound (3.2-20.2% of ncu's reference) so an instruction and a
// fence removed from a group's critical path is a cycle saved, not a queue joined.
//
// The mechanism was the open hardware question and `ba1` (`dv_234ac7262b30`,
// `harness/bench/tmem_a_operand.cu`) settled it on this exact part: SM103's
// `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` accepts
// `[a-tmem]` in place of the A descriptor, with E8M0 scale factors on both sides
// and an MN-major B, and is **bit-exact against an FP64 reference (bad=0 of 16384,
// worst rel 0.000e+00)** - the same zero error the shared-memory control form
// produces in the same job.  The layout is lane==row, four E4M3 keys per 32-bit
// word, k natural within the word, and **8 TMEM columns per 32-key k-step**, i.e.
// 32 columns for a 128-key tile: the fingerprint configuration (A[m][k] = 2^(k/32))
// returned D = 480.0000 = 32*(1+2+4+8) exactly, where strides of 16, 32 and an
// ignored address would have returned 160, 32 and 128; the reversed-byte and
// reversed-k-step writes both FAIL, so the convention is pinned, not merely
// self-consistent.  Atrex-FA's `utcmma_atmem_bsmem_h_cta1`
// (sm100_d128_fprop_kernel.cu:1470, advanced by `k*(UTCMMA_TILE_K/(4/BYTES_PER_ELEMENT))`)
// is the same substitution for the unscaled `kind::f8f6f4` and is where the stride
// arithmetic came from; it does not use block scale factors, so it could not
// answer whether this ISA allows the two together.
//
// Cost: 32 TMEM columns.  Every form that has them gets the change automatically
// through `TcgTmem::P_IN_TMEM`, which is a column-arithmetic predicate rather than
// a column-arithmetic predicate rather than a list of shapes.  `bb1`/`bb2` shipped
// it for the four forms whose map already had a 32-column window and measured frozen
// -1.82%; the 64-column score form was the one exclusion, 8 columns short, and `ba2`
// (`dv_acb64b7475e6`) then proved on this part that a `tcgen05.mma` accumulator at a
// **32-aligned but not 128-aligned** column base is legal and bit-exact, which let
// that form move its accumulator from 128 to 96 and open the window at 224 - so as
// of `bb3` every probability-carrying form publishes through TMEM.  The
// shared-memory tile is left allocated on purpose in both steps: it is not the
// occupancy limiter of any form (D=128 KSPLIT=1 is limited by its registers and its
// 256 columns, D=256 by 201-208 registers and 512 columns), so keeping `TcgLayout`
// byte-identical leaves the stores, the fence and the MMA residence as the only
// variables in the A/B.  Reclaiming it cannot buy occupancy anywhere - a second
// D=256 CTA needs <=113 KB and the form is at ~171 KB without the tile - and `pI4`
// found that ring depth changes nothing on the sequential form.
#ifndef FMHA_P_IN_TMEM
#define FMHA_P_IN_TMEM 1
#endif

// How many 128-thread warpgroups a FUSED CTA runs, per head dimension, when the
// key split's own small-launch gate has not already chosen one.
//
// The D=128 fused form ships at KSPLIT=1: 128 threads, 255 registers, 256 TMEM
// columns, two CTAs per SM, eight warps per SM and therefore **1.97 active warps
// per scheduler** (ap2, `dv_f700aec38d67`) issuing 0.47 of the 4 instructions per
// cycle the SM can.  Durable fact 9 closes every route to more warps that keeps
// the output accumulator in registers: 4 CTAs per SM would need <=128 registers
// per thread, impossible with a 128-float `oacc`, and <=128 TMEM columns,
// impossible once S, O and the scale factors coexist.  So more warps per
// scheduler means ONE CTA of 512 threads over all 512 TMEM columns, with O in
// TMEM - and that is a *key split*, which this candidate already has: KSPLIT=n
// gives n warpgroups, KHALF=128/n keys each, OCOLS=D/n output channels each, one
// score tile read exactly once (SINGLE_READ) and one PV MMA per group issued by
// tid 0 over all D columns regardless of n.
//
// What has never been measured is n=4.  The two n=2 measurements that exist both
// carry a confound: `agu1` forced O into TMEM at KSPLIT=1, which halves the warps
// per SM to four (+24.7% on s7), and the shipped KSPLIT=2 D=128 form is the
// software-pipelined one, whose four-stage ring, second score tile, cp.async
// staging and five CTA-wide barriers per group put it at 233.19 against 205.35 on
// s7 (pH34/agk2).  n=4 is the first variant that adds warps without adding a CTA
// and without removing one: 16 warps per SM, four per scheduler, the same total
// instruction stream, and per thread one quarter of the scores (32 floats, one
// `tcg_ld32`), one quarter of the exponentials and no register accumulator at
// all.  TMEM traffic per group falls threefold against KSPLIT=1's three readouts,
// which is the mechanism `ab1` priced at -36.4% of s7.
//
// MEASURED AND REJECTED (`agv1`, `dv_3e1763d15eae`, base `best/ao10_full`, one
// allocation, two rounds, 6/6 PASS in all four with zero mismatched elements,
// `<128,0,4,true>` 114 registers and `<256,0,4,true>` 128, both zero stack and
// zero spills, so this is not a spill story):
//
//   case  D    form                    base r1/r2 (us)   KSPLIT=4 r1/r2    min delta
//   s7    128  fused, 2 CTAs/SM        177.06 / 176.00   262.24 / 261.70   +48.1%
//   s1    128  fused, 2 CTAs/SM        177.89 / 177.41   271.97 / 271.90   +53.0%
//   s4    256  fused, 1 CTA/SM          87.46 /  87.84     89.06 /  89.54   +1.8%
//   s6    256  fused, 1 CTA/SM          85.31 /  85.89     86.43 /  85.76   +0.5%
//   s0    128  control, pipelined       38.18 /  38.88     38.88 /  38.88   +1.0%
//   s5    256  control, split           77.79 /  77.86     77.57 /  77.66   -0.3%
//
// **s4/s6 are the clean experiment and they say no.**  At D=256 the shipped form
// is already one CTA per SM with O in TMEM and a single score readout, so going
// from KSPLIT=2 to KSPLIT=4 changes exactly one thing: eight warps per SM become
// sixteen, i.e. two warps per scheduler become four.  Occupancy, TMEM columns,
// TMEM traffic, total instruction count and register pressure all stay put - and
// the kernel gets 0.5-1.8% SLOWER.  Doubling the warps available to each scheduler
// therefore buys nothing when the warps are lockstep copies of one another: their
// stalls are the same stall at the same place, so the scheduler still has nothing
// to issue while the CTA waits on its MMA, its barrier or its TMEM readout.
//
// s7/s1 then price what it costs to buy those warps with the second CTA: the
// D=128 fused form gives up two independent CTAs (two independent barrier
// streams, two independent MMA waits) for one lockstep CTA of sixteen warps, and
// that is +48%/+53% - far worse than the +24.7% `agu1` measured for ONE CTA of
// four warps and worse than the pipelined KSPLIT=2 form's +13.5%.  Independence
// is worth more than warp count, monotonically: 2 independent streams of 4 warps
// (shipped) > 1 stream of 8 (pipelined) > 1 stream of 4 (agu1) > 1 stream of 16
// (this).
//
// What survives is the generalisation this needed - `tcg_combine_halves` replaces
// four hard-coded two-way folds and is bit-identical at KSPLIT<=2, `SB_HALVES`
// sizes the exchange scratch, and the pipelined loop is pinned to KSPLIT=2 - so a
// future form can pick any power-of-two split.  What this closes is "more warps
// per scheduler" as a goal in itself: the only version of it left is warps with
// DIFFERENT jobs and point-to-point handoffs, where a stall in one role is issue
// slots for another.  See results/summaries/2026-10-19-agv1-ksplit4-rejected.md.
#ifndef FMHA_FUSED_KSPLIT_D128
#define FMHA_FUSED_KSPLIT_D128 1
#endif
#ifndef FMHA_FUSED_KSPLIT_D256
#define FMHA_FUSED_KSPLIT_D256 2
#endif

// Whether a CTA whose 128-row tile is mostly padding skips the dead rows' softmax
// work, and orders its rows so that "dead" is a warp-uniform property.
//
// A tile packs `Qph = 128/G` queries times `G = H/KH` heads.  Whenever `Sq` is
// smaller than one tile of queries - which `shape_range` makes common, since
// B*Q <= 16384 with KV up to 262144 pushes exactly the short-Q/wide-GQA corner -
// most of those 128 rows belong to queries that do not exist.  Frozen shape 2
// (B8 H16 KH1 Sq=1) has **16 live rows of 128** and shape 3 (B4 H32 KH8 Sq=4)
// has 16 of 128; domain d7 has 48 of 128 and shape 5 has 96 of 128.  Every one of
// those dead rows still costs a full TMEM score readout, a 128-wide maximum
// reduction, 64 `expf2` calls, an E4M3 QDQ, a probability store and a PV
// readback per key group, and none of it is read by any output, any LSE or any
// partial: the epilogue already stores under `if (valid)`.
//
// The rows were ordered head-major (`i = q0 + (m & (Qph-1))`, `h = kh*G +
// (m >> log2Qph)`), which spreads the live rows at a stride of Qph, so *every*
// warp of a short-row CTA held both live and dead rows and nothing could be
// skipped warp-uniformly.  Query-major (`i = q0 + (m >> log2G)`, `h = kh*G +
// (m & (G-1))`) makes the live rows a **prefix** of [0,128), because
// `128 == G * Qph` and G is a power of two (`plan_geometry` returns early
// otherwise).  That is a pure permutation of the same 128 rows: each row's own
// arithmetic, its Q bytes, its scale word, its output address and its LSE address
// are untouched, so every output bit is unchanged, and it is what makes
// `rowsLive = __any_sync(valid)` a whole-warp decision.
//
// Row 0 of a tile is always live, so warp 0 always has live rows: `tid == 0`
// still reaches every MMA issue, and warps 0..3 still stage the scale atoms.  The
// skip therefore covers only the per-warp arithmetic - the group maximum, the
// delayed-maximum rescale, the probabilities, the PV readback - and leaves every
// `__syncthreads()`, every `mbar_wait` and every `.sync.aligned` tcgen05 op that
// another warp depends on exactly where it was, so the mbarrier phases and the
// TMEM/smem ordering proofs are unchanged.  A dead warp parks on the same waits
// it always did and issues almost nothing between them.
//
// MEASURED (`agv2`, `dv_1487c5d6f195`, base `best/ao10_full`, one allocation, two
// rounds, 8/8 PASS in all four with zero mismatched elements) and then GATED: with
// the guard compiled into every instantiation the win and the cost land on
// different shapes.
//
//   case  D    live rows of 128  pad ratio  base r1/r2   guarded r1/r2   min delta
//   s2    128   16               8.00       41.50/42.62   38.18/39.20    **-8.0%**
//   d7    128   48               2.67       61.25/61.95   60.54/60.26    **-2.7%**
//   s3    128   16               8.00       31.55/31.78   32.45/32.10    +1.7%
//   s5    256   96               1.33       77.15/77.63   82.82/83.39    **+7.0%**
//   s9    128  1025/9 tiles      1.12      162.53/164.48  164.06/164.29   +0.1%
//   s7    128  full tiles        1.00      174.62/173.98  179.14/179.52   **+2.6%**
//   d5    128  full tiles        1.00       76.83/76.90    76.48/76.83    -0.4%
//   s0    128  full tiles        1.00       39.87/40.00    40.22/40.35    +0.9%
//
// The gain tracks the pad ratio exactly - it is -8.0% where three of four warps
// are dead and +7.0% where only one of eight is - and so does the cost: s7
// compiles the same *mapping* it always had (G=1 makes query-major and head-major
// the same permutation) and still reads +2.6%, and the register census names the
// mechanism on the shape that regressed most, `<256,2,2,true>` **206 -> 253** and
// `<256,0,2,true>` 206 -> 254, because guarding pass 1 and pass 2 separately keeps
// the 64 `SINGLE_READ` scores live across two conditional blocks instead of one
// straight line.  At 256 threads that still fits one CTA per SM, so it is not
// occupancy; it is ptxas scheduling around a register file it no longer has room
// in.
//
// Hence `PSK`: the skip is now a template parameter, and the dispatcher picks it
// only where at least half of every tile's row slots are padding
// (`2 * Sq * G <= qtiles128 * 128`).  Under `shape_range` that is exactly
// `Sq*G <= 64`, which forces `qtiles128 == 1` and therefore
// `base = B*KH <= 16*8 = 128 < resident`, i.e. every PSK-eligible launch already
// takes the split path - so gating it to the two split passes costs no coverage
// and leaves every fused instantiation, and every lightly padded split one,
// compiling the code it compiles today.
#ifndef FMHA_PAD_SKIP
#define FMHA_PAD_SKIP 1
#endif

// Unlike tcg_qk_ahead this one is NOT a compile-time predicate over (D, PASS,
// KSPLIT): agu5 measured the form winning at 3.46 waves and losing at 1.73 and
// 0.86, so whether to use it depends on the launch geometry, and both variants have
// to be compiled for the dispatcher to choose between them at runtime.  It is
// therefore an explicit template parameter on TcgTmem, TcgMinBlocks, the kernel and
// launch_tcg, defaulted to false everywhere else, and the only decision site is the
// fused D=128 branch in fmha_mxfp8_qkv.cu.

// How many 128-thread warpgroups run the group-maximum pass, per head dimension.
// That pass produces no numerator, so its TMEM allocation is 256 columns at either
// setting and two CTAs share an SM either way; KSPLIT=2 doubles the threads per
// CTA to 256 and halves the keys each of them reduces, which is the only way to
// put more warps on an SM whose third CTA is excluded by the TMEM budget.
//
// Job ap1 (dv_0a7452f443be, ncu 2025.4.1.0, --clock-control none) measured the
// KSPLIT=1 form on shape 2 as the most starved kernel in the candidate: 13.25 us
// of a 46 us shape and 20-40% of every split-path case, at 8.24% SM throughput,
// 0.10 issued warps per scheduler, 89.55% of cycles with nothing eligible, 16.34
// cycles per issued instruction and 54.9% of its stall cycles on an L1TEX
// scoreboard.  Its achieved occupancy was 10.47% against a theoretical 18.75%:
// three CTAs fit by registers (166 x 128) and by shared memory (233472 / 69632),
// but each allocates 256 of the SM's 512 TMEM columns, so the third parks in
// tcgen05.alloc and the occupancy API - which knows nothing about TMEM - reports
// three anyway.  Two CTAs of 256 threads occupy the same 512 columns and give the
// pass sixteen warps instead of eight, and they make the API's answer agree with
// the TMEM budget, so tmem_safe_smem stops having an opinion about it.
//
// The two half-maxima are combined through the existing `oSb` exchange that the
// other key-split forms already use.  `max` is exact and order independent, so the
// value written to p.bm is bit-identical to a single 128-wide reduction, which is
// what keeps the delayed-maximum recurrence - and therefore every probability -
// unchanged.
//
// The register cap that two blocks of 256 threads impose (128, against the 159-166
// the pass used when it could) turned out not to bind: ags1's census
// (dv_ad5f741c8ad1, same job as the timing below) gives **112 registers and zero
// bytes of stack for <128,1,2,false>, 106 for its TMA variant, 118 and 112 for the
// two <256,1,2,*> forms** - halving KHALF to 64 halves the live score data, so the
// split form fits under the cap with room to spare and spills nothing.
//
// What ags1 then measured, one allocation, two alternating rounds against
// `best/ao5_full`, 8/8 PASS and zero mismatched elements in all four rounds, is
// that the win belongs to D=256 alone (min-of-both-rounds, the statistic the
// harness's own SPREAD_ALERT rule asks for once a case's spread exceeds 1.3 - s2's
// does, at 1694, because sample 0 is cold):
//
//   case  D    groups/CTA  GM loop      base     new     delta
//   s5    256      ~9      generic      77.98    74.85   -4.02%
//   d3    256      ~8      generic      62.40    60.80   -2.56%
//   d7    128      4       generic      61.38    59.65   -2.82%
//   s3    128      2       generic      30.50    30.21   -0.94%
//   s2    128      4       generic      40.38    40.80   +1.03%
//   s9    128     33       pipelined   171.94   173.34   +0.82%
//   s7    128   fused control         204.70   203.46   -0.61%
//   s4    256   fused control          89.06    89.66   +0.68%
//
// The two controls bound the noise at about +-0.7%, which is ao1's floor.  Both
// D=256 cases moved past it in both rounds independently (s5 -2.47%/-2.29% and
// d3 -1.64%/-3.76% on medians), so that half is a result; the four D=128 cases
// straddle zero and disagree between rounds (d7 read +2.05% then -2.83% on
// medians, s3 +0.88% then -4.70%), so that half is not.  The mechanism is
// consistent with the split too: D=256 stages twice the bytes and issues twice the
// k-steps per group, so it has twice the latency for sixteen warps to hide, while
// the D=128 form that regressed is the one whose pipelined loop already hides the
// QK wait that the extra warps would otherwise cover.
#ifndef FMHA_GM_KSPLIT_D128
#define FMHA_GM_KSPLIT_D128 1
#endif
#ifndef FMHA_GM_KSPLIT_D256
#define FMHA_GM_KSPLIT_D256 2
#endif

// Whether the scale-factor atoms ride a 1-D `cp.async.bulk` folded into the same
// mbarrier that the K and V tiles already complete on, instead of 32-64
// sixteen-byte `cp.async`s plus a `commit_group`/`wait_group` pair of their own in
// the hot loop.  Only reachable where the tiles are already bulk (the atoms'
// completion has to be provable by the barrier the loop waits on, and the cp.async
// path has no barrier), so it applies to every TMA form - the D=128 group-maximum
// pass included, which is the kernel ap1 measured at 54.9% of its stall cycles on
// an L1TEX scoreboard.
//
// `ab1` (`dv_5d981d5c6167`) prices the whole mechanism by deleting every
// `cp_async16` from a copy of `best/ao6_full`: on s7 208.54 -> 191.49 us (-8.2%)
// and on s1 209.38 -> 201.12 (-3.9%), where K and V are already on
// `cp.async.bulk.tensor` so the only cp.async left to delete IS the atoms.  That
// was the ceiling; `agu2` (`dv_9b1966cc93b9`, base `best/ao6_full`, one
// allocation, two alternating rounds, 8/8 PASS in all four with zero mismatched
// elements) measured the mechanism and it lands at three quarters of it on the
// shapes that matter, and goes the other way at D=256:
//
//   case  D    form                       r1      r2      min     verdict
//   s7    128  fused sequential, TMA    -6.72%  -6.85%  -5.85%   win
//   s1    128  fused sequential, TMA    -6.12%  -6.05%  -6.28%   win
//   s9    128  split, pipelined GM      -3.71%  -3.10%  -3.17%   win
//   s2    128  split, generic GM        -3.32%  -1.73%  -1.41%   win
//   d7    128  split, windowed          -1.93%  -0.25%  -0.41%   neutral
//   s6    256  fused, TMA               +1.55%  +1.95%  +1.68%   REGRESSION
//   s5    256  split, TMA               +0.66%  +0.54%  +0.25%   REGRESSION
//   s0    128  fused pipelined, cp.async +0.55%  +0.70%  -0.24%   control, unchanged code
//
// Both rounds agree on every case and the control is inside +-0.7%, so the D=256
// regression is real and not drift.  Hence the D == 128 term in `SF_BULK` below.
// The asymmetry is consistent with the atom bytes being 1024 of a 16384-byte stage
// at D=128 against 2048 of a 65536-byte stage at D=256: at D=256 three extra bulk
// requests queue behind four 16 KB tensor copies on the same engine and buy back
// proportionally less, while at D=128 they replace 32 LDGSTS and a
// commit/wait pair that the loop was actually stalling on.  Getting D=256's share
// back - issuing the atoms ahead of the tiles, or only K's - is its own
// experiment, not a reason to hold back the D=128 half.
//
// The register census agrees with the mechanism: every TMA instantiation got
// cheaper and no cp.async one moved (`<128,1,1,true>` 166 -> 152,
// `<256,1,2,true>` 112 -> 104, `<256,2,2,true>` 208 -> 196, `<256,0,2,true>`
// 220 -> 216, `<128,0,1,true>` unchanged at 255, zero stack everywhere).
#ifndef FMHA_TCG_SF_BULK
#define FMHA_TCG_SF_BULK 1
#endif

// Whether a 32-key chunk's sixteen probability pairs accumulate their denominator
// into four partial accumulators and a depth-2 tree instead of one sixteen-link
// serial chain.  ap2 (`dv_f700aec38d67`) puts `wait` - a fixed-latency dependency,
// i.e. not enough independent work in flight - at 0.84 of the 4.17 cycles per
// issued instruction on s7 and 0.83 of 3.96 on s1, the largest real stall on both,
// and this accumulator is the only loop-carried dependency inside a chunk: every
// one of the sixteen `expf2` calls is independent of the others.  Three extra
// packed adds per 32 keys is the whole cost.
//
// It reassociates the denominator, so `den` differs from the serial order by about
// 1e-7 relative.  That cannot flip a probability: each E4M3 quantization is
// finished before the denominator exists, so the numerator is bit-identical.  The
// reference's own denominator is a torch reduction over the group, so this
// candidate has never reproduced its association order - it reproduces its value to
// ~1e-7, which is what the 1.5e-7 max_rel on `lse_tensor` in every official run
// already is, against rtol 0.05 and atol 0.01.
// Measured by agu4 (dv_75499b6dc49a, base dc0b0ff, one allocation, two rounds,
// 8/8 PASS in all four, zero mismatched elements, and o_tensor's max_rel_to_ref
// identical in both arms on every case while lse's moved only in its seventh
// digit - exactly the ~1e-7 reassociation this is supposed to cause).  Ungated, it
// splits cleanly by how many warpgroups a CTA has:
//
//   case  form                              r1      r2      min      KSPLIT
//   s7    D=128 fused sequential          -8.71%  -7.40%  -6.76%       1
//   s1    D=128 fused sequential          -6.24%  -6.98%  -7.10%       1
//   s9    D=128 split                     -1.54%  -1.02%  -1.17%       1
//   d5    D=128 fused sequential          +0.68%  +0.57%  -1.07%       1  (spread 1.37/1.43)
//   s2    D=128 split                     -0.39%  +4.20%  +0.16%       1  (spread 4.15/3.92)
//   s0    D=128 fused PIPELINED           +1.17%  +2.04%  +1.97%       2
//   s4    D=256 fused                     +2.78%  +3.00%  +1.31%       2
//   s6    D=256 fused                     +2.70%  +2.77%  +1.91%       2
//
// Every KSPLIT=1 case is neutral-to-better and every KSPLIT=2 case is worse in
// both rounds, so the switch is gated on `KSPLIT == 1` below.  The reading is that
// a one-warpgroup CTA has four warps and nothing else to cover a sixteen-link
// dependency, while a two-warpgroup CTA already has eight warps in one CTA and
// pays for the three extra live accumulator pairs instead of gaining from them -
// `<256,0,2,true>` went 208 -> 210 registers with them.
#ifndef FMHA_DEN_PAR
#define FMHA_DEN_PAR 1
#endif

// Whether the probability bias is a compile-time constant per head dimension rather
// than a runtime field.  reference.py's cast is `16*exp(score-normalizer)` at D=128
// (P_CAST_LOG2_SCALE=4) and unshifted at D=256, and the host publishes exactly
// `p.bias = (D == 128) ? 16.0f : 1.0f`, so at D=256 the hot loop's
// `f32x2_mul(expf2(arg), bias2)` multiplies by exactly 1.0f.  Multiplication by
// 1.0f is the identity for every finite value, every infinity and both zeros, so
// deleting it is not an approximation - but ptxas is not entitled to fold it
// itself, so the deletion has to be explicit.  ap4's census puts the softmax at
// three packed multiplies and three packed adds per key pair, so this is one of
// about twenty-two instructions per pair on the D=256 shapes (s4, s5, s6, d2, d3,
// d6, d8).
//
// Folding the D=128 bias of 16 = 2^4 into `expf2`'s `bits(b) << 23` instead of
// multiplying afterwards is a different question and is NOT covered by this
// switch: that shift is where the wraparound for very negative arguments lives, and
// only `harness/bench/expf_packed_check.cu` over all 2^32 patterns can say whether
// a folded form is still bit-exact.
// Measured by agu3 (dv_9eb308633dde, base best/ao7_full, one allocation, two
// rounds, 8/8 PASS in all four with zero mismatched elements): -1.19% on the six
// D=256 cases and -1.26% on s7/s1, seven of eight cases negative in BOTH rounds
// independently, and every output's max_abs/max_rel_to_ref/rel_l2 identical between
// the arms to the last printed digit - i.e. the arithmetic did not move, which is
// what multiplying by exactly 1.0f and reading a constant instead of a param-space
// field should do.  The census agrees: <256,0,2,true> 220 -> 208 registers and
// <256,0,2,false> 230 -> 216, zero stack, zero spills.  The D=128 half of the gain
// is instruction selection rather than a deleted multiply (its bias is 16, not 1):
// bias2 becomes an immediate and the LSE's `den / bias` becomes an exact multiply
// by 0.0625f.
#ifndef FMHA_UNIT_BIAS
#define FMHA_UNIT_BIAS 1
#endif

// Whether a bias that IS a power of two gets folded into `expf2`'s exponent field
// instead of being applied by a packed multiply afterwards.  ap6's line-attributed
// census of `<128,0,1,true>` - the form shape 7, shape 1 and d5 all run - puts
// `fmha_tcg_ptx.cuh:358`, the `mul.rn.f32x2`, at 384 instructions and 8.1% of the
// kernel, the single largest line in it, and three of those multiplies run per key
// pair: `score * scale`, `2^n * ex2(f)` inside expf, and `p * bias`.  The first two
// are the contract and the arithmetic; the third is a multiply by 16, i.e. an add of
// 4 to an exponent field that is being built by a shift anyway - one IMAD instead of
// a SHF and an FMUL2.
//
// Unlike FMHA_UNIT_BIAS this is NOT exact by inspection, and it has now been
// measured and REFUTED: job ex2 (dv_5967a35b88cf) walked all 2^32 float patterns in
// both halves of every pair and `expf2_bias<4>` differs from `expf(x) * 16.0f` on
// **1022782562 of them per half, about 24%**, first offenders 0x42b3ad80..87, i.e.
// x ~= 89.8, right at expf's overflow threshold.  The same job's `expf2_bias<0>`
// control is BIT_EXACT on finite, infinite and NaN in both halves, which is what
// makes the L=4 row the fold's fault and not the harness's - and which re-proves
// that `expf2 == expf2_bias<0>` after the refactor.
//
// The reason is that `bits(b) << 23` is not a plain exponent-field write.  expf's
// range reduction relies on that shift WRAPPING: the saturating cvt clamps its
// input, the nine surviving bits routinely exceed the eight-bit exponent field and
// land in the sign bit, and the result is only correct because ex2's argument
// compensates.  Adding `4 << 23` to the shifted pattern therefore does not add 4 to
// an exponent - for a quarter of all inputs it flips or carries into a sign bit.
// Multiplying the finished product by 16 is exact for every finite value, both
// infinities and both zeros; adding to the pattern that encodes it is not the same
// operation.
//
// So this stays 0 permanently unless someone has a different fold, and any new L has
// to be added to `kBiasL` in harness/bench/expf_packed_check.cu and pass that check
// first.  The census line that motivated it (fmha_tcg_ptx.cuh:358, 384 instructions,
// 8.1% of <128,0,1,true>) is contract arithmetic and stays.
#ifndef FMHA_FOLD_BIAS16
#define FMHA_FOLD_BIAS16 0
#endif

// The contract's probability bias, as the single definition both the host's Params
// and the kernels read, so the constant the softmax folds cannot drift from the
// value the host publishes.
template <int D>
struct ProbabilityBias {
  static constexpr float value = (D == 128) ? 16.0f : 1.0f;
};

// Whether the key-split group-maximum form asks ptxas to keep two CTAs resident
// (2) or leaves it free to use up to 255 registers and run one (1).  Only that
// form reads this switch; every other instantiation keeps the value it has always
// been compiled with, derived from its TMEM allocation.
#ifndef FMHA_GM_MIN_BLOCKS
#define FMHA_GM_MIN_BLOCKS 2
#endif

// Whether the D=128 fused key-split form uses the software-pipelined group loop.
//
// This is the switch that decides how many times a D=128 fused CTA reads its
// 128-float tiles out of TMEM per key group, and job `ab1` (`dv_5d981d5c6167`,
// one allocation, `harness/bench/ablate_apply.py` against a copy of
// `best/ao6_full`, reference-free timing, 9 samples) priced that readout on the
// two largest shapes in the suite:
//
//   ablation                            s7 median     vs base    s1 median     vs base
//   base (<128,0,1,true>)                208.54          -       209.38          -
//   notmemld  (no tcgen05.ld at all)     132.54      -36.4%      142.91      -31.7%
//   noexp     (expf2 -> identity)        130.14      -37.6%      135.97      -35.1%
//   nopstore  (no probability stores)    181.98      -12.7%      197.73       -5.6%
//   nostage   (no cp.async at all)       191.49       -8.2%      201.12       -3.9%
//   norescale (no delayed-max rescale)   210.18       +0.8%      212.35       +1.4%
//
// So two mechanisms are the whole kernel: the exponential, which the contract
// pins bit-for-bit to torch's `expf` and which `harness/bench/expf_packed_check.cu`
// already proves the packed form reproduces over all 2^32 patterns, and the TMEM
// readout, which is not contract at all - it is a consequence of where the output
// accumulator lives.
//
// With O in registers (`PIPE=0`, KSPLIT=1, 256 TMEM columns, two CTAs per SM) a
// D=128 fused CTA reads three 128-float tiles per group: the scores for the group
// maximum, the same scores again for the probabilities (`SINGLE_READ` is false
// because it needs 128 score floats live next to 128 accumulator floats), and the
// PV partial to add into the register accumulator.  With O in TMEM (`PIPE=1`,
// KSPLIT=2, 512 columns, one CTA per SM) it reads one: `SINGLE_READ` becomes true
// and the PV accumulates in TMEM until the epilogue.  That is a 3x reduction in
// the mechanism `notmemld` prices at a third of the runtime.
//
// The pipelined loop is only one of the two ways to get O into TMEM, and it is the
// only one that has been measured: pH34/agk2 put it at 233.19 against 205.35 on s7
// and agt2 put the same loop with bulk-tensor staging at 246.1.  But that form
// also carries a four-stage ring, a second score tile, cp.async staging and five
// CTA-wide barriers per group with every one of its eight warps arriving at them
// together, because one CTA per SM has no second CTA to issue while it waits.
// Setting this to 0 keeps O in TMEM and `SINGLE_READ` and drops back to the
// sequential loop - the exact combination the D=256 fused form ships with, where
// it is the fastest form on the card - so the TMEM saving can be measured without
// the pipelining and staging changes bundled into it.
// Measured and rejected by agu1 (dv_58dc70d197c2, base best/ao6_full, one
// allocation, two rounds, 7/7 PASS in all four, zero mismatched elements, both
// controls inside 0.2%): 0 costs s7 +24.7% and s1 +22.0% on min, d5 +10.2%, s0
// +7.2%.  It removes two of the three TMEM tile readouts per group that ab1 priced
// at -36.4% of s7, and still loses, because O in TMEM at D=128 needs 280-304 of the
// 512 columns and tcgen05.alloc granularity is a power of two, so the form runs one
// CTA per SM instead of two.  Two CTAs per SM is worth about +48% here against the
// -24% the readouts were worth.  See
// results/summaries/2026-10-19-agu1-o-in-tmem-sequential-rejected.md, which also
// records the only route left to one readout at two CTAs per SM: a 64-key score
// tile, S(64) + O(128) + SF(24) = 216 columns.
#ifndef FMHA_PIPE_D128
#define FMHA_PIPE_D128 1
#endif

// EXPERIMENT bl1 - the group-split ("AG") D=128 fused form.  Same 256 threads, same
// one CTA per SM, same 512 TMEM columns, same single score-tile readout per group and
// same accumulator-in-TMEM as the software-pipelined key split above, i.e.
// traffic-identical to it - but the two warpgroups take ALTERNATING WHOLE key groups
// instead of alternating halves of the same group, so the SM has two phase-offset
// streams instead of one lockstep one.  The contract's delayed maximum is what holds
// them half a group apart: group g+1's `safe` is a function of group g's maximum, so
// one warpgroup is always reading scores out of TMEM while the other is computing
// exponentials.  See Addendum 3 of
// results/summaries/2026-10-21-design-warp-specialised-d128.md for the derivation, the
// TMEM map (checked by harness/analysis/tmem_map_check.py), the ring depths and the
// two handoff-slot invariants.
//
// The hurdle is stated there as an IPC ratio rather than a percentage, because AG and
// PIPE move the same bytes: `bk1` put forced PIPE at +31.8% on s7 against `ab1`'s
// -12.1% per readout tile, so PIPE's structure costs +74% and runs at 0.271 issued
// warp-instructions per scheduler against the shipped form's 0.47.  AG beats the
// shipped form on s7 only above 0.357 - but on the five cases that already dispatch to
// the key split its competitor IS PIPE, and there the bar is any recovery at all with
// thread count, register file, TMEM columns, readout count and occupancy all equal.
// MEASURED AND PROMOTED for the launches the wave-packing argument already gives one
// CTA per SM: bl2 put it at -5.10% on frozen s0 and -7.03% on frozen s8 against the
// pipelined key split it replaces there, both rounds agreeing and three byte-identical
// controls at -0.51%.  Its scope is those launches and nothing wider; see
// FMHA_GSPLIT_FORCE_D128 for the three shapes where it was measured and rejected.
// 1 is the shipped behaviour; 0 turns the form off.
#ifndef FMHA_GSPLIT_D128
#define FMHA_GSPLIT_D128 1
#endif

// Also take the group split where the wave-packing argument does NOT ask for a key
// split, i.e. on the launches that currently run the two-CTA-per-SM three-readout
// sequential form (s7 at 512 CTAs, s1 at 256) or the 64-column form (d5 at 1024).
//
// MEASURED AND REJECTED by bl2 (dv_f343db901d9b, base best/bf2_full = 13f1535, one
// allocation, two alternating rounds, 8/8 PASS in all four runs with zero mismatched
// elements, three byte-identical controls at -0.51% as a group).  Bias-corrected
// medians, both rounds agreeing in sign and within 1.1 points of each other:
//
//   case  what AG replaced            r1      r2     mean
//   s7    sequential, 2 CTA/SM     +24.09  +25.16  +24.62%
//   s1    sequential, 2 CTA/SM     +20.32  +20.07  +20.19%
//   d5    64-column score tile     +30.73  +28.99  +29.86%
//   s0    pipelined key split       -4.86   -5.33   -5.10%
//   s8    pipelined key split       -6.54   -7.52   -7.03%
//
// So the form wins exactly where its competitor is the key split - same one CTA per SM,
// so nothing is being paid for occupancy - and loses everywhere it has to buy that
// occupancy back.  Addendum 3 stated the bar as an IPC ratio: AG and PIPE move one
// score tile per group against the sequential form's three, so against the sequential
// form AG needs IPC above 0.357 to break even, and PIPE's forced +31.8% on s7 puts
// PIPE at 0.271.  AG's +24.62% on the same shape and in the same tree puts it at
// 0.47*0.758/1.2462 = 0.286 - it recovers 5% of the distance from PIPE to break-even,
// and the missing 24% is the second CTA per SM, which no rearrangement of eight warps
// inside one CTA has now recovered in five attempts (agu1, agk2, agv1, PIPE, AG).
// d5's +29.86% is the same fact against a different incumbent: at 1024 CTAs the
// 64-column form runs two CTAs per SM and AG halves that.
//
// 0 is therefore the shipped behaviour, and the group split is taken only where the
// wave-packing argument already asked for one CTA per SM.
#ifndef FMHA_GSPLIT_FORCE_D128
#define FMHA_GSPLIT_FORCE_D128 0
#endif

// Fewest key groups the widest CTA of a launch may have before the group split is
// refused.  Each warpgroup needs at least two groups to fill its pipeline - QK(g+2) is
// issued a whole iteration ahead - and a span of one group leaves warpgroup 1 with
// nothing to do, i.e. a 128-thread CTA occupying an SM to itself.  The gate reads the
// host's `useful_key_span`, which is the WIDEST CTA's span, so a causal launch still
// runs the form on its cheap short-span tiles; those are also the tiles LPT orders
// last.  Same shape of gate as the pipelined group-maximum loop's `(cHi-cLo) >= 7`.
// 8 was a construction when `bl2` was accepted, and `bp1` then measured it against 4 -
// the only setting that moves a named case, since d9 (span 5, 64 fused CTAs) is the single
// point in the gap and d1/d4 (spans 2 and 1) stay on the key split either way.  bp1
// (dv_ec996019cbe4, base a3f9a25, one allocation, two rounds, 7/7 PASS in all four runs
// with zero mismatched elements) reads **d9 +5.52% then -2.92%**, i.e. the two rounds
// disagree in sign on a 19 us case, and its six byte-identical controls swing from -3.51%
// to +5.85% in the same allocation, so the bias correction of +0.37% is not the noise
// floor - the movement is.  Lowering the gate is therefore not measured to help and is not
// taken: 8 stays, now on a rejection rather than on a construction, and the form's win does
// not extend down to a span where a warpgroup has two or three groups.
#ifndef FMHA_GSPLIT_MIN_GROUPS
#define FMHA_GSPLIT_MIN_GROUPS 8
#endif

// EXPERIMENT br4, narrowed from br2/br3: the sequential loop's FOURTH per-group
// `__syncthreads()`, the one at the foot of the body after the PV-partial readback - removed
// ONLY where the accumulator lives in TMEM, because there the barrier is not redundant with a
// later one, it is vacuous: there is no readback at all.  Its job was to order every thread's
// `tcgen05.ld` of that partial - which at D=128 KSPLIT=1 comes out of TMEM columns [0,128),
// the same columns the next group's QK^T overwrites, because the scores are dead by then and
// that sharing is what keeps the allocation at 256 columns and two CTAs per SM - before tid 0
// issues that QK.  But the next QK issue already sits behind TWO CTA-wide barriers of the next
// iteration (the staging fence and the scale-atom fence), every thread reaches both after its
// own readback in program order, and `tcg_ld32` carries its own `tcgen05.wait::ld`, so the read
// is complete in the reading thread's order before it arrives at either.  tid 0's
// `tcg_fence_after()` on the far side is unchanged.  For every form with O in TMEM there is no
// readback at all and the barrier protects nothing whatsoever.
//
// Why the narrowing.  Two jobs measured the wide version.  `br1` (dv_223de3d08d68, two rounds,
// stacked on the bq1 tree) read s7 -0.38%, s1 -0.69%, s9 -1.01%, s4 -1.48% bias-corrected, all
// four consistent in sign.  `br3` (dv_000a09a06901, four rounds, twelve cases, single-variable
// against best/bo3_full) read every one of its eight treated cases negative on the raw median -
// s4 -1.45%, s5 -1.03%, s3 -0.91%, s6 -0.89%, s1 -0.84%, s2 -0.81%, s9 -0.59%, s7 -0.49% - and
// every one of its four byte-identical controls negative too, s0 -0.38%, s8 -0.44%, d9 -0.37%,
// d1 -2.24%, so the bias-corrected answer is -0.016% over 32 readings.  The arm bias in that
// allocation was the same size as the effect, which is the floor this campaign has now measured
// three times.  What survives it is the sub-signal: **s4 and s6 were negative in six of six rounds
// across the two jobs** (br1 -1.82/-1.13, br3 -1.51/-1.19/-1.71/-1.38 on s4) with their mins
// agreeing, while the controls flipped sign round to round.  s4 and s6 are the D=256 fused form,
// where the accumulator is in TMEM and there is no readback for the barrier to order.
//
// So this version removes the barrier exactly where the argument is vacuity rather than
// redundancy: with O in TMEM nothing at the foot of the body reads TMEM, every thread has already
// waited on this group's PV itself, the next iteration's staging writes a different ring slot than
// the one that PV read, and the next QK issue - which is what overwrites the score tile - sits
// behind two CTA-wide barriers of its own.  The D=128 sequential and split forms keep the barrier,
// because there the partial really is read out of the same columns the next QK overwrites and the
// claim is only that the next iteration's barriers subsume it.  That also makes s7, s1 and s9 -
// the three largest frozen cases - byte-identical controls for br4, which is the control set the
// wide version lacked.
//
// Kept unconditionally for `L::QK_AHEAD`, where the next group's QK is issued at the END of the
// current iteration rather than at the start of the next; that configuration is not launched
// (FMHA_TCG_QK_AHEAD is 0) and agq4 rejected it at +1.2%.  1 is the shipped behaviour; 0 removes
// the barrier where it is vacuous.  Promoted by `bt3` at 0.
#ifndef FMHA_SEQ_TAIL_BARRIER
#define FMHA_SEQ_TAIL_BARRIER 0
#endif

// EXPERIMENT bu1: the SECOND half of the wide tail-barrier removal, split off from the first so
// that the controls stay larger than the movers - the rule `br3` paid for.  Where the accumulator
// lives in registers the barrier is not vacuous but REDUNDANT: at D=128 KSPLIT=1 the PV partial is
// read out of TMEM columns [0,128), the same columns the next group's QK^T overwrites, so the
// readback really must be ordered before that issue - but the issue sits behind the next
// iteration's staging barrier AND its scale-atom barrier, every thread reaches both after its own
// readback in program order, and `tcg_ld32` carries its own `tcgen05.wait::ld`, so the read is
// complete in the reading thread's order before it arrives at either.
//
// That argument covers two launched forms, and `br3` measured both at once and could not see
// either: treating s7, s1 and s9 left only s0 and s8 as large controls, all four of its controls
// went negative along with all eight treated cases, and the bias-corrected answer over 32 readings
// was -0.016%.  So the two halves are measured separately.  THIS switch is the D=128 SPLIT pass
// (s9 132 us, s2 37, s3 31, d7 49), which leaves s7 and s1 - the two largest frozen cases, whose
// fused form keeps the barrier under the clause below - byte-identical controls alongside s4, s6,
// s0 and s8.  The follow-up job inverts it: remove the barrier for the fused form and keep it for
// the split pass, making s9 the control instead.
//
// One detail that has to hold for the split pass and not for the fused one: `rowsLive` is a
// per-WARP vote, so on a padded shape some warps execute the readback and others do not.  Both this
// barrier and the next iteration's staging barrier are outside that guard, so every thread reaches
// whichever of them exists and the ordering argument is unaffected by the divergence.
// 1 keeps the barrier (the behaviour `bt3` was accepted with); 0 removes it for the split pass.
//
// MEASURED AT 0 IN `bu1`; taken at 0 in the end, on the evidence assembled below plus `br1` and
// `br3`, because `bu2` measured the fused half cleanly and the two halves are per-instantiation
// compile-time choices that cannot interact.  What follows is the account of the job that could
// not resolve THIS half on its own.
// `bu1` (dv_44f3a2ac4e3f, base best/bt3_full, four rounds, 10/10 PASS in all eight runs,
// mismatch_zero=19 pass=29 in all eight, control bias +0.189% over 24 readings from six large
// byte-identical controls) had its target case corrupted: the BASE arm's s9 read median 168.67
// and 174.75 us in rounds 1 and 4 against its OWN mins of 134.56 and 134.50 and medians of
// 132.22/131.81 in rounds 2 and 3, i.e. two of four base rounds were ~30% slow on that one case
// while the new arm's four medians were 131.17/132.06/131.01/132.00.  The -22.4%/-24.6% those
// rounds produce is the harness's `ay2` case exactly - median and min disagree, so neither round
// can be believed - and the two usable rounds read -0.31% and -0.79%.  The other three treated
// cases read s2 +1.45%, s3 -1.58%, d7 +0.47% with signs flipping round to round, which is
// coherent rather than disappointing: s2 runs ~4 groups per CTA and d7 ~8 against s9's 33, so one
// barrier per group is a far smaller share of their runtime, and the effect should scale with
// groups per CTA.
//
// Taken at 0 because s9's two usable rounds (-0.31%, -0.79%) agree in sign with `br1`'s -1.01% and
// `br3`'s -0.59% raw on the same case, i.e. four independent readings of one case all negative, and
// because s2/s3/d7 are null rather than adverse for a reason that is understood rather than
// convenient.  It is the weaker half of the two and is quoted per-round, never as a mean of four.
#ifndef FMHA_SEQ_TAIL_BARRIER_SPLIT
#define FMHA_SEQ_TAIL_BARRIER_SPLIT 0
#endif

// EXPERIMENT bu2: the same redundancy argument applied to the D=128 FUSED sequential form - s7 and
// s1, the two largest frozen cases at ~171 and ~176 us, and the two with the longest spans (64 and
// 64 groups per CTA, so one barrier per group is at its largest share of runtime there).  This is
// the half of the wide removal `br3` could not resolve, and the reason it could not is the control
// set: `br3` treated s7, s1 AND s9 at once, which left only s0 and s8 above 35 us to estimate the
// arm bias with, and all four of its controls went negative along with all eight treated cases for
// a corrected -0.016%.  Treating the fused form alone leaves s9 (132 us), s4 (86), s6 (85), s8 (68)
// and s0 (40) byte-identical, which is the control set the rule from `br4` asks for.
//
// MEASURED AND TAKEN.  `bu2` (dv_6622ba455ef1, base best/bt3_full, one allocation, four
// alternating rounds, SELFTEST 8/8 PASS in all eight runs, mismatch_zero=15 pass=23 in all eight)
// read, bias-corrected against SIX byte-identical controls whose geomean was **+0.078%** over 24
// readings - s9 (132 us), s4 (86), s6 (85), s8 (68), s0 (40), s2 (37):
//
//   s7  -0.88  -0.63  -0.46  -1.15   mean -0.78%   mins -0.16 -0.23 -0.14 -0.19, 4/4 negative
//   s1  -0.47  -0.62  -0.83  -0.75   mean -0.67%   mins +0.02 -0.93 -0.59 -0.17, 3/4 negative
//
// **All eight treated median readings are negative and the effect is nine times the control bias**,
// the tightest separation this campaign has produced.  It also explains `br3`'s -0.016% without
// contradicting it: br3 treated s7, s1 AND s9 at once, so its only controls above 35 us were s0 and
// s8, its four controls went negative along with all eight treated cases, and its arm bias (-0.864%)
// was the size of the effect.  The effect was there; that control set could not see it.  Earlier and
// weaker readings of the same half, all the same sign: `br1` -0.38%/-0.69% over two rounds, `br3`
// -0.49%/-0.84% raw over four.
//
// With this and FMHA_SEQ_TAIL_BARRIER_SPLIT both at 0 the guard below reduces to `L::QK_AHEAD`
// alone, so every launched sequential-loop form loses the barrier and only the unlaunched
// QK-lookahead configuration keeps it.  That is behaviourally the tree `br3` ran, which is a useful
// fact to have: br3 passed 10/10 in all four runs with every correctness record equal between its
// arms and all 42 census rows identical, so the wide form has already been through a four-round
// correctness run even though its timing was unresolvable.
// 1 is the behaviour `bt3` was accepted with; 0 removes the barrier for the fused form.
#ifndef FMHA_SEQ_TAIL_BARRIER_FUSED
#define FMHA_SEQ_TAIL_BARRIER_FUSED 0
#endif

// EXPERIMENT cl1 - how the delayed-maximum rescale of the REGISTER accumulator is laid
// out, and therefore whether ptxas clones the whole main loop body to specialise it.
//
// WHY THIS KNOB EXISTS.  §131 profiled s1 under two builds that are bit-identical in
// numerics and identical in registers (247), stack (0), spills (0) and static SMEM for the
// kernel s1 and s7 actually launch, and found the slower one executes 4.66% FEWER
// instructions while taking 1.84% MORE time, because warp-cycles per issued instruction
// rose 3.29 -> 3.55.  The only stall reason that moved is `no_inst` - instruction fetch -
// and 101 of the slow arm's 278 main-bucket samples sit on ONE line: the target of the
// `@!P2 BRA` that skips the rescale below, taken once per group.  Following the addresses
// shows what that branch is hiding: its fall-through (exec 1,024 against the taken path's
// 60,928, i.e. **1.65% of group iterations**) starts with `expf2_bias`'s own constants and
// carries a SECOND complete copy of the loop body - 2,591 lines over a 70.5 KB span, with
// its own 208 FADD2, 192 FMUL2, 128 FFMA2, 130 MUFU and 128 F2FP.  So the 98.35% reach the
// primary copy by a **21-22 KB forward branch** out of a loop whose primary copy already
// spans 41.0 KB, and pay for the clone in fetch misses.  `ab2`'s `norescale` ablation
// measured -0.02% on s7 and -0.32% on s1, which priced the rescale's ARITHMETIC correctly
// and its CODE not at all.
//
// VALUES.  0 is the shipped form: `else if (alpha != 1.0f) { for t oacc[t] *= alpha; }`.
//
//   1 - UNCONDITIONAL.  Drop the test.  Bit-identical, by the argument the O_IN_TMEM branch
//       above already states for its own lanes ("multiplying by 1.0 is bit-identical to
//       skipping"), and `oacc[]` is zero-initialised before the loop so the first group's
//       `alpha == 0.0f` is a no-op as well.  There is then no branch to duplicate, at the
//       cost of +128 scalar FMUL per group (9.3% of the loop's 1,372 instructions), so this
//       is the form whose SIGN is least certain: it buys the fetch and pays the issue port,
//       and §131(b) just showed those two do not trade at parity here.
//   2 - FUSED.  Move the rescale into the PV readback as `oacc = fmaf(oacc, alpha, v)`,
//       which is the contract's own `numerator*alpha + rounded @ V` with ONE rounding
//       instead of two.  **Exactly equal to the shipped two-rounding form whenever alpha is
//       1.0f** - 98.35% of iterations - and within 0.5 ulp of it otherwise, on the OUTPUT
//       rather than on a probability, so §130(e)'s e4m3-cliff bound does not apply: 0.5 ulp
//       is ~6e-8 relative against a tolerance of 0.01 + 0.05|O|.  The common path's
//       instruction count is UNCHANGED (128 FFMA replacing 128 FADD), the branch is gone,
//       and so is the clone.  This is the form §131(f) expects to win, and it is the only
//       one that is not bit-exact, so it goes through `bitexact_ab.py`'s realized-margin
//       report rather than through an identity check.
//   3 - `__builtin_expect(alpha != 1.0f, 0)`.  Cheapest to build and the least likely to
//       work: a branch-probability hint changes which arm ptxas makes the fall-through, and
//       tail duplication is a separate decision.  Included because it is nearly free and
//       because if it does work it is the smallest possible diff.
//
// NOT INCLUDED, deliberately: making the rescale `__noinline__`.  `oacc` is a 128-element
// register array, so taking its address across a call boundary would force it to local
// memory - the exact failure §131(a)'s census exists to catch, predictable without a build.
//
// SCREEN BEFORE TIMING.  The clone is visible without a GPU: in `cuobjdump -sass` of the
// launched instantiation the contract's `F2FP.SATFINITE.E4M3` cast appears 64 times in the
// primary body and again in the clone, so its total falls by roughly a third when the
// duplication goes, and `ptxas -v` must keep reporting 0 bytes of stack.  That is what
// `experiments/clone_census.sh` counts, and §131(f)'s decision rule is to adopt whichever
// value removes the clone at the smallest increase in the primary body's line count - and
// only then to time it.
#ifndef FMHA_RESCALE_LAYOUT
#define FMHA_RESCALE_LAYOUT 0
#endif

// EXPERIMENT qs1 - how far the Q payload staging loop is unrolled, and therefore whether
// its global loads are issued AHEAD of the shared stores that consume them.
//
// WHY.  §134(b) found s5's split pass memory-latency-bound (`long_sb` 39.82%, `selected`
// 10.36%, 9.65 warp-cycles per issued instruction) with two clusters carrying 41.5% of its
// stall samples, and §135(b)2 priced the larger one at 111 of 438 `long_sb` samples = 10.1%
// of all s5 samples on TWO SASS lines: `STS.128 [R2], R16` and `STS.128 [R11], R20`, at
// exec 4,736.  §135 attributed those to the scale-factor atoms.  §137 shows the attribution
// was wrong, from the SASS immediately around them:
//
//     LDG.E.128.CONSTANT R16, desc[UR6][R4.64]     exec 3,552
//     BSYNC.RECONVERGENT B1
//     LOP3.LUT R4, R8, R9, RZ, 0xc0
//     STS.128 [R2], R16                            exec 4,736   long_sb 54   <-- waits on THAT load
//     ...
//     LDG.E.128.CONSTANT R20, desc[UR6][R4.64]     exec 3,552
//     STS.128 [R11], R20                           exec 4,736   long_sb 57
//     ISETP.NE.AND P0, PT, R12, 0x8, PT            <- loop counter compared against 8
//     VIADD R2, R2, 0x1000 / VIADD R8, R8, 0x20    <- dst += 4096, src += 32
//
// The store's source register is the load's destination, one instruction apart, and the
// counter runs to 8 in steps of 2.  That is this loop exactly: `CH = D/16 = 16` at D=256,
// `(BM*CH)/THR = (128*16)/256 = 8` iterations, `#pragma unroll 2` giving four bodies of two
// load/store pairs, once per CTA - so exec 4,736 = 4 bodies x 1,184 warps, and the atoms are
// nowhere in it (`tcg_stage_atoms` uses `cp_async16`, which lowers to `LDGSTS.E.BYPASS.128`,
// and s5's SASS has 8 of those separately).  **The Q tile's global load latency is exposed
// in full, once per CTA, because each store sits one instruction behind the load that feeds
// it.**  For s5 that lands where it hurts most: 148 CTAs at one per SM with ~3.5 groups
// each, so the prologue is a large share of a short kernel - the same asymmetry §118(c)
// measured for s0, where the prologue/epilogue costs 2.38x more in time than in instructions.
//
// VALUES.  2 is the shipped `#pragma unroll 2`.  4 and 8 widen it; 1 forbids unrolling; 0
// requests full unrolling.  A wider unroll gives ptxas 4 or 8 independent `uint4` loads to
// issue before the first store consumes one, which is the only thing that can cover this
// latency - it is not a ring-depth or a scheduling-hint question, and unlike §136's atom
// ring it touches no mbarrier, no parity slot and no ordering invariant, because the whole
// block completes before the first MMA is issued.
//
// COST AND WHY IT IS SCREENED BEFORE IT IS TIMED.  Each in-flight `uint4` is four registers,
// so batching all eight loads needs ~32 registers live.  The D=128 fused instantiation is at
// **247 of 256** (§131(a)/§133(a)), so full unrolling will spill there and 8 is likely to;
// the D=256 split instantiation runs one CTA per SM and has far more slack.  The screen is
// `experiments/clone_census.py`, which now prints for BOTH instantiations the registers,
// stack and spills and the **LDG/STS/LDGSTS interleaving in address order** - the pass
// condition is that the pattern changes from alternating (LSLSLSLS) to batched (LLLLSSSS)
// with stack and spills at zero.  Bit-exact by construction: the same bytes, the same source
// strides, the same `fp8_tile_offset` destinations, only the issue order inside a
// straight-line prologue block moves.
#ifndef FMHA_QSTAGE_UNROLL
#define FMHA_QSTAGE_UNROLL 2
#endif

// EXPERIMENT qs2 - force the Q payload staging's loads ahead of its stores IN THE SOURCE,
// because `dv_4e614d564e3b` measured that the unroll pragma does not do it where it matters.
//
// The qs1 screen (`FMHA_QSTAGE_UNROLL` 2/4/8/0, GPU-free, both instantiations censused for
// registers/stack/spills and for the LDG/STS/LDGSTS interleaving in address order) split:
//
//     fused D=128 <128,PASS_FUSED,KSPLIT=1,TMA=1>   regs 247 / stack 0 / spill 0 at EVERY value
//       unroll 2 (shipped)  SSLSLSL                  <- a 4-iteration loop of two L,S pairs
//       unroll 4            SSLSLSLSLSL              <- still alternating
//       unroll 8 and 0      SSLLLLLLLLLSSSSSSSS      <- BATCHED: nine LDG, then eight STS
//     split D=256 <256,PASS_SPLIT,KSPLIT=2,TMA=1>   regs 198 / stack 0 / spill 0 at EVERY value
//       unroll 2 (shipped)  SS LSLS L[GG-run] SL     <- the Q loop is L,S,L,S
//       unroll 4            SS LSLSLSLS L[GG-run] SL <- four sequential pairs
//       unroll 8 and 0      SS (LS)x8 L[GG-run] SL   <- EIGHT sequential pairs: fully unrolled
//                                                        and STILL not batched
//
// So ptxas hoists the loads for the D=128 fused instantiation at zero register cost -- the
// predicted spill at 247 of 256 did not happen -- and refuses to for the D=256 split one at
// every unroll value, with 58 registers to spare. That is the wrong way round: **s5 is the
// shape where the stall was measured** (111 of 438 `long_sb` samples, 10.1% of all its
// samples, on the two `STS.128` lines whose source register is the immediately preceding
// `LDG.E.128.CONSTANT`'s destination), and s1/s7 are the shapes where the prologue is only
// 7.8% of stalls and where a <=1% effect cannot be judged at all because they are
// layout-bistable (§133(g)).  Alternating L,S pairs serialise: a stalled store blocks the
// warp, so the next load is not issued until it retires, and eight pairs cost eight load
// latencies where eight batched loads cost one.
//
// VALUES.  0 = shipped, one loop.  1 = two loops over the same index range: the first fills
// a local `uint4 d[NITER]`, the second drains it to shared memory.  That is a source-level
// ordering constraint rather than a hint, so ptxas cannot interleave them, and it costs
// `NITER * 4` registers live -- 32 at NITER=8, which the D=256 split instantiation has
// (198 of 256 at one CTA per SM) and the D=128 fused one does not (247 of 256, nine spare).
// **So the screen must read the two instantiations separately, and a spill on the fused one
// means this ships gated or not at all.**  Bit-exact by construction: same bytes, same
// `p.q` strides, same `fp8_tile_offset` destinations, same `iq < p.Sq` predicate producing
// the same zero fill; only the order of independent 16-byte stores changes, and they write
// disjoint destinations.
// SCREENED AND PASSED, `dv_9fea73bec9a1` (GPU-free, two builds at 66.2 s, both launched
// instantiations censused for registers/stack/spills and for the LDG/STS/LDGSTS interleaving
// in address order).  This is §138(f)'s outcome 1, the best of the three: BOTH instantiations
// batch, at UNCHANGED resources, so the form needs no gating.
//
//     instantiation                       regs stack spill   interleave, split=0 -> split=1
//     <128,PASS_FUSED,KSPLIT=1,TMA=1>      247    0     0     SSLSLSL  -> SSLLLLLLLLLSSSSSSSS
//     <256,PASS_SPLIT,KSPLIT=2,TMA=1>      198    0     0     SSLSLSLLGGGGLGGSL
//                                                             -> SSLLLLLLLLSSSSSSSSLLGGGGLGGSL
//
// The D=256 row is the one that matters: eight loads issued before eight stores, where
// FMHA_QSTAGE_UNROLL at 2, 4, 8 and full unroll all left it as (LS)xN and never batched a
// single pair (§138(a)).  The `G` run in both rows is the six LDGSTS.E.BYPASS.128 scale-atom
// copies, which s5's profile shows carrying long_sb 0 and lg 0 -- they were never the
// problem (§137(b)/§138(c)).  Price: the D=128 function grows 4,592 -> 4,816 SASS lines
// (+4.9%), its span 71.7 -> 75.2 KB, and SHF 247 -> 266 from recomputing the destination
// addresses in the second loop; every other opcode count in the main loop is unchanged
// (FADD2 416, FMUL2 384, FFMA2 256, FFMA 277, MUFU 263, F2FP 192, FMNMX 80, FMNMX3 88,
// LDTM 20, STTM 12), so nothing in the group loop moved at all.
//
// Default 1.  The timing is NOT yet measured: §137(h)/§138(g) bound the prize at <=~0.9% of
// the fixed geomean (10.1% of s5's stall samples, and s5 is 45.024 us of 69.2336), and the
// adjudication is the same-job paired official eval against 3c1e0884 read on s5 first, since
// s5 is not one of the four layout-bistable shapes and is therefore the one place in the
// fixed suite where a <=1% candidate can actually be judged.  Revert to 0 if the frozen
// checker rejects it or the eval does not beat the incumbent.
#ifndef FMHA_QSTAGE_SPLIT
#define FMHA_QSTAGE_SPLIT 1
#endif

// pr1-P1: issue the prologue's two independent cold LDG batches (the sfq scale
// words, then the QSTAGE Q-payload load loop) ABOVE tcg_alloc, with their
// dependent stores (Q STS, sfq tcg_st1) below: the alloc+syncthreads and the
// two LDG flights overlap instead of serializing. Issue-order only — bytes,
// destinations and values identical (bit-exact). Assumes the shipped
// FMHA_QSTAGE_SPLIT=1 load/store split shape. Targets the wave-start serial
// cold chain (Q LDG ~0.8us -> sfq LDG ~0.5 -> [PDL wait] -> body's K-prime
// landing ~0.8) that forms part of the fitted 8.9us floor: gmd4 priced the
// IN-LOOP staging cover away (exactly 0.000% on s2/s3) but never touched the
// prologue, and its inter-CTA-jitter hiding cannot apply at wave start where
// every co-resident CTA issues its Q LDG synchronously (s0 fused: 0.43 waves
// = 1 CTA/SM, zero interleave available).
// Prereg: results/summaries/2026-10-25-pr1-prologue-reorder-prereg.md.
// VERDICT (pr1 screen dv_622326289bf3): mechanistically NULL ALONE — every
// band-free detector silent (s1/s9/s3 0.00%); its −1.49% screen geomean was
// favorable s0/s7 layout rolls only. SHIPPED AT 1 ANYWAY because the shipped
// artifact must be the MEASURED one: the winning H arm = LDG-order + hoist
// together, and hoist-alone was never screened. The Q/sfq overlap itself is
// below detection (<0.1us) — do not cite it as a win mechanism.
// REVERTED TO 0 (with PRIME_HOIST) after the fixed official dv_f149666d68e0
// failed correctness on ALL THREE D256 cases (4/5/6, both seeds): this flag's
// prologue block sized the sfq word batch D/32 instead of the scales block's
// DCH = D/128 — the extra tcg_st1 writes clobber SFA_P/SFB_V on D256's dense
// SF map (garbage scales into the PV MMA ⇒ 1e19-class outputs). Benign on
// D128 (clobber confined to SFB_K parity-0, rewritten every group before use
// — which is why the all-D128 screen and the 7 official D128 cases passed).
// The constant is FIXED in place; defaults went to 0 until the fixed build
// re-screened WITH D256 cases. Screen-design lesson recorded: never screen
// a whole-prologue mechanism without all four launched kernel families
// (D128/D256 × fused/split).
// RE-ADOPTED at 1 (pr2 dv_8594877d95dc, all gates passed): bit_identity
// TRUE on all six cases INCLUDING s4/s5/s6, and P1 turned out to carry a
// real D256 effect of its own — s6 (D256 fused, hoist-EXCLUDED via
// SF_AHEAD) −2.56% stable: the 256-thread prologue's two cold LDG batches
// are twice the bytes, so their overlap pays where the D128 P1-only arm
// read null. s5 −5.00% (P1 + its GM's hoist), s1 159.74 ×3 (mechanism
// reproduced), s4 −0.04% neutral.
#ifndef FMHA_PROLOG_LDG
#define FMHA_PROLOG_LDG 1
#endif

// pr1-P2: additionally hoist the bodies' slot-0 K/V/SF stage prime
// (tcg_issue_stage(..., 0, 0, cLo, ...) — byte-identical in all three loop
// bodies) to the top of the prologue so the first group's cold TMA overlaps
// the whole Q/sfq staging. Bodies skip ONLY the slot-0 ISSUE and keep every
// commit/barrier (an empty cp.async commit group is the PIPE prime's own
// idiom); SF_AHEAD forms are excluded (their prime writes TMEM atoms and
// needs the alloc first) — the generic body's normal prime arm is exactly
// the !SF_AHEAD branch, so hoist and skip pair by construction. Bit-exact.
// ADOPTED (pr1 screen dv_622326289bf3, WIN branch, all gates passed —
// bit_identity TRUE 6/6, plans identical, REG ≤251 STACK 0, checks ×2 seeds):
// s1 163.84→159.776 ×3 rounds identical = FLOOR SHIFT below every documented
// s1 state (121×1.28+8.9 → +4.84); s9 −1.7..−3.2% stable; s7 −2.5%; s0 −5.5%
// (last two carry layout-roll components — mechanism core is the band-free
// s1/s9 floor shift); s2 0.00% = the bandwidth-bound control behaving exactly
// per gmd4 Reading-1 (the mechanism moves LATENCY, not bytes). Physical model:
// the wave-start cold burst is DRAM-service-queued; K/V-TMA-first puts the
// largest transfer at the front and starts the first MMA at K-landing instead
// of after Q+sfq+K all land. The fitted 8.9us floor was ~45% recoverable
// prologue-order latency. Full attribution: the pr1 RESULTS record.
// REVERTED TO 0 pending re-validation: the ADOPTED build (3fe003d) carried
// the PROLOG_LDG D/32-vs-DCH bug and failed the fixed official
// (dv_f149666d68e0) on D256 cases 4/5/6 × both seeds (o_tensor garbage,
// relative_l2 ≈ 0.99). The HOIST mechanism itself is not implicated — the
// failing forms are exactly the ones where the P1 scales-batch clobber hits
// non-self-healing columns, and D256 fused is hoist-EXCLUDED (SF_AHEAD) —
// but the shipped artifact must be the measured one, and the measured H
// binary contained the bug. Fixed build re-screens with D256 cases before
// any re-adoption. The D128 timing evidence stands as measured (the clobber
// was benign there; 9 extra prologue instructions/thread): s1 159.776 ×3
// identical (floor shift below every documented state), s9 −1.7..−3.2%,
// s7 −2.5%, s0 −5.5%, s2 0.00% bandwidth control.
// RE-ADOPTED at 1 (pr2 dv_8594877d95dc on the FIXED build, all gates
// passed): bit_identity TRUE 6/6 including all three D256 forms; s1
// 159.74 ×3 identical (≤160.1 attribution gate MET — mechanism intact in
// the fixed build, 0.036us better than the buggy screen build); s9
// −1.70% ×3; s5 −5.00% (GM D256 prologue treated, never screened in
// pr1); s6 −2.56% (P1-only form); s0 −0.32%; s4 −0.04%. Six-case
// geomean −2.034%. Lottery note for officials: the pr2 A arm drew s0's
// FAST state vs pr1's slow — layout states are deterministic per BUILD,
// not per SOURCE across recompiles, so s0/s7 carry band risk in both
// directions; the mechanism core is s1/s5/s6/s9 (band-free, stable ×3).
#ifndef FMHA_PRIME_HOIST
#define FMHA_PRIME_HOIST 1
#endif

// EXPERIMENT sfa1 - extend pr1-P2's slot-0 prime hoist to the SF_AHEAD forms, which
// `FMHA_PRIME_HOIST` excludes.  Pre-registration, hazard analysis and decision rule:
// results/summaries/2026-10-26-sfa1-prime-hoist-prereg.md.
//
// WHY THE EXCLUSION IS STRONGER THAN ITS REASON.  The hoist's gate is
// `if constexpr (!TT::SF_AHEAD)` with the comment "SF_AHEAD forms keep the in-body prime (its
// TMEM atom writes need the alloc first)".  That is true of ONE of the prime's five
// operations -- `tcg_pipe_sf`, which writes scale atoms into TMEM through `tlane` and so does
// need `tcg_alloc` -- and false of the two that actually go to DRAM: `tcg_issue_stage` (bulk
// tensor copies of group cLo's K/V into SHARED) and the three `tcg_stage_atoms` calls
// (cp.async of the atom blocks into SHARED).  Those need only the mbarrier init and the
// async-proxy fence above, exactly as the !SF_AHEAD arm does at the same point.  So sfa1
// splits the prime along its real dependency rather than excluding the whole form: the ISSUE
// moves to the hoist point, and `cp_async_commit()` plus everything after it stays in the body
// where `tlane` exists.
//
// VALUES.  0 = shipped pr1-P2 behaviour, SF_AHEAD forms keep the whole prime in the body and
// the two edited regions are `if constexpr`-dead for them, so the emitted binary is unchanged.
// 1 = SF_AHEAD forms issue group cLo's tile and its three atom stages in the prologue, above
// `FMHA_PROLOG_LDG`'s cold LDG batches and above `tcg_alloc`; on the split key that also puts
// the cold K/V issue above `fmha_pdl_scan_wait()`, which is where the fitted 8.9us floor's
// chain currently has it ("Q LDG ~0.8us -> sfq LDG ~0.5 -> [PDL wait] -> body's K-prime
// landing ~0.8").
//
// REACH.  `tcg_sf_ahead` needs D==256, pass != PASS_GROUP_MAX, !gsplit and ksplit==2, and
// `FMHA_TCG_QK_AHEAD` is 0, so exactly TWO launched keys change: `<256,PASS_FUSED,2,...>` and
// `<256,PASS_SPLIT,2,...>`.  Fixed suite s4/s6 (fused) and s5 (split); domain d2/d3/d6/d8 are
// the four head_dim=256 cases.  Every D=128 key compiles the identical binary, which makes
// those seven fixed cases a built-in control.
//
// BIT-EXACT BY CONSTRUCTION, and the accounting argument is the part to check: the body's
// comment says "One commit group, so the loop's first `wait_group 1` still has exactly one
// older group to prove... Nothing else is outstanding at this point".  Moving the issues above
// the commit keeps that true because nothing between the new issue point and the commit issues
// a cp.async -- the only `cp_async*`/`tcg_issue_stage`/`tcg_stage_atoms` lines in 2737..3716
// are the !SF_AHEAD hoist arm (constexpr-excluded here) and :3093-3643, all inside the
// `if constexpr (GSPLIT)` arm, and SF_AHEAD requires !gsplit.  The Q payload is staged by
// LDG+`sts128` and the sfq words go straight to TMEM by `tcg_st1`; neither is a cp.async.
// Both helpers are barrier-free (`tcg_stage_atoms` is `cp_async16` only), the atom slots
// (`L::oSfK`/`oSfV`) do not overlap the prologue's stores (`L::oQ`, TMEM), and the prime's
// read of atom slot 0 stays ordered before iteration 0's refill of it by the `__syncthreads()`
// that does not move.  K/V/scale tensors are caller-produced inputs, not gm outputs, so
// reading them above the PDL scan wait is safe -- `FMHA_PROLOG_LDG` already issues the Q LDGs
// there.
//
// MEASURED AND ADJUDICATED **not_improved** -- SHIPPED AT 0, DO NOT RE-NOMINATE WITHOUT A NEW
// MECHANISM ARGUMENT.  Nominated at e23adc4d24aecc616d4a8d98f058e4b192516e2a (tree e265a602);
// the owner's controller (owner-recovery-v1, measurement_attempt 1) ran all three suites and
// recorded `status: not_improved` in runtime/PROMOTION_STATUS.json:
//
//   suite          job              candidate    control      ratio     bootstrap 95%      decision
//   fixed          dv_130cd9169b6b   64.621209   64.950510   0.994930  [0.99355,0.99635]  improved
//   domain         dv_17238d881c00   26.932362   26.948927   0.999385  [0.99578,1.00318]  needs_repeat
//   reverse_fixed  dv_b2625b024a3d   64.895043   64.820380   1.001152  [0.99874,1.00264]  not_improved
//
// Every gate passed in every suite and both arms (`full_pass: true`, `failures: {}`), and the
// bit-exactness argument held: the whole fixed-suite gain is ONE case, **s5 at per-case ratio
// 0.9500 = -5.0%**, with the other nine at 0.9991-1.0018 in the forward draw. So the mechanism
// is real and it is where the pre-registration said it would be.
//
// WHY IT STILL LOST, AND IT IS THE ORDER.  In the REVERSE draw the candidate's s5 reads
// **40.944 us against the control's 40.960** -- ratio 0.99961, i.e. **the entire win vanished**
// -- while s1, whose binary is IDENTICAL between the arms, rolled to 161.712 against 159.776 and
// cost 1.212%.  Across all five candidate draws of s5 this record has (three of mine plus the
// owner's two) the values are 38.941, 40.147, 39.031, 38.912, 40.944: **the mechanism shifts s5
// toward a fast state and does not deterministically reach it**, while the control read
// 40.742/41.031/40.942/40.960/40.960 -- slow in all five.  That is precisely the peer's warning
// in FMHA_PRIME_HOIST's own comment, "layout states are deterministic per BUILD, not per SOURCE
// across recompiles, so s0/s7 carry band risk in both directions", applied to s5, and it is why
// `JOIN.json.selection_basis` requires BOTH execution orders.
//
// THE METHOD LESSON, which is worth more than the 0.5%: my three replicates were independent
// DRAWS but not independent ORDERS -- the v2 release always evaluates the candidate first and
// exposes no reverse flag -- so three same-order A/Bs agreed with each other and with the
// owner's forward job (+0.5529%, +0.2773%, +0.4170%, +0.510%) and all four were blind to the
// one axis that decided it.  **Replicating a measurement along the axis it already varies does
// not test the axis it does not.**  Where a mechanism's effect is a state shift rather than a
// uniform latency cut, same-order replication is evidence about the draw and not about the
// order, and only the owner's reverse suite can settle it.
//
// WHAT SURVIVES, and it is not nothing.  §239's discriminator is a measurement and stands: s5's
// once-per-CTA code owns 55.45% of its stall samples at 424 stall/Minst against its own loop's
// 28.80% at 118.5, where s6's owns 23.10% at 204.9 -- so a once-per-CTA change pays in
// proportion to how much stalling already lives in once-per-CTA code, and that is a screen any
// future prologue lever can run BEFORE spending a job.  §241 narrowed the s5-vs-d3 question to
// one quantity (the cold K/V working set, 33.5 MB against 2.1 MB) after eliminating the key, the
// launch geometry, the kernel duration and the tail/floor decomposition.  And §240's
// identification of F -- F_gm 7.56 us, F_split 9.23-10.20 us, together 65.0% of s3's certified
// forward -- is untouched by this verdict and is the larger prize.
//
// Kept at 0 rather than deleted for the same reason FMHA_CTA_PHASE_STAGGER is: the code is
// proven inert at 0 (`experiments/sfa1_pairing.py` C1 -- with the knob at 0 the preprocessed,
// comment-stripped file is LINE FOR LINE the incumbent's, 1689 vs 1689 live lines) and the
// negative result is more useful with its mechanism beside it than as a commit message alone.
// Pre-registration: results/summaries/2026-10-26-sfa1-prime-hoist-prereg.md.  Full account with
// all five draws, the outlier censuses and the untouched-control arithmetic: §§235-238, 242.
#ifndef FMHA_PRIME_HOIST_SFA
#define FMHA_PRIME_HOIST_SFA 0
#endif

// DIAGNOSTICS ONLY -- FMHA_PROBE_LEVEL.  NEVER SHIP NON-ZERO: at level >= 1 every
// `fmha_tcg_kernel` exits part way through its prologue, so its outputs are garbage BY
// CONSTRUCTION and it must only ever be built through `profile_target.py --extra-flags` for a
// Duration measurement.  At the default 0 every region below is `#if 0` and the emitted code is
// unchanged -- proven, not assumed, by `experiments/sfa1_pairing.py` C2.
//
// WHAT IT PRICES.  §240 identified the per-CTA fixed cost per role (F_gm 7.56 us, F_split
// 9.23-10.20 us; together 65.0% of s3's certified forward) and §243 split it with an empty-kernel
// probe: of s3 split's 15.78 us, 4.29 us is launch/CTA-ramp/teardown floor and ~6.16 us is the
// kernel's OWN once-per-CTA code, worth 1.77% of the fixed geomean from s3 alone if halved.
// §244 attributed the STALLS in that code to lines and closed the Q-chain response, leaving three
// candidates a stall sample cannot see because a sample is not a duration.  This ladder prices
// them by returning LATER at each rung, so consecutive rungs differ by exactly one component:
//
//   level  returns after                                     prices, against the level below
//     0    (never -- the shipped kernel)                     --
//     1    kernel entry, before anything                     launch + CTA ramp + teardown floor
//     2    the `tid == 0` mbar_init block + its              mbarrier init and the async-proxy
//          fence_async_shared()                              fence
//     3    tcg_alloc + __syncthreads() + tlane +             the TMEM allocation and the CTA-wide
//          tcg_relinquish(), then tcg_dealloc                barrier that waits on ONE warp's
//                                                            tcgen05.alloc
//
// WHY EVERY RUNG IS SAFE, which is the part that has to be right before a job is spent on it.
// At level >= 1 all three async issue sites are compiled out -- the PRIME_HOIST arm, the
// PRIME_HOIST_SFA arm, and PROLOG_LDG's two cold LDG batches (whose ARRAYS are still declared and
// zero-filled, so the store side below still compiles).  So no bulk tensor copy and no cp.async is
// ever issued, and an early return cannot leave a TMA writing into shared memory that has already
// been reassigned to another CTA -- the hang-or-corruption hazard §244(e) identified for any rung
// at or after `tcg_alloc`.  The only resource a rung acquires is TMEM, and level 3 releases it with
// the same `if (tid < 32) tcg_dealloc(tmem, TT::ALLOC)` the real epilogue uses, so repeated
// launches inside one profile run cannot exhaust the 512 columns per SM.
//
// WHAT THIS LADDER DOES *NOT* PRICE, stated so the gap is not read as a null: the epilogue, which
// sits after the group loop and so cannot be reached by an early return without running the loop;
// and the cold-load latencies, which are deliberately compiled out.  §244 already bounded the
// grid-dependency waits at F_split - F_gm = 1.7-2.6 us and closed the Q chain as a target, so the
// epilogue is the one component left unpriced after this ladder, and it needs a different probe
// (run the loop, skip the stores, keep the dealloc).
#ifndef FMHA_PROBE_LEVEL
#define FMHA_PROBE_LEVEL 0
#endif

// EXPERIMENT mb1 - which arm of the `anyMask` branch ptxas makes the fall-through, and
// therefore how far the loop's HOT path has to jump every group.
//
// WHY.  §131 found that the fused D=128 loop reaches its primary (unmasked) probability
// path by a **21-22 KB forward branch taken once per group**, and that the instruction-fetch
// stall on that one target line is what differs between the builds that put s1 in its fast
// and its slow state (no_inst 39 vs 101 samples, with the same mnemonic at two non-target
// sites carrying zero in both arms).  §131(d) blamed a ptxas clone of the rescale branch;
// dv_13a386fd8d02's census REFUTES that - making the rescale unconditional (`FMHA_RESCALE_
// LAYOUT=1`) leaves the function at exactly 4,592 SASS lines and moves FMUL by 1, so no
// branch was removed and nothing merged.  The two copies of the exp are the deliberate
// `MASKED` template split documented at `fmha_tcg.cuh:1778-1785`, which pH27 measured: with
// `if (anyMask)` written inside the loop ptxas *predicated* the test and every 32-key chunk
// block carried 64 FSEL + 64 ISETP + 32 UIADD3 dead for every interior group, ~640 issued
// instructions per group out of ~2000.  So the duplication is a MEASURED trade and is not to
// be undone.  What is not a trade is WHICH ARM FALLS THROUGH: the masked variant runs once
// per warp (one diagonal group per causal row-block, 1,024 of 61,952 iterations = 1.65%) and
// the unmasked one runs on the other 98.35%, yet the rare arm is the fall-through and the
// common arm is 21 KB away.
//
// VALUES.  0 = shipped.  1 = `__builtin_expect(anyMask, 0)` on every `anyMask` test in the
// sequential fused path.  2 = 1, plus the probability branch written in the order the
// hardware actually takes it (`if (!anyMask) { unmasked } else { masked }`), because a
// probability hint tells ptxas which arm is likelier while the SOURCE ORDER is what it
// starts from - and §131's layout 3 already showed that hinting a branch whose arms are
// already in the preferred order changes nothing at all.
//
// SCREEN BEFORE TIMING, and the screen is a LAYOUT metric rather than a line count: the
// distance from the loop's hot taken branch to its target, reported by
// `experiments/clone_census.py` from `cuobjdump -sass` addresses.  Line count must NOT move
// (both arms of the split stay, by pH27), registers must stay at §131(a)'s 247 with 0 stack
// and 0 spills, and the hot forward jump must shorten.  Bit-exact by construction: the same
// two instantiations are called with the same arguments on the same condition.
#ifndef FMHA_MASK_BRANCH
#define FMHA_MASK_BRANCH 0
#endif
#if FMHA_MASK_BRANCH == 0
#define FMHA_MASK_TEST(x) (x)
#else
#define FMHA_MASK_TEST(x) (__builtin_expect(!!(x), 0))
#endif

// DIAGNOSTIC, default off, and never part of a measured or promoted build.  The group
// split is the first form in this candidate whose waits are not all CTA-wide, so a
// protocol error shows up as a kernel that never returns - and job `bl1`
// (dv_7d10faf46bc2) spent its whole 1750 s deadline that way and came back with no
// stdout at all, because a gateway job killed for exceeding its deadline returns
// neither its log nor its OSS output.  With this on, every wait in the loop is bounded:
// the first one that does not complete names itself, its group and its parity on
// stdout, latches a flag that makes every later wait return at once so the kernel
// finishes in milliseconds instead of minutes, and leaves the numbers wrong - which is
// what a diagnostic build is for.  Flip it with a source edit inside the job's private
// arm copy, never in the shipped tree.
#ifndef FMHA_GSPLIT_WATCHDOG
#define FMHA_GSPLIT_WATCHDOG 0
#endif

// Everything the bulk-tensor staging needs that the cp.async staging does not: the
// two descriptors and this CTA's (batch, kv-head) coordinates inside them.  The
// descriptors describe the WHOLE operand, so one map per forward serves every CTA
// and the tile selection is the {d, key, kv_head, batch} coordinate.
struct TmaStage {
  const CUtensorMap* k;
  const CUtensorMap* v;
  uint64_t* mbar;
  // qk1: the V ring's own barriers, non-null-distinct only in the deferred-PV
  // QK64 form, whose K and V copies are armed and waited at different points of
  // the iteration and therefore cannot share one expect_tx accounting.  Every
  // other form points this at `mbar` and issues the combined stage exactly as
  // before.
  uint64_t* mbarV;
  int32_t b;
  int32_t kh;
};

// Reports whether the loaded image can issue tcgen05; read once by the host so a
// non-arch-specific fallback build can never launch a stub kernel.
__global__ void fmha_tcg_probe(int* out) {
#ifdef FMHA_TCG_ARCH
  *out = 1;
#else
  *out = 0;
#endif
}

// ------------------------------------------------------------------- smem plan
template <int D, int PASS, int KSPLIT = 1, bool GSPLIT = false>
struct TcgLayout {
  static constexpr int BM = 128;
  static constexpr int BN = 128;
  static constexpr int DCH = D / 128;  // 128-byte swizzle chunks along d
  // The group-maximum pass only needs QK^T, so it stages no V and holds no
  // probability tile; that is what makes it cheap enough to run as a separate
  // pass over the whole key range.
  static constexpr int NV = (PASS == PASS_GROUP_MAX) ? 0 : 1;
  // Ring depth.  The software-pipelined form (see TcgTmem::PIPE) issues group
  // g+1's QK in the middle of group g's iteration and defers group g's PV wait to
  // the start of group g+1, so a tile must survive two more iterations than the
  // sequential form needs, and the buffer being filled at iteration g must not be
  // the one PV(g-2) is still reading.
  //
  // Four stages and a two-group fill is what pH-era tuning settled on, and it
  // stays: the five-stage three-group variant tried for the bulk-tensor form
  // (306385d) measured worse than four and two (s7 +6.0% against +4.9%, d5 +6.1%
  // against +3.2%), and its extra 2*BN*D = 32 KB comes out of the L1/smem pool the
  // SM shares, 166.5 -> 199.5 KB of 228 KB.  This form runs cp.async staging in any
  // case, because it is the one form at a single CTA per SM.
  // Four warpgroups are sequential by construction: the pipelined loop exists to
  // give a CTA something to issue while it waits, and a 512-thread CTA already has
  // sixteen warps for that.  See FMHA_FUSED_KSPLIT_D128.
  static constexpr bool PIPE =
      !GSPLIT && (D == 128) && (PASS == PASS_FUSED) && (KSPLIT > 1) && (KSPLIT < 4) &&
      (FMHA_PIPE_D128 != 0);
  // The pipelined group-maximum pass needs group g+1's K resident in the middle of
  // iteration g, which takes a third stage; it stages no V and its D=128 footprint
  // stays at ~68 KB, so two CTAs per SM survive.  D=256 would need 133 KB and lose
  // the second CTA, so it stays sequential.
  //
  // THAT LAST SENTENCE IS THE REASON FMHA_PIPE_GM_D256 EXISTS: at D=256 there is no
  // second CTA to lose.  plan_geometry sets `resident = (2 if D == 128 else 1) * sms`
  // and bounds the group-maximum grid by it -- `gm = resident / base`, clamped by
  // `T/2` and only ever RAISED to `ns`, which is itself <= resident/base -- so
  // `base * gm_nsplit <= resident == 148` at D=256 for EVERY legal geometry.  The gm
  // launch is therefore at 1.00 CTA/SM or below already (s5's is exactly 148 CTAs,
  // confirmed on device by dv_ebfebbdf02d1's plan_info), and 133 KB of 228 KB costs
  // no residency.  The 133 KB figure is not a guess: §109's layout model, validated by
  // reproducing three numbers this file states independently (166.5, 199.5, 183 KB),
  // computes 134.0 KB for D=256 gm at NSTAGE=3.
#ifndef FMHA_PIPE_GM_D256
#define FMHA_PIPE_GM_D256 0
#endif
// Minimum groups per gm CTA before the pipelined group-maximum loop is taken instead of
// the sequential one.  Defined here, at file scope, because its single use is inside the
// kernel body and the preprocessor runs in order.  **The shipped 7 is a WARM-PROTOCOL
// number from pI19, on a pre-`agx1` tree** -- the full provenance problem, and why it is
// worth re-measuring under v2, is documented at the use site in the group-maximum pass.
#ifndef FMHA_PIPE_GM_MIN_GROUPS
#define FMHA_PIPE_GM_MIN_GROUPS 7
#endif
// EXPERIMENT ph1 — inter-CTA phase de-synchronization for the co-resident
// 2-CTA/SM D128 classes (see the use site in the kernel prologue and
// results/summaries/2026-10-25-ph1-cta-phase-stagger-prereg.md).
// 0 = off (shipping). 1 = stagger non-GM forms (fused/split sequential).
// 2 = stagger the group-max pass only. 3 = both. NS constants ≈ half the
// measured per-group period of each form at 2 CTAs/SM (split/fused ~2.6us
// per group per CTA on s9/s1; GM ~0.77us on s9 — sp1/gmc2 anchors).
//
// MEASURED AND REJECTED — DO NOT PROMOTE (dv_3af8779c3f5e, 4 rounds,
// bit-identity TRUE on all 15 arm-case pairs): non-GM stagger LOST where
// it should have won (s1 +2.33% = +4.1us ≈ 3× the 1.3us nominal delay —
// pure added serialization in the pair regime; s9's split stagger
// persistently +0.2..+2.0us above control), and GM stagger was an EXACT
// null (s9 ±0.01% — GM's 81% no-eligible is intra-chain dependency
// latency, not partner alignment). The quantified reading: co-resident
// CTAs are ALREADY de-phased by natural memory-timing jitter, so the
// sp1 stall pools are single-CTA dependency structure and forced phase
// offsets can only ADD serialization. Fourth instance of the stall-pool
// ≠ wall-recoverable lesson (after gmd3 depth, sk3 staging, T2-D
// pipelining) and the first to bound recoverable alignment at ~zero.
// Kept at 0 with this evidence so the class is never re-derived.
#ifndef FMHA_CTA_PHASE_STAGGER
#define FMHA_CTA_PHASE_STAGGER 0
#endif
#ifndef FMHA_PHASE_STAGGER_NS
#define FMHA_PHASE_STAGGER_NS 1300
#endif
#ifndef FMHA_PHASE_STAGGER_NS_GM
#define FMHA_PHASE_STAGGER_NS_GM 400
#endif
#ifndef FMHA_PHASE_STAGGER_MIN_SPAN
#define FMHA_PHASE_STAGGER_MIN_SPAN 16
#endif
  static constexpr bool PIPE_GM =
      (PASS == PASS_GROUP_MAX) && ((D == 128) || (FMHA_PIPE_GM_D256 != 0));
  // Group-split ring depths, PER WARPGROUP.  K needs three because QK(g+2) is issued in
  // the middle of the iteration for group g and waited at the top of the next one, which
  // is the full-iteration lookahead that hides ~564 cycles of pipe time: the slots hold
  // g (in flight), g+2 (proven, about to be issued) and g+4 (filling).  V needs only two
  // because its fill is placed AFTER the previous group's PV completion wait, and that
  // wait is exactly the event that frees V(g-2) - so the shallower ring costs only that
  // V's cp.async is issued late in the iteration, and it still has the whole of the next
  // iteration's first half to land.  Slots partition by warpgroup, so the two never
  // touch the same tile: slotK(w,j) = w*3 + j%3 and slotV(w,j) = w*2 + j%2.
  static constexpr int GSTAGE_K = 3;
  static constexpr int GSTAGE_V = 2;
  static constexpr int NSTAGE = GSPLIT ? (KSPLIT * GSTAGE_V) : (PIPE ? 4 : (PIPE_GM ? 3 : 2));
  // Issuing QK one group ahead means group g+1's K has to be *proven* landed while
  // group g's PV is still in flight, and `cp.async.wait_group 1` only proves what
  // was committed before the most recent group - so K runs two groups ahead and
  // needs a third slot.  V is still consumed in the iteration after the one that
  // staged it, so its ring stays at NSTAGE and the two depths differ.  For D=256
  // fused that is 32 KB more, 183 -> 216 KB of the 228 KB an SM has, which costs
  // nothing because this form already takes all 512 TMEM columns and already runs
  // one CTA per SM.  Every other form gets NSTAGE_K == NSTAGE and an unchanged
  // layout, byte for byte.
  static constexpr bool QK_AHEAD = tcg_qk_ahead(D, PASS);
  // qk6: the mid-pipelined D=256 fused form runs K two groups ahead exactly like
  // QK_AHEAD does (its mid-iteration QK issue reads the tile the PREVIOUS head
  // staged), so it reuses the three-slot K ring; V stays on the two-slot ring with
  // its fill after the PV retirement.  Derived from the same free predicates the
  // kernel and TcgTmem use, plus this layout's own SF_AHEAD, so the ring depth and
  // the loop body cannot disagree.
  static constexpr bool MIDPIPE =
      (FMHA_D256_MIDPIPE != 0) && (D == 256) && (PASS == PASS_FUSED) && (KSPLIT == 2) &&
      tcg_sf_ahead(D, PASS, KSPLIT, GSPLIT);
  static constexpr int NSTAGE_K =
      GSPLIT ? (KSPLIT * GSTAGE_K) : ((QK_AHEAD || MIDPIPE) ? 3 : NSTAGE);
  static constexpr int NST = NSTAGE * NV;  // V pipeline stages (V exists only if NV)
  // Atoms three groups ahead of the tiles they belong to, so the loop's TMEM scale write
  // can sit above its staging barrier and delete the second per-group barrier; see
  // FMHA_SF_AHEAD_D256.  The atom ring gets its own depth because it is no longer tied to
  // a tile's: slot(g) is filled at the iteration for group g-3 and read at the one for
  // g-1, so it needs three slots where the tiles need two.  Every other form keeps
  // NSTAGE_SF == NSTAGE_K and an unchanged layout, byte for byte.
  static constexpr bool SF_AHEAD = tcg_sf_ahead(D, PASS, KSPLIT, GSPLIT);
  static constexpr int NSTAGE_SF = SF_AHEAD ? 3 : NSTAGE_K;
  // V's atoms follow the same ring as K's where they are staged ahead of V's tile, and
  // V's own tile depth everywhere else.  Equal to NST for every form but SF_AHEAD's.
  static constexpr int NST_SFV = SF_AHEAD ? NSTAGE_SF : NST;
  static constexpr size_t oQ = 0;
  static constexpr size_t oK = oQ + (size_t)BM * D;
  // K is buffered in every pass: the group-maximum pass stages no V, but it
  // still pipelines K, so its K tile must not be sized by NST.
  static constexpr size_t oV = oK + (size_t)NSTAGE_K * BN * D;
  static constexpr size_t oP = oV + (size_t)NST * BN * D;
  // The group split always publishes its probability tiles in TMEM - it needs TWO of
  // them, one per warpgroup, and that is 64 of the 512 columns its map spends exactly -
  // so its shared-memory P tile is not merely unused but unbudgeted.  That is where the
  // 16 KB comes from that keeps a ten-tile ring inside the SM's 228 KB.
  static constexpr size_t oSfQ = oP + (size_t)(GSPLIT ? 0 : NV) * BM * BN;
  static constexpr size_t oSfK = oSfQ + (size_t)512 * DCH;
  // K scale atoms follow the K tile's depth, exactly like the K tile itself - except
  // where they are staged three groups ahead, which gives them a ring of their own.
  static constexpr size_t oSfV = oSfK + (size_t)NSTAGE_SF * 512 * DCH;
  // Cross-key-half exchange for the group maximum and the denominator, used when
  // more than one 128-thread warpgroup splits one 128-key group between them: one
  // float per row per warpgroup, i.e. KSPLIT * BM floats.  Two is the floor
  // because a one-warpgroup CTA still uses the trailing slack for alignment.
  static constexpr size_t oSb = oSfV + (size_t)NST_SFV * 512 * DCH;
  // The group split uses the same scratch for a different exchange: not a reduction
  // across warpgroups that own one group between them, but a ping-pong of the running
  // state between warpgroups that own ALTERNATING groups.  Eight row-blocks - hsMs[2],
  // one slot per warpgroup because the mstate chain itself orders the next write after
  // the other warpgroup's read; hsDen[4], TWO slots per warpgroup because the den read
  // is not ordered against the next write by that chain alone; and fin[2] for the
  // epilogue's broadcast of the final state to the warpgroup that does not own it.
  static constexpr int SB_HALVES = GSPLIT ? 8 : ((KSPLIT > 2) ? KSPLIT : 2);
  // The trailing 1024 bytes absorb the alignment rounding of the UMMA window
  // base, which shifts every offset below by up to 1023 bytes.
  static constexpr size_t bytes = oSb + (size_t)SB_HALVES * BM * sizeof(float) + 1024;
};

// TMEM column map.  Scores always live in [0,128).
//
// D=128 keeps the O accumulator in registers, so the PV partial can share the
// score tile's columns (the scores are fully consumed before the PV MMA is
// issued) and a 256-column allocation leaves two CTAs resident per SM.
//
// D=256 cannot do that: a 256-float accumulator is the entire register budget,
// and 128 score + 256 output + 32 scale columns cannot be packed into 256.  So
// O lives in TMEM at [128,384), the scale factors move to [384,416) and the CTA
// takes all 512 columns - one CTA per SM, but the QK^T then runs on the
// block-scaled tensor core.  That matters because SM103 has no native FP8
// warp-level MMA: ptxas lowers `mma.sync...m16n8k32.e4m3` into two HMMA.16816
// plus an E4M3->F16 unpack of every operand byte (job pH10 census: the Phase E
// hot block is 2161 instructions of which 576 are F2FP.F16.E4M3.UNPACK_B and
// 256 are HMMA.16816.F32), and the K fragment is re-unpacked per warp.
//
// The software-pipelined form keeps TWO score tiles so that group g+1's QK can be
// issued while group g's probabilities are still being read out of the other one,
// and double-buffered scale-factor slots so the atoms of the group being issued
// never overlap the ones an in-flight MMA is reading.  For D=128 with a key split
// that is 2*128 scores + 128 output + 8 SFA_Q + 16 SFB_K + 8 SFA_P + 16 SFB_V =
// 432 of the 512 columns the form already allocates, so it costs no occupancy.
// D=256 would need 2*128 + 256 + 48 = 560 and cannot have it.
template <int D, int PASS, int KSPLIT = 1, bool QK64 = false, bool GSPLIT = false>
struct TcgTmem {
  // A key-split CTA has two warpgroups per row, so neither can own the row's
  // accumulator in registers; that is what forces O into TMEM.  D=256 has no
  // choice either way - 256 floats is the whole register budget.  The
  // group-maximum pass produces no numerator at all, so it never needs O and
  // always fits in a 256-column allocation, i.e. two CTAs per SM.
  static constexpr bool NEEDS_O = (PASS != PASS_GROUP_MAX);
  // The 64-column score tile is what buys O its way back into TMEM at D=128
  // without spending the second CTA per SM; see FMHA_QK64_D128.  It is a template
  // parameter rather than a predicate because the dispatcher picks it on launch
  // geometry, so both variants are compiled.  Note this is a PARAMETER, not a
  // member: a kernel that wants it reads its own QK64, and `TT::QK64` does not
  // exist (agu9's first attempt used it, failed the sm_103a compile, and only
  // surfaced as TCG_AVAILABLE=0 because solution.py then fell back to the plain
  // 10.3 image, where every tcgen05 body is a stub).
  static constexpr bool O_IN_TMEM = NEEDS_O && ((D > 128) || (KSPLIT > 1) || QK64);
  // Must agree with TcgLayout::PIPE, or the kernel selects a loop the shared-memory
  // layout sized no ring for.
  static constexpr bool PIPE =
      !GSPLIT && (D == 128) && NEEDS_O && (PASS == PASS_FUSED) && (KSPLIT > 1) &&
      (KSPLIT < 4) && (FMHA_PIPE_D128 != 0);
  // The group-maximum pass is a pure QK loop with no probability pass to hide an
  // MMA behind, so its wait is fully exposed - and it is 20-40% of every
  // split-path shape.  It cannot have a second score tile (128+128+24 = 280 of the
  // 256 columns that keep it at two CTAs per SM), but it does not need one: pass
  // 1 is the tile's only reader, so QK(g+1) can be issued as soon as a barrier
  // proves every warp has finished reading S(g).
  // Must stay in lockstep with the identical definition in TcgLayout above, which is
  // where FMHA_PIPE_GM_D256 is documented.
  static constexpr bool PIPE_GM =
      (PASS == PASS_GROUP_MAX) && ((D == 128) || (FMHA_PIPE_GM_D256 != 0));
  // QK(g+1) is in flight while iteration g+1 writes group g+2's K scale bytes, so
  // the K and V scale slots have to alternate by group parity wherever an MMA can
  // still be reading the other slot.  D=256 fused has the columns for it: SFA_Q at
  // 384, SFB_K 392..408, SFA_P 408..416, SFB_V 416..432 of 512.
  static constexpr bool QK_AHEAD = tcg_qk_ahead(D, PASS);
  // Atoms staged three groups ahead (see FMHA_SF_AHEAD_D256): iteration g writes group
  // g+1's atoms into TMEM while QK(g) and PV(g) are still reading the slot for g, so both
  // operands need two alternating slots exactly as PIPE's do.  Must agree with
  // TcgLayout::SF_AHEAD, or the loop writes a parity slot the map did not allocate.
  static constexpr bool SF_AHEAD = tcg_sf_ahead(D, PASS, KSPLIT, GSPLIT);
  // The group split alternates its scale slots by WARPGROUP rather than by group parity,
  // because both warpgroups have an MMA in flight at the same time and they read
  // different slots.  One slot each is enough - not two, as PIPE needs - because both of
  // its writes are ordered after the read that could conflict: the K atoms for g+2 are
  // written after the wait for QK(g), and the V atoms for g after the wait for PV(g-1).
  // That ordering is what keeps the scale region at 48 columns and the whole map at
  // exactly 512; two slots per warpgroup per operand would need 80 and would push the
  // second probability tile off the end of the allocation.
  static constexpr bool SF_ALT = PIPE || PIPE_GM || QK_AHEAD || GSPLIT || SF_AHEAD;
  static constexpr int S = 0;
  // Second score tile: PIPE's is the other half of a double-buffered pair that the two
  // warpgroups share, the group split's is warpgroup 1's own tile, single-buffered
  // WITHIN its warpgroup - QK(g+2) is issued only after a 128-thread barrier proves
  // every one of its four warps retired S(g)'s tcgen05.ld.
  static constexpr int S1 = (PIPE || GSPLIT) ? 128 : 0;
  // The 64-column score form puts its accumulator at 96 instead of 128 so that the
  // probability tile has a 32-column window above it: S=[0,64), SF=[64,96),
  // O=[96,224), P=[224,256) closes the 256-column allocation exactly, where the old
  // O=[128,256) left a 24-column gap and no window at all.  `ba2`
  // (`dv_acb64b7475e6`, `harness/bench/tmem_a_operand.cu` configs 10/11/12) proved
  // on this part that a `tcgen05.mma` accumulator at a **32-aligned but not
  // 128-aligned** column base is legal and bit-exact - bad=0 of 16384, worst rel
  // 0.000e+00, both with E8M0 scales and with unit scales, and again at a second
  // unrelated base (D at 32, A at 192) - which is what made this move available:
  // every accumulator this candidate had ever placed sat at 0, 128 or 256, and
  // nothing in the workspace had tested the alignment rule between them.
  static constexpr int O = (PIPE || GSPLIT) ? 256 : (O_IN_TMEM ? (QK64 ? 96 : 128) : 0);
  // D=256 puts O at [128,384) and the scale factors at 384; D=128 with a key
  // split only needs [128,256) for O, so its scale factors start at 256 - or at
  // 384 when the pipelined form also occupies [128,256) with the second score
  // tile.  All of them round up to a 512-column allocation, i.e. one CTA per SM.
  // With a 64-column score tile the scale factors move down to 64 and O takes
  // [128,256): 64 + 32 of scale columns + 128 of accumulator = 208 of 256, so the
  // allocation stays at 256 and two CTAs still share an SM.
  static constexpr int SFA_Q =
      (PIPE || GSPLIT) ? 384 : (QK64 ? 64 : (O_IN_TMEM ? ((D > 128) ? 384 : 256) : 128));
  static constexpr int SFB_K = SFA_Q + 8;
  static constexpr int SFA_P = SF_ALT ? (SFB_K + 16) : (SFA_Q + 16);
  static constexpr int SFB_V = SF_ALT ? (SFA_P + 8) : (SFA_Q + 24);
  // gmq1: the group-maximum pass in the 64-column form needs only S [0,64) plus
  // the SF region at [64,112) — a 128-column allocation, i.e. THREE CTAs per SM
  // (3x128 = 384 <= 512).  This is the only tcgen05 form below 256 columns; the
  // register side (REG <= 170 at 3x128 threads) is census-gated by the gmq1
  // screen.  Every other form keeps its previous allocation exactly.
  static constexpr int ALLOC = (QK64 && !NEEDS_O)
                                   ? 128
                                   : (((O_IN_TMEM && !QK64) || PIPE || GSPLIT) ? 512 : 256);
  // Conservative UPPER bound of the scale-factor region, used only as a lower bound
  // for P_TILE.  The region's exact width is 32 columns when nothing alternates
  // (SFA_Q 8, SFB_K 8, SFA_P 8, SFB_V 8) and 48 when SF_ALT doubles SFB_K and SFB_V
  // for the two parity slots; SFB_V + 16 over-reserves 8 in both cases.  That is
  // harmless here - it can only move P_TILE up - but it is why O can sit at 96 in the
  // 64-column score form even though this bound reads 104: the scale region really
  // ends at 96.  harness/analysis/tmem_map_check.py checks overlaps against the exact
  // width and P_TILE against this bound, and reports both.
  static constexpr int SF_END = SFB_V + 16;
  static constexpr int OCOLS_TM = (D > 128) ? 256 : 128;
  // The probability tile as the PV MMA's A operand: 32 columns for a 128-key E4M3
  // tile (`ba1`: 8 columns per 32-key k-step), placed above BOTH the scale-factor
  // region and the accumulator, and aligned to 32 so the `.x8` stores and the
  // k-step stride are naturally aligned.  Resolved per form, because the map above
  // is not one map:
  //
  //   form                          S        O          SF_END  P_TILE  ALLOC  fits
  //   D=128 fused/split KSPLIT=1   [0,128)  registers   168/176   192    256   yes
  //   D=128 fused PIPE  KSPLIT=2   [0,256)  [256,384)      432    448    512   yes
  //   D=256 fused/split KSPLIT=2   [0,128)  [128,384)      424    448    512   yes
  //   D=128 QK64 (64-col scores)   [0,64)    [96,224)   104/112   224    256   yes
  //   D=128 fused KSPLIT=4 (expt)  [0,128)  [128,256)      296    320    512   yes
  //   group-maximum pass           [0,128)  none        168/176   192    256   n/a
  //
  // Taking the maximum of the two lower bounds rather than only SF_END is what lets
  // the 64-column score form join the other four: its scale region ends at 104 but
  // its accumulator now ends at 224, and 224 is already 32-aligned.  The overlap
  // tests below are then redundant by construction and are kept as a guard, so a
  // future map change cannot silently allocate two operands to the same columns.
  static constexpr int P_LO = SF_END > (O_IN_TMEM ? O + OCOLS_TM : 0)
                                  ? SF_END
                                  : (O_IN_TMEM ? O + OCOLS_TM : 0);
  static constexpr int P_TILE = (P_LO + 31) & ~31;
  // The group split's second probability tile, one per warpgroup, immediately above the
  // first: 32 columns for a 128-key E4M3 tile at `ba1`'s 8 columns per k-step, so the
  // pair occupies [448,512) and closes the allocation exactly.  Meaningful only when
  // P_IN_TMEM is true below, which for this form requires the room for both.
  static constexpr int P_TILE1 = P_TILE + 32;
  static constexpr bool P_IN_TMEM =
      (FMHA_P_IN_TMEM != 0) && NEEDS_O && (P_TILE + (GSPLIT ? 64 : 32) <= ALLOC) &&
      !(O_IN_TMEM && (P_TILE < O + OCOLS_TM) && (P_TILE + 32 > O)) &&
      !((PIPE || GSPLIT) && (P_TILE < S1 + 128) && (P_TILE + 32 > S1));
  // Scale-factor slot for a group of the given parity: two alternating slots when
  // an MMA can be in flight while the other slot is written, one when not.
  static __device__ __forceinline__ uint32_t sfk_slot(uint32_t base, int par) {
    return base + (SF_ALT ? (uint32_t)(8 * par) : 0u);
  }
};
constexpr int TC_S = 0;

// Column base for an MMA the SEQUENTIAL loop issues for key group `grp`.
//
// Namespace scope rather than a TcgTmem member on purpose: `TT` is a dependent alias
// inside the kernel, so a member template would have to be spelled
// `TT::template sf_seq<...>` at every call site, and the spelling that reads naturally
// there does not parse.  Both flags are template parameters rather than runtime flags for
// a sharper reason - a form that does not stage its atoms ahead gets literally `base`
// back, whatever its own SF_ALT happens to be.  That matters because PIPE_GM sets SF_ALT
// for the pipelined loop and then falls through to the sequential one for short spans,
// where the reader is the unshifted base, so reading SF_ALT alone would move it.
//
// At AHEAD=true the write for group grp+1 happens while the slot for grp is the one this
// iteration's MMAs read, so the two alternate by group parity and ALT has to be true for
// the second slot to have been allocated; the kernel static_asserts exactly that.
template <bool AHEAD, bool ALT>
__device__ __forceinline__ uint32_t tcg_sf_seq(uint32_t base, int grp) {
  if constexpr (AHEAD && ALT) {
    return base + (uint32_t)(8 * (grp & 1));
  } else {
    return base;
  }
}

// `minBlocksPerMultiprocessor` for __launch_bounds__, i.e. how many resident CTAs
// ptxas has to leave register room for.  A form that takes all 512 TMEM columns
// can only ever have one CTA per SM, so asking for two would buy nothing and cost
// registers; a form that fits in 256 columns can have two and asks for two.  That
// reproduces the previous `KSPLIT == 1 ? 2 : 1` exactly for every form the
// dispatcher launches - KSPLIT=1 forms all allocate 256 columns and the KSPLIT=2
// ones all allocate 512 - and extends it to the one new form that splits keys
// without needing the TMEM accumulator, the group-maximum pass, whose resident
// count is what FMHA_GM_MIN_BLOCKS exists to measure.
template <int D, int PASS, int KSPLIT, bool QK64 = false, bool GSPLIT = false>
struct TcgMinBlocks {
  static constexpr bool GM_SPLIT = (PASS == PASS_GROUP_MAX) && (KSPLIT > 1);
  // gmq1: a 128-column allocation fits three per SM, so the GM-QK64 form asks
  // ptxas for three (it must then hold REG <= 170 without spills — the screen's
  // census gate).  256-column forms keep two and 512-column forms one, exactly
  // as before.
  static constexpr int value =
      GM_SPLIT ? FMHA_GM_MIN_BLOCKS
               : ((TcgTmem<D, PASS, KSPLIT, QK64, GSPLIT>::ALLOC <= 128) ? 3
               : (TcgTmem<D, PASS, KSPLIT, QK64, GSPLIT>::ALLOC <= 256) ? 2 : 1);
};

// One pipeline stage's worth of E8M0 scale atoms, and nothing else: K's `512*DCH` bytes
// and, where the pass has a V at all, V's.  Split out of `tcg_issue_stage` verbatim -
// same source strides, same destination swizzle, same `cp_async16` predicate, same
// thread-to-16-byte mapping - so that the atoms can be given a ring of their own without
// a second copy of the arithmetic drifting away from the first.
//
// `sfkstage`/`sfvstage` are RING SLOT INDICES, not group indices, and they are separate
// parameters because the caller that stages atoms ahead of the tiles they belong to (see
// FMHA_SF_AHEAD_D256) has a three-deep atom ring under a two-deep tile ring.  Every other
// caller passes the tile slot for both and gets exactly the code it had before.
//
// Atoms only ever ride `cp.async`, never a bulk copy, so this function owns no mbarrier
// traffic: the caller's `cp_async_commit()` / `cp_async_wait<N>()` pair is the whole of
// its completion mechanism.
template <int D, int PASS, int THR, bool ATOM_K = true, bool ATOM_V = true>
__device__ __forceinline__ void tcg_stage_atoms(const Params& p, int tid, int sfkstage,
                                                int sfvstage, int grp, uint32_t sSfK_u,
                                                uint32_t sSfV_u, const uint8_t* sfk_slice,
                                                int64_t v_atom_base) {
  constexpr int DCH = D / 128;
  constexpr bool WANT_V_HERE = ATOM_V && (PASS != PASS_GROUP_MAX);
  if constexpr (ATOM_K) {
    const uint8_t* atom = sfk_slice + (int64_t)grp * ((D / 32) >> 2) * 512;
    const uint32_t dst = sSfK_u + (uint32_t)(sfkstage * 512 * DCH);
#pragma unroll
    for (int c = tid; c < 32 * DCH; c += THR)
      cp_async16(dst + (uint32_t)((c >> 5) * 512 + (c & 31) * 16), atom + (int64_t)c * 16, true);
  }
  if constexpr (WANT_V_HERE) {
    const uint8_t* atom = p.sfv + ((int64_t)v_atom_base + grp) * 512;
    const uint32_t dst = sSfV_u + (uint32_t)(sfvstage * 512 * DCH);
#pragma unroll
    for (int c = tid; c < 32 * DCH; c += THR)
      cp_async16(dst + (uint32_t)((c >> 5) * 512 + (c & 31) * 16),
                 atom + (int64_t)(c >> 5) * p.colatoms * 512 + (int64_t)(c & 31) * 16, true);
  }
}

// One pipeline stage of K, V and both scale-factor atom groups.
//
// TMA=true stages K and V with one `cp.async.bulk.tensor.4d` per 128-byte swizzle
// chunk, issued by thread 0 and tracked by `tma.mbar[stage]`, instead of sixteen
// sixteen-byte cp.async per thread plus their address arithmetic and predicates.
// The bytes are the same (harness/bench/tma_layout.cu, job pI25: SWIZZLE_128B
// reproduces fp8_tile_offset exactly), so no UMMA descriptor changes, and keys
// past Skv zero-fill the same way the `valid=false` predicate does.  The
// scale-factor atoms stay on cp.async, so both completion mechanisms coexist in
// one iteration: the mbarrier proves K/V landed, cp.async.wait_group the atoms.
// ATOMS=false stages the tiles and leaves the scale atoms to a separate
// `tcg_stage_atoms` call, which is what the atoms-three-groups-ahead form needs: its
// atom ring is deeper than its tile ring, so the two cannot share one slot index.  It is
// illegal on the bulk-atom path, where the atoms ride the tile's own mbarrier and an
// expect_tx that counted them would wait forever on bytes that were never copied.
//
// STAGE_K / STAGE_V let a caller fill one operand's ring without the other's, which
// is what the QK-lookahead form needs: its K ring is three deep and its V ring two,
// so group g+2's K and group g+1's V are staged in the same iteration but land in
// rings of different depth.  Both default to true, and kstage == vstage at every
// call site that does not use the lookahead, so the other forms are unchanged.
template <int D, int PASS, int THR, bool TMA, bool STAGE_K = true, bool STAGE_V = true,
          bool ATOMS = true, bool SPLITBAR = false>
__device__ __forceinline__ void tcg_issue_stage(const Params& p, int tid, int kstage, int vstage,
                                                int grp, uint32_t sK_u, uint32_t sV_u,
                                                uint32_t sSfK_u, uint32_t sSfV_u,
                                                const uint8_t* k_base, const uint8_t* v_base,
                                                const uint8_t* sfk_slice, int64_t v_atom_base,
                                                const TmaStage& tma) {
  using L = TcgLayout<D, PASS>;
  constexpr int BN = 128;
  constexpr int CH = D / 16;  // 16-byte pieces per key row
  constexpr int DCH = D / 128;
  constexpr bool WANT_V_HERE = STAGE_V && (PASS != PASS_GROUP_MAX);
  // One barrier and one expect_tx cover the whole tile, so the bulk-copy path always
  // stages both operands together; only the cp.async path splits them.
  // qk1: SPLITBAR lifts this for the deferred-PV QK64 form, which arms K and V on
  // their own barriers because its loop issues them at different points - K ahead of
  // the loop top's barrier, V right after the previous PV retires and frees the slot.
  // Each barrier's expect_tx then counts exactly the payload this call issues
  // against it, so every wait still proves its own operand's tile AND its atoms.
  static_assert(!TMA || SPLITBAR || (STAGE_K && STAGE_V),
                "the bulk-tensor path stages K and V together unless SPLITBAR");
  const int keyBase = grp << 7;
  const uint32_t kdst = sK_u + (uint32_t)(kstage * BN * D);
  const uint32_t vdst = sV_u + (uint32_t)(vstage * BN * D);
  // Whether this stage's atoms ride the bulk path, i.e. whether this call issues
  // any cp.async at all.  D=128 only: agu2 measured D=256 at +1.68% on s6 and
  // +0.25% on s5 against -5.85%/-6.28% on s7/s1.  See FMHA_TCG_SF_BULK.
  constexpr bool SF_BULK = TMA && (D == 128) && (FMHA_TCG_SF_BULK != 0);
  static_assert(ATOMS || !SF_BULK, "ATOMS=false needs the atoms on their own cp.async stream");
  if constexpr (TMA) {
    (void)k_base;
    (void)v_base;
    // qk1: V's completion barrier when the caller arms the two operands apart;
    // every other form keeps the single combined barrier byte for byte.
    uint64_t* const vbar = SPLITBAR ? (tma.mbarV + vstage) : (tma.mbar + kstage);
    if (tid == 0) {
      constexpr uint32_t kChunk = (uint32_t)(BN * 128);
      // One barrier and one expect_tx cover everything this stage moves, so the
      // loop's single mbar_wait proves the tiles AND the atoms.  Counted from what
      // this call actually issues rather than from the pass: each operand
      // contributes one 128-byte-chunk tile payload plus, on the bulk path, its
      // 512-byte-per-chunk atom block.  For every form this candidate launches that
      // is the same number the pass-derived expression gave (two operands, or one
      // for the group-maximum pass, which stages no V).
      const uint32_t perOperand = kChunk + (SF_BULK ? 512u : 0u);
      if constexpr (SPLITBAR) {
        if constexpr (STAGE_K) mbar_expect_tx(tma.mbar + kstage, perOperand * (uint32_t)DCH);
        if constexpr (WANT_V_HERE) mbar_expect_tx(vbar, perOperand * (uint32_t)DCH);
      } else {
        const uint32_t operands = (STAGE_K ? 1u : 0u) + (WANT_V_HERE ? 1u : 0u);
        mbar_expect_tx(tma.mbar + kstage, perOperand * (uint32_t)DCH * operands);
      }
      // Coordinate order follows the descriptor's dimension order exactly:
      // {head-dim byte, kv head, key, batch}.  Getting y and z the wrong way round
      // is silent and catastrophic - the key index lands in a dimension whose
      // extent is KH, so every group past the first goes out of bounds and
      // zero-fills (job dv_9b609ab2a62f: 6 of 10 cases wrong, d7's V read back as
      // all zeros, and only the 'zero' and 'tiny' distributions still passed).
#pragma unroll
      for (int ch = 0; ch < DCH; ++ch) {
        if constexpr (!SPLITBAR || STAGE_K) {
          tma_load_4d(tma.k, kdst + (uint32_t)ch * kChunk, ch * 128, tma.kh, keyBase, tma.b,
                      tma.mbar + kstage);
        }
        if (WANT_V_HERE) {
          tma_load_4d(tma.v, vdst + (uint32_t)ch * kChunk, ch * 128, tma.kh, keyBase, tma.b,
                      vbar);
        }
      }
      if constexpr (SF_BULK) {
        // K's atom block is one contiguous 512*DCH run on both sides: the
        // destination offset `(c >> 5) * 512 + (c & 31) * 16` is just `c * 16` for
        // `c < 32*DCH`, and the source is `atom + c*16`.  V's destination is the
        // same run but its source strides by `colatoms * 512` between 128-byte
        // chunks, so it is DCH copies of 512 bytes rather than one.
        if constexpr (STAGE_K) {
          bulk_copy_1d(sSfK_u + (uint32_t)(kstage * 512 * L::DCH),
                       sfk_slice + (int64_t)grp * ((D / 32) >> 2) * 512, (uint32_t)(512 * L::DCH),
                       tma.mbar + kstage);
        }
        if constexpr (WANT_V_HERE) {
          const uint8_t* atomv = p.sfv + ((int64_t)v_atom_base + grp) * 512;
          const uint32_t vsfdst = sSfV_u + (uint32_t)(vstage * 512 * L::DCH);
#pragma unroll
          for (int g = 0; g < L::DCH; ++g) {
            bulk_copy_1d(vsfdst + (uint32_t)(g * 512), atomv + (int64_t)g * p.colatoms * 512, 512u,
                         vbar);
          }
        }
      }
    }
  } else {
#pragma unroll 2
    for (int it = 0; it < (BN * CH) / THR; ++it) {
      const int c = it * THR + tid;
      const int j = c / CH;
      const int ci = c - j * CH;
      const int jg = keyBase + j;
      const bool ok = (jg < p.Skv);
      const int jc = ok ? jg : (p.Skv - 1);
      // 128-byte swizzle chunks: a K-major UMMA descriptor requires rows exactly
      // 128 bytes apart, so d is split into chunks instead of one wide row.
      const uint32_t soff =
          (uint32_t)((ci >> 3) * (BN * 128) + fp8_tile_offset(j, (ci & 7) * 16, 128));
      if constexpr (STAGE_K) cp_async16(kdst + soff, k_base + (int64_t)jc * p.k_s + ci * 16, ok);
      if constexpr (WANT_V_HERE) {
        cp_async16(vdst + soff, v_base + (int64_t)jc * p.v_s + ci * 16, ok);
      }
    }
  }
  if constexpr (!SF_BULK && ATOMS) {
    tcg_stage_atoms<D, PASS, THR, STAGE_K, STAGE_V>(p, tid, kstage, vstage, grp, sSfK_u, sSfV_u,
                                                    sfk_slice, v_atom_base);
  }
}

// KSPLIT is the number of 128-thread warpgroups that cooperate on one 128-key
// group.  D=128 runs two CTAs per SM and already has eight warps resident, so it
// keeps KSPLIT=1.  D=256 needs all 512 TMEM columns for a single CTA, which left
// Phase F with four warps per SM issuing ~0.65 instructions/cycle against the
// ~1.7 that the eight-warp kernels reach; splitting the key axis in two gives the
// same CTA eight warps and half the softmax work each, at the cost of one extra
// barrier per group to exchange the two half-maxima.
// Pass 1 for one warpgroup's key half: the raw group maximum, which the caller
// scales once.  Both the mask and the reduction strategy are template parameters
// for the same reason as in tcg_probabilities - written as runtime `if`s, ptxas
// merged the masked and unmasked bodies into a single 715-instruction block that
// issued 256 FSEL and 254 ISETP on every group (job pH27 census).
template <int KHALF, bool MASKED, bool RED_MAX>
__device__ __forceinline__ float tcg_group_max(uint32_t tlane, int khalf, int keyLo, int lo,
                                               int hi) {
  float bmRaw = -INFINITY;
  if (RED_MAX) {
    // The reduction happens inside the TMEM load, so the FMNMX tree disappears.
    // `.red` is .sync.aligned, so it must never sit under a per-lane test (that
    // is what hung job pF8); the caller's warp vote guarantees that.
#pragma unroll
    for (int c = 0; c < KHALF / 32; ++c) {
      bmRaw = fmaxf(bmRaw, tcg_ld32_redmax(tlane + TC_S + khalf * KHALF + c * 32));
    }
    return bmRaw;
  }
#pragma unroll
  for (int c = 0; c < KHALF / 32; ++c) {
    float v[32];
    tcg_ld32(tlane + TC_S + khalf * KHALF + c * 32, v);
    if (MASKED) {
      const unsigned span = (unsigned)(hi - lo);
      unsigned d = (unsigned)(keyLo - lo) + (unsigned)(c * 32);
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        if (d > span) v[j] = -INFINITY;
        ++d;
      }
    }
    // Balanced tree, in place because these scores are dead once the maximum is
    // known.  A linear `bmRaw = fmaxf(bmRaw, v[j])` chain is KHALF-1 *dependent*
    // FMNMX - about 4 cycles each, so ~500 cycles of pure serial latency per key
    // group at KHALF=128 - and nvcc does not reassociate it.  Five levels of
    // pairwise halving plus one combine per chunk is depth 5 + KHALF/32.  max is
    // exact and order independent, so the value is unchanged.
#pragma unroll
    for (int n = 16; n >= 1; n >>= 1) {
#pragma unroll
      for (int i = 0; i < n; ++i) v[i] = fmaxf(v[i], v[i + n]);
    }
    bmRaw = fmaxf(bmRaw, v[0]);
  }
  return bmRaw;
}

// One 32-key chunk of the probability pass: the contract's E4M3 QDQ into the
// probability tile plus the unrounded denominator partial.  The scores arrive
// already in registers and already masked, so the same body serves both the pass
// that reads them out of TMEM here and the pass that read them once earlier (see
// tcg_scores).  `den2` is accumulated by reference so that the summation order -
// and therefore the rounding of the denominator - is identical in both.
//
// `PTMEM` chooses where the eight quantized words go, and it changes nothing about
// the values: the same `words[8]`, in the same key order, from the same packed
// exponential and the same `cvt_e4m3x2`.  Shared memory gets two `sts128` through
// `fp8_tile_offset`'s 128-byte swizzle, because that is what the UMMA shared-memory
// descriptor reads; TMEM gets one `tcgen05.st.32x32b.x8` at column `colBase/4` of
// this thread's lane, because that is what the A-from-TMEM form of the same MMA
// reads (`ba1` configs 2 and 9: bit-exact, 8 columns per k-step, natural byte and
// k-step order).  `pDst` is the swizzled shared-memory base in the first case and
// `tlane + TcgTmem::P_TILE` in the second, and `m` is only meaningful to the
// swizzle.
template <int BIAS_LOG2 = -1, int NACC = 1, bool PTMEM = false>
__device__ __forceinline__ void tcg_prob_chunk(const float* v, uint32_t pDst, int m, int colBase,
                                               uint64_t scale2, uint64_t nsafe2, uint64_t bias2,
                                               uint64_t& den2) {
  unsigned words[8];
  // NACC independent accumulator chains instead of one sixteen-link chain; see
  // FMHA_DEN_PAR.  At NACC=1 this is not a variant of the shipped code, it IS the
  // shipped code: no partials are declared and `den2` is accumulated in place.
  static_assert(NACC == 1 || NACC == 2 || NACC == 4, "NACC has to divide the eight w8 steps");
  uint64_t acc[NACC];
  if constexpr (NACC > 1) {
#pragma unroll
    for (int q = 0; q < NACC; ++q) acc[q] = f32x2_pack(0.0f, 0.0f);
  }
  // One 32-bit word per four keys: two hardware E4M3 pair conversions and one
  // mov.b32 of the 16-bit pair, instead of zeroing the word and OR-ing two
  // shifted halves into it (pH10 counted 32 IMAD.SHL.U32 and 8 PRMT per chunk
  // for that).  The key order, and so the denominator's summation order, is
  // unchanged.
#pragma unroll
  for (int w8 = 0; w8 < 8; ++w8) {
    unsigned short halves[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int j = 2 * w8 + s;
      // mul.rn.f32x2 and add.rn.f32x2 round exactly like the scalar forms, so
      // the scores, the shifted exponents and the probabilities are unchanged
      // bit for bit.  The exponential itself is packed too: expf2 runs the same
      // eight-step sequence CUDA's expf lowers to, with the five steps that
      // have a packed form done for both keys at once (see fmha_tcg_ptx.cuh,
      // verified exhaustively over all 2^32 patterns by
      // harness/bench/expf_packed_check.cu).
      const uint64_t arg =
          f32x2_add(f32x2_mul(f32x2_pack(v[2 * j], v[2 * j + 1]), scale2), nsafe2);
      // BIAS_LOG2 < 0 means the bias is applied by the caller's packed multiply;
      // 0 means there is none (D=256, where it is exactly 1.0f); L > 0 means 2^L is
      // already inside expf's exponent field.  See FMHA_FOLD_BIAS16.
      uint64_t pr = expf2_bias<(BIAS_LOG2 > 0 ? BIAS_LOG2 : 0)>(arg);
      if constexpr (BIAS_LOG2 < 0) pr = f32x2_mul(pr, bias2);
      // Which partial a probability lands in is fixed by w8, so the key order -
      // and every quantized probability - is untouched by the split.
      if constexpr (NACC > 1) {
        uint64_t& part = acc[(w8 * NACC) / 8];
        part = f32x2_add(part, pr);
      } else {
        den2 = f32x2_add(den2, pr);
      }
      const float2 pv = f32x2_unpack(pr);
      halves[s] = cvt_e4m3x2(pv.x, pv.y);
    }
    asm("mov.b32 %0, {%1,%2};" : "=r"(words[w8]) : "h"(halves[0]), "h"(halves[1]));
  }
  if constexpr (NACC == 4) {
    // Depth 2 instead of the serial chain's 15, for three packed adds.
    den2 = f32x2_add(den2, f32x2_add(f32x2_add(acc[0], acc[1]), f32x2_add(acc[2], acc[3])));
  } else if constexpr (NACC == 2) {
    // Depth 8 for ONE packed add and one extra register pair: agu4 says the
    // two-warpgroup forms want less chain shortening than the one-warpgroup ones,
    // not none.
    den2 = f32x2_add(den2, f32x2_add(acc[0], acc[1]));
  }
  if constexpr (PTMEM) {
    tcg_st8(pDst + (uint32_t)(colBase >> 2), words);
  } else {
    const uint4 lo4 = make_uint4(words[0], words[1], words[2], words[3]);
    const uint4 hi4 = make_uint4(words[4], words[5], words[6], words[7]);
    sts128(pDst + (uint32_t)fp8_tile_offset(m, colBase, 128), lo4);
    sts128(pDst + (uint32_t)fp8_tile_offset(m, colBase + 16, 128), hi4);
  }
}

// Pass 2 for one warpgroup's key half, reading the score tile out of TMEM.
// MASKED is a template parameter so that the interior-group copy contains no
// mask code at all.  With `if (anyMask)` written inside the loop, ptxas
// predicated the test instead of branching: the pH27 census shows each 32-key
// chunk block carrying 64 FSEL + 64 ISETP + 32 UIADD3 that are dead for every
// interior group, ~640 issued instructions per group out of ~2000.
// __noinline__ is not an option here - the 32 scores live in registers and
// cannot cross a call boundary.
template <int KHALF, bool MASKED, int BIAS_LOG2 = -1, int NACC = 1, bool PTMEM = false>
__device__ __forceinline__ uint64_t tcg_probabilities(uint32_t tlane, uint32_t pDst, int m,
                                                      int khalf, int keyLo, int lo, int hi,
                                                      uint64_t scale2, uint64_t nsafe2,
                                                      uint64_t bias2) {
  uint64_t den2 = f32x2_pack(0.0f, 0.0f);
#pragma unroll
  for (int c = 0; c < KHALF / 32; ++c) {
    float v[32];
    tcg_ld32(tlane + TC_S + khalf * KHALF + c * 32, v);
    if (MASKED) {
      // Masking the loaded score is equivalent to masking after the softmax
      // scale: the scale is positive (head_dim**-0.5), so -INFINITY*scale stays
      // -INFINITY and expf(-INFINITY - safe) is still exactly +0.
      //
      // One unsigned span test instead of `key < lo || key > hi`: the window is
      // contiguous and hi >= lo always holds (causal puts hi at the row's centre
      // and lo at most W below it), so key-lo > hi-lo is exactly the complement,
      // and starting the counter at keyLo-lo makes keys below lo wrap to a huge
      // unsigned value, which the same test also catches.  pH27's census counted
      // 2 ISETP + 2 FSEL + 1 UIADD3 per masked element; this is 1 ISETP + 1 FSEL
      // + 1 IADD3.
      const unsigned span = (unsigned)(hi - lo);
      unsigned d = (unsigned)(keyLo - lo) + (unsigned)(c * 32);
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        if (d > span) v[j] = -INFINITY;
        ++d;
      }
    }
    tcg_prob_chunk<BIAS_LOG2, NACC, PTMEM>(v, pDst, m, khalf * KHALF + c * 32, scale2, nsafe2, bias2,
                                    den2);
  }
  return den2;
}

// The same 32-key chunks fed from scores that are already in registers, i.e.
// from the single TMEM read that also produced the group maximum.
template <int KHALF, int BIAS_LOG2 = -1, int NACC = 1, bool PTMEM = false>
__device__ __forceinline__ uint64_t tcg_probabilities_pre(const float* scores, uint32_t pDst,
                                                          int m, int khalf, uint64_t scale2,
                                                          uint64_t nsafe2, uint64_t bias2) {
  uint64_t den2 = f32x2_pack(0.0f, 0.0f);
#pragma unroll
  for (int c = 0; c < KHALF / 32; ++c) {
    tcg_prob_chunk<BIAS_LOG2, NACC, PTMEM>(scores + c * 32, pDst, m, khalf * KHALF + c * 32, scale2,
                                    nsafe2, bias2, den2);
  }
  return den2;
}

// Read this warpgroup's KHALF scores out of TMEM exactly once, mask them, and
// return their maximum, leaving the values in `v` for the probability pass.
//
// pI11's single-mechanism ablation made the score readout the largest cost in the
// kernel - removing it took shape 7 from 258.66 to 167.71 us (-35.2%) and shape 1
// from 220.86 to 118.02 us (-46.6%) - and pI8 had already shown the cost is TMEM
// read bandwidth rather than wait latency, because halving the number of
// tcgen05.wait::ld per group changed nothing while removing the reads removed a
// third to a half of the runtime.  Pass 1 and pass 2 read the same tile, so
// keeping the values removes half of that traffic.
//
// The price is the FMNMX tree that `tcgen05.ld.red...max` used to do inside the
// load, and KHALF floats that stay live across the key-half exchange barrier, so
// the caller only takes this path where the output accumulator is not competing
// for the same registers.  The maximum itself is exact and order independent, and
// the mask is applied to the stored value exactly as the two-pass form applied it
// to each pass, so the numbers are unchanged.
template <int KHALF, bool MASKED, bool WANT_MAX>
__device__ __forceinline__ float tcg_scores(uint32_t tlane, uint32_t s_col, int khalf, int keyLo,
                                            int lo, int hi, float* v) {
  float bmRaw = -INFINITY;
#pragma unroll
  for (int c = 0; c < KHALF / 32; ++c) {
    tcg_ld32(tlane + s_col + khalf * KHALF + c * 32, v + c * 32);
    if (MASKED) {
      const unsigned span = (unsigned)(hi - lo);
      unsigned d = (unsigned)(keyLo - lo) + (unsigned)(c * 32);
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        if (d > span) v[c * 32 + j] = -INFINITY;
        ++d;
      }
    }
  }
  if (!WANT_MAX) return 0.0f;
  // Four independent chains instead of one: the scores have to survive for the
  // probability pass, so the in-place halving tree of tcg_group_max is not
  // available and a linear chain would be KHALF-1 dependent FMNMX (~250 cycles at
  // KHALF=64).  Four chains cut the depth to KHALF/4 + 2 without spending any
  // registers beyond the four accumulators.  max is exact and order independent.
  float b1 = -INFINITY, b2 = -INFINITY, b3 = -INFINITY;
#pragma unroll
  for (int j = 0; j < KHALF; j += 4) {
    bmRaw = fmaxf(bmRaw, v[j]);
    b1 = fmaxf(b1, v[j + 1]);
    b2 = fmaxf(b2, v[j + 2]);
    b3 = fmaxf(b3, v[j + 3]);
  }
  return fmaxf(fmaxf(bmRaw, b1), fmaxf(b2, b3));
}

// Combine one float per row across the KSPLIT key slices of a 128-key group
// through the `oSb` scratch.  Every warpgroup of a key-split CTA owns the same BM
// rows but a different BN/KSPLIT slice of the group's keys, so both the group
// maximum and the row denominator are a cross-warpgroup reduce.
//
// MAX=true is exact and order independent, so the result is bit-identical to one
// 128-wide reduction however many slices there are.  MAX=false folds the partials
// in key order, which reassociates the denominator by ~1e-7 relative - the same
// class of change the two-way fold already ships with at KSPLIT=2, five orders of
// magnitude inside the checker's rtol, and it cannot flip a quantized probability
// because every probability is rounded to E4M3 before the denominator exists.
//
// At KSPLIT=2 this reproduces the shipped expression exactly, both in the values
// and in their order, so no form this candidate already launches changes by one
// bit; at KSPLIT=1 it is never called.
template <int KSPLIT, int BM, bool MAX>
__device__ __forceinline__ float tcg_combine_halves(uint32_t sB_u, int khalf, int m, float v) {
  sts32f(sB_u + (uint32_t)((khalf * BM + m) * 4), v);
  __syncthreads();
  float r = lds32f(sB_u + (uint32_t)(m * 4));
#pragma unroll
  for (int h = 1; h < KSPLIT; ++h) {
    const float x = lds32f(sB_u + (uint32_t)((h * BM + m) * 4));
    r = MAX ? fmaxf(r, x) : (r + x);
  }
  return r;
}

// The three per-group issue/staging steps of the software-pipelined loop, as
// functions so that the pipelined and sequential forms cannot drift apart in the
// MMA descriptors they build.

// Stage one group's K (and optionally V) scale atoms into TMEM, replicated across
// the four lane subpartitions.  Only warps 0..3 take part: warps 4..7 address the
// same 128 lanes, so a second writer would race the first.
//
// ALL_WARPS lifts that guard for the group-split form, where the two warpgroups write
// DIFFERENT slots (its SFB_K/SFB_V carry one 8-column slot per warpgroup) and the
// caller passes each one its own base - so all eight warps stage at once and the lane
// slice still comes from `tlane`'s (warp & 3) term, exactly as it does for the four.
// WANT_K exists for the same form's V-only call: without it the K half would load from
// a null source and store to a null column base, i.e. into TMEM column 0, which is
// warpgroup 0's score tile.
template <int DCH, bool WANT_V, bool ALL_WARPS = false, bool WANT_K = true>
__device__ __forceinline__ void tcg_pipe_sf(uint32_t tlane, int warp, int t, uint32_t srcK,
                                            uint32_t srcV, uint32_t sfk_col, uint32_t sfv_col) {
  if (ALL_WARPS || warp < 4) {
#pragma unroll
    for (int g = 0; g < DCH; ++g) {
      if constexpr (WANT_K) {
        const uint4 wk = lds128(srcK + (uint32_t)(g * 512 + t * 16));
        tcg_st4(sfk_col + 4 * g, wk.x, wk.y, wk.z, wk.w);
      }
      if (WANT_V) {
        const uint4 wv = lds128(srcV + (uint32_t)(g * 512 + t * 16));
        tcg_st4(sfv_col + 4 * g, wv.x, wv.y, wv.z, wv.w);
      }
    }
    tcg_wait_st();
  }
}

// One key group's QK^T: D/32 block-scaled MMAs of BM x BN x 32 accumulating into
// the score tile at column `s_col`.  Called by one thread; completion is signalled
// by the caller's tcgen05.commit.
template <int D, int BM, int BN>
__device__ __forceinline__ void tcg_pipe_qk(uint32_t tmem, uint32_t s_col, uint32_t sQ_u,
                                            uint32_t kstage, uint32_t sfa_q, uint32_t sfb_k) {
#pragma unroll
  for (int kb = 0; kb < D / 32; ++kb) {
    const int chunk = kb >> 2, ink = kb & 3;
    const uint64_t da = umma_desc(sQ_u + (uint32_t)(chunk * (BM * 128)), 16, 1024) + 2 * ink;
    const uint64_t db = umma_desc(kstage + (uint32_t)(chunk * (BN * 128)), 16, 1024) + 2 * ink;
    tcg_mma_bs(tmem + s_col, da, db, tcg_idesc(BM, BN, 0, ink), (uint32_t)(kb > 0),
               sfa_q + 4 * chunk, sfb_k + 4 * chunk);
  }
}

// One key group's PV: (BN/32) x (D/128) block-scaled MMAs accumulating into the
// output tile.  `keep_first` is zero only for the very first MMA this CTA issues,
// which must overwrite because uninitialised TMEM can hold Inf/NaN and 0*Inf
// poisons the accumulator.
//
// `PTMEM` reads the probability operand out of tensor memory at `p_tmem + 8*kb`
// instead of out of the swizzled shared-memory tile `sP_u`; `ba1` established the
// 8-column k-step stride and that the two residences are numerically identical for
// this instruction.  B, the instruction descriptor (including its a_major bit),
// both scale-factor addresses and the accumulate predicate are unchanged, and the
// arithmetic order over keys is unchanged, so the accumulator is bit-identical.
template <int D, int BM, int BN, bool PTMEM = false>
__device__ __forceinline__ void tcg_pipe_pv(uint32_t tmem, uint32_t o_col, uint32_t sP_u,
                                            uint32_t p_tmem, uint32_t vstage, uint32_t sfa_p,
                                            uint32_t sfb_v, uint32_t keep_first) {
#pragma unroll
  for (int kb = 0; kb < BN / 32; ++kb) {
#pragma unroll
    for (int nc = 0; nc < D / 128; ++nc) {
      const uint64_t db =
          umma_desc(vstage + (uint32_t)(nc * (128 * BN)), 0, 1024) + (uint64_t)256 * kb;
      const uint32_t dcol = tmem + o_col + nc * 128;
      const uint32_t idesc = tcg_idesc(BM, 128, 1, kb);
      const uint32_t accum = (uint32_t)(kb > 0) | keep_first;
      if constexpr (PTMEM) {
        tcg_mma_bs_atmem(dcol, p_tmem + (uint32_t)(8 * kb), db, idesc, accum, sfa_p,
                         sfb_v + nc * 4);
      } else {
        const uint64_t da = umma_desc(sP_u, 16, 1024) + 2 * kb;
        tcg_mma_bs(dcol, da, db, idesc, accum, sfa_p, sfb_v + nc * 4);
      }
    }
  }
}

// TMA selects the bulk-tensor staging of K and V.  The two descriptors ride along
// as `__grid_constant__` parameters in every instantiation - a CUtensorMap cannot
// live inside the by-value `Params` without imposing its 64-byte alignment on the
// whole struct, and the non-TMA paths simply never read them.
#if FMHA_GSPLIT_WATCHDOG
// Latched so the FIRST stuck wait is also the only slow one: every later wait in the
// same launch returns immediately, the kernel retires with wrong numbers in a few
// milliseconds, and the job comes home with its log instead of dying on its deadline.
__device__ int gs_wdg_tripped = 0;
// tag: 1 = this warpgroup's QK, 2 = the mstate handoff, 3 = the denominator handoff,
// 4 = the step-3 PV catch-up, 5 = the step-9 PV catch-up, 6 = the epilogue PV catch-up.
__device__ __forceinline__ void gs_wait_wd(uint64_t* bar, uint32_t parity, int tag, int grp,
                                           int tid) {
  if (gs_wdg_tripped) return;
  uint32_t done = 0;
  long spins = 0;
  while (!done) {
    // Re-test the latch INSIDE the loop, and keep the bound at 100k polls.  bm2's first
    // version tested only on entry and spun 50M times, and one deadlocked launch of s0
    // cost 192,299,968 us - a single timing sample - because every resident thread paid
    // the full bound before the latch took effect.  A legitimate wait here is an MMA
    // completion or a handoff a few hundred cycles away, i.e. O(10) polls, so 100k is
    // four orders of magnitude of margin and the first tripper now costs milliseconds.
    if (gs_wdg_tripped) return;
    asm volatile(
        "{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P1;\n\t}\n"
        : "=r"(done)
        : "r"(smem_addr(bar)), "r"(parity));
    if (++spins > 100000L) {
      if ((tid & 127) == 0) {
        printf("GSWDG STUCK tag=%d tid=%d grp=%d parity=%u\n", tag, tid, grp, parity);
      }
      gs_wdg_tripped = 1;
      return;
    }
  }
}
#define GS_WAIT(BAR, PAR, TAG, GRP) gs_wait_wd((BAR), (PAR), (TAG), (GRP), tid)
#else
#define GS_WAIT(BAR, PAR, TAG, GRP) mbar_wait((BAR), (PAR))
#endif

// One 16-byte global store of four `wgt`-scaled floats, for the split pass's
// partial numerator.
//
// `ap21` (`dv_d5779d79adcd`, `MemoryWorkloadAnalysis_Tables` on s5) measured this
// store as the worst access pattern anywhere in the candidate: **"The memory access
// pattern for global stores to L1TEX might not be optimal. On average, only 4.0 of
// the 32 bytes transmitted per sector are utilized."**  The cause is the epilogue's
// own shape, which is otherwise exactly right: one row per thread, no reduction and
// no shuffles, so thread t writes row t's D floats contiguously while its neighbours
// write rows D*4 = 512 or 1024 bytes away.  Every 4-byte store from a warp therefore
// touches 32 different sectors and uses one eighth of each - 8x the sector
// transactions the bytes require.  s5's split pass is the most memory-pipe-loaded
// kernel this campaign has profiled (Mem Busy 42.21%, L1/TEX 43.14%, L2 42.21%,
// against 11.70% DRAM and 8.71% compute), and `part_a` is `nsplit*rows*D*4` = 14.5
// MB there, so those are the sectors doing it.
//
// `st.global.v4.u32` is the widest store this ISA has, so 16 bytes is the ceiling
// without changing who writes which address: it takes the sector transactions from
// 8x the data down to 2x, a 4x reduction, and cuts the store instruction count by
// the same factor.  Reaching 1x needs a transposed write through shared memory -
// consecutive threads writing consecutive 16-byte chunks - which is the follow-up
// if this pays, and would have to move `part_a`'s layout and the combine that reads
// it.  The values and their addresses are unchanged, so the partials are
// bit-identical and the combine kernel does not know the difference.
__device__ __forceinline__ void stg128_scaled(float* dst, const float* v, float wgt) {
  uint4 out;
  out.x = __float_as_uint(v[0] * wgt);
  out.y = __float_as_uint(v[1] * wgt);
  out.z = __float_as_uint(v[2] * wgt);
  out.w = __float_as_uint(v[3] * wgt);
  stg128(dst, out);
}

// pb2 port (a-side best/pb2_full): the BF16 twin - one 16-byte global store
// of EIGHT `wgt`-scaled bf16 partials.  Same values as stg128_scaled up to
// the bf16 rounding the gated format is defined by (2^-9 relative, convex-
// combination bound - see FMHA_PARTIALS_BF16), half the bytes, half the
// store instructions per row, and half the sectors the ap21 measurement
// counted.  `dst` must be 16-byte aligned: part_a's base is 256-aligned and
// every row offset is a multiple of D bf16 elements (256 B at D=128).
__device__ __forceinline__ void stg128_scaled16(__nv_bfloat16* dst, const float* v, float wgt) {
  __nv_bfloat162 packed[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 f = make_float2(v[2 * q] * wgt, v[2 * q + 1] * wgt);
    packed[q] = __float22bfloat162_rn(f);
  }
  uint4 out;
  out.x = *reinterpret_cast<unsigned*>(&packed[0]);
  out.y = *reinterpret_cast<unsigned*>(&packed[1]);
  out.z = *reinterpret_cast<unsigned*>(&packed[2]);
  out.w = *reinterpret_cast<unsigned*>(&packed[3]);
  stg128(dst, out);
}

// EXPERIMENT xt1: 1 replaces the split epilogue's per-thread part_a stores
// with a TRANSPOSED write through shared memory - the follow-up b's own
// ap21 comment named ("Reaching 1x needs a transposed write through shared
// memory - consecutive threads writing consecutive 16-byte chunks - which
// ... would have to move part_a's layout and the combine that reads it").
// nt1 (dv_90df33d6e54c, MemoryWorkloadAnalysis_Tables on the SHIPPED
// pb2t+fs7t tree) says the stores are still 2x even in bf16: s5's split
// global stores average 16.1 of 32 bytes per sector (est. speedup 5.0% of
// the kernel), d3's GM 17.0/32, s9's GM 26.1/32; the combines are already
// 28-30/32 and need nothing.  The layout does NOT move: each thread stages
// its row's scaled partials into the idle Q+K+V smem tiles (quiescent in
// the epilogue: every cp.async/TMA fill is waited and every MMA read is
// mbar-completed), then ALL threads write the tile back as consecutive
// 16-byte chunks - consecutive threads, consecutive addresses, 1x sectors.
// Values and addresses are unchanged (same wgt-scaled elements to the same
// part_a bytes), so outputs stay bit-identical and the combine cannot tell.
// Cost: one extra __syncthreads (plus the D256 staging inside the existing
// c-loop) in the epilogue only - no main-loop barrier is touched, so the
// zv2-9 pacing structure is unaffected; staging capacity is checked by
// static_assert against L::oP (128 rows x (D*elem+16) pitch <= Q+K+V).
// SHIPPED bf16-gated (xt2, default 1): dv_44c0c0f9eda2 (ungated) +
// dv_02148c2082e7 (gated confirmation) - d3 -4.87/-4.79%, s5 -1.96/-4.18%,
// s9 -0.06/-1.46% (pooled ~-0.75%), s3/s2/d7 null, every case bit-identical
// (values and addresses unchanged), engine PASS x2 seeds, census STACK:0
// with split REG 204-255 unchanged.  The fp32 tile measured +5.5% on
// s3/s2 (2x smem round trip against a ~0.2us sector tax at 2.4MB), hence
// the BF16PA compile-time gate - which the plan already restricts to
// part_a >= 6MB, exactly the byte regime where the trade pays.
#ifndef FMHA_PART_XPOSE
#define FMHA_PART_XPOSE 1
#endif

// xt3, RE-INTRODUCED ONTO THE JOIN'S TREE.  `FMHA_PART_XPOSE`'s cooperative transposed
// partials write is gated on BF16PA, and BF16PA is false everywhere because
// `FMHA_PARTIALS_BF16` ships at 0 -- so as adopted by join 20261026 the whole mechanism is
// DEAD CODE and every split shape writes its partials per-thread at 2.02x the minimum sector
// traffic (§171's census of all four D=128 split cases).  This arm's pre-join certified tree
// `b2c6fb1c` carried the fp32 arm of it, and the two trees are TEXTUALLY IDENTICAL in this
// region apart from the predicate names, so restoring it is a port and not a new mechanism.
//
// MEASURED PROVENANCE, from this arm's own source comments on `b2c6fb1c` (dv_7c925185c65b):
// `bitexact_ab.py` reported **BIT-IDENTICAL on all five shapes compared (s5, s4, s6, s2, s3),
// `o` and `lse` both `torch.equal` with ndiff 0, and the worst realized tolerance usage
// identical to six decimals (0.527612 in both arms)**.  Instructions go UP 16% (the smem round
// trip) while **cycles go DOWN 7.78%** -- trading issue slots for sector efficiency, which is
// the mechanism.  `bench.py` under the v2 ABBA protocol gave s5 42.976 -> 40.960 us (-4.69%)
// with three untouched controls flat to 0.01%.
// THE HONEST LIMIT, which is this record's own and is why the kernel number is the evidence:
// s5's forward-level -4.69% is a RUNG-OCCUPANCY shift, not a displacement of the rungs, and
// §149/row-193's resolution bound puts the smallest expressible effect on a 41 us shape at
// 2.0/41 = 4.9%.  The ncu-measured **-7.78% cycles on the isolated kernel** is order- and
// rung-independent and is what to believe.
// 0 = fp32 partials written per-thread (the join's effective behaviour).
// 1 = D=256 fp32 partials use the cooperative transposed write too.  Restored at 1.
#ifndef FMHA_PART_XPOSE_FP32
#define FMHA_PART_XPOSE_FP32 1
#endif

// xd1, RE-INTRODUCED ONTO THE JOIN'S TREE, same reasoning as FP32 above.  The D=128 register
// path's own gate.  MEASURED PROVENANCE from `b2c6fb1c`'s comments: value 1 (every D=128 fp32
// split partial) is REJECTED because it costs s2 5.7% and s3 0.9% -- the fp32 smem round trip
// exceeds the sector tax it removes at their 2.4 MB of partials (dv_44c0c0f9eda2 measured +5.5%
// at that size).  Value 2 gates on `!PSK`, which is exactly "the row slots are mostly live, so
// the loop's per-slot overhead is paid for bytes that are actually written", and **changes
// precisely two of the twenty declared shapes, s8 and s9** -- whose part_a is 16.777 and
// 16.794 MB (§246(a)), the two largest in the fixed suite.  Whole-suite effect recorded there:
// (1.073 * 1.017)^(1/10) = +0.88% of the fixed geomean, additive with FP32's +0.48% because the
// two act on disjoint keys (D=256 split vs D=128 split at PSK=0) and disjoint shapes (s5/d3 vs
// s8/s9).  Row 193's caveat applies to the s2/s3 rejection figures (each is ~0.9 rungs, i.e. at
// the resolution bound) but NOT to the s8/s9 gains, which are 2.1 and 1.05 rungs.
// 0 = per-thread write.  1 = every D=128 fp32 split partial (REJECTED: costs s2 5.7%).
// 2 = only where `!PSK`.  Restored at 2.
#ifndef FMHA_PART_XPOSE_D128
#define FMHA_PART_XPOSE_D128 2
#endif

#if FMHA_PART_XPOSE
// Row-slot remap for the cooperative transposed write: the same (i, h)
// derivation the kernel uses for its own thread, as a device function of an
// ARBITRARY slot (device lambdas would need --expt-extended-lambda, which
// the build does not pass - the first xt1 attempt compiled to tcg stubs
// through solution.py's silent arch fallback exactly on that error).
__device__ __forceinline__ int64_t xpose_row(int mm, int q0, int Qph, int log2Qph, int kh,
                                             int G, int log2G, bool pad_skip, int b, int H,
                                             int Sq) {
  const int ii = pad_skip ? (q0 + (mm >> log2G)) : (q0 + (mm & (Qph - 1)));
  const int hh = pad_skip ? (kh * G + (mm & (G - 1))) : (kh * G + (mm >> log2Qph));
  return (int64_t)((b * H + hh) * Sq) + ii;
}
__device__ __forceinline__ bool xpose_valid(int mm, int q0, int Qph, int log2G, bool pad_skip,
                                            int Sq) {
  const int ii = pad_skip ? (q0 + (mm >> log2G)) : (q0 + (mm & (Qph - 1)));
  return ii < Sq;
}
#endif

template <int D, int PASS, int KSPLIT = 1, bool TMA = false, bool QK64 = false,
          bool PSK = false, bool GSPLIT = false, bool BF16PA = false, bool FSCAN = false>
__global__ void __launch_bounds__(128 * KSPLIT,
                                  TcgMinBlocks<D, PASS, KSPLIT, QK64, GSPLIT>::value)
    fmha_tcg_kernel(const Params p, const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v) {
  // pb2 port: BF16PA selects the gated BF16 part_a format (FMHA_PARTIALS_BF16
  // doc carries the measurements and the numerics bound).  Template axis, not
  // a runtime branch: the FP32 instantiations stay byte-identical.
  static_assert(!BF16PA || PASS == PASS_SPLIT, "BF16 partials exist only on the split pass");
  // fs5t port (a-side fs1/fs3/fs4/fs5 lineage): FSCAN fuses the flat scan into
  // the split pass - each thread derives its chunk's delayed-maximum states
  // from p.bm directly (a bit-identical prefix walk + the fb/bnxt in-loop
  // recurrence) instead of replaying the scan kernel's mstate.  Template axis,
  // never a runtime branch: the non-fused instantiations stay byte-identical.
  static_assert(!FSCAN || PASS == PASS_SPLIT, "FSCAN exists only on the split pass");
#ifndef FMHA_TCG_ARCH
  // Unreachable in practice: the host only launches this kernel when
  // fmha_tcg_probe reported an arch-specific image.  Kept as a loud failure
  // rather than a silent wrong answer.
  if (threadIdx.x == 0 && blockIdx.x == 0) __trap();
  return;
#else
  (void)tmap_k;
  (void)tmap_v;
#if FMHA_PROBE_LEVEL == 1
  // DIAGNOSTICS ONLY -- see FMHA_PROBE_LEVEL.  Nothing below runs: no TMEM is allocated, no
  // mbarrier is armed, no async copy is issued, so there is nothing to leak or hang and the
  // measured Duration is the launch/CTA-ramp/teardown floor for this grid.  The static_asserts
  // below are compile-time and are still evaluated, so a probe build is as checked as a real one.
  return;
#endif
  static_assert(D == 128 || D == 256, "the tcgen05 path instantiates D=128/256");
  using L = TcgLayout<D, PASS, KSPLIT, GSPLIT>;
  using TT = TcgTmem<D, PASS, KSPLIT, QK64, GSPLIT>;
  // The group-split form is one specific point in this template space and is not
  // generalised beyond it: D=128 because its prize is the readout traffic that D=256
  // does not have (ab4: -2.5% there against -36.4% here), fused because the split
  // passes have their own state recurrence materialised in global memory, KSPLIT=2
  // because two warpgroups is what "two phase-offset streams" means, cp.async because
  // agu8 priced bulk-tensor staging at +3.49% on exactly the one-CTA-per-SM D=128 fused
  // shapes, and no QK64 because a 64-column score tile is a different trade that bi3
  // already closed.  Both probability tiles have to be in TMEM: there is no shared
  // memory P tile in this layout, and two warpgroups writing one smem tile would race.
  static_assert(!GSPLIT || (D == 128 && PASS == PASS_FUSED && KSPLIT == 2 && !TMA && !QK64 &&
                            TT::P_IN_TMEM && TT::ALLOC == 512),
                "the group-split form is D=128 fused, two warpgroups, cp.async, P in TMEM");
  // Splitting the key axis means two warpgroups own one row, so the accumulator
  // cannot live in either of their register files.
  // ... unless the pass has no accumulator at all, which is exactly what the
  // group-maximum pass is: it reduces the score tile to one float per row and
  // stores it, so it splits keys with O_IN_TMEM false and NEEDS_O false.
  static_assert(KSPLIT == 1 || TT::O_IN_TMEM || !TT::NEEDS_O,
                "key splitting needs the TMEM accumulator, or no accumulator");
  // A warpgroup's slice of a 128-key group has to stay a whole 32-key probability
  // chunk: `tcg_prob_chunk` converts exactly 32 keys into two 16-byte tile stores
  // and one E8M0 atom block, and the score readout is one `tcg_ld32` per 32 keys.
  static_assert(KSPLIT == 1 || KSPLIT == 2 || KSPLIT == 4,
                "KSPLIT has to divide BN into whole 32-key chunks");
  static_assert(L::BN / KSPLIT >= 32, "a key slice is at least one 32-key chunk");
  // The software-pipelined loops read a ring slot's scale atoms before the MMA that
  // consumes that slot's tile, which is sound while the atoms ride cp.async (whose
  // `wait_group 1` proves them) and unsound once they share the tile's mbarrier.
  // Both pipelined loops are therefore reordered under SF_BULK: the group-maximum
  // one waits on slot `nxt` above its `tcg_pipe_sf`, and the fused one waits on
  // slots `nxt` AND `cur` above its step 2, because step 2 reads the K atoms of the
  // group it is about to issue and the V atoms of the group it is about to
  // accumulate.  The MMA issues then skip the waits they used to own.
  constexpr bool WANT_V = (PASS != PASS_GROUP_MAX);
  // Whether the atoms ride the bulk path, which for a TMA instantiation is also
  // whether its loops issue any cp.async at all - and therefore whether the
  // commit_group/wait_group pair that used to follow every staging call is still
  // load-bearing or merely two instructions per thread per group.  Must agree with
  // tcg_issue_stage's own definition or a loop waits on a barrier whose expect_tx
  // counted bytes that were never copied.
  constexpr bool SF_BULK = TMA && (D == 128) && (FMHA_TCG_SF_BULK != 0);
  // Atoms staged three key groups ahead of the tiles they belong to, which is what lets
  // this loop's TMEM scale write sit above its staging barrier and the second per-group
  // CTA-wide barrier go away.  See FMHA_SF_AHEAD_D256 and
  // results/summaries/2026-10-21-design-sf-atoms-ahead.md.
  constexpr bool SF_AHEAD = L::SF_AHEAD;
  static_assert(!SF_AHEAD || TT::SF_AHEAD, "layout and TMEM map must agree on SF_AHEAD");
  // The atoms have to be on their own cp.async stream to have a ring of their own; on the
  // bulk path they ride the tile's mbarrier and cannot be staged independently of it.
  static_assert(!SF_AHEAD || !SF_BULK, "SF_AHEAD needs the atoms off the bulk path");
  // Two alternating TMEM slots per operand, because the write for g+1 happens while QK(g)
  // still reads the slot for g.  Without SF_ALT the second slot was never allocated.
  static_assert(!SF_AHEAD || TT::SF_ALT, "SF_AHEAD needs the alternating scale slots");
  // QK_AHEAD reorders this same loop for a different reason and claims the atom ring's
  // slot indices for its own; the predicate excludes it, so this is a guard, not a case.
  static_assert(!(SF_AHEAD && L::QK_AHEAD), "SF_AHEAD and QK_AHEAD both reorder the loop");
  // The parity argument that makes one write slot per group safe is that QK(g) and PV(g)
  // are both WAITED inside iteration g, which needs the accumulator in TMEM (SINGLE_READ)
  // and therefore the forms this predicate selects anyway.  Checked, not assumed.
  static_assert(!SF_AHEAD || (TT::O_IN_TMEM && !QK64), "SF_AHEAD needs O in TMEM and 128 columns");
  constexpr int BM = 128, BN = 128, KB = D / 32, DCH = D / 128;
  constexpr int THR = 128 * KSPLIT;    // threads per CTA
  constexpr int KHALF = BN / KSPLIT;   // keys this warpgroup owns per group
  constexpr int OCOLS = D / KSPLIT;    // output channels this warpgroup owns
  // EXPERIMENT rm1 - the .red maximum in the REGISTER-O sequential fused form.
  // `tcgen05.ld.red...max` writes 32 destinations that this kernel leaves dead;
  // the shipped gate excluded the one form with a register-resident O accumulator
  // on the ARGUMENT that it competes for them, never on a measurement. The
  // argument looked backwards at the allocation order: the non-red pass 1 already
  // materialises the same 32-value chunk in registers and keeps it live across a
  // ~31-deep in-place FMNMX halving tree, so the red form's dead destinations
  // live strictly shorter than the values they replace. What the swap deletes is
  // ~124 dependent-limited FMNMX per warp per key group plus the four serial
  // cross-chunk combines, i.e. exactly the class of cost agu4's FMHA_DEN_PAR
  // priced at -6.8..-8.7% on s7/s1 when it removed the denominator's serial chain
  // from THIS form (ap2: `wait`, fixed-latency dependency, is the largest stall on
  // both). pI8's bandwidth finding does not apply: it priced the pass-2 tile READ,
  // which this keeps byte-identical - .red moves the reduction into the load, it
  // does not remove any load.
  // Bit-identical by construction: max is exact and order independent, the .red
  // variant is the one the GM pass and every O-in-TMEM form already ship (pH1:
  // bad=0 of 128 lanes), the masked groups keep the register path because .red
  // cannot mask (the caller's anyMask warp vote is unchanged), and rowsLive is
  // warp-uniform wherever PAD_SKIP gates it, which is what .sync.aligned needs.
  // Scope: PASS_FUSED && !O_IN_TMEM is exactly <128,PASS_FUSED,KSPLIT=1,QK64=0,
  // GSPLIT=0> - frozen s7 and s1 and nothing else in either suite (every other
  // D=128 launch is PIPE/GSPLIT/QK64/split/GM, every D=256 launch is O_IN_TMEM),
  // so the other eighteen cases are byte-identical controls.
  //
  // MEASURED AND REJECTED - STAYS 0. dv_044f44782040 (official engine, same-job
  // A/B, --cases all --seeds 2 --compare-commit 439915b, candidate c5363a5
  // against the certified tree): compile/correctness/performance PASS in both
  // arms, mismatched_elements 0 on all ten cases in both arms (bit-identity
  // confirmed by the checker), eight byte-identical controls stable (s0/s4
  // latency identical to the digit; s2/s3/s5/s6/s8/s9 within +-0.3% median).
  // Treated: **s7 +3.92% median / +3.95% min (157.66 -> 163.84), s1 +2.54% /
  // +1.42% (159.78 -> 163.84)**; whole-suite geomean +0.648%. Both statistics
  // agree in sign on both shapes. The original register-competition argument is
  // therefore vindicated on measurement even though the live-range reasoning
  // said otherwise - at 247/256 registers the .red form's 32 dead destinations
  // per chunk cost more (allocation/scheduling pressure, or the .red LDTM's own
  // issue behaviour) than the 124 FMNMX they replace, exactly where DEN_PAR's
  // chain removal had paid. One caveat recorded honestly: s7/s1 are the
  // layout-bistable shapes (§133(g)) and both treated medians landed on 163.84,
  // a known member of the s7 slow family, so a mode re-roll may carry part of
  // the delta - but a re-roll is not a win either, and the arm is rejected on
  // either reading. Do not resubmit without a mechanism that removes the FMNMX
  // work WITHOUT the .red load (there is none in the ISA today).
#ifndef FMHA_RED_MAX_SEQ
#define FMHA_RED_MAX_SEQ 0
#endif
  constexpr bool RED_MAX_FITS = (PASS == PASS_GROUP_MAX) || TT::O_IN_TMEM ||
      ((FMHA_RED_MAX_SEQ != 0) && (PASS == PASS_FUSED) && !TT::O_IN_TMEM);
  // Reading the score tile out of TMEM once instead of once per pass removes half
  // of the kernel's single largest cost (pI11: 35.2% of shape 7, 46.6% of shape
  // 1), but it keeps KHALF scores live across the key-half exchange, which only
  // fits where no register-resident output accumulator competes for them.
  constexpr bool SINGLE_READ = (PASS != PASS_GROUP_MAX) && TT::O_IN_TMEM;

  extern __shared__ __align__(16) char smem_raw[];
  // UMMA descriptors drop the low 4 address bits and the SW128 XOR uses bits
  // [9:7], so every tile base has to be 1024-byte aligned.
  const uint32_t smem_u = (smem_window(smem_raw) + 1023u) & ~1023u;
  const uint32_t sQ_u = smem_u + (uint32_t)L::oQ;
  const uint32_t sK_u = smem_u + (uint32_t)L::oK;
  const uint32_t sV_u = smem_u + (uint32_t)L::oV;
  const uint32_t sP_u = smem_u + (uint32_t)L::oP;
  (void)(L::oSfQ);
  const uint32_t sSfK_u = smem_u + (uint32_t)L::oSfK;
  const uint32_t sSfV_u = smem_u + (uint32_t)L::oSfV;
  const uint32_t sB_u = smem_u + (uint32_t)L::oSb;
  __shared__ __align__(8) uint64_t mbar_qk;
  __shared__ __align__(8) uint64_t mbar_pv;
  // Group-split only: one QK barrier per warpgroup (each commits its own MMAs, and
  // `tcgen05.commit` tracks the prior asynchronous operations of the EXECUTING thread,
  // so a barrier per issuing thread is also a barrier per warpgroup's QK chain), plus
  // the two handoff barriers that carry (mstate) and (den) between them.  Sized to one
  // element for every other instantiation so their static shared-memory footprint - and
  // therefore their occupancy - is what it has always been.
  __shared__ __align__(8) uint64_t mbar_qk_g[GSPLIT ? 2 : 1];
  __shared__ __align__(8) uint64_t mbar_ms[GSPLIT ? 2 : 1];
  // FOUR, not two: one per warpgroup per iteration parity.  See the aliasing argument
  // at the denominator wait below - a barrier whose commits arrive two at a time between
  // one consumer's consecutive waits cannot be waited by parity at all.
  __shared__ __align__(8) uint64_t mbar_den[GSPLIT ? 4 : 1];
  // PV completion, one barrier per OWNER warpgroup.  bm2 (dv_10af290f506c) proved a
  // single shared one is not waitable by parity: the two warpgroups commit to it in
  // alternation, so between one warpgroup's consecutive waits the barrier advances by
  // TWO phases, and a wait for parity p is then answered by the phase after the one it
  // meant - which for the step-3 wait is a PV this warpgroup has not issued yet, i.e. a
  // deadlock.  Splitting by owner makes every barrier advance by exactly one phase
  // between consecutive waits on it, which is the only shape a parity wait can track.
  __shared__ __align__(8) uint64_t mbar_pv_g[GSPLIT ? 2 : 1];
  // One completion barrier per ring stage, used only by the TMA staging.  Static
  // shared memory, so the dynamic request in TcgLayout::bytes is unchanged; the
  // cp.async instantiations get one element so their static footprint - and
  // therefore their occupancy - is what it has always been.
  __shared__ __align__(8) uint64_t mbar_stage[TMA ? L::NSTAGE : 1];
  // EXPERIMENT qk1 - deferred PV in the 64-column score form.  See
  // FMHA_QK64_DEFER_PV for the full derivation.  DPV keeps the QK64 fused form's
  // ONE-readout-per-group TMEM map and 2-CTA-per-SM occupancy, and moves its PV
  // retirement from the issuing iteration's foot to the NEXT iteration's middle -
  // the PIPE loop's step-3 discipline - so the PV MMA runs under the next group's
  // staging wait, atom write, barriers and both QK half waits instead of being
  // spin-waited.  That in turn forces three relocations, all derived at the macro:
  // the V(g+1) TMA issues AFTER PV(g-1)'s retire (the two-slot V ring's slot
  // (g+1)&1 IS PV(g-1)'s slot), V(g)'s SFB_V atom write moves below the same
  // retire (single SFB_V slot; PV(g-1) was its last reader), and the K and V
  // stage barriers split (they are armed at different points, so they cannot
  // share one expect_tx accounting).  The K side, both rings' depths, the TMEM
  // column map, TcgLayout's byte count and every other form are untouched.
  constexpr bool DPV =
      (FMHA_QK64_DEFER_PV != 0) && QK64 && (PASS == PASS_FUSED) && TMA && (D == 128);
  // qk1: the V ring's own barriers.  Declared INSIDE the constexpr branch and
  // exported through a pointer: an unconditional [DPV ? 2 : 1] array leaves a
  // static-smem symbol in EVERY tcg instantiation, which shifts tmem_base_s and
  // the static footprint of forms that must stay byte-identical controls (the
  // first qk1 A/B read s3's documented bimodal median +7.06% while its min was
  // identical to the digit - a mode roll, but the controls' static layout did
  // move, and that is not a clean A/B).  A __shared__ declaration in a block
  // scope has kernel lifetime, so the pointer stays valid for the whole body.
  uint64_t* vbar_base = &mbar_stage[0];
  if constexpr (DPV) {
    __shared__ __align__(8) uint64_t mbar_stageV[2];
    vbar_base = &mbar_stageV[0];
  }
  // qk6: the mid-pipelined D=256 fused form's own ring barriers - three for the
  // two-ahead K ring, two for the V ring - under the same scoped-declaration
  // discipline DPV established (a __shared__ declaration in a block scope has
  // kernel lifetime, and the control instantiations' static footprint stays
  // byte-identical).  MIDPIPE arms K and V on separate barriers exactly like
  // DPV's SPLITBAR path, so tma.mbar points at the K array here.
  constexpr bool MIDPIPE = L::MIDPIPE && TMA;
  uint64_t* kbar_base = &mbar_stage[0];
  if constexpr (MIDPIPE) {
    __shared__ __align__(8) uint64_t mbar_stageK[3];
    __shared__ __align__(8) uint64_t mbar_stageV2[2];
    kbar_base = &mbar_stageK[0];
    vbar_base = &mbar_stageV2[0];
    // The three SFB_V slots live at 416..440 in the existing gap below P_TILE;
    // the map's SF_END bound (+16) under-reads the third slot, which is harmless
    // for P_LO but worth pinning so a future map move cannot overlap them.
    static_assert(TT::SFB_V + 24 <= TT::P_TILE,
                  "qk6: three SFB_V slots must fit below the probability tile");
  }
  __shared__ __align__(16) uint32_t tmem_base_s;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  // Threads m and 128+m own the same query row and split its keys.  Both reach
  // TMEM lane m, because a warp only ever addresses lanes [32*(warp%4), +32).
  const int m = tid & 127;  // TMEM lane == query row owned by this thread
  const int khalf = tid >> 7;

  // Descending-work dispatch order when the host asked for it; otherwise the
  // natural blockIdx mapping (see cta_index in fmha_common.cuh).
  const CtaIndex ci = cta_index(p);
  const int kvflat = ci.y;
  const int b = kvflat / p.KH;
  const int kh = kvflat - b * p.KH;
  const int G = p.group;
  const int Qph = p.qph128;
  const int log2Qph = p.log2_qph128;
  const int q0 = ci.x * Qph;
  if (q0 >= p.Sq) return;

  const uint8_t* k_base = p.k + b * p.k_b + kh * p.k_h;
  const uint8_t* v_base = p.v + b * p.v_b + kh * p.v_h;
  const uint8_t* sfk_slice = p.sfk + (int64_t)kvflat * p.Rpkv * p.Cq;
  const int64_t v_atom_base = (int64_t)kvflat * (p.cpv >> 2);
  // This CTA's tile coordinates inside the two descriptors.
  const TmaStage tma{&tmap_k, &tmap_v, kbar_base, vbar_base,
                     (int32_t)b, (int32_t)kh};
  (void)tma;

  // ---- row geometry: one query row per thread, so no cross-lane reductions
  // Query-major (see FMHA_PAD_SKIP) so the tile's dead rows are a prefix of the
  // warp range instead of a stride-Qph scatter through all four warps.  A
  // template parameter, not a runtime test: the guard itself is what costs the
  // unpadded forms registers and schedule quality.
  constexpr bool PAD_SKIP = PSK && (FMHA_PAD_SKIP != 0);
  const int log2G = PAD_SKIP ? (__ffs(G) - 1) : 0;
  const int i = PAD_SKIP ? (q0 + (m >> log2G)) : (q0 + (m & (Qph - 1)));
  const int h = PAD_SKIP ? (kh * G + (m & (G - 1))) : (kh * G + (m >> log2Qph));
  const bool valid = i < p.Sq;
  // Warp uniform once the rows are query-major, and always true for a full tile,
  // which is what every large-Sq shape gets: this costs one vote per CTA and one
  // predictable branch per group.
  const bool rowsLive = PAD_SKIP ? __any_sync(0xffffffffu, valid) : true;
  const int iClamped = valid ? i : (p.Sq - 1);
  const int off = p.Skv - p.Sq;
  const int center = iClamped + off;
  int hi = min(p.causal ? center : (p.Skv - 1), p.Skv - 1);
  int lo = max(p.has_window ? (center - p.window_left) : 0, 0);
  const int iMin = q0;
  const int iMax = min(q0 + Qph - 1, p.Sq - 1);
  const int hiMax = min(p.causal ? (iMax + off) : (p.Skv - 1), p.Skv - 1);
  const int loMin = max(p.has_window ? (iMin + off - p.window_left) : 0, 0);
  const int gLoAll = loMin >> 7;
  const int gHiAll = hiMax >> 7;
  const int64_t rowG = (int64_t)((b * p.H + h) * p.Sq) + (valid ? i : 0);

  // Key-range split.  Splits tile this CTA's useful group span, not [0,T): with
  // causal or windowed masks the useful span can be a small interior slice, and
  // tiling [0,T) leaves one split doing almost all the work.
  int cLo = gLoAll, cHi = gHiAll;
  if (PASS != PASS_FUSED) {
    const int span = gHiAll - gLoAll + 1;
    const int per = (span + p.nsplit - 1) / p.nsplit;
    cLo = gLoAll + ci.z * per;
    cHi = min(cLo + per - 1, gHiAll);
  }
  const bool any_compute = (cLo <= cHi);

#if FMHA_CTA_PHASE_STAGGER != 0
  // EXPERIMENT ph1 — inter-CTA phase de-synchronization (pre-registration:
  // results/summaries/2026-10-25-ph1-cta-phase-stagger-prereg.md).
  //
  // The co-resident CTAs of a single-wave 2-per-SM D128 launch run IDENTICAL
  // instruction streams over equal spans with no cross-CTA communication, so
  // their per-group stall phases (MMA-completion waits, barrier convergence —
  // sp1: no-eligible 46% on split128, 81% on GM) can ALIGN, idling the SM's
  // issue bandwidth in lockstep. Delaying the second co-resident member
  // (linear ID in [sms, 2*sms) — the rr-fill slot-B set pbn measured) by
  // about HALF the form's per-group period makes one CTA's compute phase
  // overlap its partner's stall phase. Bit-exact by construction: a pure
  // timing offset, no values, orders, or memory effects change. __nanosleep
  // (not a clock spin) so the delayed CTA yields issue slots while waiting.
  // Gated: D128 non-gsplit KSPLIT=1 forms only (the 2-CTA/SM classes; D256
  // is 1 CTA/SM — nothing to de-phase), per-CTA span >= MIN_SPAN (short
  // spans cannot amortize the offset), queued-class CTAs beyond the first
  // wave use the odd-wave heuristic ((dLin/sms) odd).
  if constexpr ((D == 128) && (KSPLIT == 1) && !GSPLIT) {
    constexpr bool isGM = (PASS == PASS_GROUP_MAX);
    constexpr int kMode = FMHA_CTA_PHASE_STAGGER;  // 1=non-GM, 2=GM, 3=both
    if constexpr ((kMode == 3) || (kMode == 1 && !isGM) || (kMode == 2 && isGM)) {
      if (any_compute && (cHi - cLo + 1) >= FMHA_PHASE_STAGGER_MIN_SPAN && p.pair_p > 0) {
        const unsigned dLin = (unsigned)blockIdx.x +
            (unsigned)gridDim.x * ((unsigned)blockIdx.y +
                                   (unsigned)gridDim.y * (unsigned)blockIdx.z);
        if ((((dLin / (unsigned)p.pair_p) & 1u) == 1u)) {
          __nanosleep(isGM ? (unsigned)FMHA_PHASE_STAGGER_NS_GM
                           : (unsigned)FMHA_PHASE_STAGGER_NS);
        }
      }
    }
  }
#endif

  if (PASS == PASS_GROUP_MAX) {
    // Groups outside this CTA's span are entirely masked for every one of its
    // rows, so their maximum is -inf; the first and last split record those
    // entries so the scan still sees a fully initialised array.
    // Both warpgroups of a key-split CTA own the same 128 rows, so only the first
    // of them writes: the values would be identical either way, and this loop can
    // be as long as the whole key range.
    if (valid && khalf == 0 && ci.z == 0) {
      for (int t = 0; t < gLoAll; ++t) p.bm[(size_t)t * p.rows_total + rowG] = -INFINITY;
    }
    if (valid && khalf == 0 && ci.z == p.nsplit - 1) {
      for (int t = gHiAll + 1; t < p.T; ++t) p.bm[(size_t)t * p.rows_total + rowG] = -INFINITY;
    }
  }

  if (tid == 0) {
    mbar_init(&mbar_qk, 1);
    mbar_init(&mbar_pv, 1);
    if constexpr (GSPLIT) {
      // The QK barriers take one arrival, from the tcgen05.commit of the thread that
      // issued them.  The handoff barriers take one per ROW: all 128 threads of the
      // producing warpgroup store their own row's float and then arrive, so the phase
      // completes exactly when all 128 rows of one group's state have landed, and the
      // consumer's wait is the only release/acquire pair the store needs.
      mbar_init(&mbar_qk_g[0], 1);
      mbar_init(&mbar_qk_g[1], 1);
      mbar_init(&mbar_ms[0], 128);
      mbar_init(&mbar_ms[1], 128);
      mbar_init(&mbar_den[0], 128);
      mbar_init(&mbar_den[1], 128);
      mbar_init(&mbar_den[2], 128);
      mbar_init(&mbar_den[3], 128);
      mbar_init(&mbar_pv_g[0], 1);
      mbar_init(&mbar_pv_g[1], 1);
    }
    if constexpr (TMA) {
      // One arrival per stage: thread 0 arms it with expect_tx and the bulk copy
      // completes it, so no other thread may arrive.
#pragma unroll
      for (int s = 0; s < L::NSTAGE; ++s) mbar_init(&mbar_stage[s], 1);
      // qk1: the V ring's own barriers, same one-arrival contract.
      if constexpr (DPV) {
        mbar_init(vbar_base, 1);
        mbar_init(vbar_base + 1, 1);
      }
      // qk6: MIDPIPE's own K (three-slot) and V (two-slot) ring barriers.
      if constexpr (MIDPIPE) {
        mbar_init(kbar_base, 1);
        mbar_init(kbar_base + 1, 1);
        mbar_init(kbar_base + 2, 1);
        mbar_init(vbar_base, 1);
        mbar_init(vbar_base + 1, 1);
      }
      // No tensormap acquire fence here on purpose: the descriptors arrive as
      // __grid_constant__ kernel parameters, which the driver publishes before the
      // CTA starts.  Atrex-FA's tma_descriptor_fence_acquire is for descriptors a
      // producer kernel wrote into global memory, which these are not, and
      // fence.proxy.tensormap on a param-space address is not something to hand
      // ptxas 13.1 untested (see the cvt.rn.bf16x2.e4m3x2 rejection in ag1).
      // The barrier init does need the async-proxy fence before the first copy.
      fence_async_shared();
    }
  }
#if FMHA_PROBE_LEVEL == 2
  // DIAGNOSTICS ONLY -- see FMHA_PROBE_LEVEL.  The mbarrier init and its async-proxy fence above
  // have run and nothing else has: no async copy is issued at any probe level, so this return
  // leaves nothing in flight.  Duration(level 2) - Duration(level 1) prices the init + fence.
  return;
#endif
#if FMHA_PRIME_HOIST && !FMHA_PROBE_LEVEL
  // pr1-P2: group cLo's cold slot-0 prime goes out FIRST — its TMA lands while
  // everything below (Q LDGs, sfq LDG, alloc, Q STS) still runs. The matching
  // body skips keep the issue exactly-once; SF_AHEAD forms keep the in-body
  // prime (its TMEM atom writes need the alloc).
  if constexpr (!TT::SF_AHEAD) {
    if (any_compute) {
      if constexpr (DPV) {
        // qk1: split prime - K(cLo) on mbar_stage[0], V(cLo) on mbar_stageV[0].
        // Nothing is in flight yet, so both issues are order-free; the first
        // iteration's K wait and its mid-body V wait find their own armed barrier.
        tcg_issue_stage<D, PASS, THR, TMA, true, false, true, true>(
            p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
            v_atom_base, tma);
        tcg_issue_stage<D, PASS, THR, TMA, false, true, true, true>(
            p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
            v_atom_base, tma);
      } else {
        tcg_issue_stage<D, PASS, THR, TMA>(p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u,
                                           k_base, v_base, sfk_slice, v_atom_base, tma);
      }
      if constexpr (!SF_BULK) cp_async_commit();
    }
  }
#if FMHA_PRIME_HOIST_SFA && !FMHA_PROBE_LEVEL
  // sfa1: the SF_AHEAD arm, as a SEPARATE `if constexpr` and not an `else` — with
  // `if constexpr (A) {...} else if (runtime) {...}` the else arm is taken whenever A is
  // false, so a !SF_AHEAD form would issue this prime too and the tile would be staged
  // twice into slot 0.  Written this way exactly one of the two blocks is instantiated:
  // this one for the two D=256 non-GM keys, the one above for every other form.
  //
  // Only the ISSUE moves.  `cp_async_commit()`, `cp_async_wait<0>()`, the fences, the
  // barrier and `tcg_pipe_sf` stay in the body, because `tcg_pipe_sf` writes TMEM through
  // `tlane` and `tlane` needs the alloc below.  The commit group still contains exactly
  // these three atom blocks and nothing else (see the knob's comment for the enumeration
  // of every cp.async between here and there), so the loop's first `wait_group 1` still
  // has exactly one older group to prove.
  if constexpr (TT::SF_AHEAD) {
    if (any_compute) {
      tcg_issue_stage<D, PASS, THR, TMA, true, true, false>(p, tid, 0, 0, cLo, sK_u, sV_u,
                                                            sSfK_u, sSfV_u, k_base, v_base,
                                                            sfk_slice, v_atom_base, tma);
      tcg_stage_atoms<D, PASS, THR>(p, tid, 0, 0, cLo, sSfK_u, sSfV_u, sfk_slice, v_atom_base);
      // Slots 1 .. NSTAGE_SF-1, i.e. groups cLo+1 .. cLo+NSTAGE_SF-1, verbatim from the
      // body prime this replaces.
#pragma unroll
      for (int s = 1; s < L::NSTAGE_SF; ++s) {
        if (cLo + s <= cHi) {
          tcg_stage_atoms<D, PASS, THR>(p, tid, s, s, cLo + s, sSfK_u, sSfV_u, sfk_slice,
                                        v_atom_base);
        }
      }
    }
  }
#endif
#endif
#if FMHA_PROLOG_LDG
  // pr1-P1: both cold LDG batches issue before the alloc; their stores stay
  // below it (tlane dependency). Same bytes/values, issue-order only.
  // PROLOG_DCH BUG (official dv_f149666d68e0, D256 cases 4/5/6 failed 2/2
  // seeds): this originally sized the scale-word batch D/32; the scales block
  // is ONE 32-bit word per 128-d chunk = kernel-scope DCH = D/128. The extra
  // iterations' tcg_st1 writes ran past the 8-column SFA_Q region — benign on
  // D128 (they hit only SFB_K parity-0, which every group rewrites before its
  // MMA reads it — which is why the D128-only screen and 7 official D128
  // cases passed) but FATAL on D256's dense map (garbage sfq words landed in
  // SFA_P/SFB_V, which nothing rewrites before the PV MMA). Fixed to DCH.
  constexpr int PROLOG_CH = D / 16;
  constexpr int PROLOG_QITER = (BM * PROLOG_CH) / THR;
  uint32_t sfw[DCH];
  uint4 qd[PROLOG_QITER];
#if !FMHA_PROBE_LEVEL
  if (warp < 4) {
#pragma unroll
    for (int g = 0; g < DCH; ++g) {
      const uint8_t* slice = p.sfq + (int64_t)(b * p.H + h) * p.Rpq * p.Cq;
      sfw[g] = valid ? ldg32(slice + sf_rowwise_offset(iClamped, 4 * g, p.Cq)) : 0x7f7f7f7fu;
    }
  }
#pragma unroll
  for (int it = 0; it < PROLOG_QITER; ++it) {
    const int c = it * THR + tid;
    const int r = c / PROLOG_CH;
    const int iq = PAD_SKIP ? (q0 + (r >> log2G)) : (q0 + (r & (Qph - 1)));
    qd[it] = make_uint4(0u, 0u, 0u, 0u);
    if (iq < p.Sq) {
      const int ci = c - r * PROLOG_CH;
      const int hq = PAD_SKIP ? (kh * G + (r & (G - 1))) : (kh * G + (r >> log2Qph));
      qd[it] = ldg128(p.q + b * p.q_b + (int64_t)hq * p.q_h + (int64_t)iq * p.q_s + ci * 16);
    }
  }
#else
  // DIAGNOSTICS ONLY: the two cold LDG batches are compiled out at every probe level so that an
  // early return cannot leave a global load in flight.  The ARRAYS stay declared and are filled
  // with the same values the `!valid` / `iq >= p.Sq` predicates would have produced, because the
  // store side below the alloc still has to compile -- and at levels 1-3 it is dead code anyway,
  // since every rung returns before reaching it.
  if (warp < 4) {
#pragma unroll
    for (int g = 0; g < DCH; ++g) sfw[g] = 0x7f7f7f7fu;
  }
#pragma unroll
  for (int it = 0; it < PROLOG_QITER; ++it) qd[it] = make_uint4(0u, 0u, 0u, 0u);
#endif
#endif
  if (tid < 32) tcg_alloc(&tmem_base_s, TT::ALLOC);
  __syncthreads();
  const uint32_t tmem = tmem_base_s;
  const uint32_t tlane = tmem + ((uint32_t)((warp & 3) * 32) << 16);
  if (tid < 32) tcg_relinquish();
#if FMHA_PROBE_LEVEL == 3
  // DIAGNOSTICS ONLY -- see FMHA_PROBE_LEVEL.  The allocation, the CTA-wide barrier that publishes
  // it and the relinquish have all run; nothing below has.  Duration(level 3) - Duration(level 2)
  // prices tcg_alloc + __syncthreads() + tcg_relinquish(), which is the component §244(e) named as
  // one warp's `tcgen05.alloc` with 255 other threads waiting on it.  The dealloc mirrors the real
  // epilogue's `if (tid < 32) tcg_dealloc(tmem, TT::ALLOC)` so repeated launches in one profile run
  // cannot exhaust the 512 TMEM columns per SM.  No async copy is in flight at any probe level, so
  // there is nothing to drain before exiting.
  if (tid < 32) tcg_dealloc(tmem, TT::ALLOC);
  return;
#endif

  // Where this form publishes its quantized probabilities for the PV MMA: TMEM
  // column `TcgTmem::P_TILE` when the form's column map has 32 to spare, the
  // swizzled shared-memory tile otherwise (see FMHA_P_IN_TMEM for the measurement
  // and the map).  `tlane` already carries this thread's 32-row slice and
  // `m == tid & 127` is exactly the lane it addresses, so a store at
  // `pDst + colBase/4` puts row m's 32 keys in the eight words the A-from-TMEM MMA
  // reads at `pTmem + 8*kb` - the convention `ba1` pinned on this part.  The MMA is
  // issued by tid 0 and addresses the tile without a slice, as every other TMEM
  // operand here does.
  constexpr bool PTMEM = TT::P_IN_TMEM;
  const uint32_t pDst = PTMEM ? (tlane + (uint32_t)TT::P_TILE) : sP_u;
  const uint32_t pTmem = tmem + (uint32_t)TT::P_TILE;

#if FMHA_PROLOG_LDG
  // pr1-P1 store side: the Q payload from the hoisted qd[] batch, then this
  // CTA's sfq words from sfw[] — identical values and destinations to the
  // #else path, whose loads ran here instead of above the alloc.
#pragma unroll
  for (int it = 0; it < PROLOG_QITER; ++it) {
    const int c = it * THR + tid;
    const int r = c / PROLOG_CH;
    const int ci = c - r * PROLOG_CH;
    sts128(sQ_u + (uint32_t)((ci >> 3) * (BM * 128) + fp8_tile_offset(r, (ci & 7) * 16, 128)),
           qd[it]);
  }
  if (warp < 4) {
#pragma unroll
    for (int g = 0; g < DCH; ++g) {
      tcg_st1(tlane + TT::SFA_Q + 4 * g + (warp & 3), sfw[g]);
      if (WANT_V) tcg_st1(tlane + TT::SFA_P + 4 * g + (warp & 3), 0x7f7f7f7fu);
    }
    tcg_wait_st();
  }
#else
  // ---- Q payload tile, in 128-byte K-major swizzle chunks
  {
    constexpr int CH = D / 16;
    constexpr int QITER = (BM * CH) / THR;
#if FMHA_QSTAGE_SPLIT != 0
    // qs2: all loads, then all stores.  See FMHA_QSTAGE_SPLIT for the measurement that
    // motivated it and for the register cost that decides whether it can ship ungated.
    // Only the loaded payload has to survive to the second loop.  `iq`, `ci`, `hq` and `r`
    // are all recomputed there from `it` and `tid`, which are free; keeping them in arrays
    // would cost QITER more live registers in the loop where registers are the whole
    // question (32 for qd alone at QITER=8).
    uint4 qd[QITER];
#pragma unroll
    for (int it = 0; it < QITER; ++it) {
      const int c = it * THR + tid;
      const int r = c / CH;
      const int iq = PAD_SKIP ? (q0 + (r >> log2G)) : (q0 + (r & (Qph - 1)));
      qd[it] = make_uint4(0u, 0u, 0u, 0u);
      if (iq < p.Sq) {
        const int ci = c - r * CH;
        const int hq = PAD_SKIP ? (kh * G + (r & (G - 1))) : (kh * G + (r >> log2Qph));
        qd[it] = ldg128(p.q + b * p.q_b + (int64_t)hq * p.q_h + (int64_t)iq * p.q_s + ci * 16);
      }
    }
#pragma unroll
    for (int it = 0; it < QITER; ++it) {
      const int c = it * THR + tid;
      const int r = c / CH;
      const int ci = c - r * CH;
      sts128(sQ_u + (uint32_t)((ci >> 3) * (BM * 128) + fp8_tile_offset(r, (ci & 7) * 16, 128)),
             qd[it]);
    }
#else
// FMHA_QSTAGE_UNROLL: see the knob's comment.  Written as literal `#pragma unroll N` arms
// rather than `#pragma unroll FMHA_QSTAGE_UNROLL` because macro arguments inside a pragma
// are not expanded portably; `_Pragma` with a stringified macro would be, but the explicit
// arms keep the shipped value 2 textually identical to what it replaced.
#if FMHA_QSTAGE_UNROLL == 0
#pragma unroll
#elif FMHA_QSTAGE_UNROLL == 1
#pragma unroll 1
#elif FMHA_QSTAGE_UNROLL == 4
#pragma unroll 4
#elif FMHA_QSTAGE_UNROLL >= 8
#pragma unroll 8
#else
#pragma unroll 2
#endif
    for (int it = 0; it < (BM * CH) / THR; ++it) {
      const int c = it * THR + tid;
      const int r = c / CH;
      const int ci = c - r * CH;
      const int iq = PAD_SKIP ? (q0 + (r >> log2G)) : (q0 + (r & (Qph - 1)));
      uint4 data = make_uint4(0u, 0u, 0u, 0u);
      if (iq < p.Sq) {
        const int hq = PAD_SKIP ? (kh * G + (r & (G - 1))) : (kh * G + (r >> log2Qph));
        data = ldg128(p.q + b * p.q_b + (int64_t)hq * p.q_h + (int64_t)iq * p.q_s + ci * 16);
      }
      sts128(sQ_u + (uint32_t)((ci >> 3) * (BM * 128) + fp8_tile_offset(r, (ci & 7) * 16, 128)),
             data);
    }
#endif
  }
  // ---- Q scales.  Thread m owns row m and needs the
  // four (or eight) E8M0 bytes of its own row, which the F8_128x4 atom stores as
  // one contiguous 32-bit word per 128-d chunk, so this is a single 4-byte load
  // per row and chunk, issued once per CTA.
  if (warp < 4) {
    for (int g = 0; g < DCH; ++g) {
      const uint8_t* slice = p.sfq + (int64_t)(b * p.H + h) * p.Rpq * p.Cq;
      const uint32_t word =
          valid ? ldg32(slice + sf_rowwise_offset(iClamped, 4 * g, p.Cq)) : 0x7f7f7f7fu;
      // The word column encodes m/32, i.e. this warp's lane subpartition.
      tcg_st1(tlane + TT::SFA_Q + 4 * g + (warp & 3), word);
      // P's scale for PV is the constant 1.0: the contract's bias is already
      // baked into the quantized probability before it reaches this GEMM.  The
      // group-maximum pass issues no PV, so it skips this - which is also what
      // leaves its SFA_P columns free for the pipelined form's second SFB_K slot.
      if (WANT_V) tcg_st1(tlane + TT::SFA_P + 4 * g + (warp & 3), 0x7f7f7f7fu);
    }
    tcg_wait_st();
  }
#endif

  // The O accumulator stays in registers for D=128.  Packing it as f32x2 pairs
  // saved 64 FADDs per group, but the pH2 SASS census showed ptxas paying 93
  // IMAD.MOV.U32 in the same basic block to satisfy the aligned-register-pair
  // constraint that the TMEM load destinations do not already meet: a net loss
  // of ~29 instructions per group, plus 8 bytes of stack in the SPLIT
  // instantiation.  For D=256 the accumulator lives in TMEM instead and this
  // array is dead code that ptxas removes.
  float oacc[TT::O_IN_TMEM ? 1 : D];
  if (!TT::O_IN_TMEM) {
#pragma unroll
    for (int t = 0; t < D; ++t) oacc[t] = 0.0f;
  }
  // ---- the fourth boundary's wait, placed as late as it can go.
  // Everything above this line is independent of the scan: the shared-memory barrier
  // init and its async-proxy fence, tcg_alloc and the relinquish, the Q payload staging
  // (reads p.q), the Q and P scale writes to TMEM (reads p.sfq), and the accumulator
  // clear.  Everything below reads p.mstate, which the scan produced.  The only other
  // accesses to a scan-produced buffer in this kernel are the two p.bm stores above, and
  // those are inside `if (PASS == PASS_GROUP_MAX)` -- the gm pass, which is the scan's
  // ancestor rather than its successor and is never launched with the attribute.
  // So this is where a PDL-launched split must stop: gating it on PASS_SPLIT keeps the
  // instruction out of the fused and group-max forms entirely, which launch without the
  // attribute and would pay a griddepcontrol.wait against nothing.
  if constexpr (PASS == PASS_SPLIT) fmha_pdl_scan_wait();
  float mstate = -INFINITY;
  float den = 0.0f;
  // fs5t: the fused form derives both states locally (walk at the loop head,
  // fb/bnxt recurrence in the loop); the replay loads are compiled out and
  // p.mstate is never dereferenced in that form.
  if (PASS == PASS_SPLIT && cLo > 0 && !FSCAN) {
    // Replay the exact delayed-maximum state this chunk inherits.
    mstate = p.mstate[(size_t)(cLo - 1) * p.rows_total + rowG];
  }
  // The split pass's replayed running maximum for the group it is about to run,
  // loaded one group ahead of the recurrence that consumes it.  See the loop head.
  float mt_cur = -INFINITY;
  if (PASS == PASS_SPLIT && cLo <= cHi && !FSCAN) {
    mt_cur = p.mstate[(size_t)cLo * p.rows_total + rowG];
  }
  const float scale = p.scale;
  // Compile-time per head dimension.  The host publishes the same expression into
  // p.bias from ProbabilityBias<D>, so the two cannot disagree; `bias` is only read
  // to build bias2 and for the LSE's division, both of which are dead at D=256 once
  // FMHA_UNIT_BIAS deletes the multiply.
  constexpr float kBias = ProbabilityBias<D>::value;
  // -1: apply the bias with a packed multiply, which is what a bias that is not a
  // power of two would have to do.  0: no bias at all.  L > 0: 2^L folded into
  // expf's exponent field, which FMHA_FOLD_BIAS16 gates on an exhaustive proof.
  // FMHA_BIAS_MAGIC_FOLD also selects L=4 but folds 2^L into the two MAGIC CONSTANTS
  // rather than into the shifted pattern, so `d2` stays a shift on the ALU pipe and the
  // fold costs -1 instruction AND -1 on the FMA pipe instead of being a wash.  It is
  // bit-exact and it is MEASURED WORSE (+0.710% on the fixed geomean) because it re-rolls
  // the s0/s1/s7 build mode; see its own comment and section 119.  It stays in the tree
  // at default 0 as the layout probe for section 130's mode diagnostic, not as a candidate.
  constexpr int kBiasLog2 =
      (FMHA_UNIT_BIAS == 0)
          ? -1
          : ((kBias == 1.0f) ? 0
                             : (((FMHA_FOLD_BIAS16 != 0 || FMHA_BIAS_MAGIC_FOLD != 0) &&
                                 kBias == 16.0f)
                                    ? 4
                                    : -1));
  // Four partial denominator accumulators, but only in a one-warpgroup CTA: agu4
  // measured every KSPLIT=1 case neutral-to-better (s7 -6.76%, s1 -7.10% on min)
  // and every KSPLIT=2 case worse in both rounds (s0 +1.97%, s4 +1.31%, s6 +1.91%).
  // Partial denominator accumulators, and how many: four in a one-warpgroup CTA,
  // two in a two-warpgroup one.  agu4 measured FOUR on every form and the result
  // split exactly along KSPLIT - all five KSPLIT=1 cases neutral-to-better (s7
  // -6.76%, s1 -7.10% on min) and all three KSPLIT=2 cases worse in both rounds (s0
  // +1.97%, s4 +1.31%, s6 +1.91%), with <256,0,2,true> going 208 -> 210 registers.
  // A one-warpgroup CTA has four warps and nothing else to cover a sixteen-link
  // dependency; a two-warpgroup CTA already has eight, so it wants less chain
  // shortening and pays more for the live accumulator pairs.  Two accumulators halve
  // the chain for one register pair instead of three, and they cover the five frozen
  // shapes the KSPLIT=1 gate left out (0, 4, 5, 6, 8).
  // agu6 (dv_ba2eb2693754, base best/ao8_full, two rounds, 8/8 PASS in all four,
  // zero mismatched elements) tried 2 here for KSPLIT > 1 and it lost too: on min,
  // d6 +1.89%, s4 +1.56%, s6 +1.07%, d8 +0.71%, s0 +0.24%, s8 -0.55%, geomean
  // +0.82% over the six two-warpgroup cases while the two byte-identical KSPLIT=1
  // controls read -0.75% as a group, i.e. about +1.6% corrected.  Together with
  // agu4's four-accumulator measurement (+1.31% to +1.97% on the same three forms)
  // that brackets the answer: a two-warpgroup CTA does not want its denominator
  // chain shortened at either setting, and the register movement is not the reason
  // - NACC=2 took <128,0,2,false> from 196 to 186 registers and still lost.  So
  // KSPLIT > 1 keeps the sixteen-link serial chain.
  constexpr int kDenAcc = (FMHA_DEN_PAR != 0) ? ((KSPLIT == 1) ? 4 : 1) : 1;
  const float bias = kBias;

  if constexpr (GSPLIT) {
    // =================== group-split (AG) pipelined loop =========================
    // Warpgroup `w` owns WHOLE key groups cLo+w, cLo+w+2, ... - all 128 keys and all
    // 128 output channels of each - so its score tile is read exactly once per group
    // and its accumulator never leaves TMEM.  That is the same traffic the pipelined
    // key split moves and two thirds less than the shipped form's (ab1: -36.4% for
    // deleting all three of its readout tiles, -12.1% each).  The two forms differ
    // only in SCHEDULE, and the schedule is the whole experiment: PIPE puts both
    // warpgroups on the SAME group, so all eight warps arrive at every rendezvous
    // together and the SM runs one lockstep stream, which bk1 measured at +31.8% on s7
    // against the shipped form with its CTA-wide combine worth only 0.9% of that.
    // Here the contract's own delayed maximum holds the warpgroups half a group apart -
    // group g+1's `safe` is a function of group g's maximum - so one warpgroup is
    // reading scores out of TMEM while the other computes exponentials, and ap2's 52.9%
    // of cycles with nothing eligible are filled by different work instead of by the
    // same work waiting.
    //
    // What the two warpgroups share, and how each stays ordered:
    //   * O, and therefore the PV chain, because `num = num*alpha + P@V` is not
    //     commutative and the contract's group order is part of its numerics.  Each
    //     warpgroup issues its OWN PV but only after waiting for the previous group's PV
    //     to COMPLETE, so at most one PV is ever in flight and the accumulate order is
    //     the group order by construction.  That is deliberate: it avoids needing any
    //     cross-thread issue-order guarantee from tcgen05.mma, which the two warpgroups'
    //     QK issues DO rely on being absent - those are genuinely concurrent and are safe
    //     only because they write different accumulators (S0/S1) from different operands.
    //   * (mstate, den): one float per row through shared memory, no reduction and no
    //     CTA-wide barrier.  mstate is published early, right after the maximum tree,
    //     because the next group's exponentials need it; den is published late, after the
    //     exponential pass, because nothing consumes it but the epilogue's 1/den and
    //     LSE.  hsMs needs one slot per warpgroup and hsDen two, and both follow from the
    //     handoff chain rather than from extra synchronisation: w's next mstate write is
    //     after its wait for mstate(g+1), which is after 1-w's arrive, which is after
    //     1-w's read; but w's den(g+2) write is NOT ordered after 1-w's den(g) read by
    //     that chain, so the parity slot supplies the one group of slack the chain does.
    //   * the Q tile, staged once by all 256 threads in the prologue.
    //
    // Every rendezvous inside the loop is 128 threads wide, on barrier resource 1+w:
    // "every one of my four warps retired its tcgen05.ld" and "my probability tile is
    // published" are statements about one warpgroup, and making them CTA-wide is exactly
    // what turns eight warps into one stream.  Resource 0 stays __syncthreads() in the
    // prologue and the epilogue.
    const int w = khalf;
    const int ltid = tid & 127;
    const int wbar = 1 + w;
    const uint32_t sMy = (uint32_t)(w ? TT::S1 : TT::S);
    // Two addresses for the same 32 columns, because a tcgen05.st addresses this thread's
    // lane slice while the MMA addresses the tile without one - the convention ba1 pinned
    // and the one PTMEM already uses.
    const uint32_t pMySt = pDst + (uint32_t)(32 * w);
    const uint32_t pMyMma = pTmem + (uint32_t)(32 * w);
    const uint32_t sfkSt = TT::sfk_slot(tlane + TT::SFB_K, w);
    const uint32_t sfkMma = TT::sfk_slot(tmem + TT::SFB_K, w);
    const uint32_t sfvSt = TT::sfk_slot(tlane + TT::SFB_V, w);
    const uint32_t sfvMma = TT::sfk_slot(tmem + TT::SFB_V, w);
    const uint32_t sKw = sK_u + (uint32_t)(w * L::GSTAGE_K * BN * D);
    const uint32_t sVw = sV_u + (uint32_t)(w * L::GSTAGE_V * BN * D);
    const uint32_t sSfKw = sSfK_u + (uint32_t)(w * L::GSTAGE_K * 512 * DCH);
    const uint32_t sSfVw = sSfV_u + (uint32_t)(w * L::GSTAGE_V * 512 * DCH);
    const uint32_t hsMs_u = sB_u;
    const uint32_t hsDen_u = sB_u + (uint32_t)(2 * BM * 4);
    const uint32_t finMs_u = sB_u + (uint32_t)(6 * BM * 4);
    const uint32_t finDen_u = sB_u + (uint32_t)(7 * BM * 4);
    uint64_t* msBar = &mbar_ms[1 - w];
    uint64_t* denBar = &mbar_den[2 * (1 - w)];  // + this publish's parity, at the wait
    uint64_t* pvBarOwn = &mbar_pv_g[w];
    uint64_t* pvBarOther = &mbar_pv_g[1 - w];
    uint32_t qkPhase = 0;    // parity of the next commit on mbar_qk_g[w] to wait
    uint32_t pvOwnIdx = 0;   // index of the next of MY PVs still to wait
    uint32_t pvOtherIdx = 0; // index of the next of the OTHER warpgroup's PVs to wait
    uint32_t msWait = 0;     // how many mstate publishes it has consumed
    uint32_t denWait = 0;    // how many den publishes it has consumed

    // Publish the prologue's two shared operands to the whole CTA, once per CTA rather
    // than once per group, because below this point every rendezvous is warpgroup-scoped
    // and a warpgroup that owns no groups executes none of them:
    //   * the Q tile was staged by all 256 threads through the generic proxy, but only a
    //     warpgroup that owns a group issues an MMA that reads it - and on a span of one
    //     group warpgroup 1 owns none, so it would never run the `fence.proxy.async` that
    //     publishes the half of the tile it wrote;
    //   * SFA_Q and SFA_P are TMEM columns written by warps 0-3 alone and read by BOTH
    //     warpgroups' MMAs, so warps 4-7 need a `fence::before_thread_sync` on the
    //     writers' side of a CTA-wide barrier, plus the `fence::after_thread_sync` their
    //     own MMA issue already carries.  PIPE gets both for free from its prime's
    //     `__syncthreads()`, which AG's per-warpgroup prime does not execute CTA-wide.
    if (any_compute) {
      fence_async_shared();
      tcg_fence_before();
      __syncthreads();
    }
    if (any_compute && (cLo + w <= cHi)) {
      // Prime this warpgroup's pipeline: group g0's K AND V, then group g0+2's K.  Two
      // commit groups are always created, even when the second stage is skipped, because
      // `cp.async.wait_group 1` proves everything except the most recent commit and an
      // empty commit group is still a commit group - the trick PIPE's prime uses too.
      const int g0 = cLo + w;
      tcg_issue_stage<D, PASS, 128, false>(p, ltid, 0, 0, g0, sKw, sVw, sSfKw, sSfVw, k_base,
                                           v_base, sfk_slice, v_atom_base, tma);
      cp_async_commit();
      if (g0 + 2 <= cHi) {
        tcg_issue_stage<D, PASS, 128, false, true, false>(p, ltid, 1, 0, g0 + 2, sKw, sVw,
                                                          sSfKw, sSfVw, k_base, v_base,
                                                          sfk_slice, v_atom_base, tma);
      }
      cp_async_commit();
      cp_async_wait<1>();
      fence_async_shared();  // staged K/V become UMMA operands
      wg_barrier(wbar, 128);
      tcg_pipe_sf<DCH, false, true>(tlane, warp, tid & 31, sSfKw, 0u, sfkSt, 0u);
      tcg_fence_before();
      wg_barrier(wbar, 128);
      if (ltid == 0) {
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, sMy, sQ_u, sKw, tmem + TT::SFA_Q, sfkMma);
        tcg_commit(smem_addr(&mbar_qk_g[w]));
      }
    }
    for (int grp = cLo + w, kj = 0; grp <= cHi; grp += 2, ++kj) {
      const int ksN = (kj + 1) % L::GSTAGE_K;  // K(grp+2): proven below, issued at step 5
      const int ksF = (kj + 2) % L::GSTAGE_K;  // K(grp+4): filling now
      const int vs = kj % L::GSTAGE_V;         // V(grp): consumed by this iteration's PV
      const int vsN = (kj + 1) % L::GSTAGE_V;  // V(grp+2): filled after the PV wait
      const int keyBase = grp << 7;

      // 1. K(grp+4) into the slot QK(grp-2) emptied.  That QK was issued two iterations
      //    ago and waited at the top of the previous one, so this needs no barrier: it is
      //    program order inside one warpgroup.
      if (grp + 4 <= cHi) {
        tcg_issue_stage<D, PASS, 128, false, true, false>(p, ltid, ksF, 0, grp + 4, sKw, sVw,
                                                          sSfKw, sSfVw, k_base, v_base,
                                                          sfk_slice, v_atom_base, tma);
      }
      cp_async_commit();
      // 2. One commit group may stay outstanding and it is the one just made, so
      //    K(grp+2) from the previous iteration and V(grp) from the one before have both
      //    landed - the two tiles this iteration's MMA issues read.
      cp_async_wait<1>();
      fence_async_shared();
      wg_barrier(wbar, 128);

      // 3. Retire this warpgroup's OWN previous PV, which is what frees its probability
      //    tile for step 8.  Waiting only for grp-2 here and for grp-1 at step 9 is the
      //    point of the split: the other warpgroup's PV(grp-1) is in flight right now, and
      //    this warpgroup's readout and whole exponential pass overlap it instead of
      //    waiting for it.  On its own barrier this is exactly one phase per iteration, so
      //    the parity tracks it: the count of completed commits here is at most kj, and
      //    the wait is for index kj-1.
      if (kj >= 1) {
        GS_WAIT(pvBarOwn, pvOwnIdx & 1u, 4, grp);
        ++pvOwnIdx;
        tcg_fence_after();
      }

      // 4. This group's scores, issued a whole iteration ago (or by the prime), so the
      //    wait is normally already satisfied and the tensor pipe has been busy during
      //    the staging, the atom writes and the other warpgroup's exponentials.
      GS_WAIT(&mbar_qk_g[w], qkPhase, 1, grp);
      qkPhase ^= 1u;
      tcg_fence_after();

      // ---- ONE readout of the whole 128-key score tile, and its maximum.  The mask
      // stays a template parameter: pH27 counted 64 FSEL + 64 ISETP + 32 UIADD3 per
      // 32-key chunk of dead mask code on every interior group.
      const bool needMask =
          ((unsigned)keyBase < (unsigned)lo) || ((unsigned)(keyBase + BN - 1) > (unsigned)hi);
      const bool anyMask = __any_sync(0xffffffffu, needMask);
      float scores[BN];
      const float bmRaw =
          anyMask ? tcg_scores<BN, true, true>(tlane, sMy, 0, keyBase, lo, hi, scores)
                  : tcg_scores<BN, false, true>(tlane, sMy, 0, keyBase, lo, hi, scores);

      // 5. Every warp of THIS warpgroup has retired its tcgen05.ld, so the score tile is
      //    free and the next QK can go straight into it.  128 threads arrive here, not
      //    256: the other warpgroup is somewhere inside its exponential pass and is not
      //    waited for, which is the entire difference from PIPE's tcg_combine_halves.
      wg_barrier(wbar, 128);
      if (grp + 2 <= cHi) {
        // K atoms for grp+2, written AFTER the wait above rather than before it as PIPE
        // does.  The slot they occupy is the one QK(grp) has just finished reading, and
        // that ordering is what makes one slot per warpgroup sufficient - 48 columns of
        // scale factors instead of 80, which is exactly the 32 columns the second
        // probability tile needs to close the map on 512.
        tcg_pipe_sf<DCH, false, true>(tlane, warp, tid & 31,
                                      sSfKw + (uint32_t)(ksN * 512 * DCH), 0u, sfkSt, 0u);
        tcg_fence_before();
        wg_barrier(wbar, 128);
        if (ltid == 0) {
          tcg_fence_after();
          tcg_pipe_qk<D, BM, BN>(tmem, sMy, sQ_u, sKw + (uint32_t)(ksN * BN * D),
                                 tmem + TT::SFA_Q, sfkMma);
          tcg_commit(smem_addr(&mbar_qk_g[w]));
        }
      }

      // ---- delayed maximum rescale, exactly reference.py's recurrence, with the state
      // arriving from the warpgroup that owns the previous group.  Only mstate crosses
      // here: den does not gate any exponential, so it stays behind the pass below.
      const float bm = bmRaw * scale;
      float prevMs = -INFINITY;
      if (grp != cLo) {
        // Strictly consecutive phases per barrier, so a toggling parity is sound here and
        // the counter doubles as the index of the publish being consumed.
        GS_WAIT(msBar, msWait & 1u, 2, grp);
        prevMs = lds32f(hsMs_u + (uint32_t)(((1 - w) * BM + m) * 4));
        ++msWait;
      }
      const bool upd = (bm - prevMs) > RESCALE_THRESHOLD;
      const float mn = upd ? bm : prevMs;
      const float safe = isfinite(mn) ? mn : 0.0f;
      const float alpha = isfinite(prevMs) ? expf(prevMs - safe) : 0.0f;
      mstate = mn;
      sts32f(hsMs_u + (uint32_t)((w * BM + m) * 4), mn);
      mbar_arrive(&mbar_ms[w]);

      // ---- probabilities from the registers step 4 already loaded: E4M3 QDQ with this
      // head dimension's bias, the unrounded exponential into the denominator and the
      // quantized one into the tile, in ascending key order, all unchanged from the form
      // this candidate already ships.  khalf is 0 for the tile offset because one
      // warpgroup owns all 128 keys here.
      const uint64_t scale2 = f32x2_pack(scale, scale);
      const uint64_t nsafe2 = f32x2_pack(-safe, -safe);
      const uint64_t bias2 = f32x2_pack(bias, bias);
      const uint64_t den2 =
          tcg_probabilities_pre<BN, kBiasLog2, kDenAcc, PTMEM>(scores, pMySt, m, 0, scale2,
                                                               nsafe2, bias2);
      const float2 dend = f32x2_unpack(den2);
      float prevDen = 0.0f;
      if (grp != cLo) {
        // Read the producer's OWN parity slot: denWait is both the index of the publish
        // being consumed and the producing warpgroup's iteration index, because the
        // consumer's n-th wait is always the producer's n-th publish.
        //
        // The barrier is indexed by the producer's parity too, so this wait is safe by
        // the CHAIN alone and not by a timing margin.  An mbarrier parity is not a
        // ticket - a wait for parity p is answered by ANY completed phase of that parity
        // - so a barrier that advances twice between one consumer's consecutive waits is
        // unwaitable, which is exactly what bm2 caught on the shared PV barrier.  Here
        // the producer's publish #j+2 lands on this same barrier only at its iteration
        // j+2, whose step-7 mstate wait needs the consumer's publish #j+1, which the
        // consumer makes at its iteration j+1 step 7 - strictly after its iteration j
        // step 8, i.e. after this very wait.  So at most one phase of this barrier can be
        // in flight past the one being waited, and parity (denWait >> 1) & 1 is exact.
        const uint32_t src = denWait & 1u;
        GS_WAIT(denBar + src, (denWait >> 1) & 1u, 3, grp);
        prevDen = lds32f(hsDen_u + (uint32_t)((((1 - w) * 2 + (int)src) * BM + m) * 4));
        ++denWait;
      }
      den = prevDen * alpha + (dend.x + dend.y);
      sts32f(hsDen_u + (uint32_t)(((w * 2 + (kj & 1)) * BM + m) * 4), den);
      mbar_arrive(&mbar_den[2 * w + (kj & 1)]);

      // 9. Retire the group BEFORE this one, i.e. the other warpgroup's PV, then rescale
      //    the shared accumulator and issue this group's.  One phase per iteration on the
      //    other warpgroup's own barrier, and it cannot be more than one ahead: it issues
      //    its next PV only after waiting for THIS group's, which is issued at step 10
      //    below, so at this point it has completed at most the one being waited for.
      if (grp != cLo) {
        GS_WAIT(pvBarOther, pvOtherIdx & 1u, 5, grp);
        ++pvOtherIdx;
        tcg_fence_after();
        // All 128 output channels, because one warpgroup owns the whole group here where
        // PIPE's owns 64.  ab2 measured the 4*ln(2) threshold firing ~1e-6 per row per
        // group on N(0,1) scaled scores, so the vote is almost always false and the two
        // extra tcgen05.ld/st pairs almost never execute.
        if (__any_sync(0xffffffffu, alpha != 1.0f)) {
#pragma unroll
          for (int c = 0; c < D / 32; ++c) {
            float ov[32];
            const uint32_t addr = tlane + TT::O + (uint32_t)(c * 32);
            tcg_ld32(addr, ov);
#pragma unroll
            for (int j = 0; j < 32; ++j) ov[j] *= alpha;
            tcg_st32(addr, ov);
          }
          tcg_wait_st();
        }
      }
      // V atoms for this group, written after that wait: PV(grp-2) has completed, which
      // is what frees both the tile and the slot.
      tcg_pipe_sf<DCH, true, true, false>(tlane, warp, tid & 31, 0u,
                                          sSfVw + (uint32_t)(vs * 512 * DCH), 0u, sfvSt);
      // Publish the probability tile.  In TMEM that is this thread's own wait::st, which
      // also covers the atom stores above; no async-proxy fence is needed because nothing
      // crossed into shared memory for the UMMA to read.
      tcg_wait_st();
      tcg_fence_before();
      wg_barrier(wbar, 128);
      if (ltid == 0) {
        tcg_fence_after();
        tcg_pipe_pv<D, BM, BN, PTMEM>(tmem, TT::O, sP_u, pMyMma,
                                      sVw + (uint32_t)(vs * BN * D), tmem + TT::SFA_P, sfvMma,
                                      (grp != cLo) ? 1u : 0u);
        tcg_commit(smem_addr(&mbar_pv_g[w]));
      }

      // 10. V(grp+2), issued last so the fill overlaps this warpgroup's own PV as well as
      //     the other warpgroup's next readout.  Slot vsN held V(grp-2), freed by step 3.
      if (grp + 2 <= cHi) {
        tcg_issue_stage<D, PASS, 128, false, false, true>(p, ltid, 0, vsN, grp + 2, sKw, sVw,
                                                          sSfKw, sSfVw, k_base, v_base,
                                                          sfk_slice, v_atom_base, tma);
      }
      cp_async_commit();
    }
    if (any_compute) {
      // Each warpgroup retires its OWN last PV - one phase on its own barrier, index
      // pvOwnIdx, with at most one commit outstanding, so the parity is exact - and the
      // CTA-wide barrier below is then what proves the OTHER warpgroup's last PV is also
      // done.  Neither warpgroup has to wait on a commit index it does not own, which is
      // the shape that bm2 showed a parity wait cannot express.
      if (cLo + w <= cHi) GS_WAIT(pvBarOwn, pvOwnIdx & 1u, 6, cHi);
      tcg_fence_after();
      __syncthreads();
      // The final state belongs to whichever warpgroup owns cHi; the other one holds the
      // state as of cHi-1 and needs it for 1/den, for LSE and for nothing else.  One
      // row-block each way through the scratch the loop's handoff already reserved.
      if (w == ((cHi - cLo) & 1)) {
        sts32f(finMs_u + (uint32_t)(m * 4), mstate);
        sts32f(finDen_u + (uint32_t)(m * 4), den);
      }
      __syncthreads();
      mstate = lds32f(finMs_u + (uint32_t)(m * 4));
      den = lds32f(finDen_u + (uint32_t)(m * 4));
    }
  } else if (TT::PIPE) {
    // =================== software-pipelined group loop =========================
    // Invariant at the top of iteration `grp`: QK(grp) is on the tensor pipe with
    // its commit pending on mbar_qk, PV(grp-1) is on the pipe with its commit
    // pending on mbar_pv, and the shared-memory ring holds groups grp..grp+2.
    // Both MMAs of a group are therefore issued an iteration before they are
    // waited on.  pI14's ablation priced the two exposed waits at 19.5% of shape
    // 7's runtime while the tensor pipe itself is only 12.1% busy, i.e. the pipe
    // time is not throughput, it is latency the sequential form makes every warp
    // spin on; issuing a group ahead lets the softmax of the other group fill it.
    //
    // Buffer lifetimes over the four ring stages:
    //   * step 1 fills buf(grp+2), which is the buffer group grp-2 used: its K was
    //     consumed by QK(grp-2), waited two iterations ago, and its V by
    //     PV(grp-2), waited at step 3 of the previous iteration.
    //   * buf(grp+1) holds the K that step 5 issues QK(grp+1) against and is not
    //     refilled until iteration grp+3, by which time step 4 has waited on it.
    //   * the single probability tile is rewritten at step 7 only after step 3
    //     proved PV(grp-1) finished reading it.
    //   * the scale atoms alternate by group parity, so step 2 never overwrites a
    //     slot an in-flight MMA is reading: SFB_K[par^1] was last used by
    //     QK(grp-1) and SFB_V[par] by PV(grp-2), both already waited.
    //   * the two score tiles alternate the same way, so step 5's QK(grp+1) writes
    //     the tile this iteration's pass 1 and pass 2 are no longer reading.
    uint32_t qkPhase = 0, pvPhase = 0;
    // Parity of the current use of each TMA stage barrier.  A stage is reused
    // every NSTAGE iterations, so the parity flips exactly when the ring wraps.
    uint32_t tmaPar = 0;
    if (any_compute) {
      // Prime the ring with two groups and put the first QK on the pipe.
#pragma unroll
      for (int s = 0; s < 2; ++s) {
#if FMHA_PRIME_HOIST
        // pr1-P2: slot 0 issued in the prologue; the empty commit group keeps
        // the wait<N> accounting (the PIPE prime's own idiom).
        if (s > 0 && cLo + s <= cHi) {
#else
        if (cLo + s <= cHi) {
#endif
          tcg_issue_stage<D, PASS, THR, TMA>(p, tid, s, s, cLo + s, sK_u, sV_u, sSfK_u, sSfV_u,
                                             k_base, v_base, sfk_slice, v_atom_base, tma);
        }
        if constexpr (!SF_BULK) cp_async_commit();
      }
      if constexpr (!SF_BULK) cp_async_wait<1>();
      // On the bulk path this barrier proves group cLo's scale atoms as well as its
      // K tile, and `tcg_pipe_sf` below reads those atoms, so every thread waits
      // here instead of leaving it to tid 0 at the MMA issue.  The cp.async form
      // keeps its split: `wait_group 1` proved the atoms and only the tile needs
      // the barrier.
      if constexpr (SF_BULK && TMA) mbar_wait(&mbar_stage[0], 0u);
      fence_async_shared();
      __syncthreads();
      tcg_pipe_sf<DCH, false>(tlane, warp, tid & 31, sSfK_u, 0u,
                              TT::sfk_slot(tlane + TT::SFB_K, cLo & 1), 0u);
      tcg_fence_before();
      __syncthreads();
      if (tid == 0) {
        if constexpr (TMA && !SF_BULK) mbar_wait(&mbar_stage[0], 0u);  // group cLo's K
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, (cLo & 1) ? TT::S1 : TT::S, sQ_u, sK_u, tmem + TT::SFA_Q,
                               TT::sfk_slot(tmem + TT::SFB_K, cLo & 1));
        tcg_commit(smem_addr(&mbar_qk));
      }
    }
    int cur = 0;  // ring buffer holding this group's K and V
    for (int grp = cLo; grp <= cHi; ++grp) {
      const int par = grp & 1;
      const int keyBase = grp << 7;
      const int nxt = (cur + 1) & (L::NSTAGE - 1);
      const int fill = (cur + 2) & (L::NSTAGE - 1);

      // 1. Fill the ring two groups ahead; waiting on one outstanding commit
      //    group then proves groups grp and grp+1 have both landed.
      if (grp + 2 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA>(p, tid, fill, fill, grp + 2, sK_u, sV_u, sSfK_u,
                                           sSfV_u, k_base, v_base, sfk_slice, v_atom_base, tma);
      }
      if constexpr (!SF_BULK) cp_async_commit();
      if constexpr (!SF_BULK) cp_async_wait<1>();
      // On the bulk path an atom block shares the barrier of the tile it travels
      // with, and step 2 below reads TWO slots' atoms: slot `nxt`'s K atoms, for the
      // group step 5 is about to issue, and slot `cur`'s V atoms, for this group's
      // PV.  So both barriers have to be proven here, by every thread, instead of at
      // the two MMA issues that own them on the cp.async path - where
      // `wait_group 1` proved the atoms and only the tiles needed a barrier.  Both
      // payloads were staged one and two iterations back, i.e. one and two whole
      // softmax passes ago, so in steady state both waits are already satisfied;
      // what moves is who waits and when, and the parities are unchanged because
      // `tmaPar` is not flipped until the end of the iteration.
      if constexpr (SF_BULK && TMA) {
        // Slot `nxt` was armed by the previous iteration's step 1, which is guarded
        // by `grp + 1 <= cHi` exactly as the MMA issue below is: on the last
        // iteration nothing was staged into it, so waiting would block on a barrier
        // that was never armed.  Step 2 still reads that slot's atoms, but it reads
        // stale shared memory that no MMA will consume, which is what the cp.async
        // path has always done here.  Slot `cur` needs no such guard: group grp was
        // staged two iterations ago, or by the prime for the first two groups.
        if (grp + 1 <= cHi) mbar_wait(&mbar_stage[nxt], tmaPar ^ ((nxt == 0) ? 1u : 0u));
        mbar_wait(&mbar_stage[cur], tmaPar);
      }
      fence_async_shared();  // staged K/V become UMMA operands
      __syncthreads();

      // 2. Scale atoms: K for the group step 5 issues, V for this group's PV.
      tcg_pipe_sf<DCH, true>(tlane, warp, tid & 31, sSfK_u + (uint32_t)(nxt * 512 * DCH),
                             sSfV_u + (uint32_t)(cur * 512 * DCH),
                             TT::sfk_slot(tlane + TT::SFB_K, par ^ 1),
                             TT::sfk_slot(tlane + TT::SFB_V, par));
      tcg_fence_before();
      __syncthreads();

      // 3. Retire the previous group's PV.  After this its probability tile and
      //    its V buffer are free and the accumulator may safely be rescaled.
      if (grp != cLo) {
        mbar_wait(&mbar_pv, pvPhase);
        pvPhase ^= 1u;
        tcg_fence_after();
      }

      // 4. This group's scores are ready; step 5 of the previous iteration
      //    issued them, so the wait is normally already satisfied.
      mbar_wait(&mbar_qk, qkPhase);
      qkPhase ^= 1u;
      tcg_fence_after();

      // ---- pass 1: group maximum over the masked, scaled scores
      const int keyLo = keyBase + khalf * KHALF;
      const bool needMask =
          ((unsigned)keyLo < (unsigned)lo) || ((unsigned)(keyLo + KHALF - 1) > (unsigned)hi);
      const bool anyMask = __any_sync(0xffffffffu, needMask);
      const uint32_t s_col = par ? TT::S1 : TT::S;
      float scores[KHALF];
      float bmRaw =
          anyMask ? tcg_scores<KHALF, true, true>(tlane, s_col, khalf, keyLo, lo, hi, scores)
                  : tcg_scores<KHALF, false, true>(tlane, s_col, khalf, keyLo, lo, hi, scores);
      // The group maximum spans every warpgroup's key slice; `max` is exact, so
      // the combine is bit-identical to a single 128-wide reduction.
      bmRaw = tcg_combine_halves<KSPLIT, BM, true>(sB_u, khalf, m, bmRaw);
      // The softmax scale is positive, so scaling commutes with the maximum.
      const float bm = bmRaw * scale;

      // 5. Put the next group's QK on the pipe, into the other score tile.  Its
      //    whole latency and pipe time now overlaps this group's pass 2.
      if (grp + 1 <= cHi && tid == 0) {
        // QK(grp+1) reads the ring slot two groups ahead of the one just filled.
        // With the atoms on the bulk path every thread already waited on this
        // barrier before tcg_pipe_sf read that slot's atoms, so tid 0 does not
        // repeat it.
        if constexpr (TMA && !SF_BULK) mbar_wait(&mbar_stage[nxt], tmaPar ^ ((nxt == 0) ? 1u : 0u));
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, (par ^ 1) ? TT::S1 : TT::S, sQ_u,
                               sK_u + (uint32_t)(nxt * BN * D), tmem + TT::SFA_Q,
                               TT::sfk_slot(tmem + TT::SFB_K, par ^ 1));
        tcg_commit(smem_addr(&mbar_qk));
      }

      // ---- delayed maximum rescale, exactly reference.py's recurrence
      const bool upd = (bm - mstate) > RESCALE_THRESHOLD;
      const float mn = upd ? bm : mstate;
      const float safe = isfinite(mn) ? mn : 0.0f;
      const float alpha = isfinite(mstate) ? expf(mstate - safe) : 0.0f;
      mstate = mn;
      den *= alpha;
      // O lives in TMEM, so the rescale is a load/multiply/store round trip;
      // tcgen05.ld/st are warp-wide, hence the vote, and lanes whose alpha is
      // exactly 1.0f multiply by 1.0, which is bit-identical to skipping them.
      // The first group is skipped outright: its PV overwrites the accumulator.
      if (grp != cLo && __any_sync(0xffffffffu, alpha != 1.0f)) {
#pragma unroll
        for (int c = 0; c < OCOLS / 32; ++c) {
          float ov[32];
          const uint32_t addr = tlane + TT::O + (uint32_t)(khalf * OCOLS + c * 32);
          tcg_ld32(addr, ov);
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= alpha;
          tcg_st32(addr, ov);
        }
        tcg_wait_st();
      }

      // ---- pass 2: probabilities from the registers pass 1 already loaded
      const uint64_t scale2 = f32x2_pack(scale, scale);
      const uint64_t nsafe2 = f32x2_pack(-safe, -safe);
      const uint64_t bias2 = f32x2_pack(bias, bias);
      const uint64_t den2 =
          tcg_probabilities_pre<KHALF, kBiasLog2, kDenAcc, PTMEM>(scores, pDst, m, khalf, scale2,
                                                                  nsafe2, bias2);
      {
        const float2 dend = f32x2_unpack(den2);
        den += dend.x + dend.y;
      }

      // 6. Publish the probability tile and issue this group's PV.  Its wait is
      //    deferred to step 3 of the next iteration, which is what lets the pipe
      //    stay busy across the boundary.
      // A TMEM-resident tile is published by its own `tcgen05.wait::st` plus the
      // fence/barrier pair that already follows: no async-proxy fence is needed,
      // because nothing crossed from the generic proxy into shared memory for the
      // UMMA to read.  Step 3 of the next iteration still retires PV(grp-1) before
      // step 7 rewrites the tile, so the lifetime is the one the ring's comment
      // documents, in TMEM exactly as in shared memory.
      if constexpr (PTMEM) {
        tcg_wait_st();
      } else {
        fence_async_shared();
      }
      tcg_fence_before();
      __syncthreads();
      if (tid == 0) {
        // PV(grp) reads this group's V, staged two iterations ago.  With the atoms
        // on the bulk path every thread already proved this slot above, before step
        // 2 read its V atoms, so tid 0 does not repeat it.
        if constexpr (TMA && !SF_BULK) mbar_wait(&mbar_stage[cur], tmaPar);
        tcg_fence_after();
        tcg_pipe_pv<D, BM, BN, PTMEM>(tmem, TT::O, sP_u, pTmem, sV_u + (uint32_t)(cur * BN * D),
                                      tmem + TT::SFA_P, TT::sfk_slot(tmem + TT::SFB_V, par),
                                      (grp != cLo) ? 1u : 0u);
        tcg_commit(smem_addr(&mbar_pv));
      }
      if (nxt == 0) tmaPar ^= 1u;
      cur = nxt;
    }
    if (any_compute) {
      // The last group's PV is still on the pipe; the epilogue reads O from TMEM.
      mbar_wait(&mbar_pv, pvPhase);
      pvPhase ^= 1u;
      tcg_fence_after();
      __syncthreads();
    }
  } else if (TT::PIPE_GM && !QK64 && (cHi - cLo) >= FMHA_PIPE_GM_MIN_GROUPS) {
    // gmq1: the pipelined GM body has no 64-column score form (its issue path is
    // the 128-wide tcg_pipe_qk helper), so the GM-QK64 instantiation always runs
    // the sequential body below — which is where its QK64 branch lives and which
    // is production-proven for GM at spans < 7.  The constexpr guard also deletes
    // this whole body from the GM-QK64 binary, which is what keeps its register
    // count in the 3-CTA budget.  The trade (pI19's -3.8% intra-CTA pipelining
    // prize vs three-way interleave) is priced in the gmq1 prereg.
    // Only for a span long enough to amortise the priming: pI19 measured the
    // pipelined form at -3.8% on shape 9 (33 groups per CTA) and -1.1% on shape 8
    // (9 groups) but +2.0% on domain d7, whose nsplit=64 leaves each CTA only 4
    // groups, and shape 2 (4 groups) and shape 3 (2 groups) were inside noise.
    // Both loops stay compiled for this instantiation - the group-maximum pass
    // uses 159 of its 256 available registers, so there is room - and the span is
    // uniform across the CTA, so the branch costs nothing per group.
    //
    // THE 7 IS A WARM-PROTOCOL NUMBER AND ITS PROVENANCE SHOULD BE READ BEFORE
    // TRUSTING IT.  pI19 predates both TIMING_PROTOCOL_V2 and `agx1`: the comment
    // above dates d7 at "nsplit=64", while `agx1` later raised this pass's grid from
    // 64 to 148 CTAs and the current tree gives d7 7 groups per CTA and s3 8, so two
    // of the five data points that fixed the threshold describe geometries this tree
    // no longer produces.  Under v2 the L2 is evicted before every forward, so this
    // pass reads K COLD from DRAM, and §83 measured it at **+29.6% (s8), +40.7% (s2)
    // and +65.5% (s5)** against +1.4…+6.6% for the split pass and -11% for the
    // combines.  §94 then profiled it at **11.83% occupancy, DRAM 14.70%, 22.81
    // warp-cycles per issued instruction** - latency-bound on every axis, with the
    // DRAM path five-sevenths idle.  A threshold fitted where that read hit a warm L2
    // is not obviously the threshold for a cold one, and §102(d) used this gate to
    // declare D=256 pipelining inert, which is circular when the gate is the question.
    // ============= software-pipelined group-maximum loop (D=128) ==============
    // One score tile is enough here because pass 1 is its only reader: the barrier
    // after the maximum proves every warp has retired its tcgen05.ld.red, so the
    // next group's QK can go straight onto the tensor pipe and its ~564 cycles of
    // pipe time plus completion latency run under the following iteration's
    // staging, scale-atom writes and barriers instead of stalling all eight warps.
    // The ring is three stages deep so that cp.async.wait_group 1 proves group
    // g+1's K has landed before it is issued against, and the K scale atoms
    // alternate by group parity (into the columns SFA_P would occupy - this pass
    // issues no PV, so it never writes them) so step 2 never overwrites a slot an
    // in-flight MMA is reading.
    uint32_t qkPhase = 0;
    // Parity of the current use of each TMA stage barrier.  A stage is reused
    // every NSTAGE iterations, so the parity flips exactly when the ring wraps.
    uint32_t tmaPar = 0;
    if (any_compute) {
#pragma unroll
      for (int s = 0; s < 2; ++s) {
#if FMHA_PRIME_HOIST
        // pr1-P2: slot 0 issued in the prologue; the empty commit group keeps
        // the wait<N> accounting (the PIPE prime's own idiom).
        if (s > 0 && cLo + s <= cHi) {
#else
        if (cLo + s <= cHi) {
#endif
          tcg_issue_stage<D, PASS, THR, TMA>(p, tid, s, s, cLo + s, sK_u, sV_u, sSfK_u, sSfV_u,
                                             k_base, v_base, sfk_slice, v_atom_base, tma);
        }
        if constexpr (!SF_BULK) cp_async_commit();
      }
      if constexpr (!SF_BULK) cp_async_wait<1>();
      // On the bulk path this barrier proves group cLo's scale atoms as well as its
      // K tile, and `tcg_pipe_sf` below reads those atoms, so every thread waits
      // here instead of leaving it to tid 0 at the MMA issue.  The cp.async form
      // keeps its split: `wait_group 1` proved the atoms and only the tile needs
      // the barrier.
      if constexpr (SF_BULK && TMA) mbar_wait(&mbar_stage[0], 0u);
      fence_async_shared();
      __syncthreads();
      tcg_pipe_sf<DCH, false>(tlane, warp, tid & 31, sSfK_u, 0u,
                              TT::sfk_slot(tlane + TT::SFB_K, cLo & 1), 0u);
      tcg_fence_before();
      __syncthreads();
      if (tid == 0) {
        if constexpr (TMA && !SF_BULK) mbar_wait(&mbar_stage[0], 0u);  // group cLo's K
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, TT::S, sQ_u, sK_u, tmem + TT::SFA_Q,
                               TT::sfk_slot(tmem + TT::SFB_K, cLo & 1));
        tcg_commit(smem_addr(&mbar_qk));
      }
    }
    int cur = 0;  // ring buffer holding this group's K
    for (int grp = cLo; grp <= cHi; ++grp) {
      const int par = grp & 1;
      const int keyBase = grp << 7;
      const int nxt = (cur + 1) % L::NSTAGE;
      const int fill = (cur + 2) % L::NSTAGE;

      if (grp + 2 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA>(p, tid, fill, fill, grp + 2, sK_u, sV_u, sSfK_u,
                                           sSfV_u, k_base, v_base, sfk_slice, v_atom_base, tma);
      }
      if constexpr (!SF_BULK) cp_async_commit();
      if constexpr (!SF_BULK) cp_async_wait<1>();
      // Group grp+1's atoms live in ring slot `nxt` and tcg_pipe_sf below reads
      // them.  The cp.async form proved them with `wait_group 1` and left the tile
      // barrier to tid 0 at the MMA issue; on the bulk path one barrier covers
      // both, so it moves up here and every thread waits on it.  It was armed a
      // whole iteration ago, when group grp+1 was staged, so the wait should be
      // close to free - and the last group issues no further QK and has nothing
      // armed to wait for, hence the same bound the MMA issue below uses.
      if constexpr (SF_BULK && TMA) {
        if (grp + 1 <= cHi) mbar_wait(&mbar_stage[nxt], tmaPar ^ ((nxt == 0) ? 1u : 0u));
      }
      fence_async_shared();
      __syncthreads();

      tcg_pipe_sf<DCH, false>(tlane, warp, tid & 31, sSfK_u + (uint32_t)(nxt * 512 * DCH), 0u,
                              TT::sfk_slot(tlane + TT::SFB_K, par ^ 1), 0u);
      tcg_fence_before();
      __syncthreads();

      mbar_wait(&mbar_qk, qkPhase);
      qkPhase ^= 1u;
      tcg_fence_after();

      // ---- the whole of this pass: the masked group maximum, then its store
      const int keyLo = keyBase + khalf * KHALF;
      const bool needMask =
          ((unsigned)keyLo < (unsigned)lo) || ((unsigned)(keyLo + KHALF - 1) > (unsigned)hi);
      const bool anyMask = __any_sync(0xffffffffu, needMask);
      // The group-maximum pass is the split path's repeated QK^T, so it is half
      // of what a padded shape pays for its dead rows and the same warp-uniform
      // skip applies.  `bmRaw` stays -INFINITY for a dead warp, which is what an
      // entirely masked row's maximum is anyway, and its `valid`-guarded store
      // never runs.
      float bmRaw = -INFINITY;
      if (rowsLive) {
        bmRaw = anyMask ? tcg_group_max<KHALF, true, false>(tlane, khalf, keyLo, lo, hi)
                        : tcg_group_max<KHALF, false, true>(tlane, khalf, keyLo, lo, hi);
      }
      if (KSPLIT > 1) {
        // The group maximum spans every warpgroup's key slice, so the partial
        // maxima meet in `oSb` exactly as they do in the sequential loop.  `max`
        // is exact and order independent, so the result is bit-identical to one
        // 128-wide reduction, and this barrier also proves that every warp has
        // retired its tcgen05.ld before the QK at the bottom of the iteration
        // overwrites the single score tile - which is why the KSPLIT==1 form's
        // separate barrier below becomes redundant rather than merely duplicated.
        bmRaw = tcg_combine_halves<KSPLIT, BM, true>(sB_u, khalf, m, bmRaw);
      }
      // The softmax scale is positive, so scaling commutes with the maximum.
      const float bm = bmRaw * scale;
      if (valid && khalf == 0) p.bm[(size_t)grp * p.rows_total + rowG] = bm;

      if (KSPLIT == 1) {
        // Every warp must have retired its score reads before the next group's QK
        // overwrites the single score tile.
        __syncthreads();
      }
      if (grp + 1 <= cHi && tid == 0) {
        // QK(grp+1) reads the ring slot two groups ahead of the one just filled.
        // With the atoms on the bulk path every thread already waited on this
        // barrier before tcg_pipe_sf read that slot's atoms, so tid 0 does not
        // repeat it.
        if constexpr (TMA && !SF_BULK) mbar_wait(&mbar_stage[nxt], tmaPar ^ ((nxt == 0) ? 1u : 0u));
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, TT::S, sQ_u, sK_u + (uint32_t)(nxt * BN * D),
                               tmem + TT::SFA_Q, TT::sfk_slot(tmem + TT::SFB_K, par ^ 1));
        tcg_commit(smem_addr(&mbar_qk));
      }
      if (nxt == 0) tmaPar ^= 1u;
      cur = nxt;
    }
  } else {
  if (any_compute) {
    if constexpr (SF_AHEAD) {
      // Prime for the atoms-ahead loop: group cLo's tiles on the tile ring, and THREE
      // groups of atoms on the deeper atom ring, all in one commit group.  The loop's
      // iteration for group g writes group g+1's atoms into TMEM above its staging
      // barrier, reading bytes that `wait_group 1` proved at iteration g-1 and that
      // iteration g-1's barrier then published; group cLo has no iteration -1, so its
      // atoms are written into TMEM here instead, exactly as the QK-lookahead prime
      // writes its own.  Slot(g) = (g - cLo) % NSTAGE_SF, so the three primes land in
      // slots 0, 1 and 2 and iteration 0 reads slot 1 - which no in-flight cp.async
      // targets, because iteration 0 fills slot (0+3)%3 = 0 and iteration 1 fills 1 only
      // after iteration 0's barrier has ordered this read.
#if !(FMHA_PRIME_HOIST && FMHA_PRIME_HOIST_SFA)
      if constexpr (MIDPIPE) {
        // qk6 split prime: K(cLo) and K(cLo+1) on the three-slot K ring's own
        // barriers, V(cLo) on the two-slot V ring's.  QK(cLo) issues from this
        // prime (the loop head only waits), so K(cLo) is proven below; K(cLo+1)
        // and V(cLo) stay in flight for iteration 0's mid waits, whose phase
        // formulas count them as slot arrivals #0.
        tcg_issue_stage<D, PASS, THR, TMA, true, false, false, true>(
            p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
            v_atom_base, tma);
        if (cLo + 1 <= cHi) {
          tcg_issue_stage<D, PASS, THR, TMA, true, false, false, true>(
              p, tid, 1, 0, cLo + 1, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
              v_atom_base, tma);
        }
        tcg_issue_stage<D, PASS, THR, TMA, false, true, false, true>(
            p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
            v_atom_base, tma);
      } else {
      tcg_issue_stage<D, PASS, THR, TMA, true, true, false>(p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u,
                                                            sSfV_u, k_base, v_base, sfk_slice,
                                                            v_atom_base, tma);
      }
      tcg_stage_atoms<D, PASS, THR>(p, tid, 0, 0, cLo, sSfK_u, sSfV_u, sfk_slice, v_atom_base);
      // Slots 1 .. NSTAGE_SF-1, i.e. groups cLo+1 .. cLo+NSTAGE_SF-1.  The loop's
      // iteration `iter` reads slot (iter+1) % NSTAGE_SF, so the first NSTAGE_SF-1
      // iterations all read a primed slot and only iteration NSTAGE_SF-1 reads one the
      // loop itself filled, at iteration 0.  One slot fewer here and iteration 1 would
      // read atoms the loop never staged.
#pragma unroll
      for (int s = 1; s < L::NSTAGE_SF; ++s) {
        if (cLo + s <= cHi) {
          tcg_stage_atoms<D, PASS, THR>(p, tid, s, s, cLo + s, sSfK_u, sSfV_u, sfk_slice,
                                        v_atom_base);
        }
      }
#else
      // sfa1: the tile and the three atom stages were issued in the prologue, above
      // FMHA_PROLOG_LDG's cold LDG batches and above tcg_alloc, so group cLo's K/V is in
      // flight while the Q staging runs and - on the split key - while the PDL scan wait
      // below is still pending.  This `#if` is the exact complement of the hoist's gate
      // (FMHA_PRIME_HOIST && FMHA_PRIME_HOIST_SFA, inside `if constexpr (SF_AHEAD)`), so
      // the issue happens exactly once, which is the same pairing pr1-P2 relies on for
      // the !SF_AHEAD arm below.  Everything from the commit down is unchanged: the
      // group boundary did not move, `wait<0>` still proves exactly the three atom
      // blocks, and `tcg_pipe_sf` still runs where `tlane` exists.
#endif
      // One commit group, so the loop's first `wait_group 1` still has exactly one older
      // group to prove and its accounting is unchanged.  Nothing else is outstanding at
      // this point, so waiting to zero is what proves all three atom blocks.
      cp_async_commit();
      cp_async_wait<0>();
      fence_async_shared();
      if constexpr (MIDPIPE) {
        // qk6: prove K(cLo) for the primed QK issue below.  V(cLo) and K(cLo+1)
        // stay in flight; iteration 0's mid waits find them on phase 0.
        mbar_wait(kbar_base, 0u);
        fence_async_shared();
      }
      // Publishes every thread's atom bytes to the four warps that write them to TMEM,
      // and group cLo+1's to iteration 0, which reads it before its own barrier.
      __syncthreads();
      tcg_pipe_sf<DCH, WANT_V>(tlane, warp, tid & 31, sSfK_u, sSfV_u,
                               TT::sfk_slot(tlane + TT::SFB_K, cLo & 1),
                               TT::sfk_slot(tlane + TT::SFB_V, MIDPIPE ? (cLo % 3) : (cLo & 1)));
      tcg_fence_before();
      // Orders the stores above before iteration 0's QK issue, which reads slot cLo&1.
      __syncthreads();
      if constexpr (MIDPIPE) {
        // qk6: the loop head only WAITS, so QK(cLo) issues here - the exact
        // instruction sequence the head would have issued, same descriptors,
        // same scale slots.
        if (tid == 0) {
          tcg_fence_after();
          tcg_pipe_qk<D, BM, BN>(tmem, TT::S, sQ_u, sK_u, tmem + TT::SFA_Q,
                                 tcg_sf_seq<SF_AHEAD, TT::SF_ALT>(tmem + TT::SFB_K, cLo));
          tcg_commit(smem_addr(&mbar_qk));
        }
      }
    } else {
#if FMHA_PRIME_HOIST
    // pr1-P2: slot 0 issued in the prologue — this arm is the !SF_AHEAD
    // branch, exactly matching the hoist's constexpr gate, so the issue
    // happens exactly once.  The commit stays (empty group; the wait<N>
    // accounting is unchanged).
    if constexpr (!SF_BULK) cp_async_commit();
#else
    if constexpr (DPV) {
      // qk1: split prime, matching the hoisted arm's contract exactly.
      tcg_issue_stage<D, PASS, THR, TMA, true, false, true, true>(
          p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice, v_atom_base,
          tma);
      tcg_issue_stage<D, PASS, THR, TMA, false, true, true, true>(
          p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base, sfk_slice, v_atom_base,
          tma);
    } else {
      tcg_issue_stage<D, PASS, THR, TMA>(p, tid, 0, 0, cLo, sK_u, sV_u, sSfK_u, sSfV_u, k_base,
                                         v_base, sfk_slice, v_atom_base, tma);
    }
    if constexpr (!SF_BULK) cp_async_commit();
#endif
    if constexpr (L::QK_AHEAD) {
      // QK(cLo) is issued from the prime rather than from the first iteration, so
      // its K and its scale bytes have to be proven here: stage group cLo+1's K as
      // well, then wait down to one outstanding group, which proves group cLo's
      // payload and leaves cLo+1's K in flight for the first iteration's wait.
      if (cLo + 1 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, true, false>(p, tid, 1, 0, cLo + 1, sK_u, sV_u, sSfK_u,
                                                        sSfV_u, k_base, v_base, sfk_slice,
                                                        v_atom_base, tma);
      }
      if constexpr (!SF_BULK) cp_async_commit();
      if constexpr (!SF_BULK) cp_async_wait<1>();
      fence_async_shared();
      __syncthreads();
      tcg_pipe_sf<DCH, false>(tlane, warp, tid & 31, sSfK_u, 0u,
                              TT::sfk_slot(tlane + TT::SFB_K, 0), 0u);
      tcg_fence_before();
      __syncthreads();
      if (tid == 0) {
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, TT::S, sQ_u, sK_u, tmem + TT::SFA_Q,
                               TT::sfk_slot(tmem + TT::SFB_K, 0));
        tcg_commit(smem_addr(&mbar_qk));
      }
    }
    }
  }

  // fs5t port: the per-thread prefix walk that replaces fmha_scan_flat for
  // this chunk.  Placed AFTER the staging prime and after the PDL scan wait
  // (fmha_pdl_scan_wait above covers whichever kernel produced p.bm - the
  // scan chain normally, the GM pass directly when fused - so the walk's
  // loads are ordered after GM's stores by griddepcontrol semantics).  The
  // walk's batched strided loads and the staged K/V DMAs are in flight
  // together, which is what keeps the absorbed scan off the critical path.
  // `fb` then carries bm[grp] through the loop in place of the replayed
  // mt_cur.  The recurrence is the scan kernel's FMHA_SCAN_STEP verbatim, so
  // the states are bit-identical to the ones the scan materialised.
  float fb = -INFINITY;
  if constexpr (FSCAN) {
    const int tw = min(cLo, p.T);
    {
      int t = 0;
      for (; t + 16 <= tw; t += 16) {
        const float b0 = __ldg(p.bm + (size_t)(t + 0) * p.rows_total + rowG);
        const float b1 = __ldg(p.bm + (size_t)(t + 1) * p.rows_total + rowG);
        const float b2 = __ldg(p.bm + (size_t)(t + 2) * p.rows_total + rowG);
        const float b3 = __ldg(p.bm + (size_t)(t + 3) * p.rows_total + rowG);
        const float b4 = __ldg(p.bm + (size_t)(t + 4) * p.rows_total + rowG);
        const float b5 = __ldg(p.bm + (size_t)(t + 5) * p.rows_total + rowG);
        const float b6 = __ldg(p.bm + (size_t)(t + 6) * p.rows_total + rowG);
        const float b7 = __ldg(p.bm + (size_t)(t + 7) * p.rows_total + rowG);
        const float b8 = __ldg(p.bm + (size_t)(t + 8) * p.rows_total + rowG);
        const float b9 = __ldg(p.bm + (size_t)(t + 9) * p.rows_total + rowG);
        const float b10 = __ldg(p.bm + (size_t)(t + 10) * p.rows_total + rowG);
        const float b11 = __ldg(p.bm + (size_t)(t + 11) * p.rows_total + rowG);
        const float b12 = __ldg(p.bm + (size_t)(t + 12) * p.rows_total + rowG);
        const float b13 = __ldg(p.bm + (size_t)(t + 13) * p.rows_total + rowG);
        const float b14 = __ldg(p.bm + (size_t)(t + 14) * p.rows_total + rowG);
        const float b15 = __ldg(p.bm + (size_t)(t + 15) * p.rows_total + rowG);
        if ((b0 - mstate) > RESCALE_THRESHOLD) mstate = b0;
        if ((b1 - mstate) > RESCALE_THRESHOLD) mstate = b1;
        if ((b2 - mstate) > RESCALE_THRESHOLD) mstate = b2;
        if ((b3 - mstate) > RESCALE_THRESHOLD) mstate = b3;
        if ((b4 - mstate) > RESCALE_THRESHOLD) mstate = b4;
        if ((b5 - mstate) > RESCALE_THRESHOLD) mstate = b5;
        if ((b6 - mstate) > RESCALE_THRESHOLD) mstate = b6;
        if ((b7 - mstate) > RESCALE_THRESHOLD) mstate = b7;
        if ((b8 - mstate) > RESCALE_THRESHOLD) mstate = b8;
        if ((b9 - mstate) > RESCALE_THRESHOLD) mstate = b9;
        if ((b10 - mstate) > RESCALE_THRESHOLD) mstate = b10;
        if ((b11 - mstate) > RESCALE_THRESHOLD) mstate = b11;
        if ((b12 - mstate) > RESCALE_THRESHOLD) mstate = b12;
        if ((b13 - mstate) > RESCALE_THRESHOLD) mstate = b13;
        if ((b14 - mstate) > RESCALE_THRESHOLD) mstate = b14;
        if ((b15 - mstate) > RESCALE_THRESHOLD) mstate = b15;
      }
      for (; t < tw; ++t) {
        const float b = __ldg(p.bm + (size_t)t * p.rows_total + rowG);
        if ((b - mstate) > RESCALE_THRESHOLD) mstate = b;
      }
    }
    if (cLo <= cHi) fb = __ldg(p.bm + (size_t)cLo * p.rows_total + rowG);
  }
  uint32_t qkPhase = 0, pvPhase = 0;
  // Parity of the current use of each TMA stage barrier; it flips when the
  // two-stage ring wraps, i.e. after every odd iteration.
  uint32_t tmaPar = 0;
  // qk1: the V ring's own parity counter (DPV forms only; dead elsewhere).  Its
  // slot sequence is identical to tmaPar's - one wait per iteration, flipping when
  // the two-slot ring wraps - but it is tracked apart because the wait moved.
  uint32_t vPar = 0;
  for (int grp = cLo; grp <= cHi; ++grp) {
    const int iter = grp - cLo;
    const int stage = iter & 1;
    // K's ring is one deeper than V's when QK runs a group ahead; they coincide for
    // every other form, where NSTAGE_K == NSTAGE == 2.
    const int kslot = L::QK_AHEAD ? (iter % L::NSTAGE_K) : stage;
    const int keyBase = grp << 7;
    (void)kslot;

    // Prefetch group grp+1's replayed maximum here, at the top of the body, so the
    // load has the whole group - K/V staging, the QK wait, the score readout - to
    // land before the recurrence at the bottom reads it.  The split pass is the one
    // form whose per-group `safe`/`alpha` come from memory rather than from a
    // reduction it just did, and they gate both the delayed-maximum rescale of the
    // accumulator and the probability exponent, so an unhidden load here sits on the
    // group's critical path at 11.5% achieved occupancy (`ap18`: s9's split launch
    // is `long_scoreboard` 40.9% at IPC 1.61; d3's is IPC 0.42).  One register, and
    // `bb1` freed 5-8 of them on exactly these forms.
    float mt_next = -INFINITY;
    // fs5t: the fused form prefetches bm[grp+1] where the replay form
    // prefetches mstate[grp+1] - one strided coalesced load either way,
    // consumed by the recurrence at the bottom of the iteration.
    float bnxt = -INFINITY;
    if constexpr (PASS == PASS_SPLIT) {
      if constexpr (FSCAN) {
        if (grp + 1 <= cHi) bnxt = __ldg(p.bm + (size_t)(grp + 1) * p.rows_total + rowG);
      } else if (grp + 1 <= cHi) {
        mt_next = p.mstate[(size_t)(grp + 1) * p.rows_total + rowG];
      }
    }

    if constexpr (L::QK_AHEAD) {
      // K two groups ahead, V one group ahead, in one commit group.  wait_group 1
      // below then proves exactly what this iteration consumes: K(grp+1) for the
      // QK issued at the end of it, and V(grp) for this group's PV.
      if (grp + 2 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, true, false>(
            p, tid, (iter + 2) % L::NSTAGE_K, 0, grp + 2, sK_u, sV_u, sSfK_u, sSfV_u, k_base,
            v_base, sfk_slice, v_atom_base, tma);
      }
      if (grp + 1 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, false, true>(p, tid, 0, stage ^ 1, grp + 1, sK_u, sV_u,
                                                        sSfK_u, sSfV_u, k_base, v_base, sfk_slice,
                                                        v_atom_base, tma);
      }
    } else if constexpr (SF_AHEAD) {
      // Tiles one group ahead on their own two-slot ring, atoms THREE groups ahead on a
      // three-slot one, in ONE commit group.  `wait_group 1` below therefore proves the
      // commit made at iteration iter-1, which carried atoms(grp+2) - and the atoms this
      // iteration writes to TMEM are grp+1's, proven by the wait at iteration iter-1 and
      // published to every thread by iteration iter-1's barrier.  That is the whole point:
      // the read is above THIS iteration's barrier, so the barrier can be the thing that
      // orders the write, and the second per-group barrier goes away.
      if constexpr (MIDPIPE) {
        // qk6: only K leads the loop top - two groups ahead on the three-slot ring,
        // on its own barrier, exactly the QK_AHEAD facility.  Slot (iter+2)%3's
        // previous occupant is K(grp-1), whose only reader QK(grp-1) retired at the
        // head of iteration grp-1, so the issue needs no wait.  V(grp+1) issues
        // mid-iteration instead, right after the PV(grp-1) retirement frees its
        // two-slot ring's slot (the GSPLIT rule).  The atom stream is unchanged.
        if (grp + 2 <= cHi) {
          tcg_issue_stage<D, PASS, THR, TMA, true, false, false, true>(
              p, tid, (iter + 2) % L::NSTAGE_K, 0, grp + 2, sK_u, sV_u, sSfK_u, sSfV_u,
              k_base, v_base, sfk_slice, v_atom_base, tma);
        }
        if (grp + 3 <= cHi) {
          tcg_stage_atoms<D, PASS, THR>(p, tid, (iter + 3) % L::NSTAGE_SF,
                                        (iter + 3) % L::NSTAGE_SF, grp + 3, sSfK_u, sSfV_u,
                                        sfk_slice, v_atom_base);
        }
      } else {
      if (grp + 1 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, true, true, false>(
            p, tid, stage ^ 1, stage ^ 1, grp + 1, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base,
            sfk_slice, v_atom_base, tma);
      }
      if (grp + 3 <= cHi) {
        tcg_stage_atoms<D, PASS, THR>(p, tid, (iter + 3) % L::NSTAGE_SF, (iter + 3) % L::NSTAGE_SF,
                                      grp + 3, sSfK_u, sSfV_u, sfk_slice, v_atom_base);
      }
      }
    } else if (grp + 1 <= cHi) {
      if constexpr (DPV) {
        // qk1: only K leads the loop here.  V(g+1) issues mid-iteration, right
        // after PV(g-1) retires, because the two-slot V ring puts V(g+1) in
        // PV(g-1)'s own slot.  K keeps its today ordering: QK(g-1,h1) was waited
        // inside the previous iteration, so slot (g+1)&1 is free.
        tcg_issue_stage<D, PASS, THR, TMA, true, false, true, true>(
            p, tid, stage ^ 1, stage ^ 1, grp + 1, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base,
            sfk_slice, v_atom_base, tma);
      } else {
        tcg_issue_stage<D, PASS, THR, TMA>(p, tid, stage ^ 1, stage ^ 1, grp + 1, sK_u, sV_u,
                                           sSfK_u, sSfV_u, k_base, v_base, sfk_slice, v_atom_base,
                                           tma);
      }
    }
    if constexpr (!SF_BULK) cp_async_commit();
    if constexpr (!SF_BULK) cp_async_wait<1>();
    if constexpr (SF_AHEAD) {
      // ---- group grp+1's K and V scale factors into TMEM, ABOVE the staging barrier
      // The bytes come from atom ring slot (iter+1) % NSTAGE_SF, filled by the cp.async
      // this loop issued at iteration iter-2, proven complete by iteration iter-1's
      // `wait_group 1`, and published to every thread by iteration iter-1's barrier (for
      // iteration 0, by the prime's).  So they are readable here, before this iteration
      // has synchronised at all - and that is what lets the write move up and the barrier
      // below order it, deleting the second per-group CTA-wide barrier this loop used to
      // need.  They go to the parity slot the NEXT iteration's MMAs read, `(grp+1) & 1`,
      // which is free because QK(grp-1) and PV(grp-1) - the only readers of that slot -
      // were both waited inside iteration iter-1.  Past `cHi` the source slot holds a
      // stale atom block and the destination is a slot no MMA will read, which is the
      // pattern the QK-lookahead form already documents for K and harmless for the same
      // reason: the index stays inside both rings and the bytes are never consumed.
      //
      // Placed above the tile wait rather than below it on purpose: the source bytes do
      // not depend on this group's K/V landing, so the four warps' `lds128`/`tcgen05.st`
      // pair and their `tcgen05.wait::st` run while the bulk copies are still in flight
      // instead of being exposed serially in front of the barrier.
      //
      // qk6/MIDPIPE keeps this position but rotates V's destination over THREE
      // slots ((grp+1)%3): PV(grp-1) is NOT retired here - it retires mid-iteration
      // - and the third slot keeps the write off the slot that in-flight PV(grp-1)
      // reads, because PV(grp-2) retired mid-iteration grp-2 and (grp+1)%3 differs
      // from both (grp-1)%3 and grp%3.  416..440 stays inside the gap below P_TILE
      // 448.  K's two parity slots suffice: the only in-flight SFB_K reader here is
      // QK(grp), which reads slot grp&1.  The combine barrier below orders this
      // write against the mid-iteration QK(grp+1) issue, exactly as the base form's
      // head barrier ordered it against the head issue.
      const int sfn = (iter + 1) % L::NSTAGE_SF;
      tcg_pipe_sf<DCH, WANT_V>(tlane, warp, tid & 31, sSfK_u + (uint32_t)(sfn * 512 * DCH),
                               sSfV_u + (uint32_t)(sfn * 512 * DCH),
                               TT::sfk_slot(tlane + TT::SFB_K, (grp + 1) & 1),
                               TT::sfk_slot(tlane + TT::SFB_V,
                                            MIDPIPE ? ((grp + 1) % 3) : ((grp + 1) & 1)));
    }
    // One barrier proves this group's K, its V and - with the atoms on the bulk
    // path - both scale-factor blocks, which is why the commit/wait pair above is
    // compiled out rather than merely redundant: with SF_BULK there is no cp.async
    // left in this loop for a wait_group to have anything to say about.
    // qk6/MIDPIPE has NO head tile wait, head fence or head barrier: K(grp+1) and
    // V(grp) are proven at the mid-iteration block below the readout, and the
    // key-split combine's own barrier orders those fences, the head atom stores and
    // the S-slot retirement for tid 0's mid QK(grp+1) issue.
    if constexpr (!MIDPIPE) {
    if constexpr (TMA) mbar_wait(&mbar_stage[stage], tmaPar);
    if constexpr (TMA) if (stage == 1) tmaPar ^= 1u;
    if constexpr (SF_AHEAD) {
      fence_async_shared();  // staged K/V become UMMA operands
      tcg_fence_before();
      // ONE barrier, and it now orders two things instead of one: every thread's staged
      // K and V for this group's MMAs, and the four warps' atom stores for the next
      // iteration's.  tid 0's `tcg_fence_after()` on the far side is unchanged.
      __syncthreads();
    } else {
    fence_async_shared();  // staged K/V become UMMA operands
    __syncthreads();

    // ---- this group's K and V scale factors into TMEM, replicated per lane
    {
      // With QK running a group ahead, the K scale bytes loaded here are group
      // grp+1's - the ones the QK issued at the end of this iteration will read -
      // and they go to the other of the two alternating slots so that the QK still
      // in flight keeps reading its own.  V's bytes stay with this group.  At the
      // last group the K source slot holds a stale tile; it is written to a slot no
      // MMA will read, and the index stays inside the ring.
      const int ksrc = L::QK_AHEAD ? ((iter + 1) % L::NSTAGE_K) : stage;
      const uint32_t srcK = sSfK_u + (uint32_t)(ksrc * 512 * DCH);
      const uint32_t srcV = sSfV_u + (uint32_t)(stage * 512 * DCH);
      (void)srcV;
      const int t = tid & 31;
      // Only the lookahead form alternates K's slot, and only K: its QK(grp+1) is
      // still in flight when iteration grp+1 writes group grp+2's bytes here, and
      // this loop's step ordering puts that write BEFORE the wait for QK(grp+1).  V
      // does not need to alternate because PV(grp) is waited at the end of this same
      // iteration, before the next write.  Every other form that reaches this loop
      // keeps the fixed base its own QK issue reads - which matters for PIPE_GM,
      // whose SF_ALT is true for the pipelined loop but whose short spans fall
      // through to here, where the reader below is the unshifted base.
      const uint32_t sfk_col =
          L::QK_AHEAD ? TT::sfk_slot(tlane + TT::SFB_K, stage ^ 1)
                      : (uint32_t)(tlane + TT::SFB_K);
      const uint32_t sfv_col = (uint32_t)(tlane + TT::SFB_V);
      // Warps 4..7 address the same 128 lanes as warps 0..3, so only the first
      // four warps stage the scale factors.
      if (warp < 4) {
#pragma unroll
        for (int g = 0; g < DCH; ++g) {
          const uint4 wk = lds128(srcK + (uint32_t)(g * 512 + t * 16));
          tcg_st4(sfk_col + 4 * g, wk.x, wk.y, wk.z, wk.w);
          // qk1: the DPV form's V(g) atoms have NOT landed yet - its V wait sits
          // mid-iteration, below the PV(g-1) retire the single SFB_V slot needs -
          // so this block writes K only and the V half lives at that wait.
          if (WANT_V && !DPV) {
            const uint4 wv = lds128(srcV + (uint32_t)(g * 512 + t * 16));
            tcg_st4(sfv_col + 4 * g, wv.x, wv.y, wv.z, wv.w);
          }
        }
        tcg_wait_st();
      }
    }
    tcg_fence_before();
    __syncthreads();
    }
    }  // !MIDPIPE head wait/fence/barrier

    // ---- group maximum over the masked, scaled scores (pass 1)
    // `needMask` is per lane: it depends on that row's own causal/window bounds,
    // so inside a warp it is true for the one or two boundary groups and false
    // for all the others.  Voting first makes the choice warp uniform, which is
    // also what lets the RED_MAX variant use a .sync.aligned tcgen05.ld.  These
    // declarations sit ABOVE the QK issue rather than below it because both the
    // 64-column form and the shipped one need them, and the vote is uniform either
    // way (every lane reaches it before the `tid == 0` divergence).
    const int keyLo = keyBase + khalf * KHALF;
    const bool needMask =
        ((unsigned)keyLo < (unsigned)lo) || ((unsigned)(keyLo + KHALF - 1) > (unsigned)hi);
    const bool anyMask = __any_sync(0xffffffffu, needMask);
    // The split pass replays the running maximum that the scan materialised from
    // the group-maximum pass, so it never reads a maximum of its own: for
    // PASS_SPLIT the whole reduction is dead, and where the scores are re-read by
    // pass 2 anyway (no SINGLE_READ) the TMEM loads that fed it are dead too.
    // Skipping them also drops the key-split exchange barrier, whose only
    // consumer was that same dead maximum.
    constexpr bool WANT_MAX = (PASS != PASS_SPLIT);
    float scores[SINGLE_READ ? KHALF : 1];
    float bmRaw = 0.0f;

    if constexpr (QK64) {
      // ===== 64-column score tile: two QK halves, ONE readout of the tile =====
      // Each half is an N=64 block-scaled MMA over the same Q tile and the same
      // four k-steps, reading the keys [h*64, h*64+64) of this group's staged K
      // (a 64*128 = 8192-byte offset inside each 128-byte swizzle chunk, i.e. a
      // multiple of the eight-row swizzle period, so the pattern is preserved) and
      // the SFB words [2h, 2h+2) of the atom that staging already wrote - bs3's G3b
      // decoded output column n to lane (n%32)+32*sp, word n/32 at N=64 exactly as
      // at N=128, and its G6 confirmed a +2 word offset is legal where +1 faults.
      //
      // Both halves land in the SAME 64 columns, so between them every warp must
      // have retired its tcgen05.ld; tcg_ld32 carries its own tcgen05.wait::ld, so
      // the fence-plus-barrier pair below is the whole cost of single-buffering the
      // tile.  The two halves' scores are then live together in `scores`, which is
      // what the contract needs: the group maximum spans all 128 keys before any
      // probability is computed, and the delayed maximum still fires once per
      // group.  That costs 128 registers, and they are free here precisely because
      // O moved to TMEM and `oacc[128]` became `oacc[1]`.
      bmRaw = -INFINITY;
#pragma unroll
      for (int h = 0; h < BN / 64; ++h) {
        if (h > 0) {
          tcg_fence_before();
          __syncthreads();
        }
        if (tid == 0) {
          tcg_fence_after();
          const uint32_t khalfstage =
              sK_u + (uint32_t)(stage * BN * D) + (uint32_t)(h * 64 * D);
#pragma unroll
          for (int kb = 0; kb < KB; ++kb) {
            const int chunk = kb >> 2, ink = kb & 3;
            const uint64_t da =
                umma_desc(sQ_u + (uint32_t)(chunk * (BM * 128)), 16, 1024) + 2 * ink;
            const uint64_t db =
                umma_desc(khalfstage + (uint32_t)(chunk * (BN * 128)), 16, 1024) + 2 * ink;
            tcg_mma_bs(tmem + TT::S, da, db, tcg_idesc(BM, 64, 0, ink), (uint32_t)(kb > 0),
                       tmem + TT::SFA_Q + 4 * chunk,
                       // gmq1: same parity-aware base as the 128-column form below
                       // (identity wherever SF_AHEAD is false — every form compiled
                       // today — so this is byte-identical for fused-QK64, and
                       // correct-by-construction for the GM-QK64 form whose SF_ALT
                       // is true via PIPE_GM).
                       tcg_sf_seq<SF_AHEAD, TT::SF_ALT>(tmem + TT::SFB_K, grp) + 2 * h +
                           4 * chunk);
          }
          tcg_commit(smem_addr(&mbar_qk));
        }
        mbar_wait(&mbar_qk, qkPhase);
        qkPhase ^= 1u;
        tcg_fence_after();
        // The mask is per half: a window or causal bound can fall inside either one,
        // and tcg_scores applies it to the stored value exactly as the 128-wide form
        // did, so the probability pass below sees the same numbers.
        const int hKeyLo = keyBase + h * 64;
        const bool hNeed =
            ((unsigned)hKeyLo < (unsigned)lo) || ((unsigned)(hKeyLo + 63) > (unsigned)hi);
        const bool hAny = __any_sync(0xffffffffu, hNeed);
        if constexpr (PASS == PASS_GROUP_MAX) {
          // gmq1: GM stores no scores, so it takes the max-only readers on the
          // 64-column tile — the SAME instantiations GM's D256 key-split already
          // compiles (KHALF=64) — and its `scores` array stays at size 1 (the
          // fused form's scores[128] is what pushed fused-QK64 to 255 regs; GM
          // must stay <= 170 for three CTAs per SM).  Masked halves use the
          // register path exactly as the 128-column form does (.red cannot mask);
          // the hAny warp vote keeps the .sync.aligned RED variant warp-uniform
          // (pF8).  Max is exact, so the half-merge order is bit-identical.
          bmRaw = fmaxf(bmRaw,
                        hAny ? tcg_group_max<64, true, false>(tlane, 0, hKeyLo, lo, hi)
                             : tcg_group_max<64, false, true>(tlane, 0, hKeyLo, lo, hi));
        } else {
          bmRaw = fmaxf(bmRaw,
                        hAny ? tcg_scores<64, true, true>(tlane, TC_S, 0, hKeyLo, lo, hi,
                                                          scores + h * 64)
                             : tcg_scores<64, false, true>(tlane, TC_S, 0, hKeyLo, lo, hi,
                                                           scores + h * 64));
        }
      }
    } else {
      // ---- QK^T with hardware MX block scaling.  With the lookahead this was
      // issued at the end of the previous iteration (or by the prime), so only its
      // wait is left here and the tensor pipe has been busy while this group was
      // staged.  qk6/MIDPIPE is the same discipline with the issue at the PREVIOUS
      // iteration's mid point (after its combine barrier proved the S readout
      // complete) rather than its foot, and the prime issues QK(cLo).
      if constexpr (!L::QK_AHEAD && !MIDPIPE)
      if (tid == 0) {
        tcg_fence_after();
        const uint32_t kstage = sK_u + (uint32_t)(stage * BN * D);
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
          const int chunk = kb >> 2, ink = kb & 3;
          const uint64_t da = umma_desc(sQ_u + (uint32_t)(chunk * (BM * 128)), 16, 1024) + 2 * ink;
          const uint64_t db =
              umma_desc(kstage + (uint32_t)(chunk * (BN * 128)), 16, 1024) + 2 * ink;
          tcg_mma_bs(tmem + TT::S, da, db, tcg_idesc(BM, BN, 0, ink), (uint32_t)(kb > 0),
                     tmem + TT::SFA_Q + 4 * chunk,
                     tcg_sf_seq<SF_AHEAD, TT::SF_ALT>(tmem + TT::SFB_K, grp) + 4 * chunk);
        }
        tcg_commit(smem_addr(&mbar_qk));
      }
      mbar_wait(&mbar_qk, qkPhase);
      qkPhase ^= 1u;
      tcg_fence_after();
      // A warp whose 32 rows are all padding reads no score: its maximum feeds
      // only its own dead rows' probabilities, denominator, accumulator, output
      // and LSE, every one of which the epilogue already stores under `valid`.
      // The QK wait above stays unguarded, so the mbarrier phase every thread
      // tracks still advances once per group.
      if (rowsLive) {
        if (SINGLE_READ) {
          bmRaw = FMHA_MASK_TEST(anyMask)
                      ? tcg_scores<KHALF, true, WANT_MAX>(tlane, TC_S, khalf, keyLo, lo, hi, scores)
                      : tcg_scores<KHALF, false, WANT_MAX>(tlane, TC_S, khalf, keyLo, lo, hi,
                                                           scores);
        } else if (!WANT_MAX) {
          // nothing to do: pass 2 loads and masks the scores itself
        } else if (FMHA_MASK_TEST(anyMask)) {
          bmRaw = tcg_group_max<KHALF, true, false>(tlane, khalf, keyLo, lo, hi);
        } else if (RED_MAX_FITS) {
          bmRaw = tcg_group_max<KHALF, false, true>(tlane, khalf, keyLo, lo, hi);
        } else {
          bmRaw = tcg_group_max<KHALF, false, false>(tlane, khalf, keyLo, lo, hi);
        }
      }
    }
    if constexpr (MIDPIPE) {
      // ---- qk6 MID-A: retire PV(grp-1), refill the V slot it freed, prove the
      // tiles this iteration's PV and the NEXT group's QK consume ----
      // The retirement is qk5's mechanism (bitwise-identical there; the numerics
      // never move) at its now-measured-correct position: prof4 showed the PV foot
      // wait was NOT the binding exposure once the QK wait also has slack, and the
      // rescale/P-store below plus the V and SFB_V slot rules REQUIRE it here:
      //   * V(grp+1)'s two-slot ring slot (grp+1)&1 holds V(grp-1), whose only
      //     reader PV(grp-1) is the retirement above - the GSPLIT rule verbatim;
      //   * the P(grp) store below writes the single P_TILE the in-flight PV(grp-1)
      //     still reads;
      //   * the O rescale below is a TMEM round trip on the accumulator PV(grp-1)
      //     is still accumulating.
      // K(grp+1) was issued by the PREVIOUS head (two-ahead, three-slot ring) and
      // V(grp) by the previous mid (or by the prime), so both waits find armed
      // barriers; the phase formulas count each slot's own arrivals: K slot
      // (iter+1)%NSTAGE_K sees arrival #(iter+1)/NSTAGE_K, V slot iter&1 sees
      // arrival #(iter>>1).  The fences below ride the combine barrier into tid 0's
      // MID-B issue exactly as the base form's head fence rides its head barrier.
      if (grp != cLo) {
        mbar_wait(&mbar_pv, pvPhase);
        pvPhase ^= 1u;
        tcg_fence_after();
      }
      if (grp + 1 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, false, true, false, true>(
            p, tid, 0, stage ^ 1, grp + 1, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base,
            sfk_slice, v_atom_base, tma);
      }
      if (grp + 1 <= cHi) {
        mbar_wait(kbar_base + (iter + 1) % L::NSTAGE_K,
                  (uint32_t)(((iter + 1) / L::NSTAGE_K) & 1));
      }
      mbar_wait(vbar_base + stage, (uint32_t)((iter >> 1) & 1));
      fence_async_shared();  // staged K/V become UMMA operands
      tcg_fence_before();
    }
    if (KSPLIT > 1 && WANT_MAX) {
      // The group maximum spans every warpgroup's key slice.  The barrier sits
      // outside every branch above, so warps that took different mask paths all
      // arrive at it; `max` is exact, so the combine is bit-identical to a single
      // 128-wide reduction.
      bmRaw = tcg_combine_halves<KSPLIT, BM, true>(sB_u, khalf, m, bmRaw);
    }
    if constexpr (MIDPIPE) {
      // ---- qk6 MID-B: issue QK(grp+1) under the exp/P-store slack ----
      // The combine barrier above (KSPLIT==2 here, so it always runs) proves every
      // warp retired its S(grp) readout - tcg_ld32 is self-retiring - orders the
      // loop-top atom stores and MID-A's fences, and is exactly the barrier the
      // base form's head used for this issue.  Same instruction sequence, same
      // descriptors, same scale slots: only the issue point moved, from the head
      // of iteration grp+1 (zero slack before its wait) to here, giving the wait
      // the whole exp/P-store half of this iteration plus the next head as slack.
      // The single S tile is legal to overwrite because the readout is proven done.
      if (grp + 1 <= cHi && tid == 0) {
        tcg_fence_after();
        tcg_pipe_qk<D, BM, BN>(tmem, TT::S, sQ_u,
                               sK_u + (uint32_t)(((iter + 1) % L::NSTAGE_K) * BN * D),
                               tmem + TT::SFA_Q,
                               tcg_sf_seq<SF_AHEAD, TT::SF_ALT>(tmem + TT::SFB_K, grp + 1));
        tcg_commit(smem_addr(&mbar_qk));
      }
    }
    // The softmax scale is positive, so scaling commutes with the maximum and
    // the 128 multiplies collapse into one: fl(max_j v_j * scale) is the same
    // value as max_j fl(v_j * scale), and -inf * scale stays -inf.
    const float bm = bmRaw * scale;

    if (PASS == PASS_GROUP_MAX) {
      if (valid && khalf == 0) p.bm[(size_t)grp * p.rows_total + rowG] = bm;
      __syncthreads();
      continue;
    }

    // ---- qk1 (DPV): retire PV(g-1), refill the V slot it freed, land V(g) ------
    // The PV retirement moves here from the foot of the previous iteration - the
    // PIPE loop's step-3 discipline inside the 256-column two-CTA form - so PV(g-1)
    // executes under this iteration's staging wait, K-atom write, barrier pair,
    // both QK half issues/waits and score readouts instead of being spin-waited at
    // its issue.  Every lifetime this reorders is derived at FMHA_QK64_DEFER_PV:
    // P and O need the retire above them (rescale round trip, next P store), the
    // single SFB_V slot and the V ring's slot (g-1)&1 need it above THEIR writes,
    // so the V(g+1) TMA issue and V(g)'s atom write both sit below it, and the
    // P-store barrier below orders the atom write against tid 0's PV issue exactly
    // as barrier A2 orders the loop-top K atom write against its QK issue.
    if constexpr (DPV) {
      if (grp != cLo) {
        mbar_wait(&mbar_pv, pvPhase);
        pvPhase ^= 1u;
        tcg_fence_after();
      }
      if (grp + 1 <= cHi) {
        tcg_issue_stage<D, PASS, THR, TMA, false, true, true, true>(
            p, tid, stage ^ 1, stage ^ 1, grp + 1, sK_u, sV_u, sSfK_u, sSfV_u, k_base, v_base,
            sfk_slice, v_atom_base, tma);
      }
      mbar_wait(vbar_base + stage, vPar);
      if (stage == 1) vPar ^= 1u;
      fence_async_shared();  // V(g)'s bytes become UMMA operands
      if (warp < 4) {
        const uint32_t srcVq = sSfV_u + (uint32_t)(stage * 512 * DCH);
        const int tq = tid & 31;
#pragma unroll
        for (int g = 0; g < DCH; ++g) {
          const uint4 wv = lds128(srcVq + (uint32_t)(g * 512 + tq * 16));
          tcg_st4((uint32_t)(tlane + TT::SFB_V) + 4 * g, wv.x, wv.y, wv.z, wv.w);
        }
        tcg_wait_st();
      }
    }

    // ---- delayed maximum rescale, exactly reference.py's recurrence
    float safe = 0.0f, alpha = 1.0f;
    if (rowsLive) {
    if (PASS == PASS_SPLIT) {
      // Replay the materialised running maximum so this chunk reproduces the
      // single-pass schedule bit-for-bit.  `mt` is the value prefetched at the top
      // of this iteration (or before the loop, for the first group).
      //
      // `mp` - the state entering group grp-1, which the reference's `exp(m_{g-1} -
      // m_g)` rescale factor needs - is NOT loaded.  It is bit-identical to the
      // `mstate` register this branch wrote at the end of the previous iteration:
      // that assignment is `mstate = mt`, i.e. `p.mstate[(grp-1)*rows_total + rowG]`,
      // and no other statement in this pass assigns `mstate` (the two `mstate = mn`
      // sites belong to the fused recurrence and to the pipelined loop, which
      // `TcgTmem::PIPE` restricts to PASS_FUSED).  For the first group the register
      // holds the pre-loop load of `mstate[(cLo-1)*rows_total + rowG]` when cLo > 0
      // and -INFINITY when cLo == 0, which is exactly what the `(grp > 0) ? ... :
      // -INFINITY` it replaces selected.  So this removes one of the pass's two
      // per-group global loads with no change to any value, and the prefetch above
      // hides the other.
      const float mp = mstate;
      float mt;
      if constexpr (FSCAN) {
        // fs5t: derive this group's state locally - the same recurrence step,
        // operands and order as the scan kernel's FMHA_SCAN_STEP, so `mt` is
        // bit-identical to the replayed mstate[grp].
        if ((fb - mp) > RESCALE_THRESHOLD) mstate = fb;
        mt = mstate;
        fb = bnxt;
      } else {
        mt = mt_cur;
        mt_cur = mt_next;
        mstate = mt;
      }
      safe = isfinite(mt) ? mt : 0.0f;
      alpha = isfinite(mp) ? expf(mp - safe) : 0.0f;
    } else {
    const bool upd = (bm - mstate) > RESCALE_THRESHOLD;
    const float mn = upd ? bm : mstate;
    safe = isfinite(mn) ? mn : 0.0f;
    alpha = isfinite(mstate) ? expf(mstate - safe) : 0.0f;
    mstate = mn;
    }
    den *= alpha;
    if (TT::O_IN_TMEM) {
      // O lives in TMEM, so the delayed-maximum rescale becomes a
      // load/multiply/store round trip.  tcgen05.ld/st are warp-wide, hence the
      // vote; lanes whose alpha is exactly 1.0f multiply by 1.0, which is
      // bit-identical to skipping them.  The first group of this CTA's span is
      // skipped outright: its PV MMA overwrites the accumulator (accum=0), and
      // multiplying uninitialised TMEM by alpha=0 could leave a NaN behind.
      if (grp != cLo && __any_sync(0xffffffffu, alpha != 1.0f)) {
#pragma unroll
        for (int c = 0; c < OCOLS / 32; ++c) {
          float ov[32];
          const uint32_t addr = tlane + TT::O + (uint32_t)(khalf * OCOLS + c * 32);
          tcg_ld32(addr, ov);
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= alpha;
          tcg_st32(addr, ov);
        }
        tcg_wait_st();
      }
    } else if (FMHA_RESCALE_LAYOUT == 1) {
      // Unconditional, and bit-identical: multiplying by exactly 1.0f reproduces every
      // fp32 value, and oacc[] is zero-initialised so the first group's alpha == 0.0f is a
      // no-op too.  No branch, so nothing for ptxas to duplicate.
#pragma unroll
      for (int t = 0; t < D; ++t) oacc[t] *= alpha;
    } else if (FMHA_RESCALE_LAYOUT == 2) {
      // Folded into the PV readback below as fmaf(oacc, alpha, v).  Nothing to do here, and
      // this is the arm that removes the branch WITHOUT adding an instruction to the
      // common path.
    } else if (FMHA_RESCALE_LAYOUT == 3 ? __builtin_expect(alpha != 1.0f, 0)
                                        : (alpha != 1.0f)) {
#pragma unroll
      for (int t = 0; t < D; ++t) oacc[t] *= alpha;
    }
    }  // rowsLive

    // ---- probabilities, E4M3 QDQ, and the P tile in K-major smem (pass 2)
    const uint64_t scale2 = f32x2_pack(scale, scale);
    const uint64_t nsafe2 = f32x2_pack(-safe, -safe);
    const uint64_t bias2 = f32x2_pack(bias, bias);
    // A dead warp writes no probability bytes.  Its rows of the P tile keep
    // whatever the previous group left there, the PV MMA multiplies that into
    // those rows of the accumulator, and the epilogue stores neither: O's rows
    // are independent in the PV product, so garbage in a dead row cannot reach a
    // live one.  The fence and the barrier below stay unguarded, so tid 0's PV
    // issue is still ordered after every live warp's stores.
    uint64_t den2 = f32x2_pack(0.0f, 0.0f);
    if (rowsLive) {
      if (SINGLE_READ) {
        den2 = tcg_probabilities_pre<KHALF, kBiasLog2, kDenAcc, PTMEM>(scores, pDst, m, khalf,
                                                                       scale2, nsafe2, bias2);
      } else if constexpr (FMHA_MASK_BRANCH == 2) {
        // The same two calls in the opposite order, so the arm that runs on 98.35% of
        // groups is the one ptxas lays out as the fall-through.  Values, arguments and
        // template parameters are untouched, so this is bit-exact by construction.
        if (!FMHA_MASK_TEST(anyMask)) {
          den2 = tcg_probabilities<KHALF, false, kBiasLog2, kDenAcc, PTMEM>(tlane, pDst, m, khalf,
                                                                            keyLo, lo, hi, scale2,
                                                                            nsafe2, bias2);
        } else {
          den2 = tcg_probabilities<KHALF, true, kBiasLog2, kDenAcc, PTMEM>(tlane, pDst, m, khalf,
                                                                           keyLo, lo, hi, scale2,
                                                                           nsafe2, bias2);
        }
      } else if (FMHA_MASK_TEST(anyMask)) {
        den2 = tcg_probabilities<KHALF, true, kBiasLog2, kDenAcc, PTMEM>(tlane, pDst, m, khalf,
                                                                         keyLo, lo, hi, scale2,
                                                                         nsafe2, bias2);
      } else {
        den2 = tcg_probabilities<KHALF, false, kBiasLog2, kDenAcc, PTMEM>(tlane, pDst, m, khalf,
                                                                          keyLo, lo, hi, scale2,
                                                                          nsafe2, bias2);
      }
    }
    {
      const float2 dend = f32x2_unpack(den2);
      den += dend.x + dend.y;
    }
    // Publish the probability tile.  In shared memory that is the async-proxy fence
    // the UMMA needs before it may read bytes the generic proxy wrote; in TMEM it is
    // this warp's own `tcgen05.wait::st`, because the tile never crossed proxies.
    // Either way the fence/barrier pair that follows is what orders every warp's
    // stores before tid 0's issue, and that pair is unchanged.
    if constexpr (PTMEM) {
      tcg_wait_st();
    } else {
      fence_async_shared();
    }
    tcg_fence_before();
    __syncthreads();

    // ---- PV: P from TMEM or from smem (K-major), V from smem (MN-major, no
    // transpose).  The key order, the accumulate predicate and both scale-factor
    // addresses are identical in the two residences, so the accumulator is too.
    if (tid == 0) {
      tcg_fence_after();
      const uint32_t vstage = sV_u + (uint32_t)(stage * BN * D);
      // qk6/MIDPIPE reads V's scale slot off the three-slot SFB_V ring (writer:
      // the head atom write of iteration grp-1, slot (grp)%3); every other form
      // keeps the two-parity-slot expression exactly.
      const uint32_t sfvMma =
          MIDPIPE ? (uint32_t)(tmem + TT::SFB_V + 8 * (grp % 3))
                  : tcg_sf_seq<SF_AHEAD, TT::SF_ALT>(tmem + TT::SFB_V, grp);
#pragma unroll
      for (int kb = 0; kb < BN / 32; ++kb) {
#pragma unroll
        for (int nc = 0; nc < DCH; ++nc) {
          const uint64_t db =
              umma_desc(vstage + (uint32_t)(nc * (128 * BN)), 0, 1024) + (uint64_t)256 * kb;
          // With O in TMEM the MMA accumulates across groups as well as across
          // k-blocks; the first MMA of the CTA must still overwrite, because
          // uninitialised TMEM can hold Inf/NaN and 0*Inf poisons the result.
          const uint32_t keep = (uint32_t)(kb > 0 || (TT::O_IN_TMEM && grp != cLo));
          if constexpr (PTMEM) {
            tcg_mma_bs_atmem(tmem + TT::O + nc * 128, pTmem + (uint32_t)(8 * kb), db,
                             tcg_idesc(BM, 128, 1, kb), keep, tmem + TT::SFA_P,
                             sfvMma + nc * 4);
          } else {
            const uint64_t da = umma_desc(sP_u, 16, 1024) + 2 * kb;
            tcg_mma_bs(tmem + TT::O + nc * 128, da, db, tcg_idesc(BM, 128, 1, kb), keep,
                       tmem + TT::SFA_P,
                       sfvMma + nc * 4);
          }
        }
      }
      tcg_commit(smem_addr(&mbar_pv));
      if constexpr (L::QK_AHEAD) {
        // The single score tile is free: pass 1 read it once (SINGLE_READ is true
        // whenever O lives in TMEM, which is exactly this form) and the barrier
        // above proves every warp retired its reads.  Putting the next group's QK on
        // the pipe before waiting for this group's PV is what runs two MMA chains at
        // once without the second score tile the pipelined form needs - D=256 has no
        // columns for one.  The arithmetic order on O is untouched: PV(grp-1) was
        // waited at the end of the previous iteration, this group's rescale of O
        // happened above, and PV(grp) accumulates after it, exactly as before.
        if (grp + 1 <= cHi) {
          const uint32_t knext = sK_u + (uint32_t)(((iter + 1) % L::NSTAGE_K) * BN * D);
#pragma unroll
          for (int kb = 0; kb < KB; ++kb) {
            const int chunk = kb >> 2, ink = kb & 3;
            const uint64_t da =
                umma_desc(sQ_u + (uint32_t)(chunk * (BM * 128)), 16, 1024) + 2 * ink;
            const uint64_t db =
                umma_desc(knext + (uint32_t)(chunk * (BN * 128)), 16, 1024) + 2 * ink;
            tcg_mma_bs(tmem + TT::S, da, db, tcg_idesc(BM, BN, 0, ink), (uint32_t)(kb > 0),
                       tmem + TT::SFA_Q + 4 * chunk,
                       TT::sfk_slot(tmem + TT::SFB_K, stage ^ 1) + 4 * chunk);
          }
          tcg_commit(smem_addr(&mbar_qk));
        }
      }
    }
    // qk1: the DPV form defers this retirement to the middle of the next iteration
    // (and to the post-loop epilogue guard for the last group); every other form
    // waits its PV exactly where it always has.  qk6/MIDPIPE defers it the same
    // way (to its own MID-A block), which is what keeps two MMA chains queued.
    if constexpr (!DPV && !MIDPIPE) {
      mbar_wait(&mbar_pv, pvPhase);
      pvPhase ^= 1u;
      tcg_fence_after();
    }

    // ---- num = num*alpha (already applied above) + r @ v_dec
    // With O in TMEM the PV MMA accumulated straight into it, so there is
    // nothing to read back per group.
    if (!TT::O_IN_TMEM && rowsLive) {
#pragma unroll
      for (int c = 0; c < D / 32; ++c) {  // KSPLIT==1 here: OCOLS == D
        float v[32];
        tcg_ld32(tlane + TC_S + c * 32, v);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          if (FMHA_RESCALE_LAYOUT == 2) {
            // The contract's `numerator*alpha + rounded @ V` with one rounding instead of
            // two.  Identical to the shipped form whenever alpha == 1.0f, which is 98.35%
            // of group iterations; within 0.5 ulp of it otherwise, on the OUTPUT and not on
            // a probability, so §130(e)'s e4m3-cliff bound is not engaged.
            oacc[c * 32 + j] = fmaf(oacc[c * 32 + j], alpha, v[j]);
          } else {
            oacc[c * 32 + j] += v[j];
          }
        }
      }
    }
    // See FMHA_SEQ_TAIL_BARRIER and FMHA_SEQ_TAIL_BARRIER_SPLIT.  Vacuous wherever the
    // accumulator lives in TMEM - nothing here reads TMEM, every thread has waited on this
    // group's PV itself, and the next iteration's staging targets a different ring slot than
    // the one that PV read.  Redundant rather than vacuous where the accumulator lives in
    // registers, because there the partial is read out of the same columns the next QK
    // overwrites, and the two are measured separately.  Kept unconditionally for the
    // unlaunched QK-lookahead form, whose next QK is issued inside this iteration.
    constexpr bool kTailBarrier =
        L::QK_AHEAD || (FMHA_SEQ_TAIL_BARRIER != 0) ||
        (!TT::O_IN_TMEM && ((PASS == PASS_SPLIT) ? (FMHA_SEQ_TAIL_BARRIER_SPLIT != 0)
                                                : (FMHA_SEQ_TAIL_BARRIER_FUSED != 0)));
    if constexpr (kTailBarrier) __syncthreads();
  }
  // qk1: retire the LAST deferred PV before anything reads O.  One commit per
  // iteration against one retirement per iteration offset by one, so exactly one
  // phase is outstanding here whenever the loop ran at all; the epilogue's O
  // readout below and (for the register forms, which never take DPV) the partial
  // readback are the only O consumers after the loop.  qk6/MIDPIPE has the same
  // one-outstanding-phase invariant (prime issues QK only; every iteration commits
  // one PV and retires one PV, the last retirement landing here).
  if constexpr (DPV || MIDPIPE) {
    if (cLo <= cHi) {
      mbar_wait(&mbar_pv, pvPhase);
      pvPhase ^= 1u;
      tcg_fence_after();
    }
  }
  }  // TT::PIPE / TT::PIPE_GM
  if constexpr (FSCAN) {
    // fs5t: republish what fmha_scan_flat_kernel used to publish - the final
    // delayed-maximum state per row.  Only the COMBINE reads it (PDL wait /
    // stream order after this kernel completes), never another split CTA -
    // the a-side fs1 rejection (dv_70e074a7276e) was exactly an in-kernel
    // cross-CTA mtail read racing this write.  The last chunk's walk ends at
    // gHiAll; groups past the useful span carry bm = -inf (the GM pass's
    // +-inf tails) and cannot move the state, and an EMPTY last chunk still
    // carries the full prefix walk, so every row's mtail is written exactly
    // once with exactly the flat scan's value.
    if (valid && ci.z == p.nsplit - 1) p.mtail[rowG] = mstate;
  }

  // ---- epilogue: one row per thread, so no reduction and no shuffles
  // The group split does not fold partials: one warpgroup summed a whole group's 128
  // keys, and the running denominator arrived complete in the epilogue exchange above,
  // so its den is the sequential one to the last bit rather than a reassociated fold.
  if (KSPLIT > 1 && !GSPLIT) {
    // Each warpgroup summed the probabilities of its own key slice; the row's
    // denominator is their sum, folded in key order.  The order differs from a
    // single-pass sum by ~1e-7 relative, which is five orders inside the
    // checker's rtol.
    den = tcg_combine_halves<KSPLIT, BM, false>(sB_u, khalf, m, den);
  }
  const bool live = den > 0.0f;
  const float inv = live ? (1.0f / den) : 0.0f;
  // Weight this chunk back to the global delayed-maximum reference so the
  // per-chunk partials sum to exactly the single-pass numerator/denominator.
  float wgt = 0.0f;
  if (PASS == PASS_SPLIT && any_compute && valid) {
    if constexpr (FSCAN) {
      // fs5t: store the partials UNWEIGHTED.  The normalisation - the shipped
      // exp(mA - mtail), bounded by 1 because the delayed state is monotone
      // non-decreasing - runs in the FWGT combine, from this chunk's end
      // state (part_m) and the last chunk's republished mtail, both read
      // under PDL/stream order.  No cross-CTA read inside the kernel (the
      // a-side fs1 race) and no unbounded e^{mA} (fs2' overflowed the
      // declared peaked generator: dv_bc379c3737ce).
      wgt = 1.0f;
    } else {
      const float tail = p.mtail[rowG];
      wgt = (isfinite(tail) && isfinite(mstate)) ? expf(mstate - tail) : 0.0f;
    }
  }
#if FMHA_PART_XPOSE
  // xt1 staging geometry + row remap.  The tile lives at smem_raw[0] over the
  // idle Q/K/V region; pitch carries a 16-byte pad so the 16-byte-chunk
  // phases stay bank-distinct (pitch words == 4 mod 32 for both formats).
  constexpr int XELEM = BF16PA ? 2 : 4;
  constexpr int XPITCH = D * XELEM + 16;
  // xt3: this branch is the O_IN_TMEM (D=256) epilogue, so the predicate only has to
  // decide whether an fp32 partial takes the cooperative transposed write as well.
  // See FMHA_PART_XPOSE_FP32 for why D=256 and not the D=128 register twin.
  constexpr bool XP = BF16PA || ((FMHA_PART_XPOSE_FP32 != 0) && (D == 256));
  // xd1: the D=128 register path's own gate.  See FMHA_PART_XPOSE_D128.
  constexpr bool XP128 = BF16PA || ((D == 128) &&
      (((FMHA_PART_XPOSE_D128 == 1)) ||
       ((FMHA_PART_XPOSE_D128 >= 2) && !PSK)));
  // Only the SPLIT instantiations stage (and only they have the V tile that
  // makes the room): GM D128 stages no V (oP = 64 KB < 67.6 KB of fp32
  // tile), so an unconditional assert here is a COMPILE ERROR in every
  // non-split form - which is what silently stubbed the whole tcg path in
  // the first xt1 attempt via solution.py's arch fallback (census REG:4,
  // record tcg:0, build_s 0.8).
  static_assert(PASS != PASS_SPLIT || (size_t)(128 * XPITCH) <= L::oP,
                "xt1 staging must fit in the idle Q+K+V tiles");
  const int xtotal = 128 * (D * XELEM / 16);
#endif
  // The group-maximum pass produces no numerator; the combine kernel writes
  // every element of O and LSE afterwards, so there is nothing to store here.
  if (PASS != PASS_GROUP_MAX) {
    if (TT::O_IN_TMEM) {
      // Stream the accumulator out of TMEM 32 channels at a time: a 256-float
      // row cannot live in registers next to the softmax state.  `any_compute`
      // is CTA-uniform, so the warp-wide tcgen05.ld is never taken
      // divergently, and a span this CTA did not cover has no numerator.
      // This warpgroup owns output channels [khalf*OCOLS, +OCOLS); with 8 warps
      // the two warpgroups read the same 128 TMEM lanes but disjoint columns.
#pragma unroll
      for (int c = 0; c < OCOLS / 32; ++c) {
        float ov[32];
        if (any_compute) {
          tcg_ld32(tlane + TT::O + (uint32_t)(khalf * OCOLS + c * 32), ov);
        } else {
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] = 0.0f;
        }
        if (PASS == PASS_SPLIT) {
#if FMHA_PART_XPOSE
          // xt1: stage this 32-channel block into the shared tile; the
          // cooperative global write runs after the c-loop.  BF16-only: the
          // measured fp32 tile doubles the smem round trip and LOST +5.5%
          // on the small-fp32-partial shapes (s3/s2, dv_44c0c0f9eda2) where
          // the 2x sector tax is only ~0.2us; bf16 won s5 -1.96% / d3
          // -4.87% bit-identical.
          if constexpr (!XP) {
            if (valid) {
              const size_t paOfs = ((size_t)ci.z * p.rows_total + rowG) * D +
                                   khalf * OCOLS + c * 32;
              float* pa = p.part_a + paOfs;
#pragma unroll
              for (int j = 0; j < 32; j += 4) stg128_scaled(pa + j, ov + j, wgt);
            }
          } else if (valid) {
            char* st = smem_raw + (size_t)m * XPITCH +
                       (size_t)(khalf * OCOLS + c * 32) * XELEM;
            if constexpr (BF16PA) {
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                __nv_bfloat162 pk[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                  pk[q] = __float22bfloat162_rn(make_float2(ov[u * 8 + 2 * q] * wgt,
                                                            ov[u * 8 + 2 * q + 1] * wgt));
                uint4 outv;
                outv.x = *reinterpret_cast<unsigned*>(&pk[0]);
                outv.y = *reinterpret_cast<unsigned*>(&pk[1]);
                outv.z = *reinterpret_cast<unsigned*>(&pk[2]);
                outv.w = *reinterpret_cast<unsigned*>(&pk[3]);
                *reinterpret_cast<uint4*>(st + u * 16) = outv;
              }
            } else {
#pragma unroll
              for (int u = 0; u < 8; ++u) {
                uint4 outv;
                outv.x = __float_as_uint(ov[u * 4 + 0] * wgt);
                outv.y = __float_as_uint(ov[u * 4 + 1] * wgt);
                outv.z = __float_as_uint(ov[u * 4 + 2] * wgt);
                outv.w = __float_as_uint(ov[u * 4 + 3] * wgt);
                *reinterpret_cast<uint4*>(st + u * 16) = outv;
              }
            }
          }
#else
          if (valid) {
            const size_t paOfs = ((size_t)ci.z * p.rows_total + rowG) * D +
                                 khalf * OCOLS + c * 32;
            // 16 bytes at a time; see stg128_scaled for the 4.0-of-32-bytes sector
            // measurement this replaces.  `pa` is 128-byte aligned: the workspace
            // base is, `rowG * D` is a multiple of 128 floats, and
            // `khalf * OCOLS + c * 32` is a multiple of 32 - in the BF16 twin a
            // multiple of 32 elements is 64 bytes, still 16-byte aligned.
            if constexpr (BF16PA) {
              __nv_bfloat16* pa = part_ptr<__nv_bfloat16>(p.part_a, paOfs);
#pragma unroll
              for (int j = 0; j < 32; j += 8) stg128_scaled16(pa + j, ov + j, wgt);
            } else {
              float* pa = p.part_a + paOfs;
#pragma unroll
              for (int j = 0; j < 32; j += 4) stg128_scaled(pa + j, ov + j, wgt);
            }
          }
#endif
        } else if (valid) {
          __nv_bfloat16* oRow = p.o + b * p.o_b + (int64_t)h * p.o_h + (int64_t)i * p.o_s +
                                khalf * OCOLS + c * 32;
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            __nv_bfloat162 packed[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const float2 f = make_float2(live ? ov[s * 8 + 2 * q] * inv : 0.0f,
                                           live ? ov[s * 8 + 2 * q + 1] * inv : 0.0f);
              packed[q] = __float22bfloat162_rn(f);
            }
            uint4 out;
            out.x = *reinterpret_cast<unsigned*>(&packed[0]);
            out.y = *reinterpret_cast<unsigned*>(&packed[1]);
            out.z = *reinterpret_cast<unsigned*>(&packed[2]);
            out.w = *reinterpret_cast<unsigned*>(&packed[3]);
            stg128(oRow + s * 8, out);
          }
        }
      }
#if FMHA_PART_XPOSE
      // xt1: the tile is complete (every warpgroup staged its channel half
      // of every row); one barrier publishes it, then ALL threads stream it
      // to part_a as consecutive 16-byte chunks - consecutive threads write
      // consecutive addresses, which is the 1x-sector pattern ap21's
      // follow-up called for.  Same values, same addresses as the
      // per-thread form.
      if (PASS == PASS_SPLIT && XP) {
        __syncthreads();
        constexpr int XCPR = D * XELEM / 16;
        for (int j = tid; j < xtotal; j += THR) {
          const int r = j / XCPR;
          const int c16 = j - r * XCPR;
          if (!xpose_valid(r, q0, Qph, log2G, PAD_SKIP, p.Sq)) continue;
          const uint4 v =
              *reinterpret_cast<const uint4*>(smem_raw + (size_t)r * XPITCH + c16 * 16);
          stg128(part_ptr<part_t<BF16PA>>(
                     p.part_a, ((size_t)ci.z * p.rows_total +
                      xpose_row(r, q0, Qph, log2Qph, kh, G, log2G, PAD_SKIP, b, p.H,
                                p.Sq)) *
                     D) +
                     c16 * (16 / XELEM),
                 v);
        }
      }
#endif
    } else if (PASS == PASS_SPLIT) {
#if FMHA_PART_XPOSE
      // xt1, D=128 register path: stage the whole scaled row, one barrier,
      // cooperative transposed write (see the O_IN_TMEM twin above).
      // BF16-only (measured: the fp32 smem round trip costs more than the
      // sector tax it removes at 2.4MB partials; bf16 D128 measured null
      // but harmless, and s9/d7 keep bit-identity).
      if constexpr (!XP128) {
        if (valid) {
          float* pa = p.part_a + ((size_t)ci.z * p.rows_total + rowG) * D;
#pragma unroll
          for (int d = 0; d < D; d += 4) stg128_scaled(pa + d, oacc + d, wgt);
        }
      } else {
      if (valid) {
        char* st = smem_raw + (size_t)m * XPITCH;
        if constexpr (BF16PA) {
#pragma unroll
          for (int u = 0; u < D / 8; ++u) {
            __nv_bfloat162 pk[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
              pk[q] = __float22bfloat162_rn(make_float2(oacc[u * 8 + 2 * q] * wgt,
                                                        oacc[u * 8 + 2 * q + 1] * wgt));
            uint4 outv;
            outv.x = *reinterpret_cast<unsigned*>(&pk[0]);
            outv.y = *reinterpret_cast<unsigned*>(&pk[1]);
            outv.z = *reinterpret_cast<unsigned*>(&pk[2]);
            outv.w = *reinterpret_cast<unsigned*>(&pk[3]);
            *reinterpret_cast<uint4*>(st + u * 16) = outv;
          }
        } else {
#pragma unroll
          for (int u = 0; u < D / 4; ++u) {
            uint4 outv;
            outv.x = __float_as_uint(oacc[u * 4 + 0] * wgt);
            outv.y = __float_as_uint(oacc[u * 4 + 1] * wgt);
            outv.z = __float_as_uint(oacc[u * 4 + 2] * wgt);
            outv.w = __float_as_uint(oacc[u * 4 + 3] * wgt);
            *reinterpret_cast<uint4*>(st + u * 16) = outv;
          }
        }
      }
      __syncthreads();
      {
        constexpr int XCPR = D * XELEM / 16;
        for (int j = tid; j < xtotal; j += THR) {
          const int r = j / XCPR;
          const int c16 = j - r * XCPR;
          if (!xpose_valid(r, q0, Qph, log2G, PAD_SKIP, p.Sq)) continue;
          const uint4 v =
              *reinterpret_cast<const uint4*>(smem_raw + (size_t)r * XPITCH + c16 * 16);
          stg128(part_ptr<part_t<BF16PA>>(
                     p.part_a, ((size_t)ci.z * p.rows_total +
                      xpose_row(r, q0, Qph, log2Qph, kh, G, log2G, PAD_SKIP, b, p.H,
                                p.Sq)) *
                     D) +
                     c16 * (16 / XELEM),
                 v);
        }
      }
      }
#else
      if (valid) {
        const size_t paOfs = ((size_t)ci.z * p.rows_total + rowG) * D;
        // Same 4-of-32-bytes pattern as the O_IN_TMEM branch above, and the same
        // fix.  This is the D=128 split form (s9, s2, s8, s3), whose `part_a` is
        // `nsplit*rows*128*4` (or *2 in the gated BF16 format).
        if constexpr (BF16PA) {
          __nv_bfloat16* pa = part_ptr<__nv_bfloat16>(p.part_a, paOfs);
#pragma unroll
          for (int d = 0; d < D; d += 8) stg128_scaled16(pa + d, oacc + d, wgt);
        } else {
          float* pa = p.part_a + paOfs;
#pragma unroll
          for (int d = 0; d < D; d += 4) stg128_scaled(pa + d, oacc + d, wgt);
        }
      }
#endif
    } else if (valid) {
      __nv_bfloat16* oRow = p.o + b * p.o_b + (int64_t)h * p.o_h + (int64_t)i * p.o_s;
#pragma unroll
      for (int c = 0; c < D / 8; ++c) {
        __nv_bfloat162 packed[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = make_float2(live ? oacc[c * 8 + 2 * q] * inv : 0.0f,
                                       live ? oacc[c * 8 + 2 * q + 1] * inv : 0.0f);
          packed[q] = __float22bfloat162_rn(f);
        }
        uint4 out;
        out.x = *reinterpret_cast<unsigned*>(&packed[0]);
        out.y = *reinterpret_cast<unsigned*>(&packed[1]);
        out.z = *reinterpret_cast<unsigned*>(&packed[2]);
        out.w = *reinterpret_cast<unsigned*>(&packed[3]);
        stg128(oRow + c * 8, out);
      }
    }
  }
  if (valid && khalf == 0) {
    if (PASS == PASS_SPLIT) {
      p.part_b[(size_t)ci.z * p.rows_total + rowG] = den * wgt;
      if constexpr (FSCAN) {
        // fs5t: this chunk's end state - the FWGT combine's normalisation
        // input.  Stored by every chunk, including empty ones (their
        // prefix-walked state with a zero den, whose weight therefore cannot
        // matter but stays finite: m_s <= mtail, so exp(m_s - mtail) <= 1).
        p.part_m[(size_t)ci.z * p.rows_total + rowG] = mstate;
      }
    } else if (PASS != PASS_GROUP_MAX && p.has_lse) {
      float value = -INFINITY;
      if (live && isfinite(mstate)) value = mstate + logf(den / bias);
      p.lse[b * p.l_b + (int64_t)h * p.l_h + (int64_t)i * p.l_s] = value;
    }
  }

  tcg_fence_before();
  __syncthreads();
  if (tid < 32) tcg_dealloc(tmem, TT::ALLOC);
  // TRIGGER AT THE EXIT, not the entry, and that is measured rather than cautious.  Entry
  // placement is what §37's synthetic chain used and it recovered 0.572 us per boundary,
  // but there every kernel was one CTA on a 148-SM device.  Here gm and split are sized to
  // exactly one resident wave (nsplit = resident/base), so a successor launched at the
  // primary's ENTRY steals slots from a grid that has none to spare and forces an extra
  // wave: dv_7399df02d56a measured s5 at -5.5% (gm is 148 CTAs at 1 CTA/SM and its segmax
  // successor is 33 CTAs, i.e. 22% of the machine taken away from a full wave), for a
  // whole-suite x0.9929 against x1.0062 for exit placement.  Exit placement still lets the
  // successor's grid fill during the primary's drain, which is where the remaining
  // recovery is.  Correctness does not depend on the choice: the trigger only says when the
  // dependent may LAUNCH, and its cudaGridDependencySynchronize waits for every primary
  // thread to exit and for its writes to be visible either way.
  fmha_pdl_trigger();
#endif  // FMHA_TCG_ARCH (kernel body)
}

template <int D, int PASS, int KSPLIT = 1, bool GSPLIT = false>
struct TcgSmem {
  static constexpr size_t bytes = TcgLayout<D, PASS, KSPLIT, GSPLIT>::bytes;
};

}  // namespace fmha_mxfp8
