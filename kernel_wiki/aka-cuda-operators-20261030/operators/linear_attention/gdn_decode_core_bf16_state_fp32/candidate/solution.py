"""GDN decode core (bf16 io, fp32 state) — independent CUDA C++ implementation.

Structure (promoted from dev variant v51; round-20 same-job A/B confirmed
the third consecutive reproduction: case1 -1.25%, d1 -0.73%, d9 -0.51%,
unchanged-path controls identical to 0.0%, replay checks 0/1/d0 pass):
  * k1 gdn_pre — hybrid, size-selected: t*heads >= 1024 AND (heads >= 16 OR
    t >= 256) launches gdn_pre8
    (8 heads x 256 threads, one warp per head, 4 channels/lane: uint2
    history/token, 4x uint2 weights, uint2 scratch+roll stores, float4 ws,
    alignment-guarded scalar fallbacks; q/k norm is a pure warp wredsum).
    History/token/weights stay as packed raw u32 registers with per-channel
    conversion inside the unrolled fma chain (the fully-materialized form
    used 56 regs = 4 blocks/SM at 45% occupancy; packed form restores
    occupancy: case1 -12.3us, case9 -2.1%).
    Smaller shapes keep the lean gdn_pre1 (1 head x 128 threads). Both
    compute conv/SiLU/BF16 scratch, exact conv-history roll, QK L2 norm,
    value/decay/beta into the FP32 workspace, and null-row out zeroing.
    (gdn_pre4, the 2ch/thread predecessor, remains compiled but undispatched;
    pre8 measured -6.2us on case 1 where pre4 was Compute-79%/1.9TB-s.)
  * k2 gdn_state: tile8 stages its 8KB contiguous 16-row tiles with
    cp.async.bulk (TMA) + mbarrier parity pipeline (4 buffers, 3-tile
    prologue, elected-thread
    issue, one __syncthreads per tile for buffer-reuse ordering); paired-row
    warp compute, BF16-materialized raw + gated RMSNorm;
    grid swap (value-head-fastest) for all t*hv > 512 (matrix28/29:
    fast medians 12/12 on mids, new c5/c9 min floors, zero regressions;
    4-stage prologue: c1 -2..-4us replicated, lowest c1 min recorded);
    small t*hv <= 512:
    W32/1024-thread vec32pf for t*hv <= 128 (two serial row-pair iterations,
    32-warp occupancy on 1-block/SM tiny shapes; case4 -14%), else
    W16/512-thread vec16pf (prefetch) when t*hv <= nsm*3, else vec16;
    the vec32pf tiny tier prefetches BOTH of its row pairs above the
    dependency wait (PF2): its grid never exceeds one block/SM, so the
    +16 registers are free there (matrix20/21: c4 median 14.24->12.29,
    locking the fast mode);
    all prefetch tiers AND tile8 additionally hoist the epilogue's
    z-gate (qkvz z-channels, pass-through input) and norm_weight loads
    above the dependency wait — pure kernel inputs whose late reads
    otherwise expose one DRAM round-trip at the kernel tail
    (matrix25/26: c0 -16.3%, c5 -3.5%, c8 -3.2%, c2 -4.8%, c4 fast-lock;
    tile8 stays 40 regs / 6 blocks/SM);
    scalar fallback for misaligned state slots.
  * PDL (programmatic dependent launch, sm_90+): pre kernels fire
    cudaTriggerProgrammaticLaunchCompletion() as their first statement; the
    state kernels are launched via cudaLaunchKernelEx with
    ProgrammaticStreamSerialization, prefetch pure-input state tiles
    (tile8 cp.async prologue hoisted first), then cudaGridDependencySynchronize()
    before any ws/scratch read.  Overlaps dependent launch + state prefetch
    with the producer kernel; bit-identical values.  The small-path body
    prefetches its first state row pair (float4) ABOVE the dependency wait
    (case0 -14.3%); the prefetch costs +8 registers (vec16 occupancy 4->3
    blocks/SM), so it is a template switch (PF): gdn_state_vec16pf is
    dispatched only when t*hv <= nsm*3, where the occupancy loss cannot
    create an extra wave (d4-class TH=512 shapes keep plain vec16).
  * Host: nvcc-built .so; CUDA-graph signature cache (replay recomputes from
    live buffers — no cached outputs); trimmed tensor-identity key;
    METH_FASTCALL binding with ctypes fallback; inlined __call__.

Numerical contract (frozen problem/reference.py + metadata.json, revision
reference-materialization-v1-20261013):
  * 4-tap causal depthwise convolution in FP32 on stored BF16 inputs, SiLU,
    materialized to BF16 into the qkvz scratch prefix before norm/recurrence.
  * Rolling conv history: cols <- [old col1, old col2, token] (exact BF16 copy).
  * Per-head L2 normalize q/k (eps 1e-6, FP32); q additionally scaled 128**-0.5
    after the key-head gather (equivalently before, per key head).
  * decay = exp(-exp(A_log)*softplus(a+dt_bias)) (softplus threshold 20),
    beta = sigmoid(b);  DeltaNet step per (token, value head):
      s = s*decay;  s += beta*(value - s@k) outer k;  raw = s@q
  * raw materialized to BF16 (out.dtype), then gated RMSNorm(eps 1e-6) with
    sigmoid(z) gate and norm_weight, written BF16; s written back FP32.
  * slot <= 0 (null request): only the out row is zeroed; qkvz/conv/state
    untouched. Tails, spare history columns, padding and Z stay protected.

All offsets/strides are generic int64; tensor geometry is taken from the
actual arguments, never from case identity.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch

_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <string.h>

#define EPS 1e-6f
#define QSCALE 0.08838834764831845f

__device__ __forceinline__ float wredsum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* TMA 1D bulk stage: one elected thread moves a contiguous 8KB tile
 * (16 rows x 128 fp32) global->shared, completion signalled on mbar. */
__device__ __forceinline__ void bulk_stage8k(const float* gsrc, float* sdst,
                                             unsigned long long* mbar) {
    const unsigned dst = smem_u32(sdst);
    const unsigned mb = smem_u32(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 8192;"
                 :: "r"(mb) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1], 8192, [%2];"
                 :: "r"(dst), "l"(gsrc), "r"(mb) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned long long* mbar, unsigned parity) {
    const unsigned mb = smem_u32(mbar);
    asm volatile("{\n\t.reg .pred p;\n\tLW%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra LW%=;\n\t}"
                 :: "r"(mb), "r"(parity) : "memory");
}

/* Per-token preparation: conv+SiLU+BF16 scratch+conv history roll, QK L2 norm,
 * value/decay/beta into FP32 workspace, and null-row out zeroing.
 * grid (t, 2h+hv), block 128; thread tid == channel within head.
 * head < h: q head; h <= head < 2h: k head; head >= 2h: value head. */
extern "C" __global__ void gdn_pre1(
    /* qkvz/conv/convw deliberately WITHOUT __restrict__: conv and convw are
     * the same pool (in-place history roll) and qkvz is read (token) before
     * being written (scratch) at the same address — restrict here would
     * license the compiler to reorder pool loads across pool stores (the
     * aliasing trap documented in the flashinfer fused-decode reference). */
    __nv_bfloat16* qkvz,
    const __nv_bfloat16* __restrict__ ba,
    const __nv_bfloat16* __restrict__ cw,
    const __nv_bfloat16* conv,
    __nv_bfloat16* convw,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ slots,
    const float* __restrict__ alog,
    const __nv_bfloat16* __restrict__ dtb,
    float* __restrict__ ws,
    long long t, long long h, long long hv, long long c,
    long long qkvz_s0, long long ba_s0, long long out_s0, long long out_s1,
    long long conv_sslot, long long conv_sch, long long conv_scol,
    long long cw_s0, long long slots_s)
{
    /* PDL producer: fire the programmatic launch-completion event as early as
     * possible.  The dependent state kernel performs no producer-output reads
     * before its own cudaGridDependencySynchronize(), and the event semantics
     * (all producer blocks started or exited) guarantee the producer is fully
     * scheduled before dependent blocks occupy SM slots. */
    cudaTriggerProgrammaticLaunchCompletion();
    __shared__ float red[4];
    const long long row = blockIdx.x;
    const long long head = blockIdx.y;
    const int tid = threadIdx.x;
    const int slot = slots[row * slots_s];
    if (slot <= 0) {
        if (head >= 2 * h) {
            const long long vh = head - 2 * h;
            out[row * out_s0 + vh * out_s1 + tid] = __float2bfloat16(0.0f);
        }
        return;
    }
    const long long ch = head * 128 + tid;
    const __nv_bfloat16* cs = conv + slot * conv_sslot + ch * conv_sch;
    const __nv_bfloat16 h0 = cs[0];
    const __nv_bfloat16 h1 = cs[conv_scol];
    const __nv_bfloat16 h2 = cs[2 * conv_scol];
    const __nv_bfloat16 tok = qkvz[row * qkvz_s0 + ch];
    const __nv_bfloat16* wp = cw + ch * cw_s0;
    float acc = __bfloat162float(h0) * __bfloat162float(wp[0]);
    acc += __bfloat162float(h1) * __bfloat162float(wp[1]);
    acc += __bfloat162float(h2) * __bfloat162float(wp[2]);
    acc += __bfloat162float(tok) * __bfloat162float(wp[3]);
    const float vs = __fdividef(acc, 1.0f + __expf(-acc));
    const __nv_bfloat16 vb = __float2bfloat16_rn(vs);
    qkvz[row * qkvz_s0 + ch] = vb;
    __nv_bfloat16* cw2 = convw + slot * conv_sslot + ch * conv_sch;
    cw2[0] = h1;
    cw2[conv_scol] = h2;
    cw2[2 * conv_scol] = tok;
    const float vf = __bfloat162float(vb);
    float* wsrow = ws + row * (c + 2 * hv);
    if (head >= 2 * h) {
        const long long vh = head - 2 * h;
        wsrow[head * 128 + tid] = vf;
        if (tid == 0) {
            const float b = __bfloat162float(ba[row * ba_s0 + vh]);
            const float a = __bfloat162float(ba[row * ba_s0 + hv + vh]);
            const float dt = __bfloat162float(dtb[vh]);
            const float x = a + dt;
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            const float decay = expf(-expf(alog[vh]) * sp);
            const float beta = 1.0f / (1.0f + expf(-b));
            wsrow[c + 2 * vh] = decay;
            wsrow[c + 2 * vh + 1] = beta;
        }
    } else {
        const int lane = tid & 31;
        const int wid = tid >> 5;
        float ssq = wredsum(vf * vf);
        if (lane == 0) red[wid] = ssq;
        __syncthreads();
        const float r = vf * rsqrtf((red[0] + red[1] + red[2] + red[3]) + EPS);
        wsrow[head * 128 + tid] = (head < h) ? (r * QSCALE) : r;
    }
}

extern "C" __global__ void gdn_pre4(
    __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ ba,
    const __nv_bfloat16* __restrict__ cw,
    const __nv_bfloat16* __restrict__ conv,
    __nv_bfloat16* __restrict__ convw,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ slots,
    const float* __restrict__ alog,
    const __nv_bfloat16* __restrict__ dtb,
    float* __restrict__ ws,
    long long t, long long h, long long hv, long long c,
    long long qkvz_s0, long long ba_s0, long long out_s0, long long out_s1,
    long long conv_sslot, long long conv_sch, long long conv_scol,
    long long cw_s0, long long slots_s)
{
    cudaTriggerProgrammaticLaunchCompletion();   /* PDL producer; see gdn_pre1 */
    __shared__ float red[8];
    const long long row = blockIdx.x;
    const long long heads = c / 128;
    const long long head = (long long)blockIdx.y * 4 + (threadIdx.x >> 6);
    const int p = (threadIdx.x & 63) * 2;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bool live = head < heads;
    const int slot = slots[row * slots_s];
    if (slot <= 0) {
        if (live && head >= 2 * h) {
            const long long vh = head - 2 * h;
            __nv_bfloat16* op = out + row * out_s0 + vh * out_s1 + p;
            const __nv_bfloat16 z = __float2bfloat16(0.0f);
            *(__nv_bfloat162*)op = __nv_bfloat162(z, z);
        }
        return;
    }
    float v0 = 0.0f, v1 = 0.0f;
    if (live) {
        const long long ch = head * 128 + p;
        const __nv_bfloat16* cs = conv + slot * conv_sslot + ch * conv_sch;
        __nv_bfloat16* tk = qkvz + row * qkvz_s0 + ch;
        const bool ca = (conv_sch == 1) && (conv_scol % 2 == 0)
                     && ((slot * conv_sslot) % 2 == 0);
        const __nv_bfloat162 TK = *(const __nv_bfloat162*)tk;
        __nv_bfloat162 H0, H1, H2;
        if (ca) {
            H0 = *(const __nv_bfloat162*)cs;
            H1 = *(const __nv_bfloat162*)(cs + conv_scol);
            H2 = *(const __nv_bfloat162*)(cs + 2 * conv_scol);
        } else {
            H0 = __nv_bfloat162(cs[0], cs[conv_sch]);
            H1 = __nv_bfloat162(cs[conv_scol], cs[conv_scol + conv_sch]);
            H2 = __nv_bfloat162(cs[2 * conv_scol], cs[2 * conv_scol + conv_sch]);
        }
        const float2 f0 = __bfloat1622float2(H0);
        const float2 f1 = __bfloat1622float2(H1);
        const float2 f2 = __bfloat1622float2(H2);
        const float2 ft = __bfloat1622float2(TK);
        float w0a, w1a, w2a, w3a, w0b, w1b, w2b, w3b;
        const __nv_bfloat16* wp = cw + ch * cw_s0;
        if (cw_s0 == 4) {
            const uint4 wraw = *(const uint4*)wp;
            const __nv_bfloat162 wa0 = *(__nv_bfloat162*)&wraw.x;
            const __nv_bfloat162 wa1 = *(__nv_bfloat162*)&wraw.y;
            const __nv_bfloat162 wb0 = *(__nv_bfloat162*)&wraw.z;
            const __nv_bfloat162 wb1 = *(__nv_bfloat162*)&wraw.w;
            const float2 fa0 = __bfloat1622float2(wa0);
            const float2 fa1 = __bfloat1622float2(wa1);
            const float2 fb0 = __bfloat1622float2(wb0);
            const float2 fb1 = __bfloat1622float2(wb1);
            w0a = fa0.x; w1a = fa0.y; w2a = fa1.x; w3a = fa1.y;
            w0b = fb0.x; w1b = fb0.y; w2b = fb1.x; w3b = fb1.y;
        } else {
            const __nv_bfloat16* wq = wp + cw_s0;
            w0a = __bfloat162float(wp[0]); w1a = __bfloat162float(wp[1]);
            w2a = __bfloat162float(wp[2]); w3a = __bfloat162float(wp[3]);
            w0b = __bfloat162float(wq[0]); w1b = __bfloat162float(wq[1]);
            w2b = __bfloat162float(wq[2]); w3b = __bfloat162float(wq[3]);
        }
        float accA = f0.x * w0a;
        accA += f1.x * w1a;
        accA += f2.x * w2a;
        accA += ft.x * w3a;
        float accB = f0.y * w0b;
        accB += f1.y * w1b;
        accB += f2.y * w2b;
        accB += ft.y * w3b;
        const float vsA = __fdividef(accA, 1.0f + __expf(-accA));
        const float vsB = __fdividef(accB, 1.0f + __expf(-accB));
        const __nv_bfloat16 vbA = __float2bfloat16_rn(vsA);
        const __nv_bfloat16 vbB = __float2bfloat16_rn(vsB);
        *(__nv_bfloat162*)tk = __nv_bfloat162(vbA, vbB);
        __nv_bfloat16* cw2 = convw + slot * conv_sslot + ch * conv_sch;
        if (ca) {
            *(__nv_bfloat162*)cw2 = H1;
            *(__nv_bfloat162*)(cw2 + conv_scol) = H2;
            *(__nv_bfloat162*)(cw2 + 2 * conv_scol) = TK;
        } else {
            cw2[0] = H1.x;
            cw2[conv_sch] = H1.y;
            cw2[conv_scol] = H2.x;
            cw2[conv_scol + conv_sch] = H2.y;
            cw2[2 * conv_scol] = TK.x;
            cw2[2 * conv_scol + conv_sch] = TK.y;
        }
        v0 = __bfloat162float(vbA);
        v1 = __bfloat162float(vbB);
    }
    float part = v0 * v0 + v1 * v1;
    part = wredsum(part);
    if (lane == 0) red[wid] = part;
    __syncthreads();
    if (!live) return;
    float* wsrow = ws + row * (c + 2 * hv) + head * 128 + p;
    if (head >= 2 * h) {
        const long long vh = head - 2 * h;
        *(float2*)wsrow = make_float2(v0, v1);
        if ((threadIdx.x & 63) == 0) {
            const float b = __bfloat162float(ba[row * ba_s0 + vh]);
            const float a = __bfloat162float(ba[row * ba_s0 + hv + vh]);
            const float dt = __bfloat162float(dtb[vh]);
            const float x = a + dt;
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            const float decay = expf(-expf(alog[vh]) * sp);
            const float beta = 1.0f / (1.0f + expf(-b));
            float* db = ws + row * (c + 2 * hv) + c + 2 * vh;
            db[0] = decay;
            db[1] = beta;
        }
    } else {
        const float sum = red[(wid & 6) ] + red[(wid & 6) + 1];
        const float inv = rsqrtf(sum + EPS);
        if (head < h) {
            *(float2*)wsrow = make_float2(v0 * inv * QSCALE, v1 * inv * QSCALE);
        } else {
            *(float2*)wsrow = make_float2(v0 * inv, v1 * inv);
        }
    }
}

extern "C" __global__ void gdn_pre8(
    /* see gdn_pre1: no __restrict__ on qkvz/conv/convw (aliased pools) */
    __nv_bfloat16* qkvz,
    const __nv_bfloat16* __restrict__ ba,
    const __nv_bfloat16* __restrict__ cw,
    const __nv_bfloat16* conv,
    __nv_bfloat16* convw,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ slots,
    const float* __restrict__ alog,
    const __nv_bfloat16* __restrict__ dtb,
    float* __restrict__ ws,
    long long t, long long h, long long hv, long long c,
    long long qkvz_s0, long long ba_s0, long long out_s0, long long out_s1,
    long long conv_sslot, long long conv_sch, long long conv_scol,
    long long cw_s0, long long slots_s)
{
    cudaTriggerProgrammaticLaunchCompletion();   /* PDL producer */
    const long long row = blockIdx.x;
    const long long heads = c / 128;
    const long long head = (long long)blockIdx.y * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool live = head < heads;
    const int slot = slots[row * slots_s];
    if (slot <= 0) {
        if (live && head >= 2 * h) {
            const long long vh = head - 2 * h;
            uint2* op = (uint2*)(out + row * out_s0 + vh * out_s1 + lane * 4);
            *op = make_uint2(0u, 0u);
        }
        return;
    }
    if (!live) return;
    const long long ch = head * 128 + (long long)lane * 4;
    const __nv_bfloat16* cs = conv + slot * conv_sslot + ch * conv_sch;
    __nv_bfloat16* tk = qkvz + row * qkvz_s0 + ch;
    const bool ca = (conv_sch == 1) && (conv_scol % 4 == 0)
                 && ((slot * conv_sslot) % 4 == 0);
    /* Packed-load structure: history/token/weights stay as raw u32 pairs in
     * registers (16 regs); per-channel conversion happens inside the unrolled
     * fma chain.  Halves pre8's live floats vs the fully-materialized form
     * (56 regs -> 4 blocks/SM was occupancy-starved at 45%). */
    const uint2 TKraw = *(const uint2*)tk;
    uint2 H0raw, H1raw, H2raw;
    if (ca) {
        H0raw = *(const uint2*)cs;
        H1raw = *(const uint2*)(cs + conv_scol);
        H2raw = *(const uint2*)(cs + 2 * conv_scol);
    } else {
        __nv_bfloat162 h0p[2], h1p[2], h2p[2];
        h0p[0] = __nv_bfloat162(cs[0], cs[conv_sch]);
        h0p[1] = __nv_bfloat162(cs[2 * conv_sch], cs[3 * conv_sch]);
        h1p[0] = __nv_bfloat162(cs[conv_scol], cs[conv_scol + conv_sch]);
        h1p[1] = __nv_bfloat162(cs[conv_scol + 2 * conv_sch], cs[conv_scol + 3 * conv_sch]);
        h2p[0] = __nv_bfloat162(cs[2 * conv_scol], cs[2 * conv_scol + conv_sch]);
        h2p[1] = __nv_bfloat162(cs[2 * conv_scol + 2 * conv_sch], cs[2 * conv_scol + 3 * conv_sch]);
        H0raw = *reinterpret_cast<const uint2*>(&h0p[0]);
        H1raw = *reinterpret_cast<const uint2*>(&h1p[0]);
        H2raw = *reinterpret_cast<const uint2*>(&h2p[0]);
    }
    uint4 Wraw[2];
    const __nv_bfloat16* wp = cw + ch * cw_s0;
    if (cw_s0 == 4) {
        Wraw[0] = *(const uint4*)wp;
        Wraw[1] = *(const uint4*)(wp + 8);
    } else {
        __nv_bfloat162 wtmp[8];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const __nv_bfloat16* wq = wp + (long long)u * cw_s0;
            wtmp[2 * u] = __nv_bfloat162(wq[0], wq[1]);
            wtmp[2 * u + 1] = __nv_bfloat162(wq[2], wq[3]);
        }
        Wraw[0] = *reinterpret_cast<const uint4*>(&wtmp[0]);
        Wraw[1] = *reinterpret_cast<const uint4*>(&wtmp[4]);
    }
    const __nv_bfloat16* H0e = reinterpret_cast<const __nv_bfloat16*>(&H0raw);
    const __nv_bfloat16* H1e = reinterpret_cast<const __nv_bfloat16*>(&H1raw);
    const __nv_bfloat16* H2e = reinterpret_cast<const __nv_bfloat16*>(&H2raw);
    const __nv_bfloat16* TKe = reinterpret_cast<const __nv_bfloat16*>(&TKraw);
    const __nv_bfloat16* We = reinterpret_cast<const __nv_bfloat16*>(Wraw);
    float v[4];
    __nv_bfloat16 vb[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
        float acc = __bfloat162float(H0e[u]) * __bfloat162float(We[4 * u]);
        acc += __bfloat162float(H1e[u]) * __bfloat162float(We[4 * u + 1]);
        acc += __bfloat162float(H2e[u]) * __bfloat162float(We[4 * u + 2]);
        acc += __bfloat162float(TKe[u]) * __bfloat162float(We[4 * u + 3]);
        const float s0 = __fdividef(acc, 1.0f + __expf(-acc));
        vb[u] = __float2bfloat16_rn(s0);
        v[u] = __bfloat162float(vb[u]);
    }
    {
        const __nv_bfloat162 SV01 = __nv_bfloat162(vb[0], vb[1]);
        const __nv_bfloat162 SV23 = __nv_bfloat162(vb[2], vb[3]);
        uint2 SV;
        SV.x = *reinterpret_cast<const unsigned*>(&SV01);
        SV.y = *reinterpret_cast<const unsigned*>(&SV23);
        *(uint2*)tk = SV;
    }
    __nv_bfloat16* cw2 = convw + slot * conv_sslot + ch * conv_sch;
    if (ca) {
        *(uint2*)cw2 = H1raw;
        *(uint2*)(cw2 + conv_scol) = H2raw;
        *(uint2*)(cw2 + 2 * conv_scol) = TKraw;
    } else {
        cw2[0] = H1e[0];            cw2[conv_sch] = H1e[1];
        cw2[2 * conv_sch] = H1e[2]; cw2[3 * conv_sch] = H1e[3];
        cw2[conv_scol] = H2e[0];                 cw2[conv_scol + conv_sch] = H2e[1];
        cw2[conv_scol + 2 * conv_sch] = H2e[2];  cw2[conv_scol + 3 * conv_sch] = H2e[3];
        cw2[2 * conv_scol] = TKe[0];             cw2[2 * conv_scol + conv_sch] = TKe[1];
        cw2[2 * conv_scol + 2 * conv_sch] = TKe[2]; cw2[2 * conv_scol + 3 * conv_sch] = TKe[3];
    }
    float* wsrow = ws + row * (c + 2 * hv) + head * 128 + lane * 4;
    if (head >= 2 * h) {
        const long long vh = head - 2 * h;
        *(float4*)wsrow = make_float4(v[0], v[1], v[2], v[3]);
        if (lane == 0) {
            const float b = __bfloat162float(ba[row * ba_s0 + vh]);
            const float a = __bfloat162float(ba[row * ba_s0 + hv + vh]);
            const float dt = __bfloat162float(dtb[vh]);
            const float x = a + dt;
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            const float decay = expf(-expf(alog[vh]) * sp);
            const float beta = 1.0f / (1.0f + expf(-b));
            ws[row * (c + 2 * hv) + c + 2 * vh] = decay;
            ws[row * (c + 2 * hv) + c + 2 * vh + 1] = beta;
        }
    } else {
        float ssq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        ssq = wredsum(ssq);
        const float inv = rsqrtf(ssq + EPS);
        if (head < h) {
            *(float4*)wsrow = make_float4(v[0] * inv * QSCALE, v[1] * inv * QSCALE,
                                          v[2] * inv * QSCALE, v[3] * inv * QSCALE);
        } else {
            *(float4*)wsrow = make_float4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
        }
    }
}

/* Streaming state pass per (token, value head): delta-rule update + gated
 * RMSNorm output.  grid (t, hv), block 256 (8 warps); warp w owns rows
 * w, w+8, ... of the 128x128 FP32 state; lane covers columns 4*lane..4*lane+3.
 * VEC=1 uses float4 (requires 16B alignment), VEC=0 scalar fallback. */
template <int VEC, int W, int PF>
__device__ __forceinline__ void gdn_state_body(
    const __nv_bfloat16* __restrict__ qkvz,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ nw,
    const int* __restrict__ slots,
    const float* __restrict__ ws,
    long long t, long long h, long long hv, long long c,
    long long qkvz_s0, long long out_s0, long long out_s1,
    long long st_sslot, long long slots_s, long long ratio, long long sw)
{
    const long long row = sw ? blockIdx.y : blockIdx.x;
    const long long vh = sw ? blockIdx.x : blockIdx.y;
    const int slot = slots[row * slots_s];
    if (slot <= 0) return;
    const long long kh = vh / ratio;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    __shared__ __align__(16) float smval[128];
    __shared__ __align__(16) float raws[128];
    __shared__ float red[W > 16 ? W : 16];
    const int j = lane * 4;
    float* base = state + slot * st_sslot + vh * 16384;
    /* PDL: prefetch the first row pair (pure kernel input) so its DRAM
     * latency runs concurrently with the producer kernel, then wait.
     * PF is a template switch: the prefetch costs +8 registers (vec16
     * 32->40, occupancy 4->3 blocks/SM), so it is instantiated only for
     * block counts where the occupancy loss cannot create a second wave. */
    float4 pfA, pfB, pfC, pfD;
    if (VEC && PF) {
        pfA = *(const float4*)(base + wid * 128 + j);
        pfB = *(const float4*)(base + (wid + W) * 128 + j);
        if (PF >= 2 && 2 * W < 128) {
            /* PF2: second row pair also in flight above the dependency wait.
             * Only instantiated for the tiny tier (vec32pf2, TH<=128): its
             * grid never exceeds one block/SM, so the +16 registers cost no
             * occupancy (vec16pf keeps PF1 -- matrix9 occupancy trap). */
            pfC = *(const float4*)(base + (wid + 2 * W) * 128 + j);
            pfD = *(const float4*)(base + (wid + 3 * W) * 128 + j);
        }
    }
    /* epilogue inputs are PURE kernel data (pre writes only the qkvz
     * scratch prefix; nw is never written): hoist the z-gate/norm_weight
     * loads above the dependency wait so the tail has no exposed DRAM
     * round-trip. PF tiers only (vec16 plain would lose a block/SM). */
    float zpf = 0.0f, nwpf = 0.0f;
    if (VEC && PF && tid < 128) {
        zpf = __bfloat162float(qkvz[row * qkvz_s0 + c + vh * 128 + tid]);
        nwpf = __bfloat162float(nw[tid]);
    }
    cudaGridDependencySynchronize();
    const float* wsrow = ws + row * (c + 2 * hv);
    const float4 k4 = *(const float4*)(wsrow + (h + kh) * 128 + j);
    const float4 q4 = *(const float4*)(wsrow + kh * 128 + j);
    if (tid < 128) smval[tid] = wsrow[2 * h * 128 + vh * 128 + tid];
    const float decay = wsrow[c + 2 * vh];
    const float beta = wsrow[c + 2 * vh + 1];
    __syncthreads();
#pragma unroll 4
    for (int rr = wid; rr < 128; rr += 2 * W) {
        const int rA = rr;
        const int rB = rr + W;
        float* pA = base + rA * 128 + j;
        float* pB = base + rB * 128 + j;
        float a0, a1, a2, a3, b0, b1, b2, b3;
        if (VEC) {
            const bool hit1 = (PF >= 1) && (rr == wid);
            const bool hit2 = (PF >= 2) && (rr == wid + 2 * W);
            const float4 vA = hit1 ? pfA : hit2 ? pfC : *(const float4*)pA;
            const float4 vB = hit1 ? pfB : hit2 ? pfD : *(const float4*)pB;
            a0 = vA.x; a1 = vA.y; a2 = vA.z; a3 = vA.w;
            b0 = vB.x; b1 = vB.y; b2 = vB.z; b3 = vB.w;
        } else {
            a0 = pA[0]; a1 = pA[1]; a2 = pA[2]; a3 = pA[3];
            b0 = pB[0]; b1 = pB[1]; b2 = pB[2]; b3 = pB[3];
        }
        const float dA0 = a0 * decay, dA1 = a1 * decay;
        const float dA2 = a2 * decay, dA3 = a3 * decay;
        const float dB0 = b0 * decay, dB1 = b1 * decay;
        const float dB2 = b2 * decay, dB3 = b3 * decay;
        float dotkA = dA0 * k4.x + dA1 * k4.y + dA2 * k4.z + dA3 * k4.w;
        float dotkB = dB0 * k4.x + dB1 * k4.y + dB2 * k4.z + dB3 * k4.w;
        dotkA = wredsum(dotkA);
        dotkB = wredsum(dotkB);
        const float resA = (smval[rA] - dotkA) * beta;
        const float resB = (smval[rB] - dotkB) * beta;
        a0 = fmaf(resA, k4.x, dA0); a1 = fmaf(resA, k4.y, dA1);
        a2 = fmaf(resA, k4.z, dA2); a3 = fmaf(resA, k4.w, dA3);
        b0 = fmaf(resB, k4.x, dB0); b1 = fmaf(resB, k4.y, dB1);
        b2 = fmaf(resB, k4.z, dB2); b3 = fmaf(resB, k4.w, dB3);
        float dotqA = a0 * q4.x + a1 * q4.y + a2 * q4.z + a3 * q4.w;
        float dotqB = b0 * q4.x + b1 * q4.y + b2 * q4.z + b3 * q4.w;
        dotqA = wredsum(dotqA);
        dotqB = wredsum(dotqB);
        if (lane == 0) {
            raws[rA] = __bfloat162float(__float2bfloat16_rn(dotqA));
            raws[rB] = __bfloat162float(__float2bfloat16_rn(dotqB));
        }
        if (VEC) {
            float4 vA; vA.x = a0; vA.y = a1; vA.z = a2; vA.w = a3;
            float4 vB; vB.x = b0; vB.y = b1; vB.z = b2; vB.w = b3;
            *(float4*)pA = vA;
            *(float4*)pB = vB;
        } else {
            pA[0] = a0; pA[1] = a1; pA[2] = a2; pA[3] = a3;
            pB[0] = b0; pB[1] = b1; pB[2] = b2; pB[3] = b3;
        }
    }
    __syncthreads();
    float part = 0.0f;
    if (tid < 128) { const float x = raws[tid]; part = x * x; }
    part = wredsum(part);
    if (lane == 0) red[wid] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < W; w++) s += red[w];
        red[0] = rsqrtf(s * (1.0f / 128.0f) + EPS);
    }
    __syncthreads();
    const float inv_rms = red[0];
    if (tid < 128) {
        const float gin = (VEC && PF) ? zpf
            : __bfloat162float(qkvz[row * qkvz_s0 + c + vh * 128 + tid]);
        const float g = __fdividef(1.0f, 1.0f + __expf(-gin));
        const float nwf = (VEC && PF) ? nwpf : __bfloat162float(nw[tid]);
        const float o = ((raws[tid] * inv_rms) * nwf) * g;
        out[row * out_s0 + vh * out_s1 + tid] = __float2bfloat16_rn(o);
    }
}

#define STATE_ARGS qkvz, out, state, nw, slots, ws, t, h, hv, c, \
    qkvz_s0, out_s0, out_s1, st_sslot, slots_s, ratio, sw
#define STATE_SIG \
    const __nv_bfloat16* __restrict__ qkvz, __nv_bfloat16* __restrict__ out, \
    float* __restrict__ state, const __nv_bfloat16* __restrict__ nw, \
    const int* __restrict__ slots, const float* __restrict__ ws, \
    long long t, long long h, long long hv, long long c, \
    long long qkvz_s0, long long out_s0, long long out_s1, \
    long long st_sslot, long long slots_s, long long ratio, long long sw

/* cp.async smem-staged variant: 8 tiles of 16 rows; triple-buffered;
 * thread (wid, lane) stages+consumes rows 16i+wid and 16i+wid+8, cols lane*4. */
extern "C" __global__ void gdn_state_tile8(STATE_SIG)
{
    const long long row = sw ? blockIdx.y : blockIdx.x;
    const long long vh = sw ? blockIdx.x : blockIdx.y;
    const int slot = slots[row * slots_s];
    if (slot <= 0) return;
    const float* wsrow = ws + row * (c + 2 * hv);
    const long long kh = vh / ratio;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    __shared__ __align__(16) float smval[128];
    __shared__ __align__(16) float raws[128];
    __shared__ float red[8];
    __shared__ __align__(8) unsigned long long mbar[4];
    extern __shared__ __align__(16) float tile[];  /* 4 * 16 * 128 floats */
    const int j = lane * 4;
    float* base = state + slot * st_sslot + vh * 16384;
    /* PDL: stage tiles 0 and 1 FIRST — state is a pure kernel input, so the
     * prefetch overlaps the producer kernel; ws may only be read after the
     * grid-dependency sync. */
    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(smem_u32(&mbar[b])) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
        bulk_stage8k(base, tile, &mbar[0]);
        bulk_stage8k(base + 2048, tile + 2048, &mbar[1]);
        bulk_stage8k(base + 4096, tile + 4096, &mbar[2]);
    }
    /* epilogue inputs are PURE kernel data (see v55 note): hoist z/nw
     * above the dependency wait; keeps the 8-iteration tail drain-free. */
    float zpf = 0.0f, nwpf = 0.0f;
    if (tid < 128) {
        zpf = __bfloat162float(qkvz[row * qkvz_s0 + c + vh * 128 + tid]);
        nwpf = __bfloat162float(nw[tid]);
    }
    cudaGridDependencySynchronize();
    const float4 k4 = *(const float4*)(wsrow + (h + kh) * 128 + j);
    const float4 q4 = *(const float4*)(wsrow + kh * 128 + j);
    if (tid < 128) smval[tid] = wsrow[2 * h * 128 + vh * 128 + tid];
    const float decay = wsrow[c + 2 * vh];
    const float beta = wsrow[c + 2 * vh + 1];
    __syncthreads();  /* smval ready */
    for (int i = 0; i < 8; i++) {
        const int b = i % 4;
        const int buf = b * 2048;
        /* order every thread's read of buffer (i+3)%4 (tile i-1) before the
         * elected thread refills it with the bulk copy */
        __syncthreads();
        if (tid == 0 && i + 3 < 8) {
            bulk_stage8k(base + (long long)(i + 3) * 2048,
                         tile + ((i + 3) % 4) * 2048, &mbar[(i + 3) % 4]);
        }
        mbar_wait(&mbar[b], (unsigned)((i / 4) & 1));
        const int rA = i * 16 + wid;
        const int rB = rA + 8;
        const float4 vA = *(const float4*)(tile + buf + wid * 128 + j);
        const float4 vB = *(const float4*)(tile + buf + (wid + 8) * 128 + j);
        float a0 = vA.x, a1 = vA.y, a2 = vA.z, a3 = vA.w;
        float b0 = vB.x, b1 = vB.y, b2 = vB.z, b3 = vB.w;
        const float dA0 = a0 * decay, dA1 = a1 * decay;
        const float dA2 = a2 * decay, dA3 = a3 * decay;
        const float dB0 = b0 * decay, dB1 = b1 * decay;
        const float dB2 = b2 * decay, dB3 = b3 * decay;
        float dotkA = dA0 * k4.x + dA1 * k4.y + dA2 * k4.z + dA3 * k4.w;
        float dotkB = dB0 * k4.x + dB1 * k4.y + dB2 * k4.z + dB3 * k4.w;
        dotkA = wredsum(dotkA);
        dotkB = wredsum(dotkB);
        const float resA = (smval[rA] - dotkA) * beta;
        const float resB = (smval[rB] - dotkB) * beta;
        a0 = fmaf(resA, k4.x, dA0); a1 = fmaf(resA, k4.y, dA1);
        a2 = fmaf(resA, k4.z, dA2); a3 = fmaf(resA, k4.w, dA3);
        b0 = fmaf(resB, k4.x, dB0); b1 = fmaf(resB, k4.y, dB1);
        b2 = fmaf(resB, k4.z, dB2); b3 = fmaf(resB, k4.w, dB3);
        float dotqA = a0 * q4.x + a1 * q4.y + a2 * q4.z + a3 * q4.w;
        float dotqB = b0 * q4.x + b1 * q4.y + b2 * q4.z + b3 * q4.w;
        dotqA = wredsum(dotqA);
        dotqB = wredsum(dotqB);
        if (lane == 0) {
            raws[rA] = __bfloat162float(__float2bfloat16_rn(dotqA));
            raws[rB] = __bfloat162float(__float2bfloat16_rn(dotqB));
        }
        float* pA = base + (long long)rA * 128 + j;
        float* pB = base + (long long)rB * 128 + j;
        float4 oA; oA.x = a0; oA.y = a1; oA.z = a2; oA.w = a3;
        float4 oB; oB.x = b0; oB.y = b1; oB.z = b2; oB.w = b3;
        *(float4*)pA = oA;
        *(float4*)pB = oB;
    }
    __syncthreads();
    float part = 0.0f;
    if (tid < 128) { const float x = raws[tid]; part = x * x; }
    part = wredsum(part);
    if (lane == 0) red[wid] = part;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++) ss += red[w];
        red[0] = rsqrtf(ss * (1.0f / 128.0f) + EPS);
    }
    __syncthreads();
    if (tid < 128) {
        const float inv_rms = red[0];
        const float g = __fdividef(1.0f, 1.0f + __expf(-zpf));
        const float o = ((raws[tid] * inv_rms) * nwpf) * g;
        out[row * out_s0 + vh * out_s1 + tid] = __float2bfloat16_rn(o);
    }
}

extern "C" __global__ void gdn_state_vec8(STATE_SIG) { gdn_state_body<1, 8, 0>(STATE_ARGS); }
extern "C" __global__ void gdn_state_sca8(STATE_SIG) { gdn_state_body<0, 8, 0>(STATE_ARGS); }
extern "C" __global__ void gdn_state_vec16(STATE_SIG) { gdn_state_body<1, 16, 0>(STATE_ARGS); }
extern "C" __global__ void gdn_state_vec16pf(STATE_SIG) { gdn_state_body<1, 16, 1>(STATE_ARGS); }
extern "C" __global__ void gdn_state_vec32pf(STATE_SIG) { gdn_state_body<1, 32, 1>(STATE_ARGS); }
extern "C" __global__ void gdn_state_vec32pf2(STATE_SIG) { gdn_state_body<1, 32, 2>(STATE_ARGS); }
extern "C" __global__ void gdn_state_sca16(STATE_SIG) { gdn_state_body<0, 16, 0>(STATE_ARGS); }

typedef unsigned long long u64;
#define NSIG 25
#define NSLOTS 2

static u64 g_seen_sig[NSLOTS][NSIG];
static int g_seen_count[NSLOTS];
static u64 g_exec_sig[NSLOTS][NSIG];
static cudaGraphExec_t g_exec[NSLOTS];
static int g_exec_valid[NSLOTS];
static u64 g_tick[NSLOTS];
static u64 g_clock;
static cudaStream_t g_cap;
static int g_graphs_off;

static int launch_both(const u64* a, cudaStream_t s)
{
    const void* qkvz = (const void*)a[0];
    const void* ba = (const void*)a[1];
    const void* cw = (const void*)a[2];
    const void* conv = (const void*)a[3];
    const void* alog = (const void*)a[4];
    const void* dtb = (const void*)a[5];
    void* state = (void*)a[6];
    const void* nw = (const void*)a[7];
    void* out = (void*)a[8];
    const void* slots = (const void*)a[9];
    void* ws = (void*)a[10];
    const long long t = (long long)a[11];
    const long long h = (long long)a[12];
    const long long hv = (long long)a[13];
    const long long c = (long long)a[14];
    const long long qkvz_s0 = (long long)a[15];
    const long long ba_s0 = (long long)a[16];
    const long long out_s0 = (long long)a[17];
    const long long out_s1 = (long long)a[18];
    const long long conv_sslot = (long long)a[19];
    const long long conv_sch = (long long)a[20];
    const long long conv_scol = (long long)a[21];
    const long long cw_s0 = (long long)a[22];
    const long long st_sslot = (long long)a[23];
    const long long slots_s = (long long)a[24];
    const long long heads = c / 128;
    #define PRE_ARGS \
        (__nv_bfloat16*)qkvz, (const __nv_bfloat16*)ba, (const __nv_bfloat16*)cw, \
        (const __nv_bfloat16*)conv, (__nv_bfloat16*)conv, (__nv_bfloat16*)out, \
        (const int*)slots, (const float*)alog, (const __nv_bfloat16*)dtb, \
        (float*)ws, t, h, hv, c, qkvz_s0, ba_s0, out_s0, out_s1, \
        conv_sslot, conv_sch, conv_scol, cw_s0, slots_s
    /* pre8 aggregates 8 heads per block; require enough resulting blocks to
     * fill the machine (heads>=16) or enough rows (t>=256).  d4-class shapes
     * (few heads, moderate t) keep the finer pre1 grid: measured +9% there
     * with pre8's 128-block launch. */
    if (t * heads >= 1024 && (heads >= 16 || t >= 256)) {
        const dim3 g1((unsigned)t, (unsigned)((heads + 7) / 8));
        gdn_pre8<<<g1, 256, 0, s>>>(PRE_ARGS);
    } else {
        const dim3 g1((unsigned)t, (unsigned)heads);
        gdn_pre1<<<g1, 128, 0, s>>>(PRE_ARGS);
    }
    #undef PRE_ARGS
    const bool vec = ((st_sslot % 4) == 0) && ((((u64)state) & 15ull) == 0ull);
    const long long TH = t * hv;
    const bool small = TH <= 512;
    /* vh-fastest for ALL non-small tiers (v40's single-draw pre-zpf
     * rejection revisited under lottery-aware methodology: matrix28/29
     * 12/12 fast mid medians vs ctl 9/12, new min floors, 0 regressions
     * in 13 case-jobs; c1-class was already swapped). */
    const long long sw = (TH > 512) ? 1 : 0;
    dim3 g2 = sw ? dim3((unsigned)hv, (unsigned)t) : dim3((unsigned)t, (unsigned)hv);
    /* PDL consumer launches: programmatic stream serialization lets the
     * state kernel's blocks schedule (and tile8 prefetch pure-input state
     * tiles) while the pre kernel is still running; each state kernel calls
     * cudaGridDependencySynchronize() before reading pre outputs. */
    #define ST_CALL(K, SMEM, NTHR) do { \
        cudaLaunchConfig_t cfg = {}; \
        cudaLaunchAttribute lat[1]; \
        cfg.gridDim = g2; \
        cfg.blockDim = dim3((unsigned)(NTHR)); \
        cfg.dynamicSmemBytes = (SMEM); \
        cfg.stream = s; \
        lat[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
        lat[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = lat; \
        cfg.numAttrs = 1; \
        cudaError_t le = cudaLaunchKernelEx(&cfg, K, \
            (const __nv_bfloat16*)qkvz, (__nv_bfloat16*)out, (float*)state, \
            (const __nv_bfloat16*)nw, (const int*)slots, (const float*)ws, \
            t, h, hv, c, qkvz_s0, out_s0, out_s1, st_sslot, slots_s, hv / h, sw); \
        if (le != cudaSuccess) return (int)le; \
    } while (0)
    static int g_nsm = 0;
    if (g_nsm == 0) {
        int dev = 0, nsm = 0;
        if (cudaGetDevice(&dev) == cudaSuccess
            && cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess
            && nsm > 0) {
            g_nsm = nsm;
        } else {
            g_nsm = 1;
            cudaGetLastError();
        }
    }
    if (vec) {
        if (small) {
            if (TH <= 128LL) { ST_CALL(gdn_state_vec32pf2, 0, 1024); }
            else if (TH <= (long long)g_nsm * 3) { ST_CALL(gdn_state_vec16pf, 0, 512); }
            else { ST_CALL(gdn_state_vec16, 0, 512); }
        }
        else { ST_CALL(gdn_state_tile8, 32768, 256); }
    }
    else { if (small) { ST_CALL(gdn_state_sca16, 0, 512); } else { ST_CALL(gdn_state_sca8, 0, 256); } }
    #undef ST_CALL
    return (int)cudaGetLastError();
}

static int run_sig(const u64* sig, cudaStream_t s)
{
    const long long t = (long long)sig[11];
    if (t <= 0) return 0;
    g_clock++;
    for (int i = 0; i < NSLOTS; i++) {
        if (g_exec_valid[i] && memcmp(sig, g_exec_sig[i], NSIG * sizeof(u64)) == 0) {
            g_tick[i] = g_clock;
            return (int)cudaGraphLaunch(g_exec[i], s);
        }
    }
    const int rc = launch_both(sig, s);
    if (rc != 0 || g_graphs_off) return rc;
    int idx = -1;
    for (int i = 0; i < NSLOTS; i++) {
        if (g_seen_count[i] > 0 && memcmp(sig, g_seen_sig[i], NSIG * sizeof(u64)) == 0) {
            idx = i; g_seen_count[i]++; break;
        }
    }
    if (idx < 0) {
        idx = (g_seen_count[0] <= g_seen_count[1]) ? 0 : 1;
        memcpy(g_seen_sig[idx], sig, NSIG * sizeof(u64));
        g_seen_count[idx] = 1;
    }
    if (g_seen_count[idx] < 2) return 0;
    int slot = -1;
    for (int i = 0; i < NSLOTS; i++) if (!g_exec_valid[i]) { slot = i; break; }
    if (slot < 0) {
        slot = (g_tick[0] < g_tick[1]) ? 0 : 1;
        cudaGraphExecDestroy(g_exec[slot]);
        g_exec_valid[slot] = 0;
    }
    if (!g_cap && cudaStreamCreateWithFlags(&g_cap, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        g_graphs_off = 1;
        return 0;
    }
    cudaGraph_t graph = NULL;
    cudaError_t e = cudaStreamBeginCapture(g_cap, cudaStreamCaptureModeThreadLocal);
    if (e == cudaSuccess) {
        launch_both(sig, g_cap);
        cudaError_t e2 = cudaStreamEndCapture(g_cap, &graph);
        if (e2 != cudaSuccess) e = e2;
    }
    if (e == cudaSuccess && graph != NULL) {
        cudaGraphExec_t exec = NULL;
        e = cudaGraphInstantiate(&exec, graph, 0ull);
        cudaGraphDestroy(graph);
        if (e == cudaSuccess && exec != NULL) {
            memcpy(g_exec_sig[slot], sig, NSIG * sizeof(u64));
            g_exec[slot] = exec;
            g_tick[slot] = g_clock;
            g_exec_valid[slot] = 1;
            g_seen_count[idx] = 0;
            return 0;
        }
    } else if (graph != NULL) {
        cudaGraphDestroy(graph);
    }
    cudaGetLastError();
    g_graphs_off = 1;
    return 0;
}

extern "C" int gdn_run(const unsigned long long* sig, void* stream)
{
    return run_sig(sig, (cudaStream_t)stream);
}
"""

_FASTCALL_SRC = r"""
#include <Python.h>
#include <dlfcn.h>

typedef int (*gdn_run_fn)(const unsigned long long*, void*);
static gdn_run_fn g_run = NULL;

static PyObject* fc_bind(PyObject* self, PyObject* arg) {
    const char* path = PyUnicode_AsUTF8(arg);
    if (!path) return NULL;
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        PyErr_Format(PyExc_RuntimeError, "dlopen failed: %s", dlerror());
        return NULL;
    }
    g_run = (gdn_run_fn)dlsym(handle, "gdn_run");
    if (!g_run) {
        PyErr_Format(PyExc_RuntimeError, "dlsym gdn_run failed: %s", dlerror());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* fc_run(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    const unsigned long long sig = PyLong_AsUnsignedLongLong(args[0]);
    void* stream = (void*)(uintptr_t)PyLong_AsUnsignedLongLong(args[1]);
    if ((sig == (unsigned long long)-1 || stream == (void*)(uintptr_t)-1)
            && PyErr_Occurred()) {
        return NULL;
    }
    const int rc = g_run((const unsigned long long*)(uintptr_t)sig, stream);
    return PyLong_FromLong(rc);
}

static PyMethodDef fc_methods[] = {
    {"bind", fc_bind, METH_O, NULL},
    {"run", (PyCFunction)fc_run, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fc_module = {
    PyModuleDef_HEAD_INIT, "gdn_fastcall", NULL, -1, fc_methods
};

PyMODINIT_FUNC PyInit_gdn_fastcall(void) { return PyModule_Create(&fc_module); }
"""


def _build_fastcall(cuda_lib_path, root, tag):
    """Best-effort METH_FASTCALL binder; returns run callable or None."""
    try:
        import importlib.util
        import sysconfig
        inc = sysconfig.get_paths()['include']
        if not (Path(inc) / 'Python.h').exists():
            return None
        so = root / f'gdn_fastcall_{tag}.so'
        if not so.exists():
            csrc = root / f'gdn_fastcall_{tag}.c'
            csrc.write_text(_FASTCALL_SRC)
            cc = shutil.which('gcc') or shutil.which('cc') or sysconfig.get_config_var('CC')
            cmd = [cc, '-O2', '-shared', '-fPIC', f'-I{inc}', str(csrc),
                   '-o', str(so) + '.tmp', '-ldl']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            (root / f'gdn_fastcall_{tag}.build.log').write_text(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                return None
            os.replace(str(so) + '.tmp', so)
        spec = importlib.util.spec_from_file_location('gdn_fastcall', str(so))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.bind(str(cuda_lib_path))
        return mod.run
    except Exception:
        return None


_EXT = None


def _build_ext():
    """Compile the CUDA source with nvcc once per source hash; load via ctypes."""
    global _EXT
    if _EXT is not None:
        return _EXT
    tag = hashlib.sha256(_CUDA_SOURCE.encode()).hexdigest()[:16]
    base = os.environ.get('TORCH_EXTENSIONS_DIR')
    root = Path(base) if base else Path(tempfile.gettempdir()) / 'gdn33_cache'
    root = root / 'gdn_decode_core_bf16_state_fp32'
    root.mkdir(parents=True, exist_ok=True)
    lib = root / f'gdn_{tag}.so'
    if not lib.exists():
        src = root / f'gdn_{tag}.cu'
        src.write_text(_CUDA_SOURCE)
        maj, mnr = torch.cuda.get_device_capability(0)
        arch = f'sm_{maj}{mnr}'
        nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        cmd = [nvcc, '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
               f'-arch={arch}', '-lineinfo', str(src), '-o', str(lib) + '.tmp']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        (root / f'gdn_{tag}.build.log').write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f'nvcc failed ({proc.returncode}):\n{proc.stderr[-4000:]}')
        os.replace(str(lib) + '.tmp', lib)
    handle = ctypes.CDLL(str(lib))
    fn = handle.gdn_run
    fn.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p]
    fn.restype = ctypes.c_int
    fast = _build_fastcall(lib, root, tag)
    _EXT = (handle, fn, fast)
    return _EXT


def _raw_stream(index):
    getter = getattr(torch._C, '_cuda_getCurrentRawStream', None)
    if getter is not None:
        return getter(index)
    return torch.cuda.current_stream(index).cuda_stream


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ext = _build_ext()
        self._fn = ext[1]
        self._fastrun = ext[2]
        self._ws = None
        self._ws_numel = -1
        self._checked = False
        self._cache = OrderedDict()

    def _validate(self, qkvz, ba, conv_weight, conv_state, A_log, dt_bias,
                  state, norm_weight, out, slots, t, h, hv, c):
        bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        checks = [
            (qkvz.dtype is bf16 and qkvz.stride(1) == 1 and qkvz.stride(0) % 4 == 0, 'qkvz layout'),
            (ba.dtype is bf16 and ba.stride(1) == 1, 'ba layout'),
            (conv_weight.dtype is bf16 and conv_weight.stride(1) == 1, 'conv_weight layout'),
            (conv_state.dtype is bf16 and conv_state.stride(1) == 1, 'conv_state layout'),
            (A_log.dtype is f32 and A_log.stride(0) == 1 and A_log.numel() >= hv, 'A_log layout'),
            (dt_bias.dtype is bf16 and dt_bias.stride(0) == 1, 'dt_bias layout'),
            (state.dtype is f32 and state.stride(1) == 16384 and state.stride(2) == 128
             and state.stride(3) == 1 and state.shape[2:] == (128, 128), 'state layout'),
            (norm_weight.dtype is bf16 and norm_weight.stride(0) == 1 and norm_weight.numel() >= 128,
             'norm_weight layout'),
            (out.dtype is bf16 and out.stride(2) == 1 and out.shape[1:] == (hv, 128)
             and out.stride(0) % 4 == 0 and out.stride(1) % 4 == 0, 'out layout'),
            (slots.dtype is i32 and slots.dim() == 1 and slots.shape[0] >= t, 'slots layout'),
            (hv % h == 0 and h > 0, 'head ratio'),
            (hv % 2 == 0 and (c + 2 * hv) % 4 == 0, 'workspace alignment'),
            (qkvz.shape[1] == c + hv * 128, 'qkvz width'),
            (qkvz.shape[0] >= t and out.shape[0] >= t and ba.shape[0] >= t, 'row counts'),
            (qkvz.is_cuda and state.is_cuda and out.is_cuda, 'device'),
        ]
        for ok, name in checks:
            if not ok:
                raise RuntimeError(f'Unsupported input geometry: {name}')

    def _prepare(self, key, qkvz, ba, conv_weight, conv_state, A_log, dt_bias,
                 state, norm_weight, out, metadata, slots, t):
        hv = state.shape[1]
        c = conv_weight.shape[0]
        h = (c - hv * 128) // 256
        if not self._checked:
            self._validate(qkvz, ba, conv_weight, conv_state, A_log, dt_bias,
                           state, norm_weight, out, slots, t, h, hv, c)
            self._checked = True
        need = t * (c + 2 * hv)
        if self._ws is None or self._ws_numel < need or self._ws.device != state.device:
            self._ws = torch.empty(need, dtype=torch.float32, device=state.device)
            self._ws_numel = need
        index = state.device.index
        if index is None:
            index = torch.cuda.current_device()
        sig = (ctypes.c_ulonglong * 25)(
            qkvz.data_ptr(), ba.data_ptr(), conv_weight.data_ptr(),
            conv_state.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
            state.data_ptr(), norm_weight.data_ptr(), out.data_ptr(),
            slots.data_ptr(), self._ws.data_ptr(),
            t, h, hv, c,
            qkvz.stride(0), ba.stride(0), out.stride(0), out.stride(1),
            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            conv_weight.stride(0), state.stride(0), slots.stride(0))
        refs = (qkvz, ba, conv_weight, conv_state, A_log, dt_bias, state,
                norm_weight, out, slots, metadata, self._ws, sig)
        entry = (sig, index, refs, ctypes.addressof(sig))
        self._cache[key] = entry
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return entry

    def __call__(self, qkvz, ba, conv_weight, conv_state, A_log, dt_bias,
                 state, norm_weight, out, metadata):
        t = int(metadata['num_actual_tokens'])
        if t <= 0:
            return None
        slots = metadata['non_spec_state_indices_tensor']
        key = (id(qkvz), id(conv_state), id(state), id(out), id(slots), t)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._prepare(key, qkvz, ba, conv_weight, conv_state, A_log,
                                  dt_bias, state, norm_weight, out, metadata,
                                  slots, t)
        stream = _raw_stream(entry[1])
        if self._fastrun is not None:
            code = self._fastrun(entry[3], stream)
        else:
            code = self._fn(entry[0], stream)
        if code != 0:
            raise RuntimeError(f'gdn_forward launch error {code}')
        return None

    forward = __call__
