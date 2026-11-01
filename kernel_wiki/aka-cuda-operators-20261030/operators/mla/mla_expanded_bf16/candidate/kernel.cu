// mla_expanded_bf16 — independent CUDA implementation for L20D (SM103). v2.
//
// Ragged (varlen) attention forward for the frozen D07 contract:
//   Q [total_q, heads, 192] bf16, K [total_kv, kv_heads, 192] bf16,
//   V [total_kv, kv_heads, 128] bf16, out [total_q, heads, 128] bf16,
//   optional lse [total_q, heads] fp32 in log-base-2, optional per-head sinks.
//   General right-aligned causal (visible = kv_len - q_len + i) matching
//   problem/reference.py, plus non-causal and window_left >= 0 support.
//
// Structure: FlashAttention-2 style tiled online softmax, templated on
// (BM, BN, STAGES, REUSE_Q_STAGE). cp.async multi-stage K/V pipeline;
// mma.sync m16n8k16 bf16 -> fp32; log2-domain softmax with fused fma scaling;
// stmatrix epilogue. v2 adds: multi-stage pipeline, Q-stage reuse for tight
// smem configs, and host-side config dispatch (SM-count based, no case
// recognition). MLA_CFG env (experiments only) forces a config index.
//
// No libraries: raw CUDA C++ / PTX (mma.sync, ldmatrix, stmatrix, cp.async).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>

#define DINLINE __device__ __forceinline__

namespace mla {

constexpr int DQK = 192;              // Q/K head dim
constexpr int DV  = 128;              // V head dim
constexpr int QK_CHUNKS = DQK * 2 / 16;   // 24 x 16B chunks per K/Q row
constexpr int V_CHUNKS  = DV * 2 / 16;     // 16 x 16B chunks per V row
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;   // "masked" score; finite so exp2 -> 0

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

DINLINE void cp_async16(uint32_t smem, const void* gmem, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(smem), "l"(gmem), "r"(src_size));
}
DINLINE void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DINLINE void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DINLINE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DINLINE void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                         uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               :"=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               :"r"(addr));
}
DINLINE void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                               uint32_t& r3, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      :"=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      :"r"(addr));
}
DINLINE void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2,
                         uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
               ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

DINLINE void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                      uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      :"+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      :"r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

DINLINE float exp2_fast(float x) { return exp2f(x); }

DINLINE uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// ---------------------------------------------------------------------------
// Swizzled shared-memory tile: ROWS rows x CHUNKS 16B-chunks per row,
// physical chunk = c ^ (row & 7) (128B swizzle), row-major.
// ---------------------------------------------------------------------------

template <int ROWS, int CHUNKS>
struct SmemTile {
  static constexpr int BYTES = ROWS * CHUNKS * 16;
  DINLINE static uint32_t offset(int row, int chunk) {
    return static_cast<uint32_t>(row * CHUNKS * 16 + ((chunk ^ (row & 7)) * 16));
  }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int BM, int BN, int STAGES, bool REUSE_Q_STAGE>
__global__ void __launch_bounds__((BM / 16) * 32, 1)
mla_kernel(const __nv_bfloat16* __restrict__ q,      // [total_q, heads, 192]
           const __nv_bfloat16* __restrict__ k,      // [total_kv, kvh, 192]
           const __nv_bfloat16* __restrict__ v,      // [total_kv, kvh, 128]
           __nv_bfloat16* __restrict__ out,          // [total_q, heads, 128]
           float* __restrict__ lse,                  // [total_q, heads] or null
           const float* __restrict__ sinks,          // [heads] or null
           const int32_t* __restrict__ cum_q,        // [batch+1]
           const int32_t* __restrict__ cum_kv,       // [batch+1]
           int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
           int32_t window_left, int32_t causal, int32_t has_lse,
           int32_t has_sink, float bmm1_scale, float bmm2_scale) {
  constexpr int WARPS = BM / 16;
  constexpr int THREADS = WARPS * 32;
  constexpr int QK_STEPS = DQK / 16;        // 12 k16 steps
  constexpr int NBLK = BN / 8;              // S n8 blocks per tile
  constexpr int KV_STEPS = BN / 16;         // PV k16 steps
  constexpr int DVBLK = DV / 8;             // O n8 blocks (16)
  constexpr int STAGE_CHUNKS = DV * 2 / 16 + 1;  // 17-chunk (272B) stage rows

  using QTile = SmemTile<BM, QK_CHUNKS>;
  using KTile = SmemTile<BN, QK_CHUNKS>;
  using VTile = SmemTile<BN, V_CHUNKS>;

  extern __shared__ char smem[];
  char* sq = smem;                                   // BM x 192
  char* sk = smem + QTile::BYTES;                    // STAGES x (BN x 192)
  char* sv = sk + STAGES * KTile::BYTES;             // STAGES x (BN x 128)
  char* so = REUSE_Q_STAGE ? sq : sv + STAGES * VTile::BYTES;  // BM x 272

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tile_id = blockIdx.x;
  const int h = blockIdx.y;

  // ---- Locate (batch b, q-tile m0) for this linear tile id ----------------
  int b = -1, m0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0;
    int prev_q = cum_q[0];
    int prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1];
      int nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && tile_id < acc + tiles) {
        b = i;
        m0 = (tile_id - acc) * BM;
        q_base = prev_q;
        kv_base = prev_kv;
        qlen = len_q;
        kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles;
      prev_q = nq;
      prev_kv = nkv;
    }
  }
  if (b < 0) return;  // surplus CTA

  const int group_div = heads / kv_heads;  // == 1 across the published domain
  const int h_kv = h / group_div;

  const __nv_bfloat16* q_base_ptr = q + (static_cast<int64_t>(q_base) * heads + h) * DQK;
  const __nv_bfloat16* k_base_ptr = k + (static_cast<int64_t>(kv_base) * kv_heads + h_kv) * DQK;
  const __nv_bfloat16* v_base_ptr = v + (static_cast<int64_t>(kv_base) * kv_heads + h_kv) * DV;
  __nv_bfloat16* out_base_ptr = out + (static_cast<int64_t>(q_base) * heads + h) * DV;
  float* lse_base_ptr = has_lse ? lse + q_base * heads + h : nullptr;

  const int64_t q_row_stride = static_cast<int64_t>(heads) * DQK;
  const int64_t kv_row_stride = static_cast<int64_t>(kv_heads) * DQK;
  const int64_t v_row_stride = static_cast<int64_t>(kv_heads) * DV;
  const int64_t out_row_stride = static_cast<int64_t>(heads) * DV;
  const int64_t lse_row_stride = heads;

  // ---- KV iteration range --------------------------------------------------
  const int kv_off = kvlen - qlen;                 // right alignment (0 in domain)
  int n_max = causal ? min(kvlen, kv_off + m0 + BM) : kvlen;
  if (n_max > kvlen) n_max = kvlen;
  const bool windowed = window_left >= 0;
  int n_safe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
  if (windowed) n_safe = 0;
  const int T = (n_max + BN - 1) / BN;
  const int T_full = n_safe / BN;                  // unmasked tile count
  if (T <= 0) return;                              // unreachable (kvlen>=1)

  const float sc2 = bmm1_scale * LOG2E;            // score -> log2 domain
  const float sink2 = has_sink ? sinks[h] * LOG2E : 0.0f;

  // ---- async tile loaders --------------------------------------------------
  auto load_q = [&]() {
    constexpr int NCH = BM * QK_CHUNKS;
    #pragma unroll
    for (int it = 0; it < (NCH + THREADS - 1) / THREADS; ++it) {
      int idx = tid + it * THREADS;
      if (NCH % THREADS == 0 || idx < NCH) {
        int row = idx / QK_CHUNKS;
        int c = idx % QK_CHUNKS;
        const void* src = q_base_ptr + static_cast<int64_t>(m0 + row) * q_row_stride + c * 8;
        int sz = (m0 + row < qlen) ? 16 : 0;
        cp_async16(smem_u32(sq + QTile::offset(row, c)), src, sz);
      }
    }
  };
  auto load_kv = [&](int n0, int stage) {
    char* kd = sk + stage * KTile::BYTES;
    char* vd = sv + stage * VTile::BYTES;
    constexpr int NK = BN * QK_CHUNKS;
    constexpr int NV = BN * V_CHUNKS;
    #pragma unroll
    for (int it = 0; it < (NK + THREADS - 1) / THREADS; ++it) {
      int idx = tid + it * THREADS;
      if (NK % THREADS == 0 || idx < NK) {
        int row = idx / QK_CHUNKS;
        int c = idx % QK_CHUNKS;
        const void* src = k_base_ptr + static_cast<int64_t>(n0 + row) * kv_row_stride + c * 8;
        int sz = (n0 + row < kvlen) ? 16 : 0;
        cp_async16(smem_u32(kd + KTile::offset(row, c)), src, sz);
      }
    }
    #pragma unroll
    for (int it = 0; it < (NV + THREADS - 1) / THREADS; ++it) {
      int idx = tid + it * THREADS;
      if (NV % THREADS == 0 || idx < NV) {
        int row = idx / V_CHUNKS;
        int c = idx % V_CHUNKS;
        const void* src = v_base_ptr + static_cast<int64_t>(n0 + row) * v_row_stride + c * 8;
        int sz = (n0 + row < kvlen) ? 16 : 0;
        cp_async16(smem_u32(vd + VTile::offset(row, c)), src, sz);
      }
    }
  };

  // ---- prologue: Q + first min(STAGES-1, T) KV tiles, one group per tile ---
  load_q();
  load_kv(0, 0);
  cp_commit();
  #pragma unroll
  for (int s = 1; s < STAGES - 1; ++s) {
    if (s < T) {
      load_kv(s * BN, s);
      cp_commit();
    }
  }

  // ---- per-thread row bookkeeping ------------------------------------------
  const int r0 = warp * 16 + (lane >> 2);        // fragment row (of 16)
  const int r1 = r0 + 8;
  const int c0 = (lane & 3) * 2;                 // fragment col base within n8
  const int vis0 = kv_off + m0 + r0;             // causal-visible kv index
  const int vis1 = kv_off + m0 + r1;
  const int wlo0 = windowed ? vis0 - window_left : INT32_MIN;
  const int wlo1 = windowed ? vis1 - window_left : INT32_MIN;

  float mr_0 = NEG_BIG, mr_1 = NEG_BIG;          // running row max (raw score)
  float d_0 = 0.0f, d_1 = 0.0f;                  // running denominators
  float o[DVBLK][4];                             // O accumulator
  #pragma unroll
  for (int i = 0; i < DVBLK; ++i) {
    o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  }

  uint32_t qf[4];                                  // per-k-step Q fragment

  auto q_ldsm_addr = [&](int kk) -> uint32_t {
    int grp = lane >> 3;
    int row = warp * 16 + (grp & 1) * 8 + (lane & 7);
    int chunk = 2 * kk + (grp >> 1);
    return smem_u32(sq + QTile::offset(row, chunk));
  };

  // ---- main loop -------------------------------------------------------------
  for (int n = 0; n < T; ++n) {
    int stage = n;
    // stage %= STAGES (STAGES is 2 or 3; cheap runtime modulo)
    stage = (STAGES == 2) ? (n & 1) : (n % STAGES);
    const int rem = T - 1 - n;
    if (n + STAGES - 1 < T) {
      int nstage = (STAGES == 2) ? ((n + 1) & 1) : ((n + STAGES - 1) % STAGES);
      load_kv((n + STAGES - 1) * BN, nstage);
      cp_commit();
    }
    if (rem >= STAGES - 1) {
      cp_wait<STAGES - 1>();
    } else if constexpr (STAGES >= 3) {
      if (rem == STAGES - 2) cp_wait<STAGES - 2>();
      else cp_wait<0>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    // ---- S = Q K^T (Q fragments re-read per k-step: keeps ~48 regs free) ----
    float s[NBLK][4];
    #pragma unroll
    for (int i = 0; i < NBLK; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
    {
      const char* kb = sk + stage * KTile::BYTES;
      #pragma unroll
      for (int kk = 0; kk < QK_STEPS; ++kk) {
        ldmatrix_x4(qf[0], qf[1], qf[2], qf[3], q_ldsm_addr(kk));
        #pragma unroll
        for (int nb = 0; nb < NBLK / 2; ++nb) {
          uint32_t b0, b1, b2, b3;
          int grp = lane >> 3;
          int row = nb * 16 + (grp >> 1) * 8 + (lane & 7);
          int chunk = 2 * kk + (grp & 1);
          ldmatrix_x4(b0, b1, b2, b3, smem_u32(kb) + KTile::offset(row, chunk));
          mma_bf16(s[2 * nb][0], s[2 * nb][1], s[2 * nb][2], s[2 * nb][3],
                   qf[0], qf[1], qf[2], qf[3], b0, b1);
          mma_bf16(s[2 * nb + 1][0], s[2 * nb + 1][1], s[2 * nb + 1][2], s[2 * nb + 1][3],
                   qf[0], qf[1], qf[2], qf[3], b2, b3);
        }
      }
    }

    // ---- mask (tail/diagonal tiles only), row max ----
    const int n0 = n * BN;
    if (n >= T_full) {
      #pragma unroll
      for (int i = 0; i < NBLK; ++i) {
        int j0 = n0 + i * 8 + c0;
        int j1 = j0 + 1;
        if (j0 >= kvlen || (causal && j0 > vis0) || (windowed && j0 < wlo0)) s[i][0] = NEG_BIG;
        if (j1 >= kvlen || (causal && j1 > vis0) || (windowed && j1 < wlo0)) s[i][1] = NEG_BIG;
        if (j0 >= kvlen || (causal && j0 > vis1) || (windowed && j0 < wlo1)) s[i][2] = NEG_BIG;
        if (j1 >= kvlen || (causal && j1 > vis1) || (windowed && j1 < wlo1)) s[i][3] = NEG_BIG;
      }
    }

    float tm0 = NEG_BIG, tm1 = NEG_BIG;
    #pragma unroll
    for (int i = 0; i < NBLK; ++i) {
      tm0 = fmaxf(tm0, fmaxf(s[i][0], s[i][1]));
      tm1 = fmaxf(tm1, fmaxf(s[i][2], s[i][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));

    const float nr_0 = fmaxf(mr_0, tm0);
    const float nr_1 = fmaxf(mr_1, tm1);
    const float nm2_0 = nr_0 * sc2;
    const float nm2_1 = nr_1 * sc2;
    const float alpha0 = exp2_fast(fmaf(mr_0, sc2, -nm2_0));
    const float alpha1 = exp2_fast(fmaf(mr_1, sc2, -nm2_1));
    mr_0 = nr_0;
    mr_1 = nr_1;

    // ---- P = exp2(s*sc2 - m2), row sums, bf16 A fragments ----
    float rs0 = 0.0f, rs1 = 0.0f;
    uint32_t pa[KV_STEPS][4];
    #pragma unroll
    for (int i = 0; i < NBLK; ++i) {
      float p0 = exp2_fast(fmaf(s[i][0], sc2, -nm2_0));
      float p1 = exp2_fast(fmaf(s[i][1], sc2, -nm2_0));
      float p2 = exp2_fast(fmaf(s[i][2], sc2, -nm2_1));
      float p3 = exp2_fast(fmaf(s[i][3], sc2, -nm2_1));
      rs0 += p0 + p1;
      rs1 += p2 + p3;
      int kk = i / 2;
      int half = i % 2;
      pa[kk][half * 2 + 0] = pack_bf16(p0, p1);
      pa[kk][half * 2 + 1] = pack_bf16(p2, p3);
    }
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);

    d_0 = d_0 * alpha0 + rs0;
    d_1 = d_1 * alpha1 + rs1;

    // ---- O = O * alpha + P V ----
    #pragma unroll
    for (int i = 0; i < DVBLK; ++i) {
      o[i][0] *= alpha0; o[i][1] *= alpha0;
      o[i][2] *= alpha1; o[i][3] *= alpha1;
    }
    {
      const char* vb = sv + stage * VTile::BYTES;
      #pragma unroll
      for (int kk = 0; kk < KV_STEPS; ++kk) {
        #pragma unroll
        for (int nb = 0; nb < DVBLK / 2; ++nb) {
          uint32_t b0, b1, b2, b3;
          int grp = lane >> 3;
          int row = kk * 16 + (grp & 1) * 8 + (lane & 7);
          int chunk = 2 * nb + (grp >> 1);
          ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vb) + VTile::offset(row, chunk));
          mma_bf16(o[2 * nb][0], o[2 * nb][1], o[2 * nb][2], o[2 * nb][3],
                   pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
          mma_bf16(o[2 * nb + 1][0], o[2 * nb + 1][1], o[2 * nb + 1][2], o[2 * nb + 1][3],
                   pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b2, b3);
        }
      }
    }
    __syncthreads();
  }

  // ---- epilogue -------------------------------------------------------------
  const float m2_0 = mr_0 * sc2;
  const float m2_1 = mr_1 * sc2;
  float inv0 = d_0 > 0.0f ? 1.0f / d_0 : 0.0f;
  float inv1 = d_1 > 0.0f ? 1.0f / d_1 : 0.0f;
  float lse0 = m2_0 + log2f(d_0);
  float lse1 = m2_1 + log2f(d_1);
  float extra0 = 1.0f, extra1 = 1.0f;
  if (has_sink) {
    float t0 = fmaxf(lse0, sink2) + log2f(1.0f + exp2_fast(-fabsf(lse0 - sink2)));
    float t1 = fmaxf(lse1, sink2) + log2f(1.0f + exp2_fast(-fabsf(lse1 - sink2)));
    extra0 = exp2_fast(lse0 - t0);
    extra1 = exp2_fast(lse1 - t1);
    lse0 = t0;
    lse1 = t1;
  }
  const float g0 = bmm2_scale * inv0 * extra0;
  const float g1 = bmm2_scale * inv1 * extra1;

  if (has_lse && (lane & 3) == 0) {
    if (m0 + r0 < qlen) lse_base_ptr[static_cast<int64_t>(m0 + r0) * lse_row_stride] = lse0;
    if (m0 + r1 < qlen) lse_base_ptr[static_cast<int64_t>(m0 + r1) * lse_row_stride] = lse1;
  }

  __syncthreads();  // sq safe to overwrite when REUSE_Q_STAGE
  #pragma unroll
  for (int j = 0; j < DVBLK / 2; ++j) {
    uint32_t r0r = pack_bf16(o[2 * j][0] * g0, o[2 * j][1] * g0);
    uint32_t r1r = pack_bf16(o[2 * j][2] * g1, o[2 * j][3] * g1);
    uint32_t r2r = pack_bf16(o[2 * j + 1][0] * g0, o[2 * j + 1][1] * g0);
    uint32_t r3r = pack_bf16(o[2 * j + 1][2] * g1, o[2 * j + 1][3] * g1);
    int grp = lane >> 3;
    int row = warp * 16 + (grp & 1) * 8 + (lane & 7);
    int chunk = 2 * j + (grp >> 1);
    stmatrix_x4(smem_u32(so + row * STAGE_CHUNKS * 16 + chunk * 16), r0r, r1r, r2r, r3r);
  }
  __syncthreads();

  constexpr int OCHUNKS = DV * 2 / 16;             // 16
  constexpr int NO = BM * OCHUNKS;
  #pragma unroll
  for (int it = 0; it < (NO + THREADS - 1) / THREADS; ++it) {
    int idx = tid + it * THREADS;
    if (NO % THREADS == 0 || idx < NO) {
      int row = idx / OCHUNKS;
      int c = idx % OCHUNKS;
      if (m0 + row < qlen) {
        const uint4* src = reinterpret_cast<const uint4*>(so + row * STAGE_CHUNKS * 16 + c * 16);
        uint4 val = *src;
        *reinterpret_cast<uint4*>(out_base_ptr +
                                  static_cast<int64_t>(m0 + row) * out_row_stride + c * 8) = val;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Config table
// ---------------------------------------------------------------------------

template <int BM, int BN, int STAGES, bool REUSE>
struct Config {
  static constexpr int smem_bytes() {
    int kv = STAGES * (SmemTile<BN, QK_CHUNKS>::BYTES + SmemTile<BN, V_CHUNKS>::BYTES);
    int stage = REUSE ? 0 : BM * (DV * 2 / 16 + 1) * 16;
    return SmemTile<BM, QK_CHUNKS>::BYTES + kv + stage;
  }
};

using C0 = Config<128, 64, 2, false>;   // v1 baseline
using C1 = Config<128, 64, 3, false>;   // deeper pipeline
using C2 = Config<64, 64, 2, true>;     // more CTAs, 2/SM
using C3 = Config<128, 128, 2, true>;   // wide tiles for long streams

template <typename CFG, int BM, int BN, int STAGES, bool REUSE>
int launch_one(const void* q, const void* k, const void* v, void* out, void* lse,
               const void* sinks, const void* cum_q, const void* cum_kv,
               int total_q, int batch, int heads, int kv_heads, int window_left,
               int causal, int has_lse, int has_sink, float bmm1, float bmm2,
               cudaStream_t stream) {
  constexpr int SMEM = CFG::smem_bytes();
  // Clear any error left pending by unrelated host code so the codes below
  // describe THIS launch (job dv_42647ee6bbde: first C2 launch in a second
  // .so returned rc=1 while the identical binary ran fine in the first).
  cudaError_t stale = cudaGetLastError();
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_kernel<BM, BN, STAGES, REUSE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (e != cudaSuccess) {
      if (e == cudaErrorInvalidValue && stale == cudaErrorInvalidValue)
        return 3000 + static_cast<int>(e);   // suspected stale, not ours
      return 1000 + static_cast<int>(e);
    }
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return 5000;
  dim3 grid(tiles_ub, heads, 1);
  dim3 block((BM / 16) * 32, 1, 1);
  mla_kernel<BM, BN, STAGES, REUSE><<<grid, block, SMEM, stream>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<__nv_bfloat16*>(out),
      static_cast<float*>(lse), static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1, bmm2);
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess && le == stale) return 3000 + static_cast<int>(le);
  return static_cast<int>(le);
}

inline int env_cfg() {
  static int cfg = [] {
    const char* e = getenv("MLA_CFG");
    return e ? atoi(e) : -1;
  }();
  return cfg;
}

inline int sm_count() {
  static int sms = [] {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n > 0 ? n : 128;
  }();
  return sms;
}

// Generic config choice from public shape metadata only (no case recognition).
// Measured on L20D (jobs dv_d60d713f18b2, dv_717aeef3659a): C2 (BM=64, 2 CTA/SM)
// best geomean (221us over cases 0-4,6-9); C3 (128x128) best for the longest
// streams (case5 2964 vs 3176; case4 849 vs 884). C0/C1 dominated.
inline int pick_cfg(int total_q, int batch, int heads, int max_kv_len) {
  int forced = env_cfg();
  if (forced >= 0 && forced <= 3) return forced;
  if (max_kv_len >= 8192) return 3;
  return 2;
}

}  // namespace mla

// -----------------------------------------------------------------------------
// Host entry
// -----------------------------------------------------------------------------

extern "C" int mla_forward(const void* q, const void* k, const void* v,
                           void* out, void* lse, const void* sinks,
                           const void* cum_q, const void* cum_kv,
                           int total_q, int batch, int heads, int kv_heads,
                           int window_left, int causal, int has_lse,
                           int has_sink, float bmm1_scale, float bmm2_scale,
                           int max_kv_len, void* stream) {
  using namespace mla;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (pick_cfg(total_q, batch, heads, max_kv_len)) {
    case 1:
      return launch_one<C1, 128, 64, 3, false>(q, k, v, out, lse, sinks, cum_q,
          cum_kv, total_q, batch, heads, kv_heads, window_left, causal, has_lse,
          has_sink, bmm1_scale, bmm2_scale, s);
    case 2:
      return launch_one<C2, 64, 64, 2, true>(q, k, v, out, lse, sinks, cum_q,
          cum_kv, total_q, batch, heads, kv_heads, window_left, causal, has_lse,
          has_sink, bmm1_scale, bmm2_scale, s);
    case 3:
      return launch_one<C3, 128, 128, 2, true>(q, k, v, out, lse, sinks, cum_q,
          cum_kv, total_q, batch, heads, kv_heads, window_left, causal, has_lse,
          has_sink, bmm1_scale, bmm2_scale, s);
    default:
      return launch_one<C0, 128, 64, 2, false>(q, k, v, out, lse, sinks, cum_q,
          cum_kv, total_q, batch, heads, kv_heads, window_left, causal, has_lse,
          has_sink, bmm1_scale, bmm2_scale, s);
  }
}

extern "C" int mla_config_count() { return 4; }
