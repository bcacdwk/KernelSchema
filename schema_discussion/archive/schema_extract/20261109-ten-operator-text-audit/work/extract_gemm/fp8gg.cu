// fp8_scaled_grouped_gemm — independent CUDA implementation for L20D (sm_103).
//
// Grouped NT GEMM: for each group e with rows [lo,hi) = [m_indptr[e], m_indptr[e+1]):
//   out[lo:hi, :] = (a[lo:hi] * a_scale_per_(row,kblock)) @ (b[e] * b_scale_per_(128n,128k))^T
// a: (m,k) e4m3 row-major; b: (e,n,k) e4m3; a_scale: (k/128, m) f32 (MN-major);
// b_scale: (e, k/128, n/128) f32; out: (m,n) bf16.
//
// Phase 1 design: multi-stage cp.async pipeline with warp-level
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (fp8 tensor cores),
// BLOCK_K aligned to the 128-element scale granularity: raw fp8 products are
// accumulated in tensor-core fp32 fragments per K block, then promoted into a
// master fp32 accumulator with (a_scale[row,kb] * b_scale[nt,kb]) — the same
// mathematics as the DeepGEMM sm90 1D2D promotion scheme (mechanism studied
// from the public DeepGEMM sources; all code here is an independent
// implementation, no library code is called).
//
// Numerics: e4m3 products are exact in fp32; only accumulation order differs
// from the fp32 reference, far inside the frozen tolerance (atol .01, rtol .05).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int BM = 64;          // rows per CTA tile
constexpr int BN = 128;         // cols per CTA tile (one b-scale N block)
constexpr int BK = 128;         // K per stage (one scale K block)
constexpr int STAGES = 3;
constexpr int THREADS = 256;    // 8 warps: 2 (m) x 4 (n)
constexpr int A_STAGE_BYTES = BM * BK;                                   // 8192
constexpr int B_STAGE_BYTES = BN * BK;                                   // 16384
constexpr int SFA_STAGE_OFF = A_STAGE_BYTES + B_STAGE_BYTES;             // 24576
constexpr int STAGE_BYTES = SFA_STAGE_OFF + BM * 4;                      // 24832
constexpr int EPI_ROW_BYTES = BN * 2 + 16;                               // 272
constexpr int SCHED_OFF = STAGES * STAGE_BYTES;                          // 74496
constexpr int MAX_G = 4096;                                              // group-count sanity bound

struct Params {
  const uint8_t* a;   // (m, k)
  const uint8_t* b;   // (e, n, k)
  const float* sa;    // (k/128, m)
  const float* sb;    // (e, k/128, n/128)
  const int* indptr;  // (G+1)
  uint16_t* out;      // (m, n) bf16 bits
  int m, n, k, G;
  int n_tiles;   // n / BN
  int kb_count;  // k / BK
};

DEVINL uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVINL void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async4(uint32_t dst, const void* src, bool pred) {
  const int bytes = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
DEVINL void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

DEVINL void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

DEVINL void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr),
               "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

DEVINL void mma_fp8(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVINL uint32_t pack_bf16(float x, float y) {
  __nv_bfloat162 h = __float22bfloat162_rn(make_float2(x, y));
  return *reinterpret_cast<uint32_t*>(&h);
}

// 128B swizzle: byte offset of 16B chunk `c` of row `row` within a swizzled tile.
DEVINL uint32_t swz(int row, int chunk) {
  return static_cast<uint32_t>(row) * 128u + (static_cast<uint32_t>(chunk ^ (row & 7)) * 16u);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(THREADS, 2) fp8gg_bm64(const Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // ---- schedule smem: s_ind[G+1], s_pref[G+1] ----
  int* s_ind = reinterpret_cast<int*>(smem + SCHED_OFF);
  int* s_pref = s_ind + p.G + 1;

  for (int i = tid; i <= p.G; i += THREADS) s_ind[i] = p.indptr[i];
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < p.G; ++e) {
      s_pref[e] = acc;
      const int sz = s_ind[e + 1] - s_ind[e];
      acc += (sz + BM - 1) / BM;
    }
    s_pref[p.G] = acc;
  }
  __syncthreads();
  const int W = s_pref[p.G] * p.n_tiles;

  const int wm = warp >> 2;  // 0..1 -> 32 rows each
  const int wn = warp & 3;   // 0..3 -> 32 cols each

  // ---- persistent work loop ----
  for (int w = blockIdx.x; w < W; w += gridDim.x) {
    int e = 0;
    while (s_pref[e + 1] * p.n_tiles <= w) ++e;
    const int r = w - s_pref[e] * p.n_tiles;
    const int tiles_e = s_pref[e + 1] - s_pref[e];
    const int nt = r / tiles_e;  // nt-outer order: consecutive w share the B panel
    const int mt = r % tiles_e;

    const int lo = s_ind[e];
    const int hi = s_ind[e + 1];
    const int row0 = lo + mt * BM;
    const int rows = min(BM, hi - row0);

    const int K = p.k;
    const int KB = p.kb_count;
    const int NB = p.n_tiles;

    const uint8_t* A_row = p.a + static_cast<long>(row0) * K;
    const uint8_t* B_row = p.b + (static_cast<long>(e) * p.n + nt * BN) * static_cast<long>(K);
    const float* SA_col = p.sa + row0;                                 // + kb*m + rr
    const float* SB_tile = p.sb + static_cast<long>(e) * KB * NB + nt;  // + kb*NB

    // per-thread copy plan: global element offsets (int32-safe: <= 127*7168+...) and smem offsets
    int agoff[2];
    uint32_t aoff[2];
    bool apred[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int item = tid + i * THREADS;
      const int rr = item >> 3, cc = item & 7;
      agoff[i] = rr * K + cc * 16;
      aoff[i] = rr * 128u + ((cc ^ (rr & 7)) * 16u);
      apred[i] = rr < rows;
    }
    int bgoff[4];
    uint32_t boff[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int item = tid + i * THREADS;
      const int rr = item >> 3, cc = item & 7;
      bgoff[i] = rr * K + cc * 16;
      boff[i] = A_STAGE_BYTES + rr * 128u + ((cc ^ (rr & 7)) * 16u);
    }

    auto issue_stage = [&](int kb) {
      const uint32_t base = smem_u32(smem) + (kb % STAGES) * STAGE_BYTES;
      const int kbo = kb * BK;
#pragma unroll
      for (int i = 0; i < 2; ++i)
        cp_async16(base + aoff[i], apred[i] ? (A_row + agoff[i] + kbo) : A_row, apred[i]);
#pragma unroll
      for (int i = 0; i < 4; ++i) cp_async16(base + boff[i], B_row + bgoff[i] + kbo, true);
      if (tid < BM)
        cp_async4(base + SFA_STAGE_OFF + tid * 4,
                  tid < rows ? (SA_col + kb * p.m + tid) : SA_col, tid < rows);
      cp_async_commit();
    };

    // ---- prologue ----
    const int pre = min(STAGES - 1, KB);
    for (int s = 0; s < pre; ++s) issue_stage(s);
    for (int s = pre; s < STAGES - 1; ++s) cp_async_commit();

    // ---- main loop ----
    float fin[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) fin[i][j] = 0.f;

    for (int kb = 0; kb < KB; ++kb) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      if (kb + STAGES - 1 < KB) issue_stage(kb + STAGES - 1);
      else cp_async_commit();

      const uint32_t sbase = smem_u32(smem) + (kb % STAGES) * STAGE_BYTES;
      const uint32_t As = sbase;
      const uint32_t Bs = sbase + A_STAGE_BYTES;
      const float* sfa = reinterpret_cast<const float*>(smem + (kb % STAGES) * STAGE_BYTES + SFA_STAGE_OFF);

      float acc[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const int row = wm * 32 + mi * 16 + (lane & 15);
          const uint32_t addr = As + swz(row, 2 * st + (lane >> 4));
          ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], addr);
        }
        uint32_t bf[2][4];
#pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
          const int row = wn * 32 + bi * 16 + (lane & 15);
          const uint32_t addr = Bs + swz(row, 2 * st + (lane >> 4));
          ldmatrix_x4(bf[bi][0], bf[bi][1], bf[bi][2], bf[bi][3], addr);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
          for (int bi = 0; bi < 2; ++bi) {
            mma_fp8(acc[mi * 4 + bi * 2 + 0], af[mi], bf[bi][0], bf[bi][2]);
            mma_fp8(acc[mi * 4 + bi * 2 + 1], af[mi], bf[bi][1], bf[bi][3]);
          }
        }
      }

      // ---- promote with per-(row,kb) A scale and per-(nt,kb) B scale ----
      const float sbv = __ldg(SB_tile + kb * NB);
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const float sa0 = sfa[wm * 32 + mi * 16 + (lane >> 2)];
        const float sa1 = sfa[wm * 32 + mi * 16 + (lane >> 2) + 8];
        const float sc0 = sa0 * sbv, sc1 = sa1 * sbv;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          fin[mi * 4 + ni][0] += sc0 * acc[mi * 4 + ni][0];
          fin[mi * 4 + ni][1] += sc0 * acc[mi * 4 + ni][1];
          fin[mi * 4 + ni][2] += sc1 * acc[mi * 4 + ni][2];
          fin[mi * 4 + ni][3] += sc1 * acc[mi * 4 + ni][3];
        }
      }
    }

    // ---- epilogue: fin -> bf16 -> smem (stmatrix) -> global (vectorized) ----
    cp_async_wait<0>();
    __syncthreads();

    __nv_bfloat16* Cs = reinterpret_cast<__nv_bfloat16*>(smem);  // overlays stage 0/1
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ni2 = 0; ni2 < 2; ++ni2) {
        const int t0 = mi * 4 + ni2 * 2, t1 = t0 + 1;
        const uint32_t r0 = pack_bf16(fin[t0][0], fin[t0][1]);
        const uint32_t r1 = pack_bf16(fin[t0][2], fin[t0][3]);
        const uint32_t r2 = pack_bf16(fin[t1][0], fin[t1][1]);
        const uint32_t r3 = pack_bf16(fin[t1][2], fin[t1][3]);
        const int row = wm * 32 + mi * 16 + (lane & 15);
        const uint32_t addr = smem_u32(Cs) + row * EPI_ROW_BYTES +
                              (wn * 32 + ni2 * 16) * 2u + (lane >> 4) * 16u;
        stmatrix_x4(addr, r0, r1, r2, r3);
      }
    }
    __syncthreads();

    uint16_t* O_tile = p.out + static_cast<long>(row0) * p.n + nt * BN;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int item = tid + i * THREADS;
      const int rr = item >> 4, cc = item & 15;
      if (rr < rows) {
        const uint4 v = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(Cs) + rr * EPI_ROW_BYTES + cc * 16);
        *reinterpret_cast<uint4*>(O_tile + static_cast<long>(rr) * p.n + cc * 8) = v;
      }
    }
    __syncthreads();
  }
}

extern "C" int fp8gg_max_smem() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  return v;
}

extern "C" int fp8gg_run(const void* a, const void* b, const void* sa, const void* sb,
                         const int* indptr, void* out, int m, int n, int k, int G, int nSM,
                         cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || G <= 0) return 0;
  if (G > MAX_G) return 1000 + G;
  if ((n % BN) != 0 || (k % BK) != 0) return 1001;  // contract guarantees multiples of 128

  Params p;
  p.a = static_cast<const uint8_t*>(a);
  p.b = static_cast<const uint8_t*>(b);
  p.sa = static_cast<const float*>(sa);
  p.sb = static_cast<const float*>(sb);
  p.indptr = indptr;
  p.out = static_cast<uint16_t*>(out);
  p.m = m;
  p.n = n;
  p.k = k;
  p.G = G;
  p.n_tiles = n / BN;
  p.kb_count = k / BK;

  const size_t smem = SCHED_OFF + static_cast<size_t>(G + 1) * 2 * sizeof(int) + 64;
  static int configured = 0;
  if (!configured) {
    const int maxopt = fp8gg_max_smem();
    int want = SCHED_OFF + (MAX_G + 1) * 8;
    if (want > maxopt) want = maxopt;
    if (static_cast<int>(smem) > want) return 1002;
    cudaError_t err =
        cudaFuncSetAttribute(fp8gg_bm64, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
    if (err != cudaSuccess) return static_cast<int>(err);
    configured = 1;
  }

  const long wub = (static_cast<long>(m + BM - 1) / BM + G) * static_cast<long>(p.n_tiles);
  long grid = wub < 2L * nSM ? wub : 2L * nSM;
  if (grid < 1) grid = 1;

  fp8gg_bm64<<<static_cast<int>(grid), THREADS, smem, stream>>>(p);
  return static_cast<int>(cudaGetLastError());
}
