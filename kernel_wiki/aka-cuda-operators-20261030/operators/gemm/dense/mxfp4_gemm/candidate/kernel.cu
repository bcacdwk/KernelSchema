// mxfp4_gemm — independent CUDA implementation for the frozen contract.
// D = alpha * A @ B^T + C, where A (M,K) and B (N,K) are mxfp4 operands:
//   payload: uint8, two e2m1 codes per byte (low nibble = even k), shape (rows, K/2)
//   scales : float32 powers of two (exp2(stored_ue8m0 - 127)), shape (rows, K/32)
// D is bfloat16 (M,N); C optional bfloat16 (may alias D); alpha optional float.
// Numerics: dequant value = lut[code] * scale is exact in bf16 (<=2 significand
// bits, power-of-two scale, full fp32 exponent range in bf16 incl. subnormals).
// GEMM accumulates in fp32 (bf16 mma), epilogue computes alpha*acc + c in fp32
// and rounds once (RTNE) to bf16, matching reference.py's order of operations.
// No library GEMM is used; all math is implemented here (mma.sync/ldmatrix/
// cp.async PTX). Mechanism references: NVIDIA PTX ISA; DeepGEMM (Apache-2.0)
// sources informed fragment/scale conventions only — no code copied.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdint>
#include <unordered_map>

#define DEVI __device__ __forceinline__

// ---------------------------------------------------------------------------
// PTX primitives
// ---------------------------------------------------------------------------
DEVI uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
               "r"(dst), "l"(src), "r"(src_size));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> DEVI void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

DEVI void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

DEVI void mma_bf16_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                            uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Stage 2: native tcgen05 MXFP4 path (sm_100+); helpers, kernels, launcher.
// ---------------------------------------------------------------------------
#include "tcgen05_kernels.cuh"

// ---------------------------------------------------------------------------
// Dequantization: packed e2m1 + fp32 group scale -> bf16 (exact)
// ---------------------------------------------------------------------------
// e2m1 nibble (s EEM) -> bf16 bits.
DEVI uint32_t fp4_to_bf16_bits(int c) {
  int s = (c & 8) << 12;            // sign bit3 -> bf16 bit15
  int m = c & 7;
  uint32_t b = (m >= 2) ? ((((m >> 1) + 126) << 7) | ((m & 1) << 6))
                        : (m ? 0x3F00u : 0u);  // 0.5, 0
  return b | static_cast<uint32_t>(s);
}

// One thread handles one 16-byte payload chunk = one 32-element scale group:
// reads 16 payload bytes + 1 fp32 scale, writes 32 bf16 (64 bytes).
__global__ void dequant_kernel(const uint8_t* __restrict__ qa,
                               const float* __restrict__ sa,
                               __nv_bfloat16* __restrict__ oa, long long cnta,
                               const uint8_t* __restrict__ qb,
                               const float* __restrict__ sb,
                               __nv_bfloat16* __restrict__ ob, long long cntb,
                               int gpr) {
  __shared__ uint32_t lut[256];
  const int t = threadIdx.x;
  if (t < 256) {
    lut[t] = fp4_to_bf16_bits(t & 15) | (fp4_to_bf16_bits(t >> 4) << 16);
  }
  __syncthreads();
  const long long idx = blockIdx.x * (long long)blockDim.x + t;
  const long long cnt = cnta + cntb;
  if (idx >= cnt) return;
  const uint8_t* q;
  const float* s;
  __nv_bfloat16* o;
  long long i;
  if (idx < cnta) { q = qa; s = sa; o = oa; i = idx; }
  else            { q = qb; s = sb; o = ob; i = idx - cnta; }
  const long long row = i / gpr;
  const int g = static_cast<int>(i - row * gpr);
  const uint4 payload = *reinterpret_cast<const uint4*>(q + (row * gpr + g) * 16);
  const float scale = s[row * gpr + g];
  const __nv_bfloat162 sc2 = __float2bfloat162_rn(scale);
  uint32_t out[16];
  const uint32_t bytes[16] = {
      payload.x & 0xFFu, (payload.x >> 8) & 0xFFu, (payload.x >> 16) & 0xFFu, payload.x >> 24,
      payload.y & 0xFFu, (payload.y >> 8) & 0xFFu, (payload.y >> 16) & 0xFFu, payload.y >> 24,
      payload.z & 0xFFu, (payload.z >> 8) & 0xFFu, (payload.z >> 16) & 0xFFu, payload.z >> 24,
      payload.w & 0xFFu, (payload.w >> 8) & 0xFFu, (payload.w >> 16) & 0xFFu, payload.w >> 24};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lut[bytes[j]]), sc2);
    out[j] = *reinterpret_cast<const uint32_t*>(&v);
  }
  *reinterpret_cast<uint4*>(o + (row * gpr + g) * 32) =
      *reinterpret_cast<const uint4*>(&out[0]);
  *reinterpret_cast<uint4*>(o + (row * gpr + g) * 32 + 8) =
      *reinterpret_cast<const uint4*>(&out[4]);
  *reinterpret_cast<uint4*>(o + (row * gpr + g) * 32 + 16) =
      *reinterpret_cast<const uint4*>(&out[8]);
  *reinterpret_cast<uint4*>(o + (row * gpr + g) * 32 + 24) =
      *reinterpret_cast<const uint4*>(&out[12]);
}

// ---------------------------------------------------------------------------
// Tiled bf16 GEMM: D = alpha * A @ B^T + C
//   A: (M,K) bf16 row-major, B: (N,K) bf16 row-major, D/C: (M,N) bf16.
//   BK = 64 fixed (rows are 128B); 128B XOR swizzle; cp.async multi-stage.
// ---------------------------------------------------------------------------
template <int BM, int BN, int STAGES, int WGM, int WGN, int MIN_CTA>
__global__ void __launch_bounds__(WGM * WGN * 32, MIN_CTA)
gemm_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                 const __nv_bfloat16* __restrict__ B,
                 __nv_bfloat16* __restrict__ D,
                 const __nv_bfloat16* __restrict__ C, float alpha, int has_c,
                 int M, int N, int K) {
  constexpr int BK = 64;                     // elements along K per stage
  constexpr int THREADS = WGM * WGN * 32;
  constexpr int MT = BM / WGM / 16;          // 16-row mma tiles per warp
  constexpr int NT = BN / WGN / 8;           // 8-col mma tiles per warp
  constexpr int A_STAGE = BM * BK * 2;       // bytes per A stage (row = 128B)
  constexpr int B_STAGE = BN * BK * 2;

  extern __shared__ char smem_raw[];
  char* smem_a = smem_raw;
  char* smem_b = smem_raw + A_STAGE * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int wm = warp / WGN;
  const int wn = warp % WGN;

  const int bm = blockIdx.x;                 // m tile index
  const int bn = blockIdx.y;                 // n tile index
  const int m0 = bm * BM;
  const int n0 = bn * BN;
  const int nk = K / BK;

  // ---- stage loader -------------------------------------------------------
  auto issue_load = [&](int kb) {
    const int stage = kb % STAGES;
    char* a_dst = smem_a + stage * A_STAGE;
    char* b_dst = smem_b + stage * B_STAGE;
    const int k0 = kb * BK;
#pragma unroll
    for (int i = tid; i < BM * 8; i += THREADS) {
      const int row = i >> 3;
      const int c = i & 7;
      const int grow = m0 + row;
      const bool ok = (kb < nk) && (grow < M);
      const int srow = ok ? grow : (M - 1);
      const void* src = A + ((long long)srow * K + k0) + c * 8;
      cp_async16(smem_u32(a_dst + row * 128 + ((c ^ (row & 7)) * 16)), src, ok ? 16u : 0u);
    }
#pragma unroll
    for (int i = tid; i < BN * 8; i += THREADS) {
      const int row = i >> 3;
      const int c = i & 7;
      const int grow = n0 + row;
      const bool ok = (kb < nk) && (grow < N);
      const int srow = ok ? grow : (N - 1);
      const void* src = B + ((long long)srow * K + k0) + c * 8;
      cp_async16(smem_u32(b_dst + row * 128 + ((c ^ (row & 7)) * 16)), src, ok ? 16u : 0u);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[mt][nt][j] = 0.f;

  // prologue
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    issue_load(s);
    cp_async_commit();
  }

  const int warp_m_off = wm * MT * 16;
  const int warp_n_off = wn * NT * 8;

  for (int kb = 0; kb < nk; ++kb) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kb + STAGES - 1 < nk) issue_load(kb + STAGES - 1);
    cp_async_commit();

    const int stage = kb % STAGES;
    const char* a_src = smem_a + stage * A_STAGE;
    const char* b_src = smem_b + stage * B_STAGE;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      uint32_t a[MT][4];
      uint32_t b[NT][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = warp_m_off + mt * 16 + (lane % 16);
        const int chunk = k16 * 2 + (lane / 16);
        const uint32_t addr = smem_u32(a_src + row * 128 + ((chunk ^ (row & 7)) * 16));
        ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], addr);
      }
#pragma unroll
      for (int nt2 = 0; nt2 < NT / 2; ++nt2) {
        const int row = warp_n_off + nt2 * 16 + (lane & 7) + ((lane >> 4) * 8);
        const int chunk = k16 * 2 + ((lane >> 3) & 1);
        const uint32_t addr = smem_u32(b_src + row * 128 + ((chunk ^ (row & 7)) * 16));
        uint32_t r0, r1, r2, r3;
        ldmatrix_x4(r0, r1, r2, r3, addr);
        b[nt2 * 2][0] = r0; b[nt2 * 2][1] = r1;
        b[nt2 * 2 + 1][0] = r2; b[nt2 * 2 + 1][1] = r3;
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          mma_bf16_m16n8k16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                            a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                            b[nt][0], b[nt][1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- epilogue: D = bf16(alpha * acc + C) --------------------------------
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r0g = m0 + warp_m_off + mt * 16 + (lane / 4);
    const int r1g = r0g + 8;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int col = n0 + warp_n_off + nt * 8 + 2 * (lane % 4);
      float v[4] = {acc[mt][nt][0] * alpha, acc[mt][nt][1] * alpha,
                    acc[mt][nt][2] * alpha, acc[mt][nt][3] * alpha};
      if (has_c) {
        if (r0g < M) {
          const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(C + (long long)r0g * N + col);
          const float2 c01f = __bfloat1622float2(c01);
          v[0] += c01f.x; v[1] += c01f.y;
        }
        if (r1g < M) {
          const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(C + (long long)r1g * N + col);
          const float2 c23f = __bfloat1622float2(c23);
          v[2] += c23f.x; v[3] += c23f.y;
        }
      }
      if (r0g < M) {
        const __nv_bfloat162 p = __float22bfloat162_rn(make_float2(v[0], v[1]));
        *reinterpret_cast<__nv_bfloat162*>(D + (long long)r0g * N + col) = p;
      }
      if (r1g < M) {
        const __nv_bfloat162 p = __float22bfloat162_rn(make_float2(v[2], v[3]));
        *reinterpret_cast<__nv_bfloat162*>(D + (long long)r1g * N + col) = p;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
template <int BM, int BN, int STAGES, int WGM, int WGN, int MIN_CTA>
static void launch_gemm(const at::Tensor& abuf, const at::Tensor& bbuf,
                        at::Tensor& d, const at::Tensor* c, float alpha,
                        int M, int N, int K, cudaStream_t stream) {
  static bool attr_set = false;
  const int smem = STAGES * (BM + BN) * 64 * 2;
  if (!attr_set) {
    cudaFuncSetAttribute(gemm_bf16_kernel<BM, BN, STAGES, WGM, WGN, MIN_CTA>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  dim3 grid((M + BM - 1) / BM, N / BN);
  gemm_bf16_kernel<BM, BN, STAGES, WGM, WGN, MIN_CTA>
      <<<grid, WGM * WGN * 32, smem, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(abuf.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(bbuf.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
          c ? reinterpret_cast<const __nv_bfloat16*>(c->data_ptr()) : nullptr,
          alpha, c ? 1 : 0, M, N, K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gemm launch failed");
}

void mxfp4_run(at::Tensor aq, at::Tensor asf, at::Tensor abuf, at::Tensor bq,
               at::Tensor bsf, at::Tensor bbuf, at::Tensor d, at::Tensor c,
               bool has_c, double alpha) {
  const long long M = aq.size(0);
  const long long K2 = aq.size(1);
  const long long K = K2 * 2;
  const long long N = bq.size(0);
  TORCH_CHECK(aq.is_contiguous() && bq.is_contiguous() && asf.is_contiguous() &&
              bsf.is_contiguous() && d.is_contiguous());
  TORCH_CHECK(aq.scalar_type() == at::kChar || aq.scalar_type() == at::kByte);
  TORCH_CHECK(asf.scalar_type() == at::kFloat && bsf.scalar_type() == at::kFloat);
  TORCH_CHECK(d.scalar_type() == at::kBFloat16, "only bf16 output in frozen domain");
  TORCH_CHECK(bq.size(1) == K2 && asf.size(0) == M && asf.size(1) == K / 32 &&
              bsf.size(0) == N && bsf.size(1) == K / 32 && d.size(0) == M && d.size(1) == N);
  TORCH_CHECK(K % 64 == 0 && K2 % 16 == 0 && N % 128 == 0 && M >= 1);
  TORCH_CHECK(!has_c || (c.is_contiguous() && c.scalar_type() == d.scalar_type() &&
                         c.size(0) == M && c.size(1) == N));

  auto stream = at::cuda::getCurrentCUDAStream();
  const at::Tensor* cp = has_c ? &c : nullptr;
  const float al = static_cast<float>(alpha);

  // Native block-scaled MXFP4 UMMA path on sm_100+ (bit-exact, no dequant).
  static const int cc_major = at::cuda::getCurrentDeviceProperties()->major;
  if (cc_major >= 10 && (K % 256) == 0 && (N % 128) == 0) {
    mxfp4_run_tcgen05(aq, asf, bq, bsf, d, cp, al, (int)M, (int)N, (int)K, stream);
    return;
  }

  const int gpr = static_cast<int>(K2 / 16);          // 16B chunks == scale groups per row
  const long long cnta = M * gpr;
  const long long cntb = N * gpr;
  {
    const long long total = cnta + cntb;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    dequant_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(aq.data_ptr()), asf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(abuf.data_ptr()), cnta,
        reinterpret_cast<const uint8_t*>(bq.data_ptr()), bsf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(bbuf.data_ptr()), cntb, gpr);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequant launch failed");
  }

  if (M <= 16) {
    launch_gemm<16, 64, 4, 1, 4, 4>(abuf, bbuf, d, cp, al, (int)M, (int)N, (int)K, stream);
  } else if (M <= 32) {
    launch_gemm<32, 64, 4, 2, 2, 4>(abuf, bbuf, d, cp, al, (int)M, (int)N, (int)K, stream);
  } else if (M <= 64) {
    launch_gemm<64, 64, 4, 4, 1, 3>(abuf, bbuf, d, cp, al, (int)M, (int)N, (int)K, stream);
  } else {
    launch_gemm<128, 128, 3, 2, 4, 2>(abuf, bbuf, d, cp, al, (int)M, (int)N, (int)K, stream);
  }
}

void mxfp4_run_ptr(int64_t aq, int64_t asf, int64_t bq, int64_t bsf, int64_t d,
                   int64_t c, int64_t m, int64_t n, int64_t k2, double alpha,
                   bool has_c);

// Phase-2 step-5: whole-forward hot path in C++ (tuple unpack, contiguity
// fixups, gating, checks). Python side becomes one pybind call, removing
// ~2-3us of per-call python overhead that sits INSIDE the engine's timed
// span (kbreak3: py_wall 5.4-6.5us vs ext-call wall 2.8us).
void mxfp4_forward(pybind11::object a, pybind11::object b, at::Tensor d,
                   pybind11::object c, pybind11::object alpha) {
  // tolerate tuple OR list operand pairs
  pybind11::tuple ta = pybind11::isinstance<pybind11::tuple>(a)
                           ? a.cast<pybind11::tuple>()
                           : pybind11::tuple(a);
  pybind11::tuple tb = pybind11::isinstance<pybind11::tuple>(b)
                           ? b.cast<pybind11::tuple>()
                           : pybind11::tuple(b);
  TORCH_CHECK(ta.size() == 2 && tb.size() == 2, "operands are (payload, scale) pairs");
  at::Tensor aq = ta[0].cast<at::Tensor>();
  at::Tensor asf = ta[1].cast<at::Tensor>();
  at::Tensor bq = tb[0].cast<at::Tensor>();
  at::Tensor bsf = tb[1].cast<at::Tensor>();
  if (!aq.is_contiguous()) aq = aq.contiguous();
  if (!asf.is_contiguous()) asf = asf.contiguous();
  if (!bq.is_contiguous()) bq = bq.contiguous();
  if (!bsf.is_contiguous()) bsf = bsf.contiguous();
  TORCH_CHECK(d.is_contiguous(), "d must be contiguous");
  const long long M = aq.size(0);
  const long long K2 = aq.size(1);
  const long long K = K2 * 2;
  const long long N = bq.size(0);
  TORCH_CHECK(aq.scalar_type() == at::kChar || aq.scalar_type() == at::kByte);
  TORCH_CHECK(asf.scalar_type() == at::kFloat && bsf.scalar_type() == at::kFloat);
  TORCH_CHECK(d.scalar_type() == at::kBFloat16, "only bf16 output in frozen domain");
  TORCH_CHECK(bq.size(1) == K2 && asf.size(0) == M && asf.size(1) == K / 32 &&
              bsf.size(0) == N && bsf.size(1) == K / 32 && d.size(0) == M && d.size(1) == N);
  TORCH_CHECK(K % 64 == 0 && K2 % 16 == 0 && N % 128 == 0 && M >= 1);
  const bool has_c = !c.is_none();
  at::Tensor ct = has_c ? c.cast<at::Tensor>() : d;
  if (has_c) {
    TORCH_CHECK(ct.is_contiguous() && ct.scalar_type() == d.scalar_type() &&
                ct.size(0) == M && ct.size(1) == N);
  }
  const float al = alpha.is_none() ? 1.0f : static_cast<float>(alpha.cast<double>());
  auto stream = at::cuda::getCurrentCUDAStream();
  static const int cc_major = at::cuda::getCurrentDeviceProperties()->major;
  if (cc_major >= 10 && (K % 256) == 0 && (N % 128) == 0) {
    const at::Tensor* cp = has_c ? &ct : nullptr;
    mxfp4_run_tcgen05(aq, asf, bq, bsf, d, cp, al, (int)M, (int)N, (int)K, stream);
    return;
  }
  // Non-native fallback (rare): scratch + legacy entry (revalidates, fine).
  auto opts = d.options();
  at::Tensor abuf = at::empty({M, K}, opts);
  at::Tensor bbuf = at::empty({N, K}, opts);
  mxfp4_run(aq, asf, abuf, bq, bsf, bbuf, d, ct, has_c, al);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_run", &mxfp4_run, "mxfp4 gemm fused run");
  m.def("mxfp4_run_ptr", &mxfp4_run_ptr, "mxfp4 gemm pointer fast path");
  m.def("mxfp4_forward", &mxfp4_forward, "mxfp4 whole-forward C++ hot path");
}
