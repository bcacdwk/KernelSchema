#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace mla_v1 {

constexpr int M_TILE = 128;
constexpr int NK = 64;                 // keys per tile
constexpr int KDIM = 576;              // head dim
constexpr int DDIM = 512;              // output dim
constexpr int DP = 448;                // output dims kept in TMEM
constexpr int DR = DDIM - DP;          // output dims kept in registers
#ifndef MLA_NTHREADS
#define MLA_NTHREADS 256
#endif
constexpr int NTHREADS = MLA_NTHREADS;
constexpr int NGROUP = NTHREADS / M_TILE;   // softmax column groups
constexpr int NKG = NK / NGROUP;            // keys per group
constexpr int DRG = DR / NGROUP;            // register output dims per group
// The barrier-free row maximum below indexes the score tile by whole 32-column
// TMEM chunks, so a group must own exactly one chunk. (512 threads / four
// 16-key groups were measured and are slower anyway - see
// results/v111-512threads-negative.md.)
static_assert(NKG == 32, "the softmax expects 32-key column groups");
constexpr int KSTEPS = KDIM / 32;      // 18 UMMA_K steps for QK
constexpr int PV_KSTEPS = NK / 32;     // 2 UMMA_K steps for PV
constexpr int SW = 64;                 // swizzle atom bytes
constexpr int QRB = M_TILE / 8;        // 16 row blocks in Q and P
constexpr int KRB = NK / 8;            // 8 key blocks per KV tile
constexpr int ATOMS = KDIM / SW;       // 9 dim atoms
constexpr int BLOCK_BYTES = 8 * SW;    // 512 bytes per atom
constexpr int Q_BYTES = ATOMS * QRB * BLOCK_BYTES;      // 73728
constexpr int KV_BYTES = ATOMS * KRB * BLOCK_BYTES;     // 36864
constexpr int P_BYTES = QRB * BLOCK_BYTES;              // 8192 (one key atom)
constexpr int SMEM_HEAD = 8192;   // barriers + per-row softmax exchange
static_assert(SMEM_HEAD >= 64 + 2 * NGROUP * M_TILE * 4, "SMEM head too small");
constexpr int OFF_Q = SMEM_HEAD;
constexpr int OFF_KV = OFF_Q + Q_BYTES;
constexpr int OFF_P = OFF_KV + 2 * KV_BYTES;
constexpr int SMEM_TOTAL = OFF_P + 4 * P_BYTES;         // 180224 + 1024

constexpr int TCOL_O = 0;              // DP columns
constexpr int TCOL_S = DP;             // NK columns for S, DR columns for temp O
// P is stored as E4M3 with a fixed prescale. Two limits follow from it:
//   P_SCALE * 2^RESCALE_THRESH <= 448 (E4M3 max), and weights further than
//   log2(P_SCALE)+6 below the shift fall into the E4M3 subnormal range.
// P_SCALE=64 gives a 2.75 log2 growth budget between rescales while keeping
// every weight within ~8 nat of the row maximum in the normal range.
constexpr float P_SCALE = 64.0f;
constexpr float LOG2_P_SCALE = 6.0f;
constexpr float RESCALE_THRESH = 2.75f;
#ifndef MLA_SHIFT_HEAD
#define MLA_SHIFT_HEAD 2.0f
#endif
constexpr float SHIFT_HEAD = MLA_SHIFT_HEAD;
constexpr float NEG_INF = -3.402823466e+38f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

#ifndef MLA_PART_FP16
#define MLA_PART_FP16 1
#endif
constexpr bool kPartFp16 = (MLA_PART_FP16 != 0);
constexpr int kPartWords = kPartFp16 ? (DDIM / 2) : DDIM;

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// The main kernel drains TMEM as raw bits and keeps the register output dims as
// floats; both reach the same store helper.
template <typename T>
__device__ __forceinline__ float part_val(const T* vals, int j) {
  if constexpr (std::is_same<T, float>::value) return vals[j];
  else return __uint_as_float(vals[j]);
}
// kProbe is documented with the timing-probe list further down, but it has to be
// declared here because part_store participates in probe bits 512 and 1024.
#ifndef MLA_PROBE
#define MLA_PROBE 0
#endif
constexpr int kProbe = MLA_PROBE;
#ifndef MLA_EPI_STAGE
#define MLA_EPI_STAGE 1
#endif
#if MLA_EPI_STAGE && !MLA_PART_FP16
#error "MLA_EPI_STAGE needs the 1024-byte fp16 partial row; MLA_PART_FP16 is 0"
#endif
#if MLA_EPI_STAGE
constexpr int kEpiStageStride = DDIM * 2 + 16;
#ifndef MLA_EPI_STAGE_MINROWS
#define MLA_EPI_STAGE_MINROWS 32
#endif
constexpr int kEpiStageMinRows = MLA_EPI_STAGE_MINROWS;
static_assert(M_TILE * kEpiStageStride + M_TILE * 8 <= Q_BYTES + 2 * KV_BYTES,
              "the staged epilogue tile does not fit in the freed Q + KV shared memory");
#endif

template <int N, typename T>
__device__ __forceinline__ void part_store(float* roww, int dim, const T* vals, float nrm,
                                           const void* probe_keep = nullptr) {
  static_assert(N % 8 == 0, "the vector stores below need a multiple of eight dims");
  const bool do_store = ((kProbe & 512) == 0) || (probe_keep != nullptr);
  if (kPartFp16) {
    uint32_t pk[N / 2];
    if (kProbe & 1024) {
      const uint32_t* rawv = reinterpret_cast<const uint32_t*>(vals);
#pragma unroll
      for (int j = 0; j < N / 2; ++j) pk[j] = rawv[2 * j];
    } else {
#pragma unroll
      for (int j = 0; j < N / 2; ++j)
        pk[j] = pack_h2(part_val(vals, 2 * j) * nrm, part_val(vals, 2 * j + 1) * nrm);
    }
    if (do_store) {
#pragma unroll
      for (int s = 0; s < N / 8; ++s)
        asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(roww + dim / 2 + s * 4),
                     "r"(pk[s * 4]), "r"(pk[s * 4 + 1]), "r"(pk[s * 4 + 2]), "r"(pk[s * 4 + 3]));
    }
  } else {
    float4* dst = reinterpret_cast<float4*>(roww + dim);
    if (do_store) {
#pragma unroll
      for (int s = 0; s < N / 4; ++s)
        dst[s] = make_float4(part_val(vals, s * 4), part_val(vals, s * 4 + 1),
                             part_val(vals, s * 4 + 2), part_val(vals, s * 4 + 3));
    }
  }
}
__device__ __forceinline__ float4 part_load4(const float* roww, int dim) {
  if (kPartFp16) {
    const uint2 raw = *reinterpret_cast<const uint2*>(roww + dim / 2);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    return make_float4(a.x, a.y, b.x, b.y);
  }
  return reinterpret_cast<const float4*>(roww + dim)[0];
}

__device__ __forceinline__ void loop_sync() {
  if (!(kProbe & 8)) __syncthreads();
}

// ------------------------------------------------------------------ PTX layer
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#if MLA_EPI_STAGE
// One 16-byte vector store to either destination. Both sides are a single instruction
// with no side effects but the store, so the choice predicates rather than branches.
__device__ __forceinline__ void epi_st16(bool to_smem, uint8_t* sdst, uint8_t* gdst,
                                         const uint32_t* w) {
  if (to_smem)
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(smem_u32(sdst)),
                 "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]));
  else
    asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(gdst),
                 "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]));
}

// part_store's fp16 packing with either destination, so the pack is written once and the
// per-work-item choice stays a predicate inside the store loop. Byte offsets agree with
// part_store exactly: its `roww + dim/2 + s*4` in float units is dim*2 + 16*s bytes.
template <int N, typename T>
__device__ __forceinline__ void part_store_either(bool to_smem, uint8_t* srow, float* roww,
                                                  int dim, const T* vals, float nrm) {
  static_assert(N % 8 == 0, "the vector stores below need a multiple of eight dims");
  uint32_t pk[N / 2];
#pragma unroll
  for (int j = 0; j < N / 2; ++j)
    pk[j] = pack_h2(part_val(vals, 2 * j) * nrm, part_val(vals, 2 * j + 1) * nrm);
#pragma unroll
  for (int s = 0; s < N / 8; ++s)
    epi_st16(to_smem, srow + (size_t)(dim + s * 8) * 2,
             reinterpret_cast<uint8_t*>(roww) + (size_t)(dim + s * 8) * 2, pk + s * 4);
}
#endif

__device__ __forceinline__ void tmem_alloc(uint32_t* slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
               "r"(smem_u32(slot)), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(ncols));
}
__device__ __forceinline__ void umma_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void fence_before() { if (kProbe & 1) return; asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
// Generic-proxy SMEM stores (the P tile) must be published to the async proxy
// that tcgen05.mma uses to read operands.
__device__ __forceinline__ void fence_view_async_shared() {
  if (kProbe & 1) return;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_after() { if (kProbe & 1) return; asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
}
// The async tcgen05/TMA arrive path needs a cluster-scope release after init,
// otherwise tcgen05.commit can miss the barrier and every waiter hangs.
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}
#ifndef MLA_WAIT_MODE
#define MLA_WAIT_MODE 0
#endif
#ifndef MLA_PROLOGUE_ORDER
#define MLA_PROLOGUE_ORDER 0
#endif
#ifndef MLA_SLOT_AHEAD
#define MLA_SLOT_AHEAD 0
#endif
#ifndef MLA_KV_LIN
#define MLA_KV_LIN 0
#endif
#ifndef MLA_SLOT_ORDER
#define MLA_SLOT_ORDER 0
#endif
#if MLA_SLOT_ORDER && MLA_SLOT_AHEAD
#error "MLA_SLOT_ORDER hoists within a tile; MLA_SLOT_AHEAD hoists across tiles. Pick one."
#endif
#ifndef MLA_PV_EARLY
#define MLA_PV_EARLY 2
#endif
#ifndef MLA_LO_F16
#define MLA_LO_F16 1
#endif
#ifndef MLA_FORCE_UNMASKED
#define MLA_FORCE_UNMASKED 0
#endif

#ifndef MLA_UNIFORM_BRANCH
#define MLA_UNIFORM_BRANCH 1
#endif
constexpr bool kUniformBranch = (MLA_UNIFORM_BRANCH != 0);
#ifndef MLA_PAD_FULL
#define MLA_PAD_FULL 1
#endif
constexpr bool kPadFull = (MLA_PAD_FULL != 0);
#ifndef MLA_SKIP_PAD
#define MLA_SKIP_PAD 0
#endif
constexpr bool kSkipPad = (MLA_SKIP_PAD != 0);
constexpr bool kLoF16 = (MLA_LO_F16 != 0);
#ifndef MLA_EPI_BATCH
#define MLA_EPI_BATCH 1
#endif
constexpr int kEpiBatch = MLA_EPI_BATCH;
// The same batching for the in-loop rescale drain, which pays twice per chunk
// (ld + wait + 32 mul + st + wait) and runs on every rescale event, i.e. on the
// long-KV no-split geometries the spot-check list leads with.
#ifndef MLA_RSC_BATCH
#define MLA_RSC_BATCH 1
#endif
constexpr int kRscBatch = MLA_RSC_BATCH;
static_assert(kEpiBatch >= 1 && kEpiBatch <= 4, "MLA_EPI_BATCH out of range");
static_assert(kRscBatch >= 1 && kRscBatch <= 4, "MLA_RSC_BATCH out of range");
#ifndef MLA_EPI_SKIPWARP
#define MLA_EPI_SKIPWARP 0
#endif
constexpr bool kEpiSkipWarp = (MLA_EPI_SKIPWARP != 0);
// The same predicate on the two in-loop drains: the per-tile DR accumulator read
// out of the score scratch (32 KiB per CTA per tile) and the rescale drain that
// reads, scales and writes back all DP columns (458 KiB per event). Those sit in
// the main loop, which MLA_SKIP_PAD showed is latency- rather than
// throughput-bound, so this one is a separate knob: it may well be flat.
#ifndef MLA_TDR_SKIPWARP
#define MLA_TDR_SKIPWARP 0
#endif
constexpr bool kTdrSkipWarp = (MLA_TDR_SKIPWARP != 0);
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  if (kProbe & 4) return;
#if MLA_WAIT_MODE == 1
  asm volatile("{\n.reg .pred q;\nMWAIT:\nmbarrier.test_wait.parity.shared::cta.b64 q, [%0], %1;\n"
               "@!q bra MWAIT;\n}\n" ::"r"(mbar), "r"(phase) : "memory");
#elif MLA_WAIT_MODE == 2
  asm volatile("{\n.reg .pred q;\nMWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1, %2;\n"
               "@!q bra MWAIT;\n}\n" ::"r"(mbar), "r"(phase), "n"(64) : "memory");
#elif MLA_WAIT_MODE == 3
  asm volatile(".pragma \"set knob DontInsertYield\";" ::: "memory");
  asm volatile("{\n.reg .pred q;\nMWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n"
               "@!q bra MWAIT;\n}\n" ::"r"(mbar), "r"(phase) : "memory");
  asm volatile(".pragma \"reset knob DontInsertYield\";" ::: "memory");
#else
  asm volatile("{\n.reg .pred q;\nMWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n"
               "@!q bra MWAIT;\n}\n" ::"r"(mbar), "r"(phase) : "memory");
#endif
}
__device__ __forceinline__ void umma_f8(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n" ::
               "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
                 "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
                 "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]),
                 "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
                 "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n" ::
               "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]),
               "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]),
               "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]),
               "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]),
               "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
                 "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
                 "=r"(v[14]), "=r"(v[15])
               : "r"(taddr));
}
template <int N>
__device__ __forceinline__ void tmem_ldg(uint32_t taddr, uint32_t* v) {
  if (N == 16) tmem_ld16(taddr, v);
  else tmem_ld32(taddr, v);
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
#ifndef MLA_EX2_FTZ
#define MLA_EX2_FTZ 1
#endif
__device__ __forceinline__ float ex2(float x) {
  float r;
#if MLA_EX2_FTZ
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
#else
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
#endif
  return r;
}
// Round an fp32 value onto the E4M3 mantissa grid (three mantissa bits) while
// staying in fp32. Two integer ops: half-ulp up, then mask. Half-up differs
// from the hardware's half-even only on exact ties, and the low residual still
// captures the difference either way.
__device__ __forceinline__ float rn3(float x) {
  uint32_t t = __float_as_uint(x);
  return __uint_as_float((t + 0x00080000u) & 0xFFF00000u);
}
// pack two floats into E4M3x2; the second argument lands in the low byte
__device__ __forceinline__ uint32_t cvt_e4m3x2(float hi, float lo) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return (uint32_t)r;
}
__device__ __forceinline__ uint32_t cvt_f16x2_e4m3x2(uint32_t v) {
  const uint16_t packed = (uint16_t)v;
  uint32_t r;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(packed));
  return r;
}
__device__ __forceinline__ uint32_t cvt_f16x2_f32(float hi, float lo) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t sub_f16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t v) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(v));
  return (uint32_t)r;
}

// SMEM descriptor: K-major or MN-major with SWIZZLE_64B atoms.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t sbo, uint32_t lbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;      // version = SM100
  d |= (uint64_t)4 << 61;      // layout = SWIZZLE_64B
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(int m, int n, int b_mn_major) {
  uint32_t d = 1u << 4;                            // c_format = F32
  d |= (uint32_t)(b_mn_major & 1) << 16;           // b_major
  d |= (uint32_t)((n >> 3) & 0x3F) << 17;          // n_dim
  d |= (uint32_t)((m >> 4) & 0x1F) << 24;          // m_dim
  return d;
}

// Atom-blocked tile offset: [col_atom][row_block][8 rows][64 bytes], the 16-byte
// chunk index inside an atom swizzled by ((row%8)>>1)&3.
__device__ __forceinline__ size_t tile_offset(int row, int chunk16, int row_blocks) {
  int col_atom = chunk16 >> 2;
  int in_atom = chunk16 & 3;
  int rb = row >> 3;
  int ri = row & 7;
  int cx = in_atom ^ ((ri >> 1) & 3);
  return ((size_t)col_atom * row_blocks + rb) * BLOCK_BYTES + (size_t)ri * SW + (size_t)cx * 16;
}


} // namespace mla_v1
