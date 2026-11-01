// Direct CUDA/PTX primitives for the transparent SM100 blockwise FP8 pipeline.
// Origin: FlashInfer d849eca46875, its pinned CUTLASS b46b16d00348;
// See ../baseline/LOCK.json and ../README.md for provenance.
// This file contains PTX primitives, the pipeline and the two specialized entries.
//
// Numeric order: four ascending K32 FP8 MMA instructions form each K128
// partial. Promote with one FP32 FMA using a separately rounded FP32 scale
// product, accumulate K128 blocks in ascending order, then round once to BF16.
// The instruction-internal reduction order is hardware-defined.

#ifndef FIC_PURE_CUH_
#define FIC_PURE_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace fic_pure {

// InstrDescriptor for M=128, N=128, A=E4M3, B=E4M3, C=F32, both K-major,
// dense, no negate, no saturate, max_shift=0. Bit layout from
// arch/mma_sm100_desc.hpp:411-437:
//   c_format[4,6)=1(F32)  a_format[7,10)=0(E4M3)  b_format[10,13)=0(E4M3)
//   a_major[15,16)=0(K)   b_major[16,17)=0(K)
//   n_dim[17,23)=N>>3     m_dim[24,29)=M>>4
static constexpr uint32_t IDESC_128x128 =
    (1u << 4) | (0u << 7) | (0u << 10) | (0u << 15) | (0u << 16) |
    ((uint32_t)(128 >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
static constexpr uint32_t IDESC_128x16 =
    (1u << 4) | ((uint32_t)(16 >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);

// SmemDescriptor (arch/mma_sm100_desc.hpp:98-121), K-major SWIZZLE_128B.
//   start_address[0,14) = smem_byte_addr >> 4
//   leading_byte_offset[16,30) = 1   (stride_10, canonical K-major B128)
//   stride_byte_offset[32,46)  = 64  (stride_01: 8 rows x 128 B = 1024 B = 64 u128)
//   version[46,48) = 1 (Blackwell)
//   base_offset[49,52) = 0, lbo_mode[52,53) = 0 (legacy)
//   layout_type[61,64) = 2 (SWIZZLE_128B)
static constexpr uint64_t SDESC_FIXED =
    (uint64_t(1) << 16) |      // leading_byte_offset_ = 1
    (uint64_t(64) << 32) |     // stride_byte_offset_  = 64
    (uint64_t(1) << 46) |      // version_ = 1
    (uint64_t(2) << 61);       // layout_type_ = SWIZZLE_128B

// ---------------------------------------------------------------------------
// PTX primitives
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int elect_one() {
  uint32_t pred = 0;
#if __CUDA_ARCH__ >= 1000
  asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n\t}\n"
               : "=r"(pred));
#endif
  return static_cast<int>(pred);
}

// Canonical UMMA K-major SWIZZLE_128 byte offset for one (row, col_byte) of a
// 128-row x 128-byte tile: Swizzle<3,4,3> o ((8,16),(128)) : ((128,1024),(1)).
// Bits [7,10) of the unswizzled offset hold row&7 and the swizzle XORs them
// into bits [4,7), i.e. it permutes the eight 16-byte chunks of each 128-byte
// core-matrix row.
__device__ __host__ __forceinline__ int sw128_offset(int row, int col_byte) {
  int off = (row >> 3) * 1024 + (row & 7) * 128 + col_byte;
  return off ^ (((off >> 7) & 7) << 4);
}

__device__ __forceinline__ uint64_t make_sdesc(const void* smem_ptr) {
  uint32_t a = smem_addr(smem_ptr);
  return SDESC_FIXED | uint64_t((a >> 4) & 0x3FFF);
}

// One tcgen05.mma, cta_group::1, kind::f8f6f4, both operands from SMEM.
// scaleC == 0 overwrites the TMEM accumulator, scaleC != 0 accumulates into it
// (UMMA::ScaleOut::Zero / ::One upstream).
__device__ __forceinline__ void umma_f8f6f4(uint32_t tmem_c, uint64_t desc_a,
                                            uint64_t desc_b, uint32_t idesc,
                                            uint32_t scaleC) {
  // Operand shape is exactly upstream SM100_MMA_F8F6F4_SS::fma, including the
  // four zeroed disable-mask registers passed as .b32 operands (not immediates)
  // with architectural election inside the participating MMA warp.
  uint32_t mask[4] = {0u, 0u, 0u, 0u};
  asm volatile(
      "{\n\t"
      ".reg .pred p, leader;\n\t"
      "elect.sync _|leader, 0xffffffff;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
                   "r"(smem_addr(dst_smem)), "r"(ncols));
}

__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}

__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                   "r"(taddr), "r"(ncols));
}

// Make the MMA's TMEM writes observable by this warp's tcgen05.ld, and signal
// completion through an mbarrier (upstream: accumulator_pipeline.producer_commit).
__device__ __forceinline__ void umma_commit(uint32_t mbar_smem) {
  asm volatile("{ .reg .pred leader; elect.sync _|leader, 0xffffffff; "
               "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"
               :: "r"(mbar_smem));
}

__device__ __forceinline__ void mbar_init(uint32_t mbar_smem, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_smem), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar_smem, uint32_t phase) {
  // Keep the wait entirely inside one PTX block, as the pinned barrier helper
  // does. Returning a predicate to a C++ loop introduces lane-divergent
  // convergence bookkeeping around otherwise warp-uniform pipeline roles.
  asm volatile(
      "{\n\t.reg .pred ready;\n\t"
      "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1, %2;\n\t"
      "@ready bra DONE;\n\tbra WAIT;\n\tDONE:\n\t}\n"
      :: "r"(mbar_smem), "r"(phase), "r"(0x989680u) : "memory");
}

// Generic-proxy SMEM stores must be made visible to the async proxy that
// tcgen05.mma reads through, before the MMA is issued.
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::);
}

__device__ __forceinline__ void fence_after_tmem_load() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

#define FIC_T32_OUT(v)                                                        \
  "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]),    \
      "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]),           \
      "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),       \
      "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]),       \
      "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]),       \
      "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]),       \
      "=r"(v[31])

// Read 32 consecutive TMEM columns of this warp's 32 lanes into 32 registers.
__device__ __forceinline__ void tmem_load_32x32b_x32(uint32_t* v, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : FIC_T32_OUT(v)
      : "r"(taddr));
}

// fp32 -> bf16 round-to-nearest-even, the same conversion the pinned epilogue
// performs on its way to ElementD = bfloat16.
// Same nearest-even packed conversion as the pinned native epilogue.
// PTX packs its first source into the high half and its second into the low.
__device__ __forceinline__ uint32_t bf16x2_rn(float low, float high) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(high), "f"(low));
  return packed;
}

__device__ __forceinline__ uint16_t bf16_rn(float x) {
  uint16_t out;
  asm volatile("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(out) : "f"(x));
  return out;
}

__device__ __forceinline__ void tmem_load_32x32b_x16(uint32_t* v, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
      : "r"(taddr));
}

}  // namespace fic_pure

#endif  // FIC_PURE_CUH_

// Pure CUDA/PTX restatement of the pinned SM100 blockwise mainloop.
// The arithmetic order is unchanged from kernel.cuh. Pinned pipeline depths, a
// dedicated TMA warp, a dedicated MMA warp, independent TMEM accumulator slots,
// four promotion/epilogue warps, and CLC work stealing restore the optimization
// mechanisms that the original side extraction omitted.
// No CUTLASS/CuTe implementation or device GEMM interface is linked.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace fic_pure {
constexpr int PIPE_THREADS = 384;

// Persistent ring state follows the original PipelineState increment: only
// wrap changes parity. Reconstructing it with division for every K block is
// unnecessary and obscures the independent producer/consumer progress.
template<int Stages> struct PipeCursor {
  int index=0;
  uint32_t phase=0;
  bool cycled=false;
  __device__ __forceinline__ void next() {
    if (++index==Stages) { index=0; phase^=1; cycled=true; }
  }
};

__device__ __forceinline__ void pipe_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void pipe_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void pipe_load(void* dst, const CUtensorMap* map,
                                          int x, int y, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem_addr(dst)), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void pipe_store(const CUtensorMap* map, void* src, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               "[%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_addr(src)), "r"(x), "r"(y) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Match upstream's scale-copy warp: asynchronous four-byte global-to-shared
// copies use compact scale storage, independently of the A/B TMA engine.
__device__ __forceinline__ void pipe_scale_copy(float* dst,const float* src,bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
               :: "r"(smem_addr(dst)),"l"(src),"r"(valid ? 4 : 0) : "memory");
}
__device__ __forceinline__ void pipe_scale_commit(uint32_t barrier) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
               :: "r"(barrier) : "memory");
}

// Exact pinned mainloop configuration: dense 5 input / 4 accumulator stages;
// small-M transposed 12 / 32. Scale storage is a separate compact
// ring, as upstream, so the small-M input ring fits the SM100 shared-memory cap.
template<int TILE_N> struct PipeStorage {
  static constexpr int INPUT_STAGES=TILE_N==16 ? 12 : 5;
  static constexpr int ACC_STAGES=TILE_N==16 ? 32 : 4;
  struct alignas(1024) Stage { uint8_t a[128*128]; uint8_t b[TILE_N*128]; } stage[INPUT_STAGES];
  struct alignas(16) Scale { float a[TILE_N==16 ? 16 : 128]; float b; } scale[INPUT_STAGES];
  alignas(8) uint64_t full[INPUT_STAGES],empty[INPUT_STAGES];
  alignas(8) uint64_t scale_full[INPUT_STAGES],scale_empty[INPUT_STAGES];
  alignas(8) uint64_t acc_full[ACC_STAGES],acc_empty[ACC_STAGES];
  static constexpr int SCHED_STAGES=(TILE_N==16 ? 33 : 5);
  alignas(8) uint64_t clc_full[SCHED_STAGES],clc_empty[SCHED_STAGES];
  alignas(16) uint32_t clc[SCHED_STAGES][4],tmem;
  // Separate output sector: input producer can start the next tile while
  // consumers store this tile. One sector fits alongside all pinned stages.
  // sw128_offset is relative to a 1024-byte swizzle base.
  alignas(1024) uint16_t epilogue[(TILE_N==16 ? 16 : 128)*64];
};

// Each original warp role owns a separate persistent loop. Compile-time
// specialization keeps role-private state out of the other roles' live ranges.
enum class PipeRole { AB, Scale, MMA, Promotion };
template<PipeRole ROLE, int TILE_N, bool TRANSPOSE>
__device__ __forceinline__ void pipe_role(
    PipeStorage<TILE_N>& sm, const CUtensorMap& map_a,
    const CUtensorMap& map_b, const CUtensorMap& map_d,
    const float* sfa, const float* sfb, uint16_t* out,
    int M, int N, int K, int tiles_n, int tid, int warp) {
  constexpr int INPUT_STAGES=PipeStorage<TILE_N>::INPUT_STAGES;
  constexpr int ACC_STAGES=PipeStorage<TILE_N>::ACC_STAGES;
  constexpr int SCHED_STAGES=PipeStorage<TILE_N>::SCHED_STAGES;
  const int KB=K/128;
  int work=blockIdx.x, tile=0;
  PipeCursor<INPUT_STAGES> input_state;
  PipeCursor<ACC_STAGES> accumulator_state;
  while (work>=0) {
    int tile_m = work / tiles_n, tile_n = work % tiles_n;
    // Match pinned CLC heuristic: the shorter tile dimension varies fastest.
    // The transposed small-M route already has its short dimension along N.
    if constexpr (!TRANSPOSE) {
      int tiles_m=(M+127)/128;
      if (tiles_n>tiles_m) { tile_m=work%tiles_m; tile_n=work/tiles_m; }
    }
    const int row0 = tile_m * 128;
    const int col0 = tile_n * TILE_N;

    if constexpr (ROLE == PipeRole::AB) {
      #pragma unroll 1
      for (int kt=0;kt<KB;++kt) {
        int s=input_state.index;
        if (input_state.cycled) mbar_wait(smem_addr(&sm.empty[s]),input_state.phase^1);
        pipe_expect(smem_addr(&sm.full[s]),(128+TILE_N)*128);
        pipe_load(sm.stage[s].a,&map_a,kt*128,row0,smem_addr(&sm.full[s]));
        pipe_load(sm.stage[s].b,&map_b,kt*128,0+col0,smem_addr(&sm.full[s]));
        input_state.next();
      }
    }
    // The scale producer has its own credits: A/B buffers are released by
    // MMA completion, while scales are released once promotion copies them
    // into registers. This preserves upstream's independent AB/SF pipelines.
    if constexpr (ROLE == PipeRole::Scale) {
      const int lane=tid%32;
      #pragma unroll 1
      for (int kt=0;kt<KB;++kt) {
        int s=input_state.index;
        if (input_state.cycled) mbar_wait(smem_addr(&sm.scale_empty[s]),input_state.phase^1);
        #pragma unroll
        for (int i=lane;i<(TRANSPOSE ? 16 : 128);i+=32) {
          int r=(TRANSPOSE ? col0 : row0)+i;
          int64_t offset=int64_t(r)*KB+kt;
          pipe_scale_copy(&sm.scale[s].a[i],sfa+(r<M ? offset : 0),r<M);
        }
        if (lane==0) {
          int64_t offset=int64_t(TRANSPOSE ? row0/128 : col0/128)*KB+kt;
          pipe_scale_copy(&sm.scale[s].b,sfb+offset,true);
        }
        pipe_scale_commit(smem_addr(&sm.scale_full[s]));
        input_state.next();
      }
    }
    if constexpr (ROLE == PipeRole::MMA) {
      #pragma unroll 1
      for (int kt=0;kt<KB;++kt) {
        int s=input_state.index, acc_s=accumulator_state.index;
        if (accumulator_state.cycled) mbar_wait(smem_addr(&sm.acc_empty[acc_s]),accumulator_state.phase^1);
        mbar_wait(smem_addr(&sm.full[s]),input_state.phase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint64_t da=make_sdesc(sm.stage[s].a), db=make_sdesc(sm.stage[s].b);
        constexpr uint32_t idesc = TILE_N==16 ? IDESC_128x16 : IDESC_128x128;
        const uint32_t dst=sm.tmem+acc_s*TILE_N;
        #pragma unroll
        for (int j=0;j<4;++j) umma_f8f6f4(dst,da+2*j,db+2*j,idesc,j!=0);
        umma_commit(smem_addr(&sm.acc_full[acc_s]));
        umma_commit(smem_addr(&sm.empty[s]));
        input_state.next(); accumulator_state.next();
      }
    }
    if constexpr (ROLE == PipeRole::Promotion) {
      // Keep the increase inside the producer/consumer control-flow split.
      // Reconverging with 48-register producers before this point would make
      // ptxas spill the promotion fragment despite the hardware allocation.
      asm volatile("setmaxnreg.inc.sync.aligned.u32 256;" ::: "memory");
      float acc[TILE_N];
      #pragma unroll
      for (int j=0;j<TILE_N;++j) acc[j]=0.f;
      const int row=row0+tid;
      #pragma unroll 1
      for (int kt=0;kt<KB;++kt) {
        const int acc_s=accumulator_state.index, sf_s=input_state.index;
        mbar_wait(smem_addr(&sm.scale_full[sf_s]),input_state.phase);
        float scale[TRANSPOSE ? 16 : 1];
        float sb=sm.scale[sf_s].b;
        #pragma unroll
        for (int j=0;j<(TRANSPOSE ? 16 : 1);++j) {
          int idx=TRANSPOSE ? j : tid;
          scale[j]=sm.scale[sf_s].a[idx]*sb;
        }
        pipe_arrive(smem_addr(&sm.scale_empty[sf_s]));
        mbar_wait(smem_addr(&sm.acc_full[acc_s]),accumulator_state.phase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t address=sm.tmem+acc_s*TILE_N+(uint32_t(warp*32)<<16);
        if constexpr (TRANSPOSE) {
          uint32_t part[16]; tmem_load_32x32b_x16(part,address); fence_after_tmem_load();
          #pragma unroll
          for (int j=0;j<16;++j) {
            acc[j]=fmaf(scale[j],__uint_as_float(part[j]),acc[j]);
          }
        } else {
          #pragma unroll
          for (int c=0;c<4;++c) {
            uint32_t part[32]; tmem_load_32x32b_x32(part,address+c*32); fence_after_tmem_load();
            #pragma unroll
            for (int j=0;j<32;++j) {
              asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(acc[c*32+j])
                           : "f"(scale[0]), "f"(__uint_as_float(part[j])));
            }
          }
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        pipe_arrive(smem_addr(&sm.acc_empty[acc_s]));
        input_state.next(); accumulator_state.next();
      }
      // Only promotion threads synchronize here; AB/SF/MMA keep filling
      // the next tile. Output storage is independent of their input rings.
      #pragma unroll
      for (int sector=0;sector<2;++sector) {
        if constexpr (TRANSPOSE) {
          if (tid/64==sector) {
            #pragma unroll
            for (int r=0;r<16;++r)
              sm.epilogue[sw128_offset(r,(tid%64)*2)/2]=bf16_rn(acc[r]);
          }
        } else {
          // Native epilogue writes 128 bits per instruction. Eight adjacent
          // columns stay within one 16-byte swizzle atom, so this vector is
          // aligned and eliminates the scalar halfword shared-store path.
          #pragma unroll
          for (int c=0;c<64;c+=8) {
            const int i=sector*64+c;
            uint4 packed={bf16x2_rn(acc[i],acc[i+1]),
                          bf16x2_rn(acc[i+2],acc[i+3]),
                          bf16x2_rn(acc[i+4],acc[i+5]),
                          bf16x2_rn(acc[i+6],acc[i+7])};
            *reinterpret_cast<uint4*>(sm.epilogue+sw128_offset(tid,c*2)/2)=packed;
          }
        }
        fence_async_shared();
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (tid==0) {
          pipe_store(&map_d,sm.epilogue,(TRANSPOSE ? row0 : col0)+sector*64,
                     TRANSPOSE ? col0 : row0);
        }
        // Shared output can be reused only after the TMA read completes.
        asm volatile("bar.sync 1, 128;" ::: "memory");
      }
    }
  
    
      int s=tile%SCHED_STAGES;
      mbar_wait(smem_addr(&sm.clc_full[s]),(tile/SCHED_STAGES)&1);
      uint32_t x=0,y=0,z=0,valid=0;
      asm volatile("{ .reg .pred p; .reg .b128 r; ld.shared.b128 r,[%4]; "
                   "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p,r; "
                   "selp.u32 %3,1,0,p; mov.u32 %0,0; mov.u32 %1,0; mov.u32 %2,0; "
                   "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0,%1,%2,_},r; }"
                   : "=r"(x),"=r"(y),"=r"(z),"=r"(valid) : "r"(smem_addr(sm.clc[s])) : "memory");
      pipe_arrive(smem_addr(&sm.clc_empty[s]));
      work=valid ? int(x) : -1;
    
    ++tile;
  }
}

} // namespace fic_pure
namespace fic_pure {
template<int TILE_N, bool TRANSPOSE>
static __global__ void __launch_bounds__(PIPE_THREADS, 1)
pipe_kernel(const __grid_constant__ CUtensorMap map_a,
            const __grid_constant__ CUtensorMap map_b,
            const __grid_constant__ CUtensorMap map_d,
            const float* __restrict__ sfa, const float* __restrict__ sfb,
            uint16_t* __restrict__ out,
            int M, int N, int K, int tiles_n) {
  extern __shared__ __align__(1024) uint8_t bytes[];
  auto& sm = *reinterpret_cast<PipeStorage<TILE_N>*>(bytes);
  constexpr int INPUT_STAGES=PipeStorage<TILE_N>::INPUT_STAGES;
  constexpr int ACC_STAGES=PipeStorage<TILE_N>::ACC_STAGES;
  const int tid = threadIdx.x;
  // Pinned canonical_warp_idx_sync explicitly broadcasts lane0's index so
  // the compiler can treat every role branch as warp-uniform.
  const int warp = __shfl_sync(0xffffffff, tid / 32, 0);
  // Match the original 384-thread CTA and its dynamic register allocation.
  // Promotion owns 256 registers/thread; producer/idle warps release to 48.
  if (warp == 0) { tmem_alloc(&sm.tmem, TILE_N * ACC_STAGES); tmem_relinquish(); }
  __syncthreads();
  if (warp>=4) asm volatile("setmaxnreg.dec.sync.aligned.u32 48;" ::: "memory");
  constexpr int SCHED_STAGES=PipeStorage<TILE_N>::SCHED_STAGES;
  const int KB = K / 128;
  if (tid==0) {
    for (int s=0;s<INPUT_STAGES;++s) {
      mbar_init(smem_addr(&sm.full[s]),1); mbar_init(smem_addr(&sm.empty[s]),1);
      mbar_init(smem_addr(&sm.scale_full[s]),32); mbar_init(smem_addr(&sm.scale_empty[s]),128);
    }
    for (int s=0;s<ACC_STAGES;++s) {
      mbar_init(smem_addr(&sm.acc_full[s]),1); mbar_init(smem_addr(&sm.acc_empty[s]),128);
    }
    for (int s=0;s<SCHED_STAGES;++s) {
      mbar_init(smem_addr(&sm.clc_full[s]),1);
      // 128 promotion + 1 AB + 32 MMA + 32 SF threads consume each result.
      mbar_init(smem_addr(&sm.clc_empty[s]),193);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // Choose each warp role once, outside its persistent tile loop.
  if (warp<4)
    pipe_role<PipeRole::Promotion,TILE_N,TRANSPOSE>(sm,map_a,map_b,map_d,sfa,sfb,out,M,N,K,tiles_n,tid,warp);
  else if (warp==4) {
    if (tid==128)
      pipe_role<PipeRole::AB,TILE_N,TRANSPOSE>(sm,map_a,map_b,map_d,sfa,sfb,out,M,N,K,tiles_n,tid,warp);
  }
  else if (warp==5)
    pipe_role<PipeRole::MMA,TILE_N,TRANSPOSE>(sm,map_a,map_b,map_d,sfa,sfb,out,M,N,K,tiles_n,tid,warp);
  else if (warp==6) {
  
    if (tid==192) {
      for (int q=0;;++q) {
        int s=q%SCHED_STAGES;
        if (q>=SCHED_STAGES) mbar_wait(smem_addr(&sm.clc_empty[s]),((q/SCHED_STAGES)-1)&1);
        pipe_expect(smem_addr(&sm.clc_full[s]),16);
        asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
                     :: "r"(smem_addr(sm.clc[s])), "r"(smem_addr(&sm.clc_full[s])) : "memory");
        mbar_wait(smem_addr(&sm.clc_full[s]),(q/SCHED_STAGES)&1);
        uint32_t valid;
        asm volatile("{ .reg .pred p; .reg .b128 r; ld.shared.b128 r,[%1]; "
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p,r; selp.u32 %0,1,0,p; }"
                     : "=r"(valid) : "r"(smem_addr(sm.clc[s])) : "memory");
        if (!valid) break;
      }
    }
  
  }
  else if (warp==7)
    pipe_role<PipeRole::Scale,TILE_N,TRANSPOSE>(sm,map_a,map_b,map_d,sfa,sfb,out,M,N,K,tiles_n,tid,warp);
  __syncthreads();
  if (warp==0) tmem_dealloc(sm.tmem,TILE_N * ACC_STAGES);

}
} // namespace fic_pure
