#include <tuple>
// SM103 sparse attention: 64-head WS or 128-head FP16 TCGEN05.
// No CUTLASS/CuTe dependency. Register/TMEM helpers reused from our MLA kernel.
namespace plmn_tc {
__device__ __forceinline__ unsigned sa(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void wait(unsigned b, unsigned phase) {
  asm volatile(
      "{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" ::"r"(b),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void commit(unsigned b) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(b)
      : "memory");
}
__device__ __forceinline__ void before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void publish() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ int off(int row, int byte, int rows) {
  return ((byte / 64) * (rows / 8) + row / 8) * 512 + (row % 8) * 64 +
         ((byte % 64) ^ (((row % 8) / 2) * 16));
}
__device__ __forceinline__ uint64_t desc(const void *p, int sbo = 512, int lbo = 0) {
  return ((uint64_t)(sa(p) >> 4)) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) |
         (1ull << 46) | (4ull << 61);
}
template<int Heads>
__device__ __forceinline__ unsigned idesc(int n, bool mn) {
  // Bits [7,10) and [10,13) are the A and B operand formats, both 1 = BF16 here.
  // SM103 accepts F16/F16 and BF16/BF16 but traps on a mixed pair (micro_fmt:
  // bf16f16 -> illegal instruction, bf16bf16 -> exact), so all three tiles of this
  // kernel share one format, and BF16 is the one that needs no work on Q: the TMA
  // stages the caller's bf16 Q straight into the SW64 layout this descriptor
  // describes, so there is no conversion pass over the Q tile at all. The decoded
  // KV values are exact in bf16 (e4m3 x ue8m0 needs 4 significand bits, e2m1 x
  // e4m3 needs 7, bf16 has 8); the probability tile loses three significand bits
  // against fp16, which is what the production library this question is measured
  // against does as well, and its frozen fixed/domain receipts carry the same
  // worst-case error as this candidate's (see Tile16 in kernel_mma.cuh).
  return (1u << 4) | (1u << 7) | (1u << 10) | (unsigned(mn) << 16) | ((n / 8) << 17) |
         ((Heads / 16u) << 24);
}
template<int Heads>
__device__ __forceinline__ void mma(unsigned d, uint64_t a, uint64_t b, unsigned id,
                                    unsigned accumulate) {
  if constexpr (Heads == 64) {
    asm volatile(
        "{.reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.ws.cta_group::1.kind::f16 [%0],%1,%2,%3,p,0;}" ::
            "r"(d), "l"(a), "l"(b), "r"(id), "r"(accumulate) : "memory");
  } else {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}" ::
          "r"(d),
      "l"(a), "l"(b), "r"(id), "r"(accumulate)
      : "memory");
  }
}
template<int Heads> constexpr int SMEM = 4096 + Heads * 1024 + 65536 + Heads * 128;
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t *v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
                 "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]),
                 "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t *v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"r"(v[0]),
               "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]),
               "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]),
               "r"(v[15]), "r"(taddr));
}

struct RawTile {
  uint4 a, b, c, d;
  uint64_t scales;
  bool valid, fp4;
};
__device__ __forceinline__ RawTile load_tile(plmn::FwdCtx c, int bs, int ci, int tid) {
  RawTile v{};
  const int kr = tid % 64, part = (tid / 64) * 2, e = ci * 64 + kr;
  bool second = e >= c.budget1;
  int budget = second ? c.budget2 : c.budget1;
  int slot = e < c.listn
                 ? (second ? c.idx2 : c.idx1)[(long)bs * budget + (second ? e - c.budget1 : e)]
                 : -1;
  v.valid = slot >= 0;
  if (v.valid) {
    int rec = second ? c.rec2 : c.rec1, page = second ? c.page2 : c.page1;
    v.fp4 = rec == 288;
    int width = v.fp4 ? 256 : 512, ns = v.fp4 ? 32 : 16;
    const auto *base = (second ? c.k2 : c.k1) + (long)(slot / page) * page * rec;
    const auto *data = base + (slot % page) * width + part * (v.fp4 ? 16 : 32);
    const auto *sc = base + page * width + (slot % page) * ns + part * (v.fp4 ? 2 : 1);
    v.scales = v.fp4 ? cfg32::load_scale_run<4>(sc) : cfg32::load_scale_run<2>(sc);
    v.a = *(const uint4 *)data;
    v.b = *(const uint4 *)(data + 16);
    if (!v.fp4) {
      v.c = *(const uint4 *)(data + 32);
      v.d = *(const uint4 *)(data + 48);
    }
  }
  return v;
}

// Device-scope release/acquire on the row counters that order the fused merge.
// The release publishes this CTA's partial; the acquire makes every other split's
// partial visible before it is read back.
__device__ __forceinline__ unsigned count_release(int *p) {
  unsigned old;
  asm volatile("atom.add.release.gpu.global.u32 %0,[%1],1;" : "=r"(old) : "l"(p) : "memory");
  return old;
}
__device__ __forceinline__ unsigned count_acquire(const int *p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void count_store(int *p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p), "r"(v) : "memory");
}

// o[0..15] += fs * one 16-dim slice of a split partial, held in two 16 B loads.
__device__ __forceinline__ void acc16(float *o, uint4 r0, uint4 r1, float fs) {
  const __half2 *a = reinterpret_cast<const __half2 *>(&r0);
  const __half2 *b = reinterpret_cast<const __half2 *>(&r1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __half22float2(a[i]), y = __half22float2(b[i]);
    o[2 * i] += fs * x.x;
    o[2 * i + 1] += fs * x.y;
    o[8 + 2 * i] += fs * y.x;
    o[8 + 2 * i + 1] += fs * y.y;
  }
}

// One warp merges one (request row, head) of the split partials: lane `lane` owns
// output dims [16*lane, 16*lane+16). The arithmetic is the standalone combine's
// (cfg32::plmn_combine_wide_hb32) term for term -- same per-split weights, same
// log2-domain sink rescale, same +Inf/zero rule when no split saw a token -- so
// folding the merge into the main kernel changes only when it runs. The per-split
// weight stays in the lane that owns it and is broadcast with a shuffle, which
// removes the combine's shared-memory staging without changing a value.
// `own` is this CTA's shared-memory staging row for the head (520 halves apart,
// the epilogue's own layout) and `own_split` the split index it holds; the other
// splits are read from the global partials. Keeping a CTA's own contribution in
// shared memory removes one whole write-plus-read of it per head, i.e. 1/splits of
// the merge traffic.
__device__ __forceinline__ void merge_head(const __half *__restrict__ po,
                                           const float *__restrict__ ml,
                                           const float *__restrict__ sink,
                                           __nv_bfloat16 *__restrict__ out,
                                           float *__restrict__ lse, const __half *own,
                                           int own_split, int splits, int H, int rh, int lane) {
  const long mlstride = 2;                       // the splits of one head are adjacent
  const float *mp = ml + (long)rh * splits * 2;
  float mv = cfg32::ninf(), lv = 0.f;
  if (lane < splits) {
    mv = __ldg(mp + (long)lane * mlstride);
    lv = __ldg(mp + (long)lane * mlstride + 1);
  }
  float mmax = mv;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) mmax = fmaxf(mmax, __shfl_xor_sync(0xffffffffu, mmax, d));
  const float f = (lane < splits && mv != cfg32::ninf()) ? cfg32::ex2(mv - mmax) : 0.f;
  float lsum = f * lv;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, d);
  const float lse2 = mmax + log2f(lsum);
  const int h = rh - (rh / H) * H;
  const float sk2 = __ldg(sink + h) * cfg32::LOG2E;
  const float lae2 = fmaxf(lse2, sk2) + log2f(1.f + cfg32::ex2(-fabsf(lse2 - sk2)));
  const float scale = (mmax == cfg32::ninf()) ? 0.f : cfg32::ex2(mmax - lae2);
  if (lane == 0) lse[rh] = (mmax == cfg32::ninf()) ? cfg32::pinf() : lse2 * cfg32::LN2;

  const long ostride = cfg32::D;
  const __half *pr = po + (long)rh * splits * cfg32::D + lane * 16;
  float o[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i] = 0.f;
  // This CTA's own share never went to global, so it is read straight out of the
  // epilogue's shared-memory staging row.
  acc16(o, *reinterpret_cast<const uint4 *>(own + lane * 16),
        *reinterpret_cast<const uint4 *>(own + lane * 16 + 8),
        __shfl_sync(0xffffffffu, f, own_split));
  // The peer splits are all in global. Every partial element is read exactly
  // once, by exactly one CTA, so those loads stream: an evict-first hint stops a
  // whole row of split partials from displacing anything the next forward will
  // reuse. They are also read unconditionally and weighted, because a split that
  // saw no token published zeros together with weight 0 -- skipping it instead
  // would make each load wait for its own weight before it could be issued.
  // Four splits are loaded before any of them is consumed: with one
  // load-accumulate pair per iteration only two 16 B requests per lane are ever in
  // flight and the merge crawls at ~1.3 TB/s however many CTAs are merging, while
  // a batch of four is eight requests per lane for 32 registers.
  const int peers = splits - 1;
  int j = 0;
  for (; j + 4 <= peers; j += 4) {
    uint4 v[8];
    float w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int s = j + k + ((j + k >= own_split) ? 1 : 0);
      const __half *src = pr + (long)s * ostride;
      v[2 * k] = __ldcs(reinterpret_cast<const uint4 *>(src));
      v[2 * k + 1] = __ldcs(reinterpret_cast<const uint4 *>(src + 8));
      w[k] = __shfl_sync(0xffffffffu, f, s);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc16(o, v[2 * k], v[2 * k + 1], w[k]);
  }
  for (; j < peers; ++j) {
    const int s = j + (j >= own_split);
    const __half *src = pr + (long)s * ostride;
    acc16(o, __ldcs(reinterpret_cast<const uint4 *>(src)),
          __ldcs(reinterpret_cast<const uint4 *>(src + 8)), __shfl_sync(0xffffffffu, f, s));
  }
  uint4 w0, w1;
  __nv_bfloat162 *p0 = reinterpret_cast<__nv_bfloat162 *>(&w0);
  __nv_bfloat162 *p1 = reinterpret_cast<__nv_bfloat162 *>(&w1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p0[i] = __floats2bfloat162_rn(o[2 * i] * scale, o[2 * i + 1] * scale);
    p1[i] = __floats2bfloat162_rn(o[8 + 2 * i] * scale, o[8 + 2 * i + 1] * scale);
  }
  __nv_bfloat16 *ob = out + (long)rh * cfg32::D + lane * 16;
  *reinterpret_cast<uint4 *>(ob) = w0;
  *reinterpret_cast<uint4 *>(ob + 8) = w1;
}

template<int Heads>
__global__ __launch_bounds__(512, 1) void attention(plmn::FwdCtx c, int splits, __half *po,
                                                    float *ml,
                                                    const __grid_constant__ CUtensorMap qmap,
                                                    bool use_tma, int *arrive, int *done) {
  extern __shared__ __align__(1024) unsigned char sm[];
  unsigned *alloc = (unsigned *)sm;
  unsigned long long *bar = (unsigned long long *)(sm + 16);
  unsigned long long *qbar = (unsigned long long *)(sm + 32);
  float *exchange = (float *)(sm + 64);
  int *valid = (int *)(sm + 2176);
  unsigned char *q = sm + 4096;
  unsigned char *k = q + Heads * 1024;
  unsigned char *prob = k + 65536;
  constexpr int Groups = 512 / Heads, Values = 64 / Groups;
  constexpr int Score = Heads == 64 ? 256 : 448;
  constexpr int OutputCols = Heads == 64 ? 256 : 448;
  const int tid = threadIdx.x, row = tid % Heads, bs = blockIdx.x, sp = blockIdx.y;
  // H=128 can be served either by one 128-head CTA per request row or by a pair of
  // 64-head CTAs (gridDim.z = head blocks). The pair halves the Q staging and the
  // per-CTA output tile, which is what lets the host trade splits against head
  // blocks: BS*nhb*splits CTAs either way, but nhb=2 needs only half the splits for
  // the same occupancy, and the split partials - not the attention work - are what
  // the small-batch shapes pay for. Every head index below is therefore absolute
  // (hbase + local) while every shared-memory index stays block-local.
  const int hb = blockIdx.z, nhb = gridDim.z, hbase = hb * Heads;
  // A warp accesses its own 32 TMEM datapaths. WS alternates the two
  // logical N halves between the lower and upper warp pair of each WG.
  const int group = Heads == 64 ? tid / 128 + ((tid % 128) / 64) * 4 : tid / 128;
  // WS packs the upper N half into datapaths 64..127; each N256 PV tile
  // therefore occupies 128 columns. All D512 accumulators stay in TMEM.
  const int data_row = row + (Heads == 64 ? (group / 4) * 64 : 0);
  const int score_col = Heads == 64 ? (group % 4) * Values : group * Values;
  const int output_col = (group % 4) * 16;
  const int chunks = (c.listn + 63) / 64;
  // Cyclic assignment balances both tile counts and the two KV formats. Measured on
  // this card the 528-byte MXFP8 chunk is no more expensive than the 288-byte
  // NVFP4 one (2.7 vs 3.1 us: the fp4 path decodes twice as many groups per token),
  // so weighting the two formats by their record width - which is what a
  // cost-balanced contiguous cut would do - only unbalances the chunk counts and
  // costs a whole chunk time (evidence/online-20261024/raw: sw4-n2s5 22.576 and
  // sw5-n2s5 22.848 against cyclic swp-s4-sp5 19.120 and swp-s5-sp5 19.328).
  const int first = sp, last = chunks;
  // The sparse gather is the longest-latency request this CTA issues, so it goes
  // out first: barrier initialisation, the TMEM allocation and the Q bulk copy
  // then all overlap with it instead of delaying it, and the decode below finds
  // its registers already filled.
  RawTile current{};
  if (first < last)
    current = load_tile(c, bs, first, tid);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(sa(bar)));
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(sa(qbar)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (tid < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],512;" ::"r"(sa(alloc)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  unsigned tm = *alloc, phase = 0;
  // Bulk TMA staging copies the caller's bf16 Q bytes into the same SW64 layout the
  // matrix descriptor describes. The tiles are bf16 (see idesc), so those bytes are
  // already the A operand and no in-place conversion pass follows.
  if (use_tma) {
    if (tid == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(sa(qbar)), "r"(Heads * 1024)
                   : "memory");
#pragma unroll
      for (int atom = 0; atom < 16; ++atom) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0],[%1,{%2,%3,%4}],[%5];" ::"r"(sa(q + atom * Heads * 64)),
                     "l"((unsigned long long)&qmap), "r"(atom * 32), "r"(hbase), "r"(bs),
                     "r"(sa(qbar))
                     : "memory");
      }
    }
  } else {
    // Without a tensor map, cp.async stages the same bf16 bytes into the same
    // SW64 layout. Padded heads beyond c.H are zero-filled by the size predicate.
    for (int j = tid; j < Heads * 64; j += 512) {
      int r = j / 64, d = (j % 64) * 8;
      const void *src = c.q + ((long)bs * c.H + min(hbase + r, c.H - 1)) * 512 + d;
      unsigned dst = sa(q + off(r, d * 2, Heads));
      int bytes = hbase + r < c.H ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(dst), "l"(src), "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  float m = -INFINITY, l = 0.f, tail[16];
#pragma unroll
  for (int j = 0; j < 16; ++j)
    tail[j] = 0.f;
  bool have = false;
  for (int ci = first; ci < last; ci += splits) {
    {
      const int kr = tid % 64, part0 = (tid / 64) * 2;
      if (part0 == 0)
        valid[kr] = current.valid;
#pragma unroll 1
      for (int piece = 0; piece < 2; ++piece) {
        const int part = part0 + piece;
        __align__(16) __nv_bfloat16 tmp[32];
        if (current.valid) {
          if (current.fp4)
            cfg32::decode_nvfp4_words<true>(piece ? current.b : current.a,
                                            (current.scales >> (piece * 16)) & 255,
                                            (current.scales >> (piece * 16 + 8)) & 255, tmp);
          else
            cfg32::decode_mxfp8_words<true>(piece ? current.c : current.a,
                                            piece ? current.d : current.b,
                                            (current.scales >> (piece * 8)) & 255, tmp);
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i)
            ((uint4 *)tmp)[i] = make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
          *(uint4 *)(k + off(kr, part * 64 + i * 16, 64)) = ((uint4 *)tmp)[i];
      }
    }
    // Q's staging lands asynchronously, so the first chunk's decode runs while it is
    // still in flight. The tiles are bf16 (see idesc), so the bytes the TMA staged
    // are already the MMA's A operand: there is nothing to convert and nothing to
    // wait for beyond the copy itself. Padded heads are zero-filled by the TMA's
    // out-of-bounds fill and by the cp.async size predicate.
    if (ci == first) {
      if (use_tma)
        wait(sa(qbar), 0);
      else
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }
    publish();
    before();
    __syncthreads();
    after();
    if (tid == 0) {
#pragma unroll
      for (int step = 0; step < 32; ++step) {
        int byte = step * 32;
        mma<Heads>(tm + Score, desc(q) + (byte / 64 * (Heads / 8) * 512 + (byte % 64)) / 16,
            desc(k) + (byte / 64 * 8 * 512 + (byte % 64)) / 16, idesc<Heads>(64, false), step != 0);
      }
      commit(sa(bar));
    }
    // Raw global reads for the next tile overlap current QK/softmax/PV.
    RawTile next_tile{};
    if (ci + splits < last)
      next_tile = load_tile(c, bs, ci + splits, tid);
    wait(sa(bar), phase);
    phase ^= 1;
    after();
    unsigned regs[16];
    tmem_ld16(tm + ((data_row & ~31) << 16) + Score + score_col, regs);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    float v[16], mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < Values; ++j) {
      v[j] = valid[group * Values + j] ? __uint_as_float(regs[j]) * c.sm_scale * 1.4426950408889634f
                                   : -INFINITY;
      mx = fmaxf(mx, v[j]);
    }
    exchange[group * Heads + row] = mx;
    __syncthreads();
    for (int g = 0; g < Groups; ++g)
      mx = fmaxf(mx, exchange[g * Heads + row]);
    float next = m;
    if (mx != -INFINITY && (m == -INFINITY || mx > m + 2.f))
      next = mx;
    float alpha = m == -INFINITY ? 0.f : ex2(m - next);
    m = next;
    if (have && __any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll 1
      for (int col = output_col; col < OutputCols; col += 64) {
        tmem_ld16(tm + ((data_row & ~31) << 16) + col, regs);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int j = 0; j < 16; ++j)
          regs[j] = __float_as_uint(__uint_as_float(regs[j]) * alpha);
        tmem_st16(tm + ((data_row & ~31) << 16) + col, regs);
      }
      asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    l *= alpha;
#pragma unroll
    for (int j = 0; j < Values; j += 8) {
      uint4 packed;
      auto *hp = (__nv_bfloat162 *)&packed;
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        int u = j + 2 * z;
        if constexpr (Heads == 128) {
          tail[u] *= alpha;
          tail[u + 1] *= alpha;
        }
        float p0 = (v[u] == -INFINITY) ? 0.f : ex2(v[u] - m);
        float p1 = (v[u + 1] == -INFINITY) ? 0.f : ex2(v[u + 1] - m);
        l += p0;
        l += p1;
        hp[z] = __floats2bfloat162_rn(p0, p1);
      }
      *(uint4 *)(prob + off(row, (group * Values + j) * 2, Heads)) = packed;
    }
    publish();
    before();
    __syncthreads();
    after();
    if (tid == 0) {
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        int byte = step * 32;
        uint64_t pd = desc(prob) + (byte / 64 * (Heads / 8) * 512 + byte % 64) / 16;
        int vo = (step * 16 / 8) * 512;
        mma<Heads>(tm, pd, desc(k, 512, 8 * 512) + vo / 16, idesc<Heads>(256, true), have || step != 0);
        if constexpr (Heads == 64) {
          mma<Heads>(tm + 128, pd, desc(k + 8 * 8 * 512, 512, 8 * 512) + vo / 16,
                     idesc<Heads>(256, true), have || step != 0);
        } else {
        mma<Heads>(tm + 256, pd, desc(k + 8 * 8 * 512, 512, 8 * 512) + vo / 16, idesc<Heads>(192, true),
            have || step != 0);
        mma<Heads>(tm + 448, pd, desc(k + 14 * 8 * 512, 512, 8 * 512) + vo / 16, idesc<Heads>(64, true),
            step != 0);
        }
      }
      commit(sa(bar));
    }
    wait(sa(bar), phase);
    phase ^= 1;
    after();
    if constexpr (Heads == 128) {
    tmem_ld16(tm + ((row & ~31) << 16) + 448 + group * 16, regs);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
    for (int j = 0; j < 16; ++j)
      tail[j] += __uint_as_float(regs[j]);
    }
    current = next_tile;
    have = true; // Next iteration's post-decode barrier protects TMEM reuse.
  }
  if (!have) {
    if (use_tma)
      wait(sa(qbar), 0);
    else
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }
  exchange[group * Heads + row] = l;
  __syncthreads();
  float total = 0.f;
  for (int g = 0; g < Groups; ++g)
    total += exchange[g * Heads + row];
  const long rb = ((long)bs * c.H + hbase + row) * splits + sp;
  if (group == 0 && hbase + row < c.H) {
    ml[rb * 2] = m;
    ml[rb * 2 + 1] = total;
  }
  unsigned regs[16];
#pragma unroll 1
  for (int col = output_col; hbase + row < c.H && col < (Heads == 64 ? 256 : 512); col += 64) {
    const int logical_col = Heads == 64 ? (col / 128) * 256 + (group / 4) * 128 + col % 128 : col;
    if (col < OutputCols && have) {
      tmem_ld16(tm + ((data_row & ~31) << 16) + col, regs);
      asm volatile("tcgen05.wait::ld.sync.aligned;");
    }
#pragma unroll
    for (int j = 0; j < 16; j += 8) {
      uint4 packed;
      auto *hp = (__half2 *)&packed;
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        int u = j + z * 2;
        float a = !have ? 0.f : col < OutputCols ? __uint_as_float(regs[u]) : tail[u];
        float b = !have ? 0.f : col < OutputCols ? __uint_as_float(regs[u + 1]) : tail[u + 1];
        hp[z] = __floats2half2_rn(a, b);
      }
      if (hbase + row < c.H)
        *(uint4 *)((__half *)q + row * 520 + logical_col + j) = packed;
    }
  }
  before();
  __syncthreads();
  // With the fused tail this CTA merges heads [h0,h1) itself and takes its own
  // contribution straight from the staging buffer, so those heads never have to go
  // to global. Every other head is some peer's to merge and is published here; the
  // ranges are disjoint, so a peer always finds the partials it needs.
  // With head blocks the row's peers are the `splits` CTAs of this same block, so
  // the merged range is a slice of this block's own Heads, and the staging row is
  // block-local.
  const int h0 = arrive ? hbase + (int)((long)sp * Heads / splits) : 0;
  const int h1 = arrive ? hbase + (int)((long)(sp + 1) * Heads / splits) : 0;
  for (int j = tid; j < Heads * 64; j += 512) {
    const int lr = j / 64, d = (j % 64) * 8, r = hbase + lr;
    if (r >= c.H || (r >= h0 && r < h1)) continue;
    *(uint4 *)(po + (((long)bs * c.H + r) * splits + sp) * 512 + d) =
        *(uint4 *)((__half *)q + lr * 520 + d);
  }
  if (tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,512;" ::"r"(tm));
  if (arrive == nullptr) return;   // host keeps the standalone combine kernel

  // ---- fused merge: this CTA reduces the heads it owns of its own request row ----
  // The host only takes this path when BS*splits <= nsm and the block's shared
  // memory is more than half an SM's, so exactly one CTA occupies each SM and
  // every split of a row is resident: the spin below can always be satisfied.
  // Splitting the heads (rather than the dims) keeps one warp per head, which is
  // the standalone combine's own mapping, and the cyclic chunk assignment gives
  // all splits of a row the same chunk count, so they arrive here together.
  __syncthreads();
  __threadfence();
  const int peer = bs * nhb + hb;
  if (tid == 0) count_release(arrive + peer);
  if (tid == 0) {
    while (count_acquire(arrive + peer) < (unsigned)splits) {
    }
  }
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31;
  for (int h = h0 + warp; h < h1; h += 16)
    merge_head(po, ml, c.sink, c.out, c.lse, (__half *)q + (h - hbase) * 520, sp, splits, c.H,
               bs * c.H + h, lane);
  __syncthreads();
  __threadfence();
  if (tid == 0 && count_release(done + peer) == (unsigned)(splits - 1)) {
    // Last CTA of the group: hand the counters back zeroed for the next launch.
    count_store(arrive + peer, 0);
    count_store(done + peer, 0);
  }
}
using EncodeQ = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                             const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
                             const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
template<int Heads>
bool encode_q(CUtensorMap &map, plmn::FwdCtx c) {
  static EncodeQ encode = []() {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult status;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12030, cudaEnableDefault,
                                         &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess)
      return (EncodeQ) nullptr;
    return reinterpret_cast<EncodeQ>(p);
  }();
  if (!encode)
    return false;
  cuuint64_t dims[3] = {512, (cuuint64_t)c.H, (cuuint64_t)c.BS};
  cuuint64_t strides[2] = {1024, (cuuint64_t)c.H * 1024};
  cuuint32_t box[3] = {32, Heads, 1}, steps[3] = {1, 1, 1};
  return encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16 *>(c.q), dims,
                strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
// True when one block of this geometry already takes more than half of this
// device's per-SM dynamic shared memory, i.e. exactly one CTA can be resident per
// SM. The fused merge spins on its row peers, so the host needs this guarantee
// from the real device attribute rather than from the compiled maximum.
template<int Heads>
bool one_cta_per_sm(int smem_per_sm) {
  return SMEM<Heads> * 2 > smem_per_sm;
}

// `counters` is 2*BS*nhb zeroed ints (arrive, then done) or null. Null keeps the
// standalone combine kernel; non-null lets the main kernel merge each row itself,
// which is the only launch and therefore has no second kernel's floor to pay.
// `nhb` is the number of 64-head blocks one request row is cut into (gridDim.z);
// it is 1 unless the host chose the paired 64-head geometry for H=128.
template<int Heads>
void launch_heads(plmn::FwdCtx c, int splits, __half *po, float *ml, int *counters, int nhb) {
  auto stream = at::cuda::getCurrentCUDAStream();
  static thread_local int configured_device = -1;
  if (configured_device != stream.device_index()) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(attention<Heads>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM<Heads>));
    configured_device = stream.device_index();
  }
  static thread_local CUtensorMap map{};
  static thread_local const void *last_q = nullptr;
  static thread_local int last_h = 0, last_bs = 0, last_dev = -1;
  static thread_local bool tma = false;
  if (last_q != c.q || last_h != c.H || last_bs != c.BS || last_dev != stream.device_index()) {
    tma = encode_q<Heads>(map, c);
    last_q = c.q;
    last_h = c.H;
    last_bs = c.BS;
    last_dev = stream.device_index();
  }
  int *arrive = counters;
  int *done = counters ? counters + c.BS * nhb : nullptr;
  if (counters) {
    // Fused: one kernel writes the partials, meets its row peers and merges the
    // heads it owns straight into out/lse. There is no second node to bake into a
    // graph and the caller-visible LSE address is just an argument.
    attention<Heads><<<dim3(c.BS, splits, nhb), 512, SMEM<Heads>, stream>>>(
        c, splits, po, ml, map, tma, arrive, done);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Reuse only launch structure. Input data are always read by the kernels,
  // and the fresh caller-visible LSE address is updated on every allocation.
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  if (capture == cudaStreamCaptureStatusNone) {
    auto key = [](const plmn::FwdCtx &v) {
      return std::make_tuple(v.q, v.k1, v.idx1, v.k2, v.idx2, v.sink, v.sm_scale, v.out, v.H, v.BS,
                             v.budget1, v.budget2, v.page1, v.rec1, v.page2, v.rec2, v.listn);
    };
    struct GraphState {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      cudaGraphNode_t main_node = nullptr, merge_node = nullptr;
      decltype(key(c)) args{};
      int device = -1, splits = 0, nhb = 0;
      cudaStream_t stream = nullptr;
      __half *po = nullptr;
      float *ml = nullptr;
      float *lse = nullptr;
      ~GraphState() {
        if (exec)
          cudaGraphExecDestroy(exec);
        if (graph)
          cudaGraphDestroy(graph);
      }
    };
    static thread_local GraphState state;
    int heads = c.H, rows = c.BS;
    const float *sink = c.sink;
    auto *output = c.out;
    auto *lse = c.lse;
    void *merge_args[] = {&po, &ml, &sink, &output, &lse, &splits, &heads, &rows};
    cudaKernelNodeParams merge{};
    merge.func = (void *)cfg32::plmn_combine_wide_hb32;
    merge.gridDim = dim3((c.BS * c.H + cfg32::CH - 1) / cfg32::CH);
    merge.blockDim = dim3(cfg32::CT);
    merge.kernelParams = merge_args;
    if (!state.exec || state.args != key(c) || state.device != stream.device_index() ||
        state.stream != (cudaStream_t)stream || state.splits != splits || state.nhb != nhb ||
        state.po != po || state.ml != ml) {
      if (state.exec) {
        C10_CUDA_CHECK(cudaGraphExecDestroy(state.exec));
        state.exec = nullptr;
      }
      if (state.graph) {
        C10_CUDA_CHECK(cudaGraphDestroy(state.graph));
        state.graph = nullptr;
      }
      C10_CUDA_CHECK(cudaGraphCreate(&state.graph, 0));
      plmn::FwdCtx main_ctx = c;
      main_ctx.lse = nullptr; // Main writes only private partials.
      void *main_args[] = {&main_ctx, &splits, &po, &ml, &map, &tma, &arrive, &done};
      cudaKernelNodeParams main{};
      main.func = (void *)attention<Heads>;
      main.gridDim = dim3(c.BS, splits, nhb);
      main.blockDim = dim3(512);
      main.sharedMemBytes = SMEM<Heads>;
      main.kernelParams = main_args;
      C10_CUDA_CHECK(cudaGraphAddKernelNode(&state.main_node, state.graph, nullptr, 0, &main));
      C10_CUDA_CHECK(
          cudaGraphAddKernelNode(&state.merge_node, state.graph, &state.main_node, 1, &merge));
      C10_CUDA_CHECK(cudaGraphInstantiate(&state.exec, state.graph, nullptr, nullptr, 0));
      state.args = key(c);
      state.device = stream.device_index();
      state.stream = stream;
      state.splits = splits;
      state.nhb = nhb;
      state.po = po;
      state.ml = ml;
      state.lse = lse;
    } else if (state.lse != lse) {
      C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(state.exec, state.merge_node, &merge));
      state.lse = lse;
    }
    C10_CUDA_CHECK(cudaGraphLaunch(state.exec, stream));
    return;
  }
  attention<Heads><<<dim3(c.BS, splits, nhb), 512, SMEM<Heads>, stream>>>(c, splits, po, ml, map,
                                                                         tma, arrive, done);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cfg32::plmn_combine_wide_hb32<<<(c.BS * c.H + cfg32::CH - 1) / cfg32::CH, cfg32::CT, 0, stream>>>(po, ml, c.sink, c.out,
                                                                          c.lse, splits, c.H, c.BS);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch(plmn::FwdCtx c, int splits, __half *po, float *ml, int *counters, int nhb) {
  if (nhb == 2)
    launch_heads<64>(c, splits, po, ml, counters, 2);
  else if (c.H == 64)
    launch_heads<64>(c, splits, po, ml, counters, 1);
  else
    launch_heads<128>(c, splits, po, ml, counters, 1);
}
} // namespace plmn_tc
