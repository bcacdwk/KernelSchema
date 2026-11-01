// Host TU: combine kernel, dispatch, pybind. No head-dim template kernels
// here; those are instantiated one TU per head_dim (fmha_d64/128/256.cu) so
// ninja builds them in parallel and no single ptxas run is oversized.
#include "fmha_tc.cuh"
#include <algorithm>

namespace fmha {

// ---- combine kernel: merge chunk partials -> bf16 out ----------------------
// grid = (batch*kvh, ceil(D/32)), 256 threads; 4 threads per row, 8 dims each.
// Chunk metadata loads are batched 8-deep so the serial chunk walk pipelines.
__global__ void __launch_bounds__(256) fa_combine(const __grid_constant__ Params P) {
    int bkvh = blockIdx.x;
    int kv_heads = P.heads / P.g;
    int b = bkvh / kv_heads;
    int kvh = bkvh % kv_heads;
    int q0 = __ldg(P.cu + b);
    int qlen = __ldg(P.cu + b + 1) - q0;
    int R = qlen * P.g;
    int D = P.D;
    int dtile = blockIdx.y * 32;
    int tid = threadIdx.x;
    int r = tid >> 2;   // 4 threads per row
    int qtr = tid & 3;
    __shared__ float M2s[BM];
    const float* base = P.scratch + (long long)bkvh * P.nchunks * P.chunk_stride;
    int rr = min(r, max(R - 1, 0));

    // pass 1: per-row max over chunks (batched by 8)
    float mrow = NEG;
    for (int c0 = 0; c0 < P.nchunks; c0 += 8) {
        float mb[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int c = c0 + i;
            mb[i] = (c < P.nchunks) ? base[(long long)c * P.chunk_stride + rr] : NEG;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) mrow = fmaxf(mrow, mb[i]);
    }
    #pragma unroll
    for (int off = 1; off < 4; ++off)
        mrow = fmaxf(mrow, __shfl_xor_sync(0xffffffffu, mrow, off, 4));
    if (qtr == 0) M2s[r] = (r < R) ? mrow : NEG;
    __syncthreads();
    if (r >= R) return;
    float M2 = M2s[r];

    // pass 2: accumulate rescaled partials for this thread's 8 dims
    float acc[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    float lacc = 0.f;
    long long ooff = (2LL * P.Rmax + 3) & ~3LL;
    int dloc = qtr * 8;
    for (int c0 = 0; c0 < P.nchunks; c0 += 8) {
        float mb[8], lb[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int c = c0 + i;
            const float* pc = base + (long long)c * P.chunk_stride;
            mb[i] = (c < P.nchunks) ? pc[r] : NEG;
            lb[i] = (c < P.nchunks) ? pc[P.Rmax + r] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (mb[i] == NEG) continue;
            float scale = exp2f(mb[i] - M2);
            lacc += lb[i] * scale;
            const float* orow = base + (long long)(c0 + i) * P.chunk_stride
                              + ooff + (long long)r * D + dtile + dloc;
            float4 o0 = *reinterpret_cast<const float4*>(orow);
            float4 o1 = *reinterpret_cast<const float4*>(orow + 4);
            acc[0] += o0.x * scale; acc[1] += o0.y * scale;
            acc[2] += o0.z * scale; acc[3] += o0.w * scale;
            acc[4] += o1.x * scale; acc[5] += o1.y * scale;
            acc[6] += o1.z * scale; acc[7] += o1.w * scale;
        }
    }
    float inv = 1.f / lacc;
    uint16_t* orow = P.out + (long long)(q0 + r / P.g) * P.q_out_stride
                   + (kvh * P.g + r % P.g) * D + dtile + dloc;
    #pragma unroll
    for (int e2 = 0; e2 < 4; ++e2) {
        unsigned b0 = f32_to_bf16_bits(acc[e2 * 2] * inv);
        unsigned b1 = f32_to_bf16_bits(acc[e2 * 2 + 1] * inv);
        *reinterpret_cast<unsigned*>(orow + e2 * 2) = b0 | (b1 << 16);
    }
}

} // namespace fmha

namespace fmha {

static void fmha_run_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
              torch::Tensor cu, torch::Tensor seqk, torch::Tensor bt,
              torch::Tensor ks, torch::Tensor vs,
              double softmax_scale, long long max_seqlen_q, long long max_seqlen_k,
              float* dbgptr) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16, "q must be CUDA bf16");
    TORCH_CHECK(k.scalar_type() == torch::kFloat8_e4m3fn, "k must be fp8_e4m3");
    int D = q.size(2);
    int heads = q.size(1);
    int Tq = q.size(0);
    int batch = cu.size(0) - 1;
    int kv_heads = ks.size(2);
    int page = k.size(1);
    TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == D, "q rows contiguous");
    TORCH_CHECK(out.stride(2) == 1 && out.stride(1) == D && out.stride(0) == q.stride(0), "out layout");
    TORCH_CHECK(k.stride(3) == 1 && v.stride(3) == 1, "kv rows contiguous in d");
    TORCH_CHECK((page & (page - 1)) == 0, "page_size power of two");
    TORCH_CHECK(heads % kv_heads == 0, "GQA divisibility");

    Params P = {};  // mutated below for the decode path
    P.q = reinterpret_cast<const uint16_t*>(q.data_ptr());
    P.out = reinterpret_cast<uint16_t*>(out.data_ptr());
    P.k = reinterpret_cast<const uint8_t*>(k.data_ptr());
    P.v = reinterpret_cast<const uint8_t*>(v.data_ptr());
    P.ks = ks.data_ptr<float>();
    P.vs = vs.data_ptr<float>();
    P.block_table = bt.data_ptr<int>();
    P.cu = cu.data_ptr<int>();
    P.seqk = seqk.data_ptr<int>();
    P.q_out_stride = q.stride(0);
    P.k_sb = k.stride(0); P.k_sp = k.stride(1); P.k_sh = k.stride(2);
    P.v_sb = v.stride(0); P.v_sp = v.stride(1); P.v_sh = v.stride(2);
    P.ks_sb = ks.stride(0); P.ks_sp = ks.stride(1); P.ks_sh = ks.stride(2);
    P.vs_sb = vs.stride(0); P.vs_sp = vs.stride(1); P.vs_sh = vs.stride(2);
    P.bt_stride = bt.stride(0);
    P.batch = batch; P.heads = heads; P.g = heads / kv_heads;
    P.D = D;
    P.dbg = dbgptr;
    P.page_shift = __builtin_ctz(page);
    P.sm_scale_log2e = (float)(softmax_scale * 1.4426950408889634);
    P.raw16 = ((((uintptr_t)P.k | (uintptr_t)P.v) & 15) == 0
               && (P.k_sb % 16) == 0 && (P.k_sp % 16) == 0 && (P.k_sh % 16) == 0
               && (P.v_sb % 16) == 0 && (P.v_sp % 16) == 0 && (P.v_sh % 16) == 0) ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int SS = D + 8;
    bool decode = (long long)max_seqlen_q * P.g <= BM;
    if (!decode) {
        // v4.5 mirror pairing (diagonal band) / v4.8 persistent pull (>=band).
        int sms0 = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        long long grid_orig = ((long long)(Tq + fmha::TCBM - 1) / fmha::TCBM + batch) * heads;
        const long long slots = (D == 256 ? 1 : 2) * (long long)sms0;
        const bool diag = 2LL * (long long)max_seqlen_q >= (long long)max_seqlen_k;
        P.fold = (D != 64 && diag && grid_orig >= slots && grid_orig <= 2 * slots) ? 1 : 0;
        // v4.8.4: persist only when clearly multi-wave (c9 at 1.08 waves lost
        // 4% to pull overhead; c0 2.8x / c7 2.6x waves won 15% / 35%).
        P.persist = (!P.fold && 4 * grid_orig >= 5 * slots) ? 1 : 0;
        torch::Tensor ctr;
        if (P.persist) {
            ctr = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            P.counter = ctr.data_ptr<int>();
            P.nctas = (int)std::min(grid_orig, slots);
        } else {
            P.counter = nullptr;
            P.nctas = 0;
        }
        // tcgen05 prefill for all supported head dims (sm_103a build)
        // v2.9: BNT=32 measured 0.911x on c7 (dv_b67f87b6fe7f) — doubled
        // per-tile overhead beat the 4-CTA/16-warp pool. BNT=64 restored.
        if (D == 64 && TcSmem<64, 64>::bytes() <= 232448) {
            launch_prefill_tc_64(P, Tq, TcSmem<64, 64>::bytes(), stream);
            return;
        }
        // v4.0-rt2 REJECTED (dv_5e80dd000577): 2 corr FAILs (partial group B)
        // and c9 0.504x (halved grid at 1 CTA/SM -> wave quantization),
        // c0 0.926x. Kernel kept dormant in fmha_tc.cuh; not built.
        // v4.2-v4.4 raw-staged prefill REJECTED (dv_98d8699089e2: c0 0.729x
        // even after the bank-conflict fix): L2 absorbs scattered sectors
        // cheaply (92% hit, 7% util) while the smem roundtrip adds binding
        // L1/LSU instruction work. Kernel dormant (fmha_tcr128.cu built but
        // not dispatched); v3.8 direct-register path is measured-best.
        // PF_BNT128 must match fmha_tc128.cu's instantiation: the launcher and the
        // dynamic smem size are both derived from the KV tile, and a mismatch would
        // launch with too little smem.
#ifdef PF_BNT128
        constexpr int BNT128 = 128;
#else
        constexpr int BNT128 = 64;
#endif
        if (D == 128 && TcSmem<128, BNT128>::bytes() <= 232448) {
            launch_prefill_tc_128(P, Tq, TcSmem<128, BNT128>::bytes(), stream);
            return;
        }
        if (D == 256 && TcSmem<256, 64>::bytes() <= 232448) {
            launch_prefill_tc_256(P, Tq, TcSmem<256, 64>::bytes(), stream);
            return;
        }
        const int bnp = (D == 256) ? 32 : 64;  // prefill KV tile (SmemT path)
        int smem = (BM * SS + 4 * bnp * SS) * 2 + 4 * bnp * 4 + BM * 8 + 16;
        switch (D) {
            case 64: launch_prefill_64(P, Tq, smem, stream); break;
            case 128: launch_prefill_128(P, Tq, smem, stream); break;
            case 256: launch_prefill_256(P, Tq, smem, stream); break;
            default: TORCH_CHECK(false, "unsupported head_dim ", D);
        }
        return;
    }
    // decode: cp.async ring path (BN=32, RING=3); occupancy-targeted chunks
    const int bnv = (D == 256) ? 32 : 64;   // v2 KV tile (register prefetch)
    P.Rmax = (int)max_seqlen_q * P.g;
    // v3.3: REVERT v3.2 decode experiment (dv_f3eca39546cd, geomean 0.975x):
    // occupancy-API target doubled nchunks (c3 148->296 target: 0.894x,
    // re-confirming v2.4's chunk-length preference; c2/c4 296->444: 0.93x,
    // echoing v2.1). QR16 smem shrink only pays via the larger target, so
    // qrows reverts to BM (QR mechanism kept, dispatched at BM). Target
    // back to the measured-best (D==256 ? 1 : 2) * sms.
    const int qrows = BM;
    int smem = (qrows * SS + 4 * bnv * SS) * 2 + 4 * bnv * 4 + qrows * 8 + 16;
    int base_ctas = batch * kv_heads;
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int target = (D == 256 ? 1 : 2) * sms;
    int nchunks = std::max(1, (target + base_ctas - 1) / base_ctas);
    int max_chunks = std::max(1, ((int)max_seqlen_k + bnv - 1) / bnv);
    nchunks = std::min(nchunks, max_chunks);
    int cs = std::max(bnv, (((int)max_seqlen_k + nchunks - 1) / nchunks + bnv - 1) / bnv * bnv);
    nchunks = ((int)max_seqlen_k + cs - 1) / cs;
    P.nchunks = nchunks;
    P.chunk_size = cs;
    P.chunk_stride = (((2LL * P.Rmax + 3) & ~3LL) + (long long)P.Rmax * D);
    if (nchunks > 1) {
        // v3.7: cache the scratch allocation across calls (caching-allocator
        // round trip + refcounts ~2-4us on 35-50us small decode cases).
        // Fully overwritten each call (partials) before combine reads.
        static torch::Tensor scratch_cache;
        const long long need = (long long)base_ctas * nchunks * P.chunk_stride;
        if (!scratch_cache.defined() || scratch_cache.numel() < need) {
            scratch_cache = torch::empty({need},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }
        P.scratch = scratch_cache.data_ptr<float>();
    } else {
        P.scratch = nullptr;
    }
    switch (D) {
        case 64: launch_decode_64(P, nchunks, base_ctas, smem, stream, qrows); break;
        case 128: launch_decode_128(P, nchunks, base_ctas, smem, stream, qrows); break;
        case 256: launch_decode_256(P, nchunks, base_ctas, smem, stream, qrows); break;
        default: TORCH_CHECK(false, "unsupported head_dim ", D);
    }
    if (nchunks > 1) {
        fmha::fa_combine<<<dim3(base_ctas, (D + 31) / 32), 256, 0, stream>>>(P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void fmha_run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
              torch::Tensor cu, torch::Tensor seqk, torch::Tensor bt,
              torch::Tensor ks, torch::Tensor vs,
              double softmax_scale, long long max_seqlen_q, long long max_seqlen_k) {
    fmha_run_impl(q, k, v, out, cu, seqk, bt, ks, vs, softmax_scale,
                  max_seqlen_q, max_seqlen_k, nullptr);
}

#ifdef FMHA_DBG
void fmha_run_dbg(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
                  torch::Tensor cu, torch::Tensor seqk, torch::Tensor bt,
                  torch::Tensor ks, torch::Tensor vs,
                  double softmax_scale, long long max_seqlen_q, long long max_seqlen_k,
                  torch::Tensor dbg) {
    fmha_run_impl(q, k, v, out, cu, seqk, bt, ks, vs, softmax_scale,
                  max_seqlen_q, max_seqlen_k, dbg.data_ptr<float>());
}
#endif

} // namespace fmha

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmha_run", &fmha::fmha_run, "fmha_fp8_per_token_head fused attention");
#ifdef FMHA_DBG
    m.def("fmha_run_dbg", &fmha::fmha_run_dbg, "debug variant with row-0 dump");
#endif
}
