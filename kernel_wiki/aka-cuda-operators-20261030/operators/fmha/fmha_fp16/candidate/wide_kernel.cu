#include "wide_kernel_primitives.cuh"
extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_aka_context_bf16_hd256(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __half* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, int total_bh, int max_pages_per_seq, unsigned int* __restrict__ dynamic_counter, float* __restrict__ LSE_ptr, const int* __restrict__ original_q_indptr)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 240)
    #define p_full_addr (mbar_base + 256)
    #define p_full_2_addr (mbar_base + 272)
    #define pv_done_addr (mbar_base + 288)
    #define corr_sig_addr (mbar_base + 64)
    #define corr_done_addr (mbar_base + 72)
    #define o_full_addr (mbar_base + 80)
    #define q_empty_addr (mbar_base + 88)
    #define page_full_addr (mbar_base + 96)
    #define page_empty_addr (mbar_base + 144)
    #define work_id_full_addr (mbar_base + 192)
    #define work_id_empty_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* sScale = reinterpret_cast<float*>(smem_raw + 1024);
    const int sScale_addr = smem + 1024;
    __half* smem_q_hi = reinterpret_cast<__half*>(smem_raw + 4096);
    const int smem_q_hi_addr = smem + 4096;
    __half* smem_q_lo = reinterpret_cast<__half*>(smem_raw + 36864);
    const int smem_q_lo_addr = smem + 36864;
    __half* smem_kv_hi = reinterpret_cast<__half*>(smem_raw + 69632);
    const int smem_kv_hi_addr = smem + 69632;
    __half* smem_kv_lo = reinterpret_cast<__half*>(smem_raw + 135168);
    const int smem_kv_lo_addr = smem + 135168;
    __half* smem_v_hi = reinterpret_cast<__half*>(smem_raw + 69632);
    const int smem_v_hi_addr = smem + 69632;
    __half* smem_v_lo = reinterpret_cast<__half*>(smem_raw + 135168);
    const int smem_v_lo_addr = smem + 135168;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 200704);
    const int smem_pages_addr = smem + 200704;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 200896);
    const int work_id_slot_addr = smem + 200896;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 240, 1);mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 256);mbarrier_init(smem + 264, 256);
            mbarrier_init(smem + 272, 128);mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 1);
            // --- pipeline 'kv' ---
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            // p_full_2: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'po' ---
            // page_full: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // page_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // work_id_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // work_id_empty: 1 barriers, init_count=352
            mbarrier_init(smem + 200, 352);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_softmax_0 = taddr + 64;
    const int tmem_output_hi = taddr + 128;
    const int tmem_output_lo = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_main
            unsigned int total_tiles = NUM_M_BLOCKS * total_bh;
            unsigned int sv_f16[16];
            unsigned int _phase_work_id_full_0 = 0;
            unsigned int _phase_s_full_0[2] = {0,0};
            unsigned int _phase_corr_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter = 0; tile_iter < total_tiles; tile_iter++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0);
                _phase_work_id_full_0 ^= 1;
                unsigned int tile_idx = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (tile_idx >= total_tiles) {
                    break;
                }
                unsigned int m_block;
                unsigned int bh;
                {
                    m_block = tile_idx % (unsigned int)NUM_M_BLOCKS;
                    bh = tile_idx / (unsigned int)NUM_M_BLOCKS;
                }
                if (m_block * BLOCK_M >= (unsigned int)seq_lens_q[bh]) continue;
                int seqlen_kv_bh = seq_lens_kv[bh];
                unsigned int num_n_blocks = (seqlen_kv_bh + BLOCK_N - 1) / BLOCK_N;
                int visible_end = int((m_block + 1) * BLOCK_M) + seqlen_kv_bh - seq_lens_q[bh];
                visible_end = max(1, min(seqlen_kv_bh, visible_end));
                num_n_blocks = (visible_end + BLOCK_N - 1) / BLOCK_N;
                int causal_row;
                unsigned int num_masked_iters;
                {
                    int shift = seqlen_kv_bh - seq_lens_q[bh];
                    causal_row = int(m_block * BLOCK_M) + int(warp % 4 * 32 + lane) + shift;
                    int first_row = int(m_block * BLOCK_M) + shift;
                    unsigned int full_blocks = max(0, first_row + 1) / BLOCK_N;
                    num_masked_iters = num_n_blocks > full_blocks ? num_n_blocks-full_blocks : 0;
                }
                float row_max_val = -CAKE_FMHA_INF;
                float row_sum_val = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < num_masked_iters; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (n_iter&1)*8, _phase_s_full_0[n_iter&1]);
                    _phase_s_full_0[n_iter&1] ^= 1;
                    int s_base = taddr + (n_iter&1)*384 + (warp % 4 * 32 << 16);
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(s_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(s_base + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                        : "r"(s_base + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(s_base + 96));
                    int tail_valid = min(seqlen_kv_bh - n_block * BLOCK_N, causal_row - n_block * BLOCK_N + 1);
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = tail_valid;
                        if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                        else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                        }
                    }
                    if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_1 = tail_valid - 32;
                        if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                        else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                        }
                    }
                    if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_2;
                    {
                        int _lim_2 = tail_valid - 64;
                        if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                        else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
                        }
                    }
                    if (!(_slice_lo_mask_2 & (1u << 0))) _tmem_load_0[64] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 1))) _tmem_load_0[65] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 2))) _tmem_load_0[66] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 3))) _tmem_load_0[67] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 4))) _tmem_load_0[68] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 5))) _tmem_load_0[69] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 6))) _tmem_load_0[70] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 7))) _tmem_load_0[71] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 8))) _tmem_load_0[72] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 9))) _tmem_load_0[73] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 10))) _tmem_load_0[74] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 11))) _tmem_load_0[75] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 12))) _tmem_load_0[76] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 13))) _tmem_load_0[77] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 14))) _tmem_load_0[78] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 15))) _tmem_load_0[79] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 16))) _tmem_load_0[80] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 17))) _tmem_load_0[81] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 18))) _tmem_load_0[82] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 19))) _tmem_load_0[83] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 20))) _tmem_load_0[84] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 21))) _tmem_load_0[85] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 22))) _tmem_load_0[86] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 23))) _tmem_load_0[87] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 24))) _tmem_load_0[88] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 25))) _tmem_load_0[89] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 26))) _tmem_load_0[90] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 27))) _tmem_load_0[91] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 28))) _tmem_load_0[92] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 29))) _tmem_load_0[93] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 30))) _tmem_load_0[94] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_2 & (1u << 31))) _tmem_load_0[95] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_3;
                    {
                        int _lim_3 = tail_valid - 96;
                        if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                        else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
                        }
                    }
                    if (!(_slice_lo_mask_3 & (1u << 0))) _tmem_load_0[96] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 1))) _tmem_load_0[97] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 2))) _tmem_load_0[98] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 3))) _tmem_load_0[99] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 4))) _tmem_load_0[100] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 5))) _tmem_load_0[101] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 6))) _tmem_load_0[102] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 7))) _tmem_load_0[103] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 8))) _tmem_load_0[104] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 9))) _tmem_load_0[105] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 10))) _tmem_load_0[106] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 11))) _tmem_load_0[107] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 12))) _tmem_load_0[108] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 13))) _tmem_load_0[109] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 14))) _tmem_load_0[110] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 15))) _tmem_load_0[111] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 16))) _tmem_load_0[112] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 17))) _tmem_load_0[113] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 18))) _tmem_load_0[114] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 19))) _tmem_load_0[115] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 20))) _tmem_load_0[116] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 21))) _tmem_load_0[117] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 22))) _tmem_load_0[118] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 23))) _tmem_load_0[119] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 24))) _tmem_load_0[120] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 25))) _tmem_load_0[121] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 26))) _tmem_load_0[122] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 27))) _tmem_load_0[123] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 28))) _tmem_load_0[124] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 29))) _tmem_load_0[125] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 30))) _tmem_load_0[126] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_3 & (1u << 31))) _tmem_load_0[127] = -CAKE_FMHA_INF;
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
                    uint32_t _slice_lo_mask_4;
                    {
                        int _lim_4 = valid_count;
                        if (_lim_4 <= 0) { _slice_lo_mask_4 = 0u; }
                        else if (_lim_4 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_4));
                        }
                    }
                    if (!(_slice_lo_mask_4 & (1u << 0))) _tmem_load_0[0] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 1))) _tmem_load_0[1] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 2))) _tmem_load_0[2] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 3))) _tmem_load_0[3] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 4))) _tmem_load_0[4] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 5))) _tmem_load_0[5] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 6))) _tmem_load_0[6] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 7))) _tmem_load_0[7] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 8))) _tmem_load_0[8] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 9))) _tmem_load_0[9] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 10))) _tmem_load_0[10] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 11))) _tmem_load_0[11] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 12))) _tmem_load_0[12] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 13))) _tmem_load_0[13] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 14))) _tmem_load_0[14] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 15))) _tmem_load_0[15] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 16))) _tmem_load_0[16] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 17))) _tmem_load_0[17] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 18))) _tmem_load_0[18] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 19))) _tmem_load_0[19] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 20))) _tmem_load_0[20] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 21))) _tmem_load_0[21] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 22))) _tmem_load_0[22] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 23))) _tmem_load_0[23] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 24))) _tmem_load_0[24] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 25))) _tmem_load_0[25] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 26))) _tmem_load_0[26] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 27))) _tmem_load_0[27] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 28))) _tmem_load_0[28] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 29))) _tmem_load_0[29] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 30))) _tmem_load_0[30] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_4 & (1u << 31))) _tmem_load_0[31] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_5;
                    {
                        int _lim_5 = valid_count - 32;
                        if (_lim_5 <= 0) { _slice_lo_mask_5 = 0u; }
                        else if (_lim_5 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_5));
                        }
                    }
                    if (!(_slice_lo_mask_5 & (1u << 0))) _tmem_load_0[32] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 1))) _tmem_load_0[33] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 2))) _tmem_load_0[34] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 3))) _tmem_load_0[35] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 4))) _tmem_load_0[36] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 5))) _tmem_load_0[37] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 6))) _tmem_load_0[38] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 7))) _tmem_load_0[39] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 8))) _tmem_load_0[40] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 9))) _tmem_load_0[41] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 10))) _tmem_load_0[42] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 11))) _tmem_load_0[43] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 12))) _tmem_load_0[44] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 13))) _tmem_load_0[45] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 14))) _tmem_load_0[46] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 15))) _tmem_load_0[47] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 16))) _tmem_load_0[48] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 17))) _tmem_load_0[49] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 18))) _tmem_load_0[50] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 19))) _tmem_load_0[51] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 20))) _tmem_load_0[52] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 21))) _tmem_load_0[53] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 22))) _tmem_load_0[54] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 23))) _tmem_load_0[55] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 24))) _tmem_load_0[56] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 25))) _tmem_load_0[57] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 26))) _tmem_load_0[58] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 27))) _tmem_load_0[59] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 28))) _tmem_load_0[60] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 29))) _tmem_load_0[61] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 30))) _tmem_load_0[62] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_5 & (1u << 31))) _tmem_load_0[63] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_6;
                    {
                        int _lim_6 = valid_count - 64;
                        if (_lim_6 <= 0) { _slice_lo_mask_6 = 0u; }
                        else if (_lim_6 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_6));
                        }
                    }
                    if (!(_slice_lo_mask_6 & (1u << 0))) _tmem_load_0[64] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 1))) _tmem_load_0[65] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 2))) _tmem_load_0[66] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 3))) _tmem_load_0[67] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 4))) _tmem_load_0[68] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 5))) _tmem_load_0[69] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 6))) _tmem_load_0[70] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 7))) _tmem_load_0[71] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 8))) _tmem_load_0[72] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 9))) _tmem_load_0[73] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 10))) _tmem_load_0[74] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 11))) _tmem_load_0[75] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 12))) _tmem_load_0[76] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 13))) _tmem_load_0[77] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 14))) _tmem_load_0[78] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 15))) _tmem_load_0[79] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 16))) _tmem_load_0[80] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 17))) _tmem_load_0[81] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 18))) _tmem_load_0[82] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 19))) _tmem_load_0[83] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 20))) _tmem_load_0[84] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 21))) _tmem_load_0[85] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 22))) _tmem_load_0[86] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 23))) _tmem_load_0[87] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 24))) _tmem_load_0[88] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 25))) _tmem_load_0[89] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 26))) _tmem_load_0[90] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 27))) _tmem_load_0[91] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 28))) _tmem_load_0[92] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 29))) _tmem_load_0[93] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 30))) _tmem_load_0[94] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_6 & (1u << 31))) _tmem_load_0[95] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_7;
                    {
                        int _lim_7 = valid_count - 96;
                        if (_lim_7 <= 0) { _slice_lo_mask_7 = 0u; }
                        else if (_lim_7 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_7));
                        }
                    }
                    if (!(_slice_lo_mask_7 & (1u << 0))) _tmem_load_0[96] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 1))) _tmem_load_0[97] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 2))) _tmem_load_0[98] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 3))) _tmem_load_0[99] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 4))) _tmem_load_0[100] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 5))) _tmem_load_0[101] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 6))) _tmem_load_0[102] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 7))) _tmem_load_0[103] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 8))) _tmem_load_0[104] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 9))) _tmem_load_0[105] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 10))) _tmem_load_0[106] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 11))) _tmem_load_0[107] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 12))) _tmem_load_0[108] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 13))) _tmem_load_0[109] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 14))) _tmem_load_0[110] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 15))) _tmem_load_0[111] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 16))) _tmem_load_0[112] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 17))) _tmem_load_0[113] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 18))) _tmem_load_0[114] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 19))) _tmem_load_0[115] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 20))) _tmem_load_0[116] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 21))) _tmem_load_0[117] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 22))) _tmem_load_0[118] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 23))) _tmem_load_0[119] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 24))) _tmem_load_0[120] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 25))) _tmem_load_0[121] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 26))) _tmem_load_0[122] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 27))) _tmem_load_0[123] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 28))) _tmem_load_0[124] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 29))) _tmem_load_0[125] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 30))) _tmem_load_0[126] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_7 & (1u << 31))) _tmem_load_0[127] = -CAKE_FMHA_INF;
                    float2 _reg_reduce_max2_8 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_8);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_8);
                    float new_max = _tmem_load_0_max;
                    float _max_0 = max_noftz(new_max, row_max_val);
                    new_max = _max_0;
                    float new_max_scaled = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale;
                    float selected_max;
                    float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                    if (_fma_0 >= -8.0f) {
                        selected_max = row_max_val;
                        acc_scale = 1.0f;
                        new_max_scaled = ((row_max_val == -CAKE_FMHA_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        if (row_max_val > -CAKE_FMHA_INF) {
                            float _exp2_0 = approx_exp2(_fma_0);
                            acc_scale = _exp2_0;
                        } else {
                            acc_scale = 1.0f;
                        }
                    }
                    row_max_val = selected_max;
                    sScale[warp % 4 * 32 + lane] = acc_scale;
                    mbarrier_arrive(corr_sig_addr);
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_9, _fma_c2_10);
                    int p_base = taddr + 64 + (n_iter&1)*384 + (warp % 4 * 32 << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    uint32_t _tmem_load_0_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (n_iter&1)*8);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 32] = approx_exp2(_tmem_load_0[_le + 32]);
                    }
                    uint32_t _tmem_load_0_f16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 32], _tmem_load_0[_lp*2+1 + 32]));
                        _tmem_load_0_f16_0[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_0[15])));
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 64] = approx_exp2(_tmem_load_0[_le + 64]);
                    }
                    uint32_t _tmem_load_0_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                        _tmem_load_0_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_1[15])));
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 96] = approx_exp2(_tmem_load_0[_le + 96]);
                    }
                    uint32_t _tmem_load_0_f16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 96], _tmem_load_0[_lp*2+1 + 96]));
                        _tmem_load_0_f16_2[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16_2[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (n_iter&1)*8);
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_11);
                    float _tmem_load_0_sum = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
                    row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = num_masked_iters; n_iter_1 < num_n_blocks; n_iter_1++) {
                    int n_block_1 = num_n_blocks - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr + (n_iter_1&1)*8, _phase_s_full_0[n_iter_1&1]);
                    _phase_s_full_0[n_iter_1&1] ^= 1;
                    int s_base_1 = taddr + (n_iter_1&1)*384 + (warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    float tile_max = -CAKE_FMHA_INF;
                    float tile_max_lo = -CAKE_FMHA_INF;
                    float tile_max_hi = -CAKE_FMHA_INF;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63]), "=f"(tile_max_lo)
                        : "r"(s_base_1));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95]), "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127]), "=f"(tile_max_hi)
                        : "r"(s_base_1 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float aka_tile_max = max_noftz(tile_max_lo, tile_max_hi);
                    tile_max = aka_tile_max;
                    int tail_valid_1 = seqlen_kv_bh - n_block_1 * BLOCK_N;
                    float new_max_1 = tile_max;
                    // The hardware reduction is exact for a full unmasked
                    // score tile. Tail columns retain the original masked
                    // software reduction, including all boundary predicates.
                    if (tail_valid_1 < BLOCK_N) {
                    uint32_t _slice_lo_mask_8;
                    {
                        int _lim_12 = tail_valid_1;
                        if (_lim_12 <= 0) { _slice_lo_mask_8 = 0u; }
                        else if (_lim_12 >= 32) { _slice_lo_mask_8 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_12));
                        }
                    }
                    if (!(_slice_lo_mask_8 & (1u << 0))) _tmem_load_1[0] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 1))) _tmem_load_1[1] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 2))) _tmem_load_1[2] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 3))) _tmem_load_1[3] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 4))) _tmem_load_1[4] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 5))) _tmem_load_1[5] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 6))) _tmem_load_1[6] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 7))) _tmem_load_1[7] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 8))) _tmem_load_1[8] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 9))) _tmem_load_1[9] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 10))) _tmem_load_1[10] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 11))) _tmem_load_1[11] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 12))) _tmem_load_1[12] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 13))) _tmem_load_1[13] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 14))) _tmem_load_1[14] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 15))) _tmem_load_1[15] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 16))) _tmem_load_1[16] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 17))) _tmem_load_1[17] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 18))) _tmem_load_1[18] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 19))) _tmem_load_1[19] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 20))) _tmem_load_1[20] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 21))) _tmem_load_1[21] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 22))) _tmem_load_1[22] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 23))) _tmem_load_1[23] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 24))) _tmem_load_1[24] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 25))) _tmem_load_1[25] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 26))) _tmem_load_1[26] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 27))) _tmem_load_1[27] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 28))) _tmem_load_1[28] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 29))) _tmem_load_1[29] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 30))) _tmem_load_1[30] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_8 & (1u << 31))) _tmem_load_1[31] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_9;
                    {
                        int _lim_13 = tail_valid_1 - 32;
                        if (_lim_13 <= 0) { _slice_lo_mask_9 = 0u; }
                        else if (_lim_13 >= 32) { _slice_lo_mask_9 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_13));
                        }
                    }
                    if (!(_slice_lo_mask_9 & (1u << 0))) _tmem_load_1[32] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 1))) _tmem_load_1[33] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 2))) _tmem_load_1[34] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 3))) _tmem_load_1[35] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 4))) _tmem_load_1[36] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 5))) _tmem_load_1[37] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 6))) _tmem_load_1[38] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 7))) _tmem_load_1[39] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 8))) _tmem_load_1[40] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 9))) _tmem_load_1[41] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 10))) _tmem_load_1[42] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 11))) _tmem_load_1[43] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 12))) _tmem_load_1[44] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 13))) _tmem_load_1[45] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 14))) _tmem_load_1[46] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 15))) _tmem_load_1[47] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 16))) _tmem_load_1[48] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 17))) _tmem_load_1[49] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 18))) _tmem_load_1[50] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 19))) _tmem_load_1[51] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 20))) _tmem_load_1[52] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 21))) _tmem_load_1[53] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 22))) _tmem_load_1[54] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 23))) _tmem_load_1[55] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 24))) _tmem_load_1[56] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 25))) _tmem_load_1[57] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 26))) _tmem_load_1[58] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 27))) _tmem_load_1[59] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 28))) _tmem_load_1[60] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 29))) _tmem_load_1[61] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 30))) _tmem_load_1[62] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_9 & (1u << 31))) _tmem_load_1[63] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_10;
                    {
                        int _lim_14 = tail_valid_1 - 64;
                        if (_lim_14 <= 0) { _slice_lo_mask_10 = 0u; }
                        else if (_lim_14 >= 32) { _slice_lo_mask_10 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_14));
                        }
                    }
                    if (!(_slice_lo_mask_10 & (1u << 0))) _tmem_load_1[64] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 1))) _tmem_load_1[65] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 2))) _tmem_load_1[66] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 3))) _tmem_load_1[67] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 4))) _tmem_load_1[68] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 5))) _tmem_load_1[69] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 6))) _tmem_load_1[70] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 7))) _tmem_load_1[71] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 8))) _tmem_load_1[72] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 9))) _tmem_load_1[73] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 10))) _tmem_load_1[74] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 11))) _tmem_load_1[75] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 12))) _tmem_load_1[76] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 13))) _tmem_load_1[77] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 14))) _tmem_load_1[78] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 15))) _tmem_load_1[79] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 16))) _tmem_load_1[80] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 17))) _tmem_load_1[81] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 18))) _tmem_load_1[82] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 19))) _tmem_load_1[83] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 20))) _tmem_load_1[84] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 21))) _tmem_load_1[85] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 22))) _tmem_load_1[86] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 23))) _tmem_load_1[87] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 24))) _tmem_load_1[88] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 25))) _tmem_load_1[89] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 26))) _tmem_load_1[90] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 27))) _tmem_load_1[91] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 28))) _tmem_load_1[92] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 29))) _tmem_load_1[93] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 30))) _tmem_load_1[94] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_10 & (1u << 31))) _tmem_load_1[95] = -CAKE_FMHA_INF;
                    uint32_t _slice_lo_mask_11;
                    {
                        int _lim_15 = tail_valid_1 - 96;
                        if (_lim_15 <= 0) { _slice_lo_mask_11 = 0u; }
                        else if (_lim_15 >= 32) { _slice_lo_mask_11 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_15));
                        }
                    }
                    if (!(_slice_lo_mask_11 & (1u << 0))) _tmem_load_1[96] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 1))) _tmem_load_1[97] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 2))) _tmem_load_1[98] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 3))) _tmem_load_1[99] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 4))) _tmem_load_1[100] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 5))) _tmem_load_1[101] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 6))) _tmem_load_1[102] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 7))) _tmem_load_1[103] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 8))) _tmem_load_1[104] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 9))) _tmem_load_1[105] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 10))) _tmem_load_1[106] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 11))) _tmem_load_1[107] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 12))) _tmem_load_1[108] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 13))) _tmem_load_1[109] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 14))) _tmem_load_1[110] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 15))) _tmem_load_1[111] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 16))) _tmem_load_1[112] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 17))) _tmem_load_1[113] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 18))) _tmem_load_1[114] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 19))) _tmem_load_1[115] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 20))) _tmem_load_1[116] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 21))) _tmem_load_1[117] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 22))) _tmem_load_1[118] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 23))) _tmem_load_1[119] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 24))) _tmem_load_1[120] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 25))) _tmem_load_1[121] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 26))) _tmem_load_1[122] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 27))) _tmem_load_1[123] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 28))) _tmem_load_1[124] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 29))) _tmem_load_1[125] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 30))) _tmem_load_1[126] = -CAKE_FMHA_INF;
                    if (!(_slice_lo_mask_11 & (1u << 31))) _tmem_load_1[127] = -CAKE_FMHA_INF;
                    float2 _reg_reduce_max2_16 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_16);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_16);
                    new_max_1 = _tmem_load_1_max;
                    }
                    float _max_1 = max_noftz(new_max_1, row_max_val);
                    new_max_1 = _max_1;
                    float new_max_scaled_1 = ((new_max_1 == -CAKE_FMHA_INF) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float acc_scale_1;
                    float selected_max_1;
                    float _fma_1 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled_1);
                    if (_fma_1 >= -8.0f) {
                        selected_max_1 = row_max_val;
                        acc_scale_1 = 1.0f;
                        new_max_scaled_1 = ((row_max_val == -CAKE_FMHA_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max_1 = new_max_1;
                        if (row_max_val > -CAKE_FMHA_INF) {
                            float _exp2_1 = approx_exp2(_fma_1);
                            acc_scale_1 = _exp2_1;
                        } else {
                            acc_scale_1 = 1.0f;
                        }
                    }
                    row_max_val = selected_max_1;
                    sScale[warp % 4 * 32 + lane] = acc_scale_1;
                    mbarrier_arrive(corr_sig_addr);
                    const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_18 = {-new_max_scaled_1, -new_max_scaled_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_17, _fma_c2_18);
                    int p_base_1 = taddr + 64 + (n_iter_1&1)*384 + (warp % 4 * 32 << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (_le >= 12) {
                            float2 _exp2_pair_19 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[_le*2], _tmem_load_1[_le*2 + 1]));
                            _tmem_load_1[_le*2] = _exp2_pair_19.x;
                            _tmem_load_1[_le*2 + 1] = _exp2_pair_19.y;
                        } else {
                            _tmem_load_1[_le*2] = approx_exp2(_tmem_load_1[_le*2]);
                            _tmem_load_1[_le*2 + 1] = approx_exp2(_tmem_load_1[_le*2 + 1]);
                        }
                    }
                    uint32_t _tmem_load_1_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (n_iter_1&1)*8);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (_le >= 12) {
                            float2 _exp2_pair_20 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[_le*2 + 32], _tmem_load_1[_le*2 + 1 + 32]));
                            _tmem_load_1[_le*2 + 32] = _exp2_pair_20.x;
                            _tmem_load_1[_le*2 + 1 + 32] = _exp2_pair_20.y;
                        } else {
                            _tmem_load_1[_le*2 + 32] = approx_exp2(_tmem_load_1[_le*2 + 32]);
                            _tmem_load_1[_le*2 + 1 + 32] = approx_exp2(_tmem_load_1[_le*2 + 1 + 32]);
                        }
                    }
                    uint32_t _tmem_load_1_f16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                        _tmem_load_1_f16_0[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_0[15])));
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (_le >= 12) {
                            float2 _exp2_pair_21 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[_le*2 + 64], _tmem_load_1[_le*2 + 1 + 64]));
                            _tmem_load_1[_le*2 + 64] = _exp2_pair_21.x;
                            _tmem_load_1[_le*2 + 1 + 64] = _exp2_pair_21.y;
                        } else {
                            _tmem_load_1[_le*2 + 64] = approx_exp2(_tmem_load_1[_le*2 + 64]);
                            _tmem_load_1[_le*2 + 1 + 64] = approx_exp2(_tmem_load_1[_le*2 + 1 + 64]);
                        }
                    }
                    uint32_t _tmem_load_1_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 64], _tmem_load_1[_lp*2+1 + 64]));
                        _tmem_load_1_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[15])));
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le + 96] = approx_exp2(_tmem_load_1[_le + 96]);
                    }
                    uint32_t _tmem_load_1_f16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 96], _tmem_load_1[_lp*2+1 + 96]));
                        _tmem_load_1_f16_2[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_2[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (n_iter_1&1)*8);
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                    float2 _reg_reduce_sum2_22 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_22);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_22);
                    softmax_block_sum(&_tmem_load_1[64], &_reg_reduce_sum2_22);
                    softmax_block_sum(&_tmem_load_1[96], &_reg_reduce_sum2_22);
                    float _tmem_load_1_sum = _reg_reduce_sum2_22.x + _reg_reduce_sum2_22.y;
                    row_sum_val = row_sum_val * acc_scale_1 + _tmem_load_1_sum;
                }
                sScale[warp % 4 * 32 + lane + (unsigned int)(2 * BLOCK_M)] = row_sum_val;
                sScale[warp % 4 * 32 + lane + (unsigned int)(4 * BLOCK_M)] = row_max_val;
                mbarrier_arrive(corr_sig_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_work_id_full_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_pv_done = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_1 = 0; tile_iter_1 < total_tiles_1; tile_iter_1++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_1);
                _phase_work_id_full_0_1 ^= 1;
                unsigned int tile_idx_1 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (tile_idx_1 >= total_tiles_1) {
                    break;
                }
                unsigned int m_block_1;
                unsigned int bh_1;
                {
                    m_block_1 = tile_idx_1 % (unsigned int)NUM_M_BLOCKS;
                    bh_1 = tile_idx_1 / (unsigned int)NUM_M_BLOCKS;
                }
                if (m_block_1 * BLOCK_M >= (unsigned int)seq_lens_q[bh_1]) continue;
                int seqlen_kv_bh_1 = seq_lens_kv[bh_1];
                unsigned int num_n_blocks_1 = (seqlen_kv_bh_1 + BLOCK_N - 1) / BLOCK_N;
                int visible_end_1 = int((m_block_1 + 1) * BLOCK_M) + seqlen_kv_bh_1 - seq_lens_q[bh_1];
                visible_end_1 = max(1, min(seqlen_kv_bh_1, visible_end_1));
                num_n_blocks_1 = (visible_end_1 + BLOCK_N - 1) / BLOCK_N;
                int off_q = (unsigned int)cu_seq_lens_q[bh_1] + m_block_1 * (unsigned int)BLOCK_M;
                int seqlen_q_bh = seq_lens_q[bh_1];
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    mbarrier_wait(pv_done_addr, _phase_pv_done);
                    _phase_pv_done ^= 1;
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _vote_0 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM_HALF / 16; cr_col++) {
                            int cr_addr_hi = taddr + (unsigned int)TMEM_OUTPUT_HI_OFFSET + (warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr_hi);
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                            tmem_st_x16_f32(cr_addr_hi, _tmem_load_2);
                            int cr_addr_lo = taddr + (unsigned int)TMEM_OUTPUT_LO_OFFSET + (warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr_lo);
                            const float2 _scale2_1 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                            tmem_st_x16_f32(cr_addr_lo, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + (n_iter_2&1)*8);
                    mbarrier_arrive(corr_done_addr);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                float final_sum = sScale[warp % 4 * 32 + lane + (unsigned int)(2 * BLOCK_M)];
                float final_max = sScale[warp % 4 * 32 + lane + (unsigned int)(4 * BLOCK_M)];
                float final_scale;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_0 = approx_rcp(final_sum);
                    final_scale = _rcp_0;
                } else {
                    final_scale = 0.0f;
                }
                int q_off = m_block_1 * (unsigned int)BLOCK_M;
                // tcgen05.ld.sync.aligned requires every lane of the warp,
                // including rows beyond this request's ragged tail. Only the
                // global stores below are predicated by logical row validity.
                bool valid_row=q_off+int(warp%4*32+lane)<seqlen_q_bh;

                int logical_row = (original_q_indptr[bh_1 / NUM_Q_HEADS] + q_off + int(warp % 4 * 32 + lane))*NUM_Q_HEADS + bh_1 % NUM_Q_HEADS;
                if (valid_row && LSE_ptr) LSE_ptr[logical_row] = final_sum > 0.0f ? final_max * softmax_scale_log2 + log2f(final_sum) : -CAKE_FMHA_INF;
                #pragma unroll
                for (int ce_col = 0; ce_col < HEAD_DIM_HALF / 8; ce_col++) {
                    int ce_addr = taddr + (unsigned int)TMEM_OUTPUT_HI_OFFSET + (warp % 4 * 32 << 16) + (unsigned int)(ce_col * 8);
                    float _tmem_load_4[8];
                    tmem_ld_x8(&_tmem_load_4[0], ce_addr);
                    int o_elem = logical_row * (unsigned int)HEAD_DIM + (unsigned int)(ce_col * 8);
                    {
                        const float2 _prescale2_2 = {final_scale, final_scale};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_2);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_4[0 + _ps] *= final_scale;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                        if (valid_row) *reinterpret_cast<uint4*>(&((__half*)(O_ptr + o_elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                #pragma unroll
                for (int ce_col_1 = 0; ce_col_1 < HEAD_DIM_HALF / 8; ce_col_1++) {
                    int ce_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_LO_OFFSET + (warp % 4 * 32 << 16) + (unsigned int)(ce_col_1 * 8);
                    float _tmem_load_5[8];
                    tmem_ld_x8(&_tmem_load_5[0], ce_addr_1);
                    int o_elem_1 = logical_row * (unsigned int)HEAD_DIM + (unsigned int)HEAD_DIM_HALF + (unsigned int)(ce_col_1 * 8);
                    {
                        const float2 _prescale2_3 = {final_scale, final_scale};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_5[0])[_ps], _prescale2_3);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_5[0 + _ps] *= final_scale;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                        if (valid_row) *reinterpret_cast<uint4*>(&((__half*)(O_ptr + o_elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int total_tiles_2 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_k_full=0, _phase_v_full=0;
            unsigned int _phase_work_id_full_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0[2] = {0,0};
            unsigned int _phase_p_full_2_0[2] = {0,0};
            #pragma unroll 1
            for (unsigned int tile_iter_2 = 0; tile_iter_2 < total_tiles_2; tile_iter_2++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_2);
                _phase_work_id_full_0_2 ^= 1;
                unsigned int tile_idx_2 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (tile_idx_2 >= total_tiles_2) {
                    break;
                }
                unsigned int m_block_2;
                unsigned int bh_2;
                {
                    m_block_2 = tile_idx_2 % (unsigned int)NUM_M_BLOCKS;
                    bh_2 = tile_idx_2 / (unsigned int)NUM_M_BLOCKS;
                }
                if (m_block_2 * BLOCK_M >= (unsigned int)seq_lens_q[bh_2]) continue;
                int seqlen_kv_bh_2 = seq_lens_kv[bh_2];
                unsigned int num_n_blocks_2 = (seqlen_kv_bh_2 + BLOCK_N - 1) / BLOCK_N;
                int visible_end_2 = int((m_block_2 + 1) * BLOCK_M) + seqlen_kv_bh_2 - seq_lens_q[bh_2];
                visible_end_2 = max(1, min(seqlen_kv_bh_2, visible_end_2));
                num_n_blocks_2 = (visible_end_2 + BLOCK_N - 1) / BLOCK_N;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr,_phase_k_full);_phase_k_full^=1;
                int _mma_a_lo_0 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + 0);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores_0), "r"(0));
                int _mma_a_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + 0);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores_0), "r"(1));
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr);
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_2 - 1; n_iter_3++) {
                    const unsigned int k_stage=0, v_stage=1;
                    mbarrier_wait(kv_full_addr,_phase_k_full);_phase_k_full^=1;
                    int _mma_a_lo_6 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_scores_0 + ((n_iter_3+1)&1)*384), "r"(0));
                    int _mma_a_lo_7 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_scores_0 + ((n_iter_3+1)&1)*384), "r"(1));
                    elect_commit(s_full_addr + ((n_iter_3+1)&1)*8);
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    mbarrier_wait(kv_full_addr+8,_phase_v_full);_phase_v_full^=1;
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr + (n_iter_3&1)*8, _phase_p_full_0[n_iter_3&1]);
                    _phase_p_full_0[n_iter_3&1] ^= 1;
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_2), "r"(tmem_softmax_0 + (n_iter_3&1)*384), "r"(((first_pv_flag) ? 0 : 1)));
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_3), "r"(tmem_softmax_0 + (n_iter_3&1)*384), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr + (n_iter_3&1)*8, _phase_p_full_2_0[n_iter_3&1]);
                    _phase_p_full_2_0[n_iter_3&1] ^= 1;
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_4), "r"(tmem_softmax_0 + (n_iter_3&1)*384), "r"(1));
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_5), "r"(tmem_softmax_0 + (n_iter_3&1)*384), "r"(1));
                    elect_commit(pv_done_addr);
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                mbarrier_wait(kv_full_addr+8,_phase_v_full);_phase_v_full^=1;
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr + ((num_n_blocks_2-1)&1)*8, _phase_p_full_0[(num_n_blocks_2-1)&1]);
                _phase_p_full_0[(num_n_blocks_2-1)&1] ^= 1;
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_8), "r"(tmem_softmax_0 + ((num_n_blocks_2-1)&1)*384), "r"(((first_pv_flag_1) ? 0 : 1)));
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_9), "r"(tmem_softmax_0 + ((num_n_blocks_2-1)&1)*384), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr + ((num_n_blocks_2-1)&1)*8, _phase_p_full_2_0[(num_n_blocks_2-1)&1]);
                _phase_p_full_2_0[(num_n_blocks_2-1)&1] ^= 1;
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_10), "r"(tmem_softmax_0 + ((num_n_blocks_2-1)&1)*384), "r"(1));
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_11), "r"(tmem_softmax_0 + ((num_n_blocks_2-1)&1)*384), "r"(1));
                elect_commit(kv_empty_addr+8);
                elect_commit(o_full_addr);
            }
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        { // page_offsets_main
            unsigned int total_tiles_3 = NUM_M_BLOCKS * total_bh;
            unsigned int po_stage = 0;
            unsigned int _phase_work_id_full_0_3 = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_3 = 0; tile_iter_3 < total_tiles_3; tile_iter_3++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_3);
                _phase_work_id_full_0_3 ^= 1;
                unsigned int tile_idx_3 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (tile_idx_3 >= total_tiles_3) {
                    break;
                }
                unsigned int m_block_3;
                unsigned int bh_3;
                {
                    m_block_3 = tile_idx_3 % (unsigned int)NUM_M_BLOCKS;
                    bh_3 = tile_idx_3 / (unsigned int)NUM_M_BLOCKS;
                }
                if (m_block_3 * BLOCK_M >= (unsigned int)seq_lens_q[bh_3]) continue;
                int seqlen_kv_bh_3 = seq_lens_kv[bh_3];
                unsigned int num_n_blocks_3 = (seqlen_kv_bh_3 + BLOCK_N - 1) / BLOCK_N;
                int visible_end_3 = int((m_block_3 + 1) * BLOCK_M) + seqlen_kv_bh_3 - seq_lens_q[bh_3];
                visible_end_3 = max(1, min(seqlen_kv_bh_3, visible_end_3));
                num_n_blocks_3 = (visible_end_3 + BLOCK_N - 1) / BLOCK_N;
                unsigned int kv_bh = bh_3 / (unsigned int)HEADS_PER_GROUP;
                int pt_base = kv_bh * (unsigned int)max_pages_per_seq;
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = num_n_blocks_3 - 1 - ni;
                    int pt_off = (unsigned int)pt_base + n;
                    mbarrier_wait(page_empty_addr + (po_stage) * 8, _phase_page_empty);
                    if (elect_sync()) {
                        int pg_idx = po_stage * 8;
                        smem_pages[pg_idx] = page_table[pt_off];
                        mbarrier_arrive(page_full_addr + (po_stage) * 8);
                    }
                    po_stage += 1;
                    if (po_stage == 6) { po_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_work_id_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_iter_4 = 0; tile_iter_4 < total_tiles_4; tile_iter_4++) {
                    mbarrier_wait(work_id_empty_addr, _phase_work_id_empty_0);
                    _phase_work_id_empty_0 ^= 1;
                    unsigned int _atomic_old_0 = atomicAdd(dynamic_counter, 1);
                    work_id_slot[0] = _atomic_old_0;
                    mbarrier_arrive(work_id_full_addr);
                    if (_atomic_old_0 >= total_tiles_4) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 11) {
        { // load_main
            unsigned int total_tiles_5 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_k_empty=1, _phase_v_empty=1;
            unsigned int load_po_stage = 0;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_5 = 0; tile_iter_5 < total_tiles_5; tile_iter_5++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_4);
                _phase_work_id_full_0_4 ^= 1;
                unsigned int tile_idx_4 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (tile_idx_4 >= total_tiles_5) {
                    break;
                }
                unsigned int m_block_4;
                unsigned int bh_4;
                {
                    m_block_4 = tile_idx_4 % (unsigned int)NUM_M_BLOCKS;
                    bh_4 = tile_idx_4 / (unsigned int)NUM_M_BLOCKS;
                }
                if (m_block_4 * BLOCK_M >= (unsigned int)seq_lens_q[bh_4]) continue;
                int seqlen_kv_bh_4 = seq_lens_kv[bh_4];
                unsigned int num_n_blocks_4 = (seqlen_kv_bh_4 + BLOCK_N - 1) / BLOCK_N;
                int visible_end_4 = int((m_block_4 + 1) * BLOCK_M) + seqlen_kv_bh_4 - seq_lens_q[bh_4];
                visible_end_4 = max(1, min(seqlen_kv_bh_4, visible_end_4));
                num_n_blocks_4 = (visible_end_4 + BLOCK_N - 1) / BLOCK_N;
                int off_q_1 = (unsigned int)cu_seq_lens_q[bh_4] + m_block_4 * (unsigned int)BLOCK_M;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    tma_4d_gmem2smem(smem_q_hi_addr, Q, 0, off_q_1, 0, bh_4 % NUM_Q_HEADS, q_full_addr);
                    tma_4d_gmem2smem(smem_q_lo_addr, Q, 0, off_q_1, 2, bh_4 % NUM_Q_HEADS, q_full_addr);
                }
                // Fixed K and V slots permit K lookahead independent of
                // current V lifetime: K0,V0,K1,K2,V1,K3,V2,...,Vlast.
                int previous_page=0,previous_slot=0;
                #pragma unroll 1
                for(unsigned int ni=0;ni<num_n_blocks_4;++ni) {
                    mbarrier_wait(page_full_addr+load_po_stage*8,_phase_page_full);
                    int current_page=smem_pages[load_po_stage*8];
                    int current_slot=load_po_stage;
                    ++load_po_stage;
                    if(load_po_stage==6){load_po_stage=0;_phase_page_full^=1;}
                    mbarrier_wait(kv_empty_addr+0,_phase_k_empty);_phase_k_empty^=1;
if(elect_sync()) {
  mbarrier_arrive_expect_tx(kv_full_addr+0,65536);
  tma_4d_gmem2smem(smem_kv_hi_addr+0,K,0,0,0,current_page,kv_full_addr+0);
  tma_4d_gmem2smem(smem_kv_hi_addr+16384,K,0,0,1,current_page,kv_full_addr+0);
  tma_4d_gmem2smem(smem_kv_lo_addr+0,K,0,0,2,current_page,kv_full_addr+0);
  tma_4d_gmem2smem(smem_kv_lo_addr+16384,K,0,0,3,current_page,kv_full_addr+0);
}
                    if(ni==0) {
                        mbarrier_wait(kv_empty_addr+8,_phase_v_empty);_phase_v_empty^=1;
if(elect_sync()) {
  mbarrier_arrive_expect_tx(kv_full_addr+8,65536);
  tma_4d_gmem2smem(smem_kv_hi_addr+32768,V,0,0,0,current_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_hi_addr+49152,V,0,0,1,current_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+32768,V,0,0,2,current_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+49152,V,0,0,3,current_page,kv_full_addr+8);
}
                        if(elect_sync())mbarrier_arrive(page_empty_addr+current_slot*8);
                    } else {
                        if(ni>=2) {
                            mbarrier_wait(kv_empty_addr+8,_phase_v_empty);_phase_v_empty^=1;
if(elect_sync()) {
  mbarrier_arrive_expect_tx(kv_full_addr+8,65536);
  tma_4d_gmem2smem(smem_kv_hi_addr+32768,V,0,0,0,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_hi_addr+49152,V,0,0,1,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+32768,V,0,0,2,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+49152,V,0,0,3,previous_page,kv_full_addr+8);
}
                            if(elect_sync())mbarrier_arrive(page_empty_addr+previous_slot*8);
                        }
                        previous_page=current_page;previous_slot=current_slot;
                    }
                }
                if(num_n_blocks_4>1) {
                    mbarrier_wait(kv_empty_addr+8,_phase_v_empty);_phase_v_empty^=1;
if(elect_sync()) {
  mbarrier_arrive_expect_tx(kv_full_addr+8,65536);
  tma_4d_gmem2smem(smem_kv_hi_addr+32768,V,0,0,0,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_hi_addr+49152,V,0,0,1,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+32768,V,0,0,2,previous_page,kv_full_addr+8);
  tma_4d_gmem2smem(smem_kv_lo_addr+49152,V,0,0,3,previous_page,kv_full_addr+8);
}
                    if(elect_sync())mbarrier_arrive(page_empty_addr+previous_slot*8);
                }

            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
