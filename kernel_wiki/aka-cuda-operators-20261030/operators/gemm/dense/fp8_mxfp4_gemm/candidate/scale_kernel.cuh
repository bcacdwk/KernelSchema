#pragma once
#include "kernel_helpers.cuh"

namespace deep_gemm::scale_kernel {
// Configuration comes from dispatch.cuh; local aliases preserve constant folding.
template<
    uint32_t C_kNumThreads,
    uint32_t C_BLOCK_MN,
    uint32_t C_SF_K,
    uint32_t C_BLOCK_SF_K,
    uint32_t C_kNumPsumGroups,
    bool C_kUsePsumLayout>
DG_GLOBAL void transpose_and_pack_fp32_into_ue8m0(float* sf, uint32_t* out, const uint32_t mn,
                                                       const uint32_t* grouped_layout, const uint32_t m_alignment)
{
    constexpr uint32_t kNumThreads = C_kNumThreads;
    constexpr uint32_t BLOCK_MN = C_BLOCK_MN;
    constexpr uint32_t SF_K = C_SF_K;
    constexpr uint32_t BLOCK_SF_K = C_BLOCK_SF_K;
    constexpr uint32_t kNumPsumGroups = C_kNumPsumGroups;
    constexpr bool kUsePsumLayout = C_kUsePsumLayout;

    extern __shared__ uint32_t smem_buffer[];

    // Shapes and strides
    constexpr auto kNumPackedSFK = math::constexpr_ceil_div(SF_K, 4u);
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(int));
    typedef typename utils::Vectorized<sizeof(float) * SF_K>::vec_t in_vec_t;
    constexpr auto kNumElemsPerVec = sizeof(in_vec_t) / sizeof(float);
    constexpr auto kSmemSFK = SF_K < BLOCK_SF_K ? SF_K : BLOCK_SF_K + 1;
    DG_STATIC_ASSERT(BLOCK_SF_K % 4 == 0 and BLOCK_SF_K % kNumElemsPerVec == 0, "Invalid SF block size");
    const auto num_mn_blocks = math::ceil_div(mn, BLOCK_MN);
    const auto block_mn_idx = SF_K < BLOCK_SF_K ? blockIdx.x : blockIdx.x % num_mn_blocks;
    const auto block_sf_k_idx = SF_K < BLOCK_SF_K ? 0u : blockIdx.x / num_mn_blocks;
    const auto block_sf_k_start = block_sf_k_idx * BLOCK_SF_K;
    const auto in_block_mn = min(BLOCK_MN, mn - block_mn_idx * BLOCK_MN);
    const auto in_block_sf_k = SF_K < BLOCK_SF_K ? SF_K : min(BLOCK_SF_K, SF_K - block_sf_k_start);
    const auto in_block_packed_sf_k = math::ceil_div(in_block_sf_k, 4u);
    const auto tma_aligned_mn = math::align<uint64_t>(mn, kNumTMAAlignedElems);

    // Wait for primary kernel completion before reading SFs or PSUM layout
    cudaGridDependencySynchronize();

    constexpr auto kNumPsumLayoutElems = kUsePsumLayout ? math::constexpr_align(kNumPsumGroups * 2, 4u) : 0;
    const auto group_mn_start = smem_buffer;
    const auto group_mn_end = smem_buffer + kNumPsumGroups;
    const auto sf_smem_buffer = smem_buffer + kNumPsumLayoutElems;

    // Precompute PSUM valid MN ranges into smem before the SF block.
    if constexpr (kUsePsumLayout) {
        for (uint32_t g = threadIdx.x; g < kNumPsumGroups; g += kNumThreads) {
            const auto end_g = grouped_layout[g];
            group_mn_start[g] = g == 0 ? 0 : math::align(grouped_layout[g - 1], m_alignment);
            group_mn_end[g] = end_g;
        }
        __syncthreads();
    }

    // PSUM path skips uninitialized per-group MN gaps without early return.
    const auto is_valid_mn = [&](const uint32_t& global_mn_idx) -> bool {
        if constexpr (kUsePsumLayout) {
            bool valid = false;
            for (uint32_t g = 0; g < kNumPsumGroups; ++ g)
                valid |= (global_mn_idx >= group_mn_start[g] and global_mn_idx < group_mn_end[g]);
            return valid;
        } else {
            return global_mn_idx < mn;
        }
    };

    // Shift into the group
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * kNumPackedSFK;

    // Load FP32 SFs
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block size");
    const auto local_sf = sf + static_cast<uint64_t>(block_mn_idx) * BLOCK_MN * SF_K + block_sf_k_start;
    if constexpr (SF_K < BLOCK_SF_K) {
        const auto local_sf_u32 = reinterpret_cast<uint32_t*>(local_sf);
        const auto num_values = in_block_mn * SF_K;
        const auto num_uint4 = num_values / 4;
        #pragma unroll
        for (uint32_t i = threadIdx.x; i < num_uint4; i += kNumThreads) {
            const auto& [x, y, z, w] = reinterpret_cast<const uint4*>(local_sf_u32)[i];
            ptx::st_shared(reinterpret_cast<uint4*>(sf_smem_buffer) + i, x, y, z, w);
        }
        if (const auto unaligned_idx = num_uint4 * 4 + threadIdx.x; unaligned_idx < num_values)
            ptx::st_shared(sf_smem_buffer + unaligned_idx, local_sf_u32[unaligned_idx]);
    } else {
        const auto warp_idx = threadIdx.x / 32;
        const auto lane_idx = ptx::get_lane_idx();
        for (uint32_t row = warp_idx; row < in_block_mn; row += kNumThreads / 32) {
            for (uint32_t col = lane_idx * kNumElemsPerVec; col < in_block_sf_k; col += 32 * kNumElemsPerVec) {
                const auto in_vec = *reinterpret_cast<const in_vec_t*>(local_sf + row * SF_K + col);
                const auto in_values = reinterpret_cast<const uint32_t*>(&in_vec);
                #pragma unroll
                for (uint32_t j = 0; j < kNumElemsPerVec; ++ j)
                    ptx::st_shared(sf_smem_buffer + row * kSmemSFK + col + j, in_values[j]);
            }
        }
    }
    __syncthreads();

    // Pack into UE8M0 and store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < in_block_packed_sf_k * BLOCK_MN; i += kNumThreads) {
        const auto sf_k_pack_idx = i / BLOCK_MN, mn_idx = i % BLOCK_MN;
        const auto global_mn_idx = block_mn_idx * BLOCK_MN + mn_idx;
        const auto global_sf_k_pack_idx = block_sf_k_start / 4 + sf_k_pack_idx;
        const auto in_bounds_mn = global_mn_idx < mn;
        const auto valid_mn = is_valid_mn(global_mn_idx);

        // Load shared memory
        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const auto sf_k_idx = sf_k_pack_idx * 4 + j;
            const auto global_sf_k_idx = block_sf_k_start + sf_k_idx;
            values[j] = valid_mn and global_sf_k_idx < SF_K ? ptx::ld_shared(sf_smem_buffer + mn_idx * kSmemSFK + sf_k_idx) : 0;
            // FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
            DG_TRAP_ONLY_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0);
        }

        // Pack and store
        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >>  7u);
        packed |= (values[3] <<  1u);
        // Write safe finite scale codes for PSUM gap rows; UE8M0 0xff is NaN.
        if (in_bounds_mn)
            out[global_sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}
} // namespace deep_gemm::scale_kernel
