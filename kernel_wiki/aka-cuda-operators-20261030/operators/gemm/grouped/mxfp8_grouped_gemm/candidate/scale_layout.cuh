#pragma once
#include "dispatch.cuh"

// csrc/apis/layout.hpp


// csrc/jit_kernels/impls/smxx_layout.hpp
#include <format>
#include <torch/python.h>

namespace deep_gemm {

class PackFP32IntoUE8M0Runtime final {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k, gran_k, k_alignment;
        bool use_psum_layout;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *grouped_layout;

        deep_jit::cuda::LaunchOptions options;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = precompiled::load(tag, spec_config( args.num_groups, args.options.block_dim->x, args.block_mn, args.block_packed_sf_k,
    "true", args.use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, args.options,
            args.sf, args.out, args.grouped_layout, args.mn, args.sf_k, args.packed_sf_k,
            args.gran_k, args.k_alignment
        );
    }
};

static std::tuple<int, int, int, int, int, torch::Tensor> preprocess_sf(const torch::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::kFloat);
    const auto batched_sf = dim == 2 ? sf.unsqueeze(0) : sf;

    const auto [num_sf_batches, mn, sf_k] = get_shape<3>(batched_sf);
    const auto tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& sf) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

    const auto out = torch::empty_strided({num_sf_batches, mn, sf_k},
                                          {tma_aligned_mn * sf_k, 1, tma_aligned_mn},
                                          batched_sf.options());

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        out.copy_(batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k;
        const auto smem_size = block_mn * (smem_sf_k + (1 - smem_sf_k % 2)) * static_cast<int>(sizeof(float));

        // Compile
        const auto kernel = precompiled::load("transpose_fp32", spec_config( num_threads, block_mn, sf_k, block_sf_k));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.data_ptr(), out.data_ptr(), static_cast<uint32_t>(mn)
        );
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::Tensor& sf) {
    const auto sf_reshaped = (sf.dim() == 2) ? sf.unsqueeze(0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto ue8m0_tensor = sf_reshaped.view(torch::kInt32).bitwise_right_shift(23).to(torch::kUInt8);

    // Second, make padded packed tensors
    const auto [num_sf_batches, mn, k] = get_shape<3>(sf_reshaped);
    const auto aligned_mn = get_tma_aligned_size(mn, 4);
    const auto aligned_k  = align(k, 4);

    const auto options = torch::TensorOptions().device(sf.device()).dtype(torch::kUInt8);
    auto padded = torch::zeros({num_sf_batches, aligned_mn, aligned_k}, options);
    // ReSharper disable once CppExpressionWithoutSideEffects
    padded.slice(1, 0, mn).slice(2, 0, k).copy_(ue8m0_tensor);
    padded = padded.view(-1).view(torch::kInt32).view({num_sf_batches, aligned_mn, aligned_k / 4});

    // Finally, transpose
    auto out = torch::empty_strided({num_sf_batches, aligned_mn, aligned_k / 4},
                                    {aligned_mn * (aligned_k / 4), 1, aligned_mn},
                                    at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    out = out.copy_(padded).slice(1, 0, mn);
    return (sf.dim() == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                                  const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto packed_sf_k = ceil_div(sf_k, 4);
    const auto out = torch::empty_strided({num_sf_batches, mn, packed_sf_k},
                                          {packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn},
                                          at::TensorOptions().device(batched_sf.device()).dtype(torch::kInt));

    // PSUM layout (always 2D contiguous) lets the pack kernel skip uninitialized MN gap rows
    const auto use_psum_layout = psum_layout.has_value();
    if (use_psum_layout) {
        DG_HOST_ASSERT(num_sf_batches == 1 and batched_sf.is_contiguous());
        DG_HOST_ASSERT(psum_layout->scalar_type() == torch::kInt and psum_layout->is_contiguous());
        DG_HOST_ASSERT(psum_layout->numel() > 0);
    }
    const auto m_alignment = use_psum_layout ? heuristics_runtime->get_mk_alignment_for_contiguous_layout() : 0;
    const auto num_psum_groups = use_psum_layout ? static_cast<int>(psum_layout->numel()) : 1;

    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        constexpr int block_mn = 48;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto psum_smem_elems = use_psum_layout ? align(num_psum_groups * 2, 4) : 0;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k + 1;
        const auto smem_size = block_mn * smem_sf_k * 4 + psum_smem_elems * 4;

        // Compile
        const auto kernel = precompiled::load("transpose_and_pack_fp32_into_ue8m0", spec_config( num_threads, block_mn, sf_k, block_sf_k, num_psum_groups, use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.data_ptr(), out.data_ptr(), static_cast<uint32_t>(mn),
            use_psum_layout ? psum_layout->data_ptr() : nullptr, static_cast<uint32_t>(m_alignment)
        );
    } else {
        DG_HOST_ASSERT(not use_psum_layout);
        if (mn % 4 != 0 or num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;

        // Compile and launch
        PackFP32IntoUE8M0Runtime::compile_and_launch("pack_fp32_into_ue8m0", {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            .gran_k = 128,
            .k_alignment = 128,
            .use_psum_layout = false,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .grouped_layout = nullptr,
            .options = {
                .grid_dim = dim3(ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k), 1),
                .block_dim = dim3(num_threads, 1, 1),
            }
        });
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

} // namespace deep_gemm

namespace deep_gemm::layout {

static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::variant<std::tuple<int, int, int>,
                                                                          std::tuple<int, int>>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const std::optional<bool>& is_sfa,
                                                       const bool& disable_ue8m0_cast,
                                                       const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    const auto arch_major = jit->device.get_arch_major();

    // Get granularity MN/K from recipe
    int gran_mn, gran_k;
    if (auto p = std::get_if<std::tuple<int, int, int>>(&recipe)) {
        DG_HOST_ASSERT(is_sfa.has_value());
        gran_mn = is_sfa.value() ? std::get<0>(*p) : std::get<1>(*p);
        gran_k = std::get<2>(*p);
    } else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {
        DG_HOST_ASSERT(not is_sfa.has_value());
        std::tie(gran_mn, gran_k) = *p;
    } else {
        DG_HOST_UNREACHABLE("Invalid recipe");
    }

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, x, gran_k) on SM100/SM120: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12)) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto broadcasted = gran_mn == 1 ? sf :
                                 sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted, psum_layout);
    }

    // (INT, 1, gran_k) on SM100/SM120: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(
        const torch::Tensor& sfa, const torch::Tensor& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false,
        // PSUM layout only applies to SFA (B is weights with no gaps)
        const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    // Use default recipe, if none is specified
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());

    // Must be either 'recipe' or the 'recipe_a' + 'recipe_b' pair.
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());

    // Transform SFA and SFB layout (PSUM only for SFA)
    const auto transformed_sfa = recipe.has_value() ? transform_sf_into_required_layout(sfa, m, k, recipe.value(), num_groups_a, true, disable_ue8m0_cast, psum_layout)
                                                    : transform_sf_into_required_layout(sfa, m, k, recipe_a.value(), num_groups_a, std::nullopt, disable_ue8m0_cast, psum_layout);
    const auto transformed_sfb = recipe.has_value() ? transform_sf_into_required_layout(sfb, n, k, recipe.value(), num_groups_b, false, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfb, n, k, recipe_b.value(), num_groups_b, std::nullopt, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

} // namespace deep_gemm::layout
