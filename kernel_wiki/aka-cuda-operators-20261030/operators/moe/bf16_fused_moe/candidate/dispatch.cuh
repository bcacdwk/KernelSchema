#pragma once
#include "runtime.cuh"

// csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp
#include <format>
#include <torch/python.h>

// csrc/jit_kernels/heuristics/mega_moe.hpp
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

// csrc/utils/math.hpp
// TODO: merge this file with `math.cuh` (the device part)
#include <torch/python.h>

namespace deep_gemm {

// TODO: use `torch::kFloat4_e2m1fn_x2`
constexpr auto kPackedFP4 = torch::kInt8;

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

} // namespace deep_gemm

// csrc/jit_kernels/heuristics/sm100.hpp
#include <algorithm>

// Reuse some types in the JIT modules

// csrc/jit_kernels/heuristics/common.hpp
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

// csrc/jit_kernels/heuristics/config.hpp

#include <c10/core/ScalarType.h>

namespace deep_gemm {

/// GEMM descriptors
struct GemmDesc {
    GemmType gemm_type;
    KernelType kernel_type;
    int m, n, k, num_groups;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    dg::UMMA::Major major_a;
    dg::UMMA::Major major_b;
    bool with_accumulation;

    // Requirements from users
    int num_sms, tc_util;
    std::string compiled_dims;

    // SF granularity for split-K alignment: max(gran_k_a, gran_k_b). SM120 only.
    int max_gran_k = 128;

    // False for AB-swap (transposed, stride_cd_n != 1) output: the TMA-store epilogue
    // cannot express it, so the kernel falls back to the strided-store epilogue. SM120 only.
    bool cd_n_contiguous = true;

    // SM100 m-grouped psum layout padding contract
    bool ensure_zero_padding = true;

    // Shape for heuristic generation
    int expected_m = 0, expected_n = 0, expected_k = 0, expected_num_groups = 0;
    int get_expected_m() const { return expected_m > 0 ? expected_m : m; }
    int get_expected_n() const { return expected_n > 0 ? expected_n : n; }
    int get_expected_k() const { return expected_k > 0 ? expected_k : k; }
    int get_expected_num_groups() const { return expected_num_groups > 0 ? expected_num_groups : num_groups; }

    bool is_mxf4_mma() const {
        return a_dtype == kPackedFP4 and b_dtype == kPackedFP4;
    }

    MmaKind get_mma_kind() const {
        if (a_dtype == torch::kBFloat16)
            return MmaKind::BF16;
        return is_mxf4_mma() ? MmaKind::MXF4 : MmaKind::MXFP8FP4;
    }

    int get_smem_pack_factor() const {
        return is_mxf4_mma() ? 2 : 1;
    }

    void check_validity() const {
        if (get_mma_kind() == MmaKind::BF16) {
            DG_HOST_ASSERT(a_dtype == torch::kBFloat16 and b_dtype == torch::kBFloat16);
        } else {
            DG_HOST_ASSERT(a_dtype == torch::kFloat8_e4m3fn or a_dtype == kPackedFP4);
            DG_HOST_ASSERT(b_dtype == torch::kFloat8_e4m3fn or b_dtype == kPackedFP4);
        }
        // FP8 D implies casting with dynamic per-32 UE8M0 SFD output, only exposed for batched GEMMs
        DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat or
                       (cd_dtype == torch::kFloat8_e4m3fn and gemm_type == GemmType::Batched and
                        not with_accumulation));
        DG_HOST_ASSERT(num_sms % 2 == 0);
    }

    friend std::ostream& operator << (std::ostream& os, const GemmDesc& desc) {
        MmaKind mma_kind = desc.get_mma_kind();
        os << "GemmDesc(gemm_type=" << static_cast<int>(desc.gemm_type)
           << ", kernel_type=" << static_cast<int>(desc.kernel_type)
           << ", m=" << desc.m << ", n=" << desc.n << ", k=" << desc.k
           << ", num_groups=" << desc.num_groups
           << ", major_a=" << static_cast<int>(desc.major_a)
           << ", major_b=" << static_cast<int>(desc.major_b)
           << ", mma_kind=" << static_cast<int>(mma_kind)
           << ", a_dtype=" << c10::toString(desc.a_dtype)
           << ", b_dtype=" << c10::toString(desc.b_dtype)
           << ", cd_dtype=" << c10::toString(desc.cd_dtype)
           << ", with_accumulation=" << static_cast<int>(desc.with_accumulation)
           << ", num_sms=" << desc.num_sms
           << ", tc_util=" << desc.tc_util
           << ", compiled_dims=" << desc.compiled_dims
           << ", ensure_zero_padding=" << static_cast<int>(desc.ensure_zero_padding)
           << ", expected_m=" << desc.expected_m
           << ", expected_n=" << desc.expected_n
           << ", expected_k=" << desc.expected_k
           << ", expected_num_groups=" << desc.expected_num_groups << ")";
        return os;
    }
};

/// GEMM configs
struct Layout {
    int swap_ab;
    int block_m, block_n, block_k;
    int cluster_m, cluster_n;

    int get_cluster_size() const {
        return cluster_m * cluster_n;
    }

    friend std::ostream& operator << (std::ostream& os, const Layout& layout) {
        os << "Layout(swap_ab=" << layout.swap_ab
           << ", block_m=" << layout.block_m << ", block_n=" << layout.block_n << ", block_k=" << layout.block_k
           << ", cluster_m=" << layout.cluster_m << ", cluster_n=" << layout.cluster_n << ")";
        return os;
    }
};

struct StorageConfig {
    int load_block_m, load_block_n;
    int store_block_m, store_block_n;

    int swizzle_a_mode, swizzle_b_mode;
    int swizzle_cd_mode;

    friend std::ostream& operator << (std::ostream& os, const StorageConfig& config) {
        os << "StorageConfig("
           << "load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m << ", store_block_n=" << config.store_block_n
           << ", swizzle_a_mode=" << config.swizzle_a_mode << ", swizzle_b_mode=" << config.swizzle_b_mode
           << ", swizzle_cd_mode=" << config.swizzle_cd_mode << ")";
        return os;
    }
};

struct PipelineConfig {
    int smem_size;
    int num_stages;
    int num_tma_store_stages;

    friend std::ostream& operator << (std::ostream& os, const PipelineConfig& config) {
        os << "PipelineConfig("
           << "smem_size=" << config.smem_size
           << ", num_stages=" << config.num_stages
           << ", num_tma_store_stages=" << config.num_tma_store_stages << ")";
        return os;
    }
};

struct LaunchConfig {
    int num_sms;
    int num_sms_per_cluster;
    int num_threads;

    int num_tma_threads;
    int num_math_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const LaunchConfig& config) {
        os << "LaunchConfig("
           << "num_sms=" << config.num_sms << ", num_sms_per_cluster=" << config.num_sms_per_cluster
           << ", num_threads=" << config.num_threads
           << ", num_tma_threads=" << config.num_tma_threads << ", num_math_threads=" << config.num_math_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

struct GemmConfig {
    Layout layout;
    StorageConfig storage_config;
    PipelineConfig pipeline_config;
    LaunchConfig launch_config;

    // Number of K partitions. SM120 only; set by the impl after `get_best_config`.
    int split_k_factor = 1;

    friend std::ostream& operator << (std::ostream& os, const GemmConfig& config) {
        os << "GemmConfig("
           << "layout=" << config.layout
           << ", storage_config=" << config.storage_config
           << ", pipeline_config=" << config.pipeline_config
           << ", launch_config=" << config.launch_config << ")";
        return os;
    }
};

/// Config comparators
struct LayoutInfo {
    int num_waves;
    int last_wave_util;
    int64_t num_cycles;
    Layout layout;

    friend std::ostream& operator << (std::ostream& os, const LayoutInfo& config) {
        os << "LayoutInfo("
           << "num_waves=" << config.num_waves
           << ", last_wave_util=" << config.last_wave_util
           << ", num_cycles=" << config.num_cycles << ")";
        return os;
    }
};

}  // namespace deep_gemm

// csrc/utils/layout.hpp

#include <torch/python.h>

namespace deep_gemm {

// Major-ness stuffs
template <bool kRequireContiguousBatch = true>
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if constexpr (kRequireContiguousBatch) {
        if (dim == 3)
            DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

template <bool kRequireContiguousBatch = true>
static dg::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check<kRequireContiguousBatch>(t);
    return t.stride(-1) == 1 ? dg::UMMA::Major::K : dg::UMMA::Major::MN;
}

template <bool kRequireContiguousBatch = true>
static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check<kRequireContiguousBatch>(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_fp4_requires_k_major(const torch::Tensor& a, const torch::Tensor& b) {
    return jit->device.get_arch_major() == 9 or
           (a.scalar_type() == kPackedFP4 and b.scalar_type() == kPackedFP4);
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.is_cuda());
    DG_HOST_ASSERT(t.dim() == N);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

// Returns logical shape for packed FP4 by expanding the last dimension.
template <int N>
static auto get_logical_shape(const torch::Tensor& t) {
    auto shape = get_shape<N>(t);
    if (t.scalar_type() == kPackedFP4)
        std::get<N - 1>(shape) *= 2;
    return shape;
}

static std::tuple<int, int> check_ab_fp8_fp4(const torch::Tensor& ab, const dg::UMMA::Major& major, const int& arch_major) {
    auto [mn, k] = get_shape<2>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12));
        major == dg::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(mn, k);
}

static std::tuple<int, int, int> check_grouped_ab_fp8_fp4(const torch::Tensor& ab, const dg::UMMA::Major& major, const int& arch_major) {
    auto [num_groups, mn, k] = get_shape<3>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12));
        major == dg::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(num_groups, mn, k);
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10 or arch_major == 12) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format
            std::make_tuple(1,   1, 128);   // 1D1D kernels
    }
    DG_HOST_UNREACHABLE("Unknown recipe");
}

// SF layouts
static torch::Tensor check_sf_layout(const torch::Tensor& sf,
                                     const int& mn, const int& k,
                                     const int& gran_mn, const int& gran_k,
                                     const std::optional<int>& num_groups,
                                     const bool& tma_stride_check = false,
                                     const bool& sm90_sfb_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto sf_dtype = sf.scalar_type();
    DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.size(-3) == num_groups.value());
    DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));

    // TMA stride checks: TMA aligned and MN-major
    if (tma_stride_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
        // Check contiguity in the MN direction
        DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);
        DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
    }

    // SM90 SFB must be contiguous, or contiguous after transposing the last two dimensions
    if (sm90_sfb_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.size(-2) * sf.size(-1));
        DG_HOST_ASSERT((sf.stride(-1) == 1 and sf.stride(-2) == sf.size(-1)) or
                       (sf.stride(-1) == sf.size(-2) and sf.stride(-2) == 1));
    }
    return sf;
}

} // namespace deep_gemm

namespace deep_gemm {

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        std::stringstream ss;
        ss << desc;
        const auto key = ss.str();

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm

// csrc/jit_kernels/heuristics/utils.hpp

// Reuse some types in the JIT modules

namespace deep_gemm {

static int get_num_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXF4:     return 4;
        default: DG_HOST_UNREACHABLE("Unknown MMA kind");
    }
}

template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

} // namespace deep_gemm

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16:     return {0, 0};
            case MmaKind::MXFP8FP4:
            case MmaKind::MXF4:
                return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block K is always in a fixed manner
        const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());

        // Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout or
            desc.gemm_type == GemmType::MGroupedMasked) {
            const bool swap_ab = true;
            const auto block_n = 128;
            const auto block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
            const auto cluster_m = 1;
            const auto cluster_n = ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 ? 2 : 1;
            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};
            std::vector<Layout> candidates = {layout};
            return candidates;
        }

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int swap_ab = 0; swap_ab < 2; ++ swap_ab) {
            // Block M/N candidates
            std::vector<int> block_m_candidates;
            std::vector<int> block_n_candidates;
            if (swap_ab) {
                int step = std::lcm(16, heuristics_runtime->get_block_m_multiple_of());
                int end = 256;
                for (int i = step; i <= end; i += step)
                    block_m_candidates.push_back(i);

                // TODO: consider other block N
                block_n_candidates = {128};
            } else {
                // NOTES: smaller block M can avoid TMA L2 OOB bound
                // TODO: consider block M = 256
                if (desc.m <= 32) block_m_candidates = {32};
                else if (desc.m <= 64) block_m_candidates = {64};
                else block_m_candidates = {128};

                // Small block size for small shape
                if (16 % heuristics_runtime->get_block_n_multiple_of() == 0)
                    block_n_candidates.push_back(16);
                int step = std::lcm(32, heuristics_runtime->get_block_n_multiple_of());
                // For small K, fewer store blocks improve store/compute overlap and reduce epilogue bottleneck
                int end = desc.k <= 256 ? 128 : 256;
                for (int i = step; i <= end; i += step)
                    block_n_candidates.push_back(i);
            }

            for (int cluster_m = 1; cluster_m <= 2; ++ cluster_m) {
                // After swapping, layout A/D can only do on cluster N
                if (swap_ab == 1 and cluster_m > 1)
                    continue;

                for (int cluster_n = 1; cluster_n <= 2; ++ cluster_n) {
                    // We only support cluster 2
                    if (cluster_m * cluster_n > 2)
                        continue;

                    // Only support layout A/D
                    if (swap_ab == 0 and cluster_n > 1)
                        continue;

                    // SM count must be divisible
                    if (desc.num_sms % (cluster_m * cluster_n) != 0)
                        continue;

                    for (int block_m: block_m_candidates) {
                        // Ensure large swizzle sizes (32B swizzle yields poor performance)
                        const auto swizzle_a_requirement = desc.a_dtype == kPackedFP4 ? 128 : 64;
                        // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                        const auto load_block_m_requirement = desc.major_a == dg::UMMA::Major::MN ? swizzle_a_requirement : 8;
                        if ((block_m / cluster_n) % load_block_m_requirement != 0)
                            continue;

                        // Shape must be divisible for multicast
                        if (ceil_div(desc.m, block_m) % cluster_m != 0)
                            continue;

                        for (int block_n: block_n_candidates) {
                            // Ensure large swizzle sizes (32B swizzle yields poor performance)
                            const auto swizzle_b_requirement = desc.b_dtype == kPackedFP4 ? 128 : 64;
                            // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                            const auto load_block_n_requirement = desc.major_b == dg::UMMA::Major::MN ? swizzle_b_requirement : 8;
                            if ((block_n / cluster_m) % load_block_n_requirement != 0)
                                continue;

                            // Shape must be divisible for multicast
                            if (ceil_div(desc.n, block_n) % cluster_n != 0)
                                continue;

                            // SwapAB requires block N is layout A/D' UMMA M
                            constexpr int layout_ad_m = 128;
                            if (swap_ab and block_n != layout_ad_m)
                                continue;

                            // Check tensor memory capacity
                            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, desc.get_mma_kind());
                            const auto sf_block_k = block_k / 128;
                            const auto tmem_sf_cols = desc.get_mma_kind() == MmaKind::BF16 ? 0 : sf_block_m * sf_block_k / 32 + sf_block_n * sf_block_k / 32;
                            const auto umma_n = swap_ab ? block_m : block_n;
                            if (umma_n + tmem_sf_cols > 512)
                                continue;

                            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};

                            // When neither A nor B is MN major, 128B swizzle is always feasible
                            if (desc.major_a == dg::UMMA::Major::K or desc.major_b == dg::UMMA::Major::K) {
                                const auto storage_config = get_storage_config(desc, layout);
                                if (storage_config.swizzle_a_mode != 128 or storage_config.swizzle_b_mode != 128)
                                    continue;
                            }

                            candidates.push_back(layout);
                        }
                    }
                }
            }
        }

        // Dynamic FP8 output requires the physical store atom to own complete per-32 SF groups
        if (desc.cd_dtype == torch::kFloat8_e4m3fn) {
            const auto owns_partial_sf_groups = [&](const Layout& layout) {
                const auto store_block_n = layout.swap_ab ? layout.block_n :
                    get_storage_config(desc, layout).swizzle_cd_mode;
                return store_block_n % 32 != 0;
            };
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(), owns_partial_sf_groups),
                             candidates.end());
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int layout_ad_m = 128;
        constexpr int umma_step_n = 16;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        const auto load_block_m = layout.block_m / layout.cluster_n;
        const auto load_block_n = layout.block_n / layout.cluster_m;
        const auto store_block_m = layout.swap_ab ? umma_step_n : std::min(layout_ad_m, layout.block_m);
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        const int pack_factor = desc.get_smem_pack_factor();
        const int swizzle_mode_a = get_swizzle_mode(
            (desc.major_a == dg::UMMA::Major::K ? layout.block_k : load_block_m) / pack_factor,
            c10::elementSize(desc.a_dtype));
        const int swizzle_mode_b = get_swizzle_mode(
            (desc.major_b == dg::UMMA::Major::K ? layout.block_k : load_block_n) / pack_factor,
            c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_cd = get_swizzle_mode(
            store_block_n, c10::elementSize(desc.cd_dtype));

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static int get_num_tma_store_stages(const GemmDesc& desc, const Layout& layout) {
        // With single-stage TMA stores, each SM keeps at most one bulk store in
        // flight. This paces the store traffic and improves the achieved DRAM
        // throughput when C/D flushes dominate DRAM traffic; it also frees up
        // shared memory for the A/B mainloop and halves the C/D smem read/write
        // footprint. Only enabled for k-grouped GEMMs.
        if (desc.get_mma_kind() == MmaKind::BF16 or layout.swap_ab or
            not is_k_grouped_contiguous(desc.gemm_type))
            return 2;

        const int num_k_blocks_per_group = desc.k / std::max(desc.num_groups, 1) / layout.block_k;
        const int min_k_blocks = desc.cd_dtype != torch::kFloat ? 16 :
                                 desc.with_accumulation ? 24 : 32;
        return num_k_blocks_per_group >= min_k_blocks ? 1 : 2;
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 32;

        // C/D for TMA stores
        const int num_tma_store_stages = get_num_tma_store_stages(desc, layout);
        const int smem_cd = (layout.swap_ab ? storage_config.store_block_m * storage_config.store_block_n * c10::elementSize(desc.cd_dtype)
                                            : storage_config.store_block_m * storage_config.swizzle_cd_mode) * num_tma_store_stages;

        // TODO: remove SF barriers for BF16 GEMMs
        // A/B-and-SF-transpose full, SF-TMA full, empty, and tensor memory full/empty/overlap barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        // NOTES: the last barrier is for tensor core utilization control
        const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 3 + 8;

        // Tensor memory pointer
        const int smem_tmem_ptr = 4;

        // Calculate A/B per stages
        const int pack_factor = desc.get_smem_pack_factor();
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype) / pack_factor;
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype) / pack_factor;

        // Calculate SF A/B per stages
        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(
                layout.block_m, layout.block_n, desc.get_mma_kind());
            smem_sfa_per_stage = sf_block_m * layout.block_k / 32;
            smem_sfb_per_stage = sf_block_n * layout.block_k / 32;
        }

        // Calculate stages
        int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
        int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages,
            num_tma_store_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            256,
            32, 128, 128, 128
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;
        // TODO: calculate expected cycles
        return {num_waves, last_wave_util, 0, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Single wave is always better
        if ((a.num_waves == 1 or b.num_waves == 1) and a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Doing multicast is better
        if (a.layout.get_cluster_size() != b.layout.get_cluster_size())
            return a.layout.get_cluster_size() > b.layout.get_cluster_size();

        // Smaller number of waves is better
        if (a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Larger last wave utilization is better
        if (a.last_wave_util != b.last_wave_util)
            return a.last_wave_util > b.last_wave_util;

        // More stages is better
        // Same block M, smaller block N is better
        // Same block N, smaller block M is better
        if (a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n)
            return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;

        // Less shared memory C/D, more stages is better
        return a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n;
    }
};

} // namespace deep_gemm

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8");
    return MmaKind::MXFP8FP4;
}

static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4;
}

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind) {
    // Expected tokens per expert, plus a one-sigma routing margin for the tile choice
    const float num_expected_tokens = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

    // Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
    // per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
    // A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
    int block_m = num_expected_tokens <= 10 ? 16 : 32;
    if (num_expected_tokens > 24) {
        int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
        if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
            num_blocks = 2;
        for (const int& candidate: {64, 128, 192, 240}) {
            block_m = candidate;
            if (num_blocks * candidate >= num_covered_tokens)
                break;
        }
    }
    const int store_block_m = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 : 40;
    const int block_k = 128 / get_num_mma_elem_bytes(mma_kind);

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs: 2-CTA clusters and 2 epilogue warpgroups
    return {2, block_m, store_block_m, block_k, 2 * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages * get_num_mma_elem_bytes(mma_kind);
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    const int smem_a_size_per_stage = load_block_m * block_k * num_mma_elem_bytes;
    const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bytes;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {

    // Block config
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);
    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4) : std::pair(0, 0);
    // NOTES: FP8 activations, FP8 weights, and FP4 weights unpacked to 8-bit in smem all use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 32;

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_num_mma_elem_bytes(mma_kind);
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm

// csrc/jit_kernels/impls/runtime_utils.hpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <utility>

#include <cuda.h>
#include <torch/python.h>

// csrc/jit_kernels/impls/tma_descriptor_cache.hpp
#include <array>
#include <cstdint>
#include <optional>
#include <cuda.h>

namespace deep_gemm {
// This scope covers only this host thread's up to ten descriptor constructions.
// No callback, context switch, compilation or launch occurs inside the scope.
// Every new invocation queries the driver anew. Nested scopes and exceptions
// restore the caller's scope; the context is never cached across invocations.
class TmaContextScope final {
    inline static thread_local TmaContextScope* active_ = nullptr;
    TmaContextScope* previous_;
    CUcontext context_{};
    CUresult status_;
    bool open_ = true;
public:
    TmaContextScope() : previous_(active_),
        status_(deep_jit::cuda::driver::lazy_cuCtxGetCurrent(&context_)) {
        active_ = this;
    }
    TmaContextScope(const TmaContextScope&) = delete;
    TmaContextScope& operator=(const TmaContextScope&) = delete;
    void close() noexcept {
        if (open_) {
            active_ = previous_;
            open_ = false;
        }
    }
    ~TmaContextScope() { close(); }
    static CUresult current(CUcontext* context) {
        if (!active_) return deep_jit::cuda::driver::lazy_cuCtxGetCurrent(context);
        *context = active_->context_;
        return active_->status_;
    }
};

// A tensor map encodes addresses and geometry, never tensor contents. Each
// original validation and DG_PRINT_CONFIGS call remains ahead of this helper.
// All driver arguments plus device and current context are
// compared. A changed allocation/address or layout misses; overwriting values
// at an unchanged address needs no invalidation. Entries are host-thread local,
// bounded, and contain neither Tensor owners nor kernel/launch state.
template <unsigned Rank>
static CUresult encode_tma_cached(
        CUtensorMap* output, CUtensorMapDataType dtype, unsigned rank,
        void* address, const cuuint64_t* global_dims,
        const cuuint64_t* global_strides, const cuuint32_t* box_dims,
        const cuuint32_t* element_strides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promotion,
        CUtensorMapFloatOOBfill oob, int device) {
    static_assert(Rank == 2 || Rank == 3);
    if (rank != Rank) return CUDA_ERROR_INVALID_VALUE;
    CUcontext context{};
    const auto context_status = TmaContextScope::current(&context);
    if (context_status != CUDA_SUCCESS) return context_status;
    struct Key {
        CUcontext context{};
        int device{};
        void* address{};
        CUtensorMapDataType dtype{};
        std::array<cuuint64_t, Rank> global_dims{};
        std::array<cuuint64_t, Rank - 1> global_strides{};
        std::array<cuuint32_t, Rank> box_dims{};
        std::array<cuuint32_t, Rank> element_strides{};
        CUtensorMapInterleave interleave{};
        CUtensorMapSwizzle swizzle{};
        CUtensorMapL2promotion promotion{};
        CUtensorMapFloatOOBfill oob{};
        bool operator==(const Key&) const = default;
    };
    Key key{};
    key.context = context;
    key.device = device;
    key.address = address;
    key.dtype = dtype;
    for (unsigned i = 0; i < Rank; ++i) {
        key.global_dims[i] = global_dims[i];
        key.box_dims[i] = box_dims[i];
        key.element_strides[i] = element_strides[i];
        if (i + 1 < Rank) key.global_strides[i] = global_strides[i];
    }
    key.interleave = interleave;
    key.swizzle = swizzle;
    key.promotion = promotion;
    key.oob = oob;
    struct Entry { std::optional<Key> key; CUtensorMap map; };
    thread_local std::array<Entry, 16> entries;
    thread_local unsigned next = 0;
    for (const auto& entry : entries) {
        if (entry.key && *entry.key == key) {
            *output = entry.map;
            return CUDA_SUCCESS;
        }
    }
    const auto status = deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        output, dtype, Rank, address, global_dims, global_strides, box_dims,
        element_strides, interleave, swizzle, promotion, oob);
    if (status == CUDA_SUCCESS) {
        auto& entry = entries[next];
        entry.map = *output;
        entry.key = key;
        next = (next + 1) % entries.size();
    }
    return status;
}
}  // namespace deep_gemm

namespace deep_gemm {

static std::pair<int, int> get_inner_outer_dims(const dg::UMMA::Major& major, const int& k, const int& mn) {
    return major == dg::UMMA::Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

static int get_non_contiguous_dim(const dg::UMMA::Major& major) {
    return major == dg::UMMA::Major::K ? -2 : -1;
}

static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    if (heuristics_runtime->get_ignore_compile_dims())
        return 0;

    for (const char& c: compiled_dims) {
        if (name == c)
            return dim;
    }
    return 0;
}

static std::string to_string(const dg::UMMA::Major& major) {
    switch (major) {
        case dg::UMMA::Major::K:  return "dg::UMMA::Major::K";
        case dg::UMMA::Major::MN: return "dg::UMMA::Major::MN";
    }
    DG_HOST_UNREACHABLE("Unknown major");
}

static std::string to_string(const GemmType& type) {
    switch (type) {
        case GemmType::Normal:                              return "GemmType::Normal";
        case GemmType::MGroupedContiguous:                  return "GemmType::MGroupedContiguous";
        case GemmType::MGroupedMasked:                      return "GemmType::MGroupedMasked";
        case GemmType::MGroupedContiguousWithPsumLayout:    return "GemmType::MGroupedContiguousWithPsumLayout";
        case GemmType::KGroupedContiguous:                  return "GemmType::KGroupedContiguous";
        case GemmType::KGroupedContiguousWithPsumLayout:    return "GemmType::KGroupedContiguousWithPsumLayout";
        case GemmType::Batched:                             return "GemmType::Batched";
    }
    DG_HOST_UNREACHABLE("Unknown GEMM type");
}

static std::string to_string(const at::ScalarType& dtype, const bool& fp4_unpacked_smem = true) {
    switch (dtype) {
        case torch::kInt:           return "int";
        case torch::kFloat:         return "float";
        case torch::kBFloat16:      return "dg::bfloat16_t";
        case torch::kFloat8_e4m3fn: return "dg::float_e4m3_t";
        case kPackedFP4:            return fp4_unpacked_smem ? "dg::detail::float_e2m1_unpacksmem_t"
                                                             : "dg::float_e2m1_t";
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static std::string to_string(const float& v) {
    if (std::isfinite(v)) {
        return std::format("{}0x{:a}f", std::signbit(v) ? "-" : "", std::abs(v));
    } else if (std::isinf(v)) {
        return v > 0 ? "dg::numeric_limits<float>::infinity()"
                     : "-dg::numeric_limits<float>::infinity()";
    }
    DG_HOST_UNREACHABLE("NaN input is not supported");
}

static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype,
                                                          const bool& allow_tf32,
                                                          const bool& fp4_unpacked_smem) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;

    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case kPackedFP4:            return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                                             : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode, const int& base) {
    if (base != 0) {
        DG_HOST_ASSERT(base == 32 and mode == 128);
        return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }

    DG_HOST_ASSERT(base == 0);
    switch (mode) {
        case   0:
        case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: DG_HOST_UNREACHABLE("Unsupported swizzling mode");
    }
}

static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim,
                                    const int& gmem_outer_stride,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    const auto num_gmem_outer_stride_bytes = static_cast<uint32_t>(gmem_outer_stride) * static_cast<uint32_t>(elem_size);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_outer_stride_bytes % 16u == 0u);
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);

        // Fix FP4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_inner_dim = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d (base: %d), elem size: %d, pointer: %llu\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride, swizzle_mode, swizzle_base, elem_size,
               reinterpret_cast<unsigned long long>(t.data_ptr()));
    }
    DJ_CUDA_DRIVER_CHECK(encode_tma_cached<2>(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, t.get_device()));
    return tensor_map;
}

static CUtensorMap make_tma_3d_desc(const torch::Tensor& t,
                                    int gmem_dim_0, int gmem_dim_1, int gmem_dim_2,
                                    int smem_dim_0, int smem_dim_1, int smem_dim_2,
                                    const int& gmem_stride_0, const int& gmem_stride_1,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    const auto num_gmem_stride_0_bytes = static_cast<uint32_t>(gmem_stride_0) * static_cast<uint32_t>(elem_size);
    const auto num_gmem_stride_1_bytes = static_cast<uint32_t>(gmem_stride_1) * static_cast<uint32_t>(elem_size);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_stride_0_bytes % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_stride_1_bytes % 16u == 0u);
    if (swizzle_mode != 0)
        smem_dim_0 = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_dim_0 % 128 == 0);

        // Fix fp4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_dim_0 = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1), static_cast<cuuint64_t>(gmem_dim_2),};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1), static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_0 * elem_size), static_cast<cuuint64_t>(gmem_stride_1 * elem_size)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Making 3D TMA desc: global memory: %d %d %d, shared memory: %d %d %d, outer stride: %d %d, swizzle: %d, elem size: %d\n",
               gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2,
               gmem_stride_0, gmem_stride_1, swizzle_mode, elem_size);
    }
    DJ_CUDA_DRIVER_CHECK(encode_tma_cached<3>(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        3, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, t.get_device()));
    return tensor_map;
}

static CUtensorMap make_tma_a_desc(const dg::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false,
                                   const bool& fp4_unpacked_smem = true) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == dg::UMMA::Major::K);
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32, fp4_unpacked_smem);
}

static CUtensorMap make_tma_b_desc(const dg::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_n, const int& shape_k,
                                   const int& block_n, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false,
                                   const bool& fp4_unpacked_smem = true) {
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);

    // `num_groups` is always applied into the outer dimensions
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim * num_groups,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32, fp4_unpacked_smem);
}

static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
                                    const int& shape_m, const int& shape_n,
                                    const int& block_m, const int& block_n,
                                    const int& outer_stride,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    // Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    // bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, block_m,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_sf_desc(const dg::UMMA::Major& major,
                                    const torch::Tensor& t,
                                    int shape_mn, int shape_k,
                                    const int& block_mn, const int& gran_k,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const int& smem_outer_dim = 1) {
    DG_HOST_ASSERT(major == dg::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, gran_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                            block_mn, smem_outer_dim,
                            shape_mn,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

} // namespace deep_gemm

namespace deep_gemm {

static void sm100_bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const float& activation_alpha,
    const float& activation_beta,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);

    // Make tensormap
    TmaContextScope tensor_map_context;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();

    tensor_map_context.close();  // before resolver, callback or launch

    // Compile
    const auto kernel = precompiled::load_cached("sm100_bf16_mega_moe",  num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        to_string(activation_alpha),
        to_string(activation_beta),
        fast_math ? "true" : "false");

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_weights,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_weights,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights);
}

} // namespace deep_gemm
