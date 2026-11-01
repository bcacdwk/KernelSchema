#include "dispatch.cuh"

// csrc/apis/gemm.hpp
// G08 operator closure: retained entry and registration are verbatim.

namespace deep_gemm::gemm {

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding,
                                              const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == dg::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_UNREACHABLE("Unsupported architecture: the sm90/sm120 host launcher surface is not shipped in this bundle");
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout);
    } else if (arch_major == 12) {
        DG_HOST_UNREACHABLE("Unsupported architecture: the sm90/sm120 host launcher surface is not shipped in this bundle");
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt);
}

} // namespace deep_gemm::gemm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_config_resolver", [](deep_gemm::precompiled::Store::Resolver resolver) {
        deep_gemm::precompiled::store().set_resolver(std::move(resolver));
    });
    deep_gemm::config::register_apis(m);
    deep_gemm::gemm::register_apis(m);
}
