#include "dispatch.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace deep_gemm::gemm {

// Copy exactly the logical C region; output padding is untouched.
// This non-template entry only copies bits and performs no dtype conversion.
__global__ void atrex_dense_copy_cd_16b(
        const uint4* source, uint4* destination, size_t vectors,
        size_t width_vectors, size_t source_pitch, size_t destination_pitch) {
    const size_t step = size_t(blockDim.x) * gridDim.x;
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (source_pitch == width_vectors and destination_pitch == width_vectors) {
        for (size_t i = first; i < vectors; i += step)
            destination[i] = source[i];
    } else {
        for (size_t i = first; i < vectors; i += step) {
            const size_t row = i / width_vectors;
            const size_t column = i % width_vectors;
            destination[row * destination_pitch + column] = source[row * source_pitch + column];
        }
    }
}

static bool early_return(const int& m, const int& n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    if (m == 0 or n == 0)
        return true;
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    DG_HOST_ASSERT(d.dim() == 2 and d.stride(-1) == 1);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == c->scalar_type());
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->device() == d.device());
        if (is_cd_same)
            DG_HOST_ASSERT(c->strides() == d.strides());
    }

    // Same-device N-major matrices may have different padded row pitches.
    // Copy bytes on the caller's current stream before GEMM reads its C/D.
    // The frozen input generator always supplies matching C/D shapes/types.
    if (not is_cd_same and (c.has_value() or k == 0)) {
        const auto width = static_cast<size_t>(n) * d.element_size();
        const auto destination_pitch = static_cast<size_t>(d.stride(0)) * d.element_size();
        const auto stream = at::cuda::getCurrentCUDAStream(d.get_device()).stream();
        cudaError_t status;
        if (c.has_value()) {
            const auto source_pitch = static_cast<size_t>(c->stride(0)) * c->element_size();
            const auto alignment = reinterpret_cast<std::uintptr_t>(c->data_ptr()) |
                                   reinterpret_cast<std::uintptr_t>(d.data_ptr()) |
                                   source_pitch | destination_pitch | width;
            if ((alignment & 15u) == 0u) {
                const auto vectors = size_t(m) * (width / sizeof(uint4));
                const auto blocks_needed = (vectors + 255) / 256;
                const auto blocks = static_cast<unsigned>(blocks_needed < 1024 ? blocks_needed : 1024);
                atrex_dense_copy_cd_16b<<<blocks, 256, 0, stream>>>(
                    static_cast<const uint4*>(c->data_ptr()), static_cast<uint4*>(d.data_ptr()),
                    vectors, width / sizeof(uint4), source_pitch / sizeof(uint4),
                    destination_pitch / sizeof(uint4));
                status = cudaGetLastError();
            } else {
                status = cudaMemcpy2DAsync(d.data_ptr(), destination_pitch, c->data_ptr(),
                                          source_pitch, width, m, cudaMemcpyDeviceToDevice, stream);
            }
        } else {
            status = cudaMemset2DAsync(d.data_ptr(), destination_pitch, 0, width, m, stream);
        }
        DG_HOST_ASSERT(status == cudaSuccess);
    }
    return k == 0;
}

static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims,
                         const std::optional<float>& alpha) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Dispatch into different implements

    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_UNREACHABLE("Unsupported architecture: the sm90/sm120 host launcher surface is not shipped in this bundle");
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims, alpha);
    } else if (arch_major == 12) {
        DG_HOST_UNREACHABLE("Unsupported architecture: the sm90/sm120 host launcher surface is not shipped in this bundle");
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("bf16_gemm_nt", &bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("alpha") = std::nullopt);
}

} // namespace deep_gemm::gemm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Independent BF16 GEMM: tuned SM103 pipeline and dynamic configurations";
    m.def("set_config_resolver", [](deep_gemm::precompiled::Store::Resolver resolver) {
        deep_gemm::precompiled::store().set_resolver(std::move(resolver));
    });
    deep_gemm::config::register_apis(m);
    deep_gemm::gemm::register_apis(m);
}
