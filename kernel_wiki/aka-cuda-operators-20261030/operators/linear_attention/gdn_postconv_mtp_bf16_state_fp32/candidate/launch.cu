// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Tensor validation, current-stream launch and typed dtype/head-ratio/gate dispatch.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <string>

#include "kernel.cuh"

using namespace gdn_post;

namespace {

template<typename StateT, int HeadRatio, bool SigmoidGate>
void launch_post_entry(const GdnPostConvParams& params, dim3 grid, cudaStream_t stream) {
  gdn_post_conv_mtp<StateT, HeadRatio, SigmoidGate> <<<grid, kThreads, 0, stream>>>(
          params.mixed_qkv,
          params.a,
          params.b,
          params.a_log,
          params.dt_bias,
          params.state_indices,
          params.cu_seqlens,
          params.num_accepted_tokens,
          static_cast<StateT*>(params.state),
          params.output_gate,
          params.norm_weight,
          params.out,
          params.H,
          params.HV,
          params.state_indices_width,
          params.dt_bias_type,
          params.norm_weight_is_bf16,
          params.scale,
          params.norm_eps,
          params.strides);
}

template<typename StateT, int HeadRatio>
void launch_post_gate(const GdnPostConvParams& params, dim3 grid, bool sigmoid, cudaStream_t stream) {
  if (sigmoid) launch_post_entry<StateT, HeadRatio, true>(params, grid, stream);
  else launch_post_entry<StateT, HeadRatio, false>(params, grid, stream);
}

template<typename StateT>
void launch_post_ratio(const GdnPostConvParams& params, dim3 grid, int ratio,
                       bool sigmoid, cudaStream_t stream) {
  switch (ratio) {
    case 1: launch_post_gate<StateT, 1>(params, grid, sigmoid, stream); break;
    case 2: launch_post_gate<StateT, 2>(params, grid, sigmoid, stream); break;
    case 3: launch_post_gate<StateT, 3>(params, grid, sigmoid, stream); break;
    case 4: launch_post_gate<StateT, 4>(params, grid, sigmoid, stream); break;
    case 8: launch_post_gate<StateT, 8>(params, grid, sigmoid, stream); break;
    default: TORCH_CHECK(false, "GDN decode MTP post-conv: no kernel entry for HV/H = ", ratio);
  }
}

void launch_gdn_post_conv_mtp(const GdnPostConvParams& params, int num_requests,
                              int num_value_heads, int value_heads_per_key_head,
                              bool state_is_f32, bool sigmoid_gate, cudaStream_t stream) {
  const dim3 grid(num_value_heads, num_requests);
  if (state_is_f32)
    launch_post_ratio<float>(params, grid, value_heads_per_key_head, sigmoid_gate, stream);
  else
    launch_post_ratio<__nv_bfloat16>(params, grid, value_heads_per_key_head, sigmoid_gate, stream);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "GDN decode MTP post-conv kernel launch failed: ",
              cudaGetErrorString(error));
}

}  // namespace

void fused_gdn_decode_post_conv_mtp(
    at::Tensor const& mixed_qkv, at::Tensor const& a,
    at::Tensor const& b, at::Tensor const& a_log,
    at::Tensor const& dt_bias,
    at::Tensor const& state_indices,
    at::Tensor const& cu_seqlens,
    at::Tensor const& num_accepted_tokens,
    at::Tensor& state, at::Tensor const& output_gate,
    at::Tensor const& norm_weight, at::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation) {
  using at::ScalarType;

  TORCH_CHECK(
      mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == ScalarType::BFloat16,
      "mixed_qkv must be a CUDA bfloat16 tensor");
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == ScalarType::BFloat16,
                  "a must be a CUDA bfloat16 tensor");
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == ScalarType::BFloat16,
                  "b must be a CUDA bfloat16 tensor");
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == ScalarType::Float,
                  "A_log must be a CUDA float32 tensor");
  const auto dt_bias_scalar_type = dt_bias.scalar_type();
  TORCH_CHECK(
      dt_bias.is_cuda() && (dt_bias_scalar_type == ScalarType::Float ||
                            dt_bias_scalar_type == ScalarType::BFloat16 ||
                            dt_bias_scalar_type == ScalarType::Half),
      "dt_bias must be a CUDA float32, bfloat16, or float16 tensor");
  TORCH_CHECK(
      state_indices.is_cuda() && state_indices.scalar_type() == ScalarType::Int,
      "state_indices must be a CUDA int32 tensor");
  TORCH_CHECK(
      cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == ScalarType::Int,
      "cu_seqlens must be a CUDA int32 tensor");
  TORCH_CHECK(num_accepted_tokens.is_cuda() &&
                      num_accepted_tokens.scalar_type() == ScalarType::Int,
                  "num_accepted_tokens must be a CUDA int32 tensor");
  const auto state_scalar_type = state.scalar_type();
  TORCH_CHECK(
      state.is_cuda() && (state_scalar_type == ScalarType::Float ||
                          state_scalar_type == ScalarType::BFloat16),
      "state must be a CUDA float32 or bfloat16 tensor");
  TORCH_CHECK(output_gate.is_cuda() &&
                      output_gate.scalar_type() == ScalarType::BFloat16,
                  "output_gate must be a CUDA bfloat16 tensor");
  TORCH_CHECK(norm_weight.is_cuda() &&
                      (norm_weight.scalar_type() == ScalarType::Float ||
                       norm_weight.scalar_type() == ScalarType::BFloat16),
                  "norm_weight must be a CUDA float32 or bfloat16 tensor");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == ScalarType::BFloat16,
                  "out must be a CUDA bfloat16 tensor");
  TORCH_CHECK(
      output_gate_activation == "silu" || output_gate_activation == "sigmoid",
      "output_gate_activation must be 'silu' or 'sigmoid'");

  TORCH_CHECK(mixed_qkv.dim() == 2,
                  "mixed_qkv must have shape [L, 2 * H * 128 + HV * 128]");
  const int num_tokens = static_cast<int>(mixed_qkv.size(0));
  TORCH_CHECK(num_tokens > 0,
                  "GDN decode MTP fusion requires at least one token");
  TORCH_CHECK(
      state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK,
      "state must have shape [slots, HV, 128, 128]");
  const int num_value_heads = static_cast<int>(state.size(1));
  const int64_t key_width =
      mixed_qkv.size(1) - static_cast<int64_t>(num_value_heads) * kDimV;
  TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
                  "mixed_qkv width is inconsistent with state");
  const int num_key_heads = static_cast<int>(key_width / (2 * kDimK));
  const int value_heads_per_key_head = num_value_heads / num_key_heads;
  TORCH_CHECK(
      num_value_heads % num_key_heads == 0 &&
          ((value_heads_per_key_head >= 1 && value_heads_per_key_head <= 4) ||
           value_heads_per_key_head == 8),
      "GDN decode MTP fusion requires HV/H in {1, 2, 3, 4, 8}");

  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) > 0 &&
                      state_indices.size(1) > 0 &&
                      state_indices.size(1) <= kMaxMtpTokens,
                  "state_indices must have shape [N, S] with 1 <= S <= 8");
  const int num_requests = static_cast<int>(state_indices.size(0));
  TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() == num_requests + 1,
      "cu_seqlens must have N + 1 elements");
  TORCH_CHECK(num_accepted_tokens.dim() == 1 &&
                      num_accepted_tokens.numel() == num_requests,
                  "num_accepted_tokens must have N elements");
  TORCH_CHECK(
      a.dim() == 2 && a.size(0) == num_tokens && a.size(1) == num_value_heads,
      "a must have shape [L, HV]");
  TORCH_CHECK(
      b.dim() == 2 && b.size(0) == num_tokens && b.size(1) == num_value_heads,
      "b must have shape [L, HV]");
  TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == num_value_heads,
                  "A_log must be contiguous with HV elements");
  TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == num_value_heads,
                  "dt_bias must be contiguous with HV elements");
  TORCH_CHECK(state_indices.is_contiguous(),
                  "state_indices must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  TORCH_CHECK(num_accepted_tokens.is_contiguous(),
                  "num_accepted_tokens must be contiguous");
  TORCH_CHECK(output_gate.dim() == 3 && output_gate.size(0) == num_tokens &&
                      output_gate.size(1) == num_value_heads &&
                      output_gate.size(2) == kDimV,
                  "output_gate must have shape [L, HV, 128]");
  TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV,
                  "norm_weight must be contiguous with 128 elements");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == num_tokens &&
                      out.size(1) == num_value_heads && out.size(2) == kDimV,
                  "out must have shape [L, HV, 128]");
  TORCH_CHECK(mixed_qkv.stride(1) == 1,
                  "mixed_qkv channels must be contiguous");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
                  "a and b heads must be contiguous");
  TORCH_CHECK(state.stride(0) >= num_value_heads * kDimV * kDimK &&
                      state.stride(1) == kDimV * kDimK &&
                      state.stride(2) == kDimK && state.stride(3) == 1,
                  "state must have contiguous [HV, 128, 128] slot contents");
  const int state_elements_per_copy =
      state_scalar_type == ScalarType::Float ? 4 : 8;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                      state.stride(0) % state_elements_per_copy == 0,
                  "state slots must preserve 16-byte alignment");
  TORCH_CHECK(output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV,
                  "output_gate head rows must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");

  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                 output_gate.stride(0), state.stride(0)};
  const auto* a_ptr = static_cast<const __nv_bfloat16*>(a.data_ptr());
  const auto* b_ptr = static_cast<const __nv_bfloat16*>(b.data_ptr());
  const auto* output_gate_ptr =
      static_cast<const __nv_bfloat16*>(output_gate.data_ptr());

  // dt_bias_scalar_type comes from the validation block copied above.
  const int dt_bias_type =
      dt_bias_scalar_type == at::ScalarType::Float
          ? gdn_post::kDtBiasFloat32
          : (dt_bias_scalar_type == at::ScalarType::BFloat16
                 ? gdn_post::kDtBiasBFloat16
                 : gdn_post::kDtBiasFloat16);

  c10::cuda::CUDAGuard const device_guard(mixed_qkv.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(mixed_qkv.get_device());

  GdnPostConvParams params{};
  params.mixed_qkv = static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr());
  params.a = a_ptr;
  params.b = b_ptr;
  params.a_log = static_cast<const float*>(a_log.data_ptr());
  params.dt_bias = dt_bias.data_ptr();
  params.state_indices = static_cast<const int*>(state_indices.data_ptr());
  params.cu_seqlens = static_cast<const int*>(cu_seqlens.data_ptr());
  params.num_accepted_tokens =
      static_cast<const int*>(num_accepted_tokens.data_ptr());
  params.state = state.data_ptr();
  params.output_gate = output_gate_ptr;
  params.norm_weight = norm_weight.data_ptr();
  params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  params.H = num_key_heads;
  params.HV = num_value_heads;
  params.state_indices_width = static_cast<int>(state_indices.size(1));
  params.dt_bias_type = dt_bias_type;
  params.norm_weight_is_bf16 =
      norm_weight.scalar_type() == at::ScalarType::BFloat16;
  params.scale = static_cast<float>(scale);
  params.norm_eps = static_cast<float>(norm_eps);
  params.strides = strides;

  launch_gdn_post_conv_mtp(params, num_requests, num_value_heads,
                           value_heads_per_key_head,
                           state_scalar_type == at::ScalarType::Float,
                           output_gate_activation == "sigmoid", stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gdn_decode_post_conv_mtp", &fused_gdn_decode_post_conv_mtp,
        "GDN decode/MTP post-convolution fusion (pure CUDA); mutates `state` checkpoints "
        "in place and returns the caller-owned `out` buffer",
        pybind11::arg("mixed_qkv"), pybind11::arg("a"), pybind11::arg("b"),
        pybind11::arg("a_log"), pybind11::arg("dt_bias"),
        pybind11::arg("state_indices"), pybind11::arg("cu_seqlens"),
        pybind11::arg("num_accepted_tokens"), pybind11::arg("state"),
        pybind11::arg("output_gate"), pybind11::arg("norm_weight"),
        pybind11::arg("out"), pybind11::arg("scale"), pybind11::arg("norm_eps"),
        pybind11::arg("output_gate_activation"));
}
