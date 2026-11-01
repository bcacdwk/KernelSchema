// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Convolution metadata validation, paired-feature route and generic fallback.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <optional>
#include <string>

#include "conv_kernel.cuh"

namespace sfc_conv_pure {
namespace {

int64_t next_power_of_2(int64_t value) {          // triton.next_power_of_2
  int64_t result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

struct LaunchArgs {
  const void* x = nullptr;
  const void* w = nullptr;
  const void* bias = nullptr;
  void* conv_state = nullptr;
  const int32_t* conv_state_indices = nullptr;
  const int32_t* num_accepted_tokens = nullptr;
  const int32_t* query_start_loc = nullptr;
  const int32_t* block_idx_last_scheduled_token = nullptr;
  const int32_t* initial_state_idx = nullptr;
  void* o = nullptr;
  int batch = 0;
  int dim = 0;
  int seqlen = 0;
  int state_len = 0;
  int num_cache_lines = 0;
  int64_t stride_x_seq = 0;
  int64_t stride_x_dim = 0;
  int64_t stride_x_token = 0;
  int64_t stride_w_dim = 0;
  int64_t stride_w_width = 0;
  int64_t stride_conv_state_seq = 0;
  int64_t stride_conv_state_dim = 0;
  int64_t stride_conv_state_tok = 0;
  int64_t stride_state_indices = 0;
  int64_t stride_o_seq = 0;
  int64_t stride_o_dim = 0;
  int64_t stride_o_token = 0;
  int64_t null_block_id = 0;
  bool has_bias = false;
  bool silu_activation = false;
  bool is_varlen = false;
  bool is_apc_enabled = false;
  bool is_spec_decoding = false;
  bool has_null_block = false;
  bool launch_pdl = false;
  dim3 grid{0, 0, 0};
  cudaStream_t stream = nullptr;
};

// LaunchArgs is taken by value: the launch APIs want `void**` to non-const argument
// storage, and a local copy is the honest way to provide it.
using ConvKernelFn = void (*)(ConvParams);

void launch_entry(ConvKernelFn kernel, LaunchArgs a, int block_threads = kBlockN) {
  ConvParams params{};
  params.x_ptr = a.x;
  params.w_ptr = a.w;
  params.bias_ptr = a.bias;
  params.conv_state_ptr = a.conv_state;
  params.conv_state_indices_ptr = a.conv_state_indices;
  params.num_accepted_tokens_ptr = a.num_accepted_tokens;
  params.query_start_loc_ptr = a.query_start_loc;
  params.block_idx_last_scheduled_token = a.block_idx_last_scheduled_token;
  params.initial_state_idx = a.initial_state_idx;
  params.o_ptr = a.o;
  params.batch = a.batch;
  params.dim = a.dim;
  params.seqlen_in = a.seqlen;
  params.state_len_in = a.state_len;
  params.num_cache_lines = a.num_cache_lines;
  params.stride_x_seq = a.stride_x_seq;
  params.stride_x_dim = a.stride_x_dim;
  params.stride_x_token = a.stride_x_token;
  params.stride_w_dim = a.stride_w_dim;
  params.stride_w_width = a.stride_w_width;
  params.stride_conv_state_seq = a.stride_conv_state_seq;
  params.stride_conv_state_dim = a.stride_conv_state_dim;
  params.stride_conv_state_tok = a.stride_conv_state_tok;
  params.stride_state_indices = a.stride_state_indices;
  params.stride_o_seq = a.stride_o_seq;
  params.stride_o_dim = a.stride_o_dim;
  params.stride_o_token = a.stride_o_token;
  params.null_block_id = a.null_block_id;
  params.has_bias = a.has_bias;
  params.silu_activation = a.silu_activation;
  params.is_varlen = a.is_varlen;
  params.is_apc_enabled = a.is_apc_enabled;
  params.is_spec_decoding = a.is_spec_decoding;
  params.has_null_block = a.has_null_block;
  params.launch_pdl = a.launch_pdl;
  void* launch_params[] = {&params};
  if (a.launch_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = a.grid;
    config.blockDim = dim3(block_threads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = a.stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelExC(
        &config, reinterpret_cast<const void*>(kernel), launch_params);
    TORCH_CHECK(err == cudaSuccess, "sfc conv pure: cudaLaunchKernelExC failed: ",
                cudaGetErrorString(err));
    return;
  }
  const cudaError_t err = cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel), a.grid, dim3(block_threads, 1, 1),
      launch_params, 0, a.stream);
  TORCH_CHECK(err == cudaSuccess, "sfc conv pure: cudaLaunchKernel failed: ",
              cudaGetErrorString(err));
}

template<int Width>
bool launch_np2(int np2, const LaunchArgs& a) {
  if constexpr (Width == 4) {
  // Match only semantic metadata; all other requests retain the generic route.
  if (!a.has_bias && a.silu_activation && a.is_varlen && !a.is_apc_enabled &&
      a.is_spec_decoding && a.has_null_block && a.launch_pdl && a.null_block_id == 0 &&
      a.stride_x_dim == 1 && a.stride_w_dim == 4 && a.stride_w_width == 1 &&
      a.stride_conv_state_dim == 1 && a.stride_o_dim == 1 && a.dim % 2 == 0 && a.stride_x_token % 2 == 0 && a.stride_o_token % 2 == 0 && a.stride_conv_state_seq % 2 == 0 && a.stride_conv_state_tok % 2 == 0 && reinterpret_cast<uintptr_t>(a.x) % 4 == 0 && reinterpret_cast<uintptr_t>(a.o) % 4 == 0 && reinterpret_cast<uintptr_t>(a.conv_state) % 4 == 0 && reinterpret_cast<uintptr_t>(a.w) % 8 == 0) {
    switch (np2) {
      case 4: launch_entry(causal_conv_pair<4>, a, 128); return true;
      case 8: launch_entry(causal_conv_pair<8>, a, 128); return true;
      case 16: launch_entry(causal_conv_pair<16>, a, 128); return true;
    }
  }
  }
  switch (np2) {
    case 1: launch_entry(causal_conv_generic<Width, 1>, a); return true;
    case 2: launch_entry(causal_conv_generic<Width, 2>, a); return true;
    case 4: launch_entry(causal_conv_generic<Width, 4>, a); return true;
    case 8: launch_entry(causal_conv_generic<Width, 8>, a); return true;
    case 16: launch_entry(causal_conv_generic<Width, 16>, a); return true;
    default: return false;
  }
}

bool launch_width(int width, int np2, const LaunchArgs& a) {
  switch (width) {
    case 2: return launch_np2<2>(np2, a);
    case 3: return launch_np2<3>(np2, a);
    case 4: return launch_np2<4>(np2, a);
    case 5: return launch_np2<5>(np2, a);
    case 6: return launch_np2<6>(np2, a);
    default: return false;
  }
}

// Only bf16 is instantiated, because bf16 is the only conv_state dtype any of the
// rows in scope uses: GDN05/GDN06 build conv_state as bf16 in the delivery's own
// input.py (the `_state_fp32` in GDN06's name is the *recurrent GDN state*, not
// the conv state).  The Triton original is dtype-generic; shipping an fp32/fp16
// instantiation here would be a code path nothing in scope exercises and nothing
// has verified, so it is refused instead of silently untested.
bool launch_dispatch(const LaunchArgs& a, int width, int np2, torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return launch_width(width, np2, a);
  }
  return false;
}

bool arch_supports_pdl() {                   // conv.py: current_platform.is_arch_support_pdl
  int device = 0;
  cudaGetDevice(&device);
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  return major >= 9;
}

}  // namespace

}  // namespace sfc_conv_pure

// ---------------------------------------------------------------------------
// host wrapper: conv.py `causal_conv1d_update`, same order of operations
// ---------------------------------------------------------------------------

torch::Tensor sfc_causal_conv1d_update(
    torch::Tensor x,
    torch::Tensor conv_state,
    torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::string activation,
    std::optional<torch::Tensor> conv_state_indices,
    std::optional<torch::Tensor> num_accepted_tokens,
    std::optional<torch::Tensor> query_start_loc,
    int64_t max_query_len,
    int64_t null_block_id,
    std::optional<torch::Tensor> block_idx_last_scheduled_token,
    std::optional<torch::Tensor> initial_state_idx,
    bool validate_data,
    std::optional<torch::Tensor> out_opt) {
  using sfc_conv_pure::LaunchArgs;
  namespace sfc = sfc_conv_pure;

  const bool has_null_block = true;             // conv.py: null_block_id is not None
  if (validate_data) {
    TORCH_CHECK(has_null_block, "null_block_id must be provided");
    TORCH_CHECK(x.stride(1) == 1, "x.stride(1) must be 1");
  }
  // activation normalisation: conv.py maps True->"silu", False->None, and
  // asserts anything else is in ["silu","swish"]
  TORCH_CHECK(activation == "" || activation == "silu" || activation == "swish",
              "sfc conv pure: activation must be '', 'silu' or 'swish', got '", activation, "'");
  const bool silu_activation = (activation == "silu" || activation == "swish");

  const torch::ScalarType original_x_dtype = x.scalar_type();
  x = x.to(conv_state.scalar_type());
  torch::Tensor out;
  if (!out_opt.has_value()) {
    out = x;                                    // in-place: out aliases x
  } else {
    out = *out_opt;
    TORCH_CHECK(out.sizes() == x.sizes(), "`out` shape ", out.sizes(),
                " must match `x` shape ", x.sizes(), ".");
    TORCH_CHECK(out.scalar_type() == original_x_dtype && out.device() == x.device(),
                "`out` must have the same dtype and device as the input `x`.");
  }

  const bool unsqueeze = !query_start_loc.has_value() && x.dim() == 2;
  if (unsqueeze) {
    x = x.unsqueeze(-1);                        // (batch, dim, seqlen) with seqlen == 1
    out = out.unsqueeze(-1);
  }

  int64_t batch = 0;
  int64_t dim = 0;
  int64_t seqlen = 0;
  if (!query_start_loc.has_value()) {
    TORCH_CHECK(x.dim() == 3, "sfc conv pure: x must be 3-D without query_start_loc");
    batch = x.size(0);
    dim = x.size(1);
    seqlen = x.size(2);
  } else {
    TORCH_CHECK(conv_state_indices.has_value(), "conv_state_indices is required for varlen");
    batch = conv_state_indices->size(0);
    dim = x.size(1);
    seqlen = max_query_len;
  }
  TORCH_CHECK(weight.dim() == 2, "sfc conv pure: weight must be (dim, width)");
  const int64_t width = weight.size(1);
  TORCH_CHECK(conv_state.dim() == 3, "sfc conv pure: conv_state must be (lines, dim, state_len)");
  const int64_t num_cache_lines = conv_state.size(0);

  if (validate_data) {
    TORCH_CHECK(dim == weight.size(0), "dim must equal weight.size(0)");
    TORCH_CHECK(conv_state.size(2) >= width - 1, "state_len >= width - 1");
    TORCH_CHECK(dim == conv_state.size(1), "dim must equal conv_state.size(1)");
    if (!conv_state_indices.has_value()) {
      TORCH_CHECK(conv_state.size(0) >= batch, "conv_state.size(0) >= batch");
    } else {
      TORCH_CHECK(batch == conv_state_indices->size(0), "conv_state_indices shape mismatch");
    }
    TORCH_CHECK(num_cache_lines >= batch, "num_cache_lines >= batch");
    TORCH_CHECK(weight.stride(1) == 1, "weight.stride(1) must be 1");
  }

  TORCH_CHECK(conv_state.scalar_type() == x.scalar_type() &&
                  conv_state.scalar_type() == out.scalar_type() &&
                  conv_state.scalar_type() == weight.scalar_type() &&
                  (!bias.has_value() || bias->scalar_type() == conv_state.scalar_type()),
              "sfc conv pure: x/out/weight/bias must share conv_state dtype ",
              conv_state.scalar_type(), " (the Triton original specialises per pointer dtype; "
              "mixed-dtype weights are outside the ported domain)");
  TORCH_CHECK(conv_state.scalar_type() == torch::kBFloat16,
              "sfc conv pure: instantiated for bf16 only (the only conv_state dtype "
              "GDN05/GDN06/AKA6 use), got ", conv_state.scalar_type());
  TORCH_CHECK(width >= 2 && width <= 6,
              "sfc conv pure: KERNEL_WIDTH must be in 2..6, got ", width);

  const int64_t stride_w_dim = weight.stride(0);
  const int64_t stride_w_width = weight.stride(1);

  int64_t stride_x_seq = 0;
  int64_t stride_x_dim = 0;
  int64_t stride_x_token = 0;
  int64_t stride_o_seq = 0;
  int64_t stride_o_dim = 0;
  int64_t stride_o_token = 0;
  if (!query_start_loc.has_value()) {
    stride_x_seq = x.stride(0);
    stride_x_dim = x.stride(1);
    stride_x_token = x.stride(2);
    stride_o_seq = out.stride(0);
    stride_o_dim = out.stride(1);
    stride_o_token = out.stride(2);
  } else {
    stride_x_token = x.stride(0);
    stride_x_dim = x.stride(1);
    stride_o_token = out.stride(0);
    stride_o_dim = out.stride(1);
  }

  const int64_t stride_conv_state_seq = conv_state.stride(0);
  const int64_t stride_conv_state_dim = conv_state.stride(1);
  const int64_t stride_conv_state_tok = conv_state.stride(2);
  const int64_t stride_state_indices =
      conv_state_indices.has_value() ? conv_state_indices->stride(0) : 0;

  // conv.py: with speculative decoding the effective state_len is wider
  int64_t state_len = num_accepted_tokens.has_value()
                          ? (width - 1 + (seqlen - 1))
                          : (width - 1);
  const int64_t np2_statelen = sfc::next_power_of_2(state_len);
  TORCH_CHECK(np2_statelen <= 16,
              "sfc conv pure: NP2_STATELEN must be <= 16 (instantiated 1,2,4,8,16), got ",
              np2_statelen, " from state_len=", state_len);

  LaunchArgs args;
  args.x = x.const_data_ptr();
  args.w = weight.const_data_ptr();
  args.bias = bias.has_value() ? bias->const_data_ptr() : nullptr;
  args.conv_state = conv_state.data_ptr();
  args.conv_state_indices =
      conv_state_indices.has_value() ? conv_state_indices->const_data_ptr<int32_t>() : nullptr;
  args.num_accepted_tokens =
      num_accepted_tokens.has_value() ? num_accepted_tokens->const_data_ptr<int32_t>() : nullptr;
  args.query_start_loc =
      query_start_loc.has_value() ? query_start_loc->const_data_ptr<int32_t>() : nullptr;
  args.block_idx_last_scheduled_token =
      block_idx_last_scheduled_token.has_value()
          ? block_idx_last_scheduled_token->const_data_ptr<int32_t>()
          : nullptr;
  args.initial_state_idx =
      initial_state_idx.has_value() ? initial_state_idx->const_data_ptr<int32_t>() : nullptr;
  args.o = out.data_ptr();
  args.batch = static_cast<int>(batch);
  args.dim = static_cast<int>(dim);
  args.seqlen = static_cast<int>(seqlen);
  args.state_len = static_cast<int>(state_len);
  args.num_cache_lines = static_cast<int>(num_cache_lines);
  args.stride_x_seq = stride_x_seq;
  args.stride_x_dim = stride_x_dim;
  args.stride_x_token = stride_x_token;
  args.stride_w_dim = stride_w_dim;
  args.stride_w_width = stride_w_width;
  args.stride_conv_state_seq = stride_conv_state_seq;
  args.stride_conv_state_dim = stride_conv_state_dim;
  args.stride_conv_state_tok = stride_conv_state_tok;
  args.stride_state_indices = stride_state_indices;
  args.stride_o_seq = stride_o_seq;
  args.stride_o_dim = stride_o_dim;
  args.stride_o_token = stride_o_token;
  args.null_block_id = null_block_id;
  args.has_bias = bias.has_value();
  args.silu_activation = silu_activation;
  args.is_varlen = query_start_loc.has_value();
  args.is_apc_enabled = block_idx_last_scheduled_token.has_value();
  args.is_spec_decoding = num_accepted_tokens.has_value();
  args.has_null_block = has_null_block;
  args.launch_pdl = sfc::arch_supports_pdl();
  args.grid = dim3(static_cast<unsigned>(batch),
                   static_cast<unsigned>((dim + sfc::kBlockN - 1) / sfc::kBlockN), 1);
  args.stream = c10::cuda::getCurrentCUDAStream(x.device().index());

  const c10::cuda::OptionalCUDAGuard guard(x.device());
  TORCH_CHECK(sfc::launch_dispatch(args, static_cast<int>(width),
                                   static_cast<int>(np2_statelen),
                                   conv_state.scalar_type()),
              "sfc conv pure: no instantiation for width=", width,
              " np2_statelen=", np2_statelen, " dtype=", conv_state.scalar_type());

  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out.to(original_x_dtype);
}

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
  const dim3 grid(num_requests, num_value_heads);
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
  m.def("causal_conv1d_update", &sfc_causal_conv1d_update,
        "Stage 1: causal depthwise conv1d sliding-window state update (pure CUDA, "
        "patched_native FP32 convolution product); mutates conv_state in place and, with "
        "out omitted, overwrites and returns x");
  m.def("fused_gdn_decode_post_conv_mtp", &fused_gdn_decode_post_conv_mtp,
        "Stage 2: GDN decode/MTP post-convolution fusion (pure CUDA); mutates the `state` "
        "checkpoints in place and returns the caller-owned `out` buffer");
}
