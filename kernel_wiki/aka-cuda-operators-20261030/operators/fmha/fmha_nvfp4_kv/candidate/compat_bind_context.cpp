// FMHA-COMPAT -- Torch binding for the *context* (prefill) route.
//
// Serves the four operators whose production entry is
// `flashinfer.prefill.trtllm_batch_context_with_kv_cache`: fmha_bf16 (D01), fmha_fp16 (D02),
// fmha_fp8 (D03) and fmha_nvfp4_kv (D06).  They share one forward signature and differ only in the
// dtypes the problem hands over, which is why one binding serves all four and each operator keeps
// its own `model.py` stating its own dtype contract.  The decode route (`fmha_bf16_fp8`, D04) has a
// genuinely different signature and lives in `bind_decode.cpp`.
//
// Role, and only this role: turn the problem's tensors into the raw pointers, dimensions and dtype
// ids that `abi.h` declares, check the ABI facts the problem's contract states, and report the
// launcher's status.  No computation, no allocation, no copy, no reordering and no synchronization
// happen here, so nothing is moved out of the timed region and nothing is added to it.
//
// ABI facts taken from the frozen problem contracts (metadata.json + input.py):
//   query         [total_q, heads, head_dim]        bf16 | fp16 | fp8_e4m3, head dim contiguous
//   key_cache     [pages, kv_heads, page, head_dim] bf16 | fp16 | fp8_e4m3 | packed NVFP4 uint8
//   value_cache   same shape and dtype as key_cache; V block scales are token-swizzled
//   block_tables  [batch, pages]                    int32, shared by K and V
//   seq_lens      [batch]                           int32, KV length per request
//   q_indptr      [batch + 1]                       int32, = the problem's cum_seq_lens_q
//   out           [total_q, heads, head_dim]        caller-owned guarded view: written in place and
//                                                   returned by Model.forward, alias preserved
//   lse           [total_q, heads]                  float32, optional, same guarded-view contract
//   sinks         [heads]                           float32, optional
//   kv_cache_sf   (k_block_scales, v_block_scales)  fp8_e4m3, required iff KV is packed NVFP4
//   workspace     the problem's own 128 MiB declared scratch; this route passes q_indptr explicitly
//                 and therefore never writes to it
//
// Packed NVFP4 needs no hint from Python: its payload arrives as uint8 with half the logical head
// dimension, so `uint8 -> kNvFp4` is the only encoding consistent with the tensors, and
// `host.cuh` then proves it by requiring `key_cache.size(3) * 2 == query.size(2)` and the presence
// of both block-scale tensors.  A wrong guess is a refusal with the route's own message, not a
// silently wrong answer.

#include <optional>
#include <string>

#include "compat_abi.h"
// Tensor-to-ABI conversion used only by this binding.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <optional>

#include "compat_abi.h"

namespace fmha_compat_bind {

using fmha_compat_abi::Call;
using fmha_compat_abi::Tensor;

// Must match the ids in abi.h, which launch.cu static_asserts against host.cuh's own constants.
inline int dtype_id(const at::Tensor& t, const char* what) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return fmha_compat_abi::kBFloat16;
    case at::kHalf: return fmha_compat_abi::kFloat16;
    case at::kFloat8_e4m3fn: return fmha_compat_abi::kFloat8E4M3;
    case at::kByte: return fmha_compat_abi::kNvFp4;  // packed: two e2m1 values per byte
    case at::kFloat: return fmha_compat_abi::kFloat32;
    case at::kInt: return fmha_compat_abi::kInt32;
    default:
      TORCH_CHECK(false, what, " has an unsupported dtype ", at::toString(t.scalar_type()),
                  "; the Cake FMHA compatibility route takes bf16, fp16, fp8_e4m3 or packed NVFP4");
  }
}

// Sizes and strides as the kernel addresses them: element strides, rank by rank, for rank <= 4.
// Nothing is made contiguous here -- a stride the kernel cannot express is a refusal in host.cuh,
// and quietly re-packing would be both a contract change and an untimed copy.
inline void fill(Tensor& target, const at::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda(), what, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() <= fmha_compat_abi::kMaxRank, what, " must have rank <= 4, got ", t.dim());
  target.ptr = t.data_ptr();
  target.ndim = static_cast<int>(t.dim());
  for (int i = 0; i < target.ndim; ++i) {
    target.size[i] = static_cast<long long>(t.size(i));
    target.stride[i] = static_cast<long long>(t.stride(i));
  }
  target.dtype = dtype_id(t, what);
  target.present = 1;
}

inline void fill(Tensor& target, const std::optional<at::Tensor>& maybe, const char* what) {
  if (!maybe.has_value() || !maybe->defined()) {
    target = Tensor{};  // absent: null pointer, no checks, exactly as Optional<TensorView> was
    return;
  }
  fill(target, *maybe, what);
}

inline void check_required(const Tensor& t, const char* what) {
  TORCH_CHECK(t.present != 0, what, " is required by this route");
}

// Shared by both routes: the tensor half of the call struct, filled from the problem's own
// arguments.  `q_indptr_name` only affects an error message, so a refusal names the argument the
// caller actually passed.
inline void fill_tensors(Call& call, const at::Tensor& query, const at::Tensor& key_cache,
                         const at::Tensor& value_cache, const at::Tensor& block_tables,
                         const at::Tensor& seq_lens, const std::optional<at::Tensor>& q_indptr,
                         const at::Tensor& out, const std::optional<at::Tensor>& lse,
                         const std::optional<at::Tensor>& sinks,
                         const std::optional<at::Tensor>& key_block_scales,
                         const std::optional<at::Tensor>& value_block_scales,
                         const std::optional<at::Tensor>& out_scale_factor,
                         const at::Tensor& workspace, const char* q_indptr_name) {
  fill(call.query, query, "query");
  fill(call.key_cache, key_cache, "key_cache");
  fill(call.value_cache, value_cache, "value_cache");
  fill(call.block_tables, block_tables, "block_tables");
  fill(call.seq_lens, seq_lens, "seq_lens");
  fill(call.q_indptr, q_indptr, q_indptr_name);
  fill(call.out, out, "out");
  fill(call.lse, lse, "lse");
  fill(call.attention_sinks, sinks, "sinks");
  fill(call.key_block_scales, key_block_scales, "kv_cache_sf[0]");
  fill(call.value_block_scales, value_block_scales, "kv_cache_sf[1]");
  fill(call.out_scale_factor, out_scale_factor, "out_scale_factor");

  check_required(call.query, "query");
  check_required(call.key_cache, "kv_cache[0]");
  check_required(call.value_cache, "kv_cache[1]");
  check_required(call.block_tables, "block_tables");
  check_required(call.seq_lens, "seq_lens");
  check_required(call.out, "out");
  TORCH_CHECK(workspace.is_cuda() && workspace.dtype() == at::kByte,
              "workspace_buffer must be a CUDA uint8 tensor");

  if (call.lse.present) {
    call.lse_stride_tokens = call.lse.stride[0];
    call.lse_stride_heads = call.lse.stride[1];
  }
  call.workspace_ptr = workspace.data_ptr();
  call.workspace_bytes = static_cast<long long>(workspace.numel());
}

// One place that turns a launcher status into the caller's error, so both routes refuse with the
// route's own message instead of a binding-specific paraphrase.
inline int64_t launch(const Call& call) {
  const char* message = "ok";
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int status = aka_fmha_compat_launch(&call, stream, &message);
  TORCH_CHECK(status == 0, "Cake FMHA compatibility route refused this call: ", message,
              " (code ", status, ")");
  return static_cast<int64_t>(status);
}

}  // namespace fmha_compat_bind



namespace {
using fmha_compat_bind::Call;
using fmha_compat_bind::fill_tensors;
}  // namespace

// Writes `out` (and `lse` when present) in place.  Returns nothing: the caller already owns both
// buffers and Model.forward returns those very objects, which is what preserves the alias the
// metadata declares.  The int64 result is the launcher status, so a receipt can distinguish a
// domain refusal from a launch failure without parsing an exception.
int64_t fmha_compat_context(const at::Tensor& query, const at::Tensor& key_cache,
                            const at::Tensor& value_cache, const at::Tensor& block_tables,
                            const at::Tensor& seq_lens,
                            const std::optional<at::Tensor>& q_indptr, const at::Tensor& out,
                            const std::optional<at::Tensor>& lse,
                            const std::optional<at::Tensor>& sinks,
                            const std::optional<at::Tensor>& key_block_scales,
                            const std::optional<at::Tensor>& value_block_scales,
                            const at::Tensor& workspace, double bmm1_scale, double bmm2_scale,
                            int64_t batch_size, int64_t uniform_q_len, int64_t window_left,
                            bool is_causal, bool uses_shared_paged_kv_idx, int64_t kv_layout,
                            double o_sf_scale, int64_t o_sf_start_index,
                            const std::optional<at::Tensor>& out_scale_factor) {
  Call call;
  fill_tensors(call, query, key_cache, value_cache, block_tables, seq_lens, q_indptr, out, lse,
               sinks, key_block_scales, value_block_scales, out_scale_factor, workspace,
               "cum_seq_lens_q");

  call.batch_size = static_cast<long long>(batch_size);
  call.uniform_q_len = static_cast<long long>(uniform_q_len);
  call.window_left = static_cast<long long>(window_left);
  call.is_causal = is_causal ? 1 : 0;
  call.uses_shared_paged_kv_idx = uses_shared_paged_kv_idx ? 1 : 0;
  call.kv_layout = static_cast<int>(kv_layout);
  call.bmm1_scale = bmm1_scale;
  call.bmm2_scale = bmm2_scale;
  call.o_sf_scale = o_sf_scale;
  call.o_sf_start_index = static_cast<long long>(o_sf_start_index);

  return fmha_compat_bind::launch(call);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pure-CUDA Cake FMHA compatibility route, context/prefill entry (Atrex-Bench D01/D02/D03/D06)";
  m.def("fmha_compat_context", &fmha_compat_context,
        "paged ragged GQA attention over the caller-owned out/lse buffers",
        py::arg("query"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_tables"),
        py::arg("seq_lens"), py::arg("q_indptr"), py::arg("out"), py::arg("lse"),
        py::arg("sinks"), py::arg("key_block_scales"), py::arg("value_block_scales"),
        py::arg("workspace"), py::arg("bmm1_scale"), py::arg("bmm2_scale"), py::arg("batch_size"),
        py::arg("uniform_q_len"), py::arg("window_left"), py::arg("is_causal"),
        py::arg("uses_shared_paged_kv_idx"), py::arg("kv_layout"), py::arg("o_sf_scale"),
        py::arg("o_sf_start_index"), py::arg("out_scale_factor"));
}
