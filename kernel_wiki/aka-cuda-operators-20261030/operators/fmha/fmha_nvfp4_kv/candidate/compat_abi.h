// FMHA-COMPAT -- the plain-data boundary between the Torch binding and the CUDA translation unit.
//
// Why this file exists: `host.cuh` includes `kernel.cuh`, which contains `__global__` device code
// and therefore may only be compiled by nvcc.  The binding is an ordinary C++ translation unit, so
// it cannot include either.  Rather than repeat `CompatArgs` in two places -- two copies that could
// drift, which is exactly how an ABI silently stops meaning what it says -- the binding fills this
// POD struct and `launch.cu` translates it into `fmha_compat::CompatArgs` field by field.
//
// Rules this header keeps:
//   * no CUDA device code, no templates, no exceptions: includable from a .cpp and a .cu alike;
//   * every tensor is metadata only (pointer, rank, sizes, strides, dtype id, presence).  Nothing
//     here owns memory and nothing here computes;
//   * dtype ids are the graded route's own encoding, identical to the constants in `host.cuh`.
//     They are restated here because this header cannot include `host.cuh`, and `launch.cu`
//     static_asserts the two sets agree, so a drift is a compile error and not a wrong kernel.

#ifndef FMHA_COMPAT_ABI_H_
#define FMHA_COMPAT_ABI_H_

#include <cuda_runtime_api.h>

namespace fmha_compat_abi {

// Payload dtype ids.  NVFP4 is *packed*: two e2m1 values per byte, so the physical last extent is
// half the logical head dimension.
constexpr int kBFloat16 = 0;
constexpr int kFloat16 = 1;
constexpr int kFloat8E4M3 = 2;
constexpr int kNvFp4 = 3;
// Auxiliary tensors, which are never NVFP4.  They continue the same numbering so one field can
// describe every argument.
constexpr int kFloat32 = 4;   // attention_sinks, lse
constexpr int kInt32 = 5;     // q_indptr, seq_lens, block_tables

// Paged-KV layout ids, as the graded route spelled them.
constexpr int kHndLayout = 0;  // index = page*s0 + head*s1 + token*s2 + dim*s3
constexpr int kNhdLayout = 1;  // index = page*s0 + token*s1 + head*s2 + dim*s3

constexpr int kMaxRank = 4;

struct Tensor {
  const void* ptr = nullptr;
  int ndim = 0;
  long long size[kMaxRank] = {0, 0, 0, 0};
  long long stride[kMaxRank] = {0, 0, 0, 0};
  int dtype = -1;
  int present = 0;  // 0 replaces Optional<TensorView>: absent contributes a null pointer, no checks
};

// One complete call.  Field names match `fmha_compat::CompatArgs` one for one.
struct Call {
  Tensor out;
  Tensor out_scale_factor;
  Tensor query;
  Tensor key_cache;
  Tensor value_cache;
  Tensor block_tables;
  Tensor seq_lens;
  Tensor q_indptr;  // absent => uniform decode; host.cuh materialises it into the workspace below
  Tensor attention_sinks;
  Tensor key_block_scales;    // required when the KV payload is packed NVFP4
  Tensor value_block_scales;
  Tensor lse;
  // Declared scratch from the problem's own contract (its `workspace_buffer`).  Only ever used to
  // materialise a uniform q_indptr; host.cuh refuses to write anywhere else and refuses to
  // enlarge it.
  const void* workspace_ptr = nullptr;
  long long workspace_bytes = 0;
  long long batch_size = 0;
  long long uniform_q_len = 0;  // > 0 selects the decode route's uniform indptr
  long long window_left = -1;
  int is_causal = 1;
  int uses_shared_paged_kv_idx = 1;
  double bmm1_scale = 0.0;  // sm_scale: softmax scale with the q and k scales folded in
  double bmm2_scale = 0.0;  // v_scale
  double o_sf_scale = 0.0;
  long long o_sf_start_index = 0;
  long long lse_stride_tokens = 0;
  long long lse_stride_heads = 0;
  int kv_layout = kHndLayout;
};

}  // namespace fmha_compat_abi

// Returns 0 on success, otherwise a cudaError_t value; `*message` is set to a static string naming
// the precondition that failed -- the same text the graded route threw, so a refusal is attributable
// to the same rule it always was, and a receipt can tell a domain refusal from a crash.
extern "C" int aka_fmha_compat_launch(const fmha_compat_abi::Call* call, cudaStream_t stream,
                                      const char** message);

#endif  // FMHA_COMPAT_ABI_H_
