// Torch-side ABI for the mla_absorbed_fp8 candidate. The numerical work lives
// in mla_v0.cu (SIMT fallback) and mla_v1.cu (SM103 tcgen05 path); this file
// only marshals the caller's tensors, keeps their storage/strides/aliases and
// launches on the current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <mutex>

void mla_v0_launch(const uint8_t* q, const uint8_t* kv, const int* block_tables, const int* seq_lens,
                   const int* cum_q, uint8_t* out, float* lse, const float* sinks, int heads,
                   int uniform_q, long long rows, int page_shift_unused, int page_size, int tstride,
                   float bmm1, float bmm2, cudaStream_t stream);
void mla_v1_launch(const uint8_t* q, const uint8_t* kv, const int* block_tables, const int* seq_lens,
                   const int* cum_q, uint8_t* out, float* lse, const float* sinks, int heads,
                   int uniform_q, long long rows, int batch, int page_shift, int tstride, float qk_scale,
                   float bmm2, int hilo, int pv_mode, int skip_qk, int skip_softmax, int stub_loads,
                   uint8_t* dbg_p, float* dbg_o, int dbg_p_tile,
                   int* dbg, long long items, float* workspace, long long workspace_floats,
                   int split_request, int rows_max, int max_seq_len, cudaStream_t stream);
int mla_v1_smem_bytes();
void mla_v1_prewarm_attrs();

namespace {

constexpr int kHeadDim = 576;
constexpr int kValueDim = 512;
constexpr int kMTile = 128;

struct Geometry {
  int heads = 0;
  int page_size = 0;
  int page_shift = 0;
  int tstride = 0;
  int batch = 0;
  int uniform_q = 0;
  long long rows = 0;
  long long items = 0;
};

Geometry describe(const torch::Tensor& query, const torch::Tensor& kv_cache,
                  const torch::Tensor& block_tables, const torch::Tensor& seq_lens) {
  Geometry g;
  g.heads = (int)query.size(query.dim() - 2);
  g.page_size = (int)kv_cache.size(2);
  g.tstride = (int)block_tables.size(1);
  g.rows = query.dim() == 4 ? (long long)query.size(0) * query.size(1) : (long long)query.size(0);
  g.uniform_q = query.dim() == 4 ? (int)query.size(1) : 0;
  g.batch = (int)seq_lens.size(0);
  int shift = 0;
  while ((1 << shift) < g.page_size) ++shift;
  TORCH_CHECK(g.page_size > 0 && (1 << shift) == g.page_size, "page_size must be a power of two");
  g.page_shift = shift;
  if (g.uniform_q > 0) {
    g.items = (long long)g.batch * (((long long)g.uniform_q * g.heads + kMTile - 1) / kMTile);
  } else {
    g.items = (g.rows * g.heads + kMTile - 1) / kMTile + g.batch;
  }
  return g;
}

void check_common(const torch::Tensor& query, const torch::Tensor& kv_cache, const torch::Tensor& out,
                  const torch::Tensor& block_tables, const torch::Tensor& seq_lens) {
  TORCH_CHECK(query.is_contiguous() && kv_cache.is_contiguous() && out.is_contiguous(),
              "mla candidate expects contiguous query/kv_cache/out");
  TORCH_CHECK(block_tables.is_contiguous() && seq_lens.is_contiguous(),
              "mla candidate expects contiguous index tensors");
  TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn && kv_cache.scalar_type() == at::kFloat8_e4m3fn,
              "mla candidate expects e4m3 query and kv_cache");
  TORCH_CHECK(query.size(query.dim() - 1) == kHeadDim, "mla candidate expects head_dim 576");
  TORCH_CHECK(out.size(out.dim() - 1) == kValueDim, "mla candidate expects value_dim 512");
  TORCH_CHECK(kv_cache.size(1) == 1, "mla candidate expects kv_heads == 1");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "mla candidate expects bf16 out");
}

}  // namespace

void mla_v0_forward(torch::Tensor query,
                    torch::Tensor kv_cache,
                    torch::Tensor block_tables,
                    torch::Tensor seq_lens,
                    c10::optional<torch::Tensor> cum_seq_lens_q,
                    torch::Tensor out,
                    c10::optional<torch::Tensor> lse,
                    c10::optional<torch::Tensor> sinks,
                    double bmm1_scale,
                    double bmm2_scale) {
  check_common(query, kv_cache, out, block_tables, seq_lens);
  Geometry g = describe(query, kv_cache, block_tables, seq_lens);
  mla_v0_launch(reinterpret_cast<const uint8_t*>(query.data_ptr()),
                reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()), block_tables.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                cum_seq_lens_q.has_value() ? cum_seq_lens_q->data_ptr<int>() : nullptr,
                reinterpret_cast<uint8_t*>(out.data_ptr()),
                lse.has_value() ? lse->data_ptr<float>() : nullptr,
                sinks.has_value() ? sinks->data_ptr<float>() : nullptr, g.heads, g.uniform_q, g.rows, 0,
                g.page_size, g.tstride, (float)bmm1_scale, (float)bmm2_scale,
                at::cuda::getCurrentCUDAStream());
}

namespace {

// Every host-side launch parameter except the stream, so the graph cache below
// can replay a call without recomputing or re-marshalling anything.
struct LaunchArgs {
  const uint8_t* q = nullptr;
  const uint8_t* kv = nullptr;
  const int* block_tables = nullptr;
  const int* seq_lens = nullptr;
  const int* cum_q = nullptr;
  uint8_t* out = nullptr;
  float* lse = nullptr;
  const float* sinks = nullptr;
  int heads = 0;
  int uniform_q = 0;
  long long rows = 0;
  int batch = 0;
  int page_shift = 0;
  int tstride = 0;
  float qk_scale = 0.f;
  float bmm2 = 0.f;
  int hilo = 1;
  long long items = 0;
  float* workspace = nullptr;
  long long workspace_floats = 0;
  int split_request = 0;
  int rows_max = 0;
  int max_seq_len = 0;
};

void do_launch(const LaunchArgs& a, cudaStream_t stream) {
  mla_v1_launch(a.q, a.kv, a.block_tables, a.seq_lens, a.cum_q, a.out, a.lse, a.sinks, a.heads,
                a.uniform_q, a.rows, a.batch, a.page_shift, a.tstride, a.qk_scale, a.bmm2, a.hilo,
                3, 0, 0, 0, nullptr, nullptr, 0, nullptr, a.items, a.workspace,
                a.workspace_floats, a.split_request, a.rows_max, a.max_seq_len, stream);
}

struct GraphKey {
  const void* q = nullptr;
  const void* kv = nullptr;
  const void* block_tables = nullptr;
  const void* seq_lens = nullptr;
  const void* cum_q = nullptr;
  void* out = nullptr;
  void* lse = nullptr;
  const void* sinks = nullptr;
  void* workspace = nullptr;
  long long workspace_floats = 0;
  long long rows = 0;
  long long items = 0;
  int heads = 0;
  int uniform_q = 0;
  int batch = 0;
  int page_shift = 0;
  int tstride = 0;
  int rows_max = 0;
  int max_seq_len = 0;
  int split_request = 0;
  int hilo = 0;
  float qk_scale = 0.f;
  float bmm2 = 0.f;
  cudaStream_t stream = nullptr;
};

bool key_equals(const GraphKey& a, const GraphKey& b) {
  return a.q == b.q && a.kv == b.kv && a.block_tables == b.block_tables &&
         a.seq_lens == b.seq_lens && a.cum_q == b.cum_q && a.out == b.out && a.lse == b.lse &&
         a.sinks == b.sinks && a.workspace == b.workspace &&
         a.workspace_floats == b.workspace_floats && a.rows == b.rows && a.items == b.items &&
         a.heads == b.heads && a.uniform_q == b.uniform_q && a.batch == b.batch &&
         a.page_shift == b.page_shift && a.tstride == b.tstride && a.rows_max == b.rows_max &&
         a.max_seq_len == b.max_seq_len && a.split_request == b.split_request &&
         a.hilo == b.hilo && a.qk_scale == b.qk_scale && a.bmm2 == b.bmm2 && a.stream == b.stream;
}

GraphKey make_key(const LaunchArgs& a, cudaStream_t stream) {
  GraphKey k;
  k.q = a.q;
  k.kv = a.kv;
  k.block_tables = a.block_tables;
  k.seq_lens = a.seq_lens;
  k.cum_q = a.cum_q;
  k.out = a.out;
  k.lse = a.lse;
  k.sinks = a.sinks;
  k.workspace = a.workspace;
  k.workspace_floats = a.workspace_floats;
  k.rows = a.rows;
  k.items = a.items;
  k.heads = a.heads;
  k.uniform_q = a.uniform_q;
  k.batch = a.batch;
  k.page_shift = a.page_shift;
  k.tstride = a.tstride;
  k.rows_max = a.rows_max;
  k.max_seq_len = a.max_seq_len;
  k.split_request = a.split_request;
  k.hilo = a.hilo;
  k.qk_scale = a.qk_scale;
  k.bmm2 = a.bmm2;
  k.stream = stream;
  return k;
}

constexpr int kGraphSlots = 16;
constexpr int kGraphCaptureBudget = 64;

struct GraphSlot {
  GraphKey key;
  cudaGraphExec_t exec = nullptr;
  bool captured = false;
  int hits = 0;
  unsigned long long stamp = 0;
  long long token = 0;
};

GraphSlot g_slots[kGraphSlots];
unsigned long long g_graph_clock = 0;
int g_graph_captures = 0;
cudaStream_t g_capture_stream = nullptr;
int g_graph_mode = -1;
// The most recently used graph, plus a token that identifies it. A caller that
// has already established that nothing about the launch changed can replay it
// through mla_v1_replay() without marshalling nine tensors through pybind again;
// the token is what makes that safe, since it only matches the graph that the
// last full call actually selected.
long long g_token_counter = 0;
long long g_active_token = 0;
cudaGraphExec_t g_active_exec = nullptr;
cudaStream_t g_active_stream = nullptr;
int g_stream_cache = -1;

bool stream_cached() {
  if (g_stream_cache < 0) {
    const char* env = getenv("MLA_STREAM_CACHE");
    g_stream_cache = (env != nullptr && env[0] == '0') ? 0 : 1;
  }
  return g_stream_cache == 1;
}

bool graph_enabled() {
  if (g_graph_mode < 0) {
    const char* env = getenv("MLA_GRAPH");
    g_graph_mode = (env != nullptr && env[0] == '0') ? 0 : 1;
  }
  return g_graph_mode == 1;
}

// Returns true when the call was served by a graph (replayed or freshly
// captured and launched); false means the caller must launch normally.
// One lock covers lookup, capture and replay: capture is thread-local, so a
// second thread launching on the private capture stream would corrupt it.
std::mutex g_graph_mutex;

// On success *token_out receives the identifier of the graph that was launched,
// which is what mla_v1_forward hands back to Python so a later call can replay it
// through mla_v1_replay() without re-marshalling the tensors.
bool graph_launch(const LaunchArgs& a, const GraphKey& key, cudaStream_t stream,
                  long long* token_out) {
  const std::lock_guard<std::mutex> guard(g_graph_mutex);
  *token_out = 0;
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) return false;
  if (status != cudaStreamCaptureStatusNone) return false;

  GraphSlot* slot = nullptr;
  for (int i = 0; i < kGraphSlots; ++i) {
    if (g_slots[i].hits > 0 && key_equals(g_slots[i].key, key)) {
      slot = &g_slots[i];
      break;
    }
  }
  if (slot == nullptr) {
    slot = &g_slots[0];
    for (int i = 1; i < kGraphSlots; ++i)
      if (g_slots[i].stamp < slot->stamp) slot = &g_slots[i];
    slot->key = key;
    slot->exec = nullptr;      // an evicted exec is leaked on purpose, see above
    slot->captured = false;
    slot->hits = 0;
  }
  slot->stamp = ++g_graph_clock;
  slot->hits += 1;

  if (slot->captured) {
    g_active_token = slot->token;
    g_active_exec = slot->exec;
    g_active_stream = stream;
    if (cudaGraphLaunch(slot->exec, stream) != cudaSuccess) return false;
    *token_out = slot->token;
    return true;
  }
  if (slot->hits < 2 || g_graph_captures >= kGraphCaptureBudget) return false;

  if (g_capture_stream == nullptr &&
      cudaStreamCreateWithFlags(&g_capture_stream, cudaStreamNonBlocking) != cudaSuccess) {
    return false;
  }
  mla_v1_prewarm_attrs();
  cudaGraph_t graph = nullptr;
  if (cudaStreamBeginCapture(g_capture_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  do_launch(a, g_capture_stream);
  const cudaError_t ended = cudaStreamEndCapture(g_capture_stream, &graph);
  if (ended != cudaSuccess || graph == nullptr) {
    cudaGetLastError();
    return false;
  }
  cudaGraphExec_t exec = nullptr;
  // The flags overload, spelled so it cannot bind to the deprecated
  // (error node, log buffer) signature that older toolkits also declare.
  const cudaError_t built = cudaGraphInstantiate(&exec, graph, (unsigned long long)0);
  cudaGraphDestroy(graph);
  if (built != cudaSuccess || exec == nullptr) {
    cudaGetLastError();
    return false;
  }
  ++g_graph_captures;
  slot->exec = exec;
  slot->captured = true;
  slot->token = ++g_token_counter;
  g_active_token = slot->token;
  g_active_exec = exec;
  g_active_stream = stream;
  if (cudaGraphLaunch(exec, stream) != cudaSuccess) return false;
  *token_out = slot->token;
  return true;
}

// One body for both entry points, so they cannot drift apart.
int replay_body(long long token) {
  if (token == 0 || token != g_active_token) return 0;
  cudaGraphExec_t exec = g_active_exec;
  if (exec == nullptr) return 0;
  const cudaStream_t stream = stream_cached() ? g_active_stream
                                              : (cudaStream_t)at::cuda::getCurrentCUDAStream();
  return cudaGraphLaunch(exec, stream) == cudaSuccess ? 1 : 0;
}

PyObject* replay_meth_o(PyObject* /*self*/, PyObject* arg) {
  const long long token = PyLong_Check(arg) ? PyLong_AsLongLong(arg) : -1;
  if (token == -1 && PyErr_Occurred()) return nullptr;
  return PyLong_FromLong(replay_body(token));
}
PyMethodDef g_replay_meth_o = {"mla_v1_replay_o", (PyCFunction)replay_meth_o, METH_O, nullptr};
constexpr Py_ssize_t kFcMaxArgs = 64;

struct FastCallCache {
  PyObject* self = nullptr;      // owned
  PyObject* forward = nullptr;   // owned, the UNBOUND Model.forward
  PyObject* result = nullptr;    // owned, returned on a hit
  PyObject* vals[kFcMaxArgs];    // owned, the exact argument objects
  Py_ssize_t nargs = -1;
  Py_ssize_t nkw = -1;
  cudaGraphExec_t exec = nullptr;
  cudaStream_t stream = nullptr;
  long long token = 0;
  bool armed = false;
};
FastCallCache g_fc;

void fastcall_release() {
  if (g_fc.nargs >= 0 && g_fc.nkw >= 0) {
    for (Py_ssize_t i = 0; i < g_fc.nargs + g_fc.nkw; ++i) {
      Py_XDECREF(g_fc.vals[i]);
      g_fc.vals[i] = nullptr;
    }
  }
  Py_CLEAR(g_fc.result);
  g_fc.nargs = -1;
  g_fc.nkw = -1;
  g_fc.exec = nullptr;
  g_fc.token = 0;
  g_fc.armed = false;
}

// Identity first, then value for the immutables only. PyObject_RichCompareBool on an
// arbitrary object would call Tensor.__eq__, get a tensor back, and raise on the
// ambiguous truth value, so anything that is not a float or a long falls back to
// "not the same" - which costs a slow call and never a wrong answer.
bool same_arg(PyObject* a, PyObject* b) {
  if (a == b) return true;
  if (a == nullptr || b == nullptr) return false;
  if (PyFloat_CheckExact(a) && PyFloat_CheckExact(b))
    return PyFloat_AS_DOUBLE(a) == PyFloat_AS_DOUBLE(b);
  if (PyLong_Check(a) && PyLong_Check(b))
    return PyObject_RichCompareBool(a, b, Py_EQ) == 1;
  return false;
}

PyObject* fast_call_impl(PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames) {
  const Py_ssize_t nkw = (kwnames != nullptr) ? PyTuple_GET_SIZE(kwnames) : 0;
  if (g_fc.armed && nargs == g_fc.nargs && nkw == g_fc.nkw && g_fc.exec != nullptr &&
      g_fc.exec == g_active_exec && g_fc.token == g_active_token) {
    bool same = true;
    for (Py_ssize_t i = 0; i < nargs + nkw; ++i) {
      if (!same_arg(args[i], g_fc.vals[i])) { same = false; break; }
    }
    if (same && cudaGraphLaunch(g_fc.exec, g_fc.stream) == cudaSuccess) {
      Py_INCREF(g_fc.result);
      return g_fc.result;
    }
    fastcall_release();
  }
  if (g_fc.forward == nullptr) {
    PyErr_SetString(PyExc_RuntimeError,
                    "mla_v1_fast_call was installed before mla_v1_bind_instance");
    return nullptr;
  }
  PyObject* res = nullptr;
  if (nargs >= 1 && args[0] == g_fc.self) {
    res = PyObject_Vectorcall(g_fc.forward, args, nargs, kwnames);
  } else if (nargs + nkw + 1 <= kFcMaxArgs) {
    PyObject* stack[kFcMaxArgs];
    stack[0] = g_fc.self;
    for (Py_ssize_t i = 0; i < nargs + nkw; ++i) stack[i + 1] = args[i];
    res = PyObject_Vectorcall(g_fc.forward, stack, nargs + 1, kwnames);
  } else {
    PyErr_SetString(PyExc_RuntimeError, "mla_v1_fast_call: too many positional arguments");
    return nullptr;
  }
  if (res == nullptr) return nullptr;
  if (g_active_exec != nullptr && g_active_token != 0 && nargs + nkw <= kFcMaxArgs) {
    fastcall_release();
    g_fc.exec = g_active_exec;
    g_fc.stream = g_active_stream;
    g_fc.token = g_active_token;
    g_fc.nargs = nargs;
    g_fc.nkw = nkw;
    for (Py_ssize_t i = 0; i < nargs + nkw; ++i) {
      g_fc.vals[i] = args[i];
      Py_INCREF(args[i]);
    }
    g_fc.result = res;
    Py_INCREF(res);
    g_fc.armed = true;
  }
  return res;
}

PyObject* fast_call_entry(PyObject*, PyObject* const* args, Py_ssize_t nargsf, PyObject* kwnames) {
  return fast_call_impl(args, PyVectorcall_NARGS(nargsf), kwnames);
}
PyMethodDef g_fast_call_def = {"mla_v1_fast_call", (PyCFunction)(void*)fast_call_entry,
                               METH_FASTCALL | METH_KEYWORDS, nullptr};

}  // namespace

int64_t mla_v1_bind_instance(pybind11::object self) {
  // The unbound function off the type, not a bound method: fast_call_impl supplies
  // the instance itself so that it works whichever way CPython dispatched __call__.
  PyObject* unbound = PyObject_GetAttrString((PyObject*)Py_TYPE(self.ptr()), "forward");
  if (unbound == nullptr || !PyCallable_Check(unbound)) {
    Py_XDECREF(unbound);
    PyErr_Clear();
    return 0;
  }
  fastcall_release();
  Py_XDECREF(g_fc.forward);
  Py_XDECREF(g_fc.self);
  g_fc.forward = unbound;
  g_fc.self = self.release().ptr();
  return 1;
}

pybind11::object mla_v1_make_fast_call() {
  return pybind11::reinterpret_steal<pybind11::object>(
      PyCFunction_New(&g_fast_call_def, nullptr));
}

int64_t mla_v1_replay(int64_t token) {
  return (int64_t)replay_body((long long)token);
}

// Hands Python the METH_O entry point above as a plain callable object. Called
// once, when the identity-keyed cache is filled; the object it returns is what
// every later replay call goes through.
pybind11::object mla_v1_fast_replay() {
  return pybind11::reinterpret_steal<pybind11::object>(PyCFunction_New(&g_replay_meth_o, nullptr));
}

int64_t mla_v1_forward(torch::Tensor query,
                    torch::Tensor kv_cache,
                    torch::Tensor block_tables,
                    torch::Tensor seq_lens,
                    c10::optional<torch::Tensor> cum_seq_lens_q,
                    torch::Tensor out,
                    c10::optional<torch::Tensor> lse,
                    c10::optional<torch::Tensor> sinks,
                    double bmm1_scale,
                    double bmm2_scale,
                    int64_t hilo,
                    c10::optional<torch::Tensor> workspace,
                    int64_t max_q_len,
                    int64_t max_seq_len,
                    int64_t target_ctas) {
  check_common(query, kv_cache, out, block_tables, seq_lens);
  Geometry g = describe(query, kv_cache, block_tables, seq_lens);
  const long long q_len_max = g.uniform_q > 0 ? (long long)g.uniform_q
                                             : (max_q_len > 0 ? max_q_len : g.rows);
  long long rows_max = (long long)q_len_max * g.heads;
  if (rows_max > kMTile) rows_max = kMTile;
  if (rows_max < 1) rows_max = 1;
  LaunchArgs a;
  a.q = reinterpret_cast<const uint8_t*>(query.data_ptr());
  a.kv = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
  a.block_tables = block_tables.data_ptr<int>();
  a.seq_lens = seq_lens.data_ptr<int>();
  a.cum_q = cum_seq_lens_q.has_value() ? cum_seq_lens_q->data_ptr<int>() : nullptr;
  a.out = reinterpret_cast<uint8_t*>(out.data_ptr());
  a.lse = lse.has_value() ? lse->data_ptr<float>() : nullptr;
  a.sinks = sinks.has_value() ? sinks->data_ptr<float>() : nullptr;
  a.heads = g.heads;
  a.uniform_q = g.uniform_q;
  a.rows = g.rows;
  a.batch = g.batch;
  a.page_shift = g.page_shift;
  a.tstride = g.tstride;
  a.qk_scale = (float)(bmm1_scale * 1.4426950408889634);
  a.bmm2 = (float)bmm2_scale;
  a.hilo = hilo ? 1 : 0;
  a.items = g.items;
  if (workspace.has_value() && workspace->numel() > 0) {
    a.workspace = reinterpret_cast<float*>(workspace->data_ptr());
    a.workspace_floats = (long long)(workspace->numel() * workspace->element_size() / 4);
  }
  // A negative request means "choose the split policy yourself, aiming at this
  // many CTAs overall"; mla_v1_launch turns it into a per-item split count.
  a.split_request = target_ctas > 0 ? (int)-target_ctas : 0;
  a.rows_max = (int)rows_max;
  a.max_seq_len = (int)max_seq_len;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (graph_enabled()) {
    long long token = 0;
    if (graph_launch(a, make_key(a, stream), stream, &token)) return (int64_t)token;
  }
  // No graph was launched for this call, so the C-level __call__ cache must not
  // believe one was: clearing the active token is what tells it to stay disarmed
  // rather than replay a graph that belongs to some earlier geometry.
  g_active_token = 0;
  g_active_exec = nullptr;
  fastcall_release();
  do_launch(a, stream);
  return 0;
}

int64_t mla_v1_shared_bytes() { return (int64_t)mla_v1_smem_bytes(); }



#include "kernel_simt.cuh"
#include "dispatch.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_v0_forward", torch::wrap_pybind_function(mla_v0_forward));
    m.def("mla_v1_bind_instance", torch::wrap_pybind_function(mla_v1_bind_instance));
    m.def("mla_v1_make_fast_call", torch::wrap_pybind_function(mla_v1_make_fast_call));
    m.def("mla_v1_replay", torch::wrap_pybind_function(mla_v1_replay));
    m.def("mla_v1_fast_replay", torch::wrap_pybind_function(mla_v1_fast_replay));
    m.def("mla_v1_forward", torch::wrap_pybind_function(mla_v1_forward));
}
