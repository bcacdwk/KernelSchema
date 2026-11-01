// Host-side launch binding for the FMHA candidate, as a CPython extension.
//
// Why this exists: the evaluator's measured region starts with the GPU idle
// (InputReplay restores the mutated inputs and then calls
// torch.cuda.set_rng_state, which synchronizes the stream), so host launch cost
// is part of every reported latency. Measured on the job CPU (hostbench, job
// dv_a1159241c2de): a ctypes call costs 0.5-1.0 us while a compiled-extension
// call costs 0.02-0.04 us, cudaLaunchKernel ~2.0-3.3 us, and cudaGraphLaunch
// ~1.9 us for one OR two nodes. This module removes the ctypes marshalling from
// the hot path and can collapse a main+combine launch pair into one graph
// launch.
//
// It contains NO numerics and no launch policy: a slot only remembers the
// already-packed FmhaParams buffer, the launcher entry points exported by
// fmha_kernel.cu / fmha_tc.cu, and the two host tensormaps. Those launchers
// choose the kernel, grid and split combine exactly as the ctypes path does.
#include <Python.h>
#include <cuda_runtime.h>
#include <cstdint>

typedef int (*run_fn)(const void*, void*, const void*, const void*);

struct Slot {
  const void* params;
  run_fn run;
  run_fn combine;      // optional second launch (nullptr when run does it all)
  const void* kmap;
  const void* vmap;
  cudaGraphExec_t exec;
  int graph_bad;       // capture failed or caller-captured: never retry
};

#define FMHA_MAX_SLOTS 4096
static Slot g_slots[FMHA_MAX_SLOTS];
static int g_nslots = 0;
static cudaStream_t g_side = nullptr;

static inline bool slot_ok(long i) { return i >= 0 && i < g_nslots; }

static inline int do_run(const Slot& s, cudaStream_t st) {
  int rc = s.run(s.params, (void*)st, s.kmap, s.vmap);
  if (rc == 0 && s.combine != nullptr) rc = s.combine(s.params, (void*)st, 0, 0);
  return rc;
}

// register(params, run, combine, kmap, vmap) -> slot id (negative on error)
static PyObject* ext_register(PyObject*, PyObject* const* a, Py_ssize_t n) {
  if (n != 5) {
    PyErr_SetString(PyExc_TypeError, "register(params, run, combine, kmap, vmap)");
    return nullptr;
  }
  if (g_nslots >= FMHA_MAX_SLOTS) return PyLong_FromLong(-1);
  Slot& s = g_slots[g_nslots];
  s.params = (const void*)(uintptr_t)PyLong_AsUnsignedLongLong(a[0]);
  s.run = (run_fn)(uintptr_t)PyLong_AsUnsignedLongLong(a[1]);
  s.combine = (run_fn)(uintptr_t)PyLong_AsUnsignedLongLong(a[2]);
  s.kmap = (const void*)(uintptr_t)PyLong_AsUnsignedLongLong(a[3]);
  s.vmap = (const void*)(uintptr_t)PyLong_AsUnsignedLongLong(a[4]);
  s.exec = nullptr;
  s.graph_bad = 0;
  if (PyErr_Occurred()) return nullptr;
  if (!s.params || !s.run) return PyLong_FromLong(-2);
  return PyLong_FromLong(g_nslots++);
}

// run(slot, stream) -> rc (0 ok)
static PyObject* ext_run(PyObject*, PyObject* const* a, Py_ssize_t n) {
  if (n != 2) {
    PyErr_SetString(PyExc_TypeError, "run(slot, stream)");
    return nullptr;
  }
  const long slot = PyLong_AsLong(a[0]);
  if (!slot_ok(slot)) return PyLong_FromLong(-3);
  cudaStream_t st = (cudaStream_t)(uintptr_t)PyLong_AsUnsignedLongLong(a[1]);
  if (PyErr_Occurred()) return nullptr;
  const Slot& s = g_slots[slot];
  if (s.exec != nullptr) {
    // A graph launch is illegal while the CALLER is capturing; the direct chain
    // below is capture-safe, so fall back to it in that case.
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    if (cs == cudaStreamCaptureStatusNone)
      return PyLong_FromLong((long)cudaGraphLaunch(s.exec, st));
  }
  return PyLong_FromLong((long)do_run(s, st));
}

// capture(slot, stream) -> 0 when a graph now exists for this slot. Records the
// slot's launch chain on a private side stream; the caller must already have
// executed that chain directly at least once, so no work is skipped. Refuses to
// capture while the caller's stream is itself capturing.
static PyObject* ext_capture(PyObject*, PyObject* const* a, Py_ssize_t n) {
  if (n != 2) {
    PyErr_SetString(PyExc_TypeError, "capture(slot, stream)");
    return nullptr;
  }
  const long slot = PyLong_AsLong(a[0]);
  if (!slot_ok(slot)) return PyLong_FromLong(-3);
  Slot& s = g_slots[slot];
  if (s.exec) return PyLong_FromLong(0);
  if (s.graph_bad) return PyLong_FromLong(-4);
  cudaStream_t cur = (cudaStream_t)(uintptr_t)PyLong_AsUnsignedLongLong(a[1]);
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(cur, &cs);
  if (cs != cudaStreamCaptureStatusNone) return PyLong_FromLong(-5);
  if (!g_side) {
    int rc = (int)cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking);
    if (rc) { s.graph_bad = 1; return PyLong_FromLong(rc); }
  }
  int rc = (int)cudaStreamBeginCapture(g_side, cudaStreamCaptureModeThreadLocal);
  if (rc) { s.graph_bad = 1; return PyLong_FromLong(rc); }
  const int rcr = do_run(s, g_side);
  cudaGraph_t g = nullptr;
  const int rc2 = (int)cudaStreamEndCapture(g_side, &g);
  if (rcr || rc2 || !g) {
    if (g) cudaGraphDestroy(g);
    s.graph_bad = 1;
    return PyLong_FromLong(rcr ? rcr : (rc2 ? rc2 : -1));
  }
  cudaGraphExec_t e = nullptr;
  rc = (int)cudaGraphInstantiate(&e, g, nullptr, nullptr, 0);
  cudaGraphDestroy(g);
  if (rc || !e) { s.graph_bad = 1; return PyLong_FromLong(rc ? rc : -2); }
  s.exec = e;
  return PyLong_FromLong(0);
}

static PyObject* ext_slots(PyObject*, PyObject*) {
  return PyLong_FromLong(g_nslots);
}

static PyMethodDef g_methods[] = {
    {"register_slot", (PyCFunction)ext_register, METH_FASTCALL, ""},
    {"run", (PyCFunction)ext_run, METH_FASTCALL, ""},
    {"capture", (PyCFunction)ext_capture, METH_FASTCALL, ""},
    {"slots", (PyCFunction)ext_slots, METH_NOARGS, ""},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef g_mod = {PyModuleDef_HEAD_INIT, "fmha_ext", nullptr, -1,
                                   g_methods};

PyMODINIT_FUNC PyInit_fmha_ext(void) { return PyModule_Create(&g_mod); }
