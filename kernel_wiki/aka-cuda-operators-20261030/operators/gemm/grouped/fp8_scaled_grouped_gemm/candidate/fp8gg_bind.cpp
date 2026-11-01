// fp8gg_bind.cpp — CPython fast-path launcher + C-side identity memo for the
// fp8_scaled_grouped_gemm candidate (ABI binding only; ALL numerics live in
// fp8gg_umma.cu / fp8gg.cu).
//
// Host-path evolution (measured, same-job interleaved):
//   ctypes cells            6.06us  (turn 11)
//   + METH_FASTCALL bind    4.84us  (turn 12, dv_f8749c372a4c/dv_e526df9518d3)
//   + C-side weakref memo   2.49us  (turn 16, dv_1ba4b593e011: -1.55..-1.65us
//                                    vs bind path, consistent across 2 shapes
//                                    x 2 protocols x 3 fresh-process reps;
//                                    numerics gate 20/20 + lifecycle clean)
//
// C memo soundness — IDENTICAL argument to the Python weakref memo it
// replaces: a hit requires all six stored weakrefs ALIVE and pointer-
// identical to the argument objects (PyWeakref_GetObject == arg). A torch
// tensor's data_ptr/shape/dtype/strides cannot change after construction,
// so the ptrs/dims cached at store_memo() time are exactly the live
// arguments'. Dead weakref -> Py_None -> never equal -> miss -> full Python
// revalidation. id()-reuse is impossible: identity is via weakref-resolved
// object pointers, and a freed object's weakref resolves to Py_None even if
// its address is reused (verified on-device: lifecycle 'reuse' test).
// Contents are NEVER cached; the kernel re-runs on every call.
//
// Contract: Python side only allocates, inspects public metadata, compiles/
// loads and binds the ABI — this file is pure binding glue (no tensor
// logic, no numerics, no allocations beyond small return ints).

#include <Python.h>
#include <stdint.h>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

extern "C" {
// defined in fp8gg_umma.cu (primary tcgen05 engine)
extern int fp8umma_run(const void* a, const void* b, const float* sa, const float* sb,
                       const int* indptr, void* out, int m, int n, int k, int G, int nSM,
                       int stages, cudaStream_t stream);
// defined in fp8gg.cu (fallback mma.sync engine); weak so the umma-only
// module links cleanly and launch_sync reports unavailable at runtime.
#pragma weak fp8gg_run
extern int fp8gg_run(const void* a, const void* b, const float* sa, const float* sb,
                     const int* indptr, void* out, int m, int n, int k, int G, int nSM,
                     cudaStream_t stream);
}

namespace {

// ---- C-side weakref identity memo (single slot; GIL-protected) ----
PyObject* g_wr[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
uint64_t  g_ptr[6] = {0, 0, 0, 0, 0, 0};
int       g_dims[6] = {0, 0, 0, 0, 0, 0};  // m, n, k, e, nsm, stages
PyObject* g_stream_fn = nullptr;           // strong ref; fn(dev) -> int stream
int       g_dev = 0;
int       g_valid = 0;

void clear_slot() {
    for (int i = 0; i < 6; ++i) { Py_XDECREF(g_wr[i]); g_wr[i] = nullptr; }
    g_valid = 0;
}

// try_launch6(a, b, a_scale, b_scale, m_indptr, out) -> None (miss) | rc (int)
PyObject* try_launch6(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "try_launch6 expects 6 arguments");
        return nullptr;
    }
    if (!g_valid) Py_RETURN_NONE;
    for (int i = 0; i < 6; ++i) {
        // borrowed; dead weakref yields Py_None, never equal to a tensor arg
        if (PyWeakref_GetObject(g_wr[i]) != args[i]) Py_RETURN_NONE;
    }
    if (!g_stream_fn) Py_RETURN_NONE;
    PyObject* r = PyObject_CallFunction(g_stream_fn, "i", g_dev);
    if (!r) { PyErr_Clear(); Py_RETURN_NONE; }
    unsigned long long v = PyLong_AsUnsignedLongLong(r);
    Py_DECREF(r);
    if (v == (unsigned long long)-1 && PyErr_Occurred()) { PyErr_Clear(); Py_RETURN_NONE; }
    int rc = fp8umma_run((const void*)(uintptr_t)g_ptr[0], (const void*)(uintptr_t)g_ptr[1],
                         (const float*)(uintptr_t)g_ptr[2], (const float*)(uintptr_t)g_ptr[3],
                         (const int*)(uintptr_t)g_ptr[4], (void*)(uintptr_t)g_ptr[5],
                         g_dims[0], g_dims[1], g_dims[2], g_dims[3], g_dims[4], g_dims[5],
                         (cudaStream_t)(uintptr_t)v);
    return PyLong_FromLong(rc);
}

// store_memo(a, b, a_scale, b_scale, m_indptr, out, m, n, k, e, nsm, stages)
//   -> 1 stored | 0 not-weakrefable/data_ptr failed (caller disables memo)
PyObject* store_memo(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 12) {
        PyErr_SetString(PyExc_TypeError, "store_memo expects 12 arguments");
        return nullptr;
    }
    static PyObject* s_dp = nullptr;
    if (!s_dp) s_dp = PyUnicode_InternFromString("data_ptr");
    PyObject* wr[6];
    uint64_t pv[6];
    for (int i = 0; i < 6; ++i) {
        wr[i] = PyWeakref_NewRef(args[i], nullptr);
        if (!wr[i]) {
            PyErr_Clear();
            for (int j = 0; j < i; ++j) Py_XDECREF(wr[j]);
            return PyLong_FromLong(0);
        }
        PyObject* r = PyObject_CallMethodNoArgs(args[i], s_dp);
        if (!r) {
            PyErr_Clear();
            for (int j = 0; j <= i; ++j) Py_XDECREF(wr[j]);
            return PyLong_FromLong(0);
        }
        pv[i] = PyLong_AsUnsignedLongLong(r);
        Py_DECREF(r);
        if (pv[i] == (uint64_t)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            for (int j = 0; j <= i; ++j) Py_XDECREF(wr[j]);
            return PyLong_FromLong(0);
        }
    }
    int dims[6];
    for (int i = 0; i < 6; ++i) {
        long v = PyLong_AsLong(args[6 + i]);
        if (v == -1 && PyErr_Occurred()) {
            for (int j = 0; j < 6; ++j) Py_XDECREF(wr[j]);
            return nullptr;              // programmer error: propagate
        }
        dims[i] = (int)v;
    }
    clear_slot();
    for (int i = 0; i < 6; ++i) { g_wr[i] = wr[i]; g_ptr[i] = pv[i]; g_dims[i] = dims[i]; }
    g_valid = 1;
    return PyLong_FromLong(1);
}

// set_stream_fn(fn, dev): fn(dev) must return the current stream as int.
PyObject* set_stream_fn(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "set_stream_fn expects 2 arguments");
        return nullptr;
    }
    long dev = PyLong_AsLong(args[1]);
    if (dev == -1 && PyErr_Occurred()) return nullptr;
    Py_XINCREF(args[0]);
    Py_XDECREF(g_stream_fn);
    g_stream_fn = args[0];
    g_dev = (int)dev;
    Py_RETURN_NONE;
}

}  // namespace

// launch_umma(a, b, sa, sb, indptr, out, m, n, k, G, nsm, stages, stream) -> rc
static PyObject* launch_umma(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 13) {
        PyErr_SetString(PyExc_TypeError, "launch_umma expects 13 arguments");
        return NULL;
    }
    unsigned long long v[13];
    for (int i = 0; i < 13; ++i) {
        v[i] = PyLong_AsUnsignedLongLong(args[i]);
        if (v[i] == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    }
    int rc = fp8umma_run((const void*)(uintptr_t)v[0], (const void*)(uintptr_t)v[1],
                         (const float*)(uintptr_t)v[2], (const float*)(uintptr_t)v[3],
                         (const int*)(uintptr_t)v[4], (void*)(uintptr_t)v[5],
                         (int)v[6], (int)v[7], (int)v[8], (int)v[9],
                         (int)v[10], (int)v[11], (cudaStream_t)(uintptr_t)v[12]);
    return PyLong_FromLong(rc);
}

// launch_sync(a, b, sa, sb, indptr, out, m, n, k, G, nsm, stream) -> rc
static PyObject* launch_sync(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 12) {
        PyErr_SetString(PyExc_TypeError, "launch_sync expects 12 arguments");
        return NULL;
    }
    if (fp8gg_run == 0) {
        PyErr_SetString(PyExc_NotImplementedError, "sync engine not linked in this module");
        return NULL;
    }
    unsigned long long v[12];
    for (int i = 0; i < 12; ++i) {
        v[i] = PyLong_AsUnsignedLongLong(args[i]);
        if (v[i] == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    }
    int rc = fp8gg_run((const void*)(uintptr_t)v[0], (const void*)(uintptr_t)v[1],
                       (const float*)(uintptr_t)v[2], (const float*)(uintptr_t)v[3],
                       (const int*)(uintptr_t)v[4], (void*)(uintptr_t)v[5],
                       (int)v[6], (int)v[7], (int)v[8], (int)v[9],
                       (int)v[10], (cudaStream_t)(uintptr_t)v[11]);
    return PyLong_FromLong(rc);
}

static PyMethodDef fp8ggbind_methods[] = {
    {"launch_umma", (PyCFunction)launch_umma, METH_FASTCALL, "fast umma launch"},
    {"launch_sync", (PyCFunction)launch_sync, METH_FASTCALL, "fast sync launch"},
    {"try_launch6", (PyCFunction)try_launch6, METH_FASTCALL, "C-memo hit try (stream in C)"},
    {"store_memo", (PyCFunction)store_memo, METH_FASTCALL, "store weakref identity memo"},
    {"set_stream_fn", (PyCFunction)set_stream_fn, METH_FASTCALL, "set stream getter fn(dev)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fp8ggbind_module = {
    PyModuleDef_HEAD_INIT, "fp8ggbind", NULL, -1, fp8ggbind_methods
};

PyMODINIT_FUNC PyInit_fp8ggbind(void) {
    return PyModule_Create(&fp8ggbind_module);
}
