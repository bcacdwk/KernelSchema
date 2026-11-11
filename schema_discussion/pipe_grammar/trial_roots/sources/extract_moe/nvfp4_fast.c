/* J5: the cached call, in C, behind a descriptor assigned to Model.__call__.
 *
 * Why this exists. The official eager protocol times
 *
 *     reset(); start.record(); model(**kwargs); end.record(); end.synchronize()
 *
 * with the start event executing on an otherwise idle device, and the engine
 * drives candidates as `model(**inputs.kwargs)` with args=() -- 36 keyword
 * arguments (_runtime.py: "The returned ModelInputs carries the tensor dict as
 * kwargs (args=()), so callers drive forward with model(**inputs.kwargs)").
 * Every microsecond of host work ahead of cudaGraphLaunch is therefore
 * serialised in front of the GPU work and lands one-to-one on the measured
 * latency. J1 (dv_73352f43a08e) cut that from 6.93 to 4.45 us and priced what is
 * left: 2.78-2.94 us inside moe_run, ~0.75 us binding 36 keyword arguments to 36
 * named parameters, ~0.33 us for the 16-term identity chain, ~0.2 us of ctypes
 * marshalling. `__call__ = forward`, which the candidate already does, removed
 * nn.Module._call_impl (2.97 us here) but nothing can remove the binding from a
 * PYTHON callee: CPython builds a frame and resolves every name. A C-level
 * callee receives the kwargs dict by reference and looks up only the fifteen
 * names it verifies.
 *
 * Why a descriptor. `instance(...)` goes through slot_tp_call, which binds the
 * attribute it finds only if that attribute implements tp_descr_get -- a plain
 * PyCFunction assigned to __call__ is called WITHOUT the instance (verified on
 * this interpreter: `B.__call__ = abs; B()(-3)` returns 3, while the same with a
 * python function receives the instance as its first argument). So the entry
 * point is a descriptor whose __get__ returns a small bound object holding the
 * instance, and that object's tp_call does the work. One allocation per call,
 * against ~0.75 us of argument binding.
 *
 * State. There is none: the entry is read from the instance's own `_hot`
 * attribute -- exactly the tuple the J1 python path publishes -- so its lifetime
 * is the model's and nothing here can outlive it. That matters because a global
 * entry would keep a deleted model's workspace (hundreds of MB on the large
 * shapes) alive past its shape, and the harness measures ten shapes in one
 * process.
 *
 * Safety. This file binds nothing numerical: every device byte stays in
 * nvfp4_moe.cu, and the check performed is the same identity check J1 promoted,
 * evaluated in C. The eleven objects whose data pointers enter the ABI plus the
 * four scalars that do are compared against the published entry; anything else
 * -- a positional argument, a missing key, a different object or scalar, a
 * changed stream, an unpublished or foreign-shaped entry, an unbound name list --
 * falls through to forward(), which rebuilds through the fully validated slow
 * path. Every failure mode of a mismatch between the name list and the published
 * tuple is a FALLBACK, never a wrong result: a name that does not resolve makes
 * the lookup return NULL and we fall back, and two names swapped consistently in
 * both lists still compare each object against itself. `_env_snap` is published
 * at index 15 of the verify tuple for the python path and is written exactly
 * once, in __init__, before any entry exists, so it cannot differ from the value
 * an entry was published with.
 *
 * The 21 arguments not verified have no ABI field and are read nowhere else in
 * solution.py (checked by counting every occurrence of each); that invariant is
 * recorded at _HOT_KEYS there and must be re-checked if the device side ever
 * starts to depend on one.
 *
 * No threads are created, no evaluator function is touched, and the value
 * returned is the same prebuilt [output] list the python path returns.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* Verify-tuple layout published by forward(): eleven objects, then four
 * scalars, then (index 15) the env snapshot. Must match _HOT_KEYS. */
#define NOBJ 11
#define NSCAL 4
#define NVER (NOBJ + NSCAL)

/* _hot tuple layout (solution.py, J1; CFAST2 appends 7):
 *   0 verify tuple   1 moe_run   2 int address   3 prebuilt [output]
 *   4 the ABI struct (keepalive for 2)   5 [stream, dev_idx]   6 keepalive
 *   7 raw C address of moe_run (int; CFAST2 direct call, 0 = unparsable) */
#define HOT_N 8

static PyObject *g_name[NVER];
static PyObject *g_rawstream = NULL;      /* torch._C._cuda_getCurrentRawStream */
static PyObject *g_s_forward = NULL;
static PyObject *g_s_hot = NULL;
static PyObject *g_s_last_abi = NULL;
static long long g_hits = 0, g_fallbacks = 0;
/* Why a call fell back, so a variant that never takes the C path reports the
 * reason instead of just a null result. */
enum { R_ENTRY, R_SHAPE, R_OBJ, R_SCAL, R_STREAM, R_NREASON };
static long long g_reason[R_NREASON];
/* CFAST2 (host fast-path bundle, priced by dv_e64ce57ef7cc's HOSTFAST win):
 * 1 = call moe_run through its raw C address (hot[7]) instead of the ctypes
 * function object (~0.2 us marshalling per call, J1's measurement) and skip
 * the per-call _last_abi setattr (~0.10 us) -- the publish path in
 * solution.py already stores this same ABI object on the instance and
 * republishes whenever the ABI is rebuilt, so the per-call write is
 * redundant. 0 = the exact J5 behavior. Set once at bind() from
 * NVFP4_MOE_CFAST2 (default ON), sticky per process like HOSTFAST, so A/B
 * arms are separate processes. Host-side only: no device byte changes;
 * digests identical either way. */
static int g_cfast2 = 1;

/* bind(names, rawstream): the fifteen keyword names in verify-tuple order plus
 * the raw current-stream getter. Passed in from solution.py so that file stays
 * the single source of truth for the argument list. */
static PyObject *fb_bind(PyObject *self, PyObject *args) {
    PyObject *names, *rawstream;
    int cfast2 = 1;
    if (!PyArg_ParseTuple(args, "OO|i", &names, &rawstream, &cfast2)) return NULL;
    g_cfast2 = cfast2 ? 1 : 0;
    if (!PyCallable_Check(rawstream)) {
        PyErr_SetString(PyExc_TypeError, "bind: rawstream is not callable");
        return NULL;
    }
    if (!PyTuple_Check(names) || PyTuple_GET_SIZE(names) != NVER) {
        PyErr_Format(PyExc_ValueError, "bind needs a %d-tuple of names, got %zd",
                     NVER, PyTuple_Check(names) ? PyTuple_GET_SIZE(names) : -1);
        return NULL;
    }
    for (int i = 0; i < NVER; ++i) {
        PyObject *o = PyTuple_GET_ITEM(names, i);
        const char *u;
        PyObject *intern;
        if (!PyUnicode_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "bind: names must be str");
            return NULL;
        }
        u = PyUnicode_AsUTF8(o);
        if (!u) return NULL;
        intern = PyUnicode_InternFromString(u);
        if (!intern) return NULL;
        Py_XDECREF(g_name[i]);
        g_name[i] = intern;
    }
    Py_INCREF(rawstream);
    Py_XSETREF(g_rawstream, rawstream);
    Py_RETURN_NONE;
}

/* stats() -> (bound, hits, fallbacks). An experiment needs this to prove the C
 * path actually ran: a variant that silently falls back on every call would
 * otherwise report as "no effect" and be recorded as a closed axis. */
static PyObject *fb_stats(PyObject *self, PyObject *noarg) {
    int bound = (g_rawstream != NULL);
    for (int i = 0; i < NVER; ++i) if (!g_name[i]) bound = 0;
    return Py_BuildValue("(iLL(LLLLLL))", bound, g_hits, g_fallbacks,
                         g_reason[R_ENTRY], g_reason[R_SHAPE], g_reason[R_OBJ],
                         g_reason[R_SCAL], g_reason[R_STREAM], (long long)0);
}

/* _hot[5] is a list and _hot[0] is a tuple; both are indexed positionally. */
static PyObject *seq_item(PyObject *o, Py_ssize_t i) {
    return PyList_CheckExact(o) ? PyList_GET_ITEM(o, i) : PyTuple_GET_ITEM(o, i);
}

/* The whole cached call. Returns NULL with an exception set, or the result. */
static PyObject *fast_call(PyObject *inst, PyObject *args, PyObject *kw) {
    if (g_rawstream == NULL || kw == NULL || PyTuple_GET_SIZE(args) != 0) {
        ++g_reason[R_ENTRY];
    } else {
        PyObject *hot = PyObject_GetAttr(inst, g_s_hot);
        if (hot == NULL) {
            PyErr_Clear();                     /* no entry: fall back */
            ++g_reason[R_SHAPE];
        } else if (PyTuple_CheckExact(hot) && PyTuple_GET_SIZE(hot) == HOT_N) {
            PyObject *v = PyTuple_GET_ITEM(hot, 0);
            /* _hot[5] is a LIST, not a tuple: forward() mutates st[0] in place
             * when the stream changes. Requiring a tuple here rejected every
             * call, which is what job dv_028ae8f0de7b measured as "the C path
             * is slower" -- stats() reported hits=0, fallbacks=366, i.e. the
             * A/B had priced the python path plus descriptor overhead. */
            PyObject *st = PyTuple_GET_ITEM(hot, 5);
            int ok = (PyTuple_CheckExact(v) && PyTuple_GET_SIZE(v) > NVER
                      && (PyList_CheckExact(st) ? PyList_GET_SIZE(st) >= 2
                                                : (PyTuple_CheckExact(st)
                                                   && PyTuple_GET_SIZE(st) >= 2))
                      && PyLong_CheckExact(seq_item(st, 0)));
            if (!ok) ++g_reason[R_SHAPE];
            for (int i = 0; i < NOBJ && ok; ++i) {
                PyObject *got = PyDict_GetItemWithError(kw, g_name[i]);
                if (got == NULL) {
                    if (PyErr_Occurred()) { Py_DECREF(hot); return NULL; }
                    ok = 0;                    /* absent key -> rebuild */
                } else if (got != PyTuple_GET_ITEM(v, i)) {
                    ok = 0;
                    ++g_reason[R_OBJ];
                }
            }
            for (int i = 0; i < NSCAL && ok; ++i) {
                PyObject *got = PyDict_GetItemWithError(kw, g_name[NOBJ + i]);
                int r;
                if (got == NULL) {
                    if (PyErr_Occurred()) { Py_DECREF(hot); return NULL; }
                    ok = 0;
                } else {
                    r = PyObject_RichCompareBool(got,
                                                 PyTuple_GET_ITEM(v, NOBJ + i),
                                                 Py_EQ);
                    if (r < 0) { Py_DECREF(hot); return NULL; }
                    if (!r) { ok = 0; ++g_reason[R_SCAL]; }
                }
            }
            if (ok) {
                /* A changed stream is part of the ABI, so forward() has to
                 * rebuild it; its own python fast path refreshes the boxed
                 * value, so the next call comes back here. */
                /* seq_item, not PyTuple_GET_ITEM: st is the mutable list
                 * forward() refreshes in place. Using the tuple macro here is
                 * what aborted the pre-flight self-test on job
                 * dv_f4e33cc35d7c -- an assertion in a debug build, undefined
                 * behaviour in a release one. */
                long long prev = PyLong_AsLongLong(seq_item(st, 0));
                PyObject *s = PyObject_CallOneArg(g_rawstream, seq_item(st, 1));
                long long cur;
                if (!s) { Py_DECREF(hot); return NULL; }
                cur = PyLong_AsLongLong(s);
                Py_DECREF(s);
                if (cur == -1 && PyErr_Occurred()) { Py_DECREF(hot); return NULL; }
                if (cur != prev) { ok = 0; ++g_reason[R_STREAM]; }
            }
            if (ok) {
                long code = 0;
                if (g_cfast2) {
                    /* CFAST2: call moe_run through its raw C address (hot[7],
                     * published by solution.py via ctypes.cast) instead of the
                     * ctypes function object -- J1 priced ctypes marshalling
                     * at ~0.2 us per call -- and skip the per-call _last_abi
                     * setattr (~0.10 us, J1's measurement): the publish path
                     * already stores this same ABI object on the instance and
                     * republishes whenever the ABI is rebuilt, so the
                     * per-call write was always redundant. Environ-
                     * independent savings, unlike HOSTFAST's getenv caching
                     * (officially neutral under the engine's sanitized
                     * environ, dv_421566e63cb5). */
                    long long fn = PyLong_AsLongLong(PyTuple_GET_ITEM(hot, 7));
                    long long aa;
                    if ((fn == -1 || fn == 0) && PyErr_Occurred()) {
                        Py_DECREF(hot); return NULL;
                    }
                    aa = PyLong_AsLongLong(PyTuple_GET_ITEM(hot, 2));
                    if ((aa == -1 || aa == 0) && PyErr_Occurred()) {
                        Py_DECREF(hot); return NULL;
                    }
                    if (fn == 0 || aa == 0) {
                        ok = 0; ++g_reason[R_SHAPE];   /* bad publish: rebuild */
                    } else {
                        code = ((int (*)(const void *))(intptr_t)fn)
                                   ((const void *)(intptr_t)aa);
                    }
                } else {
                    PyObject *r = PyObject_CallOneArg(PyTuple_GET_ITEM(hot, 1),
                                                      PyTuple_GET_ITEM(hot, 2));
                    if (!r) { Py_DECREF(hot); return NULL; }
                    code = PyLong_AsLong(r);
                    Py_DECREF(r);
                    if (code == -1 && PyErr_Occurred()) {
                        Py_DECREF(hot); return NULL;
                    }
                }
                if (ok) {
                    PyObject *outl;
                    if (code != 0) {
                        PyErr_Format(PyExc_RuntimeError,
                                     "moe_run failed with CUDA error %ld", code);
                        Py_DECREF(hot);
                        return NULL;
                    }
                    if (!g_cfast2) {
                        /* Diagnostics only, and PyObject_GenericSetAttr is
                         * exactly object.__setattr__: J1 measured ~0.10 us
                         * for it against ~1.00 for nn.Module.__setattr__. */
                        if (PyObject_GenericSetAttr(inst, g_s_last_abi,
                                                    PyTuple_GET_ITEM(hot, 4)) < 0) {
                            Py_DECREF(hot);
                            return NULL;
                        }
                    }
                    ++g_hits;
                    outl = PyTuple_GET_ITEM(hot, 3);
                    Py_INCREF(outl);
                    Py_DECREF(hot);
                    return outl;
                }
            }
            Py_DECREF(hot);
        } else {
            Py_DECREF(hot);
        }
    }
    ++g_fallbacks;
    {
        PyObject *f = PyObject_GetAttr(inst, g_s_forward);
        PyObject *r;
        if (!f) return NULL;
        r = PyObject_Call(f, args, kw);
        Py_DECREF(f);
        return r;
    }
}

/* ---- the bound object a descriptor lookup returns ------------------------- */
typedef struct {
    PyObject_HEAD
    PyObject *inst;
} BoundObj;

static void bound_dealloc(PyObject *o) {
    BoundObj *b = (BoundObj *)o;
    Py_XDECREF(b->inst);
    Py_TYPE(b)->tp_free(b);
}

static PyObject *bound_call(PyObject *self, PyObject *args, PyObject *kw) {
    return fast_call(((BoundObj *)self)->inst, args, kw);
}

static PyTypeObject BoundType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nvfp4_fast._Bound",
    .tp_basicsize = sizeof(BoundObj),
    .tp_dealloc = bound_dealloc,
    .tp_call = bound_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

/* ---- the descriptor assigned to Model.__call__ ---------------------------- */
static PyObject *call_descr_get(PyObject *self, PyObject *obj, PyObject *type) {
    BoundObj *b;
    if (obj == NULL) {
        Py_INCREF(self);
        return self;                            /* accessed on the class */
    }
    b = PyObject_New(BoundObj, &BoundType);
    if (b == NULL) return NULL;
    Py_INCREF(obj);
    b->inst = obj;
    return (PyObject *)b;
}

static PyTypeObject CallType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nvfp4_fast.Call",
    .tp_basicsize = sizeof(PyObject),
    .tp_descr_get = call_descr_get,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyObject *g_call_singleton = NULL;

static PyMethodDef methods[] = {
    {"bind", fb_bind, METH_VARARGS,
     "bind(names, rawstream): the 15 verified kwarg names and the stream getter"},
    {"stats", fb_stats, METH_NOARGS,
     "(bound, hits, fallbacks, (entry, shape, object, scalar, stream, spare))"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "nvfp4_fast",
    "C-level cached-call entry point for nvfp4_fused_moe (J5). Binds no "
    "numerical work; every device byte stays in nvfp4_moe.cu.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_nvfp4_fast(void) {
    PyObject *m;
    g_s_forward = PyUnicode_InternFromString("forward");
    g_s_hot = PyUnicode_InternFromString("_hot");
    g_s_last_abi = PyUnicode_InternFromString("_last_abi");
    if (!g_s_forward || !g_s_hot || !g_s_last_abi) return NULL;
    if (PyType_Ready(&BoundType) < 0) return NULL;
    if (PyType_Ready(&CallType) < 0) return NULL;
    m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    g_call_singleton = PyObject_New(PyObject, &CallType);
    if (!g_call_singleton) { Py_DECREF(m); return NULL; }
    if (PyModule_AddObject(m, "Call", g_call_singleton) < 0) {
        Py_DECREF(g_call_singleton);
        g_call_singleton = NULL;
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(g_call_singleton);                /* module holds one ref */
    return m;
}
