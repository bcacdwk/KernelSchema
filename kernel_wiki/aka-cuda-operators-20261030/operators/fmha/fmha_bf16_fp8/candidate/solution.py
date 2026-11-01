"""Independent CUDA candidate for problem 12 fmha_bf16_fp8 on L20D (sm_103).

All numerical work runs in candidate/fmha.cu, compiled here with load_inline.
This module only reads public metadata, allocates, binds the ABI and preserves
the exact call/return contract of problem/reference.py:

  * causal masking is always on (the frozen forward() exposes no causal argument
    and reference.normalized() therefore defaults it to True),
  * out and lse are written in place and the very same objects are returned,
  * lse is written whenever an lse tensor is supplied, independently of
    return_lse, because the reference does the same and the checker treats both
    as declared mutations,
  * workspace_buffer is used only as split-KV scratch (declared scratch input).

Nothing is cached across calls except allocations and the compiled module; every
forward recomputes from the current input tensors.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_SRC = "fmha.cu"
_DECL = """
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>
void fmha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor block_table,
                  torch::Tensor seq_lens, torch::Tensor cu_q, torch::Tensor out,
                  c10::optional<torch::Tensor> lse, c10::optional<torch::Tensor> sinks,
                  c10::optional<torch::Tensor> scratch, c10::optional<torch::Tensor> mirror,
                  double bmm1_scale, double bmm2_scale, int64_t window_left, int64_t max_q_len,
                  int64_t max_seq_len, int64_t cfg_override, int64_t splits_override,
                  int64_t tps_override);
std::vector<int64_t> fmha_last_plan();
void fmha_replay();
void fmha_set_debug(int64_t chunk, int64_t zp, int64_t graph, int64_t wscan);
void fmha_bind_forward(pybind11::object forward, pybind11::object names);
void fmha_bind_args(pybind11::tuple args, pybind11::object ret);
void fmha_invalidate_fast();
void fmha_set_fast(int64_t on);
pybind11::object fmha_make_fast_call();
"""


def _find_source():
    places = []
    env = os.environ.get("BATCH_CANDIDATE_DIR")
    if env:
        places.append(Path(env) / _SRC)
    places.append(Path(__file__).resolve().parent / _SRC)
    places.append(Path.cwd() / "candidate" / _SRC)
    places.append(Path.cwd() / _SRC)
    for path in places:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{_SRC} not found next to the candidate module; looked in {places}")


_EXT = None


def _extension():
    global _EXT
    if _EXT is None:
        # tcgen05 (5th-gen tensor core) instructions assemble only for the
        # architecture-specific target, so build for 10.3a. The variable is
        # restored afterwards: it is process-global and the frozen reference
        # baseline in the same process must keep the harness's own setting.
        arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3a"
        try:
            _EXT = load_inline(
                name="fmha_bf16_fp8_l20d",
                cpp_sources=[_DECL],
                cuda_sources=[_find_source().read_text()],
                functions=["fmha_forward", "fmha_last_plan", "fmha_replay",
                           "fmha_set_debug", "fmha_bind_forward", "fmha_bind_args",
                           "fmha_invalidate_fast", "fmha_set_fast",
                           "fmha_make_fast_call"],
                extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
                extra_ldflags=["-lcuda"],
                verbose=False,
            )
        finally:
            if arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return _EXT


_EXTENSION = _extension()
_FMHA = _EXTENSION.fmha_forward
_LAST_PLAN = _EXTENSION.fmha_last_plan
_REPLAY = _EXTENSION.fmha_replay
_SET_DEBUG = _EXTENSION.fmha_set_debug
_BIND_FORWARD = _EXTENSION.fmha_bind_forward
_BIND_ARGS = _EXTENSION.fmha_bind_args
_INVALIDATE_FAST = _EXTENSION.fmha_invalidate_fast
_SET_FAST = _EXTENSION.fmha_set_fast

# Tuning overrides, only used by experiments (env keeps the ABI untouched).
_CFG_OVERRIDE = int(os.environ.get("FMHA_CFG", "-1"))
_SPLITS_OVERRIDE = int(os.environ.get("FMHA_SPLITS", "0"))
# Decode per-request split balancing: -1 = promoted auto policy (device-side
# tiles-per-split balancing, host-gated to the measured-win regime; see
# make_plan), -2 = force device mode past the gate, -3 = force uniform splits,
# 0 = legacy host-formula auto target, N > 0 = force its divisor.
_TPS_OVERRIDE = int(os.environ.get("FMHA_TPS", "-1"))
_MIN_SCRATCH = 8 << 20
# Cap on the f16 KV mirror. The mirror removes the in-kernel e4m3->f16 convert
# pass for prefill (26% of prefill time in the staged mimic), but it costs two
# bytes per cached element, so past this size the C++ side falls back to
# converting in-kernel.
_MIRROR_CAP = 8 << 30


# FMHA_HOST_TRIM=0 restores the pre-trim host path for same-job paired A/B.
_TRIM = os.environ.get("FMHA_HOST_TRIM", "1") != "0"


_DBG_CHUNK = 0
_DBG_ZP = -1
_DBG_GRAPH = -1
_DBG_WSCAN = -1
# Bumped on every set_debug so the identity fast path can key on one integer: the
# combine geometry is baked into the cached launch descriptor, so a change must
# force the full path (and C++ drops the cache and any captured graph).
_DBG_GEN = 0
# Set when an override or debug knob drops the C dispatcher's binding. The Python
# identity fast paths below short-circuit before _bind_fast, so without this flag a
# single set_overrides()/set_debug() call would leave the dispatcher permanently
# unbound for the rest of the process -- which is exactly what job 8 measured: only
# the first arm in a sweep kept the C path, and every later fast=1 arm silently ran
# the Python one (host 4.5 us instead of 2.5 us).
_BIND_STALE = True


def set_debug(chunk=0, zp=-1, graph=-1, wscan=-1):
    """Experiment hook: force the combine geometry and the graph/launch choice.

    chunk <= 0, zp < 0, graph < 0 and wscan < 0 restore the promoted selection.
    Unlike the FMHA_COMBINE_ZP / FMHA_NOGRAPH / FMHA_WSCAN environment knobs
    these are per call, so one process can measure both forms on identical
    tensors. Costs the fast path one integer compare.
    """
    global _DBG_CHUNK, _DBG_ZP, _DBG_GRAPH, _DBG_WSCAN, _DBG_GEN
    # The module defaults mirror the C++ ones, so an unchanged request needs no call
    # and must not drop the cached launch descriptor.
    if (int(chunk) == _DBG_CHUNK and int(zp) == _DBG_ZP and int(graph) == _DBG_GRAPH
            and int(wscan) == _DBG_WSCAN):
        return _DBG_CHUNK, _DBG_ZP, _DBG_GRAPH, _DBG_WSCAN
    _DBG_CHUNK, _DBG_ZP, _DBG_GRAPH, _DBG_WSCAN = int(chunk), int(zp), int(graph), int(wscan)
    global _BIND_STALE
    _DBG_GEN += 1
    _SET_DEBUG(_DBG_CHUNK, _DBG_ZP, _DBG_GRAPH, _DBG_WSCAN)
    _INVALIDATE_FAST()
    _BIND_STALE = True
    return _DBG_CHUNK, _DBG_ZP, _DBG_GRAPH, _DBG_WSCAN


def set_fast(on=True):
    """Experiment hook: 0 forces the Python host path for a same-process A/B."""
    _SET_FAST(1 if on else 0)
    return bool(on)


def set_overrides(cfg=None, splits=None, tps=None):
    """Experiment hook: pin config/split choice for same-job A/B measurement."""
    global _CFG_OVERRIDE, _SPLITS_OVERRIDE, _TPS_OVERRIDE
    if cfg is not None:
        _CFG_OVERRIDE = int(cfg)
    if splits is not None:
        _SPLITS_OVERRIDE = int(splits)
    if tps is not None:
        _TPS_OVERRIDE = int(tps)
    # The C dispatcher's identity check covers the argument objects only, so an
    # override change has to drop the binding explicitly or a repeat call would
    # replay the descriptor built for the previous overrides.
    global _BIND_STALE
    _INVALIDATE_FAST()
    _BIND_STALE = True
    return _CFG_OVERRIDE, _SPLITS_OVERRIDE, _TPS_OVERRIDE


class Model(torch.nn.Module):
    """Frozen-signature entry point; see module docstring for the contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scratch = None
        self._ws_id = -1
        self._ws = None
        self._blank = None
        self._key = None
        self._refs = None
        self._mirror = None
        # Identity fast-path slots (see forward); _refs pins every object so no
        # id/identity can be recycled behind an `is` compare.
        self._fq = self._fkv = self._fbt = self._fsl = self._fcq = None
        self._fo = self._fl = self._fs = self._fw = None
        self._fmq = self._fms = self._fwl = self._fb1 = self._fb2 = None
        self._fcfg = self._fsp = self._ftps = None
        self._fdbg = None
        # Give the C dispatcher this instance's bound forward, used for every miss.
        _BIND_FORWARD(self.forward, _PARAMS)

    def _mirror_for(self, k, v, want):
        """Grow-only f16 mirror buffer sized for both paged KV tensors.

        `want` is False for decode-shaped calls, which never select a mirror
        config: allocating up to _MIRROR_CAP bytes that the kernel will not read
        is pure waste, and on a large cache it would be gigabytes of the 32 GiB
        own-memory budget spent on nothing.
        """
        if not want:
            return None
        need = 2 * (k.numel() + v.numel())
        if need <= 0 or need > _MIRROR_CAP:
            return None
        buf = self._mirror
        if buf is None or buf.numel() < need or buf.device != k.device:
            buf = torch.empty(need, dtype=torch.uint8, device=k.device)
            self._mirror = buf
        return buf

    def _bind_fast(self, local_map, ret):
        """Publish argument identity + the return object for the C dispatcher.

        Only the validated full path calls this: its tensors are caller-owned and
        pinned by _refs. The defensive _general path never binds, so a descriptor
        built from a temporary it created can never be replayed.
        """
        global _BIND_STALE
        try:
            _BIND_ARGS(tuple(local_map[n] for n in _PARAMS), ret)
            _BIND_STALE = False
        except Exception:
            _INVALIDATE_FAST()
            _BIND_STALE = True
        return ret

    @property
    def last_plan(self):
        """Launch plan of the most recent forward (experiment read-back only)."""
        return _LAST_PLAN()

    # ---- scratch and placeholder handling (cached, allocation-free per call) ----
    def _blank_tensor(self, device):
        blank = self._blank
        if blank is None or blank.device != device:
            blank = torch.empty(0, device=device, dtype=torch.float32)
            self._blank = blank
        return blank

    def _pick_scratch(self, workspace_buffer, q):
        """Prefer the declared scratch input; fall back to a cached allocation."""
        if workspace_buffer is not None and torch.is_tensor(workspace_buffer):
            if workspace_buffer.is_cuda and (
                workspace_buffer.numel() * workspace_buffer.element_size() >= _MIN_SCRATCH
            ):
                self._ws = workspace_buffer
                self._ws_id = id(workspace_buffer)
                return workspace_buffer
        if self._scratch is None or self._scratch.device != q.device:
            self._scratch = torch.empty(64 << 20, dtype=torch.uint8, device=q.device)
        self._ws = None
        self._ws_id = -2
        return self._scratch

    def forward(
        self,
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        window_left=-1,
        out=None,
        out_dtype=None,
        o_sf_scale=None,
        o_sf_vec_size=None,
        sinks=None,
        kv_layout="HND",
        enable_pdl=None,
        backend="auto",
        q_len_per_req=1,
        o_scale=1.0,
        mask=None,
        max_q_len=None,
        cum_seq_lens_q=None,
        skip_softmax_threshold_scale_factor=None,
        kv_cache_sf=None,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        bmm1_scale_log2=None,
        multi_ctas_kv_counter_buffer=None,
        enable_block_sparse_attention=False,
        cp_world=1,
        cp_rank=0,
        causal_seqlens_kv_global=None,
        bf16q_fp8kv_transform_mode=None,
        request_order=None,
        request_order_plan=None,
    ):
        # Fast path: everything the frozen input builder produces (3-D THD query,
        # explicit cum_seq_lens_q/max_q_len, caller-owned out). Tensor metadata is
        # read in C++ because each Python tensor attribute call costs ~1us and the
        # small decode shapes run in tens of microseconds total.
        #
        # Identity-key check runs FIRST: the stored key was only ever produced by
        # a full-path call that validated cum_seq_lens_q/out/max_q_len presence,
        # int max_seq_len, HND layout and a 3-D query, and the same object ids
        # guarantee those properties still hold -- so a hit skips re-deriving
        # them (query.dim() and friends cost ~0.5-1us per call, which is 5-8%
        # of the official tiny-decode medians). self._refs pins every keyed
        # object, so no id can be recycled behind the comparison.
        # The override triple is part of the identity: a descriptor replay is
        # only valid for the SAME overrides it was built with. Including them
        # (instead of requiring overrides OFF) lets experiment arms keep the
        # ~4.5us fast path -- without this, every override arm paid the ~29us
        # full host path and eager A/B medians were host-contaminated (K27
        # finding: t7c/c145 looked +35%..+180% slow; their kernels win).
        if _TRIM:
            # Short-circuit `is` chain: no tuple is built per call (~0.4-0.7us
            # saved; the slots were stored by a validated full-path call and
            # _refs pins every object, so no id can be recycled behind it).
            if (query is self._fq and kv_cache is self._fkv
                    and block_tables is self._fbt and seq_lens is self._fsl
                    and cum_seq_lens_q is self._fcq and out is self._fo
                    and lse is self._fl and sinks is self._fs
                    and workspace_buffer is self._fw
                    and max_q_len == self._fmq and max_seq_len == self._fms
                    and window_left == self._fwl and bmm1_scale == self._fb1
                    and bmm2_scale == self._fb2
                    and _CFG_OVERRIDE == self._fcfg
                    and _SPLITS_OVERRIDE == self._fsp and _TPS_OVERRIDE == self._ftps
                    and _DBG_GEN == self._fdbg
                    and kv_layout == "HND" and type(max_seq_len) is int):
                _REPLAY()
                ret = (out, lse) if return_lse else out
                return self._bind_fast(locals(), ret) if _BIND_STALE else ret
        else:
            key = (id(query), id(kv_cache), id(block_tables), id(seq_lens),
                   id(cum_seq_lens_q), id(out), id(lse), id(sinks), id(workspace_buffer),
                   max_q_len, max_seq_len, window_left, bmm1_scale, bmm2_scale,
                   _CFG_OVERRIDE, _SPLITS_OVERRIDE, _TPS_OVERRIDE, _DBG_GEN)
            if (key == self._key and kv_layout == "HND" and type(max_seq_len) is int):
                _REPLAY()
                ret = (out, lse) if return_lse else out
                return self._bind_fast(locals(), ret) if _BIND_STALE else ret
        if (
            cum_seq_lens_q is not None
            and out is not None
            and max_q_len is not None
            and type(max_seq_len) is int
            and kv_layout == "HND"
            and query.dim() == 3
        ):
            # Repeat calls with the identical argument objects reuse the launch
            # descriptor built by C++ on the first call. Identity is checked, not
            # content: the kernels re-read the current tensor contents every call,
            # and the references below keep the ids from being recycled.
            ws = self._ws
            if id(workspace_buffer) != self._ws_id:
                ws = self._pick_scratch(workspace_buffer, query)
            # Decode never selects a mirror config, so do not allocate one: the
            # largest caches in the domain are decode caches and the mirror is
            # two bytes per cached element.
            packed_rows = max_q_len * (query.shape[1] // kv_cache[0].shape[1])
            mirror = self._mirror_for(kv_cache[0], kv_cache[1], packed_rows > 128)
            _FMHA(query, kv_cache[0], kv_cache[1], block_tables, seq_lens, cum_seq_lens_q,
                  out, lse, sinks, ws, mirror, bmm1_scale, bmm2_scale, window_left,
                  max_q_len, max_seq_len, _CFG_OVERRIDE, _SPLITS_OVERRIDE, _TPS_OVERRIDE)
            self._key = None if _TRIM else key
            self._fq, self._fkv, self._fbt = query, kv_cache, block_tables
            self._fsl, self._fcq, self._fo = seq_lens, cum_seq_lens_q, out
            self._fl, self._fs, self._fw = lse, sinks, workspace_buffer
            self._fmq, self._fms, self._fwl = max_q_len, max_seq_len, window_left
            self._fb1, self._fb2 = bmm1_scale, bmm2_scale
            self._fcfg, self._fsp, self._ftps = _CFG_OVERRIDE, _SPLITS_OVERRIDE, _TPS_OVERRIDE
            self._fdbg = _DBG_GEN
            self._refs = (query, kv_cache, block_tables, seq_lens, cum_seq_lens_q, out,
                          lse, sinks, workspace_buffer, ws, mirror)
            return self._bind_fast(locals(), (out, lse) if return_lse else out)
        return self._general(
            query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
            bmm1_scale, bmm2_scale, window_left, out, sinks, kv_layout, q_len_per_req,
            max_q_len, cum_seq_lens_q, lse, return_lse,
        )

    # The harness invokes model(...) directly; aliasing __call__ to forward
    # skips nn.Module._call_impl's (empty) hook machinery, worth ~0.5-1.5us on
    # the host-bound tiny shapes. Semantics are identical: no hooks are
    # registered on this module and forward's signature is the frozen ABI.
    # Kept as the plain alias so the module stays correct (and only slower) if the
    # C dispatcher below cannot be installed. The harness invokes model(...)
    # directly, and Model.__call__ is replaced at module scope by the extension's
    # METH_FASTCALL dispatcher, which replays the cached launch descriptor after an
    # argument-identity check and vectorcalls this forward on any mismatch.
    __call__ = forward

    def _general(self, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
                 bmm1_scale, bmm2_scale, window_left, out, sinks, kv_layout, q_len_per_req,
                 max_q_len, cum_seq_lens_q, lse, return_lse):
        """Defensive path for 4-D queries, missing host metadata or NHD layouts."""
        if isinstance(kv_cache, (tuple, list)):
            k, v = kv_cache[0], kv_cache[1]
        else:
            k, v = kv_cache[0], kv_cache[1]
        if kv_layout != "HND":
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        q = query
        out_view = out
        if q.dim() == 4:
            batch, per_q = q.shape[0], q.shape[1]
            q = q.reshape(batch * per_q, q.shape[2], q.shape[3])
            if out is not None:
                out_view = out.reshape(batch * per_q, out.shape[2], out.shape[3])
            if cum_seq_lens_q is None:
                cum_seq_lens_q = (
                    torch.arange(batch + 1, dtype=torch.int32, device=q.device) * per_q
                ).to(torch.int32)
            if max_q_len is None:
                max_q_len = per_q
        if cum_seq_lens_q is None:
            batch = seq_lens.shape[0]
            per = (
                q_len_per_req
                if isinstance(q_len_per_req, int) and q_len_per_req > 0
                else q.shape[0] // max(batch, 1)
            )
            cum_seq_lens_q = (
                torch.arange(batch + 1, dtype=torch.int32, device=q.device) * per
            ).to(torch.int32)
        if max_q_len is None:
            max_q_len = q.shape[0]
        if max_seq_len is None:
            max_seq_len = block_tables.shape[1] * k.shape[2]
        elif torch.is_tensor(max_seq_len):
            max_seq_len = int(max_seq_len.item())
        if out_view is None:
            out_view = torch.empty_like(q)
        _FMHA(q, k, v, block_tables, seq_lens, cum_seq_lens_q, out_view, lse, sinks,
              self._pick_scratch(workspace_buffer, q),
              self._mirror_for(k, v, max_q_len * (q.shape[1] // k.shape[1]) > 128), bmm1_scale,
              bmm2_scale, window_left, max_q_len, max_seq_len, _CFG_OVERRIDE,
              _SPLITS_OVERRIDE, _TPS_OVERRIDE)
        return (out, lse) if return_lse else out


# ---- C-level repeat-call dispatcher ---------------------------------------
# Installed as the class __call__ so the harness's model(**kwargs) goes straight to
# C: argument-identity check, then a replay of the cached launch descriptor. Any
# mismatch vectorcalls Model.forward above, which re-validates and rebinds. The
# canonical order is forward's own signature, which the frozen input builder's dict
# mirrors key for key, so a normal call binds on its first full-path pass.
_PARAMS = tuple(
    Model.forward.__code__.co_varnames[1:Model.forward.__code__.co_argcount]
)
try:
    _FAST_CALL = _EXTENSION.fmha_make_fast_call()
except Exception:  # pragma: no cover - keep the module importable without it
    _FAST_CALL = None
if _FAST_CALL is not None:
    Model.__call__ = _FAST_CALL
