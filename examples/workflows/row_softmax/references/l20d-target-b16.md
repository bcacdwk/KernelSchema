# B16 L20D target declaration

The task owner explicitly selected the Agate production GPU alias `L20D` on
2026-11-21. This is a new row-softmax target under the original B14 parent
account. The earlier H20 target, requests and rejected diagnostic remain intact.
This document declares a target; it does not certify a worker, dev permission,
namespace isolation, a benchmark or an implementation.

The selected device is `NVIDIA L20D`, CUDA compute capability `[10,3]`,
PyTorch `2.9.0+cu130`, Triton `3.7.1`, and `torch.version.cuda == "13.0"`.
The exact runtime name for the post-diagnostic target is recorded below.
K1 must match its actual complete worker capture against these requirements.

An authenticated, non-forced `GET /v1/env/L20D` returned HTTP 200 on
2026-11-21 with these exact 411 UTF-8 bytes, without a trailing newline:

```json
{"gpu":"L20D","vendor":"nvidia","gpu_model":"NVIDIA L20D","aliases":["L20D","l20d"],"arch":"sm_103","sm_count":148,"total_memory_mb":274113,"driver_version":"580.105.08","toolchain":{"torch":"2.9.0+cu130","cuda":"13.0","triton":"3.7.1","gluon":"3.7.1","cutedsl":"4.6.1","tilelang":"0.1.13","nvcc":"13.1.115","ptxas":"13.1.115"},"source":"cache","probed_at":"2026-11-21T03:00:23.534105+00:00","probe_error":null}
```

The response SHA-256 is
`99b24a8a502c00864ccd8a77c086a8b518520eba37e5c8fd529cc91473d81dd6`.
`GET /v1/env/L20D/capabilities` also returned HTTP 200, 11936 bytes, SHA-256
`fa5a8891b3fea1087e5f43dc2b119ace718b43535df53e6331e89f91f6c90392`.
Both responses say `source=cache`; neither is a K1 worker measurement or a dev
ACL assertion. The capability response reports Triton 3.7.1 and the public dev
timeout ceiling, but no Python/bwrap absolute path or namespace permission.
The target does not pin cached SM count, memory capacity, driver or nvcc.
K1 captures actual device, driver, executable hashes and stack identity; later
qualified jobs must match that capture under the existing channel rules.

The single independent L20D diagnostic subsequently ran as `dv_37ffeb8cd2f2`
with original key `b16_l20d_diagnostic_2ebb6e8cfac34b1d822cba45cb9f5860`.
Its retained terminal response has SHA-256
`2a9ad595c4f1b4d94a54a1825a60aa67c96f6664b3589d13936a049f7656a7a5`:
`succeeded`, `execution_state=stopped`, `cleanup_pending=false`, and
`execution_quarantined=false`. The complete diagnostic stdout reports:

```json
{"name":"NVIDIA L20D        ","architecture":[10,3],"torch":"2.9.0+cu130","triton":"3.7.1","cuda_build":"13.0","mode":"cuda"}
```

The post-diagnostic target preserves those eight trailing spaces in
`runtime.name`; no normalization or comparator relaxation is introduced. The
earlier cache-based target remains retained as a separate preparation artifact.
The interpreter was `/venv/bin/python`, resolving to `/usr/bin/python3.12`,
Python 3.12.3, executable SHA-256
`1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
The trusted diagnostic returned `bwrap=null` from `shutil.which("bwrap")`.
No existing bwrap path was established, so its namespace/readonly probe did not
run. This observation does not prove every filesystem path lacks bwrap. No
tools were installed and the original six-job K1 qualification did not start.
The unqualified Demo therefore carries `remote=null`; successful dev execution
and runtime observation alone do not establish K1 qualification.

Selected immutable references, whose original text accompanies the target:

- `NVIDIA/cuda-samples@0ec046d5e99d7214cc8e9fbac2cc1e66f3019878:Common/helper_cuda.h`
  maps `(major << 4) + minor == 0xa3` to `Blackwell`, supporting the SM103
  architecture name. Its SHA-256 is
  `0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d`.
  The same commit's BSD-3-Clause `LICENSE` is retained. This source does not
  identify the L20D product alias; that name comes from the explicit target and
  the service response above.
- `triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356:docs/programming-guide/chapter-1/introduction.rst`
  explains the blocked SPMD programming model and compiler work scheduling.
- `triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356:python/tutorials/02-fused-softmax.py`
  explains fused stable softmax, row reductions and approximate exponential.
  These MIT-licensed Triton v3.5.1 documents support the algorithm and model;
  they do not establish Triton 3.7.1 or SM103 runtime compatibility. That remains
  subject to actual K1 execution and the independent reviewer.

For the fixed project-authored R0, one CTA owns a complete row, a warp is its
SIMT scheduling group, and thread-private registers retain row values through
their consumers. FP32 max, subtract, exponential, sum and divide must write the
complete supplied output tensor. No tensor-core, TMA, TMEM, asynchronous-copy or
machine-ISA obligation is added. The L20D R0 copy changes only its hardware
Contract; its Design and Realization remain the original assignment.

No L20D row-softmax microbenchmark or measured Schema result exists at this
material-freeze point. Wiki Microbench, Sota.Schema and Profile may point to this
explicit boundary and target declaration; they do not claim measured SOTA.
The original full Task comparator, FP64 golden, 72 basic calls, two-call reuse
trace, four-shape timing and all four fixed negative controls remain mandatory.
Only the original six-job K1 channel qualification can establish its execution
qualification; cached metadata and a read-only diagnostic cannot substitute.
