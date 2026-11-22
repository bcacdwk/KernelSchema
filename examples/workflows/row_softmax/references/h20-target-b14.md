# B14 H20 target declaration

This document fixes a requested row-softmax target. It does not certify a worker,
raw dev permission, namespace isolation, a benchmark, or an implementation.

The selected Agate production GPU alias is `H20`. The requested runtime is
`NVIDIA H20`, CUDA compute capability `[9,0]`, PyTorch `2.9.0+cu130`, Triton
`3.7.1`, and `torch.version.cuda == "13.0"`. K1 must compare its own complete
worker capture against these requirements before execution admission.

On 2026-11-21, an authenticated, non-forced `GET /v1/env/H20` returned HTTP 200
with the following exact 404-byte UTF-8 response (no trailing newline):

```json
{"gpu":"H20","vendor":"nvidia","gpu_model":"NVIDIA H20","aliases":["H20","h20"],"arch":"sm_90","sm_count":78,"total_memory_mb":97356,"driver_version":"580.105.08","toolchain":{"torch":"2.9.0+cu130","cuda":"13.0","triton":"3.7.1","gluon":"3.7.1","cutedsl":"4.6.1","tilelang":"0.1.13","nvcc":"13.1.115","ptxas":"13.1.115"},"source":"cache","probed_at":"2026-11-21T02:00:10.241551+00:00","probe_error":null}
```

The response SHA-256 is
`4f8889f36861266c8489376a60b9747caae7218f8eda3478e8312e6bf1f13aef`.
Its `source=cache` is preserved. This is service metadata, not K1's worker
measurement. The same read-only environment inventory contained H20 pool records
with driver `580.82.07`; no uniform driver/image identity is inferred from the
alias. The target does not pin driver or nvcc versions; K1 still captures actual
driver, executable hashes and stack identity, and subsequent qualified jobs must
match the qualification capture under the channel's existing rules.

For this small task, a CTA is the cooperative work unit assigned one complete
row; a warp is its SIMT scheduling group; registers hold thread-private values.
No tensor-core, asynchronous-copy, TMA, TMEM or machine-ISA obligation is imposed.
The programming target is Triton, with FP32 row-local max, subtract, exponential,
sum, divide and full writes into the supplied output tensor. These are target
constraints; the independent reviewer checks the final source against its exact
Schema and Task rather than treating documentation as implementation proof.

Selected reference sources:

- `NVIDIA/cuda-samples@0ec046d5e99d7214cc8e9fbac2cc1e66f3019878:Common/helper_cuda.h`
  contains the compute capability 9.0 Hopper mapping and device-property helpers.
  Its SHA-256 is `0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d`.
  Preserve the NVIDIA BSD-3-Clause license at the same commit's `LICENSE`.
- `triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356:docs/programming-guide/chapter-1/introduction.rst`
  explains Triton's blocked SPMD execution model and compiler work scheduling.
- `triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356:python/tutorials/02-fused-softmax.py`
  explains fused stable softmax, row reductions and approximate exponential.
  The selected documentation is from v3.5.1 and has MIT permission; it is not a
  claim that target Triton 3.7.1 was installed or exercised by this document.

There is no B14 H20 microbenchmark or measured H20 row-softmax Schema result at
material-freeze time. The Wiki Microbench, Sota.Schema and Profile entries may
point here to preserve that explicit boundary and the target declaration. The
published R0 design is a project-authored assignment; it is not a measured SOTA
claim. The full baseline timing and all four fixed negative controls remain
mandatory for execution qualification. An ordinary Agate allclose result, a
cached device record, or a prior different-device result cannot replace them.

This project-authored declaration contains no authentication material. Its use
is within the existing authorized KS/AKA development task; project-wide release
license decisions remain as documented in `docs/workflows/release-gaps.md`.
