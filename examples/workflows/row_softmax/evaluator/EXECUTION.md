# Implemented evaluator interface (B03)

`interface.md` is the unchanged, hash-pinned B01 historical handoff. This file
describes the implementation. `manifest.json` remains an unbound material
descriptor; `freeze_task` publishes the real canonical evaluator/Task manifests.
The complete implementation has CPU logic and transport tests. **No CUDA,
Agate service, native V0 or R0 execution qualification has been established.**

The six Task modules implement exact CPU FP32 inputs, pre-call FP32 snapshots,
the unchanged CPU binary64 golden, all 72 basic calls, and the separate two-call
reuse trace. Return identity, original out storage/view and full writes, original
x bytes/storage/view, prefix/suffix guards and elementwise tolerance are separate
checks. Actual output is read from retained original backing. Offset 3/5 is
preserved. A wrapper never writes/copies candidate output. Finite tests do not
prove arbitrary device memory safety or continuous-domain correctness.

Timing records construction, first invocation and 20 warmups separately, then
100 complete forward calls per batch, ten AB/BA pairs per shape. All 80 ordered
batch observations survive, with exact decimal start/end nanoseconds, integer
elapsed nanoseconds, median/IQR/min/max/count, and the Task's geometric mean soft
goal. Earlier correctness/constructor work is disclosed as preparation; no
timing is called cold if compilation has already occurred there. Missing or
unusable timing cannot satisfy performance. CPU timing is never CUDA evidence.

## K4 calls

Core/evaluation imports do not import torch or CUDA. These calls require the
source checkout and the K2 `Bundle`, `Publication` and Ref APIs:

```python
from kernelschema.evaluation import freeze_task, stage, RowSoftmaxTask, LocalExecutor
from kernelschema.artifacts import Bundle, Publication

p = Publication(bundle_root)
# environment_ref locates actual environment requirements/capture supplied by owner.
refs = freeze_task(p, environment=environment_ref)
source_ref = p.put_source("candidate", candidate_bytes)
p.commit([refs["task"], source_ref])
bundle = Bundle(bundle_root)
staged = stage(bundle, refs["task"], source_ref, new_stage_directory, scope=scope_ref)
```

The manifest includes the six Task modules and exact `_worker.py`, `_sandbox.py`,
`_controller.py`, `_remote.py` byte Refs. Every staged dependency is checked
against the selected Ref and supported installed implementation. Both candidate
and baseline compile directly from the verified byte buffer in a fresh process;
they never import mutable workspace `kernel.py` or a cached bare `kernel` module.
Actual loaded hashes, frozen inputs, permission probes and actual environment
are checked again at receipt consumption.

`RowSoftmaxTask(bundle, task_ref).validate_scope(schema_ref, scope_ref,
require_hardware_binding=False)` checks actual Template effective nodes/root Contract,
Task coverage IDs (not case IDs), selected shape partitions and dispatch. The
finite supported `when` form is semicolon-separated conjunctions such as
`m in {1,32}; n=128`; absence means whole Task only for root. Unknown natural
language `when`, non-root absence and variants return unknown. Explicit exclusions
fail. `require_hardware_binding=True` additionally checks for missing hardware
fields and unresolved placeholders. Concrete documentation bindings can pass
this static check; current R0's pending bindings remain unknown. Neither value
of this flag verifies referenced documentation, a device or a CUDA profile.
Actual execution qualification belongs to the selected executor and K4's
admission decision. The return has status, mappings,
requested_uncovered, task_uncovered, reasons and limitations. This checks dispatch,
not current-Schema fidelity or free-form mathematical equivalence; K4 still owns
the latter. No second Schema language or automatic prose interpretation exists.

The current evaluator always runs the full four-shape Task: 36 cases/72 basic
calls, the two-call reuse trace and all four timing shapes. It has no subset-only
mode. `RowSoftmaxTask.evaluation_capabilities()` returns `scope_mode="full_task_only"`,
the four `coverage_ids` and `subset_only=False`.
`require_evaluation_scope(scope_ref)` returns those capabilities for full scope
and raises `UnsupportedError` for a partial selection. `validate_scope` invokes
that guard as well, so partial scope cannot receive a passing evaluation
admission. `stage(..., scope=scope_ref)` applies the same guard before creating
the stage directory. K4 must pass its request's scope to staging; omitted scope
is explicitly a full-Task evaluation for standalone callers, never an inferred
subset. Static scope success and evaluator scope support still do not establish
device/profile qualification.

For the concrete CPU logic channel:

```python
executor = LocalExecutor(private_jobs_root, python=selected_python)
ready = executor.prepare(staged, include_timing=True)
facts = executor.submit(staged, operation_id=operation_id,
    submission_key=stable_key, owner=owner_id,
    reservation={"owner": owner_id, "evaluator_jobs": 1, "active_ms": limit_ms})
facts = executor.query(operation_id)
settled = executor.settle(operation_id, Publication(bundle_root))
```

`prepare` freezes actual torch-generated FP32 bytes and environment without
candidate import. `submit` durably stores the original key, stage, owner,
reservation and private job locator before spawning a detached controller.
The controller persists the original terminal after caller disconnection.
The worker installs an all-thread Linux x86_64 seccomp read-only filter before
candidate import. CPU tests prove evaluator write opens and actual candidate
write attempts are denied. This profile blocks compilation/cache writes and
explicitly **does not support CUDA**. It is a filesystem boundary, not protection
against hostile native code modifying the evaluator's own address space.

Unknown operations fence new submissions in the same executor root; repeated
submit only queries. An interrupted pre-spawn intent remains unknown even with
no PID. `settle` returns `new_receipt` plus the same original report/usage row on
replay, and recovers an interrupted publication using its original publication
ID. CPU reports are marked synthetic/qualification=none; numerical/performance
success stays unknown for the CUDA Task. Native raw bytes and CPU failures are
preserved. Lost/corrupt original state is an error, not a fresh budget.

## Agate dev channel and its unverified conditions

The implementation consumes the actual atrex-gpu-gateway skill 0.15 and installed
official client 0.14.37. It uses documented `POST /v1/jobs/dev` and original-job
`GET /v1/jobs/<id>?include_spec=true`. `AgateHTTP` uses SDK Connection/AuthProvider
objects but retains exact HTTP response bytes and performs no POST retries or
legacy-route fallback. It can load an explicitly selected config through
`AgateHTTP.from_config(config_path, profile="prod")`; credentials never enter
requests, manifests or journals. Alternatively supply an existing official
Connection and AuthProvider. Optional client installation follows that skill's
internal package index; the CPU core has no gateway dependency.

```python
from kernelschema.evaluation.agate import AgateHTTP, AgateDevExecutor

transport = AgateHTTP.from_config(config_path)
executor = AgateDevExecutor(private_jobs_root, transport)
executor.prepare(staged, profile=profile, operation_id=operation_id,
    submission_key=stable_key, owner=owner_id, reservation=reservation,
    action="probe")
facts = executor.submit(operation_id)
facts = executor.query(operation_id)
settled = executor.settle(operation_id, Publication(bundle_root))
```

`profile` specifies `channel="agate-dev-bwrap-v1"`, one `gpu` alias, absolute
remote `python` and `bwrap` paths, and for execute the actual `bwrap_sha256`.
Probe captures CUDA environment/FP32 input bytes and existing bwrap identity,
without importing the candidate. It still costs one logical evaluator job.
The caller freezes that actual capture, then prepares a separately authorized
`action="execute"` with `environment` and canonical `frozen_inputs` bytes.

Execution requires remote bwrap already installed and functional, a pinned
binary, usable user/mount/PID namespaces and GPU access. It makes `/` read-only
and allows only an external compile scratch directory to be written (device
nodes are supplied through `/dev`). The worker tests denied write opens on all
staged Python sources before candidate import. The same complete evaluator and
timing run inside this boundary. A missing prerequisite fails; the implementation
does not install bwrap, change the remote image, fall back to CPU/native allclose,
or ask a wrapper to repair out. bwrap availability/permission behavior is **not
observed on an actual Agate worker in B03**.

Dev uses `recycle=true`, `replay_policy="never"`, a stable idempotency key and a
timeout no larger than the caller reservation. Server support for never replay,
dev ACL, complete stdout retention and remote cleanup remain required actual
capabilities. Full results have a compressed, length/hash-checked frame; truncated
or absent frames cannot become passes. Original native error/terminal bytes are
kept even without a valid evaluator report. Unknown GPU milliseconds remain
null; they are not invented from wall time.

Missing acknowledgment/job ID, 404, transport loss, cancellation with pending
cleanup or quarantined execution yields `unknown`/`settle_only`. No new POST or
channel change occurs. This SDK exposes no read-only lookup by idempotency key;
if the whole acknowledgment was lost, automated recovery cannot discover the
job ID and remains fenced. A known original job can later supply its terminal;
the original receipt is absorbed once. Public request/result serialization never
substitutes for the caller's single owner, active clock, cross-channel fence,
budget admission or K2 RunContext intent. Caller must pass their tightest limits
and stop new admissions while any reservation remains unknown.

On 2026-11-20 this session installed only the small client in its isolated venv.
The explicit user home had no `.atrex/config.json`, the shell had no AGATE auth
variables, and actual prod `agate env` returned HTTP 401 missing authentication.
This is **configuration not located in this session**, not evidence that the
user's service lacks devices/dev. No GPU job was submitted and no real capability
or old L20D/H20 result was used to sign K1.real.
