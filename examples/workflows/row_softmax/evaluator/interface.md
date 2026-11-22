# Evaluator interface for K2 / K1.evaluator

This is a non-executable interface handoff. Only golden.py and baseline.py are
implemented in B01; input/check/timing/entry execution, the canonical evaluator
manifest, adapters and receipts belong to K1.evaluator after K2.interface. No
command in this directory currently claims to run the complete Task.

All local file names below are relative to the row_softmax example. They are
material locations, **not Ref objects**. K2 will snapshot bytes under one stable
`artifacts/<publication_id>/` prefix, assign real `Ref={path,sha256}` objects,
build single_file Source manifests, and publish canonical RFC 8785 JSON. Ref
paths throughout a completed bundle are relative to its request root, including
nested manifests. Never submit these draft descriptors as workflow requests.

| Input | Trusted ownership and contract |
| --- | --- |
| operation_id and durable run/request linkage | Supplied by the sole owner before any execution; replay/settlement refers to the same operation |
| Task / candidate Source / evaluator snapshots | Verified canonical Ref inputs; stage immutable actual bytes and retain source-to-native-to-loaded-code identity |
| Task contract and acceptance | task/contract.yaml and acceptance.md; domain and all side effects are hard requirements |
| Complete input and workload specifications | task/inputs.json and workload.json; 36 cases × 2 poison initializers, plus separate state trace |
| Timing definition | task/timing.md; candidate and FP32 baseline use the same full-forward recipe |
| Declared profile and permitted channel | Fixed actual supported device/stack/adapter; no silent device, backend or runtime fallback |
| Baseline Source | evaluator/baseline.py copied byte-for-byte to logical kernel.py in its own Source snapshot; used for native V0 and public performance |
| Budget reservation and original submission locator | Supplied by the owner; evaluator cannot create planner/coder work or exceed authorization |

The candidate module exposes exactly `Model()` and `forward(x=x,out=out)`.
The adapter does not supply extra files as part of C, re-encode C into a
different public identity, or let a wrapper fill/copy/repair out for it. A
first-invocation exception is a build/execution failure, not permission to
replace the starter with the baseline. `starter/kernel.py` is C0; baseline is
V0; native G and HEAD remain distinct. The starter intentionally raises.

The trusted executor must:

1. Reject unsupported shapes/dtypes/layouts and shared x/out storage before
   in-domain evaluation. Instantiate on the exact qualified CUDA device.
2. Generate each FP32 CPU case with the locked torch version. Allocate separate
   x/out backing buffers with declared guards; construct exact storage offsets
   and strides. Keep the original backing and storage alive. Record byte
   digests and signatures, preserving legal offsets through all adapters.
3. For every poison call, save pre-call x bytes and original view signatures,
   then call `golden.softmax_fp64(saved_cpu_fp32.tolist())`. The host passes
   actual saved FP32 values, promoted exactly to Python binary64. The oracle
   returns nested CPU FP64 values; it never reads post-call x or calls softmax.
4. Invoke the exact staged candidate under inference mode and synchronize before
   reading results. Inspect return identity, the original out region, original
   x/backing bytes, both view signatures and all canaries independently.
5. Reject NaN/Inf and compare every original output element with the specified
   FP64 relative-to-golden expression. Keep failures even when another hard
   item passes. Ordinary native allclose/reference Model output cannot replace
   this comparator or conceal storage replacement.
6. Run the two-call existing-case state trace on the same Model/x/out storage,
   host-overwriting x and re-poisoning out between fresh before snapshots. Keep
   its outcomes separate from the 72 basic calls.
7. Measure complete forward using timing.md. Preserve all ordered samples,
   preparation/cold/warmup details and actual environment. Report incomplete
   measurements honestly; no placeholder samples or fabricated speedup.

The candidate cannot read private evaluator source or write evaluator files.
The Task's public Contract/acceptance stay model-visible. Files visible by hash
or path are not automatically authorized for model reads. Runtime isolation and
role/tool allowlist enforcement are later implementation checks, not achieved
by this document or by placing files in separate directories.

The canonical output will be the C-defined `protocol=ks.workflow.v1`,
`kind=evaluation` report with these fields (not an implemented serializer):

| Field | Required facts |
| --- | --- |
| operation_id | Original execution operation, stable through recovery |
| subject | task_sha256, source_sha256, evaluator_sha256, inputs_sha256, environment_sha256, timing_sha256; actual exact bytes, not placeholder or guessed identities |
| build / correctness / performance | Independent pass/fail/unknown/not_run checks with reason and raw evidence; no pass/fail without evidence |
| case_ids | All 36 scheduled IDs, plus separate trace accounting; report attempted/missing calls and poison IDs rather than hiding failures |
| samples | Ref to complete raw ordered samples or null when not measured |
| raw_receipts / usage_receipts | Original adapter/backend facts and metered operation records; preserve unknown statistics and original submission locator |
| environment | Actual reported device/stack/profile/adapter snapshot, not the requested or assumed environment |

Canonical nested field details and extensions must come from K2's single codec;
this table is the handoff, not a competing schema. Include the actual subjects
in selected evidence closure so naked digest strings remain resolvable. Record
source/evaluator load digests and adapter/comparator mappings in raw facts.

Per-call raw facts must include case/trace ID, poison, input digest, original and
post-call view signatures, object/storage checks, original-region write/numeric
checks, x-byte and guard checks, and diagnostics. Record actual shape, offset,
stride and backing region bounds. No numerical status subsumes side effects.

The owner projects performance_goal from complete timing samples using the
Task's fixed goal, and combines **separate** coverage/fidelity evidence. This
evaluator does not decide design fidelity, repair code, promote native G, choose
the next episode or call a planner. Slow-but-correct can remain completed when
all other hard checks pass; unmeasured or unknown required checks cannot qualify.

Interrupted submission uses the original operation/key for query and settlement.
Timeout, cancellation acceptance or 404 does not prove non-execution. Preserve
the fence and unknown usage, avoid re-submission or channel switching, and let
the owner enforce the common budget/recovery contract. No real service behavior
or recovery capability is claimed by this interface-only handoff.
