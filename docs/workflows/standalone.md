# Standalone row-softmax workflow

This source-checkout driver calls `kernelschema.api.design`, `implement` and
`optimize`. It requires no AKA. CPU tests execute the actual driver, files,
Template checks, publications, owner and workflow APIs with explicitly synthetic
role/evaluator transports. This release has **no new business-model or GPU pass**.
The existing K3 CLI qualification is a separate runtime capability record.
The public [protocol entry](protocol.md) explains fixed inputs, checks, accounting
and recovery and links the actual executable contract.

Use the installed source checkout with Python >=3.11, PyYAML and jsonschema.
The tested interpreter is Python 3.13.12; Python 3.11 remains unverified. A complete
wheel/resource installation is not claimed. Torch/Triton are remote evaluator
requirements and are not needed to prepare requests.

```sh
KS=/absolute/path/to/KernelSchema
python "$KS/examples/workflows/row_softmax/run_demo.py" prepare --out /tmp/softmax-demo --start fixed
python "$KS/examples/workflows/row_softmax/run_demo.py" inspect --demo /tmp/softmax-demo
python "$KS/examples/workflows/row_softmax/run_demo.py" run --demo /tmp/softmax-demo
```

These commands work from any cwd. `prepare` invokes no model/GPU. With the default
unbound environment and no evaluation selection, fixed execution publishes an
honest blocked result, keeps the starter, and charges zero calls/jobs. CLI exit 0
means a result was produced; inspect its status/recovery and individual checks.

The five root request files are concrete, hash-verified requests:

| File | Meaning |
| --- | --- |
| `design.request.json` | Public Task/objective/materials, empty draft bindings |
| `fixed.request.json` | Handwritten R0, seed=null → Task starter |
| `partial.request.json` | Same R0 with the explicit manually authored partial Source (Codex, B04) |
| `verify.request.json` | Exact nonempty partial Source, verify_only, zero coder |
| `optimize.request.json` | Current R0 and objective, no invented measurements |

`prepare --start design|fixed|partial|verify` chooses the initial request. Supply
`--schema /complete/raw/directory --receipt /original/delivery.json` to explicitly
consume an existing extraction and its original receipt. No extract operation is
run or modified. Omitting the receipt also permits handwritten/curated raw inputs.
The original R0 remains unbound; a device-bound fixed run must provide a reviewed
Schema whose Contract matches the explicit target. Design derives its own locked
Contract from Task without reading R0 Design as an answer.

A design start executes design → fixed implement and may then run one optimize.
Fixed/partial starts run implement and have the same optimize gate. Only a
completed, qualified implementation advances to optimize for the supplied
objective. Ordinary build/numerical/fidelity failures, unknown checks and budget
exhaustion stop with the original implementation result; they do not establish
a design obstruction. This finite driver does not yet consume independently
authenticated design-obstruction evidence. Verify starts execute only verify_only.
Optimize uses exactly the selected settled current result, feedback and reports;
it can keep/none and stop, keep with implement/verify, or revise then act. A valid
non-none action is actually run. New Schema publication and its next request's
admission are separate persisted phases. New R never inherits an old R pass.
No useful action is fabricated to force a second round.

The planner explicitly chooses one Source, or `task/default_start` for implement
with seed=null. Verify requires exactly one nonempty Source. Original Task/Scope
and current Schema accompany all actions. `material:workflow_objective` and other
jointly granted material IDs can justify action without pretending to be measured
feedback. New REALIZATION.Feedback still needs its distinct measured provenance.

The default owner has at most 10 total model invokes, 8 evaluator jobs and 60
minutes of cumulative active time. Per-request bounds also apply. Initial writing,
coder/reviewer, qualification, failures and unknowns consume the same owner. New
Q/R/out does not create another balance. In-flight unknown is settle_only, never
permission to submit again or change channel. Internal CLI rounds/costs stay in
original receipts; the native USD threshold is not a guaranteed billing cap.

<a id="agate"></a>
## Actual Agate preparation

The built-in channel is `kernelschema.evaluation.channel:workflow_channel`; the
`selection()` helper pins its source and K1 Agate source. No custom user wrapper
is needed. Prepare [explicit target materials](target-environment.md) first.
Supply `--remote remote.json` with exactly the chosen `channel` value
`agate-dev-bwrap-v1`, authorized `gpu` alias and absolute existing remote `python`
and `bwrap` paths. Do not install tools on the worker. The official client must
already be available to the selected interpreter.

```sh
python "$KS/examples/workflows/row_softmax/run_demo.py" prepare \
  --out /tmp/softmax-real --start design --environment /materials/target.json \
  --remote /materials/remote.json
KS_AGATE_CONFIG=/explicit/existing/official/config.json \
  python "$KS/examples/workflows/row_softmax/run_demo.py" qualify --demo /tmp/softmax-real
KS_AGATE_CONFIG=/explicit/existing/official/config.json \
  python "$KS/examples/workflows/row_softmax/run_demo.py" run --demo /tmp/softmax-real
```

`KS_AGATE_CONFIG` is private local configuration selection; credential bytes are
never copied into the bundle or profile. Without it, only the standard existing
`~/.atrex/config.json` is considered. This session did not locate an authenticated
configuration; prior official env access returned HTTP 401. This does not assert
that the user's service lacks GPUs or dev permission.

Qualification uses **six logical jobs** on the same account: a CUDA environment
and fixed input probe, a complete baseline execution/timing/readonly check, then
four fixed negative Sources (no out write, x mutation, wrong value, same out object
with changed storage). Admission rechecks original raw job records and requires
the relevant hard check to reject each negative across the actual K1 adapter.
Probe/baseline alone cannot sign V01. The suite leaves at most two jobs under the
default parent for business implementation and follow-up. Budget-ended work is
partial, not a pass. A failed/unknown suite remains unqualified and retains its
original costs/locator. Repeating `qualify` recovers that same suite.

The full Task comparator, FP64 golden, 72 basic calls, two-call reuse trace and
four-shape timing remain unchanged. The channel uses K1 prepare/submit/query/
settle and the sole OwnerAccount. Native allclose is never substituted. Auth/dev
ACL, replay_policy=never, existing bwrap/namespaces/CUDA, readonly probes, complete
stdout and confirmed remote cleanup must be actually observed. Lost acknowledgment
without job_id remains fenced because the SDK has no read-only key lookup.

## Recovery and selected exports

```sh
python "$KS/examples/workflows/row_softmax/run_demo.py" continue --demo /original/demo
python "$KS/examples/workflows/row_softmax/run_demo.py" export --demo /original/demo --out /new/portable-bundle
```

Business execution and qualification hold the same driver lock and reload the
original state before admission; qualification also serializes callers on its
original child run. Recovery uses `.runtime/demo-state.json` only to find original runs and selected
Ref closures. Child run/operation state, native locator, publication IDs and the
single `.runtime/owner.json` remain authoritative; there is no second ledger.

| Interruption | Recovery |
| --- | --- |
| Fixed request written, child not yet published | Admit the already fixed request on the same owner |
| Child anchor and original backend/owner state initialized, before first result | Continue the original child and its operations |
| Child anchor exists but owner binding was never initialized or is missing | Recovery currently refuses; retained anchor alone cannot distinguish missing state from interrupted initialization |
| Child result published, driver selection not recorded | Replay original child result; do not call models again |
| Writing published / new R not admitted | Retain old result under old R; freeze/admit next fixed request on same owner |
| Model/evaluator operation unknown | Settle original operation; no replacement submit, alternate channel or budget |
| Top-level result projection lost | Reconstruct it from original child facts/publication |
| Required owner/state/native locator lost | Refuse recovery; do not mint a fresh account |

`export` selects actual request/result/Source/report/rule/method closures. It does
not recursively copy private runtime directories, CLI authentication or historical
qualification archives. Exact raw evidence selected by a report remains required;
a redacted explanation is not its replacement. A portable artifact bundle supports
reading/importing; moving it does not promise private-run recovery. Keep original
private run directories for settlement. The small public explanation here is not
an importable real GPU evidence package, because no real GPU run occurred.
