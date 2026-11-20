# Fixed FA4 F1 engineering entry

This case freezes the existing four-node Schema and original v55 Source into
normal `ks.workflow.v1` objects. CPU preparation and tests do not qualify a
model, GPU, transport, numerical result, timing result or Schema fidelity.
The source and Schema commits remain distinct; see [materials.json](materials.json).

Reference `api.implement` now supports this fixed F1 with complete Source trees,
typed K3 output, whole-source fidelity and an explicit FA4 Agate channel. These
are CPU-tested engineering paths. Real original-stack/channel/model/fidelity
qualification is still absent; AKA has not acquired this support.

From an installed source checkout with its existing Python/PyYAML dependencies:

```sh
KS=/absolute/path/to/KernelSchema
python -B "$KS/examples/workflows/fa4_decode/prepare.py" --out /tmp/fa4-f1-prepared
```

The installed-checkout command works from any cwd. The checkout must contain
both fixed commits in materials.json; preparation does not fetch missing objects.
It reads the 27 fixed Git blobs, checks their blob IDs/SHA-256/size/mode,
freezes real Task/evaluator/Materials/Scope/Schema/Source Refs, then installs a
complete stage. It imports no torch/CUDA/source module and submits no work.
`prepared.json` records all Refs plus the stage identity. `request.json` is an
original-Source `verify_only` request with zero calls/jobs and an explicitly
unqualified AKA profile advertising no execution capability. It is reviewable,
and cannot silently launch through the row-softmax channel. Execution needs a
new request with the actual adapter/profile, budgets and authority.

To create that reference request without launching anything, pass an explicit
`ClaudeProfile.as_dict()` JSON and ordinary finite budget JSON. An evaluation
selection can be made with `kernelschema.evaluation.fa4_channel.selection`.
No executable probe, credential search or model invocation occurs in prepare:

```sh
python -B "$KS/examples/workflows/fa4_decode/prepare.py" --out /tmp/fa4-reference \
  --claude-profile /selected/claude-profile.json --limits /selected/limits.json \
  --evaluation /selected/fa4-evaluation.json --intent original_verify
python -m kernelschema implement --request /tmp/fa4-reference/request.json --out /tmp/fa4-run
python -m kernelschema implement --continue /tmp/fa4-run
```

`schema_rebuild` selects no seed/starter and supplies no baseline code to the
coder. `seed_continuation` requires `--seed-bundle` and `--seed-ref` (a JSON
Source Ref) and supplies exactly that full Source. New C snapshots omit deleted
files, preserve their own per-file modes/digests, and are re-evaluated under the
same fixed Schema. `original_verify` selects the exact original C, performs no
coder call/edit/promotion, and still needs evaluator/reviewer budget. An explicit
current candidate can also be verified by an ordinary `mode=verify_only` request
whose nonempty `seed` is that C. Missing qualification returns blocked before
coder/evaluator admission; a zero-budget preparation is not the whole workflow.

The independent original-stack entry is callable in a **fresh process** on the
selected original environment. Use `stage.sha256` from prepared.json literally:

```sh
/path/to/original/python -I -B /tmp/fa4-f1-prepared/stage/evaluator/entry.py \
  --stage /tmp/fa4-f1-prepared/stage \
  --stage-sha256 sha256:<the-stage-digest> \
  --output /tmp/fa4-f1-observation.json
```

This entry checks torch 2.9.0+cu130, CUDA build 13.0, CuTe DSL 4.6.1,
cuda-python 13.1.1 and actual SM103 before importing the original source. It
records actual driver/device/Python/TVM-FFI details. The L20D alias, compiler,
hardware documents and execution-profile qualification still require binding.
There is no CPU/H20/Triton fallback, package installer, remote submission or
automatic retry. The command was not GPU-executed in this engineering batch.
The output must be outside the immutable stage. A process failure requires
settling that original operation in its owner before any subsequent admission.

The stage has 19 files: all 12 original `aka_solution/` files (six Python runtime
files), four evaluator files including actual input/reference/historical harness,
and fixed input/timing/environment descriptions. Imported Task/Schema/Scope
closure is sufficient to stage again without the original Git checkout. Original
notices and the known license gaps remain attached; no new license is granted.

## F1 and evaluator behavior

The shared root Contract remains unchanged. Only F1 B1/Q4/L64/P1/N1/S1 is
selected, with valid page zero and the complete 13-argument ABI. The original
input helper executes unchanged after seed 20261120; query BF16 randn *1.22
precedes KV BF16 randn *0.96, both cast to FP8. Scratch stays 413138944 bytes.
The fixed F1 is a Schema representative, outside the original 30 scored cases.

The evaluator executes the pinned harness's `checks`, `unchanged`, and `poison`
function AST nodes, with their original source locations. It never runs the
historical main. The reference gets separate readonly tensors, out and scratch.
Before candidate calls, the evaluator retains original out/storage/view metadata
and readonly tensor identities and bytes. Both return and that original out
must pass FP32 finite, shape/dtype and per-element `0.06 + 0.04*abs(reference)`
checks, alias the supplied view, and fully overwrite NaN poison. Scratch may
change; the other tensor inputs may not. No wrapper repairs output.

Cold import/construction and first complete forward+sync are recorded separately.
The first call includes fmha compilation/launch and native FFI probe/possible
restore; it is not a steady timing sample. After native eligibility is established,
five warmups and 40 full-forward+completion host-clock samples use the original
six tensor objects. Before/after each call the launcher/signature/thread/device/
noncapture predicates are checked. Every call is re-poisoned and fully compared,
outside timing. The original native probe's 1e-2 threshold is never substituted
for Task tolerance. No kernel-sum baseline or prior 1.1126x claim is imported.

`FA4Task.validate_scope` uses the actual Template resolver and recognizes only
the pinned four-node Schema. It returns root Contract plus direct Design context,
material coverage and `task_uncovered=[fa4_decode.shared_contract.except_f1]`.
Root binding fails; changed Schema or variant stays unknown. Runtime when and
fidelity remain separate. N1 does not prove empty parity, padded TMA/page-tail,
TMEM fences/deallocation or output staging lifetime.

## Core source format and compatibility

The additive `python_tree` format is `encoding=utf-8`, `entrypoint=solution.py`.
`files` is a name-sorted complete snapshot of 1–32 regular mode-0644 files,
at most 4 MiB total. Paths support identifier-named root Python modules or one
package-directory level of Python modules, and root `.json/.md/.txt` or `LICENSE`
text. BOM/invalid UTF-8, links, binaries, deeper trees, executable modes, missing
entrypoint and extra unlisted files reject. This is not arbitrary-source-tree
execution support. `stage_original` continues to admit only the complete pinned
twelve-file Source. `stage_candidate` freezes separate complete baseline and
candidate trees and the actual finite launcher; it never overlays old files.
Candidate execution is available only through the explicitly qualified channel.

| Core API | Boundary |
| --- | --- |
| `Publication.put_source_files(name, dict[str, FileBytes])` | Freeze exact bytes/modes with the finite tree manifest. |
| `Bundle.source_files(ref)` | Verify and return the complete name→FileBytes map. |
| `install_source(bundle, ref, new_directory)` | Atomic new snapshot installation; existing destinations reject. |
| `verify_source_directory(bundle, ref, directory)` | Exact bytes/modes/path set, including extra directories. |
| `replace_source(bundle, new_ref, directory, previous=old_ref)` | Caller-authorized, locked Linux atomic directory exchange; retry same Refs after interruption. Readers must first quiesce. Files omitted from new Source disappear. |
| `Bundle.source(ref)` / `Publication.put_source(name, bytes)` | Original single_file/kernel.py behavior is preserved; source(tree) rejects instead of truncating. |

Closure/export/import follows every explicit file. `validate_request` records
actual required source formats; `require_capabilities` checks them against both
profile and backend. Existing single_file-only backends reject python_tree.
The reference implement backend and K3 advertise this format now. Existing
single-file-only profiles and AKA adapters continue to reject it.

## Callable seams and remaining adapter work

`kernelschema.evaluation.fa4` exposes `freeze_case`, `FA4Task`, `stage_original`, `stage_candidate`,
`freeze_request`, and `freeze_observation`. `freeze_request` uses the ordinary
K2 codec/validation, with an explicit original_verify, seed_continuation or
schema_rebuild intent. Original verification selects exactly baseline C and
verify_only; seed continuation requires an explicit full Source; pure-Schema
rebuild has no seed/starter. Default role Materials contain objective/license
gaps and the explicit [candidate observation interface](candidate-interface.md).
That small interface specifies Model/13 ABI and the native launcher observation
names needed by this finite evaluator; it includes no optimized implementation.
K2/K3 reject exact nonempty baseline Python-file bodies in a `schema_rebuild`
request, including renamed Materials/text and a differently wrapped Source.
Empty package markers remain legal. This byte-identity check does not detect
rewritten excerpts; it does not turn baseline metadata into a Source grant.
Baseline metadata supplies the required Source format, not a role grant.

`freeze_observation(publication, staged, raw=..., operation_id=..., usage_receipts=...)`
verifies original stage bytes and the raw Task/R/Scope/C/evaluator/input/timing
subject map, freezes untouched raw output and actual environment, and returns a
normal evaluation Ref. Caller commits it with the original immutable request as
a subject locator. Standalone or synthetic stdout produces **unknown** build,
correctness and performance checks; fidelity and performance goal stay unknown.
It is an observation importer, not a qualified executor or automatic five-gate
promotion path. Native job/owner receipts must be supplied by the actual adapter.

## Actual channel and qualification seam

`fa4_channel.workflow_channel` is a pinned reference channel factory. It extends
the existing `AgateDevExecutor` with finite `fa4_probe` and `fa4_execute` actions;
the original submit/key/query/settle, complete compressed frame, native terminal
and usage publication remain the same implementation. Lost ACK/404 stays on the
original operation. `prepare` performs no hidden paid probe.

Explicit `fa4_channel.qualify(context, remote=...)` performs six original jobs
under the same `OwnerAccount`: original-stack probe, full baseline and four
fixed bad Sources (missing out, changed query, tolerance and original-storage
replacement). It returns retained original records for a subsequent fixed
`selection(remote=<observed remote including bwrap_sha256>, qualification=...)`.
Continue qualification by reopening that same context and owner. The account
must reserve all remaining jobs; no result or missing binding creates quota.
An initialization intent is retained before the first owner binding. Missing
that original binding or an original owner operation refuses recovery, including
a crash during initialization; retained native records cannot create a balance.
Synthetic transport is always rejected as qualification. These paid commands
have not been run here. The remote shape is:

```python
remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D",
          "python": "/selected/original/python", "bwrap": "/selected/bwrap"}
```

After qualification, reuse the original prepared `case` Refs for the new fixed
request. A fresh `freeze_case` creates a different Task identity and cannot
inherit those records. Freeze the qualified profile in that same artifact
bundle, with the original owner's descriptor as the new request's parent:

```python
# ready is the original prepared.json; owner is its existing OwnerAccount.
# qualified_profile contains the actual returned selection and fixed Claude profile.
from kernelschema import api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.budget import use_authority
from kernelschema.evaluation.fa4 import freeze_request

p = Publication(bundle_root)
profile_ref = p.put_json("qualified-profile.json", qualified_profile)
parent_ref = p.put_json("parent.json", owner.descriptor)
p.commit([profile_ref, parent_ref])
request_ref = freeze_request(Publication(bundle_root), ready["case"],
    rules=ready["rules"], profile=profile_ref, limits=business_limits,
    parent=parent_ref, intent="original_verify")
durable_write(next_request_path, Bundle(bundle_root).read(request_ref))
with use_authority(owner.path):
    result = api.implement(request=next_request_path, out=new_run_path)
```

The initial owner's total limits must cover qualification and intended business
work. No fresh parent balance is created by this handoff; insufficient remaining
capacity still blocks admission. The snippet uses the existing artifact, budget
and API functions, and does not mutate the original request/profile/run.

The source snapshot and every trusted evaluator/launcher byte, mode, path,
command, original key and profile are checked before submit and original query.
Each execution job runs baseline and candidate in separate fresh processes with
separate compile/output scratch. The preinstalled pinned bwrap denies stage and
root writes, drops capabilities and network namespaces, and exposes CUDA
devices. Every staged file receives a write-open denial probe before candidate
import. Cold, native predicates, 46 complete comparisons and 40 full-forward
samples are retained for each Source. Driver/Python/compiler/TVM-FFI and all
required original versions are captured and must match the qualified profile.
This fixed path requires `/usr/local/cuda/bin/nvcc`; it installs nothing.

The explicit trust boundary is generated code that does not attack the
evaluator's Python address space. bwrap protects files and network; it does not
isolate hostile native code from same-process tensor comparisons. Qualification
must exercise that stated boundary, full stdout and settled cleanup. Raw JSON
consistency alone is not independent proof of service execution. Missing actual
auth, namespace/CUDA permissions, original stack or qualification records blocks
admission. There is no H20/Triton substitute or permanent unsupported executor.

Fidelity receives all Python files and current root Contract/direct Design.
Every accepted source location must name a real file and exact quoted lines.
An independent reviewer is required; incomplete/unknown CuTe, TMA, TMEM,
empty-parity and page-tail evidence stays unknown. Schema static scope pass is
separate from runtime native predicates and fidelity. New FA4 variants and
design/optimize remain unsupported by this finite implementation.

AKA still needs its own Task dispatch, typed full-file model/session handoff,
complete native install/freeze with deletions, and original C-to-job/report
mapping. It must explicitly select this channel or qualify its own equivalent
boundary on the same owner, then enforce original_verify/seed/rebuild delivery.
KS support does not declare AKA F1 compatibility.

The [K7 license audit](licenses-and-gaps.md) and anonymous fixed-commit retrieval
gap remain open; see the [release checklist](../../../docs/workflows/release-gaps.md).
Current engineering tests and raw APIs do not close K8.real.
