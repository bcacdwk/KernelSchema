# Fixed FA4 F1 engineering entry

This case freezes the existing four-node Schema and original v55 Source into
normal `ks.workflow.v1` objects. CPU preparation and tests do not qualify a
model, GPU, transport, numerical result, timing result or Schema fidelity.
The source and Schema commits remain distinct; see [materials.json](materials.json).

From an installed source checkout with its existing Python/PyYAML dependencies:

```sh
python -B -m examples.workflows.fa4_decode.prepare --out /tmp/fa4-f1-prepared
```

This command reads the 27 fixed Git blobs, checks their blob IDs/SHA-256/size/mode,
freezes real Task/evaluator/Materials/Scope/Schema/Source Refs, then installs a
complete stage. It imports no torch/CUDA/source module and submits no work.
`prepared.json` records all Refs plus the stage identity. `request.json` is an
original-Source `verify_only` request with zero calls/jobs and an explicitly
unqualified AKA profile advertising no execution capability. It is reviewable,
and cannot silently launch through the row-softmax channel. Execution needs a
new request with the actual adapter/profile, budgets and authority.

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
execution support. The current case runner accepts only the complete pinned
original twelve-file Source; changed candidates fail before import.

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
No backend support declaration was expanded in this batch.

## Callable seams and remaining adapter work

`kernelschema.evaluation.fa4` exposes `freeze_case`, `FA4Task`, `stage_original`,
`freeze_request`, and `freeze_observation`. `freeze_request` uses the ordinary
K2 codec/validation, with an explicit original_verify, seed_continuation or
schema_rebuild intent. Original verification selects exactly baseline C and
verify_only; seed continuation requires an explicit full Source; pure-Schema
rebuild has no seed/starter. Default role Materials contain objective/license
gaps only. A future rebuild adapter must also suppress the optimized Task
baseline delivery exception; the current K3 tree rejection prevents accidental
baseline disclosure and is not a completed rebuild path.

`freeze_observation(publication, staged, raw=..., operation_id=..., usage_receipts=...)`
verifies original stage bytes and the raw Task/R/Scope/C/evaluator/input/timing
subject map, freezes untouched raw output and actual environment, and returns a
normal evaluation Ref. Caller commits it with the original immutable request as
a subject locator. Standalone or synthetic stdout produces **unknown** build,
correctness and performance checks; fidelity and performance goal stay unknown.
It is an observation importer, not a qualified executor or automatic five-gate
promotion path. Native job/owner receipts must be supplied by the actual adapter.

AKA/K3 handoff still needs all of the following before advertising support:

1. Route this Task to its finite scope/evaluator seam. Existing RowSoftmaxTask,
   Agate `_worker.py` and channel assumptions cannot consume this case unchanged.
2. Carry `python_tree` through actual source installation, native source identity,
   execution uploads and candidate freeze. Use `source_files` and complete
   replacement, preserving deletions and C-to-native-file mappings.
3. Add a typed full-file coder output and whole-source reviewer delivery with
   file-qualified line evidence; current K3 strings and fidelity.py kernel.py
   locations are unsupported. Enforce the three source-information boundaries.
4. Qualify original CuTe/SM103 environment, readonly isolation, complete stdout,
   original-operation budget/query/settle handling and actual Agate credentials.
   No new balance, parser or recovery journal is supplied here.
5. Reuse the full comparator/timing entry for baseline and candidate only after
   a candidate sandbox and native eligibility mechanism are supported. The
   standalone path intentionally admits original C only. Rejudge full fidelity
   against the current Schema; unknown obligations remain unknown.

The [K7 license audit](licenses-and-gaps.md) and anonymous fixed-commit retrieval
gap remain open. Current engineering tests and raw APIs do not close K8.real.
