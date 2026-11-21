# Implemented workflow protocol

This is the public entry to the implemented `ks.workflow.v1` contract. It adds
no second wire format. The executable object grammar is
[`kernelschema/contracts.py`](../../kernelschema/contracts.py); verified file
relationships are implemented by
[`artifacts.py`](../../kernelschema/artifacts.py) and
[`api.py`](../../kernelschema/api.py). The detailed
[API interface](../../kernelschema/INTERFACE.md),
[writing interface](../../kernelschema/WRITING.md),
[reference/account interface](../../kernelschema/implement/INTERFACE.md) and
[runtime interface](../../kernelschema/runtime/INTERFACE.md) describe their
implemented seams. No private planning document or runtime capture is needed
to install, read or prepare the supported source checkout.

The implement reference adapter supports the fixed row-softmax Task's entire
coverage domain and complete single `kernel.py`, plus the pinned FA4 F1 scope
and finite complete `python_tree`. The writing backends and finite driver remain
row-softmax only. [FA4 usage](../../examples/workflows/fa4_decode/README.md) gives
the actual API/channel preparation and qualification boundary. CPU tests
establish engineering behavior. They do not establish real model, CUDA,
fidelity, performance or release qualification; see
[standalone usage](standalone.md) for commands and the current missing gates.

## Entry points and fixed requests

`kernelschema.api.design`, `optimize` and `implement` are keyword-only functions.
Use either `request=<request.json>, out=<new run>` or
`continue_run=<original run>`. The `kernelschema` CLI calls those same functions.
Import functions from `kernelschema.api`; package-level operation names belong
to backend modules. `kernelschema extract -- ...` forwards the existing extract
CLI and preserves that workflow's own publication/recovery protocol.

Each request contains `protocol`, `operation`, `request_id`, `task`, `rules`,
`materials`, `scope`, `budget` and `backend`. The four object inputs are Refs.
`backend` contains `name` and a profile Ref; `budget` contains a parent account
Ref or null and request limits. Additional operation fields are:

| Operation | Fields and authority |
| --- | --- |
| design | Propose initial Schema/scope for the fixed Task and objective. |
| implement | `schema`, `seed` (Source Ref or null), `mode=code|verify_only`; the input Schema is fixed for the whole request. |
| optimize | `schema`, `feedback:[Ref]`, `evidence:[Ref]`; propose keep/revise and an explicit subsequent action. |

`request_id` is a display/correlation ID. Q is the digest of the full canonical
request; a separately admitted run with the same Q can incur new costs. A run
records its own identity, frozen inputs, backend implementation and sole owner.
Continuation cannot change any of these or recreate a lost parent balance.

The shipped implementation backends are `reference` for implement and `claude`
for design/optimize. An explicit implement backend `aka` loads its separately
installed adapter. Core/reference/writing import does not require AKA or GPU
packages. Unsupported combinations are rejected before paid work.

## Bytes, identity and object roles

Normative objects use UTF-8 RFC 8785 JCS bytes. Duplicate JSON fields, BOM,
nonfinite/floating JSON numbers, unsafe integers and Unicode surrogates are
rejected. Fractional metrics use strings. Set arrays have field-specific order
and uniqueness checks; the codec does not reorder ordinary arrays. Unknown
object fields/kinds/versions reject. `extensions.information` is descriptive;
other extension owners require explicit backend support.

A Ref has `path` and `sha256` (`sha256:` plus 64 lowercase hexadecimal digits).
Artifact paths use `artifacts/<128-bit publication ID>/...`. Reads verify real
regular-file bytes and reject symlink traversal. YAML, Markdown, source and raw
native receipts retain their original bytes; normalized candidates are separate
artifacts with their own identities.

| Object | Meaning and retained contents |
| --- | --- |
| Task | Fixed task description/contract, acceptance, workload and environment. Write-only tasks may omit evaluator and baseline (`null`); strict/reference implementation requests require them. Golden numerical reference and performance baseline are distinct. |
| Schema | Task digest, complete raw nodes including root, optional dispatch, revision, parent/change/change_report. The original Template parser determines actual inheritance and source chains. |
| Rules | Exact rule files and tools with names, versions and source Refs. Methods/checkers stay linked to real bytes. |
| Materials | Explicit `{id,file,roles,origin,license}` entries. Writing requires one `workflow_objective` visible to planner/reviewer. A grant does not permit changing Task/acceptance or using tools. |
| Scope | Task coverage IDs and `{coverage_ids,node,variant}` bindings. Row-softmax requires its full Task; FA4 requires the exact F1 point and keeps `task_uncovered`. Unsupported conditions/variants are not guessed. |
| Source | `single_file` / `kernel.py`, or finite `python_tree` / `solution.py`; UTF-8, complete name-sorted file list with mode `0644`, size and digest. Bodies are beside the manifest under `files/<name>`. The [FA4 format boundary](../../examples/workflows/fa4_decode/README.md#core-source-format-and-compatibility) limits trees to 32 files / 4 MiB, flat Python packages and root text provenance. Format support does not imply business support: profile and installed backend must each advertise the actual required format. |
| Backend profile | Explicit runtime/model/configuration/environment/native limits/capabilities; an advertised capability must also be supported by the installed backend. |
| Evaluator | Exact entrypoint/files, input/timing/environment Refs and adapter IDs. Staging verifies the actual installed trusted evaluator and complete source. |

Source is a complete snapshot, never an overlay. It has no embedded Q, history
or parent C. `seed=null` selects Task.starter_source; only when both are null is
the start empty. An invalid explicit seed cannot fall back to a baseline or
starter. Verify requires an explicit nonempty Source and performs zero coder
calls/edits/promotion; necessary evaluation/review still consumes budget.

Role inputs are host-delivered, no-tools text. They include only fixed public
assignment bytes and explicitly granted materials/Source/feedback. Evaluator
files, entrypoint and inputs are private by original Ref/byte identity, including
renamed exact copies; timing/environment descriptions remain explicitly
deliverable. Private histories are not expanded as references. The
explicit Task baseline Source exception and its exact byte/role boundary are
specified in the [runtime interface](../../kernelschema/runtime/INTERFACE.md).
Design does not receive baseline or handwritten R0 Design by default.

## Checks, results and stage handoff

A Check contains `status=pass|fail|unknown|not_run`, `receipt` and `reason`.
Pass/fail needs evidence; unknown needs a reason. Results bind run, Q and the
actual immutable subjects. They carry status, stop_reason, recovery, feedback
and cumulative request usage. CLI exit 0 means that a result was durably
published; inspect its domain status and checks.

| Result | Completion boundary |
| --- | --- |
| implement | `code.latest` and `code.qualified` are independent Source Refs. Each C has its own coverage/build/correctness/fidelity/performance assessment. Qualified requires all five passes; completed requires qualified C. `performance_goal=met|unmet|unknown` is separate from complete required measurements. |
| design/optimize | Default write-only saves readable Schema files/scope and real call records; structure/publication have file receipts, permissions/semantics are `not_run`. Completion does not establish design effect. Explicit strict mode requires all four passes. |
| optimize action | `decision=keep|revise`, reason, `next_action=none|implement|verify`, action reason, explicit input/evidence Refs. The writer proposes; the outer driver owns subsequent admission. |

The default writer saves exact YAML, rationale, change_report, original response
and native receipts. Its small content needs only raw files and a reason, or
keep/reason for optimize. Template lengths and semantic review are guidance,
not publication gates. A write-only Schema manifest explicitly marks that
mode; bundle closure checks file identity rather than running Template
resolution. Unsafe filenames or malformed output preserve the original draft.
Malformed YAML remains a partial file artifact. The file-oriented task entry
needs no evaluator, private evaluator code or locked Contract generator.

The optional standard AKA driver passes native result/feedback bytes to one
optimize iteration and implements a revision once more. It does not assess the
Schema's effect. AKA owns its original comparator and native budgets; optional
explicit owner sharing uses the existing account. The old strict action
catalog and reference qualification sequence remain opt-in compatibility
interfaces and do not gate the default writing/AKA path.

## One owner and original-operation recovery

Parent account descriptors identify an existing private OwnerAccount; they do
not carry a copyable balance. Request/parent/native limits take the tightest
applicable bound. Reserve minimum evaluation/review completion costs before
starting code work. The owner uses one active clock for serial requests and
explicit finite native groups; parent waiting and child work do not double count.
The current Claude profile cannot enforce a hard token limit and rejects finite
token budgets before launch.
Its per-call timeout defaults to 120 seconds and may be explicitly increased
in a new frozen business profile. The original parent/request/reservation and
absolute deadlines still take precedence; this never renews a failed operation
or grants a new balance. See the [runtime deadline contract](../../kernelschema/runtime/INTERFACE.md)
and [writing configuration](../../kernelschema/WRITING.md).

Reservations are durable before launch. `authorize` checks the original JCS
authorization, limits, pause/fences and deadlines at launch; `check_active`
checks an already launched original operation without granting another call or
extending its window. Cumulative native receipts are keyed by producer/operation/
receipt/sequence and absorbed only for their increment. Unknown statistics are
null, not zero. Known lower bounds survive missing statistics and overspend.

| Condition | Required behavior |
| --- | --- |
| Unresolved original process/job or constrained balance | blocked/settle_only; query/settle the same operation, preserve its locator and held budget. |
| Missing capability/materials | Explicit blocked/unsupported behavior; no fake passing executor. |
| Quiescent unfinished work with exhausted budget | partial; no new admission or replacement balance. |
| Completed or permanently ended run replay | Reconstruct its original result/receipts; no replacement model or evaluator call. |
| Missing/corrupt original identity/authority | Refuse recovery; new Q/R/out cannot recreate the balance. |

Agate qualification uses the same complete evaluator and six original logical
jobs: probe, full baseline, and four fixed bad Sources. It checks original
upload/subject/readonly/terminal facts, never native allclose in place of the
Task. Lost ACK without a job ID remains fenced; 404 is not proof of zero cost.
Malformed known-terminal output retains the original fee without granting pass.
Configuration declarations and internally consistent raw JSON alone do not prove
that a real service/model/GPU run occurred.

## Selected artifacts and release boundary

Bundle closure follows supported object semantics, actual Template raw/from
chains and explicit Source files. It does not scan prose, home, arbitrary
history or private archives. Selected report subjects need real immutable
Q/Task/R/C/rule/method/evaluator locators; a bare digest is insufficient. Missing
unrelated history is allowed; missing selected dependencies or changed original
bytes/modes rejects publication/import. Source alone need not include its old Q.

`request.json`/`result.json` are entry projections; immutable artifact history
and original private run state support recovery. A portable selected export
supports reading/importing and does not promise private-run relocation. An
explanatory redaction cannot replace bytes required by an old Ref.

The supported packaging tier is source checkout/editable. Complete wheels,
Python 3.11 compatibility, actual business-model/CUDA gates and final licensing
remain separate release work. Existing material licenses/origins are retained in
the [row-softmax source inventory](../../examples/workflows/row_softmax/references/README.md)
and its manifests; public readability does not itself grant redistribution.
The [release checklist](release-gaps.md) separates these remaining decisions
from the supported CPU/source-checkout engineering behavior.
