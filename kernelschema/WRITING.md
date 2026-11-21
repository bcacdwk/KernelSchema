# K5 shared writing interface

`kernelschema.design.workflow_backend()` and
`kernelschema.optimize.workflow_backend()` return `WritingBackend` with backend
name `claude`. Call functions only through `kernelschema.api.{design,optimize}`;
the original single CLI calls the same functions. There is no new scheduler,
ledger, executable loop or model-selected writable directory.

This is CPU engineering delivery. Tests use the actual K3 subprocess and
persisted native input/usage, with explicitly synthetic drafts/reviews. They
establish no real model, CUDA, device, numerical or performance qualification.
The first finite Task adapter supports the fixed K1 row-softmax public Task.
FA4 F1 is supported by reference implement/verify only. This writing backend
does not claim FA4 design/optimize or infer new FA4 `when`/variant semantics.
Writing uses K3's same permission-first body projection: full Rules/tool closure
and host checks are retained, known host implementations carry identity only,
and exact duplicate authorized text uses per-payload aliases. The model receives
all semantic documents through those bodies/aliases, with no text truncation.
Other Tasks require an explicit future Task/scope adapter. The initial-design Contract generator accepts the original unbound environment
or the finite `ks.row-softmax.target.v1` adapter. Explicit hardware/runtime
requirements and hash-checked original documentation stay inside Task.environment.
See [target materials](../docs/workflows/target-environment.md). A declaration
never proves an actual device; unknown formats are rejected before charging.

## Constructing a request

All values below are already frozen Refs in the same bundle. The request must
be published and its actual dependencies committed before API admission.

```python
from kernelschema import writing
from kernelschema.api import design, optimize
from kernelschema.artifacts import Publication
from kernelschema.budget import OwnerAccount, use_authority
from kernelschema.runtime import ClaudeProfile

p = Publication(bundle_root)
rules_ref = writing.freeze_rules(p)
# Load the public qualified profile file; no discovery/model call is necessary.
business_profile = writing.backend_profile(fixed_claude_profile, environment_ref)
# parameters.claude and model remain the exact fixed K3 profile.
profile_ref = p.put_json("profile.json", business_profile)
request = writing.request(
    operation="design", request_id="design-1", task=task_ref,
    rules=rules_ref, materials=materials_ref, scope=scope_ref,
    budget={"parent": parent_ref, "limits": finite_request_limits},
    profile=profile_ref,
)
request_ref = p.put_json("request.json", request)
p.commit([request_ref])
# Write the same JCS request bytes to bundle_root/request.json using durable_write.
with use_authority(original_private_parent_path):
    result = design(request=bundle_root / "request.json", out=new_run)
```

`writing.request(operation="optimize", ...)` additionally requires `schema=Ref`
and accepts explicit `feedback=[Ref]` and `evidence=[Ref]`. Lists may be empty.
A parent descriptor is `p.put_json("parent.json", owner.descriptor)` from the
original `OwnerAccount.create(...)`; `use_authority(owner.path)` attaches children
to its actual private authority. New Q, R and out directories do not reset it.
Top-level requests may use parent=null and receive their original run owner.
Finite token budgets at request/parent/native levels are rejected because the
current K3 runtime does not advertise hard token enforcement.

The per-call timeout defaults to 120000 ms. To explicitly configure a longer
business call before creating its request:

```python
from dataclasses import replace

new_claude_profile = replace(fixed_claude_profile, timeout_ms=600000)
business_profile = writing.backend_profile(new_claude_profile, environment_ref)
```

Publish that new profile and a new, not-yet-admitted Q using the construction
above, retaining the same `parent_ref` and `original_private_parent_path`.
Admission still takes the tightest remaining parent/request/native budget and
original UTC/monotonic deadlines, including the minimum reviewer/finish reserve.
Choose any required `writing_minimum_active_ms` in the new business profile
before freezing it; a longer timeout does not increase the default review
minimum or grant more budget. Never edit an admitted Q/profile or reuse a closed
scope to retry a failed call. Old operations require their original retained
backend/source installation. This new profile has a different digest and does
not inherit the historical 120-second profile's real qualification; CPU tests
alone do not establish real writing success with a longer deadline.

The standalone `Demo.create` exposes `writing_minimum_active_ms` and
`writing_finish_floor`, freezing them directly in this profile. Its explicit
`request_limits` has complete `design`/`implement`/`verify`/`optimize` entries,
frozen in the Demo config and reused for later writing and action requests.
For two 600000 ms writing windows, select planner/reviewer minima of 600000 and
allow at least 1250000 ms for the request with no extra finish floor. A nonzero
finish floor additionally requires its complete call/job/time vector to fit
the request. These are capacity checks against the same original parent; they
do not consume dummy operations or extend the account. See
[standalone usage](../docs/workflows/standalone.md) for the corresponding JSON
CLI inputs and the separate reference-loop minima.

Materials must contain exactly one reserved `workflow_objective`, at least
planner/reviewer-visible. The backend validates the current writing method,
checker/policy source and original docs/Template frozen by `freeze_rules`.
Changing the objective changes Materials and Q, without modifying Task/scope or
Task performance-goal criteria. All selected references must be explicit
Materials entries with role grants and origin/license metadata. No author cache,
repository catalog or unrelated history is scanned.

Design obtains a locked Contract from the public Task, never the handwritten
R0 Design or private evaluator. `writing.template_contract(bundle, task_ref)` provides a model-free preview of
the same Task-derived Contract; it is an engineering helper, not a new wire
contract. The known K1 golden source locator retains its actual published
repository/commit/path; a bundle hash is never invented as a repository name. The planner receives this
locked Contract, current objective, original documentation and granted material
bytes, including the original Task workload and environment. Concrete unselected
hardware/stack entries remain honestly unselected.
Optimize retains the exact current Schema Contract; every semantic review also
compares it with the original public Task and acceptance.

## Model data formats

K3's outer envelope remains `ks.claude.output.v1`: planner kind
`writing_draft`, reviewer kind `review`. Native identity, assignment, tool
permissions and usage are verified by K3. Model content cannot set host facts.

The planner's content must have exactly these fields:

```json
{
  "format": "ks.writing.draft.v1",
  "decision": "keep",
  "reason": "No supported change or follow-up is needed.",
  "selection": [{"id": "workflow_objective", "reason": "The fixed current goal."}],
  "raw": [],
  "dispatch": null,
  "bindings": null,
  "next_action": "none",
  "action_reason": null,
  "action_inputs": [],
  "action_evidence": []
}
```

For initial design use `decision=revise` and `next_action=none`, with
`action_reason=null` and empty action lists. Only optimize may propose implement
or verify. For a revision supply complete
`raw=[{"node":"root","text":"<complete YAML>"}, ...]`, inline dispatch Markdown
(or null for a legal single node), and `bindings` as a list of
`{"coverage_ids":["<coverage ID>","..."],"node":"root","variant":null}` entries.
This is the Scope's bindings list, not a complete Scope object. Cover every
fixed coverage ID exactly once using supplied nodes and supported variants.
IDs become host-chosen filenames only after strict node-ID validation;
there are no model paths. Keep requires the null/empty form and reuses the
original Schema/scope exactly. Unknown fields, duplicates, missing root, partial
raw snapshots and invalid action IDs are rejected with the raw draft retained.

The frozen `delivery/writing_input` contains a `selection_catalog` projected from
the exact planner grants. Each entry has `id`, its immutable `ref`, its runtime
`delivery_id`, and `action_id` (null when the material lacks an action grant).
`selection[].id` must copy the exact bare `id` and include a nonempty reason;
the unique `workflow_objective` is required. These namespaces are distinct:

| Use | Example |
| --- | --- |
| Draft selection ID | `workflow_objective` |
| Delivered material ID | `material/workflow_objective` |
| Action catalog ID | `material:workflow_objective` |

The parser rejects other prefixes and unauthorized IDs; it never strips a
prefix to reinterpret a failed output. The action catalog contains
`evidence:0`, `feedback:0`, etc. for exactly
the request's explicit Refs, plus `task/starter_source` when available, `task/default_start` (the Task Ref),
and `material:<id>` for material granted to both planner and reviewer. A non-none
action needs a reason, explicit inputs and nonempty authorized evidence. An
objective/document can supply that evidence without invented measurements. Verify
requires exactly one nonempty Source. Implement selects one Source or explicitly
selects `task/default_start`, meaning seed=null and Task.starter_source/empty start. Source evidence is delivered as its exact
kernel.py bytes; ordinary evidence uses the exact selected file bytes.

The planner's writing input and reviewer's `review_subject` both contain
`text_policy`, a compact projection of the actual frozen original Template's
`BUDGETS`. Its `source` Ref/hash and `symbols` identify the authority. Numeric
`maxLength`/`maxLines` entries are generated from that source, including the
300-character / 2-line Thesis, 150-character / 1-line Strategy and verb describe,
and 50-character / 1-line meaning limits. Path segments `*` select named mapping
members and `[]` select list items; non-null prefixes restrict matching strings.

Count decoded YAML strings with the original `len(value)` and
`len(value.splitlines())`: Unicode code points include whitespace/newlines,
logical lines include interior blank lines, and nothing is trimmed. YAML folding
and escapes apply before counting; `maxLength=null` only removes that entry's
character cap. Non-string values are outside these entries. The model must
self-check before returning. The original Template still validates all structure,
references and other field rules; this projection is not a second validator.

These are explicit writing-specific declarations delivered with the finite
METHOD. The current runtime has no independent response JSON Schema parameter,
and no claim is made that JSON Schema validates the YAML string's internals.
K3's outer envelope and the strict writing/Template parsers remain authoritative.
Known host implementation bodies remain omitted; this summary does not restore
them to the model input or change the common runtime delivery/profile.

Reviewer content is exactly:

```json
{
  "format": "ks.writing.review.v1",
  "subject": {"request_sha256":"...", "task_sha256":"...", "schema_sha256":"...", "rules_sha256":"...", "scope_sha256":"..."},
  "items": [{"id":"<supplied obligation ID>", "status":"unknown", "reason":"A required fact is unavailable.", "location":"<supplied exact location>", "quote":"<supplied exact text>"}]
}
```

The host supplies the complete inventory, frozen output raw and output subjects.
Every ID must occur once with exact location/quote and a reason. Missing,
malformed, mismatched-subject and unlocated results become unknown. The reviewer
must independently judge the obligation; copying its quote alone is not proof.
The finite method covers full public Task/acceptance, root Contract, selected
scope and design sections, constraints, objective, change impact and action
inputs/evidence. Nonselected designs are retained and structurally validated but
are not silently made implementation obligations. Unknown when/variant semantics
remain unknown. Independent semantic unknown/fail prevents completed status.

## Durable publication and recovery

The run fixes inputs/owner and publication IDs first. It reserves planner budget
including the review minimum before admission. K3 receives the original owner
reservation with absolute UTC/monotonic deadlines and calls the same owner's
`authorize` immediately before invocation. `writing_minimum_active_ms` in the
fixed profile may set positive planner/reviewer minima; defaults are 1/1000 ms.

The optional fixed `parameters.writing_finish_floor` preserves capacity for an
outer driver's subsequent fixed request. It is the complete flat `budget.cost()`
vector: planner/coder/reviewer/extract/model_calls/evaluator_jobs/tokens/active_ms,
all nonnegative integers, with model_calls equal to the sum of role calls.
Unknown/missing keys, bools and inconsistent totals reject before model calls.
The default is `cost()` (all zero), preserving standalone writing behavior.

Planner reservation floor is this vector plus the ordinary minimum reviewer
call/time; reviewer reservation floor retains this vector. The original owner
checks request, parent and native limits together, so the writing request's own
limits must also include the selected capacity. The floor does not consume a
dummy operation, create an account, grant the next request, or guarantee that
unrelated future work cannot spend remaining capacity after this request closes.
The outer driver settles writing, checks the same owner, then explicitly admits
its next fixed request. For code/evaluate/review, construct the floor by adding
the corresponding ordinary cost vectors component by component.

The persistent phases are original planner intent/input/terminal; retained draft;
Template normalization and whole-Task permission/scope checks; immutable precheck
and change_report; frozen raw/scope/Schema; independent reviewer input/terminal;
writing_review; result. Template is the original parser/normalizer and relation
checker, loaded through `template_api`. Each publication has its own retained ID.
Change_report refers to input Q/Task and candidate bytes; Schema refers to that
report; writing_review refers to the frozen Schema/scope and original Q. Schema
never refers back to its writing_review. There is no mutual hash dependency.

Initial design uses new_task/revision=0. A substantive same-Task design change
uses new_revision/parent+1. Mechanically equivalent representation changes use
unchanged with the same revision and a real parent. Ordinary external feedback
never relaxes Require/Forbid. New REALIZATION.Feedback requires measured immutable
provenance; unsupported authentication remains an unknown permission gate.
A rejected or unknown candidate remains immutable; repairs use a new run and
publication ID. Failed drafts without a legal Schema preserve original native
usage and raw data without fabricating a successful writing_review.

`continue_run` opens the original owner and operations even before the first
result exists. It settles original calls, reuses their receipts and publication
IDs, and reconstructs the result projection. Missing original state/authority
cannot mint quota. Unknown operations/balances return blocked/settle_only and
prevent new charges. Terminal replay and keep/none add no model fees.

## Handoff to K6 / AKA A6

A completed optimize result supplies decision, reason, next_action, action_reason,
action_inputs and action_evidence. For a non-none action, action_inputs start
with the output Schema and Scope, followed by the explicitly selected inputs.
The next driver must settle/close the prior transaction, check the same owner's
balance/fences, and construct a new fixed implement request. Verify means
verify_only with the exact selected nonempty Source. Implement means fixed-R code work using the explicitly selected Source, or
seed=null when the Task marker is selected. Old C is never selected implicitly. The writing backend does not perform
that admission or execution. `extensions.information.admitted_request` is null.
No old C pass, fidelity report, native best or promotion becomes qualification
for a new R; K4/A5 must assess the actual new request subjects.
