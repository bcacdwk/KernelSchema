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
Other Tasks require an explicit future Task/scope adapter. The initial-design
Contract generator currently requires the fixed unbound public environment; a
newly device-bound Task needs its own explicit Contract adapter. It refuses to
invent a selected device or quietly reuse the unbound skeleton.

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

For initial design use `decision=revise`; for a revision supply complete
`raw=[{"node":"root","text":"<complete YAML>"}, ...]`, inline dispatch Markdown
(or null for a legal single node), and complete `bindings` in the original Scope
shape. IDs become host-chosen filenames only after strict node-ID validation;
there are no model paths. Keep requires the null/empty form and reuses the
original Schema/scope exactly. Unknown fields, duplicates, missing root, partial
raw snapshots and invalid action IDs are rejected with the raw draft retained.

`selection` must explain real planner-granted material IDs and include the unique
objective. The action catalog contains `evidence:0`, `feedback:0`, etc. for exactly
the request's explicit Refs, plus `task/starter_source` when available. A non-none
action needs reason, explicit inputs and nonempty authorized evidence. Each implement/verify action
requires exactly one nonempty Source input. Source evidence is delivered as its exact
kernel.py bytes; ordinary evidence uses the exact selected file bytes.

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
verify_only with the exact selected nonempty Source. Implement means fixed-R
code work using its explicit input/seed. The writing backend does not perform
that admission or execution. `extensions.information.admitted_request` is null.
No old C pass, fidelity report, native best or promotion becomes qualification
for a new R; K4/A5 must assess the actual new request subjects.
