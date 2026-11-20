# K4 fixed reference loop and shared owner

The real entry point is `kernelschema.implement.reference.workflow_backend()`;
backend name `reference` uses the existing `kernelschema implement` CLI and
`kernelschema.api.implement`. This implements CPU-tested engineering only.
No model/GPU call or real Task/fidelity qualification was added in B04.

The fixed profile must contain `parameters.claude`, the complete K3 public
profile dictionary, with matching `model`. The backend never substitutes its
own implicit CLI/model configuration. `fidelity.freeze_method(publication)`
returns `file` and `tool` entries for Rules.files and Rules.tools: the exact
method bytes and actual checker source/version, both mandatory. Current complete Schema/root Contract/scope,
role-granted materials, exact current C and feedback use K3 no-tools text input.
No planner runs inside this backend. `verify_only` requires an explicit nonempty
seed, performs no coder/edits/promotion, and pays for evaluation and review.

The loop freezes a complete C before full Task staging; explicitly passes scope
to `require_evaluation_scope` and `stage`; performs an independent readonly
review; then repairs the same R from exactly that C. Source-format failure keeps
the previous complete stable C with diagnostics. A complete source with a build
defect is still its own C and gets its own failed/unknown checks. The seed ready
installation is never reapplied after admission. Intermediate source, operation,
reservation, ready/admitted inputs, terminal, assessment and feedback facts are
durable before result publication. New source publications never mutate old C.

Five separate checks gate `qualified`. Full measurements with soft-goal `unmet`
can complete. Ordinary build/numerical/fidelity defects or unknowns produce
partial and remain same-R work. Missing scope/material/capability is blocked.
Unresolved operation or constrained balance always wins as blocked/settle_only.
Completed or budget-ended runs have no further action and continue adds no fees.
The public latest/qualified fields and each C's assessments are independent;
reports explicitly retain source/request locators for selected historical closure.

## Explicit evaluation channel

There is no default passing executor. Missing selection returns a zero-call
blocked/new_request result with the seed intact. LocalExecutor's CPU checks
cannot qualify this CUDA Task; K1 Agate engineering has not acquired real service
qualification. Loading the K3 public profile does not grant GPU qualification. The repository
now supplies `kernelschema.evaluation.channel:workflow_channel` and a six-job
qualification preparer; see [standalone usage](../../docs/workflows/standalone.md).
These paths have synthetic CPU tests, with actual Agate qualification still unrun.

`profile.parameters.evaluation` optionally has exactly:

```python
{"factory": "installed_module:function", "implementation_sha256": "sha256:...",
 "parameters": {"channel_specific_fixed_configuration": "..."}}
```

The source hash is verified before the selected callable is used; its original
bytes remain required for continue. The trusted installed factory receives
`(fixed_backend_profile, private_evaluation_root)` and returns a channel with:

* `admission(context) -> {status, reason}`: validate the actual qualified runtime,
  hardware/Task/profile, complete evaluator and permission/terminal prerequisites.
  Static `validate_scope(require_hardware_binding=True)` alone is insufficient.
* `locator(operation_id) -> dict`: recoverable private locator before submission.
* `prepare(context, op)`: idempotent local preparation only; no paid GPU/model call.
  `op` includes exact staged source/evaluator, reservation, ID and locator.
* `submit(context, op, authorize=callable)`: call authorization immediately before
  the first submission; respect reservation time/deadlines; repeated submission
  must query the original operation. No alternate channel or retry key.
* `settle(context, op, publication) -> {state}`: unknown stays fenced; terminal
  includes `usage_row` and `report` Ref (or null). No paid operation during settle.
* `validate(context, op)`: verify retained native locator/identity/stage on resume.

One full-Task logical evaluation is reserved per channel operation. Prerequisite
paid probes/helpers must be separate owner operations before this channel is
qualified; `prepare` cannot hide them. Channel reports bind the actual original
Task/C/evaluator/inputs/timing/environment. Preserve full Task measurements and
put the fixed Task soft-goal result in `extensions.information.performance_goal`.
K1 executors can be wrapped at this narrow boundary without replacing their
original submit/query/settle protocol. No fabricated transport is installed by
the real factory; constructor injection exists for synthetic tests only.

## Shared single-owner authority (K5/A5 handoff)

`kernelschema.budget` is the only account/clock/receipt implementation. It is a
local Linux private state file with flock and fsynced replacement. The authority
path is private recovery configuration, never a public balance or credential.

```python
from kernelschema.budget import OwnerAccount, use_authority, cost

owner = OwnerAccount.create(private_path, owner=owner_id, limits=finite_limits)
# Freeze owner.descriptor as the child's budget.parent Ref.
with use_authority(private_path):
    child_result = api.implement(request=child_request, out=new_run)
# An external process receives the exact same path as KS_BUDGET_AUTHORITY.
```

`OwnerAccount(path)` opens only an existing authority. `for_context(context,
create=True)` creates a top-level run's original authority, or attaches a parent
through `use_authority`, explicit `authority=`, or `KS_BUDGET_AUTHORITY`. Parent
descriptors must match byte-derived identity. Missing state never creates a
replacement parent. Continue uses its durable original binding and run/Q/limits.

Stable methods:

* `register(run_id, Q, request_limits, root=...)` binds a serial request once;
  new Q/R/run does not reset the owner. Explicit new runs with equal Q charge.
* `begin(run_id)`, `end(run_id)`, `pause(bool)` maintain one activity clock;
  repeated parent/child waiting does not double count. In-flight pause/crash time
  continues until quiescence is proved. Per-run time projects only its activity.
  Reservation automatically starts activity if no explicit begin is active,
  and stops that automatic segment only when all members are quiescent. External
  callers cannot accidentally get free activity by omitting begin/end; explicit
  begin/end additionally brackets host preparation and inter-operation work.
* `reserve(run_id, operation_id, role=..., purpose=..., amount=cost(...),
  floor=cost(...), native_limits=None)` durably authorizes finite consumption
  under the tightest parent/request/native limits and minimum finish reserve.
  The current profile defaults to 1,000 ms minima for evaluation/review; fixed
  `parameters.minimum_active_ms` may select positive coder/evaluator/reviewer
  minima. Calls/jobs and the minimum active intervals are reserved before coder.
* `authorize(reservation) -> bool` checks original identity, pause/fences,
  remaining quota and expiry, then marks may-have-launched before external exec.
  Reservation includes `owner/account_id/run_id/operation_id/authorization`,
  calls/jobs/tokens/active_ms, absolute `deadline_utc_ms`, and on the real Linux
  clock `deadline_monotonic_ns` (decimal string)/`clock_boot_id`. External native
  deadlines must be tightened at the actual launch boundary, as K3 does.
* `check_active(reservation) -> bool` is the polling guard for an already
  launched native `run_bounded` operation. It ticks and persists the same owner
  clock, checks the exact original owner/account/run/reservation, active state,
  pause, original UTC/monotonic deadline and boot, and the tightest existing
  parent/request/native limits plus the original finish reserve. It requests
  zero additional cost; an admitted call that already fills a count limit can
  keep running within its original time window. The current operation and
  validated same-request group members may be normally in flight; explicit
  unknown identities/fees and unrelated unresolved operations still fence.
  Missing/corrupt authority or group/request identity, forged reservations,
  unlaunched/released/terminated operations and exhausted limits return False.
  It does not authorize again, extend the deadline, absorb fees or release any
  reservation. False means stop bounded work and settle the **original** native
  operation, never that it has terminated or may be replaced. This check does
  not query native process/job identities; the adapter records native identity
  or acknowledgment loss with `mark_unknown`. A monitor can call it through
  the same private authority without claiming or relaunching a group.
* `reserve_group(run_id, group_id, operations, floor=None, native_limits=None)`
  atomically reserves a finite native concurrent group (for example evaluator
  V0 and prefetched reviewer). Each item has `operation_id/role/purpose/amount`;
  each gets an independent original receipt. Only the original process identity
  may launch members while siblings are in flight. Another process can settle
  and release proven unstarted members, but cannot restart a half-launched group.
  Before *any* member starts, `claim_group(group_id, reservations)` permits a
  single transfer of the complete original reservation set to an external
  launcher on the same authority. A claimed/half-launched/unknown group cannot
  be claimed by a replacement process. `mark_unknown(operation_id, reason)`
  records lost acknowledgment/identity and also fences unstarted siblings.
* `release_unstarted(operation_id)` only releases a proven unlaunched reservation.
  Reference expires such an intent by releasing it and ending partial/none;
  it neither mutates the old input/deadline nor allocates a new balance.
* `absorb(run_id, cumulative_usage_row, terminated=bool, lower_bounds=None)`
  deduplicates producer/op/receipt/sequence, rejects conflicts/regression, and
  absorbs cumulative increments. Known lower bounds remain even when a latest
  statistic is null. Termination and unknown statistics are separate facts.
* `fences()`, `operations(run_id=None)`, `usage(run_id, Publication(...))` expose
  original operations and public request-only usage. Unknown limited balances
  fence new charges; unrestricted unknown tokens do not imply a running process.

This first interface admits serial requests, with serial charges or a finite
explicit native group. It does not implement a budget tree or general scheduler.
Parent/caller must not hold an unresolved charge while opening a child request;
outer waiting itself is activity on the same owner clock, not an extra charge.

## Finite reviewer

`fidelity.obligations(context)` derives mathematical/precision, ownership and
dispatch, pipeline/lifetime, Require/Forbid obligations from current selected
effective Schema nodes, retaining root Contract. New unrecognized mandatory
sections remain unknown. `review_input` emits the exact inventory and scope;
`report` checks complete IDs and exact quoted source-line locations, incorporates
narrow AST-proven violations and publishes separate coverage/fidelity results.
AST facts or readonly permissions cannot grant fidelity. Nonselected designs
are read as context but do not become mandatory implementation obligations.

The synthetic paired fixture changes R0 single-row ownership to legal multirow
Rd and obtains different fidelity for the same C. External softmax and omitted
max are design failures without invented numerical overflow on [-20,20]. Real
reviewer judgment and real kernel qualification still require subsequent gates.
