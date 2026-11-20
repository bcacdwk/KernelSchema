# K3: finite Claude CLI runtime

`kernelschema.runtime` is an optional CPU-only Python module. It implements one
model call, its role input boundary, and local Linux recovery. K4/K5 own the
business loop, checks, publication decisions and the only request/parent budget.
It imports no GPU package or other agent application. Source checkout/editable
installation uses the existing core dependencies and Python >=3.11.

## Versioned profile and actual permissions

`ClaudeProfile.inspect(requested_model=..., executable="claude", ...)` makes
read-only `--version`/`--help` probes and pins the resolved executable bytes.
`reported_models` is an explicit tuple of accepted actual model names; unknown
or different reported names reject the output. `effort="xhigh"` is the default.
`backend_profile(environment_ref, name="claude")` returns a valid core profile.
A containing business profile must freeze `parameters.claude = profile.as_dict()`
and set its `model` to the same requested model. The business backend must narrow
the primitive capabilities to the operations it actually implements.

The profile is **host-delivered, no-tools**. It cannot read/edit model-selected
paths, execute commands, browse, call MCP, delegate agents, or install candidate
files. A fresh empty private role cwd lives outside the run/repository/home tree;
its ancestors are checked for AGENTS.md, CLAUDE.md and project configuration.
All allowed material bytes arrive through stdin. A unique session UUID is
allocated before launch, and native session persistence is disabled. There is
no native context resume or cross-session hidden conversation.

The actual command is saved verbatim before launch:

```text
claude --print --verbose --output-format stream-json
  --model <requested> --effort xhigh --session-id <uuid>
  --no-session-persistence --safe-mode --tools ""
  --permission-mode dontAsk --permission-prompts none
  --strict-mcp-config --mcp-config '{"mcpServers":{}}'
  --setting-sources <user-or-empty> --settings '{"disableAllHooks":true}'
  --disable-slash-commands --no-chrome --prompt-suggestions false
  --max-budget-usd 0.50 --system-prompt <fixed-versioned-prompt>
```

`environment_auth` adds `--bare --restricted`, points the official
`CLAUDE_CONFIG_DIR` at a fresh config directory, and passes only a small explicit
environment allowlist. The caller's `HOME` is preserved unchanged; it is never
repurposed as a config directory. Authentication must already be supplied through
the caller's environment. `existing_cli_auth` lets the
official CLI use its existing user authentication/config source, with safe mode
and user-only settings. It does **not** read, copy, log or modify credentials in
Python. It does not claim an independent config root. Project/local settings,
installed customizations and tools are disabled by the documented flags.

Claude 2.1.316 still exposes executable-built-in plugins in safe mode. The
profile must explicitly declare their names in `builtin_plugins`; only exact
names with `path="builtin"` are accepted. Unlisted/user plugins, skills, any
advertised tool/MCP server, wrong permission mode, or observed tool-use event
invalidate model output. The CLI binary is part of the trusted runtime; this is
a verified model tool boundary, not an OS sandbox for a malicious CLI binary.
The empty external cwd also keeps builtin AGENTS discovery away from user/repo
materials. Never interpret permission claims inside model content as facts.

The public capabilities explicitly say `token_enforcement="unsupported"`.
Finite token budgets are rejected before launch. Timeout and call counts are
not token limits. The native USD option is set but is a native expenditure
threshold, not a host-guaranteed hard dollar ceiling (providers may report
different pricing or the CLI may check between requests). Timeout is at most
120 seconds and is further narrowed by request and owner active-time authority.
Native internal rounds and physical retries are not independent host calls;
reported rounds are recorded and unreported retries remain null.

## K4/K5 seam

```python
from kernelschema.runtime import ClaudeRuntime, RoleDelivery

runtime = ClaudeRuntime(profile)
# context is a prepared/opened K2 RunContext with the frozen profile above.
# reservation has already been durably authorized by the sole K4/K5 owner.
handle = runtime.admit(
    context, operation_id="run_owned_operation", role="coder",
    purpose="implement fixed assignment", reservation=reservation,
    source=exact_current_C,
    source_roles=("coder", "reviewer"),
    deliveries=(RoleDelivery("feedback", feedback_ref, ("coder",)),),
)
receipt = runtime.invoke(context, handle, authorize=owner_launch_check)
# The owner absorbs receipt.usage_row() once, by producer/operation/receipt ID.
```

`reservation` is a nonempty dictionary of the owner's existing durable facts;
it does not mint quota. Include `active_ms` as the remaining interval reserved
for this call. Runtime timeout is the minimum of that value, the fixed request
active limit, and the profile timeout. `owner_launch_check(intent) -> bool`
must check the same authorization, current pause/fence, active clock, deadline,
parent limits and available balance. An owner refusal starts zero processes.
The runtime additionally fences a new local operation until prior launched
operations have receipts. It has no alternate account or multi-agent scheduler.

To carry the owner's *remaining* active window across input preparation and
supervisor queueing, the reservation accepts these optional fields:

| Field | Representation and meaning |
| --- | --- |
| `deadline_utc_ms` | Strict signed safe integer, UTC Unix-epoch milliseconds; bool/float/string/null are rejected |
| `deadline_monotonic_ns` | 1–40 ASCII decimal digits in a string, same-host monotonic nanosecond cutoff |
| `clock_boot_id` | Nonempty current Linux boot ID; required exactly when the monotonic field is supplied |

At the final launch point the runtime freshly clamps profile/request/reserved
duration against each supplied absolute remaining interval. An exhausted
window raises `AdmissionDenied` without creating a launch or starting a CLI.
Bad field types, incomplete monotonic/boot pairs or a different current boot
raise `ProtocolError` before a new launch. K4/K5 should supply both UTC and
monotonic bounds to retain the owner's cutoff even if wall time moves backward
between admission and launch.

When any cutoff is supplied, `launch.json` persists the tightened `timeout_ms`
and effective `deadline_utc_ms`, `deadline_monotonic_ns`, `clock_boot_id`. The
effective monotonic bound is also recorded for UTC-only reservations, so a
later supervisor delay or wall-clock rollback cannot renew the launch window.
The supervisor rechecks before spawning its child; the child rechecks before
and after executable hashing/fsync and immediately before exec; the supervisor
waits only against the remaining original cutoff and terminates the original
process group when it expires. A late runner can produce an explicit
`not-executed.json` terminal without starting Claude. With no cutoff fields the
existing duration-only behavior is preserved; `active_ms` alone does not
subtract time spent preparing an already frozen reservation. These are narrow
per-call launch bounds, not a replacement clock or parent ledger.

`admit` freezes exact input, profile/source bindings, role, purpose, reservation,
scope/session IDs and the core intent. Re-admitting the same operation with any
changed input/reservation/purpose is an error. An interrupted preparation may
finish the same unlaunched intent; it does not replace an admitted draft.

`prepare_role_input` uses actual `materials.for_role`. Public assignment bytes
are the Task contract/acceptance, Rules files, fixed Schema raw/dispatch and
scope. It does not recursively expand prose paths, evaluator files, baselines,
parent histories or tool implementation sources. Current C is verified and
delivered only to `source_roles` (default coder/reviewer). Extra feedback/method
or source material uses `RoleDelivery(id, file, roles, kind="text"|"source")`.
Private evaluator files, entrypoint and inputs are rejected even when relabeled
or copied to a new Ref with the same bytes. Timing/environment descriptions stay
eligible for explicit public delivery. K2 and K3 use the same private-byte set.
Request/object
mutation is rejected; fixed assignment digest is stable when continuing from
a new explicitly selected C. Role inputs have independent scope IDs.

The complete Rules and tool dependency closure remains unchanged in the bundle
and in host validation. Model input omits only declared tool entrypoint bodies
and the exact shipped Template implementation set when its declared entrypoint,
version and all selected dependency bytes match the installed known sources.
These items retain their original ID, Ref, digest, size, category/authority and
`host_implementation={name,version,source}`. Unknown `.py` rules stay readable;
an explicitly granted Materials/RoleDelivery code reference stays readable too.

Every item passes its own role/private-evaluator/Source/UTF-8 checks before any
deduplication. Identical authorized bodies appear once; each duplicate retains
its original bindings with `text_alias_of` naming a body ID in the same payload.
`runtime.material_text(payload, id)` validates and resolves that direct alias
without filesystem access. The original single-file `current_source.text`
remains a canonical body. No Contract, Schema, semantic method or unknown rule
is truncated. This projection and the system-prompt alias instruction are new
CPU engineering increments, not behavior certified by earlier real CLI captures.

One narrow exception permits the exact shipped row-softmax baseline as an
explicitly granted Source whose canonical C and bytes equal
`Task.baseline_source`, even when it is also an evaluator component. Its Ref path
may differ after import. A caller-created baseline label cannot grant inputs,
entrypoint or any other private implementation. This
supports an explicit V0/baseline review or seed; it does not expose a baseline
through ordinary Materials/text, permit other evaluator code wrapped as Source,
or deliver a baseline to design by default. The current role must still be in
`source_roles` or the Source delivery's explicit grants.

`fixed_assignment.role_authority` distinguishes code, Schema proposal and
review authority. Implement fixes the input R; a coder never gains Schema
revision authority from reference text. Design/optimize planners may propose
initial/revised Schema and scope bindings within the fixed Task, coverage IDs,
ABI and hard acceptance. Input objects remain immutable. Reviewers only assess
supplied subjects/proposals. A reserved `workflow_objective` is copied into the
fixed assignment as a current-task instruction **only if that role's Materials
grant includes it**; it cannot grant tools, change Task or relax acceptance.
`verify_only` refuses coder admission even with a reservation.

Call counts in this layer retain original CLI execution evidence. The child
records `exec-intent.json` immediately before its final deadline check/exec.
A bound exec intent or native init/terminal preserves the known call count;
authentication errors, nonzero exits, timeouts and output failures after CLI
start still count one. Missing `child.json` **never implies zero**. With only a
launch/exit and no execution evidence, calls remain null with known lower bound
zero; the owner retains its reservation/fence instead of refunding it. Lost
child identity also rejects model output even if an original native terminal
and usage remain available.

After a launch, zero calls/tokens/cost are reported only for the trusted,
operation/session/process-bound `not-executed.json` plus matching supervisor
exit, with an empty native stream. That is host proof that no CLI was executed,
not a fabricated native usage report. A contradictory or unverified
not-executed claim cannot erase an observed call. Such receipts expose
`facts.not_executed=true`; their model output is absent and status remains
failed with `deadline_refused_before_exec`.

## Typed model output

The native terminal must contain one JSON object (no Markdown wrapper):

```json
{
  "protocol": "ks.claude.output.v1",
  "role": "coder",
  "assignment_sha256": "sha256:<exact input assignment digest>",
  "status": "ok",
  "kind": "candidate_source",
  "content": "<complete kernel.py text>",
  "reason": "<short explanation>"
}
```

For a fixed `python_tree` assignment the coder must instead return
`kind="candidate_files"` and content exactly
`{"format":"python_tree","entrypoint":"solution.py","files":[{"path":"solution.py","mode":"0644","content":"..."}, ...]}`.
Files must be name-sorted, unique and form the complete snapshot. The sole core
Source codec checks finite paths, UTF-8, modes, count and size. Missing files
are deleted in the next snapshot; an absolute/model-selected host path is never
accepted. `RoleOutput.candidate_files` returns the verified `FileBytes` map.
The frozen assignment and original runtime intent bind the required format;
a tree cannot silently become a `candidate_source` string.

Tree input delivers every file with exact path, mode, digest, text, entrypoint
and original Source Ref. Single-file input preserves `current_source` and its
string output interface. Implement delivery includes Task workload/environment.
Pure-Schema FA4 requests start with no Source; baseline metadata does not grant
its contents, and explicit baseline delivery in `schema_rebuild` is rejected.

Allowed pairs are coder/candidate_source (string), coder/candidate_files (the
fixed tree object), planner/writing_draft
(JSON object), and reviewer/review (JSON object). `status="blocked"` requires
null content and a nonempty reason. Unknown outer fields, wrong role/assignment,
and runtime/model/permission metadata forged into the envelope are rejected.
`RoleOutput.candidate_source` returns exact UTF-8 bytes. Writing/review objects
are untrusted proposals for K4/K5's domain validators; no nested path is an
instruction to write a file, and no claimed pass is a host verification.
The runtime reports transport success separately from model `status`.

## Durable original-operation recovery

```python
context = open_run(original_operation, original_run_root)
handle = runtime.reopen(context, original_operation_id)
receipt = runtime.settle(context, handle)  # Never invokes/authenticates a model.
```

`invoke(..., wait=False)` returns currently available facts. Repeating `invoke`
on an already launched handle only waits/settles; it never re-admits/restarts.
`settle` never starts a process. A new session from a complete stable C requires
a distinct owner-authorized operation and an original settled operation.

Before a model process can execute, the runtime saves the core/local intent,
known session, command, input, native paths and a launch fence. A detached Python
supervisor owns raw stdout/stderr and exit collection. Its child fsyncs
PID + Linux start ticks + boot ID before exec of Claude. Killing the caller does
not kill this supervisor or discard the terminal. The locator stores paths for
the original supervisor/child identities, session, cwd, logs and exit fact.
The immutable raw receipt also includes the exec-intent and any explicit
not-executed fact, together with the effective launch deadlines.

| Observed facts | Recovery behavior |
| --- | --- |
| Admitted, never launched | Zero calls; owner may still deny launch |
| Launch persisted, no discoverable process/exit yet | Unknown; preserve fence, no replacement |
| Queued runner/child reaches an expired cutoff before exec | Explicit bound not-executed receipt; zero model calls, no accepted output |
| Exit exists but child identity is missing | Preserve native/exec call evidence or unknown; reject output, never infer a refund |
| Native result emitted, original process still alive | Still in flight |
| Supervisor dead, original child alive/unknown | Still fenced |
| Both known original processes demonstrably gone, exit lost | Terminated with `exit_unobserved`; absorb any original usage, reject output |
| Timeout | Supervisor kills and waits for the original process group, then records termination |
| Exit with nonzero/truncated/missing terminal/usage | Original failed receipt; known usage retained, unknown statistics null |
| Original complete receipt replayed | Same immutable receipt, zero new call/fee |

Termination and complete accounting are separate. A terminated call with null
tokens is not automatically in flight. The owner must fence any constrained
balance it cannot determine. `active_ms` is the local invocation interval; the
sole owner maintains the request/parent active clock and must not double count.
Process identity checks assume the retained local Linux filesystem and backend
installation. Moving a private runtime, changing its executable/source/profile,
or losing its intent is not a supported recovery path.

## Receipts, including failure with no output report

`CallReceipt.facts` carries requested/reported models, session/scope, errors,
termination, calls/lower bound, total tokens, native reported cost, native
rounds, nullable physical retries and raw file references. Nonzero exit, missing
terminal, malformed/truncated output, missing usage/cost, wrong identity or
permission mismatch never yield accepted `RoleOutput`. Original raw bytes are
preserved rather than normalized. Token components are taken once from native
aggregate usage; per-model totals are not added again.

`usage_row()` is a valid core usage-index row. Its `receipt` points to a
self-contained opaque native envelope containing every original stream/input/
profile/command/PID/exit byte sequence in base64 with SHA-256. It has no hidden
external Ref dependencies. Thus **a failed design call with no Schema, C or
review can publish a normal `usage_index`, and importing that index alone keeps
all call evidence**. No synthetic successful report is needed. CPU tests import
this failure-only closure into a fresh directory and verify every original byte.
In-flight query snapshots have no immutable receipt and `usage_row()` refuses
them; the owner retains its original reservation and unknown-operation record.

Separately stored raw files are available via `evidence_refs()` for explicit
report.evidence/raw_receipts. Core intentionally does not recursively interpret
unknown native JSON. Downstream code can use this convenience list, while the
self-contained usage receipt ensures failure paths are not dependent on that
convention. Reopen validates the public envelope against private facts/output.

## Qualification tooling

The small, committable [profiles/](profiles/README.md) directory contains the
observed nonsecret CLI configuration and a four-call explanatory summary with
original digests. K4/K5 can load that profile without any local capture. The
summary is **not** a complete importable evidence bundle. Full captures under
`qualification/20261120/` remain private local archives; they are not required
source or package resources and must not be recursively submitted with runtime
source. The public summary records the CPU-only input/HOME increments separately
from the unchanged real CLI capabilities.

`python -m kernelschema.runtime.qualify --out <new-capture> --role <role>
--model <explicit-model> ...` is an explicit one-call runtime probe, not a
workflow entry point or kernel experiment. A capture uses only invented dummy
canaries, a synthetic transport Task, a fresh role scope, at most one call,
120-second timeout and the USD 0.50 native threshold. Existing output directories
refuse a new call; `--settle --out <capture>` only reads the original operation.
Cross-capture experiment permission is the caller's responsibility. The capture
archives CLI help/status, frozen source/profile/input, raw receipts and checks.
Real qualification conclusions and their limitations are recorded separately
from CPU transport tests; no fake fixture is a real-model qualification.
