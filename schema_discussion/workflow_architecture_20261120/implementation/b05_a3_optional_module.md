# B05 A3: optional Schema module and native role boundary

Status: A3 interface/CPU implementation is in the AKA worktree; no commit or push
was made by this worker. A5 fixed execution and A6 dynamic execution remain their
own work. The final integration/real qualification gates belong to A7.

## Source pair and installation

- AKA worktree: `/home/shaohanyong.shy/atrex-kernel-agent`, branch
  `feat/schema-workflows-v1`, unchanged HEAD
  `c984d141145d31795514579db213413a1bbb0775` (A1/A2 base).
- KS optional gitlink: `optional/KernelSchema`, detached at
  `1297d3d1bec6a7cdfd5246dc5d93d8fa401b9e32` for the first A3 verification.
  URL is `https://github.com/bcacdwk/KernelSchema.git`.
- The clone used local reference objects only as a transfer optimization and
  `--dissociate`; `git fsck --connectivity-only --no-dangling` passes and
  `.git/modules/optional/KernelSchema/objects/info/alternates` is absent.
  No other AKA submodule was initialized; no origin/global configuration changed.
- New owning distribution: `atrex-aka-schema==0.2.0`, depending on matching
  `atrex-aka-light==0.2.0` and existing `kernelschema==0.1.0` source/editable
  installation. It owns only `aka.workflow.schema`. The metadata-only package
  directory has a `src -> ../../src` symlink for actual editable installation.
- `src/aka/workflow/schema/release.json` contains the fixed KS SHA/public URL,
  AKA base SHA and the selected 276-file content/mode manifest. Its first digest
  is `sha256:56bba210abcacac0409888e0a5867e7368df8fd5aa69b04bbee3b6c7f3497b82`.
  The manifest excludes private runtime qualification captures. Actual imported
  KS code/resource locations are checked, without cwd assumptions or scanning.
- Final AKA code commit is intentionally not self-referentially embedded in
  this manifest. A7 records the final release pair after code is fixed.

Installation from any cwd, with AKA set to the absolute checkout:

```sh
git -C "$AKA" -c http.version=HTTP/1.1 submodule update --init -- optional/KernelSchema
python -m pip install -e "$AKA/optional/KernelSchema"
python -m pip install -e "$AKA/packages/atrex-aka-schema"
python -I -c 'from aka.workflow.schema.release import verify_release; print(verify_release())'
```

The original Framework, Legacy and Light packages remain installed through
their existing build tool. The A3 verification built those original three
distributions with `tools/build_distributions.py`, installed them into
`/tmp/schema-b05-a3-venv`, and installed KS/Schema editable. Its fresh `-I` child
from `/tmp` resolved both metadata and fixed resources correctly. Python is
3.13.12; CPU dependencies were reused via the same existing dependency roots as
A2, without downloading another interpreter, CUDA or a model stack.

## Actual switch

The only existing runtime source modified by A3 is the native public CLI parser
`src/aka/bootstrap/workflow_cli.py`. It accepts `--schema` / `--no-schema`
(default off) and `--schema-config`. Off dispatches the exact existing
`run_profile` path; no optional import, KS I/O, selection read or download.
Existing Workflow DTOs, native prompts/limits and exports are not rewritten.

On transforms the selected native Light composition's Workflow, Agent and
Problem slots into explicit optional providers. The Agent provider uses the
original Agent/NativeSession with the restricted adapter; an unregistered raw
native invocation cannot bypass its role input/launch authority. The Problem
provider is the existing A2 `SchemaProblemProvider`. Default Wiki/plugin bundles
are empty on this composition. A reconstructed launch freezes its switch and
the original selection-file hash.

```sh
aka optimize --repo-root "$AKA" --problem /absolute/bundle \
  --platform H20 --framework triton --schema \
  --schema-config /absolute/selection.json

aka optimize --repo-root "$AKA" --problem /absolute/ordinary-task \
  --platform H20 --framework triton --no-schema
```

The config is `{format:'aka.schema.selection.v1', request:'bundle/request.json',
run_root:'private/run', backend:{factory:'aka.workflow.schema.backend:create_workflow',
implementation_sha256:'sha256:<actual installed source>', parameters:{}}}`.
Paths are relative to the config's own directory; the default factory name may
be omitted. The requested factory file is located/hashed before import. Missing
fixed config, resources, profile or A5 backend produces a specific diagnostic.
No `backend.py` stub pretending to implement A5 has been installed.

## Concrete handoffs

`schema.provider.SchemaSelection.load(path)` exposes the fixed request,
provenance, normalized selection identity, original selection hash,
`problem_provider(role=...)`, `claude_profile()` and verified `backend_factory()`.
It creates no RunContext/account or paid operation.

`factory(selection, NativeModules)` returns the retained native Workflow.
`NativeModules.light(agent=..., problem=..., executor=...,
evaluator_factory=..., options=...)` constructs the real `LightWorkflow` using
the original common services. The real unchanged `ManagedRuntimeFactory.open`
creates both Evaluator and CommandEvaluator from the supplied factory.
A5 directly reuses A2's `SchemaExecution` and operation resolver; A3 adds no
parallel evaluator, Source codec or total budget.

`schema.context.RoleContext.prepare(run, role, source=...,
source_roles=..., deliveries=...)` calls the actual K3 `prepare_role_input`.
`retain(directory)` / `reopen(run, directory, expected_sha256=...)` preserve and
revalidate exact payload and grants. Fixed assignment covers initial call,
repair, rate-limit retry and fresh resume/reopen; current C changes separately
without changing assignment identity. Forged replacement input with a retained
assignment hash is rejected by reconstruction from actual granted refs.

`role.tools(original_agent, empty_scope)` uses original ToolHost and contains
only read by granted material ID, effective by fixed Schema node, and check of
the selected immutable Schema. It calls KS Template, including root Contract
for nonroot effective documents. No path, write, shell, evaluation or broad
bundle-read operation exists. Private evaluator aliases fail K3's actual input
validator. Tool tokens are revoked on close.

`schema.session.native_agent(profile)` supplies the restricted original Agent.
`RoleSession(..., authorize=..., record_locator=...)` uses its actual
NativeSession. The original owner callback persists the exact native locator
before invocation; authorization runs in the real Agent launch hook. A5 must
bind these to its original KS `OwnerAccount` operation/reservation, apply current
reservation/deadline/pause/fences, absorb complete native receipts and manage
unknown original process state. A3 has no account/clock/receipt ledger.

`session.invoke(old_prompt, purpose=...)` replaces native free-exploration text
with the fixed input. Each instance makes at most one call; same-session
`resume=True` rejects. New sessions need explicit stable C and a new authorized
operation after prior settlement. The `stdin_exec` helper verifies retained
input/CLI bytes and execs in the original Agent-owned PID, avoiding Linux's
per-argument size limit without copying K3's process supervisor/recovery loop.

`NativeRoleResult` preserves the entire original `NativeInvocation` and errors.
Its `native_source()` produces exact original
`RevisionContent({'kernel.py': utf8_bytes})`. The host can directly call
`GitCandidateWorkspace.install_source(draft, result.native_source())`.
`freeze_source(publication, candidate_workspace)` publishes exact C and returns
A2's real `SourceBinding`. CRLF, Unicode and a missing final newline survive.
A4 owns C0/seed intent-ready-admitted installation; A5 owns subsequent authorized
draft installation, original seal/evaluation/audits and final result publication.
These A3 conversion methods do not seal, evaluate, qualify or promote.

## Audited scope decisions and limits

1. The original unrestricted Claude backend uses bypass flags, so merely
   appending a Schema instruction would not protect Task/R/evaluator inputs.
   The on composition replaces its adapter and denies raw calls without the
   fixed role authority. Off retains the original behavior.
2. The first AKA native profile is intentionally smaller: no model tools,
   MCP, shell, file access, hooks, project customization or native persistence.
   It uses the original K3 public profile's concrete CLI flags and verifies
   observed permissions/model/session before accepting its typed output.
   Read/effective/check are **host inspection/context tools**; the current
   no-tools model cannot call them interactively. This narrowed choice was
   reported to the supervisor and accepted under the user's offline authority.
3. The profile admits implement coder/reviewer only, single-file Source, no
   empty start, no finite hard-token limit or native hidden-session resume.
   K3's qualification does not qualify this changed AKA native transport.
   Actual model/GPU/native profile qualification remains A5/A7 work.
4. The fixed factory remains an external A5 implementation. A3's normal CLI on
   currently reports its absence; it does not secretly start a different loop.
   A6 can consume the same fixed materials/context and A5 implementation;
   no planner loop starts from this module.
5. The only installed KS inputs used by A3 are the selected fixed snapshot.
   K4's changing worktree was read only for its interface handoff and never
   installed or relabelled a fixed release. No EXP access, model/GPU call,
   credential discovery or private qualification publication occurred.

## Validation record

First A3 run: 13 passed. Expanded original-tool/native/composition/nonroot tests:
16 passed in 31.50 s. Latest source adds canonical UUID checks, raw-native-bypass
denial, explicit changed-C assignment identity and exact native-source install;
final focused/combined results will be recorded below after completion.

The synthetic process runner uses the real original Agent and NativeSession;
its native events are explicitly fake. An additional real fresh Python child
executes the stdin shim and proves exact bytes, without invoking a model.
Permission-negative tests retain 9 synthetic native tokens even when discarding
invalid tool/model/role output. Original Task/source/Template/publication code is
real. CPU tests do not establish CUDA correctness, fidelity or three-audit PASS.

Artifacts: `/tmp/schema-b05-a3-tests.log`,
`/tmp/schema-b05-a3-combined-tests.log`, `/tmp/schema-b05-a3-install.log`,
`/tmp/schema-b05-a3-dist/`. A separate base-only AKA environment at
`/tmp/schema-b05-a3-off-venv` explicitly has no KS or optional AKA Schema import;
its real native off probe is compared with the fully installed optional state.

Full engineering contract: AKA `docs/schema-optional-module.md`.

## Fixed K4/K3 handoff update

The supervisor subsequently confirmed the publicly pushed fixed commit
`189a54153a74ac659d26652a9426bc0cc638a59a`. The final A3 gitlink and actual
editable KS installation now use this detached commit. Public GitHub fetch was
attempted twice with HTTP/1.1 and bounded low-speed checks; it failed with an
empty reply and a low-speed timeout. The exact already-confirmed immutable SHA
was then fetched from the local KS object database into the submodule's own
object store. This did not install the changing K5 worktree, add an alternate,
change the public origin or change `.gitmodules` to a local dependency.

The selected release now has 284 files, with manifest digest
`sha256:e819b9b6bc053aef20092060b638aa75a0bdaf68a6c291a41b8b3a1623bebab4`.
The new `verify_release` locates and hashes the code/resources before importing
any KS initializer. A fresh-process negative using a wrong KS package proves
its initializer is not executed.

`schema.authority.NativeCallAuthority.bind(context, original_owner,
original_reservation, source=..., episode=...)` is the A5 thin callback adapter.
It verifies the actual existing context authority and original
`OwnerAccount.operations` member (including a prior group member); it allocates
no quota. `record_locator` writes the existing RunContext intent and rejects
wrong request/source/role. `authorize` directly calls the same existing owner.
`RoleSession.from_authority` connects the callbacks and reuses the fixed K3
`runtime._local.remaining_timeout_ms` helper for original absolute deadlines.
The helper's fixed source is covered by the release manifest. There is no copied
account, timer implementation or ledger. A5 remains responsible for live pause,
native process recovery, complete receipt absorption and audits.

Pre-update combined CPU run: **50 passed in 213.48 s**, including A1/A2 and A3.
The latest pre-update focused run, including initializer-before-hash prevention,
canonical UUID/raw-call negatives and changed-C identity: **17 passed in
33.69 s**. The base-only environment explicitly confirmed both `kernelschema`
and `aka.workflow.schema` absent; real native off probe traces there and in the
optional-installed environment compared byte-for-byte equal (`cmp` exit 0),
including normalized prompts, V0 source/evaluation, limits, episode outcomes and
exports. No old A2 provenance record was rewritten. Final 189a541 pairing tests
are recorded after completion below.

## Final A3 result

The 189a541 fixed pairing passed the full A1/A2/A3 CPU set: **53 passed in
211.30 s**. A final native terminal-model mismatch negative was then added;
the complete A3 focused set passed **20 tests in 38.34 s**. The last composition
guard prevents the no-tools profile from silently replacing a caller-selected
sandbox/custom Agent; its focused test passed **1 test in 5.75 s**. That guard
is an explicit first-profile capability limit, not a claim of process sandbox
qualification. Native profile/model bytes are also required to match the fixed
request before any session starts.

The final optional package was built through the original distribution tool
(sdist, then wheel). Every owned package file in that wheel byte-matches the
final source, its release manifest selects 189a541, and the Framework wheel
contains no `aka.workflow.schema` files. KS source/editable installation remains
separate; the wheel is not presented as a resource-complete KS release.
`git diff --check` passes. The AKA branch and HEAD remain the original
`feat/schema-workflows-v1` / `c984d141145d31795514579db213413a1bbb0775`.
Only `.gitmodules` and the required fixed gitlink are staged; ordinary A3
source/docs/tests use intent-to-add for review. No commit/push was performed.

A3 interface/CPU is complete. A5 receives executable role/source/owner
integration seams, not a purported completed fixed backend. A4 seed recovery,
A5 full native fixed/audit/accounting integration, A6 dynamic composition and
A7 actual native/model/GPU qualification still require their own implementation
and evidence. Host read/effective/check tools remain noninteractive for the
current model profile; fresh explicit-Source continuation is its only supported
session continuation mode.
