# K2 interface: contracts, files and run boundaries

This is the implemented CPU handoff for `ks.workflow.v1`. The authoritative
contract remains `schema_discussion/workflow_architecture_20261120/planning/r5_contract_execution_review.md`.
K2 does not implement model calls, evaluator execution, fidelity decisions,
writing or the fixed implement loop. Current business entry points report
unsupported until the owning backend is installed. CPU synthetic fixtures
never establish runtime, Task, model, GPU or product qualification.

## Installation and entry points

Source checkout/editable on Linux, Python >=3.11:

```sh
python -m pip install -e .
kernelschema --help
kernelschema implement --request /bundle/request.json --out /new/run
kernelschema implement --continue /original/run
kernelschema extract -- --help
```

`kernelschema.api.{design,optimize,implement}` use exactly the CLI implementation.
Import these functions from `kernelschema.api`; the package-level names are
reserved for backend modules and are not callable aliases.
Each accepts keyword-only `request` (path to request.json) + `out`, or
`continue_run`. Mixing these groups fails. Return values are validated result
dictionaries; CLI exit 0 means the result projection was durably published,
not that the domain status is completed. Unsupported/admission/publishing
errors are `ProtocolError` (`UnsupportedError` is a subtype), CLI exit 2.
No import of this package loads AKA, torch, CUDA or a runtime backend.
An explicit implement request with backend `aka` lazily loads the separately
installed `aka.workflow.schema.backend.workflow_backend()`; other operations
do not discover or import AKA. Missing adapter code is an unsupported backend,
not an instruction to install it or start another top-level workflow.

`resources.resource(relative_path)`, `rule_files()` and `freeze_rules(publication)`
locate/freeze the actual checkout files independent of cwd. This release does
not promise a resource-complete wheel or a downloader. PyYAML and jsonschema
are CPU dependencies; the latter is required to start the original extract CLI.

## Canonical wire objects

`contracts.Ref` and `Check` are public TypedDicts. `validate(dict)` rejects unknown
fields/kinds/versions; it checks one object's shape. `loads(bytes)` rejects BOM,
duplicate keys, unsafe integers, floating JSON numbers, NaN/Infinity and Unicode
surrogates. `jcs(value)` emits RFC 8785 UTF-8, no newline: actual UTF-16 property
ordering and JSON escaping, safe integers only. Fractional metrics are strings.
`sha256(bytes)` and `digest(value)` both return the `sha256:`-prefixed identity.
Set-array order is checked, not silently repaired; ordinary arrays retain order.

The C-defined Task/Schema/Rules/Materials/Scope/Source/request/result, budget,
Check, usage-index and report shapes are implemented in `contracts.validate`.
These concrete auxiliary shapes fix the fields that C leaves to K2/evaluator
integration, without inventing an alternative Schema language:

| Kind | Required fields beyond protocol/kind |
| --- | --- |
| backend_profile | name, version, runtime, model (explicit strings); parameters, native_limits (objects); environment Ref; capabilities |
| evaluator | version; entrypoint Ref included in files; files sorted by path; inputs, timing, environment Refs; adapters (explicit string IDs) |
| evaluation | operation_id; C subject hashes; build/correctness/performance Checks; case_ids; samples Ref/null; raw_receipts and usage_receipts lists; environment Ref to actual reported snapshot |
| writing_review | operation_id; subject hashes request/task/output-schema/rules/output-scope; checks (structure/permissions/semantics/publication); evidence, usage_receipts |
| change_report | operation_id; subject hashes input-request/task; changes [{location,reason}]; raw (candidate byte Refs); evidence |
| feedback | subject hashes request/task/schema/source; reason; evidence |
| usage_index | entries with the exact C usage-index row fields |

`capabilities` has exactly protocols/operations/modes/source_formats/recovery/
extensions (arrays), empty_start (bool), token_enforcement
(hard/admission_only/unsupported). An actual backend declaration and its frozen
profile must both support a requested combination. A finite token budget needs
actual hard enforcement. `extensions.information` is inert descriptive content;
all other owners require explicit backend capability support. Private evaluator
files cannot be selected for model roles by Materials. Role delivery still
requires K3's actual runtime allowlist enforcement.

Optimize results include `action_reason` (null permitted when next_action=none),
`action_inputs:[Ref]`, `action_evidence:[Ref]`; an action requires nonempty explicit
inputs and a reason. All result subjects and checks are verified against actual
selected manifests/reports before publication. This is binding validation, not
proof that a raw report is trustworthy; K1/K4/K5 still own execution/review trust.

## Bundle and publication APIs

`Bundle(root).read(ref)` verifies exact regular-file bytes, no symlink traversal.
`document(ref, kind=None)` additionally validates normative JCS.
`source(ref)` returns original kernel.py bytes after hash/size/mode/UTF-8 checks.
`install_source(bundle, ref, new_directory)` installs one complete snapshot via
fsynced no-replace rename. It refuses an existing draft; K4 owns authorized seed
intent/ready/admitted transitions and later draft replacement.

`Publication(root, publication_id=None)` durably records a random 128-bit ID
before staging. `put_bytes(name,bytes,mode=0o644)`, `put_json(name,dict)` and
`put_source(name,bytes)` return stable `artifacts/<id>/...` Refs. `commit(roots,
subjects=[...])` freezes selected actual dependencies. Reopen with the same ID
for the same operation; different bytes or a different published closure need
a new publication. No snapshot is hardlinked to the mutable input/workspace.

`Bundle.closure(roots, subjects=[...])` and `import_bundle(source,destination,
roots,subjects=[...])` follow only supported object semantics: explicit Ref
fields, Source files, Template's real raw/from chains, Schema parent/report,
and original `schema_extract.delivery.v2` file-manifest edges. They do not scan
ordinary prose or all historical files. `subjects` are explicit immutable Ref
locators (often a previous request snapshot), not a hash-to-guessed-path table.
Selected evidence must locate actual Q/Task/R/C/rules/method/evaluator subjects.
Only C may be imported without its historical Q. Missing unrelated history is
allowed; missing dependencies or conflicting bytes/modes reject the transaction.

New destinations publish by no-replace directory rename. Existing destinations
preflight and stage the entire selected closure, then merge under a local flock
with fsynced rollback intent. Ordinary errors roll back. Process death leaves
`.runtime/import-intent.json`; all Bundle reads are fenced until
`recover_import(destination)` rolls back the partial merge. Raw filesystem
observers are not promised multi-file atomic visibility. This is qualified local
Linux durability (files and relevant directories fsynced), not cross-machine
recovery or a concurrent arbitrary-writer storage service.

`template_api` loads the original parser at `kernelschema._template_tools`, never
as bare `tools` and never by changing sys.path. `resolve_bundle`,
`effective_document`, `normalize_directory`, `check_directory` reuse original
parsing/normalization/check/provenance. Non-root effective documents include
root Contract. Scope node existence is checked; Task-specific when/variant/
coverage-partition semantics and execution admissibility remain with the owning
Task/writing/backend implementation.

## Backend and owner handoff

K4 exposes `kernelschema.implement.reference.workflow_backend()` for backend
`reference`; K5 exposes `kernelschema.design.workflow_backend()` and
`kernelschema.optimize.workflow_backend()` for backend `claude`. These factories
return the narrow `api.Backend` protocol:

- `capabilities(profile)`: actual supported combinations, no trial GPU invocation.
- `identity(profile)`: fixed implementation/version/source identity dictionary,
  retained in the run anchor and compared on continue; exclude credentials.
- `validate_resume(context)`: validate the actual owner ledger, all operations,
  fences, installations and native locators before execute. Missing state cannot
  allocate a replacement balance.
- `execute(context, continuing=bool) -> result`: implement the business loop,
  settle original operations before new admission, return an honest result.

`prepare_run(operation,request=...,out=...,backend_identity=...)` freezes inputs
and a canonical immutable request before returning `RunContext`. `open_run`
checks Q, operation, original request, profile, implementation and owner binding
without depending on result.json. These are engineering APIs, not execution
admission or a substitute for the owner's ledger. Explicit new runs with equal Q
have distinct run IDs and owner identities (unless a fixed parent account owns
them); there is no global dedup promise.

`RunContext` provides `root`, `run_id`, `request`, `request_sha256`, `request_ref`,
`owner`, `objects`, `bundle`; durable `save_state(namespace,dict)` /
`load_state(namespace)`; immutable `record_intent(operation_id,role,purpose,
source,reservation,submission_key,native_locator,episode=None)`; `intents()`;
`record_terminal(operation_id,facts)`; and `publish_result(result)`.
Reservation dictionaries are the sole owner's actual durable authorization;
K2 grants no quota and computes no ledger. Native roots/keys/handles belong in
intent/private state before execution, never only in the final return.
Publication facts and immutable result history survive loss of the top-level
result projection; `publish_result` reuses the original ID for identical result
bytes. K4/K5 decide how to settle and reconstruct a result from their facts.

## Extract and shared fixtures

`extract_bridge.extract(argv)` runs the unchanged original executable in a
separate interpreter and returns its exit status. `consume_raw(publication,
directory,task=Ref,origin=...,receipt=None)` snapshots actual Template raw and
optional original receipt/dependencies, returning Schema/receipt Refs plus
explicit unknown historical usage. Handwritten, curated, design and extract
sources use this same bridge without a universal P6 gate. `implement_request`
constructs the fixed public request after the caller provides the complete Task,
role materials, seed and mode. Original-code verification requires explicit
nonempty seed+verify_only; role/source information boundaries remain explicit.

The shared generator is `tests/workflows/fixtures/v1/row_softmax/build.py`.
It reads pinned K1 inputs by actual hash, emits fully verifiable synthetic
bundles, and does not copy Task criteria into a second maintained fixture.
The JCS byte vectors are shared in the same directory.
