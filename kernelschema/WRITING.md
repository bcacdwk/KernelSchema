# Schema writing

Design and optimize default to **write-only**: one bounded planner call saves
Schema YAML, a short rationale, diagnostics and the actual native call records.
`completed` means readable files were written. It does not mean correctness,
performance or the design's effect was evaluated. No reviewer or GPU runs.

## File-oriented API and CLI

```python
from kernelschema.api import design, optimize

result = design(task="/materials/task.md", out="/runs/design",
                references=["/materials/neighbor.yaml", "/materials/hardware.md"],
                runtime_profile="/materials/claude-profile.json")
# result['extensions']['information']['schema_directory'] is relative to /runs/design.
result = optimize(task="/materials/task.md", schema="/materials/current-schema",
                  feedback=["/materials/actual-feedback.txt"], out="/runs/optimize",
                  runtime_profile="/materials/claude-profile.json")
```

```sh
kernelschema design --task /materials/task.md --out /runs/design \
  --reference /materials/neighbor.yaml --runtime-profile /materials/claude-profile.json
kernelschema optimize --task /materials/task.md --schema /materials/current-schema \
  --feedback /materials/actual-feedback.txt --out /runs/optimize \
  --runtime-profile /materials/claude-profile.json
kernelschema design --continue /runs/design
```

Task, reference, objective, environment and feedback files are UTF-8 and are
copied in full. Supply only the documents and nearby designs relevant to this
task. The current Schema and feedback are delivered unchanged. Writing does not
require an evaluator or baseline implementation. Unknown target facts stay
unknown; the task description is not a device probe.

`runtime_profile` may be a `ClaudeProfile`, its dictionary or JSON file. Without
one, the entry point inspects the installed Claude CLI and uses the existing
CLI authentication, `qwen3.8-max[1m]`, high effort, a 600-second call timeout and
a USD 2.00 native stopping threshold. This threshold is not a hard billing cap.
There is no automatic retry. An explicit profile preserves its chosen model,
effort, timeout and threshold. The simpler prompt's real-model outcome has not
been established by the CPU tests.

## Files and results

The normal planner content is small:

```json
{"raw":[{"node":"root","text":"complete YAML text"}],"reason":"What was chosen or changed, and why"}
```

Every `raw` entry is saved as `<node>.yaml`. Put Markdown in the separate
`"dispatch":"complete Markdown text"` field, saved as `dispatch.md`; a raw node
named `dispatch` remains YAML. Optional `bindings` maps coverage IDs to nodes.
Optimize may return `{"decision":"keep","reason":"why"}`. There is no
required selection catalog, locked Contract wrapper, 11-field action envelope
or separate review response. Additional old envelope fields are ignored in
write-only mode. No proposed next action is executed by a writing command.

The runtime envelope still binds protocol, planner role, assignment, status and
output kind. In write-only mode the explanation can live solely in
`content.reason`; an omitted or unusable outer reason and extra outer metadata
do not reject the draft. Such metadata never supplies runtime facts. Strict
writing, coder and reviewer protocols retain their exact field requirements.

`result.schema` and `result.scope` remain the existing portable Refs. The Schema
manifest contains the exact `raw` files, optional dispatch and original parent
for a revision. `extensions.information` records:

- `schema_directory`: the directory relative to the run root, usable by a coder;
- `rationale` and `change_report`: the explanation and exact retained changes;
- `candidate_draft`: parsed content and native reply strings in
  `native_responses`, including failed calls;
- `retained_files`: the available original raw and dispatch text;
- `diagnostics`, `writing_mode`, and the explicit unevaluated qualification.

Only safe filenames, unique nodes including `root`, UTF-8 and basic nonempty
YAML mappings are required for a completed write. Template lengths, derived
Contract equivalence, dispatch semantics and performance are not acceptance
gates. Missing optional mapping hints produce diagnostics. Malformed YAML is
saved with a partial result; an unusable envelope or unsafe filename leaves the
original proposal text and a clear failure diagnosis. The host does not trim,
rewrite or discard a draft to satisfy Template text limits. Original runtime
failures retain their receipts and native outputs; they are not recast as success.
Unreadable reply JSON remains a text artifact. Failed calls, mismatched identities
and unknown native usage cannot publish a successful Schema; their available
text is retained for inspection. Existing failed runs and accounts are immutable:
this change does not reclassify or rerun them.

The writing checks report basic readability/publication and `not_run` for
semantic review. Downstream implementation consumes the files and evaluates the
actual code through AKA's standard workflow:

```sh
kernelschema implement --schema /runs/design/artifacts/OUTPUT_ID/raw \
  --problem /bench/problem --repo-root /checkouts/atrex-kernel-agent \
  --platform L20D --out /runs/implementation
```

The CLI prints the concrete Schema directory, so users need not construct the
artifact path. [`api.iterate` / `kernelschema iterate`](../docs/workflows/standalone.md#finite-automatic-iteration)
can also perform a bounded design → AKA implement → optimize → implement sequence
for the same explicit task. Schema-version rounds and the native coder's
`max_iters` are separate limits; optimize receives the original native results.

## Existing request and strict interfaces

The existing `--request`/`--continue` API, immutable Ref bundle, native receipts
and one OwnerAccount remain available. `writing.freeze_rules(publication)` and
`writing.backend_profile(profile, environment_ref)` now default to write-only.
Callers can explicitly select additional repository guidance with
`freeze_rules(..., guidance=[...])`. The default includes the writing method,
current design/optimize guide, common field guidance and starter YAML; host
checker implementations and the unrelated operation guide are not sent as
planner instructions. Explicit caller references are never truncated.

For historical strict behavior, select `writing_mode="strict"` in both
`freeze_rules` and `backend_profile`. That path retains its Task adapter,
Template precheck, independent review and four checks. Its old failed records
remain failed. Strict requests need planner/reviewer budgets; write-only defaults
to planner 1, reviewer 0, evaluator jobs 0. An explicitly supplied finish floor
or external parent still applies. The new file API creates a finite one-call
writing request; existing parent users can keep constructing requests with the
original public parent Ref and `use_authority`.
