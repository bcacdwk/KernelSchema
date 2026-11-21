# Standalone writing and implementation

Design and optimize write Schema files with one bounded planner call. Use the
[file-oriented API/CLI](../../kernelschema/WRITING.md) for an arbitrary task:

```sh
kernelschema design --task /materials/task.md --out /runs/design \
  --runtime-profile /materials/claude-profile.json
kernelschema optimize --task /materials/task.md --schema /materials/schema \
  --feedback /materials/actual-feedback.txt --out /runs/optimize \
  --runtime-profile /materials/claude-profile.json
```

The CLI prints the output directory. Keep the whole run to retain the original
proposal, rationale and native call records. Basic YAML errors preserve the
original text with diagnostics. Writing completion makes no claim about the
design's effect. There is no independent reviewer or GPU prerequisite.

## AKA standard implementation

Provide the Schema directory and a normal native Atrex-Bench problem directory:

```sh
kernelschema implement --schema /materials/schema --problem /bench/problem \
  --repo-root /checkouts/atrex-kernel-agent --platform L20D \
  --framework Triton --max-iters 4 --out /runs/implementation
```

`kernelschema.api.implement(schema=..., problem=..., repo_root=..., workspace=...,
platform=...)` calls `aka.workflow.schema.standard.implement` directly. AKA owns
its normal Light/Agent/Executor/Evaluator workflow and native results. Install
that AKA checkout in the same Python environment; follow its
[matching package installation and native Task baseline steps](https://github.com/bcacdwk/atrex-kernel-agent/blob/feat/schema-workflows-v1/docs/schema-standard-workflow.md).
`--prepare-only` prepares the
normal inputs without running the model or evaluator. Writing checks and legacy
readonly qualification do not gate this path. Native failures remain failures.

## Finite automatic iteration

The row-softmax example can prepare an isolated writing run without model or GPU
calls:

```sh
python examples/workflows/row_softmax/run_demo.py prepare \
  --out /tmp/softmax --start design --runtime-profile /materials/claude-profile.json
python examples/workflows/row_softmax/run_demo.py run --demo /tmp/softmax
```

This default stops after design and saves its actual Schema. `--start optimize
--schema /materials/schema` starts from explicit current files. `--objective`
and `--environment` accept current task instructions and target documentation.
The example's public row-softmax Task remains fixed; use the file-oriented CLI
for a different Task.

To enable automatic iteration, supply `--aka-config /materials/aka.json` when
preparing the demo. The JSON contains native AKA options:

```json
{
  "problem": "/bench/row-softmax",
  "repo_root": "/checkouts/atrex-kernel-agent",
  "platform": "L20D",
  "framework": "Triton",
  "agent_cli": "claude",
  "max_iters": 2
}
```

The finite sequence is design → standard AKA implementation → one optimize.
If optimize writes a revision, one final AKA implementation follows. Keeping
the design ends the sequence. The driver retains the complete original AKA
result and declared feedback files, including errors and unavailable timings,
for the next writer. It does not add a second comparator or fabricate passing
assessments. Native AKA owns its normal budgets by default. For an explicit
shared existing owner, add `"share_owner": true` and `"owner_limits"` with the
finite model/evaluator slots supported by AKA's optional adapter. Historical
strict-run costs and results are not rewritten or treated as credits.

Original run directories are fixed. `continue --demo /tmp/softmax` reopens the
same writing operations and retained results. An interrupted native campaign
without a saved result requires settlement in its original workspace; the
driver does not automatically replay a possibly charged operation. Use `inspect`
for the original status and `export --out /destination` for selected artifacts.

## Compatibility and current validation

`--writing-mode strict` explicitly selects the historical planner → Template
precheck → readonly reviewer path. Its `fixed`, `partial`, `verify`, `qualify`
and remote qualification interfaces remain available for old reference-backend
experiments. They are not prerequisites for file writing or standard AKA
implementation. See [the protocol](protocol.md), [target metadata](target-environment.md)
and [legacy readonly boundary](stage-ro.md) when working on that old path.

The new write-only and driver tests use synthetic model/native responses on CPU.
They verify file preservation, finite calls, original feedback, resume behavior
and the API handoff. They do not establish real model or GPU performance. A
runtime profile remains an explicit local CLI choice; no interpreter, CLI or
remote tools are downloaded by this workflow.
