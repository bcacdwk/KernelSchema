# Vector-add Schema example

[`root.yaml`](root.yaml) is an unchanged model-generated Schema for the
[native task](task.md). It was written by one real write-only `api.optimize`
call on 2026-11-21 using KernelSchema
[`907c99aa`](https://github.com/bcacdwk/KernelSchema/tree/907c99aa1589bddf73e26079f2304b83d68a8e51).
The API returned `completed` / `revise`, with no diagnostics, reviewer or GPU
call. Its input feedback described an implementation timeout with no new
candidate. This establishes a written proposal; the proposal's implementation
and performance are separate results.

The file is 9,106 bytes, SHA-256
`58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`.
It contains no native receipt, CLI stream or local runtime profile.

## Implement the supplied Schema

Use a Python environment with KernelSchema and the matching AKA packages,
your configured coding CLI and evaluation backend. Follow AKA's
[installation and native Task setup](https://github.com/bcacdwk/atrex-kernel-agent/blob/feat/schema-workflows-v1/docs/schema-standard-workflow.md).
Its [vector-add example](https://github.com/bcacdwk/atrex-kernel-agent/tree/feat/schema-workflows-v1/skills/atrex-aka-light/examples/native-atrex-bench-vector-add)
provides the actual reference, input generator, three numeric shape IDs and
measured L20D reference metadata. Copy that directory into the `data/` tree
of your real Atrex-Bench source checkout. An empty runtime marker is insufficient.
The metadata measures the PyTorch reference, not a new Triton kernel; follow
its instructions for another device, changed inputs or a fresh baseline.

Set the paths to your own checkouts and a new run directory:

```sh
KS=/path/to/KernelSchema
AKA=/path/to/atrex-kernel-agent
PROBLEM=/path/to/atrex-bench/data/native-atrex-bench-vector-add
RUNS=/path/to/new-vector-add-runs
kernelschema implement --schema "$KS/examples/workflows/vector_add/root.yaml" \
  --problem "$PROBLEM" --repo-root "$AKA" --platform L20D \
  --framework Triton --max-iters 1 --out "$RUNS/implement"
```

AKA owns the normal Agent, tools, correctness checks and native ABBA promotion.
To select Schema from AKA's own CLI, add
`--schema --schema-path /path/to/root.yaml` to the normal `aka optimize`
command. Omitting the switch or using `--no-schema` preserves the ordinary path.
This switch injects a fixed Schema into coder episodes; it does not itself
invoke design or optimize between episodes. The old `--schema-config` dynamic
controller remains a separate legacy path with its original qualifications.

## Design → implement → optimize → implement

The file APIs work for a caller-supplied task and runtime. Fill in the paths
and the native AKA settings, runtime and composition patches you intend to use.
The sequence makes at most one design and one optimize call, and at most two
native campaigns; each campaign below permits one coding iteration.

```python
import json
from pathlib import Path
from kernelschema import api

example = Path("/path/to/KernelSchema/examples/workflows/vector_add")
problem = Path("/path/to/atrex-bench/data/native-atrex-bench-vector-add")
runs = Path("/path/to/new-vector-add-runs")
writing = dict(
    task=example / "task.md",
    references=[problem / name for name in ("reference.py", "input.py", "shapes.json")],
    runtime_profile="/path/to/your-writing-runtime-profile.json",
)
native = dict(
    problem=problem, repo_root="/path/to/atrex-kernel-agent",
    platform="L20D", framework="Triton", agent_cli="claude", max_iters=1,
    # Add your existing settings=..., runtime=..., patches=(...,) as needed.
)

def published_schema(out, result):
    info = result.get("extensions", {}).get("information", {})
    relative = info.get("schema_directory")
    if result.get("status") != "completed" or not relative:
        raise RuntimeError(f"Read retained writing diagnostics: {info.get('diagnostics')}")
    return out / relative

d_out = runs / "design"
design = api.design(out=d_out, **writing)
s1 = published_schema(d_out, design)
i1 = api.implement(schema=s1, workspace=runs / "implement-1", **native)
i1_file = runs / "implement-1-result.json"
i1_file.write_text(json.dumps(i1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if i1.get("status") == "blocked":
    raise RuntimeError(f"Resolve the native run before continuing: {i1.get('workspace')}")

o_out = runs / "optimize"
optimized = api.optimize(
    schema=s1, feedback=[i1_file, *i1.get("feedback_files", [])],
    out=o_out, **writing,
)
if optimized.get("status") == "completed" and optimized.get("decision") == "revise":
    s2 = published_schema(o_out, optimized)
    i2 = api.implement(schema=s2, workspace=runs / "implement-2", **native)
    (runs / "implement-2-result.json").write_text(
        json.dumps(i2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
```

A `keep` decision ends the sequence. Preserve the full writing directories and
native results. Failed or partial writing retains diagnostics; inspect a readable
raw file explicitly before implementing it. An interrupted native campaign needs
settlement in its original workspace before further work. Native `completed`
alone, or an incumbent's carried-forward `PASS`, does not establish a new
candidate's correctness or promotion.

This snippet is a usage example, not a recorded four-step successful run.
For the existing automatic finite driver, see
[standalone workflow instructions](../../../docs/workflows/standalone.md#finite-automatic-iteration).
That driver has a fixed row-softmax task; it is not a vector-add or arbitrary-task
driver. The direct file APIs above provide the task-specific sequence without
adding a scheduler, evaluator or recovery framework.
