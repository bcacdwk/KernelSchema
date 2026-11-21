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

Use the finite `iterate` entry with the explicit task and native problem. Each
Schema version gets a new native workspace, and the original coder can run
multiple iterations within that version:

```sh
kernelschema iterate --task "$KS/examples/workflows/vector_add/task.md" \
  --problem "$PROBLEM" --repo-root "$AKA" --platform L20D \
  --runtime-profile /path/to/your-writing-runtime-profile.json \
  --reference "$PROBLEM/reference.py" --reference "$PROBLEM/input.py" \
  --reference "$PROBLEM/shapes.json" \
  --rounds 2 --max-iters 4 --out "$RUNS/new-iteration"
```

This starts with design. To begin with the supplied proposal, add
`--schema "$KS/examples/workflows/vector_add/root.yaml"`.
`--rounds 2` permits at most two Schema versions and one optimize call;
`--max-iters 4` permits up to four native coder iterations per version. A `keep`
decision ends the sequence without another implementation.

The same loop can be called directly, with your existing native settings and
composition patches:

```python
from pathlib import Path
from kernelschema import api

example = Path("/path/to/KernelSchema/examples/workflows/vector_add")
problem = Path("/path/to/atrex-bench/data/native-atrex-bench-vector-add")
result = api.iterate(
    task=example / "task.md", schema=example / "root.yaml",
    out="/path/to/new-vector-add-run", rounds=2,
    references=[problem / name for name in ("reference.py", "input.py", "shapes.json")],
    runtime_profile="/path/to/your-writing-runtime-profile.json",
    problem=problem, repo_root="/path/to/atrex-kernel-agent",
    platform="L20D", framework="Triton", agent_cli="claude", max_iters=4,
    # Add your existing settings=..., runtime=..., profile=..., patches=... as needed.
)
```

AKA also exposes this shared loop:

```sh
aka optimize --repo-root "$AKA" --problem "$PROBLEM" \
  --platform L20D --framework Triton --max-iters 4 \
  --framework-baseline never --convert-after 0 \
  --schema --schema-iterate --schema-task "$KS/examples/workflows/vector_add/task.md" \
  --schema-path "$KS/examples/workflows/vector_add/root.yaml" \
  --schema-runtime-profile /path/to/your-writing-runtime-profile.json \
  --schema-reference "$PROBLEM/reference.py" --schema-reference "$PROBLEM/input.py" \
  --schema-reference "$PROBLEM/shapes.json" \
  --schema-rounds 2 --schema-out "$RUNS/new-aka-iteration"
```

`iterations.json` lists separate writing and native stages. The complete native
result and byte-preserved declared feedback go to optimize, including terminal
implementation failures. A blocked, unsettled or interrupted native stage stops;
missing feedback or writing without a completed readable Schema also stops.
Preserve the whole run. Any existing output directory is rejected, so the loop
cannot replay an interrupted, possibly charged stage. Settle that original
operation using its own records before starting new work.

Native `completed` alone, or an incumbent's carried-forward `PASS`, does not
establish a new candidate's correctness or promotion. These commands document
the CPU-tested entry, not a recorded successful model/GPU closed loop. See the
[standalone workflow instructions](../../../docs/workflows/standalone.md#finite-automatic-iteration)
for API options and the separate fixed row-softmax example.
