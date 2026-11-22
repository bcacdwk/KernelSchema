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

## Measured solution

[`kernel.py`](kernel.py) is the unchanged 2,089-byte Triton candidate accepted
by the native host on 2026-11-21. Its SHA-256 is
`105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f`.
The independent host ABBA measured **1.503522×** geometric-mean speedup over
the PyTorch reference: 11.226588 µs baseline and 7.466858 µs candidate, for the
three shapes in [the task](task.md). Candidate full evaluation and the separate
correctness gate each passed six cases per shape with maximum absolute error 0.

In a CUDA environment with PyTorch and Triton, use it from this directory:

```python
import torch
from kernel import Model

x = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
y = Model(alpha=1.0)(x, residual)
```

The final validation used KernelSchema `13efd1554d112d56e4d767ba43b66257b2d75669`
and AKA `55bc39f3239e2d9fa65d2a799182a26df413368c`, through one `api.implement`
campaign. It supplied the earlier measured candidate as an explicit `docs`
reference; the coder materialized identical bytes in the new workspace. The
earlier finite run below generated that candidate from the supplied Schema.
This final campaign added independent measurements and host promotion without
a new Schema or writer call.

| Final campaign job | Purpose | Result |
| --- | --- | --- |
| `ev_4c605098b3bc` | New PyTorch reference V0 | Three shapes, six cases each passed |
| `ev_7102dba05dfd` | Candidate full evaluation | Three shapes, six cases each passed |
| `ev_7840da47d4aa` | Agent same-allocation ABBA | Complete and valid |
| `ev_9e107207e34c` | CandidateGate correctness-only | Three shapes, six cases each passed |
| `ev_53c06a9e4f27` | Independent host ABBA | `accepted=true`, verification `PASS`, `reused=false` |

The host timing above is AKA's projection of that single host job: one complete
A→B→B→A block, two baseline and two candidate samples per shape. Each complete
ABBA run used one correctness case; the separate full and gate jobs supplied
the six-case checks. [`results.json`](results.json) records the source hashes,
candidate/promotion commits, timestamp, receipt hashes and per-shape host timings.
Environment identity remains unknown; the measurements cover these three shapes
on this L20D allocation. The timing samples include outliers, and this single
block does not establish statistical significance or performance on other tasks.

## Implement the supplied Schema

Use a Python environment with KernelSchema and the matching AKA packages,
your configured coding CLI and evaluation backend. Follow AKA's
[matching source/package installation](https://github.com/bcacdwk/atrex-kernel-agent/blob/feat/schema-workflows-v1/docs/schema-standard-workflow.md#install-and-legacy-compatibility)
and [native Task setup](https://github.com/bcacdwk/atrex-kernel-agent/blob/feat/schema-workflows-v1/docs/schema-standard-workflow.md#native-task-and-performance-baseline).
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

A new workspace does not automatically make the previous version's kernel the
next campaign's incumbent. Each campaign starts from the supplied native
problem and ordinary AKA setup. To provide an earlier implementation as coding
reference, pass its explicit file with `--doc` (AKA: `--schema-doc`; API:
`docs=(path,)`) and keep its provenance. That supplies text to the coder, not an
automatic promotion or a guarantee that performance improves between versions.

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

Read each native stage's `schema-result.json`, version/episode feedback and
candidate source together. `candidate_commit` identifies the candidate;
`measurement_subject` and `measurement_source` distinguish its measurements
from an incumbent's carried-forward result. Native `completed` alone, or an
incumbent's carried-forward `PASS`, does not establish candidate correctness or
host promotion. Check the original correctness results and host verification
status before interpreting a speedup.

When the service omits environment identity, it stays `null` and cross-record
comparability stays `unknown`. Complete source-bound results from one native
evaluation and a complete SDK same-allocation A→B→B→A comparison can be reported;
unknown identity does not permit cross-record measurement reuse or shape minima.
See the
[standalone workflow instructions](../../../docs/workflows/standalone.md#finite-automatic-iteration)
for API options and the separate fixed row-softmax example.

## Recorded finite iteration

On 2026-11-21, a real `api.iterate` run used this unchanged Schema with
KernelSchema `13efd1554d112d56e4d767ba43b66257b2d75669` and AKA
`88d773b3acdd60ed13334dc46404e2d0caefe49f`. The first campaign generated a Triton
candidate whose full evaluation and six-case correctness check passed for all
three task shapes (maximum absolute error 0). Its Agent-issued same-allocation
ABBA passed, reporting a 2.350190× geometric-mean speedup over the PyTorch
reference within that comparison.

Host promotion failed: after the four accepted jobs (new reference V0, candidate
full evaluation, Agent ABBA, and six-case correctness), its separate host ABBA
had no remaining job allocation. No fifth job was submitted. The finite loop
passed that actual failure to one write-only optimize call, which completed
with `keep`; it stopped without a second native campaign. This demonstrates
real implementation feedback and automatic stopping, while preserving the
failed host verification. If both Agent and host perform ABBA, allow for both
jobs in the normal native budget; the four-job allocation did not cover this
observed path.

The final campaign above used five new jobs and obtained its own successful
host verification. It leaves the earlier failed host result intact. The earlier
2.350190× Agent ratio and the final 1.503522× host ratio belong to separate
allocations; they do not establish a performance change between campaigns.
