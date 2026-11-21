# Implement workflow

`kernelschema.api.implement` and `kernelschema implement` pass a raw Schema file
or directory to AKA's standard Light/Agent/Executor/Evaluator workflow:

```python
from kernelschema.api import implement

result = implement(
    schema="/materials/schema", problem="/bench/problem",
    repo_root="/checkouts/atrex-kernel-agent", platform="L20D",
    workspace="/runs/implementation", framework="Triton", max_iters=4,
)
```

```sh
kernelschema implement --schema /materials/schema --problem /bench/problem \
  --repo-root /checkouts/atrex-kernel-agent --platform L20D \
  --framework Triton --max-iters 4 --out /runs/implementation
```

Install matching AKA packages and KernelSchema source/editable in the same
Python environment; follow AKA's [installation and native Task preparation](https://github.com/bcacdwk/atrex-kernel-agent/blob/feat/schema-workflows-v1/docs/schema-standard-workflow.md).
The normal full evaluation needs real per-shape performance metadata for the
selected hardware. `prepare_only=True` / `--prepare-only` loads inputs without
calling the model or evaluator. Keep the original result and feedback: native
completion alone does not establish candidate correctness or promotion.

The [standalone driver](../docs/workflows/standalone.md) explains file writing
and the finite design → standard implement → optimize example. Writing and
legacy qualification checks do not gate standard implementation.

## Legacy request and continuation

Explicit `request` / `continue_run` selects the historical fixed-request
interface, mutually exclusive with the standard `schema` arguments:

```python
from kernelschema.api import implement
result = implement(request="/bundle/fixed.request.json", out="/new/run")
result = implement(continue_run="/new/run")
```

The legacy reference backend needs no AKA installation.
`seed=null` selects Task.starter_source; an explicit Source selects those exact
bytes. `verify_only` requires nonempty Source and starts zero coder calls.
The evaluator supports the full four-shape Task only; partial scope is rejected.

Its [readonly staging and qualification boundary](../docs/workflows/stage-ro.md)
remains an explicit compatibility path. Missing configuration/qualification
returns blocked while retaining the exact seed.
See [the reference interface](../kernelschema/implement/INTERFACE.md) for the
five independent implementation gates, original-operation recovery and budget.
