# Implement workflow

`kernelschema.api.implement` and `kernelschema implement` execute the same finite
reference backend. Source checkout/editable installation requires no AKA.

```python
from kernelschema.api import implement
result = implement(request="/bundle/fixed.request.json", out="/new/run")
result = implement(continue_run="/new/run")
```

The [standalone driver](../docs/workflows/standalone.md) freezes real requests and
runs writing, fixed implementation and the authorized next action on one owner.
`seed=null` selects Task.starter_source; an explicit Source selects those exact
bytes. `verify_only` requires nonempty Source and starts zero coder calls.
The evaluator supports the full four-shape Task only; partial scope is rejected.

The repository supplies the [Agate channel and qualification entry](../docs/workflows/standalone.md#agate).
No actual GPU qualification has been established in this delivery. Missing
configuration/qualification returns blocked while retaining the exact seed.
See [the reference interface](../kernelschema/implement/INTERFACE.md) for the
five independent implementation gates, original-operation recovery and budget.
