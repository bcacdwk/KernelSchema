# Optimize workflow

```python
from kernelschema.api import optimize
result = optimize(request="/bundle/request.json", out="/new/optimize-run")
result = optimize(continue_run="/new/optimize-run")
```

The CLI equivalent is `kernelschema optimize --request ... --out ...`, or
`kernelschema optimize --continue ...`. The factory is
`kernelschema.optimize.workflow_backend()`, name `claude`; the function lives
in `kernelschema.api`. Both writing operations use the same implementation.

Optimize reads only the explicitly selected current Schema, role-granted
Materials and declared feedback/evidence. A completed `keep` with
`next_action=none` stops normally. `implement` and `verify` describe explicit
next fixed requests for the owning driver to admit on the same account.
A published revision does not itself admit a new request or qualify any code.
See [the shared writing interface](../kernelschema/WRITING.md).
