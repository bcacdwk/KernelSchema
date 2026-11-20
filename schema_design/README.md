# Design workflow

Use the installed source checkout and the single public API:

```python
from kernelschema.api import design
result = design(request="/bundle/request.json", out="/new/design-run")
# Original operations, inputs and budget remain fixed on recovery:
result = design(continue_run="/new/design-run")
```

The CLI equivalent is `kernelschema design --request ... --out ...`, or
`kernelschema design --continue ...`. The backend factory is
`kernelschema.design.workflow_backend()`, name `claude`. The package-root
`design` name is a module; import callable functions from `kernelschema.api`.
See [the shared writing interface](../kernelschema/WRITING.md) for request
construction, exact draft/review formats, publication and limits.

This implementation has CPU engineering validation with an actual fake CLI
subprocess. It does not claim a new real model call, device binding, CUDA Task
qualification, or measured performance.
