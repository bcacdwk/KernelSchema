# Design workflow

Write a Schema from a task description and selected nearby designs/documents:

```sh
kernelschema design --task /materials/task.md --out /runs/design \
  --reference /materials/neighbor.yaml \
  --runtime-profile /materials/claude-profile.json
```

Or use `kernelschema.api.design(task=..., out=..., references=[...])`.
The default makes one planner call and saves the actual YAML, rationale,
diagnostics and call receipts. No evaluator, GPU or independent reviewer is
required. A completed write does not establish correctness or performance.
The CLI prints the Schema directory for downstream AKA implementation.

See [the shared writing interface](../kernelschema/WRITING.md) for profiles,
results, malformed-output retention, optional strict mode and the existing
`--request`/`--continue` interface. The [standalone driver](../docs/workflows/standalone.md)
can explicitly enable finite automatic iteration with AKA's standard workflow.
