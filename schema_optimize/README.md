# Optimize workflow

Write a revision from the current Schema and actual implementation feedback:

```sh
kernelschema optimize --task /materials/task.md --schema /materials/current-schema \
  --feedback /materials/actual-feedback.txt --out /runs/optimize \
  --runtime-profile /materials/claude-profile.json
```

Or use `kernelschema.api.optimize(task=..., schema=..., feedback=[...], out=...)`.
The current files and supplied feedback are delivered unchanged. One planner
call may retain the original Schema or write new YAML and an explanation.
No reviewer, GPU, improvement test or next-action envelope is required.
`completed` means the files were written; the effect has not been evaluated.

See [the shared writing interface](../kernelschema/WRITING.md) for artifacts,
diagnostics and optional strict compatibility. The [standalone driver](../docs/workflows/standalone.md)
passes AKA's original results to optimize when automatic iteration is selected.
