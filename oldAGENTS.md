# KernelSchema authoring

- The language is defined jointly by schema_template/schema.cue (package kernel) and schema_docs/1–6.
  Keep these consistent. Examples, including Markdown examples and old
  schema_wiki files, are illustrative evidence rather than overriding rules.
- Author and edit plans directly in YAML. Start with
  python3 schema_template/tools/schema.py init <directory>, then fill root.yaml.
  See schema_template/README.md for the full workflow and boundaries.
- Validate the whole task with python3 schema_template/tools/schema.py validate <directory>.
  CUE defines structural constraints; the validator reads YAML without modifying
  it. Keep local structural rules in CUE and complex references, graphs and
  notation parsing in Python; do not maintain the same check in both.
  root.yaml owns the shared Contract; each <id>.yaml contains a complete
  derived Design, matching task identity and dispatch.id, and no Contract copy.
- Increment that node's meta_data.version when its plan data changes. Coder
  feedback is edited in YAML and retains the original choice and reason.
  Prefer inline mappings for short records, inline lists for shapes and ranges,
  and multiline mathematical text. Expanded YAML is equally valid.
- Contract, Design and Realization ownership follows README and the six docs.
  The validator does not grant permission to change a Contract or redesign a
  pipeline. Changing templates is language development, distinct from filling a plan.
- Update corresponding normative Markdown and meaningful positive/negative
  tests when changing the language. Do not weaken a rule just to admit a legacy
  example. Do not migrate old YAML unless that migration is part of the task.
- Run make check before delivering changes. make setup installs the pinned
  CUE binary locally; Python's standard library is sufficient for the tools.
  CI validates maintained YAML examples and regression fixtures; historical
  schema_wiki cases are checked only when explicitly selected, without exceptions.
  Successful checks establish structure and listed relations, not numerical
  correctness, hardware feasibility, liveness or performance.
