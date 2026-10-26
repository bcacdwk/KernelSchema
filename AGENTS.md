# KernelSchema authoring

- The language is defined jointly by the kernel package in schema_template and schema_docs/1–6.
  Keep these consistent. Examples, including Markdown examples and old
  schema_wiki files, are illustrative evidence rather than overriding rules.
- Author plans in CUE, in a task directory with package plan and
  plans: kernel.#Task. Start with python3 schema_template/tools/schema.py init <directory>.
  See schema_template/README.md for the full workflow and boundaries.
- Generate YAML only with python3 schema_template/tools/schema.py build <directory>.
  Never hand-edit generated YAML or .schema-generated.json. A requested plan
  change belongs in its CUE source; increment that node's meta_data.version.
  Coder feedback is also authored in CUE and retains the original choice.
- Contract, Design and Realization ownership follows README and the six docs.
  The generator does not grant permission to change a Contract or redesign a
  pipeline. Changing templates is language development, distinct from filling a plan.
- Update corresponding normative Markdown and meaningful positive/negative
  tests when changing the language. Do not weaken a rule just to admit a legacy
  example. Do not migrate old YAML unless that migration is part of the task.
- Run make check before delivering changes. make setup installs the pinned
  CUE binary locally; Python's standard library is sufficient for the tools.
  Successful checks establish structure and listed relations, not numerical
  correctness, hardware feasibility, liveness or performance.
