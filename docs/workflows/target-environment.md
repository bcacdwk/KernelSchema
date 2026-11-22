# Explicit row-softmax target materials

`ks.row-softmax.target.v1` is a finite adapter for K1's existing public Task. It
changes no ABI, inputs, comparator, scope or timing. Design locks the Task-derived
algorithm Contract and the caller's hardware Contract; it never obtains Design
from handwritten R0. A target is a declaration, not evidence of an actual GPU.
An actual K1 capture must match every runtime expectation before channel admission.

Create a local specification containing `hardware_specification`, `runtime` and
`sources`, then freeze the selected documents without hand-writing hashes:

```sh
python /absolute/KernelSchema/examples/workflows/row_softmax/run_demo.py target \
  --spec /materials/target-spec.json --out /materials/target.json
```

`hardware_specification` contains the existing Template keys Device, Architecture,
Execution, Feature, Stack and Wiki, with explicit nonempty string leaves. Its Wiki
locators must each resolve to a selected source in the same specification. The
normal Template checker still validates the resulting complete Schema; the adapter
does not introduce another Schema parser.

`runtime` has exactly `name`, `architecture` (two integers), `torch`, `triton` and
`cuda_build`. Fill these from an explicit chosen target/version requirement or a
trusted existing capture; the tool supplies no guessed H20/CUDA/Triton defaults.
The later actual capture additionally preserves Python/binary hashes, driver,
CUDA/device information and permission probes.

Each input source has exactly `locator`, `file` and `license`. `file` is selected
relative to the specification's directory or by an explicit absolute path;
`locator` is an immutable `repository@40-hex-commit:path`. The command stores the
original UTF-8 text, actual SHA-256 and license inline in Task.environment. Thus
Task's existing environment Ref carries the complete source body without adding
new artifact dependency rules. Every Wiki locator must match one of these original
sources (optional surrounding angle brackets are accepted). Semantic support is
still checked by the independent writer/reviewer; a real hash does not prove a
claim is supported by a document.

Unknown formats, absent versions, placeholders, missing original source text,
changed hashes or unmatched Wiki locators fail before a writing call. The original
unbound environment remains supported for CPU preparation and explicit unbound
writing; it cannot qualify a CUDA channel. No real target file is published in
this delivery because a verified chosen device/stack/document set was unavailable.
