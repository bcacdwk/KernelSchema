# Source delivery and remaining release requirements

The supported engineering tier is a Linux source checkout/editable installation.
CPU preparation needs Python, PyYAML and jsonschema. It does not need AKA,
private planning documents, model captures or GPU packages. Python 3.13.12 has
been tested; Python 3.11 and a resource-complete wheel remain unverified.

The [standalone commands](standalone.md) prepare five row-softmax requests and
run the finite driver. The [FA4 commands](../../examples/workflows/fa4_decode/README.md)
prepare the pinned F1 and call the reference backend. Initial FA4 preparation
requires the two selected Git commits, while imported Task/Schema/Source closure
can be staged without that Git checkout. Anonymous retrieval of those commits
has not been established by the current engineering evidence.

Existing license facts are retained at their actual sources:

| Selected material | Retained evidence | Release requirement |
| --- | --- | --- |
| Triton v3.5.1 softmax tutorial | [Fixed inventory and upstream MIT notice](../../examples/workflows/row_softmax/references/README.md) | Preserve its original notice and source/version attribution. |
| KS workflow, Template, Schema and authored evaluator material | [Per-material origins](../../examples/workflows/row_softmax/task/materials.json) | The project owner must resolve the distribution license; no root project license or complete project notice bundle is currently supplied. |
| FA4 v55 Source and original reference/input/harness | [K7 per-file audit](../../examples/workflows/fa4_decode/licenses-and-gaps.md) | Resolve the NVIDIA BSD headers, adjacent Apache text, missing referenced notices and per-file grants. The adjacent license does not license the whole case. |
| Original CUDA/CuTe runtime stack | [Exact target requirements](../../examples/workflows/fa4_decode/task/environment.json) | Record the actual installed dependency versions, provenance and applicable distribution obligations before redistributing a runtime image. |

This page records facts and missing decisions; it grants no new license and
does not alter original third-party notices. Historical K7 missing-engineering
rows remain historical; current implementation support is described by the
FA4 entry above.

Private runtime archives are not required package resources. The package
discovery configuration excludes `kernelschema.runtime.qualification*`; only
the small public profiles and explanatory qualification summary belong in the
source delivery. Do not recursively add private captures to a source/package
or replace original evidence with a redacted explanation.

Actual business-model, CUDA, service permission, full-output/cleanup, fidelity
and performance gates remain separate from CPU tests. The historical K3 CLI
profile record does not qualify later role projection changes or the business
workflows. A real run needs a locally inspected CLI profile, an existing owner,
explicit target materials and the selected channel's original qualification
records. Missing state or qualification remains a refusal/blocked result; it
does not authorize a replacement account, a new job or a passing report.
