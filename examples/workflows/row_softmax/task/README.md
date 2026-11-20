# B01 material handoff

These are actual preparation files for row_softmax_fp32_v1. Read task.json for
the file map, contract.yaml/acceptance.md for the Task, inputs.json/workload.json
for the machine-readable partitions and cases, and timing.md/environment.md for
execution requirements. The evaluator interface is ../evaluator/interface.md.

Every JSON file labeled `material_format=kernelschema.row_softmax.materials.v1`
is **unbound material input**, not a `ks.workflow.v1` object or request. The
pretty-printed input JSON is not JCS. There is no request.json, fake Ref, Q, C
or pass receipt in this phase. Fields named repository_path locate actual bytes
relative to the checkout; paths in task.json's file map are relative to the
row_softmax example. These are deliberately distinct from bundle-relative Ref
paths. Never resolve a later bundle Ref relative to a nested manifest.

K2 can consume the materials now to implement canonical Task/Schema/Rules/
Materials/Scope/Source creation, resource resolution and publication. Its
builder must preserve raw bytes, create actual source `files/kernel.py` from
baseline or starter with mode 0644, publish at stable artifact paths, compute
real digests, sort set arrays per C and use the single RFC 8785 JCS codec.
It must not treat the draft descriptors as executable manifests or fill a
missing evaluator/profile with invented identities.

The hand-authored R0 is ../schema/root.yaml plus dispatch.md; its review fixture
is r0_review.md. It uses the published exact golden commit and pinned official
Triton reference. The Template structural/relationship check can pass while
concrete hardware and version-matched documentation remain missing. R0 is not
ready for execution admission until those environment facts are fixed. Its
placeholder hardware bindings must never masquerade as qualified facts.

The baseline is ../evaluator/baseline.py, an executable FP32 softmax+copy Model
for admitted tensors when torch is installed. It is not R0 code and not the
CPU FP64 oracle. The deliberately failing ../starter/kernel.py is C0, not V0.
The golden module accepts only a saved pre-call CPU FP32 value snapshot and
returns CPU binary64 values. Full adapter execution belongs to K1.evaluator.

Role delivery is explicit in materials.json. **Design receives neither the
hand R0 answer nor private evaluator code.** Selected R0 fixed implementation
and optimize's explicit current Schema have different delivery paths. The
physical public checkout is not a role's filesystem allowlist.

From any cwd, run the dependency-free material/CPU-oracle check:

```
python3 -B /path/to/KernelSchema/examples/workflows/row_softmax/task/check_materials.py
```

From the checkout root, use the existing Template tool (requires PyYAML):

```
python3 -B schema_template/check_yaml.py examples/workflows/row_softmax/schema/root.yaml --read-only
```

The checks do not import the reference tutorial, import torch, initialize CUDA,
run a model or assert execution qualification. They leave RNG byte generation,
GPU effects/timing, model/runtime permissions and service receipts to the
owning later gates. ../references/missing.md lists the remaining work.
