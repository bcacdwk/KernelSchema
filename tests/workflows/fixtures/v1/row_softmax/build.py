"""Generate portable, deterministic synthetic contracts from pinned K1 bytes.

This does not run or qualify any backend, evaluator, model, GPU or kernel.
"""
from pathlib import Path
import argparse
import json

from kernelschema.artifacts import Publication, durable_write
from kernelschema.contracts import PROTOCOL, ProtocolError, jcs, sha256
from kernelschema.extract_bridge import consume_raw
from kernelschema.resources import checkout_root


def build_bundle(root, operation="implement"):
    root = Path(root)
    p = Publication(root, "0123456789abcdef0123456789abcdef")
    example = checkout_root() / "examples/workflows/row_softmax"
    pinned = json.loads(Path(__file__).with_name("k1_inputs.json").read_bytes())
    for relative, expected in pinned["files"].items():
        if sha256((example / relative).read_bytes()) != expected:
            raise ProtocolError(f"K1 fixture input changed: {relative}; update only after an explicit fixed-material handoff")
    def original(relative):
        return p.put_bytes(relative, (example / relative).read_bytes())
    originals = {name: original(name) for name in (
        "task/contract.yaml", "task/acceptance.md", "task/workload.json", "task/inputs.json", "task/environment.md", "task/timing.md",
        "task/workflow_objective.md", "evaluator/interface.md", "evaluator/golden.py")}
    baseline = p.put_source("baseline", (example / "evaluator/baseline.py").read_bytes())
    starter = p.put_source("starter", (example / "starter/kernel.py").read_bytes())
    evaluator = p.put_json("evaluator.json", {"protocol": PROTOCOL, "kind": "evaluator", "version": "synthetic-interface-only",
        "entrypoint": originals["evaluator/golden.py"], "files": sorted([originals["evaluator/golden.py"], originals["evaluator/interface.md"]], key=lambda x: x["path"]),
        "inputs": originals["task/inputs.json"], "timing": originals["task/timing.md"], "environment": originals["task/environment.md"], "adapters": [],
        "extensions": {"information": {"synthetic": True, "execution": "not implemented or qualified"}}})
    task = p.put_json("task.json", {"protocol": PROTOCOL, "kind": "task", "task_id": "row_softmax_fp32_v1",
        **{key: originals["task/" + filename] for key, filename in (("contract", "contract.yaml"), ("acceptance", "acceptance.md"), ("workload", "workload.json"), ("environment", "environment.md"))},
        "evaluator": evaluator, "baseline_source": baseline, "starter_source": starter,
        "extensions": {"information": {"synthetic": True, "K1_material_commit": "35443d2f6cefe424e9adcd6d8d66dc17aa267d99", "qualification": "none"}}})
    method = p.put_bytes("rules/method.md", b"Synthetic method for contract/CPU boundaries only.\n")
    rules = p.put_json("rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": [method], "tools": []})
    materials = p.put_json("materials.json", {"protocol": PROTOCOL, "kind": "materials", "entries": [{"id": "workflow_objective", "file": originals["task/workflow_objective.md"], "roles": ["planner", "reviewer"], "origin": "fixed K1 material", "license": "project license unresolved"}]})
    scope_input = json.loads((example / "task/scope.json").read_bytes())
    scope = p.put_json("scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": scope_input["coverage_ids"], "bindings": [] if operation == "design" else scope_input["fixed_r0_bindings"]})
    caps = {"protocols": [PROTOCOL], "operations": ["design", "optimize", "implement"], "modes": ["code", "verify_only"], "source_formats": ["single_file"], "empty_start": False,
            "recovery": ["local-intent"], "token_enforcement": "unsupported", "extensions": []}
    profile = p.put_json("profile.json", {"protocol": PROTOCOL, "kind": "backend_profile", "name": "synthetic", "version": "CPU-only-fixture", "runtime": "none", "model": "none",
        "parameters": {}, "environment": originals["task/environment.md"], "capabilities": caps, "native_limits": {}, "extensions": {"information": {"synthetic": True}}})
    request = {"protocol": PROTOCOL, "operation": operation, "request_id": "synthetic-" + operation,
        "task": task, "rules": rules, "materials": materials, "scope": scope, "backend": {"name": "synthetic", "profile": profile},
        "budget": {"parent": None, "limits": {"calls": {"planner": 1, "coder": 1, "reviewer": 1, "extract": 0}, "evaluator_jobs": 1, "active_ms": 10000, "tokens": None}},
        "extensions": {"information": {"synthetic": True, "qualification": "none"}}}
    schema = None
    if operation != "design":
        schema = consume_raw(p, example / "schema", task=task)["schema"]
        request["schema"] = schema
    if operation == "implement": request.update(seed=None, mode="code")
    if operation == "optimize": request.update(feedback=[], evidence=[])
    request_ref = p.put_json("request.json", request)
    p.commit([request_ref])
    durable_write(root / "request.json", jcs(request))
    return {"root": root, "request": request, "request_ref": request_ref, "task": task, "schema": schema, "starter": starter, "baseline": baseline, "evaluator": evaluator, "profile": profile, "method": method, "originals": originals, "caps": caps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--operation", choices=("design", "optimize", "implement"), default="implement")
    args = parser.parse_args()
    result = build_bundle(args.out, args.operation)
    print(result["root"] / "request.json")
    print("Synthetic CPU contract only; execution/evaluator/model/GPU qualification: none")
