"""Standalone worker copied into each exact-byte stage (no package imports)."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types


def hashed(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load(name, path, data):
    # Compile the very bytes just verified, never import kernel by a cached name.
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def main():
    root, action = Path(sys.argv[1]).resolve(), sys.argv[2]
    manifest = json.loads((root / "stage.json").read_bytes())
    payload = {}
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink():
            raise ValueError("stage symlink forbidden")
        data = path.read_bytes()
        if hashed(data) != expected:
            raise ValueError("stage hash mismatch: " + name)
        payload[name] = data
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    modules = {name: load("ks_evaluator_" + name, root / "evaluator" / (name + ".py"), payload["evaluator/" + name + ".py"])
               for name in ("input", "golden", "check", "timing", "entry")}
    spec = json.loads(payload["inputs.json"])
    if action in ("prepare", "prepare-cuda"):
        print(json.dumps({"environment": modules["entry"].environment("cuda" if action == "prepare-cuda" else "cpu_logic"),
                          "inputs": modules["input"].freeze_inputs(spec)}, sort_keys=True, separators=(",", ":")))
        return
    execution = json.loads((root / "execution.json").read_bytes())
    frozen_data = (root / "frozen_inputs.json").read_bytes()
    if hashed(frozen_data) != execution["frozen_inputs_sha256"]:
        raise ValueError("frozen input bytes changed")
    mode = execution.get("mode", "cpu_logic")
    actual_environment = modules["entry"].environment(mode)
    if actual_environment != execution["environment"]:
        raise ValueError("prepared and actual execution environments differ")
    frozen = json.loads(frozen_data)
    sandbox = load("ks_sandbox", root / "_sandbox.py", payload["_sandbox.py"])
    permission = sandbox.restrict() if mode == "cpu_logic" else {"profile": "agate-dev-bwrap-v1", "boundary": "external bwrap readonly root, writable compile scratch"}
    permission["probes"] = sandbox.prove_readonly([root / name for name in manifest["files"] if name.endswith(".py")])
    # Baseline and candidate are both loaded after the same permission boundary.
    with contextlib.redirect_stdout(sys.stderr):
        baseline = load("ks_baseline_" + hashed(payload["baseline.py"])[7:], root / "baseline.py", payload["baseline.py"])
        try:
            candidate = load("ks_candidate_" + hashed(payload["kernel.py"])[7:], root / "kernel.py", payload["kernel.py"])
            result = modules["entry"].run(candidate.Model, baseline.Model, spec, frozen, modules, mode,
                                          include_timing=execution["include_timing"])
        except Exception as exc:
            result = {"build": "fail", "correctness": None, "timing": None,
                      "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    print(json.dumps({"profile": "local-cpu-logic-seccomp-v1" if mode == "cpu_logic" else "agate-dev-bwrap-v1",
                      "qualification": "CPU logic only; no CUDA qualification" if mode == "cpu_logic" else "finite CUDA evaluator evidence; no Schema fidelity verdict",
                      "loaded": {name: hashed(data) for name, data in payload.items()},
                      "source": manifest["source"], "evaluator": manifest["evaluator"],
                      "environment": actual_environment, "frozen_inputs_sha256": hashed(frozen_data),
                      "permissions": permission, "result": result}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
