"""Pinned-original F1 evaluator. CUDA loads only inside explicit execute().

No historical main, package installation, fallback device or performance claim.
Standalone facts require an independently qualified transport before promotion.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

READONLY = ("query", "kv_cache", "block_tables", "seq_lens")
TENSORS = ("query", "kv_cache", "workspace_buffer", "block_tables", "seq_lens", "out")


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def comparator(path, torch):
    """Reuse actual pinned helper AST nodes without executing historical main."""
    tree = ast.parse(Path(path).read_bytes(), filename=str(path))
    names = {"checks", "unchanged", "poison"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != names or len(body) != len(names):
        raise ValueError("original comparator helper inventory differs")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def view(tensor):
    return {"object": id(tensor), "storage": tensor.untyped_storage().data_ptr(),
            "storage_bytes": tensor.untyped_storage().nbytes(), "data": tensor.data_ptr(),
            "shape": tuple(tensor.shape), "stride": tuple(tensor.stride()), "offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def tensor_digest(tensor, torch):
    # Hash the full 413 MB scratch too, without an additional full-size host copy.
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 4 * 1024 * 1024):
        digest.update(raw[start:start + 4 * 1024 * 1024].cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def validate_inputs(inputs, spec, torch):
    """Reject a changed helper/ABI; this does not regenerate or resize tensors."""
    if list(inputs) != [a["name"] for a in spec["abi"]]:
        raise ValueError("13-argument input ABI/order differs")
    dtypes = {"fp8_e4m3fn": torch.float8_e4m3fn, "uint8": torch.uint8, "int32": torch.int32, "bf16": torch.bfloat16}
    for item in spec["abi"]:
        value = inputs[item["name"]]
        if item["dtype"] in dtypes:
            if list(value.shape) != item["shape"] or value.dtype != dtypes[item["dtype"]] or value.device.type != "cuda":
                raise ValueError("original input metadata differs: " + item["name"])
        else:
            expected = item["f1_value_or_generation"]
            if type(value) is not type(expected) or value != expected:
                raise ValueError("original scalar ABI differs: " + item["name"])
    if inputs["block_tables"].tolist() != [[0] + [-1] * 16433] or inputs["seq_lens"].tolist() != [64]:
        raise ValueError("F1 page map/sequence differs")
    if bool(inputs["workspace_buffer"].count_nonzero()):
        raise ValueError("original workspace must start zero")


def snapshot(inputs, torch):
    return {"out": inputs["out"], "out_view": view(inputs["out"]),
            "views": {key: view(inputs[key]) for key in READONLY},
            "readonly": {key: inputs[key].clone() for key in READONLY}}


def complete_check(result, inputs, before, expected, helpers, torch):
    """Both outputs, original destination and every readonly byte, on every call."""
    tolerance = {"atol": 0.06, "rtol": 0.04}
    original = before["out"]
    out_intact = inputs["out"] is original and view(original) == before["out_view"]
    alias = isinstance(result, torch.Tensor) and {k: v for k, v in view(result).items() if k != "object"} == {
        k: v for k, v in before["out_view"].items() if k != "object"}
    outputs = {"return": helpers["checks"](result, expected, tolerance),
               "original_out": helpers["checks"](original, expected, tolerance)}
    unchanged = helpers["unchanged"](inputs, before["readonly"])
    views = {key: view(inputs[key]) == value for key, value in before["views"].items()}
    return {"passed": out_intact and alias and all(x["passed"] for x in outputs.values()) and all(unchanged.values()) and all(views.values()),
            "outputs": outputs, "out_view_intact": out_intact, "return_aliases_original_view": alias,
            "readonly_bytes": unchanged, "readonly_views": views}


def native_state(operator, inputs, torch):
    launcher = operator._NATIVE_LAUNCHER
    if launcher is None:
        raise RuntimeError("native launcher was not established; steady q4_direct when is unproved")
    if launcher.thread_id != threading.get_ident():
        raise RuntimeError("native launcher thread differs")
    if len(launcher.source_refs) != 6 or any(inputs[key] is not original() for key, original in zip(TENSORS, launcher.source_refs)):
        raise RuntimeError("native launcher input objects differ")
    if operator._launch_signature(*[inputs[key] for key in inputs]) != launcher.signature:
        raise RuntimeError("native launcher signature differs")
    if inputs["query"].device.index != torch.cuda.current_device() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("native device/capture predicate fails")
    return launcher


def environment(required, torch):
    if not torch.cuda.is_available():
        raise RuntimeError("original CUDA stack unavailable; no CPU fallback")
    versions = {name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "cuda-python")}
    observed = {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                **versions, "compute_capability": list(torch.cuda.get_device_capability())}
    if observed != required:
        raise RuntimeError("original stack mismatch: " + json.dumps(observed, sort_keys=True))
    import tvm_ffi
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=10)
    return {**observed, "python": sys.version, "executable": sys.executable,
            "executable_sha256": sha(Path(sys.executable).read_bytes()), "os": platform.platform(),
            "device_name": torch.cuda.get_device_name(), "device_index": torch.cuda.current_device(),
            "nvidia_smi": smi.stdout, "tvm_ffi_version": str(getattr(tvm_ffi, "__version__", "unknown")),
            "tvm_ffi_module_sha256": sha(Path(tvm_ffi.__file__).read_bytes()), "qualification": "not_established"}


def verify_stage(root, expected_digest):
    data = (root / "stage.json").read_bytes()
    if sha(data) != expected_digest:
        raise ValueError("original stage digest differs")
    manifest = json.loads(data)
    expected_names = set(manifest["files"]) | {"stage.json"}
    actual_names = set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            if (Path(parent) / name).is_symlink():
                raise ValueError("stage must not contain symlinks")
        actual_names.update((Path(parent) / name).relative_to(root).as_posix() for name in files)
    if actual_names != expected_names:
        raise ValueError("stage has missing/extra files")
    for name, facts in manifest["files"].items():
        path = root / name
        if (path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != facts["sha256"]
                or path.stat().st_size != facts["size"] or (path.stat().st_mode & 0o7777) != int(facts["mode"], 8)):
            raise ValueError("staged bytes/mode differ: " + name)
    return manifest


def execute(root, manifest, *, observation=None):
    observation = {} if observation is None else observation
    import torch
    spec = json.loads((root / "f1-parameters.json").read_bytes())
    recipe = json.loads((root / "timing.json").read_bytes())
    target = json.loads((root / "environment.json").read_bytes())
    observed = environment(target["required"], torch)
    observation["environment"] = observed
    observation["cold"] = {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # No prior candidate import/native state is allowed in this fresh process.
    if any(n == "qoder_greenfield_decode_attention" or n.startswith("qoder_greenfield_decode_attention.") for n in sys.modules):
        raise RuntimeError("original source package is already imported")
    started = time.perf_counter_ns()
    sys.path.insert(0, str(root / "source"))
    candidate = load("fa4_original_solution", root / "source/solution.py")
    operator = sys.modules["qoder_greenfield_decode_attention.operator"]
    if operator._NATIVE_LAUNCHER is not None:
        raise RuntimeError("cold native cache is not empty")
    import_ns = time.perf_counter_ns() - started
    observation["cold"]["source_import_ns"] = import_ns
    generator = load("fa4_original_input", root / "evaluator/input.py")
    reference = load("fa4_original_reference", root / "evaluator/reference.py")
    helpers = comparator(root / "evaluator/harness.py", torch)
    started = time.perf_counter_ns()
    torch.manual_seed(spec["generator"]["manual_seed"])
    inputs = generator._make_inputs(**spec["generator"]["kwargs"])
    torch.cuda.synchronize()
    input_ns = time.perf_counter_ns() - started
    observation["cold"]["input_generation_ns"] = input_ns
    validate_inputs(inputs, spec, torch)
    input_digests = {key: tensor_digest(inputs[key], torch) for key in (*READONLY, "workspace_buffer")}
    observation["input_digests"] = input_digests
    observation["input_views"] = {key: view(inputs[key]) for key in TENSORS}
    observation["out_initialization"] = "original empty allocation; poisoned in place before every evaluated call"
    # The reference cannot overwrite candidate evidence or mutate its scratch.
    ref_inputs = {**inputs, **{key: inputs[key].clone() for key in READONLY}, "out": torch.empty_like(inputs["out"]),
                  "workspace_buffer": torch.zeros_like(inputs["workspace_buffer"])}
    with torch.no_grad():
        expected = reference.Model()(**ref_inputs).clone()
        torch.cuda.synchronize()
    del ref_inputs
    before = snapshot(inputs, torch)
    started = time.perf_counter_ns()
    model = candidate.Model()
    constructor_ns = time.perf_counter_ns() - started
    observation["cold"]["constructor_ns"] = constructor_ns
    checks, samples = [], []
    observation.update(checks=checks, samples_ns=samples)
    with torch.no_grad():
        helpers["poison"](inputs, ["out"])
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        result = model(**inputs)
        torch.cuda.synchronize()
        cold_ns = time.perf_counter_ns() - started
        observation["cold"]["first_forward_sync_ns"] = cold_ns
        cold_check = complete_check(result, inputs, before, expected, helpers, torch)
        checks.append({"phase": "cold", **cold_check})
        if not cold_check["passed"]:
            return {"environment": observed, "checks": checks, "samples_ns": [], "passed": False}
        launcher = native_state(operator, inputs, torch)
        counts = [("warmup", recipe["steady"]["warmup_calls"]), ("steady", recipe["steady"]["samples"])]
        for phase, count in counts:
            for index in range(count):
                helpers["poison"](inputs, ["out"])
                torch.cuda.synchronize()
                if native_state(operator, inputs, torch) is not launcher:
                    raise RuntimeError("native launcher changed before call")
                started = time.perf_counter_ns()
                result = model(**inputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter_ns() - started
                if native_state(operator, inputs, torch) is not launcher:
                    raise RuntimeError("native launcher failed or changed during call")
                check = complete_check(result, inputs, before, expected, helpers, torch)
                checks.append({"phase": phase, "index": index, **check})
                if not check["passed"]:
                    return {"environment": observed, "checks": checks, "samples_ns": samples, "passed": False}
                if phase == "steady":
                    samples.append(elapsed)
    return {"environment": observed, "input_digests": input_digests,
            "cold": {"source_import_ns": import_ns, "input_generation_ns": input_ns,
                     "constructor_ns": constructor_ns, "first_forward_sync_ns": cold_ns,
                     "includes": "normal fmha compilation/launch, native FFI probe, synchronization and any restore"},
            "checks": checks, "native_when": "observed before/after each warmup and steady call",
            "samples_ns": samples, "median_ns": statistics.median(samples), "passed": True,
            "fidelity": "unknown", "qualification": "standalone raw observation only", "performance_goal": "unknown"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage.resolve()
    if args.output.resolve().is_relative_to(root):
        raise ValueError("output must be outside immutable stage")
    manifest = verify_stage(root, args.stage_sha256)
    report = {"stage_sha256": args.stage_sha256, "subject": {key: manifest[key]["sha256"]
              for key in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")}}
    try:
        report.update(execute(root, manifest, observation=report))
    except Exception as error:
        report.update(passed=False, exception={"type": type(error).__name__, "message": str(error)},
                      qualification="not_established", fidelity="unknown")
    # Check code and input stage again after execution. No mutable import cache.
    verify_stage(root, args.stage_sha256)
    def wire(value):
        # Historical helper metrics can be NaN/Inf for intentional bad output.
        # Retain them as strings; do not lose the failing report to JSON encoding.
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, dict):
            return {k: wire(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [wire(v) for v in value]
        return value
    with args.output.open("x") as stream:
        json.dump(wire(report), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
