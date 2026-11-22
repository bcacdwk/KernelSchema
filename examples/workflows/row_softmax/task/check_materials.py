"""Check material consistency and the CPU oracle; no torch/CUDA/model imports.

This is a preparation check, not the Task evaluator or an alternate YAML parser.
Validate the R0 separately with the existing schema_template/check_yaml.py.
"""

import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys


EXAMPLE = Path(__file__).resolve().parents[1]
REPO = EXAMPLE.parents[2]


def load(relative):
    with (EXAMPLE / relative).open(encoding="utf-8") as stream:
        data = json.load(stream)
    assert data["material_format"] == "kernelschema.row_softmax.materials.v1"
    assert data["publication_status"] == "unbound_materials"
    assert "protocol" not in data, "draft material cannot masquerade as a workflow object"
    return data


def check_digest(entry, key="repository_path"):
    path = REPO / entry[key]
    raw = path.read_bytes()
    assert path.is_file() and not path.is_symlink(), path
    assert len(raw) == entry["size"], path
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == entry["sha256"], path


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def deterministic_rows(case):
    m, n = case["shape"]
    family = case["generator"]
    rows = []
    for i in range(m):
        row = []
        for j in range(n):
            if family == "constant":
                value = float(case["value"])
            elif family == "alternating":
                value = 20.0 if (i + j) % 2 == 0 else -20.0
            elif family == "single_peak":
                value = 20.0 if j == (n - 1 - 17 * i) % n else -20.0
            elif family == "near_equal":
                value = (-1) ** i * 19 + ((j % 5) - 2) * 2 ** -12
            elif family == "permuted_ramp":
                fraction = fp32(((j * (2 * i + 1) + i) % n) / (n - 1))
                value = fp32(-20 + fp32(40 * fraction))
            else:
                raise AssertionError("random values must come from the locked torch Generator")
            row.append(fp32(value))
        rows.append(row)
    return rows


def main():
    inputs = load("task/inputs.json")
    cases = inputs["cases"]
    assert len(cases) == inputs["expected_basic_case_count"] == 36
    assert len({case["case_id"] for case in cases}) == 36
    assert sum(len(case["out_initializers"]) for case in cases) == inputs["expected_basic_call_count"] == 72
    shapes = {(1, 128), (1, 1024), (32, 128), (32, 1024)}
    assert Counter(tuple(case["shape"]) for case in cases) == {shape: 9 for shape in shapes}
    partitions = set()
    for case in cases:
        m, n = case["shape"]
        partitions.add(case["coverage_id"])
        assert case["coverage_id"] == f"shape_m{m}_n{n}"
        assert case["out_initializers"] == ["nan", "negative_123"]
        assert case["layout"]["stride"] == [n, 1]
        assert case["layout"]["suffix_guard_elements"] == 8
        offsets = (case["layout"]["x_storage_offset"], case["layout"]["out_storage_offset"])
        assert offsets == ((3, 5) if case.get("seed") == 23 else (8, 8))
    for shape in shapes:
        group = [case for case in cases if tuple(case["shape"]) == shape]
        assert sorted(case["seed"] for case in group if case["generator"] == "uniform") == [17, 23]
        assert sorted(case["value"] for case in group if case["generator"] == "constant") == ["-20.0", "0.0", "20.0"]
        assert Counter(case["generator"] for case in group) == {
            "uniform": 2, "constant": 3, "alternating": 1,
            "single_peak": 1, "near_equal": 1, "permuted_ramp": 1,
        }
    by_id = {case["case_id"]: case for case in cases}
    traces = inputs["state_reuse_traces"]
    assert len(traces) == 1
    trace = traces[0]
    assert len(trace["basic_case_ids"]) == len(trace["out_initializers"]) == 2
    first, second = [by_id[id] for id in trace["basic_case_ids"]]
    assert first["shape"] == second["shape"] and first["case_id"] != second["case_id"]
    assert trace["same_model"] and trace["same_x_storage"] and trace["same_out_storage"]

    scope = load("task/scope.json")
    assert scope["coverage_ids"] == sorted(partitions)
    assert scope["fixed_r0_bindings"] == [{"coverage_ids": sorted(partitions), "node": "root", "variant": None}]
    assert scope["design_input_bindings"] == []
    workload = load("task/workload.json")
    assert {part["coverage_id"] for part in workload["coverage_partitions"]} == partitions
    assert all(part["weight"] == "1/4" for part in workload["coverage_partitions"])
    timing = workload["timing"]
    assert (timing["input_seed"], timing["warmup_calls_per_model_per_shape"],
            timing["calls_per_batch"], timing["pairs_per_shape"]) == (17, 20, 100, 10)
    assert timing["pair_order"] == ["AB", "BA"] * 5

    materials = load("task/materials.json")
    entries = materials["entries"]
    ids = [entry["id"] for entry in entries]
    assert ids == sorted(set(ids)) and ids.count("workflow_objective") == 1
    for entry in entries:
        check_digest(entry)
        if "planner" in entry["roles"]:
            assert "/schema/" not in entry["repository_path"]
            assert "/evaluator/" not in entry["repository_path"]
            assert "/starter/" not in entry["repository_path"]
    objective = next(entry for entry in entries if entry["id"] == "workflow_objective")
    assert {"planner", "reviewer"} <= set(objective["roles"])
    sources = load("references/sources.json")
    for entry in sources["external"]:
        check_digest(entry, "local_repository_path")
        assert entry["commit"] == "0add68262ab0a2e33b84524346cb27cbb2787356"
    check_digest(sources["golden"])
    for entry in sources["rule_resources"]["files"]:
        check_digest(entry)
    evaluator = load("evaluator/manifest.json")
    assert evaluator["status"] == "interface_only"
    for entry in evaluator["available_components"]:
        check_digest(entry)
    task = load("task/task.json")
    for relative in task["files"].values():
        assert (EXAMPLE / relative).is_file(), relative
    for source in (task["baseline_source_input"], task["starter_source_input"]):
        path = EXAMPLE / source["file"]
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and path.stat().st_mode & 0o777 == 0o644
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
        forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        assert [arg.arg for arg in forward.args.args] == ["self", "x", "out"]
        assert not forward.args.posonlyargs
        compile(tree, str(path), "exec")

    spec = importlib.util.spec_from_file_location("row_softmax_golden", EXAMPLE / "evaluator/golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert sys.float_info.mant_dig == 53
    checked = 0
    for case in cases:
        if case["generator"] == "uniform":
            continue
        rows = deterministic_rows(case)
        before = [row[:] for row in rows]
        assert all(math.isfinite(value) and -20 <= value <= 20 for row in rows for value in row)
        result = golden.softmax_fp64(rows)
        assert rows == before
        n = case["shape"][1]
        for i, row in enumerate(result):
            assert all(math.isfinite(value) and value > 0 for value in row)
            assert abs(math.fsum(row) - 1.0) <= 3e-16
            if case["generator"] == "constant":
                assert row == [1.0 / n] * n
            if case["generator"] == "single_peak":
                peak = (n - 1 - 17 * i) % n
                assert row.index(max(row)) == peak
                assert math.isclose(row[peak], 1 / (1 + (n - 1) * math.exp(-40)), rel_tol=3e-16)
        checked += 1
    print(f"PASS: 36 cases / 72 basic calls; 1 separate two-call state trace; {len(entries)} role materials")
    print(f"PASS: byte inventories, input/scope/timing consistency, source syntax/ABI; {checked} deterministic CPU golden cases")
    print("NOT RUN: locked torch random bytes, baseline CUDA execution, full evaluator, model, GPU, timing or service qualification")


if __name__ == "__main__":
    main()
