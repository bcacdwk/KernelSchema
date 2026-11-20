"""Actual F1 materials/Task/stage plus explicitly synthetic CPU helper checks."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kernelschema.artifacts import Bundle, Publication, import_bundle
from kernelschema.contracts import PROTOCOL, ProtocolError, UnsupportedError, jcs, sha256
from kernelschema.evaluation.fa4 import (ABI, F1, REST, FA4Task, freeze_observation, freeze_request,
    ORIGINAL, SOURCE_ROOT, stage_original)
from kernelschema.materials import require_capabilities, validate_request
from kernelschema.resources import resource
from kernelschema.template_api import resolve_bundle
from examples.workflows.fa4_decode.prepare import prepare


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("fa4-cpu") / "bundle"
    result = prepare(root, checkout=resource("AGENTS.md").parent)
    return root, result


def evaluator_module():
    path = resource("examples/workflows/fa4_decode/evaluator/entry.py")
    spec = importlib.util.spec_from_file_location("fa4_cpu_test_entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preparation_is_actual_bound_four_node_task(prepared):
    root, ready = prepared
    b, case = Bundle(root), ready["case"]
    request = b.document(ready["request"])
    objects = validate_request(b, request)
    assert objects["source_formats"] == ["python_tree"]
    assert request["seed"] == case["source"] and request["mode"] == "verify_only"
    assert request["budget"]["limits"]["evaluator_jobs"] == 0
    assert ready["gpu_jobs"] == ready["model_calls"] == 0
    assert objects["task"]["starter_source"] is None
    assert len(b.source_files(case["source"])) == 12
    assert len([p for p in b.source_files(case["source"]) if p.endswith(".py")]) == 6
    schema = b.document(case["schema"])
    assert [x["node"] for x in schema["raw"]] == ["q1_global", "q4_cluster", "q4_direct", "root"]
    resolved = resolve_bundle(b, schema)
    direct = resolved.effective_document("q4_direct")
    assert direct["CONTRACT"] == resolved.effective_document("root")["CONTRACT"]
    assert direct["DESIGN"] != resolved.effective_document("root")["DESIGN"]
    coverage = FA4Task(b, case["task"]).validate_scope(case["schema"], case["scope"])
    assert coverage["status"] == "pass" and coverage["task_uncovered"] == [REST]
    assert coverage["runtime_when"] == "not_run" and coverage["fidelity"] == "unknown"
    assert len(case["originals"]) == 27
    for entry in json.loads(resource("examples/workflows/fa4_decode/materials.json").read_bytes())["entries"]:
        assert sha256(b.read(case["originals"][entry["path"]])) == entry["sha256"]


def test_root_variant_and_full_task_do_not_pass(prepared):
    root, ready = prepared
    b, case = Bundle(root), ready["case"]
    adapter = FA4Task(b, case["task"])
    for node, variant, expected in [("root", None, "fail"), ("q4_direct", "made-up", "unknown"), ("q1_global", None, "unknown")]:
        p = Publication(root)
        scope = p.put_json("scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": [F1],
            "bindings": [{"coverage_ids": [F1], "node": node, "variant": variant}]})
        p.commit([scope])
        assert adapter.validate_scope(case["schema"], scope)["status"] == expected
    p = Publication(root)
    full = p.put_json("scope.json", {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": sorted([F1, REST]), "bindings": []})
    p.commit([full])
    with pytest.raises(UnsupportedError, match="fixed F1"):
        adapter.require_evaluation_scope(full)


def test_input_manifest_and_original_generator_execution_with_recorder(prepared):
    root, ready = prepared
    b = Bundle(root)
    spec = json.loads(b.read(ready["case"]["case_materials"]["f1-parameters.json"]))
    assert [a["name"] for a in spec["abi"]] == list(ABI)
    assert spec["generator"]["manual_seed"] == 20261120
    assert spec["generator"]["kwargs"]["workspace_size_bytes"] == 413138944
    assert spec["generator"]["kwargs"]["block_table_columns"] == 16434
    # Execute only the original pure helper bodies with a shape recorder. This
    # allocates no tensors, imports no original CUDA module and proves no GPU result.
    calls = []
    class Tensor:
        def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype
        def __mul__(self, value): calls.append(("scale", value)); return self
        def to(self, dtype): calls.append(("cast", dtype)); self.dtype = dtype; return self
        def __setitem__(self, key, value): calls.append(("table_write", key[0], value.shape))
    class Recorder:
        bfloat16, float8_e4m3fn, int32, uint8 = "bf16", "fp8", "i32", "u8"
        def randn(self, *shape, dtype, device): calls.append(("randn", shape, dtype, device)); return Tensor(shape, dtype)
        def full(self, shape, value, dtype, device): calls.append(("full", shape, value)); return Tensor(shape, dtype)
        def arange(self, start, end, dtype, device): calls.append(("arange", start, end)); return Tensor((end-start,), dtype)
        def zeros(self, size, dtype, device): calls.append(("zeros", size)); return Tensor((size,), dtype)
        def tensor(self, values, dtype, device): calls.append(("tensor", values)); return Tensor((len(values),), dtype)
        def empty(self, *shape, dtype, device): calls.append(("empty", shape)); return Tensor(shape, dtype)
    raw = b.read(ready["case"]["originals"][ORIGINAL + "input.py"])
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = {"torch": Recorder()}
    exec(compile(tree, "pinned-input-with-synthetic-recorder", "exec"), ns)
    inputs = ns["_make_inputs"](**spec["generator"]["kwargs"])
    assert list(inputs) == list(ABI)
    assert calls[:6] == [("randn", (4,16,256), "bf16", "cuda"), ("scale",1.22), ("cast","fp8"),
                         ("randn", (1,2,1,64,256), "bf16", "cuda"), ("scale",0.96), ("cast","fp8")]
    assert ("zeros",413138944) in calls and ("arange",0,1) in calls
    assert inputs["sinks"] is inputs["o_sf_scale"] is None
    assert inputs["q_len_per_req"] == 4 and inputs["max_seq_len"] == 64
    operator = ast.parse(b.read(ready["case"]["originals"][SOURCE_ROOT + "qoder_greenfield_decode_attention/operator.py"]))
    run = next(n for n in operator.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert [a.arg for a in run.args.args] == list(ABI)
    functions = {"_ceil_div", "_floor_po2", "_pick_tiles", "_select_policy"}
    operator.body = [n for n in operator.body if (isinstance(n, ast.FunctionDef) and n.name in functions)
                     or (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    policy = {}
    exec(compile(operator, "pinned-pure-host-policy-only", "exec"), policy)
    assert policy["_select_policy"](1,4,64,1) == (1,"none")
    assert policy["_pick_tiles"](4) == (16,2)


def test_exact_stage_and_import_need_no_original_repository(prepared, tmp_path):
    root, ready = prepared
    entry = evaluator_module()
    entry.verify_stage(Path(ready["stage"]["root"]), ready["stage"]["sha256"])
    case = ready["case"]
    import_bundle(Bundle(root), tmp_path / "imported", [case[k] for k in ("task", "schema", "scope")])
    stage = stage_original(Bundle(tmp_path / "imported"), case, tmp_path / "staged")
    assert stage["manifest"]["source"] == case["source"]
    assert len(stage["manifest"]["files"]) == 19
    (tmp_path / "staged/source/solution.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes/mode"):
        entry.verify_stage(tmp_path / "staged", stage["sha256"])


def test_original_only_and_backend_source_capability_early_rejection(prepared):
    root, ready = prepared
    b = Bundle(root)
    request = b.document(ready["request"])
    objects = validate_request(b, request)
    actual = {"protocols": [PROTOCOL], "operations": ["implement"], "modes": ["code", "verify_only"],
        "source_formats": ["single_file"], "empty_start": True, "recovery": [], "token_enforcement": "unsupported", "extensions": []}
    objects["profile"]["capabilities"] = deepcopy(actual)
    with pytest.raises(UnsupportedError, match="source format"):
        require_capabilities(request, objects, actual)
    p = Publication(root)
    changed = p.put_source_files("changed", {"solution.py": b.source_files(ready["case"]["source"])["solution.py"]})
    p.commit([changed])
    with pytest.raises(UnsupportedError, match="pinned original"):
        stage_original(b, {**ready["case"], "source": changed}, root / "should-not-exist")
    assert not (root / "should-not-exist").exists()


def test_source_intents_are_distinct_and_private_helpers_not_role_materials(prepared):
    root, ready = prepared
    b = Bundle(root)
    args = {"case": ready["case"], "profile": ready["profile"], "rules": ready["rules"],
            "limits": b.document(ready["request"])["budget"]["limits"]}
    rebuilt = freeze_request(Publication(root), **args, intent="schema_rebuild")
    seeded = freeze_request(Publication(root), **args, intent="seed_continuation", source=ready["case"]["source"])
    assert b.document(rebuilt)["seed"] is None
    assert b.document(seeded)["seed"] == ready["case"]["source"]
    assert b.document(rebuilt)["mode"] == b.document(seeded)["mode"] == "code"
    for request_ref, verb in ((rebuilt, b"Rebuild"), (seeded, b"Continue")):
        material_doc = b.document(b.document(request_ref)["materials"])
        objective = next(e["file"] for e in material_doc["entries"] if e["id"] == "workflow_objective")
        assert b.read(objective).startswith(verb)
    with pytest.raises(ProtocolError, match="requires"):
        freeze_request(Publication(root), **args, intent="seed_continuation")
    with pytest.raises(ProtocolError, match="without"):
        freeze_request(Publication(root), **args, intent="schema_rebuild", source=ready["case"]["source"])
    entries = b.document(ready["case"]["materials"])["entries"]
    assert {e["id"] for e in entries if e["roles"]} == {"candidate_interface", "license_gaps", "workflow_objective"}
    assert {e["id"] for e in entries if not e["roles"]} == {"k7_preflight", "pinned_inventory"}


def test_synthetic_success_never_promotes_qualification(prepared):
    root, ready = prepared
    stage = ready["stage"]
    synthetic = {"stage_sha256": stage["sha256"], "subject": {k: stage["manifest"][k]["sha256"]
        for k in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")},
        "environment": {"synthetic": True}, "passed": True, "samples_ns": [1,2,3]}
    p = Publication(root)
    report = freeze_observation(p, stage, raw=jcs(synthetic), operation_id="synthetic-only")
    p.commit([report], subjects=[ready["request"]])
    doc = Bundle(root).document(report)
    assert {doc[k]["status"] for k in ("build", "correctness", "performance")} == {"unknown"}
    assert doc["extensions"]["information"]["fidelity"] == "unknown"
    synthetic["subject"]["source"] = "sha256:" + "0" * 64
    with pytest.raises(ProtocolError, match="subject"):
        freeze_observation(Publication(root), stage, raw=jcs(synthetic), operation_id="wrong")


def test_helper_comparison_original_out_alias_finite_and_readonly(prepared):
    # Small CPU tensors test comparator logic only; they are not F1 execution.
    torch = pytest.importorskip("torch")
    root, ready = prepared
    entry = evaluator_module()
    helpers = entry.comparator(Path(ready["stage"]["root"]) / "evaluator/harness.py", torch)
    inputs = {key: torch.ones((2,2), dtype=torch.bfloat16) for key in entry.READONLY}
    inputs["out"] = torch.empty((2,2), dtype=torch.bfloat16)
    expected = torch.ones((2,2), dtype=torch.bfloat16)
    before = entry.snapshot(inputs, torch)
    helpers["poison"](inputs, ["out"])
    assert not entry.complete_check(inputs["out"], inputs, before, expected, helpers, torch)["passed"]
    inputs["out"].fill_(1)
    assert entry.complete_check(inputs["out"], inputs, before, expected, helpers, torch)["passed"]
    assert not entry.complete_check(inputs["out"].clone(), inputs, before, expected, helpers, torch)["passed"]
    inputs["query"][0,0] = 2
    assert not entry.complete_check(inputs["out"], inputs, before, expected, helpers, torch)["passed"]
    inputs["query"][0,0] = 1
    inputs["out"].set_(torch.ones((2,2), dtype=torch.bfloat16))
    assert not entry.complete_check(inputs["out"], inputs, before, expected, helpers, torch)["out_view_intact"]
    assert not helpers["checks"](expected, torch.full_like(expected, float("inf")), {"atol":0.06,"rtol":0.04})["passed"]
    # Original elementwise inequality, not torch.allclose: one bad element fails.
    bad = expected.clone(); bad[0,0] = 1.2
    assert not helpers["checks"](bad, expected, {"atol":0.06,"rtol":0.04})["passed"]


def test_prepare_process_imports_no_cuda_original_modules(tmp_path):
    code = "from examples.workflows.fa4_decode.prepare import prepare; from pathlib import Path; import sys; prepare(Path(sys.argv[1]), checkout=Path.cwd()); assert not any(n.split('.')[0] in ('torch','cutlass','cuda','qoder_greenfield_decode_attention') for n in sys.modules)"
    run = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path / "fresh")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
