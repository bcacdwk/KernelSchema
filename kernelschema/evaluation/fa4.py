"""Fixed FA4 F1 preparation. No CUDA import, submission, or inferred qualification.

This module owns the case boundary, not a second Schema parser, backend or
budget ledger. Source codec and Template inheritance use the core interfaces.
"""
from __future__ import annotations

import ast
from decimal import Decimal
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..artifacts import (Bundle, FileBytes, Publication, _OverlayBundle, _rename_new, durable_mkdir,
                        durable_write, fsync_directory, read_regular)
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, sha256, validate
from ..materials import validate_request
from ..resources import resource
from .. import template_api

EXAMPLE = "examples/workflows/fa4_decode/"
VERSION = "fa4-decode-f1.evaluator.v1"
TASK_ID = "fa4_decode_shared_contract_f1_v1"
F1 = "fa4_decode.f1.q4_direct.b1.q4.l64.p1"
REST = "fa4_decode.shared_contract.except_f1"
SOURCE_COMMIT = "3d64ad35900a0e6a66d6f93da34108fbceba0daf"
SCHEMA_COMMIT = "484902855fe7be82a74fe7c394503d1d7d222dbb"
ORIGINAL = "kernel_wiki/aka-four-problems-20261019/fa4_decode/"
SOURCE_ROOT = ORIGINAL + "aka_solution/"
SCHEMA_ROOT = "schema_template/examples/fa4_decode/"
HARNESS = "kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/fa4_decode/harness.py"
ABI = ("query", "kv_cache", "workspace_buffer", "block_tables", "seq_lens", "max_seq_len",
       "bmm1_scale", "bmm2_scale", "window_left", "sinks", "o_sf_scale", "out", "q_len_per_req")


def _case(name):
    return resource(EXAMPLE + name).read_bytes()


def originals(checkout: str | Path) -> dict[str, FileBytes]:
    """Read inventory-selected fixed Git blobs; never import original modules."""
    inventory = json.loads(_case("materials.json"))
    if (inventory["kernel_source_commit"], inventory["schema_commit"]) != (SOURCE_COMMIT, SCHEMA_COMMIT):
        raise ProtocolError("case inventory commits differ")
    payload = {}
    for entry in inventory["entries"]:
        if entry["commit"] not in (SOURCE_COMMIT, SCHEMA_COMMIT):
            raise ProtocolError("unselected material commit")
        coordinate = entry["commit"] + ":" + entry["path"]
        result = subprocess.run(["git", "-C", str(checkout), "show", coordinate], capture_output=True, check=True)
        data = result.stdout
        blob = subprocess.run(["git", "-C", str(checkout), "rev-parse", coordinate], capture_output=True, check=True).stdout.decode().strip()
        tree = subprocess.run(["git", "-C", str(checkout), "ls-tree", entry["commit"], "--", entry["path"]], capture_output=True, check=True).stdout.decode()
        if (len(data) != entry["size"] or sha256(data) != entry["sha256"] or blob != entry["git_blob"]
                or not tree.startswith("100" + entry["mode"][1:] + " blob ")):
            raise ProtocolError("original material hash/size/mode/blob differs: " + coordinate)
        payload[entry["path"]] = FileBytes(data, int(entry["mode"], 8))
    # Bind all six original Python dependencies and the 13-position ABI.
    selected = {p.removeprefix(SOURCE_ROOT) for p in payload if p.startswith(SOURCE_ROOT)}
    if len(selected) != inventory["selection"]["full_aka_solution_file_count"]:
        raise ProtocolError("incomplete original Source inventory")
    for name in inventory["selection"]["runtime_python_files"]:
        if name not in selected:
            raise ProtocolError("missing runtime dependency: " + name)
    for path in (SOURCE_ROOT + "solution.py", ORIGINAL + "reference.py"):
        tree = ast.parse(payload[path].data)
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
        method = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        if tuple(a.arg for a in method.args.args[1:]) != ABI:
            raise ProtocolError("original 13-argument ABI differs")
    return payload


def freeze_case(publication: Publication, *, checkout: str | Path) -> dict:
    """Freeze real Task, complete original Source, four-node Schema and F1 Scope."""
    payload = originals(checkout)
    put = lambda name, data: publication.put_bytes("fa4/" + name, data)
    retained = {path: put("originals/" + path, file.data) for path, file in payload.items()}
    baseline = publication.put_source_files("fa4/baseline", {
        path.removeprefix(SOURCE_ROOT): file for path, file in payload.items() if path.startswith(SOURCE_ROOT)})
    local = {name: put(name, _case(name)) for name in
             ("f1-parameters.json", "materials.json", "preflight-checks.json", "licenses-and-gaps.md", "candidate-interface.md", "task/acceptance.md",
              "task/workload.json", "task/timing.json", "task/environment.json", "evaluator/entry.py")}
    evaluator_files = [local["evaluator/entry.py"], retained[ORIGINAL + "input.py"],
                       retained[ORIGINAL + "reference.py"], retained[HARNESS]]
    evaluator = publication.put_json("fa4/evaluator.json", {
        "protocol": PROTOCOL, "kind": "evaluator", "version": VERSION,
        "entrypoint": local["evaluator/entry.py"], "files": sorted(evaluator_files, key=lambda r: r["path"]),
        "inputs": local["f1-parameters.json"], "timing": local["task/timing.json"],
        "environment": local["task/environment.json"], "adapters": ["fa4-original-stack-standalone-v1", "fa4-agate-f1-v1"],
        "extensions": {"information": {"qualification": "none", "transport": "standalone observation or explicit qualified FA4 Agate channel; no implicit launch"}}})
    task = publication.put_json("fa4/task.json", {
        "protocol": PROTOCOL, "kind": "task", "task_id": TASK_ID,
        "contract": retained[SCHEMA_ROOT + "root.yaml"], "acceptance": local["task/acceptance.md"],
        "workload": local["task/workload.json"], "environment": local["task/environment.json"],
        "evaluator": evaluator, "baseline_source": baseline, "starter_source": None})
    schema = publication.put_json("fa4/schema.json", {
        "protocol": PROTOCOL, "kind": "schema", "task_sha256": task["sha256"],
        "raw": [{"node": name, "file": retained[SCHEMA_ROOT + name + ".yaml"]}
                for name in ("q1_global", "q4_cluster", "q4_direct", "root")],
        "dispatch": retained[SCHEMA_ROOT + "dispatch.md"], "design_revision": 0,
        "parent": None, "change": "new_task", "change_report": None})
    scope = publication.put_json("fa4/scope.json", {
        "protocol": PROTOCOL, "kind": "scope", "coverage_ids": [F1],
        "bindings": [{"coverage_ids": [F1], "node": "q4_direct", "variant": None}]})
    objective = put("objective.md", b"Verify the fixed original FA4 F1 only. Retain shared Contract, exact q4_direct Design and full-forward timing. No qualification is currently established.\n")
    materials = publication.put_json("fa4/workflow-materials.json", {
        "protocol": PROTOCOL, "kind": "materials", "entries": [
            {"id": "candidate_interface", "file": local["candidate-interface.md"], "roles": ["coder", "reviewer"],
             "origin": "KS finite F1 evaluator observation interface; no baseline implementation", "license": "No original-source license grant"},
            {"id": "k7_preflight", "file": local["preflight-checks.json"], "roles": [],
             "origin": "K7 CPU-only observations; not current execution evidence", "license": "No original-source license grant"},
            {"id": "license_gaps", "file": local["licenses-and-gaps.md"], "roles": ["coder", "planner", "reviewer"],
             "origin": "K7 material audit", "license": "No additional grant; preserve original notices"},
            {"id": "pinned_inventory", "file": local["materials.json"], "roles": [],
             "origin": "K7 fixed-commit byte inventory", "license": "Original notices unchanged; licensing unresolved"},
            {"id": "workflow_objective", "file": objective, "roles": ["coder", "planner", "reviewer"],
             "origin": "K8 fixed case objective", "license": "No original-source license grant"}]})
    # Original bytes remain available without disclosing optimized Source as
    # role materials. Provenance/history is not part of a pure-Schema prompt.
    result = {"task": task, "schema": schema, "scope": scope, "source": baseline,
              "evaluator": evaluator, "materials": materials, "originals": retained,
              "case_materials": local}
    publication.commit([task, schema, scope, materials, *retained.values(), *local.values()])
    return result


class FA4Task:
    def __init__(self, bundle: Bundle, task: dict):
        self.bundle, self.reference = bundle, task
        self.task = bundle.document(task, "task")
        if self.task["task_id"] != TASK_ID:
            raise UnsupportedError("expected fixed FA4 F1 Task")
        entries = {e["path"]: e for e in json.loads(_case("materials.json"))["entries"]}
        if sha256(bundle.read(self.task["contract"])) != entries[SCHEMA_ROOT + "root.yaml"]["sha256"]:
            raise UnsupportedError("FA4 shared root Contract differs")
        for field, name in (("acceptance", "acceptance.md"), ("workload", "workload.json"), ("environment", "environment.json")):
            if bundle.read(self.task[field]) != _case("task/" + name):
                raise UnsupportedError("FA4 fixed " + field + " differs")
        self.partitions = {F1, REST}

    def evaluation_capabilities(self):
        return {"scope_mode": "fixed_f1_only", "coverage_ids": [F1], "subset_only": True,
                "task_uncovered": [REST], "source_formats": ["python_tree"], "original_source_only": False}

    def require_evaluation_scope(self, scope):
        doc = self.bundle.document(scope, "scope")
        return self.require_evaluation_scope_document(doc)

    def require_evaluation_scope_document(self, doc):
        validate(doc)
        if doc.get("kind") != "scope":
            raise ProtocolError("expected a Scope document")
        if doc["coverage_ids"] != [F1]:
            raise UnsupportedError("FA4 evaluator supports only the fixed F1 point")
        return self.evaluation_capabilities()

    def validate_scope(self, schema, scope):
        self.require_evaluation_scope(scope)
        selected = self.bundle.document(scope, "scope")
        doc = self.bundle.document(schema, "schema")
        if doc["task_sha256"] != self.reference["sha256"]:
            raise ProtocolError("Schema belongs to another Task")
        resolved = template_api.resolve_bundle(self.bundle, doc)
        pinned = {e["path"]: e["sha256"] for e in json.loads(_case("materials.json"))["entries"]}
        exact = (set(resolved.raw_task) == {"root", "q4_direct", "q4_cluster", "q1_global"}
                 and all(r["file"]["sha256"] == pinned.get(SCHEMA_ROOT + r["node"] + ".yaml") for r in doc["raw"])
                 and doc["dispatch"] is not None and doc["dispatch"]["sha256"] == pinned[SCHEMA_ROOT + "dispatch.md"])
        bindings = selected["bindings"]
        direct = bindings == [{"coverage_ids": [F1], "node": "q4_direct", "variant": None}]
        root_binding = any(b["node"] == "root" for b in bindings)
        status = "pass" if exact and direct else "fail" if exact and root_binding else "unknown"
        effective = resolved.effective_document("q4_direct") if "q4_direct" in resolved.raw_task else {}
        return {"status": status, "coverage_ids": [F1] if status == "pass" else [],
            "mappings": [{"coverage_id": F1, "node": "q4_direct", "variant": None}] if status == "pass" else [],
            "task_uncovered": [REST] if status == "pass" else sorted(self.partitions),
            "requested_uncovered": [] if status == "pass" else [F1],
            "reason": "pinned F1 material predicate; runtime when not executed" if status == "pass" else
                      "root S9 is not a direct/Q1 fallback" if status == "fail" else "unrecognized Schema/binding; no prose inference",
            "effective_contract_sha256": sha256(jcs(_wire(effective.get("CONTRACT", {})))),
            "runtime_when": "not_run", "fidelity": "unknown",
            "limitations": ["fixed F1 only", "native launcher/object/signature/thread/device/capture conditions require execution",
                "empty parity, TMA padding/page tail, TMEM and output staging lifetime remain unproved"]}


def _wire(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    return value


def freeze_request(publication: Publication, case: dict, *, rules: dict, profile: dict,
                   limits: dict, intent="original_verify", source=None, parent=None):
    """Build a normal K2 request; installed backend/role capabilities still gate it.

    No automatic profile expansion: installed backend and requested profile
    must support python_tree. Pure Schema reconstruction never grants baseline
    contents merely because the Task carries its performance-baseline Ref.
    """
    if intent not in ("original_verify", "seed_continuation", "schema_rebuild"):
        raise UnsupportedError("unknown FA4 source provenance intent")
    if intent == "original_verify" and source is not None:
        raise ProtocolError("original_verify selects the pinned original Source only")
    if intent == "seed_continuation" and source is None:
        raise ProtocolError("seed continuation requires an explicit full Source")
    if intent == "schema_rebuild" and source is not None:
        raise ProtocolError("pure Schema reconstruction starts without optimized source")
    bundle = Bundle(publication.root)
    backend = bundle.document(profile, "backend_profile")["name"]
    materials = case["materials"]
    if intent != "original_verify":
        objective = ("Continue the explicitly selected complete Source for fixed FA4 F1. Preserve the shared Contract and q4_direct Design."
                     if intent == "seed_continuation" else
                     "Rebuild fixed FA4 F1 from its Schema with no optimized Source or history delivery. Preserve the shared Contract and q4_direct Design.")
        objective_ref = publication.put_bytes("fa4/" + intent + "-objective.md", (objective + "\n").encode())
        material_doc = bundle.document(materials, "materials")
        for entry in material_doc["entries"]:
            if entry["id"] == "workflow_objective":
                entry["file"] = objective_ref
                entry["origin"] = "K8 explicit " + intent + " objective"
        materials = publication.put_json("fa4/" + intent + "-materials.json", material_doc)
    request = {"protocol": PROTOCOL, "operation": "implement", "request_id": "fa4-f1-" + intent,
        "task": case["task"], "schema": case["schema"], "scope": case["scope"], "materials": materials,
        "rules": rules, "backend": {"name": backend, "profile": profile}, "budget": {"parent": parent, "limits": limits},
        "seed": case["source"] if intent == "original_verify" else source,
        "mode": "verify_only" if intent == "original_verify" else "code",
        "extensions": {"information": {"source_intent": intent, "K8_real": "not_qualified"}}}
    validate_request(_OverlayBundle(publication.stage, publication.root), request)
    ref = publication.put_json("fa4/request.json", request)
    publication.commit([ref])
    return ref


def stage_original(bundle: Bundle, case: dict, destination: str | Path) -> dict:
    """Complete, atomic read-only-testable stage for the pinned original only.

    This standalone runner is not an untrusted-candidate sandbox or an Agate
    adapter. Changed/seed-generated Sources require a qualified backend seam.
    """
    return _stage(bundle, case, destination, original_only=True)


def stage_candidate(bundle: Bundle, case: dict, destination: str | Path) -> dict:
    """Freeze both complete Sources; execution requires the FA4 channel boundary."""
    return _stage(bundle, case, destination, original_only=False)


def execution_materials(bundle, task):
    """One trusted evaluator/baseline check, shared by admission and staging."""
    evaluator = bundle.document(task["evaluator"], "evaluator")
    if evaluator["version"] != VERSION or evaluator["environment"] != task["environment"]:
        raise UnsupportedError("unsupported FA4 evaluator/environment")
    inventory = {e["path"]: e for e in json.loads(_case("materials.json"))["entries"]}
    expected = {"entry.py": _case("evaluator/entry.py")}
    for path in (ORIGINAL + "input.py", ORIGINAL + "reference.py", HARNESS):
        expected[Path(path).name] = None
    files = {}
    if len(evaluator["files"]) != 4:
        raise ProtocolError("FA4 evaluator needs exactly four implementation files")
    for reference in evaluator["files"]:
        name = Path(reference["path"]).name
        if name not in expected or "evaluator/" + name in files:
            raise ProtocolError("FA4 evaluator component differs")
        data = bundle.read(reference)
        original_path = HARNESS if name == "harness.py" else ORIGINAL + name
        if (data != expected[name] if name == "entry.py" else sha256(data) != inventory[original_path]["sha256"]):
            raise UnsupportedError("FA4 evaluator differs from pinned trusted bytes")
        files["evaluator/" + name] = FileBytes(data, 0o644)
    if evaluator["entrypoint"] not in evaluator["files"] or Path(evaluator["entrypoint"]["path"]).name != "entry.py":
        raise ProtocolError("FA4 entrypoint differs")
    for key, name in (("inputs", "f1-parameters.json"), ("timing", "task/timing.json"), ("environment", "task/environment.json")):
        data = bundle.read(evaluator[key])
        if data != _case(name):
            raise UnsupportedError("FA4 evaluator criteria differ: " + key)
        files[Path(name).name] = FileBytes(data, 0o644)
    original_files = {p.removeprefix(SOURCE_ROOT): e for p, e in inventory.items() if p.startswith(SOURCE_ROOT)}
    baseline_files = bundle.source_files(task["baseline_source"])
    if set(baseline_files) != set(original_files) or any(sha256(f.data) != original_files[p]["sha256"] for p, f in baseline_files.items()):
        raise UnsupportedError("standalone original source differs from pinned twelve-file snapshot")
    return evaluator, files, baseline_files


def _stage(bundle, case, destination, *, original_only):
    adapter = FA4Task(bundle, case["task"])
    coverage = adapter.validate_scope(case["schema"], case["scope"])
    if coverage["status"] != "pass":
        raise UnsupportedError("F1 fixed material scope not admitted")
    source = case["source"]
    if original_only and source["sha256"] != adapter.task["baseline_source"]["sha256"]:
        raise UnsupportedError("standalone entry supports pinned original verification only")
    evaluator, files, baseline_files = execution_materials(bundle, adapter.task)
    if bundle.document(source, "source")["format"] != "python_tree":
        raise UnsupportedError("FA4 requires the complete python_tree Source")
    source_files = bundle.source_files(source)
    files.update({"source/" + name: data for name, data in source_files.items()})
    if not original_only:
        files.update({"baseline/" + name: data for name, data in baseline_files.items()})
        files["_fa4_remote.py"] = FileBytes(Path(__file__).with_name("_fa4_remote.py").read_bytes(), 0o644)
        files["_remote.py"] = FileBytes(Path(__file__).with_name("_remote.py").read_bytes(), 0o644)
    manifest = {"version": VERSION, "task": case["task"], "schema": case["schema"], "scope": case["scope"],
        "source": source, "evaluator": adapter.task["evaluator"], "inputs": evaluator["inputs"], "timing": evaluator["timing"],
        "environment": evaluator["environment"], "coverage": coverage,
        "files": {name: {"sha256": sha256(file.data), "mode": f"{file.mode:04o}", "size": len(file.data)} for name, file in files.items()}}
    if not original_only:
        manifest.update(baseline=adapter.task["baseline_source"], source_manifest=bundle.document(source, "source"),
                        baseline_manifest=bundle.document(adapter.task["baseline_source"], "source"),
                        execution_boundary="fa4-agate-f1-bwrap-v1")
    destination = Path(destination).absolute()
    durable_mkdir(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix=".ks-fa4-stage-", dir=destination.parent))
    try:
        for name, file in files.items():
            durable_write(temporary / name, file.data, mode=file.mode)
        durable_write(temporary / "stage.json", jcs(manifest))
        fsync_directory(temporary)
        _rename_new(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"root": str(destination), "sha256": digest(manifest), "manifest": manifest}


def verify_stage(staged):
    root = Path(staged["root"])
    data = read_regular(root, "stage.json")[0]
    if sha256(data) != staged["sha256"] or json.loads(data) != staged["manifest"]:
        raise ProtocolError("FA4 stage identity differs")
    manifest = staged["manifest"]
    expected = set(manifest["files"]) | {"stage.json"}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("FA4 stage symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ProtocolError("FA4 stage has missing/extra files")
    for name, facts in manifest["files"].items():
        data, mode = read_regular(root, name)
        if sha256(data) != facts["sha256"] or len(data) != facts["size"] or mode != int(facts["mode"], 8):
            raise ProtocolError("FA4 staged file differs: " + name)
    return manifest


def freeze_observation(publication: Publication, staged: dict, *, raw: bytes, operation_id: str, usage_receipts=()):
    """Bind standalone facts to immutable C/Task/evaluator/inputs/timing.

    Manual stdout cannot authenticate an external execution/permission boundary.
    Preserve reported booleans as evidence but never promote them to five gates.
    The FA4 channel has a separate qualified frame/subject/permission reporting
    seam; standalone observations cannot use it to bypass channel qualification.
    """
    verify_stage(staged)
    observation = json.loads(raw)
    if observation.get("stage_sha256") != staged["sha256"] or observation.get("subject") != {
        k: staged["manifest"][k]["sha256"] for k in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")}:
        raise ProtocolError("FA4 observation subject differs")
    evidence = publication.put_bytes("fa4/observed-raw.json", raw)
    environment = publication.put_bytes("fa4/observed-environment.json", jcs(_wire(observation.get("environment", {}))))
    unknown = {"status": "unknown", "receipt": evidence,
               "reason": "standalone observation only; original-stack transport/readonly/full-stdout qualification remains absent"}
    report = publication.put_json("fa4/evaluation.json", {"protocol": PROTOCOL, "kind": "evaluation",
        "operation_id": operation_id,
        "subject": {**{key + "_sha256": staged["manifest"][key]["sha256"] for key in ("task", "source", "evaluator", "inputs", "timing")},
                    "environment_sha256": environment["sha256"]},
        "build": unknown, "correctness": unknown, "performance": unknown, "case_ids": [F1],
        "samples": evidence if observation.get("samples_ns") else None, "raw_receipts": [evidence],
        "usage_receipts": list(usage_receipts), "environment": environment,
        "extensions": {"information": {"K8_real": "not_qualified", "fidelity": "unknown", "performance_goal": "unknown"}}})
    return report
