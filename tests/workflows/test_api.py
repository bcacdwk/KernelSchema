from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from kernelschema import api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.contracts import PROTOCOL, ProtocolError, UnsupportedError, jcs
from kernelschema.materials import require_capabilities, validate_request

IDENTITY = {"implementation": "K2 synthetic CPU owner fixture", "version": "1"}


@pytest.mark.parametrize("operation", ["design", "optimize", "implement"])
def test_anchor_before_first_result_and_wrong_operation(make_bundle, tmp_path, operation):
    f = make_bundle(operation)
    context = api.prepare_run(operation, request=f["root"] / "request.json", out=tmp_path / "run", backend_identity=IDENTITY)
    context.save_state("owner", {"synthetic": True, "reserved_calls": 1, "fence": "unsettled"})
    context.record_intent(operation_id="synthetic-1", role="coder" if operation == "implement" else "planner", purpose="CPU interruption fixture",
        source=f["starter"] if operation == "implement" else None, reservation={"synthetic": True, "calls": 1}, submission_key="original-key", native_locator={"synthetic": True, "path": "/private/native/original"})
    assert not (context.root / "result.json").exists()
    restored = api.open_run(operation, context.root, backend_identity=IDENTITY)
    assert restored.run_id == context.run_id
    assert restored.load_state("owner")["reserved_calls"] == 1
    assert restored.intents()[0]["submission_key"] == "original-key"
    with pytest.raises(ProtocolError, match="operation"): api.open_run("optimize" if operation == "implement" else "implement", context.root)
    with pytest.raises(ProtocolError, match="implementation"): api.open_run(operation, context.root, backend_identity={"version": "different"})
    with pytest.raises(ProtocolError, match="already exists"): api.prepare_run(operation, request=f["root"] / "request.json", out=context.root, backend_identity=IDENTITY)
    (context.root / ".runtime/owner-binding.json").unlink()
    with pytest.raises(ProtocolError): api.open_run(operation, context.root)


def test_same_Q_new_run_has_distinct_namespace_and_changed_request_cannot_resume(make_bundle, tmp_path):
    f = make_bundle()
    a = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "a", backend_identity=IDENTITY)
    b = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "b", backend_identity=IDENTITY)
    assert a.request_sha256 == b.request_sha256
    assert a.run_id != b.run_id and a.owner != b.owner
    changed = deepcopy(a.request); changed["budget"]["limits"]["calls"]["coder"] += 1
    durable_write(a.root / "request.json", jcs(changed), replace=True)
    with pytest.raises(ProtocolError, match="changed"): api.open_run("implement", a.root)


@pytest.mark.parametrize("entry", [api.design, api.optimize, api.implement])
def test_public_parameter_exclusivity(entry):
    for params in ({}, {"request": "x"}, {"out": "x"}, {"request": "x", "out": "y", "continue_run": "z"}, {"out": "y", "continue_run": "z"}):
        with pytest.raises(ProtocolError): entry(**params)


def test_unsupported_backend_does_not_create_run(make_bundle, tmp_path):
    f = make_bundle()
    with pytest.raises(UnsupportedError, match="not yet implemented"):
        api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_result_publication_can_reconstruct_projection(make_bundle, tmp_path):
    f = make_bundle()
    context = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "run", backend_identity=IDENTITY)
    p = Publication(context.root)
    usage = p.put_json("usage.json", {"protocol": PROTOCOL, "kind": "usage_index", "entries": [], "extensions": {"information": {"synthetic": True}}})
    p.commit([usage])
    result = {"protocol": PROTOCOL, "operation": "implement", "run_id": context.run_id, "request_id": context.request["request_id"], "request_sha256": context.request_sha256,
        "status": "partial", "stop_reason": "synthetic fixture; no execution", "usage": {"scope": "request_cumulative", "receipts": usage, "calls": {"planner": 0, "coder": 0, "reviewer": 0, "extract": 0}, "evaluator_jobs": 0, "tokens": 0, "active_ms": 0, "unsettled": []},
        "feedback": [], "recovery": "none", "code": {"latest": f["starter"], "qualified": None}, "assessments": [], "extensions": {"information": {"synthetic": True}}}
    context.publish_result(result)
    history = list((context.root / ".runtime/results").iterdir())
    (context.root / "result.json").unlink()
    restored = api.open_run("implement", context.root)
    restored.publish_result(result)
    assert list((context.root / ".runtime/results").iterdir()) == history
    assert (context.root / "result.json").read_bytes() == jcs(result)


def test_core_import_and_editable_cli_from_unrelated_cwd(tmp_path):
    script = '''import sys
class Reject:
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'aka','torch','triton','cuda'}: raise AssertionError(name)
sys.meta_path.insert(0,Reject())
import kernelschema
from kernelschema.resources import resource
from kernelschema.template_api import tools
assert resource('schema_docs/0.common.md').is_file()
assert tools().__name__ == 'kernelschema._template_tools.schema'
assert 'tools' not in sys.modules
print(kernelschema.__version__)
'''
    run = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    executable = str(Path(sys.executable).parent / "kernelschema")
    cli = subprocess.run([executable, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert cli.returncode == 0 and "design" in cli.stdout
    for op in ("design", "optimize", "implement"):
        bad = subprocess.run([executable, op, "--continue", "x", "--out", "y"], cwd=tmp_path, capture_output=True, text=True)
        assert bad.returncode != 0 and "--continue cannot use --out" in bad.stderr


def test_result_cannot_attach_C1_evaluation_to_C2(make_bundle, tmp_path):
    from test_artifacts import evaluation
    f = make_bundle()
    context = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "run", backend_identity=IDENTITY)
    p = Publication(context.root)
    native = p.put_bytes("native.json", b'{"synthetic":true,"build":"pass"}')
    report = evaluation(f)
    report["build"] = {"status": "pass", "receipt": native, "reason": "synthetic build evidence"}
    report["raw_receipts"] = [native]
    report_ref = p.put_json("evaluation.json", report)
    usage = p.put_json("usage.json", {"protocol": PROTOCOL, "kind": "usage_index", "entries": []})
    p.commit([report_ref, usage], subjects=[context.request_ref])
    not_run = {"status": "not_run", "receipt": None, "reason": "no execution"}
    assessment = {"source_sha256": f["baseline"]["sha256"], "coverage": not_run, "task_uncovered": [], "build": {"status": "pass", "receipt": report_ref, "reason": "wrong candidate"}, "correctness": not_run, "fidelity": not_run, "performance": not_run, "performance_goal": "unknown"}
    result = {"protocol": PROTOCOL, "operation": "implement", "run_id": context.run_id, "request_id": context.request["request_id"], "request_sha256": context.request_sha256,
        "status": "partial", "stop_reason": "synthetic mismatch fixture", "usage": {"scope": "request_cumulative", "receipts": usage, "calls": {"planner": 0, "coder": 0, "reviewer": 0, "extract": 0}, "evaluator_jobs": 0, "tokens": 0, "active_ms": 0, "unsettled": []},
        "feedback": [], "recovery": "none", "code": {"latest": f["baseline"], "qualified": None}, "assessments": [assessment]}
    with pytest.raises(ProtocolError, match="another actual subject"): context.publish_result(result)
    assert not (context.root / "result.json").exists()


def test_finite_tokens_require_actual_hard_enforcement(make_bundle):
    f = make_bundle(); request = deepcopy(f["request"])
    request["budget"]["limits"]["tokens"] = 10
    objects = validate_request(Bundle(f["root"]), request)
    with pytest.raises(UnsupportedError, match="hard token"): require_capabilities(request, objects, f["caps"])
