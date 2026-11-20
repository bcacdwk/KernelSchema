from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from dataclasses import replace

import yaml

from kernelschema import api, fidelity
from kernelschema.artifacts import Publication, durable_write, import_bundle
from kernelschema.contracts import PROTOCOL, jcs
from kernelschema.evaluation import RowSoftmaxTask
from kernelschema.implement.reference import ReferenceBackend
from kernelschema.resources import resource

spec = spec_from_file_location("fidelity_support", Path(__file__).parent / "fixtures/reference_support.py")
support = module_from_spec(spec)
spec.loader.exec_module(support)


def context_for(make_bundle, tmp_path, *, code=support.CODE, schema=None, label="a"):
    runtime = support.FakeRuntime()
    backend = ReferenceBackend(runtime_factory=lambda _: runtime)
    f = make_bundle(directory=tmp_path / ("bundle-" + label))
    p = Publication(f["root"])
    source = p.put_source("source", code.encode())
    p.commit([source])
    support.prepare_fixture(f, backend=backend, mode="verify_only", seed=source, schema_directory=schema)
    ctx = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / ("run-" + label), backend_identity={"synthetic": True})
    handle = runtime.admit(ctx, operation_id="review", role="reviewer", purpose="synthetic finite review",
                           source=source, reservation={"synthetic": True})
    receipt = runtime.complete(ctx, handle)
    scope = RowSoftmaxTask(ctx.bundle, ctx.request["task"]).validate_scope(ctx.request["schema"], ctx.request["scope"])
    return ctx, source, receipt, scope


def publish(ctx, source, receipt, scope):
    return fidelity.report(ctx, source=source, operation_id="review", scope_validation=scope,
                           receipt=receipt, publication=Publication(ctx.root))[1]


def test_same_C_current_R0_vs_legal_multirow_Rd(make_bundle, tmp_path):
    document = yaml.safe_load(resource("examples/workflows/row_softmax/schema/root.yaml").read_text())
    step = document["DESIGN"]["Work_schedule"]["Step"]["MAIN"]
    step["Decomposition"]["Partition"]["row"]["tile"] = 4
    step["Mapping"]["Worker_count"] = "ceildiv(m, 4)"
    step["Schedule"]["Worker_assignment"]["Preassigned"]["count"] = 4
    step["Describe"] = "Each CTA owns up to four complete rows in one launch"
    step["Decomposition"]["Worktile"]["Invariant"]["x"]["shape"] = [{"row": 4}, {"column": "n"}]
    step["Decomposition"]["Target"]["Single_tile"]["y"]["shape"] = [{"row": 4}, {"column": "n"}]
    document["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]["Info"]["Describe"] = "Load, normalize and write up to four independently owned rows"
    document["REALIZATION"]["Require"][0]["Choice"] = "One self-owned Triton kernel; each program owns up to four independent complete rows"
    rd = tmp_path / "rd"
    rd.mkdir()
    (rd / "root.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    code = support.CODE + "# MULTI_ROW synthetic design counterexample\n"
    c0 = context_for(make_bundle, tmp_path, code=code, label="r0")
    # Reuse the *same* original Task/Rules/C manifests, not a separately frozen
    # equivalent Task with different Ref prefixes.
    original, source = c0[0], c0[1]
    rd_bundle = tmp_path / "rd-bundle"
    import_bundle(original.bundle, rd_bundle, [original.request_ref])
    p = Publication(rd_bundle)
    raw = p.put_bytes("rd/root.yaml", (rd / "root.yaml").read_bytes())
    change = p.put_json("change.json", {"protocol": PROTOCOL, "kind": "change_report", "operation_id": "synthetic-r0-rd",
        "subject": {"request_sha256": original.request_sha256, "task_sha256": original.request["task"]["sha256"]},
        "changes": [{"location": "DESIGN.Work_schedule", "reason": "synthetic authorized multirow ownership counterexample"}],
        "raw": [raw], "evidence": [original.request_ref]})
    revised_schema = p.put_json("schema.json", {"protocol": PROTOCOL, "kind": "schema", "task_sha256": original.request["task"]["sha256"],
        "raw": [{"node": "root", "file": raw}], "dispatch": None, "design_revision": original.objects["schema"]["design_revision"] + 1,
        "parent": original.request["schema"], "change": "new_revision", "change_report": change})
    request = {**original.request, "schema": revised_schema, "request_id": "synthetic-rd"}
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(rd_bundle / "request.json", jcs(request))
    ctx = api.prepare_run("implement", request=rd_bundle / "request.json", out=tmp_path / "run-rd", backend_identity={"synthetic": True})
    runtime = support.FakeRuntime()
    handle = runtime.admit(ctx, operation_id="review", role="reviewer", purpose="synthetic finite review", source=source, reservation={"synthetic": True})
    scope = RowSoftmaxTask(ctx.bundle, request["task"]).validate_scope(request["schema"], request["scope"])
    cd = ctx, source, runtime.complete(ctx, handle), scope
    r0, revised = publish(*c0), publish(*cd)
    assert c0[1]["sha256"] == cd[1]["sha256"]
    assert r0["subject"]["task_sha256"] == revised["subject"]["task_sha256"]
    assert r0["fidelity"]["status"] == "fail" and revised["fidelity"]["status"] == "pass"
    assert r0["subject"]["schema_sha256"] != revised["subject"]["schema_sha256"]
    assert r0["coverage"]["status"] == revised["coverage"]["status"] == "pass"


def test_external_softmax_fails_fidelity_without_inventing_numeric_failure(make_bundle, tmp_path):
    args = context_for(make_bundle, tmp_path, code="import torch\ndef forward(x, out):\n    return torch.softmax(x, dim=-1)\n")
    report = publish(*args)
    assert report["coverage"]["status"] == "pass"
    assert report["fidelity"]["status"] == "fail"
    assert any("no numerical failure" in item["reason"] for item in report["fidelity"]["items"])


def test_omitting_max_is_current_design_failure_not_claimed_overflow(make_bundle, tmp_path):
    report = publish(*context_for(make_bundle, tmp_path, code=support.CODE + "# OMIT_MAX\n"))
    assert report["fidelity"]["status"] == "fail"
    assert "overflow" not in str(report).lower()


def test_missing_or_forged_locations_never_pass(make_bundle, tmp_path):
    ctx, source, receipt, scope = context_for(make_bundle, tmp_path)
    content = receipt.output.content
    content["items"][0]["source_locations"] = ["kernel.py:L99999"]
    modified = replace(receipt, output=replace(receipt.output, content=content))
    report = publish(ctx, source, modified, scope)
    assert report["fidelity"]["status"] == "unknown"
    assert report["coverage"]["status"] == "pass"
    content["items"].pop()
    report = publish(ctx, source, modified, scope)
    assert report["fidelity"]["status"] == "unknown"


def test_unused_external_helper_is_not_machine_proof_of_delegation(make_bundle, tmp_path):
    code = "import torch\ndef unused(x):\n    return torch.softmax(x, dim=-1)\n" + support.CODE
    report = publish(*context_for(make_bundle, tmp_path, code=code))
    assert report["fidelity"]["status"] == "pass"  # synthetic reviewer; narrow machine does not misclassify dead helper
