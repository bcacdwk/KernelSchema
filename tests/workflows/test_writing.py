"""CPU engineering only. Fake CLI is an actual process, all verdicts synthetic."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from kernelschema import api, template_api, writing
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle
from kernelschema.budget import OwnerAccount, use_authority, cost
from kernelschema.contracts import PROTOCOL, ProtocolError, digest, jcs, loads, sha256
from kernelschema.runtime import ClaudeProfile, ClaudeRuntime
from kernelschema._writing_policy import template_contract


def draft_for(f, operation, *, decision="revise"):
    # The test owns its synthetic script; actual design role input has no R0.
    doc = template_api.effective_document(Bundle(f["root"]), Bundle(f["root"]).document(f["schema"], "schema"), "root") if f["schema"] else yaml.safe_load((Path(__file__).parents[2] / "examples/workflows/row_softmax/schema/root.yaml").read_text())
    if operation == "design":
        doc["CONTRACT"] = template_contract(Bundle(f["root"]), f["task"])
    if operation == "optimize" and decision == "revise":
        doc["DESIGN"]["Work_schedule"]["Step"]["MAIN"]["Schedule"]["Worker_assignment"]["Preassigned"]["count"] = 2
    return {"format": "ks.writing.draft.v1", "decision": decision, "reason": "SYNTHETIC proposed route for CPU state validation",
            "selection": [{"id": "workflow_objective", "reason": "fixed current Task objective"}],
            "raw": [] if decision == "keep" else [{"node": "root", "text": yaml.safe_dump(doc, sort_keys=False, width=1000)}],
            "dispatch": None, "bindings": None if decision == "keep" else [{"coverage_ids": sorted(Bundle(f["root"]).document(f["request"]["scope"], "scope")["coverage_ids"]), "node": "root", "variant": None}],
            "next_action": "none", "action_reason": None, "action_inputs": [], "action_evidence": []}


@pytest.fixture
def setup_writing(make_bundle, tmp_path):
    def setup(operation="design", *, decision="revise", alter=None, review="pass", checkpoint=None, parent=None, writing_mode="strict"):
        directory = tmp_path / ("bundle-" + str(len(list(tmp_path.iterdir()))))
        f = make_bundle(operation, directory=directory)
        fake = tmp_path / ("fake-" + directory.name)
        fake.mkdir()
        script = Path(__file__).parent / "fixtures/writing/fake_claude.py"
        executable = fake / "claude"
        executable.write_text("#!" + sys.executable + "\n" + "\n".join(script.read_text().splitlines()[1:]) + "\n")
        executable.chmod(0o755)
        profile = ClaudeProfile.inspect(requested_model="synthetic-model", executable=str(executable), timeout_ms=5000)
        backend = writing.WritingBackend(operation, checkpoint=checkpoint)
        draft = draft_for(f, operation, decision=decision)
        if alter:
            alter(draft)
        (fake / "scenario.json").write_text(json.dumps({"draft": draft, "review": review}))
        p = Publication(f["root"])
        rules = writing.freeze_rules(p, writing_mode=writing_mode)
        business = writing.backend_profile(profile, f["originals"]["task/environment.md"], writing_mode=writing_mode)
        profile_ref = p.put_json("writing-profile.json", business)
        budget = {"parent": parent, "limits": {"calls": {"planner": 1, "reviewer": 1, "coder": 0, "extract": 0}, "tokens": None, "active_ms": 60000, "evaluator_jobs": 0}}
        req = writing.request(operation=operation, request_id="synthetic-writing", task=f["task"], rules=rules, materials=f["request"]["materials"], scope=f["request"]["scope"], budget=budget, profile=profile_ref, schema=f["schema"])
        ref = p.put_json("writing-request.json", req)
        p.commit([ref])
        durable_write(f["root"] / "request.json", jcs(req), replace=True)
        out = tmp_path / ("run-" + directory.name)
        return f, backend, fake, req, out
    return setup


def run(setup, monkeypatch):
    f, backend, fake, req, out = setup
    monkeypatch.setattr(api, "_backend", lambda *_: backend)
    return getattr(api, req["operation"])(request=f["root"] / "request.json", out=out)


def calls(fake):
    path = fake / "invocations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def with_material_grants(setup):
    """Synthetic role boundaries, including a private Ref with no role grant."""
    f, backend, fake, req, out = setup
    bundle = Bundle(f["root"])
    materials = bundle.document(req["materials"])
    for identifier, roles, ref in (
        ("planner_only", ["planner"], f["originals"]["task/timing.md"]),
        ("reviewer_only", ["reviewer"], f["originals"]["task/acceptance.md"]),
        ("private_only", [], bundle.document(f["evaluator"])["entrypoint"]),
    ):
        materials["entries"].append({"id": identifier, "file": ref, "roles": roles,
                                     "origin": "synthetic role-boundary test", "license": "test fixture"})
    materials["entries"].sort(key=lambda entry: entry["id"])
    p = Publication(f["root"])
    req = {**req, "materials": p.put_json("role-grants.json", materials)}
    ref = p.put_json("request-with-grants.json", req)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    return f, backend, fake, req, out


@pytest.mark.parametrize("operation,decision", [("design", "revise"), ("optimize", "keep"), ("optimize", "revise")])
def test_writing_input_exposes_exact_grants_and_original_text_budgets(setup_writing, monkeypatch, operation, decision):
    from kernelschema.runtime import material_text
    setup = with_material_grants(setup_writing(operation, decision=decision))
    f, _, fake, req, _ = setup
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result  # independent synthetic draft/review only
    planner, reviewer = calls(fake)
    supplied = loads(material_text(planner, "delivery/writing_input").encode())
    reviewed = loads(material_text(reviewer, "delivery/review_subject").encode())
    bundle = Bundle(f["root"])
    granted = [e for e in bundle.document(req["materials"])["entries"] if "planner" in e["roles"]]
    assert [e["id"] for e in supplied["selection_catalog"]] == [e["id"] for e in granted]
    delivered = {e["id"]: e for e in planner["materials"]}
    for selection, original in zip(supplied["selection_catalog"], granted):
        assert selection["ref"] == original["file"] == delivered[selection["delivery_id"]]["ref"]
        assert selection["delivery_id"] == "material/" + selection["id"]
        assert material_text(planner, selection["delivery_id"]).encode() == bundle.read(selection["ref"])
        if "reviewer" in original["roles"]:
            assert selection["action_id"] == "material:" + selection["id"]
            assert supplied["action_catalog"][selection["action_id"]] == selection["ref"]
        else:
            assert selection["action_id"] is None
            assert "material:" + selection["id"] not in supplied["action_catalog"]
    assert "material/reviewer_only" not in delivered and "material/private_only" not in delivered
    assert "selection_catalog" not in reviewed  # do not disclose planner-only grants to reviewer
    private_ref = bundle.document(f["evaluator"])["entrypoint"]
    assert private_ref["sha256"] not in jcs(supplied).decode()
    summary = supplied["text_policy"]
    assert reviewed["text_policy"] == summary
    original = template_api.tools().text_policy
    assert summary["source"] in bundle.document(req["rules"])["files"]
    assert bundle.read(summary["source"]) == Path(original.__file__).read_bytes()
    assert summary["source"]["sha256"] == sha256(Path(original.__file__).read_bytes())
    assert summary["symbols"] == ["BUDGETS", "text_size", "check_values"]
    assert len(summary["budgets"]) == len(original.BUDGETS)
    for declaration, budget in zip(summary["budgets"], original.BUDGETS):
        path = tuple(int if part == "[]" else None if part == "*" else part for part in declaration["path"])
        assert (path, declaration["maxLength"], declaration["maxLines"], declaration["prefix"]) == (
            budget.path, budget.characters, budget.lines, budget.prefix)
    assert "Unicode code points" in summary["counting"]["characters"]
    assert "len(value.splitlines())" in summary["counting"]["lines"]
    assert original.text_size("界😀\n\nA\n") == (6, 3)
    for delivery in (planner, reviewer):
        policy_item = next(e for e in delivery["materials"] if e["ref"] == summary["source"])
        assert "host_implementation" in policy_item and "text" not in policy_item
        assert any(material_text(delivery, e["id"]).encode() == writing.STRICT_METHOD
                   for e in delivery["materials"] if "host_implementation" not in e)
    assert b"neither that delivery_id nor material:<id> is a selection ID" in writing.STRICT_METHOD
    assert b"not a Scope object" in writing.STRICT_METHOD
    assert b"Only optimize can propose implement or verify" in writing.STRICT_METHOD


@pytest.mark.parametrize("identifier", ["material:workflow_objective", "material/workflow_objective", "reviewer_only", "private_only", "not_granted"])
def test_writing_selection_rejects_other_namespaces_and_ungranted_ids(setup_writing, monkeypatch, identifier):
    setup = with_material_grants(setup_writing(alter=lambda draft: draft["selection"].append(
        {"id": identifier, "reason": "SYNTHETIC forbidden selection"})))
    result = run(setup, monkeypatch)
    assert result["status"] == "failed" and result["recovery"] == "new_request"
    assert result["schema"] is None and len(calls(setup[2])) == 1
    state = api.open_run("design", setup[4]).load_state("writing")
    retained = loads(Bundle(setup[4]).read(state["draft_ref"]))["content"]
    assert retained == json.loads((setup[2] / "scenario.json").read_text())["draft"]
    assert "selection must explain actual planner-granted material IDs" in state["stop_reason"]


@pytest.mark.parametrize("identifier", ["workflow_objective", "material/workflow_objective", "material:planner_only", "material:private_only"])
def test_writing_actions_require_exact_shared_catalog_ids(setup_writing, monkeypatch, identifier):
    def alter(draft):
        draft.update(next_action="implement", action_reason="SYNTHETIC action", action_inputs=["task/default_start"], action_evidence=[identifier])
    setup = with_material_grants(setup_writing("optimize", decision="keep", alter=alter))
    result = run(setup, monkeypatch)
    assert result["status"] == "failed" and len(calls(setup[2])) == 1
    assert "action may select only explicit IDs" in result["stop_reason"]


def test_optimize_can_use_shared_material_action_id(setup_writing, monkeypatch):
    def alter(draft):
        draft.update(next_action="implement", action_reason="SYNTHETIC granted objective supports implementation",
                     action_inputs=["task/default_start"], action_evidence=["material:workflow_objective"])
    setup = setup_writing("optimize", decision="keep", alter=alter)
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result
    assert result["next_action"] == "implement" and len(calls(setup[2])) == 2
    assert result["usage"]["calls"]["coder"] == 0 and result["usage"]["evaluator_jobs"] == 0


@pytest.mark.parametrize("case", ["scope_object", "design_action", "design_keep", "unknown_field"])
def test_explicit_draft_branch_contract_stays_strict(setup_writing, monkeypatch, case):
    def alter(draft):
        if case == "scope_object":
            draft["bindings"] = {"protocol": PROTOCOL, "kind": "scope", "coverage_ids": [], "bindings": draft["bindings"]}
        elif case == "design_action":
            draft.update(next_action="implement", action_reason="SYNTHETIC unsupported initial design action",
                         action_inputs=["task/default_start"], action_evidence=["material:workflow_objective"])
        elif case == "design_keep":
            draft.update(decision="keep", raw=[], dispatch=None, bindings=None)
        else:
            draft["extra"] = "SYNTHETIC unexpected field"
    setup = setup_writing(alter=alter)
    result = run(setup, monkeypatch)
    assert result["status"] == "failed" and result["schema"] is None and len(calls(setup[2])) == 1
    state = api.open_run("design", setup[4]).load_state("writing")
    assert loads(Bundle(setup[4]).read(state["draft_ref"]))["content"] == json.loads((setup[2] / "scenario.json").read_text())["draft"]


def test_writing_private_material_cannot_be_granted_by_a_catalog(setup_writing, monkeypatch):
    setup = with_material_grants(setup_writing())
    f, backend, fake, req, out = setup
    materials = Bundle(f["root"]).document(req["materials"])
    next(e for e in materials["entries"] if e["id"] == "private_only")["roles"] = ["planner"]
    p = Publication(f["root"])
    req = {**req, "materials": p.put_json("forbidden-grant.json", materials)}
    ref = p.put_json("forbidden-request.json", req)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    with pytest.raises(ProtocolError, match="private evaluator files"):
        run((f, backend, fake, req, out), monkeypatch)
    assert not calls(fake)


def test_b24_original_failed_draft_and_all_eight_text_errors_remain(tmp_path):
    """Historical real failure, read unchanged; no operation, owner or repair."""
    from types import SimpleNamespace
    from kernelschema import _writing_policy as policy
    retained = (Path(__file__).parent / "fixtures/writing/b24-failed-draft.json").read_bytes()
    assert sha256(retained) == "sha256:66f919efa8c676cd8f241cf82563c34c41c3d8e020f5c0c1b5c9f3e5596e2892"
    draft = loads(retained)["content"]
    # Exact audited B24 planner grants, not IDs recovered by stripping its draft.
    ids = ("workflow_objective task_contract task_acceptance task_workload task_timing task_environment "
           "triton_softmax_reference triton_license reference_context catalog_relevance "
           "doc_0 doc_1 doc_2 doc_3 doc_4 doc_5 doc_6 method_design method_optimize template_guide").split()
    context = SimpleNamespace(request={"operation": "design"}, objects={
        "materials": {"entries": [{"id": identifier, "roles": ["planner", "reviewer"], "file": {}} for identifier in ids]},
        "task": {"starter_source": None}})
    context.request["task"] = {}
    with pytest.raises(ProtocolError, match="selection must explain actual planner-granted material IDs"):
        policy.parse_draft(context, draft)
    assert len(draft["selection"]) == 19 and all(e["id"].startswith("material:") for e in draft["selection"])
    raw = draft["raw"][0]["text"].encode()
    assert sha256(raw) == "sha256:8a440621fdf5d799fa52024c711800b731dff7bcb6d09d81b2917b5e3a83a3ac"
    (tmp_path / "root.yaml").write_bytes(raw)
    engine = template_api.tools()
    document = engine.yaml_source.loads(raw.decode(), "root", path=tmp_path / "root.yaml")
    with pytest.raises(engine.SchemaError) as error:
        engine.validate_documents({"root": tmp_path / "root.yaml"}, {"root": document})
    message = str(error.value)
    assert message.count("text has ") == 8
    for actual, limit in ((720, 300), (469, 300), (522, 300), (318, 150), (165, 150), (183, 150), (51, 50), (69, 50)):
        assert f"text has {actual} Unicode characters" in message
        assert f"limits are {limit} characters" in message
    assert (tmp_path / "root.yaml").read_bytes() == raw
    assert loads(retained)["content"] == draft  # strict rejection did not reinterpret fields


def test_design_actual_api_freezes_independent_output_and_replays(setup_writing, monkeypatch):
    setup = setup_writing()
    f, backend, fake, req, out = setup
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result
    assert len(calls(fake)) == 2
    assert result["usage"]["tokens"] == 52
    bundle = Bundle(out)
    schema = bundle.document(result["schema"], "schema")
    assert schema["change"] == "new_task" and schema["design_revision"] == 0
    assert schema["parent"] is None
    review = bundle.document(result["writing_checks"]["semantics"]["receipt"], "writing_review")
    assert review["subject"]["schema_sha256"] == result["schema"]["sha256"]
    assert review["subject"]["request_sha256"] == digest(req)
    assert schema["change_report"] in review["evidence"]
    planner, reviewer = calls(fake)
    assert planner["fixed_assignment"]["role_authority"]["schema_proposal"]
    assert reviewer["fixed_assignment"]["role_authority"]["review_only"]
    assert not reviewer["fixed_assignment"]["role_authority"]["schema_proposal"]
    assert not any(item["id"].startswith("schema/") for item in planner["materials"])
    assert not any(item["id"] == "current_source" for item in planner["materials"])
    assert planner["fixed_assignment"]["workflow_objective"]
    from kernelschema.runtime import material_text
    assert "Give each row to one independent CTA" not in "\n".join(material_text(planner, x["id"]) for x in planner["materials"] if "host_implementation" not in x)
    for intent in api.open_run("design", out).intents():
        assert {"deadline_utc_ms", "deadline_monotonic_ns", "clock_boot_id"} <= intent["reservation"].keys()
    (out / "result.json").unlink()
    assert api.design(continue_run=out) == result
    assert len(calls(fake)) == 2
    export = out.parent / "export"
    import_bundle(bundle, export, [result["schema"], result["writing_checks"]["semantics"]["receipt"]])
    assert Bundle(export).document(result["schema"]) == schema


@pytest.mark.parametrize("decision", ["keep", "revise"])
def test_optimize_keep_revision_and_parent(setup_writing, monkeypatch, decision):
    setup = setup_writing("optimize", decision=decision)
    f, backend, fake, req, out = setup
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result
    assert result["decision"] == decision and result["next_action"] == "none"
    schema = Bundle(out).document(result["schema"])
    if decision == "keep":
        assert result["schema"] == req["schema"] and result["scope"] == req["scope"]
    else:
        assert schema["parent"] == req["schema"] and schema["change"] == "new_revision" and schema["design_revision"] == 1
        assert "qualified" not in result
    assert api.optimize(continue_run=out) == result
    assert len(calls(fake)) == 2


@pytest.mark.parametrize("operation", ["design", "optimize"])
@pytest.mark.parametrize("boundary", ["planner_returned", "planner_settled", "change_report_frozen", "schema_frozen", "reviewer_returned", "review_frozen", "result_ready"])
def test_first_result_loss_resumes_original_operations(setup_writing, monkeypatch, operation, boundary):
    tripped = []
    def crash(label, context):
        if label == boundary and not tripped:
            tripped.append(True)
            raise RuntimeError("synthetic first-result-loss")
    setup = setup_writing(operation, checkpoint=crash)
    f, backend, fake, req, out = setup
    with pytest.raises(RuntimeError, match="first-result-loss"):
        run(setup, monkeypatch)
    assert tripped and not (out / "result.json").exists()
    count_before = len(calls(fake))
    assert count_before >= 1
    fixed = api.open_run(operation, out)
    before = fixed.load_state("writing")
    original_candidate = before["schema"]
    # A fresh backend instance has no original in-memory terminal/draft state.
    monkeypatch.setattr(api, "_backend", lambda *_: writing.WritingBackend(operation))
    result = getattr(api, operation)(continue_run=out)
    assert result["status"] == "completed", result
    assert len(calls(fake)) == 2
    assert result["usage"]["calls"]["planner"] == result["usage"]["calls"]["reviewer"] == 1
    assert result["usage"]["tokens"] == 52
    if original_candidate:
        assert result["schema"] == original_candidate
    assert getattr(api, operation)(continue_run=out) == result
    assert len(calls(fake)) == 2


@pytest.mark.parametrize("case", ["contract", "coverage", "path", "duplicate_node", "unknown_variant", "invalid_yaml", "review_unknown", "review_fail", "review_subject"])
def test_permissions_and_unknowns_retain_original_candidate(setup_writing, monkeypatch, case):
    def alter(draft):
        if case == "contract":
            doc = yaml.safe_load(draft["raw"][0]["text"])
            doc["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"]["y"] = "Any finite output"
            draft["raw"][0]["text"] = yaml.safe_dump(doc, sort_keys=False, width=1000)
        elif case == "coverage":
            draft["bindings"][0]["coverage_ids"] = draft["bindings"][0]["coverage_ids"][:-1]
        elif case == "path":
            draft["raw"][0]["node"] = "../../host"
        elif case == "duplicate_node":
            draft["raw"].append(draft["raw"][0].copy())
        elif case == "unknown_variant":
            draft["bindings"][0]["variant"] = "unmade-design-choice"
        elif case == "invalid_yaml":
            draft["raw"][0]["text"] = "not a Schema"
    review = "unknown" if case == "review_unknown" else "fail" if case == "review_fail" else "pass"
    setup = setup_writing(alter=alter, review=review)
    f, backend, fake, req, out = setup
    if case == "review_subject":
        scenario = json.loads((fake / "scenario.json").read_text())
        scenario["bad_subject"] = True
        (fake / "scenario.json").write_text(json.dumps(scenario))
    result = run(setup, monkeypatch)
    assert result["status"] != "completed", result
    state = api.open_run("design", out).load_state("writing")
    assert Bundle(out).read(state["draft_ref"])
    assert result["usage"]["calls"]["planner"] == 1
    if case in ("path", "duplicate_node", "invalid_yaml"):
        assert result["schema"] is None and len(calls(fake)) == 1
    else:
        assert result["schema"] and len(calls(fake)) == 2
    if case == "contract":
        assert result["writing_checks"]["permissions"]["status"] == "fail"
    if case in ("review_unknown", "review_subject", "unknown_variant"):
        assert result["writing_checks"]["semantics"]["status"] == "unknown"
    snapshot = Bundle(out).read(state["draft_ref"])
    assert api.design(continue_run=out) == result
    assert Bundle(out).read(state["draft_ref"]) == snapshot


@pytest.mark.parametrize("action", ["implement", "verify"])
def test_keep_explicit_same_r_action(setup_writing, monkeypatch, action):
    setup = setup_writing("optimize", decision="keep")
    f, backend, fake, req, out = setup
    p = Publication(f["root"])
    evidence = p.put_bytes("settled-feedback.txt", b"SYNTHETIC explicit settled feedback: repair same-R code or verify original C; no performance claim.")
    req = {**req, "evidence": [f["starter"], evidence]}
    ref = p.put_json("request-with-action.json", req)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    scenario = json.loads((fake / "scenario.json").read_text())
    scenario["draft"].update(next_action=action, action_reason="SYNTHETIC explicit code/verification obligation for the same design", action_inputs=["evidence:0"], action_evidence=["evidence:1"])
    (fake / "scenario.json").write_text(json.dumps(scenario))
    result = run((f, backend, fake, req, out), monkeypatch)
    assert result["status"] == "completed", result
    assert result["decision"] == "keep" and result["schema"] == req["schema"]
    assert result["next_action"] == action
    assert result["action_inputs"] == [req["schema"], req["scope"], f["starter"]]
    assert result["action_evidence"] == [evidence]
    assert result["extensions"]["information"]["admitted_request"] is None
    assert result["usage"]["calls"]["coder"] == 0 and result["usage"]["evaluator_jobs"] == 0
    delivered = calls(fake)[0]["materials"]
    assert any(x["id"] == "delivery/evidence_0" and "class Model" in x["text"] for x in delivered)
    assert any(x["id"] == "delivery/evidence_1" and "SYNTHETIC explicit" in x["text"] for x in delivered)


@pytest.mark.parametrize("missing", ["objective", "reviewer_grant", "duplicate_objective"])
def test_objective_rejected_before_any_charge(setup_writing, monkeypatch, missing):
    setup = setup_writing()
    f, backend, fake, req, out = setup
    materials = Bundle(f["root"]).document(req["materials"])
    if missing == "objective":
        materials["entries"] = []
    elif missing == "reviewer_grant":
        materials["entries"][0]["roles"] = ["planner"]
    else:
        materials["entries"].append(materials["entries"][0].copy())
    p = Publication(f["root"])
    with pytest.raises(ProtocolError):
        req = {**req, "materials": p.put_json("bad-materials.json", materials)}
        r = p.put_json("bad-request.json", req)
        p.commit([r])
        durable_write(f["root"] / "request.json", jcs(req), replace=True)
        run((f, backend, fake, req, out), monkeypatch)
    assert not out.exists() and not calls(fake)


def test_objective_change_changes_q_not_task_or_scope(setup_writing):
    setup = setup_writing()
    f, backend, fake, req, out = setup
    p = Publication(f["root"])
    objective = p.put_bytes("objective.md", b"Prefer M=32 workloads inside unchanged full-Task criteria.\n")
    materials = Bundle(f["root"]).document(req["materials"])
    materials["entries"][0]["file"] = objective
    revised = {**req, "materials": p.put_json("materials.json", materials)}
    assert digest(revised) != digest(req)
    assert revised["task"] == req["task"] and revised["scope"] == req["scope"]


def test_precheck_has_no_published_schema_and_uses_original_task_scope(setup_writing, monkeypatch):
    def check(label, context):
        if label == "precheck_frozen":
            state = context.load_state("writing")
            assert state["schema"] is None and state["change_report"] is None
            assert state["facts"]["scope"]["status"] == "pass"
            assert len(state["facts"]["scope"]["mappings"]) == 4
    result = run(setup_writing(checkpoint=check), monkeypatch)
    assert result["status"] == "completed"


def test_unknown_original_operation_fences_then_settles_without_retry(setup_writing, monkeypatch):
    from kernelschema.runtime import CallReceipt
    setup = setup_writing()
    f, backend, fake, req, out = setup
    unresolved = [True]
    base_factory = backend.runtime_factory
    class LostAcknowledgment:
        def __init__(self, profile):
            self.actual = base_factory(profile)
        def __getattr__(self, name):
            return getattr(self.actual, name)
        def invoke(self, context, handle, *, authorize):
            receipt = self.actual.invoke(context, handle, authorize=authorize)
            if unresolved[0]:
                return CallReceipt({"terminated": False, "status": "in_flight", "raw": {}}, None)
            return receipt
        def settle(self, context, handle):
            if unresolved[0]:
                return CallReceipt({"terminated": False, "status": "in_flight", "raw": {}}, None)
            return self.actual.settle(context, handle)
    backend.runtime_factory = LostAcknowledgment
    result = run(setup, monkeypatch)
    assert (result["status"], result["recovery"]) == ("blocked", "settle_only")
    assert len(calls(fake)) == 1
    again = api.design(continue_run=out)
    assert again["usage"]["unsettled"] and len(calls(fake)) == 1
    unresolved[0] = False
    final = api.design(continue_run=out)
    assert final["status"] == "completed" and len(calls(fake)) == 2
    assert final["usage"]["tokens"] == 52 and not final["usage"]["unsettled"]


@pytest.mark.parametrize("role", ["planner", "reviewer"])
def test_native_failure_has_real_usage_without_success_report(setup_writing, monkeypatch, role):
    setup = setup_writing()
    f, backend, fake, req, out = setup
    scenario = json.loads((fake / "scenario.json").read_text())
    scenario["failure"] = role
    (fake / "scenario.json").write_text(json.dumps(scenario))
    result = run(setup, monkeypatch)
    assert result["status"] != "completed"
    assert result["usage"]["calls"][role] == 1
    assert result["usage"]["tokens"] is None
    assert not result["usage"]["unsettled"]
    if role == "planner":
        assert result["schema"] is None
        assert all(c["receipt"] is None for c in result["writing_checks"].values())
    else:
        assert result["schema"] and result["writing_checks"]["semantics"]["status"] == "unknown"
    assert api.design(continue_run=out) == result
    assert len(calls(fake)) == (1 if role == "planner" else 2)


def test_finish_floor_blocks_planner_without_review_budget(setup_writing, monkeypatch):
    setup = setup_writing()
    f, backend, fake, req, out = setup
    req["budget"]["limits"]["calls"]["reviewer"] = 0
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    result = run(setup, monkeypatch)
    assert result["status"] == "partial" and result["schema"] is None
    assert not calls(fake) and result["usage"]["calls"]["planner"] == 0
    assert api.design(continue_run=out) == result


def set_finish_floor(setup, value):
    f, _, _, req, _ = setup
    p = Publication(f["root"])
    profile = Bundle(f["root"]).document(req["backend"]["profile"])
    profile["parameters"]["writing_finish_floor"] = value
    req["backend"]["profile"] = p.put_json("finish-profile.json", profile)
    ref = p.put_json("finish-request.json", req); p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)


def next_fixed_floor():
    coder = cost(role="coder", calls=1, evaluator_jobs=1, active_ms=1000)
    reviewer = cost(role="reviewer", calls=1, active_ms=1000)
    return {key: coder[key] + reviewer[key] for key in coder}


def test_writing_finish_floor_short_budget_starts_zero_calls(setup_writing, monkeypatch):
    setup = setup_writing()
    set_finish_floor(setup, next_fixed_floor())
    result = run(setup, monkeypatch)
    assert result["status"] == "partial" and not calls(setup[2])
    assert not any(result["usage"]["calls"].values())
    assert api.design(continue_run=setup[4]) == result


@pytest.mark.parametrize("value", [{}, {**cost(), "coder": True}, {**cost(), "coder": 1}, {**cost(), "extra": 0}])
def test_writing_finish_floor_rejects_malformed_vector_before_calls(setup_writing, monkeypatch, value):
    setup = setup_writing()
    set_finish_floor(setup, value)
    with pytest.raises(ProtocolError, match="writing_finish_floor"):
        run(setup, monkeypatch)
    assert not calls(setup[2])


def test_writing_finish_floor_survives_reviewer_on_same_parent(setup_writing, monkeypatch, tmp_path):
    setup = setup_writing("optimize", decision="keep")
    f, _, fake, req, out = setup
    floor = next_fixed_floor()
    limits = {"calls": {"planner": 1, "reviewer": 2, "coder": 1, "extract": 0}, "tokens": None,
              "evaluator_jobs": 1, "active_ms": 180000, "model_calls": 4}
    owner = OwnerAccount.create(tmp_path / "handoff-owner.json", owner="synthetic-handoff-owner", limits=limits)
    p = Publication(f["root"]); parent = p.put_json("parent.json", owner.descriptor); p.commit([parent])
    req["budget"] = {"parent": parent, "limits": limits}
    set_finish_floor(setup, floor)
    with use_authority(owner.path):
        result = run(setup, monkeypatch)
        assert api.optimize(continue_run=out) == result
    assert result["status"] == "completed" and len(calls(fake)) == 2
    records = owner.operations(result["run_id"])
    reviewer = next(op for op in records.values() if op["role"] == "reviewer")
    planner = next(op for op in records.values() if op["role"] == "planner")
    assert reviewer["floor"] == floor
    minimum_review = cost(role="reviewer", calls=1, active_ms=1000)
    assert planner["floor"] == {key: floor[key] + minimum_review[key] for key in floor}
    assert result["usage"]["calls"] == {"planner": 1, "reviewer": 1, "coder": 0, "extract": 0}
    # The same owner can still admit the next fixed coder while reserving its
    # evaluator/reviewer. No holding operation was inserted into writing state.
    owner.register("next_fixed", "sha256:" + "c" * 64,
        {"calls": {"planner":0,"coder":1,"reviewer":1,"extract":0}, "tokens":None,"evaluator_jobs":1,"active_ms":60000},
        root=str(tmp_path / "next-fixed"))
    reservation = owner.reserve("next_fixed", "next_coder", role="coder", purpose="synthetic next fixed work",
        amount=cost(role="coder", calls=1), floor=cost(role="reviewer", calls=1, evaluator_jobs=1, active_ms=1000))
    assert reservation["account_id"] == owner.descriptor["id"]
    owner.release_unstarted("next_coder")
    assert len(owner.operations(result["run_id"])) == 2


def test_same_parent_across_new_q_never_resets_budget(setup_writing, monkeypatch, tmp_path):
    setup = setup_writing("optimize", decision="keep")
    f, backend, fake, req, out = setup
    limits = json.loads(json.dumps(req["budget"]["limits"]))
    limits["calls"]["planner"] = limits["calls"]["reviewer"] = 2
    limits["active_ms"] = 180000
    owner = OwnerAccount.create(tmp_path / "parent-authority.json", owner="original-parent", limits=limits)
    p = Publication(f["root"])
    req["budget"]["parent"] = p.put_json("parent.json", owner.descriptor)
    ref = p.put_json("parent-request.json", req)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    with use_authority(owner.path):
        first = run(setup, monkeypatch)
        # New Q from a new objective; unchanged Task/R and original parent.
        p = Publication(f["root"])
        materials = Bundle(f["root"]).document(req["materials"])
        materials["entries"][0]["file"] = p.put_bytes("new-goal.md", b"Prefer M=32 while preserving all Task criteria.\n")
        changed = {**req, "materials": p.put_json("materials.json", materials)}
        ref = p.put_json("new-request.json", changed)
        p.commit([ref])
        durable_write(f["root"] / "request.json", jcs(changed), replace=True)
        second = api.optimize(request=f["root"] / "request.json", out=tmp_path / "second")
        third = api.optimize(request=f["root"] / "request.json", out=tmp_path / "third")
    assert first["status"] == second["status"] == "completed"
    assert first["request_sha256"] != second["request_sha256"]
    assert first["schema"] == second["schema"] == req["schema"]
    assert third["status"] == "partial" and third["usage"]["calls"]["planner"] == 0
    assert len(calls(fake)) == 4
    assert first["usage"]["tokens"] == second["usage"]["tokens"] == 52


def test_actual_factory_and_cli_continue(setup_writing, tmp_path):
    import subprocess
    setup = setup_writing("optimize", decision="keep")
    f, backend, fake, req, out = setup
    result = api.optimize(request=f["root"] / "request.json", out=out)
    assert result["status"] == "completed"
    (out / "result.json").unlink()
    command = [sys.executable, "-B", "-m", "kernelschema", "optimize", "--continue", str(out)]
    complete = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert complete.returncode == 0, complete.stderr
    assert loads((out / "result.json").read_bytes()) == result
    assert len(calls(fake)) == 2


def test_same_design_representation_does_not_increment_revision(setup_writing, monkeypatch):
    setup = setup_writing("optimize", decision="keep")
    f, backend, fake, req, out = setup
    original = Bundle(f["root"]).document(req["schema"], "schema")
    scenario = json.loads((fake / "scenario.json").read_text())
    doc = template_api.effective_document(Bundle(f["root"]), original, "root")
    doc["META_DATA"]["Describe"] = "Same mathematical, schedule and pipeline decisions; updated description"
    scenario["draft"].update(decision="revise", raw=[{"node": "root", "text": yaml.safe_dump(doc, sort_keys=False, width=1000)}],
        dispatch=Bundle(f["root"]).read(original["dispatch"]).decode(), bindings=Bundle(f["root"]).document(req["scope"])["bindings"])
    (fake / "scenario.json").write_text(json.dumps(scenario))
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result
    assert result["decision"] == "keep"
    revised = Bundle(out).document(result["schema"])
    assert revised["change"] == "unchanged" and revised["design_revision"] == original["design_revision"]
    assert revised["parent"] == req["schema"] and revised["change_report"]


def test_decimal_schema_obligation_uses_exact_quote_not_json_float(setup_writing, monkeypatch):
    def alter(draft):
        doc = yaml.safe_load(draft["raw"][0]["text"])
        doc["DESIGN"]["Math_reshape"]["Rewrite"][0]["Symbol"] = {"scale": {"meaning": "Exact quoted test-only constant", "value": 0.5}}
        draft["raw"][0]["text"] = yaml.safe_dump(doc, sort_keys=False, width=1000)
    setup = setup_writing(alter=alter)
    result = run(setup, monkeypatch)
    assert result["status"] == "completed", result
    reviewer = calls(setup[2])[1]
    text = next(x["text"] for x in reviewer["materials"] if x["id"] == "delivery/review_subject")
    assert 'yaml_decimal' in text


def test_reviewer_reads_exact_keep_bytes_not_normalized_preview(setup_writing, monkeypatch):
    setup = setup_writing("optimize", decision="keep")
    f, backend, fake, req, out = setup
    result = run(setup, monkeypatch)
    assert result["status"] == "completed"
    review_input = json.loads(next(x["text"] for x in calls(fake)[1]["materials"] if x["id"] == "delivery/review_subject"))
    original = Bundle(out).document(req["schema"])
    assert review_input["raw"] == [{"node": r["node"], "text": Bundle(out).read(r["file"]).decode()} for r in original["raw"]]
    assert review_input["dispatch"] == Bundle(out).read(original["dispatch"]).decode()


def test_complete_public_task_environment_and_workload_delivered(setup_writing, monkeypatch):
    setup = setup_writing()
    result = run(setup, monkeypatch)
    assert result["status"] == "completed"
    f = setup[0]
    task = Bundle(f["root"]).document(f["task"])
    for delivery in calls(setup[2]):
        from kernelschema.runtime import material_text
        items = {x["id"]: material_text(delivery, x["id"]) for x in delivery["materials"] if "host_implementation" not in x}
        for key in ("environment", "workload"):
            assert items["delivery/task_" + key] == Bundle(f["root"]).read(task[key]).decode()
        assert not any("evaluator/golden.py" in x["id"] for x in delivery["materials"])


def test_newly_bound_task_never_invents_a_contract(setup_writing, monkeypatch):
    from kernelschema.contracts import UnsupportedError
    setup = setup_writing()
    f, backend, fake, req, out = setup
    p = Publication(f["root"])
    task = Bundle(f["root"]).document(req["task"])
    task["environment"] = p.put_bytes("environment.md", b"SYNTHETIC newly bound device profile; no actual device claim.\n")
    req = {**req, "task": p.put_json("task.json", task)}
    r = p.put_json("new-task-request.json", req)
    p.commit([r])
    durable_write(f["root"] / "request.json", jcs(req), replace=True)
    with pytest.raises(UnsupportedError, match="ks.row-softmax.target.v1"):
        run((f, backend, fake, req, out), monkeypatch)
    assert not calls(fake)


def test_feedback_locator_cannot_fabricate_measured_provenance(setup_writing, monkeypatch):
    def alter(draft):
        doc = yaml.safe_load(draft["raw"][0]["text"])
        doc["REALIZATION"]["Prefer"] = [{"Item": "DESIGN.Work_schedule::MAIN", "Choice": "Prefer four warps when measured", "Reason": "SYNTHETIC advisory choice"}]
        doc["REALIZATION"]["Feedback"] = [{"Item": "DESIGN.Work_schedule::MAIN", "Choice": "Four warps are preferred", "Evidence": "KernelSchema@2efc92479db497e0ecf681e276ecddc72cad0de6:examples/workflows/row_softmax/evaluator/golden.py"}]
        draft["raw"][0]["text"] = yaml.safe_dump(doc, sort_keys=False, width=1000)
    result = run(setup_writing(alter=alter), monkeypatch)
    assert result["schema"], result
    assert result["status"] == "partial"
    assert result["writing_checks"]["permissions"]["status"] == "unknown"


def test_resume_rejects_changed_private_candidate_before_review(setup_writing, monkeypatch):
    def crash(label, context):
        if label == "schema_frozen":
            raise RuntimeError("synthetic crash")
    setup = setup_writing(checkpoint=crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run(setup, monkeypatch)
    f, backend, fake, req, out = setup
    context = api.open_run("design", out)
    state = context.load_state("writing")
    state["raw"][0]["text"] += "# changed after freeze\n"
    context.save_state("writing", state)
    monkeypatch.setattr(api, "_backend", lambda *_: writing.WritingBackend("design"))
    with pytest.raises(ProtocolError, match="frozen normalized preview"):
        api.design(continue_run=out)
    assert len(calls(fake)) == 1
