"""CPU engineering only. Fake CLI is an actual process, all verdicts synthetic."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from kernelschema import api, template_api, writing
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle
from kernelschema.budget import OwnerAccount, use_authority
from kernelschema.contracts import PROTOCOL, ProtocolError, digest, jcs, loads
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
    def setup(operation="design", *, decision="revise", alter=None, review="pass", checkpoint=None, parent=None):
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
        rules = writing.freeze_rules(p)
        business = writing.backend_profile(profile, f["originals"]["task/environment.md"])
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
    assert "Give each row to one independent CTA" not in "\n".join(x["text"] for x in planner["materials"])
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
        items = {x["id"]: x["text"] for x in delivery["materials"]}
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
