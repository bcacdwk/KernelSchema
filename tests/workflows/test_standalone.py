"""Real driver/API/files/owner with explicit synthetic role/evaluator transports."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from examples.workflows.row_softmax.driver import Demo as ProductionDemo, DEFAULT_LIMITS, load_runtime
from kernelschema import api, writing, template_api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.budget import OwnerAccount, cost
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.evaluation.environment import target_environment, read_target, match_target
from kernelschema.implement.reference import ReferenceBackend
from kernelschema.runtime import CallReceipt, RoleOutput

spec = importlib.util.spec_from_file_location("standalone_synthetic", Path(__file__).parent / "fixtures/reference_support.py")
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)


class Demo(ProductionDemo):
    """These historical tests explicitly exercise the retained strict driver."""
    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.setdefault("writing_mode", "strict")
        return super().create(*args, **kwargs)


class SyntheticWriting(support.FakeRuntime):
    def __init__(self, *, action="implement", decision="revise", seed="default"):
        super().__init__()
        self.action, self.decision, self.seed = action, decision, seed

    def complete(self, context, handle):
        state = context.load_state("fake_" + handle)
        role = state["fixed"]["role"]
        if role == "planner":
            initial = context.request["operation"] == "design"
            decision = "revise" if initial else self.decision
            if initial:
                # Synthetic script owns R0; production design input never reads it.
                from kernelschema.resources import resource
                doc = yaml.safe_load(resource("examples/workflows/row_softmax/schema/root.yaml").read_bytes())
                doc["CONTRACT"] = writing.template_contract(context.bundle, context.request["task"])
            else:
                doc = template_api.effective_document(context.bundle, context.objects["schema"], "root")
            if not initial and decision == "revise":
                doc["DESIGN"]["Work_schedule"]["Step"]["MAIN"]["Schedule"]["Worker_assignment"]["Preassigned"]["count"] = 2
            action = "none" if initial else self.action
            content = {"format": "ks.writing.draft.v1", "decision": decision, "reason": "SYNTHETIC target-based route; no measurement claim",
                "selection": [{"id": "workflow_objective", "reason": "fixed bounded objective"}],
                "raw": [] if decision == "keep" else [{"node": "root", "text": yaml.safe_dump(doc, sort_keys=False, width=1000)}],
                "dispatch": None, "bindings": None if decision == "keep" else [{"coverage_ids": context.objects["scope"]["coverage_ids"], "node": "root", "variant": None}],
                "next_action": action, "action_reason": "SYNTHETIC explicit starter or selected Source work" if action != "none" else None,
                "action_inputs": ["task/default_start" if self.seed == "default" else "evidence:0"] if action != "none" else [],
                "action_evidence": ["material:workflow_objective"] if action != "none" else []}
        else:
            supplied = next(x for x in state["fixed"]["delivery"]["materials"] if x["id"] == "delivery/review_subject")
            review = loads(supplied["text"].encode())
            content = {"format": "ks.writing.review.v1", "subject": review["subject"], "items": [
                {"id": o["id"], "location": o["location"], "quote": o["quote"], "status": "pass", "reason": "SYNTHETIC CPU review only"} for o in review["obligations"]]}
        output = RoleOutput(role, state["fixed"]["delivery"]["assignment_sha256"], "ok", "writing_draft" if role == "planner" else "review", content, "SYNTHETIC")
        p = Publication(context.root, state["publication_id"])
        facts = {"runtime": "synthetic-reference-v1", "producer": "synthetic:" + context.run_id, "operation_id": handle,
            "receipt_id": handle + ":terminal", "role": role, "purpose": state["fixed"]["purpose"], "terminated": True,
            "status": "succeeded", "calls": 1, "tokens": 10, "raw": {}, "scope_id": handle + "_scope",
            "requested_model": "synthetic", "reported_models": ["synthetic"]}
        receipt = p.put_bytes("receipt.json", jcs({"facts": facts, "output": asdict(output), "synthetic": True})); p.commit([receipt])
        state["terminal"] = {"facts": facts, "output": asdict(output), "receipt": receipt}
        context.save_state("fake_" + handle, state)
        return CallReceipt(facts, output, receipt)


@pytest.fixture
def synthetic(monkeypatch):
    writer = SyntheticWriting()
    class SyntheticCoder(support.FakeRuntime):
        def complete(self, context, handle):
            self.candidates = [support.CODE + "\n# SYNTHETIC exact source for " + context.request["request_id"] + "\n"]
            return super().complete(context, handle)
    coder = SyntheticCoder()
    channels = {}
    def evaluation(profile, root):
        if str(root) not in channels: channels[str(root)] = support.FakeEvaluation(root)
        return channels[str(root)]
    def backend(name, operation):
        if operation == "implement": return ReferenceBackend(runtime_factory=lambda _: coder, evaluation_factory=evaluation)
        return writing.WritingBackend(operation, runtime_factory=lambda _: writer)
    monkeypatch.setattr(api, "_backend", backend)
    return writer, coder, channels


def test_driver_two_rounds_real_api_new_schema_default_seed_and_closure(tmp_path, synthetic):
    demo = Demo.create(tmp_path / "demo")
    result = demo.run()
    assert result["phase"] == "done" and result["optimize_count"] == 1, result
    assert [h["operation"] for h in result["history"]] == ["design", "implement", "optimize", "implement"]
    requests = [Bundle(demo.root).document(h["request"]) for h in result["history"]]
    assert len({q["budget"]["parent"]["sha256"] for q in requests}) == 1
    assert requests[3]["seed"] is None and requests[3]["schema"] != requests[1]["schema"]
    schema = Bundle(demo.root).document(requests[3]["schema"])
    assert schema["parent"] == requests[1]["schema"] and schema["design_revision"] == 1
    assert len(synthetic[0].calls) == 4 and len(synthetic[1].calls) == 4
    first_code = Bundle(demo.root).document(result["history"][1]["result"])["code"]["latest"]
    second_code = Bundle(demo.root).document(result["history"][3]["result"])["code"]["latest"]
    assert first_code["sha256"] != second_code["sha256"]
    assert Bundle(demo.root).source(first_code) != Bundle(demo.root).source(second_code)
    assert sum(len(c.calls) for c in synthetic[2].values()) == 2
    before = deepcopy(result)
    assert Demo(demo.root).run() == before
    exported = tmp_path / "portable"
    demo.export(exported)
    for item in result["history"]: Bundle(exported).closure([item["result"]], subjects=[item["request"]])
    assert not (exported / ".runtime/owner.json").exists()
    # Exact original per-request subjects survive; no second account or ledger.
    assert result["history"][1]["schema"] != result["schema"]


@pytest.mark.parametrize("boundary", ["before_api", "child_result_before_driver", "writing_published", "new_schema_unadmitted"])
def test_driver_finite_recovery_before_result_and_new_r_admission(tmp_path, synthetic, boundary):
    fired = []
    def crash(label, demo):
        if label == boundary and not fired:
            fired.append(label); raise RuntimeError("synthetic driver crash")
    demo = Demo.create(tmp_path / "demo", checkpoint=crash)
    with pytest.raises(RuntimeError, match="synthetic driver crash"): demo.run()
    if boundary == "child_result_before_driver":
        (demo.root / demo.state["current"]["out"] / "result.json").unlink()
    if boundary == "new_schema_unadmitted":
        view = Demo(demo.root).summary()
        assert view["current_qualified"] is None and len(view["history"]) == 3
    result = Demo(demo.root).run()
    assert result["phase"] == "done" and len(result["history"]) == 4
    assert len(synthetic[0].calls) == 4 and len(synthetic[1].calls) == 4


@pytest.mark.parametrize("start", ["fixed", "partial", "verify"])
def test_five_requests_and_explicit_initial_source(tmp_path, synthetic, start):
    synthetic[0].action, synthetic[0].decision = "none", "keep"
    demo = Demo.create(tmp_path / "demo", start=start)
    assert set(demo.config["requests"]) == {"design", "fixed", "partial", "verify", "optimize"}
    for ref in demo.config["requests"].values(): Bundle(demo.root).closure([ref])
    result = demo.run()
    assert result["phase"] == "done"
    first = synthetic[1].inputs[0]
    ref = demo.config["seed"] if start != "fixed" else Bundle(demo.root).document(demo.config["task"])["starter_source"]
    assert next(m["text"] for m in first["materials"] if m["id"] == "current_source") == Bundle(demo.root).source(ref).decode()
    assert (synthetic[1].calls.count("coder") == 0) == (start == "verify")


@pytest.mark.parametrize("action", ["implement", "verify"])
def test_same_r_explicit_old_source_action(tmp_path, synthetic, action):
    writer = synthetic[0]
    writer.decision, writer.action, writer.seed = "keep", action, "source"
    demo = Demo.create(tmp_path / "demo", start="fixed")
    result = demo.run()
    old = Bundle(demo.root).document(result["history"][0]["result"])
    q = Bundle(demo.root).document(result["history"][-1]["request"])
    assert q["seed"] == old["code"]["latest"]
    assert q["mode"] == ("verify_only" if action == "verify" else "code")
    assert q["schema"] == result["history"][0]["schema"]


def test_default_verify_rejected_not_silent_starter(tmp_path, synthetic):
    synthetic[0].action, synthetic[0].decision = "verify", "keep"
    result = Demo.create(tmp_path / "demo", start="fixed").run()
    assert result["phase"] == "done" and len(result["history"]) == 2
    assert result["history"][-1]["status"] != "completed"


def test_owner_budget_not_reset_at_new_r(tmp_path, synthetic):
    limits = deepcopy(DEFAULT_LIMITS); limits["model_calls"] = 6
    result = Demo.create(tmp_path / "demo", limits=limits).run()
    assert result["history"][-1]["operation"] == "implement"
    assert result["history"][-1]["status"] == "partial" and result["current_qualified"] is None
    assert len(synthetic[0].calls) + len(synthetic[1].calls) == 6


@pytest.mark.parametrize("review", ["fail", "unknown"])
def test_incomplete_fixed_work_does_not_admit_optimize(tmp_path, synthetic, review):
    synthetic[1].reviews = [review]
    demo = Demo.create(tmp_path / "demo", start="fixed")
    result = demo.run()
    assert result["status"] == "partial" and result["phase"] == "done"
    assert result["optimize_count"] == 0 and not synthetic[0].calls
    assert len(result["history"]) == 1 and result["current_qualified"] is None
    original_calls = list(synthetic[1].calls)
    assert Demo(demo.root).run() == result
    assert synthetic[1].calls == original_calls


def test_original_unknown_coder_no_replacement(tmp_path, synthetic):
    synthetic[1].unknown = True
    demo = Demo.create(tmp_path / "demo", start="fixed")
    result = demo.run()
    assert result["recovery"] == "settle_only" and len(synthetic[1].calls) == 1
    assert Demo(demo.root).run()["recovery"] == "settle_only" and len(synthetic[1].calls) == 1
    out = demo.root / result["current_run"]["out"]
    context = api.open_run("implement", out)
    synthetic[1].complete(context, context.intents()[0]["operation_id"])
    synthetic[1].unknown = False
    assert Demo(demo.root).run()["phase"] == "done"
    assert synthetic[1].calls.count("coder") == 2  # original plus new R, no unknown retry


def synthetic_target():
    locator = "fixture@" + "1" * 40 + ":target.md"
    text = "SYNTHETIC source document for CPU adapter checks; no real device claim."
    return target_environment(hardware_specification={"Device": "Synthetic GPU", "Architecture": "Synthetic SM", "Execution": {"cta": "synthetic CTA", "warp": "synthetic warp"},
        "Feature": {"register": "synthetic register"}, "Stack": {"Language": "Synthetic Triton", "Toolchain": "Synthetic compiler"},
        "Wiki": {"Hardware": "<" + locator + ">", "Programming": {"Language": "<" + locator + ">", "Virtual_ISA": "<" + locator + ">", "Machine_ISA": "<" + locator + ">"},
                 "Microbench": "<" + locator + ">", "Sota": {"Kernel": "<" + locator + ">", "Schema": "<" + locator + ">"}, "Profile": "<" + locator + ">"}},
        runtime={"name": "Synthetic GPU", "architecture": [9, 0], "torch": "synthetic", "triton": "synthetic", "cuda_build": "synthetic"},
        sources=[{"locator": locator, "sha256": sha256(text.encode()), "text": text, "license": "test-only"}])


def test_typed_target_contract_locks_public_task_and_source_bytes(tmp_path, synthetic):
    target = synthetic_target()
    demo = Demo.create(tmp_path / "demo", environment=jcs(target))
    contract = writing.template_contract(Bundle(demo.root), demo.config["task"])
    assert contract["Hardware_specification"] == target["hardware_specification"]
    assert demo.run()["history"][0]["status"] == "completed"
    with pytest.raises(UnsupportedError, match="differs"): match_target(jcs(target), {"mode": "cuda"})
    target["sources"][0]["text"] += "changed"
    with pytest.raises(ProtocolError, match="bytes/hash"): read_target(jcs(target))


def existing_parent(tmp_path, *, limits=None):
    """Independent synthetic diagnostics; never opens a real validation owner."""
    owner = OwnerAccount.create(tmp_path / "private/original-owner.json", owner="synthetic-original-owner",
                                limits=deepcopy(limits if limits is not None else DEFAULT_LIMITS))
    owner.register("synthetic-diagnostic", sha256(b"synthetic diagnostic"), owner.descriptor["limits"],
                   root=str(tmp_path / "diagnostic"))
    for role in ("coder", "evaluator"):
        operation = "synthetic-prior-" + role
        reservation = owner.reserve("synthetic-diagnostic", operation, role=role, purpose="synthetic prior diagnostic",
            amount=cost(role=role, calls=int(role == "coder"), evaluator_jobs=int(role == "evaluator"), active_ms=1))
        assert owner.authorize(reservation)
        owner.absorb("synthetic-diagnostic", {"producer": "synthetic:diagnostic", "operation_id": operation,
            "role": role, "purpose": "synthetic prior diagnostic", "receipt_id": operation + ":terminal", "sequence": 1,
            "receipt": None, "state": "settled", "calls": int(role == "coder"), "evaluator_jobs": int(role == "evaluator"),
            "tokens": 0, "gpu_ms": None}, terminated=True)
    publication = Publication(tmp_path / "public-parent")
    parent = publication.put_json("original-parent.json", owner.descriptor)
    publication.commit([parent])
    return Bundle(publication.root), owner, parent


def forbid_new_owner(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("existing-parent Demo must never create an owner")
    monkeypatch.setattr(OwnerAccount, "create", forbidden)


def test_existing_parent_qualification_design_actions_recovery_and_export(tmp_path, synthetic, monkeypatch):
    from test_agate_channel import Wire, PROFILE
    limits = deepcopy(DEFAULT_LIMITS); limits["evaluator_jobs"] = 10
    bundle, owner, parent = existing_parent(tmp_path, limits=limits)
    original_operations = owner.operations()
    original_bytes = owner.path.read_bytes()
    forbid_new_owner(monkeypatch)
    target = synthetic_target()
    remote = {"channel": PROFILE, "gpu": "synthetic-gpu", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    stopped = []
    def crash(label, demo):
        if label == "writing_published" and not stopped:
            stopped.append(label); raise RuntimeError("synthetic external-owner crash")
    demo = Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle, authority=owner.path,
        limits=limits, environment=jcs(target), remote=remote, checkpoint=crash)
    assert owner.path.read_bytes() == original_bytes
    assert demo.owner.path == owner.path and demo.config["parent"] == parent
    assert not (demo.root / ".runtime/owner.json").exists()
    assert (demo.root / ".runtime/demo-anchor.json").stat().st_mode & 0o777 == 0o600
    assert Bundle(demo.root).read(parent) == bundle.read(parent)
    assert set(demo.config["requests"]) == {"design", "fixed", "partial", "verify", "optimize"}
    for ref in demo.config["requests"].values():
        assert Bundle(demo.root).document(ref)["budget"]["parent"] == parent
    # Imported public bytes suffice; only the private original authority remains live.
    bundle.root.rename(tmp_path / "archived-public-parent")
    wire = Wire(target)
    assert demo.prepare_agate(transport=wire)["status"] == "completed"
    with pytest.raises(RuntimeError, match="synthetic external-owner crash"): demo.run()
    reopened = Demo(demo.root)
    assert reopened.owner.path == owner.path
    result = reopened.run()
    assert [item["operation"] for item in result["history"]] == ["design", "implement", "optimize", "implement"]
    assert result["phase"] == "done" and result["optimize_count"] == 1
    assert result["owner"] == owner.descriptor and len(wire.posts) == 6
    for entry in reopened.state["history"]:
        context = api.open_run(Bundle(reopened.root).document(entry["result"])["operation"], reopened.root / entry["out"])
        assert context.request["budget"]["parent"] == parent
        assert context.load_state("budget")["authority"] == str(owner.path)
    qualified = api.open_run("implement", reopened.root / "qualification")
    assert qualified.request["budget"]["parent"] == parent
    assert qualified.load_state("budget")["authority"] == str(owner.path)
    requests = [Bundle(reopened.root).document(item["request"]) for item in result["history"]]
    assert requests[1]["seed"] is None and requests[3]["seed"] is None
    assert requests[1]["schema"] != requests[3]["schema"]
    operations = owner.operations()
    assert all(operations[key] == value for key, value in original_operations.items())
    assert sum(op["lower"]["calls"] for op in operations.values()) == 9
    assert sum(op["lower"]["evaluator_jobs"] for op in operations.values()) == 9
    calls = (len(synthetic[0].calls), len(synthetic[1].calls))
    assert Demo(demo.root).run() == result and calls == (4, 4)
    assert (len(synthetic[0].calls), len(synthetic[1].calls)) == calls
    exported = tmp_path / "exported"
    reopened.export(exported)
    assert Bundle(exported).read(parent) == Bundle(reopened.root).read(parent)
    assert not (exported / ".runtime").exists()
    for path in exported.rglob("*"):
        if path.is_file():
            assert str(owner.path).encode() not in path.read_bytes()
            assert original_bytes not in path.read_bytes()


@pytest.mark.parametrize("blocked_by", ["pause", "fence", "model_calls"])
def test_existing_parent_preserves_prior_usage_pause_fence_and_tight_budget(tmp_path, synthetic, monkeypatch, blocked_by):
    limits = deepcopy(DEFAULT_LIMITS)
    if blocked_by == "model_calls": limits["model_calls"] = 2
    bundle, owner, parent = existing_parent(tmp_path, limits=limits)
    if blocked_by == "pause": owner.pause()
    if blocked_by == "fence":
        reservation = owner.reserve("synthetic-diagnostic", "synthetic-unknown", role="evaluator", purpose="synthetic unknown",
                                    amount=cost(evaluator_jobs=1, active_ms=1))
        assert owner.authorize(reservation)
        owner.mark_unknown("synthetic-unknown", "SYNTHETIC unresolved original operation")
    before = owner.path.read_bytes()
    original = loads(before)
    forbid_new_owner(monkeypatch)
    demo = Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle.root, authority=owner.path)
    assert owner.path.read_bytes() == before
    result = Demo(demo.root).run()
    assert not synthetic[0].calls and not synthetic[1].calls
    assert result["status"] in ("blocked", "partial")
    after = loads(owner.path.read_bytes())
    assert after["descriptor"] == original["descriptor"] and after["operations"] == original["operations"]
    assert after["paused"] == original["paused"] and after["active_ms"] >= original["active_ms"]
    assert not (demo.root / ".runtime/owner.json").exists()
    if blocked_by == "fence":
        assert result["recovery"] == "settle_only" and result["fences"] == ["synthetic-unknown"]
        assert set(after["requests"]) == set(original["requests"])
        for key in ("request_sha256", "limits", "root"):
            assert after["requests"]["synthetic-diagnostic"][key] == original["requests"]["synthetic-diagnostic"][key]
        assert after["requests"]["synthetic-diagnostic"]["active_ms"] >= original["requests"]["synthetic-diagnostic"]["active_ms"]


@pytest.mark.parametrize("change", ["parent", "parent_bundle", "authority", "limits", "wrong_owner", "wrong_ref"])
def test_existing_parent_inputs_rejected_before_output_or_owner_changes(tmp_path, monkeypatch, change):
    bundle, owner, parent = existing_parent(tmp_path)
    other = OwnerAccount.create(tmp_path / "other/owner.json", owner="different-owner", limits=deepcopy(DEFAULT_LIMITS))
    values = {"parent": parent, "parent_bundle": bundle, "authority": owner.path}
    if change in ("parent", "parent_bundle", "authority"):
        del values[change]
        match = "requires parent, parent_bundle and authority together"
    elif change == "limits":
        values["limits"] = {**DEFAULT_LIMITS, "model_calls": DEFAULT_LIMITS["model_calls"] + 1}
        match = "limits conflict"
    elif change == "wrong_owner":
        values["authority"] = other.path
        match = "descriptor differs"
    else:
        values["parent"] = {**parent, "sha256": sha256(b"different parent bytes")}
        match = "Ref digest mismatch"
    before = owner.path.read_bytes(), other.path.read_bytes()
    forbid_new_owner(monkeypatch)
    with pytest.raises(ProtocolError, match=match): Demo.create(tmp_path / "demo", **values)
    assert not (tmp_path / "demo").exists()
    assert (owner.path.read_bytes(), other.path.read_bytes()) == before


@pytest.mark.parametrize("change", ["missing_locator", "relative_locator", "missing_authority", "wrong_owner"])
def test_existing_parent_reopen_refuses_missing_or_changed_authority(tmp_path, monkeypatch, change):
    bundle, owner, parent = existing_parent(tmp_path)
    other = OwnerAccount.create(tmp_path / "other/owner.json", owner="different-owner", limits=deepcopy(DEFAULT_LIMITS))
    demo = Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle, authority=owner.path)
    anchor = deepcopy(demo.anchor)
    if change == "missing_locator": del anchor["authority"]
    elif change == "relative_locator": anchor["authority"] = ".runtime/owner.json"
    elif change == "missing_authority": owner.path.unlink()
    else: anchor["authority"] = str(other.path)
    durable_write(demo.root / ".runtime/demo-anchor.json", jcs(anchor), mode=0o600, replace=True)
    forbid_new_owner(monkeypatch)
    with pytest.raises(ProtocolError): Demo(demo.root)
    assert not (demo.root / ".runtime/owner.json").exists()
    if change == "missing_authority": assert not owner.path.exists()


def test_existing_parent_cli_prepare_inspect_without_private_export(tmp_path, monkeypatch, capsys):
    from examples.workflows.row_softmax.run_demo import main
    bundle, owner, parent = existing_parent(tmp_path)
    ref_file = tmp_path / "parent-ref.json"
    durable_write(ref_file, jcs(parent))
    before = loads(owner.path.read_bytes())
    forbid_new_owner(monkeypatch)
    args = ["--parent-ref", str(ref_file), "--parent-bundle", str(bundle.root), "--authority", str(owner.path)]
    assert main(["prepare", "--out", str(tmp_path / "demo"), *args]) == 0
    output = capsys.readouterr()
    assert loads(output.out.encode())["owner"] == owner.descriptor
    assert str(owner.path) not in output.out + output.err
    demo = Demo(tmp_path / "demo")
    assert demo.owner.path == owner.path and demo.config["parent"] == parent
    assert main(["inspect", "--demo", str(demo.root)]) == 0
    assert str(owner.path) not in capsys.readouterr().out
    assert main(["export", "--demo", str(demo.root), "--out", str(tmp_path / "exported")]) == 0
    assert str(owner.path) not in capsys.readouterr().out
    assert Bundle(tmp_path / "exported").read(parent) == bundle.read(parent)
    assert not (tmp_path / "exported/.runtime").exists()
    after = loads(owner.path.read_bytes())
    for key in ("descriptor", "operations", "requests", "paused", "active_ms"):
        assert after[key] == before[key]
    assert main(["prepare", "--out", str(tmp_path / "incomplete"), "--parent-ref", str(ref_file)]) == 2
    assert "requires parent, parent_bundle and authority together" in capsys.readouterr().err
    assert not (tmp_path / "incomplete").exists()


def explicit_budgets():
    writing_limits = {"calls": {"planner": 1, "coder": 1, "reviewer": 2, "extract": 0},
                      "model_calls": 4, "evaluator_jobs": 1, "active_ms": 2500000, "tokens": None}
    implement_limits = {"calls": {"planner": 0, "coder": 1, "reviewer": 1, "extract": 0},
                        "model_calls": 2, "evaluator_jobs": 1, "active_ms": 1250000, "tokens": None}
    return {"request_limits": {"design": writing_limits, "optimize": {**deepcopy(writing_limits), "active_ms": 2600000},
                "implement": implement_limits, "verify": {**deepcopy(implement_limits),
                    "calls": {"planner": 0, "coder": 0, "reviewer": 1, "extract": 0}, "model_calls": 1, "active_ms": 650000}},
            "writing_minimum_active_ms": {"planner": 600000, "reviewer": 600000},
            "reference_minimum_active_ms": {"coder": 600000, "evaluator": 1000, "reviewer": 600000},
            "writing_finish_floor": {**cost(role="coder", calls=1, evaluator_jobs=1, active_ms=1201000),
                                      "reviewer": 1, "model_calls": 2}}


@pytest.mark.parametrize("action", ["implement", "verify"])
def test_explicit_budgets_frozen_across_same_parent_full_driver_and_recovery(tmp_path, synthetic, monkeypatch, action):
    from test_agate_channel import Wire, PROFILE
    limits = deepcopy(DEFAULT_LIMITS); limits["evaluator_jobs"] = 10
    bundle, owner, parent = existing_parent(tmp_path, limits=limits)
    original_operations, original_bytes = owner.operations(), owner.path.read_bytes()
    forbid_new_owner(monkeypatch)
    synthetic[0].action = action
    if action == "verify": synthetic[0].decision, synthetic[0].seed = "keep", "source"
    remote = {"channel": PROFILE, "gpu": "synthetic-gpu", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    options = explicit_budgets(); expected = deepcopy(options)
    target = synthetic_target()
    demo = Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle, authority=owner.path,
                       environment=jcs(target), remote=remote, **options)
    assert owner.path.read_bytes() == original_bytes
    options["request_limits"]["design"]["active_ms"] = 1
    options["writing_minimum_active_ms"]["planner"] = 1
    options["reference_minimum_active_ms"]["reviewer"] = 1
    options["writing_finish_floor"]["active_ms"] = 1
    assert demo.config["request_limits"] == expected["request_limits"]
    for name, ref in demo.config["requests"].items():
        q = Bundle(demo.root).document(ref)
        limit_name = "implement" if name in ("fixed", "partial") else name
        assert q["budget"]["limits"] == expected["request_limits"][limit_name]
    assert demo.prepare_agate(transport=Wire(target))["status"] == "completed"
    stopped = []
    def stop(label, current):
        if label == "writing_published" and not stopped:
            stopped.append(label); raise RuntimeError("synthetic budget recovery")
    demo.checkpoint = stop
    with pytest.raises(RuntimeError, match="synthetic budget recovery"): demo.run()
    reopened = Demo(demo.root)
    result = reopened.run()
    assert result["phase"] == "done" and [h["operation"] for h in result["history"]] == ["design", "implement", "optimize", "implement"]
    assert result["history"][-1]["status"] == "completed"
    for entry in result["history"]:
        q = Bundle(demo.root).document(entry["request"])
        limit_name = "verify" if q.get("mode") == "verify_only" else q["operation"]
        assert q["budget"] == {"parent": parent, "limits": expected["request_limits"][limit_name]}
        parameters = Bundle(demo.root).document(q["backend"]["profile"])["parameters"]
        if q["operation"] == "implement":
            assert parameters["minimum_active_ms"] == expected["reference_minimum_active_ms"]
            assert parameters["evaluation"]["parameters"]["qualification"]
        else:
            assert parameters["writing_minimum_active_ms"] == expected["writing_minimum_active_ms"]
            assert parameters["writing_finish_floor"] == expected["writing_finish_floor"]
    operations = owner.operations()
    assert all(operations[key] == value for key, value in original_operations.items())
    assert sum(op["lower"]["calls"] for op in operations.values()) == (9 if action == "implement" else 8)
    assert not (demo.root / ".runtime/owner.json").exists()
    assert Demo(demo.root).run() == result


@pytest.mark.parametrize("invalid", ["missing_stage", "unknown_stage", "bad_limit", "unknown_minimum", "bool_minimum", "zero_minimum", "huge_minimum", "incomplete_floor", "inconsistent_floor", "negative_floor"])
def test_explicit_budgets_invalid_before_output_or_owner_change(tmp_path, monkeypatch, invalid):
    bundle, owner, parent = existing_parent(tmp_path)
    before = owner.path.read_bytes()
    forbid_new_owner(monkeypatch)
    options = explicit_budgets()
    if invalid == "missing_stage": del options["request_limits"]["verify"]
    elif invalid == "unknown_stage": options["request_limits"]["followup"] = options["request_limits"]["implement"]
    elif invalid == "bad_limit": options["request_limits"]["design"]["active_ms"] = True
    elif invalid == "unknown_minimum": options["writing_minimum_active_ms"]["coder"] = 600000
    elif invalid == "bool_minimum": options["reference_minimum_active_ms"]["reviewer"] = True
    elif invalid == "zero_minimum": options["writing_minimum_active_ms"]["planner"] = 0
    elif invalid == "huge_minimum": options["writing_minimum_active_ms"]["planner"] = 2 ** 64
    elif invalid == "incomplete_floor": del options["writing_finish_floor"]["tokens"]
    elif invalid == "inconsistent_floor": options["writing_finish_floor"]["model_calls"] = 1
    else: options["writing_finish_floor"]["active_ms"] = -1
    with pytest.raises(ProtocolError):
        Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle, authority=owner.path, **options)
    assert not (tmp_path / "demo").exists() and owner.path.read_bytes() == before


@pytest.mark.parametrize("tight", ["request", "parent"])
def test_explicit_writing_minima_and_finish_floor_refuse_insufficient_budget_without_calls(tmp_path, synthetic, tight):
    options = explicit_budgets()
    parent_limits = deepcopy(DEFAULT_LIMITS)
    if tight == "request": options["request_limits"]["design"]["active_ms"] = 2400000
    else: parent_limits["active_ms"] = 2400000
    bundle, owner, parent = existing_parent(tmp_path, limits=parent_limits)
    original = owner.operations()
    result = Demo.create(tmp_path / "demo", parent=parent, parent_bundle=bundle, authority=owner.path, **options).run()
    assert result["status"] == "partial" and result["phase"] == "done"
    assert not synthetic[0].calls and not synthetic[1].calls
    assert owner.operations() == original


def test_explicit_budgets_cli_freezes_public_inputs(tmp_path, monkeypatch, capsys):
    from examples.workflows.row_softmax.run_demo import main
    bundle, owner, parent = existing_parent(tmp_path)
    before = loads(owner.path.read_bytes())
    forbid_new_owner(monkeypatch)
    ref_file = tmp_path / "parent.json"; durable_write(ref_file, jcs(parent))
    args = ["prepare", "--out", str(tmp_path / "demo"), "--parent-ref", str(ref_file),
            "--parent-bundle", str(bundle.root), "--authority", str(owner.path)]
    options = explicit_budgets()
    for name, value in options.items():
        path = tmp_path / (name + ".json"); durable_write(path, jcs(value))
        args.extend(["--" + name.replace("_", "-"), str(path)])
    assert main(args) == 0
    assert str(owner.path) not in capsys.readouterr().out
    demo = Demo(tmp_path / "demo")
    assert demo.config["request_limits"] == options["request_limits"]
    assert Bundle(demo.root).document(demo.config["writing_profile"])["parameters"]["writing_finish_floor"] == options["writing_finish_floor"]
    assert Bundle(demo.root).document(demo.config["reference_profile"])["parameters"]["minimum_active_ms"] == options["reference_minimum_active_ms"]
    after = loads(owner.path.read_bytes())
    for key in ("descriptor", "operations", "requests", "paused", "active_ms", "active_ns"):
        assert after[key] == before[key]
