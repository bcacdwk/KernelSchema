"""Real driver/API/files/owner with explicit synthetic role/evaluator transports."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from examples.workflows.row_softmax.driver import Demo, DEFAULT_LIMITS, load_runtime
from kernelschema import api, writing, template_api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.evaluation.environment import target_environment, read_target, match_target
from kernelschema.implement.reference import ReferenceBackend
from kernelschema.runtime import CallReceipt, RoleOutput

spec = importlib.util.spec_from_file_location("standalone_synthetic", Path(__file__).parent / "fixtures/reference_support.py")
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)


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
