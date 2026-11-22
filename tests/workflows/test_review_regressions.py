"""Independent boundary regressions. All transports/receipts are synthetic."""
from copy import deepcopy
from dataclasses import replace
import shutil

import pytest

from kernelschema import api, fidelity
from kernelschema.artifacts import Bundle, FileBytes, Publication, durable_write
from kernelschema.budget import OwnerAccount
from kernelschema.contracts import ProtocolError, jcs, loads, sha256
from kernelschema.evaluation.fa4 import freeze_observation
from kernelschema.evaluation.fa4_channel import FA4Channel, selection
from kernelschema.materials import validate_request
from kernelschema.resources import resource
from kernelschema.runtime import RoleDelivery, material_text, prepare_role_input
from examples.workflows.fa4_decode.prepare import prepare
from test_fa4_bridge import setup as bridge_setup
from test_fidelity import context_for, publish
from test_standalone import synthetic  # same real driver with explicit fake transports


@pytest.fixture(scope="module")
def fa4_prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("review-fa4") / "bundle"
    return root, prepare(root, checkout=resource("AGENTS.md").parent)


def test_rebuild_rejects_renamed_and_rewrapped_baseline_code(fa4_prepared, tmp_path, monkeypatch):
    _, _, _, request = bridge_setup(fa4_prepared, tmp_path, monkeypatch, intent="schema_rebuild")
    c = api.prepare_run("implement", request=request, out=tmp_path / "run", backend_identity={"synthetic": True})
    baseline = c.bundle.source_files(c.objects["task"]["baseline_source"])
    body = baseline["qoder_greenfield_decode_attention/operator.py"]
    p = Publication(c.root)
    renamed = p.put_bytes("renamed-public.md", body.data)
    wrapped = p.put_source_files("renamed-source", {"solution.py": FileBytes(b"# synthetic entry\n", 0o644), "helper.py": body})
    empty_package = p.put_source_files("independent-empty-package", {"solution.py": FileBytes(b"# independent\n", 0o644), "pkg/__init__.py": FileBytes(b"", 0o644)})
    p.commit([renamed, wrapped, empty_package])
    for arguments in ({"deliveries": (RoleDelivery("original", renamed, ("coder",)),)}, {"source": wrapped}):
        with pytest.raises(ProtocolError, match="optimized baseline"):
            prepare_role_input(c, "coder", **arguments)
    p = Publication(c.root)
    materials = deepcopy(c.objects["materials"])
    materials["entries"].append({"id": "original_code", "file": renamed, "roles": ["coder"], "origin": "synthetic negative", "license": "none"})
    materials["entries"].sort(key=lambda e: e["id"])
    ref = p.put_json("bad-materials.json", materials); p.commit([ref])
    with pytest.raises(ProtocolError, match="optimized baseline"):
        validate_request(c.bundle, {**c.request, "materials": ref})
    assert material_text(prepare_role_input(c, "coder", source=empty_package), "current_source/pkg/__init__.py") == ""


@pytest.mark.parametrize("change", ["alias_text", "alias_size", "body_size", "body_host", "cycle", "missing"])
def test_alias_rejects_ambiguous_or_inconsistent_payload(change):
    payload = {"materials": [
        {"id": "body", "text": "abc", "sha256": sha256(b"abc"), "size": 3},
        {"id": "alias", "text_alias_of": "body", "sha256": sha256(b"abc"), "size": 3}]}
    body, alias = payload["materials"]
    if change == "alias_text": alias["text"] = "contradictory"
    elif change == "alias_size": alias["size"] = 99
    elif change == "body_size": body["size"] = 99
    elif change == "body_host": body["host_implementation"] = {"name": "synthetic"}
    elif change == "cycle": body["text_alias_of"] = "alias"
    else: alias["text_alias_of"] = "absent"
    with pytest.raises(ProtocolError): material_text(payload, "alias")


def test_fidelity_checks_every_supplied_quote():
    answer = {"source_locations": ["helper.py:L1"], "evidence": [
        {"location": "helper.py:L1", "quote": "x=1"}, {"location": "missing.py:L999", "quote": "invented"}]}
    assert not fidelity._located(answer, {"helper.py": ["x=1"]})


@pytest.mark.parametrize("change", ["unhashable_id", "reason_object", "extra_coverage"])
def test_malformed_review_retains_unknown_report(make_bundle, tmp_path, change):
    c, source, receipt, scope = context_for(make_bundle, tmp_path)
    content = deepcopy(receipt.output.content)
    if change == "unhashable_id": content["items"][0]["id"] = []
    elif change == "reason_object": content["items"][0]["reason"] = {"claim": "pass"}
    else: content["coverage"].append({**content["coverage"][0], "coverage_id": "nonexistent"})
    report = publish(c, source, replace(receipt, output=replace(receipt.output, content=content)), scope)
    assert report["fidelity" if change != "extra_coverage" else "coverage"]["status"] == "unknown"


class MalformedTerminal:
    def __init__(self): self.posts = 0
    def identity(self): return {"synthetic": True, "name": "independent malformed terminal fixture"}
    def request(self, method, path, body=None):
        if method == "POST":
            self.posts += 1
            return {"status": 200, "body": b'{"job_id":"synthetic-job"}'}
        return {"status": 200, "body": jcs({"job_id": "synthetic-job", "status": "succeeded",
            "finished_at": "2026-11-20T00:00:00Z", "cleanup_pending": False,
            "result": {"exit_code": 0, "stdout": "not a complete frame"}})}


@pytest.mark.parametrize("task", ["row", "fa4"])
@pytest.mark.parametrize("loss", ["binding_and_account", "operations", "operations_and_intents"])
def test_qualification_never_recreates_lost_original_account(fa4_prepared, tmp_path, task, loss):
    if task == "row":
        from test_agate_channel import context
        from kernelschema.evaluation.channel import qualify
        _, original, _, remote = context(tmp_path)
        root, request = original.root, deepcopy(original.request)
    else:
        from kernelschema.evaluation.fa4_channel import qualify
        root, ready = fa4_prepared
        request = Bundle(root).document(ready["request"])
        remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    request["budget"] = {"parent": None, "limits": {"calls": dict.fromkeys(("planner", "coder", "reviewer", "extract"), 0),
        "evaluator_jobs": 6, "active_ms": 60000, "tokens": None}}
    p = Publication(root); ref = p.put_json("lost-owner-request.json", request); p.commit([ref])
    path = root / ("lost-owner-" + loss + "-" + ref["sha256"][7:] + ".json"); durable_write(path, jcs(request))
    c = api.prepare_run("implement", request=path, out=tmp_path / "lost-owner-run", backend_identity={"synthetic": True})
    wire = MalformedTerminal()
    first = qualify(c, remote=remote, transport=wire)
    assert first["usage"]["evaluator_jobs"] == 1 and wire.posts == 1
    owner = OwnerAccount.for_context(c)
    binding = c.root / ".runtime/state/budget.json"
    retained_owner = owner.path.with_suffix(".retained")
    shutil.copy2(owner.path, retained_owner)
    before = retained_owner.read_bytes()
    if loss == "binding_and_account":
        owner.path.unlink()
        binding.rename(binding.with_suffix(".retained"))
    else:
        damaged = loads(before); damaged["operations"] = {}
        durable_write(owner.path, jcs(damaged), mode=0o600, replace=True)
        if loss == "operations_and_intents":
            (c.root / ".runtime/intents").rename(tmp_path / "retained-core-intents")
    with pytest.raises(ProtocolError, match="owner binding missing|owner operation"):
        qualify(api.open_run("implement", c.root), remote=remote, transport=wire)
    if loss == "binding_and_account":
        assert not owner.path.exists() and not binding.exists()
    assert retained_owner.read_bytes() == before and wire.posts == 1


def test_qualification_bootstrap_without_binding_refuses_reinitialization(fa4_prepared, tmp_path, monkeypatch):
    from kernelschema.evaluation.fa4_channel import qualify
    root, ready = fa4_prepared
    c = api.prepare_run("implement", request=root / "request.json", out=tmp_path / "bootstrap", backend_identity={"synthetic": True})
    original = OwnerAccount.for_context.__func__
    def interrupt(cls, *args, **kwargs):
        raise RuntimeError("synthetic interruption before original binding")
    monkeypatch.setattr(OwnerAccount, "for_context", classmethod(interrupt))
    remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    wire = MalformedTerminal()
    with pytest.raises(RuntimeError, match="before original binding"):
        qualify(c, remote=remote, transport=wire)
    monkeypatch.setattr(OwnerAccount, "for_context", classmethod(original))
    with pytest.raises(ProtocolError, match="owner binding missing"):
        qualify(api.open_run("implement", c.root), remote=remote, transport=wire)
    assert wire.posts == 0 and not (c.root / ".runtime/owner-account.json").exists()


def test_unqualified_reference_replay_reuses_empty_usage_projection(fa4_prepared, tmp_path, monkeypatch):
    remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    channel = FA4Channel({"parameters": {"evaluation": selection(remote=remote)}}, tmp_path / "channel")
    _, runtime, _, path = bridge_setup(fa4_prepared, tmp_path, monkeypatch, evaluation=channel)
    run = tmp_path / "run"
    result = api.implement(request=path, out=run)
    artifacts = sorted(str(p) for p in (run / "artifacts").rglob("*") if p.is_file())
    assert api.implement(continue_run=run) == result
    assert api.implement(continue_run=run) == result
    assert sorted(str(p) for p in (run / "artifacts").rglob("*") if p.is_file()) == artifacts
    assert result["status"] == "blocked" and result["usage"]["evaluator_jobs"] == 0 and runtime.calls == []


def test_pending_same_schema_request_does_not_inherit_old_qualification(tmp_path, synthetic):
    from examples.workflows.row_softmax.driver import Demo
    synthetic[0].decision, synthetic[0].action, synthetic[0].seed = "keep", "verify", "source"
    def interrupt(label, demo):
        if label == "new_schema_unadmitted": raise RuntimeError("synthetic pending verify")
    demo = Demo.create(tmp_path / "demo", start="fixed", checkpoint=interrupt)
    with pytest.raises(RuntimeError, match="pending verify"): demo.run()
    view = Demo(demo.root).summary()
    assert view["current_run"]["operation"] == "implement"
    assert view["current_qualified"] is None
    original = Bundle(demo.root).document(view["history"][0]["result"])
    assert original["code"]["qualified"] is not None  # still preserved as history


def test_standalone_observation_rejects_extra_stage_file(fa4_prepared, tmp_path):
    _, ready = fa4_prepared
    staged = deepcopy(ready["stage"])
    root = tmp_path / "stage"
    shutil.copytree(staged["root"], root)
    staged["root"] = str(root)
    (root / "source/extra.py").write_text("# unlisted\n")
    raw = jcs({"stage_sha256": staged["sha256"], "subject": {key: staged["manifest"][key]["sha256"]
        for key in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")}})
    with pytest.raises(ProtocolError, match="extra"):
        freeze_observation(Publication(tmp_path / "observation"), staged, raw=raw, operation_id="synthetic")
