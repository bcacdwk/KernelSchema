"""CPU interface tests. Every model/Agate receipt here is synthetic."""
from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys

import pytest

from kernelschema import api, fidelity
from kernelschema.artifacts import Bundle, FileBytes, Publication, durable_write, import_bundle
from kernelschema.contracts import PROTOCOL, ProtocolError, digest, jcs, sha256
from kernelschema.evaluation.fa4 import F1, REST, freeze_request, stage_candidate
from kernelschema.implement.reference import ReferenceBackend
from kernelschema.resources import resource
from kernelschema.runtime import RoleOutput, prepare_role_input, RoleDelivery, material_text
from examples.workflows.fa4_decode.prepare import prepare

spec = spec_from_file_location("fa4_synthetic_support", Path(__file__).parent / "fixtures/reference_support.py")
support = module_from_spec(spec)
spec.loader.exec_module(support)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("fa4-bridge") / "bundle"
    return root, prepare(root, checkout=resource("AGENTS.md").parent)


def typed(files):
    return {"format": "python_tree", "entrypoint": "solution.py", "files": [
        {"path": name, "mode": "0644", "content": data} for name, data in sorted(files.items())]}


def setup(prepared, tmp_path, monkeypatch, *, intent="original_verify", source=None, runtime=None, checkpoint=None, evaluation=None):
    root, ready = prepared
    runtime = runtime or support.FakeRuntime(reviews=["unknown"])
    evaluator = evaluation or support.FakeEvaluation(tmp_path / "eval", statuses=["unknown"])
    backend = ReferenceBackend(runtime_factory=lambda _: runtime, evaluation_factory=lambda *_: evaluator, checkpoint=checkpoint)
    p = Publication(root)
    profile = p.put_json("profile.json", {"protocol": PROTOCOL, "kind": "backend_profile", "name": "reference", "version": "synthetic-fa4",
        "runtime": "synthetic", "model": "synthetic", "parameters": {}, "native_limits": {},
        "environment": Bundle(root).document(ready["case"]["task"])["environment"], "capabilities": backend.capabilities({})})
    p.commit([profile])
    request = freeze_request(Publication(root), ready["case"], rules=ready["rules"], profile=profile,
        limits={"calls": {"planner": 0, "coder": 2, "reviewer": 2, "extract": 0}, "evaluator_jobs": 2, "active_ms": 120000, "tokens": None},
        intent=intent, source=source)
    path = tmp_path / "input.json"
    # API locates artifacts relative to the request's original bundle root.
    path = root / ("request-" + request["sha256"][7:] + ".json")
    durable_write(path, Bundle(root).read(request))
    monkeypatch.setattr("kernelschema.implement.reference.workflow_backend", lambda: backend)
    return backend, runtime, evaluator, path


def test_typed_output_uses_only_core_finite_source_codec():
    value = {"protocol": "ks.claude.output.v1", "role": "coder", "assignment_sha256": "sha256:" + "a" * 64,
             "status": "ok", "kind": "candidate_files", "content": typed({"solution.py": "# exact\n", "pkg/helper.py": "x=1\n"}), "reason": "synthetic"}
    parse = lambda v: RoleOutput.parse(v, role="coder", assignment_sha256=value["assignment_sha256"], source_format="python_tree")
    assert parse(value).candidate_files["pkg/helper.py"].data == b"x=1\n"
    assert parse(value).candidate_source is None
    bad = []
    for path in ("/tmp/evil.py", "../evil.py", "pkg/deep/evil.py", "evil.sh"):
        x = deepcopy(value); x["content"]["files"][0]["path"] = path; bad.append(x)
    x = deepcopy(value); x["content"]["files"][0]["mode"] = "0755"; bad.append(x)
    x = deepcopy(value); x["content"]["files"].append(x["content"]["files"][0]); bad.append(x)
    x = deepcopy(value); x["kind"] = "candidate_source"; x["content"] = "# truncated"; bad.append(x)
    x = deepcopy(value); x["content"]["entrypoint"] = "pkg/helper.py"; bad.append(x)
    x = deepcopy(value); x["content"]["files"][0]["content"] = "\ud800"; bad.append(x)
    for x in bad:
        with pytest.raises(ProtocolError):
            parse(x)
    with pytest.raises(ProtocolError):
        RoleOutput.parse(value, role="coder", assignment_sha256=value["assignment_sha256"])


def test_api_original_verify_current_whole_source_and_report_closure(prepared, tmp_path, monkeypatch):
    _, runtime, evaluator, request = setup(prepared, tmp_path, monkeypatch)
    result = api.implement(request=request, out=tmp_path / "run")
    assert result["status"] == "partial" and result["code"]["qualified"] is None
    assert runtime.calls == ["reviewer"] and len(evaluator.calls) == 1
    assert result["code"]["latest"] == prepared[1]["case"]["source"]
    source_items = [i for i in runtime.inputs[0]["materials"] if i["category"] == "source"]
    assert len(source_items) == 12 and len([i for i in source_items if i["path"].endswith(".py")]) == 6
    for item in source_items:
        assert item["mode"] == "0644" and sha256(material_text(runtime.inputs[0], item["id"]).encode()) == item["sha256"]
    a = result["assessments"][0]
    assert a["task_uncovered"] == [REST] and a["fidelity"]["status"] == "unknown"
    import_bundle(Bundle(tmp_path / "run"), tmp_path / "export", [a["fidelity"]["receipt"], result["code"]["latest"]])
    assert len(Bundle(tmp_path / "export").source_files(result["code"]["latest"])) == 12
    assert api.implement(continue_run=tmp_path / "run") == result


def test_api_rebuild_exact_next_c_and_complete_tree_deletion(prepared, tmp_path, monkeypatch):
    first = typed({"solution.py": "# synthetic C1\n", "old.py": "# delete me\n"})
    second = typed({"solution.py": "# synthetic C2\n"})
    runtime = support.FakeRuntime(candidates=[first, second], reviews=["unknown"])
    _, runtime, evaluator, request = setup(prepared, tmp_path, monkeypatch, intent="schema_rebuild", runtime=runtime)
    result = api.implement(request=request, out=tmp_path / "run")
    assert runtime.calls == ["coder", "reviewer", "coder", "reviewer"]
    assert not [i for i in runtime.inputs[0]["materials"] if i["category"] == "source"]
    assert "_NATIVE_LAUNCHER" in material_text(runtime.inputs[0], "material/candidate_interface")
    baseline_files = Bundle(prepared[0]).source_files(prepared[1]["case"]["source"])
    text = jcs(runtime.inputs[0]).decode()
    assert baseline_files["qoder_greenfield_decode_attention/operator.py"].data.decode() not in text
    second_coder = [i for i in runtime.inputs[2]["materials"] if i["category"] == "source"]
    assert {i["path"]: i["text"] for i in second_coder} == {i["path"]: i["content"] for i in first["files"]}
    assert set(Bundle(tmp_path / "run").source_files(result["code"]["latest"])) == {"solution.py"}
    stage = tmp_path / "run/.runtime/reference/stages" / evaluator.calls[-1]
    assert (stage / "source/solution.py").read_text() == "# synthetic C2\n"
    assert not (stage / "source/old.py").exists()
    assert result["usage"]["calls"]["coder"] == 2 and result["usage"]["evaluator_jobs"] == 2
    assert api.implement(continue_run=tmp_path / "run") == result


def test_explicit_seed_and_no_first_result_recovery(prepared, tmp_path, monkeypatch):
    p = Publication(prepared[0]); seed = p.put_source_files("seed", {"solution.py": FileBytes(b"# selected seed\n", 0o644)})
    p.commit([seed])
    runtime = support.FakeRuntime(candidates=[typed({"solution.py": "# next C\n"})], reviews=["unknown"])
    interrupted = []
    def checkpoint(label, context):
        if label == "coder_settled" and not interrupted:
            interrupted.append(True)
            raise RuntimeError("synthetic caller crash after original paid fact")
    backend, runtime, _, request = setup(prepared, tmp_path, monkeypatch, intent="seed_continuation", source=seed,
                                         runtime=runtime, checkpoint=checkpoint)
    with pytest.raises(RuntimeError, match="caller crash"):
        api.implement(request=request, out=tmp_path / "run")
    assert not (tmp_path / "run/result.json").exists()
    backend.checkpoint = lambda *args: None
    result = api.implement(continue_run=tmp_path / "run")
    assert [i for i in runtime.inputs[0]["materials"] if i["category"] == "source"][0]["text"] == "# selected seed\n"
    assert result["usage"]["calls"]["coder"] == 2  # recovered first call, one later authorized repair
    assert runtime.calls.count("coder") == 2


def test_rebuild_rejects_baseline_delivery_and_multifile_locations(prepared, tmp_path, monkeypatch):
    _, _, _, request = setup(prepared, tmp_path, monkeypatch, intent="schema_rebuild")
    context = api.prepare_run("implement", request=request, out=tmp_path / "run", backend_identity={"synthetic": True})
    baseline = context.objects["task"]["baseline_source"]
    assert not [i for i in prepare_role_input(context, "coder")["materials"] if i["category"] == "source"]
    with pytest.raises(ProtocolError, match="optimized baseline"):
        prepare_role_input(context, "coder", source=baseline)
    with pytest.raises(ProtocolError, match="optimized baseline"):
        prepare_role_input(context, "coder", deliveries=(RoleDelivery("hidden", baseline, ("coder",), kind="source"),))
    lines = {"solution.py": ["# entry"], "pkg/helper.py": ["x = 1", "return x"]}
    answer = {"source_locations": ["pkg/helper.py:L1-L2"], "evidence": [{"location": "pkg/helper.py:L1-L2", "quote": "x = 1\nreturn x"}]}
    assert fidelity._located(answer, lines)
    for location, quote in (("kernel.py:L1", "# entry"), ("solution.py:L1", "x = 1"), ("pkg/helper.py:L2-L3", "return x")):
        assert not fidelity._located({"source_locations": [location], "evidence": [{"location": location, "quote": quote}]}, lines)


def test_production_channel_missing_qualification_is_blocked(prepared, tmp_path, monkeypatch):
    from kernelschema.evaluation.fa4_channel import FA4Channel, selection
    remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/original/python", "bwrap": "/usr/bin/bwrap"}
    _, runtime, _, request = setup(prepared, tmp_path, monkeypatch)
    context = api.prepare_run("implement", request=request, out=tmp_path / "context", backend_identity={"synthetic": True})
    channel = FA4Channel({"parameters": {"evaluation": selection(remote=remote)}}, tmp_path / "eval")
    assert channel.admission(context)["status"] == "unknown"
    assert "qualification records" in channel.admission(context)["reason"]
    assert runtime.calls == []
    _, runtime, _, path = setup(prepared, tmp_path, monkeypatch, evaluation=channel)
    result = api.implement(request=path, out=tmp_path / "blocked-run")
    assert result["status"] == "blocked" and result["usage"]["evaluator_jobs"] == 0 and not any(result["usage"]["calls"].values())
    assert result["assessments"][0]["task_uncovered"] == [REST]


def test_fa4_method_locations_name_existing_selected_schema_sections(prepared, tmp_path, monkeypatch):
    _, _, _, request = setup(prepared, tmp_path, monkeypatch, intent="schema_rebuild")
    c = api.prepare_run("implement", request=request, out=tmp_path / "run", backend_identity={"synthetic": True})
    locations = [location for item in fidelity.obligations(c) for location in item["schema_locations"]]
    assert "q4_direct::DESIGN.Math_reshape.Rewrite[2]" in locations
    assert "q4_direct::META_DATA.Dispatch.when" in locations
    assert not any("F1_review" in location for location in locations)


def test_actual_k3_cli_tree_output_and_original_reopen(prepared, tmp_path):
    from kernelschema.runtime import ClaudeProfile, ClaudeRuntime
    root, ready = prepared
    fake = tmp_path / "claude"
    original = Path(__file__).parent / "fixtures/runtime/fake_claude.py"
    fake.write_text("#!" + sys.executable + "\n" + "\n".join(original.read_text().splitlines()[1:]) + "\n")
    fake.chmod(0o755)
    (tmp_path / "scenario.json").write_text('{"mode":"success"}')
    profile = ClaudeProfile.inspect(requested_model="synthetic-model", executable=str(fake), timeout_ms=3000)
    runtime = ClaudeRuntime(profile)
    p = Publication(root)
    config = profile.backend_profile(Bundle(root).document(ready["case"]["task"])["environment"], name="reference")
    profile_ref = p.put_json("runtime-profile.json", config); p.commit([profile_ref])
    request = freeze_request(Publication(root), ready["case"], rules=ready["rules"], profile=profile_ref,
        intent="seed_continuation", source=ready["case"]["source"],
        limits={"calls": {"planner":0,"coder":1,"reviewer":0,"extract":0}, "evaluator_jobs":0, "active_ms":60000,"tokens":None})
    path = root / "cli-tree-request.json"; durable_write(path, Bundle(root).read(request))
    context = api.prepare_run("implement", request=path, out=tmp_path / "run", backend_identity=runtime.identity())
    handle = runtime.admit(context, operation_id="synthetic_tree", role="coder", purpose="synthetic CPU tree delivery",
        source=ready["case"]["source"], reservation={"synthetic":True,"calls":1,"active_ms":3000})
    receipt = runtime.invoke(context, handle, authorize=lambda _: True)
    assert receipt.ok, receipt.facts
    assert len(receipt.output.candidate_files) == 12
    assert receipt.output.candidate_files["solution.py"].data.endswith(b"# synthetic CLI continuation\n")
    assert runtime.settle(context, runtime.reopen(context, handle.operation_id)).receipt == receipt.receipt
    assert len((tmp_path / "invocations.jsonl").read_text().splitlines()) == 1
