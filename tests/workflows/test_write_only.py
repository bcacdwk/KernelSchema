"""Finite writing behavior; all model/native outcomes in this file are synthetic."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelschema import api, cli, writing
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle
from kernelschema.contracts import jcs, loads
from kernelschema.runtime import material_text
from examples.workflows.row_softmax.driver import Demo
from test_writing import setup_writing, run, calls
from test_standalone import synthetic


TEXT = "META_DATA:\n  name: synthetic\nDESIGN:\n  decision: one worker per row\n"


def proposal(text=TEXT):
    return {"raw": [{"node": "root", "text": text}], "reason": "SYNTHETIC writing rationale; no measurement claim"}


def simple(setup_writing, operation="design", content=None):
    def alter(draft):
        draft.clear()
        draft.update(proposal() if content is None else content)
    setup = setup_writing(operation, writing_mode="write_only", alter=alter)
    f, backend, fake, request, out = setup
    request["budget"]["limits"]["calls"]["reviewer"] = 0
    p = Publication(f["root"])
    ref = p.put_json("one-call.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    return setup


def test_one_call_writes_long_text_without_template_or_reviewer(setup_writing, monkeypatch):
    original = TEXT + "  explanation: " + "long useful explanation " * 100 + "\n"
    setup = simple(setup_writing, content=proposal(original))
    def forbidden(*args, **kwargs):
        pytest.fail("write-only must not derive Contract, precheck or review")
    monkeypatch.setattr(writing.policy, "contract_for", forbidden)
    monkeypatch.setattr(writing.policy, "parse_draft", forbidden)
    monkeypatch.setattr(writing.policy, "precheck", forbidden)
    result = run(setup, monkeypatch)
    assert result["status"] == "completed"
    assert result["usage"]["calls"] == {"planner": 1, "reviewer": 0, "coder": 0, "extract": 0}
    assert result["usage"]["evaluator_jobs"] == 0
    assert result["writing_checks"]["semantics"]["status"] == "not_run"
    bundle = Bundle(setup[4])
    manifest = bundle.document(result["schema"])
    assert bundle.read(manifest["raw"][0]["file"]).decode() == original
    info = result["extensions"]["information"]
    assert (setup[4] / info["schema_directory"] / "root.yaml").read_text() == original
    assert "not been evaluated" in bundle.read(info["rationale"]).decode()
    delivered = calls(setup[2])
    assert [item["role"] for item in delivered] == ["planner"]
    body = json.loads(material_text(delivered[0], "delivery/writing_input"))
    assert set(body) == {"mode", "output", "note"}
    assert not any("IMPLEMENT_SCHEMA.md" in str(item.get("ref")) for item in delivered[0]["materials"])
    assert api.design(continue_run=setup[4]) == result
    assert len(calls(setup[2])) == 1
    publication = Publication(setup[4])
    selected = publication.put_json("selected.json", result)
    context = api.open_run("design", setup[4])
    publication.commit([selected], subjects=[context.request_ref])
    exported = setup[4].parent / "exported"
    import_bundle(bundle, exported, [selected], subjects=[context.request_ref])
    for reference in [info["candidate_draft"], info["rationale"], *info["retained_files"], *info["call_artifacts"]]:
        assert Bundle(exported).read(reference) == bundle.read(reference)


@pytest.mark.parametrize("content,expected", [
    (proposal("DESIGN: [unfinished\n"), "partial"),
    ({"raw": [{"node": "../escape", "text": TEXT}], "reason": "synthetic invalid path"}, "failed"),
])
def test_bad_format_retains_original_text(setup_writing, monkeypatch, content, expected):
    setup = simple(setup_writing, content=content)
    result = run(setup, monkeypatch)
    assert result["status"] == expected
    info = result["extensions"]["information"]
    assert info["diagnostics"]
    assert Bundle(setup[4]).read(info["retained_files"][0]).decode() == content["raw"][0]["text"]
    if expected == "partial":
        manifest = Bundle(setup[4]).document(result["schema"])
        assert Bundle(setup[4]).read(manifest["raw"][0]["file"]).decode() == content["raw"][0]["text"]
    assert len(calls(setup[2])) == 1


def scenario(setup, **changes):
    path = setup[2] / "scenario.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


@pytest.mark.parametrize("outer", [{}, {"reason": None, "extra_note": "SYNTHETIC optional metadata"}])
def test_write_only_explanation_need_not_be_repeated_in_outer_wrapper(setup_writing, monkeypatch, outer):
    content = {**proposal(), "dispatch": "# Dispatch\nRead root.yaml.\n"}
    setup = simple(setup_writing, content=content)
    scenario(setup, omit_outer_reason=True, outer=outer)
    result = run(setup, monkeypatch)
    assert result["status"] == "completed"
    bundle = Bundle(setup[4])
    info = result["extensions"]["information"]
    manifest = bundle.document(result["schema"])
    assert bundle.read(manifest["dispatch"]).decode() == content["dispatch"]
    retained = loads(bundle.read(info["candidate_draft"]))
    assert retained["content"] == content
    envelope = json.loads(retained["native_responses"][0])
    assert envelope["content"] == content and envelope.get("reason") is None
    assert result["usage"]["calls"]["reviewer"] == 0 and len(calls(setup[2])) == 1
    assert api.design(continue_run=setup[4]) == result


@pytest.mark.parametrize("dispatch,expected", [("# Dispatch\n- use `root.yaml`\n", "partial"), ("DESIGN: {route: root}\n", "completed")])
def test_raw_dispatch_node_is_never_guessed_to_be_markdown(setup_writing, monkeypatch, dispatch, expected):
    content = proposal()
    content["raw"].append({"node": "dispatch", "text": dispatch})
    setup = simple(setup_writing, content=content)
    scenario(setup, omit_outer_reason=True)
    result = run(setup, monkeypatch)
    assert result["status"] == expected
    bundle = Bundle(setup[4])
    manifest = bundle.document(result["schema"])
    assert manifest["dispatch"] is None
    files = {item["node"]: bundle.read(item["file"]).decode() for item in manifest["raw"]}
    assert files == {"root": TEXT, "dispatch": dispatch}
    if expected == "partial":
        assert any("dispatch.yaml" in d for d in result["extensions"]["information"]["diagnostics"])
    assert len(calls(setup[2])) == 1


@pytest.mark.parametrize("changes,error", [
    ({"outer": {"role": "coder"}}, "invalid_role_output"),
    ({"outer": {"assignment_sha256": "sha256:" + "0" * 64}}, "invalid_role_output"),
    ({"outer": {"kind": "candidate_source"}}, "invalid_role_output"),
    ({"init": {"tools": ["Read"]}}, "tool_or_mcp_permission_mismatch"),
    ({"exit_code": 7}, "nonzero_exit"),
    ({"terminal": {"subtype": "error_during_execution", "is_error": True}}, "native_terminal_failure"),
    ({"terminal": {"usage": None}}, "missing_or_invalid_usage"),
])
def test_unaccepted_native_reply_is_retained_without_publishing_schema(setup_writing, monkeypatch, changes, error):
    content = proposal()
    setup = simple(setup_writing, content=content)
    scenario(setup, omit_outer_reason=True, **changes)
    result = run(setup, monkeypatch)
    assert result["status"] in ("failed", "blocked") and result["schema"] is None
    bundle = Bundle(setup[4])
    info = result["extensions"]["information"]
    retained = loads(bundle.read(info["candidate_draft"]))
    assert retained["content"] == content and retained["ok"] is False
    assert bundle.read(info["retained_files"][0]).decode() == TEXT
    assert json.loads(retained["native_responses"][0])["content"] == content
    assert any(error in diagnostic for diagnostic in info["diagnostics"])
    receipt = loads(bundle.read(retained["receipt"]))
    assert receipt["status"] == "failed" and error in receipt["errors"]
    if error == "missing_or_invalid_usage":
        assert receipt["tokens"] is None and not receipt["accounting_complete"]
    assert len(calls(setup[2])) == 1
    assert result["usage"]["calls"]["reviewer"] == 0


def test_malformed_reply_text_remains_a_portable_artifact(setup_writing, monkeypatch):
    setup = simple(setup_writing)
    response = '{"content":{"raw":[{"node":"root","text":"unfinished'
    scenario(setup, response=response)
    result = run(setup, monkeypatch)
    assert result["status"] == "failed" and result["schema"] is None
    bundle = Bundle(setup[4])
    info = result["extensions"]["information"]
    retained = loads(bundle.read(info["candidate_draft"]))
    assert retained["content"] == response and retained["native_responses"] == [response]
    publication = Publication(setup[4])
    selected = publication.put_json("retained-failure.json", result)
    context = api.open_run("design", setup[4])
    publication.commit([selected], subjects=[context.request_ref])
    exported = setup[4].parent / "exported-failure"
    import_bundle(bundle, exported, [selected], subjects=[context.request_ref])
    for ref in [info["candidate_draft"], *info["retained_files"], *info["call_artifacts"]]:
        assert Bundle(exported).read(ref) == bundle.read(ref)


@pytest.mark.parametrize("keep", [True, False])
def test_optimize_preserves_current_schema_and_actual_feedback(setup_writing, monkeypatch, keep):
    content = {"decision": "keep", "reason": "The actual failure is in code; keep this design"} if keep else proposal()
    setup = simple(setup_writing, "optimize", content=content)
    f, _, fake, request, out = setup
    previous = Bundle(f["root"]).document(request["schema"])
    original = [Bundle(f["root"]).read(e["file"]) for e in previous["raw"]]
    p = Publication(f["root"])
    actual_feedback = b"SYNTHETIC native failure record: wrong values for shape 37x513; timing unavailable.\n"
    feedback = p.put_bytes("actual-feedback.txt", actual_feedback)
    request["feedback"] = [feedback]
    ref = p.put_json("with-feedback.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    result = run(setup, monkeypatch)
    assert result["status"] == "completed"
    assert result["decision"] == ("keep" if keep else "revise")
    assert result["next_action"] == "none"
    assert [Bundle(out).read(e["file"]) for e in previous["raw"]] == original
    assert material_text(calls(fake)[0], "delivery/feedback_0").encode() == actual_feedback
    assert bool(result["schema"] == request["schema"]) == keep
    if not keep:
        assert Bundle(out).document(result["schema"])["parent"] == request["schema"]


def test_file_api_needs_no_evaluator_and_preserves_all_inputs(tmp_path, setup_writing, monkeypatch):
    setup = simple(setup_writing)
    profile = Bundle(setup[0]["root"]).document(setup[3]["backend"]["profile"])["parameters"]["claude"]
    task = tmp_path / "task.md"
    task.write_text("SYNTHETIC task: row sum, exact ABI and bounds supplied here.\n")
    reference = tmp_path / "neighbor.md"
    reference.write_text("SYNTHETIC explicitly selected neighboring design.\n")
    out = tmp_path / "direct"
    result = api.design(task=task, out=out, runtime_profile=profile, references=[reference])
    assert result["status"] == "completed"
    context = api.open_run("design", out)
    assert context.objects["task"]["evaluator"] is None
    assert context.objects["task"]["baseline_source"] is None
    delivered = calls(setup[2])[-1]
    assert material_text(delivered, "task/contract") == task.read_text()
    assert material_text(delivered, "material/reference_0000") == reference.read_text()
    schema_dir = out / result["extensions"]["information"]["schema_directory"]
    (setup[2] / "scenario.json").write_text(json.dumps({"draft": {"decision": "keep", "reason": "SYNTHETIC keep"}}))
    feedback = tmp_path / "feedback.txt"
    feedback.write_text("SYNTHETIC actual measurement unavailable.\n")
    optimized = api.optimize(task=task, schema=schema_dir, feedback=[feedback], runtime_profile=profile, out=tmp_path / "direct-optimize")
    assert optimized["status"] == "completed" and optimized["decision"] == "keep"
    assert material_text(calls(setup[2])[-1], "delivery/feedback_0") == feedback.read_text()


def test_default_demo_stops_after_writing_without_gpu(tmp_path, synthetic):
    demo = Demo.create(tmp_path / "demo")
    result = demo.run()
    assert result["status"] == "completed"
    assert [entry["operation"] for entry in result["history"]] == ["design"]
    assert len(synthetic[0].calls) == 1
    assert not synthetic[1].calls and not synthetic[2]
    request = Bundle(demo.root).document(demo.config["requests"]["design"])
    assert request["budget"]["limits"]["calls"]["reviewer"] == 0
    materials = Bundle(demo.root).document(request["materials"])
    assert not any(e["id"].startswith(("doc_", "method_")) for e in materials["entries"])
    assert Demo(demo.root).run() == result


def test_aka_driver_runs_one_optimize_with_original_feedback(tmp_path, synthetic, monkeypatch):
    native_calls = []
    def native(**kwargs):
        assert "owner_account" not in kwargs  # Native AKA budgets are the default.
        path = Path(kwargs["workspace"])
        path.mkdir(parents=True)
        feedback = path / "feedback.txt"
        feedback.write_text("SYNTHETIC AKA measurement: 3.25 us; native comparator passed.\n")
        native_calls.append(kwargs)
        return {"status": "completed", "stop_reason": "SYNTHETIC standard result", "latency_us": 3.25,
                "workspace": str(path), "feedback_files": [str(feedback)], "artifacts": {"feedback": str(feedback)}}
    monkeypatch.setattr(api, "implement", native)
    demo = Demo.create(tmp_path / "demo", aka_config={"problem": "/synthetic/problem", "repo_root": "/synthetic/aka", "platform": "cuda", "max_iters": 1})
    result = demo.run()
    assert result["status"] == "completed" and result["optimize_count"] == 1
    assert len(native_calls) == 2 and len(synthetic[0].calls) == 2
    assert [e["operation"] for e in result["history"]] == ["design", "optimize"]
    assert not synthetic[1].calls and not synthetic[2]
    optimized = api.open_run("optimize", demo.root / demo.state["history"][1]["out"])
    assert optimized.bundle.read(optimized.request["feedback"][0]).find(b"3.25") >= 0
    assert optimized.bundle.read(optimized.request["feedback"][1]).find(b"native comparator passed") >= 0
    assert Demo(demo.root).run() == result


def test_standard_implement_api_and_cli_forward_to_aka(monkeypatch, tmp_path):
    observed = []
    def implement(**kwargs):
        observed.append(kwargs)
        return {"status": "prepared", "stop_reason": "SYNTHETIC preparation"}
    original = api.importlib.import_module
    monkeypatch.setattr(api.importlib, "import_module", lambda name: SimpleNamespace(implement=implement) if name == "aka.workflow.schema.standard" else original(name))
    result = api.implement(schema="/schema", problem="/problem", repo_root="/aka", workspace="/workspace", platform="cuda", prepare_only=True)
    assert result["status"] == "prepared"
    assert cli.main(["implement", "--schema", "/schema", "--problem", "/problem", "--repo-root", "/aka", "--platform", "L20D", "--out", str(tmp_path), "--prepare-only"]) == 0
    assert len(observed) == 2 and observed[1]["workspace"] == str(tmp_path)


def test_file_cli_runs_without_request_bundle(tmp_path, setup_writing, capsys):
    setup = simple(setup_writing)
    task = tmp_path / "task.md"
    task.write_text("SYNTHETIC task with explicit bounds and ABI.\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(Bundle(setup[0]["root"]).document(setup[3]["backend"]["profile"])["parameters"]["claude"]))
    out = tmp_path / "cli-writing"
    assert cli.main(["design", "--task", str(task), "--out", str(out), "--runtime-profile", str(profile)]) == 0
    assert "Schema: " in capsys.readouterr().out
    result = loads((out / "result.json").read_bytes())
    assert result["status"] == "completed" and result["usage"]["calls"]["reviewer"] == 0


def test_extract_cli_still_calls_original_bridge(monkeypatch):
    from kernelschema import extract_bridge
    arguments = []
    monkeypatch.setattr(extract_bridge, "extract", lambda argv: arguments.append(argv) or 17)
    assert cli.main(["extract", "--", "--original-argument"]) == 17
    assert arguments == [["--original-argument"]]
