"""Finite wiring only: synthetic writer/native results, no model or GPU."""
import json
from pathlib import Path

import pytest

from kernelschema import api, cli, ProtocolError


@pytest.fixture
def inputs(tmp_path):
    task = tmp_path / "task.md"
    task.write_text("Explicit native task; no invented workload\n")
    schema = tmp_path / "root.yaml"
    schema.write_bytes(b"DESIGN: original\r\n")
    return dict(task=task, schema=schema, out=tmp_path / "run", problem=tmp_path,
                repo_root=tmp_path, platform="CPU-fixture", max_iters=4)


def proposal(out, text="DESIGN: proposed\n"):
    raw = out / "artifacts/raw"
    raw.mkdir(parents=True)
    (raw / "root.yaml").write_text(text)
    return {"status": "completed", "stop_reason": "files written", "decision": "revise",
            "extensions": {"information": {"schema_directory": "artifacts/raw", "diagnostics": []}}}


def native(workspace, status="completed", **extra):
    workspace.mkdir()
    feedback = workspace / "original.json"
    feedback.write_bytes(b'{"correctness":"FAIL","timing":null,"details":"all retained"}\r\n')
    return {"status": status, "exit_code": 0 if status == "completed" else 1,
            "stop_reason": "synthetic terminal", "workspace": str(workspace),
            "feedback_files": [str(feedback)], "versions": [{"correctness": "PASS", "subject": "incumbent"}],
            "native_state": {"episodes": 4}, **extra}


def test_from_task_preserves_failed_feedback_and_full_native_campaign(inputs, monkeypatch):
    inputs.pop("schema")
    events, results, originals = [], [], []
    options = dict(settings={"token_budget": 900}, runtime={"shape_batch_size": 1}, patches=("native.json",))

    def design(**kw):
        events.append("design")
        assert kw["task"] == inputs["task"] and kw["runtime_profile"] == "explicit-writing.json"
        return proposal(kw["out"])

    def implement(**kw):
        events.append("implement")
        assert kw["max_iters"] == 4
        assert all(kw[name] == value for name, value in options.items())
        assert "owner_account" not in kw
        originals.append((kw["schema"] / "root.yaml", (kw["schema"] / "root.yaml").read_bytes()))
        result = native(kw["workspace"], "failed" if not results else "completed")
        results.append(result)
        return result

    def optimize(**kw):
        events.append("optimize")
        assert json.loads(kw["feedback"][0].read_text()) == results[0]
        assert kw["feedback"][1].read_bytes() == Path(results[0]["feedback_files"][0]).read_bytes()
        assert json.loads(kw["feedback"][1].read_text())["correctness"] == "FAIL"
        return proposal(kw["out"], "DESIGN: revised after actual failure\n")

    monkeypatch.setattr(api, "design", design)
    monkeypatch.setattr(api, "implement", implement)
    monkeypatch.setattr(api, "optimize", optimize)
    result = api.iterate(**inputs, rounds=2, runtime_profile="explicit-writing.json", **options)
    assert events == ["design", "implement", "optimize", "implement"]
    assert len({r["workspace"] for r in results}) == 2
    assert result["status"] == "completed" and "version limit" in result["stop_reason"]
    assert all(path.read_bytes() == data for path, data in originals)
    assert "correctness" not in result and "qualified" not in result
    assert json.loads((inputs["out"] / "iterations.json").read_text()) == result
    for stage in result["stages"]:
        assert Path(stage["result_file"]).is_file()


def test_existing_schema_keep_stops_without_design_or_candidate_success(inputs, monkeypatch):
    original = inputs["schema"].read_bytes()
    seen = []
    monkeypatch.setattr(api, "design", lambda **kw: pytest.fail("existing Schema must skip design"))
    monkeypatch.setattr(api, "implement", lambda **kw: seen.append(kw) or native(kw["workspace"], "failed"))
    monkeypatch.setattr(api, "optimize", lambda **kw: {"status": "completed", "decision": "keep", "stop_reason": "keep"})
    result = api.iterate(**inputs, rounds=5)
    assert len(seen) == 1 and seen[0]["schema"] == inputs["schema"]
    assert result["status"] == "failed" and result["stop_reason"] == "optimize kept the current Schema"
    assert result["schema"] == str(inputs["schema"])
    assert inputs["schema"].read_bytes() == original


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_invalid_rounds_cannot_start_a_stage(inputs, monkeypatch, rounds):
    monkeypatch.setattr(api, "implement", lambda **kw: pytest.fail("invalid rounds executed"))
    with pytest.raises(ProtocolError, match="positive integer"):
        api.iterate(**inputs, rounds=rounds)
    assert not inputs["out"].exists()


def test_one_version_keeps_native_iters_without_an_optimize_call(inputs, monkeypatch):
    monkeypatch.setattr(api, "optimize", lambda **kw: pytest.fail("version limit exceeded"))
    monkeypatch.setattr(api, "implement", lambda **kw: native(kw["workspace"]))
    result = api.iterate(**inputs, rounds=1)
    assert [s["operation"] for s in result["stages"]] == ["implement"]
    assert result["stages"][0]["feedback"]


@pytest.mark.parametrize("stage_status", ["completed", "failed", "partial", "blocked"])
def test_writing_without_files_stops(inputs, monkeypatch, stage_status):
    inputs.pop("schema")
    monkeypatch.setattr(api, "design", lambda **kw: {"status": stage_status, "stop_reason": "no output"})
    monkeypatch.setattr(api, "implement", lambda **kw: pytest.fail("writing supplied no Schema"))
    result = api.iterate(**inputs)
    assert result["status"] != "completed"
    assert [s["operation"] for s in result["stages"]] == ["design"]


@pytest.mark.parametrize("relative", ["missing", "../../outside"])
def test_revision_without_a_usable_local_artifact_stops(inputs, monkeypatch, relative):
    calls = []
    monkeypatch.setattr(api, "implement", lambda **kw: calls.append(kw) or native(kw["workspace"]))
    monkeypatch.setattr(api, "optimize", lambda **kw: {"status": "completed", "decision": "revise",
        "extensions": {"information": {"schema_directory": relative}}})
    result = api.iterate(**inputs)
    assert result["status"] == "failed" and len(calls) == 1
    assert json.loads((inputs["out"] / "iterations.json").read_text())["status"] == "failed"


def test_unsettled_writing_keep_is_not_a_completed_decision(inputs, monkeypatch):
    monkeypatch.setattr(api, "implement", lambda **kw: native(kw["workspace"]))
    monkeypatch.setattr(api, "optimize", lambda **kw: {"status": "completed", "decision": "keep",
        "usage": {"unsettled": True}})
    result = api.iterate(**inputs)
    assert result["status"] == "blocked" and "unsettled" in result["stop_reason"]


@pytest.mark.parametrize("status,extra", [("blocked", {}), ("pending", {}), ("interrupted", {}),
                                         ("failed", {"usage": {"unsettled": True}})])
def test_unsettled_native_result_stops_after_retaining_feedback(inputs, monkeypatch, status, extra):
    monkeypatch.setattr(api, "implement", lambda **kw: native(kw["workspace"], status, **extra))
    monkeypatch.setattr(api, "optimize", lambda **kw: pytest.fail("unsettled stage must stop"))
    result = api.iterate(**inputs)
    assert result["status"] == "blocked"
    assert Path(result["stages"][0]["feedback"][0]["file"]).is_file()


def test_missing_declared_feedback_is_not_silently_dropped(inputs, monkeypatch):
    monkeypatch.setattr(api, "implement", lambda **kw: native(kw["workspace"],
        feedback_files=[str(kw["workspace"] / "missing.json")]))
    monkeypatch.setattr(api, "optimize", lambda **kw: pytest.fail("missing feedback must stop"))
    result = api.iterate(**inputs)
    assert result["status"] == "blocked" and "feedback" in result["stop_reason"]
    assert Path(result["stages"][0]["result_file"]).is_file()


@pytest.mark.parametrize("interrupted", [False, True])
def test_existing_output_never_replays_a_stage(inputs, monkeypatch, interrupted):
    calls = []
    def implement(**kw):
        calls.append(kw)
        if interrupted:
            kw["workspace"].mkdir()
            (kw["workspace"] / "receipt.txt").write_text("possibly charged; inspect original")
            raise KeyboardInterrupt()
        return native(kw["workspace"])
    monkeypatch.setattr(api, "implement", implement)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            api.iterate(**inputs)
        summary = json.loads((inputs["out"] / "iterations.json").read_text())
        assert summary["status"] == summary["stages"][0]["status"] == "interrupted"
    else:
        api.iterate(**inputs, rounds=1)
    before = (inputs["out"] / "iterations.json").read_bytes()
    with pytest.raises(ProtocolError, match="already exists"):
        api.iterate(**inputs)
    assert len(calls) == 1
    assert (inputs["out"] / "iterations.json").read_bytes() == before


@pytest.mark.parametrize("populated", [False, True])
def test_existing_directory_is_rejected_before_any_call(inputs, monkeypatch, populated):
    inputs["out"].mkdir()
    if populated:
        (inputs["out"] / "original.txt").write_text("unrelated original output")
    monkeypatch.setattr(api, "implement", lambda **kw: pytest.fail("existing output executed"))
    with pytest.raises(ProtocolError, match="already exists"):
        api.iterate(**inputs)
    assert not (inputs["out"] / "iterations.json").exists()


def test_cli_passes_version_and_native_iteration_limits_separately(inputs, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(api, "iterate", lambda **kw: seen.append(kw) or {"status": "completed", "stop_reason": "keep"})
    assert cli.main(["iterate", "--task", str(inputs["task"]), "--out", str(inputs["out"]),
        "--schema", str(inputs["schema"]), "--problem", str(inputs["problem"]),
        "--repo-root", str(inputs["repo_root"]), "--platform", "CPU-fixture", "--rounds", "2",
        "--max-iters", "7", "--runtime-profile", "writing.json"]) == 0
    assert seen[0]["rounds"] == 2 and seen[0]["max_iters"] == 7
    assert seen[0]["runtime_profile"] == "writing.json" and seen[0]["owner_account"] is None
    assert "iterations.json" in capsys.readouterr().out
