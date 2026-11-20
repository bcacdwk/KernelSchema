from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import subprocess
import sys

import pytest

from kernelschema import api
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle
from kernelschema.budget import OwnerAccount, use_authority
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs
from kernelschema.implement.reference import ReferenceBackend

spec = spec_from_file_location("reference_support", Path(__file__).parent / "fixtures/reference_support.py")
support = module_from_spec(spec)
spec.loader.exec_module(support)


def setup(make_bundle, tmp_path, monkeypatch, *, runtime=None, evaluator=None, mode="code", limit=2, checkpoint=None):
    runtime = runtime or support.FakeRuntime()
    evaluator = evaluator or support.FakeEvaluation(tmp_path / "eval")
    backend = ReferenceBackend(runtime_factory=lambda _: runtime, evaluation_factory=lambda *_: evaluator, checkpoint=checkpoint)
    f = make_bundle()
    seed = None
    if mode == "verify_only":
        p = Publication(f["root"])
        seed = p.put_source("seed", support.CODE.encode())
        p.commit([seed])
    request = support.prepare_fixture(f, backend=backend, mode=mode, limit=limit, seed=seed)
    monkeypatch.setattr("kernelschema.implement.reference.workflow_backend", lambda: backend)
    return backend, runtime, evaluator, f, request


def test_real_api_loop_synthetic_full_task_and_soft_goal_completed(make_bundle, tmp_path, monkeypatch):
    backend, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "completed" and result["recovery"] == "none"
    assert result["code"]["latest"] == result["code"]["qualified"]
    selected = next(a for a in result["assessments"] if a["source_sha256"] == result["code"]["qualified"]["sha256"])
    assert selected["performance_goal"] == "unmet"
    assert runtime.calls == ["coder", "reviewer"] and len(evaluator.calls) == 1
    assert result["usage"]["calls"] == {"planner": 0, "coder": 1, "reviewer": 1, "extract": 0}
    assert result["usage"]["tokens"] == 20
    again = api.implement(continue_run=tmp_path / "run")
    assert again == result and runtime.calls == ["coder", "reviewer"]
    export = tmp_path / "export"
    import_bundle(Bundle(tmp_path / "run"), export, [result["code"]["qualified"], selected["fidelity"]["receipt"]], subjects=[api.open_run("implement", tmp_path / "run").request_ref])
    assert Bundle(export).source(result["code"]["qualified"]) == support.CODE.encode()


def test_verify_only_exact_seed_zero_coder_and_promotions(make_bundle, tmp_path, monkeypatch):
    _, runtime, _, f, request = setup(make_bundle, tmp_path, monkeypatch, mode="verify_only")
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "completed" and runtime.calls == ["reviewer"]
    assert result["code"]["latest"] == request["seed"]
    assert result["extensions"]["information"]["promotions"] == 0


def test_fidelity_fail_repairs_same_r_from_exact_latest_C(make_bundle, tmp_path, monkeypatch):
    runtime = support.FakeRuntime(candidates=[support.CODE + "# first\n", support.CODE + "# second\n"], reviews=["fail", "pass"])
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch, runtime=runtime)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "completed" and runtime.calls == ["coder", "reviewer", "coder", "reviewer"]
    context = api.open_run("implement", tmp_path / "run")
    coder_intents = [i for i in context.intents() if i["role"] == "coder"]
    assert context.bundle.source(coder_intents[1]["source"]).endswith(b"# first\n")
    assert result["usage"]["calls"]["planner"] == 0
    assert len(result["assessments"]) == 3


@pytest.mark.parametrize("boundary", ["seed_intent", "seed_ready", "operation_planned", "operation_reserved", "coder_admitted", "coder_settled", "coder_finished", "evaluation_ready", "evaluation_admitted", "evaluator_finished", "reviewer_settled", "fidelity_intent", "reviewer_finished", "result_ready"])
def test_first_result_crash_recovers_original_ops_without_duplicate_charges(make_bundle, tmp_path, monkeypatch, boundary):
    fired = []
    def crash(label, _):
        if label == boundary and not fired:
            fired.append(label)
            raise RuntimeError("synthetic crash")
    backend, runtime, evaluator, f, _ = setup(make_bundle, tmp_path, monkeypatch, checkpoint=crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert not (tmp_path / "run/result.json").exists()
    backend.checkpoint = lambda *_: None
    result = api.implement(continue_run=tmp_path / "run")
    assert result["status"] == "completed"
    assert runtime.calls == ["coder", "reviewer"] and len(evaluator.calls) == 1
    assert result["usage"]["calls"]["coder"] == 1 and result["usage"]["evaluator_jobs"] == 1


def test_unknown_original_blocks_and_later_terminal_settles_once(make_bundle, tmp_path, monkeypatch):
    runtime = support.FakeRuntime(unknown=True)
    _, runtime, evaluator, f, _ = setup(make_bundle, tmp_path, monkeypatch, runtime=runtime)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert (result["status"], result["recovery"]) == ("blocked", "settle_only")
    assert result["usage"]["calls"]["coder"] is None
    api.implement(continue_run=tmp_path / "run")
    assert runtime.calls == ["coder"] and not evaluator.calls
    context = api.open_run("implement", tmp_path / "run")
    runtime.complete(context, context.intents()[0]["operation_id"])
    runtime.unknown = False
    result = api.implement(continue_run=context.root)
    assert result["status"] == "completed" and runtime.calls == ["coder", "reviewer"]


def test_settled_check_unknown_is_partial_and_can_repair(make_bundle, tmp_path, monkeypatch):
    evaluator = support.FakeEvaluation(tmp_path / "eval", statuses=["unknown", "pass"])
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch, evaluator=evaluator)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "completed" and len(evaluator.calls) == 2
    assert runtime.calls.count("coder") == 2


def test_minimum_finish_reserve_prevents_any_coder_charge(make_bundle, tmp_path, monkeypatch):
    _, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch)
    request["budget"]["limits"]["calls"]["reviewer"] = 0
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "partial" and not runtime.calls and not evaluator.calls


def test_corrupt_seed_and_missing_owner_never_fallback(make_bundle, tmp_path, monkeypatch):
    _, runtime, _, f, request = setup(make_bundle, tmp_path, monkeypatch, mode="verify_only")
    seed = f["root"] / request["seed"]["path"]
    seed.with_name("files").joinpath("kernel.py").write_bytes(b"changed")
    with pytest.raises(ProtocolError):
        api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert not runtime.calls


def test_missing_ledger_on_continue_rejects_new_balance(make_bundle, tmp_path, monkeypatch):
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch)
    api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    (tmp_path / "run/.runtime/owner-account.json").unlink()
    with pytest.raises(ProtocolError):
        api.implement(continue_run=tmp_path / "run")
    assert runtime.calls == ["coder", "reviewer"]


def test_missing_request_contribution_is_not_recreated(make_bundle, tmp_path, monkeypatch):
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch)
    api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    authority = tmp_path / "run/.runtime/owner-account.json"
    state = json.loads(authority.read_bytes())
    state["requests"] = {}
    durable_write(authority, jcs(state), mode=0o600, replace=True)
    with pytest.raises(ProtocolError, match="contribution missing"):
        api.implement(continue_run=tmp_path / "run")
    assert runtime.calls == ["coder", "reviewer"]


def test_default_factory_public_profile_cli_blocks_without_charging(make_bundle, tmp_path):
    from kernelschema.resources import resource
    backend = ReferenceBackend()
    f = make_bundle()
    request = support.prepare_fixture(f, backend=backend)
    profile = Bundle(f["root"]).document(request["backend"]["profile"])
    claude = json.loads(resource("kernelschema/runtime/profiles/claude-code-2.1.316-qwen3.8-max-no-tools.v1.json").read_bytes())
    profile["parameters"]["claude"] = claude
    profile["model"] = claude["requested_model"]
    p = Publication(f["root"])
    request["backend"]["profile"] = p.put_json("profile.json", profile)
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    result = subprocess.run([sys.executable, "-B", "-m", "kernelschema", "implement", "--request", str(f["root"] / "request.json"), "--out", str(tmp_path / "run")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    projection = json.loads((tmp_path / "run/result.json").read_bytes())
    assert projection["status"] == "blocked" and projection["recovery"] == "new_request"
    assert not any(projection["usage"]["calls"].values()) and projection["usage"]["evaluator_jobs"] == 0
    assert not api.open_run("implement", tmp_path / "run").intents()


def test_invalid_candidate_retains_stable_seed_and_no_evaluator_charge(make_bundle, tmp_path, monkeypatch):
    runtime = support.FakeRuntime(candidates=["\ufeff# damaged source\n"])
    _, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch, runtime=runtime)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "partial" and not evaluator.calls
    task = Bundle(f["root"]).document(request["task"])
    assert result["code"]["latest"] == task["starter_source"]
    assert result["extensions"]["information"]["diagnostics"]


def test_subset_rejected_before_operations(make_bundle, tmp_path, monkeypatch):
    _, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch)
    p = Publication(f["root"])
    scope = Bundle(f["root"]).document(request["scope"])
    scope["coverage_ids"] = scope["coverage_ids"][:1]
    scope["bindings"][0]["coverage_ids"] = scope["coverage_ids"]
    request["scope"] = p.put_json("scope.json", scope)
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    with pytest.raises(UnsupportedError, match="full Task"):
        api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert not runtime.calls and not evaluator.calls


def test_live_unknown_evaluator_does_not_call_reviewer_or_resubmit(make_bundle, tmp_path, monkeypatch):
    evaluator = support.FakeEvaluation(tmp_path / "eval", unknown=True)
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch, evaluator=evaluator)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["recovery"] == "settle_only" and len(evaluator.calls) == 1
    api.implement(continue_run=tmp_path / "run")
    assert runtime.calls == ["coder"] and len(evaluator.calls) == 1
    evaluator.unknown = False
    result = api.implement(continue_run=tmp_path / "run")
    assert result["status"] == "completed" and len(evaluator.calls) == 1


def test_first_ready_seed_change_is_not_overwritten(make_bundle, tmp_path, monkeypatch):
    def crash(label, _):
        if label == "seed_intent":
            raise RuntimeError("stop before seed")
    backend, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch, checkpoint=crash)
    with pytest.raises(RuntimeError):
        api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    seed_dir = tmp_path / "run/.runtime/reference/seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "kernel.py").write_bytes(b"changed draft")
    backend.checkpoint = lambda *_: None
    with pytest.raises(ProtocolError, match="never overwrite"):
        api.implement(continue_run=tmp_path / "run")
    assert (seed_dir / "kernel.py").read_bytes() == b"changed draft" and not runtime.calls


def test_same_parent_same_q_new_run_charges_without_copying_parent_history(make_bundle, tmp_path, monkeypatch):
    _, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch)
    owner = OwnerAccount.create(tmp_path / "parent.json", owner="fixed-parent", limits=request["budget"]["limits"])
    p = Publication(f["root"])
    request["budget"]["parent"] = p.put_json("parent.json", owner.descriptor)
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    with use_authority(owner.path):
        first = api.implement(request=f["root"] / "request.json", out=tmp_path / "run1")
        second = api.implement(request=f["root"] / "request.json", out=tmp_path / "run2")
        third = api.implement(request=f["root"] / "request.json", out=tmp_path / "run3")
    assert first["status"] == second["status"] == "completed"
    assert first["request_sha256"] == second["request_sha256"] and first["run_id"] != second["run_id"]
    assert first["usage"]["calls"]["coder"] == second["usage"]["calls"]["coder"] == 1
    assert third["status"] == "partial" and third["code"]["qualified"] is None
    assert third["usage"]["calls"]["coder"] == 0
    assert runtime.calls == ["coder", "reviewer", "coder", "reviewer"] and len(evaluator.calls) == 2


def test_synthetic_projection_keeps_latest_and_qualified_subjects_separate(make_bundle, tmp_path, monkeypatch):
    backend, _, _, f, _ = setup(make_bundle, tmp_path, monkeypatch)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    context = api.open_run("implement", tmp_path / "run")
    state = context.load_state("reference")
    p = Publication(context.root)
    latest = p.put_source("unassessed", support.CODE.encode() + b"# synthetic projector-only newer C\n")
    p.commit([latest])
    state.update(latest=latest, phase="synthetic-projector-only", status="partial")
    backend._assessment(state, latest, context)
    projection = backend._result(context, state, OwnerAccount.for_context(context))
    api.validate_result(context, projection)
    assert projection["code"] == {"latest": latest, "qualified": result["code"]["qualified"]}
    assessment = next(a for a in projection["assessments"] if a["source_sha256"] == latest["sha256"])
    assert assessment["correctness"]["status"] == "not_run"


def test_real_K3_fake_process_uses_shared_authorization_and_records_unknown_review(make_bundle, tmp_path, monkeypatch):
    from kernelschema.runtime import ClaudeProfile, ClaudeRuntime
    fake_root = tmp_path / "cli"
    fake_root.mkdir()
    script = Path(__file__).parent / "fixtures/runtime/fake_claude.py"
    fake = fake_root / "claude"
    fake.write_text("#!" + sys.executable + "\n" + "\n".join(script.read_text().splitlines()[1:]) + "\n")
    fake.chmod(0o755)
    (fake_root / "scenario.json").write_text('{"mode":"success","seconds":0}')
    profile = ClaudeProfile.inspect(requested_model="synthetic-model", executable=str(fake), timeout_ms=3000)
    runtime = ClaudeRuntime(profile)
    _, _, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch, runtime=runtime, limit=1)
    p = Publication(f["root"])
    request["backend"]["profile"] = p.put_json("claude.json", profile.backend_profile(f["originals"]["task/environment.md"], name="reference"))
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert result["status"] == "partial" and result["code"]["qualified"] is None
    assert result["usage"]["calls"]["coder"] == result["usage"]["calls"]["reviewer"] == 1
    assert result["usage"]["tokens"] == 52 and len(evaluator.calls) == 1
    assert len((fake_root / "invocations.jsonl").read_text().splitlines()) == 2


def test_terminated_job_without_report_is_unknown_not_unrun(make_bundle, tmp_path, monkeypatch):
    class Reportless(support.FakeEvaluation):
        def settle(self, *args):
            return {**super().settle(*args), "report": None}
    evaluator = Reportless(tmp_path / "eval")
    _, runtime, _, f, _ = setup(make_bundle, tmp_path, monkeypatch, evaluator=evaluator, limit=1)
    result = api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    latest = next(a for a in result["assessments"] if a["source_sha256"] == result["code"]["latest"]["sha256"])
    assert latest["build"]["status"] == latest["correctness"]["status"] == "unknown"
    assert result["status"] == "partial" and result["recovery"] == "none"
    assert result["usage"]["evaluator_jobs"] == 1


@pytest.mark.parametrize("where", ["parent", "native"])
def test_finite_parent_or_native_tokens_rejected_before_charging(make_bundle, tmp_path, monkeypatch, where):
    from copy import deepcopy
    _, runtime, evaluator, f, request = setup(make_bundle, tmp_path, monkeypatch)
    limit = deepcopy(request["budget"]["limits"])
    limit["tokens"] = 100
    p = Publication(f["root"])
    if where == "parent":
        owner = OwnerAccount.create(tmp_path / "parent.json", owner="parent", limits=limit)
        request["budget"]["parent"] = p.put_json("parent.json", owner.descriptor)
    else:
        owner = None
        profile = Bundle(f["root"]).document(request["backend"]["profile"])
        profile["native_limits"]["workflow_limits"] = limit
        request["backend"]["profile"] = p.put_json("profile.json", profile)
    ref = p.put_json("request.json", request)
    p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    with use_authority(owner.path if owner else tmp_path / "unused"):
        with pytest.raises(UnsupportedError, match="finite parent/native token"):
            api.implement(request=f["root"] / "request.json", out=tmp_path / "run")
    assert not runtime.calls and not evaluator.calls
