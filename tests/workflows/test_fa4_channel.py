"""Synthetic frames and actual Agate state machine; zero network/GPU calls."""
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from kernelschema.artifacts import Bundle, Publication, FileBytes
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, sha256
from kernelschema.evaluation.agate import AgateDevExecutor, MARKER, validate_prepared_request, interpret_terminal
from kernelschema.evaluation.fa4 import stage_candidate, _case
from kernelschema.evaluation.fa4_channel import _inner, complete, read_qualification, selection, qualify
from kernelschema.resources import resource
from examples.workflows.fa4_decode.prepare import prepare


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    root = tmp_path_factory.mktemp("fa4-channel") / "bundle"
    result = prepare(root, checkout=resource("AGENTS.md").parent)
    staged = stage_candidate(Bundle(root), result["case"], root / "candidate-stage")
    return root, result, staged


def environment():
    return {**json.loads(_case("task/environment.json"))["required"],
            "python": "synthetic Python 3.x", "executable": "/original/python", "executable_sha256": "sha256:" + "4" * 64,
            "nvidia_smi": "synthetic SM103, UUID, driver", "tvm_ffi_version": "synthetic", "tvm_ffi_module_sha256": "sha256:" + "5" * 64,
            "compiler": "synthetic nvcc", "qualification": "not_established"}


def observation(inner):
    c = {"passed": True, "out_view_intact": True, "return_aliases_original_view": True,
         "readonly_bytes": dict.fromkeys(("query", "kv_cache", "block_tables", "seq_lens"), True),
         "readonly_views": dict.fromkeys(("query", "kv_cache", "block_tables", "seq_lens"), True),
         "outputs": {k: {"passed": True, "elements": 16384, "failing_elements": 0} for k in ("return", "original_out")}}
    return {"stage_sha256": digest(inner), "subject": {k: inner[k]["sha256"] for k in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")},
            "permissions": {"boundary": "fa4-agate-f1-bwrap-v1", "probes": [
                {"path": name, "write_open_denied": True} for name in ["stage.json", *inner["files"]]]},
            "environment": environment(), "input_digests": dict.fromkeys(("query", "kv_cache", "block_tables", "seq_lens", "workspace_buffer"), "sha256:" + "6" * 64),
            "build": "pass", "passed": True, "cold": dict.fromkeys(("source_import_ns", "input_generation_ns", "constructor_ns", "first_forward_sync_ns"), 10),
            "checks": [{"phase": "cold", **deepcopy(c)}, *[{"phase": phase, "index": i, **deepcopy(c)} for phase, n in (("warmup", 5), ("steady", 40)) for i in range(n)]],
            "samples_ns": [100] * 40, "native_when": "observed before/after each warmup and steady call"}


def execution(intent):
    if intent["action"] == "fa4_probe":
        return {"action": "fa4_probe", "exit_code": 0, "environment": environment(), "bwrap_sha256": "sha256:" + "7" * 64}
    m = intent["stage"]["manifest"]
    reports = {}
    for label in ("baseline", "candidate"):
        inner = _inner(m, label)
        reports[label] = {"files": inner["files"], "stage_sha256": digest(inner), "observation": json.dumps(observation(inner)),
                          "exit_code": 0, "stdout": "synthetic stdout", "stderr": "synthetic stderr"}
    return {"action": "fa4_execute", "exit_code": 0, "boundary": "fa4-agate-f1-bwrap-v1",
            "stage_sha256": intent["stage"]["sha256"], "loaded": m["files"],
            "bwrap_sha256": intent["profile"]["bwrap_sha256"], "reports": reports}


def terminal(value):
    raw = jcs(value)
    frame = MARKER + json.dumps({"sha256": sha256(raw), "bytes": len(raw), "gzip_base64": base64.b64encode(gzip.compress(raw)).decode()})
    return {"job_id": "synthetic-job", "status": "succeeded", "finished_at": "2026-11-20T00:00:00Z", "cleanup_pending": False,
            "result": {"exit_code": 0, "stdout": frame}}


class Transport:
    def __init__(self):
        self.posts = 0
        self.native = None
        self.unknown = False
    def identity(self): return {"synthetic": True, "name": "CPU-only FA4 fake transport"}
    def request(self, method, path, body=None):
        if method == "POST":
            self.posts += 1
            return {"status": 200, "body": b'{"job_id":"synthetic-job"}'}
        if self.unknown:
            return {"status": 404, "body": b'{}'}
        return {"status": 200, "body": json.dumps(self.native).encode()}


def job(ready, tmp_path):
    transport = Transport()
    executor = AgateDevExecutor(tmp_path / "jobs", transport)
    intent = executor.prepare(deepcopy(ready[2]), profile={"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/original/python",
        "bwrap": "/usr/bin/bwrap", "bwrap_sha256": "sha256:" + "7" * 64}, operation_id="synthetic_operation", submission_key="synthetic_operation",
        owner="synthetic-owner", reservation={"owner": "synthetic-owner", "evaluator_jobs": 1, "active_ms": 60000},
        environment=environment(), action="fa4_execute")
    transport.native = terminal(execution(intent))
    return executor, transport, intent


def test_upload_and_real_original_job_settlement_are_complete_and_synthetic(ready, tmp_path):
    executor, transport, intent = job(ready, tmp_path)
    request = json.loads((executor._job(intent["operation_id"]) / "request.json").read_bytes())
    assert len([n for n in request["files"] if n.startswith("_ks_stage/source/")]) == 12
    assert len([n for n in request["files"] if n.startswith("_ks_stage/baseline/")]) == 12
    validate_prepared_request(intent, request)
    executor.submit(intent["operation_id"], authorize=lambda: True)
    settled = executor.settle(intent["operation_id"], Publication(ready[0]))
    report = Bundle(ready[0]).document(settled["report"])
    assert {report[k]["status"] for k in ("build", "correctness", "performance")} == {"unknown"}
    assert settled["usage_row"]["evaluator_jobs"] == 1
    assert executor.settle(intent["operation_id"], Publication(ready[0]))["report"] == settled["report"]
    assert transport.posts == 1


@pytest.mark.parametrize("change", ["delete", "worker", "mode", "command", "source_identity"])
def test_upload_tampering_rejected_even_with_new_request_digest(ready, tmp_path, change):
    executor, _, intent = job(ready, tmp_path)
    request = json.loads((executor._job(intent["operation_id"]) / "request.json").read_bytes())
    if change == "delete": del request["files"]["_ks_stage/source/solution.py"]
    elif change == "worker": request["files"]["_ks_stage/_fa4_remote.py"] += "# altered\n"
    elif change == "command": request["command"] += " extra"
    elif change == "mode": intent["stage"]["manifest"]["files"]["source/solution.py"]["mode"] = "0755"
    else: intent["stage"]["manifest"]["source"]["sha256"] = "sha256:" + "b" * 64
    intent["request_sha256"] = digest(request)
    with pytest.raises(ProtocolError): validate_prepared_request(intent, request)


@pytest.mark.parametrize("change", ["subject", "permission", "inputs", "environment", "truncated", "wrong_file"])
def test_incomplete_or_mismatched_original_output_cannot_pass(ready, tmp_path, change):
    _, _, intent = job(ready, tmp_path)
    value = execution(intent)
    raw = value["reports"]["candidate"]
    obs = json.loads(raw["observation"])
    if change == "subject": obs["subject"]["source"] = "sha256:" + "0" * 64
    elif change == "permission": obs["permissions"]["probes"].pop()
    elif change == "inputs": obs["input_digests"]["query"] = "sha256:" + "0" * 64
    elif change == "environment": obs["environment"]["compute_capability"] = [9, 0]
    elif change == "wrong_file": raw["files"].pop("source/solution.py")
    raw["observation"] = json.dumps(obs)
    native = terminal(value)
    if change == "truncated": native["result"]["stdout"] = native["result"]["stdout"][:-10]
    result, diagnostic = interpret_terminal(intent, native)
    assert result is None and diagnostic


def test_unknown_original_job_only_queries_and_retains_one_fee(ready, tmp_path):
    executor, transport, intent = job(ready, tmp_path)
    transport.unknown = True
    assert executor.submit(intent["operation_id"], authorize=lambda: True)["state"] == "unknown"
    assert executor.submit(intent["operation_id"], authorize=lambda: pytest.fail("must not authorize again"))["state"] == "unknown"
    transport.unknown = False
    result = executor.settle(intent["operation_id"], Publication(ready[0]))
    assert result["state"] == "terminal" and result["usage_row"]["evaluator_jobs"] == 1 and transport.posts == 1


def test_complete_measurement_requires_every_forward_and_native_predicate(ready):
    obs = observation(_inner(ready[2]["manifest"], "candidate"))
    assert complete(obs)
    for key in ("checks", "samples_ns"):
        value = deepcopy(obs); value[key].pop(); assert not complete(value)
    value = deepcopy(obs); value["checks"][-1]["readonly_bytes"]["query"] = False; assert not complete(value)
    value = deepcopy(obs); value["native_when"] = "not_run"; assert not complete(value)
    value = deepcopy(obs); value["cold"]["first_forward_sync_ns"] = 0; assert not complete(value)
    from kernelschema.evaluation.fa4_channel import correctness_status
    value = deepcopy(obs); value.update(checks=obs["checks"][:1], passed=False, native_when="not_run", exception={"message":"native predicate unavailable"})
    assert correctness_status(value) == "unknown" and not complete(value)
    value["checks"][0]["passed"] = False
    assert correctness_status(value) == "fail"


def test_remote_launcher_constructs_two_fresh_readonly_stages_without_cuda(ready, tmp_path, monkeypatch):
    from kernelschema.evaluation import _fa4_remote as remote
    binary = tmp_path / "synthetic-bwrap"; binary.write_bytes(b"synthetic launcher never executed")
    calls = []
    def run(argv, **kwargs):
        stage = Path(argv[argv.index("--stage") + 1])
        output = Path(argv[argv.index("--output") + 1])
        inner = json.loads((stage / "stage.json").read_bytes())
        assert "--require-readonly" in argv and "--unshare-all" in argv and "--clearenv" in argv
        assert argv.count("--ro-bind") == 2 and "--setenv" in argv
        assert not output.is_relative_to(stage)
        output.write_text(json.dumps(observation(inner)))
        calls.append({"stage": str(stage), "output": str(output), "source": inner["source"]})
        return type("SyntheticCompleted", (), {"returncode": 0})()
    monkeypatch.setattr(remote.subprocess, "run", run)
    result = remote.run(Path(ready[2]["root"]), {"bwrap": str(binary), "bwrap_sha256": sha256(binary.read_bytes()),
        "python": "/original/python", "timeout_s": 60, "deadline_utc_ms": None})
    assert len(calls) == 2 and calls[0]["stage"] != calls[1]["stage"] and calls[0]["output"] != calls[1]["output"]
    assert set(result["reports"]) == {"baseline", "candidate"}


def test_six_job_qualification_uses_original_owner_and_never_qualifies_synthetic(ready, tmp_path):
    from kernelschema import api
    from kernelschema.artifacts import durable_write
    from kernelschema.budget import OwnerAccount
    class QualificationTransport(Transport):
        def __init__(self):
            super().__init__(); self.natives = {}; self.unknown = True
        def request(self, method, path, body=None):
            if method == "POST":
                self.posts += 1
                key = "synthetic_" + str(self.posts)
                m = json.loads(body["files"]["_ks_stage/stage.json"])
                intent = {"stage": {"manifest": m, "sha256": digest(m)}, "action": body["command"].split()[-1],
                          "profile": {"bwrap_sha256": "sha256:" + "7" * 64}}
                value = execution(intent)
                if intent["action"] == "fa4_execute":
                    code = body["files"]["_ks_stage/source/solution.py"]
                    name = next((n for n in ("missing_out", "changed_input", "tolerance", "storage") if "# Fixed negative qualification control: " + n in code), None)
                    if name:
                        raw = value["reports"]["candidate"]
                        obs = json.loads(raw["observation"]); obs["passed"] = False; obs["checks"] = obs["checks"][:1]
                        obs["checks"][0]["passed"] = False
                        if name in ("missing_out", "tolerance"): obs["checks"][0]["outputs"]["original_out"]["passed"] = False
                        elif name == "changed_input": obs["checks"][0]["readonly_bytes"]["query"] = False
                        else: obs["checks"][0]["out_view_intact"] = False
                        raw["observation"] = json.dumps(obs); raw["exit_code"] = 2
                self.natives[key] = {**terminal(value), "job_id": key}
                return {"status": 200, "body": json.dumps({"job_id": key}).encode()}
            if self.unknown: return {"status": 404, "body": b'{}'}
            return {"status": 200, "body": json.dumps(self.natives[path.split("/")[-1].split("?")[0]]).encode()}
    root, result, _ = ready
    request = Bundle(root).document(result["request"])
    request["budget"]["limits"].update(evaluator_jobs=6, active_ms=120000)
    p = Publication(root); ref = p.put_json("qualification-request.json", request); p.commit([ref])
    path = root / "qualification-request.json"; durable_write(path, Bundle(root).read(ref))
    c = api.prepare_run("implement", request=path, out=tmp_path / "run", backend_identity={"synthetic": True})
    remote = {"channel": "agate-dev-bwrap-v1", "gpu": "L20D", "python": "/original/python", "bwrap": "/usr/bin/bwrap"}
    transport = QualificationTransport()
    first = qualify(c, remote=remote, transport=transport)
    assert first["recovery"] == "settle_only" and transport.posts == 1
    transport.unknown = False
    result = qualify(c, remote=remote, transport=transport)
    assert result["status"] == "blocked" and result["qualification"] is None and "synthetic" in result["reason"]
    assert result["usage"]["evaluator_jobs"] == 6 and transport.posts == 6
    assert len(OwnerAccount.for_context(c).operations(c.run_id)) == 6
    assert qualify(c, remote=remote, transport=transport)["usage"]["evaluator_jobs"] == 6 and transport.posts == 6
