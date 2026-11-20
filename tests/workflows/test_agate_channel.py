"""Synthetic in-memory wire only: no SDK auth, remote service, CUDA or fees."""
import base64
import gzip
import json
from pathlib import Path
import time

import pytest

from examples.workflows.row_softmax.driver import Demo
from kernelschema import api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.budget import use_authority
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.evaluation.agate import MARKER, PROFILE, AgateDevExecutor
from kernelschema.evaluation.channel import AgateChannel, qualify, read_qualification, selection, BAD_SOURCES, BAD_CHECKS
from kernelschema.evaluation.task import stage
from kernelschema.runtime import AdmissionDenied
from test_standalone import synthetic_target


def frame(value):
    raw = jcs(value)
    return MARKER + json.dumps({"bytes": len(raw), "sha256": sha256(raw), "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode()})


class Wire:
    """Injected test transport, never installed/configured by the real factory."""
    def __init__(self, target):
        self.target = target
        self.posts, self.native = [], {}
        self.unknown = False
        self.lost_ack = False
        self.accept_negative = False
    def identity(self):
        # Exercise actual admission logic without enabling any production bypass.
        return {"fixture": "SYNTHETIC in-memory wire; no service qualification"}
    def request(self, method, path, payload=None):
        if method == "POST":
            self.posts.append(payload)
            job = "synthetic_" + str(len(self.posts))
            files = payload["files"]
            manifest = loads(files["_ks_stage/stage.json"].encode())
            action = payload["command"].split()[-1]
            if action == "probe":
                observed = {"environment": {"mode": "cuda", **self.target["runtime"]},
                            "inputs": {"torch_version": self.target["runtime"]["torch"], "cases": [{"synthetic": True} for _ in range(36)]}}
                execution = {"action": "probe", "exit_code": 0, "stdout": jcs(observed).decode(), "stderr": "", "bwrap_sha256": sha256(b"synthetic-bwrap")}
            else:
                settings = loads(files["_ks_stage/execution.json"].encode())
                bad = next((name for name, data in BAD_SOURCES.items() if files["_ks_stage/kernel.py"].encode() == data), None)
                if self.accept_negative: bad = None
                worker = {"loaded": manifest["files"], "source": manifest["source"], "evaluator": manifest["evaluator"],
                    "environment": settings["environment"], "frozen_inputs_sha256": settings["frozen_inputs_sha256"], "profile": PROFILE,
                    "permissions": {"probes": [{"write_open_denied": True} for name in manifest["files"] if name.endswith(".py")]},
                    "result": {"build": "pass", "correctness": {"passed": bad is None, "basic_calls": [
                        {"case_id": "synthetic_" + str(i // 2), "checks": {BAD_CHECKS[bad]: False} if bad else {"synthetic": True}} for i in range(72)]},
                               "timing": {"complete": True, "soft_goal": "unmet", "synthetic": True} if bad is None else None}}
                execution = {"action": "execute", "exit_code": 0, "stdout": jcs(worker).decode(), "stderr": "", "bwrap_sha256": sha256(b"synthetic-bwrap")}
            self.native[job] = {"job_id": job, "status": "succeeded", "finished_at": "2026-11-20T00:00:00Z", "cleanup_pending": False,
                "fixture": "SYNTHETIC; no real execution", "result": {"exit_code": 0, "stdout": frame(execution), "stderr": ""}}
            if self.lost_ack: raise ConnectionError("SYNTHETIC lost acknowledgment")
            return {"status": 202, "body": jcs({"job_id": job})}
        if self.unknown: return {"status": 404, "body": b'{}'}
        job = path.split("/")[-1].split("?")[0]
        return {"status": 200, "body": jcs(self.native[job])}


def context(tmp_path):
    target = synthetic_target()
    remote = {"channel": PROFILE, "gpu": "synthetic-gpu", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    demo = Demo.create(tmp_path / "demo", start="fixed", environment=jcs(target), remote=remote)
    q = Bundle(demo.root).document(demo.config["requests"]["fixed"])
    q["budget"]["limits"]["evaluator_jobs"] = 7
    p = Publication(demo.root); ref = p.put_json("qualification-request.json", q); p.commit([ref])
    durable_write(demo.root / "qualification.request.json", jcs(q))
    with use_authority(demo.owner.path):
        ctx = api.prepare_run("implement", request=demo.root / "qualification.request.json", out=demo.root / "qualification", backend_identity={"synthetic": "qualification test"})
    return demo, ctx, target, remote


def test_qualification_same_k1_owner_replay_and_concrete_channel(tmp_path, monkeypatch):
    demo, ctx, target, remote = context(tmp_path)
    wire = Wire(target)
    with use_authority(demo.owner.path): result = qualify(ctx, remote=remote, transport=wire)
    assert result["status"] == "completed" and result["usage"]["evaluator_jobs"] == 6
    with use_authority(demo.owner.path): replay = qualify(ctx, remote=remote, transport=wire)
    assert replay["usage"]["evaluator_jobs"] == 6 and len(wire.posts) == 6
    pinned = {**remote, "bwrap_sha256": sha256(b"synthetic-bwrap")}
    env, frozen = read_qualification(result["qualification"], context=ctx, remote=pinned, transport=wire.identity())
    assert env["name"] == "Synthetic GPU" and len(loads(frozen)["cases"]) == 36
    profile = Bundle(demo.root).document(demo.config["reference_profile"])
    profile["parameters"]["evaluation"] = selection(remote=pinned, qualification=result["qualification"])
    channel = AgateChannel(profile, ctx.root / ".runtime/next-channel", transport=wire)
    # Existing handwritten R0 is intentionally unbound. Target declaration never
    # silently edits it or grants actual GPU admission.
    assert channel.admission(ctx)["status"] == "unknown"
    # Isolate wrapper's live path after static scope was independently checked;
    # all actual K1 identity/frame/owner logic below remains in use.
    from kernelschema.evaluation import RowSoftmaxTask
    monkeypatch.setattr(RowSoftmaxTask, "validate_scope", lambda *a, **k: {"status": "pass"})
    from kernelschema import template_api
    original_effective = template_api.effective_document
    def bound_hardware(*args, **kwargs):
        document = original_effective(*args, **kwargs)
        document["CONTRACT"]["Hardware_specification"] = target["hardware_specification"]
        return document
    monkeypatch.setattr(template_api, "effective_document", bound_hardware)
    assert channel.admission(ctx)["status"] == "pass"
    from kernelschema.budget import cost
    reservation = demo.owner.reserve(ctx.run_id, ctx.run_id + "_wrapper", role="evaluator", purpose="synthetic wrapper", amount=cost(evaluator_jobs=1, active_ms=1000))
    op = {"id": ctx.run_id + "_wrapper", "source": ctx.objects["task"]["starter_source"], "reservation": reservation}
    op["stage"] = stage(ctx.bundle, ctx.request["task"], op["source"], ctx.root / "wrapper-stage", scope=ctx.request["scope"])
    op["locator"] = channel.locator(op["id"])
    channel.prepare(ctx, op)
    channel.submit(ctx, op, authorize=lambda: demo.owner.authorize(reservation))
    settled = channel.settle(ctx, op, Publication(ctx.root))
    demo.owner.absorb(ctx.run_id, settled["usage_row"], terminated=True)
    report = ctx.bundle.document(settled["report"])
    assert report["subject"]["source_sha256"] == op["source"]["sha256"]
    assert report["extensions"]["information"]["performance_goal"] == "unmet"
    channel.validate(ctx, op)
    channel.submit(ctx, op, authorize=lambda: pytest.fail("replay cannot reauthorize"))
    assert len(wire.posts) == 7


def test_qualification_unknown_original_job_and_lost_ack(tmp_path):
    demo, ctx, target, remote = context(tmp_path)
    wire = Wire(target); wire.unknown = True
    with use_authority(demo.owner.path): result = qualify(ctx, remote=remote, transport=wire)
    assert result["recovery"] == "settle_only" and len(wire.posts) == 1
    with use_authority(demo.owner.path): assert qualify(ctx, remote=remote, transport=wire)["recovery"] == "settle_only"
    assert len(wire.posts) == 1
    wire.unknown = False
    with use_authority(demo.owner.path): assert qualify(ctx, remote=remote, transport=wire)["status"] == "completed"
    assert len(wire.posts) == 6


def test_missing_qualification_and_expired_submission_zero_posts(tmp_path):
    demo, ctx, target, remote = context(tmp_path)
    wire = Wire(target)
    with pytest.raises(UnsupportedError, match="missing Agate"):
        read_qualification(None, context=ctx, remote=remote, transport=wire.identity())
    executor = AgateDevExecutor(tmp_path / "jobs", wire)
    staged = stage(ctx.bundle, ctx.request["task"], ctx.objects["task"]["baseline_source"], tmp_path / "stage", scope=ctx.request["scope"])
    executor.prepare(staged, profile=remote, operation_id="expired", submission_key="expired", owner="synthetic", action="probe",
        reservation={"owner": "synthetic", "evaluator_jobs": 1, "active_ms": 10000, "deadline_utc_ms": time.time_ns() // 1000000 - 1})
    with pytest.raises(AdmissionDenied): executor.submit("expired", authorize=lambda: pytest.fail("expired launch"))
    assert not wire.posts


def test_suite_does_not_promote_baseline_when_adapter_accepts_bad_code(tmp_path):
    demo, ctx, target, remote = context(tmp_path)
    wire = Wire(target); wire.accept_negative = True
    with use_authority(demo.owner.path): result = qualify(ctx, remote=remote, transport=wire)
    assert result["status"] == "blocked" and result["qualification"] is None
    assert "negative control" in result["reason"] and result["usage"]["evaluator_jobs"] == 6
    assert ctx.bundle.read(result["evidence"])
    with use_authority(demo.owner.path): replay = qualify(ctx, remote=remote, transport=wire)
    assert replay["status"] == "blocked" and len(wire.posts) == 6


def test_lost_qualification_ack_stays_fenced_without_second_post(tmp_path):
    demo, ctx, target, remote = context(tmp_path)
    wire = Wire(target); wire.lost_ack = True
    with use_authority(demo.owner.path): first = qualify(ctx, remote=remote, transport=wire)
    with use_authority(demo.owner.path): second = qualify(ctx, remote=remote, transport=wire)
    assert first["recovery"] == second["recovery"] == "settle_only"
    assert first["usage"]["unsettled"] and len(wire.posts) == 1


def test_driver_qualification_entry_preserves_same_parent_and_selected_records(tmp_path):
    target = synthetic_target()
    remote = {"channel": PROFILE, "gpu": "synthetic-gpu", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    demo = Demo.create(tmp_path / "demo", environment=jcs(target), remote=remote)
    wire = Wire(target)
    result = demo.prepare_agate(transport=wire)
    assert result["status"] == "completed" and demo.summary()["qualification_usage"]["evaluator_jobs"] == 6
    q = api.open_run("implement", demo.root / "qualification").request
    assert q["budget"]["parent"] == demo.config["parent"] and q["budget"]["limits"]["evaluator_jobs"] == 6
    replay = Demo(demo.root).prepare_agate(transport=wire)
    assert replay["status"] == "completed" and len(wire.posts) == 6
    demo.export(tmp_path / "selected")
    assert Bundle(tmp_path / "selected").read(demo.state["qualification_evidence"])


def test_remote_queue_deadline_refuses_worker_start(tmp_path, monkeypatch):
    from kernelschema.evaluation import _remote
    profile = {"python": "/unused/python", "bwrap": "/unused/bwrap", "timeout_s": 30, "deadline_utc_ms": 5000}
    (tmp_path / "remote.json").write_bytes(jcs(profile))
    monkeypatch.setattr(_remote, "__file__", str(tmp_path / "_remote.py"))
    monkeypatch.setattr(_remote.sys, "argv", ["_remote.py", "probe"])
    monkeypatch.setattr(_remote.time, "time_ns", lambda: 6000 * 1000000)
    monkeypatch.setattr(_remote.subprocess, "run", lambda *a, **k: pytest.fail("expired remote job cannot start a worker"))
    with pytest.raises(TimeoutError, match="owner deadline"): _remote.main()
    monkeypatch.setattr(_remote.time, "time_ns", lambda: 2500 * 1000000)
    assert _remote.remaining_timeout(profile) == 2


def test_unbound_target_refused_before_any_qualification_charge(tmp_path):
    remote = {"channel": PROFILE, "gpu": "synthetic-gpu", "python": "/synthetic/python", "bwrap": "/synthetic/bwrap"}
    demo = Demo.create(tmp_path / "demo", remote=remote)
    wire = Wire(synthetic_target())
    with pytest.raises(UnsupportedError, match="ks.row-softmax.target.v1"):
        demo.prepare_agate(transport=wire)
    assert not wire.posts and not demo.owner.operations()
