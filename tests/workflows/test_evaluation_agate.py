"""Documented dev wire exercised with fake HTTP, never a GPU/service receipt."""
import base64
import gzip
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from kernelschema.artifacts import Bundle, Publication, import_bundle
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs, loads, sha256
from kernelschema.evaluation.agate import AgateDevExecutor, AgateHTTP, MARKER, PROFILE, REJECTION_RECORDS, _unframe

spec=importlib.util.spec_from_file_location("agate_fixture",Path(__file__).parent/"fixtures/evaluation/build.py")
fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
REJECTION_BODY = (Path(__file__).parent / "fixtures/evaluation/gateway-rejection.raw").read_bytes()


def reject_post(transport, *, status=400, body=REJECTION_BODY):
    def request(method, path, payload=None):
        assert method == "POST", "pre-job rejection must not query a nonexistent worker"
        transport.posts.append((path, payload))
        return {"status": status, "body": body}
    transport.request = request


def frame(value):
    raw=jcs(value)
    return MARKER+json.dumps({"bytes":len(raw),"sha256":sha256(raw),"gzip_base64":base64.b64encode(gzip.compress(raw,mtime=0)).decode()})


class FakeTransport:
    def __init__(self):
        self.posts=[]; self.gets=[]; self.status=404; self.lose_ack=False
        self.raw=b'{"reason":"job_not_found"}'
        self.channel="synthetic-gateway"
    def identity(self): return {"channel":self.channel,"synthetic":True}
    def request(self,method,path,payload=None):
        if method=="POST":
            self.posts.append((path,payload))
            if self.lose_ack: raise ConnectionError("response lost after accepted POST")
            return {"status":202,"body":b'{ "job_id" : "dv_original", "status":"queued" }'}
        self.gets.append(path)
        return {"status":self.status,"body":self.raw}
    def terminal(self,stdout=None,**changes):
        native={"job_id":"dv_original","status":"succeeded","finished_at":"2026-11-20T00:00:00Z","cleanup_pending":False,
                "result":{"exit_code":0,"stdout":stdout or frame({"action":"probe","exit_code":0,"stdout":"{}","stderr":""}),"stderr":""}}
        native.update(changes); self.status=200; self.raw=json.dumps(native,indent=2).encode()


def prepared(tmp_path):
    f,s=fixture.staged(tmp_path)
    transport=FakeTransport(); executor=AgateDevExecutor(tmp_path/"jobs",transport)
    arguments=dict(profile={"channel":PROFILE,"gpu":"H20","python":"/usr/bin/python3","bwrap":"/usr/bin/bwrap","synthetic":True},
        operation_id="probe",submission_key="frozen-key",owner="parent",reservation={"owner":"parent","evaluator_jobs":1,"active_ms":10000},action="probe")
    executor.prepare(s,**arguments)
    return f,s,transport,executor,arguments


def test_exact_dev_request_unknown_404_cancel_then_original_terminal(tmp_path):
    f,s,t,e,args=prepared(tmp_path)
    assert e.submit("probe")["state"] == "unknown"
    request=t.posts[0][1]
    assert t.posts[0][0] == "/v1/jobs/dev"
    assert request["files"]["_ks_stage/kernel.py"].encode() == f["bundle"].source(f["candidate"])
    assert request["idempotency_key"] == "frozen-key" and request["replay_policy"] == "never" and request["recycle"]
    assert e.submit("probe")["state"] == "unknown" and len(t.posts)==1
    t.terminal(status="cancelled",cleanup_pending=True)
    assert e.query("probe")["state"] == "unknown"
    e.prepare(s,**{**args,"operation_id":"replacement","submission_key":"other-key"})
    with pytest.raises(ProtocolError,match="fences"): e.submit("replacement")
    t.terminal()
    observed=e.query("probe")
    assert observed["state"] == "terminal" and observed["execution"]["action"] == "probe"
    assert (e.root/"probe/terminal.raw").read_bytes() == t.raw
    settled=e.settle("probe",Publication(f["bundle"].root))
    assert settled["new_receipt"] and settled["report"] is None
    assert settled["usage_row"]["evaluator_jobs"] == 1 and settled["usage_row"]["gpu_ms"] is None
    assert not e.settle("probe",Publication(f["bundle"].root))["new_receipt"]
    assert len(t.posts)==1


def test_lost_submit_ack_never_posts_again_or_switches(tmp_path):
    _,_,t,e,_=prepared(tmp_path); t.lose_ack=True
    assert e.submit("probe")["state"] == "unknown"
    assert e.submit("probe")["state"] == "unknown"
    assert len(t.posts)==1 and not t.gets
    t.channel="replacement-channel"
    with pytest.raises(ProtocolError,match="channel"): e.query("probe")


def test_original_http_rejection_settles_zero_jobs_and_exports_exact_provenance(tmp_path):
    pytest.importorskip("atrex_gateway_client")
    f, staged, _, _, arguments = prepared(tmp_path)
    requests = []
    def opener(request, timeout):
        requests.append((request.method, request.full_url, request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(REJECTION_BODY))
    connection = SimpleNamespace(url="https://example.invalid", environment_id="", route_label="", headers=lambda: {})
    http = AgateHTTP(connection, SimpleNamespace(headers=lambda: {}), opener=opener)
    executor = AgateDevExecutor(tmp_path / "http-jobs", http)
    executor.prepare(staged, **arguments)
    facts = executor.submit("probe")
    assert facts["state"] == "terminal" and facts["outcome"] == "rejected_before_job"
    assert "job_id" not in facts and "native" not in facts and facts["execution"] is None
    before = {name: (executor.root / "probe" / name).read_bytes() for name in REJECTION_RECORDS}
    assert before["acceptance.raw"] == REJECTION_BODY and loads(before["acceptance-status.json"]) == {"status": 400}
    assert not (executor.root / "probe/terminal.raw").exists()
    assert requests == [("POST", "https://example.invalid/v1/jobs/dev", jcs(loads(before["request.json"])))]
    fixture_origin = loads((Path(__file__).parent / "fixtures/evaluation/gateway-rejection.json").read_bytes())
    assert sha256(REJECTION_BODY) == fixture_origin["body_sha256"] and len(REJECTION_BODY) == fixture_origin["body_bytes"]
    http.opener = lambda *a, **k: pytest.fail("saved rejection recovery must not use the network")
    restored = AgateDevExecutor(executor.root, http)
    assert restored.submit("probe") == facts  # Same original key, no new authorization or POST.
    settled = restored.settle("probe", Publication(f["bundle"].root))
    assert settled["report"] is None and settled["outcome"] == "rejected_before_job"
    row = settled["usage_row"]
    assert (row["calls"], row["evaluator_jobs"], row["tokens"], row["gpu_ms"]) == (0, 0, 0, 0)
    assert row["receipt_id"] == row["receipt"]["sha256"] == facts["receipt_id"]
    # Exporting the ordinary usage index alone retains all original proof bytes.
    p = Publication(f["bundle"].root)
    usage = p.put_json("selected-usage.json", {"protocol": "ks.workflow.v1", "kind": "usage_index", "entries": [row]})
    p.commit([usage]); import_bundle(f["bundle"], tmp_path / "selected", [usage])
    evidence = loads(Bundle(tmp_path / "selected").read(row["receipt"]))
    assert evidence == facts["rejection"]
    assert evidence["response"]["details"]["rule_ids"] == ["process_root_escape"]
    assert evidence["submitted_request_sha256"] == sha256(requests[0][2])
    for name, raw in before.items():
        assert evidence["original_records"][name].encode() == raw
        assert evidence["original_sha256"][name] == sha256(raw)
        assert (executor.root / "probe" / name).read_bytes() == raw
    assert restored.settle("probe", Publication(f["bundle"].root)) == {**settled, "new_receipt": False}


@pytest.mark.parametrize("status,error_class", [(400, "validation"), (422, "validation"), (401, "auth"), (403, "auth")])
def test_complete_pre_job_validation_auth_refusals(tmp_path, status, error_class):
    f, _, transport, executor, _ = prepared(tmp_path)
    body = jcs({**loads(REJECTION_BODY), "error_class": error_class})
    reject_post(transport, status=status, body=body)
    assert executor.submit("probe")["outcome"] == "rejected_before_job"
    assert executor.settle("probe", Publication(f["bundle"].root))["usage_row"]["evaluator_jobs"] == 0
    assert len(transport.posts) == 1 and not transport.gets


@pytest.mark.parametrize("status,body", [
    *((s, REJECTION_BODY) for s in (202, 404, 409, 429, 500, 502)),
    ("400", REJECTION_BODY), (400, b'{"reason":"dev_sensitive_target_denied"}'),
    (400, b'{"error_class":"validation",'), (400, b'{}'),
    (400, jcs({**loads(REJECTION_BODY), "error_class": "infra"})),
    (400, jcs({k: v for k, v in loads(REJECTION_BODY).items() if k != "trace_id"})),
    (400, jcs({**loads(REJECTION_BODY), "job_id": ""})),
    (400, jcs({**loads(REJECTION_BODY), "details": {"job_id": "dv_existing"}})),
    (400, jcs({**loads(REJECTION_BODY), "details": {"cleanup_pending": True}})),
    (400, jcs({**loads(REJECTION_BODY), "status": "queued"})),
    (400, REJECTION_BODY[:-1] + b',"job_id":"dv_existing","job_id":null}'),
])
def test_incomplete_ambiguous_or_non_admission_errors_stay_unknown(tmp_path, status, body):
    f, staged, transport, executor, arguments = prepared(tmp_path)
    reject_post(transport, status=status, body=body)
    assert executor.submit("probe")["state"] == "unknown"
    assert executor.settle("probe", Publication(f["bundle"].root))["recovery"] == "settle_only"
    executor.prepare(staged, **{**arguments, "operation_id": "next", "submission_key": "next"})
    with pytest.raises(ProtocolError, match="fences"):
        executor.submit("next")
    assert len(transport.posts) == 1 and not transport.gets
    assert not (executor.root / "probe/absorption.json").exists()


@pytest.mark.parametrize("missing", ["acceptance-status.json", "submission-started.json", "submitted_request_sha256", "submission_key"])
def test_rejection_requires_original_post_status_and_request_binding(tmp_path, missing):
    _, _, transport, executor, _ = prepared(tmp_path)
    reject_post(transport)
    executor.submit("probe")
    if missing.endswith(".json"):
        (executor.root / "probe" / missing).unlink()
    else:
        path = executor.root / "probe/submission-started.json"
        started = loads(path.read_bytes()); started[missing] = "changed-original-binding"
        path.write_bytes(jcs(started))
    assert executor.query("probe")["state"] == "unknown"
    assert not (executor.root / "probe/absorption.json").exists()


def test_rejection_does_not_fence_a_separately_authorized_operation(tmp_path):
    _, staged, transport, executor, arguments = prepared(tmp_path)
    reject_post(transport)
    executor.submit("probe")
    executor.prepare(staged, **{**arguments, "operation_id": "other", "submission_key": "other-key"})
    assert executor.submit("other")["outcome"] == "rejected_before_job"
    assert len(transport.posts) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 500])
def test_accepted_job_later_http_error_cannot_become_zero_jobs(tmp_path, status):
    f, _, transport, executor, _ = prepared(tmp_path)
    transport.status, transport.raw = status, REJECTION_BODY
    assert executor.submit("probe")["state"] == "unknown"
    assert executor.query("probe")["job_id"] == "dv_original"
    transport.terminal()
    settled = executor.settle("probe", Publication(f["bundle"].root))
    assert settled["usage_row"]["evaluator_jobs"] == 1 and "outcome" not in settled
    assert len(transport.posts) == 1
    # Damaged/replaced acknowledgment cannot erase an already observed locator.
    (executor.root / "probe/acceptance.raw").write_bytes(REJECTION_BODY)
    (executor.root / "probe/acceptance-status.json").write_bytes(jcs({"status": 400}))
    with pytest.raises(ProtocolError, match="accepted job acknowledgment changed"):
        executor.query("probe")


def test_original_unknown_recovers_only_from_saved_refusal_with_one_owner_receipt(tmp_path, monkeypatch):
    from kernelschema.budget import OwnerAccount, cost
    f, staged, transport, _, arguments = prepared(tmp_path)
    clock = [1000]
    monkeypatch.setattr("kernelschema.evaluation.agate.time.time_ns", lambda: clock[0] * 1000000)
    limits = {"calls": {"coder": 0, "reviewer": 0, "planner": 0, "extract": 0},
              "model_calls": 0, "evaluator_jobs": 1, "tokens": 0, "active_ms": 10000}
    account = OwnerAccount.create(tmp_path / "synthetic-owner.json", owner="parent", limits=limits, clock=lambda: clock[0])
    account.register("run", sha256(b"synthetic-request"), limits, root=str(f["bundle"].root))
    account.begin("run")
    reservation = account.reserve("run", "probe", role="evaluator", purpose="probe", amount=cost(evaluator_jobs=1, active_ms=1000))
    executor = AgateDevExecutor(tmp_path / "owned-jobs", transport)
    executor.prepare(staged, **{**arguments, "reservation": reservation})
    reject_post(transport)
    executor.submit("probe", authorize=lambda: account.authorize(reservation))
    # Emulate the old adapter's unknown classification without touching any real run.
    account.mark_unknown("probe", "old adapter could not classify complete HTTP refusal")
    assert account.fences() == ["probe"]
    clock[0] += 357
    transport.request = lambda *a, **k: pytest.fail("original response alone must recover")
    recovered = AgateDevExecutor(executor.root, transport).settle("probe", Publication(f["bundle"].root))
    row = recovered["usage_row"]
    assert account.absorb("run", row, terminated=True)
    account.end("run")
    clock[0] += 1000
    assert not account.absorb("run", row, terminated=True)
    usage = account.usage("run", Publication(f["bundle"].root))
    assert usage["active_ms"] == 357 and usage["evaluator_jobs"] == 0 and usage["unsettled"] == []
    assert all(v == 0 for v in usage["calls"].values()) and usage["tokens"] == 0
    op = account.operations()["probe"]
    assert op["terminated"] and not op["released"] and op["history"] == [row] and op["reservation"] == reservation
    assert len(account.operations()) == len(transport.posts) == 1
    assert loads(account.path.read_bytes())["active_since"] is None


def test_rejection_publication_recovers_original_receipt_after_interruption(tmp_path, monkeypatch):
    import kernelschema.evaluation.agate as agate
    f, _, transport, executor, _ = prepared(tmp_path)
    reject_post(transport)
    executor.submit("probe")
    original_write = agate.durable_write
    def interrupted(path, data, **kwargs):
        if path.name == "absorption.json":
            raise OSError("synthetic crash after original receipt publication")
        return original_write(path, data, **kwargs)
    with monkeypatch.context() as interrupted_process:
        interrupted_process.setattr(agate, "durable_write", interrupted)
        with pytest.raises(OSError, match="synthetic crash"):
            executor.settle("probe", Publication(f["bundle"].root))
    binding = loads((executor.root / "probe/publication.json").read_bytes())
    transport.request = lambda *a, **k: pytest.fail("publication recovery is offline")
    settled = executor.settle("probe", Publication(f["bundle"].root))
    assert binding["id"] in settled["usage_row"]["receipt"]["path"]
    assert not executor.settle("probe", Publication(f["bundle"].root))["new_receipt"]
    assert len(transport.posts) == 1


def test_lost_ack_with_late_original_job_locator_still_costs_one_job(tmp_path):
    f, _, transport, executor, _ = prepared(tmp_path)
    transport.lose_ack = True
    assert executor.submit("probe")["state"] == "unknown"
    # Only a retained original acknowledgment can supply the missing locator.
    (executor.root / "probe/acceptance.raw").write_bytes(b'{"job_id":"dv_original"}')
    (executor.root / "probe/acceptance-status.json").write_bytes(jcs({"status": 202}))
    transport.terminal()
    assert executor.submit("probe")["state"] == "terminal"
    assert executor.settle("probe", Publication(f["bundle"].root))["usage_row"]["evaluator_jobs"] == 1
    assert len(transport.posts) == 1


def test_prior_job_receipt_prevents_zero_job_reinterpretation_after_record_loss(tmp_path):
    f, _, transport, executor, _ = prepared(tmp_path)
    transport.terminal()
    executor.submit("probe")
    settled = executor.settle("probe", Publication(f["bundle"].root))
    job = executor.root / "probe"
    receipt = (job / "absorption.json").read_bytes()
    assert settled["usage_row"]["evaluator_jobs"] == 1
    for path in [job / "accepted-job.json", job / "terminal.raw", *job.glob("query-*.raw")]:
        path.unlink()
    (job / "acceptance.raw").write_bytes(REJECTION_BODY)
    (job / "acceptance-status.json").write_bytes(jcs({"status": 400}))
    transport.request = lambda *a, **k: pytest.fail("lost original job state cannot launch or query another job")
    assert executor.settle("probe", Publication(f["bundle"].root))["state"] == "unknown"
    assert (job / "absorption.json").read_bytes() == receipt and len(transport.posts) == 1


def test_truncation_keeps_original_terminal_without_pass(tmp_path):
    f,_,t,e,_=prepared(tmp_path)
    t.terminal(stdout=frame({"action":"probe","exit_code":0})[:-12])
    result=e.submit("probe")
    assert result["state"]=="terminal" and result["execution"] is None and result["diagnostic"]
    assert e.settle("probe",Publication(f["bundle"].root))["report"] is None


@pytest.mark.parametrize("body", [b"<html>temporary gateway error", b"{", b"[]", b"null"])
def test_malformed_original_ack_is_unknown_without_resubmit(tmp_path, body):
    _, _, transport, executor, _ = prepared(tmp_path)
    original = transport.request
    def request(method, path, payload=None):
        response = original(method, path, payload)
        return {"status": 502, "body": body} if method == "POST" else response
    transport.request = request
    assert executor.submit("probe")["state"] == "unknown"
    assert executor.submit("probe")["state"] == "unknown"
    assert len(transport.posts) == 1 and not transport.gets
    assert (executor.root / "probe/acceptance.raw").read_bytes() == body


@pytest.mark.parametrize("body", [b"{", b"[]", b"null"])
def test_malformed_query_preserves_locator_until_original_terminal(tmp_path, body):
    _, _, transport, executor, _ = prepared(tmp_path)
    transport.status, transport.raw = 200, body
    assert executor.submit("probe")["state"] == "unknown"
    assert not (executor.root / "probe/terminal.raw").exists()
    transport.terminal()
    assert executor.submit("probe")["state"] == "terminal"
    assert len(transport.posts) == 1


def test_original_query_rejects_changed_prepared_upload(tmp_path):
    _, _, transport, executor, _ = prepared(tmp_path)
    assert executor.submit("probe")["state"] == "unknown"
    path = executor.root / "probe/request.json"
    request = json.loads(path.read_bytes())
    request["files"]["_ks_stage/kernel.py"] += "\n# changed original prepared bytes\n"
    path.write_bytes(jcs(request))
    transport.terminal()
    with pytest.raises(ProtocolError, match="prepared request changed"):
        executor.query("probe")
    assert len(transport.posts) == 1


@pytest.mark.parametrize("stdout", [MARKER + "[]", frame([]), frame({"action": "probe", "exit_code": 0})[:-1]])
def test_malformed_evaluator_frame_settles_original_fee_without_report(tmp_path, stdout):
    fixture_value, _, transport, executor, _ = prepared(tmp_path)
    transport.terminal(stdout=stdout)
    result = executor.submit("probe")
    assert result["state"] == "terminal" and result["execution"] is None
    settled = executor.settle("probe", Publication(fixture_value["bundle"].root))
    assert settled["usage_row"]["evaluator_jobs"] == 1 and settled["report"] is None
    assert len(transport.posts) == 1


def test_execute_requires_actual_capture_and_preserves_wrong_source_as_unknown(tmp_path):
    _,s,t,e,args=prepared(tmp_path)
    profile={**args["profile"],"bwrap_sha256":sha256(b"synthetic-bwrap")}
    changed={**args,"profile":profile,"operation_id":"execute","action":"execute"}
    with pytest.raises(UnsupportedError,match="actual CUDA"): e.prepare(s,**changed)
    inputs=jcs({"torch_version":"synthetic","cases":[{"synthetic":True} for _ in range(36)]})
    environment={"mode":"cuda","torch":"synthetic"}
    e.prepare(s,**changed,frozen_inputs=inputs,environment=environment)
    worker={"loaded":s["manifest"]["files"],"source":{**s["manifest"]["source"],"sha256":sha256(b"old candidate")}}
    t.terminal(stdout=frame({"action":"execute","exit_code":0,"bwrap_sha256":profile["bwrap_sha256"],"stdout":jcs(worker).decode()}))
    result=e.submit("execute")
    assert result["execution"] is None and "differs" in result["diagnostic"]


def test_official_auth_connection_raw_http_no_retry():
    sdk=pytest.importorskip("atrex_gateway_client")
    from atrex_gateway_client.auth import NoAuth
    from atrex_gateway_client.profiles import Connection
    calls=[]
    class Response(io.BytesIO):
        status=202; headers={}
    def opener(request,timeout):
        calls.append(request)
        return Response(b'{ "job_id" : "dv_one" }')
    client=AgateHTTP(Connection("https://example.invalid"),NoAuth(),opener=opener)
    raw=client.request("POST","/v1/jobs/dev",{"idempotency_key":"same"})
    assert raw["body"]==b'{ "job_id" : "dv_one" }' and len(calls)==1
    assert calls[0].data==jcs({"idempotency_key":"same"})


def test_official_env_pair_without_default_file_and_secret_free_identity(tmp_path, monkeypatch):
    pytest.importorskip("atrex_gateway_client")
    from atrex_gateway_client import profiles
    from kernelschema.evaluation.channel import transport_from_config
    monkeypatch.setattr(profiles, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("KS_AGATE_CONFIG", raising=False)
    monkeypatch.setenv("AGATE_AK", "test-env-access")
    monkeypatch.setenv("AGATE_SK", "test-env-secret")
    client = transport_from_config()
    assert client.connection.url == profiles.PROFILES["prod"]
    assert client.auth.headers()["Access-Key"] == "test-env-access"
    assert client.auth.secret_key == "test-env-secret"
    identity = json.dumps(client.identity())
    assert "test-env-access" not in identity and "test-env-secret" not in identity
    assert client.identity()["post_retries"] == 0


def test_official_explicit_file_and_named_route_with_env_precedence(tmp_path, monkeypatch):
    pytest.importorskip("atrex_gateway_client")
    from kernelschema.evaluation.channel import transport_from_config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ak": "test-file-access", "sk": "test-file-secret", "profiles": {
        "routed": {"url": "https://example.invalid", "environment_id": "h20-test", "route_label": "route-test"}}}))
    monkeypatch.setenv("KS_AGATE_CONFIG", str(path))
    monkeypatch.delenv("AGATE_AK", raising=False)
    monkeypatch.delenv("AGATE_SK", raising=False)
    client = transport_from_config("routed")
    assert client.auth.access_key == "test-file-access" and client.auth.secret_key == "test-file-secret"
    assert client.connection.headers()["x-agate-environment-id"] == "h20-test"
    assert client.identity()["route_label"] == "route-test"
    monkeypatch.setenv("AGATE_AK", "test-env-access")
    monkeypatch.setenv("AGATE_SK", "test-env-secret")
    client = transport_from_config("routed")
    assert client.auth.access_key == "test-env-access" and client.auth.secret_key == "test-env-secret"
    assert client.connection.environment_id == "h20-test"
    monkeypatch.setenv("KS_AGATE_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(UnsupportedError, match="explicit Agate configuration"):
        transport_from_config()


@pytest.mark.parametrize("pair", [{}, {"AGATE_AK": "test-access"}, {"AGATE_SK": "test-secret"},
                                  {"AGATE_AK": "", "AGATE_SK": "test-secret"}])
def test_official_missing_or_incomplete_auth_refused(tmp_path, monkeypatch, pair):
    pytest.importorskip("atrex_gateway_client")
    from atrex_gateway_client import profiles
    monkeypatch.setattr(profiles, "CONFIG_PATH", tmp_path / "missing.json")
    for key in ("AGATE_AK", "AGATE_SK"):
        monkeypatch.delenv(key, raising=False)
    for key, value in pair.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(UnsupportedError, match="complete AGATE_AK/AGATE_SK"):
        AgateHTTP.from_config()
    if pair:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ak": "test-file-access", "sk": "test-file-secret"}))
        with pytest.raises(UnsupportedError, match="complete AGATE_AK/AGATE_SK"):
            AgateHTTP.from_config(path)
