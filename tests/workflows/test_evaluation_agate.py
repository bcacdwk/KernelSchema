"""Documented dev wire exercised with fake HTTP, never a GPU/service receipt."""
import base64
import gzip
import importlib.util
import io
import json
from pathlib import Path

import pytest

from kernelschema.artifacts import Publication
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs, sha256
from kernelschema.evaluation.agate import AgateDevExecutor, AgateHTTP, MARKER, PROFILE, _unframe

spec=importlib.util.spec_from_file_location("agate_fixture",Path(__file__).parent/"fixtures/evaluation/build.py")
fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


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
