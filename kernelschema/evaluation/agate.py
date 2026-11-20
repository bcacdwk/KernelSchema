"""Bounded Agate dev adapter, based on skill 0.15/client 0.14.37 wire fields.

CPU transport-tested, not service qualified. No POST retry or legacy route.
"""
from __future__ import annotations
import base64
import gzip
import importlib.metadata
import io
import json
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request

from ..artifacts import Publication, durable_mkdir, durable_write, read_regular
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from .task import verify_stage

PROFILE = "agate-dev-bwrap-v1"
MARKER = "KS_EVALUATOR_V1 "


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AgateHTTP:
    """Official SDK Connection/AuthProvider, documented HTTP, exact response bytes.

    SDK submit_job discards original wire bytes and may retry. This transport
    uses its public auth/connection objects, with exactly one request attempt.
    Credentials stay inside the auth provider and are never journaled.
    """
    def __init__(self, connection, auth, *, timeout=30, opener=None):
        parsed=urllib.parse.urlsplit(connection.url)
        if parsed.scheme not in ("https","http") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ProtocolError("Agate connection must be an HTTP origin without embedded credentials")
        self.connection, self.auth, self.timeout = connection, auth, timeout
        self.opener=opener or urllib.request.build_opener(_NoRedirect()).open

    @classmethod
    def from_config(cls, config_path, *, profile="prod", timeout=30):
        from atrex_gateway_client.auth import AkSkAuth
        from atrex_gateway_client.profiles import load_config, resolve_profile
        config = load_config(Path(config_path))
        if not config.get("ak") or not config.get("sk"):
            raise UnsupportedError("selected configuration does not supply Agate AK/SK")
        return cls(resolve_profile(profile,config), AkSkAuth(config["ak"],config["sk"]),timeout=timeout)

    def identity(self):
        return {"url":self.connection.url,"environment_id":self.connection.environment_id,
                "route_label":self.connection.route_label,"sdk":importlib.metadata.version("atrex-gateway-client"),"post_retries":0}

    def request(self, method, path, payload=None):
        headers = {**self.auth.headers(), **self.connection.headers(), "content-type":"application/json"}
        request = urllib.request.Request(self.connection.url.rstrip("/") + path, method=method,
                    data=None if payload is None else jcs(payload), headers=headers)
        try:
            response = self.opener(request,timeout=self.timeout)
        except urllib.error.HTTPError as response:
            with response: return {"status":response.code,"body":response.read()}
        with response:
            if self.connection.environment_id and response.headers.get("x-agate-environment-id") != self.connection.environment_id:
                raise ProtocolError("routed environment mismatch; never replay mutation")
            return {"status":response.status,"body":response.read()}


def _unframe(stdout):
    if not isinstance(stdout,str) or not stdout.startswith(MARKER) or len(stdout) > 48 * 1024 * 1024:
        raise ProtocolError("complete evaluator frame absent or too large")
    try:
        envelope=json.loads(stdout[len(MARKER):])
        if type(envelope["bytes"]) is not int or not 0 <= envelope["bytes"] <= 32 * 1024 * 1024:
            raise ValueError("invalid length")
        with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(envelope["gzip_base64"],validate=True))) as stream:
            data=stream.read(32 * 1024 * 1024 + 1)
        if len(data) != envelope["bytes"] or sha256(data) != envelope["sha256"]:
            raise ValueError("truncated or changed evaluator frame")
        return loads(data)
    except (KeyError,ValueError,OSError) as error:
        raise ProtocolError("incomplete evaluator frame: " + str(error)) from error


class AgateDevExecutor:
    def __init__(self, root, transport):
        self.root=Path(root).absolute()
        durable_mkdir(self.root)
        self.transport=transport

    def capabilities(self):
        return {"profile":PROFILE,"engineering":"dev prepare/submit/query/settle implemented",
                "service_qualified":False,"Task_qualification":False,
                "requires":["configured auth and dev ACL", "server replay_policy=never", "pinned remote bwrap/namespaces/CUDA",
                            "frozen actual environment and FP32 input bytes", "complete framed stdout", "confirmed remote terminal/cleanup"],
                "missing_sdk_operation":"read-only lookup by idempotency_key; lost acknowledgment without job_id stays unknown"}

    def _job(self, operation_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",operation_id): raise ProtocolError("unsafe operation_id")
        return self.root / operation_id

    def locator(self, operation_id):
        """Original private locator available before submission acknowledgment."""
        return {"root":str(self._job(operation_id)),"profile":PROFILE,"transport":self.transport.identity()}

    def prepare(self, staged, *, profile, operation_id, submission_key, owner, reservation,
                frozen_inputs=None, environment=None, action="execute"):
        """No network; freeze exact request/key/subjects before POST.

        Probe captures only trusted CUDA environment/inputs, never candidate. It
        still costs one logical job. Execute consumes that actual prior capture.
        Caller owns the reservation, active clock and cross-channel fences.
        """
        verify_stage(staged)
        if profile.get("channel") != PROFILE or action not in ("probe","execute"):
            raise UnsupportedError("only explicit Agate dev+bwrap profile implemented")
        ms=reservation.get("active_ms")
        if not owner or not submission_key or reservation.get("owner") != owner or reservation.get("evaluator_jobs") != 1:
            raise ProtocolError("explicit caller owner/reservation/stable key required")
        if type(ms) is not int or ms < 1000: raise ProtocolError("reservation must allow at least 1s")
        ms = min(ms, 1200000)  # one dev job never exceeds the channel's 20min bound
        if any(not isinstance(profile.get(k),str) or not profile[k].startswith("/") for k in ("python","bwrap")):
            raise ProtocolError("absolute remote python/bwrap paths required")
        if (action == "execute" and not re.fullmatch(r"sha256:[0-9a-f]{64}",profile.get("bwrap_sha256","") or "")) or not profile.get("gpu"):
            raise ProtocolError("fixed GPU and actual bwrap hash required")
        root=Path(staged["root"])
        files={"_ks_stage/"+name:read_regular(root,name)[0].decode() for name in staged["manifest"]["files"]}
        files["_ks_stage/stage.json"]=jcs(staged["manifest"]).decode()
        files["_ks_stage/remote.json"]=jcs({**{k:profile.get(k) for k in ("python","bwrap","bwrap_sha256")},"timeout_s":ms//1000,
            "deadline_utc_ms":reservation.get("deadline_utc_ms")}).decode()
        if action == "execute":
            if not environment or environment.get("mode") != "cuda" or frozen_inputs is None:
                raise UnsupportedError("execute requires prior actual CUDA environment/input capture")
            inputs=loads(frozen_inputs)
            if inputs.get("torch_version") != environment.get("torch") or len(inputs.get("cases",[])) != 36:
                raise ProtocolError("frozen inputs/environment mismatch")
            files["_ks_stage/frozen_inputs.json"]=frozen_inputs.decode()
            files["_ks_stage/execution.json"]=jcs({"mode":"cuda","profile":PROFILE,"include_timing":True,
                "environment":environment,"frozen_inputs_sha256":sha256(frozen_inputs)}).decode()
        request={"spec":{"target_hardware":[profile["gpu"]],"num_gpus":1},
                 "command":shlex.join([profile["python"],"-I","-B","_ks_stage/_remote.py",action]),"files":files,"env_vars":{},
                 "timeout_s":ms//1000,"recycle":True,"replay_policy":"never","dev_intent":"custom_harness","idempotency_key":submission_key}
        intent={"operation_id":operation_id,"submission_key":submission_key,"owner":owner,"reservation":reservation,
                "profile":profile,"transport":self.transport.identity(),"stage":staged,"request_sha256":digest(request),
                "action":action,"environment":environment,"frozen_inputs_sha256":None if frozen_inputs is None else sha256(frozen_inputs)}
        job=self._job(operation_id)
        if job.exists():
            if loads(read_regular(job,"intent.json")[0]) != intent: raise ProtocolError("prepared intent differs")
        else:
            durable_mkdir(job)
            durable_write(job / "request.json",jcs(request)); durable_write(job / "intent.json",jcs(intent))
        return intent

    def submit(self, operation_id, *, authorize=None):
        job=self._job(operation_id); intent=loads(read_regular(job,"intent.json")[0])
        if intent["transport"] != self.transport.identity(): raise ProtocolError("original Agate channel changed")
        if (job / "submission-started.json").exists(): return self.query(operation_id)
        for old in self.root.iterdir():
            if old.is_dir() and old != job and (old/"submission-started.json").exists() and not (old/"terminal.raw").exists():
                raise ProtocolError("unknown original Agate job fences new submissions")
        request=loads(read_regular(job,"request.json")[0])
        if digest(request) != intent["request_sha256"]: raise ProtocolError("prepared request changed")
        reservation = intent["reservation"]
        remaining = min(reservation["active_ms"], request["timeout_s"] * 1000)
        if "deadline_utc_ms" in reservation:
            remaining = min(remaining, reservation["deadline_utc_ms"] - time.time_ns() // 1000000)
        if "deadline_monotonic_ns" in reservation:
            if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != reservation["clock_boot_id"]:
                raise ProtocolError("original Agate reservation belongs to another boot")
            remaining = min(remaining, (int(reservation["deadline_monotonic_ns"]) - time.monotonic_ns()) // 1000000)
        if remaining < 1000 or (authorize is not None and not authorize()):
            from ..runtime import AdmissionDenied
            raise AdmissionDenied("original Agate launch deadline exhausted or owner refused")
        request["timeout_s"] = remaining // 1000
        durable_write(job/"submission-started.json",jcs({"submission_key":intent["submission_key"],"request_sha256":intent["request_sha256"],
            "submitted_request_sha256":digest(request), "timeout_s":request["timeout_s"]}))
        try:
            response=self.transport.request("POST","/v1/jobs/dev",request)
            durable_write(job/"acceptance.raw",response["body"])
            durable_write(job/"acceptance-status.json",jcs({"status":response["status"]}))
        except Exception as error:
            durable_write(job/"submission-error.json",jcs({"type":type(error).__name__,"message":str(error)[:2000]}))
        return self.query(operation_id)

    def query(self, operation_id):
        job=self._job(operation_id); intent=loads(read_regular(job,"intent.json")[0])
        if intent["transport"] != self.transport.identity(): raise ProtocolError("cannot switch original channel")
        unknown={"state":"unknown","operation_id":operation_id,"submission_key":intent["submission_key"],"recovery":"settle_only"}
        if not (job/"acceptance.raw").exists():
            return {**unknown,"reason":"missing acknowledgment/job_id; no read-only key lookup API"}
        acceptance=json.loads(read_regular(job,"acceptance.raw")[0]); job_id=acceptance.get("job_id")
        if not isinstance(job_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",job_id):
            return {**unknown,"reason":"no original job locator"}
        unknown["job_id"]=job_id
        if (job/"terminal.raw").exists(): raw=read_regular(job,"terminal.raw")[0]
        else:
            try: response=self.transport.request("GET","/v1/jobs/"+job_id+"?include_spec=true")
            except Exception as error: return {**unknown,"reason":type(error).__name__}
            raw=response["body"]; event=job/("query-"+sha256(raw)[7:]+".raw")
            if not event.exists(): durable_write(event,raw)
            if response["status"] != 200: return {**unknown,"reason":"HTTP "+str(response["status"])}
        native=json.loads(raw)
        if native.get("job_id") != job_id: raise ProtocolError("different original job")
        if native.get("status") not in ("succeeded","failed","cancelled") or not native.get("finished_at") or native.get("cleanup_pending") is not False or native.get("execution_quarantined",False):
            return {**unknown,"reason":"remote terminal/quiescence unconfirmed"}
        if not (job/"terminal.raw").exists(): durable_write(job/"terminal.raw",raw)
        execution, diagnostic = interpret_terminal(intent, native)
        return {"state":"terminal","operation_id":operation_id,"job_id":job_id,"receipt_id":sha256(raw),
                "native":native,"execution":execution,"diagnostic":diagnostic}

    def settle(self, operation_id, publication: Publication):
        facts=self.query(operation_id)
        if facts["state"] != "terminal": return facts
        return _publish(self,operation_id,publication,facts)


def interpret_terminal(intent, native):
    """Same complete K1 frame/subject checks for live and retained qualification."""
    execution=None; diagnostic=None; result=native.get("result") or {}
    if native["status"] == "succeeded" and result.get("exit_code") == 0:
        try:
            execution=_unframe(result.get("stdout"))
            if execution.get("action") != intent["action"]: raise ProtocolError("different evaluator action")
            if execution.get("exit_code") == 0 and intent["action"] == "execute":
                worker=loads(execution["stdout"]); manifest=intent["stage"]["manifest"]
                if (execution.get("bwrap_sha256") != intent["profile"]["bwrap_sha256"] or
                    worker["loaded"] != manifest["files"] or worker["source"] != manifest["source"] or
                    worker["evaluator"] != manifest["evaluator"] or worker["environment"] != intent["environment"] or
                    worker["frozen_inputs_sha256"] != intent["frozen_inputs_sha256"] or worker["profile"] != PROFILE or
                    len(worker["permissions"]["probes"]) != sum(name.endswith(".py") for name in manifest["files"]) or
                    not all(p["write_open_denied"] for p in worker["permissions"]["probes"])):
                    raise ProtocolError("actual evaluator subjects/environment/permission evidence differs")
                execution["worker"]=worker
        except (ProtocolError,KeyError,ValueError) as error:
            diagnostic=str(error); execution=None
    return execution, diagnostic

AgateExecutor = AgateDevExecutor


def _publish(executor, operation_id, publication, facts):
    job=executor._job(operation_id); intent=loads(read_regular(job,"intent.json")[0])
    if (job/"absorption.json").exists():
        prior=loads(read_regular(job,"absorption.json")[0])
        if prior["receipt_id"] != facts["receipt_id"] or prior["bundle_root"] != str(publication.root):
            raise ProtocolError("receipt replay conflicts with original publication")
        return {**prior,"new_receipt":False}
    selected=job/"publication.json"
    if selected.exists():
        binding=loads(read_regular(job,"publication.json")[0])
        if binding["root"] != str(publication.root): raise ProtocolError("original bundle changed")
        publication=Publication(binding["root"],binding["id"])
    else: durable_write(selected,jcs({"root":str(publication.root),"id":publication.publication_id}))
    prefix="evaluation/"+operation_id
    native=publication.put_bytes(prefix+"/agate-terminal.json",read_regular(job,"terminal.raw")[0])
    acceptance=publication.put_bytes(prefix+"/agate-acceptance.json",read_regular(job,"acceptance.raw")[0])
    row={"producer":"kernelschema.agate:"+str(executor.root),"operation_id":operation_id,"role":"evaluator",
         "purpose":"row-softmax "+intent["action"],"receipt_id":facts["receipt_id"],"sequence":1,"receipt":native,
         "state":"settled","calls":0,"evaluator_jobs":1,"tokens":0,"gpu_ms":None}
    usage=publication.put_json(prefix+"/usage.json",{"protocol":PROTOCOL,"kind":"usage_index","entries":[row]})
    # Exact original terminal retained even if no valid worker report is available.
    report=None; roots=[usage,acceptance]; execution=facts["execution"]
    if execution and execution.get("worker"):
        worker=execution["worker"]; result=worker["result"]; manifest=intent["stage"]["manifest"]
        raw=publication.put_bytes(prefix+"/worker.json",jcs(worker))
        env=publication.put_bytes(prefix+"/environment.json",jcs(worker["environment"]))
        frozen=publication.put_bytes(prefix+"/frozen-inputs.json",loads(read_regular(job,"request.json")[0])["files"]["_ks_stage/frozen_inputs.json"].encode())
        samples=publication.put_bytes(prefix+"/samples.json",jcs(result["timing"])) if result.get("timing") else None
        check=lambda status,reason:{"status":status,"reason":reason,"receipt":raw if status in ("pass","fail") else None}
        correct=result.get("correctness")
        document={"protocol":PROTOCOL,"kind":"evaluation","operation_id":operation_id,
            "subject":{"task_sha256":manifest["task"]["sha256"],"source_sha256":manifest["source"]["sha256"],
                "evaluator_sha256":manifest["evaluator"]["sha256"],"inputs_sha256":manifest["inputs"]["sha256"],
                "timing_sha256":manifest["timing"]["sha256"],"environment_sha256":env["sha256"]},
            "build":check(result["build"],"actual selected CUDA worker"),
            "correctness":check("pass" if correct and correct["passed"] else "fail" if correct else "unknown","complete original-storage Task suite"),
            "performance":check("pass" if result.get("timing") and result["timing"]["complete"] else "unknown","complete forward timing; soft goal separately reported"),
            "case_ids":sorted({c["case_id"] for c in correct["basic_calls"]}) if correct else [],"samples":samples,
            "raw_receipts":[native,acceptance,raw,frozen],"usage_receipts":[usage],"environment":env,
            "extensions":{"information":{"performance_goal":(result.get("timing") or {}).get("soft_goal","unknown")}}}
        if intent["profile"].get("synthetic") or intent["transport"].get("synthetic"):
            document["extensions"]={"information":{"synthetic":True,"qualification":"none"}}
            for name in ("build","correctness","performance"):
                document[name]=check("unknown","synthetic transport exercise; no CUDA qualification")
        report=publication.put_json(prefix+"/report.json",document); roots.append(report)
    publication.commit(roots,subjects=[intent["stage"]["manifest"]["task"],intent["stage"]["manifest"]["source"]])
    settled={"state":"terminal","receipt_id":facts["receipt_id"],"report":report,"usage_row":row,
             "bundle_root":str(publication.root),"new_receipt":True,"diagnostic":facts["diagnostic"]}
    durable_write(job/"absorption.json",jcs(settled))
    return settled
