"""Bounded Agate dev adapter, based on skill 0.15/client 0.14.37 wire fields.

CPU transport-tested, not service qualified. No POST retry or legacy route.
"""
from __future__ import annotations
import base64
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request

from ..artifacts import Publication, durable_mkdir, durable_write, read_regular, source_manifest
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from .task import verify_stage
from ._stage_ro import PROFILE as STAGE_RO_PROFILE, INHERITED_PROFILE, PROFILES as STAGE_RO_PROFILES, TOOLS, NAMESPACES, CAPS

PROFILE = "agate-dev-bwrap-v1"
MARKER = "KS_EVALUATOR_V1 "
REJECTION_RECORDS = ("intent.json", "request.json", "submission-started.json", "acceptance-status.json", "acceptance.raw")


def interpret_rejection(record):
    """Recognize only a complete original dev POST validation/auth refusal.

    This is operation termination before job creation, never a worker terminal.
    Missing/ambiguous envelopes or submission provenance remain unknown. Exact
    original bytes are embedded so an exported usage receipt is self-contained.
    """
    if set(record) != set(REJECTION_RECORDS):
        return None
    try:
        intent, request, started, status, response = (loads(record[name]) for name in REJECTION_RECORDS)
        validate_prepared_request(intent, request)
        if (not isinstance(status, dict) or set(status) != {"status"}
                or type(status["status"]) is not int or status["status"] not in (400, 401, 403, 422)
                or not isinstance(response, dict)):
            return None
        required = {"error_class", "reason", "message", "details", "trace_id"}
        if (set(response) not in (required, required | {"job_id"}) or response.get("job_id") is not None
                or response["error_class"] not in ("validation", "auth")
                or any(not isinstance(response[k], str) or not response[k].strip() for k in ("reason", "message", "trace_id"))
                or not isinstance(response["details"], dict)):
            return None
        # A locator/acceptance or execution fact conflicts with pre-job refusal,
        # including one nested in details. Unknown error schemas stay fenced.
        def accepted(value):
            if isinstance(value, dict):
                return any((k in ("job_id", "job_ids", "job", "locator", "job_locator", "accepted", "accepted_at",
                                 "status", "job_status", "execution", "execution_phase", "execution_quarantined",
                                 "cleanup_pending", "started_at", "finished_at", "result") and v is not None)
                           or accepted(v) for k, v in value.items())
            return isinstance(value, list) and any(accepted(v) for v in value)
        if accepted(response["details"]):
            return None
        if (set(started) != {"submission_key", "request_sha256", "submitted_request_sha256", "timeout_s"}
                or started["submission_key"] != intent["submission_key"]
                or started["request_sha256"] != intent["request_sha256"]
                or type(started["timeout_s"]) is not int or not 1 <= started["timeout_s"] <= request["timeout_s"]
                or digest({**request, "timeout_s": started["timeout_s"]}) != started["submitted_request_sha256"]):
            return None
        return {"format": "ks.agate.rejection.v1", "outcome": "rejected_before_job",
                "operation_id": intent["operation_id"], "submission_key": intent["submission_key"],
                "http_status": status["status"], "response": response,
                "method": "POST", "path": "/v1/jobs/dev", "transport": intent["transport"],
                "submitted_request_sha256": started["submitted_request_sha256"],
                "original_sha256": {name: sha256(record[name].encode()) for name in REJECTION_RECORDS},
                "original_records": record}
    except (ProtocolError, KeyError, TypeError, AttributeError):
        return None


def rejection_diagnostic(rejection):
    response = rejection["response"]
    return ("gateway rejected before job: HTTP " + str(rejection["http_status"]) + " "
            + response["error_class"] + "/" + response["reason"])


def stage_ro_profile(profile):
    required = {"channel", "gpu", "tool_sha256", *TOOLS}
    if not isinstance(profile, dict) or set(profile) != required or profile["channel"] not in STAGE_RO_PROFILES:
        raise ProtocolError("stage-ro requires exact channel/gpu/python/unshare/mount/setpriv/tool_sha256 fields")
    if not isinstance(profile["gpu"], str) or not profile["gpu"].strip():
        raise ProtocolError("stage-ro requires an explicit GPU alias")
    if any(not isinstance(profile[k], str) or not profile[k].startswith("/") for k in TOOLS):
        raise ProtocolError("stage-ro requires absolute existing tool paths")
    hashes = profile["tool_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(TOOLS) or any(
            not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in hashes.values()):
        raise ProtocolError("stage-ro requires all four fixed tool hashes before probe")
    return profile


def remote_settings(profile, reservation):
    names = ("channel", *TOOLS, "tool_sha256") if profile["channel"] in STAGE_RO_PROFILES else ("python", "bwrap", "bwrap_sha256")
    return {**{k: profile.get(k) for k in names}, "timeout_s": min(reservation["active_ms"], 1200000) // 1000,
            "deadline_utc_ms": reservation.get("deadline_utc_ms")}


def execution_settings(profile, environment, frozen_sha256):
    return {"mode": "cuda", "profile": profile["channel"], "include_timing": True,
            "environment": environment, "frozen_inputs_sha256": frozen_sha256}


def protected_files(intent):
    manifest = intent["stage"]["manifest"]
    files = {**manifest["files"], "stage.json": digest(manifest),
             "remote.json": digest(remote_settings(intent["profile"], intent["reservation"]))}
    if intent["action"] == "execute":
        files.update({"frozen_inputs.json": intent["frozen_inputs_sha256"],
            "execution.json": digest(execution_settings(intent["profile"], intent["environment"], intent["frozen_inputs_sha256"]))})
    return files


def validate_inherited_proc(permission, before, after):
    proc = permission["proc"]
    outer, helper = proc["outer"], proc["helper"]
    if ("fresh_proc_pid" in permission or proc["mode"] != "inherited-restricted"
            or proc["private_pid_view"] is not False or proc["mounts_unchanged"] is not True
            or type(proc["namespace_pid"]) is not int or proc["namespace_pid"] != 1
            or any(type(pid) is not int or pid <= 0 for pid in (proc["proc_self_pid"], outer["pid"], helper["pid"]))
            or proc["proc_self_pid"] == 1
            or len({proc["proc_self_pid"], outer["pid"], helper["pid"]}) != 3
            or any(type(fd) is not int or fd <= 2 for fd in (outer["directory_fd"], outer["file_fd"]))
            or outer["directory_fd"] == outer["file_fd"]
            or not Path(outer["cwd"]).is_absolute()
            or helper["namespaces"] != {**after, "pid": before["pid"]}
            or not re.fullmatch(r"[0-9a-f]{16}", helper["permitted_caps"]) or not int(helper["permitted_caps"], 16)
            or helper["inherited_fds"] != [] or helper["cwd"] != permission["scratch"]):
        raise ProtocolError("inherited proc/outer/helper evidence differs")
    pidfds = helper["pidfds"]
    if (not isinstance(pidfds, list) or len(pidfds) > 1
            or any(type(p["fd"]) is not int or p["fd"] <= 2 or p["target"] != "anon_inode:[pidfd]"
                   or type(p["pid"]) is not int or p["pid"] != helper["pid"] for p in pidfds)):
        raise ProtocolError("unshare helper self-pidfd evidence differs")
    rows = proc["mounts"]
    if (not isinstance(rows, list) or not rows or not any(r["path"] == "/proc" and r["type"] == "proc" for r in rows)
            or any(not Path(r["path"]).is_relative_to("/proc") or r["propagation"] for r in rows)):
        raise ProtocolError("unchanged inherited proc mount evidence absent")
    expected = {"outer.root", "outer.cwd", "outer.directory_fd", "outer.file_fd",
                "helper.root", "helper.cwd", "helper.fd0", "helper.fd1", "helper.fd2",
                *("helper.fd" + str(p["fd"]) for p in pidfds)}
    denied = proc["denied"]
    if (len(denied) != len(expected) or {p["target"] for p in denied} != expected
            or any(type(p["errno"]) is not int or p["errno"] not in (1, 13) for p in denied)):
        raise ProtocolError("outside proc write-alias denial evidence absent")
    probes = proc["self_probes"]
    if (len(probes) != 3 or {p["via"] for p in probes} != {"root", "cwd", "fd"}
            or any(type(p["errno"]) is not int or p["errno"] != 30 for p in probes)):
        raise ProtocolError("own proc stage EROFS evidence absent")


def validate_stage_ro_evidence(intent, execution, worker):
    profile = stage_ro_profile(intent["profile"])
    permission = worker["permissions"]
    expected = protected_files(intent)
    before, after = permission["namespaces_before"], permission["namespaces_after"]
    if (execution["profile"] != profile["channel"] or execution["tool_sha256"] != profile["tool_sha256"]
            or worker["profile"] != profile["channel"] or permission["profile"] != profile["channel"]
            or permission["scope"] != "complete_evaluator_stage" or permission["whole_root_readonly"] is not False
            or permission["stage_files"] != expected or permission["tool_sha256"] != profile["tool_sha256"]
            or set(before) != set(NAMESPACES) or set(after) != set(NAMESPACES)
            or any(not re.fullmatch(n + r":\[[0-9]+\]", before[n]) or not re.fullmatch(n + r":\[[0-9]+\]", after[n]) or before[n] == after[n] for n in NAMESPACES)
            or permission["private_mounts"] is not True
            or set(permission["caps"]) != set(CAPS) | {"NoNewPrivs"}
            or any(int(permission["caps"][name], 16) for name in CAPS) or permission["caps"]["NoNewPrivs"] != "1"
            or permission["inherited_fds"] != [] or permission["mount_aliases"] != []
            or not permission["scratch"].startswith("/tmp/ks-compile-") or permission["cwd"] != permission["scratch"]
            or permission["scratch_writable"] is not True):
        raise ProtocolError("actual stage-ro boundary evidence differs")
    if profile["channel"] == INHERITED_PROFILE:
        validate_inherited_proc(permission, before, after)
    elif type(permission["fresh_proc_pid"]) is not int or permission["fresh_proc_pid"] != 1 or "proc" in permission:
        raise ProtocolError("fresh proc evidence absent or changed")
    probes = permission["probes"]
    if (len(probes) != len(expected) + 1 or {p["path"] for p in probes} != {*expected, ".ks-stage-write-probe"}
            or any(p["write_open_denied"] is not True or type(p["errno"]) is not int or p["errno"] != 30 for p in probes)):
        raise ProtocolError("complete stage-ro EROFS probes absent")


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
    def from_config(cls, config_path=None, *, profile="prod", timeout=30):
        from atrex_gateway_client.auth import AkSkAuth
        from atrex_gateway_client.profiles import CONFIG_PATH, load_config, resolve_profile
        if config_path is not None and not Path(config_path).is_file():
            raise UnsupportedError("explicit Agate configuration is not an existing file")
        config = load_config(CONFIG_PATH if config_path is None else Path(config_path))
        # The official CLI prefers AGATE_AK/AGATE_SK to file credentials. Keep
        # that order, but require one complete pair rather than mixing sources.
        if "AGATE_AK" in os.environ or "AGATE_SK" in os.environ:
            ak, sk = os.environ.get("AGATE_AK"), os.environ.get("AGATE_SK")
        else:
            ak, sk = config.get("ak"), config.get("sk")
        if not all(isinstance(value, str) and value.strip() for value in (ak, sk)):
            raise UnsupportedError("Agate authentication requires a complete AGATE_AK/AGATE_SK or configuration ak/sk pair")
        return cls(resolve_profile(profile,config), AkSkAuth(ak,sk),timeout=timeout)

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
    except (KeyError,ValueError,TypeError,OSError,EOFError) as error:
        raise ProtocolError("incomplete evaluator frame: " + str(error)) from error


def validate_prepared_request(intent, request):
    """Bind the actual uploaded bytes/command to the original K1 stage and owner.

    Live submission and retained qualification use this same check. Matching a
    report to an intent alone does not prove which worker bytes were uploaded.
    """
    if intent.get("action") in ("fa4_execute", "fa4_probe"):
        from .fa4_channel import validate_prepared
        return validate_prepared(intent, request)
    try:
        if digest(request) != intent["request_sha256"]:
            raise ProtocolError("prepared request changed")
        manifest, profile = intent["stage"]["manifest"], intent["profile"]
        if profile["channel"] in STAGE_RO_PROFILES:
            stage_ro_profile(profile)
        files = request["files"]
        ms = min(intent["reservation"]["active_ms"], 1200000)
        extras = {"_ks_stage/stage.json": jcs(manifest).decode(),
            "_ks_stage/remote.json": jcs(remote_settings(profile, intent["reservation"])).decode()}
        if intent["action"] == "execute":
            frozen = files["_ks_stage/frozen_inputs.json"]
            if sha256(frozen.encode()) != intent["frozen_inputs_sha256"]:
                raise ProtocolError("uploaded frozen inputs changed")
            extras.update({"_ks_stage/frozen_inputs.json": frozen,
                "_ks_stage/execution.json": jcs(execution_settings(profile, intent["environment"], intent["frozen_inputs_sha256"])).decode()})
        elif intent["action"] != "probe":
            raise ProtocolError("unknown original evaluator action")
        expected_names = {"_ks_stage/" + name for name in manifest["files"]} | set(extras)
        if set(files) != expected_names or any(files[name] != data for name, data in extras.items()):
            raise ProtocolError("uploaded stage/settings differ from original intent")
        if any(sha256(files["_ks_stage/" + name].encode()) != value for name, value in manifest["files"].items()):
            raise ProtocolError("uploaded evaluator/source bytes differ from original stage")
        if digest(source_manifest(files["_ks_stage/kernel.py"].encode())) != manifest["source"]["sha256"]:
            raise ProtocolError("uploaded kernel differs from original Source")
        expected = {"spec": {"target_hardware": [profile["gpu"]], "num_gpus": 1},
            "command": shlex.join([profile["python"], "-I", "-B", "_ks_stage/_remote.py", intent["action"]]),
            "files": files, "env_vars": {}, "timeout_s": ms // 1000, "recycle": True,
            "replay_policy": "never", "dev_intent": "custom_harness", "idempotency_key": intent["submission_key"]}
        if jcs(request) != jcs(expected):
            raise ProtocolError("uploaded command/profile/replay policy differs from original intent")
    except (KeyError, TypeError, AttributeError) as error:
        raise ProtocolError("malformed original prepared request") from error
    return request


class AgateDevExecutor:
    def __init__(self, root, transport):
        self.root=Path(root).absolute()
        durable_mkdir(self.root)
        self.transport=transport

    def capabilities(self):
        return {"profile":PROFILE,"profiles":[PROFILE, *STAGE_RO_PROFILES],"engineering":"dev prepare/submit/query/settle implemented",
                "service_qualified":False,"Task_qualification":False,
                "requires":["configured auth and dev ACL", "server replay_policy=never", "selected pinned remote tools/namespaces/CUDA",
                            "frozen actual environment and FP32 input bytes", "complete framed stdout", "confirmed remote terminal/cleanup or bound explicit pre-job refusal"],
                "missing_sdk_operation":"read-only lookup by idempotency_key; lost acknowledgment without job_id stays unknown"}

    def _job(self, operation_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",operation_id): raise ProtocolError("unsafe operation_id")
        return self.root / operation_id

    def locator(self, operation_id, *, profile=PROFILE):
        """Original private locator available before submission acknowledgment."""
        return {"root":str(self._job(operation_id)),"profile":profile,"transport":self.transport.identity()}

    def prepare(self, staged, *, profile, operation_id, submission_key, owner, reservation,
                frozen_inputs=None, environment=None, action="execute"):
        """No network; freeze exact request/key/subjects before POST.

        Probe captures only trusted CUDA environment/inputs, never candidate. It
        still costs one logical job. Execute consumes that actual prior capture.
        Caller owns the reservation, active clock and cross-channel fences.
        """
        if action in ("fa4_execute", "fa4_probe"):
            from .fa4_channel import prepare
            return prepare(self, staged, profile=profile, operation_id=operation_id, submission_key=submission_key,
                           owner=owner, reservation=reservation, environment=environment, action=action)
        verify_stage(staged)
        if profile.get("channel") not in (PROFILE, *STAGE_RO_PROFILES) or action not in ("probe","execute"):
            raise UnsupportedError("only explicit Agate bwrap or stage-ro profile implemented")
        ms=reservation.get("active_ms")
        if not owner or not submission_key or reservation.get("owner") != owner or reservation.get("evaluator_jobs") != 1:
            raise ProtocolError("explicit caller owner/reservation/stable key required")
        if type(ms) is not int or ms < 1000: raise ProtocolError("reservation must allow at least 1s")
        ms = min(ms, 1200000)  # one dev job never exceeds the channel's 20min bound
        if profile["channel"] in STAGE_RO_PROFILES:
            stage_ro_profile(profile)
        else:
            if any(not isinstance(profile.get(k),str) or not profile[k].startswith("/") for k in ("python","bwrap")):
                raise ProtocolError("absolute remote python/bwrap paths required")
            if (action == "execute" and not re.fullmatch(r"sha256:[0-9a-f]{64}",profile.get("bwrap_sha256","") or "")) or not profile.get("gpu"):
                raise ProtocolError("fixed GPU and actual bwrap hash required")
        root=Path(staged["root"])
        files={"_ks_stage/"+name:read_regular(root,name)[0].decode() for name in staged["manifest"]["files"]}
        files["_ks_stage/stage.json"]=jcs(staged["manifest"]).decode()
        files["_ks_stage/remote.json"]=jcs(remote_settings(profile, reservation)).decode()
        if action == "execute":
            if not environment or environment.get("mode") != "cuda" or frozen_inputs is None:
                raise UnsupportedError("execute requires prior actual CUDA environment/input capture")
            inputs=loads(frozen_inputs)
            if inputs.get("torch_version") != environment.get("torch") or len(inputs.get("cases",[])) != 36:
                raise ProtocolError("frozen inputs/environment mismatch")
            files["_ks_stage/frozen_inputs.json"]=frozen_inputs.decode()
            files["_ks_stage/execution.json"]=jcs(execution_settings(profile, environment, sha256(frozen_inputs))).decode()
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
                if self._rejection(old) is None:
                    raise ProtocolError("unknown original Agate job fences new submissions")
        request=loads(read_regular(job,"request.json")[0])
        validate_prepared_request(intent, request)
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
            durable_write(job/"submission-error.json",jcs({"type":type(error).__name__,
                "message":"transport outcome unknown; original identity retained"}))
        return self.query(operation_id)

    def _rejection(self, job):
        # Never reinterpret a later query failure or previously observed job as
        # zero jobs, even if a private acknowledgment was subsequently damaged.
        if any((job/name).exists() for name in ("accepted-job.json", "terminal.raw")) or any(job.glob("query-*.raw")):
            return None
        if (job/"absorption.json").exists():
            prior = loads(read_regular(job, "absorption.json")[0])
            if prior.get("outcome") != "rejected_before_job" or prior.get("usage_row", {}).get("evaluator_jobs") != 0:
                return None
        if not all((job/name).exists() for name in REJECTION_RECORDS):
            return None
        try:
            rejection = interpret_rejection({name: read_regular(job, name)[0].decode() for name in REJECTION_RECORDS})
        except UnicodeError:
            return None
        if rejection and (rejection["operation_id"] != job.name or rejection["transport"] != self.transport.identity()):
            raise ProtocolError("original rejected submission identity changed")
        return rejection

    def query(self, operation_id):
        job=self._job(operation_id); intent=loads(read_regular(job,"intent.json")[0])
        if intent["transport"] != self.transport.identity(): raise ProtocolError("cannot switch original channel")
        validate_prepared_request(intent, loads(read_regular(job,"request.json")[0]))
        unknown={"state":"unknown","operation_id":operation_id,"submission_key":intent["submission_key"],"recovery":"settle_only"}
        if not (job/"acceptance.raw").exists():
            return {**unknown,"reason":"missing acknowledgment/job_id; no read-only key lookup API"}
        try:
            acceptance=json.loads(read_regular(job,"acceptance.raw")[0])
        except (ValueError, UnicodeError):
            return {**unknown,"reason":"malformed original acknowledgment; no read-only key lookup API"}
        job_id=acceptance.get("job_id") if isinstance(acceptance,dict) else None
        if (job/"accepted-job.json").exists():
            locator=loads(read_regular(job,"accepted-job.json")[0])
            if locator != {"job_id":job_id,"acceptance_sha256":sha256(read_regular(job,"acceptance.raw")[0])}:
                raise ProtocolError("original accepted job acknowledgment changed")
        if not isinstance(job_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",job_id):
            rejection=self._rejection(job)
            if rejection is not None:
                return {"state":"terminal","outcome":"rejected_before_job","operation_id":operation_id,
                        "submission_key":intent["submission_key"],"receipt_id":digest(rejection),
                        "rejection":rejection,"execution":None,"diagnostic":rejection_diagnostic(rejection)}
            return {**unknown,"reason":"no original job locator"}
        if not (job/"accepted-job.json").exists():
            durable_write(job/"accepted-job.json",jcs({"job_id":job_id,"acceptance_sha256":sha256(read_regular(job,"acceptance.raw")[0])}))
        unknown["job_id"]=job_id
        if (job/"terminal.raw").exists(): raw=read_regular(job,"terminal.raw")[0]
        else:
            try: response=self.transport.request("GET","/v1/jobs/"+job_id+"?include_spec=true")
            except Exception as error: return {**unknown,"reason":type(error).__name__}
            raw=response["body"]; event=job/("query-"+sha256(raw)[7:]+".raw")
            if not event.exists(): durable_write(event,raw)
            if response["status"] != 200: return {**unknown,"reason":"HTTP "+str(response["status"])}
        try:
            native=json.loads(raw)
        except (ValueError, UnicodeError):
            return {**unknown,"reason":"incomplete original job response"}
        if not isinstance(native,dict):
            return {**unknown,"reason":"original job response is not an object"}
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
    if intent.get("action") in ("fa4_execute", "fa4_probe"):
        from .fa4_channel import interpret
        return interpret(intent, native)
    execution=None; diagnostic=None; result=native.get("result") or {}
    if not isinstance(result,dict):
        return None, "native result is not an object"
    if native.get("output_truncated") is True or result.get("output_truncated") is True:
        return None, "native output is truncated"
    if native["status"] == "succeeded" and result.get("exit_code") == 0:
        try:
            execution=_unframe(result.get("stdout"))
            if not isinstance(execution,dict): raise ProtocolError("evaluator output is not an object")
            if execution.get("action") != intent["action"]: raise ProtocolError("different evaluator action")
            stage_ro = intent["profile"]["channel"] in STAGE_RO_PROFILES
            if execution.get("exit_code") == 0 and (intent["action"] == "execute" or stage_ro):
                worker=loads(execution["stdout"]); manifest=intent["stage"]["manifest"]
                if (worker["loaded"] != manifest["files"] or worker["source"] != manifest["source"] or
                        worker["evaluator"] != manifest["evaluator"] or worker["profile"] != intent["profile"]["channel"]):
                    raise ProtocolError("actual evaluator subjects/environment/permission evidence differs")
                if stage_ro:
                    validate_stage_ro_evidence(intent, execution, worker)
                elif (execution.get("bwrap_sha256") != intent["profile"]["bwrap_sha256"] or
                        len(worker["permissions"]["probes"]) != sum(name.endswith(".py") for name in manifest["files"]) or
                        not all(p["write_open_denied"] for p in worker["permissions"]["probes"])):
                    raise ProtocolError("actual evaluator permission evidence differs")
                if intent["action"] == "execute":
                    if worker["environment"] != intent["environment"] or worker["frozen_inputs_sha256"] != intent["frozen_inputs_sha256"]:
                        raise ProtocolError("actual evaluator environment/frozen inputs differ")
                    execution["worker"]=worker
        except (ProtocolError,KeyError,ValueError,TypeError,AttributeError) as error:
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
    rejected = facts.get("outcome") == "rejected_before_job"
    native=(publication.put_bytes(prefix+"/agate-rejection.json",jcs(facts["rejection"])) if rejected else
            publication.put_bytes(prefix+"/agate-terminal.json",read_regular(job,"terminal.raw")[0]))
    acceptance=publication.put_bytes(prefix+"/agate-acceptance.json",read_regular(job,"acceptance.raw")[0])
    row={"producer":"kernelschema.agate:"+str(executor.root),"operation_id":operation_id,"role":"evaluator",
         "purpose":intent["action"],"receipt_id":facts["receipt_id"],"sequence":1,"receipt":native,
         "state":"settled","calls":0,"evaluator_jobs":0 if rejected else 1,"tokens":0,"gpu_ms":0 if rejected else None}
    usage=publication.put_json(prefix+"/usage.json",{"protocol":PROTOCOL,"kind":"usage_index","entries":[row]})
    # Exact original terminal retained even if no valid worker report is available.
    report=None; roots=[usage,acceptance]; execution=facts["execution"]
    if execution and intent["action"] == "fa4_execute":
        from .fa4_channel import publish_report
        report = publish_report(publication, intent, execution, native=native, acceptance=acceptance, usage=usage)
        roots.append(report)
    elif execution and execution.get("worker"):
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
    if rejected:
        settled["outcome"] = "rejected_before_job"
    durable_write(job/"absorption.json",jcs(settled))
    return settled
