"""Finite F1 Agate adapter, using the existing original-job/account machinery."""
from pathlib import Path
import fcntl
import json
import os
import re
import shlex
import statistics
from uuid import uuid4

from ..artifacts import Publication, durable_mkdir, durable_write, read_regular, FileBytes, source_files_manifest
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from .agate import AgateDevExecutor, PROFILE, _unframe
from .channel import _remote_profile, transport_from_config, _record
from .fa4 import FA4Task, F1, ABI, _case, verify_stage, stage_candidate, execution_materials
from ..budget import OwnerAccount, cost, BudgetDenied
from ..runtime import AdmissionDenied

BOUNDARY = "fa4-agate-f1-bwrap-v1"
FORMAT = "ks.fa4.qualification.v1"
TRUST = "generated-code-no-evaluator-tampering; readonly-root-stage; writable-compile-output; cuda-devices; no-network"
BAD = {"missing_out": "out.fill_(float('nan'))", "changed_input": "query.zero_()",
       "tolerance": "out.add_(1000)", "storage": "out.set_(out.clone())"}
PHASES = ("probe", "baseline", *BAD)


def negative_files(bundle, source, name):
    files = bundle.source_files(source)
    suffix = ("\n# Fixed negative qualification control: " + name + "\n_KSOriginalModel = Model\n"
              "class Model(_KSOriginalModel):\n    def forward(self, " + ", ".join(ABI) + "):\n"
              "        result = super().forward(" + ", ".join(ABI) + ")\n"
              "        " + BAD[name] + "\n        return out\n")
    return {**files, "solution.py": FileBytes(files["solution.py"].data + suffix.encode(), 0o644)}


def _failed_control(name, observation):
    checks = observation.get("checks", [])
    if not checks or checks[0].get("phase") != "cold" or checks[0].get("passed") is not False:
        return False
    first = checks[0]
    if name in ("missing_out", "tolerance"):
        return first.get("outputs", {}).get("original_out", {}).get("passed") is False
    if name == "changed_input":
        return first.get("readonly_bytes", {}).get("query") is False
    return first.get("out_view_intact") is False


def read_qualification(value, *, context, remote, transport):
    if (not isinstance(value, dict) or set(value) != {"format", "trust_boundary", *PHASES}
            or value["format"] != FORMAT or value["trust_boundary"] != TRUST):
        raise UnsupportedError("FA4 qualification needs original probe/baseline/four controls and explicit trust boundary")
    if transport.get("synthetic"):
        raise UnsupportedError("synthetic FA4 transport cannot qualify execution")
    task = FA4Task(context.bundle, context.request["task"])
    _, evaluator_files, baseline_files = execution_materials(context.bundle, task.task)
    expected_files = {name: f.data for name, f in evaluator_files.items()}
    expected_files.update({name: Path(__file__).with_name(name).read_bytes() for name in ("_remote.py", "_fa4_remote.py")})
    expected_files.update({"baseline/" + name: f.data for name, f in baseline_files.items()})
    environment = None
    for phase in PHASES:
        record = value[phase]
        if set(record) != {"intent.json", "request.json", "acceptance.raw", "terminal.raw"}:
            raise ProtocolError("FA4 original qualification record incomplete")
        intent, request = loads(record["intent.json"]), loads(record["request.json"])
        validate_prepared(intent, request)
        native, ack = json.loads(record["terminal.raw"]), json.loads(record["acceptance.raw"])
        if (native.get("job_id") != ack.get("job_id") or not ack.get("job_id") or native.get("status") != "succeeded"
                or not native.get("finished_at") or native.get("cleanup_pending") is not False or native.get("execution_quarantined", False)):
            raise UnsupportedError("FA4 qualification original job is not fully settled")
        if intent["transport"] != transport or intent["profile"] != (remote if phase != "probe" else {k: v for k, v in remote.items() if k != "bwrap_sha256"}):
            raise ProtocolError("FA4 qualification original transport/profile changed")
        manifest = intent["stage"]["manifest"]
        if any(manifest[k] != context.request[k] for k in ("task", "schema", "scope")) or manifest["evaluator"] != task.task["evaluator"] or manifest["baseline"] != task.task["baseline_source"]:
            raise ProtocolError("FA4 qualification belongs to another fixed assignment")
        source_files = negative_files(context.bundle, task.task["baseline_source"], phase) if phase in BAD else baseline_files
        expected = {**expected_files, **{"source/" + name: file.data for name, file in source_files.items()}}
        if manifest["files"] != {name: {"sha256": sha256(data), "size": len(data), "mode": "0644"} for name, data in expected.items()}:
            raise ProtocolError("FA4 qualification trusted evaluator or fixed control bytes differ")
        execution, diagnostic = interpret(intent, native)
        if execution is None:
            raise UnsupportedError("FA4 qualification incomplete: " + str(diagnostic))
        if phase == "probe":
            if intent["action"] != "fa4_probe" or execution["bwrap_sha256"] != remote["bwrap_sha256"]:
                raise ProtocolError("FA4 original probe action/bwrap differs")
            environment = execution["environment"]
        else:
            if intent["action"] != "fa4_execute" or intent["environment"] != environment:
                raise ProtocolError("FA4 qualification environment changed after probe")
            base, candidate = [execution["reports"][label]["parsed"] for label in ("baseline", "candidate")]
            if not complete(base) or execution["reports"]["baseline"]["exit_code"] != 0:
                raise UnsupportedError("FA4 fixed baseline did not pass complete full-forward suite")
            if phase == "baseline":
                if not complete(candidate) or execution["reports"]["candidate"]["exit_code"] != 0:
                    raise UnsupportedError("FA4 baseline candidate did not pass")
            elif not _failed_control(phase, candidate):
                raise UnsupportedError("FA4 control was not rejected by its original predicate: " + phase)
    return environment


def selection(*, remote, qualification=None, connection_profile="prod"):
    _remote_profile(remote)
    return {"factory": "kernelschema.evaluation.fa4_channel:workflow_channel",
            "implementation_sha256": sha256(Path(__file__).read_bytes()),
            "parameters": {"remote": remote, "qualification": qualification, "connection_profile": connection_profile,
                "trust_boundary": TRUST, "implementations": implementation_hashes()}}


def implementation_hashes():
    return {name: sha256(Path(__file__).with_name(name).read_bytes())
            for name in ("agate.py", "fa4.py", "_fa4_remote.py", "_remote.py")}


def _request(intent, files):
    profile, reservation = intent["profile"], intent["reservation"]
    ms = min(reservation["active_ms"], 1200000)
    return {"spec": {"target_hardware": [profile["gpu"]], "num_gpus": 1},
        "command": shlex.join([profile["python"], "-I", "-B", "_ks_stage/_fa4_remote.py", intent["action"]]),
        "files": files, "env_vars": {}, "timeout_s": ms // 1000, "recycle": True,
        "replay_policy": "never", "dev_intent": "custom_harness", "idempotency_key": intent["submission_key"]}


def _settings(intent):
    return {**{k: intent["profile"].get(k) for k in ("python", "bwrap", "bwrap_sha256")},
            "timeout_s": min(intent["reservation"]["active_ms"], 1200000) // 1000,
            "deadline_utc_ms": intent["reservation"].get("deadline_utc_ms")}


def validate_prepared(intent, request):
    try:
        manifest = intent["stage"]["manifest"]
        if manifest["execution_boundary"] != BOUNDARY or digest(manifest) != intent["stage"]["sha256"]:
            raise ProtocolError("FA4 stage boundary/identity differs")
        files = request["files"]
        extras = {"_ks_stage/stage.json": jcs(manifest).decode(), "_ks_stage/remote.json": jcs(_settings(intent)).decode()}
        if set(files) != {"_ks_stage/" + name for name in manifest["files"]} | set(extras):
            raise ProtocolError("FA4 upload path set differs")
        for name, facts in manifest["files"].items():
            data = files["_ks_stage/" + name].encode()
            if sha256(data) != facts["sha256"] or len(data) != facts["size"] or facts["mode"] != "0644":
                raise ProtocolError("FA4 upload bytes/mode differ")
        for prefix, key in (("source", "source"), ("baseline", "baseline")):
            bodies = {name.removeprefix(prefix + "/"): FileBytes(files["_ks_stage/" + name].encode(), 0o644)
                      for name in manifest["files"] if name.startswith(prefix + "/")}
            if source_files_manifest(bodies) != manifest[key + "_manifest"] or digest(source_files_manifest(bodies)) != manifest[key]["sha256"]:
                raise ProtocolError("FA4 uploaded complete Source identity differs")
        for name, data in extras.items():
            if files[name] != data:
                raise ProtocolError("FA4 uploaded settings differ")
        if digest(request) != intent["request_sha256"] or jcs(request) != jcs(_request(intent, files)):
            raise ProtocolError("FA4 original command/profile/key differs")
    except (KeyError, TypeError, AttributeError, UnicodeError) as error:
        raise ProtocolError("malformed FA4 prepared request") from error
    return request


def prepare(executor, staged, *, profile, operation_id, submission_key, owner, reservation, environment, action):
    verify_stage(staged)
    _remote_profile(profile)
    if profile["channel"] != PROFILE or not owner or reservation.get("owner") != owner or reservation.get("evaluator_jobs") != 1:
        raise ProtocolError("FA4 requires original owner reservation and Agate dev profile")
    if type(reservation.get("active_ms")) is not int or reservation["active_ms"] < 1000:
        raise ProtocolError("FA4 reservation requires at least one second")
    if action == "fa4_execute" and not re.fullmatch(r"sha256:[0-9a-f]{64}", profile.get("bwrap_sha256", "")):
        raise UnsupportedError("FA4 execute requires actual pinned bwrap")
    intent = {"operation_id": operation_id, "submission_key": submission_key, "owner": owner, "reservation": reservation,
              "profile": profile, "transport": executor.transport.identity(), "stage": staged,
              "action": action, "environment": environment, "frozen_inputs_sha256": None}
    files = {"_ks_stage/" + name: read_regular(Path(staged["root"]), name)[0].decode()
             for name in staged["manifest"]["files"]}
    files.update({"_ks_stage/stage.json": jcs(staged["manifest"]).decode(), "_ks_stage/remote.json": jcs(_settings(intent)).decode()})
    request = _request(intent, files)
    intent["request_sha256"] = digest(request)
    validate_prepared(intent, request)
    job = executor._job(operation_id)
    if job.exists():
        if loads(read_regular(job, "intent.json")[0]) != intent:
            raise ProtocolError("FA4 original prepared intent changed")
    else:
        durable_mkdir(job)
        durable_write(job / "request.json", jcs(request))
        durable_write(job / "intent.json", jcs(intent))
    return intent


def _inner(manifest, label):
    key = "baseline" if label == "baseline" else "source"
    files = {}
    for name, facts in manifest["files"].items():
        if name.startswith(key + "/"):
            files["source/" + name.split("/", 1)[1]] = facts
        elif not name.startswith(("source/", "baseline/")) and name not in ("_remote.py", "_fa4_remote.py"):
            files[name] = facts
    return {**{k: manifest[k] for k in ("version", "task", "schema", "scope", "evaluator", "inputs", "timing", "environment", "coverage")},
            "source": manifest[key], "files": files}


def _environment(value):
    required = json.loads(_case("task/environment.json"))["required"]
    if (not isinstance(value, dict) or any(value.get(k) != v for k, v in required.items())
            or any(not value.get(k) or value[k] == "unknown" for k in
                   ("python", "executable", "executable_sha256", "nvidia_smi", "tvm_ffi_version", "tvm_ffi_module_sha256", "compiler"))):
        raise UnsupportedError("FA4 original stack/driver/Python/compiler/TVM-FFI capture incomplete")
    return value


def interpret(intent, native):
    try:
        result = native.get("result")
        if native.get("status") != "succeeded" or not isinstance(result, dict) or result.get("exit_code") != 0:
            return None, "original FA4 launcher did not complete"
        execution = _unframe(result.get("stdout"))
        manifest = intent["stage"]["manifest"]
        if execution.get("action") != intent["action"] or execution.get("exit_code") != 0:
            raise ProtocolError("FA4 action/outer terminal differs")
        if intent["action"] == "fa4_probe":
            _environment(execution["environment"])
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", execution.get("bwrap_sha256", "")):
                raise ProtocolError("FA4 bwrap observation absent")
            return execution, None
        if (execution.get("boundary") != BOUNDARY or execution.get("stage_sha256") != intent["stage"]["sha256"]
                or execution.get("loaded") != manifest["files"] or execution.get("bwrap_sha256") != intent["profile"]["bwrap_sha256"]
                or set(execution["reports"]) != {"baseline", "candidate"}):
            raise ProtocolError("FA4 outer files/permission boundary differs")
        for label in ("baseline", "candidate"):
            raw = execution["reports"][label]
            inner = _inner(manifest, label)
            observation = json.loads(raw["observation"])
            if (raw["files"] != inner["files"] or raw["stage_sha256"] != digest(inner)
                    or observation["stage_sha256"] != digest(inner)
                    or observation["subject"] != {k: inner[k]["sha256"] for k in ("task", "schema", "scope", "source", "evaluator", "inputs", "timing")}):
                raise ProtocolError("FA4 inner Source/Task stage differs")
            permission = observation["permissions"]
            expected_probes = {"stage.json", *inner["files"]}
            probes = permission["probes"]
            if (permission["boundary"] != BOUNDARY or len(probes) != len(expected_probes)
                    or {p["path"] for p in probes} != expected_probes or not all(p["write_open_denied"] is True for p in probes)):
                raise ProtocolError("FA4 complete readonly evidence absent")
            _environment(observation["environment"])
            if intent["environment"] is not None and observation["environment"] != intent["environment"]:
                raise ProtocolError("FA4 actual original environment changed")
            raw["parsed"] = observation
        baseline, candidate = [execution["reports"][label]["parsed"] for label in ("baseline", "candidate")]
        if baseline["environment"] != candidate["environment"]:
            raise ProtocolError("FA4 baseline/candidate environments differ")
        if candidate.get("input_digests") is not None and candidate["input_digests"] != baseline.get("input_digests"):
            raise ProtocolError("FA4 actual generated inputs differ")
        return execution, None
    except (ProtocolError, UnsupportedError, KeyError, TypeError, ValueError, AttributeError) as error:
        return None, str(error)


def comparisons_complete(observation):
    checks = observation.get("checks", [])
    expected = [("cold", None), *[("warmup", i) for i in range(5)], *[("steady", i) for i in range(40)]]
    return (set(observation.get("input_digests", {})) == {"query", "kv_cache", "block_tables", "seq_lens", "workspace_buffer"}
        and all(isinstance(v, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", v) for v in observation.get("input_digests", {}).values())
        and [(c.get("phase"), c.get("index")) for c in checks] == expected
        and all(c.get("passed") is True and c.get("out_view_intact") is True and c.get("return_aliases_original_view") is True
            and all(c.get(k, {}).get(name) is True for k in ("readonly_bytes", "readonly_views") for name in ("query", "kv_cache", "block_tables", "seq_lens"))
            and all(c.get("outputs", {}).get(name, {}).get("passed") is True
                    and c["outputs"][name].get("elements") == 16384 and c["outputs"][name].get("failing_elements") == 0
                    for name in ("return", "original_out")) for c in checks))


def correctness_status(observation):
    if any(c.get("passed") is False for c in observation.get("checks", [])):
        return "fail"
    return "pass" if comparisons_complete(observation) else "unknown"


def complete(observation):
    samples, cold = observation.get("samples_ns", []), observation.get("cold", {})
    return (observation.get("passed") is True and observation.get("build") == "pass" and comparisons_complete(observation)
        and len(samples) == 40 and all(type(n) is int and n > 0 for n in samples)
        and all(type(cold.get(k)) is int and cold[k] > 0 for k in ("source_import_ns", "input_generation_ns", "constructor_ns", "first_forward_sync_ns"))
        and observation.get("native_when") == "observed before/after each warmup and steady call")


def publish_report(publication, intent, execution, *, native, acceptance, usage):
    manifest = intent["stage"]["manifest"]
    baseline, candidate = [execution["reports"][label]["parsed"] for label in ("baseline", "candidate")]
    raw = publication.put_bytes("fa4/execution.json", jcs(execution))
    env = publication.put_bytes("fa4/environment.json", jcs(candidate["environment"]))
    both = all(complete(o) and execution["reports"][label]["exit_code"] == 0
               for label, o in (("baseline", baseline), ("candidate", candidate)))
    synthetic = bool(intent["transport"].get("synthetic"))
    def check(status, reason):
        return {"status": "unknown" if synthetic else status, "receipt": raw,
                "reason": "synthetic transport; no FA4 qualification" if synthetic else reason}
    timing = {"baseline": baseline.get("samples_ns", []), "candidate": candidate.get("samples_ns", []),
              "baseline_cold": baseline.get("cold"), "candidate_cold": candidate.get("cold"), "complete": both}
    if both:
        timing["baseline_over_candidate"] = str(statistics.median(timing["baseline"]) / statistics.median(timing["candidate"]))
    samples = publication.put_bytes("fa4/samples.json", jcs(timing))
    return publication.put_json("fa4/evaluation.json", {"protocol": PROTOCOL, "kind": "evaluation",
        "operation_id": intent["operation_id"], "subject": {**{k + "_sha256": manifest[k]["sha256"]
            for k in ("task", "source", "evaluator", "inputs", "timing")}, "environment_sha256": env["sha256"]},
        "build": check("pass" if candidate.get("build") == "pass" else "fail", "candidate import/construction on original stack"),
        "correctness": check(correctness_status(candidate), "full F1 forward/storage/readonly suite; missing runtime evidence is unknown"),
        "performance": check("pass" if both else "unknown", "separate fresh baseline/candidate cold and 40 full-forward samples"),
        "case_ids": [F1], "samples": samples, "raw_receipts": [native, acceptance, raw], "usage_receipts": [usage],
        "environment": env, "extensions": {"information": {"synthetic": synthetic, "performance_goal": "unknown",
            "baseline": manifest["baseline"], "trust_boundary": TRUST, "qualification": "none" if synthetic else "original-qualified-channel"}}})


class FA4Channel:
    def __init__(self, profile, root, *, transport=None):
        self.settings = profile["parameters"]["evaluation"]["parameters"]
        if set(self.settings) != {"remote", "qualification", "connection_profile", "trust_boundary", "implementations"}:
            raise ProtocolError("FA4 channel settings differ")
        if self.settings["trust_boundary"] != TRUST or self.settings["implementations"] != implementation_hashes():
            raise ProtocolError("FA4 installed execution boundary differs")
        self.root, self.transport, self.executor, self.environment = Path(root), transport, None, None

    def admission(self, context):
        try:
            task = FA4Task(context.bundle, context.request["task"])
            if task.validate_scope(context.request["schema"], context.request["scope"])["status"] != "pass":
                raise UnsupportedError("FA4 fixed F1 scope is not admitted")
            execution_materials(context.bundle, task.task)
            if self.settings["qualification"] is None:
                raise UnsupportedError("FA4 requires original-stack probe, full baseline and four negative readonly/comparator qualification records")
            self.transport = self.transport or transport_from_config(self.settings["connection_profile"])
            self.environment = read_qualification(self.settings["qualification"], context=context,
                remote=self.settings["remote"], transport=self.transport.identity())
            self.executor = AgateDevExecutor(self.root, self.transport)
            return {"status": "pass", "reason": "FA4 original-stack full F1 channel records verified"}
        except UnsupportedError as error:
            return {"status": "unknown", "reason": str(error)}

    def _ready(self, context):
        if self.executor is None:
            admission = self.admission(context)
            if admission["status"] != "pass":
                raise UnsupportedError(admission["reason"])

    def locator(self, operation_id):
        return self.executor.locator(operation_id)

    def prepare(self, context, op):
        self._ready(context)
        return self.executor.prepare(op["stage"], profile=self.settings["remote"], operation_id=op["id"],
            submission_key=op["id"], owner=context.owner["id"], reservation=op["reservation"],
            environment=self.environment, action="fa4_execute")

    def validate(self, context, op):
        self._ready(context)
        verify_stage(op["stage"])
        if self.executor.locator(op["id"]) != op["locator"]:
            raise ProtocolError("FA4 original job locator changed")
        self.prepare(context, op)

    def submit(self, context, op, *, authorize):
        self.validate(context, op)
        return self.executor.submit(op["id"], authorize=authorize)

    def settle(self, context, op, publication):
        self._ready(context)
        return self.executor.settle(op["id"], publication)


def workflow_channel(profile, private_evaluation_root):
    return FA4Channel(profile, private_evaluation_root)


def qualify(context, *, remote, transport=None):
    """Explicit six original jobs under the same owner; never invoked by admission.

    Returns original records for a later fixed profile. Resume this same context
    after an unknown original job. It cannot fabricate missing authority/state.
    """
    lock = os.open(context.root / ".runtime/fa4-qualification.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _qualify(context, remote=remote, transport=transport)
    finally:
        os.close(lock)


def _qualify(context, *, remote, transport):
    _remote_profile(remote)
    if "bwrap_sha256" in remote:
        raise ProtocolError("FA4 qualification starts with a probe, not an invented bwrap digest")
    task = FA4Task(context.bundle, context.request["task"])
    if task.validate_scope(context.request["schema"], context.request["scope"])["status"] != "pass":
        raise UnsupportedError("FA4 qualification requires the fixed F1 scope")
    account = OwnerAccount.for_context(context, create=True)
    transport = transport or transport_from_config()
    executor = AgateDevExecutor(context.root / ".runtime/fa4-qualification/jobs", transport)
    if (context.root / ".runtime/state/fa4_qualification.json").exists():
        state = context.load_state("fa4_qualification")
    else:
        state = {"remote": remote, "transport": transport.identity(), "phase": "probe", "operations": {}, "records": {}, "qualification": None,
                 "implementations": {**implementation_hashes(), "fa4_channel.py": sha256(Path(__file__).read_bytes()),
                                     "entry.py": sha256(_case("evaluator/entry.py"))}}
        context.save_state("fa4_qualification", state)
    if state["remote"] != remote or state["transport"] != transport.identity():
        raise ProtocolError("FA4 original qualification channel changed")
    if state.get("implementations") != {**implementation_hashes(), "fa4_channel.py": sha256(Path(__file__).read_bytes()),
                                        "entry.py": sha256(_case("evaluator/entry.py"))}:
        raise ProtocolError("FA4 original qualification implementation changed")
    for phase, record in state["records"].items():
        if _record(executor, state["operations"][phase]["id"]) != record:
            raise ProtocolError("FA4 original qualification record changed or disappeared")
    account.begin(context.run_id)
    try:
        while state["phase"] != "done":
            phase = state["phase"]
            if phase not in state["operations"]:
                state["operations"][phase] = {"id": context.run_id + "_fa4_qualify_" + phase, "publication_id": uuid4().hex}
                context.save_state("fa4_qualification", state)
            op = state["operations"][phase]
            reservation = account.operations(context.run_id).get(op["id"], {}).get("reservation")
            if reservation is None:
                remaining = len(PHASES) - PHASES.index(phase) - 1
                reservation = account.reserve(context.run_id, op["id"], role="evaluator", purpose="FA4 qualification " + phase,
                    amount=cost(evaluator_jobs=1, active_ms=1000), floor=cost(evaluator_jobs=remaining, active_ms=1000 * remaining))
            source = task.task["baseline_source"]
            if phase in BAD:
                publication = Publication(context.root, op["publication_id"])
                source = publication.put_source_files("control", negative_files(context.bundle, source, phase))
                publication.commit([source])
            root = context.root / ".runtime/fa4-qualification" / ("stage-" + phase)
            if root.exists():
                manifest = loads(read_regular(root, "stage.json")[0])
                staged = {"root": str(root), "manifest": manifest, "sha256": digest(manifest)}
                verify_stage(staged)
            else:
                staged = stage_candidate(context.bundle, {**{k: context.request[k] for k in ("task", "schema", "scope")}, "source": source}, root)
            selected, environment = remote, None
            if phase != "probe":
                record = state["records"]["probe"]
                probe, diagnostic = interpret(loads(record["intent.json"]), json.loads(record["terminal.raw"]))
                if probe is None:
                    raise UnsupportedError("original FA4 probe incomplete: " + str(diagnostic))
                selected = {**remote, "bwrap_sha256": probe["bwrap_sha256"]}
                environment = probe["environment"]
            context.record_intent(operation_id=op["id"], role="evaluator", purpose="FA4 qualification " + phase,
                source=source, reservation=reservation, submission_key=op["id"], native_locator=executor.locator(op["id"]))
            executor.prepare(staged, profile=selected, operation_id=op["id"], submission_key=op["id"],
                owner=context.owner["id"], reservation=reservation, environment=environment,
                action="fa4_probe" if phase == "probe" else "fa4_execute")
            if not account.operations(context.run_id)[op["id"]]["launched"]:
                executor.submit(op["id"], authorize=lambda: account.authorize(reservation))
            settled = executor.settle(op["id"], Publication(context.root))
            if settled["state"] != "terminal":
                account.mark_unknown(op["id"], "original FA4 qualification job unknown")
                return {"status": "blocked", "recovery": "settle_only", "qualification": None,
                        "reason": "original FA4 qualification job unknown; no replacement submission",
                        "usage": account.usage(context.run_id, Publication(context.root))}
            account.absorb(context.run_id, settled["usage_row"], terminated=True)
            context.record_terminal(op["id"], {"report": settled.get("report"), "usage": settled["usage_row"], "terminated": True})
            record = _record(executor, op["id"])
            state["records"][phase] = record
            observed, diagnostic = interpret(loads(record["intent.json"]), json.loads(record["terminal.raw"]))
            if observed is None:
                raise UnsupportedError("FA4 original qualification output incomplete: " + str(diagnostic))
            state["phase"] = PHASES[PHASES.index(phase) + 1] if phase != PHASES[-1] else "done"
            if state["phase"] == "done":
                candidate = {"format": FORMAT, "trust_boundary": TRUST, **state["records"]}
                read_qualification(candidate, context=context, remote=selected, transport=transport.identity())
                state["qualification"] = candidate
            context.save_state("fa4_qualification", state)
    except (BudgetDenied, AdmissionDenied, UnsupportedError) as error:
        if "op" in locals():
            owned = account.operations(context.run_id).get(op["id"])
            if owned and not owned["launched"]:
                account.release_unstarted(op["id"])
        state.update(phase="done", error=str(error))
        context.save_state("fa4_qualification", state)
    finally:
        account.end(context.run_id)
    return {"status": "completed" if state["qualification"] else "blocked", "recovery": "none",
            "qualification": state["qualification"], "reason": state.get("error"),
            "usage": account.usage(context.run_id, Publication(context.root))}
