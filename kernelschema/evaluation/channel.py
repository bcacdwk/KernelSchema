"""K1 Agate dev channel for the fixed reference backend, with paid qualification.

No API key is frozen. AGATE_AK/AGATE_SK or the official SDK configuration supplies
authentication; KS_AGATE_CONFIG explicitly selects an existing file. Qualification
stores original nonsecret job evidence, not a pass flag; admission reinterprets it
through K1's same frame/subject validator.
"""
from pathlib import Path
import fcntl
import json
import os
import re
import time
from uuid import uuid4

from ..artifacts import Publication, read_regular, source_manifest
from ..budget import OwnerAccount, cost, BudgetDenied
from ..contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from .agate import AgateDevExecutor, AgateHTTP, PROFILE, STAGE_RO_PROFILE, stage_ro_profile, interpret_terminal, validate_prepared_request
from .environment import match_target, read_target
from .task import RowSoftmaxTask, stage, verify_stage
from ..runtime import AdmissionDenied

FORMAT = "ks.agate.qualification.v1"
BAD_BODIES = {
    "missing_out": "return out",
    "changed_x": "out.copy_(torch.softmax(x, dim=-1)); x.zero_(); return out",
    "tolerance": "out.fill_(0.5); return out",
    "storage": "out.set_(torch.softmax(x, dim=-1)); return out",
}
BAD_CHECKS = {"missing_out": "out_fully_written", "changed_x": "x_bytes", "tolerance": "elementwise_tolerance", "storage": "out_storage"}
BAD_SOURCES = {name: ("# Fixed negative qualification control: " + name + "\nimport torch\nclass Model(torch.nn.Module):\n    def forward(self, x, out):\n        " + body + "\n").encode() for name, body in BAD_BODIES.items()}
PHASES = ("probe", "execute", *BAD_SOURCES)


def _remote_profile(remote):
    required = {"channel", "gpu", "python", "bwrap"}
    if not isinstance(remote, dict) or set(remote) not in (required, required | {"bwrap_sha256"}):
        raise ProtocolError("remote requires only channel/gpu/python/bwrap and optional observed bwrap_sha256; credentials belong in the SDK config")
    if remote["channel"] != PROFILE or not isinstance(remote["gpu"], str) or not remote["gpu"].strip():
        raise ProtocolError("remote needs the supported channel and explicit GPU alias")
    if any(not isinstance(remote[k], str) or not remote[k].startswith("/") for k in ("python", "bwrap")):
        raise ProtocolError("remote Python/bwrap paths must be absolute")
    if "bwrap_sha256" in remote and not re.fullmatch(r"sha256:[0-9a-f]{64}", remote["bwrap_sha256"]):
        raise ProtocolError("remote bwrap digest must come from the original probe")
    return remote


def selection(*, remote, qualification=None, connection_profile="prod"):
    _row_remote_profile(remote)
    return {"factory": "kernelschema.evaluation.channel:workflow_channel",
            "implementation_sha256": sha256(Path(__file__).read_bytes()),
            "parameters": {"remote": remote, "qualification": qualification, "connection_profile": connection_profile,
                           "agate_sha256": sha256(Path(__file__).with_name("agate.py").read_bytes())}}


def _row_remote_profile(remote):
    # FA4 still imports _remote_profile and remains bwrap-only.
    return stage_ro_profile(remote) if isinstance(remote, dict) and remote.get("channel") == STAGE_RO_PROFILE else _remote_profile(remote)


def transport_from_config(connection_profile="prod"):
    path = os.environ.get("KS_AGATE_CONFIG")
    try:
        return AgateHTTP.from_config(path, profile=connection_profile)
    except ImportError as error:
        raise UnsupportedError("optional official atrex-gateway-client is not installed in this interpreter") from error


def _record(executor, op):
    job = executor._job(op)
    return {name: read_regular(job, name)[0].decode() for name in ("intent.json", "request.json", "acceptance.raw", "terminal.raw")}


def _read_record(record):
    if set(record) != {"intent.json", "request.json", "acceptance.raw", "terminal.raw"}:
        raise ProtocolError("incomplete original qualification record")
    intent = loads(record["intent.json"].encode())
    request = loads(record["request.json"].encode())
    validate_prepared_request(intent, request)
    native = json.loads(record["terminal.raw"])
    acceptance = json.loads(record["acceptance.raw"])
    if digest(request) != intent["request_sha256"] or native.get("job_id") != acceptance.get("job_id") or not native.get("job_id"):
        raise ProtocolError("qualification original identity mismatch")
    if native.get("status") != "succeeded" or not native.get("finished_at") or native.get("cleanup_pending") is not False or native.get("execution_quarantined", False):
        raise UnsupportedError("qualification lacks an original successful settled job")
    execution, diagnostic = interpret_terminal(intent, native)
    if not execution or execution.get("exit_code") != 0:
        raise UnsupportedError("qualification original evaluator incomplete: " + str(diagnostic))
    if intent["action"] == "probe":
        try:
            observed = loads(execution["stdout"].encode())
            environment, inputs = observed["environment"], observed["inputs"]
            if (environment["mode"] != "cuda" or inputs["torch_version"] != environment["torch"]
                    or len(inputs["cases"]) != 36
                    or (intent["profile"]["channel"] == PROFILE and not re.fullmatch(r"sha256:[0-9a-f]{64}", execution.get("bwrap_sha256") or ""))):
                raise ValueError("capture incomplete")
        except (ProtocolError, KeyError, TypeError, AttributeError, ValueError) as error:
            raise UnsupportedError("qualification original probe lacks complete CUDA/input/tool capture") from error
    return intent, request, execution


def read_qualification(value, *, context, remote, transport):
    _row_remote_profile(remote)
    if not isinstance(value, dict) or set(value) != {"format", *PHASES} or value["format"] != FORMAT:
        raise UnsupportedError("missing Agate probe/baseline/four-negative qualification; run the paid prepare command on the same owner")
    probe, _, pe = _read_record(value["probe"])
    execute, request, ee = _read_record(value["execute"])
    if probe["action"] != "probe" or execute["action"] != "execute":
        raise ProtocolError("qualification action mismatch")
    if probe["transport"] != transport or execute["transport"] != transport or transport.get("synthetic") or remote.get("synthetic"):
        raise UnsupportedError("qualification transport changed or is synthetic")
    observed = loads(pe["stdout"].encode())
    environment, frozen = observed["environment"], jcs(observed["inputs"])
    if execute["profile"] != remote or probe["profile"] != {k: v for k, v in remote.items() if k != "bwrap_sha256"}:
        raise ProtocolError("qualification remote profile changed")
    if ((remote["channel"] == PROFILE and pe.get("bwrap_sha256") != remote.get("bwrap_sha256"))
            or execute["environment"] != environment or execute["frozen_inputs_sha256"] != sha256(frozen)):
        raise ProtocolError("qualification capture/tool changed")
    manifest = execute["stage"]["manifest"]
    task = context.objects["task"]
    if manifest["task"] != context.request["task"] or manifest["evaluator"] != task["evaluator"] or manifest["source"] != task["baseline_source"]:
        raise UnsupportedError("qualification must use this exact Task/evaluator/baseline")
    # All stage bytes, including original trusted workers, must still be installed.
    expected = {Path(r["path"]).name: context.bundle.read(r) for r in context.bundle.document(task["evaluator"])["files"]}
    for name, raw in expected.items():
        staged_name = "evaluator/" + name if name in ("baseline.py", "check.py", "entry.py", "golden.py", "input.py", "timing.py") else name
        if manifest["files"].get(staged_name) != sha256(raw):
            raise ProtocolError("qualification evaluator bytes changed")
    if request["files"].get("_ks_stage/frozen_inputs.json", "").encode() != frozen:
        raise ProtocolError("qualification frozen inputs changed")
    match_target(context.bundle.read(task["environment"]), environment)
    result = ee.get("worker", {}).get("result", {})
    if result.get("build") != "pass" or not (result.get("correctness") or {}).get("passed") or not (result.get("timing") or {}).get("complete"):
        raise UnsupportedError("qualification requires full baseline correctness, timing, readonly probes and CUDA")
    for name, code in BAD_SOURCES.items():
        intent, _, execution = _read_record(value[name])
        subject = intent["stage"]["manifest"]
        if intent["action"] != "execute" or intent["profile"] != remote or intent["transport"] != transport or intent["environment"] != environment or intent["frozen_inputs_sha256"] != sha256(frozen):
            raise ProtocolError("negative qualification profile/capture changed")
        if subject["source"]["sha256"] != digest(source_manifest(code)) or any(subject[k] != manifest[k] for k in ("task", "evaluator", "inputs", "timing")):
            raise ProtocolError("negative qualification did not run the exact fixed bad Source/Task")
        if subject["files"] != {**manifest["files"], "kernel.py": sha256(code)}:
            raise ProtocolError("negative qualification evaluator bytes changed")
        correctness = execution.get("worker", {}).get("result", {}).get("correctness") or {}
        calls = correctness.get("basic_calls", [])
        if correctness.get("passed") is not False or len(calls) != 72 or not any(c.get("checks", {}).get(BAD_CHECKS[name]) is False for c in calls):
            raise UnsupportedError("actual adapter has not rejected negative control: " + name)
    return environment, frozen


class AgateChannel:
    def __init__(self, profile, root, *, transport=None):
        self.settings = profile["parameters"]["evaluation"]["parameters"]
        if set(self.settings) != {"remote", "qualification", "connection_profile", "agate_sha256"}:
            raise ProtocolError("unknown Agate channel configuration")
        if self.settings["agate_sha256"] != sha256(Path(__file__).with_name("agate.py").read_bytes()):
            raise ProtocolError("pinned K1 Agate implementation changed")
        self.root, self.transport = Path(root), transport
        self.executor = None
        self.capture = None

    def admission(self, context):
        try:
            task = RowSoftmaxTask(context.bundle, context.request["task"])
            task.require_evaluation_scope(context.request["scope"])
            if task.validate_scope(context.request["schema"], context.request["scope"], require_hardware_binding=True)["status"] != "pass":
                raise UnsupportedError("Schema hardware/stack must be explicitly bound to the target")
            self.transport = self.transport or transport_from_config(self.settings["connection_profile"])
            self.capture = read_qualification(self.settings["qualification"], context=context,
                remote=self.settings["remote"], transport=self.transport.identity())
            from .. import template_api
            hardware = template_api.effective_document(context.bundle, context.objects["schema"], "root")["CONTRACT"]["Hardware_specification"]
            if hardware != read_target(context.bundle.read(context.objects["task"]["environment"]))["hardware_specification"]:
                raise UnsupportedError("Schema hardware Contract differs from the fixed Task target")
            self.executor = AgateDevExecutor(self.root, self.transport)
            return {"status": "pass", "reason": "original full K1 Agate qualification verified for this Task, target and channel"}
        except UnsupportedError as error:
            return {"status": "unknown", "reason": str(error)}

    def _ready(self, context):
        if self.executor is None:
            admission = self.admission(context)
            if admission["status"] != "pass": raise UnsupportedError(admission["reason"])

    def locator(self, operation_id):
        return self.executor.locator(operation_id, profile=self.settings["remote"]["channel"])

    def prepare(self, context, op):
        self._ready(context)
        environment, frozen = self.capture
        return self.executor.prepare(op["stage"], profile=self.settings["remote"], operation_id=op["id"],
            submission_key=op["id"], owner=context.owner["id"], reservation=op["reservation"],
            environment=environment, frozen_inputs=frozen)

    def submit(self, context, op, *, authorize):
        self.validate(context, op)
        # Owner authorizes the original absolute cutoff at the last local point.
        # The submitted duration is tightened too; preparation cannot renew it.
        return self.executor.submit(op["id"], authorize=authorize)

    def settle(self, context, op, publication):
        self._ready(context)
        return self.executor.settle(op["id"], publication)

    def validate(self, context, op):
        self._ready(context)
        verify_stage(op["stage"])
        if self.locator(op["id"]) != op["locator"]:
            raise ProtocolError("original Agate locator changed")
        self.prepare(context, op)  # K1 checks the identical prepared intent, no network.


def _deadline(reservation):
    remaining = reservation["active_ms"]
    if "deadline_utc_ms" in reservation:
        remaining = min(remaining, reservation["deadline_utc_ms"] - time.time_ns() // 1_000_000)
    if "deadline_monotonic_ns" in reservation:
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != reservation["clock_boot_id"]:
            raise ProtocolError("Agate authorization belongs to another boot")
        remaining = min(remaining, (int(reservation["deadline_monotonic_ns"]) - time.monotonic_ns()) // 1_000_000)
    if remaining < 1000:
        from ..runtime import AdmissionDenied
        raise AdmissionDenied("original Agate submission deadline exhausted")
    return min(remaining, 1200000)


def workflow_channel(profile, private_evaluation_root):
    return AgateChannel(profile, private_evaluation_root)


def qualify(context, *, remote, transport=None):
    """Six finite jobs (probe, baseline and four bad controls), same owner.

    Caller creates/opens an ordinary K2 context with the fixed complete Task and
    original parent authority. Reopen this same context on unknown; never retry
    a missing acknowledgment with a new key. No model invocation occurs here.
    """
    lock = os.open(context.root / ".runtime/agate-qualification.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _qualify(context, remote=remote, transport=transport)
    finally:
        os.close(lock)


def _qualification_account(context, *, state_path, native_root):
    """Initialize once; retained qualification facts cannot mint another owner."""
    intents = context.intents()
    marker = context.root / ".runtime/state/qualification_owner.json"
    started = marker.exists() or state_path.exists() or native_root.exists() or bool(intents)
    expected = {"state_namespace": state_path.stem}
    if marker.exists():
        if context.load_state("qualification_owner") != expected:
            raise ProtocolError("original qualification owner initialization differs")
    elif not started:
        # A crash here has the same conservative missing-binding behavior as
        # ordinary workflow bootstrap. This is an intent, not a budget operation.
        context.save_state("qualification_owner", expected)
    account = OwnerAccount.for_context(context, create=not started)
    records = account.operations(context.run_id)
    if not state_path.exists() and (records or intents or native_root.exists()):
        raise ProtocolError("original qualification state missing; cannot restart")
    for intent in intents:
        record = records.get(intent["operation_id"])
        if record is None or jcs(record["reservation"]) != jcs(intent["reservation"]):
            raise ProtocolError("original qualification owner operation missing or changed")
    if state_path.exists():
        # Terminal native facts remain a witness even if both the core intent
        # directory and ledger operations were lost. Never project those paid
        # records as an empty/zero-use account.
        core = {i["operation_id"]: i for i in intents}
        for retained in context.load_state(state_path.stem)["records"].values():
            native = loads(retained["intent.json"])
            operation_id = native["operation_id"]
            original, record = core.get(operation_id), records.get(operation_id)
            if (original is None or record is None or not record["history"] or not record["terminated"]
                    or jcs(native["reservation"]) != jcs(original["reservation"])
                    or jcs(native["reservation"]) != jcs(record["reservation"])):
                raise ProtocolError("original qualification owner operation/core intent missing or changed")
    return account


def _qualify(context, *, remote, transport=None):
    _row_remote_profile(remote)
    if "bwrap_sha256" in remote:
        raise ProtocolError("qualification begins with an unpinned remote profile; probe records the actual bwrap digest")
    read_target(context.bundle.read(context.objects["task"]["environment"]))
    root = context.root / ".runtime/agate-qualification"
    path = context.root / ".runtime/state/agate_qualification.json"
    account = _qualification_account(context, state_path=path, native_root=root)
    transport = transport or transport_from_config()
    executor = AgateDevExecutor(root / "jobs", transport)
    if path.exists(): state = context.load_state("agate_qualification")
    else:
        state = {"remote": remote, "transport": transport.identity(), "operations": [], "phase": "probe", "records": {}, "qualification": None}
        context.save_state("agate_qualification", state)
    if state["remote"] != remote or state["transport"] != transport.identity():
        raise ProtocolError("original qualification profile changed")
    for phase, record in state["records"].items():
        if phase not in PHASES or _record(executor, context.run_id + "_qualify_" + phase) != record:
            raise ProtocolError("original qualification record changed or disappeared")
    if state["qualification"]:
        actual_remote = loads(state["qualification"]["execute"]["intent.json"].encode())["profile"]
        read_qualification(state["qualification"], context=context, remote=actual_remote, transport=transport.identity())
    account.begin(context.run_id)
    try:
        while state["phase"] != "done":
            phase = state["phase"]
            opid = context.run_id + "_qualify_" + phase
            op = next((o for o in state["operations"] if o["id"] == opid), None)
            if op is None:
                op = {"id": opid, "publication_id": uuid4().hex}
                state["operations"].append(op); context.save_state("agate_qualification", state)
            reservation = account.operations(context.run_id).get(opid, {}).get("reservation")
            if reservation is None:
                reservation = account.reserve(context.run_id, opid, role="evaluator", purpose="Agate qualification " + phase,
                    amount=cost(evaluator_jobs=1, active_ms=1000), floor=cost(evaluator_jobs=len(PHASES)-PHASES.index(phase)-1, active_ms=1000*(len(PHASES)-PHASES.index(phase)-1)))
            source = context.objects["task"]["baseline_source"]
            if phase in BAD_SOURCES:
                p = Publication(context.root, op["publication_id"])
                source = p.put_source("negative-" + phase, BAD_SOURCES[phase]); p.commit([source])
            destination = root / ("stage-" + phase)
            if destination.exists():
                manifest = loads(read_regular(destination, "stage.json")[0]); staged = {"root": str(destination), "manifest": manifest, "sha256": digest(manifest)}
                verify_stage(staged)
            else:
                staged = stage(context.bundle, context.request["task"], source, destination, scope=context.request["scope"])
            selected_remote = remote
            arguments = {}
            if phase != "probe":
                _, _, execution = _read_record(state["records"]["probe"])
                observed = loads(execution["stdout"].encode())
                match_target(context.bundle.read(context.objects["task"]["environment"]), observed["environment"])
                selected_remote = {**remote, "bwrap_sha256": execution["bwrap_sha256"]} if remote["channel"] == PROFILE else remote
                arguments = {"environment": observed["environment"], "frozen_inputs": jcs(observed["inputs"])}
            context.record_intent(operation_id=opid, role="evaluator", purpose="Agate qualification " + phase,
                source=source, reservation=reservation,
                submission_key=opid, native_locator=executor.locator(opid, profile=selected_remote["channel"]))
            executor.prepare(staged, profile=selected_remote, operation_id=opid, submission_key=opid,
                owner=context.owner["id"], reservation=reservation, action="probe" if phase == "probe" else "execute", **arguments)
            owned = account.operations(context.run_id)[opid]
            if not owned["launched"]:
                _deadline(reservation)
                executor.submit(opid, authorize=lambda: account.authorize(reservation))
            settled = executor.settle(opid, Publication(context.root))
            if settled["state"] != "terminal":
                account.mark_unknown(opid, "original qualification operation is unknown")
                return {"status": "blocked", "recovery": "settle_only", "qualification": None,
                        "reason": "original qualification operation is unknown; no replacement submission",
                        "usage": account.usage(context.run_id, Publication(context.root))}
            account.absorb(context.run_id, settled["usage_row"], terminated=True)
            context.record_terminal(opid, {"report": settled.get("report"), "usage": settled["usage_row"], "terminated": True})
            record = _record(executor, opid)
            state["records"][phase] = record
            try: _read_record(record)
            except UnsupportedError as error:
                state.update(phase="done", error=str(error)); context.save_state("agate_qualification", state); break
            state["phase"] = PHASES[PHASES.index(phase) + 1] if phase != PHASES[-1] else "done"
            if state["phase"] == "done":
                candidate = {"format": FORMAT, **state["records"]}
                read_qualification(candidate, context=context, remote=selected_remote, transport=transport.identity())
                state["qualification"] = candidate
            context.save_state("agate_qualification", state)
    except (UnsupportedError, BudgetDenied, AdmissionDenied) as error:
        if 'opid' in locals():
            owned = account.operations(context.run_id).get(opid)
            if owned and not owned["launched"]: account.release_unstarted(opid)
        state.update(phase="done", error=str(error))
        context.save_state("agate_qualification", state)
    finally:
        account.end(context.run_id)
    publication = Publication(context.root)
    evidence = publication.put_bytes("qualification-original-records.json", jcs({"format": FORMAT, "records": state["records"],
        "qualification": state["qualification"], "error": state.get("error")}))
    publication.commit([evidence])
    return {"status": "completed" if state["qualification"] else "blocked", "recovery": "none",
            "qualification": state["qualification"], "reason": state.get("error"), "evidence": evidence,
            "usage": account.usage(context.run_id, Publication(context.root))}
