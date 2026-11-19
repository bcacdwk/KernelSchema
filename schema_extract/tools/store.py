"""The authoritative v2 run record and immutable artifact checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import stat
import time
from uuid import UUID, uuid4

from . import __version__
from .common import (STATE_GUARD, ExtractError, RunnerBusy, StaleResult, SupervisorBusy, copy_artifacts, digest, json_bytes,
                     fsync_directory, manifest, read_json, seal, sync_tree, verify_manifest, write_json, rules_root)
from .artifacts import (align_source_identity, approval_projection, artifact_manifest, check_stage_snapshot, design_context,
                        load_yaml, normalize_candidate, source_context, validate_artifact_sources, write_yaml)
from .runtime import EXECUTOR_CAPABILITIES, execution_boundary, verify_execution
from .stages import (NODE_STOPS, call_limit, actual_sources, call_view, effective_nodes, new_scope, next_call, next_calls, node_signature,
                     approval_is_current, scope_for, scope_key, session_group, session_scope,
                     selection_signature, writable_files, writing_order, invalidate_final, invalidate_node, invalidate_stage)


STATE_FORMAT = "schema_extract.run.v2"
SUMMARY_FORMAT = "schema_extract.summary.v1"
SUPERVISOR_FORMAT = "schema_extract.supervisor.v1"


def _state_stamp(path: Path) -> dict:
    """A small observation token, never an integrity or approval decision."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ExtractError("state.json is not a regular file")
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @staticmethod
    def execution_config(state: dict) -> dict:
        config = state.get("config", {})
        if config.get("backend") not in {"claude", "codex"} or any(
                not isinstance(config.get(key), str) or not config[key]
                for key in ("agent_binary", "model", "worker_effort", "reviewer_effort")):
            raise ExtractError("run must register backend, executable, model and both role efforts")
        if not Path(config["agent_binary"]).is_absolute():
            raise ExtractError("registered agent executable must be an absolute path")
        return config

    def context(self, state: dict, attempt: dict | None = None) -> dict:
        from .snapshots import verify_run_inputs
        lock = read_json(self.run_dir(state["run_id"]) / "inputs.lock.json")
        verify_run_inputs(state, lock)
        self.execution_config(state)
        if attempt is not None and (attempt.get("run_id") != state["run_id"] or
                attempt.get("input_fingerprint") != state["input_fingerprint"] or
                attempt.get("attempt_id") not in state["attempts"]):
            raise ExtractError("attempt does not belong to the current frozen run")
        from .runtime import executor_lock
        executor = executor_lock(self, state, attempt, lock=lock)
        return {"source_lock": lock, "rules_lock": lock, "executor_lock": executor,
                "identity": {"source_fingerprint": lock["fingerprint"], "rules_id": lock["rules"]["id"],
                             "executor_rules_id": executor["rules"]["id"]}}

    def initialize(self) -> None:
        marker = self.root / "format.json"
        if marker.exists():
            if read_json(marker) != {"format": "schema_extract.store.v2"}:
                raise ExtractError("state root belongs to another format")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "supervisor.lock").open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SupervisorBusy("supervisor is busy; control transaction was not committed") from exc
            try:
                if marker.exists():
                    if read_json(marker) != {"format": "schema_extract.store.v2"}:
                        raise ExtractError("state root belongs to another format")
                elif any(path.name != "supervisor.lock" for path in self.root.iterdir()):
                    raise ExtractError("state root is nonempty and is not a v2 store")
                else:
                    write_json(marker, {"format": "schema_extract.store.v2"})
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run_ids(self) -> list[str]:
        """Only one directory level; attempts and checkpoints are not an index."""
        directory = self.root / "runs"
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.iterdir() if path.is_dir() and not path.is_symlink())

    def read_supervisor(self) -> dict:
        path = self.root / "supervisor.json"
        if not path.exists():
            return {"format": SUPERVISOR_FORMAT, "revision": 0, "active_hold": None,
                    "holds": {}, "starts": {}, "controls": {}, "maintenance_requests": {},
                    "retries": {}, "observations": {}}
        value = read_json(path)
        if not isinstance(value, dict) or value.get("format") != SUPERVISOR_FORMAT:
            raise ExtractError("unsupported supervisor control ledger")
        for name in ("holds", "starts", "controls", "maintenance_requests", "retries", "observations"):
            if not isinstance(value.get(name), dict):
                raise ExtractError("invalid supervisor field: " + name)
        return value

    @contextmanager
    def supervisor(self):
        """Short control transaction. Never acquire a run lock or wait here."""
        self.initialize()
        with (self.root / "supervisor.lock").open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SupervisorBusy("supervisor is busy; control transaction was not committed") from exc
            try:
                value = self.read_supervisor()
                before = deepcopy(value)
                yield value
                if value != before:
                    value["revision"] = before["revision"] + 1
                    value["updated_at"] = time.time()
                    write_json(self.root / "supervisor.json", value)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def control_transaction(self, *, on_wait=None):
        started = time.monotonic() if on_wait is not None else None
        while True:
            transaction = self.supervisor()
            try:
                ledger = transaction.__enter__()
                break
            except SupervisorBusy:
                if on_wait is not None:
                    on_wait(time.monotonic() - started)
                time.sleep(0.05)
        try:
            yield ledger
        except BaseException as exc:
            if not transaction.__exit__(type(exc), exc, exc.__traceback__):
                raise
        else:
            transaction.__exit__(None, None, None)

    def writer_status(self, run_id: str) -> str:
        """Observe a lock without creating a file or changing run authority."""
        path = self.run_dir(run_id) / "runner.lock"
        try:
            with path.open("r") as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return "held"
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                return "free"
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unknown"

    def assert_startable(self, run_id: str, ledger: dict | None = None) -> None:
        ledger = self.read_supervisor() if ledger is None else ledger
        if ledger["active_hold"]:
            raise ExtractError("global hold is active: " + ledger["active_hold"])

    def append_control(self, ledger: dict, run_id: str, action: str, *, hold_id=None, binding=None) -> dict:
        """Caller holds supervisor.lock; requests also bind independent cancel delivery."""
        previous = ledger["controls"].get(run_id, {})
        request = {"format": "schema_extract.control.v2", "run_id": run_id,
                   "id": str(uuid4()), "action": action, "created_at": time.time(),
                   "control_revision": previous.get("revision", 0) + 1,
                   "origin": "hold" if hold_id else "manual", "hold_id": hold_id}
        if binding is not None:
            request["binding"] = binding
        write_json(self.root / "requests" / run_id / (request["id"] + ".json"), request)
        ledger["controls"][run_id] = {"revision": request["control_revision"], "id": request["id"],
                                      "action": action, "hold_id": hold_id, "at": request["created_at"],
                                      "deliveries": deepcopy(previous.get("deliveries", {}))}
        return request

    def hold_member(self, ledger: dict, state: dict) -> dict | None:
        """Register the gate's actual effect, including late-arriving Runners."""
        hold_id = ledger["active_hold"]
        if not hold_id:
            return None
        run_id = state["run_id"]
        hold = ledger["holds"][hold_id]
        existing = hold["members"].get(run_id)
        if existing and not (existing["status"] == "queued_blocked" and state["status"] in {"running", "ready"}):
            return existing
        latest = ledger["controls"].get(run_id, {})
        # Resume may have passed its entry gate while the saved state was still
        # queued. Upgrade the original hold effect at the Runner's next boundary.
        member = existing or {"original_status": state["status"], "state_revision": state.get("state_revision"),
                  "prior_control_revision": latest.get("revision", 0), "status": "preserved",
                  "gate_capable": "hold.v1" in state.get("executor", {}).get("capabilities", [])}
        hold["members"][run_id] = member
        commit = ledger.get("commits", {}).get(run_id)
        if (commit and commit["control_revision"] < hold["revision"] and commit["checkpoint"] == state["checkpoint"] and
                commit["target"] == state["output"] and state["status"] == "ready" and not state["active_attempts"]):
            member.update(status="completed_before_hold", publication=deepcopy(commit))
            return member
        pending_manual = any(not request.get("hold_id") and request["id"] not in state["control"]["requests"]
                             for request in self.control_requests(run_id))
        if state["published"] or state["status"] in {"paused", "cancelled"} or pending_manual or (
                state["control"].get("action") in {"pause", "cancel"} and not state["control"].get("hold_id")) or (
                latest.get("action") in {"pause", "cancel"} and not latest.get("hold_id")):
            member["status"] = "preserved"
            if pending_manual:
                member["reason"] = "已落盘的手动请求先于本 hold；其控制账本可能尚未提交。"
            return member
        if state["status"] == "queued":
            member["status"] = "queued_blocked"
        elif state["status"] in {"running", "ready"} or state.get("active_attempts"):
            request = self.append_control(ledger, run_id, "pause", hold_id=hold_id)
            member.update(status="pause_requested", request_id=request["id"],
                          control_revision=request["control_revision"])
        return member

    def run_dir(self, run_id: str) -> Path:
        try:
            if str(UUID(run_id)) != run_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError) as exc:
            raise ExtractError("run ID must be its exact UUID") from exc
        return self.root / "runs" / run_id

    def load(self, run_id: str) -> dict:
        path = self.run_dir(run_id) / "state.json"
        guard = STATE_GUARD.get()
        value = guard.read_state() if guard and path.parent == guard.directory else read_json(path)
        if not isinstance(value, dict) or value.get("format") != STATE_FORMAT or value.get("run_id") != run_id:
            raise ExtractError("unsupported or mismatched state; start a new extraction")
        if not {"active_attempts", "reopen_requests", "selected_designs", "writing_plan"} <= value.keys():
            raise ExtractError("run does not implement the current six-stage protocol")
        self.execution_config(value)
        return value

    def save(self, state: dict) -> None:
        if state.get("format") != STATE_FORMAT:
            raise ExtractError("cannot save another state format")
        directory = self.run_dir(state["run_id"])
        guard = STATE_GUARD.get()
        guard = guard if guard and guard.directory == directory else None
        if guard:
            guard.verify()
        revision = state.get("state_revision", 0)
        if type(revision) is not int or revision < 0:
            raise ExtractError("invalid state revision")
        saved = {**state, "state_revision": revision + 1, "updated_at": time.time()}
        write_json(directory / "state.json", saved)
        state.update(state_revision=saved["state_revision"], updated_at=saved["updated_at"])
        if guard:
            guard.written(json_bytes(saved))
        # The authoritative commit is complete. A failed observer projection
        # must not roll it back or turn a completed invocation into a retry.
        try:
            stamp = _state_stamp(directory / "state.json")
            projection = summary(self, state)
            projection["state_file"] = stamp
            write_json(directory / "summary.json", projection)
        except Exception:
            # read_summary detects the old/missing token and falls back once.
            pass

    def read_summary(self, run_id: str, *, fallback=True) -> dict:
        """Read one projection plus one state stat; fallback never validates nodes."""
        directory = self.run_dir(run_id)
        observation = {"observed_at": time.time(), "source": "summary", "status": "unknown",
                       "deep_verified": False}
        try:
            value = read_json(directory / "summary.json")
            if (not isinstance(value, dict) or value.get("format") != SUMMARY_FORMAT or
                    value.get("run_id") != run_id or type(value.get("state_revision")) is not int):
                raise ExtractError("unsupported or mismatched summary projection")
            if value.get("state_file") == _state_stamp(directory / "state.json"):
                observation["status"] = "current"
                return {**value, "observation": observation}
            observation.update(status="stale", reason="state changed after this projection")
        except (ExtractError, OSError) as exc:
            observation["reason"] = str(exc)
        if not fallback:
            return {"format": SUMMARY_FORMAT, "run_id": run_id, "observation": observation}
        observation["source"] = "state"
        return {**summary(self, self.load(run_id)), "observation": observation}

    @contextmanager
    def writer(self, run_id: str):
        """One Runner owns all phase decisions and serial incremental commits."""
        path = self.run_dir(run_id) / "runner.lock"
        with path.open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunnerBusy("another Runner owns this run") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def request_control(self, run_id: str, action: str) -> dict:
        """Submit an immutable request outside the run; never edit its state."""
        state = self.load(run_id)
        if action not in {"pause", "cancel"}:
            raise ExtractError("control action must be pause or cancel")
        with self.supervisor() as ledger:
            binding = {"input_fingerprint": state["input_fingerprint"],
                       "executor_digest": state["executor"]["digest"]}
            previous = ledger["controls"].get(run_id, {})
            if action == "cancel" and previous.get("action") == action and previous.get("binding") == binding:
                request = read_json(self.root / "requests" / run_id / (previous["id"] + ".json"))
                return {**request, "committed": True, "reused": True}
            request = self.append_control(ledger, run_id, action, binding=binding)
            ledger["controls"][run_id]["binding"] = binding
        return {**request, "committed": True}

    def control_requests(self, run_id: str) -> list[dict]:
        directory = self.root / "requests" / run_id
        requests = []
        for path in directory.glob("*.json"):
            value = read_json(path)
            if (path.is_symlink() or value.get("format") != "schema_extract.control.v2" or
                    value.get("run_id") != run_id or value.get("action") not in {"pause", "cancel"} or
                    path.name != value.get("id", "") + ".json" or
                    type(value.get("created_at")) not in {int, float}):
                raise ExtractError("invalid control request: " + str(path))
            requests.append(value)
        return sorted(requests, key=lambda item: (item["created_at"], item["id"]))

    def create(self, lock: dict, output: Path, *, queued=False,
               run_id: str | None = None, created_at: float | None = None) -> dict:
        if not (self.root / "format.json").exists():
            write_json(self.root / "format.json", {"format": "schema_extract.store.v2"})
        run_id = run_id or str(uuid4())
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "state.json").exists():
            raise ExtractError("run already exists: " + run_id)
        now = time.time() if created_at is None else created_at
        state = {"format": STATE_FORMAT, "run_id": run_id,
                 "operator_id": lock["source"]["bindings"]["id"], "output": str(output.resolve()),
                 "input_fingerprint": lock["fingerprint"], "snapshot": lock["snapshot"],
                 "source_identity": {key: lock["source"][key] for key in ("id", "repository_id", "ref", "ref_kind")},
                 "status": "queued" if queued else "running", "stage": "P1", "role": "worker", "node": "root",
                 "mode": "normal", "selected_designs": None, "writing_plan": None, "plan_draft": None,
                 "selection_signature": None, "writing_signature": None, "merged_checkpoint": None,
                 "revision_scope": [], "pending_rechecks": [], "recoverable_checkpoint": None,
                 "plan_revision": 1, "created_at": now,
                 "executor": {"version": __version__, "rules_id": lock["rules"]["id"],
                              "digest": digest({name: value for name, value in lock["rules"]["files"].items()
                                                if name.startswith("schema_extract/") and name.endswith(".py")}),
                              "capabilities": list(EXECUTOR_CAPABILITIES)},
                 "reopening": None, "reopen_requests": {}, "plan_history": [],
                 "stale_results": {}, "unconfirmed_attempts": {}, "p6_starts": 0, "final_review_started": False,
                 "config": lock["config"],
                 "call_additions": [],
                 "sessions": {},
                 "stages": {stage: new_scope() for stage in ("P1", "P2", "P3", "P4", "P5", "P6")},
                 "nodes": {"root": {"id": "root", "file": "root.yaml", "status": "draft", "merged": False,
                                     "draft_checkpoint": None,
                                     "approval": None, "approved_sha256": None,
                                     "generation": 0, "approved_generation": None}},
                 "checkpoint": None, "attempts": [], "active_attempts": {}, "unsettled_attempts": {},
                 "accepted_receipts": {}, "reason": None, "feedback": [],
                 "attempt_usage": {}, "recovered_attempts": {}, "session_handoffs": [],
                 "control": {"action": None, "requests": {}, "resume_status": None},
                 "published": False, "publication": None, "exports": []}
        write_json(directory / "inputs.lock.json", lock)
        initial = directory / (".initial-" + uuid4().hex)
        initial.mkdir()
        shutil.copyfile(rules_root(lock) / "schema_template/starter.yaml", initial / "root.yaml")
        state["checkpoint"] = self.checkpoint(state, initial, attempt_id=None, expected_files={"root.yaml"})
        state["nodes"]["root"]["draft_checkpoint"] = state["checkpoint"]
        shutil.rmtree(initial)
        self.save(state)
        with self.supervisor() as ledger:
            ledger["starts"].setdefault(run_id, {"run_id": run_id, "output": state["output"],
                                                "operator_id": state["operator_id"], "created_at": state["created_at"],
                                                "authorized": False, "status": "preexisting"})
        fsync_directory(directory.parent)
        fsync_directory(self.root)
        return state

    def checkpoint(self, state: dict, artifacts: Path, *, attempt_id: str | None, expected_files,
                   stage: str | None = None, node: str | None = None, parent: str | None = None,
                   call: dict | None = None, preparation: dict | None = None) -> str:
        files = artifact_manifest(artifacts, expected=expected_files)
        identity = {"format": "schema_extract.checkpoint.v2", "run_id": state["run_id"],
                    "stage": stage or state["stage"], "node": node or state["node"], "attempt_id": attempt_id,
                    "plan_revision": state["plan_revision"], "parent": parent or state["checkpoint"],
                    "input_fingerprint": state["input_fingerprint"], "files": files}
        if call is not None:
            identity.update({key: call[key] for key in ("maintenance_task_id", "purpose", "maintenance_step") if key in call})
        if state.get("execution_binding"):
            identity["execution_binding"] = state["execution_binding"]
        if preparation:
            identity["preparation"] = preparation
        identity["selection_signature"] = state.get("selection_signature")
        checkpoint_id = digest(identity)
        base = self.run_dir(state["run_id"]) / "checkpoints"
        destination = base / checkpoint_id
        if destination.exists():
            self.verify_checkpoint(state, checkpoint_id)
            return checkpoint_id
        temporary = base / (".building-" + uuid4().hex)
        temporary.mkdir(parents=True)
        (temporary / "artifacts").mkdir()
        for name in files:
            shutil.copyfile(artifacts / name, temporary / "artifacts" / name)
        verify_manifest(temporary / "artifacts", files)
        write_json(temporary / "checkpoint.json", {**identity, "id": checkpoint_id})
        seal(temporary)
        sync_tree(temporary)
        os.rename(temporary, destination)
        fsync_directory(base)
        return checkpoint_id

    def verify_checkpoint(self, state: dict, checkpoint_id: str | None = None) -> tuple[Path, dict]:
        checkpoint_id = checkpoint_id or state["checkpoint"]
        if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_id):
            raise ExtractError("invalid checkpoint ID")
        directory = self.run_dir(state["run_id"]) / "checkpoints" / checkpoint_id
        info = read_json(directory / "checkpoint.json")
        if info.get("id") != checkpoint_id or digest({k: v for k, v in info.items() if k != "id"}) != checkpoint_id:
            raise ExtractError("checkpoint identity mismatch")
        for key in ("run_id", "input_fingerprint"):
            if info[key] != state[key]:
                raise ExtractError(f"checkpoint {key} mismatch")
        if info.get("execution_binding") and info["execution_binding"] not in state.get("execution_bindings", []):
            raise ExtractError("checkpoint belongs to an unknown executor binding")
        artifact_manifest(directory / "artifacts", expected=info["files"])
        verify_manifest(directory / "artifacts", info["files"])
        return directory / "artifacts", info

    @execution_boundary
    def verify_approval_origin(self, state: dict, approval: dict | None) -> tuple[dict, dict]:
        """Resolve an approval to its accepted receipt, Runner gate, and immutable bytes."""
        if not isinstance(approval, dict):
            raise ExtractError("missing effective approval")
        if approval["input_fingerprint"] != state["input_fingerprint"]:
            raise ExtractError("approval no longer matches the fixed inputs")
        attempt_id = approval["attempt_id"]
        if state["accepted_receipts"].get(attempt_id) != approval["receipt_sha256"]:
            raise ExtractError("approval is not bound to an accepted receipt")
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt_id
        receipt = read_json(directory / "receipt.json")
        validation = read_json(directory / "validated.json")
        invocation = read_json(directory / "attempt.json")
        self.context(state, invocation)
        verify_execution(self, state, attempt=invocation)
        if any(approval[key] != invocation[key] for key in ("stage", "node", "attempt_id", "input_fingerprint")):
            raise ExtractError("approval scope differs from its invocation")
        if approval.get("execution_binding") != invocation.get("execution_binding"):
            raise ExtractError("approval executor differs from its original invocation")
        if approval["generation"] != invocation["generation"] or approval["plan_revision"] != invocation["plan_revision"]:
            raise ExtractError("approval invocation generation/revision identity changed")
        from .receipts import call_identity
        if receipt.get("identity") != call_identity(invocation) or invocation.get("role") != "reviewer":
            raise ExtractError("approval is not the independent review of its registered invocation")
        if digest(receipt) != approval["receipt_sha256"] or receipt["decision"] != "approve":
            raise ExtractError("approval receipt was changed or did not approve")
        from .stages import writing_signature
        stage = invocation["stage"]
        selected = selection_signature(receipt["plan"]) if stage == "P2" else invocation.get("selection_signature")
        writing = writing_signature(receipt["plan"]) if stage == "P4" else invocation.get("writing_signature")
        if approval.get("selection_signature") != selected or approval.get("writing_signature") != writing:
            raise ExtractError("approval decision signatures differ from their original review")
        if stage in {"P3", "P5"} and (approval.get("node_signature") != invocation["node_signature"] or
                                       validation.get("node_signature") != invocation["node_signature"]):
            raise ExtractError("approval selected instance differs from its original review")
        if approval.get("applies_to_node"):
            entry = next((entry for entry in invocation["selected_designs"] if entry["id"] == approval["applies_to_node"]), None)
            if not entry or approval.get("node_signature") != node_signature(entry):
                raise ExtractError("scoped final approval differs from the actually reviewed instance")
        if stage in {"P3", "P5"} or approval.get("applies_to_node"):
            subject = approval.get("applies_to_node", invocation["node"])
            provenance = validation.get("design_resolution", {}).get("nodes", {}).get(subject, {})
            if (provenance.get("validation") != "strict" or any(approval.get(key) != provenance.get(field) for key, field in
                    (("effective_identity", "effective_identity"), ("source_manifest", "source_manifest"), ("design_origins", "origins")))):
                raise ExtractError("approval provenance differs from its original strictly checked use")
        if validation["receipt_sha256"] != approval["receipt_sha256"] or validation["check"]["exit_code"] != 0:
            raise ExtractError("approval has no matching successful Runner gate")
        if invocation.get("mode") == "adjudicate" and validation.get("adjudication_changed"):
            raise ExtractError("an adjudication edit requires its new independent final signoff")
        if invocation.get("mode") == "signoff":
            from .stages import stage_files, writing_signature
            _, base = self.verify_checkpoint(state, invocation["adjudication_checkpoint"])
            producer = base.get("attempt_id")
            if not producer or producer not in state["accepted_receipts"]:
                raise ExtractError("signoff has no accepted adjudication candidate")
            previous = self.run_dir(state["run_id"]) / "attempts" / producer
            adjudication = read_json(previous / "receipt.json")
            edited = read_json(previous / "validated.json")
            if (adjudication["identity"].get("mode") != "adjudicate" or adjudication["decision"] != "approve" or
                    digest(adjudication) != state["accepted_receipts"][producer] or not edited.get("adjudication_changed") or
                    edited["artifact_digests"] != base["files"] or edited["check"]["exit_code"] != 0 or
                    any(adjudication["identity"][key] != invocation[key] for key in ("stage", "node", "generation"))):
                raise ExtractError("signoff does not bind the exact edited candidate and final stage decision")
            compared = stage_files(invocation["stage"], node_entry={"id": invocation["node"]}, revision_scope=invocation["revision_scope"])
            if any(base["files"].get(name) != invocation["dependency_digests"].get(name) for name in compared):
                raise ExtractError("signoff target changed after adjudication")
            signature = selection_signature if invocation["stage"] == "P2" else writing_signature
            if invocation["stage"] in {"P2", "P4"} and signature(adjudication["plan"]) != signature(receipt["plan"]):
                raise ExtractError("signoff changed the adjudicated stage decision")
        _, checkpoint = self.verify_checkpoint(state, approval["checkpoint"])
        if checkpoint["files"] != approval["artifact_digests"] or checkpoint["files"] != validation["artifact_digests"]:
            raise ExtractError("approval content differs from its checked immutable checkpoint")
        if approval.get("projection") != validation.get("approval_projection") or not approval.get("projection"):
            raise ExtractError("approval has no matching stage subject projection")
        return receipt, checkpoint

    @execution_boundary
    def verify_effective_approval(self, state: dict, approval: dict | None, *, directory: Path | None = None) -> tuple[dict, dict]:
        receipt, checkpoint = self.verify_approval_origin(state, approval)
        if not approval_is_current(state, approval):
            raise ExtractError("approval belongs to a superseded scope or selection")
        if directory is None:
            pointer = state["checkpoint"] if state["stage"] in {"P1", "P2", "P3", "P4", "P6"} else state["merged_checkpoint"]
            directory, _ = self.verify_checkpoint(state, pointer)
        stage = approval["stage"]
        plan = state.get("selected_designs") if stage == "P2" else state.get("writing_plan") if stage == "P4" else None
        projection = approval_projection(stage, directory,
            parser_root=rules_root(self.context(state)["rules_lock"]), plan=plan,
            selected_designs=state.get("selected_designs"), node=approval["node"])
        if projection["subject_sha256"] != approval["projection"]["subject_sha256"]:
            if stage != "P5" or not self._aligned_context(state, approval["node"], approval,
                                                        projection["source_context"]):
                raise ExtractError("approved stage subject changed: " + stage)
        return receipt, checkpoint

    def verify_approval(self, state: dict, approval: dict | None, *, directory: Path | None = None) -> tuple[dict, dict]:
        """Execution admission uses current applicability, not just old evidence."""
        return self.verify_effective_approval(state, approval, directory=directory)

    def _reviewed_context(self, state: dict, approval: dict, node_id: str) -> dict:
        validation = read_json(self.run_dir(state["run_id"]) / "attempts" / approval["attempt_id"] / "validated.json")
        context = validation.get("design_resolution", {}).get("nodes", {}).get(node_id)
        if not context or context.get("validation") != "strict":
            raise ExtractError("node approval lacks strictly checked actual source context: " + node_id)
        return context

    @staticmethod
    def source_identity(state: dict, node_id: str) -> str | None:
        node = state["nodes"][node_id]
        return (node.get("source_context") or node.get("approval") or {}).get("effective_identity")

    def _aligned_context(self, state: dict, node_id: str, approval: dict, current: dict) -> bool:
        pointer = state["nodes"][node_id].get("applicability")
        if not pointer or pointer.get("approval_attempt_id") != approval["attempt_id"]:
            return False
        path = self.run_dir(state["run_id"]) / "attempts" / pointer["attempt_id"] / "validated.json"
        validation = read_json(path)
        if (state["accepted_receipts"].get(pointer["attempt_id"]) != validation.get("receipt_sha256") or
                validation.get("check", {}).get("exit_code") != 0):
            raise ExtractError("current applicability has no accepted successful Runner validation")
        record = validation.get("retained_approvals", {}).get(node_id)
        if not record or digest(record) != pointer["record_sha256"]:
            raise ExtractError("current approval applicability differs from its Runner validation")
        if (record["approval_attempt_id"] != approval["attempt_id"] or record["generation"] != state["nodes"][node_id]["generation"]
                or record["current_context"]["effective_identity"] != current["effective_identity"]):
            return False
        _, checkpoint = self.verify_checkpoint(state, record["checkpoint"])
        if any(checkpoint["files"].get(name) != checksum for name, checksum in current["source_manifest"].items()):
            return False
        previous = self._reviewed_context(state, approval, node_id)
        strict = record["current_context"]
        used = ("file", "raw_sha256", "origins", "contract_sha256", "region_sha256", "use_context_sha256", "source_manifest", "effective_identity")
        if any(strict.get(key) != current.get(key) for key in used):
            return False
        proof = align_source_identity(previous, strict)
        return proof["equivalent"] and proof == record["equivalence"]

    @execution_boundary
    def verify_node(self, state: dict, node_id: str, *, merged_info: dict | None = None) -> dict:
        """Verify this node's actual content and source identities in the current merge."""
        if node_id not in effective_nodes(state):
            raise ExtractError("node is not a current approved merge: " + node_id)
        node = state["nodes"][node_id]
        merged, current_info = self.verify_checkpoint(state, state["merged_checkpoint"])
        if merged_info is not None and merged_info != current_info:
            raise ExtractError("node verification received an obsolete merged checkpoint")
        merged_info = current_info
        approval = node["approval"]
        receipt, approval_checkpoint = (self.verify_approval_origin(state, approval) if approval.get("applies_to_node")
                                         else self.verify_approval(state, approval, directory=merged))
        if approval["stage"] == "P5":
            entry = next(entry for entry in state["selected_designs"] if entry["id"] == node_id)
            if (approval["node"] != node_id or approval["generation"] != node["generation"] or
                    approval["node_signature"] != node_signature(entry)):
                raise ExtractError("node review belongs to another node or superseded selection")
        elif approval["stage"] == "P6":
            if node_id not in receipt["affected_nodes"]:
                raise ExtractError("final review did not explicitly recheck this node")
        elif node_id != "root" or approval["stage"] != "P3":
            raise ExtractError("node has no scoped Design approval")
        if (approval_checkpoint["files"].get(node["file"]) != node["approved_sha256"] or
                merged_info["files"].get(node["file"]) != node["approved_sha256"]):
            raise ExtractError("node approval does not match its current merged bytes: " + node_id)
        graph = source_context(merged, parser_root=rules_root(self.context(state)["rules_lock"]), nodes=[node_id])
        current = graph["nodes"][node_id]
        if current["effective_identity"] != approval.get("effective_identity") and not self._aligned_context(state, node_id, approval, current):
            raise ExtractError("approved node's actual source identity changed: " + node_id)
        if self.source_identity(state, node_id) != current["effective_identity"]:
            raise ExtractError("node does not record its current actual source identity: " + node_id)
        for name, checksum in current["source_manifest"].items():
            source = name.removesuffix(".yaml")
            if source == node_id:
                continue
            if source not in effective_nodes(state) or checksum != merged_info["files"].get(name):
                raise ExtractError("approved node uses a source without current approval: " + source)
            provider = state["nodes"][source]["approval"]
            self.verify_approval_origin(state, provider)
            if (self.source_identity(state, source) != graph["nodes"][source]["effective_identity"] or
                    state["nodes"][source]["approved_sha256"] != merged_info["files"][name]):
                raise ExtractError("source approval differs from current actual provenance: " + source)
            if (provider.get("effective_identity") != graph["nodes"][source]["effective_identity"] and
                    not self._aligned_context(state, source, provider, graph["nodes"][source])):
                raise ExtractError("source identity drift lacks current equivalence evidence: " + source)
        return approval_checkpoint

    @execution_boundary
    def validate_attempt(self, state: dict, attempt: dict) -> None:
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt["attempt_id"]
        if read_json(directory / "attempt.json") != attempt:
            raise ExtractError("agent modified invocation identity")
        self.context(state, attempt)
        config = self.execution_config(state)
        if any(attempt.get(key) != config[key] for key in ("backend", "model")) or (
                attempt.get("effort") != config[attempt["role"] + "_effort"] or
                attempt.get("executable") != config["agent_binary"] or
                attempt.get("session_scope") != session_scope(attempt)):
            raise ExtractError("invocation execution or session identity differs from this run")
        if attempt["stage"] in {"P4", "P6"} and (
                attempt.get("selection_signature") != selection_signature(state["selected_designs"])):
            raise StaleResult("approved selection changed while the invocation ran")
        maintenance_call = attempt.get("purpose") == "maintenance"
        if maintenance_call:
            from .maintenance import validate_attempt as validate_maintenance_attempt
            validate_maintenance_attempt(state, attempt)
        if attempt["run_id"] != state["run_id"] or attempt["input_fingerprint"] != state["input_fingerprint"]:
            raise ExtractError("invocation fixed input identity changed")
        key = scope_key(attempt["stage"], attempt["node"])
        if state["active_attempts"].get(key) != attempt["attempt_id"]:
            raise StaleResult("receipt is not from the active invocation for this scope")
        if attempt["stage"] == "P5":
            node = state["nodes"].get(attempt["node"])
            entry = next((entry for entry in state.get("selected_designs") or [] if entry["id"] == attempt["node"]), None)
            if not node or not entry or attempt["generation"] != node["generation"] or attempt["node_signature"] != node_signature(entry):
                raise StaleResult("node membership or selected instance changed while the invocation ran")
            if state.get("reopening") or not all(state["stages"][stage]["status"] == "approved" for stage in ("P1", "P2", "P3", "P4")):
                raise StaleResult("upstream review is not currently valid")
            if not maintenance_call and attempt["base_checkpoint"] != node["draft_checkpoint"]:
                raise StaleResult("this node's draft checkpoint changed")
            if not maintenance_call and (attempt["role"], attempt["mode"]) != (node["role"], node["mode"]):
                raise StaleResult("this node's role or review mode changed")
            known_sources = actual_sources(node) & {entry["id"] for entry in state["selected_designs"]}
            if known_sources - effective_nodes(state):
                raise StaleResult("an actual source is awaiting its independent repair or review")
        elif not maintenance_call:
            if any(attempt[key] != state[key] for key in ("stage", "node", "role", "mode", "plan_revision")):
                raise StaleResult("upstream stage invocation no longer matches its current scope")
            if attempt["generation"] != scope_for(state, attempt)["generation"] or attempt["base_checkpoint"] != state["checkpoint"]:
                raise StaleResult("stage generation or checkpoint changed")
        if attempt["stage"] in {"P3", "P5"}:
            entry = next((entry for entry in state["selected_designs"] if entry["id"] == attempt["node"]), None)
            if not entry or attempt.get("node_signature") != node_signature(entry):
                raise StaleResult("the current selected instance changed")

    def validate_sources(self, state: dict, attempt: dict, workspace: Path, *, graph: dict | None = None,
                         reviewed_graph: dict | None = None) -> dict:
        """Only adopted definitions must remain provided, independently approved and current."""
        parser_root = rules_root(self.context(state, attempt)["rules_lock"])
        graph = graph or source_context(workspace, parser_root=parser_root, nodes=[attempt["node"]])
        reviewed = reviewed_graph or graph
        submitted = reviewed["nodes"][attempt["node"]]
        offered = attempt.get("source_identities", {})
        used = {}
        for name, checksum in submitted["source_manifest"].items():
            source = name.removesuffix(".yaml")
            if source == attempt["node"]:
                continue
            if source not in offered or attempt["dependency_digests"].get(name) != checksum:
                raise StaleResult("adopted source was not provided in this invocation: " + source)
            if source not in effective_nodes(state):
                raise StaleResult("adopted source no longer has current independent approval: " + source)
            if reviewed["nodes"][source]["effective_identity"] != offered[source]:
                raise StaleResult("adopted source differs from the read and approved invocation material: " + source)
            try:
                self.verify_node(state, source)
            except ExtractError as exc:
                raise StaleResult("adopted source approval or identity changed: " + source + ": " + str(exc)) from exc
            used[source] = self.source_identity(state, source)
        _, merged_info = self.verify_checkpoint(state, state["merged_checkpoint"])
        drift = any(merged_info["files"].get(name) != checksum for name, checksum in submitted["source_manifest"].items()
                    if name != attempt["node"] + ".yaml")
        drift = drift or any(offered[source] != identity for source, identity in used.items())
        alignment = None
        if drift and reviewed_graph is None:
            temporary = self.run_dir(state["run_id"]) / (".source-check-" + uuid4().hex)
            try:
                self.merged_workspace(state, workspace, temporary, node_id=attempt["node"])
                checked = check_stage_snapshot(temporary, parser_root=parser_root, stage="P5",
                                               selected_designs=state["selected_designs"], node=attempt["node"])
                if checked["exit_code"]:
                    raise StaleResult("current adopted definitions do not form a strict completed snapshot")
                graph = checked["design_resolution"]
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        current = graph["nodes"][attempt["node"]]
        if drift:
            alignment = align_source_identity(submitted, current)
            if not alignment["equivalent"]:
                raise StaleResult("actual definitions, shared Contract or complete use context changed while the invocation ran")
        for source, identity in used.items():
            if graph["nodes"][source]["effective_identity"] != identity:
                raise StaleResult("adopted source has no currently applicable provenance: " + source)
        return {"design_resolution": graph, "reviewed_design_resolution": reviewed,
                "source_alignment": alignment, "adopted_source_identities": used,
                "dependency_digests": {name: checksum for name, checksum in current["source_manifest"].items()
                                       if name != attempt["node"] + ".yaml"}}

    def refresh_approvals(self, state: dict, workspace: Path, checkpoint_id: str, attempt: dict, validation: dict) -> None:
        """Retain authentic approvals only after checking the actual current use."""
        parser_root = rules_root(self.context(state)["rules_lock"])
        planned = {entry["id"] for entry in state.get("selected_designs") or []}
        projections = {}
        for stage in ("P2", "P4"):
            if int(stage[1:]) <= int(attempt["stage"][1:]):
                continue
            approval = state["stages"][stage].get("approval")
            if not approval or approval["attempt_id"] == attempt["attempt_id"]:
                continue
            plan = state["selected_designs"] if stage == "P2" else state["writing_plan"]
            projection = approval_projection(stage, workspace, parser_root=parser_root, plan=plan,
                                             selected_designs=state["selected_designs"])
            if (projection["subject_sha256"] != approval["projection"]["subject_sha256"] or
                    not approval_is_current(state, approval)):
                self.verify_approval_origin(state, approval)
                projections[stage] = {"previous_approval": approval["attempt_id"],
                                      "previous_subject": approval["projection"]["subject_sha256"],
                                      "current_subject": projection["subject_sha256"]}
                invalidate_stage(state, stage)
        validation["invalidated_projections"] = projections
        if projections:
            state.setdefault("feedback", []).append({"code": "stage_subject_changed", "stages": projections})
        retained, invalidated = {}, []
        for node_id in sorted(planned):
            node = state["nodes"][node_id]
            approval = node.get("approval")
            if not approval or approval["attempt_id"] == attempt["attempt_id"]:
                continue
            self.verify_approval_origin(state, approval)
            previous = self._reviewed_context(state, approval, node_id)
            checked = check_stage_snapshot(workspace, parser_root=parser_root, stage="P3" if node_id == "root" else "P5",
                                           selected_designs=state["selected_designs"], node=node_id)
            current = checked.get("design_resolution", {}).get("nodes", {}).get(node_id)
            proof = align_source_identity(previous, current) if current and not checked["exit_code"] else None
            if (not proof or not proof["equivalent"] or not approval_is_current(state, approval)):
                if node_id == "root":
                    invalidate_stage(state, "P3")
                else:
                    invalidate_node(node, require_worker=bool(actual_sources(node) - planned))
                if current:
                    node["source_context"] = deepcopy(current)
                node.setdefault("feedback", []).append({"code": "actual_use_changed", "source_check": checked,
                                                         "previous_approval": approval["attempt_id"], "equivalence": proof})
                invalidated.append(node_id)
                continue
            node["source_context"] = deepcopy(current)
            if (proof["current_identity"] != approval.get("effective_identity") or
                    approval.get("selection_signature") != state.get("selection_signature") or
                    approval.get("writing_signature") not in {None, state.get("writing_signature")}):
                record = {"approval_attempt_id": approval["attempt_id"], "generation": node["generation"],
                          "checkpoint": checkpoint_id, "current_context": current, "equivalence": proof,
                          "selection_signature": state.get("selection_signature"), "writing_signature": state.get("writing_signature")}
                retained[node_id] = record
                node["applicability"] = {"attempt_id": attempt["attempt_id"], "approval_attempt_id": approval["attempt_id"],
                                         "record_sha256": digest(record)}
        validation["retained_approvals"] = retained
        validation["invalidated_uses"] = invalidated
        if invalidated or projections:
            invalidate_final(state, invalidated)

    def _adjudication_candidates(self, state: dict, call: dict, before: Path) -> list[dict]:
        """Prepare versions, current materials and canonical bytes before any selection."""
        from .artifacts import _erase_fields, _same, validate_plan
        from .stages import stage_files
        context = self.context(state)
        parser_root = rules_root(context["rules_lock"])
        scope = scope_for(state, call)
        view = call_view(state, call)
        entry = next((item for item in state.get("selected_designs") or [] if item["id"] == call["node"]), None)
        own_signature = node_signature(entry) if entry and call["stage"] in {"P3", "P5"} else ""
        saved = [*scope["candidates"], {"checkpoint": view["checkpoint"], "plan": state.get("plan_draft"),
                                      "node_signature": own_signature}]
        prepared, seen = [], set()
        for candidate in saved:
            if candidate["checkpoint"] in seen:
                continue
            seen.add(candidate["checkpoint"])
            if own_signature and candidate.get("node_signature") != own_signature:
                continue
            temporary = self.run_dir(state["run_id"]) / (".adjudication-" + uuid4().hex)
            original, normalized = temporary / "original", temporary / "normalized"
            try:
                source, _ = self.verify_checkpoint(state, candidate["checkpoint"])
                plan = deepcopy(candidate.get("plan"))
                if call["stage"] in {"P2", "P4"}:
                    validate_plan(call["stage"], plan, parser_root=parser_root,
                                  selected_designs=state.get("selected_designs") if call["stage"] == "P4" else None)
                copy_artifacts(before, original)
                original.chmod(0o755)
                for path in original.iterdir():
                    path.chmod(0o644)
                version_adjustments = []
                for name in stage_files(call["stage"], node_entry=entry, revision_scope=state["revision_scope"]):
                    if not (source / name).is_file():
                        raise ExtractError("candidate has no completed stage file: " + name)
                    copy_artifacts(source / name, original / name)
                    if name.endswith(".yaml"):
                        current = load_yaml(before, name, parser_root=parser_root)
                        selected = load_yaml(original, name, parser_root=parser_root)
                        different = not _same(_erase_fields(current, []), _erase_fields(selected, []))
                        version = max(current["META_DATA"]["Version"] + int(different), selected["META_DATA"]["Version"])
                        if selected["META_DATA"]["Version"] != version:
                            version_adjustments.append({"path": name, "before": selected["META_DATA"]["Version"], "after": version})
                            selected["META_DATA"]["Version"] = version
                            write_yaml(original / name, selected, parser_root=parser_root)
                validation = normalize_candidate(before, original, normalized, parser_root / "schema_template/starter.yaml",
                    stage=call["stage"], role="worker", decision="complete", node_entry=entry if call["stage"] == "P5" else None,
                    selected_designs=plan if call["stage"] == "P2" else state.get("selected_designs"), revision_scope=state["revision_scope"],
                    first_draft=call["stage"] == "P5" and not state["nodes"][call["node"]].get("authored", False))
                validation.update(origin="runner_candidate_preparation", original_checkpoint=candidate["checkpoint"],
                                  reading_checkpoint=view["checkpoint"], version_adjustments=version_adjustments)
                validate_artifact_sources(normalized, context["source_lock"], decision="complete", parser_root=parser_root,
                                          completed_nodes=None if call["stage"] == "P6" else [call["node"]])
                if call["stage"] == "P5":
                    graph = validation["check"]["design_resolution"]
                    adopted = set(graph["nodes"][call["node"]]["source_manifest"]) - {call["node"] + ".yaml"}
                    for name in adopted:
                        source_id = name.removesuffix(".yaml")
                        if source_id not in effective_nodes(state) or graph["nodes"][source_id]["effective_identity"] != self.source_identity(state, source_id):
                            raise ExtractError("candidate needs an unapproved or changed actual source")
                checkpoint_id = self.checkpoint(state, normalized, attempt_id=None, expected_files=validation["artifact_digests"],
                                                stage=call["stage"], node=call["node"], parent=view["checkpoint"])
                prepared.append({"checkpoint": checkpoint_id, "original_checkpoint": candidate["checkpoint"], "plan": plan,
                                 "files": manifest(normalized), "node_signature": own_signature,
                                 "normalization": validation})
            except ExtractError:
                # Incomplete or inapplicable old drafts remain history. They
                # are not candidates a reviewer can silently adopt.
                continue
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return list({item["checkpoint"]: item for item in prepared}.values())

    @execution_boundary
    def prepare_attempt(self, state: dict, session_id: str | None, *, call: dict | None = None) -> dict:
        config = self.execution_config(state)
        verify_execution(self, state, executor=True)
        call = call or next_call(state)
        if call is None:
            raise ExtractError("no dependency-ready call can be prepared")
        if state.get("execution_binding"):
            call = {**call, "execution_binding": state["execution_binding"]}
        key = scope_key(call["stage"], call["node"])
        if key in state["active_attempts"]:
            raise ExtractError("cannot prepare a second attempt before confirmed termination")
        if not any(all(call.get(field) == ready.get(field) for field in
                       ("stage", "node", "role", "mode", "purpose", "maintenance_task_id", "maintenance_step"))
                   for ready in next_calls(state)):
            raise ExtractError("cannot prepare a call outside the current dependency-ready set")
        from .maintenance import prepare_workspace
        prepare_workspace(self, state, call)
        if call.get("purpose") != "maintenance" and call["stage"] == "P5":
            self.start_node(state, call["node"])
        view = call_view(state, call)
        if call.get("purpose") != "maintenance":
            prerequisite_directory, _ = self.verify_checkpoint(state, state["merged_checkpoint"] if call["stage"] == "P5" else view["checkpoint"])
            for required in ("P1", "P2", "P3", "P4"):
                if int(required[1:]) >= int(call["stage"][1:]):
                    break
                if state["stages"][required]["status"] != "approved":
                    raise ExtractError("missing independent prerequisite approval: " + required)
                self.verify_approval(state, state["stages"][required]["approval"], directory=prerequisite_directory)
        scope = scope_for(state, call)
        if scope["started_at"] is None:
            scope["started_at"] = time.time()
        if scope["baseline_checkpoint"] is None:
            scope["baseline_checkpoint"] = view["checkpoint"]
            scope["candidates"].append({"checkpoint": view["checkpoint"], "plan": state.get("plan_draft"),
                                       "generation": scope["generation"], "node_signature": None})
        mode = call.get("mode", "normal")
        if mode in {"adjudicate", "signoff"}:
            counter = "adjudications" if mode == "adjudicate" else "signoffs"
            if scope[counter]:
                raise ExtractError("adjudication/signoff is a single call, never restarted")
            scope[counter] += 1
        scope["calls"] += 1
        group = session_group(state, call)
        state["sessions"].setdefault(group, {"worker": None, "reviewer": None})
        if session_id != state["sessions"][group][call["role"]]:
            raise ExtractError("requested session does not match this exact stage/node/role/operation")
        artifacts, checkpoint = self.verify_checkpoint(state, view["checkpoint"])
        attempt_id = str(uuid4())
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt_id
        directory.mkdir(parents=True)
        shutil.copytree(artifacts, directory / "workspace")
        for path in (directory / "workspace").rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        (directory / "workspace").chmod(0o755)
        invocation = {key: view[key] for key in ("run_id", "input_fingerprint", "stage", "role", "node", "plan_revision")}
        entry = next((entry for entry in state.get("selected_designs") or [] if entry["id"] == call["node"]), None)
        scoped_session = session_scope(call)
        invocation.update({"attempt_id": attempt_id, "base_checkpoint": view["checkpoint"],
                           "response_format": "schema_extract.proposal.v1",
                           "dependency_digests": checkpoint["files"], "session_id": session_id,
                           "session_group": group, "mode": mode, "writable_files": writable_files(state, call),
                           "session_scope": scoped_session,
                           "session_directory": str(self.run_dir(state["run_id"]) / "sessions" / digest(scoped_session)),
                           "backend": config["backend"], "model": config["model"], "executable": config["agent_binary"],
                           "effort": config[call["role"] + "_effort"],
                           "generation": state["nodes"][call["node"]]["generation"] if call["stage"] == "P5" else scope["generation"],
                           "node_signature": node_signature(entry) if entry and call["stage"] in {"P3", "P5"} else "",
                           "selected_designs": deepcopy(state["selected_designs"]),
                           "revision_scope": state["revision_scope"],
                           "workspace": str(directory / "workspace"), "directory": str(directory),
                           "created_at": time.time()})
        if checkpoint.get("preparation"):
            invocation["preparation"] = deepcopy(checkpoint["preparation"])
        invocation.update({key: call[key] for key in ("maintenance_task_id", "purpose", "maintenance_step") if key in call})
        if state.get("execution_binding"):
            invocation["execution_binding"] = state["execution_binding"]
        if call["stage"] == "P5" and state["nodes"][call["node"]].get("baseline_context_checkpoint"):
            invocation["baseline_context_checkpoint"] = state["nodes"][call["node"]]["baseline_context_checkpoint"]
        if mode == "adjudicate":
            invocation["adjudication_candidates"] = self._adjudication_candidates(state, call, artifacts)
        elif mode == "signoff":
            invocation["adjudication_checkpoint"] = scope["adjudication"]["checkpoint"]
        if call["stage"] in {"P3", "P4", "P5", "P6"}:
            invocation["selection_signature"] = selection_signature(state["selected_designs"])
        if call["stage"] in {"P5", "P6"}:
            invocation["writing_signature"] = state["writing_signature"]
        for filename in invocation["writable_files"]:
            path = directory / "workspace" / filename
            if not path.exists():
                path.touch()
        if call["stage"] in {"P5", "P6"}:
            invocation["source_identities"] = {}
            for node_id in sorted(effective_nodes(state)):
                if node_id == call["node"] and call["stage"] == "P5":
                    continue
                node = state["nodes"][node_id]
                if node["file"] in checkpoint["files"]:
                    self.verify_node(state, node_id)
                    invocation["source_identities"][node_id] = self.source_identity(state, node_id)
            invocation.update(design_context(state, invocation, directory / "workspace",
                parser_root=rules_root(self.context(state)["rules_lock"])))
        from .prompts import prepare_attempt_materials
        context = self.context(state)
        invocation.update(prepare_attempt_materials(state, context["source_lock"], invocation,
                          rules_lock=context["rules_lock"], executor_lock=context["executor_lock"]))
        handoff = next((item for item in reversed(state.get("session_handoffs", []))
                        if item["session_group"] == group and item["role"] == call["role"] and
                        not item.get("next_attempt_id")), None)
        if handoff:
            handoff["next_attempt_id"] = attempt_id
            invocation["recovery_handoff"] = deepcopy(handoff)
        write_json(directory / "attempt.json", invocation)
        if call.get("purpose") == "maintenance":
            from .maintenance import task_for
            task_for(state, call).setdefault("attempts", []).append({"attempt_id": attempt_id, "step": call["maintenance_step"]})
        if call["stage"] == "P5":
            state["nodes"][call["node"]]["writing_started"] = writing_order(state, call["node"])
        state["attempts"].append(attempt_id)
        state["active_attempts"][key] = attempt_id
        state.setdefault("unsettled_attempts", {})[attempt_id] = "prepared"
        if call["stage"] == "P6" and not state["final_review_started"]:
            state["final_review_started"] = True
            state["p6_starts"] += 1
        self.save(state)
        return invocation

    def _node_draft(self, state: dict, node_id: str):
        node = state["nodes"][node_id]
        candidates = [node.get("draft_checkpoint"), node.get("last_draft_checkpoint"),
                      (node.get("approval") or {}).get("checkpoint"),
                      (node.get("last_approval") or {}).get("checkpoint"), state.get("merged_checkpoint")]
        for previous in dict.fromkeys(item for item in candidates if item):
            directory, info = self.verify_checkpoint(state, previous)
            if node["file"] in info["files"]:
                return previous, directory, info
        return None

    def prepare_shared_metadata(self, state: dict, checkpoint_id: str, call: dict) -> str:
        """Present current existing drafts read-only to the P1 root author."""
        selected = {entry["id"] for entry in state.get("selected_designs") or []}
        if not selected - {"root"}:
            return checkpoint_id
        source, info = self.verify_checkpoint(state, checkpoint_id)
        temporary = self.run_dir(state["run_id"]) / (".shared-metadata-" + uuid4().hex)
        temporary.mkdir()
        try:
            expected = set()
            for name in info["files"]:
                if name == "dispatch.md" or name.removesuffix(".yaml") in selected:
                    shutil.copyfile(source / name, temporary / name)
                    expected.add(name)
            for node_id in sorted(selected - {"root"}):
                material = self._node_draft(state, node_id)
                if material:
                    previous, directory, old_info = material
                    node = state["nodes"][node_id]
                    shutil.copyfile(directory / node["file"], temporary / node["file"])
                    expected.add(node["file"])
                    if actual_sources(node) - selected and not node.get("baseline_context_checkpoint"):
                        if any(name.endswith(".yaml") and name.removesuffix(".yaml") not in selected
                               for name in old_info["files"]):
                            node["baseline_context_checkpoint"] = previous
            if artifact_manifest(temporary, expected=expected) == info["files"]:
                return checkpoint_id
            return self.checkpoint(state, temporary, attempt_id=None, stage="P1", node="root",
                                   parent=checkpoint_id, call=call, expected_files=expected)
        finally:
            shutil.rmtree(temporary)

    def shared_drafts(self, state: dict, checkpoint_id: str) -> None:
        """Keep the checked P1 draft bytes without granting Design approval."""
        _, info = self.verify_checkpoint(state, checkpoint_id)
        for entry in state.get("selected_designs") or []:
            node = state["nodes"][entry["id"]]
            if node["id"] == "root" or node["file"] not in info["files"]:
                continue
            if node.get("draft_checkpoint") != checkpoint_id:
                node["last_draft_checkpoint"] = node.get("draft_checkpoint") or node.get("last_draft_checkpoint")
                node["draft_checkpoint"] = checkpoint_id

    def start_node(self, state: dict, node_id: str, *, call: dict | None = None) -> None:
        """Refresh reading material while preserving the node's own raw draft."""
        node = state["nodes"][node_id]
        parser_root = rules_root(self.context(state)["rules_lock"])
        merged, checkpoint = self.verify_checkpoint(state, state["merged_checkpoint"])
        selected = {entry["id"] for entry in state["selected_designs"]}
        if not actual_sources(node) - selected:
            node.pop("baseline_context_checkpoint", None)
        for source in sorted(effective_nodes(state)):
            self.verify_node(state, source, merged_info=checkpoint)
        temporary = self.run_dir(state["run_id"]) / (".preparing-" + uuid4().hex)
        temporary.mkdir()
        try:
            preparation = None
            for name in checkpoint["files"]:
                if name == node["file"]:
                    continue
                shutil.copyfile(merged / name, temporary / name)
            own = None
            material = self._node_draft(state, node_id)
            if material:
                previous, old, old_info = material
                own = old / node["file"]
                if actual_sources(node) - selected and any(name.endswith(".yaml") and name != node["file"]
                        and name not in checkpoint["files"] for name in old_info["files"]):
                    node["baseline_context_checkpoint"] = previous
            if own:
                shutil.copyfile(own, temporary / node["file"])
                # Restoring an existing raw draft uses the same approved shared
                # metadata as creating a starter. The old immutable raw remains
                # historical; only this new, still unapproved working copy changes.
                from .artifacts import _synchronize_metadata
                old = {"root": load_yaml(own.parent, "root.yaml", parser_root=parser_root)}
                new = {"root": load_yaml(merged, "root.yaml", parser_root=parser_root)}
                synchronization = _synchronize_metadata(own.parent, temporary, old, new,
                                                         parser_root=parser_root, nodes=[node_id])
                if synchronization["nodes"]:
                    shared = state["stages"]["P1"]["approval"]
                    self.verify_approval(state, shared, directory=merged)
                    preparation = {"kind": "shared_metadata_initialization", "original_checkpoint": previous,
                                   "p1_approval": {key: shared[key] for key in ("attempt_id", "receipt_sha256", "checkpoint")},
                                   "root_sha256": checkpoint["files"]["root.yaml"], "synchronization": synchronization}
            else:
                starter = parser_root / "schema_template/starter.yaml"
                derived = load_yaml(starter.parent, starter.name, parser_root=parser_root)
                root = load_yaml(merged, "root.yaml", parser_root=parser_root)
                derived.pop("CONTRACT")
                for field in ("Name", "Describe"):
                    derived["META_DATA"][field] = deepcopy(root["META_DATA"][field])
                derived["META_DATA"]["Version"] = 1
                derived["META_DATA"]["Dispatch"] = {"id": node_id, "when": None}
                write_yaml(temporary / node["file"], derived, parser_root=parser_root)
            if node.get("draft_checkpoint"):
                _, old_info = self.verify_checkpoint(state, node["draft_checkpoint"])
                if old_info["files"] == manifest(temporary):
                    return
            node["draft_checkpoint"] = self.checkpoint(state, temporary, attempt_id=None, stage="P5", node=node_id,
                                                       parent=node.get("draft_checkpoint") or state["merged_checkpoint"], call=call,
                                                       preparation=preparation, expected_files=set(checkpoint["files"]) | {node["file"]})
            node["authored"] = node.get("authored", False)
        finally:
            shutil.rmtree(temporary)

    def merged_workspace(self, state: dict, workspace: Path, destination: Path, *, node_id: str) -> Path:
        """Adopt only the reviewed target raw into the latest merged snapshot."""
        merged, _ = self.verify_checkpoint(state, state["merged_checkpoint"])
        filename = state["nodes"][node_id]["file"]
        copy_artifacts(merged, destination)
        for path in destination.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        destination.chmod(0o755)
        copy_artifacts(workspace / filename, destination / filename)
        return destination


def summary(store: Store, state: dict) -> dict:
    """Project recorded progress without reading or approving any artifact."""
    planned = {entry["id"] for entry in state.get("selected_designs") or []}
    approved = effective_nodes(state)
    groups = {key: [] for key in ("ready", "in_flight", "revision", "approved", "waiting", "blocked")}
    waits = {entry["id"]: entry["wait_for"] for entry in state.get("writing_plan") or []}
    ready = ({call["node"] for call in next_calls(state)}
             if state["status"] == "running" and state["stage"] != "P6" and
             not (state.get("control") or {}).get("action") else set())
    active = state.get("active_attempts", {})
    details = {}
    observed = planned | {"root"} | {key.removeprefix("P5:") for key in active if key.startswith("P5:")}
    for node_id in sorted(observed):
        node = state["nodes"][node_id]
        attempt_id = active.get("P5:" + node_id)
        if node_id == "root":
            attempt_id = active.get(state["stage"])
        if attempt_id:
            groups["in_flight"].append(node_id)
        elif node_id in approved:
            groups["approved"].append(node_id)
        elif node["status"] in NODE_STOPS:
            groups["blocked"].append(node_id)
        elif node_id in ready:
            groups["ready"].append(node_id)
        elif set(waits.get(node_id, [])) - approved:
            groups["waiting"].append(node_id)
        if node.get("stage", {}).get("status") in {"revision", "adjudicating"}:
            groups["revision"].append(node_id)
        details[node_id] = {"wait_for": waits.get(node_id, []), "status": node["status"], "role": node.get("role"),
                            "mode": node.get("mode", "normal"), "attempt_id": attempt_id,
                            "reason": node.get("reason"), "generation": node["generation"],
                            "planned": node_id in planned, "instance": node.get("instance"),
                            "provider_reopen": node.get("provider_reopen")}
    scopes = {**state["stages"], **{"P5:" + node: state["nodes"][node]["stage"]
                                   for node in observed if "stage" in state["nodes"][node]}}
    scope_fields = ("status", "calls", "revisions", "elapsed_seconds", "started_at",
                     "adjudications", "signoffs")
    from .audit import usage_summary, usage_totals
    from .observation import observation_totals
    rows = [{"run_id": state["run_id"], "origin_run_id": state["run_id"], "attempt_id": key,
             "backend": state["config"]["backend"],
             "usage": usage_summary({"backend": state["config"]["backend"], "counter_scope": "unknown"}),
             **state.get("attempt_usage", {}).get(key, {})} for key in state.get("attempts", [])]
    return {key: state.get(key) for key in ("run_id", "operator_id", "status", "stage", "role", "node",
                                          "checkpoint", "output", "published", "reason")} | {
        "format": SUMMARY_FORMAT, "input_fingerprint": state["input_fingerprint"],
        "execution_binding": state.get("execution_binding"),
        "source_identity": deepcopy(state.get("source_identity")),
        "execution": {key: state.get("config", {}).get(key) for key in
                      ("backend", "worker_effort", "reviewer_effort", "model", "agent_binary")},
        "executor": deepcopy(state.get("executor", {"version": None, "digest": None, "rules_id": None,
                                                     "capabilities": []})),
        "state_revision": state.get("state_revision"), "generated_at": time.time(),
        "created_at": state.get("created_at"), "updated_at": state.get("updated_at"),
        "content_verified": False,
        "mode": state.get("mode", "normal"), "state_directory": str(store.run_dir(state["run_id"])),
        "approved_stages": [key for key, value in state["stages"].items() if value["status"] == "approved"],
        "approved_nodes": sorted(approved), "nodes_by_status": groups, "node_details": details,
        "active_attempts": dict(active), "unsettled_attempts": dict(state.get("unsettled_attempts", {})),
        "runner": state.get("runner"),
        "unconfirmed_attempts": dict(state.get("unconfirmed_attempts", {})),
        "control": state.get("control"), "publication": state.get("publication"),
        "exports": state.get("exports", []),
        "elapsed_seconds": sum(item.get("elapsed_seconds", 0) for item in state.get("attempt_usage", {}).values()),
        "provider_usage": usage_totals(rows),
        "root_usage": observation_totals([row for row in rows if row.get("backend") == "codex" or
                                           (row.get("usage") or {}).get("backend") == "codex"]),
        "calls": {"limit": call_limit(state), "revision_limit": state["config"]["revision_limit"],
                  "additions": deepcopy(state.get("call_additions", [])),
                  "scopes": {key: {field: value.get(field) for field in scope_fields}
                             for key, value in scopes.items()}},
        "completed_node_count": len(approved), "plan_revision": state["plan_revision"],
        "p6_starts": state.get("p6_starts", 0), "planned_node_count": len(planned),
        "attempt_count": len(state.get("attempts", [])),
        "last_attempt_id": next(iter(state.get("attempts", [])[::-1]), None),
        "merged_checkpoint": state.get("merged_checkpoint")}
