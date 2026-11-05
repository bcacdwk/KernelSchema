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
from .common import (STATE_GUARD, ExtractError, RunnerBusy, StaleResult, copy_artifacts, digest, json_bytes,
                     fsync_directory, manifest, read_json, seal, sync_tree, verify_manifest, write_json, write_text, rules_root)
from .artifacts import DESIGN_REGIONS, _same, design_context, load_yaml, validate_artifact_sources, write_yaml
from .budget import call_limit, review_seconds
from .runtime import (EXECUTOR_CAPABILITIES, STATE_FORMAT as RUNTIME_STATE_FORMAT,
                      execution_boundary, verify_execution)
from .revisions import (STATE_FORMAT as SEMANTIC_STATE_FORMAT, CONTEXT_FIELDS, resolve_context,
                        verify_effective_approval as verify_effective_context_approval)
from .stages import (NODE_STOPS, call_view, effective_nodes, new_scope, next_call, next_calls, node_signature,
                     scope_for, scope_key, session_group, writable_files)


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

    def initialize(self) -> None:
        marker = self.root / "format.json"
        if marker.exists():
            if read_json(marker) != {"format": "schema_extract.store.v2"}:
                raise ExtractError("state root belongs to another format")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "supervisor.lock").open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
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
                    "holds": {}, "starts": {}, "controls": {}, "continuations": {},
                    "retries": {}, "auto_restart": {}, "observations": {}}
        value = read_json(path)
        if not isinstance(value, dict) or value.get("format") != SUPERVISOR_FORMAT:
            raise ExtractError("unsupported supervisor control ledger")
        for name in ("holds", "starts", "controls", "continuations", "retries", "auto_restart", "observations"):
            if not isinstance(value.get(name), dict):
                raise ExtractError("invalid supervisor field: " + name)
        return value

    @contextmanager
    def supervisor(self):
        """Short control transaction. Never acquire a run lock or wait here."""
        self.initialize()
        with (self.root / "supervisor.lock").open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
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
        if run_id in ledger["continuations"]:
            raise ExtractError("run has a reserved successor; use " + ledger["continuations"][run_id]["run_id"])
        self.assert_constructed(run_id, ledger)
        if ledger["active_hold"]:
            raise ExtractError("global hold is active: " + ledger["active_hold"])

    def assert_constructed(self, run_id: str, ledger: dict | None = None) -> None:
        """Both dispatch and further continuation require committed inheritance."""
        ledger = self.read_supervisor() if ledger is None else ledger
        if any(value["run_id"] == run_id and value["status"] != "ready" for value in ledger["continuations"].values()):
            raise ExtractError("successor is still being constructed; its original budget is not yet committed")

    def append_control(self, ledger: dict, run_id: str, action: str, *, hold_id=None) -> dict:
        """Caller holds supervisor.lock; requests remain the Runner's input."""
        previous = ledger["controls"].get(run_id, {})
        request = {"format": "schema_extract.control.v2", "run_id": run_id,
                   "id": str(uuid4()), "action": action, "created_at": time.time(),
                   "control_revision": previous.get("revision", 0) + 1,
                   "origin": "hold" if hold_id else "manual", "hold_id": hold_id}
        write_json(self.root / "requests" / run_id / (request["id"] + ".json"), request)
        ledger["controls"][run_id] = {"revision": request["control_revision"], "id": request["id"],
                                      "action": action, "hold_id": hold_id, "at": request["created_at"]}
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
        if not isinstance(value, dict) or value.get("format") not in {STATE_FORMAT, RUNTIME_STATE_FORMAT, SEMANTIC_STATE_FORMAT} or value.get("run_id") != run_id:
            raise ExtractError("unsupported or mismatched state; v1 runs must use their frozen executor")
        if "active_attempts" not in value or "reopen_requests" not in value:
            raise ExtractError("this older serial run must use its frozen executor; import its checkpoint into a new run")
        return value

    def save(self, state: dict) -> None:
        if state.get("format") not in {STATE_FORMAT, RUNTIME_STATE_FORMAT, SEMANTIC_STATE_FORMAT}:
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
        self.load(run_id)
        if action not in {"pause", "cancel"}:
            raise ExtractError("control action must be pause or cancel")
        with self.supervisor() as ledger:
            return self.append_control(ledger, run_id, action)

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

    def create(self, lock: dict, output: Path, *, seed: dict | None = None, queued=False,
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
                 "mode": "normal", "plan": None, "plan_draft": None, "merged_checkpoint": None,
                 "revision_scope": [], "pending_rechecks": [], "recoverable_checkpoint": None,
                 "plan_revision": 1, "created_at": now,
                 "executor": {"version": __version__, "rules_id": lock["rules"]["id"],
                              "digest": digest({name: value for name, value in lock["rules"]["files"].items()
                                                if name.startswith("schema_extract/") and name.endswith(".py")}),
                              "capabilities": list(EXECUTOR_CAPABILITIES)},
                 "reopening": None, "reopen_requests": {}, "plan_history": [],
                 "stale_results": {}, "unconfirmed_attempts": {}, "p6_starts": 0, "final_review_started": False,
                 "config": lock["config"], "deadline": now + lock["config"]["operator_seconds"],
                 "budget_extensions": [],
                 "sessions": {group: {"worker": None, "reviewer": None}
                              for group in ("P1_P2", "P3", "P4", "P6", "P6_signoff")},
                 "stages": {stage: new_scope() for stage in ("P1", "P2", "P3", "P4", "P5", "P6")},
                 "nodes": {"root": {"file": "root.yaml", "parent": "", "status": "draft", "merged": False,
                                     "draft_checkpoint": None, "approved_checkpoints": {},
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
        if seed:
            imported = directory / "imported"
            shutil.copytree(seed["artifact_directory"], imported / "artifacts")
            verify_manifest(imported / "artifacts", seed["checkpoint_record"]["files"])
            for name, value in (("checkpoint.json", seed["checkpoint_record"]), ("inputs.lock.json", seed["input_lock"]),
                                ("task.json", seed["task_record"]), ("provenance.json", seed["provenance"])):
                write_json(imported / name, value)
            root = load_yaml(imported / "artifacts", "root.yaml", parser_root=rules_root(lock))
            starter = load_yaml(initial, "root.yaml", parser_root=rules_root(lock))
            adjustments = []
            hardware = starter["contract"]["hardware_specification"]
            if not _same(root["contract"].get("hardware_specification"), hardware):
                root["contract"]["hardware_specification"] = deepcopy(hardware)
                root["meta_data"]["version"] += 1
                adjustments.append("Hardware placeholder restored to the current starter; original bytes remain in imported/artifacts.")
            write_yaml(initial / "root.yaml", root, source=imported / "artifacts/root.yaml",
                       parser_root=rules_root(lock))
            validate_artifact_sources(initial, lock, decision="continue")
            state["imported_draft"] = {**seed["provenance"], "artifacts": str(imported / "artifacts"),
                                       "initial_adjustments": adjustments, "approval_inherited": False}
            state["feedback"] = [{"detail": "按当前P1独立重审导入的Contract；旧Design/索引/派生只作后续阅读线索。",
                                   "prior_issues": seed["task_record"].get("issues", [])}]
            seal(imported)
            sync_tree(imported)
        state["checkpoint"] = self.checkpoint(state, initial, attempt_id=None)
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

    def checkpoint(self, state: dict, artifacts: Path, *, attempt_id: str | None,
                   stage: str | None = None, node: str | None = None, parent: str | None = None,
                   call: dict | None = None) -> str:
        files = manifest(artifacts)
        identity = {"format": "schema_extract.checkpoint.v2", "run_id": state["run_id"],
                    "stage": stage or state["stage"], "node": node or state["node"], "attempt_id": attempt_id,
                    "plan_revision": state["plan_revision"], "parent": parent or state["checkpoint"],
                    "input_fingerprint": state["input_fingerprint"], "files": files}
        if state.get("semantic_revision"):
            identity["semantic_revision"] = state["semantic_revision"]
        if call is not None:
            identity.update({key: call[key] for key in CONTEXT_FIELDS if key in call})
        if state.get("semantic_revision") and attempt_id:
            invocation = read_json(self.run_dir(state["run_id"]) / "attempts" / attempt_id / "attempt.json")
            identity.update({key: invocation[key] for key in CONTEXT_FIELDS if key in invocation})
        checkpoint_id = digest(identity)
        base = self.run_dir(state["run_id"]) / "checkpoints"
        destination = base / checkpoint_id
        if destination.exists():
            self.verify_checkpoint(state, checkpoint_id)
            return checkpoint_id
        temporary = base / (".building-" + uuid4().hex)
        temporary.mkdir(parents=True)
        shutil.copytree(artifacts, temporary / "artifacts")
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
        if state.get("execution_binding") or state.get("semantic_revision"):
            resolve_context(self, state, invocation)
        if any(approval[key] != invocation[key] for key in ("stage", "node", "attempt_id", "input_fingerprint")):
            raise ExtractError("approval scope differs from its invocation")
        if approval["generation"] != invocation["generation"] or approval["plan_revision"] != invocation["plan_revision"]:
            raise ExtractError("approval invocation generation/revision identity changed")
        if invocation.get("semantic_revision"):
            from .receipts import call_identity
            if receipt.get("identity") != call_identity(invocation):
                raise ExtractError("approval receipt semantic context differs from its invocation")
            for key in CONTEXT_FIELDS:
                if approval.get(key) != invocation.get(key) or validation.get(key) != invocation.get(key):
                    raise ExtractError("approval or gate semantic context changed: " + key)
        if digest(receipt) != approval["receipt_sha256"] or receipt["decision"] != "approve":
            raise ExtractError("approval receipt was changed or did not approve")
        if validation["receipt_sha256"] != approval["receipt_sha256"] or validation["check"]["exit_code"] != 0:
            raise ExtractError("approval has no matching successful Runner gate")
        _, checkpoint = self.verify_checkpoint(state, approval["checkpoint"])
        if invocation.get("semantic_revision") and any(checkpoint.get(key) != invocation.get(key) for key in CONTEXT_FIELDS):
            raise ExtractError("approval checkpoint semantic context changed")
        if checkpoint["files"] != approval["artifact_digests"] or checkpoint["files"] != validation["artifact_digests"]:
            raise ExtractError("approval content differs from its checked immutable checkpoint")
        return receipt, checkpoint

    @execution_boundary
    def verify_effective_approval(self, state: dict, approval: dict | None) -> tuple[dict, dict]:
        return verify_effective_context_approval(self, state, approval)

    def verify_approval(self, state: dict, approval: dict | None) -> tuple[dict, dict]:
        """Execution admission uses current applicability, not just old evidence."""
        return self.verify_effective_approval(state, approval)

    @execution_boundary
    def verify_node(self, state: dict, node_id: str, *, merged_info: dict | None = None) -> dict:
        """Current validity depends on this node and its ancestors, never siblings."""
        if node_id not in effective_nodes(state):
            raise ExtractError("node is not a current approved merge: " + node_id)
        node = state["nodes"][node_id]
        receipt, approval_checkpoint = self.verify_approval(state, node["approval"])
        approval = node["approval"]
        if approval["stage"] == "P5":
            entry = next(entry for entry in state["plan"] if entry["id"] == node_id)
            if (approval["node"] != node_id or approval["generation"] != node["generation"] or
                    approval["node_signature"] != node_signature(entry)):
                raise ExtractError("node review belongs to another node or superseded selection")
        elif approval["stage"] == "P6":
            if node_id not in receipt["affected_nodes"]:
                raise ExtractError("final review did not explicitly recheck this node")
        elif node_id != "root" or approval["stage"] != "P3":
            raise ExtractError("node has no scoped Design approval")
        if merged_info is None:
            _, merged_info = self.verify_checkpoint(state, state["merged_checkpoint"])
        if (approval_checkpoint["files"].get(node["file"]) != node["approved_sha256"] or
                merged_info["files"].get(node["file"]) != node["approved_sha256"]):
            raise ExtractError("node approval does not match its current merged bytes: " + node_id)
        from .revisions import effective_dependency_digests
        dependencies = effective_dependency_digests(self, state, approval, approval_checkpoint)
        parent = node["parent"]
        while parent:
            ancestor = state["nodes"][parent]
            if parent not in effective_nodes(state) or dependencies.get(ancestor["file"]) != merged_info["files"].get(ancestor["file"]):
                raise ExtractError("approved node ancestor dependency changed: " + node_id)
            parent = ancestor["parent"]
        return approval_checkpoint

    @execution_boundary
    def validate_attempt(self, state: dict, attempt: dict) -> None:
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt["attempt_id"]
        if read_json(directory / "attempt.json") != attempt:
            raise ExtractError("agent modified invocation identity")
        if state.get("execution_binding") or state.get("semantic_revision"):
            resolve_context(self, state, attempt)
        if attempt.get("semantic_revision") != state.get("semantic_revision"):
            raise StaleResult("invocation semantic context was superseded")
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
            entry = next((entry for entry in state.get("plan") or [] if entry["id"] == attempt["node"]), None)
            if not node or not entry or attempt["generation"] != node["generation"] or attempt["node_signature"] != node_signature(entry):
                raise StaleResult("node membership, parent or selection changed while the invocation ran")
            if state.get("reopening") or not all(state["stages"][stage]["status"] == "approved" for stage in ("P1", "P2", "P3", "P4")):
                raise StaleResult("upstream review is not currently valid")
            if not maintenance_call and attempt["base_checkpoint"] != node["draft_checkpoint"]:
                raise StaleResult("this node's draft checkpoint changed")
            if not maintenance_call and (attempt["role"], attempt["mode"]) != (node["role"], node["mode"]):
                raise StaleResult("this node's role or review mode changed")
            _, merged = self.verify_checkpoint(state, state["merged_checkpoint"])
            parent = node["parent"]
            while parent:
                ancestor = state["nodes"][parent]
                self.verify_node(state, parent, merged_info=merged)
                if attempt["dependency_digests"].get(ancestor["file"]) != merged["files"].get(ancestor["file"]):
                    raise StaleResult("approved parent or Contract changed while the invocation ran")
                parent = ancestor["parent"]
        elif not maintenance_call:
            if any(attempt[key] != state[key] for key in ("stage", "node", "role", "mode", "plan_revision")):
                raise StaleResult("upstream stage invocation no longer matches its current scope")
            if attempt["generation"] != scope_for(state, attempt)["generation"] or attempt["base_checkpoint"] != state["checkpoint"]:
                raise StaleResult("stage generation or checkpoint changed")

    @execution_boundary
    def prepare_attempt(self, state: dict, session_id: str | None, *, call: dict | None = None,
                        timeout_seconds: float | None = None) -> dict:
        if state.get("execution_binding") or state.get("semantic_revision"):
            verify_execution(self, state, executor=True)
        call = call or next_call(state)
        if call is None:
            raise ExtractError("no dependency-ready call can be prepared")
        key = scope_key(call["stage"], call["node"])
        if key in state["active_attempts"]:
            raise ExtractError("cannot prepare a second attempt before confirmed termination")
        from .stages import next_calls
        if not any(all(call.get(field) == ready.get(field) for field in
                       ("stage", "node", "role", "mode", "purpose", "maintenance_task_id", "maintenance_step"))
                   for ready in next_calls(state)):
            raise ExtractError("cannot prepare a call outside the current dependency-ready set")
        from .maintenance import prepare_workspace
        prepare_workspace(self, state, call)
        if call.get("purpose") != "maintenance" and call["stage"] == "P5":
            self.start_node(state, call["node"])
        view = call_view(state, call)
        scope = scope_for(state, call)
        if scope["started_at"] is None:
            scope["started_at"] = time.time()
            scope["deadline"] = scope["started_at"] + state["config"]["stage_seconds"]
            scope["adjudication_reserved_seconds"] = state["config"]["adjudication_seconds"]
            scope["signoff_reserved_seconds"] = state["config"]["adjudication_seconds"] if call["stage"] == "P6" else 0
            scope["review_reserved_seconds"] = review_seconds(state["config"])
        if scope["baseline_checkpoint"] is None:
            scope["baseline_checkpoint"] = view["checkpoint"]
            scope["candidates"].append({"checkpoint": view["checkpoint"], "plan": state.get("plan_draft")})
        mode = call.get("mode", "normal")
        if mode in {"adjudicate", "signoff"}:
            counter = "adjudications" if mode == "adjudicate" else "signoffs"
            if scope[counter]:
                raise ExtractError("adjudication/signoff is a single call, never restarted")
            scope[counter] += 1
        scope["calls"] += 1
        group = session_group(state, call)
        state["sessions"].setdefault(group, {"worker": None, "reviewer": None})
        artifacts, checkpoint = self.verify_checkpoint(state, view["checkpoint"])
        attempt_id = str(uuid4())
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt_id
        directory.mkdir(parents=True)
        shutil.copytree(artifacts, directory / "workspace")
        for path in (directory / "workspace").rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        (directory / "workspace").chmod(0o755)
        invocation = {key: view[key] for key in ("run_id", "input_fingerprint", "stage", "role", "node", "plan_revision")}
        entry = next((entry for entry in state.get("plan") or [] if entry["id"] == call["node"]), None)
        invocation.update({"attempt_id": attempt_id, "base_checkpoint": view["checkpoint"],
                           "response_format": "schema_extract.proposal.v1",
                           "dependency_digests": checkpoint["files"], "session_id": session_id,
                           "session_group": group, "mode": mode, "writable_files": writable_files(state, call),
                           "generation": state["nodes"][call["node"]]["generation"] if call["stage"] == "P5" else scope["generation"],
                           "node_signature": node_signature(entry) if entry and call["stage"] == "P5" else "",
                           "revision_scope": state["revision_scope"],
                           "workspace": str(directory / "workspace"), "directory": str(directory),
                           "created_at": time.time()})
        if state.get("semantic_revision"):
            invocation["semantic_revision"] = state["semantic_revision"]
        invocation.update({key: call[key] for key in CONTEXT_FIELDS if key in call})
        if timeout_seconds is not None:
            invocation["timeout_seconds"] = timeout_seconds
        if state.get("execution_binding"):
            invocation["execution_binding"] = state["execution_binding"]
        if call["stage"] in {"P5", "P6"}:
            invocation.update(design_context(state, invocation, directory / "workspace",
                parser_root=rules_root(resolve_context(self, state)["rules_lock"])))
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
        state["attempts"].append(attempt_id)
        state["active_attempts"][key] = attempt_id
        state.setdefault("unsettled_attempts", {})[attempt_id] = "prepared"
        if call["stage"] == "P6" and not state["final_review_started"]:
            state["final_review_started"] = True
            state["p6_starts"] += 1
        self.save(state)
        return invocation

    def start_node(self, state: dict, node_id: str, *, call: dict | None = None) -> None:
        """Refresh read-only ancestors; an authored node always keeps its own body."""
        node = state["nodes"][node_id]
        parser_root = rules_root(resolve_context(self, state)["rules_lock"])
        merged, checkpoint = self.verify_checkpoint(state, state["merged_checkpoint"])
        ancestors, parent = [], node["parent"]
        while parent:
            value = state["nodes"][parent]
            self.verify_node(state, parent, merged_info=checkpoint)
            ancestors.append(value["file"])
            parent = value["parent"]
        temporary = self.run_dir(state["run_id"]) / (".preparing-" + uuid4().hex)
        temporary.mkdir()
        try:
            for name in ancestors:
                shutil.copyfile(merged / name, temporary / name)
            parent_file = state["nodes"][node["parent"]]["file"]
            own = None
            candidates = [node.get("draft_checkpoint"), node.get("last_draft_checkpoint"),
                          (node.get("approval") or {}).get("checkpoint"),
                          (node.get("last_approval") or {}).get("checkpoint"), state["merged_checkpoint"]]
            for previous in dict.fromkeys(item for item in candidates if item):
                old, old_info = self.verify_checkpoint(state, previous)
                if node["file"] in old_info["files"]:
                    own = old / node["file"]
                    break
            if own:
                shutil.copyfile(own, temporary / node["file"])
            else:
                derived = load_yaml(merged, parent_file, parser_root=parser_root)
                derived.pop("contract", None)
                derived.pop("realization", None)
                derived["design"] = {region: {"from_parent": node["parent"]} for region in DESIGN_REGIONS}
                derived["meta_data"]["version"] = 1
                derived["meta_data"]["dispatch"] = {"id": node_id, "parent": node["parent"], "when": ""}
                write_yaml(temporary / node["file"], derived, source=merged / parent_file, parser_root=parser_root)
            if node.get("draft_checkpoint"):
                _, old_info = self.verify_checkpoint(state, node["draft_checkpoint"])
                if old_info["files"] == manifest(temporary):
                    node["parent_sha256"] = checkpoint["files"][parent_file]
                    return
            node["draft_checkpoint"] = self.checkpoint(state, temporary, attempt_id=None, stage="P5", node=node_id,
                                                       parent=node.get("draft_checkpoint") or state["merged_checkpoint"], call=call)
            node["parent_sha256"] = checkpoint["files"][parent_file]
            # The initial clone is a saved checkpoint too. A request to reopen
            # before writing anything must not turn that clone into a first draft.
            node["authored"] = node.get("authored", False)
        finally:
            shutil.rmtree(temporary)

    def merged_workspace(self, state: dict, workspace: Path, destination: Path, *, node_id: str) -> Path:
        """Serial commit input: a node proposal cannot replace ancestors or siblings."""
        merged, info = self.verify_checkpoint(state, state["merged_checkpoint"])
        filename = state["nodes"][node_id]["file"]
        private = manifest(workspace)
        for name, value in private.items():
            if name != filename and info["files"].get(name) != value:
                raise ExtractError("node dependencies changed before serial merge")
        copy_artifacts(merged, destination)
        for path in destination.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        destination.chmod(0o755)
        copy_artifacts(workspace / filename, destination / filename)
        return destination

    def handoff(self, state: dict, attempt: dict, text: str, lock: dict) -> None:
        directory = self.run_dir(state["run_id"]) / "stages" / attempt["stage"]
        directory.mkdir(parents=True, exist_ok=True)
        context = resolve_context(self, state, attempt)
        body = (f"Checkpoint: {state['checkpoint']}\nSource: {context['source_lock']['source']['id']}\n"
                f"Rules: {context['rules_lock']['rules']['id']}\n\n{text.rstrip()}\n")
        write_text(directory / "handoff.md", body)
        sync_tree(directory)
        fsync_directory(directory.parent)


def summary(store: Store, state: dict) -> dict:
    """Project recorded progress without reading or approving any artifact."""
    planned = {entry["id"] for entry in state.get("plan") or []}
    approved = effective_nodes(state)
    groups = {key: [] for key in ("ready", "in_flight", "revision", "approved", "waiting_parent", "blocked")}
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
        elif node["parent"] and node["parent"] not in approved:
            groups["waiting_parent"].append(node_id)
        if node.get("stage", {}).get("status") in {"revision", "adjudicating"}:
            groups["revision"].append(node_id)
        details[node_id] = {"parent": node["parent"], "status": node["status"], "role": node.get("role"),
                            "mode": node.get("mode", "normal"), "attempt_id": attempt_id,
                            "reason": node.get("reason"), "generation": node["generation"],
                            "planned": node_id in planned, "identity_migration": node.get("identity_migration"),
                            "provider_reopen": node.get("provider_reopen")}
    scopes = {**state["stages"], **{"P5:" + node: state["nodes"][node]["stage"]
                                   for node in observed if "stage" in state["nodes"][node]}}
    budget_fields = ("status", "calls", "revisions", "elapsed_seconds", "started_at", "deadline",
                     "adjudications", "signoffs")
    return {key: state.get(key) for key in ("run_id", "operator_id", "status", "stage", "role", "node",
                                          "checkpoint", "output", "published", "reason")} | {
        "format": SUMMARY_FORMAT, "input_fingerprint": state["input_fingerprint"],
        "source_identity": deepcopy(state.get("source_identity")),
        "execution": {key: state.get("config", {}).get(key) for key in ("worker_effort", "reviewer_effort", "model", "codex_binary")},
        "executor": deepcopy(state.get("executor", {"version": None, "digest": None, "rules_id": None,
                                                     "capabilities": []})),
        "state_revision": state.get("state_revision"), "generated_at": time.time(),
        "created_at": state.get("created_at"), "updated_at": state.get("updated_at"),
        "content_verified": False,
        "mode": state.get("mode", "normal"), "state_directory": str(store.run_dir(state["run_id"])),
        "approved_stages": [key for key, value in state["stages"].items() if value["status"] == "approved"],
        "approved_nodes": sorted(approved), "nodes_by_status": groups, "node_details": details,
        "active_attempts": dict(active), "unsettled_attempts": dict(state.get("unsettled_attempts", {})),
        "runner": state.get("runner"), "continuation": state.get("continuation"),
        "execution_binding": state.get("execution_binding"),
        "execution_bindings": list(state.get("execution_bindings", [])),
        "unconfirmed_attempts": dict(state.get("unconfirmed_attempts", {})),
        "control": state.get("control"), "publication": state.get("publication"),
        "exports": state.get("exports", []),
        "elapsed_seconds": sum(item.get("elapsed_seconds", 0) for item in state.get("attempt_usage", {}).values()),
        "budget": {"deadline": state.get("deadline"),
                   "limits": {key: state.get("config", {}).get(key) for key in
                              ("operator_seconds", "stage_seconds", "call_seconds", "max_calls",
                               "revision_limit", "adjudication_seconds")} | {"max_calls": call_limit(state)},
                   "extensions": [{key: item[key] for key in ("id", "run_id", "origin", "at", "seconds", "calls")}
                                  for item in state.get("budget_extensions", [])],
                   "scopes": {key: {field: value.get(field) for field in budget_fields}
                              for key, value in scopes.items()}},
        "imported_checkpoint": state.get("imported_draft", {}).get("checkpoint"),
        "legacy_usage": state.get("imported_draft", {}).get("usage"),
        "completed_node_count": len(approved), "plan_revision": state["plan_revision"],
        "p6_starts": state.get("p6_starts", 0), "planned_node_count": len(planned),
        "attempt_count": len(state.get("attempts", [])),
        "last_attempt_id": next(iter(state.get("attempts", [])[::-1]), None),
        "merged_checkpoint": state.get("merged_checkpoint")}
