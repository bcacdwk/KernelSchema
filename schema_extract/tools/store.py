"""The authoritative v2 run record and immutable artifact checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import time
from uuid import UUID, uuid4

from .common import (RUN_WRITES, ExtractError, RunnerBusy, StaleResult, copy_artifacts, digest, json_bytes,
                     fsync_directory, manifest, read_json, seal, sync_tree, verify_manifest, write_json, write_text, rules_root)
from .artifacts import _same, load_yaml, validate_artifact_sources, write_yaml
from .stages import (call_view, effective_nodes, new_scope, next_call, node_signature,
                     scope_for, scope_key, session_group, writable_files)


STATE_FORMAT = "schema_extract.run.v2"


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def run_dir(self, run_id: str) -> Path:
        try:
            if str(UUID(run_id)) != run_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError) as exc:
            raise ExtractError("run ID must be its exact UUID") from exc
        return self.root / "runs" / run_id

    def load(self, run_id: str) -> dict:
        path = self.run_dir(run_id) / "state.json"
        guard = RUN_WRITES.get()
        value = guard.read_state() if guard and path.parent == guard.directory else read_json(path)
        if value.get("format") != STATE_FORMAT or value.get("run_id") != run_id:
            raise ExtractError("unsupported or mismatched state; v1 runs must use their frozen executor")
        if "active_attempts" not in value or "reopen_requests" not in value:
            raise ExtractError("this older serial run must use its frozen executor; import its checkpoint into a new run")
        return value

    def save(self, state: dict) -> None:
        if state.get("format") != STATE_FORMAT:
            raise ExtractError("cannot save another state format")
        write_json(self.run_dir(state["run_id"]) / "state.json", state)

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
        request = {"format": "schema_extract.control.v2", "run_id": run_id,
                   "id": str(uuid4()), "action": action, "created_at": time.time()}
        write_json(self.root / "requests" / run_id / (request["id"] + ".json"), request)
        return request

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

    def create(self, lock: dict, output: Path, *, seed: dict | None = None, queued=False) -> dict:
        run_id = str(uuid4())
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True)
        now = time.time()
        state = {"format": STATE_FORMAT, "run_id": run_id,
                 "operator_id": lock["source"]["bindings"]["id"], "output": str(output.resolve()),
                 "input_fingerprint": lock["fingerprint"], "snapshot": lock["snapshot"],
                 "status": "queued" if queued else "running", "stage": "P1", "role": "worker", "node": "root",
                 "mode": "normal", "plan": None, "plan_draft": None, "merged_checkpoint": None,
                 "revision_scope": [], "pending_rechecks": [], "recoverable_checkpoint": None,
                 "plan_revision": 1, "created_at": now,
                 "reopening": None, "reopen_requests": {}, "plan_history": [],
                 "stale_results": {}, "unconfirmed_attempts": {}, "p6_starts": 0, "final_review_started": False,
                 "config": lock["config"], "deadline": now + lock["config"]["operator_seconds"],
                 "sessions": {group: {"worker": None, "reviewer": None}
                              for group in ("P1_P2", "P3", "P4", "P6", "P6_signoff")},
                 "stages": {stage: new_scope() for stage in ("P1", "P2", "P3", "P4", "P5", "P6")},
                 "nodes": {"root": {"file": "root.yaml", "parent": "", "status": "draft", "merged": False,
                                     "draft_checkpoint": None, "approved_checkpoints": {},
                                     "approval": None, "approved_sha256": None,
                                     "generation": 0, "approved_generation": None}},
                 "checkpoint": None, "attempts": [], "active_attempts": {},
                 "accepted_receipts": {}, "reason": None, "feedback": [],
                 "attempt_usage": {}, "recovered_attempts": {}, "session_handoffs": [],
                 "control": {"action": None, "requests": {}, "resume_status": None},
                 "published": False, "publication": None, "exports": []}
        write_json(directory / "inputs.lock.json", lock)
        initial = directory / "initial"
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
        fsync_directory(directory.parent)
        fsync_directory(self.root)
        return state

    def checkpoint(self, state: dict, artifacts: Path, *, attempt_id: str | None,
                   stage: str | None = None, node: str | None = None, parent: str | None = None) -> str:
        files = manifest(artifacts)
        identity = {"format": "schema_extract.checkpoint.v2", "run_id": state["run_id"],
                    "stage": stage or state["stage"], "node": node or state["node"], "attempt_id": attempt_id,
                    "plan_revision": state["plan_revision"], "parent": parent or state["checkpoint"],
                    "input_fingerprint": state["input_fingerprint"], "files": files}
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
        if guard := RUN_WRITES.get():
            guard.tree_written(destination / "artifacts", files)
            # Bind to the exact JSON already authored, never to a refresh's
            # later read of a possibly changed checkpoint record.
            guard.written(destination / "checkpoint.json", json_bytes({**identity, "id": checkpoint_id}))
        return checkpoint_id

    def verify_checkpoint(self, state: dict, checkpoint_id: str | None = None) -> tuple[Path, dict]:
        checkpoint_id = checkpoint_id or state["checkpoint"]
        if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_id):
            raise ExtractError("invalid checkpoint ID")
        directory = self.run_dir(state["run_id"]) / "checkpoints" / checkpoint_id
        info = read_json(directory / "checkpoint.json")
        if digest({k: v for k, v in info.items() if k != "id"}) != checkpoint_id:
            raise ExtractError("checkpoint identity mismatch")
        for key in ("run_id", "input_fingerprint"):
            if info[key] != state[key]:
                raise ExtractError(f"checkpoint {key} mismatch")
        verify_manifest(directory / "artifacts", info["files"])
        return directory / "artifacts", info

    def verify_approval(self, state: dict, approval: dict | None) -> tuple[dict, dict]:
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
        if any(approval[key] != invocation[key] for key in ("stage", "node", "attempt_id", "input_fingerprint")):
            raise ExtractError("approval scope differs from its invocation")
        if approval["generation"] != invocation["generation"] or approval["plan_revision"] != invocation["plan_revision"]:
            raise ExtractError("approval invocation generation/revision identity changed")
        if digest(receipt) != approval["receipt_sha256"] or receipt["decision"] != "approve":
            raise ExtractError("approval receipt was changed or did not approve")
        if validation["receipt_sha256"] != approval["receipt_sha256"] or validation["check"]["exit_code"] != 0:
            raise ExtractError("approval has no matching successful Runner gate")
        _, checkpoint = self.verify_checkpoint(state, approval["checkpoint"])
        if checkpoint["files"] != approval["artifact_digests"] or checkpoint["files"] != validation["artifact_digests"]:
            raise ExtractError("approval content differs from its checked immutable checkpoint")
        return receipt, checkpoint

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
        parent = node["parent"]
        while parent:
            ancestor = state["nodes"][parent]
            if parent not in effective_nodes(state) or approval_checkpoint["files"].get(ancestor["file"]) != merged_info["files"].get(ancestor["file"]):
                raise ExtractError("approved node ancestor dependency changed: " + node_id)
            parent = ancestor["parent"]
        return approval_checkpoint

    def validate_attempt(self, state: dict, attempt: dict) -> None:
        directory = self.run_dir(state["run_id"]) / "attempts" / attempt["attempt_id"]
        if read_json(directory / "attempt.json") != attempt:
            raise ExtractError("agent modified invocation identity")
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
            if attempt["base_checkpoint"] != node["draft_checkpoint"]:
                raise StaleResult("this node's draft checkpoint changed")
            if (attempt["role"], attempt["mode"]) != (node["role"], node["mode"]):
                raise StaleResult("this node's role or review mode changed")
            _, merged = self.verify_checkpoint(state, state["merged_checkpoint"])
            parent = node["parent"]
            while parent:
                ancestor = state["nodes"][parent]
                self.verify_node(state, parent, merged_info=merged)
                if attempt["dependency_digests"].get(ancestor["file"]) != merged["files"].get(ancestor["file"]):
                    raise StaleResult("approved parent or Contract changed while the invocation ran")
                parent = ancestor["parent"]
        else:
            if any(attempt[key] != state[key] for key in ("stage", "node", "role", "mode", "plan_revision")):
                raise StaleResult("upstream stage invocation no longer matches its current scope")
            if attempt["generation"] != scope_for(state, attempt)["generation"] or attempt["base_checkpoint"] != state["checkpoint"]:
                raise StaleResult("stage generation or checkpoint changed")

    def prepare_attempt(self, state: dict, session_id: str | None, *, call: dict | None = None) -> dict:
        call = call or next_call(state)
        if call is None:
            raise ExtractError("no dependency-ready call can be prepared")
        key = scope_key(call["stage"], call["node"])
        if key in state["active_attempts"]:
            raise ExtractError("cannot prepare a second attempt before confirmed termination")
        from .stages import next_calls
        if not any(all(call[field] == ready[field] for field in ("stage", "node", "role", "mode")) for ready in next_calls(state)):
            raise ExtractError("cannot prepare a call outside the current dependency-ready set")
        if call["stage"] == "P5" and state["nodes"][call["node"]]["draft_checkpoint"] is None:
            self.start_node(state, call["node"])
        view = call_view(state, call)
        scope = scope_for(state, call)
        if scope["started_at"] is None:
            scope["started_at"] = time.time()
            scope["deadline"] = scope["started_at"] + state["config"]["stage_seconds"]
            scope["baseline_checkpoint"] = view["checkpoint"]
            scope["candidates"].append({"checkpoint": view["checkpoint"], "plan": state.get("plan_draft")})
            scope["adjudication_reserved_seconds"] = state["config"]["adjudication_seconds"]
            scope["signoff_reserved_seconds"] = state["config"]["adjudication_seconds"] if call["stage"] == "P6" else 0
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
                           "dependency_digests": checkpoint["files"], "session_id": session_id,
                           "session_group": group, "mode": mode, "writable_files": writable_files(state, call),
                           "generation": state["nodes"][call["node"]]["generation"] if call["stage"] == "P5" else scope["generation"],
                           "node_signature": node_signature(entry) if entry and call["stage"] == "P5" else "",
                           "revision_scope": state["revision_scope"],
                           "workspace": str(directory / "workspace"), "directory": str(directory),
                           "created_at": time.time()})
        handoff = next((item for item in reversed(state.get("session_handoffs", []))
                        if item["session_group"] == group and item["role"] == call["role"] and
                        not item.get("next_attempt_id")), None)
        if handoff:
            handoff["next_attempt_id"] = attempt_id
            invocation["recovery_handoff"] = deepcopy(handoff)
        write_json(directory / "attempt.json", invocation)
        state["attempts"].append(attempt_id)
        state["active_attempts"][key] = attempt_id
        if call["stage"] == "P6" and not state["final_review_started"]:
            state["final_review_started"] = True
            state["p6_starts"] += 1
        self.save(state)
        return invocation

    def start_node(self, state: dict, node_id: str) -> None:
        """Copy only the approved root/ancestors and this node's direct parent."""
        node = state["nodes"][node_id]
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
            derived = load_yaml(merged, parent_file, parser_root=rules_root(state))
            derived.pop("contract", None)
            version = 1
            previous = node.get("last_draft_checkpoint")
            if previous:
                old, old_info = self.verify_checkpoint(state, previous)
                if node["file"] in old_info["files"]:
                    version = load_yaml(old, node["file"], parser_root=rules_root(state))["meta_data"]["version"]
            derived["meta_data"]["version"] = version
            derived["meta_data"]["dispatch"] = {"id": node_id, "parent": node["parent"], "when": ""}
            write_yaml(temporary / node["file"], derived, source=merged / parent_file,
                       parser_root=rules_root(state))
            node["draft_checkpoint"] = self.checkpoint(state, temporary, attempt_id=None, stage="P5", node=node_id,
                                                       parent=state["merged_checkpoint"])
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
        body = (f"Checkpoint: {state['checkpoint']}\nSource: {lock['source']['id']}\n"
                f"Rules: {lock['rules']['id']}\n\n{text.rstrip()}\n")
        write_text(directory / "handoff.md", body)
        sync_tree(directory)
        fsync_directory(directory.parent)
