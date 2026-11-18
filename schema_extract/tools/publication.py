"""Atomic, non-replacing delivery of one independently reviewed checkpoint."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from .common import (ExtractError, digest, fsync_directory, json_bytes, manifest, read_json, sha256,
                     sync_tree, verify_manifest, write_json, rules_root)
from .artifacts import check_complete, validate_artifact_sources
from .runtime import execution_boundary, verify_execution


def rename_new(source: Path, target: Path) -> None:
    """Atomic no-replace includes an already existing empty directory or symlink."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        code = function(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        code = function(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise ExtractError("this platform has no supported atomic no-replace directory rename")
    if code:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), str(target))
        raise OSError(error, os.strerror(error), str(target))


@execution_boundary
def publish(store, state: dict, output: Path, *, export=False) -> dict:
    from .maintenance import publication_barrier
    from .runner import _final_inputs
    if (state["status"] != "ready" or state["stages"]["P6"]["status"] != "approved" or state["control"]["action"] or
            state.get("pending_rechecks") or state.get("reopening") or
            any(state.get(name) for name in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts"))):
        raise ExtractError("only the settled, independently reviewed final checkpoint can be published")
    if publication_barrier(store, state):
        raise ExtractError("a control request, drain or maintenance prevents publication and export")
    lock = verify_execution(store, state, executor=True)
    artifacts, checkpoint = store.verify_checkpoint(state)
    receipt, reviewed = store.verify_approval(state, state["stages"]["P6"]["approval"], directory=artifacts)
    if checkpoint["id"] != reviewed["id"] or checkpoint["id"] != state["merged_checkpoint"]:
        raise ExtractError("final checkpoint differs from its current independent P6 approval")
    approval_attempt = read_json(store.run_dir(state["run_id"]) / "attempts" /
                                 receipt["identity"]["attempt_id"] / "attempt.json")
    verify_execution(store, state, attempt=approval_attempt)
    if approval_attempt["stage"] != "P6" or approval_attempt["role"] != "reviewer":
        raise ExtractError("delivery requires the current full P6 reviewer receipt")
    _final_inputs(store, state, artifacts)
    checked = check_complete(artifacts, rules_root(lock), selected_designs=state["selected_designs"])
    if checked["exit_code"] != 0 or checked["excluded_nodes"]:
        raise ExtractError("final directory did not pass complete publication checks: " +
                           "; ".join(checked["diagnostics"]))
    validate_artifact_sources(artifacts, lock, decision="approve", parser_root=rules_root(lock))
    output = output.expanduser().absolute()
    if (output.is_symlink() or output.resolve().is_relative_to(store.root) or
            store.root.is_relative_to(output.resolve()) or output.resolve() == Path(lock["source"]["origin_repository"])):
        raise ExtractError("publication target must be separate from runtime and source repository roots")
    output = output.resolve()
    proof = {"format": "schema_extract.delivery.v2", "run_id": state["run_id"],
             "checkpoint": checkpoint["id"], "input_fingerprint": state["input_fingerprint"],
             "source": {key: lock["source"][key] for key in ("id", "ref", "ref_kind", "repository_id", "bindings")},
             "rules_id": lock["rules"]["id"], "executor": state["executor"],
             "files": checkpoint["files"], "selected_designs": state["selected_designs"],
             "selection_signature": state["selection_signature"], "writing_signature": state["writing_signature"],
             "design_sources": checked["design_resolution"]["nodes"],
             "approval": {"attempt_id": receipt["identity"]["attempt_id"], "receipt_sha256": digest(receipt),
                          "attempt_sha256": digest(approval_attempt), "backend": approval_attempt["backend"],
                          "model": approval_attempt["model"], "effort": approval_attempt["effort"],
                          "session_scope": approval_attempt["session_scope"]},
             "check_policy": "skip-hardware", "checked_nodes": checked["checked_nodes"]}
    publication_id = digest({"proof": proof, "target": str(output)})
    directory = store.run_dir(state["run_id"]) / "publications"
    directory.mkdir(exist_ok=True)
    intent = {"id": publication_id, "target": str(output), "checkpoint": checkpoint["id"],
              "input_fingerprint": state["input_fingerprint"], "proof": proof, "export": export}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".schema-extract-" + uuid4().hex)
    try:
        shutil.copytree(artifacts, temporary)
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        temporary.chmod(0o755)
        write_json(temporary / ".extraction/receipt.json", proof)
        expected = {**checkpoint["files"], ".extraction/receipt.json": sha256(json_bytes(proof))}
        verify_manifest(temporary, expected)
        sync_tree(temporary)
        verify_manifest(artifacts, checkpoint["files"])
        if store.load(state["run_id"]) != state:
            raise ExtractError("Runner state changed before publication")
        with store.control_transaction() as ledger:
            if store.load(state["run_id"]) != state:
                raise ExtractError("Runner context changed before the publication rename")
            if publication_barrier(store, state, ledger):
                raise ExtractError("control request, drain or maintenance fenced the publication rename")
            # Full P6/effective-approval checks ran under the run writer above.
            # This short transaction rechecks their exact state/context and
            # checkpoint identities, then linearizes rename against requests.
            denied = ledger["active_hold"]
            pending_control = state["control"]["action"] or next((request["action"] for request in
                store.control_requests(state["run_id"]) if request["id"] not in state["control"]["requests"]), None)
            if denied:
                store.hold_member(ledger, state)
                status = "held"
            elif pending_control:
                status = "controlled"
            else:
                try:
                    rename_new(temporary, output)
                    fsync_directory(output.parent)
                    status = "published"
                except FileExistsError:
                    status = "existing"
                if status == "published":
                    ledger.setdefault("commits", {})[state["run_id"]] = {
                        "publication_id": publication_id, "target": str(output), "checkpoint": checkpoint["id"],
                        "at": time.time(), "control_revision": ledger["revision"] + 1}
        if status == "existing":
            try:
                same = manifest(output) == expected
            except ExtractError:
                same = False
            status = "published" if same else "conflict"
        if status in {"held", "controlled"}:
            from .recovery import consume_controls
            consume_controls(store, state)
            pending_control = state["control"]["action"]
            if not export and state["control"]["action"]:
                state["control"].update(resume_status="ready", stopped_at=time.time())
                state["status"] = "cancelled" if state["control"]["action"] == "cancel" else "paused"
        record = {**intent, "status": status, "artifacts": str(artifacts),
                  "hold_id": denied, "control_action": pending_control,
                  "reason": None if status == "published" else "全局 hold 阻止发布。" if denied else
                            "当前 pause/cancel 请求阻止发布。" if pending_control else "目标已有不同内容；目标与已审checkpoint均已保留。"}
        # The target may already be visible if the Runner dies here. Replay
        # checks its complete manifest and writes this same receipt once.
        write_json(directory / (publication_id + ".json"), record)
        if export:
            state["exports"] = [item for item in state["exports"] if item["id"] != publication_id] + [record]
        else:
            state["publication"] = record
            state["published"] = status == "published"
        store.save(state)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
