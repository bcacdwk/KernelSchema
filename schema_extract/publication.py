"""Atomic, non-replacing delivery of one independently reviewed checkpoint."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from .common import ExtractError, digest, fsync_directory, manifest, read_json, sync_tree, write_json
from .snapshots import verify_snapshot


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


def publish(store, state: dict, output: Path, *, export=False) -> dict:
    if state["status"] != "ready" or state["active_attempts"]:
        raise ExtractError("only the final independently reviewed ready checkpoint can be published")
    lock = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
    verify_snapshot(lock)
    receipt, reviewed = store.verify_approval(state, state["stages"]["P6"]["approval"])
    artifacts, checkpoint = store.verify_checkpoint(state)
    if checkpoint["id"] != reviewed["id"] or checkpoint["id"] != state["merged_checkpoint"]:
        raise ExtractError("final checkpoint differs from its effective independent P6 approval")
    output = output.absolute()
    imported = state.get("imported_draft")
    if imported and output.resolve().is_relative_to(Path(imported["source_task"]).parent.parent):
        raise ExtractError("publication target must be outside the legacy runtime storage")
    if (output.is_symlink() or output.resolve().is_relative_to(store.root) or
            store.root.is_relative_to(output.resolve()) or output.resolve() == Path(lock["source"]["origin_repository"])):
        raise ExtractError("publication target must be separate from runtime and source repository roots")
    output = output.resolve()
    proof = {"format": "schema_extract.delivery.v2", "run_id": state["run_id"],
             "checkpoint": checkpoint["id"], "input_fingerprint": state["input_fingerprint"],
             "source": {key: lock["source"][key] for key in ("id", "ref", "repository_id", "bindings")},
             "rules_id": lock["rules"]["id"], "files": checkpoint["files"],
             "approval": {"attempt_id": receipt["identity"]["attempt_id"], "receipt_sha256": digest(receipt)},
             "check_policy": "skip-hardware"}
    if lock["source"].get("aliases"):
        proof["source"]["aliases"] = lock["source"]["aliases"]
    if state.get("imported_draft"):
        proof["imported_from"] = {key: state["imported_draft"][key]
                                  for key in ("task_key", "checkpoint", "fingerprint", "files", "usage")}
    publication_id = digest({"proof": proof, "target": str(output)})
    directory = store.run_dir(state["run_id"]) / "publications"
    directory.mkdir(exist_ok=True)
    intent = {"id": publication_id, "target": str(output), "checkpoint": checkpoint["id"],
              "input_fingerprint": state["input_fingerprint"], "proof": proof, "export": export}
    write_json(directory / (publication_id + ".intent.json"), intent)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".schema-extract-" + uuid4().hex)
    try:
        shutil.copytree(artifacts, temporary)
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        temporary.chmod(0o755)
        write_json(temporary / ".extraction/receipt.json", proof)
        expected = manifest(temporary)
        sync_tree(temporary)
        try:
            rename_new(temporary, output)
            fsync_directory(output.parent)
            status = "published"
        except FileExistsError:
            try:
                same = manifest(output) == expected
            except ExtractError:
                same = False
            status = "published" if same else "conflict"
        record = {**intent, "status": status, "artifacts": str(artifacts),
                  "reason": None if status == "published" else "目标已有不同内容；目标与已审checkpoint均已保留。"}
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
