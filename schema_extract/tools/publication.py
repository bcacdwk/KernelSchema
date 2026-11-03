"""Atomic, non-replacing delivery of one independently reviewed checkpoint."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from uuid import uuid4

from .common import (ExtractError, digest, fsync_directory, json_bytes, manifest, read_json, sha256,
                     sync_tree, verify_manifest, write_json)
from .snapshots import verify_run_inputs


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


def _same_inode(path: Path, device: int, inode: int, *, directory=False) -> bool:
    try:
        value = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return (value.st_dev, value.st_ino) == (device, inode) and (
        not directory or stat.S_ISDIR(value.st_mode))


def _empty_marker(path: Path) -> bool:
    try:
        value = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISREG(value.st_mode) and value.st_size == 0


def _same_scaffold(path: Path, scaffold: dict) -> bool:
    """Require the durable post-move state, not a possibly reused inode."""
    try:
        value = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISDIR(value.st_mode) and (
        value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns) == (
            scaffold["device"], scaffold["inode"],
            scaffold.get("directory_mtime_ns"), scaffold.get("directory_ctime_ns"))


def _restore_marker(output: Path, backup: Path, scaffold: dict) -> None:
    """Put back only the saved marker, without replacing concurrent contents."""
    if not backup.exists() and not backup.is_symlink():
        return
    try:
        if not output.exists() and not output.is_symlink():
            output.mkdir()
        elif not _same_scaffold(output, scaffold):
            return
        rename_new(backup, output / ".gitkeep")
        fsync_directory(output)
        fsync_directory(output.parent)
    except OSError:
        # A competing writer may have replaced the directory or marker. Keep
        # the saved file at its intent-derived sibling path for recovery.
        pass


def _prepare_scaffold(output: Path, backup: Path, intent: dict, intent_path: Path) -> None:
    """Consume only a marker scaffold, including its interrupted empty state."""
    try:
        value = output.lstat()
        if not stat.S_ISDIR(value.st_mode):
            return
        entries = sorted(path.name for path in output.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    marker = output / ".gitkeep"
    if entries == [".gitkeep"] and _empty_marker(marker):
        if backup.exists() or backup.is_symlink():
            return
        # Keep the original inode alive while observing its post-move times;
        # deleting and recreating the path cannot supply a replacement inode.
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(directory_fd)
            if (opened.st_dev, opened.st_ino, opened.st_mtime_ns, opened.st_ctime_ns) != (
                    value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns):
                return
            marker_stat = marker.lstat()
            intent["scaffold"] = {"device": value.st_dev, "inode": value.st_ino,
                                  "marker_device": marker_stat.st_dev, "marker_inode": marker_stat.st_ino}
            write_json(intent_path, intent)
            try:
                rename_new(marker, backup)
            except (FileNotFoundError, FileExistsError):
                return
            os.fsync(directory_fd)
            fsync_directory(output.parent)
            moved = os.fstat(directory_fd)
            intent["scaffold"].update(directory_mtime_ns=moved.st_mtime_ns,
                                      directory_ctime_ns=moved.st_ctime_ns)
            # A crash before this write leaves an uncertain empty directory.
            # Preserve its marker backup and report conflict on replay.
            write_json(intent_path, intent)
            if not _empty_marker(backup) or not _same_inode(
                    backup, marker_stat.st_dev, marker_stat.st_ino):
                _restore_marker(output, backup, intent["scaffold"])
                return
        finally:
            os.close(directory_fd)
    scaffold = intent.get("scaffold")
    if not scaffold or not _empty_marker(backup) or not _same_inode(
            backup, scaffold["marker_device"], scaffold["marker_inode"]):
        return
    if not _same_scaffold(output, scaffold):
        return
    try:
        # rmdir is deliberately non-recursive: a concurrent new entry wins.
        output.rmdir()
        fsync_directory(output.parent)
    except OSError as exc:
        if exc.errno not in {errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR}:
            raise


def publish(store, state: dict, output: Path, *, export=False) -> dict:
    if state["status"] != "ready" or state["active_attempts"]:
        raise ExtractError("only the final independently reviewed ready checkpoint can be published")
    lock = read_json(store.run_dir(state["run_id"]) / "inputs.lock.json")
    verify_run_inputs(state, lock)
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
    intent_path = directory / (publication_id + ".intent.json")
    if intent_path.exists():
        previous = read_json(intent_path)
        if {key: value for key, value in previous.items() if key != "scaffold"} == intent:
            if "scaffold" in previous:
                intent["scaffold"] = previous["scaffold"]
    write_json(intent_path, intent)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".schema-extract-" + uuid4().hex)
    backup = output.with_name("." + output.name + ".schema-extract-scaffold-" + publication_id)
    published = False
    try:
        shutil.copytree(artifacts, temporary)
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        temporary.chmod(0o755)
        write_json(temporary / ".extraction/receipt.json", proof)
        expected = {**checkpoint["files"], ".extraction/receipt.json": sha256(json_bytes(proof))}
        verify_manifest(temporary, expected)
        sync_tree(temporary)
        if store.load(state["run_id"]) != state:
            raise ExtractError("Runner state changed before publication")
        with store.supervisor() as ledger:
            denied = ledger["active_hold"]
            if state["run_id"] in ledger["continuations"]:
                raise ExtractError("a superseded run cannot publish")
            if denied:
                store.hold_member(ledger, state)
                status = "held"
            else:
                _prepare_scaffold(output, backup, intent, intent_path)
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
        if status == "held":
            from .recovery import consume_controls
            consume_controls(store, state)
            if not export and state["control"]["action"]:
                state["control"].update(resume_status="ready", stopped_at=time.time())
                state["status"] = "cancelled" if state["control"]["action"] == "cancel" else "paused"
        published = status == "published"
        record = {**intent, "status": status, "artifacts": str(artifacts),
                  "hold_id": denied,
                  "reason": None if status == "published" else "全局 hold 阻止发布。" if denied else "目标已有不同内容；目标与已审checkpoint均已保留。"}
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
        if scaffold := intent.get("scaffold"):
            if published:
                if _empty_marker(backup) and _same_inode(
                        backup, scaffold["marker_device"], scaffold["marker_inode"]):
                    backup.unlink()
                    fsync_directory(output.parent)
            elif _same_inode(backup, scaffold["marker_device"], scaffold["marker_inode"]):
                _restore_marker(output, backup, scaffold)
        if temporary.exists():
            shutil.rmtree(temporary)
