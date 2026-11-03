"""Shared filesystem roots, immutable manifests, and identity primitives."""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from uuid import uuid4


# Only Store's current Runner thread may advance its trusted state bytes.
# Agent transports and concurrent control submitters do not own this guard.
STATE_GUARD = ContextVar("schema_extract_state_guard", default=None)


class ExtractError(ValueError):
    """An actionable input, state, or proposal error."""


class StaleResult(ExtractError):
    """A genuine invocation whose dependencies were superseded while it ran."""


class RunnerBusy(ExtractError):
    """The run's existing Runner will consume any newly queued control request."""


def rules_root(record: dict | None = None) -> Path:
    """The active tool tree, or one run's fixed schema rules and executor."""
    return Path(record["snapshot"]) / "rules" if record is not None else Path(__file__).resolve().parents[2]


def source_root(record: dict) -> Path:
    return Path(record["snapshot"]) / "source"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExtractError(f"cannot read JSON {path}: {exc}") from exc


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    body = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def copy_artifacts(source: Path, destination: Path) -> None:
    """Copy a Runner-created working file or directory."""
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(directory: Path) -> None:
    """Flush payload bytes and directory entries before publishing a pointer."""
    directories = [directory]
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ExtractError(f"cannot persist a symlink in an immutable tree: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        else:
            raise ExtractError(f"cannot persist a special file: {path}")
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fsync_directory(path)


def relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ExtractError(f"invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ExtractError(f"invalid relative path: {value!r}")
    return value


def regular_file(root: Path, name: str) -> Path:
    relative_name(name)
    path = root / name
    for part in [path, *path.parents]:
        if part == root.parent:
            break
        if part.is_symlink():
            raise ExtractError(f"symlink is not an allowed input/artifact: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ExtractError(f"missing regular file: {name}")
    return path


def manifest(directory: Path) -> dict[str, str]:
    """Reject links and special files, including links to allowed files."""
    files = {}
    if not directory.is_dir() or directory.is_symlink():
        raise ExtractError(f"invalid directory: {directory}")
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ExtractError(f"unsupported file type: {name}")
        if stat.S_ISREG(mode):
            files[name] = sha256(path.read_bytes())
    return files


def seal(directory: Path) -> None:
    for path in directory.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    directory.chmod(0o555)


def verify_manifest(directory: Path, expected: dict[str, str]) -> None:
    actual = manifest(directory)
    if actual != expected:
        changed = sorted(name for name in actual.keys() | expected.keys()
                         if actual.get(name) != expected.get(name))
        raise ExtractError(f"fixed contents changed: {', '.join(changed[:12])}")
