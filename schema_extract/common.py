"""Small filesystem and identity primitives shared by the four boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from uuid import uuid4


class ExtractError(ValueError):
    """An actionable input, state, or proposal error."""


class StaleResult(ExtractError):
    """A genuine invocation whose dependencies were superseded while it ran."""


class RunnerBusy(ExtractError):
    """The run's existing Runner will consume any newly queued control request."""


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


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
