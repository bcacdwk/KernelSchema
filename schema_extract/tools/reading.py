"""Bounded, lossless line ranges from a fixed tree; runnable as a standalone file."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def read_range(root: Path, path: str, start_line: int = 1, end_line: int | None = None,
               *, max_bytes: int = 6000, cursor: str | None = None,
               expected_sha256: str | None = None) -> dict:
    root = root.resolve(strict=True)
    name = PurePosixPath(path)
    if not path or name.is_absolute() or ".." in name.parts or name.as_posix() != path or "\\" in path:
        raise ValueError("path must be a plain path relative to the fixed root")
    target = root / path
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("symlinks are not readable evidence")
    if not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError("file is outside the fixed tree or missing")
    before = target.stat()
    data = target.read_bytes()
    after = target.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size) != (
            after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size):
        raise ValueError("file changed during read")
    file_hash = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise ValueError("file digest does not match fixed input")
    lines = data.decode("utf-8").splitlines(keepends=True)
    end_line = min(start_line + 79, len(lines)) if end_line is None else end_line
    if not 1024 <= max_bytes <= 24000 or not 1 <= start_line <= end_line <= len(lines):
        raise ValueError("invalid line range or page budget (1024..24000 bytes)")
    binding = hashlib.sha256(encoded([str(root), path, file_hash, start_line, end_line])).hexdigest()
    line, column = start_line, 0
    if cursor:
        try:
            saved = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            if saved["binding"] != binding or set(saved) != {"binding", "line", "column"}:
                raise ValueError()
            line, column = saved["line"], saved["column"]
            if type(line) is not int or type(column) is not int or not start_line <= line <= end_line or not 0 <= column < len(lines[line - 1]):
                raise ValueError()
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("invalid or stale range cursor") from exc

    def continuation(next_line, next_column):
        if next_line > end_line:
            return None
        return base64.urlsafe_b64encode(encoded({"binding": binding, "line": next_line, "column": next_column})).decode()

    result = {"format": "schema_extract.read.v2", "root": str(root), "path": path,
              "sha256": file_hash, "start_line": start_line, "end_line": end_line,
              "fragments": [], "next_cursor": None}
    while line <= end_line:
        remaining = lines[line - 1][column:]
        low, high, best = 1, len(remaining), 0
        while low <= high:
            size = (low + high) // 2
            complete = size == len(remaining)
            fragment = {"line": line, "column": column + 1, "text": remaining[:size], "ends_line": complete}
            candidate = {**result, "fragments": result["fragments"] + [fragment],
                         "next_cursor": continuation(line + 1, 0) if complete else continuation(line, column + size)}
            if len(encoded(candidate)) <= max_bytes:
                best, low = size, size + 1
            else:
                high = size - 1
        if best == 0:
            if not result["fragments"]:
                raise ValueError("page metadata exceeds the budget")
            break
        complete = best == len(remaining)
        result["fragments"].append({"line": line, "column": column + 1, "text": remaining[:best], "ends_line": complete})
        line, column = (line + 1, 0) if complete else (line, column + best)
        result["next_cursor"] = continuation(line, column)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--path", required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int)
    parser.add_argument("--max-bytes", type=int, default=6000)
    parser.add_argument("--cursor")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        result = read_range(args.root, args.path, args.start_line, args.end_line,
                            max_bytes=args.max_bytes, cursor=args.cursor,
                            expected_sha256=args.expected_sha256)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(encoded(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
