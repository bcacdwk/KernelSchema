"""Lossless, bounded pages of fixed local UTF-8 sources, using only stdlib.

Run this file directly with ``python3 -B`` so it also works from an empty,
read-only artifact workspace. Repeat the same --requests JSON with --cursor
to continue. A page's max_bytes covers its compact JSON, trailing newline,
and one additional JSON-string wrapping (as used by command tools).

In v1 output, a fragment with only ``line`` and ``text`` is a complete line
starting at column 1. Omitted ``column`` defaults to 1; omitted ``ends_line``
defaults to true. Partial lines carry ``ends_line: false``, and continuations
carry their original character column. ``remaining.request_index`` points to
the unchanged input request; its omitted column also defaults to 1. A source's
resolved absolute path and SHA appear once per file group, not again in
``remaining``. Text, including line endings, is never shortened or normalized.

Add ``"find": ["literal", ...]`` to a request to locate case-sensitive
substrings within its selected lines. Its file group reports ``matching_lines``
as 1-based line numbers and ``total_matches`` as the full count of matching
lines in the range. Each matching line is reported once, without its body.
Read a returned line or its context with a normal start_line/end_line request.

Cursor codec v2 checks the version, batch digest and exact position together
for accidental damage. A valid cursor whose inputs have changed is stale.
To start a new batch after an old v1 cursor, omit --cursor.

This bounds one reader invocation, not an arbitrary outer batch of tools.
Use one invocation per tool result with enough output allowance for the page
and the tool's own metadata. No read state or temporary files are written.
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys


DEFAULT_MAX_BYTES = MAX_PAGE_BYTES = 6000
MIN_PAGE_BYTES = 1024
FORMAT = "schema_extract.read_page.v1"
# Version, raw SHA-256 batch digest, request index, line index, character offset.
_CURSOR_VERSION = 2
_CURSOR_BODY = struct.Struct(">B32sQQQ")
_CURSOR_CHECK_BYTES = 12


class ReadingError(ValueError):
    """A failed read produces no partial page or silently advanced cursor."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def serialize_page(page: dict) -> str:
    return _json(page) + "\n"


def output_size(page: dict) -> int:
    """Bytes after one extra JSON-string wrapper, including the CLI newline."""
    return len(_json(serialize_page(page)).encode("utf-8"))


def _label(value) -> str:
    text = str(value)
    return text if len(text) <= 240 else text[:237] + "..."


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReadingError("invalid_request", f"{name} must be an integer >= {minimum}")
    return value


def _stat_key(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_file(path: Path) -> bytes:
    """Detect replacement or writes during the read, including same-size writes."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReadingError("invalid_path", f"not a regular file: {_label(path)}")
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise ReadingError("read_failed", f"cannot read {_label(path)}: {_label(exc.strerror)}") from exc
    if _stat_key(before) != _stat_key(after) or _stat_key(after) != _stat_key(current):
        raise ReadingError("file_changed", f"file changed during read: {_label(path)}; start a new read")
    return data


@dataclass
class _Source:
    path: str
    sha256: str
    lines: list[str]
    start: int
    end: int
    matches: list[int] | None

    @property
    def first_line_index(self) -> int:
        return (self.matches[0] if self.matches else self.start) - 1


def _sources(roots, requests) -> tuple[list[_Source], bytes]:
    if not roots:
        raise ReadingError("invalid_request", "at least one allowed root is required")
    try:
        if any(not Path(root).is_absolute() for root in roots):
            raise ReadingError("invalid_path", "allowed roots must be absolute directories")
        allowed = [Path(root).resolve(strict=True) for root in roots]
        if any(not root.is_dir() for root in allowed):
            raise ReadingError("invalid_path", "allowed roots must be existing directories")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, ReadingError):
            raise
        raise ReadingError("invalid_path", "allowed roots must be existing directories") from exc
    if not isinstance(requests, list) or not requests:
        raise ReadingError("invalid_request", "requests must be a nonempty list of file ranges")
    result, binding, cache = [], [], {}
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or set(request) - {"path", "start_line", "end_line", "find"}:
            raise ReadingError("invalid_request", f"request {index} must contain only path/start_line/end_line/find")
        terms = request.get("find")
        if "find" in request and (not isinstance(terms, list) or not terms or any(
                not isinstance(term, str) or term.splitlines() != [term] for term in terms)):
            raise ReadingError("invalid_request", f"request {index} find must be a nonempty list of nonempty single-line literals")
        name = request.get("path")
        if not isinstance(name, str) or not name or not Path(name).is_absolute():
            raise ReadingError("invalid_path", f"request {index} path must be absolute")
        try:
            path = Path(name).resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ReadingError("invalid_path", f"cannot resolve {_label(name)}") from exc
        if not any(path.is_relative_to(root) for root in allowed):
            raise ReadingError("invalid_path", f"path is outside allowed roots: {_label(name)}")
        start = _integer(request.get("start_line", 1), "start_line", 1)
        end = request.get("end_line")
        if end is not None:
            _integer(end, "end_line", start)
        if path not in cache:
            data = _read_file(path)
            try:
                # keepends preserves CRLF, lone CR and Unicode line separators.
                # The line numbering matches str.splitlines used by receipts.
                lines = data.decode("utf-8").splitlines(keepends=True)
            except UnicodeDecodeError as exc:
                raise ReadingError("invalid_encoding", f"source is not UTF-8: {_label(path)}") from exc
            cache[path] = (hashlib.sha256(data).hexdigest(), lines)
        sha256, lines = cache[path]
        if start > len(lines) + 1:
            raise ReadingError("invalid_range", f"request {index} starts after EOF (file has {len(lines)} lines)")
        actual_end = min(end, len(lines)) if end is not None else len(lines)
        matches = None
        descriptor = [str(path), start, end, sha256]
        if terms is not None:
            matches = [number for number in range(start, actual_end + 1)
                       if any(term in lines[number - 1] for term in terms)]
            descriptor.append(terms)
        result.append(_Source(str(path), sha256, lines, start, actual_end, matches))
        binding.append(descriptor)
    try:
        digest = hashlib.sha256(_json(binding).encode("utf-8")).digest()
    except UnicodeEncodeError as exc:
        raise ReadingError("invalid_request", "requests must contain valid UTF-8 text") from exc
    return result, digest


def _cursor(binding: bytes, position: tuple[int, int, int]) -> str:
    body = _CURSOR_BODY.pack(_CURSOR_VERSION, binding, *position)
    check = hashlib.sha256(body).digest()[:_CURSOR_CHECK_BYTES]
    return base64.urlsafe_b64encode(body + check).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[bytes, int, int, int]:
    """Validate the public damage check before comparing with current inputs."""
    try:
        if not isinstance(cursor, str) or len(cursor) > 256:
            raise ValueError("bad cursor")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != cursor:
            raise ValueError("noncanonical cursor")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ReadingError("invalid_cursor", "invalid read cursor; use the unchanged cursor from a previous page") from exc
    if len(raw) != _CURSOR_BODY.size + _CURSOR_CHECK_BYTES:
        if raw.startswith(b'[1,"'):
            raise ReadingError("unsupported_cursor_version", "cursor version 1 has no integrity check; start a new batch without --cursor")
        raise ReadingError("invalid_cursor", "invalid read cursor length; copy the unchanged cursor from a previous page")
    body, check = raw[:_CURSOR_BODY.size], raw[_CURSOR_BODY.size:]
    if hashlib.sha256(body).digest()[:_CURSOR_CHECK_BYTES] != check:
        raise ReadingError("invalid_cursor", "read cursor integrity check failed; copy the unchanged cursor from a previous page")
    version, binding, index, line, column = _CURSOR_BODY.unpack(body)
    if version != _CURSOR_VERSION:
        raise ReadingError("unsupported_cursor_version", f"unsupported cursor version {version}; start a new batch without --cursor")
    return binding, index, line, column


def _position(saved: tuple[bytes, int, int, int] | None, binding: bytes,
              sources: list[_Source]) -> tuple[int, int, int]:
    if saved is None:
        return (0, sources[0].first_line_index, 0)
    previous, index, line, column = saved
    if previous != binding:
        raise ReadingError("stale_cursor", "requests or file contents changed; start a new batch without --cursor")
    if index < len(sources):
        source = sources[index]
        if source.matches:
            valid = column == 0 and line + 1 in source.matches
        elif source.matches == [] or source.start > source.end:
            valid = line == source.start - 1 and column == 0
        else:
            valid = source.start - 1 <= line < source.end and column < len(source.lines[line])
        if valid:
            return index, line, column
    raise ReadingError("invalid_cursor", "cursor position is outside the selected material; start a new batch without --cursor")


def read_page(requests: list[dict], *, roots, cursor: str | None = None,
              max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    """Return one exact page; repeat requests and next_cursor until complete.

    ``files[].fragments`` carry 1-based original line numbers and Unicode
    character columns (omitted ``column`` means 1). Their text preserves all
    original characters, including line endings. Omitted ``ends_line`` means
    true: the complete physical line has now been delivered; a final line does
    not need a newline. Only incomplete fragments carry ``ends_line: false``.
    ``remaining`` identifies the next unread character by ``request_index``,
    line and column (also defaulting to 1), and counts later input requests.
    A request with ``find`` locates any listed case-sensitive literal in the
    selected range. It reports only ``matching_lines`` for this page and the
    range's ``total_matches``. Its continuation points to the next matching
    line, or the requested start when an empty group is still to be delivered.

    Every selected file is hashed on each call. A checksum covers the cursor
    version, batch digest and coordinates. A damaged cursor fails separately
    from a valid cursor whose files changed, even with unchanged size and mtime.
    Files are read into memory; no persistent index or background process is
    needed. The ceiling limits a page only, never the total material read.
    """
    _integer(max_bytes, "max_bytes", MIN_PAGE_BYTES)
    if max_bytes > MAX_PAGE_BYTES:
        raise ReadingError("invalid_request", f"max_bytes must be <= {MAX_PAGE_BYTES}; continue with next_cursor")
    saved = _decode_cursor(cursor) if cursor is not None else None
    sources, binding = _sources(roots, requests)
    position = _position(saved, binding, sources)
    replay_cursor = _cursor(binding, position)
    files = []
    progressed = False

    def page(at):
        index, line, column = at
        complete = index == len(sources)
        remaining = None if complete else {
            "request_index": index, "line": line + 1, "through_line": sources[index].end,
            "following_requests": len(sources) - index - 1,
        }
        if remaining is not None and column:
            remaining["column"] = column + 1
        return {"format": FORMAT, "cursor": replay_cursor, "max_bytes": max_bytes,
                "files": files, "remaining": remaining,
                "next_cursor": None if complete else _cursor(binding, at), "complete": complete}

    def next_file(index):
        index += 1
        return (index, sources[index].first_line_index if index < len(sources) else 0, 0)

    def fitting_count(available, candidate):
        # Each character or line number costs at least one byte. Completing a
        # file can remove metadata, so check the endpoint before binary search.
        low, high = 0, min(available, max_bytes)
        if output_size(page(candidate(high))) <= max_bytes:
            return high
        high -= 1
        while low < high:
            middle = (low + high + 1) // 2
            if output_size(page(candidate(middle))) <= max_bytes:
                low = middle
            else:
                high = middle - 1
        return low

    while position[0] < len(sources):
        index, line, column = position
        source = sources[index]
        new_group = not files or files[-1]["request_index"] != index
        if new_group:
            group = {"request_index": index, "path": source.path, "sha256": source.sha256,
                     "start_line": source.start, "end_line": source.end, "total_lines": len(source.lines)}
            if source.matches is None:
                group["fragments"] = []
            else:
                group.update(matching_lines=[], total_matches=len(source.matches))
            files.append(group)
        if source.matches == [] or source.start > source.end:
            after = next_file(index)
            if output_size(page(after)) <= max_bytes:
                position, progressed = after, True
                continue
            files.pop()
            break
        if source.matches is not None:
            offset = bisect_left(source.matches, line + 1)

            def match_candidate(count):
                end = offset + count
                files[-1]["matching_lines"] = source.matches[offset:end]
                if end == len(source.matches):
                    return next_file(index)
                return (index, source.matches[end] - 1, 0)

            count = fitting_count(len(source.matches) - offset, match_candidate)
            if not count:
                files.pop()
                break
            position, progressed = match_candidate(count), True
            if position[0] == index:
                break
            continue
        fragment = {"line": line + 1, "text": ""}
        if column:
            fragment["column"] = column + 1
        files[-1]["fragments"].append(fragment)
        text = source.lines[line]

        def candidate(length):
            end = column + length
            fragment["text"] = text[column:end]
            if end < len(text):
                fragment["ends_line"] = False
                return (index, line, end)
            fragment.pop("ends_line", None)
            return (index, line + 1, 0) if line + 1 < source.end else next_file(index)

        count = fitting_count(len(text) - column, candidate)
        if not count:
            files[-1]["fragments"].pop()
            if new_group:
                files.pop()
            break
        position, progressed = candidate(count), True
        if column + count < len(text):
            break
    if not progressed:
        raise ReadingError("page_too_small", "source metadata does not fit this page; increase max_bytes or use shorter source paths")
    result = page(position)
    assert output_size(result) <= max_bytes
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ReadingError("invalid_request", _label(message))


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="allowed absolute root; repeat for sources and artifacts")
    parser.add_argument("--requests", required=True, help='JSON ranges with path, optional start_line/end_line and optional find:["literal", ...]')
    parser.add_argument("--cursor", help="next_cursor from the preceding page; repeat unchanged --requests")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help=f"JSON-wrapped bytes per page ({MIN_PAGE_BYTES}..{MAX_PAGE_BYTES})")
    try:
        args = parser.parse_args(argv)
        try:
            requests = json.loads(args.requests)
        except (ValueError, TypeError) as exc:
            raise ReadingError("invalid_request", "requests must be valid JSON") from exc
        result = read_page(requests, roots=args.root, cursor=args.cursor, max_bytes=args.max_bytes)
    except ReadingError as exc:
        result = {"error": exc.code, "message": _label(exc)}
        sys.stdout.buffer.write(serialize_page(result).encode("utf-8"))
        return 2
    sys.stdout.buffer.write(serialize_page(result).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
