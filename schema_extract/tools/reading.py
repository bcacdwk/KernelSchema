"""Bounded, lossless line ranges from a fixed tree; runnable as a standalone file."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile


def stage_check(record: dict, *, workspace: Path | None = None, rules: Path | None = None,
                baseline: Path | None = None, source: Path | None = None) -> dict:
    """Preview only: a worker's normalization never writes to the attempt workspace."""
    from schema_extract.tools.artifacts import (CandidateRejected, _early_check, artifact_manifest, check_stage_snapshot,
                                               load_root, normalize_candidate, validate_artifact_sources)
    from schema_extract.tools.common import ExtractError, manifest, verify_manifest

    if record.get("format") != "schema_extract.stage_check.v1":
        raise ExtractError("unsupported stage-check materials")
    workspace = workspace or Path(record["workspace"])
    rules = rules or Path(record["rules_root"])
    source = source or Path(record["source_root"])
    if source.name != "source":
        raise ExtractError("the frozen source directory must end in source")
    source_lock = {"snapshot": str(source.parent), "source": record["source"]}
    verify_manifest(source, record["source"]["files"])
    original = artifact_manifest(workspace, expected=record["artifact_files"],
        additions={"dispatch.md"} if (record["stage"], record["role"]) == ("P2", "worker") else ())
    result = {"kind": "read_only_stage_check", "stage": record["stage"], "read_only": True,
              "skip_hardware": True, "raw_passed": False, "corrections": [], "exit_code": 1}
    arguments = {"parser_root": rules, "stage": record["stage"],
                 "selected_designs": record["selected_designs"], "node": record["node"],
                 "checked_nodes": record.get("checked_nodes")}
    try:
        if record["role"] == "reviewer":
            check = check_stage_snapshot(workspace, **arguments)
            validate_artifact_sources(workspace, source_lock, decision="complete", parser_root=rules,
                                      nodes=check.get("checked_nodes") or [record["node"]])
            result.update(check=check, exit_code=check["exit_code"], raw_passed=check["exit_code"] == 0)
        else:
            baseline = baseline or Path(record["baseline"])
            with tempfile.TemporaryDirectory(prefix="extract-check-") as temporary:
                candidate = Path(temporary) / "preview"
                validation = normalize_candidate(baseline, workspace, candidate, rules / "schema_template/starter.yaml",
                    stage=record["stage"], role="worker", decision="continue",
                    selected_designs=record["selected_designs"], node_entry=record.get("node_entry"),
                    revision_scope=record.get("revision_scope"), first_draft=record.get("first_draft", False),
                    checked_nodes=record.get("checked_nodes"), maintenance_fields=record.get("maintenance_fields"),
                    baseline_materials=Path(record["baseline_materials"]) if record.get("baseline_materials") else None)
                result.update(corrections=validation["corrections"],
                              pending_corrections=validation["artifact_digests"] != original,
                              original_artifact_digests=original,
                              preview_artifact_digests=validation["artifact_digests"])
                if record["stage"] == "P2":
                    _early_check(load_root(candidate, parser_root=rules), parser_root=rules, stage="P2")
                    check = {"kind": "phase_fields", "stage": "P2", "exit_code": 0, "diagnostics": [],
                             "selection": "awaiting_proposal", "checked_nodes": ["root"]}
                else:
                    check = check_stage_snapshot(candidate, **arguments)
                validate_artifact_sources(candidate, source_lock, decision="complete", parser_root=rules,
                                          nodes=check.get("checked_nodes") or [record["node"]])
                result.update(check=check, preview_passed=check["exit_code"] == 0,
                              exit_code=check["exit_code"] or (1 if result["pending_corrections"] else 0),
                              raw_passed=check["exit_code"] == 0 and not result["pending_corrections"] and record["stage"] != "P2")
        result["source_validation"] = {"status": "verified", "source_id": record["source"]["id"],
                                       "repository_id": record["source"]["repository_id"], "ref": record["source"]["ref"]}
    except CandidateRejected as exc:
        result.update(diagnostics=[str(exc)], corrections=exc.validation.get("corrections", []),
                      check=exc.validation.get("check"), exit_code=(exc.validation.get("check") or {}).get("exit_code", 1))
    except (ExtractError, ValueError) as exc:
        result["diagnostics"] = [str(exc)]
    if artifact_manifest(workspace, expected=original) != original:
        raise ExtractError("artifact bytes changed during the read-only check")
    return result


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _read_file(root: Path, path: str, expected_sha256: str | None):
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
    return root, data, file_hash


def read_evidence(root: Path, path: str, start_line: int, end_line: int,
                  *, quote_offset: int = 0, expected_sha256: str | None = None) -> dict:
    """Generate a precise excerpt from the caller's range, without choosing evidence."""
    _, data, file_hash = _read_file(root, path, expected_sha256)
    lines = data.decode("utf-8").splitlines()
    if (type(start_line) is not int or type(end_line) is not int or
            not 1 <= start_line <= end_line <= len(lines) or end_line - start_line >= 12):
        raise ValueError("invalid evidence range (maximum 12 lines)")
    selected = "\n".join(lines[start_line - 1:end_line])
    if type(quote_offset) is not int or not 0 <= quote_offset < len(selected):
        raise ValueError("quote_offset must select a character within the evidence range")
    quote = selected[quote_offset:quote_offset + 320]
    if not quote.strip():
        raise ValueError("evidence excerpt is empty; select the relevant range/quote_offset")
    return {"path": path, "sha256": file_hash, "line_start": start_line,
            "line_end": end_line, "quote": quote}


def read_range(root: Path, path: str, start_line: int = 1, end_line: int | None = None,
               *, max_bytes: int = 6000, cursor: str | None = None,
               expected_sha256: str | None = None) -> dict:
    root, data, file_hash = _read_file(root, path, expected_sha256)
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
    parser.add_argument("--root", type=Path)
    parser.add_argument("--path")
    parser.add_argument("--check-materials", type=Path, help="run the registered read-only stage check")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int)
    parser.add_argument("--max-bytes", type=int, default=6000)
    parser.add_argument("--cursor")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--evidence", action="store_true", help="generate an exact short evidence anchor")
    parser.add_argument("--quote-offset", type=int, default=0,
                        help="zero-based character offset within the selected evidence lines")
    args = parser.parse_args(argv)
    try:
        if args.check_materials:
            if not args.read_only or not args.skip_hardware or args.root or args.path or args.evidence or args.cursor:
                parser.error("--check-materials requires --read-only --skip-hardware and takes no reading options")
            result = stage_check(json.loads(args.check_materials.read_text(encoding="utf-8")))
            sys.stdout.buffer.write(encoded(result))
            return result["exit_code"]
        if args.root is None or args.path is None or args.read_only or args.skip_hardware:
            parser.error("line reading requires --root and --path")
        if args.evidence:
            if args.end_line is None or args.cursor is not None:
                raise ValueError("--evidence requires --end-line and does not use --cursor")
            result = read_evidence(args.root, args.path, args.start_line, args.end_line,
                                   quote_offset=args.quote_offset, expected_sha256=args.expected_sha256)
        else:
            if args.quote_offset:
                raise ValueError("--quote-offset requires --evidence")
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
