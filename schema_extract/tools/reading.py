"""Exact evidence excerpts and read-only checks against fixed extraction inputs."""

from __future__ import annotations

import argparse
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
    from schema_extract.tools.common import ExtractError, verify_manifest

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
                    notices = _early_check(load_root(candidate, parser_root=rules), parser_root=rules, stage="P2")
                    check = {"kind": "phase_fields", "stage": "P2", "exit_code": 0, "diagnostics": [],
                             "selection": "awaiting_proposal", "checked_nodes": ["root"], "notices": notices}
                else:
                    check = check_stage_snapshot(candidate, **arguments)
                validate_artifact_sources(candidate, source_lock, decision="complete", parser_root=rules,
                                          nodes=check.get("checked_nodes") or [record["node"]])
                result.update(check=check, preview_passed=check["exit_code"] == 0,
                              exit_code=check["exit_code"] or (1 if record["stage"] == "P6" and result["pending_corrections"] else 0),
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
            not 1 <= start_line <= end_line <= len(lines)):
        raise ValueError(f"invalid evidence range (file has {len(lines)} lines)")
    selected = "\n".join(lines[start_line - 1:end_line])
    if type(quote_offset) is not int or not 0 <= quote_offset < len(selected):
        raise ValueError("quote_offset must select a character within the evidence range")
    quote = selected[quote_offset:quote_offset + 320]
    if not quote.strip():
        raise ValueError("evidence excerpt is empty; select the relevant range/quote_offset")
    return {"path": path, "sha256": file_hash, "line_start": start_line,
            "line_end": end_line, "quote": quote}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--path")
    parser.add_argument("--check-materials", type=Path, help="run the registered read-only stage check")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--evidence", action="store_true", help="generate an exact short evidence anchor")
    parser.add_argument("--quote-offset", type=int, default=0,
                        help="zero-based character offset within the selected evidence lines")
    args = parser.parse_args(argv)
    try:
        if args.check_materials:
            if not args.read_only or not args.skip_hardware or args.root or args.path or args.evidence:
                parser.error("--check-materials requires --read-only --skip-hardware and takes no reading options")
            result = stage_check(json.loads(args.check_materials.read_text(encoding="utf-8")))
            sys.stdout.buffer.write(encoded(result))
            return result["exit_code"]
        if (not args.evidence or args.root is None or args.path is None or args.end_line is None or
                args.read_only or args.skip_hardware):
            parser.error("evidence reading requires --evidence, --root, --path and --end-line")
        result = read_evidence(args.root, args.path, args.start_line, args.end_line,
                               quote_offset=args.quote_offset, expected_sha256=args.expected_sha256)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(encoded(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
