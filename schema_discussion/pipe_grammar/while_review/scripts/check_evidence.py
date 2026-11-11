#!/usr/bin/env python3
"""Verify local evidence bytes, source locations and document examples; no GPU."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
CACHE = {}
CHECKED = set()
VERSIONS = {}
ORIGINAL = {}
COUNTS = {"source_ranges": 0, "fragment_hashes": 0, "git_blobs": 0,
          "local_links": 0, "yaml_examples": 0}


def locate(value):
    if not isinstance(value, str) or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return None
    p = Path(value)
    for candidate in ([p] if p.is_absolute() else [REPO / p, STUDY / p]):
        if candidate.is_file():
            return candidate.resolve()
    return None


def raw(path):
    if path not in CACHE:
        CACHE[path] = path.read_bytes()
    return CACHE[path]


def recorded_version(path, expected=None):
    expected = expected or ORIGINAL.get(path.resolve())
    if expected and hashlib.sha256(raw(path)).hexdigest() != expected:
        return VERSIONS.get((path.resolve(), expected), path)
    return path


def check_hash(path, expected):
    path = recorded_version(path, expected)
    assert hashlib.sha256(raw(path)).hexdigest() == expected, f"SHA256 mismatch: {path}"
    CHECKED.add(path)


def check_range(path, span, expected=None):
    assert len(span) == 2 and all(type(x) is int for x in span), (path, span)
    lines = raw(path).decode("utf-8").splitlines(keepends=True)
    start, end = span
    assert 1 <= start <= end <= len(lines), (path, span, len(lines))
    COUNTS["source_ranges"] += 1
    if expected:
        fragment = "".join(lines[start - 1:end]).encode("utf-8")
        assert hashlib.sha256(fragment).hexdigest() == expected, (path, span)
        COUNTS["fragment_hashes"] += 1


def walk(value, report=None):
    if isinstance(value, list):
        for child in value:
            walk(child, report)
        return
    if not isinstance(value, dict):
        return
    report = locate(value.get("report")) or report
    path = next((p for key in ("local_snapshot", "local_path", "path")
                 if (p := locate(value.get(key))) is not None), None)
    if path:
        path = recorded_version(path, value.get("sha256"))
        if "sha256" in value:
            check_hash(path, value["sha256"])
        if "bytes" in value:
            assert len(raw(path)) == value["bytes"], path
        if "git_blob_sha" in value:
            data = raw(path)
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            assert hashlib.sha1(blob).hexdigest() == value["git_blob_sha"], path
            COUNTS["git_blobs"] += 1
        span = None
        for first, last in (("line_start", "line_end"), ("start_line", "end_line")):
            if first in value and last in value:
                span = [value[first], value[last]]
                break
        if span is None and isinstance(value.get("lines"), list):
            if len(value["lines"]) == 2 and all(type(x) is int for x in value["lines"]):
                span = value["lines"]
        if span:
            check_range(path, span, value.get("fragment_sha256"))
        if "context_start_line" in value and "context_end_line" in value:
            check_range(path, [value["context_start_line"], value["context_end_line"]],
                        value.get("context_sha256"))
        if type(value.get("line")) is int:
            first = value["line"]
            # 05_build_data.py records an eleven-line header context.
            last = min(first + 10, len(raw(path).decode("utf-8").splitlines())) \
                if "head_fragment_sha256" in value else first
            check_range(path, [first, last], value.get("head_fragment_sha256"))
        for key in ("line_spans", "consulted_line_spans"):
            for i, item in enumerate(value.get(key, [])):
                hashes = value.get("line_spans_sha256", [])
                check_range(path, item, hashes[i] if i < len(hashes) else None)
    if report and "report_sha256" in value:
        check_hash(report, value["report_sha256"])
    for child in value.values():
        walk(child, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((STUDY / "manifest.json").read_text())
    for entry in manifest.get("document_snapshots", []):
        original = (REPO / entry["original_path"]).resolve()
        VERSIONS[original, entry["sha256"]] = (STUDY / entry["local_path"]).resolve()
    ORIGINAL.update({(REPO / path).resolve(): expected
                     for path, expected in manifest["protected_documents"].items()})
    for path, expected in manifest["protected_documents"].items():
        check_hash(REPO / path, expected)
    for path, expected in manifest.get("artifacts", {}).items():
        check_hash(STUDY / path, expected)
    data_files = sorted((STUDY / "data").glob("*.json"))
    data_files += sorted((STUDY / "sources").glob("*.json"))
    for path in data_files:
        if path.name == "validation.json":
            continue
        walk(json.loads(path.read_text()))
    documents = [STUDY / "METHOD.md", *sorted((STUDY / "reports").glob("*.md"))]
    if (STUDY / "README.md").exists():
        documents.append(STUDY / "README.md")
    for path in documents:
        text = path.read_text()
        assert text.count("```") % 2 == 0, f"Unbalanced fences: {path}"
        for block in re.findall(r"^```yaml\n(.*?)^```", text, re.M | re.S):
            yaml.safe_load(block)
            COUNTS["yaml_examples"] += 1
        for link in re.findall(r"\]\(([^)]+)\)", text):
            target = link.split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            assert (path.parent / target).exists(), (path, target)
            COUNTS["local_links"] += 1
    drift = [path for path, expected in manifest["protected_documents"].items()
             if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != expected]
    result = {"passed": True, "unique_hashed_files": len(CHECKED), **COUNTS,
              "protected_documents": len(manifest["protected_documents"]),
              "live_document_drift_preserved_as_snapshots": drift,
              "limits": "Integrity, line-range and document-syntax checks only. No algorithm, GPU or general termination proof."}
    if args.write_report:
        (STUDY / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
