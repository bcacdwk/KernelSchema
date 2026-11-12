"""Verify research inputs, report evidence ranges, and local Markdown links."""

from pathlib import Path
import argparse
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "schema_discussion/phase_grammar"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    corpus = json.loads((BASE / "corpus.json").read_text())
    errors = []
    totals = {"hashes": 0, "evidence_ranges": 0, "quotes": 0, "local_links": 0}
    hashed = {}
    records = [corpus["prior_manifest"], *corpus["reports"]]
    records += [{"path": r["snapshot"], "sha256": r["sha256"]} for r in corpus["rules"]]
    for operator in corpus["operators"].values():
        records += [{"path": n["snapshot"], "sha256": n["sha256"]}
                    for n in operator["nodes"] if n.get("snapshot")]
        records += operator["sources"]
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"Missing input: {record['path']}")
            continue
        digest = sha(path)
        if digest != record["sha256"]:
            errors.append(f"Changed input: {record['path']}")
        hashed[str(path.resolve())] = digest
        totals["hashes"] += 1

    cache = {}

    def read(path):
        if path not in cache:
            cache[path] = path.read_text().splitlines(keepends=True)
        return cache[path]

    def check_tree(value, owner, key_path=""):
        if isinstance(value, list):
            for i, child in enumerate(value):
                check_tree(child, owner, f"{key_path}[{i}]")
            return
        if not isinstance(value, dict):
            return
        if {"path", "start_line", "end_line"} <= set(value):
            path = ROOT / value["path"]
            location = f"{owner}:{key_path}"
            if not path.is_file():
                errors.append(f"Missing evidence file: {location}: {value['path']}")
            else:
                lines = read(path)
                start, end = value["start_line"], value["end_line"]
                if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(lines):
                    errors.append(f"Invalid range: {location}: {start}-{end}, file has {len(lines)} lines")
                else:
                    totals["evidence_ranges"] += 1
                    quote = value.get("quote")
                    if quote is not None:
                        if not isinstance(quote, str) or not quote or quote not in "".join(lines[start-1:end]):
                            errors.append(f"Quote mismatch: {location}: {value['path']}:{start}-{end}")
                        else:
                            totals["quotes"] += 1
                    if str(path.resolve()) not in hashed:
                        hashed[str(path.resolve())] = sha(path)
        for key, child in value.items():
            check_tree(child, owner, f"{key_path}.{key}")

    reports = sorted((BASE / "data").glob("[0-9][0-9]_*.json"))
    for path in reports:
        check_tree(json.loads(path.read_text()), path.name)

    for path in [BASE / "README.md", BASE / "METHOD.md", *sorted((BASE / "reports").glob("*.md"))]:
        document = path.read_text()
        targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", document)
        targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", document, re.M)
        for target in targets:
            target = target.strip().strip("<>")
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            target = re.sub(r":\d+(?:[-–]\d+)?$", "", target)
            linked = Path(target) if target.startswith("/") else path.parent / target
            if not linked.exists():
                errors.append(f"Missing link: {path.relative_to(BASE)} -> {target}")
            else:
                totals["local_links"] += 1
    result = {"status": "pass" if not errors else "fail", "totals": totals,
              "report_data_files": [str(p.relative_to(BASE)) for p in reports],
              "errors": errors,
              "additional_evidence_hashes": {str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else p: digest
                                            for p, digest in hashed.items()}}
    if args.write_report:
        (BASE / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "totals", "report_data_files", "errors")}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
