"""Validate fixed inputs, quoted evidence, links, and YAML sketch syntax."""

from pathlib import Path
import argparse
import hashlib
import json
import re

import yaml


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "schema_discussion/phase_grammar/view_review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    inputs = json.loads((BASE / "inputs.json").read_text())
    corpus = json.loads((ROOT / inputs["parent_corpus"]["path"]).read_text())
    expected = {inputs["parent_corpus"]["path"]: inputs["parent_corpus"]["sha256"]}
    expected.update({r["snapshot"]: r["sha256"] for r in inputs["rules"]})
    for operator in corpus["operators"].values():
        expected.update({n["snapshot"]: n["sha256"] for n in operator["nodes"] if n.get("snapshot")})
        expected.update({s["path"]: s["sha256"] for s in operator["sources"]})
    errors = []
    totals = {"input_hashes": 0, "evidence_ranges": 0, "quotes": 0, "local_links": 0, "yaml_sketches": 0}
    for path, digest in expected.items():
        full = ROOT / path
        if not full.is_file() or sha(full) != digest:
            errors.append(f"missing/changed fixed input: {path}")
        totals["input_hashes"] += 1

    def inspect(value, owner):
        if isinstance(value, list):
            for i, child in enumerate(value):
                inspect(child, f"{owner}[{i}]")
        elif isinstance(value, dict):
            if {"path", "start_line", "end_line"} <= value.keys():
                path = ROOT / value["path"]
                if not path.is_file():
                    errors.append(f"missing evidence: {owner}: {path}")
                else:
                    lines = path.read_text().splitlines(keepends=True)
                    start, end = value["start_line"], value["end_line"]
                    if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(lines):
                        errors.append(f"invalid range: {owner}: {start}-{end}")
                    else:
                        totals["evidence_ranges"] += 1
                        if "quote" in value:
                            quote = value["quote"]
                            if not isinstance(quote, str) or not quote or quote not in "".join(lines[start-1:end]):
                                errors.append(f"quote mismatch: {owner}")
                            else:
                                totals["quotes"] += 1
            for key, child in value.items():
                inspect(child, f"{owner}.{key}")

    for path in sorted((BASE / "data").glob("[0-9][0-9]_*.json")):
        inspect(json.loads(path.read_text()), path.name)
    for path in [BASE / "README.md", BASE / "BRIEF.md", *sorted((BASE / "reports").glob("*.md"))]:
        text = path.read_text()
        targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text)
        targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, re.M)
        for target in targets:
            target = target.strip().strip("<>")
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            target = re.sub(r":\d+(?:[-–]\d+)?$", "", target.split("#", 1)[0])
            linked = Path(target) if target.startswith("/") else path.parent / target
            pending_report = args.write_report and linked.resolve() == (BASE / "data/validation.json").resolve()
            if not linked.exists() and not pending_report:
                errors.append(f"missing link: {path.name}: {target}")
            else:
                totals["local_links"] += 1
        for block in re.finditer(r"^```ya?ml\s*\n(.*?)^```\s*$", text, re.M | re.S):
            try:
                list(yaml.load_all(block.group(1), Loader=UniqueLoader))
                totals["yaml_sketches"] += 1
            except Exception as exc:
                errors.append(f"invalid YAML sketch: {path.name}: {exc}")
    live_rules = {r["path"]: sha(ROOT / r["path"]) == r["sha256"] for r in inputs["rules"]}
    result = {"status": "pass" if not errors else "fail", "totals": totals,
              "live_rules_match_capture": live_rules, "errors": errors,
              "scope": "Source identity, exact citations, local-link targets and sketch YAML syntax; no validation of complete proposed schemas or GPU behavior."}
    if args.write_report:
        (BASE / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
