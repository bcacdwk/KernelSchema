"""Check research input identity, local citations, and YAML sketch syntax."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "schema_discussion/flow_review"


class UniqueLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"duplicate explicit YAML key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def main():
    corpus = json.loads((OUT / "corpus.json").read_text())
    errors, counts = [], Counter()
    expected = {corpus["prior_manifest"]["path"]: corpus["prior_manifest"]["sha256"]}
    for rule in corpus["rules"]:
        expected[rule["snapshot"]] = rule["sha256"]
        expected[rule["path"]] = rule["sha256"]
    for operator in corpus["operators"].values():
        for node in operator["nodes"]:
            if node.get("snapshot"):
                expected[node["snapshot"]] = node["sha256"]
    for sources in corpus["sources"].values():
        for source in sources:
            expected[source["path"]] = source["sha256"]
    for survey in corpus["surveys"]:
        expected[survey["path"]] = survey["sha256"]

    def inspect_evidence(value, owner):
        if isinstance(value, list):
            for i, child in enumerate(value):
                inspect_evidence(child, f"{owner}[{i}]")
        elif isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                relative, digest = value["path"], value["sha256"]
                if relative in expected and expected[relative] != digest:
                    errors.append(f"conflicting input hash: {owner}: {relative}")
                expected[relative] = digest
            ranges = []
            if {"path", "start_line", "end_line"} <= value.keys():
                ranges.append((value["start_line"], value["end_line"], value.get("quote")))
            elif "path" in value and isinstance(value.get("lines"), list) and len(value["lines"]) == 2:
                ranges.append((*value["lines"], None))
            if "path" in value and isinstance(value.get("principal_ranges"), list):
                ranges.extend((start, end, None) for start, end in value["principal_ranges"])
            for start, end, quote in ranges:
                path = ROOT / value["path"]
                if not path.is_file():
                    errors.append(f"missing evidence: {owner}: {path}")
                else:
                    lines = path.read_text().splitlines(keepends=True)
                    if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(lines):
                        errors.append(f"invalid evidence range: {owner}: {start}-{end}")
                    else:
                        counts["evidence_ranges"] += 1
                        if quote is not None:
                            if not quote or quote not in "".join(lines[start - 1:end]):
                                errors.append(f"quote mismatch: {owner}")
                            else:
                                counts["exact_quotes"] += 1
            for key, child in value.items():
                inspect_evidence(child, f"{owner}.{key}")

    for path in sorted((OUT / "data").glob("[0-9][0-9]_*.json")):
        inspect_evidence(json.loads(path.read_text()), path.name)
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"changed or missing input: {relative}")
        counts["hashes"] += 1
        if path.suffix in {".yaml", ".yml"} and path.is_file():
            try:
                yaml.load(path.read_text(), Loader=UniqueLoader)
                counts["input_yaml"] += 1
            except Exception as exc:
                errors.append(f"invalid source YAML: {relative}: {exc}")
    for path in sorted(OUT.rglob("*.md")):
        if path.parent.name == "rules":
            continue
        content = path.read_text()
        for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip().strip("<>")
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            file_target = target.split("#", 1)[0]
            match = re.search(r":(\d+)(?:[-–](\d+))?$", file_target)
            if match:
                file_target = file_target[:match.start()]
            linked = Path(file_target) if file_target.startswith("/") else path.parent / file_target
            if linked.resolve() == (OUT / "data/validation.json").resolve():
                counts["local_links"] += 1
                continue
            if not linked.exists():
                errors.append(f"broken local link in {path.relative_to(OUT)}: {target}")
                continue
            counts["local_links"] += 1
            if match and linked.is_file():
                total_lines = len(linked.read_text().splitlines())
                start, end = int(match[1]), int(match[2] or match[1])
                if not 1 <= start <= end <= total_lines:
                    errors.append(f"invalid citation lines in {path.relative_to(OUT)}: {target}")
                counts["line_citations"] += 1
        for block in re.finditer(r"^(?P<fence>`{3,}|~{3,})ya?ml[ \t]*\n(?P<body>.*?)^(?P=fence)[ \t]*$", content, re.M | re.S):
            try:
                list(yaml.load_all(block["body"], Loader=UniqueLoader))
                counts["yaml_sketches"] += 1
            except Exception as exc:
                errors.append(f"invalid YAML sketch in {path.relative_to(OUT)}: {exc}")
    for path in sorted((OUT / "sketches").glob("*.yaml")):
        try:
            list(yaml.load_all(path.read_text(), Loader=UniqueLoader))
            counts["yaml_sketches"] += 1
        except Exception as exc:
            errors.append(f"invalid YAML sketch in {path.relative_to(OUT)}: {exc}")
    result = {"status": "fail" if errors else "pass", "counts": dict(counts), "errors": errors,
              "scope": "Input hashes, evidence range bounds and supplied exact quotes, explicit duplicate YAML keys, local link targets, citation line bounds, and YAML sketch parsing. Semantic evidence is reviewed separately; no full new-schema validation or GPU verification."}
    (OUT / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
