"""Check review materials and preserve the captured formal rules."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import yaml

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


class UniqueLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"duplicate explicit key: {key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def main():
    inputs = json.loads((BASE / "inputs.json").read_text())
    expected, errors, counts = {}, [], Counter()

    def collect(value, owner):
        if isinstance(value, list):
            for i, child in enumerate(value):
                collect(child, f"{owner}[{i}]")
        elif isinstance(value, dict):
            file_path = value.get("path", value.get("snapshot"))
            if isinstance(file_path, str) and isinstance(value.get("sha256"), str):
                path, digest = file_path, value["sha256"]
                if path in expected and expected[path] != digest:
                    errors.append(f"conflicting hash: {owner}: {path}")
                expected[path] = digest
                if isinstance(value.get("snapshot"), str):
                    expected[value["snapshot"]] = digest
            start = value.get("start_line", value.get("line_start"))
            end = value.get("end_line", value.get("line_end"))
            if start is None and end is None and isinstance(value.get("lines"), list) and len(value["lines"]) == 2:
                start, end = value["lines"]
            if "path" in value and start is not None and end is not None:
                path = ROOT / value["path"]
                if not path.is_file():
                    errors.append(f"missing evidence: {owner}: {path}")
                else:
                    lines = path.read_text().splitlines(keepends=True)
                    if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(lines):
                        errors.append(f"invalid evidence range: {owner}")
                    else:
                        counts["evidence_ranges"] += 1
                        if "quote" in value:
                            quote = value["quote"]
                            if not isinstance(quote, str) or not quote or quote not in "".join(lines[start - 1:end]):
                                errors.append(f"quote mismatch: {owner}")
                            else:
                                counts["exact_quotes"] += 1
            for key, child in value.items():
                collect(child, f"{owner}.{key}")

    collect(inputs, "inputs")
    for path in sorted((BASE / "data").glob("[0-9][0-9]_*.json")):
        collect(json.loads(path.read_text()), path.name)
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"changed or missing input: {relative}")
        counts["hashes"] += 1
    for path in sorted(BASE.rglob("*.md")):
        if path.parent.name in {"rules", "inputs"}:
            continue
        content = path.read_text()
        targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", content)
        targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", content, re.M)
        for target in targets:
            target = target.strip().strip("<>")
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", target) or target.startswith("#"):
                continue
            target_file = target.split("#", 1)[0]
            match = re.search(r":(\d+)(?:[-–](\d+))?$", target_file)
            if match:
                target_file = target_file[:match.start()]
            linked = Path(target_file) if target_file.startswith("/") else path.parent / target_file
            if linked.resolve() == (BASE / "data/validation.json").resolve():
                counts["local_links"] += 1
                continue
            if not linked.exists():
                errors.append(f"broken link in {path.relative_to(BASE)}: {target}")
                continue
            counts["local_links"] += 1
            if match and linked.is_file():
                length = len(linked.read_text().splitlines())
                if not 1 <= int(match[1]) <= int(match[2] or match[1]) <= length:
                    errors.append(f"invalid citation in {path.relative_to(BASE)}: {target}")
                counts["line_citations"] += 1
        for block in re.finditer(r"^(?P<fence>`{3,}|~{3,})ya?ml[ \t]*\n(?P<body>.*?)^(?P=fence)[ \t]*$", content, re.M | re.S):
            try:
                list(yaml.load_all(block["body"], Loader=UniqueLoader))
                counts["yaml_sketches"] += 1
            except Exception as exc:
                errors.append(f"invalid YAML in {path.relative_to(BASE)}: {exc}")
    for path in sorted((BASE / "sketches").glob("*.yaml")):
        try:
            list(yaml.load_all(path.read_text(), Loader=UniqueLoader))
            counts["yaml_sketches"] += 1
        except Exception as exc:
            errors.append(f"invalid YAML in {path.relative_to(BASE)}: {exc}")
    result = {"status": "fail" if errors else "pass", "counts": dict(counts), "errors": errors,
              "scope": "Material hashes, supplied exact quotes, evidence/citation bounds, links and YAML parsing only. Design necessity is assessed by the six agents; no GPU or complete-schema validation."}
    (BASE / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
