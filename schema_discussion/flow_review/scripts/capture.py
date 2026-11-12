"""Capture current rules and reuse the existing, immutable ten-operator inputs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "schema_discussion/flow_review"
PRIOR = ROOT / "schema_extract/archive/20261109-ten-operator-text-audit/data/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "corpus.json").exists():
        raise SystemExit("Capture already exists; do not silently refresh research inputs.")
    for folder in ["rules", "reports", "data", "sketches"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text())
    corpus = {
        "format": "kernelschema.flow_review.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_at_start": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "prior_manifest": {"path": str(PRIOR.relative_to(ROOT)), "sha256": digest(PRIOR)},
        "rules": [],
        "operators": prior["operators"],
        "sources": {},
        "surveys": [],
    }
    for path in sorted((ROOT / "schema_docs").glob("[0-6].*.md")):
        dest = OUT / "rules" / path.name
        shutil.copyfile(path, dest)
        corpus["rules"].append({"path": str(path.relative_to(ROOT)), "snapshot": str(dest.relative_to(ROOT)), "sha256": digest(path)})
    count = Counter()
    for opid, operator in corpus["operators"].items():
        for node in operator["nodes"]:
            if not node.get("snapshot"):
                continue
            path = ROOT / node["snapshot"]
            assert digest(path) == node["sha256"], path
            count[node["category"]] += 1
            if opid.startswith("template_"):
                current = ROOT / node["source_file"]
                node["matches_current_template"] = current.read_bytes() == path.read_bytes()
                assert node["matches_current_template"], current
    old_sources = json.loads((ROOT / "schema_discussion/phase_grammar/corpus.json").read_text())
    for opid, operator in old_sources["operators"].items():
        corpus["sources"][opid] = operator.get("sources", [])
        for source in corpus["sources"][opid]:
            path = ROOT / source["path"]
            assert digest(path) == source["sha256"], path
    for relative in [
        "schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md",
        "schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md",
    ]:
        path = ROOT / relative
        corpus["surveys"].append({"path": relative, "sha256": digest(path)})
    corpus["file_categories"] = dict(count)
    (OUT / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"operators": len(corpus["operators"]), "file_categories": dict(count), "rule_snapshots": len(corpus["rules"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
