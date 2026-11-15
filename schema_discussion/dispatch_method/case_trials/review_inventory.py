"""Read-only inventory of the trial documents and their cited comparisons.

Run from the repository root:
  python schema_discussion/dispatch_method/case_trials/review_inventory.py
No extraction, schema rewriting, or GPU execution is performed.
"""

from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import unquote

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORT = HERE.parent / "06_case_trials.md"
CASES = {name: HERE / name / "dispatch.md" for name in ("moe", "fa4_prefill")}
ARTIFACTS = REPO / (
    ".schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/"
    "checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/"
    "artifacts"
)


def section(text, start):
    tail = text.split(start + "\n", 1)[1]
    return tail.split("\n## ", 1)[0]


def load_yaml(path):
    return yaml.safe_load(path.read_text())


def effective_area(name, area):
    doc = load_yaml(ARTIFACTS / (name + ".yaml"))
    value = doc["design"][area]
    if isinstance(value, dict) and set(value) == {"from_parent"}:
        return effective_area(value["from_parent"], area)
    return value


def changed_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            path = prefix + "/" + str(key)
            result.extend([path] if key not in a or key not in b else changed_paths(a[key], b[key], path))
        return result
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [path for i, (x, y) in enumerate(zip(a, b)) for path in changed_paths(x, y, prefix + "/" + str(i))]
    return [] if a == b else [prefix]


documents = {path: path.read_text() for path in [*CASES.values(), REPORT]}
corpus = "\n".join(documents.values())
counts = {}
for name, path in CASES.items():
    text = documents[path]
    mains = section(text, "## 拟保留的完整主线")
    local_heading = next(line for line in text.splitlines() if line.startswith("## 一次记录"))
    locals_text = section(text, local_heading)
    records = re.findall(r"^### ([^\n]+)\n\n(.*?)(?=\n### |\Z)", locals_text, re.M | re.S)
    definitions = Counter(record_id for record_id, _ in records)
    body_occurrences = {record_id: corpus.count(body.strip()) for record_id, body in records}
    counts[name] = {
        "proposed_full_main_entries": len(re.findall(r"^\| `", mains, re.M)),
        "local_record_entries": len(records),
        "record_definitions": dict(definitions),
        "whole_record_body_occurrences": body_occurrences,
        "extra_whole_record_restatements": sum(max(0, value - 1) for value in body_occurrences.values()),
        "new_complete_yaml_files": len(list(path.parent.glob("*.yaml"))) + len(list(path.parent.glob("*.yml"))),
    }

broken_links = []
for path, text in documents.items():
    targets = re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, re.M)
    targets += re.findall(r"\]\(([^\s)]+)\)", text)
    for target in targets:
        if target.startswith(("http://", "https://", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        target = re.sub(r":\d+$", "", target)
        resolved = Path(target) if target.startswith("/") else path.parent / target
        if not resolved.exists():
            broken_links.append({"document": str(path.relative_to(REPO)), "target": target})

fa = REPO / "schema_template/examples/fa4_prefill"
root = load_yaml(fa / "root.yaml")["design"]
packed = load_yaml(fa / "packed_persistent.yaml")["design"]
single = load_yaml(fa / "single_persistent_dual_load.yaml")["design"]
static = load_yaml(fa / "small_static_lpt.yaml")["design"]
static_phase = static["tile_pipeline"]["phase"]["attention"]
areas = ("math_reshape", "work_schedule", "tile_pipeline")
checks = {
    "fa_persistent_math_equal": packed["math_reshape"] == single["math_reshape"],
    "fa_persistent_pipeline_equal": packed["tile_pipeline"] == single["tile_pipeline"],
    "fa_persistent_schedule_changed_paths": changed_paths(packed["work_schedule"], single["work_schedule"]),
    "fa_persistent_math_equals_root": packed["math_reshape"] == root["math_reshape"],
    "fa_static_pipeline_equals_root": static["tile_pipeline"] == root["tile_pipeline"],
    "fa_static_has_workcoord": "WorkCoord" in static_phase["resource"],
    "fa_static_first_load_dependencies": {
        role: static_phase["role"][role]["waits_for"]["data"] for role in ("LoadQK", "LoadV")
    },
    "moe_rebuild_flat_areas_equal": {
        area: effective_area("fused_small_flat", area) == effective_area("no_scan_fused_small_flat", area)
        for area in areas
    },
}

print(json.dumps({"counts": counts, "broken_local_file_links": broken_links, "source_checks": checks}, ensure_ascii=False, indent=2))
raise SystemExit(bool(broken_links))
