#!/usr/bin/env python3
"""Read-only MoE audit. Run from any cwd; stdout is the reproducible JSON evidence.

This parses archived YAML, follows only the existing from_parent references,
and counts existing records. It does not run extraction, a checker, or a kernel.
YAML equality is parsed-value equality, not a claim of semantic equivalence.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
RUN = REPO / ".schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029"
CHECKPOINT = "92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb"
RAW = RUN / "checkpoints" / CHECKPOINT / "artifacts"
INITIAL = REPO / "schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe"
NODES = REPO / "schema_extract/archive/six-operators/20261109-pause/nodes.csv"
ATTEMPTS = REPO / "schema_extract/archive/six-operators/20261109-usage-review/attempts.csv"
REGIONS = ("math_reshape", "work_schedule", "tile_pipeline")
PREFIXES = (
    "init_probe_double_verify_", "init_probe_verify_", "init_no_scan_",
    "init_probe_", "probe_double_verify_", "probe_verify_",
    "double_verify_", "no_scan_", "verify_", "probe_",
)


def relative(p):
    return str(Path(p).relative_to(REPO))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def changes(a, b, path=""):
    """Structural paths; list-length change counts as a single changed path."""
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        result = [path + "/" + str(k) for k in sorted(a.keys() ^ b.keys())]
        for k in sorted(a.keys() & b.keys()):
            result.extend(changes(a[k], b[k], path + "/" + str(k)))
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (list size {len(a)} -> {len(b)})"]
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in changes(x, y, path + "/" + str(i))]
    return [] if a == b else [path]


def documents(directory):
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(directory.glob("*.yaml"))}


def resolver(docs):
    def resolve(node, region, seen=()):
        if node in seen:
            raise ValueError("cyclic reference")
        value = docs[node]["design"][region]
        if isinstance(value, dict) and set(value) == {"from_parent"}:
            parent = docs[node]["meta_data"].get("dispatch", {}).get("parent", "root")
            assert value["from_parent"] == parent
            return resolve(parent, region, seen + (node,))
        return value
    return lambda node: {region: resolve(node, region) for region in REGIONS}


def compare(docs, a, b):
    effective = resolver(docs)
    da, db = effective(a), effective(b)
    # Historical artifacts use phase; these keys are data, not current syntax.
    pa = da["tile_pipeline"].get("phase", {})
    pb = db["tile_pipeline"].get("phase", {})
    return {
        "a": a, "b": b,
        "equal_effective_regions": [r for r in REGIONS if da[r] == db[r]],
        "equal_pipeline_phases": [k for k in sorted(pa.keys() & pb.keys()) if pa[k] == pb[k]],
        "added_pipeline_phases": sorted(pb.keys() - pa.keys()),
        "removed_pipeline_phases": sorted(pa.keys() - pb.keys()),
        "different_design_paths": changes(da, db),
        "equal_realization": docs[a].get("realization") == docs[b].get("realization"),
        "different_realization_paths": changes(docs[a].get("realization"), docs[b].get("realization")),
        "equal_metadata": docs[a]["meta_data"] == docs[b]["meta_data"],
        "coverage": "All effective Design fields, including parameters and action prose; metadata and realization checked separately. No semantic normalization.",
    }


def totals(rows):
    return {
        "attempts": len(rows),
        "unique_nodes": len({r["node"] for r in rows}),
        "cost_30m_usd": str(sum((Decimal(r["cost_30m_usd"] or "0") for r in rows), Decimal(0))),
        "cli_seconds": sum(float(r["cli_seconds"] or "0") for r in rows),
        "processing_seconds": sum(float(r["processing_seconds"] or "0") for r in rows),
        "by_role": dict(sorted(Counter(r["role"] for r in rows).items())),
        "by_response_decision": dict(sorted(Counter(r["response_decision"] for r in rows).items())),
        "by_outcome": dict(sorted(Counter(r["outcome"] for r in rows).items())),
    }


state = json.loads((RUN / "state.json").read_text())
lock = json.loads((RUN / "inputs.lock.json").read_text())
inventory = [r for r in read_csv(NODES) if r["operator"] == "moe"]
attempts = [r for r in read_csv(ATTEMPTS) if r["operator"] == "MoE"]
latest = documents(RAW)
initial = documents(INITIAL)
assert {r["node"] for r in inventory} == {r["id"] for r in state["plan"]}
assert set(latest) == {r["node"] for r in inventory if r["category_zh"] == "已批准"}

groups = defaultdict(list)
for row in inventory:
    node = row["node"]
    prefix = next((p for p in PREFIXES if node.startswith(p)), "BASE")
    groups[prefix].append({
        "node": node,
        "suffix": node[len(prefix):] if prefix != "BASE" else node,
        "category": row["category_zh"],
    })

fc1 = ("mono_arrival", "mono_fused", "mono_split", "selfprep", "w1_fused",
       "w1_split", "ws12", "ws_arrival", "ws_fused", "ws_split")
fc2 = ("flat", "mono", "redf", "w1", "ws")
product = {f"{a}_{b}" for a in fc1 for b in fc2}
assert product <= set(initial)

reference_nodes = {
    n: {r: latest[n]["design"][r]["from_parent"] for r in REGIONS
        if set(latest[n]["design"][r]) == {"from_parent"}}
    for n in sorted(latest) if n != "root"
}
reference_nodes = {n: refs for n, refs in reference_nodes.items() if refs}

source_matches = {}
for name in ("solution.py", "nvfp4_moe.cu", "nvfp4_fast.c"):
    path = "kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/" + name
    expected = lock["source"]["files"][path]
    current = digest(REPO / path)
    source_matches[path] = {"sha256": current, "locked_sha256": expected, "equal": current == expected}

selected_file_checks = []
for r in inventory:
    if r["source_file"]:
        file_path = Path(r["source_file"])
        selected = latest.get(r["node"])
        if selected is None:
            selected = yaml.safe_load(file_path.read_text())
        selected_file_checks.append({
            "node": r["node"], "category": r["category_zh"], "source": relative(file_path),
            "sha256": digest(file_path), "matches_inventory": digest(file_path) == r["own_sha256"],
            "lines": len(file_path.read_text().splitlines()),
            "when_nonempty": bool(selected.get("meta_data", {}).get("dispatch", {}).get("when")),
        })

revision_evidence = []
for r in attempts:
    if r["response_decision"] != "revise":
        continue
    directory = REPO / r["evidence_directory"]
    receipt = directory / "receipt.json"
    result_path = receipt if receipt.exists() else directory / "response.json"
    response = json.loads(result_path.read_text())
    revision_evidence.append({
        "attempt_id": r["attempt_id"], "version": r["version"], "stage": r["stage"],
        "node": r["node"], "response_file": relative(result_path),
        "accepted_receipt_present": receipt.exists(), "summary": response.get("summary"),
        "issues": response.get("issues"),
    })

p5_latest = [r for r in attempts if r["version"].startswith("2.5.0") and r["stage"] == "P5"]
rejected = []
for r in p5_latest:
    p = REPO / r["evidence_directory"] / "rejected.json"
    if p.exists():
        rejected.append({"node": r["node"], "role": r["role"], "attempt_id": r["attempt_id"],
                         "cost_30m_usd": r["cost_30m_usd"], "file": relative(p),
                         **json.loads(p.read_text())})

report = {
    "scope": "Read-only archived counts and parsed YAML comparisons. No extraction, checker, GPU validation, cost projection, or new approval.",
    "inputs": {relative(p): digest(p) for p in (NODES, ATTEMPTS, RUN / "state.json", RUN / "inputs.lock.json")},
    "source_matches": source_matches,
    "initial": {"yaml_count": len(initial), "fc1_rows": fc1, "fc2_columns": fc2,
                "product_count": len(product), "other_nodes": sorted(set(initial) - product)},
    "paused": {"run": relative(RUN), "status": state["status"], "stage": state["stage"],
               "checkpoint": CHECKPOINT, "planned": len(state["plan"]),
               "categories": dict(sorted(Counter(r["category_zh"] for r in inventory).items())),
               "raw_at_checkpoint": len(latest), "selected_files": selected_file_checks,
               "prefix_groups": {k: {"count": len(v), "nodes": sorted(v, key=lambda x: x["node"])} for k, v in sorted(groups.items())},
               "approved_ref_nodes": len(reference_nodes),
               "approved_ref_regions": sum(map(len, reference_nodes.values())),
               "approved_references": reference_nodes},
    "latest_comparisons": [compare(latest, a, b) for a, b in (
        ("fused_small_flat", "no_scan_fused_small_flat"),
        ("no_scan_fused_big", "init_no_scan_fused_big"),
        ("fused_small_mono", "fused_small_flat"),
        ("fused_small_mono", "fused_small_cp"),
        ("root", "no_scan_fused_big"), ("root", "fused_small"),
        ("fused_small", "split_store"), ("root", "fused_serial"),
        ("verify_root", "double_verify_root"),
    )],
    "initial_comparisons": [compare(initial, a, b) for a, b in (
        ("mono_fused_mono", "mono_fused_flat"),
        ("mono_fused_mono", "mono_split_mono"),
        ("mono_split_mono", "mono_arrival_mono"),
        ("mono_split_mono", "mono_split_redf"),
        ("ws_split_mono", "selfprep_mono"),
        ("bf16_tc", "bf16_warp"),
    )],
    "cost": {"basis": "Existing deduplicated attempts.csv, user-supplied 30-minute cache-write pricing; not a bill or clean-slate extraction experiment.",
             "all_moe": totals(attempts),
             "moe_2_5_series": totals([r for r in attempts if r["version"].startswith("2.5.0")]),
             "moe_2_5_p5": totals(p5_latest),
             "p5_2_5_rejected_proposals": rejected,
             "revision_evidence": revision_evidence},
}
print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
