#!/usr/bin/env python3
"""Read-only inventory for the dispatch discussion; no extraction or checks run.

From repository root:
  python schema_discussion/dispatch_method/evidence_cross_operator/inventory.py > \
    schema_discussion/dispatch_method/evidence_cross_operator/inventory.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WIKI = ROOT / "schema_wiki/archive/six-operators"
OPS = "aka-cuda-operators-20261030/operators"
SAMPLES = {
    "fa4_prefill": "schema_template/examples/fa4_prefill",
    "fa4_decode": "schema_template/examples/fa4_decode",
    "gdn_template": "schema_template/examples/gdn_prefill",
    "qsa_prefill": "schema_template/examples/qsa_prefill",
}
WIKI_OPS = {
    "gdn": "linear_attention/gdn_prefill_bf16_state_bf16",
    "mla": "mla/mla_expanded_fp8",
    "gemm": "gemm/grouped/fp8_scaled_grouped_gemm",
    "sparse": "sparse_attention/dual_cache_latent_bf16",
    "fmha": "fmha/fmha_mxfp8_qkv",
    "moe": "moe/nvfp4_fused_moe",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def item(path: Path) -> dict:
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    meta = doc["meta_data"]
    return {
        "path": relative(path),
        "sha256": sha(raw),
        "bytes": len(raw),
        "version": meta["version"],
        "source": meta.get("source"),
        "id": meta.get("dispatch", {}).get("id", "root"),
        "parent": meta.get("dispatch", {}).get("parent"),
        "empty_when": meta.get("dispatch", {}).get("when") == "",
    }


def directory(path: Path) -> dict:
    nodes = [item(p) for p in sorted(path.glob("*.yaml"))]
    return {"path": relative(path), "yaml_count": len(nodes), "nodes": nodes}


def diff_paths(a, b, at="design"):
    if type(a) is not type(b):
        return [at]
    if isinstance(a, dict):
        out = []
        for key in sorted(a.keys() | b.keys()):
            out.extend([at + "." + key] if key not in a or key not in b
                       else diff_paths(a[key], b[key], at + "." + key))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [at]
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in diff_paths(x, y, f"{at}[{i}]")]
    return [] if a == b else [at]


result = {
    "purpose": "Snapshot counts and source identity, not independent-design counts or correctness proof.",
    "count_rule": "One directory-level YAML per node; do not add initial and revised versions. Pause rows exclude retired history; selected files may be empty-when scaffolds.",
    "templates": {k: directory(ROOT / v) for k, v in SAMPLES.items()},
    "initial_2_2_2": {k: directory(WIKI / "initial" / OPS / v)
                      for k, v in WIKI_OPS.items()},
    "latest_published": {},
}
for name, sub in WIKI_OPS.items():
    version = "2.2.7" if name == "gdn" else "2.5.0"
    path = WIKI / "revisions" / version / OPS / sub
    if not path.exists():
        path = WIKI / "initial" / OPS / sub
    result["latest_published"][name] = directory(path)

csv_path = ROOT / "schema_extract/archive/six-operators/20261109-pause/nodes.csv"
rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
result["pause_csv_sha256"] = sha(csv_path.read_bytes())
result["pause"] = {}
for name in WIKI_OPS:
    all_rows = [r for r in rows if r["operator"] == name]
    active = [r for r in all_rows if r["category_zh"] != "已退役历史节点"]
    selected = []
    for row in active:
        source = row["source_file"]
        info = {"id": row["node"], "category": row["category_zh"],
                "recorded_version": row["node_version"],
                "selected_checkpoint": row["selected_checkpoint"]}
        if source and Path(source).exists():
            info["selected_yaml"] = item(Path(source))
            info["matches_recorded_sha256"] = info["selected_yaml"]["sha256"] == row["own_sha256"]
        selected.append(info)
    result["pause"][name] = {
        "plan_count": len(active),
        "approved_count": sum(r["category_zh"] == "已批准" for r in active),
        "retired_rows_excluded": len(all_rows) - len(active),
        "status_counts": dict(Counter(r["category_zh"] for r in active)),
        "selected_yaml_files": sum("selected_yaml" in n for n in selected),
        "empty_when_files": sum(n.get("selected_yaml", {}).get("empty_when", False) for n in selected),
        "nodes": selected,
    }

template_count = sum(x["yaml_count"] for x in result["templates"].values())
result["totals"] = {
    "four_templates": template_count,
    "six_initial": sum(x["yaml_count"] for x in result["initial_2_2_2"].values()),
    "ten_with_initial_wiki": template_count + sum(x["yaml_count"] for x in result["initial_2_2_2"].values()),
    "ten_with_latest_published_wiki": template_count + sum(x["yaml_count"] for x in result["latest_published"].values()),
    "six_pause_planned": sum(x["plan_count"] for x in result["pause"].values()),
    "six_pause_approved": sum(x["approved_count"] for x in result["pause"].values()),
    "six_pause_selected_yaml_files": sum(x["selected_yaml_files"] for x in result["pause"].values()),
}

base = ROOT / "schema_template/examples/fa4_prefill"
a = yaml.safe_load((base / "packed_persistent.yaml").read_text())["design"]
b = yaml.safe_load((base / "single_persistent_dual_load.yaml").read_text())["design"]
result["prefill_pair"] = {
    "a": relative(base / "packed_persistent.yaml"),
    "b": relative(base / "single_persistent_dual_load.yaml"),
    "equal_regions": [region for region in a if a[region] == b[region]],
    "different_paths": diff_paths(a, b),
    "semantic_note": "The differing strategy text selects binary versus linear lookup over the same prefix table. Semantic equivalence is argued separately using the fixed source; equality of parsed regions alone is not that proof.",
}

result["source_checks"] = []
old_ref = "01fff90d714abc6e84904ea9c44523ba66d34a0e"
old_root = "kernel_wiki/aka-four-problems-20261019"
for suffix in [
    "fa4_prefill/aka_solution/flash_attn/cute/interface.py",
    "fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py",
    "fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py",
    "fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py",
    "fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py",
    "gdn_prefill/aka_solution/gated_delta_net_chunked.py",
    "qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py",
    "qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py",
]:
    path = old_root + "/" + suffix
    data = subprocess.check_output(["git", "show", old_ref + ":" + path], cwd=ROOT)
    result["source_checks"].append({"path": path, "ref": old_ref,
                                    "sha256": sha(data), "current_worktree_file_exists": (ROOT / path).exists()})

for suffix in ["linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu",
               "sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu",
               "fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu",
               "fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh",
               "fmha/fmha_mxfp8_qkv/candidate/fmha_common.cuh"]:
    path = "kernel_wiki/" + OPS + "/" + suffix
    ref = "6e4ca91bd91d998cac99e8dadd83838e153c679d"
    data = subprocess.check_output(["git", "show", ref + ":" + path], cwd=ROOT)
    result["source_checks"].append({"path": path, "ref": ref, "sha256": sha(data),
                                    "current_matches_fixed_git_source": data == (ROOT / path).read_bytes()})

for run_id, suffix in [
    ("531567f1-bcae-45ad-acf6-fd674c055fdc", "mla/mla_expanded_fp8/candidate/solution.py"),
    ("d4e41124-e29c-458c-a1ce-2a9bf87ee0a8", "gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_umma.cu"),
]:
    lock = json.loads((ROOT / ".schema_extract/v2/runs" / run_id / "inputs.lock.json").read_text())
    path = "kernel_wiki/" + OPS + "/" + suffix
    fixed = Path(lock["snapshot"]) / "source" / path
    data = fixed.read_bytes()
    result["source_checks"].append({"path": path, "ref": lock["source"]["ref"],
                                    "snapshot": relative(fixed), "sha256": sha(data),
                                    "current_matches_frozen_source_file": data == (ROOT / path).read_bytes()})

print(json.dumps(result, ensure_ascii=False, indent=2))
