#!/usr/bin/env python3
"""Read-only catalog/search for the imported kernel library; stdlib, no CUDA imports.

The source snapshot is immutable. `render` emits an apply_patch patch for derived
indexes; `check` checks freshness, links, and every imported file's SHA-256.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCH = "aka-cuda-operators-20261030"
SOURCE = ROOT / BATCH
CATEGORIES = {
    "gemm": "矩阵乘法 GEMM",
    "moe": "融合专家计算 MoE",
    "fmha": "多头注意力 FMHA",
    "mla": "多头潜在注意力 MLA",
    "sparse_attention": "稀疏注意力",
    "linear_attention": "线性注意力 GDN / KDA",
}
SUBCATEGORIES = {
    "dense": "普通 GEMM",
    "grouped": "分组 GEMM",
    "grouped_masked": "带掩码的分组 GEMM",
}
STATUS = {
    "certified": "原包认证",
    "passed_sampled": "采样验收通过",
    "accepted_by_user_with_performance_waiver": "人工接受性能例外",
}
AUDIT = {
    "exact_supplied_problem": "7 个题目文件逐字节核对",
    "exact_accepted_problem_core": "验收核心题目绑定已核对",
    "compatible_latest_receipt_missing": "待最终回执核验",
}
PRECISIONS = ("bf16", "fp16", "fp8", "mxfp8", "mxfp4", "nvfp4")
STAGES = ("prefill", "prefix", "decode", "mtp")

# Editorial summaries are grounded in each problem/metadata.json description,
# input.py, reference.py, shape_range.json and shape_cases.json. They classify contracts, not a new
# correctness/performance claim. Keep architecture-specific details in source.
NOTES = {
    "bf16_gemm": "BF16 A/B；D = αABᵀ + C，可选 C 与 D 别名",
    "bf16_nvfp4_gemm": "BF16 × NVFP4 普通 GEMM，原生载荷与缩放布局",
    "fp8_scaled_gemm": "带显式缩放因子的 FP8 普通 GEMM",
    "fp8_mxfp4_gemm": "FP8 × MXFP4 普通 GEMM，含可选 C 累加",
    "mxfp4_gemm": "MXFP4 普通 GEMM，含可选 C 累加",
    "mxfp8_gemm": "MXFP8 普通 GEMM，含可选 C 累加",
    "bf16_grouped_gemm": "BF16 M 分组连续 / PSUM GEMM",
    "fp8_scaled_grouped_gemm": "带缩放的 FP8 分组 GEMM，以 indptr 定义分组",
    "fp8_mxfp4_grouped_gemm": "FP8 × MXFP4 M 分组连续 / PSUM GEMM",
    "mxfp4_grouped_gemm": "MXFP4 M 分组连续 / PSUM GEMM",
    "mxfp8_grouped_gemm": "MXFP8 M 分组连续 / PSUM GEMM",
    "bf16_grouped_gemm_masked": "BF16 专家分组 GEMM，只写各组有效前缀",
    "fp8_mxfp4_grouped_gemm_masked": "FP8 × MXFP4 专家分组 GEMM，只写有效前缀",
    "mxfp4_grouped_gemm_masked": "MXFP4 专家分组 GEMM，只写有效前缀",
    "mxfp8_grouped_gemm_masked": "MXFP8 专家分组 GEMM，只写有效前缀",
    "bf16_fused_moe": "预路由 BF16 专家 FC1 → 激活 → FC2 → 加权归约",
    "fp8_mxfp4_fused_moe": "预路由 FP8 / MXFP4 融合专家计算与加权归约",
    "fp8_scaled_fused_moe": "FP8 块缩放专家 SwiGLU FFN 与加权归约",
    "mxfp8_fused_moe": "预路由 MXFP8 融合专家计算与加权归约",
    "nvfp4_fused_moe": "NVFP4 路由 MoE；FC2 前含激活量化",
    "fmha_bf16": "BF16 Q/K/V → BF16 O；分页 MHA / GQA / MQA",
    "fmha_bf16_fp8": "BF16 Q + FP8 KV；per-tensor 缩放",
    "fmha_fp16": "FP16 Q/K/V → FP16 O；分页注意力",
    "fmha_fp8": "FP8 Q/K/V；标量缩放，普通 FMHA 几何",
    "fmha_fp8_per_token_head": "BF16 Q + FP8 KV；每 token / head 缩放，O-only",
    "fmha_mxfp8_qkv": "MXFP8 Q/K/V block32；连续布局，无分页 / ragged",
    "fmha_nvfp4_kv": "FP8 Q + NVFP4 KV；group16 缩放",
    "fmha_nvfp4_qkpv": "BF16 外部 Q/K/V，内部 NVFP4 QK 与 PV；非因果变长",
    "mla_absorbed_fp8": "吸收式 FP8 MLA：Q/K D576，V 为 K 前 512 维",
    "mla_expanded_bf16": "展开式 BF16 MLA，Q/K D192、V D128，连续 ragged prefill",
    "mla_expanded_fp8": "展开式 FP8 MLA，Q/K D192、V D128，连续 ragged prefill",
    "mla_nvfp4": "NVFP4 dense MQA：Q/K/V D512，K=V；分页 decode / MTP",
    "dual_cache_latent_bf16": "BF16 单 / 双 latent KV 缓存，联合 softmax",
    "dual_cache_latent_fp8": "FP8 单 / 双 latent KV 缓存，独立 QK / PV descale",
    "paged_indexed_latent_fp8": "物理 token 索引访问分页 FP8 latent KV；D512 / D576",
    "paged_latent_mxfp8_nvfp4": "BF16 Q + MXFP8 滑窗 KV + NVFP4 压缩 KV",
    "gdn_conv_mtp_bf16_state_bf16": "卷积 → SiLU → GDN 递推 → gated RMSNorm；BF16 检查点",
    "gdn_conv_mtp_bf16_state_fp32": "卷积 → SiLU → GDN 递推 → gated RMSNorm；FP32 检查点",
    "gdn_postconv_mtp_bf16_state_bf16": "卷积后 GDN 递推与 gated RMSNorm；BF16 检查点",
    "gdn_postconv_mtp_bf16_state_fp32": "卷积后 GDN 递推与 gated RMSNorm；FP32 检查点",
    "gdn_decode_core_bf16_state_fp32": "普通 decode ABI：卷积、GDN 递推与原位 gated RMSNorm",
    "gdn_prefill_bf16_state_bf16": "GDN packed prefill；只读 BF16 初态，可选新 FP32 终态",
    "gdn_prefill_bf16_state_fp32": "GDN packed prefill；只读 FP32 初态，可选新 FP32 终态",
    "kda_raw_prefill_state_bf16": "KDA 原始门控 prefill；逐通道衰减、BF16 终态与检查点",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stages_for(op, category):
    if category == "fmha":
        return ["prefill"] if op == "fmha_nvfp4_qkpv" else list(STAGES)
    if category == "mla":
        if "expanded" in op:
            return ["prefill"]
        return ["decode", "mtp"] if op == "mla_nvfp4" else list(STAGES)
    if category == "sparse_attention":
        return ["decode", "mtp"] if op == "paged_latent_mxfp8_nvfp4" else ["prefill", "decode", "mtp"]
    if category == "linear_attention":
        if "prefill" in op:
            return ["prefill"]
        return ["decode", "mtp"] if "mtp" in op else ["decode"]
    return []  # Do not infer serving stages for matrix/FFN contracts.


def build():
    manifest = read(SOURCE / "MANIFEST.json")
    if set(NOTES) != {op["operator"] for op in manifest["operators"]}:
        raise ValueError("Editorial summaries must cover exactly the manifest's operators")
    entries = []
    for op in manifest["operators"]:
        name = op["operator"]
        directory = SOURCE / op["directory"]
        meta = read(directory / "problem/metadata.json")
        note = NOTES[name]
        source_description = meta["description"].split("\n\n", 1)[0]
        if name in {"mla_expanded_bf16", "mla_expanded_fp8"}:
            # Mark only the superseded clause in the imported excerpt. The
            # surrounding scope, lengths and optional outputs remain searchable.
            source_description = source_description.replace(
                "H12/16/24/32/48/64/96/128；ragged首次causal prefill",
                "〔头数与因果模式见本题 problem/shape_range.json〕", 1)
            domain = read(directory / "problem/shape_range.json")["input_kwargs"]["properties"]
            heads = ", ".join(str(h) for h in sorted(domain["heads"]["values"]))
            note += f"，Hq=Hkv ∈ {{{heads}}}，支持因果和非因果模式，可选 attention sinks"
        summary = read(directory / "performance/SUMMARY.json")
        primary = next(r for r in summary["measurements"] if r["id"] == summary["primary_measurement"])
        formats = set(name.split("_state_")[0].split("_")) & set(PRECISIONS)
        formats.update({
            "fmha_nvfp4_kv": ["fp8"],
            "fmha_nvfp4_qkpv": ["bf16"],
            "fmha_fp8_per_token_head": ["bf16"],
            "paged_latent_mxfp8_nvfp4": ["bf16"],
            "kda_raw_prefill_state_bf16": ["bf16"],
        }.get(name, []))
        paths = {key: f"{BATCH}/{op['directory']}/{suffix}" for key, suffix in {
            "readme": "README.md", "candidate": "candidate",
            "entrypoint": "candidate/solution.py", "metadata": "problem/metadata.json",
            "input": "problem/input.py", "reference": "problem/reference.py",
            "production": "problem/solution.py", "shape_cases": "problem/shape_cases.json",
            "shape_range": "problem/shape_range.json", "performance": "performance/SUMMARY.json",
        }.items()}
        paths["acceptance"] = f"{BATCH}/{op['performance']['record']}"
        for key, filename in [("precision", "PRECISION.md"), ("variants", "VARIANTS.md"),
                              ("user_acceptance", "USER_ACCEPTANCE.md")]:
            if (directory / filename).is_file():
                paths[key] = f"{BATCH}/{op['directory']}/{filename}"
        unsupported = {r["id"]: [row["shape_id"] for row in r["rows"]
                                if row["status"] == "baseline_unsupported"]
                       for r in summary["measurements"]}
        entries.append({
            "id": name, "title": op["display_name"], "category": op["category"],
            "subcategory": op["subcategory"], "summary": note,
            "source_description": source_description,
            "precision_tags": [p for p in PRECISIONS if p in formats],
            "state_dtype": name.split("_state_")[1] if "_state_" in name else None,
            "stages": stages_for(name, op["category"]), "source_tags": op["tags"],
            "solution_origin": op["solution_origin"],
            "acceptance_status": summary["acceptance_status"],
            "problem_verification": op["problem_verification"],
            "requires_receipt": op["problem_requires_latest_receipt_confirmation"],
            "performance_waiver": summary["performance_waiver"],
            "metric": primary["metric"], "comparison": primary["comparison"],
            "primary_measurement": primary["id"],
            "measurement_groups": len(summary["measurements"]),
            "primary_row_status_counts": dict(Counter(row["status"] for row in primary["rows"])),
            "unsupported_baseline_shapes": {k: v for k, v in unsupported.items() if v},
            "fixed_case_count": len(read(directory / "problem/shape_cases.json")),
            "interface": op["interface"], "dependencies": op["dependencies"], "paths": paths,
        })
    entries.sort(key=lambda e: (list(CATEGORIES).index(e["category"]), e["subcategory"] or "", e["id"]))
    return {"schema": "kernel_wiki.catalog.v1", "batch": BATCH,
            "manifest": f"{BATCH}/MANIFEST.json", "manifest_sha256": sha(SOURCE / "MANIFEST.json"),
            "operator_count": len(entries), "operators": entries}


def link(entry):
    return f"[{entry['id']}]({entry['paths']['readme']})"


def render_catalog(catalog):
    lines = ["# 44 个算子总索引", "",
             "[首页](README.md) · [多维索引](FACETS.md) · [分类与阅读方法](GUIDE.md) · [验证与计时口径](EVIDENCE.md)", "",
             "由 `tools/catalog.py render` 从原清单和中文分类说明生成。每个名字直达逐题 README；继续进入源码、合同、用例和性能记录。", "",
             "精度列是主要输入／量化格式，未穷举输出、scale 和累加 dtype；状态精度单列。阶段表示合同用途，不表示阶段下的所有组合都已验收。", ""]
    for category, title in CATEGORIES.items():
        subset = [e for e in catalog["operators"] if e["category"] == category]
        lines.extend([f"## {title}（{len(subset)}）", ""])
        for subcategory in (SUBCATEGORIES if category == "gemm" else [None]):
            group = [e for e in subset if e["subcategory"] == subcategory]
            if subcategory:
                lines.extend([f"### {SUBCATEGORIES[subcategory]}（{len(group)}）", ""])
            lines.extend(["| 算子 | 计算边界 | 格式 / 状态 | 阶段 | 来源 / 验收 |",
                          "| --- | --- | --- | --- | --- |"])
            for e in group:
                precision = " / ".join(p.upper() for p in e["precision_tags"])
                if e["state_dtype"]:
                    precision += f"；state {e['state_dtype'].upper()}"
                status = f"{e['solution_origin']}；{STATUS[e['acceptance_status']]}"
                if e["requires_receipt"]:
                    status += "；**待回执**"
                lines.append(f"| {link(e)} | {e['summary']} | {precision} | {' / '.join(e['stages']) or '—'} | {status} |")
            lines.append("")
    return "\n".join(lines)


def render_facets(catalog):
    entries = catalog["operators"]
    lines = ["# 多维检索索引", "", "[首页](README.md) · [全部 44 题](CATALOG.md) · [命令行组合筛选](README.md#命令行检索)", "",
             "同一算子可出现在多个分组；源码只有一份。格式与阶段标签是检索入口，完整 ABI、缩放布局和支持域见每题合同。", ""]
    groups = [
        ("主要输入与量化格式", [(p.upper(), [e for e in entries if p in e["precision_tags"]]) for p in PRECISIONS]),
        ("递推状态精度", [(p.upper(), [e for e in entries if e["state_dtype"] == p]) for p in ["bf16", "fp32"]]),
        ("注意力阶段", [(p, [e for e in entries if p in e["stages"]]) for p in STAGES]),
        ("实现来源", [(p, [e for e in entries if e["solution_origin"] == p]) for p in ["AKA", "REFERENCE"]]),
        ("候选验收状态", [(label, [e for e in entries if e["acceptance_status"] == key]) for key, label in STATUS.items()]),
        ("题目与验收的绑定", [(label, [e for e in entries if e["problem_verification"] == key]) for key, label in AUDIT.items()]),
        ("主要测量口径与对照", [
            ("NCU kernel-sum / 生产实现", [e for e in entries if e["metric"] == "ncu_kernel_sum" and e["comparison"]["kind"] == "production"]),
            ("CUDA event forward / 历史候选", [e for e in entries if e["metric"] == "cuda_event_forward" and e["comparison"]["kind"] == "historical_candidate"]),
            ("有生产对照缺测项", [e for e in entries if e["unsupported_baseline_shapes"]]),
        ]),
    ]
    for title, buckets in groups:
        lines.extend([f"## {title}", ""])
        for label, bucket in buckets:
            lines.extend([f"### {label}（{len(bucket)}）", ""])
            lines.extend(f"- {link(e)}" for e in bucket)
            lines.append("")
    return "\n".join(lines)


def products(catalog):
    return {"catalog.json": json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            "CATALOG.md": render_catalog(catalog), "FACETS.md": render_facets(catalog)}


def check(catalog):
    for name, expected in products(catalog).items():
        if not (ROOT / name).is_file() or (ROOT / name).read_text(encoding="utf-8") != expected:
            raise ValueError(f"Stale index: {name}; run `render` and apply its patch")
    provenance = read(ROOT / "IMPORT.json")
    for name, expected in provenance["source_metadata_sha256"].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"Pinned import metadata changed: {name}")
    checksums = [line.split(maxsplit=1) for line in (SOURCE / "CHECKSUMS.sha256").read_text().splitlines()]
    for digest, name in checksums:
        if sha(SOURCE / name) != digest:
            raise ValueError(f"Imported bytes changed: {name}")
    expected_names = {name for _, name in checksums} | {"CHECKSUMS.sha256"}
    ignored = {".DS_Store", "__pycache__", ".ruff_cache"}
    actual_names = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*")
                    if p.is_file() and not ignored.intersection(p.relative_to(SOURCE).parts)}
    if actual_names != expected_names or any(p.is_symlink() for p in SOURCE.rglob("*")):
        raise ValueError("Imported file inventory differs from the pinned snapshot")
    if (len(expected_names) != provenance["source_file_count"]
            or sum((SOURCE / name).stat().st_size for name in expected_names) != provenance["source_bytes"]
            or len(catalog["operators"]) != provenance["operator_count"]):
        raise ValueError("Import counts differ from the pinned inventory")
    for entry in catalog["operators"]:
        for target in entry["paths"].values():
            if not (ROOT / target).exists():
                raise ValueError(f"Missing catalog target: {target}")
    for doc in ROOT.rglob("*.md"):
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if target and not re.match(r"^[a-zA-Z]+:", target) and not (doc.parent / target).exists():
                raise ValueError(f"Broken Markdown link: {doc.relative_to(ROOT)} -> {target}")
    print(json.dumps({"status": "passed", "operators": len(catalog["operators"]),
                      "source_files": len(expected_names), "derived_indexes": 3,
                      "gpu_evaluation_rerun": False}, ensure_ascii=False))


def emit_patch(catalog):
    chunks = ["*** Begin Patch"]
    for name, content in products(catalog).items():
        target = ROOT / name
        if target.exists():
            old = target.read_text(encoding="utf-8")
            if old == content:
                continue
            chunks.extend([f"*** Update File: {target}", "@@"])
            chunks.extend("-" + line for line in old.splitlines())
        else:
            chunks.append(f"*** Add File: {target}")
        chunks.extend("+" + line for line in content.splitlines())
    chunks.append("*** End Patch")
    print("\n".join(chunks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="筛选算子；不同条件取交集，关键词不区分大小写")
    listing.add_argument("query", nargs="*", help="ID、中文说明、原合同描述、来源标签中的关键词（AND）")
    listing.add_argument("--family", choices=CATEGORIES)
    listing.add_argument("--subcategory", choices=SUBCATEGORIES)
    listing.add_argument("--precision", type=str.lower, choices=PRECISIONS)
    listing.add_argument("--state", type=str.lower, choices=["bf16", "fp32"])
    listing.add_argument("--stage", type=str.lower, choices=STAGES)
    listing.add_argument("--origin", type=str.upper, choices=["AKA", "REFERENCE"])
    listing.add_argument("--status", choices=STATUS)
    listing.add_argument("--audit", choices=AUDIT)
    listing.add_argument("--metric", choices=["ncu_kernel_sum", "cuda_event_forward"])
    listing.add_argument("--comparator", choices=["production", "historical_candidate"])
    listing.add_argument("--needs-receipt", action="store_true")
    listing.add_argument("--waiver", action="store_true")
    listing.add_argument("--json", action="store_true")
    showing = commands.add_parser("show", help="按稳定算子 ID 查看接口、路径和验证状态")
    showing.add_argument("id")
    showing.add_argument("--json", action="store_true")
    commands.add_parser("check", help="检查索引、链接、导入清单和全部原始哈希")
    commands.add_parser("render", help="输出派生索引的 apply_patch 补丁，不修改文件")
    args = parser.parse_args()
    catalog = build()
    entries = catalog["operators"]
    if args.command == "check":
        check(catalog)
    elif args.command == "render":
        emit_patch(catalog)
    elif args.command == "show":
        entry = next((e for e in entries if e["id"] == args.id), None)
        if entry is None:
            parser.error(f"Unknown operator: {args.id}; use `list` to search")
        if args.json:
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        else:
            print(f"{entry['id']}\n{entry['summary']}")
            print(f"来源: {entry['solution_origin']} | 验收: {STATUS[entry['acceptance_status']]} | 题目: {AUDIT[entry['problem_verification']]}")
            print(f"测量: {entry['metric']} | 对照: {entry['comparison']['kind']}")
            print("导入元数据摘录：" + entry["source_description"])
            for key, target in entry["paths"].items():
                print(f"{key}: {ROOT / target}")
            print("interface: " + json.dumps(entry["interface"], ensure_ascii=False))
    else:
        filters = {"category": args.family, "subcategory": args.subcategory,
                   "state_dtype": args.state, "solution_origin": args.origin,
                   "acceptance_status": args.status, "problem_verification": args.audit, "metric": args.metric}
        selected = []
        for entry in entries:
            if any(value is not None and entry[key] != value for key, value in filters.items()):
                continue
            if args.precision and args.precision not in entry["precision_tags"]:
                continue
            if args.stage and args.stage not in entry["stages"]:
                continue
            if args.comparator and args.comparator != entry["comparison"]["kind"]:
                continue
            if args.needs_receipt and not entry["requires_receipt"]:
                continue
            if args.waiver and entry["acceptance_status"] != "accepted_by_user_with_performance_waiver":
                continue
            haystack = json.dumps({k: v for k, v in entry.items() if k not in {"paths", "interface", "dependencies"}}, ensure_ascii=False).casefold()
            if not all(term.casefold() in haystack for term in args.query):
                continue
            selected.append(entry)
        if args.json:
            print(json.dumps(selected, ensure_ascii=False, indent=2))
        else:
            print(f"{len(selected)} operator(s)")
            for e in selected:
                flags = "; 待回执" if e["requires_receipt"] else ""
                print(f"{e['id']}\t{e['solution_origin']}\t{STATUS[e['acceptance_status']]}{flags}\t{e['summary']}")
                print(f"  {ROOT / e['paths']['readme']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"catalog: {error}", file=sys.stderr)
        sys.exit(1)
