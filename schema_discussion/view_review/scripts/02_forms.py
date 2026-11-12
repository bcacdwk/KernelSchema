#!/usr/bin/env python3
"""Build this investigation's evidence and finite checks, not a schema validator.

Only the six frozen roots and four named branches are structurally scanned.
No GPU work, source mutation, or full-tree re-extraction is performed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[4]
BASE = "schema_discussion/phase_grammar/view_review"
ARCHIVE = "schema_extract/archive/20261109-ten-operator-text-audit"
Y = f"{ARCHIVE}/inputs"
S = f"{ARCHIVE}/work"
PATHS = {
    "rule": f"{BASE}/rules/5.design.tile_pipeline.md",
    "mathrule": f"{BASE}/rules/3.design.math_reshape.md",
    "gdn": f"{Y}/extract_gdn/root.yaml",
    "gemm": f"{Y}/extract_gemm/root.yaml",
    "mla": f"{Y}/extract_mla/root.yaml",
    "sparse": f"{Y}/extract_sparse/root.yaml",
    "fmha": f"{Y}/extract_fmha/root.yaml",
    "moe": f"{Y}/extract_moe/root.yaml",
    "direct": f"{Y}/extract_moe/direct.yaml",
    "mma_sync": f"{Y}/extract_gemm/mma_sync.yaml",
    "tc_packed": f"{Y}/extract_sparse/tc_packed.yaml",
    "xpose": f"{Y}/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml",
    "gdncu": f"{S}/extract_gdn/gdn_kernel.cu",
    "gdnh": f"{S}/extract_gdn/gdn_common.cuh",
    "gemmcu": f"{S}/extract_gemm/fp8gg_umma.cu",
    "mmacu": f"{S}/extract_gemm/fp8gg.cu",
    "mlacu": "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py",
    "sparsecu": f"{S}/extract_sparse/dsv4_kernel.cu",
    "fmhacu": f"{S}/extract_fmha/source/candidate/fmha_tcg.cuh",
    "moecu": f"{S}/extract_moe/source_nvfp4_moe.cu",
}


def evidence(key: str, start: int, end: int) -> dict:
    path = PATHS[key]
    lines = (ROOT / path).read_text().splitlines()
    assert 1 <= start <= end <= len(lines)
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "quote": "\n".join(lines[start - 1 : end]),
    }


def evs(*ranges: tuple[str, int, int]) -> list[dict]:
    return [evidence(*r) for r in ranges]


SELECTED = {
    "extract_gdn": [],
    "extract_gemm": ["mma_sync"],
    "extract_mla": [],
    "extract_sparse": ["tc_packed"],
    "extract_fmha": ["tcg_d128_flat_tma_xpose_narrow"],
    "extract_moe": ["direct"],
}

# These are representative actually read ranges, not claims of full-file review.
READ_RANGES = {
    "gdn": [[433, 675], [685, 690]],
    "gemm": [[147, 180], [252, 329], [398, 489]],
    "mla": [[165, 223], [360, 433], [466, 621]],
    "sparse": [[168, 260], [295, 318], [375, 510]],
    "fmha": [[169, 196], [451, 534], [545, 582], [594, 648], [680, 702]],
    "moe": [[256, 299], [373, 395], [625, 686], [725, 760], [840, 875]],
    "direct": [[160, 211], [276, 390]],
    "mma_sync": [[135, 234]],
    "tc_packed": [[225, 421]],
    "xpose": [[486, 573], [600, 654]],
    "gdncu": [[1893, 1937], [2490, 2515], [2535, 2557], [2810, 2846]],
    "gdnh": [[27, 76]],
    "gemmcu": [[311, 349], [583, 602], [632, 668], [738, 768]],
    "mmacu": [[23, 43], [174, 215], [255, 290]],
    "mlacu": [[440, 490], [807, 953], [962, 1035], [1070, 1110], [1210, 1313], [1333, 1379], [1390, 1430]],
    "sparsecu": [[865, 905], [974, 1000], [1150, 1178], [1415, 1470], [1937, 2000], [2198, 2237], [2488, 2510], [3530, 3588], [4460, 4475]],
    "fmhacu": [[1815, 1975], [2987, 3008], [3270, 3325], [3388, 3450], [4886, 4923], [5115, 5185], [5380, 5430]],
    "moecu": [[1920, 1980], [2210, 2245], [4344, 4372], [4480, 4536], [4660, 4722], [4904, 4926], [5045, 5070]],
}


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def inventory() -> dict:
    inputs = json.loads((ROOT / BASE / "inputs.json").read_text())
    corpus_path = inputs["parent_corpus"]["path"]
    assert digest(corpus_path) == inputs["parent_corpus"]["sha256"]
    corpus = json.loads((ROOT / corpus_path).read_text())
    rules = []
    for r in inputs["rules"]:
        assert digest(r["snapshot"]) == r["sha256"]
        rules.append({"path": r["snapshot"], "sha256": r["sha256"], "matches": True})
    nodes = []
    for op, branches in SELECTED.items():
        for n in corpus["operators"][op]["nodes"]:
            if n["node"] not in ["root", *branches]:
                continue
            assert digest(n["snapshot"]) == n["sha256"]
            d = yaml.safe_load((ROOT / n["snapshot"]).read_text())
            regions = [
                r
                for ph in d["design"]["tile_pipeline"]["phase"].values()
                for r in ph.get("resource", {}).values()
            ]
            views = [v for r in regions for v in (r.get("view") or {}).values()]
            nodes.append({
                "operator": op,
                "node": n["node"],
                "scope": "root" if n["node"] == "root" else "branch",
                "category": n["category"],
                "selected_basis": n.get("selected_basis"),
                "path": n["snapshot"],
                "sha256": n["sha256"],
                "hash_matches": True,
                "structural_scan": {
                    "storage_entries": len(regions),
                    "view_entries": len(views),
                    "slot_index_fields": sum("slot_index" in v for v in views),
                    "part_fields": sum("part" in v for v in views),
                    "view_dtype_fields": sum("dtype" in v for v in views),
                },
            })
    sources = []
    pattern = re.compile(r"__shared__|ring|staging|dtype|float8|BFloat16|bfloat16|tmem|smem")
    reverse = {p: k for k, p in PATHS.items()}
    for op, o in corpus["operators"].items():
        for s in o["sources"]:
            assert digest(s["path"]) == s["sha256"]
            lines = (ROOT / s["path"]).read_text().splitlines()
            key = reverse.get(s["path"])
            sources.append({
                "operator": op,
                "path": s["path"],
                "sha256": s["sha256"],
                "hash_matches": True,
                "matches_source_lock_in_corpus": s["matches_source_lock"],
                "keyword_scan_line_count": len(lines),
                "keyword_hits": sum(bool(pattern.search(line)) for line in lines),
                "substantive_read_ranges": READ_RANGES.get(key, []),
            })
    assert len(nodes) == 10 and len(sources) == 18
    return {
        "current_rule_inputs": f"{BASE}/inputs.json",
        "rule_hash_checks": rules,
        "corpus": {"path": corpus_path, "sha256": digest(corpus_path), "hash_matches": True},
        "scope": "六 root 与四个 approved 分支；仅这十稿全量结构扫描，相关数学/存储/动作/源码段落实读。十八源码校验 hash 并做关键词扫描，其中八份实读相关计算/布局段。未重审 172 稿，未读取另一位新代理报告，未运行 GPU。",
        "status_limit": "approved 是原 working checkpoint 状态，不是新规则迁移通过。review_candidate、needs_authoring、未开始节点不在本次分支实读范围。MLA MODE1 的 FP16 路径只是同一冻结源码内的旁证，不冒充 root 的 MODE0 设计范围。",
        "nodes": nodes,
        "sources": sources,
        "representative_schema_read_ranges": [{"path": PATHS[k], "ranges": ranges} for k, ranges in READ_RANGES.items() if k in ["gdn", "gemm", "mla", "sparse", "fmha", "moe", "direct", "mma_sync", "tc_packed", "xpose"]],
        "navigation_only": ["schema_discussion/phase_grammar/reports/01_storage.md"],
    }


def finite_checks() -> dict:
    # Same byte cardinality does not identify the same TMEM row/column region.
    rowwise = {r * 512 + b for r in range(128) for b in range(128, 256)}
    flat = set(range(16384, 32768))
    assert len(rowwise) == len(flat) == 16384 and rowwise != flat
    # Counterexample to resetting a persistent ring at each item.
    lengths = [1, 2, 1]
    continuous = []
    local = []
    n = 0
    for w, length in enumerate(lengths):
        for t in range(length):
            continuous.append([w, t, n, n % 3])
            local.append(t % 3)
            n += 1
    assert [r[-1] for r in continuous] != local
    # GEMM's two terminal empty groups are actions, not two extra data tiles.
    kb = 4
    by_old_identity = [(t, p) for t in range(2) for p in range(kb + 2)]
    collapsed = [t * kb + p for t, p in by_old_identity]
    collisions = {str(n): [list(e) for e, k in zip(by_old_identity, collapsed) if k == n] for n in set(collapsed) if collapsed.count(n) > 1}
    assert len(by_old_identity) == 12 and len(set(collapsed)) == 10
    # These two configurations occur in the frozen root's allowed cfg list.
    cfg2 = {"HB": 64, "KC": 64, "Stages": 2, "D": 512}
    cfg4 = {"HB": 32, "KC": 64, "Stages": 3, "D": 512}
    for cfg in [cfg2, cfg4]:
        cfg["gather_slot_bytes"] = 2 * cfg["KC"] * cfg["D"]
        cfg["arena_bytes"] = cfg["Stages"] * cfg["gather_slot_bytes"]
        cfg["query_dense_bytes"] = 2 * cfg["HB"] * cfg["D"]
        cfg["partial_full_stride_capacity_bytes"] = 4 * cfg["HB"] * cfg["D"]
        assert cfg["partial_full_stride_capacity_bytes"] <= cfg["arena_bytes"]
    assert cfg2["partial_full_stride_capacity_bytes"] > cfg2["gather_slot_bytes"]
    # Partial overlap is not an equivalence relation.
    spans = {"Gather(0)": [0, 65536], "OutputStage": [0, 66560], "Gather(1)": [65536, 131072]}
    def intersects(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])
    assert intersects(spans["Gather(0)"], spans["OutputStage"])
    assert intersects(spans["OutputStage"], spans["Gather(1)"])
    assert not intersects(spans["Gather(0)"], spans["Gather(1)"])
    # Source layout budgets, not a proof of instruction address legality.
    budgets = {
        "mla_score_slot_fp32": 128 * 128 * 4,
        "mla_three_slots": 3 * 128 * 128 * 4,
        "mla_P_mode0_fp8_dense": 128 * 128,
        "mla_P_mode1_fp16_dense_source_only": 128 * 128 * 2,
        "gdn_Ti_logical_dense_bytes": 64 * 64 * 2,
        "gdn_Ti_materialized_tile_bytes": 10 * 16 * 16 * 2,
        "mma_sync_input_slot": (64 + 128) * 128 + 4 * 64,
        "mma_sync_output_padded_capacity": 64 * (128 * 2 + 16),
        "tc_packed_hash_plus_max_indices": 8192 * 8 + 2560 * 4,
        "tc_packed_ring": 2 * 64 * 512 * 2,
        "fmha_split_operand_arena": 5 * 128 * 128,
        "fmha_split_partial_padded_capacity": 128 * (128 * 4 + 16),
        "moe_fc1_used_tmem_columns": 128 + 2 * 32,
        "moe_fc1_reserved_tmem_columns": 256,
    }
    assert budgets["gdn_Ti_materialized_tile_bytes"] == 5120
    assert budgets["mma_sync_output_padded_capacity"] <= budgets["mma_sync_input_slot"]
    assert budgets["fmha_split_partial_padded_capacity"] <= budgets["fmha_split_operand_arena"]
    return {
        "kind": "有限算术、集合与实例身份反例；不是完整事件图、PTX 布局或新语法验证",
        "budgets": budgets,
        "sparse_root_configurations": {"cfg2": cfg2, "cfg4": cfg4},
        "rowwise_vs_flat_same_bytes": {"each_bytes": len(flat), "intersection_bytes": len(rowwise & flat), "different": True},
        "persistent_vs_reset": {"work_lengths": lengths, "continuous_rows_w_t_n_slot": continuous, "reset_local_slots": local},
        "terminal_identity_collision_if_real_tile_flatten_is_reused": {"KB": kb, "old_count": len(by_old_identity), "incorrect_distinct_count": len(set(collapsed)), "collisions": collisions},
        "partial_overlap_is_not_transitive": spans,
        "source_and_scope": "cfg2/cfg4 来自 Sparse root 的 P.cfg 范围及 kCfgs；MLA N=[1,2,1] 与 GEMM KB=4 是语义小模型，不声称对应某条实测 trace。所有预算与源码绑定证据见 F01/F02/F03/F05/F06。",
    }


def normalized_examples() -> list[dict]:
    # Five unchanged current header fields count as five assignments. Each
    # value binding and each per-value attribute counts once; axes/endpoints
    # remain one semantic field, not a character/indentation count.
    cases = [
        ("single_gemm_F", [], [], [], "无 view，5→5；原规则已满足。没有减少任何独立设计事实。"),
        ("moe_Exchange", [{}, {"part": "[0,K)"}], [{}, {}], ["FinalRoutes.part"], "8→7；同址关系和两个用途保留，只把不影响存活关系的起始/终止位置交给实现。"),
        ("mla_ScoreProbability", [{"slot_index": "n%3"}, {"slot_index": "n%3", "part": "column[32,64)", "dtype": "fp8_e4m3fn"}], [{"slot_index": "n%3"}, {"slot_index": "n%3", "part": "column[32,64)"}], ["P.dtype"], "保守主草图11→10，只减少已定类型复写。进一步省P.part可到9，但必须先证明跨warp/每行的S读取退休与tiny的关系；不能仅凭本warp先读后写认证，若需新增全量等待则不采用这项省略。两项槽映射保留，不新增父继承规则。"),
        ("sparse_ring", [{"part": "Gather slot interval", "dtype": "bf16"}, {"part": "Q prefix", "dtype": "bf16"}, {"part": "Partial prefix", "dtype": "fp32"}], [{"part": "Gather slot interval"}, {}, {}], ["Gather.dtype", "Q_ring.dtype", "PartialStage.dtype", "Q_ring.part", "PartialStage.part"], "14→9；保留逐 Gather 的区域轮转，Q/Partial 格式由已有 prepare/finish 的 Q/Acc 绑定确定。数据绑定和已知 materialization extent 仍需核对，不能把它们藏入无定义的标签。"),
        ("gemm_mma_sync_InputRing", [{"slot_index": "p%3"}, {"slot_index": "p%3", "part": "A"}, {"slot_index": "p%3", "part": "B"}, {"slot_index": "p%3", "part": "AS", "dtype": "fp32"}, {"slot_index": 0, "dtype": "bf16"}], [{"slot_index": "p%3"}, {"slot_index": "p%3", "part": "A"}, {"slot_index": "p%3", "part": "B"}, {"slot_index": "p%3", "part": "AS"}, {"slot_index": 0}], ["AS.dtype if source binding resolved", "Output.dtype if output binding resolved"], "20→18 是已回查 fill/convert 绑定后的手工草图，未实现自动解析；不同映射、三个同时存活子区与终端空组全部保留。减少两处类型复写，独立槽/共享/布局事实未减少。"),
        ("fmha_root_TensorMemory", [{"part": "ScoreHalf"}, {"part": "QueryScale", "dtype": "uint8"}, {"part": "KeyScale", "dtype": "uint8"}, {"part": "UnitScale", "dtype": "uint8"}, {"part": "ValueScale", "dtype": "uint8"}, {"part": "NumState"}, {"part": "Prob8", "dtype": "fp8_e4m3fn"}], [{"part": "ScoreHalf"}, {"part": "QueryScale"}, {"part": "KeyScale"}, {"part": "UnitScale"}, {"part": "ValueScale"}, {"part": "NumState"}, {"part": "Prob8"}], ["Prob8.dtype", "four scale view dtype fields"], "24→19 保留原父区域时删除五处视图格式声明；Prob8 在上游定义，scale 的 E8M0 atom 表示在既有 realization 绑定，事实并未丢掉。七个不重叠子区也可按现行规则分列，此时固定容量字段升至35，取消七项位置选择；不是减少总填写字段。"),
    ]
    out = []
    for name, before, after, removed, comment in cases:
        def count(views):
            return 5 + len(views) + sum(len(v) for v in views)
        out.append({"id": name, "current_header_assignments": 5, "current_view_attributes": before, "proposed_view_attributes": after, "current_count": count(before), "proposed_count": count(after), "removed": removed, "interpretation": comment})
    return out


def build() -> dict:
    findings = [
        {
            "id": "F01",
            "claim": "上游数值 dtype、父容量单位、源码打包载体是三件事。view 可只定义占用区域而完全删除 dtype；父 fp32/uint8 仍解释容量轴。不能由逻辑 shape*dtype 普遍推导实际物化字节。",
            "recommendation": "view 不再定义类型，也不继承父类型。数值格式沿用值/转换产物；本地 HashTable 或硬件 atom 格式在创建动作 describe 或既有 realization 唯一定义。父 dtype 保留为容量坐标单位。",
            "evidence": evs(("mathrule", 81, 88), ("mathrule", 202, 208), ("rule", 374, 388), ("mla", 199, 213), ("mlacu", 464, 470), ("mlacu", 997, 1000), ("mlacu", 1029, 1032), ("gdn", 264, 265), ("gdn", 580, 585), ("gdnh", 50, 65), ("tc_packed", 363, 369), ("fmha", 680, 684), ("gdn", 688, 690)),
            "limits": "MODE1 FP16 仅源内旁证。uint32 打包寄存器不自动把逻辑 P 变成 uint32。Ti 的5120字节打包须有 retained tiles/format 绑定，不能由删 dtype 自动获得，也不能把 dense 8192 当真占用。",
        },
        {
            "id": "F02",
            "claim": "多个真实复用只是同一容量包络的先后独占；较小用途没有必要指定首字节。现行『只占一部分就必须 part』比这些设计约束更细。",
            "recommendation": "省略 part 表示整个槽是该值的保守复用/冲突包络，值仅须能容纳且不必填满。整区交接无需额外 reuse/epoch 语法。仅当并存或局部读者释放依赖某个子区时才精化 part。",
            "evidence": evs(("rule", 368, 372), ("direct", 291, 306), ("direct", 372, 375), ("moecu", 1963, 1976), ("moecu", 2218, 2237), ("mla", 479, 491), ("mlacu", 1080, 1099), ("sparse", 385, 393), ("sparse", 453, 460), ("sparse", 489, 498), ("sparsecu", 1170, 1173), ("sparsecu", 1432, 1435)),
            "limits": "包络不是实际写集合，不能因没有 part 就把较小值解释成铺满整槽，也不能把未用字节自动借给并发值。源绑定仍可在 realization 记录精确偏移。MLA跨warp/每行退休未完成证明，主草图保留P.part；本条的可直接省略实例以Exchange和Sparse已有CTA join为依据。",
        },
        {
            "id": "F03",
            "claim": "同一父公式只对共同版本时钟有效；单槽常量、工作内索引和跨工作连续索引不能靠重命名统一。局部 view 形参的 alpha-renaming 不改变事件，改动作签名/flatten 域则可能改变身份和 terminal。",
            "recommendation": "保留逐值 slot_index，不设按名称/参数位置的隐式模运算，也不为少数重复公式新增父模板和覆盖继承。可以机械重命名局部形参，但必须保持其绑定函数；不改事件索引来省 view 字段。",
            "evidence": evs(("rule", 330, 334), ("rule", 364, 372), ("rule", 611, 621), ("gdn", 575, 586), ("mma_sync", 163, 168), ("mma_sync", 189, 205), ("mma_sync", 223, 226), ("moe", 651, 676), ("moecu", 4491, 4514), ("mlacu", 1259, 1277), ("mlacu", 1367, 1374)),
            "limits": "小模型证明特定错误归一化会碰撞，未证明所有合法重参数化都无效；若以后引入显式单参模板，必须另算形参/覆盖和绑定成本。",
        },
        {
            "id": "F04",
            "claim": "是否不重叠是流水线事实，绝对起点常是实现选择。FMHA root 的前一轮 PV 与当前 QK 并存，不能把全部 TMEM 视图改成整区轮流使用；独立子区可分列，但并不总更短。",
            "recommendation": "保留不相交容量与读取完成约束；纯独立区域可用已有平级 storage 声明让 Coder 排列。存在局部重叠或跨子区后用时，仍以同一父区域的 part 表达必要拓扑。",
            "evidence": evs(("rule", 340, 356), ("fmha", 495, 507), ("fmha", 594, 613), ("fmhacu", 2991, 3005), ("fmhacu", 4888, 4908), ("gdn", 651, 659), ("gdncu", 2502, 2511), ("gdncu", 2822, 2839), ("mla", 479, 491), ("mlacu", 852, 858), ("mlacu", 1080, 1099)),
            "limits": "对固定源码，已有 part 是事实记录；晚定起点不等于证明任意地址合法。GDN 最末 mini-chunk 的读退休缺口不会因字段省略消失。MLA的首32列tiny读与[32,64)的P列本来不相交；本地先读后写不等于全部warp先读后写，且tb=tb0的既有row-address问题尚未解决，主草图不省该part。",
        },
        {
            "id": "F05",
            "claim": "逐值 flat storage + shared_with 只在简单两值整区复用显得浅；复杂 arena 仍需 backing 身份、容量一次计账、逐值区域和引用作用域。重叠关系本身不是等价关系。",
            "recommendation": "不采用全局 flat+alias 新体系，不用每个属性各一张并行表。一个容量条目和其中逐值记录，保留了关联局部性；简单单值不写 view。",
            "evidence": evs(("rule", 354, 358), ("rule", 405, 407), ("tc_packed", 246, 267), ("tc_packed", 381, 394), ("sparsecu", 1946, 1951), ("sparsecu", 2210, 2235), ("sparsecu", 3530, 3545), ("xpose", 503, 512)),
            "limits": "这是对同例所需事实的表达成本比较，不是证明所有可能 flat 语言都不可行。增加 backing 表能表达，但只是把 view 的父子关系转成引用，没有删掉事实。",
        },
        {
            "id": "F06",
            "claim": "晚定偏移仍能核对容量和复用，但须保留容量上限、已知物化足迹、并存关系、异步写和最后读者。max(单值 bytes) 及数学值 shape*dtype 都不足以给出普遍物理账。",
            "recommendation": "按父容量计一次；整区交接核对每个活动集合能放下，分区核对边界/不相交；released 仍覆盖必要读者且覆盖前无未完成旧写。额外 padding/分配粒度在既有容量或 realization 记录，不造假值。",
            "evidence": evs(("sparsecu", 878, 885), ("sparsecu", 4463, 4472), ("mma_sync", 201, 208), ("mmacu", 33, 38), ("mmacu", 263, 280), ("xpose", 604, 612), ("fmhacu", 5394, 5424), ("moe", 669, 676), ("moecu", 4349, 4367)),
            "limits": "容量算术是必要条件。未运行 GPU，未证明所有 shape 域、线程地址、swizzle、对齐和原语完成保证。FMHA empty split 的 Query 写者退休问题单列保留，不能把『空』当作无存储事件。",
        },
        {
            "id": "F07",
            "claim": "同一数学值在不同层级的副本，以及跨动作保留的寄存器双缓冲，不能当重复声明删除。它们支撑提前释放和重叠，view 本身不承担事件同步。",
            "recommendation": "保留独立 storage 身份及每次访问绑定；单值缓冲按现行同名缺省省 view。不要为格式推导省掉副本，也不把普通指令寄存器载体逐一登记。",
            "evidence": evs(("rule", 177, 186), ("mla", 387, 424), ("gemm", 315, 329), ("gemm", 426, 446), ("gemmcu", 592, 601), ("gemmcu", 654, 665), ("fmha", 548, 555), ("fmhacu", 3278, 3299), ("fmhacu", 3316, 3320)),
            "limits": "八份相关源码的局部实读支撑这些身份/生存期判断；不声称编译器实际寄存器峰值等于所有逻辑值量之和。",
        },
    ]
    proposals = [
        {
            "id": "A",
            "name": "同一 storage 容量头 + 可选精简 view；省略 part 为整槽保守复用包络",
            "selected": True,
            "claim": "保持一个结构，删除值类型复写和无关定位。whole reuse 是 view 的缺省，不新增第二种任选表达。",
            "recommendation": "slot_index 在 depth>1 时仍逐值明确；part 仅在空间共存/局部复用或源绑定要求时精化；彻底删除 view.dtype，格式在既有值/动作/实现绑定中唯一说明；父 dtype 继续解释 shape 的容量单位。",
            "format_authority": [
                "view 只命名占用区域，没有数值解释字段，也不存在从父 dtype 继承的类型。数学值/工作 tile 或转换产物沿用其既有表示。",
                "本地对象或硬件 packing 没有上游数值表示时，在创建它的现有动作 describe 或对应 realization 绑定处定义一次；多个消费者引用同一对象/绑定，不分头规定不同格式。HashTable 的8192个uint64原子字已有 build 契约，E8M0 atoms 已有 realization，故这些例子无需补另一张类型表。",
                "实际物化子集、压缩打包和 layout carrier 不由 dtype 唯一决定。所依赖的足迹/容量与实现绑定必须留下。若某稿的唯一格式事实仅在旧 view.dtype 中，迁移必须移到对应现有绑定；这算移动事实，不算消除设计决定。",
            ],
            "part_omission_contract": [
                "省略只把整槽视为潜在冲突范围，表示这一用途取得该槽的复用生命周期；较小逻辑值无须填满。",
                "所访问/保留内容须在包络内；两个需要并存的值不能因未写 part 自动推断不重叠。",
                "源写与旧值必要读取都退休后才能覆盖；包络不生成依赖。精确源偏移可保留在 realization，若设计需要局部读者释放则继续写 part。",
            ],
            "same_case_trials": normalized_examples(),
            "finite_checks": finite_checks(),
            "evidence": evs(("rule", 340, 358), ("rule", 368, 388), ("direct", 372, 375), ("sparse", 489, 498), ("mla", 209, 213), ("tc_packed", 363, 369), ("fmha", 680, 684)),
            "unproven": "这是建议草图，不是现行语法，也不是十稿完整迁移。尚未实现格式绑定解析、全域容量求解、事件图展开、TMEM 地址合法性或源码修复。",
        },
        {
            "id": "A_type_fallback",
            "name": "与A对照：上游默认，未绑定时父类型默认/显式view.dtype覆盖",
            "selected": False,
            "claim": "也能表达同例，但同时维护上游类型、父容量类型和本地 override 的优先级，且不能说明 uint8 scale 的 E8M0 atom packing。",
            "recommendation": "优先选择A的区域职责单一化。保留 fallback 并未证明是核心信息的唯一载体。",
            "same_case_trials": {
                "ScoreProbability": "P 需上游 FP8 优先于父 FP32；与当前继承默认相反，要增加冲突/优先级规则。",
                "sparse_ring": "Q_ring/PartialStage 不与数学符号同名，需要额外决定由动作绑定推格式还是写 override。区域-only 方案直接要求既有访问绑定完整，不新增继承分支。",
                "tc_packed_HashTable": "uint64 看似必须本地 override，但 build 已明确8192-word uint64。删 view.dtype 不丢这项事实。",
                "fmha_scales": "view.uint8 只给载体宽度，不能表达 E8M0 atom；真正的格式契约本来就在 realization。",
            },
            "evidence": evs(("mla", 393, 394), ("sparse", 444, 455), ("sparse", 489, 493), ("tc_packed", 363, 369), ("fmha", 680, 684)),
        },
        {
            "id": "B",
            "name": "把 slot/type/part 提升到 storage 级",
            "selected": False,
            "claim": "单值可以浅一层；多值若属性相异就必须恢复逐值记录或引入并行关联表/默认覆盖机制。共同父公式只在一个共同版本时钟上省复写。",
            "recommendation": "不作为统一规范。MLA S/P 能合并 n%3；GDN FinalState=0、GEMM Output=0 与其余环转不同，Sparse 跨原槽的阶段覆盖也不是一条父轮转式。",
            "same_case_trials": {
                "single_F": "与A相同，无 view 可提升。",
                "Exchange": "values:[WarpCounts,FinalRoutes] 可表达，保存相同两用途/同址事实；只换了列表容器。",
                "ScoreProbability": "values:[S(n),P(n)] + slot_index:n%3 可少一项重复映射；需定义父公式如何对各值签名绑定，不能靠自由变量碰巧同名。这只聚合slot，P.part仍需逐值位置，不能把它无证据省略。",
                "sparse_ring": "Q/Partial 为整 arena，Gather 是环内子区；父公共 part 会改变含义，需逐值记录。",
                "mma_sync_InputRing": "p%3 与 Output(t) 的0不能共用唯一父式；增加例外后又有一套逐值属性/覆盖规则。",
                "fmha_TensorMemory": "类型与区域各异；分属性映射会把同一个值的事实拆开关联。",
            },
            "capacity_bytes_variant": "将 shape+dtype 改为单个 size_bytes 对全槽 SP 可再少一容量属性，但有 part 时失去 row/column 坐标及其单位；必须补 shape/stride/逐行 region，不能当作普遍删父 dtype 的胜利。",
            "evidence": evs(("gdn", 575, 586), ("mma_sync", 163, 168), ("sparse", 385, 393), ("rule", 388, 388)),
        },
        {
            "id": "C",
            "name": "逐值 flat storage + shared_with/backing",
            "selected": False,
            "claim": "整区两个值能表达；一般案例将一层嵌套换成 backing 引用、容量所有权、部分重叠和跨 phase 作用域规则。",
            "recommendation": "不引入两类 primary/alias 条目或第二张 backing 表。保持容量与其用途靠在一起。",
            "same_case_trials": {
                "single_F": "直接条目与现有相同，无收益。",
                "Exchange": "FinalRoutes.shared_with=WarpCounts 简短，但容量仅主条目计账、次条目继承 location/depth 的规则必须新增。",
                "ScoreProbability": "S_tmem/P_tmem 需要共享容量、相同三槽映射及与 S_reg/P_reg 的不同身份；两值 dtype 已来自上游，扁平并未进一步减少事实。",
                "sparse_ring": "Gather 具有多个可并存版本，Q/Partial 覆盖整个 arena 的不同份额；shared_with 指值、槽还是总 backing 必须分清。",
                "mma_sync_InputRing": "Stage/A/B/AS/Output 既有整组又有子区，same-address 等价类不足；若回补 part+backing，则已重建 view 关系。",
                "fmha_TensorMemory": "互不重叠时可直接用现行平级 storage；无须新增共享关系。",
            },
            "evidence": evs(("tc_packed", 246, 267), ("mla", 387, 412), ("direct", 188, 199), ("direct", 372, 375), ("mma_sync", 158, 168)),
        },
        {
            "id": "D",
            "name": "只有整区分时复用组，不再允许 part",
            "selected": False,
            "claim": "Exchange 与完整阶段切换很合适；有并存子区时必须新增分区/嵌套 arena，或错误地串行化本来允许的流水线。",
            "recommendation": "只吸收整区语义为A的缺省；局部例外仍用原有 part，不新增 epochs/modes/bundles 任意组合的第二语言。",
            "same_case_trials": {
                "single_F": "单个用途即可，原有表示已满足。",
                "Exchange": "直接成立；轮换不用位置。",
                "ScoreProbability": "只有在同槽S全部所需读取交接得到证明后才可用；当前tiny首32列与P列的分区不能自动删除，也不能混同另一个Accumulator或寄存器副本。",
                "sparse_ring": "整个阶段的切换成立，稳态多个 Gather 槽必须额外表达，整区组本身不够。",
                "mma_sync_InputRing": "输入与输出是两阶段，输入阶段仍有三个槽及 A/B/AS 子区。",
                "fmha_TensorMemory": "ScoreHalf/NumState/Prob8/scales 并存，纯轮流使用语义不成立。",
            },
            "evidence": evs(("sparse", 453, 488), ("fmha", 495, 507), ("fmha", 594, 613), ("xpose", 604, 612)),
        },
    ]
    return {"coverage": inventory(), "findings": findings, "proposals": proposals}


def verify_quotes(obj):
    count = 0
    if isinstance(obj, dict):
        if {"path", "start_line", "end_line", "quote"} <= obj.keys():
            actual = "\n".join((ROOT / obj["path"]).read_text().splitlines()[obj["start_line"] - 1 : obj["end_line"]])
            assert actual == obj["quote"], obj["path"]
            count += 1
        for value in obj.values():
            count += verify_quotes(value)
    elif isinstance(obj, list):
        count += sum(verify_quotes(value) for value in obj)
    return count


if __name__ == "__main__":
    result = build()
    quote_count = verify_quotes(result)
    out = ROOT / BASE / "data/02_forms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "selected_yaml": len(result["coverage"]["nodes"]), "sources_hashed": len(result["coverage"]["sources"]), "evidence_quotes_checked": quote_count, "same_case_counts": [[e["id"], e["current_count"], e["proposed_count"]] for e in result["proposals"][0]["same_case_trials"]], "finite_checks_passed": True}, ensure_ascii=False))
