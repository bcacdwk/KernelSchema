#!/usr/bin/env python3
"""Build this audit's evidence JSON; verify fixed input hashes and literal quotes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[3]
BASE = "schema_extract/archive/20261109-ten-operator-text-audit/"
STUDY = "schema_discussion/phase_grammar/"
RULE = STUDY + "sources/rules/"
corpus = json.loads((ROOT / STUDY / "corpus.json").read_text())


def inp(op, node="root"):
    return BASE + f"inputs/extract_{op}/{node}.yaml"


def ev(path, start, end, quote=None):
    raw = "\n".join((ROOT / path).read_text().splitlines()[start - 1:end])
    if quote is None:
        quote = raw.strip()
    assert quote and quote in raw, (path, start, end, quote)
    return {"path": path, "start_line": start, "end_line": end, "quote": quote}


GDN = BASE + "work/extract_gdn/gdn_kernel.cu"
GEMM = BASE + "work/extract_gemm/fp8gg_umma.cu"
SYNC = BASE + "work/extract_gemm/fp8gg.cu"
MLA = STUDY + "sources/kernel/extract_mla/solution.py"
SPARSE = BASE + "work/extract_sparse/dsv4_kernel.cu"
FMHA = BASE + "work/extract_fmha/source/candidate/fmha_tcg.cuh"
FMHA_HOST = BASE + "work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu"
MOE = BASE + "work/extract_moe/source_nvfp4_moe.cu"
D5 = RULE + "5.design.tile_pipeline.md"
D4 = RULE + "4.design.work_schedule.md"

source_ranges = {
    GDN: [[285,405],[2174,2228],[2310,2358],[2450,2555],[2625,2675],[2725,2877],[3248,3277]],
    GEMM: [[646,790],[887,931],[964,1065],[1705,1820]],
    SYNC: [[25,45],[253,302]],
    MLA: [[905,949],[1082,1107],[1252,1365],[1638,1692],[1884,1905],[2006,2029]],
    SPARSE: [[1110,1245],[1407,1460],[6128,6208],[6340,6406]],
    FMHA: [[1821,1877],[2989,3015],[4684,4782],[4888,4933],[5058,5134],[5166,5183],[5374,5435],[5515,5521]],
    FMHA_HOST: [[801,838],[1178,1205]],
    MOE: [[1698,1738],[1760,1815],[4418,4443],[4835,4925],[5130,5195],[7719,7787],[8323,8363],[9450,9508]],
}
branch_ranges = {
    ("gemm", "mma_sync"): [[130,245]],
    ("gemm", "wide_recovery_full"): [[145,255],[450,484]],
    ("sparse", "probe_hidden_packed"): [[285,450]],
    ("fmha", "tcg_d128_flat_tma_xpose_narrow"): [[559,650]],
    ("moe", "verify_only"): [[1,298]],
}
roots = []
source_hashes = []
for op, item in corpus["operators"].items():
    node = next(n for n in item["nodes"] if n["node"] == "root")
    actual = hashlib.sha256((ROOT / node["snapshot"]).read_bytes()).hexdigest()
    assert actual == node["sha256"]
    roots.append({"operator": op, "path": node["snapshot"], "status": node["category"],
                  "sha256": actual, "read_scope": "full_text", "line_count": len((ROOT / node["snapshot"]).read_text().splitlines())})
    for src in item["sources"]:
        actual = hashlib.sha256((ROOT / src["path"]).read_bytes()).hexdigest()
        assert actual == src["sha256"]
        source_hashes.append({"path": src["path"], "sha256": actual, "matches_corpus": True,
                              "corpus_matches_source_lock": src["matches_source_lock"],
                              "semantic_ranges_read": source_ranges.get(src["path"], []),
                              "read_scope": "selected_source_ranges" if src["path"] in source_ranges else "hash_check_only"})
branches = []
for (op, name), ranges in branch_ranges.items():
    node = next(n for n in corpus["operators"]["extract_" + op]["nodes"] if n["node"] == name)
    assert hashlib.sha256((ROOT / node["snapshot"]).read_bytes()).hexdigest() == node["sha256"]
    branches.append({"operator": "extract_" + op, "node": name, "path": node["snapshot"],
                     "status": node["category"], "ranges_read": ranges,
                     "read_scope": "full_text" if name == "verify_only" else "targeted_excerpts"})

old_ids = {
    "extract_gdn": ["F12", "F16"], "extract_gemm": ["F10", "F12"],
    "extract_mla": ["F09", "F11", "F18"], "extract_sparse": ["F05", "F07", "F09"],
    "extract_fmha": ["F14", "F19"], "extract_moe": ["F08", "F23", "F24", "F25"],
}
coverage = {
    "method": STUDY + "METHOD.md", "corpus": STUDY + "corpus.json",
    "roots": roots, "branches": branches,
    "corpus_state_counts": dict(Counter(n["category"] for op in corpus["operators"].values() for n in op["nodes"])),
    "semantic_material_total": 172, "scaffold_total": 5, "not_started_total": 43,
    "review_extent": "Six roots fully read, verify_only fully read, four other branches read at the listed counterexample excerpts. No claim of semantic review of all 172 authored nodes.",
    "source_hashes": source_hashes, "new_source_hashes": [],
    "rules_read": [
        {"path": D5, "scope": "full_text"}, {"path": D4, "scope": "full_text"},
        {"path": RULE + "0.common.md", "scope": "relevant naming, alias, inheritance and extension sections"},
        {"path": RULE + "3.design.math_reshape.md", "ranges": [[88,130]]},
        {"path": RULE + "6.realization.md", "ranges": [[1,115]]},
    ],
    "prior_work_reused": {
        "reports": [x["path"] for x in corpus["reports"]],
        "catalog": BASE + "data/findings_catalog.json",
        "annotations": [{"path": BASE + f"annotations/{op}.json", "selected_finding_ids": [op + "-" + f for f in ids],
                         "other_material": "root node_review and selected removal_candidates"} for op, ids in old_ids.items()],
        "operator_reports": [{"path": BASE + f"operators/{op}.md", "scope": "findings, mechanisms, limits; long all-node tables not repeated"} for op in old_ids],
        "mechanical_locator": STUDY + "data/root_fields.json",
        "length_statistics": "Root maximum verb-contract lengths are reused from SCHEMA_VERBOSITY.md; no new whole-corpus text census.",
    },
    "limits": [
        "Old field names and execution-mode spellings are treated as archived syntax, not current-schema defects.",
        "No issue-history census: absence of a filed issue is not used as evidence of sufficiency.",
        "Eight source files were examined semantically at selected ranges; all 18 corpus source files were hash checked only beyond those ranges.",
        "No GPU, performance experiment, resumed extraction, source fix, formal-doc edit or new approval.",
        "Sketches are partial proposals in current notation, not complete standalone plans or mechanically proven equivalent rewrites.",
    ],
}

findings = [
    {
        "id": "PG04-01", "operator": ["extract_gdn", "extract_sparse"], "category": "implementation_detail",
        "related_definitions": ["resource", "domain"],
        "claim": "精确线程/寄存器 lowering 可以删；GDN 两个扫描 warp 不等 Q/K、Sparse 的 Q 与 KV 就绪边界及整块读者退休必须保留，不能把整段 prepare 一并删掉。",
        "recommendation": "删线程 0..63、具体 float4 寄存器轮换及普通 barrier 编号；以短 describe 保留参与组、扫描与传输重叠、所有读者退休、版本/区域绑定。控制上的 Active 用 if，单 role 的头组遍历可用局部 for；资源视图所需的 chunk 索引继续登记。",
        "retained_facts": ["g/beta 写者发布后扫描组可开始，尚未等到 Q/K；Gram 前全 CTA 汇合", "Q_ring 先读完再让给 KV；prepare 只保证 Q 就绪及 KV 已发起", "最后状态仍从 stage 0 写出，保留下一片读取与当前片写出重叠的设计事实"],
        "removed_facts": ["普通指令操作数/临时寄存器编号与逐线程地址", "等价实现中的 lane 0、线程 0..63 选择细节", "描述未使用的末轮临时变量赋值"],
        "why_not_missing": "现行 docs5 已明确原语内部线程协作可留一句语义；真正独立发布边界仍可拆动作/role。这里没有必须新增公共定义的最小反例。",
        "prior_leads": ["SCHEMA_VERBOSITY.md: longest GDN/Sparse contracts", "extract_gdn-F10", "extract_sparse-F09"],
        "evidence": [
            ev(inp("gdn"),498,508), ev(inp("gdn"),661,669),
            ev(GDN,338,359), ev(GDN,384,394), ev(GDN,2855,2864),
            ev(inp("sparse"),444,460), ev(SPARSE,1150,1174), ev(SPARSE,1198,1206),
            ev(D5,20,20), ev(D5,328,328), ev(D5,674,674),
        ],
    },
    {
        "id": "PG04-02", "operator": "extract_gdn", "category": "source_or_draft_issue",
        "related_definitions": ["resource", "domain"],
        "claim": "最长 finish_chunk 的核心困难是源码 final-mini 的读者退休缺保证；别名区域和轮转槽本来已在 resource.view 中，不是缺一种资源字段。",
        "recommendation": "保留 full/mini 数学差异、RhsOutput 与 stage-0 别名、非末轮在下一 advance 退休及最后 mini 的具体限制；移除解释性反复。不得因合并成一个 serial 动作而声称所有 warp 已读完，也不在调查中替源码加屏障。",
        "minimal_counterexample": "当最终 mini chunk 的 (c_n-1)%NSTAGE=0 且请求 FinalState，快 warp 的 stage-0 写入可早于另一 warp 的 Kc row-0 读取；只改 view 或添加一个未落实的 release 名称不能提供保证。",
        "why_not_missing": "现有 view、release/waiting 足以描述有依据的安全复用；缺少的是该路径落实依赖的源码保证。FMHA xpose 已有类似显式 reuse 边与单独来源限制，说明应区分设计关系和实现偏差。",
        "prior_leads": ["extract_gdn-F12", "SCHEMA_VERBOSITY.md: finish_chunk and FMHA source-defect separation"],
        "evidence": [
            ev(inp("gdn"),638,660), ev(inp("gdn"),565,586),
            ev(GDN,2499,2512), ev(GDN,2543,2552), ev(GDN,2820,2839),
            ev(GDN,2320,2345), ev(GDN,2653,2660),
            ev(inp("fmha","tcg_d128_flat_tma_xpose_narrow"),604,612),
            ev(inp("fmha","tcg_d128_flat_tma_xpose_narrow"),642,649), ev(FMHA,5412,5424),
            ev(D5,405,405),
        ],
    },
    {
        "id": "PG04-03", "operator": "extract_gemm", "category": "concise_semantics",
        "related_definitions": ["resource", "domain"],
        "claim": "先归还 shared 源槽、后完成 global 目标写入，以及跨 tail 保留全部旧 store 的读者集合，可以用一句 release 契约和现有模式保留。",
        "recommendation": "按 noun 分开 DotPair 与 Staging 的 release 描述；Staging 只需声明等待全部先前 bulk store 结束读源并汇合写者。保留 async store 和 worker 末尾最终完成等待；不要新增通用 read_done 事件。",
        "minimal_counterexample": "full tile 后是 tail，再到 full：只等待上一 worktile 的 store 会漏掉更早 full tile 的在途读取；改等 store.done 则额外等目标落地，丢失允许重叠。",
        "retained_facts": ["all previously issued bulk source reads", "tail 不使用 Staging 且不清历史", "最终目的写完成", "mma_sync 的 Output 是 InputRing 第 0 槽的 BF16 view"],
        "why_not_missing": "read completion 与 destination completion 的分离已由 release/store 的基础语义明确；历史集合是原语保证的适用范围，宜短写而非强制把每个 queue bookkeeping 状态结构化。",
        "prior_leads": ["extract_gemm-F10", "extract_gemm-F12"],
        "evidence": [
            ev(inp("gemm"),439,457), ev(inp("gemm"),458,467),
            ev(GEMM,1015,1018), ev(GEMM,1033,1055),
            ev(inp("gemm","mma_sync"),158,168), ev(inp("gemm","mma_sync"),201,209),
            ev(SYNC,263,267), ev(SYNC,283,296),
            ev(D5,123,128), ev(D5,141,141),
        ],
    },
    {
        "id": "PG04-04", "operator": "extract_mla", "category": "concise_semantics",
        "related_definitions": ["resource", "domain"],
        "claim": "S/P 同时具有 TMEM view 和寄存器副本，动作必须指明实际副本；这类非结构化动作绑定合理，不能由 noun 同名推出，也不能据同名把两块资源合并。",
        "recommendation": "describe 只绑定源动作、版本、资源/view、目标与必要原语顺序。保留前 PV 读 P 先于后 QK 覆盖的保证；删已由 depth/slot_index/等待传递给出的安全证明。当前规则不再从同名动作自动生成数据边，迁移时补全真实前驱。",
        "minimal_counterexample": "仅保留 load:Scores(n) 而不绑定 ScoreProbability.S(n) 到 ScoreRegisters.S(n)，不能判定读的是哪份 S。仅靠 async_ordered 的完成先后也不能推出片内读先写后；强加 PV.done 到后 QK 会改变重叠。",
        "why_not_missing": "docs5 183–190 已把动作访问绑定明确交给 describe；405/228 也允许有依据的原语顺序保证，view 负责静态存储关系即可。",
        "prior_leads": ["extract_mla-F09", "extract_mla-F11"],
        "evidence": [
            ev(inp("mla"),387,412), ev(inp("mla"),479,491), ev(inp("mla"),565,574),
            ev(MLA,918,945), ev(MLA,1090,1100), ev(MLA,1273,1286), ev(MLA,1302,1330),
            ev(D5,172,184), ev(D5,228,228), ev(D5,479,486),
        ],
    },
    {
        "id": "PG04-05", "operator": "extract_fmha", "category": "existing_structure",
        "related_definitions": ["resource", "domain"],
        "claim": "旧稿把 PV.done 定成只发起、再在 retire/drain 描述完成，当前可以直接用 async matmul 与 await/waiting；无需新中间事件或第四类公共表。",
        "recommendation": "matmul 的 done 恢复为结果可用/本动作所需读者结束；previous/final await 在原位置等待。两种独立 QK/PV completion 不能因同 role 就合成一个 async_ordered 序列。z 只在单 role 的两个 score-half 遍历中使用，可移成局部 for(z=0:2)。",
        "minimal_counterexample": "在 PV 发起处把动作设为 serial 并真等完，将取消 PV(g-1) 与当前两半 QK 的重叠；保留 async_unordered 与中部 await 则能保留它。",
        "why_not_missing": "当前三种 mode、两事件和 released/prepared 边已经能表达；改变的是旧稿的粒度/完成契约，不是字段名字。",
        "prior_leads": ["SCHEMA_VERBOSITY.md: FMHA custom verbs", "extract_fmha-F13"],
        "evidence": [
            ev(inp("fmha"),594,621), ev(inp("fmha"),644,658), ev(inp("fmha"),462,464),
            ev(FMHA,4708,4737), ev(FMHA,4899,4921), ev(FMHA,5081,5094), ev(FMHA,5121,5128), ev(FMHA,5166,5178),
            ev(D5,119,128), ev(D5,202,208),
        ],
    },
    {
        "id": "PG04-06", "operator": "extract_fmha", "category": "other_layer",
        "related_definitions": ["resource"],
        "claim": "102912 字节起始 SMEM 请求及为 TMEM 占用调整的实际 reservation 会改变容量，不能和普通未使用实现字节一起一删了之；但不应为它制造假数学值或额外 resource。",
        "recommendation": "资源账保持设计中的 live allocation/view；realization 以一句绑定实际 launch 动态 SMEM 请求、初始 102912 字节及 256 TMEM columns 的容量原因。缓存成员、重试过程和 error clearing 可交 Coder/来源；如要结构化实际 reservation，应另研究 launch/realization 的预算接口。",
        "retained_facts": ["每 CTA 实际动态 SMEM 请求与 live-resource 字节和不同", "从 102912 字节开始的实际占用选择", "256 TMEM columns 约束", "helper API 失败时不能声称已获驻留保证"],
        "why_not_missing": "docs5 明说 total_size 不含实现额外开销；当前 realization 可以承载与容量相关的最小实现条件。这是预算归属边界，非三个公共定义缺少任意 unused-byte 对象。",
        "prior_leads": ["extract_fmha-F19"],
        "evidence": [
            ev(inp("fmha"),685,702), ev(inp("fmha"),466,494), ev(FMHA,1822,1852),
            ev(FMHA_HOST,817,838), ev(FMHA_HOST,1183,1195),
            ev(D5,350,354), ev(RULE + "6.realization.md",5,13),
        ],
    },
    {
        "id": "PG04-07", "operator": "extract_moe", "category": "existing_structure",
        "related_definitions": ["resource", "domain", "milestone"],
        "claim": "Hoist 改变 scale copy 前是否等待的事实已被当前 if/waiting 覆盖；四个 epilogue 是同一 CTA 内四个 role，不能用成员通配符或内容 alias 把它们变成一个完成者。",
        "recommendation": "用 if(t>0 and Hoist==0) 放置 Drain，保留所有四个 release 前驱及首个 matmul 的无条件复用边。删除 q 的 0/1 假循环及重复解释；不登记 barrier 编号/存储。跨 kernel 的相同字段 alias 只复用文本，资源仍是新 worker 的分配。",
        "minimal_counterexample": "把前驱简化成 Epilogue[*] 会只选择 worker 的 CTA 成员，不能选到 Epilogue0/1/2；把 Hoist=1 时也等待在 scale copy 前会消除源码允许的跨工作重叠。",
        "why_not_missing": "docs5 已有对应条件等待规则和 worker-member 语义。GEMM 的四次恢复提交用 alias 复用 phase 正文仍是四次完整计算，这不是少写字段就能删除的工作。",
        "prior_leads": ["SCHEMA_VERBOSITY.md: MoE conditional scale-copy wait", "extract_moe-F17", "extract_gemm-F04"],
        "evidence": [
            ev(inp("moe"),758,795), ev(inp("moe"),814,848), ev(inp("moe"),850,870),
            ev(MOE,4855,4867), ev(MOE,4891,4915), ev(MOE,4431,4435), ev(MOE,5143,5155),
            ev(inp("gemm","wide_recovery_full"),150,169), ev(inp("gemm","wide_recovery_full"),252,254), ev(GEMM,1757,1762), ev(GEMM,1802,1816),
            ev(D5,293,293), ev(D5,566,568), ev(D5,650,662),
        ],
    },
    {
        "id": "PG04-08", "operator": ["extract_mla", "extract_gdn"], "category": "other_layer",
        "related_definitions": ["resource", "milestone"],
        "claim": "host 元数据读回、建表、上传之后才能启动主 kernel，以及跨调用复用同一 global workspace，都是真关系，但不是 docs5 的片上资源/局部事件。",
        "recommendation": "MLA 在 docs4/调用前提保留 D2H→host 建 Geo→H2D→main 启动及 metadata 版本；GDN 用一句说明 workspace 共用、消费区每次重写、同流复用和无跨流排他。若想让这些依赖机器可见，另议 docs4/调用边界，不给 host upload 伪造 GPU phase 或局部 milestone。",
        "minimal_counterexample": "同样的一条 main launch，在 upload 前启动而在首次读表时才等待，与上传完成后才允许 kernel 启动有不同资源占用和可重叠关系；当前 phase milestone 不能命名 CPU/upload 的完成。",
        "why_not_missing": "这是跨层范围未定。当前语言允许 strategy 保留核心事实，未证明需要把完整分配器/缓存状态机纳入微流水线。retention invocation 也不等于进程分配身份。",
        "prior_leads": ["extract_mla-F18", "extract_gdn-F16", "SCHEMA_DISCUSSION.md sections 2 and 5"],
        "evidence": [
            ev(inp("mla"),321,338), ev(MLA,1669,1681), ev(MLA,1895,1898), ev(MLA,2007,2024),
            ev(inp("gdn"),379,383), ev(GDN,3254,3274),
            ev(D4,27,29), ev(D4,285,297), ev(D4,371,375), ev(D5,498,504),
        ],
    },
    {
        "id": "PG04-09", "operator": "extract_sparse", "category": "other_layer",
        "related_definitions": ["milestone", "resource"],
        "claim": "caller-prior fork 与两个输出生产者到 caller-next 的 join 不能由内部三条 launch 推出；它们应是 docs4/调用接口的边界关系，不能冒充 milestone 的局部 all-of。",
        "recommendation": "保留 prefix/remainder 独立推进、remainder 等 probe、caller 后续工作等两段输出且 host 不为输出阻塞。图缓存键、计数和失效代码退出微流水线；首轮 capture 的 caller-stream completion 边界仍需在正确层保留。",
        "minimal_counterexample": "仅余 probe、prefix、remainder<-probe 三条内部 launch 时，可以抢读 caller 未写完的输入或让 caller 后续工作提前消费 remainder。添加 prefix.kernel_done→remainder 虽避免某些抢跑，却改变真实并行关系。",
        "why_not_missing": "milestone 只命名本 phase 的已有局部事件，无法也不应该凭名称创造调用者端点；是否新增外部边界语法是另一层的明确研究题。",
        "prior_leads": ["extract_sparse-F05", "SCHEMA_DISCUSSION.md section 2"],
        "evidence": [
            ev(inp("sparse","probe_hidden_packed"),299,326), ev(inp("sparse","probe_hidden_packed"),428,434),
            ev(SPARSE,6153,6163), ev(SPARSE,6188,6191),
            ev(inp("sparse"),557,571), ev(SPARSE,6343,6364), ev(SPARSE,6379,6403),
            ev(D5,500,504), ev(D4,375,377),
        ],
    },
    {
        "id": "PG04-10", "operator": "extract_moe", "category": "source_or_draft_issue",
        "related_definitions": ["resource", "milestone"],
        "claim": "verify_only 的 HostObserved 可能来自先前调用；把这句话删掉或改称当前 verify_done 会伪造代际与完成保证。增加 phase resource/milestone 也不能修复 CPU reset 与旧 GPU writer 的竞态。",
        "recommendation": "保留无代际 mirror、host reset 未等旧 writer、HostObserved 非当前 Clean/非 kernel_done 的短限制；device 同流复用与 host 观察分别陈述。要引入 generation/tag 或新同步先修设计/源码，不把它当作格式补全。",
        "minimal_counterexample": "上一 full graph 的 GPU mirror 写 1 可以覆盖本次 CPU 写的 -1；当前 verify 未完成时 host 已读 1 并省略 full graph。",
        "why_not_missing": "源本身没有请求标签或该等待关系。真实物理共享属于调用边界；没有证据表明三类公共定义应代替该边界或替源码补安全性。",
        "prior_leads": ["extract_moe-F23", "extract_moe-F24"],
        "evidence": [
            ev(inp("moe","verify_only"),145,158), ev(inp("moe","verify_only"),263,275),
            ev(MOE,9478,9489), ev(MOE,1760,1786),
            ev(D5,500,500), ev(RULE + "3.design.math_reshape.md",102,108),
        ],
    },
]

naming = [
    {"current": "resource", "recommended": "buffer", "reason": "在 depth/location/shape/view 的语境中，一眼能看出登记的是装值的片上存储；可包括寄存器状态和累加器，避免 resource 泛指 SM/引擎/锁。",
     "alternatives_rejected": [{"name": "resource", "reason": "含义过宽，与硬件资源、执行角色混淆。"}, {"name": "allocation", "reason": "容易让人以为这是分配动作或 allocator 生命周期，而这里是静态存储账。"}, {"name": "storage", "reason": "更像存储介质总称，与子字段 location 含义接近；若希望弱化数据流缓冲的印象，它是合理备选。"}],
     "scope_note": "改名不扩到 global/host/cross-call，也不因此登记 barrier 存储或 compiler temporaries。"},
    {"current": "domain", "recommended": "index", "reason": "实例键就是后续循环/视图共用的索引身份，值给出取值范围或序列；比 domain 直接，也保留 worktile/flatten 两种非简单区间的含义。",
     "alternatives_rejected": [{"name": "range", "reason": "容易把 flatten、运行时 worktile 序列误读成静态整数上下界。"}, {"name": "loop", "reason": "登记不执行循环；执行位置属于 for。"}, {"name": "shared_index", "reason": "较长；某些索引因 resource view 而公共，并非多个 role 在使用。"}],
     "scope_note": "不暗含 role 锁步，不把 role 本地索引都提升为公共登记。"},
    {"current": "milestone", "recommended": "event_group", "reason": "直接显示值是已有事件的 all-of 集合；比抽象进度词 milestone 更容易猜到结构。",
     "alternatives_rejected": [{"name": "event", "reason": "更短，但容易被当成一个新 CUDA event 或独立完成时点，隐藏其组合/别名性质。"}, {"name": "checkpoint", "reason": "常暗示保存状态或全体停点。"}, {"name": "signal", "reason": "与实际通知动作冲突；命名事件组并不发送通知。"}],
     "scope_note": "只在跨 phase 引用局部事件时登记；不重命名 kernel_done/launch_ready，也不扩成 host/caller 事件表。"},
]

sketches = [
    {
        "id": "PG04-S1", "finding_ids": ["PG04-03"], "status": "partial proposal in current notation",
        "description": "只展示 GEMM Staging 的 release/store 声明；完整 pipe 中 release→填入 Output→store 的位置、尾块路径与最终目的完成等待继续保留。",
        "code": textwrap.dedent('''\
            using_verb:
              release:
                Staging:
                  execution_mode: serial
                  describe: "Return OutputStaging after all earlier full-tile stores have finished reading it, then rendezvous all epilogue writers."
              store:
                Output:
                  execution_mode: async_unordered
                  describe: "Copy OutputStaging.Output(t,f) to this worktile's owned out region; done guarantees the global writes are visible."
            '''),
        "retained": ["全部历史读源，不仅上一工作", "源可复用早于目标写完", "所有写者汇合", "原 Output(t,f) 区域与版本、full/tail 条件、末尾 drain"],
        "removed_or_relocated": ["first full 没有历史的同义说明", "尾块和下一工作重叠的重复解释", "同一 verb 描述中混入无关 DotPair 的含义；改为另一个 noun 条目"],
        "not_claimed": "不把 worker 内部 per-row 提前 release 自动提升成多 warp 锁步；DotPair 的更细成员时序不由本片段改写。",
        "evidence": [ev(inp("gemm"),439,467), ev(GEMM,1015,1018), ev(GEMM,1033,1055)],
    },
    {
        "id": "PG04-S2", "finding_ids": ["PG04-05"], "status": "partial proposal in current notation",
        "description": "FMHA 将 pv:NumState(g)、retire:Previous(g)、drain:LastPV 对应到下列三个动作。PreviousPV 仍在当前两次 QK/readout 之后、V refill/rescale/Prob8 写之前；FinalPV 仍在循环之后。其余动作也须按八 verb 登记，本片段不构成整份计划。",
        "code": textwrap.dedent('''\
            using_verb:
              matmul:
                PV:
                  execution_mode: async_unordered
                  describe: "Apply value_output for group g: read Prob8(g), UnitScale and ValueScale(g) from their TensorMemory views, and Value(g) from its SMEM view; update TensorMemory.NumState from the rescaled state prepared by compute:StateProb(g), except that the first group initializes it without reading the old value. Done covers NumState and all of this PV's operand reads."
              await:
                PreviousPV:
                  execution_mode: serial
                  describe: "Wait for PV(g-1); at g=cLo there is no previous PV and this action has no data effect."
                FinalPV:
                  execution_mode: serial
                  describe: "Wait for PV(cHi) before final NumState readout."
            waiting:
              released:
                - "{done await:PreviousPV(g)} ← [{done matmul:PV(g-1)}]"
              prepared:
                - "{done await:FinalPV} ← [{done matmul:PV(cHi)}]"
            '''),
        "retained": ["PV 跨入下一组两半 QK 的重叠", "Value/ValueScale/Prob8 全部读完及 NumState 完成", "首组不读旧 NumState；首 PreviousPV 无数据效应", "循环末尾等最后 PV"],
        "removed_or_relocated": ["done denotes enqueue only 的自定义完成规则", "PV parity 翻转细节交 realization/Coder", "单 role 的 z: [0,2] 改为局部 for(z=0:2)，g 因资源 view 继续公共登记"],
        "not_claimed": "当前输入就绪边、所有 view 和 QK 的完成机制仍需保留；不能给 QK/PV 伪造共同 async_ordered 序列。",
        "evidence": [ev(inp("fmha"),594,626), ev(FMHA,4899,4921), ev(FMHA,5121,5128), ev(FMHA,5173,5178)],
    },
    {
        "id": "PG04-S3", "finding_ids": ["PG04-07"], "status": "partial proposal in current notation",
        "description": "MoE MMA 的局部控制与复用边。load:ScaleTM 对应旧 prepare 的纯复制，matmul 对应原 MMA；保留现有完整输入绑定、就绪边、模式和原语次序。三个 using_verb 条目须在完整迁移稿中登记。",
        "code": textwrap.dedent('''\
            pipe:
              - "for(t)":
                  - "if(t > 0 and Hoist == 0)":
                      - "await:Drain(t)"
                  - "for(g[t])":
                      - "load:ScaleTM(g)"
                      - "matmul:PayloadA(g)@PayloadB(g)"
            waiting:
              released:
                - "{done await:Drain(t)} ← [{done Epilogue::release:Accumulator(t-1)}, {done Epilogue0::release:Accumulator(t-1)}, {done Epilogue1::release:Accumulator(t-1)}, {done Epilogue2::release:Accumulator(t-1)}]"
                - "{issue matmul:PayloadA(g[t,0])@PayloadB(g[t,0])} ← [{done Epilogue::release:Accumulator(t-1)}, {done Epilogue0::release:Accumulator(t-1)}, {done Epilogue1::release:Accumulator(t-1)}, {done Epilogue2::release:Accumulator(t-1)}]"
            '''),
        "retained": ["Hoist=0 非首工作在 scale copy 前等四个旧读者", "Hoist=1 允许先拷 scale，但首个 MMA 仍等四个 release", "四个独立 warp role，局部尾部可各自继续", "t/g 与原槽位映射及初末边界"],
        "removed_or_relocated": ["0/1 q 伪循环和 q 实参", "解释 optional wait 的重复长句", "不会被任何 loop/view 使用的公共 c 登记可在完整迁移时删除"],
        "not_claimed": "Epilogue[*] 只选择 worker 成员，不是四个 epilogue warp；不得用它替换上面的四个 role。entry/retire 和数据就绪边未在此局部片段重印。",
        "evidence": [ev(inp("moe"),758,795), ev(inp("moe"),814,848), ev(MOE,4855,4906), ev(D5,566,568)],
    },
]

data = {"coverage": coverage, "findings": findings, "naming": naming, "sketches": sketches}
categories = {"existing_structure", "concise_semantics", "underspecified_rule", "missing_capability", "other_layer", "implementation_detail", "source_or_draft_issue"}
for finding in findings:
    assert finding["category"] in categories
    assert finding["claim"] and finding["recommendation"] and finding["evidence"]
for item in findings + sketches:
    for evidence in item["evidence"]:
        ev(evidence["path"], evidence["start_line"], evidence["end_line"], evidence["quote"])
try:
    import yaml
    for sketch in sketches:
        yaml.safe_load(sketch["code"])
except ImportError:
    pass
out = ROOT / STUDY / "data/04_prose_gaps.json"
out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(out.relative_to(ROOT)), "findings": len(findings), "sketches": len(sketches),
                  "evidence_quotes_checked": sum(len(x["evidence"]) for x in findings + sketches),
                  "source_hashes_checked": len(source_hashes), "root_hashes_checked": len(roots),
                  "categories": dict(Counter(f["category"] for f in findings))}, ensure_ascii=False))
