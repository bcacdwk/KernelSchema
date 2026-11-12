#!/usr/bin/env python3
"""Reproduce this audit's raw milestone census and exact evidence excerpts.

Reads fixed corpus inputs; writes only data/03_events.json. This is not a schema
validator, effective-document expansion, or semantic approval of the census.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "schema_discussion/phase_grammar"
CORPUS = json.loads((BASE / "corpus.json").read_text())
INPUT = "schema_extract/archive/20261109-ten-operator-text-audit/inputs"
WORK = "schema_extract/archive/20261109-ten-operator-text-audit/work"
RULE = "schema_discussion/phase_grammar/sources/rules"


def ev(path: str, first: int, last: int) -> dict:
    lines = (REPO / path).read_text().splitlines(keepends=True)
    assert 1 <= first <= last <= len(lines), (path, first, last)
    return {"path": path, "start_line": first, "end_line": last,
            "quote": "".join(lines[first - 1:last])}


def root(op: str) -> str:
    return f"{INPUT}/extract_{op}/root.yaml"


def branch(op: str, node: str) -> str:
    return f"{INPUT}/extract_{op}/{node}.yaml"


F = branch("fmha", "tcg_d128_segmented_tma_direct_narrow")
M = branch("moe", "double_verify_root")
S = branch("sparse", "probe_hidden_packed")
G = branch("gdn", "segmented")
GC = f"{WORK}/extract_gdn/gdn_kernel.cu"
MC = f"{WORK}/extract_moe/source_nvfp4_moe.cu"
SC = f"{WORK}/extract_sparse/dsv4_kernel.cu"
LC = "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py"
FC = f"{WORK}/extract_fmha/source/candidate/fmha_tcg.cuh"
FTC = f"{WORK}/extract_fmha/source/candidate/fmha_tc.cuh"
D4 = f"{RULE}/4.design.work_schedule.md"
D5 = f"{RULE}/5.design.tile_pipeline.md"

root_ranges = {"gdn": [[320, 690]], "gemm": [[1, 48], [194, 546]],
               "mla": [[286, 652]], "sparse": [[277, 575]],
               "fmha": [[383, 702]], "moe": [[365, 1026]]}
branch_ranges = {
    ("gdn", "segmented"): [[211, 250], [282, 370], [482, 674]],
    ("fmha", "tcg_d128_segmented_tma_direct_narrow"): [[1, 114], [229, 784]],
    ("moe", "double_verify_root"): [[1, 85], [358, 706], [932, 1135],
                                      [1170, 1198], [1348, 1380], [1584, 1650]],
    ("sparse", "probe_hidden_packed"): [[1, 38], [298, 542], [654, 716], [970, 1030]],
}
source_ranges = {
    GC: [[400, 429], [1990, 2058], [2170, 2233], [3423, 3477]],
    f"{WORK}/extract_gemm/fp8gg_umma.cu": [[728, 792], [1009, 1058], [1852, 1886]],
    LC: [[1630, 1695], [1880, 1906], [1992, 2032]],
    SC: [[6090, 6198]],
    f"{WORK}/extract_fmha/source/candidate/fmha_common.cuh": [[84, 116]],
    FTC: [[1368, 1450]],
    FC: [[3445, 3502], [5495, 5520]],
    f"{WORK}/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu": [[1640, 1698]],
    MC: [[185, 220], [1715, 1792], [4488, 4528], [4834, 4930],
         [9144, 9182], [9189, 9246], [9462, 9505]],
}

roots = []
branches = []
sources = []
census = {}
for op_id, op in CORPUS["operators"].items():
    name = op["name"]
    node_lookup = {n["node"]: n for n in op["nodes"]}
    rn = node_lookup["root"]
    d = yaml.load((REPO / rn["snapshot"]).read_text(), Loader=yaml.CSafeLoader)
    roots.append({"operator": name, "path": rn["snapshot"], "category": rn["category"],
                  "node_version": rn["node_version"], "sha256": rn["sha256"],
                  "read_ranges": root_ranges[name],
                  "reading_scope": "完整 work_schedule、tile_pipeline 与 realization；非完整 Contract 数学等价审阅",
                  "phase_count": len(d["design"]["work_schedule"]["phase"]),
                  "milestone_declarations": sum(len(p.get("milestone", {})) for p in
                                                d["design"]["tile_pipeline"]["phase"].values())})
    for (bo, bn), ranges in branch_ranges.items():
        if bo != name:
            continue
        n = node_lookup[bn]
        branches.append({"operator": name, "node": bn, "path": n["snapshot"],
                         "category": n["category"], "node_version": n["node_version"],
                         "sha256": n["sha256"], "read_ranges": ranges,
                         "reading_scope": "事件相关上下文与所列源码对照；未声称全分支数学/数值审阅"})
    for s in op["sources"]:
        if s["path"] not in source_ranges:
            continue
        observed = hashlib.sha256((REPO / s["path"]).read_bytes()).hexdigest()
        assert observed == s["sha256"], s["path"]
        sources.append({"operator": name, "path": s["path"], "sha256": observed,
                        "logical_path": s["logical_path"], "source_ref": s["source_ref"],
                        "matches_source_lock": s["matches_source_lock"],
                        "read_ranges": source_ranges[s["path"]]})
    declarations = []
    eligible = [n for n in op["nodes"] if n["semantic_eligible"]]
    for n in eligible:
        doc = yaml.load((REPO / n["snapshot"]).read_text(), Loader=yaml.CSafeLoader)
        for phase, p in doc.get("design", {}).get("tile_pipeline", {}).get("phase", {}).items():
            for alias, events in p.get("milestone", {}).items():
                declarations.append({"node": n["node"], "category": n["category"],
                                     "phase": phase, "name": alias, "events": events})
    census[name] = {
        "eligible_design_files": len(eligible),
        "eligible_by_category": dict(collections.Counter(n["category"] for n in eligible)),
        "local_milestone_files": len({r["node"] for r in declarations}),
        "local_milestone_declarations": len(declarations),
        "declarations_by_category": dict(collections.Counter(r["category"] for r in declarations)),
        "event_group_sizes": dict(sorted(collections.Counter(len(r["events"]) for r in declarations).items())),
        "indexed_alias_keys": sum("(" in r["name"] for r in declarations),
    }

findings = [
    {
        "id": "EV01", "operator": ["gdn", "fmha"], "category": "concise_semantics",
        "related_common": ["milestone"],
        "claim": "对外事件别名有实际复用价值；GDN 八个值共用一次延后的 PDL wait。启动许可、全 grid 数据完成与每个消费者等待点互不等价，不能用一个 done 字段合并。",
        "recommendation": "保留可选别名，建议字段改名 event。start_after、ready、wait_at 继续分别记录启动、生产就绪、消费约束；GDN 的 ready: prep.kernel_done 与 wait_at 不删。直接 role 引用可作为后续单次引用简写候选，本轮不增加第二套正式寻址。",
        "evidence": [ev(root("gdn"), 326, 338), ev(root("gdn"), 552, 553),
                     ev(root("gdn"), 601, 609), ev(GC, 2211, 2225),
                     ev(F, 659, 668), ev(FC, 3463, 3474), ev(FTC, 1371, 1378),
                     ev(D4, 290, 295), ev(D4, 369, 379)],
    },
    {
        "id": "EV02", "operator": ["gdn", "fmha", "moe"], "category": "underspecified_rule",
        "related_common": ["milestone", "domain"],
        "claim": "定义位于 phase 不意味着事件只有一份。动作身份已有 worker/member/实参，但 milestone 的实例化及键形参规则只给原则；多 consumer 只列部分 wait_at 时其余 consumer 是否保留默认也未说清。真实案例包含不一一对应的生产/消费工作块，以及无 payload worktile 仍发生的入口 guard。",
        "recommendation": "补短规则：别名是一族事件，成员 selector 只选当前所解析 worker 的成员；handoff 按数学值/版本/区域映射选择相关生产与消费 worker，全部相关生产结果均须满足。形参由引用实参代入，不捕获 role 循环；prologue 事件由 pipe 实际存在性决定，不因 worktile 域空而删去。明确每个 consumer 的默认/显式 wait_at 规则，start_after 引用局部别名时须给出整个提交可用的覆盖范围。没有真实案例要求新 scope/map 字段，本轮不增。",
        "evidence": [ev(D5, 174, 177), ev(D5, 498, 504), ev(D5, 557, 568),
                     ev(D4, 299, 305), ev(D4, 371, 373), ev(F, 295, 308),
                     ev(F, 318, 336), ev(G, 284, 289), ev(G, 321, 340),
                     ev(M, 1187, 1196), ev(M, 608, 626)],
    },
    {
        "id": "EV03", "operator": ["moe"], "category": "underspecified_rule",
        "related_common": ["milestone"],
        "claim": "生产侧的 all-of 事件组可以汇成一个前驱；消费侧 wait_at 需要就绪条件分别约束组内每个事件，不能仅约束它们最后全部完成的时刻。MoE fc1/fc2 的七角色 guard 组是实存例子。现有实例动作契约含真实 wait，不能据此声称源程序有此竞态；待补的是分组引用的共同语义。",
        "recommendation": "明说 ready/start_after 是对前驱集合取全部，wait_at 展开为对每个列出事件的约束。不能用最早一名、最晚一名或角色名通配猜测全部参与者；有条件时依动作实际发生域核对，已选事件缺失不能悄悄丢弃。无需新增 any/all 运算或 barrier 字段。",
        "minimal_counterexample": "把 MoE 的七个 guard 等待点记为 c0..c6，就绪为 r。仅规定 r 先于 max(c0..c6)，允许 c0<r<c6；本意需要对每个 j 都有 r<=cj。取两名角色即已足够显示两种解释不同。",
        "evidence": [ev(M, 398, 404), ev(M, 1366, 1368), ev(M, 1601, 1603),
                     ev(M, 1187, 1196), ev(MC, 208, 214),
                     ev(D5, 498, 504), ev(D4, 293, 295), ev(D5, 621, 626)],
    },
    {
        "id": "EV04", "operator": ["moe"], "category": "existing_structure",
        "related_common": ["milestone"],
        "claim": "double_verify_root 的 Clean.guard 等待点组在这个节点是可推导的重复：count 整个提交已等 verify.kernel_done，之后所有提交经 launch_ready 链接；Clean 在任何这些入口读取前已完成。七个 alias 含十九个原事件引用，并非缺少组语法才必须逐项写。",
        "recommendation": "可删除此节点 Clean.wait_at 及仅供它使用的七个 guarded 别名，保留 Clean 的 producer/consumer/storage/ready、启动边、每个真实 guard 读取与其空工作行为。该删法只对已经证明启动边支配全部读取的节点成立，不能推广到 GDN/FMHA 的延后等待。",
        "preserved_facts": ["第二次验证产生 Clean", "七个消费 phase 与全部已提交 worker 仍存在", "guard 在 payload 前读取当前 verdict", "Active=0 不制造 payload worktile", "PDL 与 kernel_done 的原保证"],
        "evidence": [ev(M, 374, 404), ev(M, 1035, 1048), ev(M, 1187, 1196),
                     ev(M, 1366, 1368), ev(M, 1601, 1603),
                     ev(MC, 9226, 9241), ev(MC, 185, 214), ev(D5, 477, 486)],
    },
    {
        "id": "EV05", "operator": ["moe", "gemm", "fmha"], "category": "existing_structure",
        "related_common": ["milestone", "resource", "domain"],
        "claim": "若实际关系在同一 phase 内，应该用 waiting/if 和真实动作表达，不能因出现多事件组或 release 就上升为 milestone。MoE 四 epilogue 的条件 Drain 已有结构；GEMM 源缓冲 read-complete 与目标 global-complete 也已有不同动作承载。FMHA root 的旧 enqueue-done 写法需按新八动词/模式迁移，不是缺一个局部里程碑。",
        "recommendation": "将 MoE 的 0/1 q 伪循环改为 if(t>0 and Hoist==0) 下 await:Drain(t)，保持首个 MMA 的四前驱。完成/源读完的差别留在 store/release 与 waiting，barrier 地址、编号、phase-bit 编码交 realization/Coder；不在 phase 事件表抄全部内部动作。",
        "evidence": [ev(root("moe"), 758, 795), ev(root("moe"), 814, 824),
                     ev(MC, 4855, 4871), ev(MC, 4894, 4915),
                     ev(root("gemm"), 439, 467),
                     ev(root("gemm"), 30, 38),
                     ev(f"{WORK}/extract_gemm/fp8gg_umma.cu", 1015, 1039),
                     ev(f"{WORK}/extract_gemm/fp8gg_umma.cu", 1055, 1055),
                     ev(root("fmha"), 577, 593), ev(D5, 190, 192), ev(D5, 226, 228), ev(D5, 568, 568)],
    },
    {
        "id": "EV06", "operator": ["mla"], "category": "other_layer",
        "related_common": ["milestone", "resource"],
        "claim": "MLA Geo 由 host 在 side stream 上传，main launch 要等上传完成；这是实际执行前提，目前只能在 schedule.strategy 写清。它不是 main 中某个 role 的事件，milestone 不能引用它。能写进文字不等于 flow 已有这个外部端点。",
        "recommendation": "作为调用/调度外边界需求保留：一条短规则注明 host 建表、上传完成→main 启动，并将实际 copy/event API 留在必要实现绑定。若希望结构可检查，另议外部前提/完成端点；不要虚构 GPU upload phase 或把等待移进 main 来凑 milestone。该源码侧流 metadata 输入本身是否满足调用者生产完成前提，须另查调用约定，不能由新别名补保证。",
        "evidence": [ev(root("mla"), 291, 292), ev(root("mla"), 320, 336),
                     ev(LC, 1646, 1681), ev(LC, 1895, 1898), ev(LC, 2008, 2024),
                     ev(D4, 27, 29), ev(D4, 367, 373), ev(D5, 504, 504)],
    },
    {
        "id": "EV07", "operator": ["sparse"], "category": "other_layer",
        "related_common": ["milestone"],
        "claim": "Sparse hidden probe 分支的输入 fork 和输出 join 是真实外部边界：probe 与 prefix 等 caller 既有工作；caller 后续使用等 prefix 和 remainder 两路。flow 只登记 remainder 等 probe，余下关系在 thesis/describe，内部 milestone 没有 caller 端点可引用。",
        "recommendation": "承認调度外边界结构尚缺共同写法，保留精简的前后关系并单独讨论共同调用约定或外部事件引用。不可把 prefix 与 remainder 串行、添加虚构 merge/kernel 或让 host 同步来替代；probe 的 host 观察也不能改叫 kernel_done。事件对象缓存方式与 DisableTiming 等普通 API 细节不必进入事件别名。",
        "evidence": [ev(S, 299, 326), ev(S, 427, 444), ev(S, 532, 536),
                     ev(SC, 6153, 6163), ev(SC, 6175, 6191),
                     ev(D4, 260, 271), ev(D5, 498, 504)],
    },
    {
        "id": "EV08", "operator": ["fmha", "moe"], "category": "source_or_draft_issue",
        "related_common": ["milestone", "resource"],
        "claim": "事件别名不能补出本不存在的区域/可见性保证。FMHA 的 scan wait 后仍有未按非空 split 保护的入口 MState 读取，旧稿明确不承诺该额外读取的初始化/安全；MoE 最后到达者的 pinned-host verdict 也不是全部输出存储完成。改变事件名或把 publish.done 当 kernel_done 会掩盖来源边界。",
        "recommendation": "保留真实保证与源码限制的区分，源异常另核对/修复。短总则足够：别名只选择已有事件，不创建动作、通知、可见性或初始化；保证来自对应动作与绑定原语。不要为这些限制新增 read/visible/host_done 事件种类，也不把源码缺保证包装为新字段已解决。",
        "evidence": [ev(F, 667, 673), ev(FC, 3474, 3488),
                     ev(M, 1007, 1018), ev(MC, 1760, 1789),
                     ev(D5, 196, 196), ev(D5, 498, 504), ev(D5, 183, 190)],
    },
]

sketches = [
    {
        "id": "ES01", "operator": "gdn", "status": "建议草图，非现行规范变更；旧动词转换只展示所需片段",
        "purpose": "保留八次复用的别名并改为 event；比较直接引用的新语法成本。",
        "current_excerpt": "milestone:\n  prep_ready: [\"Main/join_prep:Prepared@done\"]\n",
        "candidate": "event:\n  prep_wait: [\"{done MAIN[0]::await:Prepared}\"]\n\n# handoff 中八个值继续各自引用同一等待位置\nTi: {producer: prep, consumer: [chain], storage: global, ready: prep.kernel_done, wait_at: chain.prep_wait}\nQc: {producer: prep, consumer: [chain], storage: global, ready: prep.kernel_done, wait_at: chain.prep_wait}\n",
        "direct_reference_alternative": "# 仅比较用；phase::Role[...] 的跨 phase 寻址是新提案，当前不接受\nTi: {producer: prep, consumer: [chain], storage: global, ready: prep.kernel_done, wait_at: \"{done chain::MAIN[0]::await:Prepared}\"}\nQc: {producer: prep, consumer: [chain], storage: global, ready: prep.kernel_done, wait_at: \"{done chain::MAIN[0]::await:Prepared}\"}\n",
        "preserves": ["prep grid completion", "chain input-only prework", "one actual PDL wait", "eight independent mathematical values"],
        "assessment": "直接引用对单次引用节省一个声明，对此例会在八处暴露内部 role/action 路径；不建议为省一行去掉可复用别名层。event 与 prep_wait 两处改名均为建议。",
        "evidence": [ev(root("gdn"), 331, 338), ev(root("gdn"), 552, 553), ev(root("gdn"), 601, 609)],
    },
    {
        "id": "ES02", "operator": "moe", "status": "使用现有字段的删减候选；须按节点核对支配关系",
        "purpose": "删除 double_verify_root 中已由启动链保证的七个消费别名，保留实际 guards。",
        "current_excerpt": "Clean:\n  producer: verify\n  consumer: [count, scan, scatter, gather, fc1, fc2, finalize]\n  storage: global\n  ready: verify.kernel_done\n  wait_at: [count.guarded, scan.guarded, scatter.guarded, gather.guarded, fc1.guarded, fc2.guarded, finalize.guarded]\n",
        "candidate": "Clean:\n  producer: verify\n  consumer: [count, scan, scatter, gather, fc1, fc2, finalize]\n  storage: global\n  ready: verify.kernel_done\n",
        "additional_edit": "仅删此 Clean.wait_at 和七个仅被它引用的 guarded 定义；保留 count.start_after: verify.kernel_done、全部后继 launch_ready 链、入口 guard 动作及空工作语义。",
        "assessment": "两个七角色组和五个单角色组共十九个事件引用在此节点可省。不是证明多事件组本身没用，也不是删除原语；Active=0 的入口仍真实发生。",
        "evidence": [ev(M, 374, 404), ev(M, 1187, 1196), ev(M, 1366, 1368), ev(M, 1601, 1603)],
    },
    {
        "id": "ES03", "operator": "moe", "status": "当前 for/if/waiting 结构的局部迁移草图，非新字段；其余动作声明/依赖仍须保留",
        "purpose": "四角色条件 Drain 留在本地等待图，去掉人工 q 域，不另建 milestone。",
        "current_excerpt": "- repeat: {q: [0, \"min(1,t)*(1-Hoist)\"]}\n  steady: [\"wait:Drain(t,q)\"]\n",
        "candidate": "pipe:\n  - \"for(t)\":\n      - \"if(t > 0 and Hoist == 0)\":\n          - \"await:Drain(t)\"\n      - \"for(g[t])\":\n          - \"load:ScaleTM(g)\"\n          - \"matmul:PayloadA(g)@PayloadB(g)\"\nwaiting:\n  released:\n    - \"{done await:Drain(t)} ← [{done Epilogue[0]::release:Accumulator(t-1)}, {done Epilogue0[0]::release:Accumulator(t-1)}, {done Epilogue1[0]::release:Accumulator(t-1)}, {done Epilogue2[0]::release:Accumulator(t-1)}]\"\n    - \"{issue matmul:PayloadA(g[t,0])@PayloadB(g[t,0])} ← [{done Epilogue[0]::release:Accumulator(t-1)}, {done Epilogue0[0]::release:Accumulator(t-1)}, {done Epilogue1[0]::release:Accumulator(t-1)}, {done Epilogue2[0]::release:Accumulator(t-1)}]\"\n",
        "assessment": "Drain 用 serial await，ScaleTM 用声明了累计矩阵完成依据的搬运动作；本草图仅显示控制/复用边，原先 prepared 前驱、初始化、回收、动作 describe 均需照常声明。四角色列表可继续用 YAML anchor 共享；它不创建新事件。",
        "evidence": [ev(root("moe"), 758, 795), ev(MC, 4855, 4871), ev(MC, 4894, 4915)],
    },
]

out = {
    "coverage": {
        "basis": "sources/rules 当前快照；六个暂停 working root 和四个有意选择的分支，均回查固定源码；未读取本轮其他新代理报告",
        "rules_read": [f"{RULE}/0.common.md", D4, D5, f"{RULE}/6.realization.md"],
        "roots": roots, "branches": branches, "sources": sources,
        "reports_read": CORPUS["reports"], "new_source_copies_created": [],
        "source_hash_note": "上表记录实际读取的九份固定源码 SHA256，全部与共同 corpus 和其源锁一致；未另增源码副本。",
        "mechanical_census": {
            "algorithm": "对 corpus.semantic_eligible=true 的172份 raw YAML 用 CSafeLoader读取，仅统计本地 design.tile_pipeline.phase.*.milestone 条目；YAML alias 按字段位置计入，不展开 from_parent、不按运行次数展开，不作为作者重复输入量。",
            "denominator": {"material_files": 177, "semantic_eligible_designs": 172, "scaffolds_excluded": 5, "not_started_excluded": 43},
            "operators": census,
        },
        "limits": ["六 root 全部读过相关完整 W/T/R；四分支只审阅所列事件上下文，未声称172份全量语义审阅。",
                   "approved/review_candidate/needs_authoring 是冻结语料状态，非当前八动词等新规则的合格判定。",
                   "没有运行 GPU、schema 提取或正式检查器，没有修改正式规范/算子/运行状态。",
                   "未找到必须扩充 milestone 内部结构的真实最小反例，不等于证明该层对所有未来设计完备。",
                   "未发现真实索引别名键只说明此次 raw 语料没有覆盖它，不能支持删除索引能力。"],
    },
    "findings": findings,
    "naming": {
        "resource": {"recommended": "storage", "reason": "限定片上存储占用，涵盖 SMEM/TMEM/寄存器状态，不暗示算力引擎或 CUDA 资源对象。",
                     "alternatives": [{"name": "buffer", "assessment": "常用且短，但易让持久状态、累加器看成临时传输缓冲。"},
                                      {"name": "resource", "assessment": "保留兼容阅读习惯，但范围比当前存储账宽。"},
                                      {"name": "allocation", "assessment": "精确但长，易把分配器/API 身份带进设计。"}]},
        "domain": {"recommended": "index", "reason": "条目实际给共享索引的身份和有序域；比 domain 更易快速猜到用途。",
                   "alternatives": [{"name": "domain", "assessment": "数学术语准确，熟悉集合/迭代域者可保留。"},
                                    {"name": "range", "assessment": "会把 worktile 序列和 flatten 误读成整数上下界。"},
                                    {"name": "loop", "assessment": "暗示这里创建循环/锁步，实际循环在 pipe。"}]},
        "milestone": {"recommended": "event", "reason": "简短惯用，能覆盖生产就绪、消费等待与索引事件族；定义首句明确只是对外别名。",
                      "alternatives": [{"name": "event_group", "assessment": "最直接点出集合，但比 milestone 更长，单元素多数时略重。"},
                                       {"name": "milestone", "assessment": "能表达公开执行进度，但易误读为全 phase 只有一次的时点。"},
                                       {"name": "mark", "assessment": "短但暗示插入标记动作，猜不到 all-of 引用。"},
                                       {"name": "sync", "assessment": "不采用：误导为创建同步，而不是命名已有保证。"}],
                      "instance_names": "生产者用 x_ready；消费者 await.done 用 x_wait、读取 issue 用 x_read。名字不能替代显式 issue/done。"},
    },
    "sketches": sketches,
}

for f in findings:
    assert f["category"] in {"existing_structure", "concise_semantics", "underspecified_rule",
                              "missing_capability", "other_layer", "implementation_detail", "source_or_draft_issue"}
for block in findings + sketches:
    for e in block["evidence"]:
        original = ev(e["path"], e["start_line"], e["end_line"])["quote"]
        assert e["quote"] and e["quote"] in original

destination = BASE / "data/03_events.json"
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(destination.relative_to(REPO)),
                  "roots": len(roots), "branches": len(branches), "sources": len(sources),
                  "findings": len(findings), "sketches": len(sketches),
                  "milestone_declarations": sum(x["local_milestone_declarations"] for x in census.values()),
                  "indexed_alias_keys": sum(x["indexed_alias_keys"] for x in census.values())}, ensure_ascii=False))
