"""Build task 05's evidence bundle from the frozen corpus; no source edits."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = "schema_extract/archive/20261109-ten-operator-text-audit/"
RULE = "schema_discussion/phase_grammar/sources/rules/"
CORPUS = json.loads((ROOT / "schema_discussion/phase_grammar/corpus.json").read_text())
Y = {k: v["root"] for k, v in CORPUS["operators"].items()}
S = {
    "gdn": BASE + "work/extract_gdn/gdn_kernel.cu",
    "gemm": BASE + "work/extract_gemm/fp8gg_umma.cu",
    "mla": "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py",
    "sparse": BASE + "work/extract_sparse/dsv4_kernel.cu",
    "fmha": BASE + "work/extract_fmha/source/candidate/fmha_tcg.cuh",
    "moe": BASE + "work/extract_moe/source_nvfp4_moe.cu",
}
B = {
    "direct": BASE + "inputs/extract_moe/direct.yaml",
    "verify": BASE + "inputs/extract_moe/verify_fused_big_flat.yaml",
    "xpose": BASE + "inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml",
    "hidden": BASE + "inputs/extract_sparse/probe_hidden_packed.yaml",
}


def evidence(path, start, end):
    lines = (ROOT / path).read_text().splitlines(keepends=True)
    assert 1 <= start <= end <= len(lines), (path, start, end)
    return {"path": path, "start_line": start, "end_line": end,
            "quote": "".join(lines[start - 1:end]).rstrip("\n")}


root_spans = {
    "extract_gdn": [[1, 30], [308, 690]],
    "extract_gemm": [[1, 38], [181, 546]],
    "extract_mla": [[1, 16], [276, 652]],
    "extract_sparse": [[1, 23], [266, 575]],
    "extract_fmha": [[1, 36], [369, 702]],
    "extract_moe": [[1, 29], [351, 1026]],
}
branch_spans = {
    "direct": [[183, 421]],
    "verify": [[282, 338], [553, 633], [732, 821], [956, 994],
               [1040, 1075], [1185, 1221]],
    "xpose": [[208, 233], [293, 373], [471, 665]],
    "hidden": [[298, 389], [443, 544], [1025, 1055]],
}
source_spans = {
    "gdn": [[2205, 2248], [2448, 2555], [2800, 2879], [3252, 3266]],
    "gemm": [[284, 355], [638, 666], [730, 764], [1008, 1044]],
    "mla": [[493, 511], [1450, 1488], [1664, 1688], [1890, 1902], [2004, 2022]],
    "sparse": [[6138, 6196]],
    "fmha": [[1689, 1711], [1818, 1853], [3300, 3488], [4386, 4439],
             [4450, 4478], [4688, 4720], [5222, 5248], [5382, 5448], [5515, 5521]],
    "moe": [[180, 220], [1182, 1230], [1914, 1992], [2210, 2242], [4202, 4230],
            [4840, 4918]],
}

roots = []
for op, spans in root_spans.items():
    node = next(n for n in CORPUS["operators"][op]["nodes"] if n["node"] == "root")
    roots.append({"operator": op, "node": "root", "path": node["snapshot"],
                  "category": node["category"], "node_version": node["node_version"],
                  "sha256": node["sha256"], "read_ranges": spans,
                  "scope": "完整读取相关 work_schedule、tile_pipeline、realization 与 ledger；另读当前选择域。未声称逐项重审完整 Contract 数学。"})
branches = []
for label, path in B.items():
    op = path.split("/")[-2]
    node = next(n for n in CORPUS["operators"][op]["nodes"] if n["snapshot"] == path)
    branches.append({"operator": op, "node": node["node"], "path": path,
                     "category": node["category"], "node_version": node["node_version"],
                     "sha256": node["sha256"], "read_ranges": branch_spans[label],
                     "scope": "针对所列边界精读相关上下文，不声称全文语义批准。"})
sources = []
for op, path in S.items():
    actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    locked = next(s for s in CORPUS["operators"]["extract_" + op]["sources"] if s["path"] == path)
    assert actual == locked["sha256"]
    sources.append({"operator": op, "path": path, "sha256": actual,
                    "matches_corpus_sha256": True,
                    "corpus_matches_source_lock": locked["matches_source_lock"],
                    "read_ranges": source_spans[op], "new_copy_created": False})

coverage = {
    "task": "05_names_minimality",
    "rules": [RULE + "0.common.md", RULE + "4.design.work_schedule.md",
              RULE + "5.design.tile_pipeline.md", RULE + "6.realization.md"],
    "method": "schema_discussion/phase_grammar/METHOD.md",
    "corpus": "schema_discussion/phase_grammar/corpus.json",
    "required_prior_reports_read": CORPUS["reports"],
    "roots": roots, "branches": branches, "source_evidence_hashes": sources,
    "semantic_sample": {"root_count": 6, "selected_branch_count": 4,
                        "status": "所选均为历史 approved；不等于符合本轮当前规则。"},
    "locator_only": {
        "method": "读取 corpus 节点与源码目录；rg 扫描六类 inputs 的 order/storage/milestone 以定位；另解析 gdn/segmented、moe/fused_big_mono、moe/verify_fused_big_mono 的结构摘要后未选作精读分支。",
        "not_semantic_reviews": True,
        "issue_lookup": "搜索 schema_extract/issues 中 milestone/wait_at/单 role/共享索引/domain；读取 capture-invocation-boundary、conditional-flow-topology、source-defect-abstraction-boundary 的相关条目。该关键词查询不构成全局未提报证明。"
    },
    "constraints": ["未读取本轮其他代理产物或综合 README", "未运行 GPU、提取器或 checker 迁移",
                    "未新增源码副本；源码 hash 均与 corpus 一致", "未把 172 有设计稿称为本人的语义审阅数"],
    "overall_judgment": "本样本未发现必须增加第四种 phase 公共定义的反例。保留三种可选定义；它们对存储身份、索引身份、事件接口分别负责。三者够用不等于单靠这三者就能描述 kernel，也不证明所有 kernel 完备。",
}

findings = [
    {
        "id": "N1", "operator": ["moe", "gdn", "gemm"], "category": "existing_structure",
        "claim": "存储登记不能由逻辑值或动作 noun 代替；身份按 worker 实例延续，而非按 phase 名重建。MoE direct 的 Census 与 Exchange 跨两个 phase 留在同一 CTA；root FC1/FC2 的同名 PayloadA 则是两次 kernel 的独立分配。",
        "recommendation": "保留并改名 storage。保留 location、depth、shape/dtype、必要 view 与成员范围；一块分配只记一次。沿用 launch.order/handoff 判断存续；存储名字或 YAML alias 均不提供依赖。",
        "other_definitions": "index 只给版本命名，event 只给动作事件命名，都不能给出分配身份、容量或重叠区域。",
        "elimination_trial": "删除 MoE direct 的 resource，剩余 Census 数学标量和 handoff 仍不知道该值留在每个 CTA 的 SMEM；Exchange 的 WarpCounts/FinalRoutes 共用也无法从数学值推出。把这些写到 math 或 describe 只是移动存储账。",
        "evidence": [evidence(B["direct"], 188, 199), evidence(B["direct"], 291, 306),
                     evidence(B["direct"], 372, 375), evidence(S["moe"], 1929, 1932),
                     evidence(S["moe"], 1966, 1985), evidence(S["moe"], 2217, 2237),
                     evidence(Y["extract_moe"], 376, 383),
                     evidence(Y["extract_moe"], 850, 869), evidence(RULE + "4.design.work_schedule.md", 324, 324)],
    },
    {
        "id": "N2", "operator": ["moe", "fmha", "mla"], "category": "underspecified_rule",
        "claim": "只许多个 role 或资源 view 使用 phase domain 的条件过窄：MoE root gather 只有一个 role、没有登记资源，却需要 t:worktile 来沿用上游分配序列。另一方面 FMHA root 的 z 仅作两半 QK 局部循环，可直接移进 for。",
        "recommendation": "改名 index，并把适用条件改为：多个使用处需要共同解释的索引，或需要具名 worktile/flatten 有序序列时登记；只在一个局部循环中用到的整数范围内联。允许一个 role；明确是索引域声明，不是可变计数器或锁步调度。",
        "other_definitions": "单 role 的 g 仍被 Key/Value 的 view 使用，不能因只有一个 role 一并删掉。事件实参与 view 版本均沿用相同索引身份。",
        "elimination_trial": "FMHA z:[0,2] 可消去而保留 for(z=0:2)，这是消去重复；MoE t:worktile 若消去，只能重抄 worker_assignment 的起点/步长/终止，或另增局部 worktile 域语法。MLA flatten 若换手写累计 N(w)，则把同一关系散到循环、槽号和跨工作边中。",
        "evidence": [evidence(RULE + "5.design.tile_pipeline.md", 320, 332),
                     evidence(Y["extract_moe"], 473, 475), evidence(Y["extract_moe"], 617, 631),
                     evidence(S["moe"], 1189, 1203), evidence(Y["extract_fmha"], 461, 482),
                     evidence(Y["extract_fmha"], 644, 657), evidence(S["fmha"], 4706, 4712),
                     evidence(Y["extract_mla"], 363, 380), evidence(S["mla"], 1456, 1483)],
    },
    {
        "id": "N3", "operator": ["gdn", "moe"], "category": "underspecified_rule",
        "claim": "milestone 是局部事件接口，单事件、七个角色的事件组、消费侧 await.done 都有真实用途。生产聚合的全部完成与消费目标的逐事件约束必须分开解释；仅约束一组最晚成立时点不足以保护其中较早的读取。",
        "recommendation": "改名 event，保持已有事件引用列表与全部满足规则。补清：ready 展开为全部相关生产事件；wait_at 对每个列出的实际目标事件施加就绪约束。显式 role/member、工作索引与区域匹配继续沿用；不新增 runtime 通知、全局 alias、all/any 字段或事件种类。",
        "other_definitions": "事件可命名存储释放或输入等待，但 event 本身不分配存储、不生成索引、不产生可见性。",
        "elimination_trial": "删除 GDN prep_ready 并省掉 wait_at 会把 prep 完成等待移到消费 worktile 入口，失去先读 initial_state 的安排；允许 work_schedule 直接引用 role::await 又只是把局部事件接口耦合进调度语法。MoE 七角色列表还会随每个调用处展开。",
        "logical_witness": "若生产就绪在 5、两个消费端点在 2 和 7，只约束全部消费端点已发生的聚合时点 >=5 仍容许第一个消费过早。数字是规则反例，不是实际测量。所选 MoE 分支还存在启动/PDL 传递保护，本条不声称观察到该分支发生过早读取。",
        "scope_note": "MoE 的七个角色都在一个 CTA 成员内；Epilogue[*] 不等价于 Epilogue、Epilogue0、Epilogue1、Epilogue2。[*] 选择 worker 成员，不选择 warp 或 role。",
        "evidence": [evidence(Y["extract_gdn"], 330, 338), evidence(Y["extract_gdn"], 550, 553),
                     evidence(Y["extract_gdn"], 601, 609), evidence(S["gdn"], 2211, 2225),
                     evidence(B["verify"], 315, 317), evidence(B["verify"], 615, 630),
                     evidence(B["verify"], 981, 983), evidence(S["moe"], 208, 214),
                     evidence(S["moe"], 4215, 4218), evidence(RULE + "5.design.tile_pipeline.md", 500, 504),
                     evidence(RULE + "4.design.work_schedule.md", 290, 295)],
    },
    {
        "id": "N4", "operator": ["fmha", "gemm"], "category": "existing_structure",
        "claim": "slot/view 映射和容量是主干事实，不能因其像地址公式就交给 Coder 自行选择；但 total_size 是派生校验值，不是另一份独立设计决定，也不是 kernel 实際总预留。",
        "recommendation": "保留必要 slot_index/part/dtype 与 total_size 的现有校验等式，可由工具生成总字节数，不增加第二套 bytes/stride/lifetime 字段。本轮不提出把 total_size 改成省略字段。区分设计登记的逻辑存储账与 realization 中有因果作用的实际预留；视图映射不规定 lane 地址或 swizzle。",
        "other_definitions": "view 用 index 的版本作参数，但不捕获某个 role 当前游标；reuse 需要动作/waiting，不能从总字节数或区域相交推导同步。",
        "measured_or_derived": {"kind": "静态推导，非性能测量", "sample": "FMHA root D=128, Br=128, KeyGroup=128, 单 CTA",
                               "registered_smem_bytes": 83968, "source_initial_smem_request_bytes": 102912,
                               "difference_bytes": 18944, "difference_components": [16384, 512, 1024, 1024],
                               "tmem_bytes": 131072,
                               "algorithm": "SMEM=128*128 + 2*128*128 + 2*128*128 + 2*512 + 2*512；TMEM=128*256*4；初始请求采用 root realization 与 TcgLayout 同一公式。未把动态 occupancy 调整后的请求量等同 102912。"},
        "facets": [{"category": "concise_semantics", "item": "异步写源已读完即可复用、目标写入仍在途：保留一条明确完成契约与必要 release/waiting。"},
                   {"category": "other_layer", "item": "额外但影响驻留的实际 SMEM reservation 在 realization 保留；不冒充新的逻辑值分配。"}],
        "evidence": [evidence(Y["extract_fmha"], 466, 507), evidence(Y["extract_fmha"], 688, 702),
                     evidence(S["fmha"], 1822, 1836), evidence(S["fmha"], 1849, 1852),
                     evidence(B["xpose"], 503, 512), evidence(B["xpose"], 605, 611),
                     evidence(S["fmha"], 5225, 5246), evidence(Y["extract_gemm"], 439, 457),
                     evidence(S["gemm"], 1015, 1018), evidence(S["gemm"], 1033, 1039),
                     evidence(RULE + "5.design.tile_pipeline.md", 346, 350)],
    },
    {
        "id": "N5", "operator": ["fmha", "gdn", "moe"], "category": "source_or_draft_issue",
        "claim": "存储别名不提供读写排序。FMHA xpose 空 split 与 GDN 最终 mini chunk 都有旧读者未退场即覆盖的来源风险；这不构成第四种 phase 定义。MoE 的条件 drain 已在当前工作稿放回显式前驱，不应继续称为尚只藏在文字中的新缺口。",
        "recommendation": "保留有据设计关系和短来源限制，迁移时将已有依赖表达为 waiting；不能靠 storage/event 名字自动补 join。MoE 条件等待用已有 if + await + released 转写，不再使用 0/1 假循环。没有足够来源保证时如实保留 source_or_draft_issue。",
        "other_definitions": "resource、domain、milestone 本身都不能修复来源缺少同步；事件必须指向真实动作保证。",
        "facets": [{"category": "existing_structure", "item": "FMHA stage_partials.issue 等待 prepare.done，以及 MoE 条件 drain 都已有表达位置。"},
                   {"category": "concise_semantics", "item": "原语自身的有序读后写保证可以短 describe 说明并绑定 realization；不强迫逐指令加边。"}],
        "evidence": [evidence(B["xpose"], 569, 580), evidence(B["xpose"], 604, 612),
                     evidence(B["xpose"], 642, 649), evidence(S["fmha"], 3351, 3365),
                     evidence(S["fmha"], 3471, 3482),
                     evidence(S["fmha"], 4462, 4467), evidence(S["fmha"], 5394, 5397),
                     evidence(S["fmha"], 5412, 5424), evidence(Y["extract_gdn"], 649, 660),
                     evidence(S["gdn"], 2499, 2512), evidence(S["gdn"], 2543, 2551),
                     evidence(S["gdn"], 2827, 2839), evidence(Y["extract_moe"], 781, 795),
                     evidence(S["moe"], 4855, 4868), evidence(S["moe"], 4903, 4915)],
    },
    {
        "id": "N6", "operator": ["gdn", "fmha"], "category": "implementation_detail",
        "claim": "记录尺度可以变小：GDN final_store 把最后一次未使用的 current 赋值写进契约，无助于数学、容量或交接；FMHA 的不可达越界 guard 和 barrier parity 编码也不值得在 phase 公共定义新增字段。",
        "recommendation": "删除最后一次无效寄存器赋值的叙述，保留转置目标、分批下一批读取先于当前批存储的重叠关系、staging 生命周期与真正完成边界。普通地址、lane 算式、barrier 编号交给 Coder；影响占用的 reservation 或真实发布/复用 join 不能一起删。",
        "other_definitions": "三类公共定义只描述必要身份与约束；不把 Coder 的局部临时变量当第四种公共状态。",
        "facets": [{"category": "concise_semantics", "item": "批次预读与写出的必要相对关系可用一句动作说明保留，无需补一套线程级循环 DSL。"}],
        "evidence": [evidence(Y["extract_gdn"], 661, 669), evidence(S["gdn"], 2855, 2864),
                     evidence(Y["extract_fmha"], 532, 541), evidence(RULE + "6.realization.md", 11, 13)],
    },
    {
        "id": "N7", "operator": ["mla", "sparse", "gdn", "moe"], "category": "other_layer",
        "claim": "MLA Geo 上传后的 launch 等待、Sparse caller fork/join，以及跨调用缓存身份/复用都有真实语义，但不能由 worker 的 storage/index/event 冒充。源码确实执行外部准备与事件队列，问题不等于 phase 表达力不足。",
        "recommendation": "工作表的工作排序留在 work_schedule；外部供给/输出可用性归公共调用边界与调度；必要 CUDA 原语和物理缓存约定留 realization/调用约定。短句保留当前明确关系，若需要结构化外部引用应单独讨论上层范围，不能添加伪 host phase 或把 global cache 放入片上 storage。",
        "other_definitions": "event 是局部已存在动作的出口，不凭空制造外部 host 上传、后继 caller 工作或跨调用版本；index 的 worktile 也不是 process cache 序号。",
        "issue_status": "指定旧报告已讨论 MLA/Sparse 外部边界及跨调用状态；capture-invocation-boundary 与 conditional-flow-topology 也已有归档。未把它们宣称为本轮首次发现的未提报问题。N2/N3 为本轮规则澄清建议，关键词查找无匹配不能证明全仓从未提报。",
        "evidence": [evidence(Y["extract_mla"], 320, 338), evidence(S["mla"], 1669, 1681),
                     evidence(S["mla"], 1895, 1898), evidence(S["mla"], 2007, 2011),
                     evidence(B["hidden"], 299, 311), evidence(B["hidden"], 315, 326),
                     evidence(S["sparse"], 6153, 6163), evidence(S["sparse"], 6188, 6191),
                     evidence(Y["extract_gdn"], 379, 383), evidence(S["gdn"], 3254, 3264),
                     evidence(Y["extract_moe"], 372, 374)],
    },
]

naming = {
    "method": "设计者从覆盖范围、术语冲突、长度和可能误读比较，未做用户可用性测试或词频实验。保留公共规范的单数字段风格。",
    "choices": [
        {"current": "resource", "selected": "storage", "definition": "Named on-chip allocations and their views for this worker.",
         "candidates": [
             {"name": "resource", "length": 8, "assessment": "现名宽于实际内容，易让人期待 Tensor Core、barrier 或 occupancy 资源也在此登记。"},
             {"name": "storage", "length": 7, "assessment": "覆盖 SMEM/TMEM/register 及长期状态，惯用且直接；需在定义中明确片上与 worker 范围。与 handoff.storage 的职责相关，路径层级可区分。"},
             {"name": "buffer", "length": 6, "assessment": "更短、GPU 中常见，但易使人只想到搬运数组或暂存，不自然涵盖寄存器标量、状态和混合解释区。"}]},
        {"current": "domain", "selected": "index", "definition": "Named iteration indices and ordered sequences.",
         "candidates": [
             {"name": "domain", "length": 6, "assessment": "数学上准确，但孤立初读时用途较泛；不熟悉索引域的读者不易猜到。"},
             {"name": "index", "length": 5, "assessment": "直接提示键是索引名，兼容整数范围、worktile 与 flatten；定义必须强调范围/序列声明而非当前可变游标。"},
             {"name": "range", "length": 5, "assessment": "整数半开区间容易理解，但 worktile 与 flatten 是有序序列，且上游已有 range 表示规模允许范围，冲突更大。"}]},
        {"current": "milestone", "selected": "event", "definition": "Names for existing local events, used by cross-phase handoffs.",
         "candidates": [
             {"name": "milestone", "length": 9, "assessment": "英语常见，但易联想到重要进度或阶段末点，对消费等待、中段释放、单事件略带多余暗示。"},
             {"name": "event", "length": 5, "assessment": "贴近 issue/done 引用，单事件和复合事件都自然；须定义为静态引用命名而非创建 CUDA event/notification。"},
             {"name": "marker", "length": 6, "assessment": "可表达标记任意位置，但对多个角色的 AND 组及就绪保证不如 event 明确，也容易像无语义的调试标记。"}]},
    ],
    "migration": "只替换三个 phase 字段名及对应的资源引用路径，保留条目结构、事件端点、索引形式和所有必要关系。storage/index/event 并非新的执行指令。",
}

sketches = [
    {
        "id": "S1", "operator": "moe", "source": B["direct"],
        "label": "建议片段；现名版已将旧资源字段按当前规则归一化，两版仅改 resource 名与引用路径。",
        "before": """flow:
  launch: [{phase: [prelude, direct], order: per_worker}]
  handoff:
    Census: {producer: prelude, consumer: [direct], storage: tile_pipeline.phase.prelude.resource.Census}
phase:
  prelude:
    resource:
      Census: {depth: 1, location: SMEM, shape: [], dtype: int32, total_size: 4}
      Exchange:
        depth: 1
        location: SMEM
        shape: [element: 16]
        dtype: int32
        total_size: 64
        view:
          WarpCounts: {}
          FinalRoutes: {part: {element: [0, K]}}
""",
        "after": """flow:
  launch: [{phase: [prelude, direct], order: per_worker}]
  handoff:
    Census: {producer: prelude, consumer: [direct], storage: tile_pipeline.phase.prelude.storage.Census}
phase:
  prelude:
    storage:
      Census: {depth: 1, location: SMEM, shape: [], dtype: int32, total_size: 4}
      Exchange:
        depth: 1
        location: SMEM
        shape: [element: 16]
        dtype: int32
        total_size: 64
        view:
          WarpCounts: {}
          FinalRoutes: {part: {element: [0, K]}}
""",
        "preserves": "flow 属于 work_schedule，phase 存储片段属于 tile_pipeline；分区包装与角色动作省略，非完整提交稿。direct 使用前一 phase 的 Census 与 Exchange.FinalRoutes，不重复分配。",
        "evidence": [evidence(B["direct"], 188, 199), evidence(B["direct"], 291, 306), evidence(B["direct"], 372, 375)],
    },
    {
        "id": "S2", "operator": "fmha", "source": Y["extract_fmha"],
        "label": "建议片段；从 root 的 g/z 和 Key 视图抽取，不增加动作、通知或字段结构。",
        "before": """domain:
  g: [cLo, 'cHi+1']
  z: [0, 2]
resource:
  Key:
    depth: 2
    location: SMEM
    shape: [key: KeyGroup, channel: D]
    dtype: fp8_e4m3fn
    total_size: '2*KeyGroup*D'
    view: {'Key(g)': {slot_index: '(g-cLo)%2'}}
""",
        "after": """index:
  g: [cLo, 'cHi+1']
storage:
  Key:
    depth: 2
    location: SMEM
    shape: [key: KeyGroup, channel: D]
    dtype: fp8_e4m3fn
    total_size: '2*KeyGroup*D'
    view: {'Key(g)': {slot_index: '(g-cLo)%2'}}
""",
        "local_loop_change": "同一 role 的现行语法转写中，仅将内层 for(z) 换成 for(z=0:2)，原内层完整动作列表原位保留。外层 for(g)、两次 ScoreHalf 计算/读取顺序、原语读完屏障均不变。",
        "preserves": "单 role 仍可以有 phase index，因为 g 为资源版本映射提供共同含义；z 的删除只消去一次局部整数域登记。",
        "evidence": [evidence(Y["extract_fmha"], 462, 476), evidence(Y["extract_fmha"], 644, 657), evidence(S["fmha"], 4706, 4712)],
    },
    {
        "id": "S3", "operator": "gdn", "source": Y["extract_gdn"],
        "label": "建议片段；将旧 join_prep 动作按当前八 verb 转写为 await:Prepared 后，对比的唯一差异是 milestone→event。",
        "before": """chain:
  milestone:
    prep_ready: ['{done MAIN[0]::await:Prepared}']
""",
        "after": """chain:
  event:
    prep_ready: ['{done MAIN[0]::await:Prepared}']
""",
        "handoff_unchanged": "Ti: {producer: prep, consumer: [chain], storage: global, ready: prep.kernel_done, wait_at: chain.prep_ready}",
        "preserves": "同一个已有 PDL 等待的 done，在 initial_state/PRE0 独立输入准备之后。其余七个 handoff 使用同一名称。没有新通知、没有产物别名、没有把 wait_at 移到入口或 kernel 尾部。",
        "evidence": [evidence(Y["extract_gdn"], 330, 338), evidence(Y["extract_gdn"], 550, 553), evidence(Y["extract_gdn"], 601, 609)],
    },
]

bundle = {"coverage": coverage, "findings": findings, "naming": naming, "sketches": sketches}
for item in findings + sketches:
    for ev in item["evidence"]:
        actual = evidence(ev["path"], ev["start_line"], ev["end_line"])
        assert actual == ev
        assert ev["quote"] in (ROOT / ev["path"]).read_text()

out = ROOT / "schema_discussion/phase_grammar/data/05_names_minimality.json"
out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"written": str(out.relative_to(ROOT)), "findings": len(findings),
                  "roots": len(roots), "branches": len(branches),
                  "sources_hash_checked": len(sources),
                  "evidence_quotes_checked": sum(len(f["evidence"]) for f in findings + sketches)}, ensure_ascii=False))
