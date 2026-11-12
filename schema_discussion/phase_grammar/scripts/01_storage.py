#!/usr/bin/env python3
"""Build the storage audit evidence file; verify fixed-input hashes and quotes.

This is an evidence check, not a schema validator or a GPU test.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "schema_discussion/phase_grammar"
corpus = json.loads((BASE / "corpus.json").read_text())
paths = {
    "rule5": "schema_discussion/phase_grammar/sources/rules/5.design.tile_pipeline.md",
    "rule4": "schema_discussion/phase_grammar/sources/rules/4.design.work_schedule.md",
    "rule6": "schema_discussion/phase_grammar/sources/rules/6.realization.md",
    "gdn_cu": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu",
    "gdn_h": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_common.cuh",
    "gemm_cu": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu",
    "mla_cu": "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py",
    "sparse_cu": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu",
    "fmha_cu": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh",
    "fmha_host": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu",
    "moe_cu": "schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu",
}
for key, op in corpus["operators"].items():
    paths[op["name"]] = op["root"]
paths["moe_direct"] = next(n["snapshot"] for n in corpus["operators"]["extract_moe"]["nodes"] if n["node"] == "direct")
paths["fmha_xpose"] = next(n["snapshot"] for n in corpus["operators"]["extract_fmha"]["nodes"] if n["node"] == "tcg_d128_flat_tma_xpose_narrow")

expected = {r["snapshot"]: r["sha256"] for r in corpus["rules"]}
source_paths = set()
for op in corpus["operators"].values():
    for node in op["nodes"]:
        if node["snapshot"]:
            expected[node["snapshot"]] = node["sha256"]
    for source in op["sources"]:
        expected[source["path"]] = source["sha256"]
        source_paths.add(source["path"])


def ev(key, first, last=None, quote=None):
    path = paths[key]
    data = (ROOT / path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == expected[path], (path, "fixed input hash changed")
    last = first if last is None else last
    fragment = "\n".join(data.decode().splitlines()[first - 1:last])
    quote = fragment if quote is None else quote
    assert quote and quote in fragment, (path, first, last, quote)
    result = {"path": path, "start_line": first, "end_line": last, "quote": quote}
    if path in source_paths:
        result["sha256"] = digest
    return result


findings = [
    {
        "id": "ST01", "operator": ["mla", "gdn", "moe"],
        "category": "existing_structure",
        "claim": "slot_index、part、dtype 分别决定版本落槽、别名区域和字节解释；三者不是重复信息。MLA 的 P(n) 位于每行第 32..64 个 FP32 单元对应的字节内，不能从三槽、P 的逻辑形状或复用等待推出。GDN 的混合类型 stage 和 MoE 的三槽 payload／两槽 scale 同样需要明确映射。",
        "recommendation": "保留 view 的三项语义；slot_index 可缩名为 slot，但不加 n%depth 默认值。单槽、整槽、同类型且与上游值同名的资源沿用现有隐式绑定；只有改名能使 Maximum→M、Denominator→D 这类单值资源省掉空 view，异地副本和真正别名继续保留。",
        "other_definitions": {"domain": "共享索引提供 n 的身份与边界，不决定 n 落哪个槽。", "milestone": "不参与静态区域解释；事件和 waiting 另外保证覆盖安全。"},
        "evidence": [
            ev("mla", 387, 406),
            ev("mla", 413, 424),
            ev("mla_cu", 464, 470),
            ev("mla_cu", 1090, 1100),
            ev("gdn", 571, 586),
            ev("gdn_cu", 1902, 1907),
            ev("gdn_h", 64, 65),
            ev("moe", 639, 668),
            ev("moe_cu", 4908, 4921),
            ev("moe_cu", 7765, 7778),
            ev("rule5", 362, 386),
        ],
    },
    {
        "id": "ST02", "operator": ["sparse"],
        "category": "existing_structure",
        "claim": "Sparse 的同一 arena 在开头放 Q、稳态放 Gather 的多槽环、末尾放 FP32 partial；整个 ring 以 depth=1 的字节区域加 part 表达是有意义的选择。末尾区域可以跨原先的 KV 槽边界，不能机械改成 Stages 份独立资源。现有 part 已能表达这个真实关系，未发现需要另加 alias、arena、region_map 字段。",
        "recommendation": "保留一个 backing 区域和三个明确的 view，计容量一次；说明 depth 是所选分配单位的槽数，不必等于每个阶段的逻辑缓冲深度。保留 Q 读完后允许首次 KV 覆盖、最后全部 KV 读完后允许 partial staging 的交接；swizzle 和地址实现不进入 view。",
        "other_definitions": {"domain": "c 与 c0 决定当前 Gather 区域；声明本身不推进 c。", "milestone": "此例在一个 role 内，不必增加跨 phase 事件组。"},
        "evidence": [
            ev("sparse", 385, 393),
            ev("sparse", 444, 455),
            ev("sparse", 486, 498),
            ev("sparse_cu", 878, 884),
            ev("sparse_cu", 1162, 1173),
            ev("sparse_cu", 1424, 1435),
            ev("sparse_cu", 1452, 1465),
        ],
    },
    {
        "id": "ST03", "operator": ["gemm", "fmha", "sparse"],
        "category": "concise_semantics",
        "claim": "暂存寄存器不能一律当编译器临时量删除。GEMM 的双槽 DrainRegister 使后一次读与当前 FMA 重叠，并使全部 TMEM 读完后、最后一次 FMA 前归还 ProductRing；FMHA 的 QueryPrefetch 把 Q 读取提前到 TMEM 分配前；Sparse 的 Q_cached 按 key-group 复制并跨整段 recurrence 存活。这些分别改变重叠、生命周期或容量。",
        "recommendation": "保留上述设计级副本、槽数和必要的角色绑定；动作说明只点名从哪个 view 读入哪个副本、何时源可归还。普通 fma 中间标量、转换打包变量、barrier 存储、指令寄存器编号不建资源条目。role.register 是总预算，不能再把资源的逻辑寄存器字节额外加到它上面。",
        "other_definitions": {"domain": "GEMM 的 d 控制 Fragment 的双缓冲，n 控制 TMEM 产品双缓冲，不能互相代替。", "milestone": "归還边界应是局部 release/done；只有上游要引用才另导出事件组。"},
        "evidence": [
            ev("gemm", 315, 322),
            ev("gemm", 426, 446),
            ev("gemm", 481, 489),
            ev("gemm_cu", 592, 606),
            ev("gemm_cu", 653, 665),
            ev("fmha", 548, 555),
            ev("fmha_cu", 3278, 3299),
            ev("fmha_cu", 3316, 3320),
            ev("sparse", 395),
            ev("sparse", 453, 455),
            ev("sparse_cu", 1162, 1173),
            ev("rule5", 352, 358),
        ],
    },
    {
        "id": "ST04", "operator": ["moe"],
        "category": "existing_structure",
        "claim": "MoE/direct 在同一次 per_worker 提交中把 Census/Exchange 登记在 prelude，direct 读取 Census 并把 Exchange.WarpCounts 已用完的区域改放 FinalRoutes，证明单 phase 登记并不意味着单 phase 寿命。相反，root 的 fc1/fc2 是不同 kernel；YAML alias 只复制定义，不给它们同一物理存储身份。",
        "recommendation": "保留现有 launch.order + handoff.storage + 一处资源/view 定义，必要读写在动作 describe 绑定，不新增 resource.scope、owner_phase 或跨 kernel 的同名共享约定。可在资源节增加通往现有跨 phase 条款的短链接。",
        "other_definitions": {"domain": "direct 的 worktile 序号属于后续工作，不延长旧循环索引的作用域。", "milestone": "Census 使用 handoff 默认就绪与入口等待；不需要为默认交接强建 milestone。"},
        "evidence": [
            ev("moe_direct", 188, 199),
            ev("moe_direct", 291, 306),
            ev("moe_direct", 372, 375),
            ev("moe_cu", 1929, 1932),
            ev("moe_cu", 1960, 1976),
            ev("moe_cu", 2218, 2237),
            ev("moe", 375, 395),
            ev("moe", 850, 870),
            ev("rule5", 492, 500),
        ],
    },
    {
        "id": "ST05", "operator": ["fmha", "gemm"],
        "category": "existing_structure",
        "claim": "total_size 按现行规则完全由 depth×单槽字节×持有 worker 成员数确定，是校验/展示值，没有独立设计选择。FMHA root 登记的 SMEM 值容量为 83968 字节，源码初始动态 SMEM 请求为 102912 字节，之后还可能按 occupancy 查询增大；二者不能混称 kernel 总占用或驻留证明。六 root 使用旧字段，没有手填 total_size，不能据旧稿统计其填写错误率。",
        "recommendation": "倾向把 total_size 从作者必填改成工具派生展示，或至少允许省略且填写时严格校验；若保留短字段名可用 bytes。仍按 location 分开、按 worker 成员计，view 不再计量；要保留实现预留与寄存器预算的独立核对，不把多 CTA 合计拿去比单 SM 容量。无需另加 slot_size。",
        "other_definitions": {"domain": "不参与静态字节等式；不能按总循环长度重复计槽。", "milestone": "事件决定存续和复用，不提供实际占用测量。"},
        "calculation": {
            "fmha_registered_smem": "Query 16384 + Key 2*16384 + Value 2*16384 + KeyAtom 2*512 + ValueAtom 2*512 = 83968",
            "fmha_initial_dynamic_request": "83968 + unused P 16384 + unused Q-scale 512 + exchange reservation 1024 + alignment allowance 1024 = 102912",
            "limitation": "纯算术复核固定源码的初始请求；不包含全部静态 SMEM、动态调整或实测寄存器，不作 GPU 驻留测量。"
        },
        "evidence": [
            ev("rule5", 338, 350),
            ev("fmha", 466, 494),
            ev("fmha", 685, 702),
            ev("fmha_cu", 1822, 1852),
            ev("fmha_host", 818, 833),
            ev("fmha_host", 1183, 1195),
            ev("gemm", 273, 300),
            ev("gemm_cu", 124, 129),
            ev("gemm_cu", 315, 325),
        ],
    },
    {
        "id": "ST06", "operator": ["mla", "moe", "fmha"],
        "category": "underspecified_rule",
        "claim": "建议澄清“片上分配账”的粒度：设计上独立占用的区域不必逐一对应分配 API 调用。MLA 源码只申请一次 512 列 TMEM，根稿把不重叠的 S/P 区与 Y 区分别登记；强求 API 一次一条会把简单的三槽结构改写成更多偏移表达式。真实实现的 padding/reservation 也不等于新的数学值：MoE/direct 的 InputCache 含 H/8 未用 padding，而 MLA TinyY 逻辑量没有抄入源码 130 的行距。反过来，MoE root fc1 仅用 192 列却按有效分配粒度申请 256 列 TMEM，不能把这 64 列削掉后声称降低了占用。",
        "recommendation": "补一句：登记设计占用的区域，别名只记一次；互不重叠的子区可以分别列账，API 打包与线程布局由 realization/Coder 处理。纯布局余量不新增 view/伪数学值；若它显著改变容量或驻留（FMHA 的大块预留），在 realization 保留简短的实际请求/约束和理由。已决定的最小有效分配粒度仍决定资源容量，例如 MoE TMEM 的 256 列。不要把未知编译器复用假定为资源已共用，也不要把各逻辑值生命周期的简单求和声称为物理寄存器峰值。",
        "other_categories": ["implementation_detail", "other_layer"],
        "classification_note": "需要补清的是账的单位，不是新分配能力。barrier 编号、纯布局余量属实现；影响实际驻留的请求/兼容条件属 realization。旧稿包含 padding 并不自动证明它违反当前规则或必须加字段。",
        "other_definitions": {"domain": "局部/共享索引不负责物理地址或 allocation API 身份。", "milestone": "实际存续与读者交接仍由动作/依赖表达，不增加分配生命周期事件表。"},
        "evidence": [
            ev("mla", 387, 400),
            ev("mla_cu", 464, 470),
            ev("mla_cu", 817, 819),
            ev("mla_cu", 844),
            ev("moe_direct", 324, 331),
            ev("moe_direct", 349, 353),
            ev("moe_cu", 1924, 1928),
            ev("mla", 443, 448),
            ev("mla_cu", 1080, 1084),
            ev("moe", 669, 676),
            ev("moe_cu", 4349, 4367),
            ev("moe_cu", 7765, 7778),
            ev("fmha", 688, 702),
            ev("rule5", 350, 358),
            ev("rule5", 403, 407),
            ev("rule6", 5, 13),
        ],
    },
    {
        "id": "ST07", "operator": ["gdn", "moe", "mla"],
        "category": "other_layer",
        "claim": "跨调用 workspace/cache 的真实身份、初始化和复用条件不能丢，但不能自动变成 docs5 的存储声明缺口。GDN 的 grow-only global scratch、MoE/direct 的自清零 RouteAccum/Arrivals、MLA 的跨调用 queue word，都超出了 worker 内的片上分配范围。",
        "recommendation": "保留短的全局物化、当前版本/初值和顺序复用约定，归到数学、work_schedule 或调用边界；源码没有的跨 stream 保证如实保留。不要给 storage 添加 process/cache/device-key/allocator 字段。本轮不提议新的跨调用语法。",
        "other_definitions": {"domain": "单次 worker 的工作序号不是 invocation/cache epoch。", "milestone": "局部事件组不能凭空提供另一次调用的排他或外部 stream 同步。"},
        "evidence": [
            ev("gdn", 330, 338),
            ev("gdn", 379, 383),
            ev("gdn_cu", 3254, 3274),
            ev("moe_direct", 126, 145),
            ev("moe_cu", 2239, 2256),
            ev("mla", 342, 352),
            ev("mla", 626, 633),
            ev("mla_cu", 1456, 1466),
            ev("mla_cu", 1730, 1737),
            ev("rule4", 288, 311),
        ],
    },
    {
        "id": "ST08", "operator": ["gdn", "fmha"],
        "category": "source_or_draft_issue",
        "claim": "别名区域已能声明，不等于覆盖顺序已经安全。GDN final mini-chunk 在 stage 0 仍被其他 warp 读 Kc 时可写 FinalState；FMHA 空 split 在 Query 写者完成前可覆写同一 arena。原稿把问题写成长说明，主因是来源保证或旧动作边界，并非 view 无法表达共用。",
        "recommendation": "保留来源限制，使用已命名的读完/准备完成事件和 waiting.released 表达可证明的设计前驱；不能用新增 resource 字段、默认初始化或一条 milestone 假装源码已同步。FMHA 分支已有 prepare→stage_partials 的设计边；GDN 若要改为完整安全设计需先明确对应读者完成边界，不能在此次研究中擅自修源或认证。",
        "other_definitions": {"domain": "首尾/空域决定哪些读写存在；不可用版本索引掩盖不同 warp 的先后。", "milestone": "事件组只命名现有事件，不能创造缺少的 join 或可见性。"},
        "evidence": [
            ev("gdn", 577, 586),
            ev("gdn", 637, 659),
            ev("gdn_cu", 2499, 2511),
            ev("gdn_cu", 2543, 2552),
            ev("gdn_cu", 2819, 2839),
            ev("fmha_xpose", 503, 512),
            ev("fmha_xpose", 604, 612),
            ev("fmha_xpose", 642, 649),
            ev("fmha_cu", 5230, 5246),
            ev("fmha_cu", 3346, 3365),
            ev("fmha_cu", 5198, 5204),
            ev("fmha_cu", 5388, 5396),
            ev("fmha_cu", 5412, 5424),
            ev("rule5", 405, 407),
        ],
    },
]

naming = {
    "resource": {
        "recommended": "storage",
        "why": "直接表示存储及其区域，覆盖 SMEM、TMEM、register 中的值、状态与缓冲；不暗示 malloc/API 粒度。使用权由 acquire/release/依赖约束这份 storage 或区域，不应塞进静态账的新字段。",
        "candidates": [
            {"name": "storage", "assessment": "首选；范围最接近登记内容，常用，七个字母。"},
            {"name": "buffer", "assessment": "易懂且短，但容易让人只想到搬运缓冲，对寄存器累计状态和 TMEM 分区需多解释。"},
            {"name": "allocation", "assessment": "强调容量与身份，但容易误解成真实分配调用或动态生命周期对象；比 resource 更长。"},
            {"name": "resource", "assessment": "太宽，容易把 warp、引擎、barrier、分配许可也装进同一表。"},
        ],
        "permit_boundary": "FMHA 在 prepare 中 relinquish allocation permit 后继续使用 TMEM，到 finish 才 deallocate；许可归还、区域读者归还、物理销毁必须区分。",
        "evidence": [ev("fmha", 548, 555), ev("fmha", 636, 643), ev("fmha_cu", 3316, 3320), ev("rule5", 127, 132)],
    },
    "domain": {
        "recommended": "index",
        "why": "本表登记共享索引的身份和取值规则；index 比 domain 更容易猜中用途，也覆盖 worktile 序号与 flatten。",
        "candidates": [
            {"name": "index", "assessment": "首选；要在定义中说明这里是具名索引表，不是一次下标值。"},
            {"name": "range", "assessment": "最容易理解整数区间，但对动态 worktile 与 flatten 含义偏窄。"},
            {"name": "domain", "assessment": "数学上准确，初读容易与算子适用域/输入范围混淆。"},
        ],
        "boundary": "不生成循环，不分配 worktile，不定义物理槽，不要求各 role 锁步。",
        "evidence": [ev("mla", 364, 380), ev("rule5", 318, 334)],
    },
    "milestone": {
        "recommended": "event",
        "why": "推荐短而惯用的 event，但规范必须首句说明它是对外可引用的复合事件名，成立条件为列表中全部已有事件发生。",
        "candidates": [
            {"name": "event", "assessment": "首选；短，和 ready/wait_at 引用自然衔接，需防止误读为新建 CUDA event。"},
            {"name": "milestone", "assessment": "保留也可；里程碑的聚合含义不错，但不像内核/依赖术语那么直观。"},
            {"name": "event_group", "assessment": "更明确暴露复合性，但更长；若最看重无歧义，可选它。"},
            {"name": "ready", "assessment": "不推荐；消费者 wait_at 也可能引用 issue，不能都叫数据就绪。"},
        ],
        "boundary": "只命名现有事件组，不产生同步、数据可见性、内存或跨调用排他；不要求每个 phase 填写。",
        "evidence": [ev("gdn", 552, 553), ev("gdn", 330, 338), ev("rule5", 498, 500), ev("rule4", 290, 295)],
    },
    "view": {
        "recommended": "保留 view，slot_index 可改 slot",
        "why": "view 是惯用存储术语；part 已短，dtype 明确。压缩可省项和冗余单值别名，避免把三项编码为难读字符串。",
    },
}

sketches = [
    {
        "id": "SK01", "operator": "mla", "status": "proposal_not_current_schema",
        "changes": ["resource→storage", "domain→index", "slot_index→slot", "total_size 改为派生展示", "Maximum/Denominator 改用上游 M/D 名，省掉单槽空 view"],
        "preserved": "同一连续 n 域、三槽 S/P 共用、每行的 FP32 字节区域和 FP8 解释；ScoreRegisters、PackedRegisters 和原有等待不在片段中展示，仍保留。",
        "yaml": """index:
  w: worktile
  t: [0, "N(w)"]
  n: {flatten: [w, t]}
storage:
  ScoreProbability:
    depth: 3
    location: TMEM
    shape: [row: Bq, column: Bk]
    dtype: fp32
    view:
      "S(n)": {slot: "n%3"}
      "P(n)": {slot: "n%3", part: {column: [32, 64]}, dtype: fp8_e4m3fn}
  M: {depth: 1, location: register, shape: [query_token: Bq], dtype: fp32}
  D: {depth: 1, location: register, shape: [query_token: Bq], dtype: fp32}
""",
        "evidence": [ev("mla", 364, 394), ev("mla", 413, 424)],
    },
    {
        "id": "SK02", "operator": "sparse", "status": "proposal_not_current_schema",
        "changes": ["resource→storage", "total_size 改为派生展示", "保留一个字节 arena，展示不能强行删去 part 的反例"],
        "preserved": "Q/Gather/FP32 staging 同一容量，Gather 按 c-c0 轮转，PartialStage 可以跨原 KV 槽；不增加存储或隐含同步。",
        "yaml": """storage:
  ring:
    depth: 1
    location: SMEM
    shape: [byte: "2*Stages*KC*D"]
    dtype: uint8
    view:
      "Gather(c)": {part: {byte: ["2*((c-c0)%Stages)*KC*D", "2*(((c-c0)%Stages)+1)*KC*D"]}, dtype: bf16}
      Q_ring: {part: {byte: [0, "2*HB*D"]}, dtype: bf16}
      PartialStage: {part: {byte: [0, "4*HB*D"]}, dtype: fp32}
""",
        "description_sketch": "prepare: When Q uses ring slot zero, read ring.Q_ring into Q_cached and join all query readers before reusing the ring. finish: After all Gather reads and fills retire, write Acc to ring.PartialStage and publish it before the global copy.",
        "description_limit": "只替换这两个动作涉及 ring 的句子，保留其他输入、控制、发布与数学绑定；不是整条动作契约的完整替代。",
        "evidence": [ev("sparse", 385, 393), ev("sparse", 444, 455), ev("sparse", 486, 498)],
    },
    {
        "id": "SK03", "operator": "moe/direct", "status": "proposal_not_current_schema",
        "changes": ["resource→storage，引用路径同步改名", "total_size 改为派生展示", "其他设计关系沿用已有结构，没有新 scope 字段"],
        "preserved": "一批 worker 先 prelude 后 direct；Census 片上交接；Exchange 的第二用途不重复登记存储。direct 的 Last 条件、动作序列和跨 CTA 原子协议仍保留。",
        "yaml": """work_schedule:
  flow:
    launch: [{phase: [prelude, direct], order: per_worker}]
    handoff:
      Census: {producer: prelude, consumer: [direct], storage: tile_pipeline.phase.prelude.storage.Census}
tile_pipeline:
  phase:
    prelude:
      storage:
        Census: {depth: 1, location: SMEM, shape: [], dtype: int32}
        Exchange:
          depth: 1
          location: SMEM
          shape: [element: 16]
          dtype: int32
          view:
            WarpCounts: {}
            FinalRoutes: {part: {element: [0, K]}}
""",
        "description_sketch": "direct.Main gather: Write final route indices to Exchange.FinalRoutes after prelude has consumed Exchange.WarpCounts; join the CTA before reading those indices.",
        "evidence": [ev("moe_direct", 188, 199), ev("moe_direct", 291, 306), ev("moe_direct", 372, 375)],
    },
]

all_nodes = [n for o in corpus["operators"].values() for n in o["nodes"]]
roots = []
notes = {
    "gdn": "重点阅读 Design 与 realization，核对 stage、rhs/output 别名、mini 尾部和全局 workspace。",
    "gemm": "阅读 dispatch 与 Design/realization，核对三槽输入、两槽 TMEM、寄存器读出和提前归还。",
    "mla": "阅读 Design/realization，核对 S/P 字节重解释、多种连续环、geometry 和输出副本。",
    "sparse": "阅读 Design/realization，核对 Q/Gather/PartialStage 复用、Q 副本和跨 kernel global partials。",
    "fmha": "阅读 Design/realization，核对 TMEM 分区、Q 预取、半幅 score、SMEM 额外请求。",
    "moe": "阅读 Design/realization，核对 packed payload/scale 环、寄存器片段、分 kernel 身份。",
}
for op in corpus["operators"].values():
    node = next(n for n in op["nodes"] if n["node"] == "root")
    roots.append({"operator": op["name"], "path": node["snapshot"], "sha256": node["sha256"], "status": node["category"], "basis": "旧审阅 approved；不表示已认证当前 0–6 规则", "review_scope": notes[op["name"]]})

coverage = {
    "method": "schema_discussion/phase_grammar/METHOD.md",
    "corpus": "schema_discussion/phase_grammar/corpus.json",
    "fixed_population": {
        "planned_nodes": len(all_nodes),
        "category_counts": dict(collections.Counter(n["category"] for n in all_nodes)),
        "authored_design_material": sum(n["semantic_eligible"] for n in all_nodes),
        "material_without_authored_design": sum(n["material_present"] and not n["semantic_eligible"] for n in all_nodes),
        "not_started": sum(n["category"] == "not_started" for n in all_nodes),
        "not_a_semantic_review_count": True,
    },
    "roots": roots,
    "branches": [
        {"operator": "moe", "node": "direct", "path": paths["moe_direct"], "status": "approved", "reviewed_lines": [1, 421], "reason": "同 worker 跨 phase 存储复用的反例样本"},
        {"operator": "fmha", "node": "tcg_d128_flat_tma_xpose_narrow", "path": paths["fmha_xpose"], "status": "approved", "reviewed_lines": [492, 657], "reason": "只深入审阅 split 资源/动作/等待，检验别名区域与空 split 读写边界"},
    ],
    "lexical_only_branch_searches": [
        "全体有 YAML 节点中的跨 phase 资源引用/多成员关键词扫描；不计为语义审阅",
        "gdn/segmented、moe/fused_small、sparse/tc_heads、sparse/tc_packed 的资源定位扫描；未作为完整分支结论样本",
    ],
    "prior_reports_read": [
        {"path": corpus["reports"][0]["path"], "ranges": [[1, 9], [80, 89], [174, 240]], "use": "复用文字/结构/来源限制三分法和定位，关键事实回到 YAML/源码"},
        {"path": corpus["reports"][1]["path"], "ranges": [[1, 231]], "use": "重点复核动作说明、外部边界与跨调用状态，未把旧提案当现有字段"},
    ],
    "rules_read": [paths["rule5"], paths["rule4"], paths["rule6"], "schema_discussion/phase_grammar/sources/rules/0.common.md"],
    "source_review": "以下源码仅审阅存储、使用权、关键读写与调度所需段落；搜索命中不代表逐行全文件审阅。MLA 的 CUDA 内嵌在 solution.py。",
    "sources": [{"path": p, "sha256": expected[p], "basis": "corpus 固定源码；无新增源码文件"} for p in paths.values() if p in source_paths],
    "limits": [
        "未运行 GPU，未重新提取算子，未按新字段校验旧稿。",
        "所审六 root 与两处分支的 worker 都是单 CTA；未用这些样本认证 resource.member 的部分成员或异构成员容量规则。现有 member 未发现可删依据。",
        "未发现本层必须新增字段的 missing_capability；结论仅限本次有据样本。",
        "未读取其他独立代理报告，未修改正式资料或运行状态。",
    ],
}

result = {"coverage": coverage, "findings": findings, "naming": naming, "sketches": sketches}
out = BASE / "data/01_storage.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def walk_evidence(obj):
    if isinstance(obj, dict):
        if all(k in obj for k in ["path", "start_line", "end_line", "quote"]):
            yield obj
        for value in obj.values():
            yield from walk_evidence(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_evidence(value)


count = 0
for evidence in walk_evidence(result):
    content = (ROOT / evidence["path"]).read_text().splitlines()
    fragment = "\n".join(content[evidence["start_line"] - 1:evidence["end_line"]])
    assert evidence["quote"] in fragment
    count += 1
assert coverage["fixed_population"]["authored_design_material"] == 172
assert coverage["fixed_population"]["material_without_authored_design"] == 5
assert coverage["fixed_population"]["not_started"] == 43
assert 16384 + 2 * 16384 + 2 * 16384 + 2 * 512 + 2 * 512 == 83968
assert 83968 + 16384 + 512 + 1024 + 1024 == 102912
print(f"Wrote {out.relative_to(ROOT)}; checked {count} evidence quotes, fixed-input hashes and two footprint calculations.")
