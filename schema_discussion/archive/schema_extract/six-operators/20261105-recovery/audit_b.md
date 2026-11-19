# KernelSchema 六算子恢复与质量独立审计 B

> 归档说明：六算子恢复阶段的历史记录。正文中的运行状态、预算、命令和分工对应记录时点，不作为新的执行授权。 [归档索引](../../README.md)。

审计日期：2026-11-05 UTC。审计者：独立子代理 B。根目录：`/home/shaohanyong.shy/KernelSchema`。本报告只提出方案，未启动提取、恢复、修订、运行层升级或发布；未改 schema/kernel/规则/runtime/Git。没有读取另一位审计者的报告，也没有联系或调用其他子代理。

## 结论

六项不能统一选择“重提取”，也不能统一选择“旧稿直接保留”。首选分流如下：

| 算子 | 首选内容动作 | 为什么 | 正式运行处理 |
|---|---|---|---|
| fp8_scaled_grouped_gemm | **patch** 2.2.4 校正版的 root 成员表达，三节点独立 review | 数值、分块、缓存和 graph 错误范围描述有较高复用价值；仍有一个明确的角色成员声明错误 | 已完成 run 不可 revise/full-restart；旧发布保留。要新正式回执须新目录 extract；工具没有 v2 已发布稿导入并仅重审入口 |
| fmha_mxfp8_qkv | **review + 保稿恢复** 现有 43 节点计划；按问题局部 patch | 新稿已实际修复旧八节点的主要条件拓扑问题；29 节点有效，另外14节点也均有写作历史，重做损失大 | 先合法结算终止 fence；旧语义桥接 executor → 当前规则 semantic revision → 按依赖恢复 → 当前 P6 → 新目录 export |
| mla_expanded_fp8 | **patch** Contract 的三个 nullable 字段及三个下游绑定，随后完整单节点 review | 核心数值路径保留了容易误读的 tiny 额外计算、gain=256、半精度分母等，没必要重写设计 | 已发布，不接受 revise；新目录人工补丁复审不能冒充 Runner 发布；需要正式新发布则新 extract |
| dual_cache_latent_bf16 | **review + 保稿恢复** 现有八节点计划；P1 任务域需重点复核 | 新 root/mma_direct 已拆开 combine；tc_heads 有较完整草稿；控制槽重叠是源缺陷，当前规范允许保留证据后继续抽象提取 | 先结算 fence；桥接旧语义 executor 后 semantic revision；先 P1–P4 review，再父先子后完成六个未有效节点，最后 P6 |
| gdn_prefill_bf16_state_bf16 | **keep 2.2.7 内容，必要时 review**；不恢复旧两节点 | 六节点已经针对输入域、缓存/直接 QK、分段和 Perm 做过实质修正；抽查未发现必须重写的设计问题 | ready + 正式 export 已完成；不再 resume。若要求全树按2.5.0重新背书，应新目录重审/新 extract |
| nvfp4_fused_moe | **保留默认 scope 后继的 P1/P2 与已有 root 草稿；review 后继续 P2–P6** | 旧60节点不只有冗余，更混入条件删提交/删成员；默认 scope 后继已识别 graph/ABI/验证的真实提交实例，重启会丢掉这些工作 | 最新 run 只 P1 有效、P2 reviewer 因配额错误停止；先桥接及 semantic revision。若扩大至非默认控制，必须另建 source scope/output，不能在此 run 偷扩 |

这里的 keep 是保留内容和历史；本审计**没有**给任何旧批准出具满足 `covered_changes/covered_issues` 的 adoption，也不声称完成当前2.5.0的 P6。特别是 FMHA/MoE 大树，本次做的是恢复决策审计和有针对性的内容核验，不是全有效全文终审。

## 规则与工具边界

已读取 `EXTRACT_SCHEMA.md`、`schema_extract/README.md`、`REVIEW.md`、`tools/README.md`、六阶段页，以及 schema_docs 0/1/3/4/5/6 的相关完整字段与语义章节。实际 `python3 -B schema_extract/extract.py --version` 返回 **2.5.0**；Git HEAD 为 `63e4376`。

关键约束据当前文件而非历史总结：

- `schema_extract/README.md` 的“三种冻结身份”区分 source/config、rules、executor。普通 resume 固定原规则，工作树更新不作用于旧 run；source/任务/config/check_policy 不能靠 semantic revision 改变。
- 同文件“恢复与交付”：ready/published、有 export、已提交发布 rename 的 run 不接受 semantic revision。当前 `tools/continuation.py:144–147` 同样禁止对这些 run 使用 `continue --full-restart`。历史2.2.x从已发布项创建 successor 的行为不能照搬成当前命令。
- `tools/revision_control.py:18,76–82` 要求冻结 executor 已支持 `semantic_revision.v1`。本次三个未完成当前 run 均不支持；它们的能力只有 summary/state_guard/hold/scoped_restart/continuation/audit/proposal/budget_extension。
- `tools/runtime.py:310–315` 明确禁止把 `design_region_derivation.v1` 能力变化注册为 compatible runtime。直接把当前2.5.0工作树设为旧2.2.x的兼容目标，既违反语义边界，也被工具拦截。
- `schema_docs/0.common.md`：root Design 全本地；派生三区可完整本地或 `from_parent: 直接父ID`。旧全本地稿仍合法；现有 raw 不自动转换、不用父正文覆盖子稿；不能跳祖先/兄弟/跨目录引用。真实相同的整区才适合引用，引用不抹除拓扑差异。
- `when` 在完整 Contract 域独立求值；parent 只给直接设计来源。P4 改父要求显式 identity migration；provider 变化需重评后代，P6 当前资格失效；P6 必须读全树有效内容，各节点参数绑定、选择域、跨区接口和 realization 均覆盖。
- `schema_extract/REVIEW.md`：完整旧稿先 review；有据问题才 patch/reextract；未完成稿仍继续作者逻辑步骤，维护不能批准残稿。有限维护的字段权限不缩小 reviewer 全文审阅责任。
- 外部 comparator 未提供公式/基准/归一化时，用 tolerance 明确来源记录，不采用默认简写；不因这个已知缺口无限阻塞提取。源缺陷若不要求重定算法、精度、逻辑依赖、容量/生命周期等，记录后可完成抽象 Design；不能把坏地址作为 require，也不能据此批准残稿。

## 六项现状矩阵与真实身份

原批次映射已独立核实：`.schema_extract/preparations/six-families.json` 的六个 preflight 项与实际发布 receipt 一致；历史用户名称 `fp8_scaled_group_gemm` 对应 catalog ID `fp8_scaled_grouped_gemm`，其余五个 ID 同名。原六项77节点，推荐版本用 GEMM 2.2.4 和 GDN 2.2.7 后为81节点。

`R(UUID)` 以下表示 `.schema_extract/v2/runs/UUID/`；每个状态的权威文件是 `R/state.json`，固定身份是 `R/inputs.lock.json`，合入产物是 `R/checkpoints/<checkpoint>/artifacts/`。附录提供当前重点路径和全历史简表。

| 算子 | 正式发布/推荐内容 | 后续真实状态（2026-11-05读 state + inspect） | 完成与缺失 | 停止原因及限制 |
|---|---|---|---|---|
| GEMM | 原3节点 `4aac1ea4-94fa-4b4c-93ca-d48a8eb08996` /2.2.2；推荐3节点 `b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b` /2.2.4，目录 `schema_wiki/revisions/2.2.4/.../gemm/grouped/fp8_scaled_grouped_gemm` | 两者 ready/published；推荐 checkpoint `301b901245cd78343bca60556eabfe08c29939e3bc90720321db7da1f0c429eb` | root、umma_wide、mma_sync 均正式完成 | 无未完成后继；原稿 graph 销毁错误范围已由推荐版修正。无需恢复原2.2.1 comparator阻塞 run |
| FMHA | 原8节点 `acb68909-aaaa-4bc2-8b16-94904752af8e` /2.2.2，原 catalog输出目录，ready/published | 后继 `45ea6608-66b2-4915-b47d-59d2c115602e` /2.2.7，P5 `termination_unconfirmed`，原 source/角色/任务域不变，checkpoint `812602cc9003742176580b62ea078814a926c3a8d230aa42da277aa2ad48e85f` | P1–P4 approved；43计划节点中29有效，14未有效；无当前 P6 | 7 cli_error、5 pending、2 termination_unconfirmed。两次失败调用因403模型配额上限退出，但进程组归属当时不明而留下 fence。不能把“runner已死/现在没PID”当作已结算 |
| MLA | 原单root `e3f7704f-061f-48ff-8131-cc317afe6003` /2.2.2，ready/published，checkpoint `2a16c9dcfb5d65b04ee50108349ec9040de6ee479a7c055729f7110a52461ede` | 无后续修订run | 1/1正式完成 | 新版 MLA 校注收集能力不等于旧快照已有校注；原 snapshot 的 candidate/problem文件均与当前对应文件逐字节相同，旧身份必须保留 |
| Sparse | 原3节点 `d5a73c01-7c3a-45b2-a3ce-0d5eec634808` /2.2.2，ready/published；现阅读入口为 errata，原dispatch字节不变 | 后继链见下；最新 `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7` /2.2.9，P5 `termination_unconfirmed`，checkpoint `0876fd66dce267bb9c46cd383bddc47ddfb87d53521d0c2a26425d534d11a356` | P1–P4 approved；8计划节点中 root/mma_direct 有效；tc_heads 和 probe_serial_mma 有稿；另4未写 | tc_heads的最后调用退出后仍 fenced；probe_serial_mma cli_error；其后代等待父。最新停止不是旧源槽位问题本身；该问题2.2.9已澄清提取边界 |
| GDN | 原2节点 `97902388-eda0-4cbd-b8a6-ba71440e86ab` /2.2.2。推荐6节点 `45f9bb51-77f3-4285-b7a7-251cf2744fa1` /2.2.7，目录 `schema_wiki/revisions/2.2.7/.../linear_attention/gdn_prefill_bf16_state_bf16` | ready；原output的 publication conflict、published=false，但 exports中的正式新目录已 published，非未交付 | 6/6；P6 attempt `1283b632-a802-4625-b666-a0eb3dce449d`；checkpoint `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d` | 无剩余节点。不能因顶层 published=false重新恢复；本次所选P6 audit --verify仍 verified |
| MoE | 原60节点 `3563c3b7-2b7c-4967-9d41-9c2042e53d2c` /2.2.2，ready/published，errata仅索引更正 | 新source scope任务 `53d4bebe-882d-4a66-98b1-287d132dd6f3` /2.2.7 → `2c4207c8-000e-43ea-8da0-2eb6a6079aee` /2.2.8 → 最新 `feb0c655-4e13-43ac-817d-1f457f31c029` /2.2.10，P2 reviewer `cli_error` | 当前仅P1 approved；P2有24行dispatch草案、1025行root保留稿；P3/P4历史用量保留但当前pending；plan=None，不能报作“已有一个完整节点”或“60节点修订完成” | 最后attempt `9eb6167a-1108-4b9e-ba7c-714786b0ae5b` 确认终止；403 gpt-6-astra模型配额上限、CLI retry耗尽；无active/unconfirmed/unsettled；source含default_scope新增输入，不能混同原60节点scope |

三项未完成任务的公开 inspect 均显示 runner lock free、alive=false、successor=null、revision_gate_capable=false，且全局 hold=`15837dcb-5f2f-4e3a-a4d8-d99631834cd6`。各原deadline在本次审计时已过去；暂停/等待不停止预算时钟，未来继续必须核实并追加授权预算，不能把错误状态等同于还有足够余额。当前未检查供应商配额是否已恢复，不发送模型调用探测。

非权威文件 `.schema_extract/trials/six-families/post-batch/active-artifact-repairs.json` 尚写 FMHA running/12/43、Sparse running/1/8、MoE running/P1；它们只是历史观察。`COMPLETION.json` 的“no unfinished”对应旧批次完成时点，也不能覆盖后续任务。

### 实际未完成节点

FMHA的14项：

- `cli_error`：tcg_d128_flat_cp_xpose_wide、tcg_d128_fscan_cp_direct_wide、tcg_d128_fscan_tma_xpose_narrow、tcg_d128_segmented_cp_direct_wide、tcg_d128_segmented_cp_xpose_narrow、tcg_d256_segmented_cp_narrow、tcg_d256_segmented_tma_wide。
- `pending`：tcg_d128_fscan_cp_xpose_narrow、tcg_d128_fscan_cp_xpose_wide、tcg_d128_fscan_tma_xpose_wide、tcg_d128_segmented_cp_xpose_wide、tcg_d256_segmented_cp_wide。
- `termination_unconfirmed`：tcg_d128_segmented_tma_xpose_wide（attempt `85cfe993-9236-4aec-bb0d-c76a276262af`）、tcg_d256_flat_cp_wide（`01c1e189-86a7-4e22-8006-da604bba958a`）。

14项均 `authored=true`、当前generation=1。13项保留last_approval，只有 fscan_cp_xpose_wide没有；9项有当前draft_checkpoint，5项pending当前draft指针为空但不能忽略历史稿。它们不属于“从未做过”，旧批准也不属于“当前有效”。已核实root验收记录重开要求保留源码限定的实验判据；因此下游代际变化/复核有真实原因。只看节点曾approved会虚增完成数。

Sparse的6项：`tc_heads`（上述未结算调用 `adbba921-2156-4bfd-9fc1-6b8359f3a9f1`）、`probe_serial_mma`（cli_error，有当前稿）、`tc_packed`、`probe_serial_packed`、`probe_hidden_packed`、`probe_hidden_mma`（后四authored=false）。依赖为 root → mma_direct → tc_heads → tc_packed → probe_serial_packed → probe_hidden_packed → probe_hidden_mma；probe_serial_mma直接派生mma_direct。

Sparse完整后继链：原发布 d5a73… → 059ed88a… /2.2.6 paused/P1 → 2d252bf9… /2.2.7 blocked/P5（当时3个有效，tc_direct源缺陷阻塞）→ b6d50336… /2.2.9 cli_error/P1 → 8f1be03a… /2.2.9。各次均不继承批准，账目仍保留。GDN中间9699c87b… /2.2.6 cancelled/P1是已被接替的历史任务，不应复活。

## 逐算子内容判断

### 1. fp8_scaled_grouped_gemm

审阅定位：推荐2.2.4三节点的metadata/when、root Contract/数学、root和wide的工作提交与角色主干、mma_sync主干和各节点realization；回查固定 `candidate/solution.py`、`fp8gg_umma.cu`、输入构造及 reference。没有将旧P6等同本轮全树批准。

可保留：Contract清楚区分物理group rows与generator的logical_group_sizes，全物理行写回、out身份和guard不被掩盖；FP8原始值和FP32 scale定义标准答案。math_reshape按128 reduction scale block先独立FP32 partial，再scale产品和ordered FMA，记录参考与候选舍入位置变化及padded行额外工作。root持久CTA、三槽operand ring、两槽TMEM pair、epilogue提前归还product-slot而保留寄存器fragment；wide 256列和mma.sync路径保持真正不同设计。realization的descriptor/swizzle/实际shared-capacity交接有作用，不能随意删去。

**明确补丁 B-GEMM-1**：推荐 `root.yaml:427` 的 `design.tile_pipeline.phase.main.role.MMA.info.member` 为 `warp[1:3]`，但同文件433–434说明dual=0时仅warp1、warp2在此role不活动。原发布root同字段在368行同样为`warp[1:3]`。`schema_docs/5.design.tile_pipeline.md:54`要求“不承担动作的warp不登记”，成员整数端点规则允许参数化。源码 `fp8gg_umma.cu:415` 是 `((warp == 1 || (warp == 2 && p.dual)) && lane == 0)`；428以后分dual；1830定义host双发射门。因此应在新创作副本将此role范围改为已定义整数参数的实际集合，例如 `warp[1:2+dual]`，并复查setup/barrier arrival count及动作约定。不是修改kernel，也不需要改Contract或新增一个dual节点：角色数量已经合法参数化，两个partial的逻辑输入/输出、资源、主提交集合不变。Setup中全CTA初始化成员仍保留；不能把setup参与推成MMA参与。

同类范围核对覆盖两份正式GEMM树的所有节点：**仅两代root的MMA.info.member**发现此问题；wide的Setup/Producer/MMA/Epilogue分别固定为0:10、8:9、9:10、0:8，mma_sync固定协作0:8，无同类inactive-by-dual范围。原发布保留原字节；首选只在以2.2.4为基础的新稿改root，不回写两份历史产物。

graph校正可保留：推荐root约290行明确销毁返回码忽略，检查的是create/add-node/instantiate/launch；源码1726–1727、1783–1784、1847–1848无销毁状态分支。host的weakref参数memo、device0属性和缓存键漏检查是源行为/限制，不应为了“正确”修饰成每次重新验证元数据。当前三节点都全本地；没有完全相同的直接父整区，强制转换引用无明显收益。

范围：内容动作patch P3:root的上述member/相邻必要动作约定 → 三节点完整review/P6。若采用正式新run，则P1–P6按新extract流程重新批准，旧内容只能作为阅读参考，不能伪造旧批准移植。

### 2. fmha_mxfp8_qkv

可复用的核心不是“普通softmax”：Contract解包F8_128x4 E8M0，明确V scales与Q/K不同布局；按128-key数值组延迟更新最大值、P经过E4M3舍入而分母使用未舍入概率，D128 gain16/D256 gain1，LSE单独按自然对数标准答案。新版root补充0*Inf/NaN语义，并已用nullable表达可选LSE、window及seq_kv_lens；这些比原8节点Contract质量更好。

旧产物的确定问题仍真实存在：`split_tcg.yaml:214–220`固定列出segment launch及segmax handoff，269–272却说TwoPass=false时不提交并令scan直接依赖gm；当前flow规则不授权正文删边/换producer。旧root将TMA成功与cp.async失败保留同一路径，但它们的PV退休/搬运组织有差异。旧Contract的lse_tensor、window_left、seq_kv_lens、LSE也未用nullable；最新2.2.7 root已修这组问题。不要回到旧八节点补几个注释后声称完成。

最新43节点计划有具体依据，不是为2.5.0复制文本而创建：源码 `fmha_mxfp8_qkv.cu:1543–1760`实际有gm、flat/fused/segmented扫描、split和combine组合，`maps.valid`控制TMA/cp；D128 PSK与转置写回、窄/宽combine也改变角色与物化。新dispatch给每个节点直接父及独立完整选择。已抽查当前root、fused_d128_cp、tc_split_flat/segmented、tcg的flat/fscan、TMA/cp与direct/xpose代表稿的条件、flow和差异；这些内容值得保留，再逐节点review。

风险仍须专门核对：原源码mma.sync group-max grid使用qtiles128（1979）而计算行tile按64，源码覆盖不足不能虚构多发CTA修复；empty split的unwritten PartA、descriptor失败回退、PDL依赖、masked/nonfinite乘积、graph记录与回放固定选择均应保持真实。当前root验收来源记录曾重开，先核对它已有的限定实验标准，不将“未提供完整外部comparator”扩大成“完全无已知误差公式”。

首选 P1/P2/P3/P4各review，已有29有效节点和其余完整候选各P5 review；缺稿/未完成稿keep正文继续authoring，反馈只patch相应字段。全部恢复后新P6。不能仅检查差异或when：43节点当前有效全文、每个local realization仍须覆盖。当前29节点里，7个数学区、7个work_schedule区与直接父解析对象完全相等；2.5.0以后可选转换这些整区节约阅读，但不是修复前提，已有稿默认不动。原有slot/producer拓扑不相同的pipeline不能为了压缩强共享。

### 3. mla_expanded_fp8

该单root的技术内容有较高价值。Contract限定prepared Q/K/V而不是projection/RoPE/cache写入；在线计算区分FP16 exp/四条FP16 sum chain与E4M3 PV；`gain_log2=8`准确反映源码`kLog2_448=8`其实是256，不能误写448。tiny路径的FP32额外PV/分母只替换selected输出，**regular P8/PV仍执行**，源码1078–1087及后续issuer路径可核实。流式队列、几何双槽、KVST跨work连续环、TMEM S/P别名、correction输出读取归还早于最终global store等均值得保留。源码头部“tiny bypass entirely”的注释不应覆盖已读实际代码。

**明确补丁 B-MLA-1**：仅一份正式root，当前缺少nullable的完整字段集合为：

1. `contract.algorithm_requirement.input.attention_sinks`（root28–31）；
2. `contract.algorithm_requirement.input.lse`（root33–38）；
3. `contract.algorithm_requirement.output.LSE`（root52）。

`schema_docs/1.contract.algorithm_requirement.md`“nullable 与 None”（217附近）明确缺省/false不增加None分支，shape/dtype只描述非None对象；不能靠meaning、range或default表达缺席。原root65–67以正文定义optional，故虽然checker通过，当前语言仍不完整。源码 `problem/input.py:101,113,123,129`实际以sink/return_lse生成None或Tensor；候选`solution.py:2100–2101,2159`接受None并按return_lse返回out或(out,lse)。局部补丁应给三项`nullable: true`，在compute明确return_lse=false时逻辑LSE=None、没有写入及没有tuple第二项，并保留true时lse必需的相关约束。可省略参数的default是否保留按已绑定接口证据处理，不能将nullable误写成可省略权限。

此补丁**不改变提交/分支**：sink和可选LSE在同一个kernel局部运算/写回中处理，原稿已有when-present条件；无需拆三个新plan，不改变QK/PV主体。python_none单值输入、`sage_attn_sfs`的tuple(None,None,None,None)不是可选数值张量，不应批量加nullable。

**明确补丁 B-MLA-2**：root165–167的Bq/Bk/N使用`binding: ref:design.work_schedule::...`。当前math规范172–173把ref限定为上游/本区已定义，下游提供者使用bind；这三项由work_schedule确定，应改`bind:`并核对目标。当前正式六算子树检索只在MLA发现这三处此类下游ref绑定。局部表达修正，不重做数学。

realization中的tcgen05 ordered-completion、TMEM读写等待与TMA128/64swizzle/tiny scratch匹配是必要实现交接；不要在其reason中补出未在设计存在的同步。跨stream元数据、table覆写和重入缺少协议是源限制，dispatch已披露，应保留。单root无需转换引用。本次所选原P6 audit仍verified仅证明历史证据完整，不能豁免这两个当前表达补丁。

范围：P1 Contract patch + P3三个binding patch → root全文review → 单root P6；已发布run不可能原UUID semantic revision，正式当前发布需新目录extract。

### 4. dual_cache_latent_bf16

Contract关于两池ID坐标、SW保留128列、压缩起点不随有效SW长度移动、重复ID保留多重性、一份KV同时为K/V、联合softmax与sink分母的主干可以复用。旧root270–278/328–331固定列combine却正文要求S=1删掉launch/handoffs；旧probe_dual把互斥packed/mma及隐藏/串行probe汇成结构并集。这是schema表达错误，不是源kernel固有缺陷。

新root限定P.splits>1并固定attention→combine；mma_direct为单kernel；新八节点计划将tc_heads与tc_packed、串行probe与隐藏probe、两种实际结果分开，方向正确。已读最新root Contract、root/mma_direct调度主干、tc_heads草稿的角色/同步/realization代表区及全部八节点计划；tc_heads草稿通过当前3文件子集checker，但还没有独立P5批准，不能据此解锁tc_packed。

**P1必须review的任务域疑点**：最新root47–55仍给bmm1_scale/bmm2_scale默认1.0、sinks允许None，约99–104同时承认实际构造固定scale=D^-0.5/1且sink总存在。固定 `problem/input.py:116–120`与shape_range没有sink可选开关，None来自reference额外接口。当前P1规则禁止仅因签名扩展任务接口，GDN修订已遵循同原则。本次不假装已证明全部范围来源；恢复前应独立核对本任务是否有允许更宽接口的scope声明。若仅上述绑定，则局部patch Contract以准确固定任务域，再沿全树复核；不要称这是“为实现缩小Contract”，这是核正转录范围。若确实要提取更广通用API，那是source/scope变更，另建任务。该问题不足以否定已写数学与设计主体。

源槽位问题已经独立看代码：`dsv4_kernel.cu:2111–2115`使bar_land与tmem_base_s同址且ukey紧跟，2497–2515同时初始化landing、分配TMEM、读base；与声明的5*STG barrier容量及后续gather/QK使用不符。按当前REVIEW源缺陷边界，容量、逻辑对象/使用点足够支持抽象Design，不必先求出GPU损坏值。要保留TMEM、完整UnionKey容量、landing/validity和读写依赖，不能把坏地址写入require，也不能用issue已归档代替完成这些段。旧2.2.7的tc_direct blocked理由已不适用于当前规则，但普通resume旧run仍不会自动获得新边界。

实际probe必须保留：源码`probe_begin/probe_end`、隐藏分支约6148–6163已经在side stream提交probe并发prefix；有界poll未齐仍可能有正均值而接受packed。新dispatch清楚记录R的FP64累加转FP32、prefix和余段producer、fork/join以及外图录制固定host选择；保留这些而不是恢复旧probe_dual并集。未提供read-retirement或跨调用host gate保护，不得补造。

首选当前8f1…正式修订到2.5.0，P1–P4 review，已完整候选review，未写后四节点keep/正常P5。tc_heads → tc_packed的父先子后次序不可跳过；完整P6还缺全部6节点资格。无需为语法升级重新提取所有两池数学。

### 5. gdn_prefill_bf16_state_bf16

推荐使用2.2.7六节点，原两节点只保留历史。固定输入 `problem/input.py:243–246`确实提供cu_seqlens/chunk_indices/chunk_offsets并norm=false；六节点Contract已移除旧稿的`int32 or python_none`等扩展和normalize路径，并为core_attn_out/FinalState/CoreWritten使用nullable。

内容质量较好：逐token delta recurrence与chunk64三角逆/状态链分层；Q/K/V跨请求padding读与g/beta有效尾掩码区分；hierarchical inverse各层不同乘法顺序、额外零权重项不被数值简化；KS/QS双累加分组、Uh与DU分别从U舍入、mini-row分支、FP32 carried state和BF16 operand snapshot准确区分。work_schedule定义真实prep→chain或prep→pass_a→pass_c、PDL payload wait，GQA共享图像的producer及消费者人数；六节点分别处理Qc/Kc是否物化和Perm是否存在。

已核对六份when、flow、handoff、parent以及代表pipeline差异，回查 `gdn_kernel.cu:3221–3249`的Gc/Gw选择、3316–3321的direct-QK、3341–3354的Perm/PRE0。`segmented_ordered`对N=8/T>=15880的化简有源公式支持，不能叠加父segmented的N<8条件；这是独立when的重要正例。

源问题仍是源问题：root约294–302保留非连续core临时写后无copyback；约654–658保留final mini-chunk stage0写与Kc读取未退休的竞态，不把一个大动作正文视为跨warp顺序。源材料支持的必要设计约束仍应保留；无需GPU复现来重新完成schema。既有六节点全本地整区间没有完全相等的父区，强制引用转换收益低且容易混淆不同消费者/域。

首选keep发布内容，独立当前规则review可排在有确定问题的算子之后。本审计未逐行完整读六个pipeline全文，因此不宣称2.5.0 P6通过；仅有历史2.2.7终审及本次其证据verify成功。要出具当前规则全树批准，需新目录全覆盖review；不能对ready/exported原run revise。

### 6. nvfp4_fused_moe

旧60节点的可复用知识：prepared FP4 payload/scale反置换、up/gate行置换、两种激活、block16 QDQ与E4M3 scale、每route FC2先BF16物化再按权重做FP32归约；这些数学事实无需因新scope重推。核心流水中路由count/scan/scatter、gather、FC1、activation/QDQ、FC2、finalize和split-K/last-arriver确有实质差异。

但不能只做“60份转from_parent”：旧`ws_fused_ws.yaml` count/scan/scatter在232以后正文说only outside Small、gather说only AMODE=2，flow仍列结构并集；Epilogue503/592登记`warp[3:11]`而thesis要求实际`warp[3:3+EPI]`，gather419登记0:32而Nt可64/128等。源参与集合可以用整数端点表达，不能借最大集合正文删成员；旧`gemm1_clamp_limit`还用`fp32_or_python_none`代替nullable。节点数量不是首要错误，覆盖当前真实默认可达路径才是。

默认scope后继有实质进展：scope文件明确“默认实现选择下可达全部路径”，包含zero probe、形状PDL门、graph建立/回放、cache复用、已有fallback/错误出口，完整动态输入域不缩为十样本。最新root旧P3稿1025行仍在，P2草案重新把代表root限定为复用ABI、cfg97/cfg2=81、无zero验证的大路由PDL融合主干。草案列出small准备、serial、split-K、FC2、direct、verify-only、重复验证、ABI建立前缀和异常提交前缀；这些不是“稍后随便造若干分支”，P4还必须核实可达组合。

尤其值得保留的新发现：源码`nvfp4_moe.cu:9533–9538`无论BeginCapture返回是否成功仍调用私有stream的run_pipeline；9577失败后再caller stream run_pipeline。因此确有“同一请求已实际提交两组主干”的路径，不能把第一组一概当纯录图或另一次算子请求。ZERO2干净图后dirty/超时再完整图也可能发生两次验证；源码快照不能通过文字删掉其中一次。`solution.py:1169–1485`默认自动cfg/route_hist/small_thresh/zero probe以及ABI重用决定了可达分支，旧60节点的手工cfg笛卡尔积不能直接搬成默认scope计划。

首选保留当前default-scope lineage：P1独立review（不自动采纳旧approve），P2从当前24行草案继续完整review，P3保留root草稿按当前P2选择修补、完整重审，P4重审默认可达整树，P5首次新节点可利用2.5.0整区引用，最后全树P6。P3正文存在不代表当前P3完成；现在没有正式P4计划。不要仅为省成本采纳53d/2c旧P3/P4批准，当前阶段和问题不同。

如果目标改回涵盖手工cfg/cfg2、诊断或实验开关：新增source/scope任务，不继续feb0…扩大范围。原60节点仍可作被核实的阅读线索，不能继承其批准。

## 合法、可执行的后续命令方案（本次均未执行）

以下只列已从当前CLI定义/help核实的语法。`RUN`、`CHECKPOINT`、`BRIDGE`、`PLAN`代表实施时选定的真实值；现在没有生成compatibility review或semantic proposal，不预先伪造独立签字。

### A. 先结算旧调用，保留控制和账目

1. 读取权威state、公开inspect，按其active/unconfirmed登记的attempt核对process/result。当前三个run的公开读取：

```sh
python3 -B schema_extract/extract.py inspect --run 45ea6608-66b2-4915-b47d-59d2c115602e 8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7 feb0c655-4e13-43ac-817d-1f457f31c029 --json
python3 -B schema_extract/extract.py status RUN --full
python3 -B schema_extract/extract.py audit RUN --attempt ATTEMPT --verify --json
```

2. 真正开始恢复前，可用`pause RUN`让原冻结Runner在不派发新工作的控制下接管并对账；已读2.2.7冻结`runner.py:948–965`，consume_controls后会_reconcile。若确需终止在途调用使用`cancel RUN`，但不能把cancel返回0当作终止已确认。直到active/unconfirmed/unsettled全空且旧writer/launch已结清，再进入升级。现在inspect没有PID仍显示unknown，不能手工清fence、编辑state或删除attempt。全局hold继续保留，后续只有在正式获准接续时释放对应hold；释放不能越过更晚的pause/cancel。

3. 预算与外部配额分别核实；当前错误是配额而非源码缺失。配置里的model/effort冻结，不能通过resume加不存在的`--model`换模型。需要改变config时新建任务。未来追加示例：`resume RUN --add-seconds N --add-calls M`，每次实际调用是一次追加授权；默认continue不接受这些追加。

### B. 未完成旧run采用2.5.0

**不能直接调用revise成功，也不能直接compatible升级2.5.0。** 现有snapshot浅层版本清单只到2.2.10，没有任何已登记的2.4.1/旧语义桥接runtime。真实可用的工程基底是Git `60285f0` 的 **2.4.1**：已读其`tools/__init__.py`及`revision_control.py`，支持semantic_revision.v1、旧inline Design，不含2.5新整区语言。但这不表示整份2.4.1自动compatible：逐文件比较发现它相对FMHA 2.2.7有15个非executor文件差异（含common/work_schedule/tile_pipeline），Sparse2.2.9有7个、MoE2.2.10有6个（含review/P5/工具）。

因此要保进度，先在runtime之外制作、验证一个**确实保留各run原语义**的桥接目标：以2.4.1执行器的实际代码为基底，使用原snapshot的schema_docs/starter/checker/阶段职责与提示语义，核对完整差异；必要时回移纯runtime修复。不能只改版本号/能力常量，也不能只复制几个文件就宣称通过。至少验证旧raw/checkpoint/receipt可解析、原批准依赖语义不变、旧字段检查未变、账目/单次决定不重置、原source/config锁不变、semantic revision的request/drain/prepare/apply可工作。工程验证用隔离合成状态，不能先改真实run“试一试”。

桥接制作不在本次只读任务范围；报告不声称它已经存在或通过独立兼容审阅。真实桥接完成后，按当前入口注册：

```sh
python3 -B schema_extract/extract.py compatibility-report RUN --checkpoint CHECKPOINT --target-repo BRIDGE > REPORT.json
python3 -B schema_extract/extract.py compatibility-report RUN --checkpoint CHECKPOINT --target-repo BRIDGE --review REVIEW.json
python3 -B schema_extract/extract.py upgrade-runtime RUN --report REPORT.json --review REVIEW.json --target-repo BRIDGE
python3 -B schema_extract/extract.py revise request RUN --reason '采用2.5.0并按本次独立范围评估保留和复核已有稿' --target-repo /home/shaohanyong.shy/KernelSchema
python3 -B schema_extract/extract.py revise prepare RUN
python3 -B schema_extract/extract.py revise prepare RUN --plan PLAN.json
python3 -B schema_extract/extract.py revise apply RUN
python3 -B schema_extract/extract.py revise status RUN
```

REVIEW必须逐项覆盖完整差异及SHA，真实独立身份/记录，preserves七项全部有据；unknown或新语义不得标runtime_fix。PLAN在排空后根据实际返回的scope、稿件、批准、差异生成，独立选择keep/review/patch/reextract。已完整候选优先review；不充分的旧批准不adopt；未开始范围keep。明确写入字段，例如MLA补丁的Contract/三个binding；对FMHA/Sparse尚未确定的问题不要给全目录无限改写权。最终仍需当前P6。

桥接后才可用当前`resume RUN --scope P5:NODE`作普通单节点恢复；现有三个旧executor缺`scoped_recovery.v2`，当前CLI477明确拒绝。不能将旧`scoped_restart.v1`误认为已支持当前scope语义。maintenance task内不能使用普通provider_reopen扩大权限。

若不愿承担桥接工程，未完成run可以明确选择：

```sh
python3 -B schema_extract/extract.py continue RUN --full-restart --checkpoint CHECKPOINT --reason '明确放弃当前批准并从P1重审'
python3 -B schema_extract/extract.py resume SUCCESSOR_UUID
```

这也要求旧调用结清，保留预算/用量且不继承批准；当前实现封存旧树，root内容可保留，派生节点资格归零。它对FMHA代价很大，不是默认推荐；对于仅P1有效的MoE，若桥接成本超过重新P1审核成本，可作为有明确理由的替代。但不得对已ready/published GDN/GEMM/MLA使用此命令。

### C. 已完成产物的新目录重审/发布

可以在独立创作目录保留raw的root/branches/dispatch做人工review与有限patch，使用当前checker；不得复制旧`.extraction/receipt.json`冒充新批准，不覆写历史目录。这个产物是“当前规则审阅稿”，不是Runner已发布结果。

若必须得到当前正式Runner P6/receipt，现有工具的合法入口是新`extract`，没有“v2发布目录导入为已完成P3/P5”“seed checkpoint”“迁移批准”参数。`import-legacy`只接受已终止v1任务，从P1重审，不适用于这些v2发布项。示例：

```sh
python3 -B schema_extract/extract.py extract --source /path/to/fixed-source.json --output /path/to/new-delivery --queue
python3 -B schema_extract/extract.py resume NEW_RUN_UUID
```

fixed-source.json用`schema_extract.source.v2`，repository指向原snapshot/source，directory可为`.`，角色使用原lock中的完整相对路径；scope严格复用原lock所列文件。省略revision使用实际固定字节。新source identity可以不同，但要逐文件核对原源码/角色/目标未变；不要为方便调用当前catalog而悄悄收进未绑定的新校注或范围。旧schema可作审阅参考，但不是代码已支持的自动seed/approval import。

未完成FMHA/Sparse的output仍占用原发布目录；完成后原发布会conflict而保持ready，可用已核实入口：

```sh
python3 -B schema_extract/extract.py export RUN --output /path/to/new-delivery
```

MoE目前绑定`schema_wiki/revisions/2.2.7/.../moe/nvfp4_fused_moe`，目录尚无交付；若采用2.5.0后不希望版本路径误导，最终export到新目录并在导航写真实receipt版本。发布目标须不存在或仅有零字节普通.gitkeep，不能预先创建普通空输出目录。export只交付同一完整已审结果，不重新审阅也不改变run绑定输出。

### D. 输入scope改变

MoE `default_scope`已作为新source输入建立独立53d lineage；不是从原60节点run semantic revision收窄来的。继续它保留该范围。如果要换到全控制/新源码/新任务/新模型配置，使用新source描述和新output extract；不能改inputs.lock、对现有run加scope文件或借target-repo替换source。其他三项历史接续也需保留原source/config，而新规则的来源缺陷边界只影响表达与审阅方式，不允许修kernel。

## 优先级与风险

1. 首先保存本报告与权威身份、保留全局hold，对三个未完成run进行合法终止对账。否则任何upgrade/revise/full-restart都不应开始。
2. 可以独立推进小内容修订评审：MLA三个nullable/三个bind、GEMM单role成员；GDN保留现状。它们的修改成本低，但正式新发布仍受工具入口限制。
3. 工程上先准备一套真实旧语义桥接方案并独立验证；若按每个冻结基线保留旧规范，须逐run形成完整差异review，不能共享一张泛化“兼容证书”。
4. 保稿续作优先FMHA（已有29/43有效、14都有历史）、再Sparse（2/8有效但tc_heads草稿可用）。两者先P1–P4的scope/依赖复核，尤其Sparse接口域，避免后半树写完后再次全树失效。
5. MoE最后完成P2/P4的默认可达计划，再按父依赖完成P5。不要预先承诺节点数；可能小于旧60，也可能因ABI/错误前缀多于直觉。

主要不确定性：没有完成FMHA43、MoE60、GDN6全部pipeline全文及所有源码支路的P6级审阅；没有验证所有default MoE组合可达性；没有GPU运行；没有核实当前外部配额已恢复；桥接executor尚未构造/测试，不能把已有2.4.1commit等同现成兼容交付。具体字节差异、准入条件、旧会话退休和审核成本需在正式prepare返回实际frontier后确定。

## 实际执行的只读验证与证据

- 当前版本/help和相关CLI/runtime/continuation/revision控制源码读取；六类初始preflight映射、catalog路径、全部23个相关run的浅层state概要，未遍历打印attempt/snapshot全集。
- 八份正式交付receipt逐文件SHA：**94个artifact文件全部匹配**。当前checker检查八个目录（86个YAML，含两份历史与修订并存树），全部exit0；均`--skip-hardware`。这不构成语义通过。
- 当前FMHA合入29文件、Sparse合入2文件、Sparse tc_heads的3文件草稿依赖目录，当前checker均exit0；checker称whole task checked只表示目录直属现有YAML，不证明43/8计划齐全。
- 对六个重点run，输入锁中candidate/problem文件与当前对应字节比对，分别11/13/8/9/10/18文件，无差异。源码证据因此可用当前kernel路径的同一行号定位，但身份仍以原snapshot为准。
- 公开inspect三未完成run、定向读取其失败attempt process/result/events；未执行任何新模型/GPU请求。事件明确403模型配额错误，FMHA/Sparse额外保留终止未确认。
- GDN最终P6 `1283…` 和MLA最终P6 `3f0c…` 的 `audit --attempt --verify --json`均exit0、verification=verified，errors/unavailable=[]。仅该两次调用及其输入/结果checkpoint，不声称全run audit。
- 字段/结构抽查如正文；按解析后的整区比较，只评估可能共享内容，未实际转换；所有现有审查树都是全本地Design。
- `git status --short`在收口时为空；本次仅在/tmp写报告/辅助证据，不改AGENTS.md（保持空）。

辅助证据：

- `/tmp/kernelschema-recovery-audit-20261105-b-runs.json`
- `/tmp/kernelschema-recovery-audit-20261105-b-inspect.json`
- `/tmp/kernelschema-recovery-audit-20261105-b-checks.json`
- `/tmp/kernelschema-recovery-audit-20261105-b-draft-checks.json`
- `/tmp/kernelschema-recovery-audit-20261105-b-gdn-audit.json`
- `/tmp/kernelschema-recovery-audit-20261105-b-mla-audit.json`

## 附录：重点固定路径

### fp8_scaled_grouped_gemm

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/a42ac9bf81434da90b1a322bc2bfba2777bb87f8fe9a4aefffc844116b3d0bdc`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b/checkpoints/301b901245cd78343bca60556eabfe08c29939e3bc90720321db7da1f0c429eb/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/.schema_extract/trials/six-families/verification/2.2.4/fp8_scaled_grouped_gemm`

### fmha_mxfp8_qkv

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/138d555b31ab141c588c735d7b260b5b139f3188daa26393fd447ef066ffa3b8`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/812602cc9003742176580b62ea078814a926c3a8d230aa42da277aa2ad48e85f/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv`

### mla_expanded_fp8

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/e3f7704f-061f-48ff-8131-cc317afe6003`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/b97eec335391249ef6078863930904b62a52dcd4dbf8a0ad72a0021c35a91fd6`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/e3f7704f-061f-48ff-8131-cc317afe6003/checkpoints/2a16c9dcfb5d65b04ee50108349ec9040de6ee479a7c055729f7110a52461ede/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8`

### dual_cache_latent_bf16

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/ece761e6f1d0183735c539148e66c1965a62f883d7152ad3d20b70d66c1d0160`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/0876fd66dce267bb9c46cd383bddc47ddfb87d53521d0c2a26425d534d11a356/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16`

### gdn_prefill_bf16_state_bf16

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/0b48041c858b11c265d3ea0c03ec39714fe16909c958ea31e4eb25ab55266b31`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/checkpoints/349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16`

### nvfp4_fused_moe

- run：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029`
- snapshot：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/8b281029059476dcfe78d7588953be485983ff2d084fdc1216ff44d2e3b565f0`；其中`source/`与`rules/`分别是来源和规则冻结树。
- 当前checkpoint产物：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/ed3b30e50697640fc0b86c8205bddf0e10bdc4308134b7bb1596e9c29c622354/artifacts/`
- 原output绑定：`/home/shaohanyong.shy/KernelSchema/schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe`

## 附录：全部相关run简表

| 算子 | UUID | 冻结版本 | 状态/阶段 | predecessor |
|---|---|---|---|---|
| dual_cache_latent_bf16 | `059ed88a-3be3-4642-b578-8ed5959eed37` | 2.2.6 | paused/P1 | d5a73c01-7c3a-45b2-a3ce-0d5eec634808 |
| fmha_mxfp8_qkv | `166fc15f-e49e-42ef-b429-c7f507f5ed53` | 2.2.1 | blocked/P1 | 无 |
| nvfp4_fused_moe | `2c4207c8-000e-43ea-8da0-2eb6a6079aee` | 2.2.8 | invalid_receipt/P4 | 53d4bebe-882d-4a66-98b1-287d132dd6f3 |
| dual_cache_latent_bf16 | `2d252bf9-d9a3-4fa4-96d2-90a98e25eb34` | 2.2.7 | blocked/P5 | 059ed88a-3be3-4642-b578-8ed5959eed37 |
| mla_expanded_fp8 | `2dfb565c-28a6-48de-9cdc-ab31f1758e4a` | 2.2.1 | blocked/P1 | 无 |
| nvfp4_fused_moe | `3563c3b7-2b7c-4967-9d41-9c2042e53d2c` | 2.2.2 | ready/P6 | d2be6c2b-8a06-4f4b-9f23-61ed6cda8db1 |
| fmha_mxfp8_qkv | `45ea6608-66b2-4915-b47d-59d2c115602e` | 2.2.7 | termination_unconfirmed/P5 | acb68909-aaaa-4bc2-8b16-94904752af8e |
| gdn_prefill_bf16_state_bf16 | `45f9bb51-77f3-4285-b7a7-251cf2744fa1` | 2.2.7 | ready/P6 | 9699c87b-4ae0-412a-b966-defc914407ef |
| fp8_scaled_grouped_gemm | `4aac1ea4-94fa-4b4c-93ca-d48a8eb08996` | 2.2.2 | ready/P6 | b259d14c-3496-4802-88df-6f4632d90fd0 |
| nvfp4_fused_moe | `53d4bebe-882d-4a66-98b1-287d132dd6f3` | 2.2.7 | cli_error/P4 | 无 |
| gdn_prefill_bf16_state_bf16 | `67778ed7-dde4-41a4-aaa5-34a6d58228f4` | 2.2.1 | blocked/P1 | 无 |
| dual_cache_latent_bf16 | `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7` | 2.2.9 | termination_unconfirmed/P5 | b6d50336-fe0e-4689-a52a-a50ffefcbd6a |
| gdn_prefill_bf16_state_bf16 | `9699c87b-4ae0-412a-b966-defc914407ef` | 2.2.6 | cancelled/P1 | 97902388-eda0-4cbd-b8a6-ba71440e86ab |
| gdn_prefill_bf16_state_bf16 | `97902388-eda0-4cbd-b8a6-ba71440e86ab` | 2.2.2 | ready/P6 | 67778ed7-dde4-41a4-aaa5-34a6d58228f4 |
| fmha_mxfp8_qkv | `acb68909-aaaa-4bc2-8b16-94904752af8e` | 2.2.2 | ready/P6 | 166fc15f-e49e-42ef-b429-c7f507f5ed53 |
| fp8_scaled_grouped_gemm | `b259d14c-3496-4802-88df-6f4632d90fd0` | 2.2.1 | blocked/P1 | 无 |
| fp8_scaled_grouped_gemm | `b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b` | 2.2.4 | ready/P6 | 无 |
| dual_cache_latent_bf16 | `b6d50336-fe0e-4689-a52a-a50ffefcbd6a` | 2.2.9 | cli_error/P1 | 2d252bf9-d9a3-4fa4-96d2-90a98e25eb34 |
| nvfp4_fused_moe | `d2be6c2b-8a06-4f4b-9f23-61ed6cda8db1` | 2.2.1 | blocked/P1 | 无 |
| dual_cache_latent_bf16 | `d5a73c01-7c3a-45b2-a3ce-0d5eec634808` | 2.2.2 | ready/P6 | e6b585c3-f060-45cc-8f41-aba22512717a |
| mla_expanded_fp8 | `e3f7704f-061f-48ff-8131-cc317afe6003` | 2.2.2 | ready/P6 | 2dfb565c-28a6-48de-9cdc-ab31f1758e4a |
| dual_cache_latent_bf16 | `e6b585c3-f060-45cc-8f41-aba22512717a` | 2.2.1 | blocked/P1 | 无 |
| nvfp4_fused_moe | `feb0c655-4e13-43ac-817d-1f457f31c029` | 2.2.10 | cli_error/P2 | 2c4207c8-000e-43ea-8da0-2eb6a6079aee |
