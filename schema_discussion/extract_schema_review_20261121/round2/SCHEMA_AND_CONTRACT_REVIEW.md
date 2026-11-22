# 第二轮：Schema、协议契约与最小修复复审

2026-11-21；基线 HEAD `484902855fe7be82a74fe7c394503d1d7d222dbb`。这是独立复审与可执行计划，**产品修改尚未实施，也不是新 P6 批准**。只写本报告及 `schema_contract_evidence/`；未改正式 YAML、规范、工具、receipt、run、issue 台账或 kernel，未启动提取、恢复、monitor、模型、GPU，未提交或 push。现有用户改动和 `workflow_architecture_20261120` 均保留；后者不是本轮的已批准设计。

结论：第一轮主要归因成立，但应补齐修复范围、压缩实现方案。明确待修的是 **9 份 YAML 和 2 份 dispatch**；当前证据不能支持“先前修过，所以正文已完整”。Tolerance 的 100 字符确实与已知来源记录冲突，单改文本不能可靠解决；只需调整这一项现有预算，不改字段结构。`concurrent` 和短路问题可以用短规范澄清收口，不必新增调度机制或再次要求用户选择。

## 1. 独立覆盖与证据层级

已读第一轮两份报告及两个 FINDINGS、原 [ISSUE_RECAP](../../extract_twelve_20261119/ISSUE_RECAP.md)、[FINAL_REPORT](../../extract_twelve_20261119/FINAL_REPORT.md)、[DELIVERY_INDEX](../../extract_twelve_20261119/DELIVERY_INDEX.json)，并按问题读取当前正文、原回执、固定源码和检查实现。

独立按 DELIVERY_INDEX 的精确 12 个 run 扫描 **372 份 `receipt.json`**，得到 **24 个 unique candidate ID、104 次出现**。逐项比对第一轮映射的回执位置和 SHA，全部一致，无遗漏或多列。这个结果证明候选覆盖完整，**不证明候选建议都对，也不证明 33 份设计全域正确**。逐项处置见第 8 节。

另核对当前 **33 YAML、12 dispatch、12 publication receipt**：45 个成品文件仍与各自旧 publication proof 匹配，审计前后 57 个文件 SHA 不变。这证明第一轮发现针对的就是现行已发布字节，不能用“也许已在别处改过”回避。

可复现证据：[audit.py](schema_contract_evidence/audit.py)、[result.json](schema_contract_evidence/result.json)。本轮没有重新逐算子通审全部数学、动态域和 CUDA 同步；原回执只作为定位和当时判断的证据。其他第二轮的控制、规范化事务与安全恢复结论由 [CORRECTNESS_REVIEW](CORRECTNESS_REVIEW.md) 主审，本报告只明确交界。

## 2. 明确需要的正文纠正

以下 `S-*` 沿用第一轮编号。`accept` 表示同意事实和必要行动；`refine` 表示保留问题但修订范围或方案；`reject` 只否定指定提议；`defer` 明确不在本轮实施范围。

### S-01/S-02：masked GEMM 三项一起修，不能只补 min

**accept，并细化数值表述。** [当前 root](../../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/root.yaml) 第 112 行的 `eff=masked_m[e]-128*mb in [1,128]` 是残留正文错误。固定 `candidate/kernel.cu:1181–1190` 两个编译分支都先截到 `bm`，再向上对齐 16；本节点 `bm=128`。合法 `masked_m=256, mb=0` 在现行正文得到 256，源实现为 128。

建议将三处含义一次写正确：

1. `eff=min(128, masked_m[e]-128*mb)`，`eff_a=16*ceil_div(eff,16)`，只对实际产生的非空 worktile 使用。
2. 第 118 行与 Contract `y` 的对应只能涉及 `i_l<eff`。`eff<=i_l<eff_a` 确实计算、窄化和写回，但 Contract 未给这些 inactive 行定义 `y`；不能让一个未定义结果参加等式。保留实际访问域、有效结果域与已分配 capacity 的区别，不恢复“padding 永不写回”。
3. 第 39、89、117–118 行的“解码/乘积全精确、只改变归约次序”过强。`2^100 * 2^100=2^200` 超出 FP32，`2^-127 * 2^-23=2^-150` 是 FP32 最小次正规数的一半；这些单独可表示的操作数可由 Contract 允许的 payload/scale 构成。位数少且尺度为二次幂不足以证明全域精确。

Contract 写真实的 FP32 解码求值和转换；Design 写实际 block-scaled datapath、归约与舍入。可以说明展开尺度后在实数代数上是相同点积，但不能将它升级为所有允许输入上的 FP32 恒等式或必达容差的保证。**不缩小 Contract、不改阈值、不据这些反例宣称 GPU 已失败。** 这只是撤销不成立的文字保证；未提供的全域等价界仍未提供。

本轮已用 CPU 枚举 capacity=128..1024 的全部 128 倍数、每个 `masked_m=0..capacity`，共 **4616 个输入组合**。修正索引公式满足 `1<=eff<=eff_a<=128`、写回不越 capacity、相邻 mb 不重叠、尾部额外行至多 15。只证明该明确索引公式，不证明 TMA/GPU 行为。

### S-03：m/l 副本结论正确，root page_regs 也有同类漏账

**refine：将第一轮“顺便核对 page_regs”收敛成已证实的具体修正。** [FP8 FMHA root](../../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/root.yaml):330–341 与 [decode](../../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/decode_split_kv.yaml):221–232 的计量不能按数学值去重。

固定 `candidate/fmha.cu:421,582–596` 为每线程 `rm[MT][2], rl[MT][2]`；`row0=m0+16*mt+(lane>>2)`，四 lane 保留同行状态。第 1192 行只让 `(lane&3)==0` 发布 m/l，并没有消除其余 lane 在计算期间的状态。

| 存储 | 当前 | 按已登记值的同时存活副本计 | 必要改动 |
| --- | ---: | ---: | --- |
| root `m_run`、`l_run`，各自 | 512 B | `128线程 × MT=2 × 2 × 4 = 2048 B` | shape 增已有机制可表达的复制轴，total_size 同步；Math 的 128 行形状不变 |
| decode `m_run`、`l_run`，各自 | 256 B | `128线程 × MT=1 × 2 × 4 = 1024 B` | 保留 `slice:4`，另计同一行的四 lane 副本 |
| root `page_regs` | 512 B | `4 warp × 32 keys × 4 words × 4 = 2048 B` | 登记四 warp 私有副本；第 360 行动作本来已称 private page_regs |

`fmha.cu:608–645` 每个 warp 的 `MetaRegs mr` 都保存整块元数据；root 的 WM=4、WK=1，四份副本明确。decode 为 BN=16，只有每 warp 的 16 lane 执行 metadata 的写入和消费；其现有 `4 slices ×16 keys ×4 words=1024 B` 是活跃 payload 账，**不要再机械乘四**。若要讨论所有线程槽的 SIMT 分配，完整槽位容量为 2048 B，但最终编译器如何分配、复用及计入角色预算不能由这份逻辑 payload 表反推；不为此补做编译/硬件调查。

DOC5:355–363 已有容量公式，只需把“逻辑总量”澄清为**登记的值区域在成员内同时存活的独立副本总量，不能将相同值的副本去重；整 kernel 寄存器分配仍由角色预算/Realization 表达**。不要额外乘一次 `Worker.unit_count`、不要把 Storage 区域与 `Role.Register` 重复累加，也不要用源码数组字节数宣称最终寄存器数或驻留数已经得到证明。

### S-01/S-05：保留 KDA 精度分流，删去虚假的 null 地址要求

**accept。** [prepare_chain](../../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml):46 的 `ki_f32` 分流正确。`prepare.cuh:803–817` 从 FP32 `ki_pair` 直接算 `kr_pair`，`839–855` 才分别窄化；root 的 restore 遍则重读已窄化 BF16 `ki`。两族不能为了统一写法合并。

同文件第 581 行 Forbid 一概要求“state index pointer、use_state_indices、两个 FP32 state pointers 保持 null/zero”不成立。`runtime.py:99` 建立一个元素的 `unused_indices`，`solution.py:208` 传入，`chain_launch.cu:281` 写入其地址；它不是 null。`use_state_indices=0` 与两个 FP32 指针为 null 是另外的事实。

最小修法：Choice 只保留“不启用状态索引间接寻址或 FP32 state；CHAIN 仍按 request 解析 state slot”的实质约束，去掉对未读占位地址的约束。无需新 nullable/alias 类型，也无需把未用 ABI 全部逐项变成实现要求。`slot_start` 仅在 cadence>0 使用的条件可在 meaning 顺手说清；Math/Pipe 已有守卫，不升级为框架功能。

### 其他已经发生的内容修正：保留，不重复制造门禁

| 原问题组 | 本轮复核与去向 | 边界 |
| --- | --- | --- |
| dense GEMM 全 K 归约 | `weight_band_mma.yaml:33–49` 的 `g=16*t+4*q+u`、终点 `4*block_count-1` 完整覆盖；对 5 个多阶段规模做组号枚举，各组恰一次。保留修法。 | 不回退单 stage 简式；没有重新证明所有浮点误差。 |
| dense stream-K partial load | `pair_stream_k.yaml:252,281–283` 把实际同 lane 普通读取记 serial，只在 trailing segment 存在；删去更宽条件域的不存在前驱正确。 | 不能据此把任意 async 改 serial；无需要新增“保险等待”。 |
| FP8 MoE 空提交组、小形状 decode；mixed Sparse 单槽复用；BF16 MoE 原子序号；MLA TMEM permit/释放 | 现有动作存在性、serial、退休/复用规则已能表达，保留原修订。对修改处用空域、单点、恒假 guard 定向核对；不新增动作类别。 | 有限图不会从源码自动发现漏写动作；本轮不声称重新穷尽每条流水线。 |
| MLA/Sparse CTA/warp/head 分工、TMEM，FMHA lse 归属，GEMM 双 CTA 与两次 32B store | 保留实际分工与字节账；FP8 FMHA 上述残留单列，不从某一旧修复推断全账正确。 | 不开展全仓库硬件资源审计。 |
| GDN when/out 对齐、vec==0 记录 | 三份当前 when 均保留 out 基址 4B 对齐与 stride 偶数，vec==0 只取消状态向量访问要求。保留。 | 源 host 标量输出分支由奇 stride 触发，不能虚构“基址不对齐即回退”。 |
| FMHA 短 KV/wave-cap/生产路径、GEMM SM 数方向/工作区回退、MoE 队列/finalize 分区 | 原审阅及当前 dispatch 定义承担条件，继续保留；没有证据要求新 selector 字段。 | Contract、来源支持域、节点 when、实测覆盖仍分开。 |
| absorbed MLA 默认值/workspace、KDA scale=0 | 23 个允许省略的默认值和 workspace 写入白名单是正确方向；不把被入口拒绝的 out=None 升为合法调用。scale=0 的已记录差异留在 Gap。 | 不修改 kernel/reference，不用收窄任务域掩盖来源差异。 |

这里的“保留”不是为未逐条复演的源实现再签正确性证书。第一轮已有分层和 P3/P5 核对方法足够；仅在现有段落合并补充“完整归约与窄化点、三个访问范围、独立副本、退化条件动作和最终退休”几个容易漏的落点，**不另建全局 checklist、语义证明器或逐算子字符串规则**。

## 3. S-07：预算冲突是真的，最小修法仍是字符串

**accept 问题，refine 精确修复清单。** `text_policy.py:58` 为 Tolerance 定义 100 Unicode 字符/1 逻辑行；DOC1:256–265 与 P1:35 要求已知阈值、基准、局部标签完整保留。以下六字段即使去空格、把小数压缩为 `.01`，也已有 **105 字符**，尚未写 source-only、缺口或 no-defaults：

```text
atol=.01,rtol=.05,max_mismatch_fraction=.001,max_rel_l2=.01,relative_to=candidate,mode=elementwise_and_l2
```

因此“只把 mode 补回再压缩几字”不是可靠方案。将字段名改得难以识别、把正文事实移到旧回执、把显式 null 写成 absent，都没有解决契约冲突。

`mla_absorbed_fp8/performance/ACCEPTANCE.json:268–294` 明确给出 out 的六字段，以及 lse 的 `max_rel_l2:null`、相同 mode。当前 root:125–126 缺 mode，且 lse 写 `max_rel_l2 absent`。这是**提取遗漏/改写已知事实**，不是来源未提供；来源没有交付的完整比较器定义仍然是独立缺口，不能从标签 `elementwise_and_l2` 推导出完整公式、计算精度、非有限值或检查路由。

独立扫描本批每个已绑定 acceptance JSON 的具名 tolerance 后，需恢复标签的精确范围如下；不是仅 absorbed MLA 一项：

| root 所属算子 | 对应 Contract 输出 | 已知待恢复内容 |
| --- | --- | --- |
| `bf16_fused_moe` | `y_result` ← `input.y` | mode 与 `max_rel_l2=null` |
| `mla_absorbed_fp8` | `attention_output` ← out/return[0]；`attention_lse` ← lse/return[1] | 两者的 mode；lse 的 null 状态，撤掉 absent |
| `mla_nvfp4` | `out`、`lse` | mode、`max_rel_l2=null`；恢复 `max_mismatch_fraction` 全标签 |
| `paged_indexed_latent_fp8` | `out` | mode；恢复 `max_mismatch_fraction`、`max_rel_l2` 全标签 |
| `kda_raw_prefill_state_bf16` | `attention_output`、`final_state`、`checkpoint_states` | mode 与 `max_rel_l2=null` |

合计 **5 个 root、9 个输出**。来源文件 SHA、实际 JSON pointer、所有出现过的输出名和现行字符串均保存在 `result.json`，可逐项核对。其他 7 个绑定验收 JSON 未发现该六字段组合；这不意味着完整比较器已经存在，也不允许统一给它们补 mode 或默认 null。

最小产品修改：

- 仅把 `BUDGETS` 中 `Golden_reference.Tolerance.*` 改为已有档位 **300 字符/2 逻辑行**，继续使用 `text_size/check_values`；不新增结构字段、DSL、动态豁免或第二份预算表。175 字符的一行完整来源记录已由现有 helper 验证可容纳；300/2 给准确缺口说明留空间，150/1 仍不够。
- 按固定来源恢复上述文本，保留数值、输出/别名对应、`relative_to=candidate`、显式 null；没有键才写 absent，不能由 null 推断该比较不执行或通过。
- P1/DOC1 无需重写验收体系；只在现有来源记录段明确“长度冲突不得靠丢已声明字段解决”，保留未知比较器限制。已有 `comparators.validate_comparisons` 的 output/relative_to/atol/rtol 绑定继续使用。
- **不新增全标签自然语言解析 gate。** 现行 comparators 本来没有证明 mode、L2、mismatch 组合语义；预算修复也不证明未来模型必不遗漏。此批验收用来源对象与九条改后文本的逐项保真复核，后续仍由独立 reviewer 负责已有事实是否完整。

KDA 最终回执 `5ab1ac5e…` 的 `unresolved` 曾称六字段“逐字保留”，现行正文却缺两项。这说明旧批准可能遗漏明确字段，不能拿该结论替代本轮原文对照；也不能为了自动验证这句话而构建新的比较器平台。

## 4. S-08/S-09 与 X-01～X-06：对齐义务，不增加一层契约系统

| 项目 | 独立结论 | 最短可靠处理 |
| --- | --- | --- |
| handoff | 硬门禁是 **P2/P3/P4 且 approve**：非空、1200 字符、20 非空行；`response.md` 和 JSON Schema 缺数值。 | 在 `receipts.py` 把已有谓词/数值收成小函数或常量，让 validator、schema description、`prompts.build_prompt` 使用同一来源；保持原阈值。 |
| 来源类别 | P3/P5 要 implementation 与 host，同址不抵消类别义务；实现入口对应 `bindings.entrypoint`。 | 在动态材料投影现有 STAGES/bindings 为类别→path/symbol 提示，错误显示缺类及期望位置；同一范围由角色分别登记两类。不要自动复制引文或新建 evidence-ID 协议。 |
| 坐标/索引 | 真实行范围、0-based 字符偏移、数组索引有效性均是现有可靠边界。 | 保留准确字节读取，返回原 JSON path、实际行数、选段长度、索引上界；不 clamp、不改 offset=0、不删除坏锚点移位、不搜索近似文本代选。 |
| Version | DOC0:37/43、P5、`artifacts._versions` 一致；GDN 首稿写 2 是作者错误。 | 提示一句：`first_draft=true` 的未 authored Version=1 占位仍是首稿；共享元数据同步后已有更高版本按现规则。工具不自动钳到 1。 |
| YAML 文字预算 | 除 Tolerance 外，没有证据需要普遍放宽；已按解码后的 Unicode 字符/逻辑行计数。 | 用现有 checker/helper 精简重复内容。不要把 handoff 的“非空行”套给 YAML，也不在每阶段抄整张 BUDGETS。 |
| dispatch 基线链 | 正式变体必须直链当前完整 YAML；P2/P4 允许未来计划。 | 在 P2/P4 现有“仅计划”段补明：未来基线的线索放“范围”；仅已存在完整基线进入正式变体条目。P6 集中落实。不给未来节点造空 YAML，不伪挂 root。 |

本轮 [receipt_matrix.py](schema_contract_evidence/receipt_matrix.py) 使用现有隔离 fixture，经真实 `normalize_response/validate_receipt` 测 **23 例**：P2/P3/P4 approve 的 1200/1201 个 Unicode 字符、20/21 非空行、夹空行、空 handoff；同阶段 revise 的 3000 字符；P6 对同类长度无该限制。所有结果符合现行谓词；当前 JSON Schema 对这些形状均接受，说明 provider schema 现在只承担结构子集。P5 未在该矩阵重跑；其阶段不命中代码谓词，原 GEMM P5 2841 字符/8 行的合法实例证据仍在 [恢复记录](../../extract_twelve_20261119/recovery/gemm_p5_split_k_anchor/README.md)。

**不强推 provider 条件 JSON Schema。** 最小实现先共享 description/prompt 的完整义务和 Runner 的权威校验；明确 provider 只校验支持的结构子集即可。若以后证实所支持 backend 都能执行条件关键字，再从同一规则派生，不能无条件在 handoff 上加 maxLength，也不能另写非空行正则。`validate_shape` 当前不实现 maxLength/组合条件，因此只添加 schema 关键字不等于两端已一致。

生成描述时只构造 handoff 字段自己的 schema 副本；现行 `TEXT` 是多字段共享的字典，不能直接给全局 TEXT 加交接限制或说明。数值仍只有一个维护位置。

X-04 的独立机械错误累计与 schema_errors 细目传递有价值；应复用现有校验逻辑，不另设一个允许更多输入的“预检成功”通道。**三件事分开**：坏回执不被接纳/批准；保留草稿须独立通过固定输入、checkpoint、Version、Source 和写入权限门禁；是否让原角色重交按真实失败分类处理。无效路径枚举或错误 decision 被拒绝，不等于已经越界读取或获得批准，不能一概变成人工硬停；固定字节漂移、调用身份与权限/native 证据问题也不能一概当普通格式。精确分类和 X-05 自动补正预算采用同伴第二轮结论，不在本报告重复建表。

无需产品化的轻问题：

- `--skip-hardware` 明确跳过硬件区及其预算，不能偷偷改 flag 成“有时仍报 gate”。P1 Execution 的 launch 配置、SMEM 数量等越层细节可精简，或归 Design/Realization；不启动硬件材料补齐。
- flow scalar 漏引号仍拒绝；可在已有 source-style 诊断中加提示，优先级低，不猜测重写 YAML。
- layout warning 可补首个差异区域及可写归属；`pending_corrections` 已允许正常提交，不要求 worker 手工消除只读区域的 warning，不扩大权限。
- `.gitkeep` 与输出叶目录不存在的约定用启动说明解决，保留 no-replace；不加自动删目录或合并模式。CLI 模型告警不阻断的事实保留，不加模型重试/替换机制，也不从告警消失推断实际上下文/effort。
- 辅助脚本误读 JSON、参数位置、摘要类型、usage 未结算，属于调用方错误；README 给现有接口示例即可。已有补正/失败计数的轻量展示可保留为小改，不做通用适配器或第二套控制入口。

## 5. S-10/S-11：短规范澄清足够

### 短路：accept；不改 evaluator

`pipeline/expressions.py:255–264` 已从左到右短路。实际运行得到 `false and 1%0==0 → false`、`true or 1%0==0 → true`、`true and 1%0==0 → division by zero`，见 `result.json`。

DOC5 的 if 段补一句：“and/or 从左到右短路，未求值分支不要求其整数表达式在本次调用有定义；实际求值的表达式仍须有效。”DOC0 如需交叉引用仅链接此处；不把该规则泛化成 shape/索引可无定义。不改 KDA 的 cadence 守卫，不添计数器、guard 字段或第二表达式解释器。

### concurrent：refine；不视为必须用户另选的产品分叉

第一轮指出真实措辞张力正确，但把它升级为新的用户语义选择过重。DOC4:248 已以 step 标识工作提交与 worker 实例，:260/277 已说明无隐含屏障、无同时驻留保证；DOC5:479–481 与当前 checker 又按逻辑 worker 身份区分局部 Storage。现行 `rules/schedule.py:222–225` 明确：concurrent worker 身份不同，片上可达性/可见性/寿命须靠实际保证判断，concurrent 本身既不证明也不一律拒绝。有限图只展开单 step 的代表 worker，不是跨 step 存续证明器。

BF16 MoE 的实际实例已有充分描述：[root](../../../schema_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/root.yaml):248,453,492,502–511 指明 GEMM EPI 在 sentinel/TMEM 退休后由同一物理 warp 承担 COMBINE，并执行 grid/dispatch 会合；源 `kernel.cuh:1177–1182,1229–1244` 对得上。146 CTA、73 双 CTA worker、1168 warp worker 是**不同 step 的逻辑映射**，不能把它们的物理容量相加，也不能用现有前三个 order 强行要求映射相同。

推荐只改 DOC4:25/277 与 DOC5:479–481 的相关句子，保持四个 enum：

> concurrent 的各 step 使用分别映射的逻辑 worker 群体；不隐含 step 列表顺序或同时驻留。不同逻辑 worker 可在明确退出/交接后先后使用同一物理执行单元；必须写清物理对应关系、不能同时活跃的范围，以及旧存储和在途访问的退休。concurrent 本身不提供这些保证。

“可独立推进”应理解为没有 enum 自动施加整体串行次序，仍受已有 Handoff、动作和显式物理接管约束；不能把它写成无条件并发能力。DOC5 保留前三个 order 的共用 Storage 规则；不同逻辑 worker 复用物理地址，不自动成为同一存储身份。

这澄清现有实例和检查边界，**不放松容量/寿命责任，也不加新 order、跨 step 调度图或资源接管注册表**。可用该 BF16 MoE 的小型说明作为正例，并保留“同一 warp 同时承担两份无交接工作”不能成立的反例；不宣称普通 checker 能自动证明它。只有用户另行要求 concurrent 必须物理互斥，才需要新表达能力的设计选择；当前不应主动制造这个分叉。上述仍是待实施建议，不把它记成规范已经修订。

## 6. S-09：已发布修订与 current 元数据的最小流程

**accept 身份边界，refine 实施方式：复用旧六项格式，不复刻旧六项整场归档。** 旧 [六算子当前索引](../../schema_audit_and_cleanup_20261119/CURRENT_SCHEMA_INDEX.json) 和现有 `schema_wiki.current_schema_manifest.v1` 已区分 `runner_p6_receipt:false`、人工源审阅/结构检查、当前 SHA/Version/Source 与 historical_receipt。这足以维护本批修订；不需要新发布器或回执种类。

建议未来实际修订按以下顺序收口：

1. 固定改前成品摘要、Version、来源身份及旧 receipt；在本批维护证据下保全本次改动涉及的旧文件，或引用已核实可还原的旧 checkpoint。旧 run/publication proof/DELIVERY_INDEX 保持历史原义。
2. 对下表明确文件作一次有限修订，核对实际 provider/consumer。按 DOC0:43，**YAML 数据内容变化递增该文件 Version**；纯格式不递增；只改 root Contract 时仅 root 增版，未改文件的有效内容追溯需同时固定真实 provider 字节。修改 prose 字段也是数据修改，不能只因“编辑性”就豁免 YAML 版本。
3. 按来源独立复核改动及实际影响，运行受影响目录的 `--read-only --skip-hardware` 检查；dispatch 编辑用现有目录/链接校验。记录检查器/规则实际版本或字节、命令、退出码和改后文件清单，不把旧 100 字符规则的检查结果冒作新稿通过记录。
4. 对改动目录新增或更新 `.extraction/current.json`，复用既有格式，绑定**整个当前目录**的 YAML/dispatch 摘要、Version、Source/源文件 SHA、有限审阅范围、checker 证据和旧 receipt 路径+SHA。`runner_p6_receipt:false`；未改文件仍列入当前闭包，使 root 或 from provider 的变化不会被单文件旧批准掩盖。
5. 新的批次当前索引只指向这些 current 清单，明确当前人工修订和旧 P6 两种证据范围；原十二项 DELIVERY_INDEX 不改签。没有 current 的未改目录继续以匹配实际字节的 receipt 为证据。旧六项已归档的 receipt 保持现路径；本批 receipt 留原路径作历史即可，**不为一致外观再搬迁几百份过程文件**。

每个算子的 current 是当前内容入口，批次索引只是指针/摘要投影，不再维护另一份可独立修改的批准事实。DOC/Template/预算变化属于新的规则快照；不能把它们塞进既有未完成 run 的 Extract Python-only 修复续接。此次十二项已发布，更无恢复迁移需求。

`current.json` 是可核验的当前内容与审阅清单，不是 Runner 可消费的新批准。`publication.py` 仍按原 checkpoint/receipt 的完整 manifest 做 no-replace；不要为了手工修稿让旧 run 自动 republish、覆盖目录或通过 current 跳过发布检查。若未来明确要求新 Runner P6 认证，才另走相应审阅流程；本轮语义修正和文案维护无需重开全部 extract。

明确修订清单（版本以目前字节为基准，实施时再次读取）：

| 算子/文件 | 当前 Version → 建议下一版 | 内容 |
| --- | --- | --- |
| masked GEMM `root.yaml` | 4 → 5 | clamp、active/padding 等式域、数值精确性声明 |
| FP8 FMHA `root.yaml` | 4 → 5 | m/l 及 page_regs 副本 |
| FP8 FMHA `decode_split_kv.yaml` | 2 → 3 | m/l lane 副本 |
| BF16 MoE `root.yaml` | 4 → 5 | 完整 Tolerance；无需为 S-11 重拆设计 |
| absorbed MLA `root.yaml` | 4 → 5 | 两条 Tolerance，修 null/absent |
| NVFP4 MLA `root.yaml` | 2 → 3 | 两条完整 Tolerance |
| indexed Sparse `root.yaml` | 4 → 5 | 完整 Tolerance |
| KDA `root.yaml` | 3 → 4 | 三条完整 Tolerance |
| KDA `prepare_chain.yaml` | 2 → 3 | Forbid 的占位指针断言；保留 ki_f32 分流 |
| mixed Sparse、GDN 各自 `dispatch.md` | 无 YAML 增版 | 将过程口吻改成当前来源覆盖与限制；更新当前摘要 |

例如 mixed Sparse “本次未运行 GPU、未构建扩展、未执行验收”，可归入维护证据；正文表达“所列阈值和计时来自固定源码及随包回执；正确性仅覆盖来源的 SM103 采样”。GDN “已记 source_gap issue/无 GPU、不复现”同理。**不能删掉**具体 fence/屏障风险、比较器/逐输出/机制 A/B 缺口、NCU 6/10 和 shape 0/1/3/9 的来源性能例外；也不能把文案清理扩成删除规范中的 P1/P6 职责或测试里的失败语义。

## 7. 工程顺序、验收与可删项

先后依赖应尽量少：

1. **保真与提交事实先修。** S-06 规范化歧义、X-08 控制提交事实、X-15 P2 检查对象采用同伴第二轮方案。S-06 已被同伴推进到隔离的真实 normalization transaction，不能仅以本批作者绕名或静态检查通过宣称已修。回执安全分类必须区分拒绝批准、保稿、允许补正，避免粗分整类错误。
2. **同批改 Tolerance 预算与九份已知 YAML。** 同时落实必要 DOC5/并发/短路澄清，按实际来源保真验证，不先产生一套尚无法通过预算的新正式正文。此工作不依赖新增外部比较器、GPU 或 kernel 修复。
3. **对齐现有反馈与提示。** X-01～X-04 共用数值、类别绑定和诊断细目；X-05 只在明确可补正的类别内按既有 scope 调用预算推进，保留历史累计和独立 reviewer/单裁决预留。不上新策略平台，不默认恢复旧停点或已发布 run。
4. **一次完成当前清单与文案收口。** 新旧证据明确分开，受影响目录及依赖闭包可核验。缓存/超越样本/monitor 新代际等由同伴按各自证据单独修，不能为了“验证已修”启动本批已停 monitor。

验收按实际改动选择，不堆镜像测试：

| 改动 | 必要验收 |
| --- | --- |
| masked | 本轮索引穷举可复用；逐段源对照与 active-only 关系检查；数值反例只否定文字保证 |
| FMHA/KDA | 上述固定数组/线程算术和原始数据流核对；正确副本轴与 total_size；Forbid 不再约束无意义地址 |
| Tolerance | 九条文本与对应原 JSON 的数值/标签/null/输出别名逐项一致；完整文本在新预算通过，超限仍准确报 path/count |
| handoff/证据 | 阶段×decision 边界，Unicode/空白行，合法同址双类别、缺入口类别、越界索引/offset、固定来源漂移；诊断不能改变原数组/引文身份 |
| 现有 artifact 规则 | 首稿 1、正常续修、更高版本 starter 同步、只读 provider、P2/P4 未来计划与 P6 当前直链；不放宽权限或造假基线 |
| current 清单 | 旧 receipt 和已保全旧字节不变；当前全部文件/来源/Version/闭包与摘要一致；编辑证据与语义审阅范围可区分；没有伪造 P6 |

明确可删/不新增：比较器 DSL、Tolerance 对象结构、多消费者 storage 新映射、新并发 enum/调度图、通用 alias 类型、AST 证据范围证明器、第二份预算/错误策略表、全域自动语义 checker、所有动作补等待、自动引用/坐标补齐、输出目录合并器、模型告警自动换模型、为纯文案重开 extract、为了清理把所有历史文件统一搬家。

没有新的必要用户语义选择。当前唯一外部范围边界仍是主代理已说明的“本轮先交审计/可执行计划”；不能把本报告自行转换成产品实施授权。用户若另要求更低的普通补正成本上限或新的 Runner P6 认证，再处理那一项明确取舍。

## 8. 24 个 framework candidate 的独立去向

次数是本轮从原 receipt 重算；每个 ID 均有去向。回执坐标复用第一轮已核验映射，不复制一份 104 条历史清单。

| ID | 次数 | 第二轮处置 |
| --- | ---: | --- |
| `concurrent-order-sequential-warp-reuse` | 8 | **refine S-11**：短规范澄清，拒绝新 enum/强制用户分叉 |
| `execution-summary-budget-unenforced` | 2 | **reject 改 skip-hardware gate**；保留显式豁免，按层次精简摘要 |
| `extent-input-name-overlap` | 3 | **accept 文档示例**：真实 scalar Input 与另名 extent 的等式；不放开双重对象身份 |
| `flow-scalar-comma-reported-as-unknown-field` | 5 | **defer 可选诊断**：可提示引号原因，不改解析结果/严重度 |
| `handoff-single-storage-per-value` | 4 | **reject 新结构**：当前 Math:23/35 已声明 routing_tables 聚合，Flow:256–259 的分量与聚合体去向明确；不能照搬“无处登记”推断 |
| `if-condition-guarded-divisor` | 2 | **accept 澄清**：evaluator 已短路，实际测试确认 |
| `inplace-output-dual-registration` | 3 | **refine**：保留异名 Input buffer/Output result；拒绝候选建议的“只登记 Output” |
| `inplace-output-naming-doc-gap` | 1 | **accept 短例**：同上，Tolerance 按 Output 名对应 |
| `layout-correction-without-field-path` | 4 | **defer 可选诊断**：首差异/权限归属；不变成手工全清 warning 的义务 |
| `math-identifier-case-fold-collision` | 5 | **accept，修法取同伴第二轮**：不能只靠 related 单行补丁或安全测试旧通过 |
| `p2-future-baseline-record-leads` | 8 | **accept 提示澄清**：未来线索留范围，当前正式条目直链 |
| `p2-planned-baseline-variant-placement` | 11 | **accept 同一修法**，不建计划条目新类型 |
| `p2-revise-empty-plan-misattributed` | 4 | **refine X-15**：检查真实 worker 草案；候选所称已批准选择回滚不成立 |
| `p2-revise-empty-plan-snapshot-diagnostic` | 2 | **accept X-15 归因**；保留 approve 与 plan_draft 一致要求 |
| `p2-variant-leads-need-prose-home` | 12 | **accept 同一提示澄清**，不另建永久线索表 |
| `pipe-guard-parameter-range-coverage` | 1 | **refine 审阅方法**：选明确空/单点/恒假边界；不承诺图工具自动穷尽源码 |
| `receipt-impl-entry-anchor` | 2 | **accept X-02 投影**：任务绑定入口与节点 Source 分清 |
| `register-storage-replica-accounting` | 3 | **refine S-03**：除 m/l 外落实 root page_regs；不把活跃 payload 当编译器全部分配 |
| `required-evidence-bound-entry-category-hint` | 4 | **refine X-02**：类别/path 机械门禁；候选的 symbol AST 范围门禁不采纳 |
| `required-evidence-category-binding-same-path` | 3 | **accept X-02**，拒绝工具自行复制类别判断 |
| `template-identifier-case-fold-subscript` | 2 | **refine S-06**：具体机制不能只引用候选；显式注册 I 的样例可拒绝不等于开放数学安全，取同伴证据 |
| `tolerance-record-budget` | 3 | **accept S-07**，恢复 indexed 标签 |
| `tolerance-record-length-budget` | 5 | **accept S-07**，absorbed mode 有源且 lse 为 null |
| `tolerance-source-record-budget` | 7 | **accept S-07**；补明五算子九输出范围，未知定义仍未知 |

## 9. 本轮实际验证与未验证

- `audit.py` 成功：候选覆盖/回执 SHA、45 成品与旧 proof、57 文件保全；4616 个 masked 域输入；5 个 full-K 组号样本；Tolerance 105/175 字符反例及建议预算 helper；4 个短路案例。
- `receipt_matrix.py` 成功：23 例真实现行 receipt gate 与 JSON Schema 对照，隔离目录自动清理。准备 P4 reviewer fixture 时最初未刷新 checkpoint/结论集合，受现有门禁拒绝；纠正测试夹具后再跑，不归为产品缺陷。
- 8 项既有定向离线测试通过，见 [existing_tests.log](schema_contract_evidence/existing_tests.log)：文本计量、hardware 豁免、Contract 预算/判据、计划与当前链接、证据类别、坐标、真实固定引文、比较输出与基准绑定。第一次测试调用漏 `PYTHONPATH=schema_template`，3 项 import error、其余 5 项通过；[原日志](schema_contract_evidence/existing_tests_initial_invocation_error.log)保留，按仓库测试入口修正后 8 项全部通过。
- 未实施上述计划；未验证新提示能降低真实模型返工率、provider 条件 schema 能力、所有动态语义/跨 step 活性、GPU 正确性或性能。既有测试通过和旧批准均不能抵消本报告的明确残留。

不再扩展全域源审计；剩余工作应是按上述范围实施并验证，或由后续 high/xhigh 将计划删减、整合入 discussion。删除冗余时仍须保留数学身份、来源事实、权限和新旧批准范围四条边界。
