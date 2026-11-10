# extract_fmha：全量文字审计 v1.1

结论：现稿依赖自由叙述才能确定部分核心设计，主要是动作的隐含输入、异步提交与真实完成、所有 warp/异步读者的退休、内部 collective、批加载顺序、空分支以及动态 shared-memory 请求。**这不等于证明需要新增字段或事件**：公式和实现原语是有意开放语义；PV 完成、批加载、角色与别名有现有结构写法，复杂选择绑定有复用成本。五份空选择条件和一份 wide/narrow 混稿属于现稿完整性问题；硬件 Contract 占位则是冻结 DOC2 第13行明确允许的 `--skip-hardware` 提取边界，不是违规，也不要求本轮 worker 补硬件。

本报告不授予新批准。输入来自 pause inventory 的 current-plan 最新选定稿：43 份有内容 YAML，26 approved、4 review_candidate、13 needs_authoring；没有把草稿转作批准或把空值算成简洁成果。冻结 manifest 的 run 是 `45ea6608-66b2-4915-b47d-59d2c115602e`，working checkpoint 是 `e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938`。43 份 SHA256 全部与 manifest 相符。

来源为 `source@6e4ca91bd91d998cac99e8dadd83838e153c679d` 下 `kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/`。已用 `git show` 取出 16 份源码/题目/记录，与当前字节逐份相等；见 [source_identity.json](../work/extract_fmha/source_identity.json)。亲读的依据包括参考数学、prepare/compile、host plan/dispatch、graph 路由、TCG/TC 缓冲及关键计算、空 split、GSPLIT、合并和 occupancy helper；没有将未亲读的全部指令或所有编译分支声称为已证明。

规则为审计目录冻结的 README、DOC0/1/3/4/5/6、REVIEW 与示例边界，遵循 [AUDIT_PROMPT.md](../AUDIT_PROMPT.md) v1.1。机器明细见 [annotations](../annotations/extract_fmha.json)；统一校验见 [audit_summary](../data/extract_fmha/audit_summary.json)。

## 完整覆盖及统一统计

逐组阅读并人工分类了全部 **1,397** 个精确文本组；每个使用节点、字段路径与 alias 作用域均检查。结构核对通过可复核的精确对象登记完成：111 种符号属性、36 种工作定义、8 种 flow、8 种 domain、143 种 resource、52 种 role/pipe、12 种 milestone、9 种 realization 引用列表。登记保留全部使用节点和路径，位于 [work/structures](../work/extract_fmha/structures/)。这是精确去重后的全量阅读，不是抽样。

所有字符均按归一化空白后的字符串长度计数，包含公式/引用，不称纯自然语言字数。英文 YAML 与中文 dispatch 分列；raw 与 effective 不相加。

| 范围 | YAML 数 | raw 全字符串声明 | raw 全字符串实例 | 开放字段声明字符 | 开放字段实例字符 | effective Design 字符 |
|---|---:|---:|---:|---:|---:|---:|
| approved | 26 | 724,204 | 733,240 | 623,966 | 632,669 | 638,730 |
| review_candidate | 4 | 124,372 | 124,372 | 106,664 | 106,664 | 110,149 |
| needs_authoring | 13 | 423,227 | 424,530 | 362,147 | 363,450 | 384,023 |
| 全部 YAML | 43 | 1,271,803 | 1,282,142 | 1,092,777 | 1,102,783 | 1,132,902 |
| root 单独 | 1 | 30,706 | 30,706 | 28,384 | 28,384 | 17,143 |

YAML 有 1,344 个开放文本组、441,396 个唯一字符、5,071 次声明。dispatch 有 53 组/53 次声明、14,208 字符。合计 455,604 个唯一字符、5,124 次声明、1,106,985 声明字符。1,244 个 verb 声明展开为 1,266 个实例；root 是 13 个 verb/5,824 字符，中位 438、最大 619 字符。长度不参与分类或删减决定。

所有 43 个节点的 Design 三区均本地定义，**没有 from_parent**；effective provider 都是自身。raw 与 effective 的差异仅是观察范围/alias 展开，不存在跨文件隐式继承。approved 中 fused_d128_group 的 Even/Odd alias 引入 8,703 个开放字段实例字符，needs_authoring 中另一组 alias 引入 1,303；不能把实例重复当作可删除的源声明。跨文件完全相同文字仍各算一次声明，不因重复就具备删除许可。

`scalars.jsonl` 的剩余字符串已核对：主要是 dtype/shape、retention、binding、role/pipe/事件引用、参数表达式和来源；没有发现库存遗漏的重要开放正文。hardware 的 null 骨架不是字符串，不能被文字 census 的零面积掩盖。

## 每个节点的覆盖

下面的“已读”指本轮审计阅读与绑定检查完成，不表示源实现正确或节点获得批准。全部节点共同保留获准跳过硬件检查的研究限制；五份空 when 的适用性留未决。

| 节点 | manifest 状态 | 版本 | launch | verb 声明/实例 | 本轮检查 |
|---|---|---:|---:|---:|---|
| fused_d128_cp | approved | 2 | 1 | 12/12 | 已读；本地三区/引用/资源/事件 |
| fused_d128_group | approved | 3 | 1 | 19/38 | 已读；Even/Odd、空 Odd、cold 限制 |
| fused_d128_key | approved | 3 | 1 | 15/15 | 已读；本地三区/引用/资源/事件 |
| fused_d256_cp | approved | 2 | 1 | 11/11 | 已读；本地三区/引用/资源/事件 |
| fused_d256_tma | approved | 2 | 1 | 15/15 | 已读；本地三区/引用/资源/事件 |
| root | approved | 9 | 1 | 13/13 | 已读；本地三区/引用/资源/事件 |
| tc_fused | approved | 3 | 1 | 11/11 | 已读；本地三区/引用/资源/事件 |
| tc_split_flat_narrow | approved | 3 | 4 | 31/31 | 已读；本地三区/引用/资源/事件 |
| tc_split_flat_wide | approved | 2 | 4 | 29/29 | 已读；本地三区/引用/资源/事件 |
| tc_split_segmented_narrow | approved | 2 | 5 | 36/36 | 已读；本地三区/引用/资源/事件 |
| tc_split_segmented_wide | approved | 2 | 5 | 35/35 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_cp_direct_narrow | approved | 2 | 4 | 31/31 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_cp_direct_wide | approved | 2 | 4 | 30/30 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_cp_xpose_narrow | review_candidate | 2 | 4 | 32/32 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_cp_xpose_wide | needs_authoring | 1 | 4 | 32/32 | 已读；when 空，适用域未决 |
| tcg_d128_flat_tma_direct_narrow | approved | 2 | 4 | 31/31 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_tma_direct_wide | approved | 2 | 4 | 30/30 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_flat_tma_xpose_narrow | approved | 4 | 4 | 32/32 | 已读；设计 reuse 边与源缺 join 分列 |
| tcg_d128_flat_tma_xpose_wide | review_candidate | 2 | 4 | 31/31 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_cp_direct_narrow | approved | 2 | 3 | 28/28 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_cp_direct_wide | needs_authoring | 2 | 3 | 28/28 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_cp_xpose_narrow | needs_authoring | 1 | 3 | 31/31 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_cp_xpose_wide | needs_authoring | 1 | 3 | 29/29 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_tma_direct_narrow | approved | 2 | 3 | 28/28 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_tma_direct_wide | approved | 2 | 3 | 28/28 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_tma_xpose_narrow | review_candidate | 2 | 3 | 29/29 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_fscan_tma_xpose_wide | needs_authoring | 1 | 3 | 29/29 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_cp_direct_narrow | approved | 2 | 5 | 36/36 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_cp_direct_wide | needs_authoring | 2 | 5 | 37/37 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_cp_xpose_narrow | needs_authoring | 1 | 5 | 35/35 | 已读；when 空，适用域未决 |
| tcg_d128_segmented_cp_xpose_wide | needs_authoring | 1 | 5 | 36/36 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_tma_direct_narrow | approved | 2 | 5 | 36/36 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_tma_direct_wide | approved | 2 | 5 | 36/36 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_tma_xpose_narrow | review_candidate | 2 | 5 | 35/35 | 已读；本地三区/引用/资源/事件 |
| tcg_d128_segmented_tma_xpose_wide | needs_authoring | 1 | 5 | 35/35 | 已读；when 空，适用域未决 |
| tcg_d256_flat_cp_narrow | approved | 2 | 4 | 30/30 | 已读；本地三区/引用/资源/事件 |
| tcg_d256_flat_cp_wide | needs_authoring | 2 | 4 | 30/30 | 已读；wide 数学/narrow 管线冲突 |
| tcg_d256_flat_tma_narrow | approved | 3 | 4 | 30/30 | 已读；本地三区/引用/资源/事件 |
| tcg_d256_flat_tma_wide | approved | 2 | 4 | 30/30 | 已读；本地三区/引用/资源/事件 |
| tcg_d256_segmented_cp_narrow | needs_authoring | 1 | 5 | 33/33 | 已读；when 空，适用域未决 |
| tcg_d256_segmented_cp_wide | needs_authoring | 1 | 5 | 33/36 | 已读；本地三区/引用/资源/事件 |
| tcg_d256_segmented_tma_narrow | approved | 2 | 5 | 33/33 | 已读；本地三区/引用/资源/事件 |
| tcg_d256_segmented_tma_wide | needs_authoring | 1 | 5 | 33/33 | 已读；when 空，适用域未决 |

## 全字段分类与适用性

主分类是整个字段的类别，不能把 MIXED 的全部面积算作可删或必需。下面为去重组主分类；声明处的判断使用 occurrence override，因此声明面积与主分类面积不可直接换算。

| 类别 | 精确组 | 唯一字符 | 按 occurrence 判断的声明数 | 声明字符 |
|---|---:|---:|---:|---:|
| DUPLICATE | 19 | 2,523 | 97 | 11,426 |
| NONCORE | 0 | 0 | 0 | 0 |
| STRUCTURABLE | 2 | 192 | 8 | 768 |
| ESSENTIAL_TEXT | 17 | 6,488 | 47 | 20,561 |
| FORMAL | 148 | 65,612 | 1,215 | 204,720 |
| SOURCE_LIMIT | 7 | 2,821 | 8 | 3,028 |
| EXPLANATION | 574 | 78,229 | 1,983 | 182,827 |
| MIXED | 629 | 299,739 | 1,482 | 577,126 |
| UNRESOLVED | 1 | 0 | 284 | 106,529 |

有 279 个 occurrence override：274 个来自五份空 when 草稿的核心适用性，5 个来自 cp-wide 的 narrow 文字/管线冲突。加上空 when 自身的五次声明，共 **284 次声明未决**、106,529 字符；主分类只有 1 个 UNRESOLVED 组（空 when，0 字符），但计入 occurrence override 后，**134 个精确文本组含至少一个未决上下文**（`groups_with_unresolved_contexts=134`）。这些是不同统计口径；284 次未决声明仍计入全部正文，不从分母删除。

统一 `scope_statistics`：

| 范围 | 精确组 | 唯一字符 | 非 FORMAL 且含核心的主分类组 | 对应已判定声明 | 对应声明字符 |
|---|---:|---:|---:|---:|---:|
| yaml | 1,344 | 441,396 | 614 | 1,459 | 571,859 |
| math | 531 | 134,285 | 83 | 243 | 92,954 |
| main_WTR | 715 | 254,377 | 484 | 1,168 | 429,883 |
| dispatch_when | 39 | 43,636 | 38 | 38 | 43,636 |
| dispatch_md | 53 | 14,208 | 5 | 5 | 981 |

这里的核心面积是**这些字段含核心内容**，并非这些字符全都不可删。YAML 的 1,459 次已判定非正式核心声明中，approved 为 953 次/359,550 字符，review_candidate 为 168 次/66,792 字符，needs_authoring 为 338 次/145,517 字符。五个空域节点的相关使用处没有被统一 `contexts_checked` 标签掩盖；具体 override 解释了已检查内容和仍缺的域依据。

## 全部机制 findings

以下每项的完整 unit IDs、所有使用节点、证据、最小反例和结构片段均在机器明细中。覆盖集合互有交叉，不能相加当独立缺口数。节点数不含 dispatch；“+索引”表示另覆盖 dispatch。

| Finding | 覆盖精确组 / YAML 节点 | 判断 |
|---|---:|---|
| extract_fmha-F01 | 101 / 43 | 正式公式、nullable/副作用和边界是有意开放载体；字符串不是漏洞。 |
| extract_fmha-F02 | 489 / 43 +索引 | 必填标题与摘要多无独占事实；可短写，不能删成空字段。 |
| extract_fmha-F03 | 30 / 43 +索引 | 验收基准、未知误差界与来源限制必须保留，不套默认比较器。 |
| extract_fmha-F04 | 54 / 43 +索引 | HostPlan/实际probe/MapsValid及capture范围；跨文件选择定义存在复用成本。 |
| extract_fmha-F05 | 1 / 5 | 五份空 when 草稿；现有字段未完成，不能猜域。 |
| extract_fmha-F06 | 116 / 43 | native MXFP8、BF16、FTZ/FMA、分母树序属于数值定义，部分解释重复。 |
| extract_fmha-F07 | 135 / 43 | ledger成本/单位/范围与by正式引用分开；不能按长短删数学关系。 |
| extract_fmha-F08 | 28 / 43 | 输出math尾部返回None/不读旧目标重复root契约，可局部删。 |
| extract_fmha-F10 | 39 / 36 | 四类源边界：bm覆盖孔、xpose空split竞争、GSPLIT cold未退役、废弃越域/尾部读取。 |
| extract_fmha-F11 | 2 / 1 | DOC2允许硬件null占位和skip-hardware；仅限制硬件/容量研究结论，不是违规或语言缺字段。 |
| extract_fmha-F12 | 397 / 43 | 自定义动作中的隐含输入、到达/退休/完成、collective与空分支是核心。 |
| extract_fmha-F13 | 5 / 3 | PV真实完成可用现有async长形式和done边表达；保持原等待位置。 |
| extract_fmha-F14 | 15 / 32 | 别名存储、环槽和所有读者已可用resource.view/边表达；源缺陷另列。 |
| extract_fmha-F16 | 5 / 14 | 64/16/1批扫描可用现有repeat、视图和produce表达；展开未必更短。 |
| extract_fmha-F17 | 38 / 39 | PDL许可/数据可见和晚wait已有flow/milestone；解释可缩。 |
| extract_fmha-F19 | 9 / 38 | 动态SMEM预留/occupancy请求是独占主要容量事实，Realization有意开放。 |
| extract_fmha-F20 | 74 / 43 | 原语/descriptor/API细节保留兼容事实，不能只凭换API判核心缺口。 |
| extract_fmha-F21 | 63 / 43 | Br/ScanThreads/WorkId/WrittenG/TailRows在strategy定义，已有表达式与引用接口。 |
| extract_fmha-F22 | 19 / 37 | 自然网格与单tile归属重复partition/order/preassigned，可省可选strategy。 |
| extract_fmha-F23 | 7 / 6 | D256 cp-wide草稿wide数学与narrow容量/角色冲突；不制造新设计。 |

**现成结构与开放语义的界线。** 数学/formula、动作本身的内在保证和具体实现兼容条件有意开放；没有把它们计作已证“必须扩语言”。已有结构可承载启动/交接、成员范围、存储视图和事件：例如 `ready: flat_scan.kernel_done` 与 `wait_at: split.scan_ready` 分别保留生产完成与消费等待位置，`scan_ready: [Main/dependency:Scan@done]` 保留等待完成，而不是等待发起。跨文件的 when 不能用 from_parent 共享，参数/函数的紧凑共享有表达成本；本轮没有证明必须新增固定字段或事件。

F13 的局部候选把 root 的 pv 长形式设为 async，令其 done 真正覆盖已提交乘积/操作数读取，并使 `retire:Previous(g)@done` 等 `pv:NumState(g-1)`、`drain:LastPV@done` 等最后一个 PV。首轮移位边自然裁去。原 pipe 中两次 QK→retire 的位置保持，因此没有提前排空 PV。该候选没有直接更改 schema，也没有计入删字收益；全短稿仍需独立等价审阅。

F14 的 split `TensorMemory.view` 已让 `DotTile` 与 `GroupPV` 共用 `[0,128)`，并将 QueryScale/KeyScale/UnitScale/ValueScale/Prob8 定位到其余列；`OperandArena` 和 `OperandScratch` 已将 Query/Key/Value 与 PartA staging 纳入同一物理分配。alias 只复用内容，不能凭其产生存储共享。不能用两个独立 resource 替代已有 alias，也不能忽略尚在途的另一个读者。

F16 给出完整的 B16 数学视图/retention、Batch.view、原 j 域和 `load:B16(j)` 后逐项 `produce:Mscan(...)` 的候选，保持整批载入先于递推；末尾 `tail:Mscan(t)` 可直接使用已有通用 `produce:Mscan(t)`。这说明是表达成本选择，不是给未定义新动词起名字。

## 源边界与关键反例

- **空分支与首次/末次。** 空 repeat 没有主体事件，已列的 prime/retire/tail/finish 不因此消失；首 PV overwrite 不读未初始化 NumState，首 retire 为 no-op。cp.async 的末次空 commit 会改变 wait-group 计数，不能当作空文本删除。D128 xpose 的空 split 仍写 PartA/PartB，FSCAN 最后 split 即使空仍发布 PartM/MTail。所有 materialized partial 在 combine 都被消费，零权重不消除 NaN/Inf。
- **所有读者与复用。** D256/TMA 的三槽 K、二槽 V、三槽 atom、不同 TMEM scale 环有各自退休距离；不能用单条“上一组完成”取代。score/PV TMEM、HalfMax/DenPart exchange 和 staging arena 分别检查了末读者。GSPLIT 的 Even/Odd 精确 alias 使用按 w 与 j 解释，Odd 可无组但仍有 prologue/epilogue；cold 的全 CTA 拷贝不能被局部 role 的 wait-group 当作已完成。
- **源缺陷不是新设计。** `cu:1979` 的 mma.sync group-max grid 使用 qtiles128，而 TC 每 CTA 仅 64 行，后部 bm 未写；四个 tc_split 稿保留 PriorMax 和临时 O/LSE 写。D128 xpose 的源在 Query 写后没有空分支前置 join，`tcg.cuh:5396–5424` 的 staging 后屏障不能修复此前竞争。approved `tcg_d128_flat_tma_xpose_narrow` 中已有 `prepare.done → stage_partials.issue` 的 reuse 边，是设计所需而源未兑现的约束；其他对应稿明确记录实际损坏字节。不能把两者混称源码保证，也不能为“完整”而新造清零/屏障。
- **越数学域及无消费者读取。** 入口 `cLo-1` 可能超出逻辑 MState，但实际读仍发生、值被丢弃；尾部 atom 可能旧/未初始化，写入无后续矩阵消费者的槽。报告没有因此保证任意地址读取安全，也没有新增初始化。相关合法地址/无后续影响的来源范围以正文限制为准。
- **跨工作与跨调用。** 各 CTA 只分配一份 worktile，无 persistent refill。局部资源不跨 kernel 或调用；global partial 保留至声明消费者结束。prepared `input.py:205` 先 compile，因此一次能力 probe 在准备期间完成；不把未准备 lazy forward 的 probe 隐去。`cu:2241` 只有 BeginCapture 成功后才记录 launch_sequence，失败时没有该私有 stream 的计算前缀；成功回放与直接提交使用相同 flow。GraphLaunch 失败内部是否部分执行未获独立证明，不宣称此故障边界已验证。
- **草稿差异。** `tcg_d256_flat_cp_wide` 的 when/数学为四组 wide merge，但仍有“narrow selection Ns<=128”、4 warp、SharedDen[128] 和串行全部 Ns 归并。它在 manifest 为 needs_authoring，不能据名字/长文判已完成，更不能拿其短管线压低复杂度。

## 保守删改候选

25 条精确 quote→replacement 候选覆盖 84 次批准声明，区间互不重叠；按唯一候选算 2,597 字符，按声明量算 **8,584 字符**，全部属于 approved。其中特定自然网格说明省略贡献 W/T/R 区 6,556 字符，数学尾部/免责声明缩写贡献 2,028 字符。dispatch、root、review_candidate 和 needs_authoring 没有计入删减收益。没有交付经独立审阅的完整短稿，因此只称保守候选量，不称安全删除量。

| 原 quote | replacement | 唯一保留依据 |
|---|---|---|
| `Grid=(Sq,H,B), worker_id=i+Sq*(h+H*b). Each CTA completes exactly one row.` | 删除整个可选 schedule.strategy | 本节点 batch/head/query 或 row partition、ascending order、worker_count=R、preassigned `{start: worker_id, count: 1}` 与 refill/exit。 |
| `Initial destination bytes are not read. All valid caller destinations are written and the host returns None.` | 空串 | root.compute 的外部存储/返回契约；原 math 仍保留每条输出公式与非空字段。 |
| `Workspace retains its written bytes; the Python call returns None.` | `Workspace retains its written bytes.` | workspace 行为仍留本段；返回 None 在 root 唯一定义。 |
| `The shared tolerance source records do not provide a design guarantee.` | 空串 | 同字段仍有具体近似与 unknown 误差，root.tolerance 明确限制最终验收。 |

完整 quote、适用 occurrence 和结构来源行号见机器候选，不把同句在未完成稿中的出现也算收益。长的 E16/概率/分母树公式、cold/空 split 来源限制、operand arrival/retirement 定义、SMEM 请求算法均未按长度截断。

## 限制与校验

统一脚本 `python3 -B scripts/summarize.py --operator extract_fmha` 返回 VALID：1,397/1,397 标注、43/43 节点、20 findings、25 候选。校验确认格式、覆盖、quote 唯一性/不重叠与证据可定位，不是 GPU 或完整短稿语义证明。

本轮未改输入、原 schema、规范、Runner、AGENTS.md 或运行状态，未运行 GPU。现有批准不证明固定源 kernel 正确；获准 skip-hardware 不赋予默认硬件/容量保证，也不构成本轮补写硬件的要求。确切来源和数值限制、草稿域缺项、API 核心重要性 uncertain、结构化候选未独立审阅，均保留在 annotations.limits 与逐节点 notes。
