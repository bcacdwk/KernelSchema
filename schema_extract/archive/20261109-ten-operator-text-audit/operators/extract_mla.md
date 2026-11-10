# extract_mla：全量文字审计（AUDIT_PROMPT v1.1）

本算子仍依赖文字定义部分核心设计，主要是 host 几何表及其时点、动态队列协议、复合动作的读写对象，以及原语内部的读先于写保证。现有结构已表达多数角色、容量、循环、发布与复用边；不能把重复的解释和有意开放的数学/动作语义一并算成结构缺口。

已人工审阅 **176/176 精确文本组、2/2 节点和完整 dispatch**，并核对 605 个字符串标量中的 421 个非开放文本目录项。完整分类与每条定位见 [annotations/extract_mla.json](../annotations/extract_mla.json)，统计校验见 [audit_summary.json](../data/extract_mla/audit_summary.json)。`summarize.py --operator extract_mla` 返回 VALID，UNRESOLVED=0；另有 10 组 `core=uncertain`，不表示所有问题均已有确定结论。

## 样本、来源与覆盖

| 节点 | 状态/版本 | raw 文本组 / 声明 | raw 开放字符 | raw 全字符串字符 | 有效 Design 来源 |
|---|---|---:|---:|---:|---|
| root | approved / v4 | 147 / 156 | 23,580 | 26,725 | 数学/W/T 均 root |
| bootstrap | approved / v1 | 28 / 28 | 7,588 | 9,180 | 数学 root；W/T bootstrap |
| dispatch.md | 当前冻结索引 | 11 / 11 | 1,260 | 单列 | 不适用 |

无骨架、未开始或 review_candidate 节点。approved 是冻结提取状态，不是 GPU 或全域数值验证。root SHA256 为 `ca4a71d1b6a460822f4967257d6374ddae17c12b827c3bef8717cba990e4898f`；bootstrap 为 `1513208118579f4ebf63830493c5a2486af4cf9de61efb11bda84665c9563ff2`。规则采用审计目录冻结版本，而非以前的批准理由。

两份 meta_data.source 均指向 `SchemaExtractSource_f91523a50e3d@228934a2253bde4d6f0c4463dd77aeada1bf07a8`。已按本 run 的 inputs.lock 对私有 snapshot 内 **24 个本算子文件**逐一核对 SHA256；candidate/solution.py 为 `78cb2427001cd23d296e907f40f28c9743ba46471eb17ee80a38d584bf45dff8`。亲读了两节点涉及的 kernel/host/入口、reference、输入与域/验收材料；不宣称逐行复刻所有未选路径。完整路径与摘要见 [source_identity.json](../work/extract_mla/source_identity.json)。snapshot 无独立 `.git`，从目录向上找到的主仓 HEAD 不能充当 source ref，本报告仅引用与 lock 字节匹配的 snapshot 行号。

bootstrap 的整区绑定逐项核对如下；它不继承 root 的选择条件或 Realization：

| provider 项 | bootstrap 使用检查 |
|---|---|
| tiles 与 Bq/Bk | root:160–177 的两个 `bind:design.work_schedule` 均解析至 bootstrap:34 的本地 128；q/k 域来自共享 Contract，qv/N 与 Geo 一致。 |
| scores | 六组 FP8 QK、FP32 S、有限 sentinel 与延后 scale 相同；本地 Q/K dtype 引用闭合。 |
| weights | M/D 初值、冻结阈值、FP16 四条 sum 链、E4M3 P 相同；本地 Alpha 两槽、S/P 三槽与相应发布/复用匹配。 |
| value_state | 首 PV 初始化、后续 rescale 与四组 PV 相同；bootstrap 仅 t 域，不误用 root 的跨工作 n。 |
| tiny | q=k≤32 推出 a=0、N=1；原主 QK/PV 仍执行，Final 读 S/V 并写 TinyY/TinyD，Correction 选择同一来源。 |
| output | Fsum/Fmax、sink、BF16 和可选 LSE 相同；bootstrap 整体 store.done 覆盖所有写回，不继承 root 的跨工作早 release。 |
| symbol/retention/ledger | QTile/KTile/VTile、S/M/D/Alpha/P/Y/TinyD/TinyY/Fsum/Fmax 的 shape/dtype/实例范围均在本地资源或角色消费中闭合；成本账仍是同一工作计算。 |
| 本地实现绑定 | main.MMA、Loader、Softmax+Correction 三条 require 均在 bootstrap 本地存在，分别核对选择与理由，未以父批准代替。 |

root 覆盖五个角色，bootstrap 覆盖四个角色。逐项检查 N=1 时中间 repeat 为空、首 PV 无旧累加输入、末轮 Final→P→PV→输出、root terminal 发布但不产生下一 worktile、跨工作 ring/descriptor、所有几何读者 latch、全部行组输出读完后复用。bootstrap 第二个 Geometry 槽未使用仍登记容量；不能因无下份工作就偷偷减槽。Sinks/LSE 只改 payload/写回，未取消已有计算事件。

## 统一统计

字符采用归一化后的 `len`，含数学式、标识与来源叙述，均不是纯自然语言字数。“核心字段”下列均按 summarize 的 `category != FORMAL && core in {some,all}` 口径；字符面积不是不可压缩量。

| 范围 | 精确组 | 去重字符 | 声明数 | 声明字符 | 非 FORMAL 核心组 / 声明 | 该类声明字符面积 | 保守候选节省 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W：工作调度 | 9 | 3,142 | 12 | 4,798 | 3 / 5 | 3,675 | 979 |
| T：微流水线 | 36 | 9,586 | 36 | 9,586 | 29 / 29 | 8,121 | 1,066 |
| R：实现交接 | 6 | 779 | 12 | 1,558 | 1 / 2 | 246 | 0 |
| W/T/R 合计 | **51** | **13,507** | **60** | **15,942** | **33 / 36** | **12,042** | **2,045** |
| 数学 Design | 54 | 6,954 | 62 | 7,015 | 3 / 3 | 698 | 160 |
| 硬件 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| dispatch.when（英文 YAML） | 2 | 1,169 | 2 | 1,169 | 2 / 2 | 1,169 | 0 |
| dispatch.md（中文索引） | 11 | 1,260 | 11 | 1,260 | 1 / 1 | 104 | 0 |
| 全部 YAML | 165 | 28,539 | 184 | 31,168 | 40 / 43 | 15,819 | 2,315 |
| YAML＋dispatch | **176** | **29,799** | **195** | **32,428** | **41 / 44** | **15,923** | **2,315** |

W/T/R 的核心声明在 root 为 20/34，bootstrap 为 16/26，均有覆盖。数学区 54 组中，公式、ID、成本等 FORMAL 与近似误差正文分开；不能把字符串数学计入语言漏洞。硬件为提取时的全 null 占位，零字不能证明硬件表达完整。W/T/R 分项严格复用 scope_statistics 条件，细表在 [scope_breakdown.json](../work/extract_mla/scope_breakdown.json)。

| 整字段类别 | 组数 | 去重字符 | 声明数 | 声明字符 |
|---|---:|---:|---:|---:|
| FORMAL | 40 | 6,343 | 49 | 6,426 |
| MIXED | 37 | 13,625 | 39 | 15,164 |
| EXPLANATION | 69 | 4,587 | 74 | 5,233 |
| DUPLICATE | 14 | 1,445 | 14 | 1,445 |
| SOURCE_LIMIT | 6 | 1,818 | 6 | 1,818 |
| ESSENTIAL_TEXT | 10 | 1,981 | 13 | 2,342 |
| STRUCTURABLE / NONCORE / UNRESOLVED | 0 | 0 | 0 | 0 |

未把任何包含核心定义的 MIXED 整段称为可删。没有找到可用完整结构替代其**整字段所有核心意义**的组，故 STRUCTURABLE 为零；仍有局部现有结构替代/消除重复的 findings。ESSENTIAL_TEXT 也不等于“已经证明必须扩语言”：其中包括有意开放的动作/原语绑定，两组原语绑定的核心重要性仍 uncertain。

raw 全字符串总量为 35,905（不含 dispatch）；没有 alias 展开膨胀。十个跨文件相同文本组额外形成 2,546 字符，同文件 ledger.by/值域的相同文本多声明形成 83 字符；共 2,629。它们只是精确同文，不是可删证明，尤其 Realization 不继承。

有效 Design 与 raw 独立报告：root 全 Design 字符为 17,537，bootstrap 从 raw 7,351（含 `from_parent` 值的 4 字符）变为有效 14,882；合计有效 32,419。开放数学声明 raw 仅 root 7,015 字符，effective 两节点各 7,015，共 14,030；有效 Design 开放字符合计 28,414。W/T/R 均本地定义，raw/effective 的该三类数字不变。以上不与 raw 相加，也未另把共享 Contract 算成 bootstrap 新声明。

## 18 个去重机制与判断

完整 unit IDs、所有节点锚点、反例和结构片段均在 annotations.findings；下表覆盖全部机制。

| Finding | 结论 |
|---|---|
| F01 | 必填 meaning/name/thesis 的简要解释与 ledger.by/step 正式身份分开；前者可短写，后者不能删。 |
| F02 | 可见性、sink、域、非数值单例类型和输入 range 是正式数学/参数内容，不是自由叙述缺口。 |
| F03 | 缓冲 meaning 重复 compute[3]，可缩；compute 内逻辑覆盖、旧值无关、guard 与跨调用状态仍重要。不以仅 Python 对象包装变化作为核心反例。 |
| F04 | tolerance 的未知比较器定义、源拒绝/缺陷必须保留证据边界。 |
| F05 | Geo 的稳定排序门控、Scatter 及缓存来源独占工作次序/槽数选择；`worktile_order: Item` 仅消费已定义顺序，不能把 N 冒充独立工作轴。开放公式可精炼，但不是已消除语义。 |
| F06 | FP16/FP8、fma、FTZ、tiny 路径及未知误差界要保留；重复的 value_state.accumulate 可省略，数学公式不能按长度裁。 |
| F07 | 成本 ledger 与完整 padding/tiny 计算范围是正式关系；baseline 只定义比较口径。 |
| F08 | invariant/iteration、flatten、slots、pipe 与默认依赖已承载多项重复说明；局部删除不能推成无文字全稿。 |
| F09 | Scores、Weights、Final、KV、Item 的复合读写对象须由 verb 绑定；resource.view 不会自动给动作生成读写集合。候选是更紧凑组成表示，现状为有意开放语义。 |
| F10 | 原子 pop、terminal 及最后 pop 重置的跨调用私有队列协议依赖文字；request 的“发信号即返回”决定重叠窗口。图已给 request→publish 及全部旧读者 latch→reuse。 |
| F11 | 旧 PV 读 P 在后 QK 覆盖前完成的原语保证不可由 done 顺序推出；增加 PV.done→QK 会加强顺序。3/4 槽通知 API 的差别本身不算已证核心缺口。 |
| F12 | root OutputState 的复合对象和 tiny/普通行边界仍需定义；已有 `release` 能表达全部源读完但尾写仍在途，不能误判缺少 read_done 事件。 |
| F13 | 半块读取/输出 chunk 内部安排可按真实发布边界再结构化；需先有数学视图和完成依据，不能只造 loadHalf 名字。128+64 TMA 分段未被自动认定核心。 |
| F14 | 三条本地 Realization 的选择与理由分开核对；只有第一条的额外读先写后保证有明确核心反例，其余 API 绑定保留来源，核心重要性 uncertain。 |
| F15 | TMEM 行地址/descriptor 寿命、对齐与硬件拒绝、未选 FP16 PV 路径保留限制。L2 persisting hint 在源码有而 YAML 未写，仅列候选实现信息，未证明漏核心；必要时已有 realization.prefer 可承载。 |
| F16 | bootstrap 全数学 provider、参数、retention、事件和本地实现绑定逐项闭合，无需复制数学或继承 root 选择域。 |
| F17 | 384 launch 线程不能从 active-role 并集推回；空硬件资料不足以量化多余 warp 的驻留代价，保留来源事实，不据此断言新字段缺口。 |
| F18 | host D2H→建表→H2D→主核等待以及录图绑定缺少紧凑的现有计算事件对应，是唯一 missing_field_or_event **研究候选**。不能虚构 host 计算 phase 或未定义 table_event milestone；现状仍由合法 strategy 承担，并未证明必须扩语言。 |

6 条 structural_gap findings 主要落在开放语义，另有表示成本候选；不是 6 个已证语言缺陷。没有漏掉结构性机制后只报告 Scatter：队列、复合读写、早回收、别名原语顺序、初始化/终止和跨调用均独立检查。

现有结构的代表性承担方式如下（原稿已有，不是新增动作名）：

```yaml
# root：跨工作环
n: {flatten: [w, t]}
# Keys / Values view
slot_index: n%KVSlots
# Queue：重用另一槽前，全部旧描述符读者均已捕获
'publish:Next(w+1)':
  ['Softmax/latch:Item(w)', 'Correction/latch:Item(w)',
   'MMA/latch:Item(w)', 'Loader/latch:Item(w)']
# Correction：源读完后归还，尾部写仍可继续
steady: ['store:OutputPrefix(w)', 'release:OutputState(w)', 'store:OutputTail(w)']
```

片段仅展示已有关系，不是可独立运行的新 schema；其他 pipe、data 边及 verb 对象定义必须保留。

## 保守删减候选与限制

**17 条精确 quote→replacement，按合并区间计 raw 声明节省 2,315 字符**：root 1,464、bootstrap 851；W 979、T 1,066、数学 160、metadata 110，dispatch 与 R 为零。按每组候选计一次为 1,897 字符。候选不重叠，未删除互为依据的两个定义；这只是局部保守量，未生成或独立审阅完整短稿，不能称全稿等价压缩已完成。

代表性短写：

- `QTile waits for the initial Q transfer; it does not drain unrelated operations.` → `Wait for QTile.` 保留原 serial 长形式、pipe 位置和唯一 load 来源；无隐式排空已由 DOC5 定义。
- 两个 Softmax.publish 段删去 `The two-slot scale relay is safe because ...` 的完整证明句；保留 Alpha/P 对象绑定、两槽 view、MMA 原语保证和全部现有边。
- `Products issue in one ordered tensor stream, and a completion notification covers all earlier products.` → `Products issue in one ordered tensor stream.` role 已使用 async_ordered_completion，Realization 仍保留累积完成绑定；旧操作数读先于后写的独有句子也保留。
- bootstrap.mapping.strategy → `Each CTA has 384 threads.` WorkCount、worker_count 与一份预分配已在结构；没有直接删除未知的 launch 线程数事实。
- 省略 value_state 的重复 `arithmetic.accumulate` 项，数学和仍保留的 `First PV initializes Y` 承担首轮初始化；该可选映射为空时须整体省略。

最长的 host schedule.strategy、MMA.mma 和输出契约都仍含必要定义，不能整体删除。候选不取消 terminal/rescale/no-op 完成事件，不把 release 改成 store.done，也不以 barrier API 名不同制造核心反例。

限制：未跑 GPU、未修源代码、未改原 schema/规范/Runner；源 TMEM 行地址与 descriptor 复用偏差仍妨碍把该设计当作正确实现保证；比较器缺定义与近似误差未知仍在。capture/replay 解释依据当前 DOC0 绑定规则及固定 host 入口，并无本轮实测。源码 L2 hint 的增益只见源码陈述，现有 `prefer` 足以记录必要时的短绑定；本审计不拿这一遗漏证明新语言字段必要。
