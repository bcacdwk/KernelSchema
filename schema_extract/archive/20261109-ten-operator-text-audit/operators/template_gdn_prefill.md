# template_gdn_prefill：全量文字审计 v1.1

**结论：当前两份设计仍依赖自由叙述确定部分核心执行语义；同时存在可以直接精简、或迁入已有结构的内容。** W/T/R（work_schedule、tile_pipeline、realization）共50个精确文本组、65次声明，其中27组、33次声明含仍由文字承担的核心事实，另1组/1次声明的独立核心重要性未决。它们不是全部“语言缺字段”：合成动作、捕获/归还及原语兼容性多为有意开放语义；外部输出别名缺少专用结构；Inverse逐层warp分工主要是紧凑表示成本；P数量与raw预取的部分编排已有可行结构写法。通知流身份是已核实的源码协议事实，现有证据不足以证明其需专用核心字段。数学公式和选择谓词另计，不以保留公式证明文字不可压缩。

## 身份、范围与覆盖

采用本目录冻结的 `AUDIT_PROMPT.md` v1.1、README、DOC0–6及REVIEW，不以历史批准替代本轮判断。读取两份完整YAML、共享Contract、全部本地Design/Realization及dispatch；对最初148组及统一补入的10组逐项人工标注，检查每个声明作用域。完整标注、逐组理由/证据与全部finding见 [annotations](../annotations/template_gdn_prefill.json)；统一校验结果见 [audit_summary](../data/template_gdn_prefill/audit_summary.json)。`summarize.py`结果为 **VALID，158/158组、2/2节点、0个UNRESOLVED整字段**。这不消除下文的来源限制。

| 节点 | 状态/版本 | 完整覆盖与实际差异 |
| --- | --- | --- |
| root | template / 7 | `B>1`；共享Contract；index_cast→gdn有Cu32 handoff；每worktile更新序列末界；Gate/Beta分槽，Recurrence四通知流；各role持久处理预分配序列。 |
| m64_single_sequence | template / 6 | `B=1`；使用root Contract，三个Design区均本地；仍提交无消费者的index_cast；固定`[0,T)`；raw标量双槽、processed五槽、四chunk前瞻；Recurrence共用ready流。 |

`parent: root`只记派生关系，不叠加父选择条件。两谓词在`1<=B<=16`有效域互斥且覆盖；zero/continuation初态及长度是否对齐均留在节点内。没有空骨架、未开始节点、YAML alias或`from_parent`。两节点的math在Cu32消费及输出索引上不同，schedule/pipeline也有实际差异；不能把跨文件复制直接当作可删内容，也不能用现行整区引用局部覆盖。raw与effective Design一致；单文件阅读时补入共享Contract属于阅读视图，不再加进raw统计。

来源是 `01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/solution.py::Model.forward`。通过只读`git show`取得并亲读selected文件的调度、分派、本例活跃role/原语路径，以及 `reference.py`、`input.py`；源快照、SHA256和原路径见 [source_manifest](../work/template_gdn_prefill/source_manifest.json)。未用旁边可读性较好的另一个实现文件替代selected源码行号。模板仅是规范化示例，不是GPU、语义或性能已验证基线。

## 统一统计

字符均为压缩连续空白后的Python字符串长度，包含公式、引用和标识，**不是自然语言字数或token数**。声明量按各YAML源标量一次；本例无alias，声明量等于实例量。

| 度量 | root | single | 两YAML合计 |
| --- | ---: | ---: | ---: |
| 所有字符串声明/实例字符 | 14,772 | 10,642 | 25,414 |
| v1.1开放文本声明/实例字符 | 11,017 | 7,731 | 18,748 |
| 开放文本声明次数 | 142 | 97 | 239 |
| W/T/R所有字符串 | 7,016 | 6,280 | 13,296 |
| W/T/R开放文本 | 4,320 | 3,898 | 8,218 |
| math开放文本 | 3,725 | 3,692 | 7,417 |
| 硬件开放文本 | 604 | 0（共享root） | 604 |
| effective Design字符串实例字符 | 9,932 | 9,447 | 19,379 |
| verb声明数 / 字符 | 19 / 2,329 | 17 / 2,326 | 36 / 4,655 |

dispatch另为中文导航8组、542字符；英文YAML及其正式记号不与之混称。全算子去重后158组、14,676字符；含dispatch的声明总数247、字符19,290。W/T/R去重50组、6,631字符，硬件另为9组、604字符。158组中有39组、47次声明的`MIXED/ESSENTIAL_TEXT/STRUCTURABLE`已确认含核心片段；其中硬件7组/7次，W/T/R为27组/33次，其余为Contract或数学近似说明。另1个MIXED组（root通知API/流身份，143字符、1次声明）标`core: uncertain`；其文本已审阅，未把核心重要性未决混作未审阅字段。

| 主类别 | 精确组数 | 去重字符面积 | 全声明次数 | 全声明字符面积（含dispatch） |
| --- | ---: | ---: | ---: | ---: |
| EXPLANATION | 73 | 3,953 | 116 | 6,073 |
| DUPLICATE | 16 | 1,459 | 21 | 2,049 |
| MIXED | 25 | 4,692 | 28 | 5,033 |
| ESSENTIAL_TEXT | 14 | 926 | 18 | 1,303 |
| FORMAL | 27 | 3,367 | 60 | 4,521 |
| STRUCTURABLE | 2 | 119 | 3 | 151 |
| SOURCE_LIMIT | 1 | 160 | 1 | 160 |
| NONCORE / UNRESOLVED | 0 / 0 | 0 | 0 | 0 |

这些是整字段分类面积，不是可安全删除字符。`MIXED`中源限制、数值事实与重复说明共存；`ESSENTIAL_TEXT`也不是已证明需扩展语言。短标题与账本标签保留阅读价值，不能从类别直接推导删除所有必填字段。`ledger.by`和新增`math.step`是正式引用/身份，已逐项按FORMAL处理。

## 全部机制与表达判断

下表列出所有17个去重finding。数字为“涉及精确组/声明次数”，不同finding可重叠，不能相加；全部unit ID、节点、最小删除反例、实际候选写法及证据在JSON中。

| finding | 覆盖 | 判断 |
| --- | --- | --- |
| F01 摘要/对象解释 | 70/110；两节点与dispatch | thesis、meaning、导航及部分ledger重述当前公式/结构；可以短写，不因跨文件相同就删除定义。 |
| F02 正式关系 | 29/63；两节点 | 偶数padding、分层inverse、BF16舍入、32-token分组、谓词和正式引用均为核心数学或量化账本，不是文字漏洞。 |
| F03 外部别名/副作用 | 9/10；root | 返回顺序、`O is output`、`S_final is output_state`和仅这两输入可变靠文字；shape/ref属性、内部resource.view不能表达外部对象身份。删除后“返回新同值tensor而原buffer不变”成为最小反例。 |
| F04 输入与计时域 | 6/7；root | 分布/normalization/前缀初态可用正式关系；contiguous CUDA与计时边界仍是开放任务环境语义。尚无专用benchmark/layout字段。 |
| F05 已知来源限制 | 5/6；两节点 | blocked FP32 oracle、fast近似无误差界、root旧descriptor在途读取顺序、工具链记录不全必须保留；不能添同步假定源码已安全。 |
| F06 P与工作编号 | 2/2；两节点 | `worker_count: P`可改成已有表达式`min(worktile_count,num_sm)`，`step: worker_count`；runtime查询仍需绑定。linear id由worktile_order推导，148-SM只是来源实例。 |
| F07 已有边界/默认 | 11/15；两节点 | index_cast提交、零初态、Beta归还、跨t游标及不排空已有flow/pipe/reuse/DOC5承载。single无Cu32消费者不意味着转换不存在。 |
| F08 matmul重复 | 4/8；两节点 | 零累加/累加目标由mm32/mma32、累积完成由mode决定；保留短产物绑定和operand-read完成契约。 |
| F09 捕获与别名寿命 | 10/13；两节点 | 资源同址和容量已结构化；KK/QK/U捕获进register、提前归还、row/column Beta不同寿命仍靠动作契约。边不定义“read复制到哪里”。 |
| F10 合成发布与单读复用 | 7/7；两节点 | Gate/Scalar/StateOperand/UpdateOperand/Pair成员不是由名字产生；一次S读取同时快照并原地衰减也不由view推出。当前属开放动作语义，结构化读写/发布成员可作为候选。 |
| F11 通知流身份 | 6/6；两节点 | 收窄为`open_semantics / medium`：保留root四流、single共用ready流及省略acquire的源码事实。未找到只改变barrier/API就必然新增非传递等待的反例，也无一阶成本证据；撤回“缺专用通知流核心字段”结论。 |
| F12 Inverse成员分工 | 1/2；两节点 | 8/32/64层每chunk两warp、16层四warp及层间rendezvous；静态member不能逐层变化。可拆固定role与sync，但完整表示成本较高，未做等价短稿。 |
| F13 raw标量内部编排 | 3/3；single | 四head视图、首两次load与c+2条件预取可写入已有math/repeat/事件；当前prepare把内部异步动作压成文字。事务barrier/warp fence仍需原语契约。 |
| F14 rescale已有替代 | 1/2；两节点 | 用既有`compute:QSscaled(n)`替代自定义`rescale:QS(n)`，同步改前驱引用；math与Arena别名已给全部语义。 |
| F15 原语绑定 | 13/16；两节点 | tcgen05、warp mma.sync、TMA descriptor、bulk-store完成及raw fence属于必要开放实现绑定；既有require.item负责定位，不是闭合的原语语义表。 |
| F16 硬件开放语义 | 9/9；root | warp=32、CTA组成、register/TMEM机制及实现栈身份不能由键名推导；wiki为pending。cluster是设备空间说明，本例无cluster执行，不据此删除共享硬件事实。 |
| F17 跨调用workspace | 3/3；root | 源码按编译配置/device缓存可变TensorMap workspace；外部多stream调用是否被约束未核实。共享物理存储不因每次调用获得新事件身份而自动安全；未宣称已证race。single用embedded descriptor，不依赖同一mutable workspace。 |

已有结构的两个直接候选示例（建议，未修改原YAML）：

```yaml
# F06：保留runtime查询定义与当前worktile_order
mapping:
  strategy: num_sm is the runtime device SM count.
  worker: {unit: CTA, unit_count: 1}
  worker_count: min(worktile_count,num_sm)
schedule:
  worker_assignment:
    preassigned: {start: worker_id, step: worker_count}
    refill: none
    exit: assigned_done
```

F14把pipe中的`rescale:QS(n)`和所有对应事件引用改为`compute:QSscaled(n)`；保留`compute:QSscaled(n)@issue`等待`MMA1/matmul:QS(n)@done`，MMA1的Y等待新的compute完成。`QSscaled=(QS⊙Gamma)*fp32(scale)`及Arena同址view保持不变，使用已定义通用`compute`，不是仅替换为另一个未定义动作名。

F13可先定义`RawScalar(c)[field,i,r]`为head组`4*floor(h/4)+r`的原始alpha/beta视图，再以`repeat: {j: [0,2]}`发射首两个异步`load:RawScalar(n[t,j])`。后续捕获完成后用`repeat: {r: [min(c+2,Np),min(c+3,Np)]}`发射有效的下一次预取，再算本轮G/Gamma。该片段显示现有语法能承载编排；捕获的目的值、TMA域外零填充转为逻辑Alpha=1及warp可见性仍须保留定义，不能把新名称当成完整解法。

F11的收窄依据是具体依赖链，而非“规范允许文字”：`StateOperand(n)`已等上一`S.done`，其累积完成涵盖旧KS/QS的Sbar读取；当前`U.done`保护R被Ubar覆盖，上一S完成也覆盖R/Udec的旧读者。single的`prepare(c+5)`在`store:O(c).done`之后，后者传递覆盖旧scalar读者。因此，一个在这些既有退役点归还的full/empty实现，其acquire可只等待已满足条件。物理barrier数量及附加指令可能不同，但不能单凭差异就证明新的执行依赖或主要性能变化。single合并Ubar/Udec发布的事实已有`UpdateOperand`动作与data边，不能再作为通知存储身份缺口的证据。保留源码协议记录，但不提出新增通知流字段；详见 [F11修订依据](../work/template_gdn_prefill/F11_revision.md)。

## 边界和读者退役

`L_b>0`且`Np=2*ceil(L_b/(2*C))>=2`，无空序列；但ScalarOutput的中间循环在Np=2/4时为空，首部`min(4,Np)`和末部stores仍执行。synthetic chunk保留动作和完成事件，只抑制域外数据写出。Alpha=1、Beta=0、Q/K/V=0在数学层定义；最终state是`S(Np-1)`，不能简单裁成最后一个有效token动作。zero初态是显式输入的零值，不是省略状态加载或使用源码的`None`分支。

root的`n=flatten[t,c]`让Q/K/V等环跨worktile连续；同一工作内state依赖用`n[t,c+1]`，不会错误进位到下一份工作。每t先load initial_state，末尾store final state等待本t最后一次S完成。DOC5默认跨worktile不排空，最终worker安全回收与每工作drain不同。single在来源148-SM设备上提交64 CTA、一CTA一worktile；不能把这个运行实例偷偷提升为任意设备资源下的常量。

K/Q复用同时考虑两个issuer：MMA0成对QK的累积完成，分别再加MMA1的S或QS完成。GateRing覆盖等待Inverse与Recurrence两个读者；BetaRing等Tinv；单槽O等bulk-store完成。Arena中KK/QK、KS/U、QS/QSscaled/Y、R/Ubar的同址关系均读取并对照early capture和复用链。single五槽Scalar的旧读者退役由`store:O(c)`完成传递到`prepare(c+5)`；不能只因一个读者读完就归还全部Scalar。raw双槽在读取到register、warp rendezvous/fence后才重发c+2。

selected源码没有自有graph capture/replay或fallback计算分派。外层纯capture不产生GPU完成事件，实际回放才生成调用实例；这采用DOC0共享规则，无需为录制另造空plan。缓存workspace跨调用的物理复用另见F17，未将其解释为已有跨调用安全证明。

## 保守短写与限制

机器明细有 **19条精确quote→replacement**，覆盖18个文本组；每条quote在归一化字段中唯一，适用occurrence明确且区间不重叠。候选节省为英文YAML声明 **1,516字符**（root 868，single 648），中文dispatch **100字符**，合计 **1,616**；按精确组去重的节省为1,186。1516约占YAML开放文本8.1%，是保守候选量，未生成/批准完整短稿，不能称等价压缩已验证。

| 原文片段 | 候选 | 保留依据 |
| --- | --- | --- |
| `Residual after two separate BF16 roundings.` | `Residual.` | R公式仍显式双bf16；两节点共68字符。 |
| `Issue the named upstream product; KS/QS/U start from zero, Y accumulates into QSscaled, S into Sdec. Completion covers this issuer's prior products and operand reads.` | `Compute the named upstream matrix product; done covers operand reads.` | mm32/mma32与execution_mode；两节点共194字符。 |
| `All roles follow the same preassigned sequence; ring cursors continue across worktiles while S reloads each worktile's initial_state. No per-worktile global drain.` | 删除该可选strategy | preassigned、flatten、每t load、DOC5跨工作默认；163字符。 |
| `Raw stages use c%2; write G/Gamma/Beta into processed stage c%5. ` | 删除该局部句 | 两个resource的slot_index已明确；65字符，保留前后raw读写顺序与ready-only说明。 |

长度457/472字符的两节点Recurrence.publish均为MIXED；336字符的single raw prepare也为MIXED。它们含独占单读、合成对象或内部次序，不能按长度截断；通知身份本身的独立核心重要性不再视为已证。一个自定义动词即使多数内容重复，也需保留合法短契约或完成全部动作/引用迁移。

仍未独立核验fast近似误差界、TensorMapManager旧请求顺序、安装版PyTorch转换kernel的512元素容量、全套低层inverse汇编、并发调用的stream安全及pending硬件资料。SOURCE_LIMIT仅1个整字段不表示只有1处限制，其余位于MIXED。所有结论均为离线结构/文字审计；原schema、规范、Runner和GPU运行未改动。
