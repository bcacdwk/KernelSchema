# 当前 check_yaml 的文本膨胀缺口

**这个 demo 不是全局最长；“满额”只描述固定结构中有预算的字段。** 本轮观察复核快照为 475 行、41,284 个 Unicode 字符、41,308 UTF-8 字节，159/160 个有字符预算的字段到顶；最长物理行仍为第 27 行的 607 字符。388 个字符串值中，160 个有字符限额，12 个仅限单行，216 个没有文本预算，具名键另计。当前 `Input.k.range[0]` 为一个中文逗号 `，`，因此 README/CSV 的 160/160 不适用于本快照；CSV 的字符数仅这一项与实读结果不同。本轮读取内容的 SHA-256 前缀为 `67799c816bd0`；统计不代表后续版本。

报告从结构规则、字符串联合类型、列表元素、表达式、引用及具名键逐项核对 [BUDGETS][budgets]，并参考 DOC0–6、FIELD_MAP 和 PLAN。“校验放行”不表示内容符合字段原意。第 6 节保留原审计起点（475 行、41,433 字符、160/160 到顶）的探针证据；第 7 节记录当前具体观察。本轮只更新本报告，未写入工具、规范、demo 或统计附件。

为缩短路径：`A = CONTRACT.Algorithm_requirement`，`H = CONTRACT.Hardware_specification`，`M = DESIGN.Math_reshape`。
`W = DESIGN.Work_schedule`，`WS = W.Step.<STEP>`；`P = DESIGN.Tile_pipeline`，`PS = P.Step.<STEP>`。
`<name>` 表示具名条目，`[]` 表示列表元素；下列字段均使用当前拼写。

## 1. 值得优先补限额的位置

| 优先级 | 具体位置 | 当前机器检查 | 自然扩写风险与判断 |
| --- | --- | --- | --- |
| 高 | `H.Stack.Language / Template / Toolchain` | 非空已知字符串；每项仅限 1 逻辑行，无字符上限；Template 还检查对应 Wiki 项同现 | 环境身份很容易写成版本、组件、编译过程及兼容说明。**值得补字符限额**。[硬件结构][hardware] |
| 高 | `M.Rewrite[].Symbol.<name>.retention` | 值须以 `carry_forward` 或 `keep_value` 开头并有非空尾段；出现时还要求 shape/dtype。不解析实例范围，无字符限额 | 分号后容易继续解释寿命、消费者、初始化与收尾。**值得补限额**；不能把前缀正则当成完整语法约束。[实现][retention] |
| 高 | `WS.Schedule.Worker_assignment.exit` 的字符串及列表元素；`WS.Decomposition.Target.Tile_reduction.input / output` 的字符串及列表元素、`method`；`W.Flow.Handoff.<name>.initialize` 的字符串分支 | `reference()` 只查非空并排除 `unknown:`、`<` 开头；引用列表另查非空和重复。这些位置没有进一步解析引用目标 | 很容易把退出条件、合并过程或初始化理由直接写成长段。**值得逐字符串补限额**。`exit` 并不是只接受 `assigned_done` 的封闭枚举。[值检查][reference]、[字段接线][work-fields] |
| 高 | 全部直接填写的 `dtype`（完整位置见下一节）；`PS.Storage.<name>.location` | dtype 排除部分占位及非法 None 混写，并检查可确定的上游冲突；不是类型枚举。location 仅要求非空文本，没有硬件名称查表 | `fp32 (解释…)`、`SMEM (解释…)` 可带很长的格式、精度或生命周期说明。**值得补限额**；类型名和真实硬件标识仍应保持完整。[dtype][attribute]、[location][storage-fields] |
| 高 | `A.Constant.<name>.value`、`M.Rewrite[].Symbol.<name>.value` 的字符串，以及下一节列出的整数表达式位置 | 固定值字符串仅做已知文本检查；Symbol.value 还递归接受数组中的字符串。整数表达式只对能识别的整数字面量检查数值下界 | “看起来是数字字段”仍可能接纳 `64 (解释…)`、长公式或整段推导。**值得为字符串分支补适度限额**，保留合法数学记法。[整数][integer]、[标量/数组][literal] |
| 中 | `M.Rewrite[].Symbol.<name>.binding`；Math 的 `shape / dtype`、Work_schedule 数据块 `dtype` 中允许的 `ref:` / `bind:` 分支 | binding 检查前缀；属性引用检查 `.shape` / `.dtype` 后缀。可定位的声明路径会查目标、属性及循环，但公式/strategy 中的语义目标可以保留为未解析 | 提供者后容易补括号说明；不能将这些分支全部归为严格短引用。**可补限额**，但保留完整真实目标。[目标解析][target] |
| 高 | `META_DATA.source` 不含 `@` 的材料来源写法 | 非空已知文本，不能只写 pending；没有字符或行数上限 | 材料标题容易扩展成来源综述、推导过程或沿革，甚至多个段落。**值得补来源摘要的限额**。[实现][meta-source] |
| 较低 | `A.Golden_reference.source`、`H.Wiki` 全部叶子、`REALIZATION.Feedback[].evidence`，以及含 `@` 的 `META_DATA.source` | 仓库、修订、路径形式；按字段要求固定版本/入口。路径段禁止换行及通配符，但允许空格、括号等；Wiki 另限单行。均无字符上限 | 比自由正文更受约束，但 `path (用途说明…)` 仍可能被整体当作路径接受。**防止附加说明有价值，真实长路径不能截断或伪造**；不宜与 Stack 正文同等收紧。[仓库引用][repository] |

机器目前不能普遍区分“有效类型/数学内容/真实外部名称”与其后附加的解释。Storage 的已知 dtype 字节表只用于容量核算，不是合法 dtype 的封闭表。这里的建议仅是现有字段的篇幅约束，不要求 Python 表达式或新增 DSL。

## 2. 容易漏看的字符串分支，完整落点

下列描述记录均有 `dtype`；其 `shape` 列表中，每个 `[]` 内的轴大小也都允许无字符限额的整数表达式：

- `A.Input.<name>`、`A.Output.<name>`、`A.Compute[].Symbol.<name>`。
- `M.Rewrite[].Symbol.<name>`。
- `WS.Decomposition.Worktile.Invariant.<name>`、`WS.Decomposition.Worktile.Iteration.<name>`、`WS.Decomposition.Target.Single_tile.<name>`。
- `PS.Storage.<name>`。

| 其余位置 | 哪一部分仍可写长 | 已有约束 |
| --- | --- | --- |
| `A.Constant.<name>.value`；`M.Rewrite[].Symbol.<name>.value` | 字符串本身；后者数组中任意字符串叶子 | 非空、已知值；Constant 对可识别表达式做运行时依赖和循环检查，不理解的记法不因此报错 |
| `WS.Decomposition.Partition.<name>.size / tile / range[]`；`WS.Decomposition.Worktile.Invariant.<name>.offset[]` | 整数表达式字符串 | range 恰好两个端点，offset 长度匹配 shape；可确定数值关系仍检查 |
| `WS.Mapping.Worker / Placement` 的 `unit_count`；`WS.Mapping.worker_count / member[].<axis>[]` | 计数及成员区域端点中的表达式字符串 | 成员数、两端点和可确定区间关系仍检查 |
| `WS.Schedule.Worker_assignment.Preassigned.start / stride / count`；`WS.Schedule.Worktile_order[].group` | 数量、起点、步长表达式 | start+stride 与 start+count 互斥，已知范围仍检查 |
| `PS.Index.<name>.definition[]`；`PS.Storage.<name>.depth / total_size`；其 `View.<value>.slot_index / region.<axis>[]` | 索引域端点、槽号、容量和区域端点表达式 | 列表域恰好两端；区域轴存在性、已知槽/容量边界和可识别索引作用域仍检查 |
| `PS.Role.<ROLE>.Info.member`；`Pipe` 动作/控制键；`Waiting`、`PS.Event` 的事件字符串；`W.Flow.Launch[].start_after`、`Handoff.<name>.ready / wait_at` | 成员选择的端点、动作/事件实参、`if` 条件、`for` 的范围与切片表达式 | 外壳、括号、动词、声明与参数个数受约束；表达式内部仍开放，不应将整个事件语法当成自由正文 |

[integer()][integer] 对非整数字面量的字符串直接保留；[表达式检查][expression]只核对能识别的索引/域使用。[Pipe 解析器][syntax]也明确不验证通用数学语法。因此这些字段可能多写，但其合法复杂度与普通说明段落不同，不能一律压成极短标签。

`Math.Symbol.shape / dtype` 中的 `unknown:<reason>` 已限 **150 字符 / 1 行**；该预算带前缀条件，**不会**给普通 dtype、shape 轴大小或 `ref:` / `bind:` 引用自动加同样的上限。

## 3. 名称长度：与正文分开看

- **较值得关注：`H.Execution`、`H.Feature` 的具名 key。** [实现][hardware]使用 `key=text`，只查非空，允许空格、括号和整句。值虽已限 100 字符，键没有字数预算；作者可能将机制说明写进名称。
- `WS.Mapping.Worker.unit / Placement.unit` 和 `PS.Role.<ROLE>.Info.member` 中的硬件名会精确查 `Execution`，不能单独附加解释；但登记表的长键会原样成为可引用名称。这只能证明名称一致，不能证明它是实际硬件标识。[查表][unit-registration]
- 其余自取 key/名称，包括 Constant、输入输出、Symbol、Partition、数据块、Handoff、Index、Storage、View/Event 基名与形参、noun、轴、step/role、节点 ID，均没有通用长度上限，但有大小写/标识符规则；部分还有唯一性、存在性和引用一致性检查。普通括号说明不能直接加到这些标识里，风险主要是过长描述性名称。**具名 key 没有预算，与 `META_DATA.name`、`Compute[].name`、`Rewrite[].name` 已限 50 字符是两回事**；`meaning` 的 50 字符也不限制其父 key。[标识符检查][identifier]只检查字符形式。
- 当前 Constant 的 `minimum_finite_input_value_allowed_…`、`maximum_finite_input_value_allowed_…`、`default_exact_attention_score_scale_…`、`default_exact_zero_bias_value_…` 四个键均为 **150 字符**，`head_dim` 为 8；完整原名见 [root.yaml 第 18 行起][demo-constants]。Math_reshape 的最长 Symbol 为 `partial_attention_numerator_after_online_exponential_rescaling_…_without_recomputing_any_saved_contribution`，**279 字符**；其次为 `iteration_count` 15、`packed_weights` 14、`running_value`/`partial_value` 13，见 [Symbol 原文][demo-symbol]。这批极长名是合成 demo 为填满 range/default/item 而设置的压力内容，[README][demo-readme]已说明；同时也确证真实写作缺少自取名称长度约束，值得纳入下一轮。不能把压力构造都归因于现有正文预算漏挂，也不能直接以 `name` 属性的 50 字符预算冒充 key 已有限制。
- **Refill 的机制名属于后一类：必须是小写标识符；规则正文已经限 150 字符。** 不能把机制名与规则值混成一个未封顶的说明字段。

## 4. 每条受限，不等于条目总量受限

[named()/sequence()][containers]一般只检查形状、非空和需要时的唯一性，没有业务条数或合计字符预算。正常可增长的位置包括：

- **最容易扩成许多段：** `A.Compute[]`、`M.Rewrite[]`、`REALIZATION.Require / Prefer / Forbid / Feedback[]`、`H.Execution / Feature`、`WS.Schedule.Worker_assignment.Refill`。单个 math 最多 1,000 字符，但还可以增加数学步骤；各 Rewrite 还可能带自己的 Symbol、Arithmetic、Ledger。
- **登记表可以继续增加：** 两区 Step、Role、Using_verb 下的 noun，以及 Constant、输入输出、Symbol、Partition、各数据块、Handoff、Index、Storage、View、Event。相应 meaning/describe 各自受限，不限制全表。
- **结构列表也可增长：** shape 轴、Symbol.value 数组、Mapping.member、Worktile_order、Launch/交接引用列表、Event/Waiting 及前驱集合、Pipe 动作和控制体。它们仍须满足维度、计数、唯一性与引用关系；两端点区间不是任意长列表。

`Feature` 的“通常不超过十条”是[文档指导][feature-count]，当前机器没有十条硬限制。任务最多十份完整 YAML 则确实检查。
另有每文档 100,000 个展开节点、解析深度及有效设计节点容量护栏；它们按结构规模计数，不是字符预算，也不能阻止一个属性写成长文。[解析容量][capacity]、[有效设计计数][resolution-count]
这些事实解释了为什么没有内容意义上的全文“写满”。复杂算子的必要条目不因此算滥用；不建议为了一个机械最长值而删掉必须定义的对象或步骤。

## 5. 不需要优先处理的项与预算路径核对

本次主要是未配置预算的开放位置，未发现既定预算因普通可选字段或复用形态而失效。固定记录中的额外字段、YAML 注释也会被拒绝。

- 真正封闭的是 `direction`、launch `order`、`execution_mode`、`Refill: none`、Index 的 `worktile` 分支、八个 verb、事件 `issue/done` 等；这些值不能扩写成长说明。
- Contract 的局部属性引用、明确的已登记字段路径、step/producer/consumer、Handoff.storage、Worktile_order.by、Tile_reduction.dimension、flatten 域名、Design.from 等还有相应的格式和目标检查。除前文单列的表达式实参、未解析语义目标及名称长度外，不应都算成自由文本缺口。
- 已设预算的可选/嵌套路径没有发现常规漏套：Using_verb.describe、各类 meaning、Refill 的映射值、range/default 的字符串和 range 列表端点、REALIZATION.item 的单值和列表元素、Feedback.choice、unknown 属性都被相应规则选中。[逐路径选择][selection]
- 正常 CLI 对目录内每个 raw 文档检查预算，再解析 from；跨 step 名称不会改变模式匹配，alias 在各实际字段上下文检查。没有发现普通整区复用或别名把已封顶正文变成不检查的情况。[入口][raw-budget]
- 当前规范和结构中没有独立的 `where` 字段；写在 math 内的 where 说明随 math 计数，写进上述开放表达式则随该表达式处理。`--skip-hardware` 是明确豁免；`--show-effective` 明示未验证，均不计作本次缺口。

## 6. 少量完整检查复现

使用审计起点 demo 的临时副本，各组只改下列值，保留三 step、硬件和 dispatch。每组均先执行默认 `check_yaml.py`，再执行 `--read-only`；**六组的两次退出码全部为 0**。没有使用 `--skip-hardware`、standalone 或未验证接口。缺口结论不依赖 demo 的具体总字符数。

| 组 | 写长的字段与实测字符数 | 默认检查 / 只读复查 |
| --- | --- | --- |
| Stack | `H.Stack.Toolchain`：501 字符 / 1 行 | 0 / 0；排版后全文 41,905 字符，已超过原 demo |
| Retention | `M.Rewrite[1].Symbol.running_max.retention`：471 / 1 行 | 0 / 0 |
| Exit | `WS=ATTENTION` 的 `Schedule.Worker_assignment.exit[1]`：476 / 1 行 | 0 / 0 |
| 属性 | `A.Input.scale.dtype`：371；`A.Input.q.shape[0].query`：311；`PS=ATTENTION` 的 `Storage.query_rows.location`：315 | 0 / 0；均为属性值后接普通括号解释 |
| Binding | `M.Rewrite[0].Symbol.bk.binding`：386 / 1 行 | 0 / 0；`bind:DESIGN.Work_schedule::bk (解释…)` |
| 来源 | `META_DATA.source`：524 / 2 行；`REALIZATION.Feedback[0].evidence`：419 / 1 行 | 0 / 0；分别是来源综述、仓库路径后接括号解释 |

对照：Refill.atomic_counter 的 190 字符说明、`Symbol.scores.dtype` 的 198 字符 `unknown:` 值均返回 **1**，准确报告 **150 字符 / 1 行** 上限。所有临时任务已删除；在该探针批次开始与结束时，原 demo 四个文件的 SHA-256 相同。

以下可重现第一组；只写临时目录，命令在仓库根目录运行：

```bash
/opt/venv/python3.13.12/bin/python3 -B - <<'PY'
import pathlib, subprocess, sys, tempfile
sys.path.insert(0, 'schema_template')
from tools import yaml_source
from tools.diagnostics import encode_value
demo = pathlib.Path('schema_discussion/template_tooling/text_limit_demo')
data = yaml_source.load(demo / 'root.yaml', 'root').value
data['CONTRACT']['Hardware_specification']['Stack']['Toolchain'] = 'NVIDIA CUDA Toolkit 13.0 nvcc; the compiler lowers the selected CUDA C++ source to the Blackwell target and supplies the assembler used for the device binary; this entry also records the language front end, template instantiation behavior, host integration, device linking, library search configuration and diagnostic conventions so that another implementer can reproduce the same environment without inferring those choices from the kernel source or from a compiler installed elsewhere on the machine'
with tempfile.TemporaryDirectory(prefix='schema_text_gap_') as folder:
    root = pathlib.Path(folder) / 'root.yaml'
    root.write_text(encode_value(data) + '\n', encoding='utf-8')
    root.with_name('dispatch.md').write_bytes((demo / 'dispatch.md').read_bytes())
    for flags in ([], ['--read-only']):
        subprocess.run([sys.executable, '-B', 'schema_template/check_yaml.py', *flags, str(root)], check=True)
PY
```

## 7. 具体排版观察与后续处理

以下均核对当前原文，使用 `/opt/venv/python3.13.12/bin/python3 -B`、同一 `yaml_source.loads` 和无损 formatter 做内存验证；未运行全套测试或写回 YAML。

### range、长名称与 item

`CONTRACT.Algorithm_requirement.Input` 的字符串 range/default **已限 150 字符 / 1 行**，二端点列表按每个字符串端点检查，没有列表合计预算。[当前 Input][demo-input] 中 q/v/bias 的端点均为 150/150；k 为 1/150，首端点实际原文是 `range: [，, maximum_finite_input_value_allowed_…]`；scale 的表达式 range 为 150，scale/bias 的 default 引用各为 150。长 Constant 引用叠加 shape 等同行属性造成宽行，属于名称与组合宽度问题，不是 range 漏挂上限。

`k.range[0]` 的中文逗号会被 loader 读成字符串；单独调用现有 [input_range 规则][input-range]无诊断，当前文档文本策略检查也为 0 条诊断。**仅标点串被当作表达式接受，是字符/形式补漏的具体证据**，不表示它是有意义的下界；后续应在保留合法数学符号的前提下处理，不能用全局禁 Unicode 代替判断。本轮不恢复或猜测该端点。

四个 `REALIZATION` 区段各自的 [item][demo-item] 都是 **300 字符**：`DESIGN.Math_reshape::` 的 21 字符加上述 279 字符 Symbol。单值与列表元素的 300/2 预算都已经命中；名字预算与引用可读性需要补，不能认定 item 漏限额或为凑两行折断引用。

### 单引号：已确认必要的包裹与正文内部字符

| 实际位置与简短原文 | 同一 loader 的内存试验 | 结论 |
| --- | --- | --- |
| [Arithmetic.approx 第 134 行][demo-arithmetic]：`approx: 'Apply …; unknown: the synthetic layout fixture …'` | 处于 block mapping；去掉外层单引号后，`unknown: ` 的冒号加空格触发 `mapping values are not allowed here` | 包裹引号必要，非清理漏项；正文中没有额外单引号或数学撇号 |
| Using_verb 下 **10 个 describe 值**：第 364、365、368、401、403、404、406、408、440、441 行，例如 [第 364 行][demo-verb]的 `describe: 'Load key_block(t,i) from contract k …'` | 全在 flow `{execution_mode: …, describe: …}` 记录中；逐个仅去包裹、保留解码正文，10/10 均改变映射结构。`key_block(t,i)` 的括号不保护逗号；小样例被拆为 `describe: Load key_block(t` 和额外的 `i): null`。同样正文改放 block 的 `describe:` 行时，10/10 可无引号且值不变 | 引号落在说明值，不在 noun 名称；既定行内排版下全部必要。不能为去引号删逗号或改写已有 150 字符正文 |
| [release.input_slot 第 373 行][demo-apostrophe]的 `Return this member's use …`；[signal.part_done 第 408 行][demo-escaped-apostrophe]的 `'Notify … member''s … released, allowing …'` | 前者原本裸写；后者单引号标量内的 `''` 解码为一个 `'`，外层因逗号而必要 | 两处正文的 `'` 都是英文所有格，非待删除的 YAML 包裹；当前规范允许内部字符，不是新增缺陷 |

补做的同 resolver 小样例中，去掉 `'true'`/`'3.0'` 的包裹分别改变为 bool/Decimal，去掉 `'{done signal:ready}'` 的包裹改变为映射；`'operand(i)'` 的单独括号则可裸写。可见应按实际 block/flow、分隔符和类型判断。本快照内存调用 `format_document` 返回与源文本完全一致的结果；上述目标未发现能无损去除却遗留的包裹引号。

### 固定小记录与 assignment 的紧凑边界

当前 formatter 的 [RECORDS][layout-records] 未纳入 Mapping.Worker/Placement、Worktile_order 条目和 Preassigned；这些位置只列入字段排序，[member][layout-member]则已经有明确的行内规则。下一轮补有限路径规则即可，不改字段或内容。

- **Mapping：** 三个 step 的 Worker/Placement 当前各占三行，worker_count 已单行；ATTENTION/MERGE 的 member 列表也已各自单行。建议每个子字段各一行，保留 Mapping 层级及独立 strategy。以 [ATTENTION 第 214 行起][demo-mapping]的子字段为例，当前：

  ```yaml
  Worker:
    unit: CTA
    unit_count: 2
  worker_count: 1
  Placement:
    unit: cluster
    unit_count: 1
  member: [{query: [0, 64]}, {query: [64, 128]}]
  ```

  拟议（连同未改的 strategy 在内，内存回读类型、值和完整结构相同，Mapping 局部结构检查通过）：

  ```yaml
  Worker: {unit: CTA, unit_count: 2}
  worker_count: 1
  Placement: {unit: cluster, unit_count: 1}
  member: [{query: [0, 64]}, {query: [64, 128]}]
  ```

  这不是把四个子字段或带 150 字符 strategy 的整个 Mapping 压到同一行；无需为此新增容器。[DOC4][mapping-doc]已有 Worker/Placement 小记录范例，采用这一边界无需用户另行裁定。

- **Worktile_order：** [当前四个条目][demo-order]都展开为 2–3 行，可各一行。ATTENTION 的 `by: part`、`group: 2`、`direction: ascending` → `- {by: part, group: 2, direction: ascending}`，下一项 → `- {by: part, direction: ascending}`；保留列表及条目顺序。两项完整内存回读一致。字段仍沿用现有 `by / group / direction` 次序；`by` 的合法列表分支也应按已知属性形状保留。

- **Worker_assignment：** [ATTENTION 第 230 行起][demo-assignment]当前 Preassigned 占三行、Refill 占两行（一个机制、150 字符规则）、exit 占三行（两个引用）。可以分别变为以下三个字段行；完整内存回读类型/值/结构相同，assignment 局部结构检查通过：

  ```yaml
  Preassigned: {start: 0, count: 1}
  Refill: {atomic_counter: 'The LOAD role atomically increments a counter initialized to one; a result below parts assigns the next partition, otherwise it saves the final ticket'}
  exit: [assigned_done, terminal_ticket]
  ```

  PREPARE/MERGE 的 `Refill: none` 和标量 `exit: assigned_done` 已经单行；Preassigned 的 count/stride 两种互斥记录都可行内，其整数表达式尚无字符上限。Refill 的规则值已经限 150/1，机制 key 无名称预算；上例因规则含逗号，合行后须补外层引号。含原缩进的 Refill 行由当前规则行的 180 字符变为 190 字符，不能为了更窄而改掉已限额正文。建议一个机制时整体行内，多个机制时保留 Refill 映射展开、各机制规则单独一行。exit 当前合行仅 50 字符，但其字符串/列表元素无字符预算、条数也无上限；多项或长引用时整字段合行可能很宽，不能承诺普遍单行。建议当前短双项可合行，多项仍每项一行；普适短列表边界与 exit 预算在下一轮一起定，不把整个 Worker_assignment 压成一行。

**后续范围：** 已确认待补的是自取名称长度、仅标点表达式的字符/形式约束，以及上述固定路径的紧凑排版；range/item/Refill 既有预算不重设、不改正文。必要包裹引号和合法所有格不计 gap。本轮没有必须立即请用户裁定的布局歧义；下一轮计划中需明确自取名称的具体上限及真实外部标识例外、exit 多项列表的合行边界，统一实施后再更新 demo/统计。

[budgets]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/text_policy.py:34
[hardware]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/hardware.py:7
[retention]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/structure.py:83
[reference]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:211
[work-fields]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/structure.py:123
[attribute]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:238
[storage-fields]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/structure.py:220
[integer]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:120
[literal]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:148
[target]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/symbols.py:63
[meta-source]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:202
[repository]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:176
[expression]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/pipeline.py:84
[syntax]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/pipeline_syntax.py:17
[unit-registration]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/declarations.py:87
[containers]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:74
[feature-count]: /home/shaohanyong.shy/KernelSchema/schema_docs/2.contract.hardware_specification.md:169
[capacity]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/yaml_source.py:293
[resolution-count]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/design_resolution.py:84
[selection]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/text_policy.py:101
[raw-budget]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/schema.py:19
[identifier]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:30
[demo-constants]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:18
[demo-symbol]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:153
[demo-readme]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/README.md:18
[demo-input]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:23
[demo-item]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:449
[demo-arithmetic]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:134
[demo-verb]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:364
[demo-apostrophe]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:373
[demo-escaped-apostrophe]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:408
[layout-records]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/yaml_format.py:25
[layout-member]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/yaml_format.py:168
[demo-mapping]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:214
[mapping-doc]: /home/shaohanyong.shy/KernelSchema/schema_docs/4.design.work_schedule.md:181
[demo-order]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:224
[demo-assignment]: /home/shaohanyong.shy/KernelSchema/schema_discussion/template_tooling/text_limit_demo/root.yaml:230
[input-range]: /home/shaohanyong.shy/KernelSchema/schema_template/tools/rules/values.py:278
