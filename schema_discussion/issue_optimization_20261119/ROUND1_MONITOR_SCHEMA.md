# 第一轮 max 审计：monitor、Schema 表达与诊断

2026-11-19，北京时间。审计代理：`/root/issue_r1_monitor_schema_max`，`gpt-6-astra / max`。本报告覆盖分配的全部 10 个稳定 ID，不以归档状态排除问题。

结论：保留现有局部修复，补齐三项待决说明及其必要实例修订；监控还须处理一个本轮真实复现的旧观察覆盖新观察问题，并补上等待反馈。normalizer、名称冲突诊断及 Version 权限提示没有发现必须重写的根因缺口。MLA、GEMM 的已知表达缺口须在本轮限域修正，不能仅修改规则或归档 issue 就结案。

## 审计范围与不变量

- 基线为 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。检查了实际 `git status`、tracked diff、相关 untracked issue、讨论证据与六份正式 root；没有 staged 改动。相关 Python、测试和两个 Version prompt 的工作树差异均已逐项阅读。尚未实现的 DOC1/DOC2/DOC5/P1 指引没有被当作已完成。
- 已读仓库 AGENTS、`schema_extract/README.md`、`schema_extract/issues/README.md`、旧 [ISSUE_DECISION_REVIEW](../extract_six_full_20261118/ISSUE_DECISION_REVIEW.md)、各 ID 的正式记录及关联调查。任务所称 `issues/README.md` 在本仓库的实际位置为 `schema_extract/issues/README.md`。
- [RUNS](../extract_six_full_20261118/RUNS.json) 与 [DELIVERY_INDEX](../extract_six_full_20261118/DELIVERY_INDEX.json) 确认：6 项逻辑交付、15 份设计，5 项正常 P6、FMHA 为用户授权小修。FMHA 原自动状态仍是 `invalid_receipt/P6/published:false`；取消的 Sparse successor 不恢复。monitor 已 disabled、generation 8。本轮没有必要操作这些运行。
- [USER_DECISION](../extract_six_full_20261118/recovery_reassessment/USER_DECISION.md) 的工具故障续接边界继续有效；其“其他 issue 等指挥”已经被本轮全面优化授权取代。不得用本轮规则说明改动给旧运行换语义规则，不得伪填旧审批或抹掉待审计标记。
- 本代理只写本报告。验证使用只读现有产物、内存对象、临时 state 和自建短子进程；没有启动模型、GPU、真实 clock，未执行 extract/resume/cancel/线上 monitor，未改 runtime、正式 issue、规则或 wiki。

## 10 项覆盖与实施决定

| 稳定 ID | 原状态 | 当前质量与结论 | 本轮动作 |
| --- | --- | --- | --- |
| `monitor-registration-lock-retry` | resolved | 仅重试取锁，正文不重放；真实 SIGINT 后可续。持续等待静默，缺少长期可见反馈。 | 必做：限于前台 ack/绑定 scan 的等待提示；保留原重试与身份门禁，固化真实中断验证。 |
| `monitor-wait-stale-observation` | resolved | ack 时间与 claim owner 的旧观察已修；scan 的提交等待仍可误标采样时间、覆盖新观察，本轮已复现。 | 必做：采样/完成时间分开，latest 不被较旧观察覆盖；保留活 owner/未知 owner 不抢占。 |
| `monitor-beijing-time-display` | pending | `wakeup.py:400` 仍生成 UTC；人读观察输出仍打印 epoch。 | 必做：统一人读北京时间，原数值时间、身份与调度保持不变。 |
| `normalizer-diff-not-visible` | resolved | 首差异、Unicode 字符位置、局部上下文和省略号均符合问题目标；转换逻辑未变。 | 已足够：保留实现与现有测试，不扩为完整 diff 产品。 |
| `inplace-output-input-name-diagnostic` | resolved | 冲突双方路径明确；条件式提示保留两个角色，不推断 alias、不改名或删对象。 | 已足够：保留实现；不放宽统一命名空间。 |
| `maintenance-version-bump-outside-field-cap` | resolved | 两处 prompt 与现有 `_erase_fields`、`_versions`、P6 scope 相符。 | 已足够：不增加 Version 写入口，不改门禁。精简轮可合并 P6 的重复版本句。 |
| `tuple-abi-input-typing` | declined | 拒绝新增类型机制合理；MLA 的容器 dtype 与元素 shape 混用仍未完全解决。 | 必做实例修订，规范样例合并进下述 non-tensor ID；保留原 decline 所针对的主张。 |
| `inplace-output-input-registration` | declined | 现有规范已要求双角色；六份正式 root 已有对应 Input/Output，MoE 的同一对象关系明确。 | 已足够：不新增规范义务。FMHA/Sparse 的更深语义核对交最终专项组。 |
| `non-tensor-parameter-registration` | pending | DOC1 仍没有解释容器、元素、dtype 对象之间的层次；GEMM 已有写法可自洽。 | 必做：少量现有字段样例，明确原子对象/逻辑元素两种表达；连同 MLA 实例验收。 |
| `location-storage-level-registration` | pending | P1 与 DOC5 的词汇关联缺口仍在；GEMM 的 register 尚无清楚的所属单元摘要。 | 必做：DOC2/DOC5/P1 对齐最小命名口径，补 GEMM 既有 thread 摘要。 |

## 必做一：监控等待、观察与北京时间

### 保留已有正确边界

`Store.control_transaction()`（`store.py:135–149`）只捕获事务进入时的 `SupervisorBusy`，锁外等待 0.05 秒，再次取得锁后才执行一次正文。`Clock.ack`、绑定 `scan` 的 claim/commit/异常清理四处复用它，方向正确。线程、提醒、代际、ack 前提及 claim ID 仍在当前 ledger 下检查；daemon tick/observe、start/stop、pause/cancel 等仍即时 busy。不得将业务拒绝或正文异常放入重试循环。

ack 的时间已在锁内读取；scan 也不再使用等待前的进程表判断新 claim。`_alive(...) is not False` 对不可确认的 owner 保守拒绝抢占，这个三值语义必须保留。

### 已复现的遗漏：等提交锁后，旧观察被标成新观察

位置：`wakeup.py:321–332`、`monitor.py:207–234`。当前先 `inspect_runs()`，等待 commit 锁，取得锁后以新 `now` 调用 `record_scan()`。后者无采样新旧检查，直接覆写 `observations.latest`。

本轮在临时 Store 中运行当前代码，单次 `inspect_runs` 返回 rev1；在它等待 commit 锁时，模拟已有 daemon 观察提交 rev2：

| 事件 | 虚拟 epoch | 状态 |
| --- | ---: | --- |
| 前台实际采样 | 1000 | rev1，行内 `observation.observed_at=1000` |
| 等锁期间另一观察提交 | 1060 | latest 更新为 rev2 |
| 前台获得锁并提交 | 1120 | 返回 `observed_at=1120`，latest 反退为 rev1 |

实测 `inspect_call_count=1`，`completed_at=1120`、`next_due_at=1420`。因此不是重复执行观察，而是提交时间冒充采样时间及投影覆盖顺序有误。这还会把锁等待计入 `unchanged_since`/`possible_stall` 的观察间隔，制造不实的观察结论。没有证据称线上已发生相同覆盖，报告仅声明隔离复现。

目标方案：

1. 在锁外观察完成时保存采样时间；行已有 `observation.observed_at` 时保留该真实时间。`record_scan` 使用采样时刻，`_complete` 的 `completed_at` 与 `next_due_at = completed_at + interval` 继续使用实际完成时间。初次 scan、daemon observe、手工 scan 与绑定 scan 的命名含义保持一致。
2. 给 latest 的每个 run 保留足以比较采样新旧的时间，较旧采样不得覆盖更晚采样；不能只用全局时间，因为 daemon 是分批/限域观察。也不能仅比较 `state_revision`：同一修订号下进程存活与锁状态仍可能变化。
3. 仅新的观察参与 latest/changes 的提交；新增采样元数据不能使每次无变化观察都变成 `changes`。旧扫描的返回值可保留其实际采样及完成时间，但不能宣称它是最新状态。`unchanged_since` 与停滞提示用真实采样时间，不能用取锁等待放大。
4. 不在持锁区重新执行整轮 `inspect_runs`，不为等锁重放观察。claim owner 的即时检查只需读取该 PID 的 boot/start 身份与状态；可让 `_alive` 在未给 process table 时复用现有 `process_identity` + `process_alive`，避免在全局锁里扫描整个 `/proc`。不可退化成只看 PID 或只用 `kill(pid, 0)`。

### 等待反馈与真实中断

本轮用临时 Store、自建 Python 子进程和真实 flock/SIGINT 验证了两条现有路径：

- ack 等锁时中断：子进程退出，ledger 字节未变，ack 仍为空；释放锁后同一提醒 ack 可成功。
- scan 已提交 claim、等 commit 锁时中断：ack 与未完成 claim 保留，检查次数仍为 0；原子进程真实退出后，同 wake 的下一次 scan 回收 claim 并完成，随后重复 scan 返回 `reused=true`。恢复时只观察一次，最终次数为 1。

这证明现有中断核心可用，不需要新增恢复队列或强行清除 claim。应区分“同一次锁重试不重复观察”与“进程退出后重新采样未提交的只读观察”；后者是正确恢复，不应为了字面不重读而缓存不明旧结果。

最小改动是在 `control_transaction` 增加可选、默认无行为的 `on_wait(elapsed)`，仅在取锁失败、锁外调用；前台 Clock/CLI 提供节流后的 stderr 提示，说明等待的 ack/scan 阶段、已等待时长及可中断后续同一 ID。用 monotonic 计等待时长，短竞争不刷屏，持续等待定期可见。具体提示间隔是显示细节，不是超时或终止预算。不要改变其他调用者行为，不增加公共 CLI 参数、重试框架或后台进程。

不要为 SIGINT 增加一个可能再次无限等锁的必做清理。若 CLI 美化 KeyboardInterrupt，不能未经对账输出 `committed:false`：中断可能发生在提交成功、输出之前；应指导用同一 ID/status 核对。

### 北京时间仅位于展示边界

采用一个纯 formatter，固定 `timezone(timedelta(hours=8))`，人读形式如 `2026-11-20 00:00:00 北京时间（UTC+8）`。不修改进程 TZ，不依赖带历史夏令时的地区时区，不把 epoch 加八小时后回存。

必改位置：

- `Clock.message` 的到期时间；`prompt/operations/wakeup.md` 提醒 Supervisor 的后续简报采用同一时区。
- `cli._print_observations` 的 `at`、`last`（它同时服务非 JSON inspect/scan）；绑定 scan 的完成/下次时间也应有清楚的人读展示。
- status/start/ack/stop 当前默认就是 JSON。保留原 key/value/type，可只在输出副本添加关键 `*_display` 字符串，覆盖到期、签收、采样、完成、下次时间；不要把这些派生串写入 supervisor ledger。`--json` 的原机器字段不重解释或替换。

内部 `now()`、epoch、时间间隔、`_item` 的摘要输入、reminder/thread/generation/claim、queue 补送次数和真实触发时刻不变。停用状态不能因为保存着旧 next_due 就被描述为仍在计时。旧报告的 UTC 证据和旧回执保持原字节，不批量“换时区修复”历史。

必要定向验证：现有竞争/stop/代际测试继续通过；增加上述旧观察覆盖复现及同 revision 不同进程观察、限域批次互不回退、无变化不误报 changes；真实子进程覆盖等锁 SIGINT、死 claim 重试和提交后输出前中断的幂等；等待提示出现在 stderr 且不污染 JSON。时间验证覆盖 UTC 日界/年界、None、原 epoch 与 reminder ID 未变，既有 300/1800/120 秒间隔不变。只在临时 state 中测试，禁止用已停用的真实 monitor 验收。

## 必做二：非 Tensor 表达及 MLA 实例

DOC1 已提供开放 dtype，不需要类型注册表、复合字段或扩大 `python_none`。缺的是逻辑层次说明，不能将“checker 接受字符串”当作类型语义已明确。

在 `1.contract.algorithm_requirement.md` 的 shape/类型说明附近补一段原则和少量样例：

- 有逻辑元素轴的数值序列：shape 描述元素位置，dtype 是每个逻辑元素的类型；meaning 说明 Python tuple 等调用者对象及顺序。例：三整数 tuple 用 `shape: [{component: 3}]`、`dtype: python_int`。
- 整体作为一个原子值登记的非数值对象：使用 `shape: []`，dtype 命名整个对象的已定义类型，meaning/必要 Math 解释其值。这里的逻辑标量不等于“Python 对象必须是基本标量”；不能用这种写法隐藏计算实际需要的元素/成员定义。
- `torch.dtype` 对象与其指代的张量格式分开：`shape: []`、`dtype: torch.dtype`，例如 `range: == torch.bfloat16`；不能把这个参数本身写为 BF16 数据元素。
- 含 None 的 tuple 不是整个对象 None；`nullable` 只控制整个对象，`default` 只控制省略后的取值。没有证据时不从默认 tuple、样本或未使用推导固定值域。`python_none` 仍只表示单个 None，shape 必须为 `[]`。

样例是现有字段的解释，不形成新的封闭类型语法。既有 DOC1 的调用者全对象登记义务、唯一命名、数值域和 nullable 检查保持不变。

### 当前实例的明确判定

**GEMM 的非 Tensor 条目无需改。** `root.yaml:19` 的三元素逻辑轴配 `python_int`，并通过 tuple 等值 range 表达 `(1,128,128)`；`:22` 的 `torch.dtype` 是整个标量对象，其值等于 `torch.bfloat16`，输出另登记 BF16。两者没有混用容器类型与元素格式，不应为“统一所有 tuple 的写法”重写。

**MLA 需要修订，不应继续称已完全无歧义。** `root.yaml:36` 的 `python_tuple` 是整个容器名，`shape: [{element: 4}]` 却采用元素轴；它与相邻 `python_int` 元素表达共用 shape 但换了 dtype 层次。旧维护 review/verify 确实批准过这种取舍，原 singleton 错误也确已消除；本轮发现的剩余表达问题不能据此撤销历史批准或假称不存在。

第二轮的最小目标：

1. 将 `Input.sage_attn_sfs` 明确为一个整体 Python tuple 值：`shape: []`、保留 `dtype: python_tuple` 与真实 default；meaning 明说它是未使用的 SageAttention tuple 参数，区分默认成员值与有效域。不要仅靠 default 把所有显式值限制为四个 None，也不添加 `nullable`。当前固定实现和 reference 都不读取它的成员；若需要说明默认 tuple 的四成员，放在该条目含义或既有默认表达中，不再给“整个 tuple 类型”配元素轴。
2. `Input.num_elts_per_sage_attn_blk` 保留 `shape: [{element: 4}]`、`dtype: python_int`、`default: '(0,0,0,0)'` 的元素表达；只在 meaning 明确 Python tuple 的对象含义。不得新增“全部整数只能为 0”的 range。这里与原子 tuple 使用不同表达是有意且可说明的层次选择，不要求虚假的统一。
3. 按本文件内容修订递增 root Version；没有证据需要改 Compute、Design、Output、其他 ABI 或 dispatch 的设计事实。

依据：固定 `problem/input.py:106–129`、`candidate/solution.py:2105–2106` 与其实际下游调用、`problem/reference.py:78` 及 normalized/returned 链；shape_range/metadata 未声明这两个参数的固定值域。旧维护任务本来也明确禁止据生成样本/default 收窄有效域。MoE 的 `tuple(int32, fp32)` 已通过 `Compute.Routing` 明确定义逻辑每槽 pair 与两个 tensor 的对应，不能因为 dtype 中出现 tuple 就机械套用 MLA 的整改。

验收应由独立读者从当前字段唯一还原“一个对象/若干逻辑元素”、默认行为与值域，而不只看 checker 退出码。可以用现有 `validate_algorithm` 验证文档例子的结构，但不新增解析任意 Python 容器的 checker，也不为几个说明句编写镜像实现的测试。

## 必做三：存储名与所属执行单元

`DOC2.Execution` 已能在摘要中说明执行单元的物理资源；DOC5 的 location 又是开放存储名。最小一致规则是：在实际所属执行单元摘要里登记给定材料明确的存储名和所属范围，例如 thread 的 `register`、CTA 的 `smem`/`tmem`；`Storage.location` 沿用该名称。名称不必是一个独立 Execution 键，纯存储不因此成为 Worker/Role 的执行者。

具体修改：

- DOC2 的提取例外段、Execution 说明：最小登记包含材料已明确的资源名称及其所属单元，仍不是容量/设备能力调查。
- DOC5 的 `Storage.location` 一行及必要的相邻说明：资源名可以来自所属执行单元的摘要；不要求为了 register 新造执行层级。
- P1 `第 3 段：硬件占位`：登记材料明确、后续需要引用的存储名称/范围；资源容量、编译后寄存器分配、spill 证明与其他占位不因此成为提取前置。缺真实名称时沿既有定义责任回 P1，不能让 P3/P5 越权补 Contract。

**GEMM 需要一个小实例修订。** 当前 `Hardware_specification.Execution` 列出 smem/tmem，却没有清楚给 `Storage.location: register` 关联所属单元。给既有 `thread` 摘要补“thread-local register values”的真实名称与范围即可，复核 root/wide 中 sb_regs、row_acc 等使用处。源码 `candidate/fp8gg_umma.cu:813–815,832–836` 给出局部累加数组与明确 register prefetch 意图；它不证明编译后无 spilling。不要新增 `Execution.register`，不要把数组槽位小计升级为总寄存器或容量保证，不改已有设计存储账。

只修改 root Contract 时仅递增 root Version；wide 原文件与设计无需改。两个节点的完整目录检查和来源对应一起核对。其他算子已能在摘要看到部分存储归属，不借本 issue 批量重写硬件事实；FMHA/Sparse 的完整一致性由最终专项组按同一规则检查。

规范验收必须保持普通非 `--skip-hardware` 检查的既有义务。此次仅澄清命名，不能删掉普通硬件校验，也不能把提取中的占位或 `--skip-hardware` 写成硬件保证。

## 已足够的修法与必须保留的验收边界

### normalizer 与名称冲突诊断

`yaml_format._shown_change` 仅替换 warning 的展示，不触碰候选转换；长字段给首个解码 Unicode 字符差异的 1-based 位置、前 24/后 72 字符上下文及省略标记，短字段仍完整展示。identifier/reference 与 terminal period 共用实现，保留原 Diagnostic 路径和类别。原 [NORMALIZER_WARNING](../extract_six_full_20261118/NORMALIZER_WARNING.md) 的真实 968 字符案例与字节一致性依据有效；无需借本轮添加全量 diff、日志产品或截断猜测。

`symbols.contract` 复用已有 locations 表后，常量/输入/输出/Symbol 的统一唯一性集合没有缩小；运行时 extent 仍不能撞对象名。Input/Output 同名才给保留双角色、异名和条件性写回提示；shape/dtype 一致不是 alias 证明，不同也不豁免重名。旧 [IO_NAME_DIAGNOSTIC](../extract_six_full_20261118/IO_NAME_DIAGNOSTIC.md) 的组合验证说明了范围，本轮源码阅读未发现须增加字段或改合法性的理由。

### Version 与原地 I/O

`_erase_fields` 允许获准内容修改随带 Version，`_versions` 要求内容变化递增；P6 显式 scope 仍只接受目录及指定非 root 内容字段。当前 maintenance/P6 prompt 已讲清 worker 更新、Runner 校验，不是 Runner 代填。原权限代码 AST 与本轮提示的边界吻合，无需为了提示问题放宽 scope。

保留验收：合法非 root 内容连同 Version 可改；改内容不增版本仍拒；root 和 reviewer 仍只读；未授权 metadata 不得顺带改；显式 P6 `META_DATA.Version` scope 仍拒。P6 inventory 修复由运行组负责，不把它与 Version scope 混为一项。

DOC1 的 Input/Output 义务已经完备。当前六份正式 root 均登记写回输入角色；MoE `output_buffer → output` 的 Compute 明确旧值不读、全量覆写、同对象 list 返回。Sparse 当前 `out_buffer` 已有 nullable/default 且 range 明确有限/±inf/NaN，FMHA 已有 o_buffer/lse_buffer；这项原缺项已经处理。其他五项的 guarded 有限初态不能被错误推广成 Sparse 的有限限制，签名的更宽可选分支也不能自动扩大任务范围。原 decline 应继续针对“需要新增双登记义务”的主张，不否定历史实例修正。

## 第二轮单一文件所有者与执行顺序

运行审计组及根代理已确认以下单一所有者边界；本轮双方均不编辑共享代码。

| 所有者 | 文件/范围 | 接口约定 |
| --- | --- | --- |
| 运行组 | `store.py`、`cli.py`、`schema_extract/README.md`、`tools/README.md` | 统一落地这些共用文件。store 的唯一监控需求是可选 `on_wait(elapsed)`；CLI 整合等待 stderr、人读时间/输出副本 display。普通恢复与控制行为不变。 |
| monitor/schema 组 | `wakeup.py`；`monitor.py` 的 scan/record_scan 观察段；`test_wakeup_retry.py` 与必要观察回归 | 负责采样/提交、过时投影、单 PID claim 观察和 formatter；不改 monitor 的 cancel/recovery 控制段。 |
| monitor/schema 组 | `prompt/operations/wakeup.md`、DOC1/DOC2/DOC5、`prompt/phases/01_contract.md` | 同线程简报时区、非 Tensor 样例、存储命名说明。 |
| monitor/schema 组 | MLA/GEMM 的两份 root 及本轮定点修订记录 | 限域表达修正、Version、前后摘要、原 receipt 来源、现行检查。FMHA/Sparse 不抢改。 |
| 已修部分保留 | `yaml_format.py`、`rules/symbols.py`、两个 Version prompt 及相邻现有测试 | 默认不增源码。后续精简只能消除重复，不删语义边界。 |
| 根代理 | 当前 wiki/Supervisor 索引整合；第四轮 issue 台账/归档 | 用真实实现与验证收口，不将旧归档状态当验收。 |

### 两组可直接并行采用的接口

```python
# 运行组：其余调用者继续无参数使用。
Store.control_transaction(self, *, on_wait=None)
# on_wait(elapsed_seconds: float) -> None

# monitor/schema 组：现有 now/sender 测试接口保留。
Clock.__init__(self, store, repo, *, now=None, sender=None, on_wait=None)
# on_wait(phase: str, elapsed_seconds: float) -> None
# phase 固定为 ack / scan_claim / scan_commit / scan_cleanup。

# monitor/schema 组提供，运行组在 CLI 引用；均不读取或写入 ledger。
format_beijing_time(timestamp) -> str
clock_display(result: dict) -> dict
```

`Store` 仅在可选 callback 存在时测量该次事务进入的 monotonic elapsed；每次 `SupervisorBusy` 后、0.05 秒 sleep 之前调用，回调时未持 supervisor 锁。callback 不参与授权、重试条件或成功判定，返回值忽略，正文及 `__exit__` 异常不调用它来重放事务。Clock 将四个固定 phase 连同 elapsed 转发；默认 `None` 与当前行为一致。

CLI 为前台 monitor 创建节流回调：等待满 1 秒首次写 stderr，之后每满 30 秒最多一次，`flush=True`；传入 `Clock(..., on_wait=...)`。这些数值仅控制展示，不能终止等待；其余 CLI 命令和 `_clock` 不安装回调。

formatter 返回带北京时间/UTC+8 标注的完整日期与秒；None 显示“未记录”，原 None 不改。`clock_display` 返回深拷贝，只为以下已存在的绝对时间补同名 `_display`：根层的 observed_at/next_due_at；根层或 scan.clock 下 check、last_scan 的 started_at/due_at/ack_at/observed_at/completed_at；reminder 的 due_at/next_attempt_at/first_queued_at/last_queued_at 及 ack.at；顶层 ack.at；daemon.heartbeat_at。绑定 scan 的 clock.next_due_at 同样覆盖。不要递归转换任意 `_at` 键、run 证据、PID 身份、duration、elapsed/overdue/age 或历史证据对象；缺字段不制造原始值。

CLI 仅在打印 monitor 命令返回值时调用 `clock_display`，包括原有 JSON 形式；所有原 key/value/type 保持不变，新增 display 只是输出副本，不能反向传回调度。`inspect --json` 完全沿原输出；`_print_observations` 用 formatter 处理人读的 at/last。由 monitor/schema 组提供 helper 并测试副本不变性，运行组只负责调用位置，双方无需等待对方选择接口。

建议顺序：先确定 Store 可选回调和 CLI 展示接口；monitor/schema 组实现观察修正、展示 helper 与规则/两个实例；运行组整合共享入口；各自定向检查，根代理统一跑必要的全量离线检查；第三轮只精简，第四轮核实精简未删门禁/说明后更新原稳定 ID。无需在实施中再询问用户类型系统或硬件调查偏好。

### 正式产物修订及证据交付

用户已授权本轮全 issue 优化，MLA/GEMM 不再等另一轮许可，也不通过旧 extract 重新跑。第二轮须在修稿前保留原 root 和 receipt 的字节/摘要，记录具体字段、固定源码依据、原批准只覆盖原字节、修订作者与独立核验范围；完成后记录新的根文件摘要与检查结果。可以使用本轮一个明确的修订记录及现有 .extraction 附加证据位置，不另建通用迁移机制。

当前阅读入口必须能追到这份修订记录，清楚区分原正常 P6 交付和本轮人工限域表达修订；原 receipt 不得悄悄被当作新字节的批准。旧 runtime、checkpoint、P6 approval、调用数及历史 batch delivery 证据不修改。给根代理提供逐算子的 before/after manifest、原 receipt 路径/摘要、修订记录路径，供当前 wiki/Supervisor 索引更新。不能只写勘误而留正式 root 不改，也不能生成伪造的新 P6。

## 本轮实际验证与后续交接

实际执行：

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=schema_template python3 -B -m unittest tools.tests.test_yaml tools.tests.test_normalization tools.tests.test_schema -q`：**76 项通过**。涵盖现有 normalizer 新测试及 Contract/只读 API 等相邻规则；没有编写或运行模型能力回归。
- 现行 checker 对正式 MLA root、GEMM root 分别执行 `--read-only --skip-hardware`：退出均为 0，分别覆盖 1/2 YAML，规范化预览均为 0 改动、dispatch 通过。该结果证明结构/格式可检查，不能据此免除上述语义表达修订。
- 当前 monitor 的旧观察覆盖确定性复现及真实 flock/SIGINT 两场景：结果如前文。测试只触及临时目录和自建子进程。

根代理正在统一运行 extract 全量离线基线，本报告不重复等待或宣称该结果。第二轮应将本轮新发现变成必要回归，旧诊断/说明改动通过现有相关测试即可，不为低影响文字逐句添加镜像测试。

最终 FMHA/Sparse 专项交接：重点复核其正式 Input/Output 的 None/分配/同对象返回、Sparse 目的地旧值 NaN 的完整有效域、各 Storage.location 的所属范围、寄存器设计槽位与编译保证的区别。保留已交付 continuation_audit 的原始待审计事实；专项审查只能记录自己真实完成的范围，不能将工具修复、模板通过或本报告替代两份最终独立全文审阅。FMHA 原自动失败与取消的 Sparse successor 保持关闭。
