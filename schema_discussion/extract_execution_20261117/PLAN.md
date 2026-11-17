# Extract 执行工具实施计划

建议按四轮、每轮两位全新 MAX 实施；之后两位全新 XHIGH 独立验收、一轮新 MAX 修复、原 XHIGH 复验。先固定小接口和安全边界，再接六阶段，最后收口恢复、交付与实际 CLI 验证。
本轮只形成计划。基线 `27f36e5`；不改 `schema_wiki`、不重建 schema_docs/Template 字段和格式规则，不运行模型、Extract 或 GPU。实装后的临时小样本验收与批量重提取分开。

## 1. 整合决定

- 采用现行六阶段方法：P2 独立审定机制、root 和有限代表；含 root ≤10，保留关键知识前提下越少越好。参数、局部变化和已知组合只进入 `dispatch.md`。
- P4 落实已定取舍和写作安排；P5 的内容共享由实际整区 `from` 决定。选择关系、拟议等待、实际来源分开，不能把它们改名后继续当单父树。
- 所有阶段的 worker/reviewer 默认 Claude Code `qwen3.8-max[1m]`、`effort=max`；保留显式 Codex 后端。Supervisor 的 Codex queue 提醒不属于阶段后端。
- 按用户最新要求，每阶段、角色独立新会话，P5 另按节点分开；取消现有 P1→P2 worker 共会话特例。同一作用域内续写/退修可精确 resume，P6 按需才启动 worker。
- 保留有界裁决与限域修稿，否决报告 A 的“删除 reviewer 裁决补丁”建议：现行工具协议明确授权 reviewer 提出补丁、Runner 应用，P6 改稿另有独立签收；它与实现代码堆补丁不是同一问题。
- 不新增强制 PNG 交付或绘图框架。正式交付要求是 raw YAML 集合、中文目录及交付凭据；Template 图功能按需使用，有限图检查与全目录结构验收分别记录。
- 唯一正式脚本改为 `schema_extract/extract_yaml.py`，删除旧名入口；全部固定模型指令归入 `schema_extract/prompt/`，现有 phases 随之迁入。

## 2. 目标目录与模块职责

```text
schema_extract/
  extract_yaml.py                 # 唯一正式 CLI 脚本
  prompt/
    common.md                    # 表达重点、材料使用及共同边界
    roles/                       # worker、reviewer
    phases/                      # 六阶段页，修正链接和会话说明
    operations/                  # 恢复、裁决/签收、review_eval、Supervisor 提醒
  tools/                         # 沿用平铺模块；仅新增 backends.py、isolation.py
```

| 模块 | 实装后职责及精简边界 |
| --- | --- |
| `artifacts.py`、`receipts.py`、`comparators.py` | 当前字段、Template 适配、阶段权限、规范化事务、原稿/最终证据和逐输出验收；不复制 Template parser/resolver/formatter。 |
| `stages.py`、`store.py`、`runner.py` | 选择与安排、实际来源调度、单写者、候选/批准、定点返回、合入和状态提交；Runner 只编排这些明确操作。 |
| `agent.py`、新增 `backends.py`、`isolation.py` | 共用进程生命周期；两个明确后端分别构造命令、解释原生结果；专用 Linux 隔离包装，不建 registry/容器平台。 |
| `prompts.py`、`snapshots.py`、`reading.py` | 选取冻结提示、组装动态材料、冻结实际输入和提供真实来源定位；不再内嵌大段模型指令。 |
| `processes.py`、`recovery.py`、`budget.py`、`audit.py`、`observation.py` | 保留启动门、进程身份、取消结算、可信恢复、原预算和有据用量；Codex 观测留在后端适用范围。 |
| `maintenance.py`、`runtime.py` | 仅保留当前 run 的限域维护和冻结执行身份核验；规则/执行器跨版本换绑与旧批准迁移退出主链。 |
| `publication.py`、`standalone.py`、`monitor.py`、`wakeup.py`、`cli.py`、`review_eval.py` | 保留原子交付、独立视图、轻量观察与已有 Supervisor 控制；全部实际模型入口共用后端协议。 |

删除 `legacy.py`、`compatibility.py` 及对应 import/report/upgrade 命令；`continuation.py`、`revisions.py`、`revision_control.py` 的跨版本语义采用、predecessor/successor、旧批准 adoption 和迁移分支退出。
先把仍有价值的当前 run 上下文核验、排空屏障、维护与发布屏障收回上述模块，再删除失用模块；不凭文件名一刀切掉正确性职责。
删除 `parent/from_parent`、确定性父表、换父迁移、旧 standalone exporter fallback、旧包布局探测及其专属测试。保留当前运行的修订记录、候选、原始事件、checkpoint、退修/裁决计数与预算。
旧 run 和 archive 留在原处，不转换、不回写、不要求新入口恢复；不保留兼容壳、旧字段别名或双模式路径。

## 3. 先固定四个小接口

1. **阶段决定**：复用 proposal/receipt 的 `plan`。P2 条目最小为 `{id, instance}`，instance 是可复查的具体调用/模板实例定位；P4 条目为 `{id, wait_for}`，只排列已选非 root，等待项可零个或多个。详细机制、差额、依据仍在目录及既有 conclusions/evidence/handoff。
2. **产物检查**：`artifacts` 提供阶段快照校验、规范化候选事务和当前来源上下文；返回检查对象、诊断、原/最终摘要、实际 origins/manifest/identity，沿用既有 validation 存储，不另建来源数据库。
3. **调度与影响**：`stages` 接受 P2 选择、P4 安排及从当前 raw 解析的实际来源信息，返回就绪调用/受影响范围；`store` 保存原草稿和批准投影，`runner` 唯一执行及提交。
4. **调用与结果**：`agent.run(attempt, prompt, schema, timeout)` 返回 backend、退出/终止状态、原生完成状态、session、response、usage 和原始日志位置；model/effort 分 requested/reported。Runner 不判断某家 CLI 的事件名。

P2 选择与 P4 安排分别绑定其独立 reviewer 回执；state 只保留运行所需投影和摘要，不重复保存长文字。P4 不得增删 ID 或更换 P2 已定实例，变更须有据返回 P2。
P4 每个已选非 root 恰有一项安排；等待只指向已选 ID，拒绝自依赖和循环，root 已完成。不把“暂无实际 from 的首稿”误当作无权开写，也不让拟议等待自动变成共享内容。
阶段批准按职责投影：P1 的共享前提、P2 的选择/来源、P3 的 root Design/when、P4 的安排、P5 的节点有效内容各自核验；后阶段合法增补不抹掉前阶段依据，也不把旧全文 SHA 当作永远不许变化的锁。
本次 attempt 仍绑定 nonce、当前决定、写范围、输入/checkpoint、代际、规则/执行身份；批准另绑定最终 raw、实际来源身份及真实 reviewer 回执。

## 4. 六阶段执行及返回

| 阶段 | 工作边界与放行对象 |
| --- | --- |
| P1 | 写任务元数据、算法 Contract、`CONTRACT.Hardware_specification.Execution` 的最小有据登记，其余硬件与 starter 相同；独立核对全部输出、验收依据与已知缺项，不要求完成 Design。 |
| P2 | 新 worker/reviewer；只改 root Source/Dispatch/Version 与目录草案。Reviewer 审机制、真实 root、每个代表相对有效基线的新增主体价值及记录去处；须降级时先由 worker 对齐目录与集合再批准。 |
| P3 | 新角色对；完成 root 本地 Design 三区、必要 Realization、自身实际 when/Version，Contract 与所选身份只读；按完整 root 有效全文审阅。 |
| P4 | 新角色对；只写目录和既有计划/交接，读完整 root 与相关基线，落实 P2。未来文件标“仅计划”，不创建占位 YAML、不制造链接，不重新惯例选型。 |
| P5 | 每节点新角色对；按拟议等待准备工作，只写自己的 YAML。首次从 starter 建自身身份和待写区域，不复制父正文、不自动写三条 from；已有 raw 原样续写。实际共享仅引用已存在、已审来源。 |
| P6 | 全新 reviewer 先独立审全部最终成果，再读必要过程记录反查知识去向；需修改才按明确文件/字段启动 worker。完成全目录检查和最终全文/目录审阅后才 ready。 |

P2 的候选研究数可多于十；完整集合仍 ≤10。程序核数量、身份、权限和证据，独特机制与局部差额是否足够由 reviewer 判断，不按关键词、源码分支数或文案长度替代。
P5 审共享 Contract、各区 raw 引用链及完整使用上下文：范围、绑定、数值、跨区接口、工作归属、事件、存储寿命、Realization 和输入到输出贯通；provider 获批不证明 consumer 适用。
同一当前共用定义可读一次，全部使用节点仍覆盖；Contract 只在 root，when/Version/Realization 不继承。单 root 仍完成 P4 与 P6，P5 显式跳过。

| 实质影响 | 返回及保留规则 |
| --- | --- |
| 普通新记录、未改变义务的未展开候选 | 当前工作继续；P3 交 P4，P5 交 P6 集中落实和审阅，不新增完整 YAML。 |
| 仅本节点整区本地化或换成已审同一区域 | 当前 P5 核对全文与实际图；不自动回 P4，P6 对齐目录共享摘要。 |
| 已定代表的登记、写作等待或未来安排改变 | 回 P4 修受影响安排；保留其他稿件、批准和计数。 |
| 具体事实/关键关系表明机制或代表判断错误 | 回 P2 修相应判断，再由 P4 按需落实；不默认重选 root 或所有代表。 |
| provider 定义错误；来源正确但不适用于当前绑定 | 前者回实际定义所在 P3/P5，后者修 consumer；资格来自真实 origins，不限“祖先”。 |
| 共享 Contract 错误；真实任务/来源改变 | 前者回 P1 并复核实际受影响范围；后者另建运行。P6 越界问题同样走所属阶段。 |

已有固定源码的具体误读也可返回，不强制“必须首次发现新行”。保留稳定问题/证据去重、原预算和代际，防止同一请求反复重开取得新额度。
完整旧稿可先交新 reviewer，未完成稿回 worker；原批准真实性与当前适用性分别核验，维护不替代缺失的正常阶段或最终 P6。

## 5. 规范化、来源和批准的闭环

正常候选统一执行：**原稿取证 → Template 整体规范化候选 → 权限与来源比较 → 受控副本原子落地 → 严格重读检查 → 独立 review**。

- 先保存原 bytes、差异及 worker 引文；路径/文件权限先核，字段权限以 Template 规范化后的语义比较核对。可确定的大小写、引号、末句号不应在规范化之前被要求手工修好。
- `normalize_documents/normalize_directory` 只生成候选；其 `source_hashes` 指原输入，候选自己的 hash/行号不等于已落地 raw。Extract 不另写规范化规则。
- 对一致的当前快照一起规范化，涵盖本候选、root、必要 provider；修改已审定义时纳入其全部真实使用者的引用同步。候选所需改动超出当前权限则整体不落地，按影响返回，不能截断关联修正。
- 原始摘要未变、全部候选改动在授权范围内，才写入 Runner 暂存副本；完成落盘重读、严格检查及 checkpoint 后提交状态指针。无变化不重写，纯格式修正不伪造 Version 增长。
- worker 原回执仍锚定原稿；既有 validation 连接原/最终摘要、corrections 和新 checkpoint。Reviewer 从最终 raw 取证，禁止挪用候选虚拟位置或把旧行号冒充新证据。
- Reviewer 普通调用始终只读；已批准的字节不再隐式规范化。后续合法修订产生新候选和对应批准，发布不偷偷改字节。

| 检查场景 | 检查范围，不放松的门禁 |
| --- | --- |
| P1 | 复用 Template 算法/结构规则的适用检查及阶段权限；不要求未写 Design 完成，硬件只校本阶段登记边界。 |
| P2/P4 | 用当前字段与目录阶段门禁核对选择、记录、链接、实际/未来文件集合；P4 支持未开始的计划，不要求未来 YAML 存在。 |
| P3/P5 | `validate_documents` 严格检查受控完成快照，加适用的 `validate_local_repositories`；root/目标及所有解析必需来源齐全。另核真实目录的阶段不变量和文件清单，不伪造完整 dispatch。 |
| P6/发布/正式 standalone | `validate_resolved_file` 等完整目录 API 覆盖全部直属 YAML、dispatch、实际图和可确定来源；选定集合精确等于最终集合，不能沿用中间检查的排除项。 |

P3/P5 可排除未完成或已失效的无关草稿，但记录 checked/excluded 范围，所需 provider 不可排除；不能把“零 I/O 快照 API 不读 dispatch”变成跳过目录/文件完整性的理由。
模型按需调用 CLI 时明确 `--read-only --skip-hardware`；只读模式检查的是规范化预览，有未落地修正的返回值按诊断解释，不冒充最终 raw 已通过。
`source_manifest(node)` 只含 root、自身和同名区域实际链；**不是所有可读文件，也不是整个节点 from 图的可达闭包**。加载检查材料须补齐 provider 其他区域继续引用的节点；全图查环包括跨区边。
实际内容依赖从 raw `from`、`origins` 和共享 Contract 计算。可提供其他已审设计作只读素材，但不能把全部素材变成批准依赖；P4 `wait_for` 同样不生成永久内容依赖。
合入只采用目标 raw；重算最新实际图，核对真正采用的来源在调用时已提供且已审、当前仍适用，退休依赖失效的在途结果。无关兄弟完成不使当前结果过期。
来源全文件 SHA 变化先核当前 provenance：即使变化位于未使用区域，也不能沿用旧 identity。能机械证明实际定义、Contract 与使用上下文不变时，在既有 validation 记录当前身份及等价依据；否则独立复审相关使用者，不退回整棵虚构父树。

保留现有一次 `select/patch/blocked` 裁决。当前 patch 是 `{path, old, new}` 的唯一匹配文本替换，并非字段操作；Runner 应用后先走同一 Template 规范化、权限核验、原子落地与严格重检，自动大小写/引号/标点修正不记裁决失败。选择现有候选时，其版本/来源调整及规范化须在交给裁决前完成。
凡裁决产生改稿，复用现有 signoff 通道，由全新 reviewer 审最终 raw 及本阶段有效全文，只有此次签收形成最终批准；将当前仅 P6 的触发条件推广至其他裁决改稿范围。直接选择实际读过且不再变化的规范候选无需额外签收；正常修稿与维护仍是 worker→独立 reviewer。
这是一次有界签收，不新增裁决循环：`budget.py` 在原 scope 额度内预留一个签收调用及时间，沿用仅 approve/blocked、无新增重试的限制，恢复/重开不重置已用次数；未签收候选不得解锁后续或发布。

## 6. Claude、隔离与提示

Claude 命令使用参数数组、文件 stdin 和 `-p --model qwen3.8-max[1m] --effort max --output-format stream-json --verbose --json-schema <JSON>`；环境中的 effort 与参数一致，不叠加固定 thinking budget。
默认后端只用于新建任务；调用、恢复、后台和审计核验已登记的 backend/model/effort/session 身份，缺字段明确拒绝，不默认为 Qwen，也不为旧状态设计迁移。
新会话读取真实 session_id，同作用域续接只用精确 UUID；稳定逻辑 cwd、各角色独立 session 存储与每次独立 attempt 配合。中断先确认终止，再退休会话、以 checkpoint 和短交接新建；不猜最近会话。
成功须同时满足退出 0、明确成功终结事件、无调用错误、session 一致、有效 structured_output、nonce/schema 和产物门禁。普通文本、截断 JSONL、缺终结、冲突 session 均不放行；恢复也用同一 parser，不伪造 Codex 事件或静默 fallback。
用量保留后端原口径及证据，区分每调用和累计会话、缓存读/写、服务报告费用；按 attempt 去重。未知项不填 0，Claude 不套 Codex cache 子集公式，报告费用不等于账单；失败/恢复不重置预算。
隔离先做专用 Linux user/mount namespace 包装：仅暴露必要材料和 CLI 运行依赖，冻结源码/规则、非授权产物只读，Runner 状态及其他宿主写路径不可访问；worker 仅授权产物位置可写，reviewer 产物全只读。CLI 会话与临时存储另设最小可写目录，字段权限仍由 Runner 差异门禁检查。
将挂载视图、可写目录、身份/能力收紧及退出后进程清理作为整体实测；防目录替换、符号链接与重新挂载逃逸。不能只靠 chmod、工具名单或 namespace 创建成功声明隔离完成，也不建通用沙箱平台。
工具侧禁内部委派及非必要 hooks/plugins/MCP，reviewer 只开放读取能力；真实取证/检查优先由 Runner 提供快照或固定只读 helper。保留已配置认证及 API 连接，设置和凭据不写进 prompt/审计；如需网络约束，说明实际边界，不以禁 Bash 代替网络隔离。
固定提示盘点覆盖阶段、角色、退修、恢复交接、裁决/签收、维护、`review_eval`、monitor 工程交接和 wakeup；机械 schema/枚举/普通错误诊断留代码，不造复杂模板引擎。
`prompts.py` 只装配身份、范围、可读材料、当前决定、入口、反馈和必要交接；每次调用显式给齐，不能依赖新模型继承对话。动态信息走每次 stdin，不仅更新 Claude 首次系统提示快照。
共同提示简短提醒文本表达重点/必要补充、变量名避免生僻自造缩写；长度、大小写、排版明细交 Template，不逐项占 prompt。`provided_documents` 标记已嵌入材料，冻结 prompt 全树及实际 `prompt.txt`。
P6 先独立理解成果，再读取 P2/P4/P5 的必要交接定位未落实事实；保留材料指针和阅读顺序，不一律删除 handoff，也不把前审结论当作正文或本次结论。
框架问题沿既有 `framework_candidates`：worker 提候选，reviewer 独立确认，Supervisor 按稳定 ID/根因处理；汇总只把已确认项当正式待办，无候选不加报告，不另建台账。

## 7. 四轮双 MAX 的实施顺序

每轮使用两位全新 MAX；表内共享文件只归指定 owner。另一位提交接口需求，不能同轮同时编辑 `runner.py/store.py/stages.py`。每轮退出时合并检查接口与实际证据，不以临时兼容层掩盖缺项。

| 轮次 | MAX A | MAX B | 退出条件 |
| --- | --- | --- | --- |
| 1：接口与基础 | 独占 `artifacts/receipts/comparators`：当前字段、最小 plan 协议、阶段权限、Template 事务及批准身份。 | 独占 `agent/backends/isolation/processes/observation`：双后端结果、会话、环境和最小隔离；需要改原生解析时同时负责 `audit`。 | 四接口固定；临时文件及假 CLI 证明允许写/拒越界、只读 reviewer、规范化身份、成功/失败结果和终止确认；不宣称六阶段已通。 |
| 2：正常六阶段 | 独占 `stages/store/runner`：P1–P6 主路径、选择与等待、DAG、合入、批准；接第一轮 API，并将 Runner 内固定提示调用切向 B 提供的文件。 | 独占 `prompts/prompt/snapshots/cli/review_eval`：全部提示外置、会话材料、默认值、冻结、入口改名；一并迁出 monitor/wakeup/recovery 的固定指令。 | 假双后端跑通单 root 和少量多来源正常闭环；P4 不扩张集合，P6 收到知识去向材料；冻结/后台调用都使用新入口和默认配置。 |
| 3：返回与交付 | 独占 `runner/stages/store/maintenance/budget`：真实影响返回、当前 run 限域修稿、裁决候选规范化与一次签收预留、在途退休及原预算保留；断开旧迁移调用。 | 独占 `publication/standalone/runtime/recovery/audit/monitor/wakeup`：可信恢复、用量、发布/视图、监督；A 断开依赖后再删除兼容/跨版本迁移模块。 | provider 修复只影响真实使用者，普通记录不重开；恢复不串会话/重计预算，P6 与最终目录一致，发布冲突及来源漂移拒绝，旧兼容活路径清除。 |
| 4：集成与文档 | 负责阶段/Template/来源/批准链的集成修复及对应既有测试，继续独占 `runner/stages/store/artifacts/receipts/maintenance/budget`。 | 负责执行器/隔离/恢复/交付集成及对应测试，统一活动引用和 `README/REVIEW/tools README/EXECUTOR_ALIGNMENT/EXTRACT_SCHEMA`，补实际 CLI 验证证据。 | 必要测试通过，新入口/冻结/提示完整；受控实测说明确切覆盖与限制，当前文档如实标明已实现能力。 |

第四轮如需改双方接口，先由主代理确定责任再串行落地。活动脚本、测试、文档、冻结启动、monitor/wakeup/review_eval 同步新名；不批量替换历史 archive 或 schema_wiki。

## 8. 精简验证与独立验收

离线验证集中保护行为，不为搬文件写镜像测试：

1. 单 root 与多代表完整六阶段；11 份拒绝、参数项不晋升、P4 选择不漂移、未来文件无假链接、P6 实有集合精确且目录完整。
2. P1 Execution/P2 Source/P3 when 的合法权限，以及 P5 provider/目录、P6 scope 越界拒绝；规范化必要联动不拆散，原稿和最终证据可复核，批准后零隐式改写。
3. 三区不同来源、多跳链、跨区环、未审来源、来源漂移、无关兄弟变化；实际区来源与图加载闭包不同，定点返工保留其他稿件/额度，迟到结果无权合入。
4. 双后端成功/截断/缺 structured_output/session 错配/错误 nonce、精确续接、超时取消/未确认终止；普通、维护、裁决及 review_eval 共用结果规则。
5. 恢复与单写者、已用预算/一次决定、用量去重/未知；裁决补丁的可确定格式修正自动完成、最终候选一次独立签收及预算不重置；发布幂等/冲突、standalone 当前来源和全目录前置检查。

先运行受影响的现有测试，第四轮再完整跑适用 Extract suite；Template 仅做相关公共 API 回归，不新增其规则测试矩阵。删除只服务旧格式、旧父树及跨版本迁移的测试。
真实 CLI 先只补已有证据缺口：用最终动态 proposal schema 验新 worker 写文件、独立 reviewer 只读、同作用域精确 resume，以及真实 session/result/usage 解析。隔离的越界/目录替换/链接/取消/子进程退出先在临时目录离线实测，再结合所选 CLI 验会话持久化。
若有限 smoke 尚不能证明 Runner 集成，第四轮运行一份临时小来源、固定调用/墙钟上限的真实 Claude 全链路样本；经 P1–P6、目录、批准和临时发布闭环，不批量提取 schema_wiki、不用 GPU。替身通过与模型语义审阅分别报告。
只在实际需要有限事件关系验证且有显式样本时调用现有 `pipeline.configuration/analyze`；要 PNG 才用现有 GraphOutput。样本缺值/分析未完成明确记录，不编参数、不新增默认硬门禁；图/独立视图在 raw 创作目录外产生并携带来源凭据。

随后两位全新 XHIGH **各自读完整最终 diff 和必要证据**：一位重点验六阶段、Template、来源、权限与设计知识去向；另一位重点验后端、隔离、提示、恢复、预算/用量和交付。分别出具体文件、失败条件及复现/证据，不互相代签。
再一轮全新 MAX 按意见修复，共享文件仍单一 owner；原两位 XHIGH 对最终改动逐项复验，必要时补受影响验证。全部门禁与意见闭合后，才把执行器标为可用于正式提取。

## 9. 已证实、后续核实与待决

已证实：当前规范/Template 已定义唯一字段、整区 DAG、候选规范化与严格 API；现有 Extract 仍是旧字段/父树并直接绑定 Codex。依据两份独立报告及本次关键源码核对。
本机已有记录仅证明 Claude `2.1.316` 的小 schema 读/写、精确 resume 基本路径；报告 B 的 unshare 只证明 user+mount namespace 创建成功，**未证明完整隔离或完整提案协议**。
后续技术核实由实施完成：所选隔离下的实际权限与取消、完整动态 schema、模型/effort/session/result/usage 实际回报、受控全链路。失败应报告具体事实并修复，不静默切换后端或降低门禁。
当前没有需另行向用户选择的设计分歧；本计划供审阅，后续按实施授权推进。

依据：[规划 A](01_workflow_plan.md)、[规划 B](02_executor_plan.md)、[六阶段方法](../../EXTRACT_SCHEMA.md)、[REVIEW](../../schema_extract/REVIEW.md)、[裁决协议](../../schema_extract/tools/README.md#退修与裁决)、[Template API](../../schema_template/tools/schema.py)、[来源身份](../../schema_template/tools/design_resolution.py)、[图操作](../../schema_template/README.md#微流水线图)。
