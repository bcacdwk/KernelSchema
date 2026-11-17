# Extract 对接独立规划 B：执行器、提示与全链路

规划基线：`27f36e5`，2026-11-17。只读代码、规范、定向归档和本机 CLI 帮助；未读另一规划报告，未运行模型、GPU 或真实提取。

## 结论

直接更新当前 Extract，建议四轮实施，每轮两位新 MAX；随后两位新 XHIGH 独立验收、一轮新 MAX 修正、原 XHIGH 复验。
核心保留项是独立审阅、冻结材料、单写者、可验证 checkpoint、进程结算、有限预算与原子发布；单父计划、旧格式和换版迁移支路不是新流程的前提。
默认 worker/reviewer 统一为 Claude Code `qwen3.8-max[1m]`、`effort=max`，Codex 保留显式可选后端；Supervisor 和同线程提醒仍使用 Codex。
唯一外部脚本定为 `schema_extract/extract_yaml.py`，直接替换 `extract.py`；内部模块继续平放在 `tools/`，不增加多提供者插件框架。

## 已核实的依据与缺口

| 位置 | 当前事实与实施含义 |
| --- | --- |
| `EXECUTOR_ALIGNMENT.md` | 已有 Claude 2.1.316 的 high 读取、max 精确续接、xhigh 写入及小型 structured_output 实测；未证明完整提案 schema、正式隔离和六阶段可用。 |
| `agent.py:CodexCLI.command/run` | 命令、事件成功判断及用量观测均绑定 Codex；参数数组、stdin 文件、启动门、超时和取消结算值得保留。 |
| `runner.py:_session/_reconcile/run` | UUID 检查和角色分离已有基础；实例化直接使用 CodexCLI，恢复还直接判断 `turn.completed`。 |
| `processes.py`、`recovery.py` | 启动门、boot/start 身份、所有权确认、终止 fence、时间高水位已有实际恢复价值，不能随后端替换丢掉。 |
| `stages.py:STAGES/session_group/next_calls` | P1/P2 worker 共用 `P1_P2`，P5 只等待直接父；需要独立阶段会话和真实来源依赖。 |
| `store.py:prepare_attempt/start_node` | 每次复制 workspace，所有普通文件设为可写；按祖先刷新。复制与 chmod 不能证明 Claude 的 OS 隔离。 |
| `artifacts.py:validate_artifacts/validate_plan/check_complete` | 仍读小写旧字段、冻结整个硬件区、七字段单父计划；外部 checker 未带 `--read-only`，可能跨阶段自动写回。 |
| `receipts.py`、`stages.py:_approve` | proposal→机器生成 receipt、nonce、来源引文和摘要应保留；P5 evidence/批准目前仍围绕 child 与 direct_parent。 |
| `prompts.py:build_prompt` | 固定指令大量内嵌；含单父/from_parent/迁移语义，并向 reviewer 剔除全部 handoff，不能满足 P6 去向核对。 |
| `review_eval.py:evaluation_prompt/evaluate` | 另一处完整 reviewer 提示与直接 CodexCLI 调用，必须纳入统一入口，不能只改正常 Runner。 |
| `wakeup.py:message`、`monitor.py:classify` | 有直接投递给 Supervisor 的固定指令、恢复建议和工程交接；runner/recovery 的回填文字也可能进入下一轮模型输入。 |
| `runtime.py`、`revisions.py`、`compatibility.py`、`legacy.py` | 冻结身份检查和当前修订能力与大量兼容、迁移、前任继承支路混合，需要分开保留和删除。 |
| `audit.py:parse_events/result_usage`、`observation.py` | Codex stream 与原生 session/SQLite 观测相互校验；不能把 Claude cache 字段硬塞进其“缓存是输入子集”的口径。 |
| `snapshots.py:rule_files/frozen_invocation` | 当前只枚举 phases，未冻结新 prompt 树；入口路径、冻结 PYTHONPATH、后台启动和阅读辅助需一起改。 |

本机只读核对：Claude Code `2.1.316`，Codex CLI `0.166.0`；Codex 的现有全局权限参数接 `exec/resume --help` 均退出 0。
Claude 帮助有 `--effort max`、`--json-schema`、`stream-json`、精确 `--resume`、`--restricted`、`--safe-mode`；这些不是已验证的 OS 隔离。
Claude `--system-prompt-snapshot` 默认保留首次系统提示；每次变化的身份、反馈、交接和阶段状态必须放 stdin，不能只更新 append-system-prompt。
本机没有 bwrap/podman/docker/firejail/nsjail；`unshare --user --map-root-user --mount true` 退出 0，仅证明创建相关 namespace 可行。
Codex `queue --help` 明确：投递成功后恢复失败仍可能退出成功；已有 queue/ack/scan 区分有价值，不把排队成功当 Supervisor 已处理。
OpenAI Docs 官方搜索与页面由主代理尝试后均为 403；本规划的 Codex 能力依据限于安装版帮助，不声称已验证在线最新文档。
定向依据见归档 `20261109-usage-review/ROOT_ANALYSIS.md`：重复材料加载、恢复首轮大上下文和局部回修有实际成本；不能由此承诺固定前缀的缓存节省比例。
归档暂停记录中的 MoE 为 154 个计划节点、96 个批准节点，支持取消按配置展开；不要求本次修复或恢复该历史 run。

## 先确定四个小接口

1. **调用接口**：既有 attempt 增加明确 backend、实际可执行文件/版本、requested model/effort、会话作用域、权限描述；`run(attempt,prompt,schema,timeout)` 仍由同一进程生命周期实现。
2. **结果接口**：统一 outcome、退出码、终止确认、实际 session、backend 原生完成状态、response、原始 trace 定位和 usage；Runner/audit 不再判断具体事件名。
3. **产物接口**：Template 返回的 raw 文档、候选规范化、实际来源图及检查诊断有明确边界；批准只绑定最终重新读取的字节与有效来源身份。
4. **调度接口**：P2/P4 的选定集合/写作安排与 P5 的实际 `from` 图分开；保留既有 `plan` 回执载体，去掉 parent 权威，不新增永久清单。

这些接口放在对应既有模块，必要时只新增一个小 `backends.py` 和一个专用隔离模块；不新增泛型 registry、provider 插件、事件总线或第二份状态库。
并行修改前由主代理确定函数参数和机器字段；同一轮 runner/store 等共享文件只归一人，依赖方用约定接口，轮末再合并检查。

## 六阶段如何落成 Python

- P1 只允许任务元数据、算法 Contract 和 `CONTRACT.Hardware_specification.Execution` 有据登记；其余硬件值与 starter 一致。P1 的局部算法检查不要求完整 Design。
- P2 新建 worker/reviewer 会话；允许 dispatch 草案及 root 的 Source/Dispatch/Version。Reviewer 在既有 conclusions 审定机制、root、完整候选与记录去处，含 root 至多十份。
- P3 写实际选定 root 的本地 Design 三区、必要 Realization 和自身 when；Contract 只读，范围补写不触发重选 root。
- P4 只写 dispatch 和既有 plan/交接；沿已定选择安排最多九个非 root，未来文件仍标仅计划，不预生成占位 YAML 或假链接。
- P5 按实际已存在且已审的来源推进；三区允许不同 provider。每个节点仅写自身 YAML，本地化或改用已审相同区通常在自身闭环，不自动退回 P4。
- P6 全新 reviewer 先审全部产物，再从 P2/P4/P5 既有记录反查知识去向；必要时新 worker 按明确文件/字段集中补 dispatch 和限域修稿，随后完整终审。
- 数量、文件集合、图无环、链接、权限等由工具检查；独特机制、记录是否足够、范围与输入到输出的贯通由 reviewer 判断，不能用关键词启发式代替。
- 返回按真实影响：Contract→P1，机制/取舍→P2，root 定义→P3，未来安排→P4，provider 定义→对应 P3/P5；普通新记录交 P6。
- 修 provider 只复核真实引用该定义的使用者；provider 正确但对当前绑定不适用时修 consumer。相关记录也要复核，未受影响的正文保留。

## Template 接入与批准身份

直接采用当前规范字段；不加大小写别名或旧字段转换层。调用冻结 Template 的公开 API，不复制 resolver、预算规则或格式算法。
`normalize_directory/normalize_documents` 产生 `NormalizedTask`；其 `source_hashes` 是原磁盘字节，candidate 文档的 hash/位置属于候选，不能混用。
每次先验证真实 worker 差异属于授权范围，再生成完整任务规范化候选；Runner 只落盘本次授权范围内的安全修正，保留来源比较和无变化不写回。
文件权限先核 raw diff，字段权限通过规范化候选比较语义差异；不因可自动修正的大小写而要求作者先手工过关，也不让格式修正掩盖受保护字段的内容改动。
若规范化必须改只读 provider 或另一阶段字段，保留候选与诊断，走实际定义的修订入口；不能截断跨文件修正后声称整个任务已规范化。
落盘后重新读取并运行严格 API，再创建 checkpoint/receipt；reviewer 从这个最终 checkpoint 取证。Reviewer 输入需要修正时先退修，不静默改它已审的字节。
Worker 提案的原始选锚先绑定提交 raw；后续规范化的原/最终摘要与修正关系进入既有 validation，不能将旧行号或引文冒充新 raw 的证据。最终批准来自读过最终快照的 reviewer。
`validate_documents` 是严格零 I/O，文件调用还需 `validate_dispatch`、可确定的本地仓库检查；P1/P2/P4 按本阶段完成范围使用 API，不能让未来草稿污染当前门禁。
CLI 预览统一使用 `--read-only --skip-hardware`；只读预览有未应用修正会返回 1，不能误记工具故障，也不能把候选检查通过记为 raw 批准。
读取 `ResolvedTask.origins/dependency_nodes/source_manifest/effective_identity` 建图；effective 视图仅作阅读，证据锚点仍是当前 raw 引用行、中间链及真实定义。
一次批准保留 raw 文件摘要、实际来源 manifest/chain、effective identity、规则和执行身份、独立 reviewer 会话及回执摘要。
Template 来源身份包含 raw 全文件摘要：即使改动发生在 provider 未被使用的区域，也必须核对当前 provenance；内容影响与来源身份变化不能混为一件事。
等价使用区可以在现有 validation 中记录当前适用性依据；实际内容变化才触发对应使用者复审，不能把旧 identity 原样冒充当前批准。P6 始终覆盖最终全目录。

## Claude/Qwen 的最小完整接入

`agent.py` 只负责共有启动/取消/结算；两个明确 backend 各负责参数、环境、原生事件及结构化结果解释。所有普通、续写、格式修复、维护、裁决、签收和 review-eval 共用它。
CLI 提供 `--backend claude|codex`、相应可执行文件选择、model 与角色 effort；默认固定 Claude/Qwen/max。后台、恢复不重新取工作树默认值，不自动换模型或后端。
Claude 使用 `-p --model 'qwen3.8-max[1m]' --effort max --output-format stream-json --verbose --json-schema <JSON>`；Python 参数数组和文件 stdin，不拼接 shell。
子进程将 `CLAUDE_CODE_EFFORT_LEVEL` 与参数一致，去掉冲突的固定 thinking budget；配置仅读取所需项，凭据及敏感 endpoint 不写 argv、prompt、回执或审计摘要。
新建调用读取真实 session_id；续写只传精确 UUID，禁止最近会话猜测。session key 至少含 backend、阶段/节点、角色和操作作用域；跨阶段/角色/后端不得复用。
保留每个角色独立、可持续的 session 存储及稳定逻辑 cwd；attempt 仍独立落盘。普通同作用域续写复用，确认中断后的旧会话退休，由 checkpoint 与短交接建立新会话。
Claude 的最终 result 要满足退出 0、成功 subtype、无调用错误、session 一致、有效 structured_output 与 nonce/schema 校验；只把 structured_output 写入 response.json。
普通 result 文本、缺失终结事件、多个冲突 session、截断/损坏 JSONL 都不能放行；恢复路径通过同一 backend parser 重建结果，不伪装成 Codex turn.completed。
分别记录 requested/reported model 与 effort、CLI 版本、会话和权限实情；未观察到的值标未知，CLI 请求字符串不证明底层服务实际采用了哪个模型。
权限最小方案是专用 Linux namespace 包装器：冻结来源/规则/证据只读，worker 仅本阶段产物可写，reviewer 产物只读，CLI 自用会话/临时目录单独可写。
隔离必须防止角色通过目录替换、符号链接、重新挂载或可写 shell 越权；不能靠文件 chmod 或 allowedTools 声称完成。实施时先用临时目录证明所选实现，再纳入启动门。
CLI 再关闭内部委派、自动插件/hooks/MCP 等额外能力，设置必要文件工具和免交互策略；restricted/safe-mode 的配置读取影响要核实，不能因此丢失已配置认证。
推荐先让 Runner 生成只读来源定位、引文预览和 effective 视图，角色用文件工具读取；确需命令时只开放固定 reading/checking helper，不给 reviewer 任意 shell。
API 网络与模型可调用工具的网络权限分别说明；既有模型服务连接要可用，不能把关闭 Bash 或 network_access 声称为完整网络隔离。
用量保留 Claude 原生 `usage/modelUsage`、缓存读/写、服务报告费用及来源；独立区分每调用与会话累计，不能直接相加造成续接重复。
Codex 保留现有有界原生观测；Claude 无依据时只报告已记录小计/未知。不把 Claude 缓存计数强制解释为 Codex 输入子集，不把 reported cost 当提供者账单。
按 `(run_id,attempt_id)` 去重，费用未知不填 0；墙钟、CLI、处理时间及已知等待分列，原预算不因失败、暂停、续写或恢复重置。

## 提示词外置与精简

将 phases 移至 `schema_extract/prompt/phases/`，其余按需使用 `common/roles/operations/`；只用能表达用途的文件名，不为每句文字建一个文件。
`prompts.py` 只选模板、读取冻结文档和组装动态数据；角色职责、返回要求、恢复/维护/裁决/签收及阶段说明均在 MD，不把固定长指令移成 Python 常量。
完整盘点 `review_eval.evaluation_prompt`、`wakeup.message`、monitor 工程交接、runner/recovery 回填指令、maintenance/revision 的模型可见文字，以及 JSON schema 的模型指令性 description。
普通异常文本和 CLI 帮助可以留 Python；凡作为 Agent 下一步指令复用的固定文字必须来自 prompt MD。删除的兼容/迁移功能不搬一份死提示进去。
动态包只给身份/阶段/角色、可读材料、精确写范围、当前批准前提、源码入口、必要真实来源闭包、反馈与关键交接；静态材料已嵌入时明确标记，避免再读一遍。
Reviewer 先读当前产物再看过程记录；不提供作者推理当结论，但要提供当前冻结决定和必须反查的事实。尤其 P6 不能继续一律删除 handoff。
Worker 提交 framework_candidates；Reviewer 在原 conclusions/issues 中认定是否需要维护，Supervisor 查重并维护既有 issues 注册表，不建立另一组报告或计分表。
字数表、casing、格式细目交 Template 自动处理，模型根据具体诊断修订；提示只写任务关键补充，少复述可见路径和原字段。
`snapshots.rule_files` 递归冻结 prompt 全树，生成的 prompt.txt 保留最终实际输入；源码、规则、模板缺失时明确报错，不回读工作树补齐。

## 四轮实施分工与验收

**第一轮：接口和可用基础，两位新 MAX。**
- A：`artifacts/receipts/comparators` 对接规范字段、Template API、阶段写权限和批准身份；提出去 parent 的最小 plan 结构与 P2/P4 分工，暂不并行改 Runner。
- B：`agent/backends/processes/observation` 建立两后端共有生命周期和结果接口，完成默认 Claude 的命令、会话、环境及隔离最小实现；只用假 CLI 和临时文件验收。
- 本轮出口：调用、结果、产物、调度四接口固定；正常写/只读、越界、取消后终止确认、structured_output 与 nonce 正反例通过。共享字段变更由主代理统一落一次。

**第二轮：六阶段与全部提示，两位新 MAX。**
- A：`stages/runner/store` 落实 P1–P6、选定集合、实际来源图与合入，接第一轮 backend；拆开 P1/P2 会话，P5 保留 raw 草稿并按实际来源刷新。
- B：`prompts/prompt/cli/snapshots/review_eval` 外置所有提示，统一入口与默认值，修阶段链接与冻结清单；review-eval 复用调用协议，不形成第二套执行器。
- 本轮出口：假 Claude 跑通单 root 到 P6，以及少量多 provider 节点；P4 不创建未来 YAML，P6 收到既有新事实，后台配置与前台一致。

**第三轮：修订、恢复与审计减重，两位新 MAX。**
- A：收敛 `revisions/maintenance` 到当前规则下有据返回和受限修稿，更新 `publication/standalone` 的实际文件集合、来源和有效批准；保留当前 run 预算与历史尝试。
- B：`runtime/audit/monitor/wakeup/recovery` 使用统一结果与身份，接 Claude 用量口径和 Supervisor 提示；清除 import-legacy、compatibility-report、upgrade-runtime 及对应兼容模块。
- A 先移除语义换版/前任迁移调用，再由 B 删除失去用途的 `continuation/revision_control` 支路；两人不同时修改共享模块。若文件仍有当前作用则保留精简部分。
- 本轮出口：多 provider 改动只影响真实使用者；正常记录不重开 P2；暂停/恢复/格式修复不重计预算或串会话；旧 run/archive 只留原地，不做转换或回归要求。

**第四轮：完整收口，两位新 MAX。**
- A：更新现有有价值的离线集成用例，覆盖双后端、新六阶段、规范化身份、来源漂移、陈旧结果、进程 fence 与发布幂等；删除专测已移除协议的用例。
- B：全仓引用与模型提示盘点、入口/冻结调用验证、说明精简；统一 `README/REVIEW/tools README/EXECUTOR_ALIGNMENT` 的当前事实，不改 schema_docs 字段语法或 Template 规则。
- 本轮出口：必要现有测试和 CLI smoke 通过、外置提示不缺材料、无旧命令或单父假设的活路径；阶段、后端、会话、权限、检查和用量报告都与实际一致。
- 实现授权后只在需要补足接口证据时做有界 Claude smoke：完整动态提案 schema、权限下文件读写、同作用域精确续接。已有三次基础 smoke 不机械重跑；不启动真实算子提取。

两位新 XHIGH 分别验“六阶段/Template/批准”与“执行器/权限/提示/恢复/费用”，都读取最终 diff 和关键证据；问题仍进原有审阅结果。
随后一轮新 MAX 修具体验收意见，原两位 XHIGH 对最终内容复验；新增代码或发现缺口才扩展测试，不以测试数量或角色轮次代替完成标准。

## 必要验收与真实决策项

- 六阶段：单 root 仍终审；配置、局部和组合不会生成数十个完整节点；11 份拒绝，<=10 仅为上限，每份价值仍须独立审定。
- 来源：三区不同 provider、多跳链与跨区环；provider 未审不可引用，源变更拒绝陈旧批准，本节点本地化与正确但不适用的来源分别处理。
- 规范化：只读零写入、授权文件修正、跨文件越界、候选行号、重读后身份、raw/effective 引文；无关草稿不导致当前阶段误失败。
- 执行：新阶段新会话，同角色受限续接，session 不匹配/缺 structured_output/超时/取消/截断结果全部拒绝；普通和维护/裁决路径同样验。
- 权限：临时目录内实测允许写、禁止写、只读 helper、session 持久化、目录/链接逃逸和退出后子进程；namespace 创建成功不是这些项目的替代品。
- 费用：Claude 单次/续接/缺统计，Codex 原生有界观测，重复结果去重；不编造 cache、费用、模型或已确认终止。
- 当前已可直接决定：入口改名、默认 Qwen/max、阶段独立会话、四轮分工、删除兼容层、全部提示外置，不应再逐项向用户索取确认。
- 真需用户决定只在事实触发时出现：用户明确要求逐套完整交付超过十个独特设计；真实任务域/来源改变；现有机器无法提供所需隔离而必须接受更弱执行边界；实际试跑预算超出已有授权。
- 完整提案 schema、模型/usage 实际回报和所选隔离下续接是否成立属于实施验证项，不是先向用户抛出的设计选择；验证失败不静默降级。

## 给主代理的十点摘要

1. 推荐四轮、每轮两位新 MAX，接口先固定；两位 XHIGH 双线验收后再一轮 MAX 修正和原人复验。
2. 保留进程/预算/冻结/回执的价值，删单父计划及兼容、旧格式、换版迁移支路，不处理旧运行。
3. 默认 Claude `qwen3.8-max[1m]` + max；Codex 显式可选，Supervisor 提醒仍走 Codex。
4. `agent.py` 统一生命周期，两个小 backend 解释命令、事件、结果；恢复、裁决、review-eval 同样接入。
5. P1/P2 必须拆会话，所有阶段/角色独立；动态信息每次走 stdin，避免 Claude 系统提示快照固定旧内容。
6. 现机 namespace 仅做了创建验证；正式文件/进程隔离与会话读写仍须离线实测，不等同工具 allowlist。
7. P2 选择少量完整代表，P4 落写作安排，P5 依赖实际三区 from，P6 集中记录并终审。
8. Template 候选、落盘 raw、effective 来源及批准身份分开，规范化不越阶段权限、不暗改 reviewer 已审字节。
9. 提示外置覆盖 prompts、review_eval、wakeup、monitor 及恢复回填，P6 必须读既有新事实，复用原回执。
10. 最终只留 `schema_extract/extract_yaml.py` 外部入口；用量缺项保留未知，验收不做真实重提取或无边界测试扩张。
