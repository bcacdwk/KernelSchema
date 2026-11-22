# 独立 Design / Optimize / Implement / Workflow 最终实施计划

2026-11-22，xhigh 终审定稿。本文件是实施依据，自足取代本轮研究与中间方案；`06_decisions.md` 保留用户决定，`FINAL_REVIEW.md` 保留终审判断。
用户已授权定稿后直接实施、限定清理、commit、默认 Qwen/Agate 实跑和持续查修，无需等待再次确认。本轮仅完成计划检查，以下验证均是待执行要求。

## 实施中的已记录修订

以下正文保留定稿时方案。真实运行后的局部选择已由实现和必要验证替代，相应行为以本段及模块使用说明为准：

- [Issue 02](ISSUE_02.md)：Design/Optimize 的完整文件不再嵌入一个大 JSON 返回。选材只读，写作允许候选目录内直属 YAML/dispatch 的 native Write/Edit，最终只回小决定；host 保全草稿、检查输入和 Contract、规范化交付。无新稿的失败不盲修。Implement 的交稿方式未改。已通过一次真实权限/小交稿探针，但仍须业务实跑。
- [Issue 03](ISSUE_03.md)：Workflow 增加已有 Design `--reference` 的简单透传及恢复身份检查，仅用于无 Schema 的初始 Design；示例另提供真实 Agate 缓存环境资料，固定 Task/eval 不变。不新增环境探测运行时，不把缓存当本次候选观测。

- [Issue 04](ISSUE_04.md)：实际默认 Qwen/max 的完整写作需要 1775 秒，因此四入口默认单次等待统一为有界、可覆盖的 3600 秒，原运行的已固定上限不变。DOC 阅读接受本次真正返回正文的精确 Read 或 Grep 命中，拒绝只有文件名、计数、空结果或错误的查询；不再把工具名差异误当未阅读。

- [Issue 05](ISSUE_05.md)：真实 Implement 暴露必要语言绑定缺口；新增可追溯的 Triton3.7.1 官方资料，由新 Design 正式绑定，旧 Contract/失败不回填。Implement 首建新增可重复 `--reference` 保存已绑定的补充资料；Workflow 将普通资料继续传给它，Schema 近邻仍仅供 Design。补充资料与修复反馈分开，反馈以固定实物路径供 Optimize 读取；不自动联网抓 Wiki，不增加恢复或评测平台。

- [Issue 06](ISSUE_06.md)：正式服务的harness使用atrex_bench，本地atrex别名由adapter转换，固定eval不变。增加显式、严格限于已知无job的422拒绝且仅此harness修正的同目录恢复；保全旧attempt，复用完整候选而不重调模型，原预算不清零。已有或未知job仍只按原身份恢复。

模型、effort、调用总额、核心职责、Task/eval 与保护区均保持；正文中旧的 900 秒默认已被上述实测修订替代。当前执行与结果见 [EXECUTION](EXECUTION.md)。

## 1. 已定范围与最小文件

- Design 从 Task 与可用近邻写 Schema；Optimize 读完整 Schema、kernel 和真实反馈，决定保持、修实现或改设计；Implement 自有 coder 与 Agate 闭合当前设计；Workflow 串联同一入口。
- 首版消费已确定的 reference、输入/案例和误差判据，不新增自动 oracle/测试生成业务。仅有明确文字时 Design 可以工作，实测材料缺项不能伪造。
- 一个 kernel 入口覆盖任务必要域与分支；同域备选不默认全部实现。root 不隐式兜底、文件顺序不决定选路、测试样本不缩小任务合法域。
- 四目录各自拥有业务；无根共享包、公开 Python API、旧 CLI/API 兼容壳、AKA 导入/安装/调用或反向 Workflow 依赖，不导入 Extract runtime。
- `schema_extract/`、`EXTRACT_SCHEMA.md`、`schema_docs/`、`schema_template/`、`schema_wiki/`、`kernel_wiki/` 及正式发布凭据原样保留；不修改 AKA，不访问 AKA_experiment 及变体。
- 不引入独立 Reviewer、写作效果验收、qualification、owner 账户、promotion、排名或跨运行预算平台。

```text
schema_design/{design_schema.py,tools/runner.py,prompt/design.md,README.md,tools/tests/test_design.py}
schema_optimize/{optimize_schema.py,tools/runner.py,prompt/optimize.md,README.md,tools/tests/test_optimize.py}
schema_implement/{implement_kernel.py,tools/coder.py,tools/agate.py,prompt/implement.md,README.md,tools/tests/test_implement.py}
schema_workflow/{workflow.py,README.md,tools/tests/test_workflow.py,examples/vector_add/}
```

Workflow 单脚本无模型、prompt、SDK 或 Schema parser，以当前 Python、绝对脚本路径、`shell=False` 调三个 CLI；不另拆 runner 或 demo driver。
根 `DESIGN_SCHEMA.md`、`OPTIMIZE_SCHEMA.md`、`IMPLEMENT_SCHEMA.md` 更新职责与入口；规范仍只读使用现有 DOC/Template，避免复制多套规则。

## 2. 可以直接编码的 CLI

```sh
python3 schema_design/design_schema.py --task T/task.md --out R/design
python3 schema_implement/implement_kernel.py --task T/task.md --eval-task T/eval --schema S --gpu L20D --out R/implement
python3 schema_implement/implement_kernel.py --resume R/implement --feedback F
python3 schema_optimize/optimize_schema.py --task T/task.md --schema S --kernel K --feedback F --out R/optimize
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --gpu L20D --out R
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --schema S --gpu L20D --out R
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --schema S --kernel K --feedback F --gpu L20D --out R
python3 schema_workflow/workflow.py --resume R
```

- 默认 `claude` CLI、`qwen3.8-max[1m]`、effort `max`，与 Extract 一致；允许 `--model/--effort` 覆盖，正常入口不强制填写 model，不加 backend/逐阶段模型矩阵。
- 三业务入口有 `--max-calls`，默认 Design/Optimize/Implement 为 3/2/6；`--timeout-s` 默认 900，Implement 同值分别限制一次 coder 和一次远端等待。
- Workflow 有 `--max-versions` 默认 3（正整数，含初始 Schema）、`--max-calls-per-schema` 默认 6；Design 总计最多 3，每版 Optimize 总计最多 2，超时采用上述默认。
- Design 可重复 `--reference`；Implement 可接 `--kernel` seed 和重复 `--feedback`；Optimize/Workflow 的 `--feedback` 也可重复，保留原文。
- `--schema` 接目录或其中 YAML，但总是读取完整任务；`--kernel` 接完整源码目录或真正自足文件，包含必需 wrapper/dispatcher/helper/构建源码。
- Optimize 通常必需 kernel；真实 Implement `design_gap` 且尚无源码时可省略，以其结果为反馈，不用空文件冒充 kernel。
- Workflow 新运行必需 Task/out；Schema 起点直接 Implement；Schema+kernel+反馈起点先 Optimize。仅文字任务可先 Design，再具体报告 eval/gpu 缺项。
- `--out` 必须全新、不覆盖或包含输入、不落入保护区；resume 只沿原输入/模型/上限/目录，Implement 仅允许新增修复反馈，不清零计数。
- 入口按自身 `__file__` 定位 checkout；help/import 不建运行、不读认证、不初始化模型/SDK。初始化后 stdout 只输出一行绝对 `result.json` 路径，进度去 stderr。
- 退出码 `0/1/2=complete/incomplete/interrupted`；参数错误没有结果文件。Workflow 核对已知路径、stdout、退出码与结果一致，不从日志猜成功。

## 3. 真实阅读、只读模型与 host 收稿

每个模块只限定导入自己的代码和只读 Template；避免裸导入同名 `tools`。Template 使用 `schema_template/tools/schema.py` 的读取、规范化规划和校验能力，不复制 parser。
所选参考与输入带齐 root、节点、dispatch、完整中间 `from` 链和共享使用者；消费 `ResolvedTask.effective_document/source_manifest/origins_dict/effective_identity`，不把展开副本回写成原节点。
必要外部材料一次固定原字节、原位置与摘要；已有不可变 attempt 可只读引用。复制期间变化即停，反馈相对引用按原位置解析并带齐实物，不能只抄路径。
DOC 与其他外部绑定材料记录原位置/摘要；来源定位按原上下文核对，不能因复制到运行目录就声称已做原仓库检查。
DOC0=`schema_docs/0.common.md` 与本模块根方法进入首请求；DOC1–6 分别为同目录 `1.contract.algorithm_requirement.md`、`2.contract.hardware_specification.md`、`3.design.math_reshape.md`、`4.design.work_schedule.md`、`5.design.tile_pipeline.md`、`6.realization.md`。
DOC1–6、`schema_template/README.md`、`schema_template/starter.yaml`、任务绑定资料和完整源码给实际可 Read 的绝对路径；按当前数学、调度、流水线、实现问题实际读取，不只给禁用工具后的链接。

host 使用 argv 列表和 stdin，`Popen(..., shell=False, start_new_session=True)`；现有 native CLI 的命令形态如下，`SCHEMA` 为本地收包 JSON schema：

```text
claude -p --model MODEL --effort EFFORT --output-format stream-json --verbose --json-schema SCHEMA
  --safe-mode --restricted --disable-slash-commands --no-chrome --setting-sources ""
  --strict-mcp-config --mcp-config {"mcpServers":{}} --settings {"disableAllHooks":true}
  --permission-mode dontAsk --permission-prompts none --tools Read,Glob,Grep --allowedTools Read,Glob,Grep
  --disallowedTools Bash,Edit,Write,Agent,Task,Skill,WebFetch,WebSearch --add-dir MATERIAL_DIR
```

每个必要材料目录各传一次 `--add-dir`；cwd 是当前运行目录，不授予认证目录读取。CLI 权限不是 OS 隔离承诺；模型无写文件/GPU/网络工具，只有 host 物化产物与提交评测。
**禁用 settings 不能丢 Qwen 连接：**保持原 HOME、现有 `CLAUDE_CONFIG_DIR` 与原 CLI 登录来源；在子进程内以进程环境优先，补读该配置目录（默认 `~/.claude`）`settings.json.env` 的连接字段。
保留 `ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_CUSTOM_HEADERS`、大小写代理变量及已配置的 `NODE_EXTRA_CA_CERTS/SSL_CERT_FILE/SSL_CERT_DIR`；仅传给 CLI，不放进 prompt、读取目录、产物、摘要或日志，不复制认证文件到运行区。
设定本次 effort 和适当输出容量（现有 Extract 为 `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`）；关闭非必要流量/自动更新，不带入用户 hooks/MCP/插件执行。实际 CLI 不支持参数时具体排障，不重新做资格任务。
本地收包为完整 `files[{path,content}]`、notes/缺项；Optimize 加 decision/required_fix，coder 加入口与落实说明。Design 选材调用仅交来源/用途；这不是新的公开协议。
保留真正最终回答、实际 Read/Glob/Grep 路径与必要退出/诊断；不存思维正文、完整认证环境或重复调用账本。材料过大不得静默截断。
host 拒绝绝对/逃逸/重复路径、软链接与缺失入口/root；写作只接完整节点 YAML/dispatch，源码交完整文件集合，不接 patch/shell 指令作为交付。
先保留原稿，再对新候选 `normalize_directory`，只应用新输出中的候选文本，随后严格重读实际落盘文件并 `validate_directory`；原材料不改，不默认 skip-hardware、不画图或增加证明器。
Optimize 在规范化前后都按解析值核对完整 root CONTRACT；漂移留稿拒收，不静默回填。真实 Template 错误交同一写作者在余量内修，warning 如实记录。
每次模型调用前原子保存 calls_used，选材/坏返回/失败/超时都计数；超时停止并回收进程组，未知存活进程不启动替代调用。
写作只保留本次调用/session/PID 等恢复定位，不建跨运行写作会话引擎；Implement 已确认 coder 退出后可在原余量内继续新调用。

## 4. 小产物与交接字段

Design/Optimize 只保留 `input/、drafts/NN.json、schema/、notes.md、result.json`；draft 容纳原回答、读取记录与诊断，keep/repair 引用未改输入 Schema。
Implement 只保留 `inputs/、state.json、result.json、attempts/NN/`；attempt 按进展包含 `kernel/、notes.md、request.json、response.json、feedback.json` 和必要原稿，不造根候选副本。
未产出 kernel 的缺项/设计诊断也进该 attempt 的 feedback；源码一旦固定不再改，终态反馈一旦定稿不再覆盖，修复产生新 attempt。

| 模块 | result.json 业务字段 |
| --- | --- |
| Design | `status, stop_reason, calls_used, schema, notes` |
| Optimize | Design 字段加 `decision, required_fix`；无有效决定时 decision=null |
| Implement | `status, stop_reason, calls_used, schema, kernel, feedback` |
| Workflow | `status, stop_reason, last_usable, latest_attempt` |

路径相对其所属结果/反馈；下游转绝对路径，Workflow 转存时重新相对化。无可接收 Schema/完整 kernel 则为 null，坏稿仍保留。
Implement feedback 单处记录入口、完整源码摘要、固定 Schema 位置/摘要、落实说明、逐 case 正确性/时延/单位/统计、实际环境和原 request/job/response 位置。
源码推理与运行观测分开；未返回的环境信息留空，不拿请求值冒充观测。不重复 checks/performance/evaluation/next 或构造通用指标输入。
停因：`needs_input`（缺材料/能力）、`design_gap`、`implementation_failed`、`evaluation_failed`、`write_failed`、`input_changed`；运行未结清为 interrupted，定位细节在 state/feedback。
模型次数耗尽仍归实际写作/实现失败并保留 calls_used；Workflow 因上限结束用 `limit`，正常结束用 `done`，不能把所有失败改名 limit。
`required_fix` 只表示有具体、尚未解决的 Task/Contract/设计落实必需问题，notes 指向相应材料/代码；性能改善假设或单纯未达期望速度不是这个布尔值。
repair 恒为 true，keep 恒为 false，revise 必须明确填写；诊断已确认必需问题而新稿失败时也可保留 true。没有可信决定/诊断的坏响应不能推导安全。

## 5. Design 与 Optimize 的完整业务

Design host 读 Task、`kernel_wiki/` catalog 和 `schema_wiki/` 目录，实际发现 root/dispatch，目录占位不算 Schema，catalog 外现成 Schema 也可选。
通常第一调用按数学结构、调度/流水线、布局/精度与设备条件找至多三个互补近邻并实际读说明；host 装配完整来源，第二调用写，第三调用只修具体诊断。
空目录直接写；无适用近邻仍可从明确 Task 设计。坏参考可舍弃，不以来源资格、标签交集或历史成绩阻断任务。
借鉴交代条件与差异，组合核对数值路径、资源寿命和生产消费接口；跨任务内容改为本地完整区域，`from` 仅做任务内合法整区共享。
写一套覆盖必要范围的设计与分支，明确重叠选路。语义/关键前提不明则留下具体缺项与原稿，不发明假决定。
Optimize 实际读固定 Task、完整 Schema/kernel/反馈；Implement result 仅作入口，还须读其指向的源码、落实说明和原实测材料。
核对反馈绑定、案例/环境/正确性/计量范围，区分观察、推断与提案；外部成绩是起始证据，不自动变成本次验证，不造加速比。
只改 DESIGN、必要 REALIZATION/dispatch；Task、CONTRACT 解析值、oracle、kernel、原反馈不变。共享区域修改处理全部使用者，局部引用修改须完整本地化。

| decision | 含义 |
| --- | --- |
| keep | 无值得采用的设计或代码修复；保持原 Schema，不代表旧/失败 kernel 已通过本次评测。 |
| repair | 原设计不变，有具体代码落实问题；notes 给修复建议，交同版 Implement。 |
| revise | 完整新 Schema 有实质解析内容变化；说明变化及待验证收益，删除节点须说明并同步 dispatch/from。 |

非法稿、Contract 漂移或关键缺项为 incomplete+decision=null；同内容重排不能称 revise 换版本，不用 keep 隐藏写作失败。
有效文件检查到此结束写作；不增加独立 Reviewer 或性能验收，不把 Template 检查称为设计正确性证明。

## 6. Implement 自有 coder 与独立 Agate

先固定 Task、完整 Schema 和 `eval/`：`reference.py` 的 `Model(**init_kwargs).forward(**inputs)`、`input.py` 的 `_make_inputs(**input_kwargs)`、非空 `shapes.json`、实际所需 metadata/比较器及 `eval.json`。
`eval.json` 沿 SDK 名称保存 languages、requirements、harness/Bench pin、options 内误差和计时设置；保留既有实际比较器，不能用 SDK 默认容差覆盖它。
缺项/任务与栈或 GPU 冲突时不启动 coder/GPU；首版支持正式 SDK 可完整表达的 Python Model/Triton 路径，不能表达的构建/比较器明确说明，不偷换 Task。
coder 消费完整 raw/effective/from、固定验收、当前源码和最近反馈；只能返回源码与说明，不改 Schema/判据、不生成成绩。
默认复用旧源码的副本，必要时重写；返回新完整文件集才是新候选，旧 seed/旧成绩不算本次实现。完整外部可变 seed 按输入规则固定，不另造镜像层。
coder 对数学、布局、调度、同步、资源寿命、Require/Forbid 和必要分支核对落实，notes 写节点/条件/源码位置与缺口；不加通用静态证明表。
完整候选固定、计算整体摘要后 host 一次提交全部固定 cases 的正确性与计时；真实 helper/dispatcher/构建依赖整包交付，不能只送入口。
ABI/编译/数值或代码落实问题，在同 Schema 累计余量内修，每次新 attempt 后全案例重验，不拼接不同候选成绩。
已知漏分支/违反要求/未完成不能因数值通过结束；代码可修则继续，必须改设计则 design_gap，尚无 kernel 也保留真实诊断给 Optimize。
无已知必需缺口、全部固定 cases passed 且必要计时齐全才 complete；不宣称样本证明全部动态域，不另开 Implement 性能搜索。

`tools/agate.py` 延迟导入正式 `atrex-gateway-client`，沿用现有连接配置构造 `Client(base_url, auth=AkSkAuth(...))`；不记录凭据、不依赖 AKA 或 shell agate。
已核对的接口基准为 SDK 0.14.37：`build_eval_request_from_content(candidate, reference, gpu, name=..., options=..., idempotency_key=...)`、`submit_job("eval", request)`、`get_job(job_id, wait=True, timeout<=30, include_spec=True)`。
实际安装发行版本可以不同，记录验证版本并适配其正式接口；不硬编码旧 `/tmp` 源码路径、不 vendor SDK。host 不安装 torch/Triton/Bench/Ray，远端使用现成 Bench。
自足文件可传完整文本；多文件使用 `{archive, entry_point}` 与正式上传能力，`prepare_uploads(gpu, files, kind="eval")/upload_file`，保留预留 job ID 与不含签名 URL 的文件引用。
提交前保存固定 request、候选摘要、稳定 idempotency key 和动作；返回 job ID 立即落盘。上传预留与已提交区分，已知未提交可续原上传/请求，提交身份不明不能重新发一个任务。
已知 job 只查询原 job；超时/断连为 interrupted。提交回包丢失仅在服务保证同 key 取回原任务时重发同请求，否则继续定位，不换 key 或因短时 404 重投。
默认不取消；取消受理不等于终态。job succeeded 仍须逐 case passed、比较器确已执行、必要时延齐全；缺案例/时延与基础设施终态失败按实际 evaluation_failed 保留。
不隐藏终态自动重投。真实基础设施故障由后续排障明确解决，旧 request/job/终态留存，不将其当编译问题交 coder 改算法。
Implement 持原目录锁；resume 核对输入/候选摘要，先结清未完动作，再消耗新增且未消费的修复反馈，完成且无新反馈直接复用结果。

## 7. Workflow：三起点、有限收尾与恢复

运行目录仅 `state.json,result.json,design/,vNNN/implement/,vNNN/optimize/NN/`；每版一个 Implement 目录，每次 Optimize 一个新目录。
state 只存固定参数/输入摘要、版本/子结果位置、当前动作、已消费反馈与本次成功实测反馈的决定位置；持本运行锁，动作前/结果后原子写，不另建编排数据库或事件账本。
初始 Schema=v001；只有有效 revise 才登记下一版，新版接最近完整源码副本并重新完整评测。repair 不增版本，旧版用量不清零。
同版 Implement 总是 resume 原目录；Optimize 按唯一子运行路径求和，传本版剩余次数，恢复不重复加计数，未知结果不能按 calls_used=0 另开。
外部起点先 Optimize 时最多给 1 次，保留同版另 1 次读取本次反馈；外部决定不计作本次实测优化证据，坏稿照实失败而不偷借余量。

1. Task 起点 Design→Implement；Schema 起点直接 Implement；第三起点先 Optimize，原 kernel/反馈仅作 seed 与外部证据。
2. Implement complete 建立本次可用配对，并在有剩余次数时 Optimize 同一源码/反馈；design_gap 或有完整 kernel 的 implementation_failed 也可交真实反馈，未成源码的普通坏返回不能冒称 design_gap。
3. repair：原 Implement 存在就带该反馈 resume；第三起点尚无目录则以原 Schema/完整 seed/该反馈首建一次。累计 coder 用尽时 incomplete+limit。
4. revise：有版本余量就登记完整新 Schema 并 Implement，全案例重验；同内容稿拒绝，旧成绩不继承。
5. keep：本版已有本次成功实现可结束；已有本次真实失败则保留失败，不能因 keep 再开实现或改成成功。
6. 第三起点 keep 且尚无本次 Implement，仍首建本版唯一 Implement 并验证原 Schema/seed；成功后用预留 Optimize 调用读本次反馈，再按本节收尾。
7. needs_input/evaluation_failed/write_failed/input_changed 停在具体原因；interrupted 保留原动作，Implement 只原目录 resume。没有新终态反馈不再 Optimize。

**有限完成策略：**至少有本次已验证 kernel 和一次基于本次已完成全案例正确性/计时的 Implement 反馈的有效 Optimize 决定；外部反馈、单纯设计诊断或写出 Schema 不满足这条主链。
无版本余量时仍允许真实反馈决定：revise 且 required_fix=false 是明确的未实施可选提案，保留原有效配对，`complete+limit`；不伪造 keep、不声称该提案已实现或加速。
required_fix=true 的 repair/revise、最新真实失败/缺项或未结清运行不能这样收尾；分别 incomplete 或 interrupted，剩余具体问题保持可见。
Optimize 次数用尽时，只有本次主链已成立、当前已验证且无未解决必需问题才 complete+limit；没有有效本次反馈决定则 incomplete+limit，不重开目录洗次数。
`last_usable={schema,kernel,feedback}` 直接指固定 Implement inputs/attempt 文件，不指可被 resume 改写的 result；`latest_attempt` 指最新模块 result，两者可不同。
后验发现必需设计/代码缺陷（Optimize required_fix=true 或明确 design_gap）时保守清空 last_usable，保留原文件与原因；不自动挑历史候选替代。后续完整重验成功才建立新配对。
optional 新提案/新失败不把旧成绩绑定给新 Schema；最新真实失败仍 incomplete，即使保留未受否定的旧配对。未实施提案由 latest_attempt 定位，不建立 best 排行榜。
complete 只承诺有限流程的可用交付；性能目标是否达到据真实测量说明，无基线不造 speedup，keep 不是性能收益。
恢复复用已完成阶段；仅动作已登记且子输出未创建才启动原动作。有输出无可信终态先定位原进程/结果，不重放半份写作，不盲目重新提交远端任务。

## 8. 两轮三位首批 owner，之后按真实故障派新代理

主代理统筹，具体工作交全新 `gpt-6-astra / xhigh`；最多两位同时运行，共用 checkout 按写区交接，不新增 worktree。

| 轮次 | Owner / 写区 | 必须交接 |
| --- | --- | --- |
| 第一轮并行 A | Design/Optimize 全目录与对应根方法 | 两独立入口、真实阅读/收稿、必要 CPU 回归、实际参数/结果 |
| 第一轮并行 B | Implement 全目录、根 IMPLEMENT 方法、唯一 vector_add 示例 | 固定验收材料、自有 coder/SDK 闭环、必要回归和实际依赖 |
| 第二轮 C | A/B 停写后接整合权；Workflow、根配置/README、限定清理 | 同入口编排、删除后验证、首次简短 commit |
| 后续 D 起 | 新 xhigh 接真实运行及具体故障模块 | 修复、受影响验证、闭环证据和后续提交 |

A/B 不改根 README/pyproject/.gitignore 或对方代码；同模块、根文件和运行目录始终单一写入者。交接只列代码/结果、已做验证、剩余问题、未结清 job。
三位是首批分工，不是总人数上限；不得用产品 incomplete/interrupted 作为本次项目提前停止的理由。

## 9. 根配置、唯一例子与必要 CPU 检查

保留项目元数据与 Python>=3.11；删旧 project.scripts/包发现，`[tool.setuptools] packages=[]`，源码 checkout 入口，不承诺 wheel/SDK 产品。
保留 `PyYAML>=6.0,<7`、`jsonschema>=4,<5` 和 pytest extra；新增 implement extra 声明实际正式 SDK 依赖，记录实测版本，不强制获取旧精确 pin。
如实际借用第三方源码/文本，随该材料保留原版权/许可/NOTICE；SDK 未有再分发依据，不 vendor，不因重排另立项目许可证。
pytest 默认收集四新模块 `tools/tests`；根 README 改导航和实际命令，保留 `#2-schema-是什么` 与保护区仍引用的有效路径，删除旧产品/API/AKA 运行要求。
保留 `.schema_extract/`、`/.worktrees/` 与既有缓存/Wiki 下载 ignore，删除 qualification 专用项；新运行根 `/.schema_workflow/` 忽略，不提交捕获或凭据。
用现有合适 Python；若有旧 editable/AKA/PYTHONPATH 污染，用它建轻量 venv，不下载新 Python，不改 AKA/其他项目环境、不做环境矩阵。核对真实 import 来源即可。
唯一例子为 `schema_workflow/examples/vector_add/`：Task、固定 eval 包与 README；Implement 只读普通文件，不 import Workflow。B 在产品 coder 前固定验收并写清来源/比较器。
例子为连续 FP32 vector_add，合法域 `1<=N<=65536`；两设计域 `1..1024` / `513..65536`，重叠选前者；案例 `1,512,513,1024,1025,4097,65536`，数字 case ID，Schema 有真实合法 from。
reference 为 FP32 相加，输入与误差判据在模型前固定；seed/具体文件实现按正式 Bench 形态落定。产品不生成 oracle，不强造 helper；自然产生的依赖必须整包送测。
CPU 重点：来源/Contract 不漂移、真实 DOC/源码读取、坏稿留存、完整多文件送测、同设计修复、全案例/计时判断、已知缺口不能通过、源码反馈绑定、已知/未知 job 原位恢复。
Workflow 伪 CLI 覆盖三起点、外部 keep 后本次验证、无 kernel design_gap、repair 首建/续原目录、revise 重验、累计上限、可选末版提案完成/必需缺陷失败、不可变配对与完成不重跑。
另测关闭 settings 后 Qwen 连接字段仍传到子进程且不进产物；只做必要行为测试，不测目录布局、私有 helper 或提示措辞。
删除旧根后，实际解释器不能导入旧 `kernelschema`/`aka`，四入口从其他 cwd 可 help/使用材料；一次显式跑 Template `test_design_resolution.py` 与 Extract `test_standalone.py` 相关 CPU 回归。
首组通过即推进；只有修复、失败或未覆盖实际风险才补测，不对每模块重复整套 GPU 链。

## 10. 限定清理 → 首次 commit → 实跑 → 修复提交

C 先记录当前 Git 基线与用户已有改动，仅复核下述目标状态/活动引用和正式引用，不重扫全盘/历史捕获。保护区及 `schema_wiki/**/.extraction/` 完整保留。
正式 current/receipt 所引用的历史凭据同样保护，包括 `schema_discussion/archive/schema_wiki/extraction_records/` 中被引记录；根 `.schema_extract/` 不是这些正式凭据。
1. 整合新代码、prompt、示例与根说明，确认已不消费旧产品后，同批移除仅仓库根 `kernelschema/、tests/、docs/、examples/`，不匹配其他层级同名目录。
2. 删除旧运行区前，只保全下面单稿原字节到本目录 `historical_draft/root.yaml`，核对大小与 SHA；旁边 README 标“未验收历史稿”，写源路径、身份、复制日期与未发布事实。

```text
源：.schema_extract/v2/runs/a362b85a-dce6-4ad4-b38e-3a186dd87c1c/attempts/88f07fb8-23d7-4c16-a368-69399bfffcd2/workspace/root.yaml
目标：schema_discussion/standalone_modules_20261122/historical_draft/root.yaml
大小：13,490 字节
SHA-256：611786083f0da2b6b9aca4a789a28f05b7a7af419b55d779370667afad54b7dd
```

源变更/缺失或保全稿冲突时只暂停对应删除并查清，不覆盖、不扩成整套归档；不改 YAML、不发布 Wiki、不当新示例。
3. 复核两已登记 worktree 的普通/ignored 状态、HEAD 已并入保留分支、无锁/活动引用后，正常执行 `git worktree remove .worktrees/schema-workflows-v1` 和 `git worktree remove /tmp/ks-opt-20261103-ryxoz_iu/baseline`。
先移 Git 登记，不先删目录；不用默认 force/prune/解锁/杀进程，不删分支。仅已确认可丢的 ignored 缓存/qualification 残留可定点删后重试；新改动/未合并提交/未知内容暂停对应项。
登记移除后确认 `.worktrees/.tooling/` 无引用，再删旧 k1/k2 环境、缓存与 delivery/evidence 残留，最后只删空 `.worktrees/`；不删新增 worktree，不扩大到 `/tmp` 父目录或 SDK 参考副本。
4. 单稿已保全、无活动运行/锁/正式引用冲突后删除仅根 `.schema_extract/`，保留 ignore，Extract 以后可重建它。
5. 本文件吸收必需事实后，删本目录 `01_current_modules.md、02_independent_implement.md、03_end_to_end_workflows.md、04_cleanup_and_sequence.md、05_outline_and_decisions.md、07_design_optimize_plan.md、08_implement_plan.md、09_workflow_plan.md、10_execution_cleanup_plan.md、11_slim_writing_workflow.md、12_slim_implement.md、13_slim_execution.md、14_minimal_product_plan.md`。
保留 `FINAL_PLAN.md、FINAL_REVIEW.md、06_decisions.md、historical_draft/、EXECUTION.md` 与本目录 README；README 改保留件导航，06 仅修已删链接/阶段陈述、保留用户决定原意。其他 discussion/AGENTS/Git 历史及正式引用记录不在删除范围。
6. 完成上述必要 CPU 与删除后路径/import 检查，核对保护区无本次改动、worktree 登记正确，首次简短 commit（如 `Implement standalone schema workflows`）；仅提交授权变更，不夹用户原改动、认证/大捕获，不 push。
7. 新 xhigh 用新运行目录执行下列默认命令；目录若已存在换新的名字，不能借此清洗旧运行次数或未结清 job：

```sh
python3 schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --gpu L20D --out .schema_workflow/runs/vector_add-001
```

取得本次 **Design 选材/读 DOC→自有 Implement 完整入口全案例正确与真实计时→Optimize 实读该代码/反馈作决定**；自然 repair/revise 有余量时按产品规则继续，不能伪造 keep/revise 或旧成绩。
这条主链同时证明三个独立入口与 Workflow；缺必要分支证据只补最小调用，失败/中断其余路径用 CPU 证明，不故意制造 GPU 故障。未发生真实修复就说明修复仅经 CPU 验证。
`EXECUTION.md` 记录命令、实际 Python/SDK/model、简短故障/修复、检查结果及产物位置；保存固定 Schema/完整 kernel、原 request/job/response、逐 case 与实际读取记录，不再造审计账本。
问题由新 xhigh 查修，运行未知先结清原 job；SDK/连接沿正式发行和现有配置解决，不重问已确认模型/Agate、不重用已耗尽旧运行账户、不偷换 Task/判据/栈。
实跑返回失败不等于项目完成：修代码/配置后重跑受影响 CPU 与真实候选，必要独立新运行标明原因且保留原结果；最终修复和必要记录再次 commit，直到三个独立能力与 Workflow 主链实际成立。
仅授权内无法消除的外部缺项才能如实保留具体失败与续接位置，继续其他可做工作，不能用“已记录”代替可解决闭环或宣称全完成。
交付须同时具备真实已验证 kernel、本次真实反馈优化决定、必要分支证据、限定清理完成、保护区未改、最终代码/记录已提交；首次 commit 只是实跑起点。
