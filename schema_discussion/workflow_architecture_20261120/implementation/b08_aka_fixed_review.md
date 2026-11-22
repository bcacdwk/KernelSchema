# B08 — AKA fixed 独立审阅与 A7.fixed 发行工程

2026-11-20 UTC。本报告仅覆盖 AKA 固定 backend 的独立审查、定点修复、原生运输探针和发行工程，不签 A7.fixed 真实业务完成。

## 源码身份与范围

- AKA 工作树 `/home/shaohanyong.shy/atrex-kernel-agent`，分支 `feat/schema-workflows-v1`，review base/HEAD 始终为 `4be322b0635a2c5023077eb6b1308752d9b61996`。原 upstream 为 `alibaba/atrex-kernel-agent`、基线 `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`；发行 fork 为 `bcacdwk/atrex-kernel-agent`。
- KS 子模块保持 detached `b1049e19403238125d4210dd275a8f3e2874ba47`，未消费另一代理的活跃源码。实际 `verify_release()` 的 manifest 为 `sha256:d108ae06e45fb405eb9ba6c900404550b454958d53b628ad8a90e01df16e5da1`。
- 读取了适用 AGENTS、现行 C/V、A1–A5 公开文档、B07 报告及实际原 native/controller/account/evaluator 代码。未访问、修改或运行 EXP，未派生代理，未修改私有计划、AGENTS、分支或全局配置，未提交/推送。
- 仅修改 AKA 源码、对应 CPU 测试和公开说明；此文件是唯一写入原 KS main 的报告。KS 公共问题仅以具体源码与证据交给主代理协调。
- 使用既有 Python 3.13.12 环境 `/tmp/schema-b05-a3-venv`，没有新解释器、服务、GPU 镜像或凭据搜索。

## 独立发现并修复的问题

| 问题 | 复现与修复 |
| --- | --- |
| 收尾恢复永久跳过原付费操作 | 原 controller 在 production/native `_finish` 前已置 `phase=done`。在 finish 的 `native_invocation_retained` 边界中断，重启直接 `publish`，`finished=false` 且原操作永远不吸收。新增实际 production 反例首先失败；现使用 durable `finishing`、冻结收尾前结果、恢复原 finish 后才转 done。未结算 fence 不得标 finish 完成。 |
| production 恢复同 C 的 Ref 路径漂移 | 修复前一问题后反例继续失败：重新 `publish_native_source` 得到同摘要但新路径的 Ref，`tx.plan` 拒绝“original operation key changed subject”，原 native production wrapper 又把它转成 violation。现核对源码摘要并沿用该操作原 Ref；原角色输入、会话与费用不变。反例最终通过，收尾回执仅吸收一次、模型不重调。 |
| 派生观察文件写盘失败丢失完整 native 回执 | `NativeSession` 原先先写 `public_output.record_invocation`，失败时 optional owner 尚未拿到已完成的 stdout/stderr/usage。新增低层原 NativeSession 反例先失败。现先调用原 optional complete-receipt hook，随后写派生观察；默认无 hook 的原路径结果不变。 |
| 具体 Agate 服务缺 K6 选择时构造崩溃 | `Services` 原来无条件读取 `profile.parameters.evaluation`。新 native capture 可以合法完成而 K6 尚未配置，不能因此抛 KeyError。现明确保留未选 channel，准入返回缺原 K6 full-Task/六作业资格的 unknown/blocked，不伪造 CPU/GPU 通过。 |
| 真实完整 Demo Materials 无法进入 AKA | 四包安装 smoke 使用实际 K6 Demo 的 fixed request，`SchemaProblemProvider` 将整个 evaluator 依赖闭包视为私有，误拒明确授予 coder 的 `task_environment`、`task_timing`。新增两项反例先失败；现仅允许 Task.environment/evaluator.timing 的原公开描述字节，任何 evaluator.files/entrypoint/inputs 仍优先禁止（包括别名）。snapshot/私有输入与两项公开描述共 5 项最终检查通过。 |
| 资格流程仅有消费器，没有真实捕获操作 | 新增有限 `aka.workflow.schema.native_probe`，复用原 `RoleCalls`、`NativeSession` 和同一 `OwnerAccount`；Transactions 仅增加私有 namespace 参数，不增加余额、时钟或另一个 controller。coder/reviewer 至多各一次，失败停止，continuation 不替换已启动操作。保存原 input、exec-intent、invocation 与账户 receipts；校验实际 no-tools init、模型/session、RoleOutput、完整 token usage 与原进程终止。 |

原 native 资格消费同时加强为选择原 input/exec/invocation 三类 Ref，核其 role、input hash、assignment/request/source、session/operation；冻结原 Agent/provider/process/runtime/Claude normalizer 等实现身份。注入的 CPU transport 明确 `synthetic=true`，不可经 `select_request` 选成真实资格。`select_request` 从实际捕获生成新的 immutable request/selection 与 host-only Materials，不要求用户手写“pass JSON”，不复制 live owner 或更改 admitted request。

## 原控制流与三条 audit 核对

- 原 `aka workflow`、`aka optimize` 的 `--schema/--schema-config` 使用同一个 `FixedController`；KS 显式 `backend=aka` 将原 `RunContext` 直接交给 `Backend.execute`，沿用原 OwnerAccount 和 native campaign writer lease，没有嵌套 `api.implement`。
- default/off 仍不导入 Schema/KS，不访问其 checkout、不联网。独立从 upstream `bf5557a` 的 Git 对象导出原 `src`，使用同一个原 Campaign CPU 探针对照本轮 current-off，normalized prompt、源、limits/outcomes、exports、setup/finish 完全一致。
- V0 仍是原 native setup/measure/CommandEvaluator 的 Task baseline；C0 在 V0 后由 A4 安装。跨 e 使用准确未晋升终态 C，G/HEAD 独立。保留原 admitted draft，不从恢复重新盖 seed；verify_only 无 coder/改码 episode。
- 普通 seal 后/ABBA 前 durable qualification、未提交恢复保留同一绑定、已提交恢复从 commit-bound audit 吸收原资格/usage 三条路径均保留。已提交 audit 缺资格仍保留 native history 并停止，不补问模型改历史。
- 原 Task 评测、独立 fidelity、native production、ABBA、finish 都有独立原操作。Same-allocation ABBA 只在明确 synthetic 服务中验证原 schedule/score/promotion；具体 Agate 不选择尚未资格的 comparison，不把不同 allocation timing 拼成 ABBA。
- 原 owner 承担 parent/request/native 最紧限额、finish floor、group、check_active、原绝对 timeout 与累计 receipt。未知 token 保持 null；原调用终止不代签 sibling 或 GPU cleanup。core-anchor 到 budget binding 的初始化缺口继续保守失败，不凭缺文件造账或自动新 run。

## CPU / CLI 验证

源代码冻结后运行一次完整适用 `tests`；最后安装 smoke 揭示 Task metadata 边界后仅修该处并补受影响验证，没有重复整库。完整结果与安装 smoke 在报告最终更新处记录。

| 检查 | 结果与证据 |
| --- | --- |
| 收尾反例修复前 | `/tmp/schema-b08-aka-before.log`：1 failed，17.83s。 |
| 回执先于派生观察反例修复前 | `/tmp/schema-b08-native-before.log`：1 failed，3.22s。 |
| 第一轮修复增量 | `/tmp/schema-b08-aka-fixes.log`：1 failed / 1 passed，19.92s；额外失败揭示 production Ref 漂移，原记录保留。 |
| 收尾最终定点 | `/tmp/schema-b08-finish-fixed.log`：1 passed，18.09s。 |
| 新 native probe CPU | `/tmp/schema-b08-probe-reviewed.log`：3 passed，16.43s；有限两调用、原回执丢返回恢复、未知 usage 不签资格、synthetic 不可选择。此前两项失败仅为 usage projection 新 Ref 导致 replay 不等；改用原 operations revision 的稳定 usage publication 后通过。 |
| 唯一完整 A1–A5 CPU 轮 | `/tmp/schema-b08-all-cpu.log`：**119 passed，714.16s，0 skipped**；发生在最后 Task public metadata 修复之前。 |
| 完整 Demo 安装烟测最初失败 | `/tmp/schema-b08-installed-smoke.log`：原 `aka workflow` 在 model/评测前报 `Materials cannot expose private evaluator bytes`。失败目录完整保留，没有用缺 budget binding 重建原 run；后续是明确新 request_id 的零调用安装测试。 |
| public metadata 反例与增量 | `/tmp/schema-b08-public-descriptors-before.log`：2 failed，2.81s；`...-fixed.log`：**5 passed，2.93s**，包括公开 environment/timing、private entrypoint/inputs 同字节别名、snapshot。 |
| upstream/current off 对照 | `/tmp/schema-b08-off-comparison.json`：equal=true；原始 upstream trace 与本轮 full suite 产生的 off trace 一致。 |
| wheel/source 原字节一致 | `/tmp/schema-b08-source-artifacts.json`：全部四包的 `aka/*` 与当前源码逐文件对照无差异；wheel 不含 KS、private runtime 或 qualification capture。 |

本轮实际原 CLI 两个 command 均有 fresh-process 测试；测试中的 model/GPU/ABBA 传输明确 synthetic，不能充当真实资格。新 `tests/test_schema_native_probe.py` 被仓库现有 `tests` ignore 规则覆盖，文件已创建且纳入本轮 pytest；主代理提交时需显式 `git add -f`，本代理未改变 index。

## 真实 MODEL：一次 coder 失败，零 GPU

本批实际新增 **1 次 AKA coder invoke，0 次 reviewer，0 GPU/Agate 作业**。未使用剩余授权硬试；主代理明确要求保留失败并先诊断输入。

- 使用现存官方 Claude `2.1.316`，executable SHA `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824` 与固定 public profile 相同，请求/允许报告模型 `qwen3.8-max[1m]` / `qwen3.8-max`，effort xhigh、timeout_ms=120000、native USD threshold=`0.50`。
- 从固定 KS Demo 的真实 full fixed request 准备独立 probe child，绑定其**原父账户** `/tmp/schema-b08-native-real/prepared/.runtime/owner.json`。此 child 不运行 fixed业务controller、不评测、不安装 candidate。
- 原 `NativeSession` 到 120 秒配置超时后终止进程；`exit_status=-9`、`timed_out=true`。完整原 stdout 265,629 bytes、stderr 82 bytes 及 NativeInvocation 已保留。实际 init 为 tools=[]、mcp_servers=[]、dontAsk、匹配 model/session；没有完整 result，唯一验收错误 `native_terminal_unavailable`。
- 原 exec-intent 的 PID/boot/start_ticks 与 operation/session 已在费用前保留；终后复核 `process_alive=false`。原账户为 coder=1、reviewer=0、evaluator_jobs=0、active_ms=124369（包含准备/记账，非声称模型恰耗 124369ms），unsettled=[]。
- **tokens=null，CLI reported USD=null，独立账单未验证。** 无完整终态统计不能记为 0，也不能从 partial stdout 猜费用。实际原生阈值并非宿主保证的硬美元上限。
- qualification=null、status=blocked、recovery=none；`--continue` 实際重放结果完全一致、无新模型操作。未签 AKA transport qualification、Task/fidelity/GPU/ABBA 或 A7.fixed。

原证据根 `/tmp/schema-b08-native-real`；命令日志 `/tmp/schema-b08-native-real-capture.log`、`...-replay.log`；仅选定事实摘要 `/tmp/schema-b08-native-observation-fixed.json`。失败前的零调用准备 import 失败及一次本地诊断脚本语法失败均保留日志，没有额外模型/GPU调用。不要把整个 private capture 根放入公开源码。

## 1 MB 角色输入诊断（已交主代理/KS owner）

原 input 保持不变，生成的清单仅含 ID、原 Ref 路径、类别与大小，位于 `/tmp/schema-b08-native-input-sizes.json`、`/tmp/schema-b08-native-input-summary.json`。

| 组成 | UTF-8 正文字节 |
| --- | ---: |
| Rules | 687,103 |
| role Materials | 276,673 |
| Task/Schema assignment | 14,474 |
| current Source | 377 |
| probe 说明 | 358 |
| 全部正文 | 978,985 |
| 实际 JSON input（含编码/元数据） | 1,012,929 |

Rules 中有 **29 个 Template checker Python 实现，409,961 bytes**：包括 `normalization.py` 54,557，`yaml_format.py` 35,993，`yaml_source.py` 35,581，`rules/symbols.py` 29,882 等。另有 **11 组完全重复正文，额外 254,428 bytes**：Task Contract/acceptance、implement method、7 篇 schema_docs 与 template_guide，分别作为 Rules/assignment 和 Materials 重复投递。

固定 b104 的 `runtime/claude.py:prepare_role_input` 直接投递所有 Rules.files；`writing.py:163–164` 又要求 Template 全部实现位于 Rules.files。权限/版本验证所需 checker 字节应保留在原闭包，可由 KS owner 使用明确 host-only/tool source 或角色投递选择，不能直接截断必要 Contract/Schema、也不能按后缀猜测任意规则是否可省。仅移出 checker 实现正文并去精确重复的理论正文为 314,596 bytes；这是输入构成诊断，不是新 profile/新输入已经验证的声明。AKA 未跨仓修复、未回写原失败资格；新输入若再模型验证需另有明确有界决定。

## A7.fixed 发行门与 A6 接口

新增公开 `docs/schema-source-release.md` 给出固定 upstream/fork/KS 配对、最小安装、native_probe/continue、自动选择原资格 Ref、两条业务入口、原父账户和 scope 边界。AKA 继续使用原 LICENSE/NOTICE，不给 KS/第三方重新授权。

四个 owning distributions 已用原 `tools/build_distributions.py` ownership 规则本地 `build --no-isolation` 执行 sdist→wheel。最初工件与索引 `/tmp/schema-b08-aka-dist`、`/tmp/schema-b08-source-artifacts.json` 保留；最后仅重建改变的 Schema 包，最终工件 `/tmp/schema-b08-aka-dist-final` 与完整索引 `/tmp/schema-b08-evidence.json`。不把 package version 0.2.0 当最终 source commit；本批未提交，最终固定发行 SHA 由主代理审查后记录。

**门状态：** A5.interface/CPU 审阅工程已完成适用修复；A7.fixed 的包/来源/文档工程可交接，但真实 native coder/reviewer、实际 GPU full Task、current-R独立fidelity、production交付与真实同 allocation comparison 尚未齐，A7.fixed 不签 completed。Agate 当前配置/auth未在本批验证，不据此宣称服务没有GPU。首发仅 Linux/source-checkout、row-softmax full_task_only、single_file/kernel.py、Triton/never/no conversion；Python3.11、F1/多文件与 planner 不扩范围。

A6 必须继承原 `Backend.execute(context)` / `FixedController.execute()` 与公共 result；保留原 writer lease、owner authority、request/profile/implementation 身份。旧事务结算/关闭后才构造新不可变 R/mode 请求，普通 build/numerical/fidelity fail 或 unknown 留在同 R；资格不等于 native promotion，C 不等于 G。`finishing` 是付费恢复阶段，不能把结果已合格当作收尾已结算。原 audit 缺资格和 bootstrap binding缺失继续 fail-closed。A6 不可使用本次超时的运输记录签 qualification。

## 最终验证更新

完成：**119 项全 CPU + 最后 5 项 Task 边界增量**；四包实际安装后从 `/tmp` 使用 `python -I` 执行 **5 个命令**：`aka workflow --schema`、`aka optimize --schema`、KS 显式 backend=aka、KS continue、native_probe 原失败 continue。三条业务路径均真实进入同一 fixed backend 并发布规范 blocked/zero-fee result，使用原父账户；probe 仍是原 1 次 coder/unknown usage，没有新调用。

原命令/输出/结果：`/tmp/schema-b08-installed-smoke-final/commands.json`、`summary.json` 及 `/tmp/schema-b08-installed-smoke-final.log`。Schema 实际从 venv `site-packages/aka/workflow/schema/backend.py` 导入，KS import/resource root 为固定 AKA 子模块。完成 wheel 检查后仅将 Schema 恢复至原 editable 开发安装，Framework/Legacy/Light 保留本轮同源 wheel；恢复日志 `/tmp/schema-b08-schema-editable-restore.log`。

| 最终 wheel | SHA-256 |
| --- | --- |
| Framework | `9354664f3820e0b721fe002acf62b20a6c8ee810a12ca7f0afaaa064a587b40d` |
| Legacy | `a967c30761e2616fec64eb10126d02f8b5a40a544ee2d1dc3bbc6128ff739ce8` |
| Light | `c0cb37a7a0bc7424cdad1d711775ca81c33dfae6dbe59f72eb172cd2a3571b6b` |
| Schema（含最后 Task 修复） | `2a16b768b925829d28a9c2e2e3fa3f5795ea7835cc0dab98aa9574a21177cbd7` |

所有 wheel 的 AKA payload 与最终当前源码逐文件相同，均有原 LICENSE/NOTICE，不打包 KS 或 private capture。最终 14 个 AKA 源码/测试/文档文件及全部证据摘要在 `/tmp/schema-b08-evidence.json`；AST、本地文档链接、`git diff --check` 通过，HEAD/分支与 KS gitlink 未改。

另有一项 **KS 共享权限边界诊断已精确交主代理/runtime owner**：b104 的 `materials.validate_request` 和 `runtime.prepare_role_input` 仅检查 evaluator.files，单独把 evaluator.inputs Ref 作为 coder Materials 可进入 role payload（18,977 bytes）。零模型/零GPU复现 `/tmp/schema-b08-ks-input-alias-diagnostic.json` 只输出名称/大小。AKA 业务 A2 已阻断该类 inputs/别名；此批未改 KS 共用 runtime，未来固定升级需由其 owner 确认并修该边界，不能把公开 environment/timing 描述与实际 inputs 混为一类。该诊断没有给模型发送私有输入，未改变本次原捕获。
