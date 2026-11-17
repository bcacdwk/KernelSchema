# R4 A：流程、Template 与批准链集成收口

本轮局部实现与离线验证已完成，停止写入，交后续两位新 XHIGH 独立验收。没有修改 Template 字段/格式规则、Schema Wiki、旧 run 或个人配置，没有 Git commit、GPU 或正式提取。A 没有调用真实模型。

用户已取消耗时的真实 P1–P6 全流程试跑。B 已确认相关调用终止、会话退休、无存活或未知登记进程；两次试跑均没有阶段批准或发布。本报告不将离线整链或真实单次 CLI 能力验证称为真实全流程验收。

## P1 共享任务元数据出口

修复 `R3_A_metadata_boundary.txt` 中已复现的死结：已有非 root 后，P1 合法更正 `META_DATA.Name/Describe` 无法整组落地，而 P5 又无权改它们。

- P1 worker 的可写文件仍只有 `root.yaml`。`normalize_candidate` 先封存原稿、核原始文件权限、调用现有 Template 规范化；随后 Runner 在同一受控候选事务内将规范值同步到当前已有非 root。同步只涉及 `Name/Describe` 和逐文件精确 `Version + 1`，保护自身 Source、Dispatch、Design、Realization 及其他内容，P5 的身份保护不变。
- `shared_metadata_sync` 留在既有 validation，逐文件记录同步字段的前后值、原始/规范化/最终摘要、版本和受保护语义摘要；worker 回执仍锚定原稿。纯规范化、未改变共享值的文件不自动递增版本。缺 Describe 或错误 META_DATA 映射按候选拒绝处理，不抛裸 KeyError。
- P1 准备阶段将当前选定节点的已有最新稿作为只读成员纳入快照，包括已审稿、未合入完整稿、未完成 starter，以及引用已退休 provider 的待修 raw。只检查 P1 应完成的 root 来源；其他稿件已填写的非空 Source 仍须符合固定来源，不因尚未填写 Source 或 Design 就失去元数据同步出口。
- P1 reviewer 引用快照内每个 YAML，审最终共同前提。审后合入复制实际已审字节，不再生成或规范化；更新草稿指针时保留旧 checkpoint。未完成正文保持 worker，完整正文优先交 reviewer，P1 不授予 P3/P5 正文批准。
- 共享值变化后沿现有 `refresh_approvals/align_source_identity` 核实际使用上下文：需要复审的 P3/P5 保存原批准与旧身份，再形成当前批准；P2 选型与 P4 安排投影不变时保留其批准。没有另建批准系统，也没有将来源的新 SHA 直接替换进旧批准。
- 退休节点以后重新选入，准备新的 P5 待审副本时，用当前真实 P1 批准初始化共享元数据。版本按已有 raw 递增，`checkpoint.preparation` 绑定原 checkpoint、当前 P1 receipt/批准 checkpoint、root SHA 及同一同步记录；历史 raw 和批准不改，正文保留并由当前 P5 独立审。这里不重复回 P1 审已经批准的共享值。
- 退休 provider 的完整历史链保持原 checkpoint，上述同步不恢复其当前来源资格。consumer 仍须显式本地化或改用当前已审来源；历史材料只作为 baseline，不能借当前同名 provider 重解释原链。

相应局部接口：`validate_artifact_sources(..., completed_nodes=...)` 明确本阶段必须完成的节点；所有非空来源始终检查，P6 仍检查全量完成。Store 的 P1 快照/草稿指针和 P5 初始化都复用当前 checkpoint，没有旧字段别名或迁移入口。

## 原子写与最终产物门禁

与 B 的真实 Claude Edit/Write 验证对齐：worker 使用本次私有可写目录，非许可现存文件只读 bind；原生同目录临时文件完成 rename 后可正常采纳，reviewer 整体只读。

`artifact_manifest` 在读取、复制或散列原稿前，先完整检查直属条目：只接受普通单链接文件，拒绝目录（含空目录）、符号链接、硬链接、特殊文件及非法产物名。之后才读内容，并按阶段文件差异和 Template 规范化后的字段权限拒绝新增/删除/超范围文件或保护内容变更。已消失的原生临时文件不计为产物。

门禁已接 Runner 正常原稿、裁决选择、规范化、失败/中断草稿恢复、receipt 原稿、只读 helper、Store checkpoint 创建及复核。B 同步 audit 在归档原稿前调用它，并确认 Agent/recover_result 不在此前读取 workspace。反例使用禁止 `Path.read_bytes` 的观察器证明目录/符号链接/硬链接/残留临时文件先被拒绝；B 的同字节硬链接归档反例也通过。

裁决 select 采用调用前已封存的规范候选与权限证明，重核其基线、成功检查和实际字节；P1 派生同步不会被当成 worker 对非 root 的直接写权限。批准后不再调版本或重新规范化。实际 patch 继续走现有事务和独立一次签收，原 scope 预算不重置。

## 保留测试转换与提示

| 测试 | 当前承接行为 |
| --- | --- |
| `test_proposals` | P2 `{id, instance}`、P4 `{id, wait_for}`、真实中文目录/仅计划登记、P6 全文件与文档判断、nonce/原稿证据/审计摘要；不再依赖固定父表文本。 |
| `test_issue_evidence` | 格式/结构问题自己的文件锚点、来源问题不得借用该例外、回转固定来源证据、P4 独立 reviewer 不改安排。 |
| `test_review_protocol` | 六阶段角色的当前完整输入、选择与安排分开、空值含义、冻结全文、材料阅读顺序、限域维护/裁决/签收、独立 session 和稳定问题有界返回。 |
| `test_scoped_recovery`、`test_recovery_budget` | 当前选择/实例与实际来源授权、旁支在途 fence、pause/clock 控制、原预算与全部阶段的签收预留；原生失败真实性另由真实离线 current_recovery 覆盖。 |
| `test_source_gaps` | 当前字段与小写对象名、来源记录不掩盖缺项、逐输出比较基准/阈值、nullable IO 和严格 P1 完成检查。 |

只删除旧父链/确定性父表、continuation 和执行身份换绑的专属断言；现有真实来源图、当前维护/返回和冻结材料用例承接其仍有意义的不变量。没有 skip，也没有继承旧 TestCase 重跑整套。

根据 B 的真实工具误用，公共 `operations/check.md` 直接给出单独执行的 `/run/extract/check`，说明目录/阶段/只读与 skip-hardware 已定位、退出码从工具结果读取，不拼接 echo/分号/管道/重定向。动态材料提供唯一明确的 `/rules/EXTRACT_SCHEMA.md` 导航且实际只读挂载。没有放宽 shell 策略或增加模板系统。P5 的 first_draft 提示同时与当前 Version 对齐，元数据同步过的未完成稿不被误导去重置版本。

## 验证与完整集成状态

完整日志、命令和最终字节变化清单见 [R4_A_checks.txt](R4_A_checks.txt)。

- 一致副本 `/tmp/extract-r4-integrated-vuw3zx8t`：完整 Extract discover **125 项通过，574.513 秒**。包括双后端真实离线命名空间、原子写/只读保护、六阶段调度、多节点来源、签收/预算/回转、实际发布/重放/归档/恢复与监督。
- 最终副本 `/tmp/extract-r4-final-btrhx9io`：针对最后的提示、first_draft 和非法 P1 字段拒绝补验 **25 项通过，60.500 秒**；这些是已有用例的复验，不加到 125 的独立用例数。
- 少量相关 Template API 回归 **6 项通过，0.966 秒**：原输入 SHA 与规范化候选分离、类型/别名恢复、硬件占位、实际来源 identity、跨区环与完整目录登记。第一次使用错误测试 cwd 导致 `check_yaml` ImportError，切到 Template 自身目录后通过，未改 Template。
- 最终副本清单含 169 个仓库文件与 2 个模型阻断脚本；169 个仓库文件与当前工作区逐字节一致；所辖生产模块语法检查、`git diff --check` 通过。测试使用临时状态和显式离线 binary，进程 PATH 的 claude/codex 为拒绝脚本，未意外调用模型。

真实 proc 依赖的用例采用 B 已验证的显式系统 bootstrap，挂载私有 proc 后，在任何仓库 Python 开始前降回原 UID/GID 并设 NNP。产品不含 sudo，也未修改用户配置。

与 B 的隔离、原生结果、审计、交付及恢复接口已在同一套件集成通过。本轮没有尚未通过的适用离线检查。真实 Claude/Codex 工具读写、完整 schema、精确 resume/独立 reviewer，以及真实已知正反例 reviewer 的证据见 [R4_B.md](R4_B.md)；它们与真实阶段批准分开。确定性多节点 fixture 不证明真实代表的独特机制。

真实完整试跑按用户要求停止，**P1–P6 真实模型全链未验收**；终止证据见 [R4_B_stopped_trials.json](R4_B_stopped_trials.json)。后续仍需执行计划中的独立验收与必要修复，本轮不自签全系统最终完成。
