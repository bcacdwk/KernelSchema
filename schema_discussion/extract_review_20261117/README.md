# Extract 审阅与精简

基线：`be4fc48`（extract dispatch refine），开始时工作区干净。

## 目标与范围

审阅当前 Extract 的流程、prompt、脚本和使用说明：先修实际缺口，再删重复或无用逻辑，最后独立确认精简没有削弱功能。前三轮重点为 `schema_extract/` 及入口说明；用户追加的第四轮核查覆盖 changelog 所有当前有效改动，包括 Docs、Template 工具/示例和 Extract，确认遗漏后统一补齐。

- 维持六阶段、有限完整设计、实际整区 `from`、独立 reviewer、当前 Template 规则及统一入口。
- 默认 Claude Code `qwen3.8-max[1m]`、两角色 `max`，保留显式 Codex 后端。
- 不恢复自建隔离，不增加兼容层、历史格式迁移、新框架或无依据的防御检查。
- schema_docs 的正式字段规则固定，只读。第四轮只补已确定要求在工具、现行示例与其他说明中的落实遗漏，不由历史 changelog 反向改字段规则。不改 Kernel，不重写 archive 和此前的审阅证据；明确延期事项单列状态，不冒充已完成。
- 不运行真实模型全流程、GPU、提取批次、queue 提醒或发布。以必要的离线小检查验证具体改动，不为文案或机械改名堆测试。
- 直接修复已确认的问题；没有证据的疑虑标为待核，不作为已发现缺陷。固定 prompt 仍放在 Markdown。
- 两位子代理并行时遵守文件所有权；跨界修改先由主代理协调。不得覆盖他人工作，不提交 Git commit。

## 轮次

| 轮次 | effort | 分工与目的 |
| --- | --- | --- |
| 1 | max × 2 | 阶段/批准/状态；调用/会话/恢复/进程 |
| 2 | max × 2 | 产物/来源/Template/发布；prompt/文字/实际材料对齐 |
| 3 | max × 2 | 跨模块复核、剩余接口与异常路径，修复具体遗漏 |
| 4 | max × 2 | 对照 changelog 与最终决定逐条核查全部当前有效改动：Docs、Template 工具/示例、作者入口及 Extract；排除已否决/被替代项，单列延期项，补齐确认遗漏 |
| 5 | high × 2 | 精简流程及产物代码；精简执行及控制代码 |
| 6 | high × 2 | 从新视角清理剩余重复、无用分支与说明 |
| 7 | high × 2 | 从实际使用价值主动删减无必要功能、代码、测试与防御性文字；以仓库目标判断，简化现有设计 |
| 8 | xhigh × 2 | 独立验收功能完整性和精简合理性，必要时第二轮收尾 |

每轮使用新代理。各报告只记录：已确认问题、实际改动、验证结果和未决项；说明精简保留的关键行为。最终汇总见 `CHANGELOG.md`。

## 进度

- 第一轮完成：修复写作次序反复阻塞、维护决定协议矛盾、未启动调用的会话误退役及启动失败处理；后端参数创建前校验。报告见 `R1_A_FLOW.md`、`R1_B_RUNTIME.md`。
- 第二轮完成：早期文本检查/提示等级、冻结指导与导航、standalone 代码身份，以及回执/阶段文字对齐。报告见 `R2_A_ARTIFACTS.md`、`R2_B_PROMPTS.md`。
- 第三轮完成：维护草稿/计划保持、派发前 pause/cancel、失败调用审计及后台启动早退。报告见 `R3_A_CONTROL.md`、`R3_B_INTEGRATION.md`。
- 第四轮完成：全部有效 changelog 已分别映射至 Docs/Template/案例及 Extract/作者入口；补齐旧字段引用与隔离说明残留，未改正式字段规则。报告见 `R4_A_CHANGELOG_TEMPLATE.md`、`R4_B_CHANGELOG_EXTRACT.md`。
- 第一轮 high 完成：删除无用状态投影/薄包装，合并相同批准动作与同一时点重复核验；必要边界保持。报告见 `R5_A_FLOW_PRUNE.md`、`R5_B_RUNTIME_PRUNE.md`。
- 第二轮 high 完成：删除无效参数链、未使用分支/导入、恒真审计开关、转发包装及无写入者状态候选；保留实际核验。报告见 `R6_A_ARTIFACTS_PRUNE.md`、`R6_B_CONTROLS_PRUNE.md`。
- 第三轮 high 完成：删除 `.gitkeep` 特殊接管、普通分页 reader、重复 handoff 文件、inspect 启动选项及独立提醒 probe 流程，统一到现有直接路径；删减相应状态、提示与测试。报告见 `R7_A_CORE_PRUNE.md`、`R7_B_OPERATIONS_PRUNE.md`。
- xhigh 独立验收完成并签收：补齐未执行调用的恢复交接回退、迟到提醒的审计身份及测试调用同步，两位各九项定向用例通过。报告见 `R8_A_FINAL_FLOW.md`、`R8_B_FINAL_RUNTIME.md`。
- 本轮完成，无已确认未修项；简版交付见 [CHANGELOG.md](CHANGELOG.md)。
- 用户追加第三轮 high：允许主动删除无独立价值的功能/测试/代码，强调简洁直白的表达和协作判断；并非只删死代码，正式字段规则仍固定。
- 精简前源码/提示/说明快照保存在 `/tmp/schema-extract-pre-prune-ego_8hhb`（95 文件，含 SHA256 清单），供最终复核比较，不作为新运行身份或产品模块。
- 当前轮次为四轮 MAX、三轮 high，最后 xhigh 验收；每轮两位新代理。
