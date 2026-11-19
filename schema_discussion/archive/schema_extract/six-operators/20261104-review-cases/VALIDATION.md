# 2.2.5 Reviewer 验证记录

> 归档说明：2.2.5 时点的验证记录，样例、来源与复核说明已一并归入[本目录](README.md)。[归档索引](../../README.md)。

2026-11-04，当前文档审阅方法完成离线验证与真实模型验证。模型调用沿用已配置的 Codex provider/model，未指定模型覆盖，reviewer effort 为 xhigh。测试不包含 GPU 运行。

离线结果：42 项 Schema Extract 测试通过，148 项 schema/template 测试通过。覆盖当前文档输入隔离、四项批准结论、文件证据范围、上游规范错误重开及其去重、共享 Contract 展开、单文件检查、输出冲突保护，以及回归样例和评分器。

真实语义回归分两批执行，共 10 个不同样例，每例一个全新只读会话、一次调用、最多 240 秒；两批使用相同的冻结规则身份：

`bcd0148d35ef62a46d335e8608cb1a74d3b728eb1a4d2c52272a8bd18196614d`

| 样例 | 预期 | 实际 |
| --- | --- | --- |
| valid_copy | approve | approve |
| valid_custom_action | approve | approve |
| valid_optional_noop | approve | approve |
| valid_source_gap | approve | approve |
| history_definition | revise，定位历史定义依赖 | 符合 |
| missing_definition | revise，定位未定义对象 | 符合 |
| phantom_phase | revise，定位正文取消已声明提交 | 符合 |
| phantom_members | revise，定位正文取消已声明成员 | 符合 |
| lower_layer_override | revise，定位下层改写 Contract | 符合 |
| review_history_in_thesis | revise，定位无关审阅历史 | 符合 |

4 个合法样例均通过，6 个缺陷样例均退修并定位到相应字段和问题类别。合法 no-op 的审阅明确区分“动作没有数据效果”与“动作或事件不存在”；历史记录样例的审阅确认计算与结构本身完整，同时指出 thesis 中的批准、反馈和待办必须移出。这些结果表明本次样例中的自由表达得到保留。

## 真实 GDN 的隔离终审

另对已发布的 `gdn_prefill_bf16_state_bf16` 两节点及索引，使用生产 `build_prompt`、当前规范、固定原源码和正式 proposal/receipt 校验执行了一次隔离 P6。没有向 reviewer 提供前序批准或交接。

检查器仍返回 0。Reviewer 在约 284 秒后返回 `reopen`，要求重开 P3，并明确定位两类 `schema_conformance`：

- normalize 始终出现在 flow 中，正文却在关闭标志时取消提交、handoff 和完成依赖；源码确实不提交该 kernel，不能改写成 no-op 来掩盖。
- Qc/Kc/Perm 的固定 handoff 与 KQTMA、无排列时的生产/消费区域说明不一致，需要在所属层修正，不能由正文删除关系。

返回通过正式回执校验。输入副本及原发布文件的字节均保持不变；这是独立审阅验证，没有修改原 run 状态或批准，也没有完成其后续 P3 修订。

## 证据与边界

完整本地证据位于以下 Git 忽略目录，包含冻结输入、真实提示、CLI 事件、响应、结果及原始 usage：

- `.schema_extract/review-eval/20261104-reviewer-smoke-01/report.json`：2 例通过。
- `.schema_extract/review-eval/20261104-reviewer-suite-01/report.json`：其余 8 例通过。
- `.schema_extract/review-eval/20261104-real-gdn-01/review.json`：真实 GDN 的隔离终审与回执。

原始 GDN P6 调用的完整证据审计再次通过，历史回执仍可验证。两个语义回归批次的墙钟分别约 242 秒和 892 秒，真实 GDN 调用与第二批有重叠；这些时长不能相加当作总墙钟。用量保留原值，不汇总未经确认的 token 口径或费用。

本次验证支持 Reviewer 机制修复，不将历史产物的具体表达问题视为已经修复，也不保证未来所有模型调用都不漏检。代码、方法或模型发生实质变化时继续使用本回归集，并追加新的真实反例。
