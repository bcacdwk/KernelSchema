# Extract 与 Schema 系统审计：已完成

**最终入口：[FINAL_PLAN.md](FINAL_PLAN.md)。审计及方案已完成，产品修复全部待实施。** 本次仅更新本 discussion；没有修改 Extract/Template、规范、正式 YAML/dispatch/receipt、旧运行态、kernel 或 issue 注册表。十二项已交付、monitor 已停的状态保留。

最终推荐四包：**修正采用对象与提交事实 → 定点更正 Schema、规范与当前字节清单 → 贯通回执义务、诊断与原角色补正 → 修正观察与监控代际**。后面三包可独立验收。没有必须用户决定的技术或语义分叉，本轮没有将方案自行扩展为产品实施。

- [FINAL_PLAN.md](FINAL_PLAN.md)：已有修正的评价、确认缺陷、唯一推荐修法、文件/顺序/必要验收、发布与迁移边界；已做检查与未来测试分开。
- [COVERAGE.json](COVERAGE.json)：15 个 X、11 个 S、24 候选/104 次出现、ISSUE_RECAP 全部类别及既有稳定修正的最终去向。
- [FINAL_REVIEW.md](FINAL_REVIEW.md)：high 的 16 组删减裁定、两条验收补充与最终风险/决策状态。
- [FINAL_VALIDATION.json](FINAL_VALIDATION.json)：319 个保护文件 SHA、前轮依据/旁支保全、JSON/覆盖/链接的最终轻量验证。

## 已完成的流程与依据

全流程按用户指定为 **2+2 max → 1 high → 1 xhigh**；每轮使用新的代理，主代理统览进度。前轮报告保留原位，后轮结论收敛到最终入口。

| 轮次 | 新代理 | 完成内容 | 原始产物 |
| --- | --- | --- | --- |
| 1 | 2 个 max | Extract 机制与 Schema 内容分工审计，复审既有修正 | [EXTRACT_AUDIT.md](round1/EXTRACT_AUDIT.md)、[SCHEMA_AUDIT.md](round1/SCHEMA_AUDIT.md)及 FINDINGS |
| 2 | 2 个 max | 独立反证、补齐真实采用/提交和来源契约证据 | [CORRECTNESS_REVIEW.md](round2/CORRECTNESS_REVIEW.md)、[SCHEMA_AND_CONTRACT_REVIEW.md](round2/SCHEMA_AND_CONTRACT_REVIEW.md)及 DECISIONS/evidence |
| 3 | 1 个 high | 将完整方案收为四包，主动删减重复机制和可选工作 | [SIMPLIFIED_PLAN.md](round3/SIMPLIFIED_PLAN.md)、[TRIM_DECISIONS.json](round3/TRIM_DECISIONS.json) |
| 4 | 1 个 xhigh | 复核删减、补齐验收对象闭环，完成最终方案与覆盖索引 | 本目录 FINAL_PLAN / COVERAGE / FINAL_REVIEW / FINAL_VALIDATION |

事实起点是十二项的 [ISSUE_RECAP](../extract_twelve_20261119/ISSUE_RECAP.md)、[FINAL_REPORT](../extract_twelve_20261119/FINAL_REPORT.md)、[DELIVERY_INDEX](../extract_twelve_20261119/DELIVERY_INDEX.json)。基线 HEAD 为 `484902855fe7be82a74fe7c394503d1d7d222dbb`，详见 [BASELINE.json](BASELINE.json)。此前已有的 AGENTS.md、十二项产物和 workflow_architecture_20261120 等工作区改动保留；旁支讨论不是本轮已批准设计。

## 范围与证明边界

本轮只管 Extract/schema 的工具、协议、提示、规范、Template 与正文表达。kernel_wiki 自身缺陷、性能问题和来源资料补齐明确排除；Schema 仍须准确保留风险、例外和缺口，不能虚构源码保证或通过缩小任务域掩盖它们。

没有启动提取、真实模型/GPU、恢复、monitor 或发布，没有提交或 push。固定来源、角色独立、真实引用、权限、预算与最新用户控制继续有效。旧批准只证明原字节；未来正式内容更正必须有 Version、完整目录检查、有限语义复核和 current 清单，不能改签历史或伪造新 P6。

既有测试与离线反例的范围在最终计划中列明；它们不证明四包已实施、不证明 33 稿全域正确，也不证明真实模型返工率已下降。
