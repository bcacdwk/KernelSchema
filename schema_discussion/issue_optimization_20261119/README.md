# 六算子提取后的 issue 优化与产物复核

本轮工作已完成。上一轮 Git 变动涉及的 15 个正式 issue 均经过两名 max 审计、两名 xhigh 实现、两名 high 精简及一名 xhigh 删减复核；随后两名 xhigh 分别完成 FMHA、Sparse 的授权人工审修。

## Issue 结果

15 项全部按原稳定 ID 归档：**12 resolved、2 declined、1 partially_adopted**。具体决定见[最终处置表](ROUND4_REVIEW_AND_ARCHIVE.md)和[覆盖记录](ISSUE_COVERAGE.json)。

- 续接区分保存进度的真实性与稿件完成度，保留合法未完成稿、最新 P6 草稿、原批准和调用账；修复快照并发复用及审计标记准确性。
- 巡检按真实采样时间保留最新观察，提供锁等待反馈，用户时间统一为北京时间，内部 epoch、提醒身份和调度间隔保持不变。
- 补清非 Tensor 对象/元素层次、存储名称归属，并修正 MLA/GEMM 的相关实例。已有充分的回执、权限和诊断修法保留，重复实现与说明已删减。

续接的部分采纳只表示产品范围限定：支持同源码、同任务与同语义规则下的 extract 工具修复；不承诺一般源码/任务/语义迁移。两个 declined 保留“不必新增类型机制或重复 I/O 规则”的决定，相关实例问题已按现有规范处理。

## 当前 Schema

六项交付仍共 15 份设计，当前字节与凭据见 [SCHEMA_REVIEW_INDEX.json](SCHEMA_REVIEW_INDEX.json)，阅读入口见 [Schema Wiki](../../schema_wiki/README.md)。

| 算子 | 本轮结果 |
| --- | --- |
| FMHA | [专项审修完成](SCHEMA_AUDIT_FMHA.md)：修正 None 缩放参数与准备期缩放角色、workspace 写保护矛盾、有效位数、状态副本容量、split 适用条件和过强数值声明；三稿及目录检查通过。 |
| Sparse | [专项审修完成](SCHEMA_AUDIT_SPARSE.md)：修正寄存器保证、存储归属、双副本计数与跨 stage 暂存视图；四稿及目录检查、37处存储等式核对通过。 |
| MLA / GEMM | [限域修订与复核通过](ROUND4_REVIEW_AND_ARCHIVE.md)：分别澄清 tuple 表达与 thread/register 名称关联；正式 root 及完整目录检查通过。 |
| MoE / GDN | 保持原提取交付字节，本轮没有追加专项语义审修。 |

FMHA/Sparse 各由同一位 xhigh 承担本次 worker/reviewer 全流程。原提取回执、旧运行失败事实和 201 次登记调用保留；人工修订有独立的记录和当前 manifest，不产生新的 extract/P6 批准。FMHA 正式外部比较器本体及全动态域数值证明仍缺来源，Sparse 已知源控制区布局缺陷仍未修 kernel；正文和审修记录已限定这些事实。

## 验证

- 最终 extract 离线套件：**25 个模块、221 项全部通过**，见[汇总](FINAL_EXTRACT_TESTS.json)。
- 相关 Template：**76 项通过**；四个发生修订的算子均完成完整目录检查。
- 六项当前 manifest、六份原回执和八份历史状态摘要核对通过。框架代码与规则在最终测试及 Schema 审修期间保持稳定，见[最终核验](FINAL_VERIFICATION.json)。
- 未重新启动提取、巡检或 GPU 验证；未提交或推送。

实施依据与复核记录：[R1运行](ROUND1_RUNTIME.md)、[R1巡检与规范](ROUND1_MONITOR_SCHEMA.md)、[R2运行](ROUND2_RUNTIME.md)、[R2巡检与规范](ROUND2_MONITOR_SCHEMA.md)、[R3运行精简](ROUND3_RUNTIME.md)、[R3规范精简](ROUND3_MONITOR_SCHEMA.md)、[R4复核归档](ROUND4_REVIEW_AND_ARCHIVE.md)。
