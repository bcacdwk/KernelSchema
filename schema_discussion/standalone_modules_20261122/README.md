# 独立功能重构：保留记录

2026-11-22。用户已确认四个独立功能目录、自有 Implement 与 Agate、同 CLI Workflow、删除旧根产品和限定历史清理，并授权正式实现、commit 后真实 Qwen/Agate 持续验证。

max 调研/具体计划、high 精简与最终 xhigh 检查已完成。过程稿已由最终计划吸收后删除；本目录只保留最终依据、用户决定、实际执行记录和唯一未验收历史稿。真实闭环是否成立只按执行记录中的实际产物判断。

- [最终实施计划](FINAL_PLAN.md)：功能、入口、恢复、有限收尾与清理边界。
- [最终审阅](FINAL_REVIEW.md)：裁剪与必要补回的判断。
- [用户决定](06_decisions.md)：已确认范围与持续实施授权。
- [实施与真实运行记录](EXECUTION.md)：代码、CPU、清理、提交及后续 Qwen/Agate 事实。
- 真实修复：[来源材料](ISSUE_01.md)、[native 交稿](ISSUE_02.md)、[环境参考](ISSUE_03.md)、[阅读证据与等待时间](ISSUE_04.md)、[语言绑定与资料交接](ISSUE_05.md)、[服务harness与拒绝请求恢复](ISSUE_06.md)。
- [唯一未验收历史稿](historical_draft/README.md)：原字节、身份、SHA 与未发布边界。

实施起点为 `main@9f95eb30122678fab855f6855a322a8f7291dd7f`；固定示例材料先提交到 `2c9c346cdd0373821b633012bdfece446c766d65`。不修改 Extract、规范、Template、两 Wiki 或正式发布凭据。其他旧 discussion 保持历史身份，尤其正式 current/receipt 引用的记录不在本次删除范围。
