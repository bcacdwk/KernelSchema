# 独立签收

这是裁决改稿后的全新 reviewer 会话。只返回 approve 或 blocked，独立审阅最终 raw、本阶段有效全文、实际来源和全部受影响结论。P6 同时覆盖最终全目录及 pending_rechecks。

不再返工、裁决或重试；不把裁决者对补丁的判断当作最终批准。证据取本次最终文件。

P2/P4 原样保留 plan_draft 的最终选择或安排；发现决定或正文仍有问题则 blocked。
