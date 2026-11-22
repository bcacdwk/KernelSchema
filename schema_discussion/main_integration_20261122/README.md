# KernelSchema main 整合与提取修订

两轮 A/B、C/D 的授权实施及独立验证已完成，当前为 **implementation_complete_ready_for_main_delivery**。根代理按 D 已审精确清单完成 Git 提交、正常推送与远端 SHA 核验。原批准字节、旧审计计划与原 receipt 保持历史原义。

- [最终实施报告](FINAL_REPORT.md)：实际变化、测试结果及限制。
- [实施覆盖](IMPLEMENTATION_COVERAGE.json)：26 发现、24 候选/104 出现、20 recap、11 既有修正、16 删减与 FC01/02。
- [公开历史说明](PUBLIC_HISTORY.md)及[公开 allowlist](HISTORY_PUBLICATION.json)：695 原件逐项保全、五旧候选及必要新证据，clone 的缺失材料边界。
- [W2 冻结对象](W2_SUBJECT.json)、[独立有限复核](W2_REVIEW.json)、[本批九 current 索引](../extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json)。
- [第一轮屏障](W1_GATE.json)、[第一轮收口](ROUND1_GATE.json)、[A](ROUND1_A.md)、[B](ROUND1_B.md)、[C](ROUND2_C.md)、[D](ROUND2_D.md)。
- [仲裁计划](ARBITRATION_PLAN.md)与[机器执行映射](execution_map.json)保留原计划快照；实际状态看[progress](progress.json)，唯一表外精确归属见[OWNERSHIP_CHANGES](OWNERSHIP_CHANGES.json)。

原 57 文件基线为 `41b7db0`，workflow 正常合并为 `6c3859e`，保留 37 个自然提交；W1/W3/W4 为 `1a221e4`、`d444101`、`f46b14a`。仅根负责 Git 写动作；AGENTS 用户改动、旧运行态与未公开捕获保留本地。没有真实模型/GPU/Extract/monitor 或 AKA 修改。
