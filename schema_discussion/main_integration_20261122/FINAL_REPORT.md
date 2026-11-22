# main 整合与修订交付

授权的两轮 A/B、C/D 实施和独立验证已完成，产品及已审公开材料可由根代理精准提交并正常推送 main。原审计 FINAL_PLAN 的“待实施”仍保留其历史含义；本报告及 [IMPLEMENTATION_COVERAGE](IMPLEMENTATION_COVERAGE.json) 给出实际实施结果。最终 Git/远端 SHA 由根的交付记录绑定，不在自身提交中虚构自己的 commit SHA。

原十二项 57 文件已在 `41b7db0dd636f01a1d8bc490bc159c2d65b7b034` 保留批准字节；`6c3859e58cbc25e4094590c5e50f39ffcdb773a3` 正常合入 workflow 的 37 个自然提交。W1 `1a221e4` 修数学身份、P2 候选与控制提交事实；W3 `d444101` 统一 handoff/证据义务、安全完整诊断和普通补正的既有预算；W4 `f46b14a` 区分缓存/样本、当前/历史观察与 clock 代际。W2 九 YAML 各增一版、两 dispatch 更正来源边界，加入有限规则澄清与九 current；所有产品和覆盖文件由包含它们的 Git 提交绑定。

| 验证对象 | 本轮实际结果 | 证据 |
| --- | --- | --- |
| A：W1/W3 最终共同树 | 186 个不同 CPU 测试通过 | [ROUND1_A](ROUND1_A.md) |
| B：W1/W4 最终共同树 | 46 个 CPU 测试通过 | [ROUND1_B](ROUND1_B.md) |
| C：W2 | 64 项 Template 测试、九完整目录 checker/dispatch 通过，19 YAML raw=preview | [ROUND2_C](ROUND2_C.md) |
| D：有限源复核 | 九更正 YAML、两 dispatch、九 Tolerance/十继承消费者及真实 from 通过 | [W2_REVIEW](W2_REVIEW.json) |
| D：共享 workflow | 187 个不同用例闭环；修复三条旧文档 pin 后 69 例回归通过，另一个 strict CLI 夹具显式选 strict 后单例通过；117 例不受影响结果复用 | [workflow 验证](round2_d/workflow_verification.json) |
| D：current/bridge | 九完整清单与索引一致，27 个实际目录 provenance 断言通过 | [终检](round2_d/current_verification.json) |

[Schema Wiki](../../schema_wiki/README.md) 保留原六项/15 YAML，并加入十二项/33 YAML，共 **18 项/48 YAML**。本批九 current 覆盖完整 19 YAML/9 dispatch，明确 `runner_p6_receipt:false`；另外三目录的 14 YAML/3 dispatch 仍匹配原 receipt。12 原 receipt、原 DELIVERY_INDEX 和固定 Source 未改，新清单不重签旧 P6。

共享接口没有新增门禁。默认 design/optimize 仍 write_only，strict 冻结实际 Template/规范并拒旧规则漂移；Extract bridge 透传原 CLI；标准 implement 保持原 AKA 委托，iterate 保持按版本计轮、keep 停止、失败反馈与未结算/异常停止及不重放。唯一额外产品兼容变更是经根划归的 row_softmax 公共材料清单三条文档 size/SHA 更新，保留 driver 身份检查。

695 个旧原件原位保全；277 个允许保护路径只有授权变化；12 state、supervisor、原 receipt 和用户 AGENTS 摘要相符。42 archive 元信息排除后未访问目标。历史公开仅五个旧候选和精确必要新证据，未公开 690 个旧原件、账户或 raw/native/思维捕获。[HISTORY_PUBLICATION](HISTORY_PUBLICATION.json) 给逐项去向与 allowlist；[公开历史说明](PUBLIC_HISTORY.md)明确 clone 不含旧运行账/冻结 run、未公开历史、原 /tmp 环境/fixture，SHA 不能代替缺失内容。

没有真实模型、GPU、Extract、monitor 调用、AKA 改动或新的 P6。masked 的 4616 例是旧索引证据复用，未重算。结构与有限来源检查不证明全部 33 稿、全域数值/性能、编译器寄存器分配或真实模型返工成本；比较器缺口、GDN 同步风险、mixed Sparse 哈希未确认和性能例外仍如实保留。批准计划的保留项和明确排除项均已收口，无额外执行轮或隐藏必做待办。
