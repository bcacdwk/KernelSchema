# ROUND1 A：采用对象、P2 与回执补正

W1（S-06、X-15）和全部 W3（X-01～05）已实现。最终共同冻结树上 **186 个不同测试全部通过**，四个独立进程退出码均为 0；含本方 W1 全量回归，不累计开发重跑或先前 W1 的用例数。产品与测试现保持冻结，交主代理精确提交。

证据入口：[最终验证与源码/测试 SHA](round1_a/final_validation.json)、[共同冻结](ROUND1_FROZEN.json)、[W1 验收](w1_a.json)、[W1 屏障提交](W1_GATE.json)。W1 产品提交为 `1a221e440f6f64cb83c4179f1cb9d22b92c3e812`。

## 行为与边界

| 项目 | 实现与实际断言 |
| --- | --- |
| S-06 | 声明原始拼写、目标和来源分别保留；Math 只同步明确原始绑定，未登记临时量保留。真实 `normalize_candidate → accept_result` 的 `V=v+1; y=V+v` 与 `:=`、原输入 v/V 四分支，v=1 均得 3，最终仍待审；反向捕获整组拒绝。局部/外层、双局部、Pipe/formal/comprehension、共享 provider/不同 consumer、安全同对象改名、幂等和事务恢复均覆盖。 |
| X-15 | 普通与维护 P2 `revise/plan=[]` 检查已读的 `state.plan_draft`，保留真实反馈并转 worker，不增加假 check_failures；approve 相等、reviewer raw 只读、坏目录与 adjudication 自有候选门保留。首稿记录为 pending_review。 |
| X-01 | receipts 中的同一条件/数字供 gate、schema 的独立副本及 invocation 使用；仅 P2/P3/P4 approve 要求非空、1200 Unicode 字符/20 非空行。阶段×decision、Unicode、空白行与 P5 长 handoff 实际批准通过。 |
| X-02 | 类别/locator 从当前 STAGES 和固定 bindings 投影；implementation 对应 entrypoint，同址 host 不抵消该类，每类任一允许 locator 足够。角色仍选择真实引用并写语义结论。 |
| X-03/04 | 沿原 shape/normalize/receipt 路径合并安全独立的坐标、类别、索引、handoff 错误；保留原数组索引与实际计数。schema_errors 原路径与解释送到反馈；输出截断报告省略数量/字符和未检查部分。坏容器前置失败不读依赖材料；来源漂移仍硬拒绝。每个新提案完整重验。 |
| X-05 | native 双通道矛盾优先归有限 CLI 路径，即使同时 schema 错误。仅普通 receipt repair 去额外累计 cap；第三次有余额可继续，用新 nonce 和原角色，旧次数/用量不清零。错误决定或清单外选择不执行、不越界读；坏 nonce、身份、权限、来源/字节与未确认终止不放宽。 |

坏回执拒绝批准、可信草稿未批准保留、原角色是否可重交分别有断言。调用仍受原 scope `call_exhaustion` 限制，保留独立 reviewer、裁决和签收预留、单次裁决/签收、维护以及 pause/cancel/hold 等用户控制。没有新增预算、parser、validator 平台或诊断服务；复用原校验路径与调用门。

## 最终验证

共同冻结 `ROUND1_FROZEN.json` SHA256 为 `0eed4276bfe21214f7b4f3955e3a17e91a4a26ce9fd9c2a64976fea38fb5fce4`。22 个本方产品/验收输入在测试结束后逐 SHA 重核一致。解释器为 `/opt/venv/python3.13.12/bin/python`（3.13.12），显式 main PYTHONPATH，禁写 pyc，PATH 中没有真实 claude/codex。精确命令、耗时及日志 SHA 见最终验证 JSON。

| 独立进程 | 模块 | 通过 |
| --- | --- | ---: |
| Template | `tools.tests.test_normalization`（PYTHONPATH 同时加入 main/schema_template，保持 check_yaml 的相同模块身份） | 31 |
| W1 Extract 回归 | artifact_transactions、stage_flow、maintenance_receipts | 43 |
| W3 协议/证据 | proposals、receipt_repair_policy、call_limits、backends_runtime、prompt_materials、review_protocol、issue_evidence、source_gaps | 82 |
| 采用/恢复 | current_returns、current_recovery | 30 |

先前 W1 启动方式与两个 fixture/旧期望修正完整记录在 w1_a.json，最终已重跑全部受影响模块。W3 开发阶段修正了多行反馈的 JSON 断言与精简 prompt fixture 的完整 schema 身份；最终 186 项来自同一冻结树。开发失败日志建议 local-only，公开证明使用最终四份日志与此清单即可。

## 精确改动与限制

本方产品仅 `schema_template/tools/normalization.py`、`schema_extract/tools/{runner,prompts,receipts,reading,agent}.py`、`schema_extract/prompt/operations/{response,receipt_repair}.md`。测试改动仅分配范围内的 normalization、artifact_transactions、stage_flow、maintenance_receipts、proposals、receipt_repair_policy、backends_runtime、review_protocol；其余分配模块作回归输入。精确路径/SHA 由最终清单列出。

全部使用新临时根、假 backend/CLI 和 CPU；未运行真实模型、GPU、生产 Extract 或 monitor，未改旧运行态、旧 receipt/账户/check 计数、固定来源提交、原稿、AGENTS 或 kernelschema。没有访问 AKA、AKA_experiment 或 archive 目录，没有执行仓库 Git 写动作。该证据不证明历史生产 33 稿数学正确、GPU/全域语义，也不声称真实模型返工成本已下降。无本方必需项遗留。
