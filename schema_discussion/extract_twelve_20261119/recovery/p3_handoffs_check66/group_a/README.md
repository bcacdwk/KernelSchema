# 巡检 66：P3 handoff 恢复 A 组

本轮仅控制原 `mla_nvfp4` 和 `fmha_bf16` 两个 run。两项都已完成**一次实际提交的普通恢复**，并真实启动原 P3 reviewer 的下一次调用；**P3 尚未批准，新回执待正常巡检核验**。

| 算子 | 原 P3 调用 | 最新失败 handoff | 恢复后调用 | 新 attempt |
| --- | --- | --- | --- | --- |
| mla_nvfp4 | 4/32 | 3241 字符、5 非空行 | 5/32 | `d9bca411-63f2-4658-adf4-9659ea81152f` |
| fmha_bf16 | 7/32 | 1235 字符、4 非空行 | 8/32 | `5b5ce7b1-c13d-46ab-919f-78be956a7eab` |

两项恢复前均为 `P3 / root / reviewer / normal / invalid_receipt`，`receipt_repairs=3`、`invalid_receipts=3`、冻结 `retry_limit=2`，没有追加调用。`call_limit_or_single_decision` 是观察分类，不能据此认定 32 次预算耗尽。冻结门禁仍要求非空 handoff 不超过 1200 字符和 20 非空行，直接计数与原生结果一致。

两项各有 21 项恢复前置检查和 37 项恢复保存检查，全部通过。历史调用已真实退出、后代结清、用量登记且 settled；旧 Runner 不存活、锁 free，无 active/unconfirmed/unsettled 调用或控制、hold、维护屏障。最新失败调用的原生成功、exit=0、schema/nonce、响应匹配及精确 session 均有效。正式 `audit --attempt ... --verify` 的 errors 均为空，但状态为 **partial**，明确保留 `receipt_not_accepted; artifact validation not completed`，不构成旧回执批准。

MLA 在北京时间 **2026-11-20 06:43:29** 启动恢复。FMHA 同批发起的第一个 CLI 请求返回 `busy / committed=false`，控制事务未提交；已核实该 CLI 退出、state 字节完全不变、仍为 7 次、没有新 attempt。Supervisor 明确授权仅重放这个未提交请求；在与 B 组确认不再竞争控制入口后，于 **06:44:41** 单独重放成功。FMHA 因而有两次 CLI 命令记录，但仅一次已提交恢复、仅一个新增模型调用。没有重发已经成功启动的 MLA。

新调用均有真实 native init，精确 session 与原 reviewer 一致；正式 inspect 确认 Runner 活跃、锁 held、调用 running。新 prompt 含对应最新失败 attempt 与原 `<=1200 characters, <=20 nonempty lines` 诊断，prompt 摘要及新 nonce 已核对。原输入锁、source/rules/executor 身份、配置、32 次上限、checkpoint、P1/P2 批准、选择/节点/计划、旧 accepted receipts、旧失败证据和逐调用用量均保持。

本轮未新建 run、未增额、未提交维护或 runtime 修复，未直接修改 state、旧 receipt、批准、prompt、kernel 或规则，未进行 GPU 测试，也未 ack/scan 或修改 monitor 时钟。两个 run 后续仍须通过原 P3 及全部剩余门禁；若新结果重复失败，先记录准确长度与证据，不在本轮循环恢复。

详细证据：

- [MLA 记录](mla_nvfp4/README.md)：本 scope 首次暴露长度诊断，和先前已通过的 P2 handoff 故障分开。
- [FMHA 记录](fmha_bf16/README.md)：2270 → 1235 字符的明显收敛及明确未提交请求的顺序重放。
- 各目录的 `diagnosis.json`、`before_history.json`、`before_audit.json`、`recovery_verification.json` 提供完整事实、计数和检查结果；`before_state.json`、`after_state.json`、`failed_*.json`、`resumed_*.json` 为原始证据副本或对应原生材料。
