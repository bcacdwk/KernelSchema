# B07 / 原在途操作的预算检查接口

2026-11-20。按当前 `KernelSchema/.worktrees/schema-workflows-v1` 工作树补充
`OwnerAccount.check_active(reservation) -> bool`，供 A5 的原 native
`run_bounded` 在执行中轮询。没有新模型、GPU、网络调用或另一份账本。

## 接口语义

实现位于 `kernelschema/budget.py`。调用使用原 authority 的 flock 事务，
推进并持久化同一个 owner 活动时钟；不再调用 authorize，不新增额度、
operation、receipt、收费、deadline，也不释放预约。已分配的 calls/jobs
已经在 held totals 内，因此不会再把当前调用加一次；全部调用额度已被
合法预约占满时，原操作仍可在剩余时间窗口内运行。

返回 True 需要同时满足：

- 传入 reservation 与原持久授权的 JCS 字节完全一致，含原 owner、account、
  operation/run 身份；原请求贡献、Q/root 和活动段仍存在。
- 原操作已经 launch，尚未 released/terminated；owner 未 paused。
- 原 UTC 截止未到，原 monotonic 截止/boot（若存在）一致且未到；保留原
  最低收尾预留，活动窗口必须还有正余额。
- 原 parent/request/native 限额与已有用量、预约、finish reserve 相容；
  受限余额不能确定或实际累计超限则拒绝。
- 本操作或同一请求的已批准 group 可正常在途。组的原身份、成员、费用
  计划及预约关联必须完整；明确 mark_unknown、缺身份、非本组未结算操作
  或受限费用 unknown 仍 fence。统计 null 不自动等于进程仍在途：已终止
  成员的非受限 token unknown 可以保留，有限 token 余额不明则拒绝。

不同/伪造预约（包括 Python 中相等但 JCS 不同的 `True`/`1`）、未启动、
已释放或已终止、暂停、耗尽及可检测的 authority/请求/组损坏返回 False。
缺原 authority 时不会创建替代账户。正常暂停/限额拒绝也会持久化本次
clock tick，在途时间不会因 False 或 pause 而停止累计。

False 的消费方式是停止继续执行有界工作，并查询/结算**原** native 操作。
它不是已经终止、已退款或可以换 operation/channel 的证据。
本方法不查询 native PID/job 状态；adapter 发现原生身份/提交确认丢失时
须调用现有 mark_unknown。通过同一私有 authority 轮询不要求重新 claim
group，不允许借轮询转交或重新启动成员。

## 改动与兼容

仅修改：

- `kernelschema/budget.py`：新增 check_active；原 `_check` 增加内部
  `active_operations` 过滤参数，默认空值保持原 reserve/authorize 行为；
  继续复用 `_check`、`_totals`、原事务与时钟。
- `kernelschema/implement/INTERFACE.md`：增加消费说明。
- `tests/workflows/test_budget.py`：原 13 项保留，新增 30 项针对性检查。
- 本新报告。

未改变 authority 格式或 VERSION，未迁移、重建或改写任何已有账户。
没有修改 K6/writing/driver/runtime/evaluation/其它测试、原交付报告、
私有 archive 或 Git。

## 实际 CPU 验证

使用已有 Python 3.13 CPU 环境：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_budget.py -q
```

最终 **43 passed in 2.81s**；包含原预算全部回归和新增的正常单操作、
并发组共用时钟、暂停/恢复、零重复准入/计费、伪造 reservation、原 UTC
及 monotonic/boot、三层最紧 counter 上限、受限与非受限 unknown token、
finish reserve、已关闭/未知操作、缺 authority/请求 Q/活动段/组身份等。

首轮新测试发现 pause 拒绝以异常离开事务，使当次时钟推进未立即持久化；
已改为在原事务内返回 False，后续完整组验证通过。在最终 43 项前一轮
为 **42 passed in 2.85s**，再补原请求 Q 缺失负例后得到上述最终结果。
三份授权源码/说明文件的空白检查通过。没有为本补丁重复运行整个仓库，
也没有任何新模型或 GPU 资格声明。
