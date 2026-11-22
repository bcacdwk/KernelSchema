# B04：K4 接入所需的单调用期限与缺 PID 计费修复

本次只修复 KS Claude 单调用运行时的启动期限与结算边界，没有新增模型/GPU 调用、时钟账本或调度器。公开 profile 配置、B03 报告、旧真实 capture/source/历史声明保持不变。修改仅涉及工作树 `kernelschema/runtime/{claude.py,_runner.py,_local.py,INTERFACE.md}` 与 `tests/workflows/test_claude_runtime.py`。

## K4 可直接使用的预约字段

`reservation.active_ms` 保持原有单调用预约时长，新增可选字段：

| 字段 | 规范 |
| --- | --- |
| `deadline_utc_ms` | 严格有符号安全整数，UTC Unix epoch 毫秒；不接受 bool、float、string 或 null |
| `deadline_monotonic_ns` | 1–40 位 ASCII 十进制字符串；同一主机 monotonic 纳秒截止 |
| `clock_boot_id` | 非空字符串，必须与上一字段成对，并匹配当前 Linux boot ID |

字段类型/成对关系在 admit 验证，启动及 runner/exec 关口再次检查当前 boot。UTC 或 monotonic 可单独提供（monotonic 必须带 boot）；K4 同时给两种截止能防止 admit→invoke 期间墙钟回拨延长原窗口。类型/boot 错误为 `ProtocolError`；有效剩余不足一毫秒或已耗尽为 `AdmissionDenied`，不创建 launch、不启动 CLI。

最终启动点重新取：

```text
min(profile timeout,
    fixed request active_ms,
    reservation active_ms,
    deadline_utc_ms - current UTC,
    deadline_monotonic_ns - current monotonic)
```

仅对实际提供的截止取 min，余量向下取整到毫秒。存在任一截止时，`launch.json` 保存收紧后的 `timeout_ms`、有效 UTC/monotonic 截止和 boot ID。即使 owner 只给 UTC，启动后也固定一条对应的同机 monotonic 截止，不能因为 supervisor 排队或后来的墙钟回拨再次得到原完整时长。

supervisor 在实际创建 child 前重新核剩余；child 在读入 launch 后、执行文件 hash/fsync 后且 exec 前再次核剩余；supervisor 等待原 child 时持续使用原截止的剩余，并在耗尽时 kill 原进程组、wait 后记录终态。超时仍是本地进程终止事实，不伪造缺失的 native 用量。没有截止字段的路径保留旧 duration-only 行为；**K4 要保护含输入准备时间的 owner 总 active 余额，必须传新截止字段，不能只传已冻结的 active_ms。**

这只是从唯一 owner 接收不可延长的当次授权界限，不创建新总额、累计 active 时钟、暂停状态或另一个账本。原 owner 的 authorize 回调、pause/fence、预约去重和预算结算职责不变。

## 缺 child 身份不退款

修复前，exit 已存在但 `child.json` 丢失可能得到 calls=0；这会把真实调用当未发生。现在：

- child 在最终 exec 关口持久记录与原 operation/session/process 绑定的 `exec-intent.json`。
- 原 exec 意图或原 native init/terminal 仍能证明调用时，calls 与已知下界保留 1；原 tokens/CLI-reported cost 保留。
- 只有 launch/exit、缺乏执行证据时，calls 为 null、已知下界为 0，而非确认零调用；owner 保留未知/预约，不得据此退款或放行受限余额。
- 缺 child 身份时一律拒绝接受模型输出；不能凭一个成功 native result 掩盖原定位事实损坏。

有截止的 runner 迟到且尚未执行 CLI 时，受信 supervisor/child 才会写 `not-executed.json`。只有该事实匹配原 operation/session/process、匹配 exit 的明确标记，且 native stream 为空，才能确认 calls/tokens/cost 为 0。它是“未调用 CLI”的宿主证据，不是补造的 native usage。`facts.not_executed=true`，状态 failed，错误为 `deadline_refused_before_exec`，没有模型输出。

若“未执行”声称与原 native 事件矛盾或身份不符，记录 `unverified_not_executed_claim`，不冲销原调用。有效截止、exec 意图和明确未执行原件都进入原自包含 usage receipt；依旧无需业务成功报告才能保存失败事实。缓存重放仍复用原 receipt，不重复调用或结算。

## 针对性 CPU 验证

使用已有 Python 3.13.12 的 k2-cpu 环境，只运行受影响 runtime 子集，没有全量无差别重跑：

```sh
python -B -m pytest tests/workflows/test_claude_runtime.py -q \
  -k 'deadline or missing_child or success_raw_usage or unknown_original_process or timeout_kills or actual_caller_death or receipt_cache_loss or failed_call_usage_index' -x
```

**26 passed, 40 deselected，46.79 秒。** 覆盖字段严格性、admit 后过期零启动、有效 timeout 缩短、实际 subprocess runner 排队过期不启动 CLI、迟启动只获原剩余、child exec gate 再查、同机 monotonic 在墙钟回拨时仍拒绝、boot 变化拒绝，以及实际 fake CLI 完成/首次 settle 前删 child 身份（含同时删 exec 记录）不退款、不伪成功。

随后收紧了“已证明未 exec”时的错误分类，避免将无 CLI 的截止拒绝误标成非零 CLI 退出，并增加矛盾的未执行声明反例：

```sh
python -B -m pytest tests/workflows/test_claude_runtime.py -q \
  -k 'deadline_runner_queued or missing_child or not_executed_claim or original_failures_are_recorded_without_retry and nonzero or timeout_kills' -x
```

**7 passed, 60 deselected，14.60 秒。** 这些结果是 CPU 运行时工程验证；既有四次真实 CLI 资格只保留其原 profile/source/input 身份和原适用范围，不声称本次新截止/缺 PID 故障分支已实际付费验证。

## 交接和修改边界

- API 签名未变：K4 将新字段放入原 owner 的 reservation，仍调用 admit/invoke/settle。
- upfront 期限耗尽抛 `AdmissionDenied`；已启动的 supervisor 迟到则通过原 operation 的明确未执行 receipt 回收，不能新开操作绕过未知 fence。
- `calls=null` 不等于 0；本地终止与统计完整仍分开。K4 应按已有 unknown/余额不明规则保留 fence。
- 只有更新后的实现可产生新事实；私有 run 仍要求原实现身份，不提供静默跨版本恢复。
- 没有修改 `profiles/`、`qualification/20261120/`、B03 文件、core、K4 业务代码、共享 fixture 或 Git 配置，没有移动已结束 run。
