# 第二轮 D：外层接口、材料与集成复核

2026-11-18。以 `f48823b` 及保留的真实修复为基线。未修改 schema_docs、案例 YAML/PNG、Kernel、旧 run 或历史报告；未提交、推送或发起真实模型/queue/monitor。

## 实际改动与接口

- CLI 删除 Extract/review-eval 的时间及显式输出参数，删除 `--add-seconds`。`resume --add-calls` 使用 `stages.validate_add_calls`，向 launch 传整数 `add_calls`；冻结能力为 `call_extension.v1`。`--scope` 与释放 hold 不追加次数。
- 显式 resume 在 `reserve_launch` 前重新交付/观察 pending cancel，包含已提交但尚未登记 delivery 的当前 cancel。未确认退出时明确拒绝启动。Runner 负责真实 attempt 结清；CLI 不清 fence、不采纳旧提案。`SupervisorBusy` 仍如实报告控制未提交或交付未确认。
- runtime、receipt、prompt 与 audit 删除专用输出身份、时间预算及旧 budget 投影。动态提示使用 `calls`；audit 保留实际耗时、usage、次数追加、控制交付观察与原有 starts/retries 等待证据。Claude 的 128000 默认及环境指纹由第一轮 backend 提供。
- `maintenance.validate_attempt(..., current_checkpoint=None)` 默认继续严格比较原 base；仅 Runner 已验证同一次 settled worker 的 result checkpoint、父 checkpoint、原始 repair 摘要与文件证据后传当前 checkpoint。role、step、generation 与 task 状态校验保留。consume/mark_drained 共用 `Store.control_transaction()`，只在获取锁处退让；对外 submit 仍单次尝试。
- audit 在既有证据集合中增加 gate 绑定的 `cli-tmp/version.json` 与版本 stdout/stderr；取消交付以独立观察投影随所选证据封存。回执身份中只删除已无合法来源的 scalar-null 分支，`reopen` object/null 与正式 nullable IO 不变。
- 保留 worker 规范化预览成功但 raw 未批准、JSON 与代码/Markdown 的类型边界、精确 comparisons 键名、真实行范围/offset/摘录校验，以及原生 session 身份独立于提案采纳。

## 对第一轮及 C 的独立复核

确认所有 Claude 调用复用 backend 默认；版本查询进入 gate 且可显式取消，输出流直接写文件，正常观察失败不杀模型。`recovery.verified_result` 的 `configuration_digest` 校验保留。clock 的在途发送、迟到结果及锁忙均沿原记录处理，观察与心跳不等待 queue 完成。

发现旧 pending cancel 在后来 pause 后会被 `deliver_cancel` 永远判 superseded，导致 resume guard 无法释放。已由 C 修复：同冻结身份的历史 delivery 可以继续核真实退出；最新 pause/hold 下只观察，不重放旧 KILL。C 的对应反例通过。C 另负责 pending result 的真实 gate 闭合对账与长耗时检查，不用年龄释放或假 exit。最后复核了仅未 finalized 的双摘要重放：必须是 `exit_source=cli-exit.json`，匹配旧/新 result 摘要、同次 gate、实际退出及原生事件；audit 的一般摘要不匹配仍拒绝。发布最终提交改用短事务锁退让，取得锁后仍重新检查最新控制；取消 pending 不计作格式失败。对应短反例由 C 验证。

## 相对 origin 必需的文字事实变化

保留远端段落、链接与行文结构，只替换失效事实：

- `schema_extract/README.md`：去 `ps` 依赖及三个秒数参数；说明 backend 默认 128000、耗时/token 仅统计、calls-only 追加；更新 cancel 的 KILL/pending/confirmed 和显式 resume 顺序；删除 monitor 自动重启选项，说明单份在途投递及 queue 故障边界；review-eval 示例去秒数。
- `schema_extract/tools/README.md`：调用接口去 timeout；终止依据改为 Linux `/proc` 归属；删除裁决时间预留和检查超时拒收，保留调用预留与真实检查结果。
- `EXTRACT_SCHEMA.md`、`schema_extract/REVIEW.md` 与 maintenance/normal_worker/receipt_repair/recovery/reopen/retry/monitor_handoff/wakeup 操作提示：执行预算只剩调用次数，相应原词局部替换。wakeup 的后台恢复句改为 Supervisor 决定恢复，ack/already_acknowledged 后仍 scan 的原语义保留。
- 保留合并时已有的必要差异：两处删除任意 12 行上限；README 的 acceptance 类型边界；check 的 worker preview 退出码事实；P1 的 comparisons 精确字段。P1 局部标签不能推出组合公式的原文完全保留。

维护 fields 上限、ready/发布区分、standalone 范围、review-eval 标量任务边界与其他 schema 规则均未重写。后续 high 只需精简本轮新增句子，不必改历史证据。

## 定向验证

20 个不同的短离线用例已通过：`test_maintenance_receipts` 6 项、`test_cli_recovery` 6 项，以及 prompt calls 1 项、P1 preview 1 项、JSON/nullable 3 项、真实证据范围/offset/摘录 3 项。使用现有临时 scalar fixture 和 fake native CLI，只调用相关步骤，不跑完整算子提取。

首批 11 项中 8 项直接通过；一项因并行改源码被严格 frozen-executor 校验拒绝，两项新夹具断言按真实接口修正（patch 后步骤名为 verify；同配置两输出的输入指纹相同）。随后固定生产源码窗口，3 项复验与 9 项必要接口/材料检查共 **12 tests / 46.296s / OK**；没有放宽冻结校验或重跑已通过组。

版本证据补齐后仅复跑现有配置/audit 用例，**1 test / 25.583s / OK**，实际归档含 version JSON、stdout 和 stderr，配置前置校验与默认 128000 保持通过。C 最后接口修正完成后，D 的受影响 Python 文件 `py_compile`、全工作树 `git diff --check` 及生产残留字段扫描通过；只剩 Runner 事件轮询的 `timeout=0.1` 和 SQLite 关闭 busy wait 的 `timeout=0`，均不是执行上限。未验证服务端接受 128000 或真实 queue→ack→scan；离线结果不代表这两个在线闭环通过。后续 high 文字精简与新 xhigh 总审由 Supervisor 安排。
