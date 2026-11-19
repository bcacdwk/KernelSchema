# 第三轮精简：运行与续接

2026-11-19。代理 `/root/issue_r3_runtime_high`，`gpt-6-astra / high`。

对照 HEAD、优化前快照和第二轮完整实现，运行链路已基本收敛。本轮只删冗余，没有新增机制、重做方案或删减独立测试覆盖。

| 文件 | 精简及依据 |
| --- | --- |
| `tools/publication.py` | 删除临时 markers 变量及空列表分支。有效 binding 已强制逐项采用 marked continuation，因此有 binding 时必有交付审计标记。 |
| `tools/snapshots.py` | 合并两个格式分支重复的 snapshot 根目录赋值；格式、身份、manifest、封印、并发复用检查原样保留。 |
| `tools/artifacts.py`、`tools/receipts.py` | 压短解释性注释；完整 inventory 和错误分类逻辑不变。 |
| `tools/tests/test_executor_repairs.py` | 删除未使用的 target lock 读取和重复说明测试用途的注释。 |
| `tools/tests/test_current_recovery.py` | 删除随后被覆盖的 termination 赋值，仍实际调用并断言确认终止。 |
| `README.md`、`tools/README.md` | 去掉重复叙述。Supervisor 手册保留命令、完整授权字段及用户决策含义；执行协议通过链接引用并集中说明保存证据、提交和冻结边界。 |

`executor_repairs.py` 的来源/权限核验、原已哈希 scope 材料、各类保存稿、准确 stage/attempt 标记及提交前重检均有独立职责，不删。`runtime.py`、Store/CLI 身份与锁等待接口、P6 共用保存 helper，以及 maintenance 空轮询、scoped retry 反馈、receipt repair、retained scope 四项原修复已足够简洁，本轮保留。

未完成诊断不形成批准；原 calls、approval、evidence 不改。pause/cancel/hold/在途 fence、普通阶段和 P6 gate、严格同内容封印复用不变。没有恢复无标记授权或 `--review`。未修改 AGENTS、正式 issue 状态、正式产物、原始回执、历史运行或两层快照。

定向验证使用仓库根目录、`PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin` 和 `PYTHONDONTWRITEBYTECODE=1`：

```sh
python -B -m unittest \
  schema_extract.tools.tests.test_executor_snapshots \
  schema_extract.tools.tests.test_executor_repairs.ExecutorRepairTests.test_user_instruction_continues_p6_and_delivers_deferred_audit_marker \
  schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_unstarted_gate_preserves_previous_session_even_when_result_is_missing -q
```

**5 tests / 51.473s / OK**，日志 `/tmp/r3-runtime-targeted.log`。覆盖同内容并发快照及不完整/可写/链接负例、读取漂移、真实目标正常发布与审计标记、未启动调用真实终止确认及原会话续接。日志含两条假 gate 子进程 ResourceWarning；确认终止断言通过。另完成全部 6 个改动 Python 文件 AST 解析及本组 `git diff --check`，均通过。

生产编辑完成；全量离线验收由根代理统一执行。本轮没有真实 extract/resume/monitor/cancel、模型、GPU、commit 或 push。
