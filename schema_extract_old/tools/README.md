# 提取工具维护

根目录的 `runner.py` 接收命令，`supervisor.py` 是独立巡检入口。实现按职责分开。

| 模块 | 职责 |
| --- | --- |
| `planning.py`、`records.py` | 选择与冻结输入、复用草稿、保存任务和不可变检查点 |
| `execution.py`、`review_context.py` | 推进阶段、绑定审阅、恢复和导出 |
| `prompts.py`、`protocol.py` | 按阶段准备提示，校验回执身份、行号与引文 |
| `codex_cli.py`、`reading.py` | 调用 Codex 与确认终止，通过原生 `exec_command` 按原行号读取或定位源码 |
| `supervisor.py`、`scheduling.py` | 只读巡检、派发排队任务和可选的 macOS 定时唤醒 |
| `framework.py` | 问题提报、查重与归档决议 |

`.schema_extract/tasks/<key>/task.json` 是任务状态的唯一依据。检查点先完整落盘，再更新任务指针。恢复先确认旧进程终止。执行器按仓库相对路径核对冻结摘要，默认只冻结提取所用的运行模块。

读取工具单页上限仍为 6,000 字节。agent 用 `--pages 2` 在一次工具回合内读取最多两页，下一组从末页游标继续，也可从该组首游标按单页重放。

Contract 审阅从冻结的数值记录引用显式的 `relative_to`。比较精度和非有限值规则缺少依据时标为未知。多节点任务中，若已有独立审阅、来源和规则相同、节点与选择关系未变，且共享 Contract 只改了容差，新的 reviewer 聚焦受该判据影响的数值断言。证据不足时回到完整审阅，每个节点仍须取得当前检查点有效的独立结论。

修改工具后用新计划验证。继续原版本时，从 `inputs.lock.json` 取 `snapshot` 作工作目录，`--repo` 指向对应版本的仓库，`--state-root` 指向原运行区。这两个参数均放在子命令前。

回归命令在仓库根目录执行。

```sh
python3 -B -m unittest discover -s schema_extract/tools/tests -t .
```
