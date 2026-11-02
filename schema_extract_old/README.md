# Schema Extract

从选定的 kernel 源码写出可追溯的 schema。**Less is more。** 每个节点的设计都应能独立读懂。同一节点已有定义就直接引用，保留影响计算、调度或流水线的细节，审阅只提出有实质影响的修改。[逐段提取方法](../EXTRACT_SCHEMA.md)

## 跑一次提取

需要 Python 3.10+、PyYAML 和已登录的本地 Codex CLI。从仓库根目录运行，`--work-root` 指向仓库外的暂存目录。

```sh
python3 -m schema_extract discover
python3 -m schema_extract plan --id fp8_scaled_gemm --run-id fp8_trial \
  --work-root /absolute/path/to/work --output-root schema_wiki/my_collection
python3 -m schema_extract run fp8_trial
python3 -m schema_extract status fp8_trial
python3 -m schema_extract export fp8_trial --id fp8_scaled_gemm
```

`plan` 只冻结显式选择的算子、规范、检查器和执行器。作者按 Contract、root、派生节点、索引推进，在同一会话内续写。独立 reviewer 审阅已接受检查点的副本，索引小修可复用未变节点的有效结论。检查与审阅通过后任务达到 `ready`，再显式导出。

硬件段保留模板占位，检查使用 `--skip-hardware`。提取不做 GPU 验收。轮次、时长、总 token 和 schema 正文长度均无硬性上限，写到足以还原设计即可。

## 查看与恢复

- `status RUN_ID` 查看阶段、检查点和活跃调用，`--full` 查看完整任务记录。
- `pause RUN_ID` 在调用边界暂停，`cancel RUN_ID` 请求终止。用 `resume RUN_ID --id OPERATOR` 恢复中断的任务，或用 `retry RUN_ID --id OPERATOR --reason '原因'` 重试已结束的未就绪任务。恢复从已接受的检查点建立新会话。
- `plan --seed-run OLD_RUN` 可在来源、规范和检查政策匹配时复用已接受草稿，并重新独立审阅。
- `candidates RUN_ID` 汇总本轮框架改进建议，`issues` 列出待决项。算子自身的缺陷与资料缺口随任务记录保留。[入账规则](issues/README.md)

主入口是 [runner.py](runner.py)，也可直接执行 `python3 schema_extract/runner.py --help`。可选巡检使用 `python3 -m schema_extract.supervisor tick RUN_ID`。macOS 上用 `enable`、`disable` 管理 15 分钟调度。巡检只读，可请求启动已排队的工作。[工具维护与测试](tools/README.md)
