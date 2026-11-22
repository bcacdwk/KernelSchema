# Workflow

`workflow.py` 用当前 Python 和绝对脚本路径，调用独立 Design、Implement、Optimize 的同一 CLI。它负责版本、累计调用和恢复，不读取 Schema 语义，不初始化模型或 Agate。

## 三种起点

```sh
# Task → Design → Implement → Optimize
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --gpu L20D --out R
# 已有完整 Schema → Implement → Optimize
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --schema S --gpu L20D --out R
# 已有 Schema、完整源码和反馈 → Optimize，再做本次 Implement 验证
python3 schema_workflow/workflow.py --task T/task.md --eval-task T/eval --schema S --kernel K --feedback F --gpu L20D --out R
python3 schema_workflow/workflow.py --resume R
```

新运行必须给 Task 和全新 out；Schema 可以是目录或其中 YAML，kernel 可以是完整源码目录或自足文件，feedback 可重复。只有文字任务时可以先 Design，之后 Implement 具体报告缺少的固定 eval/GPU。外部成绩只作起始证据。

默认模型 `qwen3.8-max[1m]`、effort `max`；`--model/--effort` 统一传到三个入口。`--max-versions` 默认 3（含初始版），`--max-calls-per-schema` 默认 6；Design 最多 3 次，每版 Optimize 合计最多 2 次。`--timeout-s` 默认 900，交业务 CLI 限制单次写作与评测等待。第三起点第一次 Optimize 最多 1 次，预留另一次读取本次实测。

## 交付与有限收尾

stdout 一行绝对 `result.json` 路径；退出码 `0/1/2` 对应 complete/incomplete/interrupted，进度去 stderr。参数错误不创建运行目录。result 有 `status, stop_reason, last_usable, latest_attempt`；路径相对 result 所在目录。

- 每版只有一个 Implement 目录；repair 带原反馈 resume，同版 coder 次数不清零。
- 只有有效 revise 才登记下一版，沿用最近完整源码作 seed，重新全案例评测。
- keep 只有在本版本次 Implement 成功后才可完成；外部起点 keep 仍必须先验证。真实失败后的 keep 保留失败。
- complete 至少需要本次成功实现与一次读取本次正确性/计时反馈的有效 Optimize 决定。到达版本上限时，可选 revise 保留为未实施提案，并可 complete+limit；必需修复、真实失败和未结清运行不能这样收尾。
- `last_usable` 直接指固定 Implement Schema、完整 kernel 与终态 feedback。required_fix 或 design_gap 会保守清空配对；以后完整重验成功才能重建。可选新提案失败可保留旧配对，但总体仍 incomplete。

## 恢复与产物

```text
R/
  state.json
  result.json
  design/
  v001/implement/
  v001/optimize/01/
  v001/optimize/02/
  v002/...
```

Workflow 持目录锁，动作前和结果后原子记录。resume 固定原参数并核对原输入、已交付不可变文件；完成状态直接复用。Implement 中断只在原目录恢复，原 job 的查询与候选核对由 Implement 负责。已创建却没有可信终态的 Design/Optimize 不重放，结果具体指向需定位的子运行和进程。不能用新的子目录把未知调用当作零次。

源码依赖和安装见[根 README](../README.md#独立-design--implement--optimize--workflow)，固定任务见 [vector_add](examples/vector_add/README.md)。CPU 测试：

```sh
python3 -m pytest schema_workflow/tools/tests -q
```
