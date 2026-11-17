# 只读自检

需要定位结构、权限或规范化问题时，单独执行 invocation.check_command 给出的本次固定命令。

本次目录、阶段、规则、来源和 `--read-only --skip-hardware` 已由 Runner 定位。命令不接受额外参数，不追加 `echo`、分号、管道或重定向；退出码和诊断直接读取工具结果。Template 文档中面向普通作者的 Python 自检示例，在当前调用内由此 helper 承接。

阶段 helper 只读取当前文件，规范化预览写入本次临时目录，不改本次 workspace。返回中的 `corrections` 和 `preview_passed` 不表示 raw 已落地或已批准；Runner 接收提案后完成受控规范化、严格重读和独立审阅。

P1 不要求未来 Design；P2 不要求未来代表文件；P4 检查完整 root 和目录中的已定选择及未来计划。P2 worker 的选择只在最终 proposal 中提交，自检会明确 selection=awaiting_proposal，Runner 再核对精确 plan。

检查器只说明实际覆盖范围，不能代替语义审阅；不要求 worker 与 reviewer 重复运行同一检查。
