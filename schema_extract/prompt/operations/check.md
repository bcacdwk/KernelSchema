# 只读自检

需要定位结构、权限或规范化问题时，单独执行 invocation.check_command 给出的本次固定命令。本次目录、阶段、规则、来源和 `--read-only --skip-hardware` 已由 Runner 定位；不传额外参数，不追加 `echo`、分号、管道或重定向（Template 文档中的普通自检示例在此均由该 helper 承接）。

helper 不修改本次 workspace。P1–P5 worker 在临时目录的规范化预览与来源检查通过时返回 0；`pending_corrections=true` 的机械格式修正不表示 raw 已落地或已批准，满足本阶段提交要求即可 complete，由 Runner 接收提案后受控落地、严格重读并交独立审阅。结构、权限、来源或需人工精简的文本问题返回非零；reviewer 与 P6 严格检查当前 raw。checks 记录实际退出码。

P1 不要求未来 Design；P2 不要求未来代表文件（worker 自检显示 selection=awaiting_proposal，最终选择随 proposal 提交后由 Runner 核对精确 plan）；P4 检查完整 root 和目录中的已定选择及未来计划。检查器不能代替语义审阅，也不要求 worker 与 reviewer 重复运行同一检查。
