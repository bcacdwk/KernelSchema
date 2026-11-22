# Issue 03：把已有服务环境资料交给初始 Design

`vector_add-002` 已自然终结，三次 Qwen 调用后仍无可验收 Schema，候选 GPU 评测为 0 次。已有读取成功与重复超时不能确定唯一根因。本修复只补交已有环境事实，不声称解决全部交稿超时；native Write/Edit 交付修复另见 [ISSUE_02.md](ISSUE_02.md)。

主代理此前用正式 SDK 0.14.37 只读调用 `get_env('L20D')`，未创建 GPU job。公开材料 [environment.json](../../schema_workflow/examples/vector_add/environment.json) 保留白名单事实、获取时间、缓存来源和限制；它与原记录逐项相同，已由主代理单独提交为 `51f6b2399d29bafbeacf401de45afa7011160dc8`（`Add environment`）。没有携带凭据、签名 URL 或私有配置。

该记录于 `2026-11-22T12:29:54.586150+00:00` 获取，`source=cache`、`probed_at=2026-11-22T11:48:31.084555+00:00`、`probe_error=null`。设备身份为 `NVIDIA L20D / sm_103`；PyTorch `2.9.0+cu130`、CUDA `13.0`、Triton `3.7.1`、nvcc/ptxas `13.1.115` 均来自返回字段，未由营销型号推断架构。它不表示本次候选的观测、未来稳定分配、完整硬件/编程语义或正确性与性能结果；Python 版本未报告。这些限制写在参考文件中，会一起进入写作者材料。

Workflow 新增可重复 `--reference`，绝对化后只透传初始 Design，并纳入原有 config、输入摘要和恢复检查。普通文件按文件固定；Schema 目录或其中的文件按 Design 已有 root/dispatch 文件边界固定完整目录摘要。没有引入 SDK、Schema parser、环境查询层或新公共对象协议。`--schema` 与 `--reference` 同时提供会明确报参数错误，resume 不能替换参考；原有三起点和模型/次数/超时默认值不变。

两份 README 增加从 Task 开始、附带真实环境文件的命令；手写目标 Schema 没有作为 Design 参考。手写 root 版本由 1 升到 2，只补已知实现栈与来源，tiled 原文件保持不变；Golden_reference 仍指 `2c9c346cdd0373821b633012bdfece446c766d65`。环境元数据不能充当完整手册，Wiki 未绑定入口保留可见缺项。dispatch 使用完整真实 Git 定位符，避免复制进 Implement 后将原位置的相对链接误当作快照内材料。

验证使用 `/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python`：

```sh
python -m pytest -q schema_workflow/tools/tests/test_workflow.py
# 30 passed in 7.16s

python schema_template/check_yaml.py schema_workflow/examples/vector_add/schema/root.yaml --read-only
# 2 YAML; 0 would change; structure/text/from/relations/dispatch passed
```

新增 CPU 用例覆盖重复引用、相对路径、Schema 节点文件与目录的整体摘要、文件或兄弟内容改变后拒绝 resume、恢复参数不替换、输出不能重叠输入、非 Design 起点不静默忽略参考。真实 Design CLI 的材料装配也已执行，仅替换模型调用，确认环境原字节、SHA-256、manifest 绑定及完整内容实际进入写请求；该 CPU 用例不声称真实模型阅读或交付。既有真实 Implement→Optimize 材料交接 CPU 用例继续通过。

完整 Template 检查没有 `--skip-hardware`，没有修改 Template。`git diff --check` 通过；相对修复前 `c352c6b` 的保护区和固定 Task/eval 六文件 diff 为零，公开环境字节与其提交及原 SDK 记录一致。此代理未运行新模型/GPU、未再联网、未 stage/commit/push，未写共享 EXECUTION 或其他模块。真实主链仍须由主代理提交后使用新运行验证，001/002 的次数与失败结论保持不变。
