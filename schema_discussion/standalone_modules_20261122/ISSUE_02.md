# Issue 02：大稿交付只能一次返回，且无稿超时仍盲修

`vector_add-002` 的 write 2/3 在 900 秒后退出 143；19 条公开 Read/Grep 均 completed，DOC1–6、Template README/starter、manifest 和 QSA 格式案例都实际可读。`final_answer` 为空，`structured_output=null`。本轮没有旧路径访问拒绝，Issue 01 修复的来源/读取缺口不能解释本轮全部失败。自然 repair 3/3 同样 900 秒后退出 143，最终回答为空、结构化稿为 null，14 条 Read 全部完成。i06 确认 Workflow 最终退出 1、`incomplete/write_failed`、Design `calls_used=3/schema=null`，相关 PID/进程组全部退出且没有 GPU job 后才交出产品写权。`unrecognized_model` stderr 提示也出现在成功选材中，不能据此判断写作失败原因。原 001/002 的证据、计数与结论不改。

已确认的交付缺陷：Design/Optimize 要求将全部 YAML 作为字符串嵌入单个最终 `StructuredOutput`，而写作者没有 Write/Edit。方法要求逐层写入、自检，运行接口却不能保全中间文件；无最终返回时 host 只有 null，仍自动发起同样的写作 repair。公开材料不足以证明长 JSON 是模型迟迟不结束的唯一原因，也没有读取隐藏思维来推测原因。

修复使用现成 native CLI 的文件工具方式，不导入 Extract runtime，不复制其多阶段流程：

- 选材继续只读。写作只允许 Write/Edit 本调用 `drafts/NN/` 中直属的 `.yaml/.yml` 与 `dispatch.md`；native 单层文件规则配合 host 目录/文件名检查，不新增业务命名或数量限制。
- 最终小 JSON 仅给文件名、notes、missing，Optimize 另给决定。host 从真正落盘字节收稿；未列入最终集合的文件保留为草稿，不进入交付。
- host 复核固定输入摘要、公开完成写工具的目标、候选直接文件/普通文件/链接/大小边界，再沿原有完整集合、Contract、规范化和严格 Template 检查。原始候选不被规范化覆盖。
- 有实际部分稿时，新 repair 复制到下一候选目录后继续；无新候选内容或最终决定的失败直接保留 `incomplete/write_failed`，不重复付费等待。没有成功终态的文件不能被当作成功交付。
- 增加公开 Write/Edit 路径/结果、工具错误、原生错误、终态状态/用量和调用耗时；不保存隐藏思维、原生大捕获或认证。没有新增监控平台。

本次调整明确替代 FINAL_PLAN 第 3 节中 Design/Optimize“仅只读工具、host 从大 JSON 物化全部文件”的初始选择；模型、effort、900 秒上限、调用总额、规范、固定 Task/eval 与验收不变。根方法只同步自动化入口与产物说明，设计方法不改。

硬件材料是独立问题：原固定 Task/eval 给出了 L20D、Python/PyTorch/Triton 与必要算法/分支/验收，但未给 Architecture 和具体栈版本；DOC2 允许 Wiki 入口占位，不等于可以编造主体事实。手写目标例的 Contract 不能代替独立设备资料，更不能作为从零 Design 答案投喂。主代理另取得 Agate `get_env('L20D')` 服务缓存资料，含 `arch=sm_103`、Triton 3.7.1、PyTorch 2.9.0+cu130 等；应作为独立原始材料交付，不能记作本次候选的 GPU 观测。

准备阶段曾为最小 Contract 对照读取手写例 root 前 110 行，连带显示其首段 Design；没有使用或投喂这些设计内容，随后停止读取该文件。修复材料不包含目标 Schema。

原 002 终态交接后已应用。受影响 CPU 检查：

```sh
/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python -m pytest -q schema_design/tools/tests schema_optimize/tools/tests
# 38 passed, 8 subtests passed in 25.58s
```

测试覆盖部分稿复制与原稿保全、无稿超时不重复、输入被修改即拒收、候选链接/非 YAML/子目录拒收、小响应收取真实文件、Contract 漂移，以及不能用 keep 隐藏已失败的候选修订。公开错误、写路径、用量保留而思维正文与 Write 内容载荷不进调用记录。

按主代理授权，只进行了一次真实 native 权限/交稿探针：`/tmp/kernelschema-native-delivery-probe-pxb3sq_0/` 保存 request、`drafts/01.json`、假输入及 `result.json`。模型/effort 为原默认值，探针单次上限 120 秒，43.874 秒后退出 0，diagnostics 为空。真实工具记录显示候选 root Write 与 Edit 均 completed；假 input Write 和 `candidate/nested/escape.yaml` Write 均实际尝试并得到权限错误。假输入原字节不变、嵌套文件不存在，最终 `StructuredOutput` 正常返回 `files=["root.yaml"]`、notes、missing。探针未运行任何 Schema 业务或 GPU，不能作为完整大稿的效果证明。

native 公开 `modelUsage` 自报 contextWindow=1000000、maxOutputTokens=32000；host 的输出 env 仍为 128000。该自报值不是本轮业务超时根因的证据，不据此改模型、effort 或扩大 900 秒。

`git diff --check` 通过。未 stage/commit/push；未修改原 001/002、Extract/DOC/Template/wiki、固定 Task/eval 或 Implement。完整 Schema 以及真实 Design→Implement→Optimize 主链仍待后续验证。
