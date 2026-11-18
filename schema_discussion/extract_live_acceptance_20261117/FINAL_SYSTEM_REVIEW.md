# 最后一轮整体复核

2026-11-18。基于 `f48823b` 和前轮未提交改动复核，未重做大调研，未提交或推送。

## 额外落实

- 统一取消语义：后续 pause 不撤销同次执行的 cancel；旧 confirmed 交付保持不变，现有 launch/control revision 防止旧请求越过显式 resume 误伤新执行。resume 也检查被后续 pause 覆盖、尚无 delivery 记录的 cancel；新执行可登记新的 cancel。
- monitor 不再仅凭保存的 `termination.confirmed` 报告已退出，须同时核实当前进程身份及真实 gate 闭合。删除了一处无用的 invocation 读取。
- Runner 的普通与单范围恢复均结清 provisional result，再按更新后的 CLI/处理耗时高水位落账；不漏掉 pending 后的结清时间，不重复累加同一调用。
- queue 复用 `child_exited`，在后代观察/结清前保留主进程身份锚，最后才回收退出状态。修复主进程快速退出、同组后代尚未登记时永久失去归属证明的窗口；没有按年龄重发或杀停，也没有凭空表补造退出证明。

维持无时间截止、真实取消与 fence、环境指纹/冻结身份、未 finalized 双摘要转换、调用与一次性决定限制、独审/P5/P6、维护保稿重提和严格篡改拒绝。没有新守护、恢复框架或旧 run 兼容层。本轮未再改活跃 MD/prompt；远端文字事实变化仍以前轮两份实施/精简记录为准。

## 验证

14 个不同短定向用例通过：取消交付 6 项、当前退出观察 1 项、cancel→结清→resume 及耗时 1 项、queue 在途/迟到/stop 与真实小进程后代 4 项、CLI 在 pause 后补核未交付 cancel 1 项、scope provisional 结清与去重 1 项。产品/受影响测试编译及 `git diff --check` 通过。此前 R1/R2 集合未重跑、未相加；未跑真实算子提取或大 suite。

## 在线小验收

仅使用 `/tmp/extract-final-review-aw6kj08o` 的全新空环境，原 run 未启动。安全配置、退出与 usage 见 `FINAL_SYSTEM_REVIEW_ONLINE.json`；不封存凭证、headers、私有 home 或隐藏推理。

- Claude：默认 `qwen3.8-max[1m]` / `max`，环境 `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`，普通工具为空，CLI 2.1.316。初次探针的合法 JSON Schema 只写 `const: true`；内置 StructuredOutput 四次传入字符串，最终 exit 0 但 `structured_output=null`，43.03 秒、output 1287 tokens，不能算通过。经 Supervisor 追加授权，唯一一次修正探针明示 `type: boolean`，18.31 秒 exit 0、唯一原生 success、结构化响应精确 `{"ok":true}`；2 个原生 turns，output 533、input 6、cache creation 1364 tokens。未降档，两次调用均自行退出。
- 同 flags 的一次本地假 API 捕获实际请求 `max_tokens=128000`、`stream=true`。假 API 未返回上限元数据，CLI 的 `modelUsage.maxOutputTokens` 仍为 32000；内嵌代码也确认它来自本地 `age(model).default`，不能当作请求值或服务端限额。真实服务短 JSON 已通过；实际参数只在离线传输中捕获，未拦截真实认证请求，也未证明填满 128K 或最大容量。
- 提醒：绑定主线程 `01a1c900-a487-76a1-92ee-990cbff08276`，空 state，保持原 300 秒首提醒。真实入队一次，queue exit 0，消息 `01a1ed22-fd63-7ae1-84a3-f26a2ab198bf`；当前活跃主线程未实际接收，未代 ack/scan，故在线闭环未验证。没有第二次投递。

两次模型、queue 及 clock 的进程/后代均经实际观察确认退出，本地假 API 已关闭并 join。clock 已停用，generation 由 1 变为 2，锁 free；保留临时目录中仅四个最小 monitor 状态/锁文件，供迟到提醒识别停用状态。私有模型 home、原始事件、临时脚本与其他临时文件已删除。当前全部产品改动仍未提交；本轮已停止编辑和临时进程。
