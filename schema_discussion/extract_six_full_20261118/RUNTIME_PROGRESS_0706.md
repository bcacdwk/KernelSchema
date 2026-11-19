# 六项 P1 native-call 只读进度抽样

观察窗口：2026-11-18 07:06–07:08 UTC；统计截止约 07:07:39。读取登记的 run/attempt 日志及 workspace 元数据；仅汇总工具调用、结果和系统事件类型，未读取/转述模型隐含思维，未修改运行状态或产物。下表调用数为当前 native attempt 内部工具调用次数，非外层 calls 预算。

| 算子 | 可见推进 / 当前工具数 | 原稿与自检 | 诊断 |
|---|---|---|---|
| FP8 grouped GEMM | 15 Read，1 Write，1 Bash；07:05:24 写入；最长工具间隔 29.6 分钟 | root.yaml 10,396 B；07:05:36 自检失败：out 重名，多字段超过字数限制 | 已完成首稿，进入内容/格式修正；无权限拒绝 |
| NVFP4 MoE | 15 Read，2 Write，2 Bash；06:50/07:00 两版原稿；最长工具间隔 30.3 分钟 | root.yaml 18,932 B；两次自检失败；第二次诊断已缩减为 output 重名及两个 Math 超 1000 字 | 有可见修订进展，尚未通过自检；无权限拒绝 |
| FMHA MXFP8 | 29 Read，18 Grep，5 Bash；最后工具 06:57:50 成功读取 symbols.py；最长已闭合工具间隔 14.1 分钟 | root.yaml 仍为初始 1,522 B / 06:16:30；没有 Write 或自检 | 仍在阅读规则/生成；曾两次 git cat-file 被拒、两次路径探测失败，之后大量 Read/Grep 成功，未见连续失败死循环 |
| MLA expanded FP8 | 23 Read，5 Grep，2 Bash；07:05:07–27 成功读取 receipts/comparators；最长工具间隔 16.5 分钟 | root.yaml 仍为初始 1,522 B / 06:16:35；没有 Write 或自检 | 有规则阅读活动但未落稿；历史两次不存在路径，未见权限拒绝 |
| Sparse dual cache | worker 07:03:09 StructuredOutput 成功，已进入 reviewer；reviewer 14 Read、6 Bash | worker 06:49:31 preview_passed=true；reviewer workspace 已有归一化稿 11,563 B / 07:03:12 | 可确认跨角色推进；reviewer 07:05:33 Python 读取记录被拒，07:05:45 改用 Read 同文件成功 |
| GDN prefill | 截止 07:07:39 共 31 工具；07:07:25 首次 Write 成功；最长工具间隔 16.3 分钟 | root.yaml 12,100 B；07:07:35 自检失败，报告多字段长度超限且 source_validation=verified | 从初始稿进入自检；06:35 的复合 Bash 被拒后，06:36 直接执行 helper 成功返回结构化校验失败 |

当前六个 native attempt 的每次工具调用都有对应结果，没有待返回工具。单次工具最长 2.815 秒；每个当前 attempt 累计工具等待仅约 1.7–6.1 秒。五个初始 worker 的 events 绝大多数为 `system:thinking_tokens` 元事件，且存在 14–30 分钟的工具间隔；可观察证据表明长耗时主要位于 native 模型生成与间歇性规则阅读/修稿，不是文件工具等待。该元事件持续出现只能证明生成活动，不能证明内容正确或必将完成；FMHA/MLA 尚未落稿，需继续按新 Write、helper 结果、StructuredOutput 或阶段变化观察。

真实权限拒绝最小定位：FMHA attempt `ff6a95e9-68ce-40e6-a28d-7ed938a7a3f4` 在 06:47:19、06:48:46 的复合 `git cat-file`；Sparse worker `05a2fdf9-06aa-4669-8253-8ab14f264491` 06:41:57 及 reviewer `b56c1d5b-c970-4c99-beb9-a2287811153c` 07:05:33 的 `python3 -c`/`python3 -I -c`；GDN `4100a420-2747-496e-b8d0-95946a779ca1` 06:35:11 的 `cat helper; ... python3 -c`。均立即返回 `dontAsk` 拒绝；GDN 并非直接执行 helper 被拒。`schema_extract/tools/backends.py:174–190` 明确仅把固定 helper 加入 Bash allowlist，且设置 `--permission-mode dontAsk`，因此这些扩展命令拒绝符合当前工具权限形态，不能归类为整个工具通道故障。未发现需要通过扩大权限、重启或介入代写来解除的现存工具挂起。

六项 stderr 均有一次启动时 `unrecognized_model: qwen3.8-max[1m]`，之后已有大量成功工具调用，Sparse 已结束 worker；仅凭此提示没有证据判定模型调用失败。自检产生的非零退出码均有明确 schema 诊断，应由当前 worker 处理；本次未审批 schema 内容、未运行额外自检、未启动模型、未处理 Supervisor 锁。
