# 两例证据末行 +1：只读核对

结论：**两例 Claude Read 的真实工具回显都在末尾换行后显示一个带编号的空 EOF 行，恰好是提案所选末行；不是本工具把证据绑定到错误版本。** 不应仅归因为模型随手写错。以下只引用文件字节、tool_use/tool_result 和最终结构化输出，不转述模型内部思考。

| 项目 | FMHA P1 reviewer | Sparse P2 worker |
| --- | --- | --- |
| run | `fb8b3758-92d7-4093-aa18-c2e5028f0a43` | `8f0844f9-f279-4781-a19d-0216b7fc4ef5` |
| attempt | `8bee8bce-75e6-4f97-af52-dbcdfe16384b` | `65b0cb22-3de1-4bcc-9d7e-741be71046ea` |
| 提案范围 | `artifact_evidence[15] root.yaml:167–205` | `artifact_evidence[3] dispatch.md:47–110` |
| workspace 字节数 / LF 数 / CR 数 | 14644 / 204 / 0 | 27691 / 109 / 0 |
| `decode("utf-8").splitlines()` 行数 | 204 | 109 |
| 文件 SHA256 | `fb5e2d49641504b94f3fb963f86884e58fe0484542e54751c62b1b0cbee819c5` | `f23b0e117f69ea703ab572e574761c7f2ff6695a59453c54d0219cc09639a555` |

两文件均以单个 LF 结束，最后一条实际文本行非空。Read 的相关回显如下（`\t` 表示回显中的制表符）：

```text
FMHA events.jsonl:52  Read toolu_4cdef7a68a27451f81065ca9，读取本 attempt workspace/root.yaml
FMHA events.jsonl:53  tool_result 尾部：
203\t            Pipe:
204\t              - compute:output
205\t

Sparse events.jsonl:30278  Read toolu_b3388ef4be014db192a7135c，workspace/dispatch.md，offset=103，limit=10
Sparse events.jsonl:30279  tool_result 尾部：
109\t- 条件与依据：接口相容性由源码保证（run_tc 的 SPLIT 分支与 launch_combine<TCM,false>），用于给 tc tile 定价单 token shape；无生产晋升记录，不按配对新建节点。
110\t
```

上述 `events.jsonl` 均位于对应 run 的 `attempts/<attempt>/`。FMHA 回显的 1–204 行与当前 workspace 逐行一致；Sparse 回显的 103–109 行也逐行一致。目标文件相关 Bash 只有 Sparse 的 `awk` 表格管道符计数调用（事件 29473 / 29475），结果为权限拒绝，没有实际行数输出。

规范化与版本核对：

- FMHA 前一 worker `8bd974a0-cfb3-4620-9f85-5a8144242f02` 的 workspace/original 为 389 行、SHA256 `8ed89bfa5437b35b6590aea30b384572f2eeaa5cdf1812c920b87f9b0d5f0304`；normalized 为表中 204 行版本。它自己的 receipt 仍绑定 389 行原稿 SHA256。本 reviewer 的 workspace、base checkpoint `35e911f3e04548299f5e701ab38aae5a9acff0003b8ea562f79a27a26bf05a31` 和该 normalized 的字节完全一致；reviewer 实际 Read 的也是此版本。
- Sparse 将原生 Write 与随后 9 次 Edit 在内存依次重放，结果与当前 workspace 字节完全一致。每次 Edit 的替换前后换行数相同（8 次 0→0、1 次 2→2）。失败 attempt 没有 original/normalized 目录；越界拒收发生在创建它们之前，不能由规范化重排行号引起。
- 两例 native result 的 `structured_output`（事件 16801 / 36259）与 `result.json.response`、`response.json` 完全相同；205 / 110 在模型最终提案中已经存在，没有被接收层改写。

两次运行各自冻结的 `runner.py:540,558,561,581` 都先以 workspace 调用 `normalize_response`、生成并校验原始证据，之后才保存 original、规范化 worker 候选；`receipts.py:204–205` 将模型选择的起止行原样交给 `read_evidence`，绑定该文件的 SHA256。冻结 `reading.py:112–115` 与当前源码的范围规则一致：使用 `splitlines()`，要求 `1 <= start <= end <= len(lines)`。尾部 LF 不额外产生可引用的空行。

因此严格拒收仍正确。当前源码已实现的有限 receipt 修复可返回准确范围错误，由原角色按真实文件重新选取；无需新增恢复机制、允许末行 +1 或由工具自动截断证据。当前冻结运行沿用主 Supervisor 已获授权的公开恢复入口。本核对仅新增本文，未改代码、冻结规则、运行状态或产物，未调用模型或恢复操作。
