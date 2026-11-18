LIVE08 — 非 JSON 验收材料被 `relative_to` 扫描误判

结论：已修复当前产品把 Python 验收代码强制按 JSON 解析的问题。真实 `result.response` 的临时副本已越过原失败点，但**原完整回执仍不通过**：它的 6 条 `comparisons.output` 全是说明句，不是当前 `Golden_reference.Tolerance` 的精确键名。这一后续拒绝符合现有契约，未放宽门禁或代改模型提案。新 run 需要模型自行生成正确绑定，并完整完成 P1–P6 与独立审阅。

真实对象为 QSA r02 run `388e08c1-eef9-463d-bc95-ee9bfee09cb6`，P1 worker attempt `bf98dff0-9de7-4e91-b597-859828120fc3`，冻结快照 `2ca75e82d3c85056c8d795f78a8997663b39eae44b9bbd8735da170fa39d043e`。最终原生结果为 exit 0 / success，提案 decision 为 complete。只读取最终公开提案、运行记录、源码和稿件，未读取 events 或模型隐藏推理。

根因及修改：

- [comparators.py](../../schema_extract/tools/comparators.py) 的 `declarations()` 原先扫描所有 acceptance 文件，只要文本正则命中 `"relative_to": "..."` 就执行 `json.loads`。固定 Python 比较器 `correctness.py:334` 的字典成员触发了这一分支；这不能证明整份文件为 JSON。源描述同时绑定 metadata.json、correctness.json 和 correctness.py 合法，README 原示例也允许 acceptance.md。
- `declarations()` 现在只对 `.json` 后缀文件（大小写不敏感）做结构化解析，先严格读取 JSON，再按已有具名输出及 `tolerance.relative_to` 逻辑定位声明。坏 JSON 即使不含该文本也拒绝，错误为 `invalid acceptance JSON: PATH`。Decimal 阈值比较、逐输出绑定、声明锚点、基准与公式矛盾检查均保留。Python / Markdown 按普通固定来源证据使用，不解释程序，也不把任意文本当作声明。
- [receipts.py](../../schema_extract/tools/receipts.py) 的 P1 入口和回执字段未改；真实文件哈希、来源范围、行号、原文引用、证据类别及独立 reviewer 要求仍按原逻辑校验。
- [README](../../schema_extract/README.md) 增加一段验收格式说明；[P1 提示](../../schema_extract/prompt/phases/01_contract.md) 增加一段 `comparisons` 的精确名称、JSON 声明锚点和无声明时 `[]` 的填写说明。旧提示只要求逐输出核验，未解释这两个字符串字段应填键名。固定 schema 字段与语义规范未改。
- [定向回归](../../schema_extract/tools/tests/test_source_gaps.py) 扩充混合 JSON / Python / Markdown 场景，证明非 JSON 材料可作真实 receipt 证据；同一混合场景仍拒绝错误 `relative_to` 与 `atol`，坏 JSON 两种情况均拒绝。

真实副本验证：把冻结来源的全部 463 个文件、冻结 rules、该 attempt 的 original 与 normalized 目录复制到临时目录，保持来源绑定和稿件字节原样。用未改的 `result.response` 调用 `normalize_response`，结果与已有 receipt.json 完全相同，再调用 `validate_receipt`；仅在内存中的状态副本恢复该已结清 attempt 当时的 active 登记，没有更换 run_id、nonce、来源摘要或提案。旧比较器从冻结副本加载，新比较器使用当前产品；两者使用相同冻结语义规则和稿件。

| 检查 | 实际结果 |
| --- | --- |
| 旧比较器 + 原提案 | 重现 `explicit relative_to declarations require readable acceptance JSON: .../atrex_bench/eval/correctness.py` |
| 新比较器声明解析 | 通过；正确读取 correctness.json 的 `out` 与 `input.out` 两组声明，各有 32 处；首次行号分别为 28 / 49 |
| 两组 JSON 声明的内容 | 均为 `relative_to=candidate`、`atol=0.000032`、`rtol=0.02`；Python 来源仍保留在绑定与固定 manifest 内 |
| 新比较器 + 同一原提案完整回执 | 拒绝：`comparison bindings need a declared output and unique output/source_output pair` |
| 同一 normalized 稿的冻结 P1 `check_stage_snapshot` | exit_code=0，diagnostics=[]，notices=[] |
| 验证后核对 | 真实 state、inputs.lock、attempt、result、original/workspace/normalized 稿件摘要均不变；463 个冻结来源文件与冻结 rules manifest 均未变 |

后续拒绝的具体证据：当前 Tolerance 只有 `result` 一键。提案第一条 output 是以 `root.yaml Golden_reference.Tolerance.result: abs(...)` 开头的长说明，source_output 同样是多句说明；另五条还把 Constant、Input、Compute、Hardware 等事实放进 comparisons。6 条 output 均不等于 `result`。必须由新 worker 按实际 JSON 输出身份、当前 Contract 输出和声明行证据重新填写，由 reviewer 独立判断映射；本次没有改写任何一条提案来制造通过。

关键原始摘要：

| 文件 | SHA-256 |
| --- | --- |
| result.json | `32ad3431b4d8a1bd205c3ff48b4984e0d2d313671e7505dd11b6488585de17df` |
| original / workspace / normalized 中的 root.yaml（三者相同） | `a56c8f8da992737a621d46ba62f72384dba95b6031e86a2d190181882e88c1a6` |
| inputs.lock.json | `827efb33924151ee990a15c483330e734477da7d9313d42b6d63bfd3feb810d9` |
| state.json（本次验证前后） | `e29711747c9aa721d7ad31dde02abb750cf51d94e721029052c1d2ac4a4f12da` |

定向测试命令：

```sh
python3 -B -m unittest schema_extract.tools.tests.test_source_gaps schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_current_contract_comparator_keeps_output_and_operand_evidence
```

结果：8 tests，3.180s，OK。`git diff --check` 通过。没有运行全套测试、GPU 或真实模型。临时回放脚本与结果放在 `/tmp/live08_replay.py`、`/tmp/live08_replay_result.json`，副本目录自动清理；它们不参与产品或任何 run。

新 run 复用旧稿的可行性建议（仅只读核查，未执行）：

1. 用修复后的当前产品和完整原 source descriptor 新建 QSA run，先 `extract ... --queue`，采用新输出目录并保持已授权的 Claude Code / qwen3.8-max[1m] / max 配置与足够的完整流程预算。旧 r02 的执行器被冻结，普通 resume 不会采用本次修复，不能改旧快照以省一次新建。
2. 新 run 尚无 P1 调用时，用现有 `maintain NEW_RUN --issue-id ... --tasks FILE` 提交明确的 P1 / root / patch 任务，fields 限 P1 允许字段。在 reason 中内嵌上一份 Qwen root.yaml 原文、旧 run/attempt 和摘要，并明确“仅作撰写参考，未通过回执或独立审阅；逐项回查新 run 固定来源；由新 worker 写入自己的授权 workspace；不得继承旧回执、批准或证据索引”。同时点明本报告的 comparisons 填写错误。
3. 不仅给旧 run 的文件路径：新调用只登记当前 workspace、来源、规则与材料阅读目录，旧 run 目录不是正式材料入口。也不要把旧草稿加进 acceptance / reference / scope 等来源角色。reason 内嵌原文使用已有任务文本通道，不新增 seed、迁移或外部材料系统。
4. 该通道可以让新 P1 参考旧稿写作，但不会把旧稿装成初始 checkpoint。模型仍须完成当前提案，随后完整经过 P1–P6。`maintain` 命令会申请后台 Runner；排队后提交该请求即可，不需要同时再启动另一个 Runner。

上述建议依据现有代码：`maintenance._validate_tasks()` 校验明确 scope/fields/reason；`submit_request()` 把原始文本随新 run_id 与 input_fingerprint 封存；`mark_drained()` 对尚未完成的 P1 设置 waiting，`_resume_normal()` 经 `_normal_feedback()` 把 reason 送入普通 worker 的 feedback；`prompts.render_prompt()` 将该 feedback 原文放入 invocation。`cli.py` 的 maintain 分支启动同一冻结 Runner，维护不继承其他 run 的 checkpoint，也不授予阶段批准。此处只核查机制可行性，没有提交维护请求、制作 task 文件或启动 run。

本次只改 comparators.py、test_source_gaps.py、README、P1 阶段提示和本报告。没有修改 receipts.py 的既有未提交变更，没有修改 `.schema_extract` 中任何 snapshot/state/checkpoint/原稿，没有修改模型撰写的 schema，没有操作 FA4 r02、撤销 LIVE01/05/06/07 工作或提交 Git。验证完成后停止写入。
