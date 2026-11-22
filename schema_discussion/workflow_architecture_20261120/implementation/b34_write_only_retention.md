# B34：write-only 输出与原稿保留

2026-11-21 UTC。在 KS worktree `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`、分支 `feat/schema-workflows-v1`、基线 `ad90f7b9adb2f83d01302ea9fa4462390f464745` 完成窄修复，源码已冻结，未 commit/push。真实模型调用0、GPU jobs0、真实 owner 操作0；无新权限验收、reviewer 或效果验收。未访问禁止实验仓、修改 AKA、原 extract 或 Template，也未安装为活跃依赖。

## 结果与用户可见行为

只对冻结为 `write_only` 的 planner，外层 `reason` 可省略，解释直接使用实际 `content.reason`；不可用的外层 reason 与额外外层字段不再阻止保留 Schema。额外字段不提供可信运行时事实。protocol、role、assignment、status、kind 与 content 类型仍校验；原生会话、进程、权限、模型身份、费用和用量判断保持。选择器写入原生 intent，并与冻结输入的 writing mode 对照。strict、coder、reviewer 继续使用原完整外层字段协议。

写作层先保留实际回复，再决定能否发布。现有 `candidate_draft` 指向的 `draft.json` 新增 `native_responses` 字符串数组，保存原 terminal/assistant 文本；能解析的 content 与各 raw 原文在失败时也保留。未完整解析的 JSON 仍可从该文件和完整 native stream 读取。失败、身份错配、未知用量不因此变成成功；没有对旧调用重试、续额或改账。

已授权且成功的调用继续使用现有 Schema/Scope/Publication：可读 YAML 完成写出；坏 YAML 原样保存并给 partial 与诊断。`retained_files` 保留 raw 和独立 dispatch 原文，`call_artifacts` 保留原生记录，现成 export 闭包包含这些产物。原回复在现有 draft 中以字符串保存，避免把 `ks.claude.output.v1` 根对象误当作规范工作流对象；没有改通用 artifacts/contracts 或建立第二套恢复平台。

短产物约定已与 parser 对齐：每条 `raw:[{node,text}]` 都是 `<node>.yaml`；Markdown 用独立 `"dispatch":"完整 Markdown"` 字段，保存为 `dispatch.md`。合法的 `raw.node=dispatch` 仍是 YAML，不依据名字或文本猜格式。独立 dispatch 做 UTF-8 可写性检查；没有增加语义门。

## 原 B33 只读重解释

离线脚本 [reinterpret_b33.py](/tmp/schema-b34-write-only-retention-20261121/reinterpret_b33.py) 在独立 [reinterpreted-b33](/tmp/schema-b34-write-only-retention-20261121/reinterpreted-b33) 运行。它读取原 native events/profile/input/intent/exit，核对原 assignment/input/profile 摘要，确认原 exit0、无 timeout/spawn error，然后只从原 `fixed_assignment.writing_mode` 构造当前 parser 的离线选择器。原 intent/输入/结果没有补字段或改写。

脚本显式禁止 `ClaudeRuntime.admit/invoke/settle` 和 `OwnerAccount.for_context`。使用独立、标为 `not_a_runtime_receipt=true` 的 interpretation fixture 调用 `_draft/_normalize/_publish`，不执行公开 API、旧 run continue 或实际结算。此处是新解释下的产物测试，**不是旧 failed run 的恢复成功或第二次模型调用**。

[summary](/tmp/schema-b34-write-only-retention-20261121/reinterpretation-summary.json) 显示：旧选择器仍为 `invalid_role_output`；当前 write-only parser errors=[]；产物状态为 **partial**，仅保留原 dispatch 的 YAML 可读性诊断和无覆盖映射提示。permissions/semantics=not_run。费用仍引用原一次调用 32,392 tokens / USD0.53898625，不产生新账。

| 原节点 | 新离线 fixture 原样文件 | bytes / SHA-256 |
| --- | --- | --- |
| root | [root.yaml](/tmp/schema-b34-write-only-retention-20261121/reinterpreted-b33/artifacts/3097a951230441a09e3aa3260f919ff1/raw/root.yaml) | 8076 / `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c` |
| dispatch | [dispatch.yaml](/tmp/schema-b34-write-only-retention-20261121/reinterpreted-b33/artifacts/3097a951230441a09e3aa3260f919ff1/raw/dispatch.yaml) | 748 / `9452d314ed8554bc8dc7c085f166096143cea3ad5d5c52bd49f750bae8f5e000` |

两份文本与 B33 已逐字提取的原件完全相同。root 可供下游单独读取；包含错误标注 dispatch 的整个目录仍不宣称全部可读。[draft.json](/tmp/schema-b34-write-only-retention-20261121/reinterpreted-b33/artifacts/8da9c014e3d743d9bc53dee01ff458ee/draft.json) 保留实际 content、完整 terminal reply 与 assistant 文本；[rationale.md](/tmp/schema-b34-write-only-retention-20261121/reinterpreted-b33/artifacts/d70d4033718d46889186aa1eb1b5f2bb/rationale.md) 保存解释。没有宿主手改 Schema 或将第二 raw 改名为 Markdown。

## 原件不变与针对性验证

已对 B33 design（含原私有运行时/账本文件）、retained-native-draft、summary/return/source manifest 共 **90 文件**逐字核对，无增删或 SHA 变化。见 [原始清单](/tmp/schema-b34-write-only-retention-20261121/original-b33-hashes.json)（SHA `ec0c2d9ac1b996f6b7d608a5b250986bb5531de63260bd1987aba773545cfd39`）与 [核对结果](/tmp/schema-b34-write-only-retention-20261121/original-b33-unchanged.json)。核心原件：

| 原件 | SHA-256 |
| --- | --- |
| `design/result.json`（仍 failed） | `b761acd8aa7c190dd680604cc2f0fa2fc762523852200f8d13c585857835a0dd` |
| 原 `claude-receipt.json`（仍 failed / invalid_role_output） | `4134b37ca82a814ecd98700c0fabc71f1d4ad9915360fd4d7911e640b23e5699` |
| 原 `claude-events.jsonl` | `0b47ea08f0d72cf1e0f63a4fd1c2ea0b4bc2d796895dd75c0d6667cb346941c9` |
| 原 `usage.json` | `e4881c5cc3d7679e63c9f5ae43b76c93574c28fa35f3f81484706e38157a2af6` |

只用现有 `.worktrees/.tooling/k1-evaluation/bin/python`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，未安装/下载或重跑全库。**60 个针对性用例通过**：

- [targeted-fixed-tests.log](/tmp/schema-b34-write-only-retention-20261121/targeted-fixed-tests.log)、同名 XML：32 passed / 55.60秒，write-only、原文保留/导出/continue、缺/坏外层 reason、额外字段、dispatch 明确格式、wrong role/assignment/kind、nonzero exit、native failure、unknown usage，以及 strict/coder/reviewer 的 exact-field 负例。
- [compat-tests.log](/tmp/schema-b34-write-only-retention-20261121/compat-tests.log)、同名 XML：27 passed / 79.68秒，原 strict design/optimize、原 planner/reviewer 失败、原 coder 成功/重放、原 runtime 全类失败和冲突用量路径。
- [permission-test.log](/tmp/schema-b34-write-only-retention-20261121/permission-test.log)、同名 XML：1 passed / 3.29秒，write-only 的实际 native tools 权限不符仍失败、保留原回复。

最初测试暴露了独立原回复 `.txt` 被通用闭包按根 protocol 校验的问题；已收敛为现有 draft 的字符串字段，修复后上述失败用例全部通过。首次失败日志仍保留，没有抹去。`git diff --check` 通过。测试中的 fake CLI/owner 为独立 CPU synthetic fixture，不是新真实调用。

## 冻结源码

精确 [source-manifest.json](/tmp/schema-b34-write-only-retention-20261121/source-manifest.json)，SHA `f50778c1378e10f59960b2327678f8975930f2160842313955dd7f92555063da`；八文件副本在 [frozen-patch](/tmp/schema-b34-write-only-retention-20261121/frozen-patch)，[source.patch](/tmp/schema-b34-write-only-retention-20261121/source.patch) SHA `1197ee7fff1a0ac89653d89bf4679afe567e471b059e2dca56d27b73d97a2080`。仅下列源码由本 agent 修改，根可精确提交后再固定后续 optimize 版本：

| 文件 | SHA-256 |
| --- | --- |
| `kernelschema/WRITING.md` | `06ebef94e451ea6ac84a884554a407a7d9cf4852e10597c8686c089d8af0d307` |
| `kernelschema/_writing_policy.py` | `a8a17caddc3dbc981f3ee58cd4a2550769a4e5317dc59c2f54c858bdd8a553d6` |
| `kernelschema/runtime/INTERFACE.md` | `09697ec238f3b5dd584088eae9a535373314d04967270e598426067a5d250ce2` |
| `kernelschema/runtime/claude.py` | `2212e9d44ffcccba7e38be2d4f82b9aa0c05e854fc2928558576f024ffa3a41e` |
| `kernelschema/writing.py` | `8c848bc7bbdbb66e0d194d35b7c02b0e3b79670e96b25457f52950a0bfe41211` |
| `tests/workflows/fixtures/writing/fake_claude.py` | `f480793ad0719bc4d03ad568967f45da2504987523bb0f959059d98eb5b6dc83` |
| `tests/workflows/test_claude_runtime.py` | `5eed59a2cf9fd1a461ff7dade1ccb93cdbb5d45a05e2782f03cbffd12be69367` |
| `tests/workflows/test_write_only.py` | `69f2263bbe93fa0d0e9588df2824650f62ba0b17a3f6c73b894a18e65821f5e0` |

不提供将旧 failed result 改为 completed 的能力。已有产物的离线提取/重解释只写新目录并标明来源；后续真实 optimize 应使用根新固定的提交与一次新的授权调用。
