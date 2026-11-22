# B24：固定新 driver 的真实独立 design

2026-11-21 UTC。通过固定 KS `6db007b3290b59f542316f12fd647028f4901696` 的公开 `Demo.create(start="design", remote=None)` / `Demo.run()`，沿原 B14 parent 执行了唯一一次真实业务 design。**长预算入口真实生效，planner 原生调用成功，但 K5 draft 格式校验失败；没有成功 Schema、独立 reviewer、implement、GPU 或完整 kernel 链。** 本轮首个明确业务失败后停止，没有另开请求重试、修改原输出或代签成功。

新增真实模型 **planner1 / reviewer0 / coder0**，GPU jobs **0**；原生报告 **142547 tokens、1.44354875 USD**，本请求 active **555801 ms / 555801282259 ns**。全部原进程及 operation 已终止，原 parent 无 active/fence，收费 slot 已交回根。CLI 费用是原报告值，非账单；原全账 tokens 和总费用仍 unknown。

证据根为 [/tmp/schema-b24-real-design-20261121](/tmp/schema-b24-real-design-20261121)。主要入口是 [机器交接](/tmp/schema-b24-real-design-20261121/b24-final-handoff.json)、[真实执行审计](/tmp/schema-b24-real-design-20261121/audit.json)、[静态失败诊断](/tmp/schema-b24-real-design-20261121/static-diagnosis.json) 和 [选定导出描述](/tmp/schema-b24-real-design-20261121/selected-export.json)。本项没有修改生产源码、提交、推送、派生代理、访问禁止的实验仓、安装依赖或运行无关测试；原 main 仅写本报告。

固定执行安装使用自己的 [Git archive 快照](/tmp/schema-b24-real-design-20261121/fixed-ks)，显式把该路径置于 import 首位；解释器复用既有 `.worktrees/.tooling/k1-evaluation/bin/python`，依赖为 PyYAML 6.0.3、jsonschema 4.26.0、官方 SDK 0.14.37。本轮没有调用 SDK。没有把 B18 的旧 77ad editable 安装冒称 6db，也未消费另一代理正在修改的 evaluation 源码。170 个相关 Git blob 逐字核验，B23 五个冻结文件均核 SHA/长度；api、budget、writing/policy、runtime、reference、evaluation、driver 与原 Template 实际模块全部来自此快照。源码 manifest SHA 为 `826c51509447e7e4b2b8c922dad9ed34604f3deb7d784882132eb07ebc70bc8e`。

已读适用 AGENTS、B21/B22/B23 报告、B23 freeze/准备脚本/inputs 以及现行 standalone、WRITING、runtime 接口。自己的 [准备与零调用预检](/tmp/schema-b24-real-design-20261121/prepare_and_preflight.py) 通过公开 `Demo.create` 使用 B23 pure-design 提案；根据根本轮指令显式设 `remote=None`，没有调用 B23 helper 中的远程配置分支或 `prepare_agate`。

准备前后原 authority 字节相同，SHA 为 `4baedd020df29b3be58f1e544e7dd6ff067c8b6ba85598bf0d1273bd37f6abcb`。当时原 7 个 operation 全 terminated、无 fence、active_run/active_since=null；累计 model3/jobs3/388841ms，剩 21 models/13 jobs/10411159ms，roles 剩 planner6/coder6/reviewer15/extract0。没有新 owner、注册、预约或 active 计时。预检通过后才发出本次唯一收费调用。

| 冻结预算入口 | 本次值 |
| --- | --- |
| design / optimize 请求 | 各 1250000ms，planner1/reviewer1，model_calls2，jobs0，tokens不限 |
| writing minimum | planner600000ms / reviewer600000ms |
| writing_finish_floor | 完整 `cost()` 向量全零 |
| implement 请求 | 1500000ms，coder1/reviewer1，model_calls2，job1 |
| verify 请求 | 800000ms，reviewer1，model_calls1，job1 |
| reference minimum | coder600000 / evaluator120000 / reviewer600000ms |
| 原 parent 限额 | 24 models / 16 jobs / 10800000ms；planner6/coder8/reviewer16/extract0；tokens不限 |

未触发的 implement/verify 预算只是新 Demo 冻结参数，不是执行资格。新 reference profile 不含 evaluation selection，qualification=null；无论预算有余量与否，均不能签 GPU 资格。

真实 target 精确复用 B23 已核 B16 资料：SM10.3，runtime name 为 `NVIDIA L20D` 加末尾 **8 个空格**，Torch2.9.0+cu130、Triton3.7.1、CUDA13.0；target SHA 为 `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f`。这是保留的真实目标/provenance，本轮未重新探测设备。原数学、ABI、domain、FP64 golden、全部 acceptance/workload、完整 forward timing、baseline/starter/evaluator 和公开近邻参考闭包均保留，原 `workflow_objective.md` 逐字不变。

本 Demo 的 Task Ref SHA 为 `43f22ed08f6b4b948250af97e303c15471dc03716094e130ba685e45bd193ebd`。它不同于 B22 的 Task Ref，是因为新的 publication ID 改变了 Ref 路径；准备脚本第一次误把两次发布的 Task 摘要应相等作为断言，在任何收费调用之前被拒。随后仅修正本项辅助检查为完整递归内容闭包比较，并复核全部 Task 原叶子字节一致；未改生产 Task、Demo/Q 或 authority。重新打开同一个尚未准入 Demo，未创建第二 Demo。

runtime 精确对象从 B21 业务 profile 的 `parameters.claude` 复用，SHA 为 **`3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530`**：CLI2.1.316、qwen3.8-max[1m]、xhigh、2.00 USD threshold、600000ms、existing_cli_auth/no-tools。官方 inspect/version/help 与已配置 auth status 成功；未读、复制或导出凭据。B21 原运输资格未被当作本次 KS writing/review 资格。

Planner 零调用预览与实际 input 都是 **414153 bytes**，含 69 个 material entries：24 正文、16 alias、29 个仅身份 host implementation。实际每项正文/alias SHA 与权限核对通过，完整 Task/Rules/Template/参考资料没有截断。五个示例请求虽冻结了 R0，design Q 本身没有 schema；实际 planner 没有收到 R0 Design、current_source、实际优化 Code、私有 evaluator 或 parent 历史/authority。预览与实际投递正文完全一致，仅独立发布的 delivery Ref 路径不同。

唯一执行命令为：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -I -B \
  /tmp/schema-b24-real-design-20261121/run_design.py
```

[runner](/tmp/schema-b24-real-design-20261121/run_design.py) 先检查固定源与原 owner SHA、持久化唯一 execution intent，再调用原 `Demo.run`；不串联私有写作流程、不修改原 phase/status。Demo 为 [/tmp/schema-b24-real-design-20261121/standalone-demo](/tmp/schema-b24-real-design-20261121/standalone-demo)。唯一 run_id 为 **`bed95147d41e42ab9e59222fae15fa73`**，operation 为同前缀 `_planner`。

| 实际原件 | 事实 |
| --- | --- |
| session | `a7e94e99-0e12-46b1-8ad4-e26185f65c88` |
| planner reservation | amount.active_ms600000，effective allowance649543ms，另保留 reviewer1 / 600000ms floor |
| 原 launch | timeout600000ms、原绝对 UTC/monotonic cutoff、`--max-budget-usd 2.00` |
| 原 CLI 终态 | subtype=success，is_error=false，returncode0，timed_out=false，spawn_error=null |
| K3 receipt | status=succeeded，errors=[]，native_rounds1；physical_retries 未报告，保持 null |
| 原进程 | child PID1145326/start_ticks163046760；supervisor PID1145307/start_ticks163046735；均已终止 |
| 实际权限 init | qwen3.8-max[1m]，dontAsk；tools/MCP/skills 均空，builtin plugin 身份由原 parser 校验 |
| 原 CLI active | 551559ms；加宿主原请求工作后为555801ms |
| tokens | input6 + cacheCreation113067 + cacheRead0 + output29474 =142547 |
| CLI reported USD | `1.44354875`；原 threshold 不是硬账单上限 |

原 RoleReceipt 为 `artifacts/a13eec6ec5ab472aa156f0f2eae164be/claude-receipt.json`，SHA **`c762d8046bad5006aa7de66528b97f03fdfc01d5bd242a59f3c68701944e2f1e`**。实际 input SHA 为 `52908fe2bbfb993aefea3d58edf6f15f93d70ca60de66f30d1afb3ea9dea05d6`，原 native terminal 行 SHA 为 `3afec5a5ec5a38239801814131732ae7c9b158fafc2b104b046a297c1d48888c`。所有 input/profile/command/events/exit/进程原件 Ref 及 SHA 列在 audit.roles[0].raw；receipt 内 original base64 逐字复核，既有原 parser 重解析一致。

原生运输成功后，K5 在 `parse_draft` 拒绝：

```text
selection must explain actual planner-granted material IDs
```

原 draft `ok=true`，包含全部12个所需字段、19个带非空 reason 的 selection 项和一个10006-byte、188行的 root YAML。其19个 selection ID **全部**采用了 `material:` 前缀，例如 `material:workflow_objective`、`material:task_contract`、`material:triton_softmax_reference`。这些名称来自同一投递中的 action catalog；而 `selection[].id` 原 parser 必须匹配 Materials entry 的 bare ID，如 `workflow_objective`。投递本身又使用第三种 `material/workflow_objective` ID。去掉 `material:` 后19项都对应实际获准材料，但**本项没有对输出做去前缀或重新准入**。

这是明确的 ID 用法错误；三个命名空间同时出现、缺少独立 bare-ID selection catalog，是可以最小改善的提示契约问题。不能据此放宽 allowlist 或偷偷解释既有失败输出。原 retained draft 为 `artifacts/3fdf03e96aed4247b6b69e065dd47c7f/draft.json`，SHA **`66f919efa8c676cd8f241cf82563c34c41c3d8e020f5c0c1b5c9f3e5596e2892`**。

生产流程在此首失败处终止，没有到 normalize/precheck、Schema publication 或独立 reviewer。原 design result 是 failed/recovery=new_request，schema=null、scope=null，structure/permissions/semantics/publication 均 unknown 且无 receipt；不能称为“已成功 design”。Demo 也以 phase=done/status=failed/recovery=new_request 结束，history 只有这一项 design；没有进入 implement 准入。通用 checkpoint 标签 `writing_published` 不代表已发布 Schema，实际 result/state 才是事实。

为定位后续最小问题，本项只在自己的诊断目录，对**同一原始 raw、未改一字**调用原 Template 做静态检查；没有账户/API admission、模型或生产 state 写入。这不是本 run 的 precheck/independent review，静态结果不能反向填进原业务结果。原 Template 另报以下8处长度超限：

| 原 raw 位置 | 实际字符数 / 既有限制 |
| --- | --- |
| Math_reshape.Thesis | 720 / 300 |
| Work_schedule.Thesis | 469 / 300 |
| Tile_pipeline.Thesis | 522 / 300 |
| Work_schedule.Step.MAIN.Decomposition.Strategy | 318 / 150 |
| Partition.row.meaning | 51 / 50 |
| Storage.row_block.meaning | 69 / 50 |
| Role.MAIN.Info.Using_verb.load.row.describe | 165 / 150 |
| Role.MAIN.Info.Using_verb.store.out.describe | 183 / 150 |

完整原错误保存在 [static-diagnosis.json](/tmp/schema-b24-real-design-20261121/static-diagnosis.json)。精确未改 raw 为 [diagnostics/original-raw/root.yaml](/tmp/schema-b24-real-design-20261121/diagnostics/original-raw/root.yaml)，SHA `8a440621fdf5d799fa52024c711800b731dff7bcb6d09d81b2917b5e3a83a3ac`；该文件是**失败模型原稿，不是通过的 Schema**。静态检查已在首轮 Template 错误处停止，没有手工缩短、反复修稿或推定其余结构/语义通过。

根要求的实际输入/响应契约补审见 [prompt-contract-audit.json](/tmp/schema-b24-real-design-20261121/prompt-contract-audit.json)。实际 `delivery/writing_input` 只有 `action_catalog/format/locked_contract/note/objective_id` 五个字段，**没有 bare-ID selection_catalog**；仅 `objective_id=workflow_objective` 和 METHOD 的 `selection [{id,reason}] including workflow_objective` 暗示 bare ID。19个带前缀 ID 的对应 bare ID 全部在本次 planner allowlist 内，且19项 action_catalog Ref 均与该获准 Materials Ref 相同；逐项原 Ref、角色授权、投递ID和映射列在该补审的 `mappings`。

本次响应没有单独传入带 enum/maxLength 的 JSON Schema，原 CLI command 也没有 `--json-schema`；输入只声明 `output_protocol=ks.claude.output.v1`，外层格式由 system prompt、draft字段由 METHOD 文本说明，K3/写作 parser 在返回后验证。**实际所有投递正文均没有 `maxLength` 字样，METHOD 与 writing_input 也没有300/150/50数值长度摘要。** 真正限制来自固定 [Template text_policy.py](/tmp/schema-b24-real-design-20261121/fixed-ks/schema_template/tools/text_policy.py:34) 的 `BUDGETS`；该文件在实际 payload 中仅以 host implementation 身份投递，正文没有给 planner。因此不能把这些数值限制称为已由一个显式响应 schema 告知并强制模型遵守。原公开 docs 仍保留其“一句/简述”等字段说明，未被删减。

后续最小建议是：在新固定 planner writing input 中显式列 bare-ID `selection_catalog`；METHOD/WRITING 明确区分 bare selection、`material/` delivery、`material:` action IDs；从原 Template 的权威 `BUDGETS` 投影出数据化字符/行数限制，并要求模型在原输出内自检。保持 strict parser、Template 与原 Task 判据不变。本项只给建议，没有改生产代码，也没有以新 scope 继续调用模型；未来修复与真实验证由根另行明确安排。

原 parent 新增且仅新增1个 operation、1个 request；旧7个 operation 和全部旧 request 对象逐字保持，包括 B17 append-only sequence2 token-unknown 勘误。最终本项快照如下：

| 账本项 | 结算后事实 |
| --- | --- |
| 累计 / 剩余 models | 4 / 20 |
| 累计 / 剩余 jobs | 3 / 13 |
| 累计 / 剩余 active_ms | 944642 / 9855358 |
| 剩余角色 | planner5 / coder6 / reviewer15 / extract0 |
| 本轮精确 active 增量 | 555801282259ns；同新 request 精确ns一致 |
| 全账 tokens | unknown；370133 仅为已知数学下界 |
| 全部 operation | 8个全终；active_run=null、active_since=null、fences=[] |

原 descriptor/首次限额没有改变。本项结算后 snapshot SHA 为 `f5cf57f0bc32b4f218d9a81b7c21a1e18bfa2e3872de1ef819ddaaa1b27b0cad`；slot 已交回根，后续收费前须读取真实 owner，不能把此快照冒称未来余额。各旧 scope/result/Q/profile 均未修改，GPU qualification 仍未建立。

[终态恢复验证](/tmp/schema-b24-real-design-20261121/recovery-check.json) 分别通过原 `Demo.run()` terminal replay 和原 `api.design(continue_run=原step-0)`：结果相同，operations/requests/descriptor/active_ns/active_ms/paused 不变，新增模型0、GPU0、operation0、active_ns0。只有原 owner 的正常 idle clock 基准允许刷新，未重新执行已完成 planner。

[design-selected-export](/tmp/schema-b24-real-design-20261121/design-selected-export) 通过原 `import_bundle` 导出选定 result、原 Q subject、完整 Task/Rules/Materials、原 draft、RoleReceipt 与全部原生证据闭包，共116个文件、5926424 bytes；export 描述 SHA 为 `02d8f6c5e8e7cf31025f2194b25c7d80b22b29594560ba7c0284008c537d835a`。公开 Ref/SHA 和每文件清单均已校验，未携带运行时 authority、其私有路径或认证材料。该导出可用其 `roots`/`subjects` 经公开 `import_bundle(Bundle(source), destination, roots, subjects=subjects)` 消费作为**本次真实失败设计证据**；没有合法 Schema Ref 可直接作为通过的 fixed-R design 使用。

验证范围为170 Git blob/5文件 freeze、实际 import/Template来源、原 Task完整闭包、exact profile及官方只读认证检查、实际角色输入/原收据/终态/进程/用量、旧账不变、零调用恢复、选定闭包导出与一次未改原稿的静态 Template 诊断。未跑全库测试；未把 CPU 诊断、B21 native资格或原生success替代本次 Template/语义/独立review/GPU门。**本轮完成了真实独立尝试与诚实收口，业务 design 和整体 kernel 链均未通过。**
