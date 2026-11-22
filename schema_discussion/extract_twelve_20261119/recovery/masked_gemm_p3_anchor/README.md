# masked GEMM：P3 绑定实现入口回执恢复

原 run：`c8dd5678-2935-4efa-943a-02e848d03611`；算子：`fp8_mxfp4_grouped_gemm_masked`。本次只控制这个原 run；没有控制其他算子或 monitor 时钟。

## 根因与停机依据

恢复前为 `P3 / root / reviewer / normal / invalid_receipt`，失败调用 `73af655b-77bb-4b4c-9bdd-af53d68ef4e3`，checkpoint `c00e34f2e7cbd89d1e3e0a24002812d49a15c52c36b80ef56df952a4d5d5a384`。P1、P2 已批准，P3 approval 为空。旧响应虽为 approve，但未被 accepted_receipts 接纳。

实际调用为 **P3 6/32**，实质退修 1 次，`receipt_repairs=3`、`invalid_receipts=3`；冻结 `retry_limit=2`，无调用追加、裁决或签收。这是同阶段累计补正计数达到停止条件，并非 32 次调用用尽。

绑定的 `entrypoint` 和 `host` 均为 `candidate/solution.py::Model.forward`。最新回执共 73 条源码证据，其中 `[2]` 用 `category=host` 引用该文件 116–135 行（从 116 行即函数定义开始，quote_offset=0），`[3]` 用 host 引用 87–114 行校验逻辑；全部 implementation 证据都在 `candidate/kernel.cu`。因此确实缺少 implementation 类别下的绑定入口引用。冻结 `receipts.py:402–406` 要求每种必需类别至少覆盖自己的绑定路径；CUDA 下游实现和已有 host 类别不会替代这一项。所有源码证据的路径、行范围和偏移均有效，错误在类别覆盖，不是越界、绑定漂移或验证器误判。

源码入口 116–135 行确实执行参数解包、校验与缓存、调用 `_fn(..., _raw_stream(...))` 和返回 None；上游 Python 入口与 CUDA 下游实现应由 reviewer 自行核实后建立实现证据链。同 run 的 worker 调用 `54454743-f272-4c71-88c0-f3432fd282d7` 已独立使用该入口的 implementation 与 host 两项证据完成补正并获接纳，说明无需工具修复或放宽门禁。Supervisor 没有代写新证据或判断。

原 P3 调用链：

| 次序 | 调用 | 角色 | 结果 |
| --- | --- | --- | --- |
| 1 | ee1a4eaa-e935-4db4-9eb8-364357b03bb2 | worker | 缺 implementation 绑定入口；合法草稿由 Runner 保留；补正累计 1 |
| 2 | 54454743-f272-4c71-88c0-f3432fd282d7 | worker | 加入绑定入口 implementation 证据，complete 接纳 |
| 3 | 2d771b0d-8bae-4690-9e06-ac7b903420c3 | reviewer | 有效 revise，实质退修累计 1 |
| 4 | 4e08ac98-c35d-4552-b079-c8fafc975e42 | worker | 修改完成，complete 接纳 |
| 5 | fdd0e1f1-3f39-4d36-9ed7-a47d9def48f8 | reviewer | root.yaml 第 447–513 行越界，实际 512 行；补正累计 2 |
| 6 | 73af655b-77bb-4b4c-9bdd-af53d68ef4e3 | reviewer | 末行修为 512 后，暴露缺 implementation 绑定入口；补正累计 3 后停机 |

两次 reviewer 响应的 73 条 source_evidence **类别、路径、行号和偏移完全相同**，note 文案有变化，因此不能说完整 JSON 完全相同。缺入口原已存在，先被产物行号错误遮住。另有后置门禁风险：最新 handoff 为 **2684 字符、6 非空行**，超过冻结 `receipts.py:462–464` 的 1200 字符上限；此前仍未执行到这一检查。该诊断链已保留，未开展额外框架审计或修改代码。

最新原生结果 completed / exit=0 / success=true，无解析错误，session ready，真实 CLI exit=0，后代已结清、termination.confirmed=true、survivors=[]。14 条历史 attempt 用量均 settled。正式 inspect 确认 active/unconfirmed/unsettled 全空，旧 Runner 不存活且锁 free，launch finished，无 pause/cancel/hold/maintenance 屏障。

正式 `audit --attempt 73af... --verify --json` 退出 0，verification.errors=[]；其状态为 **partial**，unavailable 明确记录 `receipt_not_accepted; artifact validation not completed`。审计证明可得证据一致，不构成旧回执批准或完整产物门禁通过。

## 一次受控恢复

授权来自本会话用户持续指令与 Supervisor 明确委派：普通问题按推荐处理并记录；原 run 已结清、无控制屏障、既有 32 次预算内允许一次有针对性的普通恢复。Supervisor 也已获告后置 handoff 问题，并明确继续这一恢复策略。

北京时间 **2026-11-20 06:12:10**，通过标准 CLI 执行唯一一次 `resume c8dd5678-2935-4efa-943a-02e848d03611`，显式指定 repo 和 `.schema_extract/v2`，完整 argv 保存在 `resume_launch.json`。未调用 repair-runtime，未重开 run，未增加额度，未直接改 state/旧回执/批准/源码/规则/提示，未修改 kernel 或进行 GPU 测试。普通 resume 没有自由反馈参数；本次仅由原 Runner 向新调用传递已经登记的真实入口缺项诊断和完整动态材料，未注入或改写旧证据。

新调用为 **`7a388176-110f-48b7-8275-a0796adc2e83`**，仍是 `P3 / root / reviewer / normal`，第 **7/32** 次调用。精确 session **`bca9e483-b406-44e1-8dae-114a217c5cbd`** 同时获 attempt、invocation.resume_session_id 和原生 system/init 确认。新 prompt 包含旧失败 attempt ID 与 `missing bound implementation entry anchor` 原诊断。

06:12:48 的针对性核验全部通过：输入锁、来源/规则执行器身份、配置与预算、checkpoint、选择与草稿、P1/P2 批准、accepted_receipts、14 条旧账目、旧调用证据字节保持；调用只增 1，补正/退修/裁决/签收计数未清零。Runner 存活且锁 held；新调用为正常在途，unconfirmed 为空。旧无效 approve 仍未接纳，P3 approval 仍为空。

观察时已有 66 条原生事件，尚无新 result。**本次确认的是恢复运行，尚未宣称 P3 获批。** 新 reviewer 仍需自行修正证据，并满足 handoff 及全部后续原门禁。若再次失败，先分析新结果；本次不循环 resume。

## 证据文件

- `before_state.json`、`before_inputs.lock.json`、`before_inspect.json`：恢复前权威状态副本、冻结输入锁与正式观察。
- `diagnosis.json`、`source_anchor_coordinates.json`：真实账目/时间线/绑定角色/全部源码坐标与引文，以及历史证据摘要。
- `failed_*.json`：最新失败响应、恢复诊断、真实退出、完成状态、调用和会话证据。
- `before_audit.json`：正式有限范围审计结果（partial，无错误）。
- `resume_launch.json`、`resume.stdout`、`resume.stderr`：唯一一次正式恢复的命令与日志。
- `after_inspect.json`、`after_state.json`、`resumed_attempt.json`、`resumed_invocation.json`、`resumed_prompt_diagnostic_excerpt.txt`、`recovery_verification.json`：恢复后的运行状态与原计数/会话/证据保留核验。
