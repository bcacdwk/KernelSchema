# GEMM P5 绑定入口回执缺项：单范围恢复记录

本次已通过正式入口恢复 `P5:split_k_reduce` 一次。原 Runner 已消费请求，新 reviewer 已真实启动并收到原诊断；尚未取得本次新回执，不将启动视为节点批准。

- Run：`2b64ba84-64b3-4d10-86fe-65c351ce571c`，算子 `bf16_nvfp4_gemm`。
- 原失败 attempt：`dafb1609-4488-4cc3-9d70-0bc770cc508f`。
- 唯一恢复请求：`5c214e5e-794d-4078-a0ec-4edd2c2d764b`，`consumed`。
- 新 reviewer attempt：`2ec6e8ac-f534-4c44-8e19-7a87aa84a390`。
- 正式命令于北京时间 2026-11-20 13:19:10 返回 0；13:19:11 由原 Runner 消费。

## 原因与准入

最新 proposal 与 Runner 规范化 receipt 已将 `split_k_reduce.yaml:272–276` 修为 `272–275`；实际 raw 共 275 行。重新核验最新 62 条源码锚点与 40 条产物锚点均通过固定字节、引文和行范围检查。

下一道真实诊断是 `missing bound implementation entry anchor`。冻结绑定的 entrypoint 为 `candidate/solution.py::Model.forward`，函数实际范围是 77–112 行。回执对该文件 77–81 和 111–112 行的两条引用均标为 `host`；58 条 `implementation` 引用全部指向 `candidate/kernels.cu`，因此缺少 **implementation 类别下的绑定入口路径**。proposal 无独立 symbol 字段，函数身份通过冻结绑定和源码 AST 核实。冻结 `response.md` 明确要求“每类先覆盖对应绑定入口”，冻结 `receipts.py` 以 category 与绑定 path 检查；此次为普通回执缺项，无证据表明是验证器误判。Runner 对 proposal 补充实际 quote、sha256、source_id，原 category/path/range 保持一致。

停止分类 `call_limit_or_single_decision` 对应 `receipt_repair_limit_reached`，不是节点 32 次额度耗尽。恢复前为 `calls=4/32`、`invalid_receipts=3`、`receipt_repairs=3`、`revisions=0`；本次未追加或退还次数。

原调用为 `P5 / split_k_reduce / reviewer / normal`，精确 session 为 `7601fdcb-278c-4db2-903e-cc57432a32e4`。冻结 `verified_result` 重读原生事件和真实 exit，确认 exit 0、唯一 native success、schema/nonce 与保存响应一致、session 精确匹配。账目 `settled=true`；gate 的 `descendants_reaped=true`，Linux 进程身份复核原 gate 与已登记后代均已退出，目标没有 active/unconfirmed/unsettled 调用。

冻结 `scoped_restart_evidence` 成功，`settlement_required=false`。当前实例仍为 cfg144 的 `ws10sk::w4a16_ws10sk_kernel + ws10sk::sk_reduce_kernel` 链，generation 0、plan_revision 1，首次派发的 `wait_for=[umma_single_cta]` 沿用原准入。当前 raw 无 Design `from`；实际内容依赖只有批准 root，SHA 为 `34b026f90a32e90c183cda8d61093533883f3a32ec567de3f6dc6eb1296a8134`，有效来源身份为 `fde3ecd18546760eb42f8fa0e21f7512a66123f48ce218c7a65aee57de1e170a`。P1–P4 批准有效，无 pause/cancel/hold、待处理控制或维护屏障。

原 handoff 为 2841 字、8 个非空行。冻结 1200 字/20 行机械门禁仅适用于 P2/P3/P4 的 approve，**不适用于本次 P5**；不将长度推断为后置失败，也未改写动态反馈。

## 唯一控制操作与恢复后确认

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_extract/extract_yaml.py resume 2b64ba84-64b3-4d10-86fe-65c351ce571c --scope P5:split_k_reduce --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

命令一次成功，无 busy 重放。原 Runner 新增且仅新增上述一个 reviewer 调用。新 gate 已 released，PID `1859610` 的登记身份存活；原生 init 确认原精确 session 和 `qwen3.8-max[1m]`。新 prompt 第 338 行包含真实诊断 `missing bound implementation entry anchor`，prompt SHA 与 invocation 一致。

定向前后核对通过：

- 目标 calls 从 4 增至 5，上限仍为 32；invalid_receipts 与 receipt_repairs 均仍为 3，其余退修/裁决/签收计数不变。
- 健康兄弟 `pair_two_pass` 仍由原 attempt `8ffd010b-8170-4b86-89ed-7bf199840e7a` 运行；gate 身份未变且仍存活，calls 仍为 4，未收到暂停、取消或强制重启。
- inputs.lock、source/config/rules/executor 身份、原 Runner、实例、generation、写作安排、原稿及 source context 保持；原有效 P1–P4/节点批准、accepted receipts 和已结清调用账目保留。
- 343 项原登记、回执、结果、候选与 checkpoint 文件摘要核对无变化；run/merged checkpoint 仍为 `282bba4e4a7a0bc9c82c1a8e48975fd622e635d62f31a1d45a800fc1665283e2`，目标稿件 checkpoint 仍为 `0884de4fe98f5115c9b8cfe8114b30784e0041614bc66e4cc8be87c6857090a7`。

本次未执行无 scope 恢复、维护、增额、直接 state/旧证据编辑、工具或 kernel 修复、GPU 操作、新建 run，亦未执行 monitor ack/scan 或 clock 操作。新调用如再次失败，应先分析新的真实回执，不循环 resume。

## 证据索引

- `diagnosis.json`、`bound_entry/`、`frozen_protocol/`：绑定入口、原生到规范化坐标、全部锚点检查及适用规则。
- `preflight.json`、`native_result_check.json`、`failed_attempt/`：冻结恢复资格、实际依赖、终止身份及原失败证据。
- `state.before.json`、`state.pre_submit.json`、`state.after.json`、`status.before.json`、`inputs.lock.before.json`：运行状态与身份基线。
- `supervisor.before.json`、`supervisor.after.json`、`resume.*`：控制基线、唯一正式命令及 consumed 记录。
- `postflight.json`、`immutable_manifest.before.json`：调用、批准、账目和证据保持检查。
- `new_attempt/`、`sibling_before/`：新会话启动与诊断送达、健康兄弟原进程身份。
- `preflight.py`、`postflight.py`：仅向本证据目录写文件的定向核验脚本；没有运行状态写入函数。

取证脚本初次执行曾将 proposal/receipt 的 Runner 补充字段误作完全相等比较，及过早读取尚未结算的新 attempt_usage；均在只读检查处停止。按真实结构修正后核验通过，未因此重发控制请求或新增模型调用。
