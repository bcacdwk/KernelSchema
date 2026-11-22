# Issue 06：Agate harness 枚举适配与已拒绝请求恢复

`vector_add-006` 的 Design 已完成，Implement 第一次 coder 调用正常交付完整 kernel，`gaps=[]`。首次 Agate `submit_job('eval', request)` 返回 `422 / request_validation_failed / validation`，没有创建 job；Implement 和 Workflow 如实停在 `incomplete / evaluation_failed`，`calls_used=1`。原 `attempts/01` 的候选、草稿、notes、request、response 和 feedback 保持。

## 字段级根因与无调度复验

正式 `atrex-gateway-client 0.14.37` 的 builder 只负责组织 envelope，服务端仍是请求 schema 权威。SDK 公开 `Client.validate_eval(request)` 文档声明该接口检查声明和内联源，不上传或调度 job，因此无需重放原提交来查错。

2026-11-22 UTC，通过既有 `agate.connect` 和授权配置路径调用原请求的无调度验证，得到唯一字段错误：

```json
{
  "loc": ["body", "harness"],
  "type": "enum",
  "msg": "Input should be 'atrex_bench' or 'sol_execbench'"
}
```

固定 eval 使用历史名称 `harness: atrex`，适配器直接透传造成上述错误。只把 wire 中的该值改成 `atrex_bench` 后服务接受验证；修复后的实际 builder 用原候选、固定 eval、原名称和原 key 重建请求，逐字段差异也只有 `harness`。服务返回 `valid=true, scope=declaration_and_inline_source`。诊断期间 `submit_job` 调用为 0，模型调用为 0；该验证不代表 GPU 正确性或计时通过。

## 适配与诊断修复

`agate.build_request` 将固定配置的 `atrex` 映射为 `atrex_bench`，也接受已使用正式名称的固定配置。不改 Task/eval 原字节、7 个案例、误差阈值、benchmark 模式/迭代数、版本或 `options.timeout_s=900`。

`service_error` 保留有上限的结构化 `validation_errors`：字段位置、错误类型和裁剪/脱敏后的消息。原 exception 字符串、payload、input、ctx、认证字段和签名 URL 不落盘。已验证 enum 错误仍可直接定位 harness，同时测试验证私密 input/ctx/URL/认证消息不会进入持久诊断。

## 同目录显式恢复

Implement 和 Workflow 都接受 `--resume R --recover-rejected-request`。Workflow 只重排当前版本的原 Implement 目录，不理解 Schema 或 SDK 校验内容。Implement 要求旧终态确为 `evaluation_failed`，固定 response 为 `rejected / 422 / request_validation_failed / validation`，没有未决 action、已知 job、上传 reservation 或 `oss_files`，原 inline 候选完整，`gaps=[]`，Schema/kernel/notes/反馈绑定及固定 manifest 一致，且没有待消费的新 repair feedback。

重建请求的实质差异严格限定为本次 `harness: atrex → atrex_bench`；body 不变或其他字段变化均不允许用这个入口重复提交。所有检查在创建新 attempt 前完成。新 attempt 复制原 kernel/notes 字节，复用原 Schema/eval/reference 快照；coder 不调用，`calls_used` 不变，即使 coder 余量已用尽也可以恢复该已交完整候选的评测。原 attempt、原 request/key、原拒绝 response/feedback 不覆盖。新的修正请求使用新 key，`state.request_recoveries` 与新 attempt 的 `recovery.json` 记录原 key/请求摘要、source attempt、原结果和累计次数。

新成功 feedback 只绑定新 attempt 的测量证据和原固定材料，不把旧拒绝反馈导入 Optimize 的当前候选闭包。complete 不重跑。任何未决提交或已有 job 拒绝显式开关，使用普通 resume 沿原请求/key/ID 继续；Workflow 一次恢复已启动后如中断，普通 resume 不再次传递恢复开关。

## CPU 与真实运行边界

```sh
python -m pytest -q schema_implement/tools/tests/test_implement.py schema_workflow/tools/tests/test_workflow.py
# 78 passed in 39.74s

python -m pytest -q schema_implement/tools/tests/test_implement.py \
  -k 'real_sdk_builder or recovery or rejected_harness or rejected_request or validation_errors'
# 13 passed, 32 deselected in 6.27s
```

检查覆盖原 attempt 所有字节保全、calls 不增、固定 inputs 不变、完整候选与 notes 复用、无变化 body 拒绝、其他 wire 参数变化拒绝、错误状态/原因/Schema 绑定/gaps 拒绝、已知 job/未知 acknowledgement/上传 reservation 拒绝，以及普通恢复继续原 job/key。真实 Workflow/Implement/Optimize CLI 装配使用明确的 CPU 模型与服务 fixture，证实同版本原 Implement 目录恢复后，新成功 feedback 和全部固定 references 能正常进入 Optimize；旧拒绝不混入本次候选绑定。该测试不声称实际模型或 GPU 验收。

另外已将真实 006 state 与 manifest 交给本次实际恢复检查函数，在其第一个目录创建动作前强制停止：全部前置检查通过，`attempts/02` 尚不存在，原 state/result/拒绝证据摘要不变，`calls_used=1`。仅用于校验入口可用，没有实际提交或模型调用。

本次工具调用只用了 SDK 无调度验证，未使用获准的原 key 诊断提交重放额度。真实恢复由主代理提交代码后通过正常 Workflow 入口启动；本记录截止时真实 GPU 正确性/计时与 Optimize 均未发生。共享 EXECUTION 由主代理维护。
