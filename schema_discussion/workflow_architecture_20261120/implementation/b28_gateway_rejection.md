# B28：Agate 明确入队前拒绝的恢复与结算

2026-11-21 UTC。KS `feat/schema-workflows-v1` 基线 `5341470ba5ac03aea857b6480cc7a813c0e3682f` 上已完成有界 transport 修复：完整、明确且绑定原 POST 的 validation/auth 拒绝可沿原 operation/key 结算 jobs=0、停止原 active 计时；资格仍为 blocked，不产生 worker terminal、Task fail、correctness 或权限通过证明。源码尚未提交/推送，交由根精确提交。

本项新增真实模型/GPU/网络调用和真实 OwnerAccount API 操作均为 **0**。全部执行测试使用 fake opener、内存 wire 或独立临时 synthetic owner；唯一真实响应 fixture 是从 B27 已导出的207字节原拒绝复制，未请求服务。没有读取凭据、认证头，或读取/枚举/修改/运行 AKA_experiment；没有派生代理。writing、Template、预算格式、Task/Schema、CUDA/worker/sandbox 判据与 AKA 均未改。

原始证据先核了 AGENTS、B27 最终报告与 handoff、rejection-audit、完整 selected-export 的112文件/1495024字节，以及 B14 diagnostic_owner.py 的既有拒绝结算规则。B27 原 HTTP400 / validation / dev_sensitive_target_denied / process_root_escape 响应完整，无 job_id；没有 queued、worker 或 terminal。新 parser 对原导出五件套进行只读解释，正确恢复原 key 和 submitted_request SHA；没有调用原 executor、结算、重投或修改它。

本次实现和公开接口影响：

- 仅识别 HTTP400/401/403/422、完整 gateway envelope：error_class 为 validation/auth，reason/message/trace_id 非空，details 为对象，无任何 accepted/job/locator/执行事实；其他 schema、重复 JSON key、缺 status 或不完整原件仍 unknown。
- 原 intent/request/submission-start 必须一致，核原 key、prepared SHA、实际 tightened timeout 和 submitted_request SHA。原有 accepted locator、query、terminal 或已记 jobs 的 absorption 收据阻止改判 jobs=0。
- 保留既有 operation `state=terminal`，新增具体 transport `outcome=rejected_before_job`；settle 的 report=None，既有 usage_index 的 settled row 记录 calls/jobs/tokens/gpu_ms 全0。没有新增公共 contract kind、预算字段或第二账本。这里 terminal 只说明原提交 operation 已结束，不声称存在远程 worker 终态。
- `ks.agate.rejection.v1` 是普通原始 receipt 字节，内嵌原 HTTP status/body、request、intent、submission-start 的精确字符串和逐项 SHA。普通 usage_index 单独导出即可保全原证明闭包，不依赖任意 byte 文档的隐式 Ref 遍历。
- 曾观察到的 accepted job 增加私有 `accepted-job.json`，绑定 job_id 和原 ACK SHA。GET 4xx/404、5xx、断网、丢 ACK、未知 body 继续 unknown/settle_only；迟到原 ACK 仍只接回原 job、计 jobs1。不会重 POST、换 key 或创建替代 operation。
- RowSoftmax 和共享 transport 的 FA4 qualification 只增加同一拒绝原件变体的消费：沿原 OwnerAccount.absorb/end 与 RunContext.record_terminal 收口，保留实际 elapsed active；返回 blocked、qualification=None、recovery=none，停止后续阶段。已完成资格重开不重新 begin active clock。正常原 job 原件/判据不变。
- 网络异常 journal 仅保留异常类和固定的 unknown 说明，避免原异常文本包含认证数据；HTTP 原响应字节未重写。

验证没有全库重跑。解释器为既有 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python`；使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，没有安装依赖。直接运行 pytest 曾因环境自动插件 `platformdirs.pytest_plugin` 缺失而未收集，随后禁用自动加载。

三份定点测试累计覆盖 **91个不同用例**，无未解决失败：

- 首次 `test_evaluation_agate.py test_agate_channel.py test_fa4_channel.py`：86 passed、1 failed /194.25s。唯一失败来自新增 synthetic owner 夹具使用1970年时钟而没有对齐 transport 截止时钟；只修测试的注入时钟，没有放宽生产截止。该次 RowSoftmax/FA4 的32个测试全部通过。
- 修正夹具并补入 publication crash 与迟到ACK测试后，完整 `test_evaluation_agate.py`：**58 passed /36.99s**。
- 最后对“先前已有job收据、其他原件损坏仍不可退成0”的新增 guard 及拒绝导出/中断publication两项相关回归：**3 passed /3.63s**，其中1个新用例、2个重复验证。

具体覆盖完整400拒绝和4种HTTP分类；原status/body/provenance逐字不变；请求/key/timeout绑定缺失；网络/丢ACK/5xx/404/不完整响应持续fence；已accepted job后来的4xx仍计jobs1；receipt crash恢复同publication；旧unknown仅凭已存原件恢复、0新网络/0新op；synthetic owner精确记录357ms active、重复吸收不收费、预约终止；拒绝后的qualification无后续POST且非资格；usage单独导出自包含原证据。`git diff --check` 通过。

B27 原五份HTTP/POST原件、原Q、原demo state、原qualification state均与本项初始SHA一致。其原 operation history/latest 与原selected-export的唯一jobs0 receipt逐项相同，operation仍终止；没有再次吸收。父账整个文件SHA在本项期间发生变化，根已明确归因为获授权B29同parent真实design调用，因此本报告不声称整个父账文件静止。本项不操作或调整B29账。

原 B27 手工结算收据和旧公开结果继续保留；本修复不是原scope的重试许可或自动迁移。未知原件无法补成拒绝；普通auth细节体或未来未支持错误schema仍保守 unknown。原GPU资格仍未成立，平台 process_root_escape 控制没有被绕过，原权限探针没有删改。

证据目录：[本项只读审计与测试日志](/tmp/schema-b28-gateway-rejection)。关键文件为 `original-records-before.json`、`original-records-audit.json`、`original-proof-interpretation.json`、`targeted-first.log`、`transport-final.log`、`receipt-guard-final.log`、`source-manifest.json`。B27 原response SHA保持 `d8249a8c12a54200f06e07526712b86ce49ce6c02ef6cda14249d061211d24ea`。

冻结文件清单：

| worktree 相对路径 | bytes | SHA-256 |
| --- | ---: | --- |
| `kernelschema/evaluation/agate.py` | 38411 | `c11985faa9a79819a8bf7d110aa91620f4bac1efed1bb51d9c04107655829bee` |
| `kernelschema/evaluation/channel.py` | 24491 | `ff4673f5ab80185c334e086e9aa11c767be747e9b9bc38ee5f30890487f67d16` |
| `kernelschema/evaluation/fa4_channel.py` | 30672 | `de0f396b518f2a5e9ef5e3a93548c45e6896a7eb4d228af46e890aa9341758c0` |
| `tests/workflows/test_evaluation_agate.py` | 24689 | `f906f25e4666f414ee07400ec8276ac96b19013b3abca37c6a31cc4d7f5862a7` |
| `tests/workflows/test_agate_channel.py` | 20244 | `8f2227f6239f1036ac8d64c4745932470180511908f1d579cce59bf2ed358887` |
| `tests/workflows/test_fa4_channel.py` | 16395 | `279acd7a043e91b8d9a1082cf7224d27d581df87afd7fefc421fce561e4219a5` |
| `tests/workflows/fixtures/evaluation/gateway-rejection.raw` | 207 | `d8249a8c12a54200f06e07526712b86ce49ce6c02ef6cda14249d061211d24ea` |
| `tests/workflows/fixtures/evaluation/gateway-rejection.json` | 543 | `79a6b76656ecc817ef43088cfb519a53231e70d91f425b4b42021484a1e98c93` |
| `examples/workflows/row_softmax/evaluator/EXECUTION.md` | 17357 | `6b9f43a74bf13fce0ec590d35727b51e6dfa07af45287a485a23d8d39b79ebac` |
