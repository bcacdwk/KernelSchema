# B32：标准 AKA / Agate 真实执行

2026-11-21。本轮只验证新标准 native Atrex-Bench Task，不替旧 strict Task、副作用或 sandbox 保证签收。未运行模型；未修改 KS writing/API；未访问 AKA_experiment；未提交或推送。

## 固定来源与范围

- AKA 基础提交：`9dfe8c66b7ccca8081e4a781896fe86c5d8af9ac`。已读标准文档、Light plugin、原 LegacyProblemProvider / LegacyExecutor / LegacyEvaluator 及 GPU gateway skill。
- 官方 bench：`alibaba/atrex-bench@1fa1f0d8a663093a22946270e7d6accb00c33698`，Apache-2.0；来源及 archive hash 延续 B31 `provenance.json`。
- 私有执行目录：`/tmp/schema-b32-standard-real-20261121`。保留 B31 原脚本、原请求和完整 Task 副本。
- 官方 SDK 0.14.37 从已安装 SDK 复制至私有 `fixed-sdk`，没有重写 HTTP/auth。预算源码从 B18 固定目录复制至 `fixed-ks`，45 个源文件逐字节匹配 KS `77ad741ca5b088d24f4cd23018eb0efcd491f15f`；清单在 `dependency-freeze.json`。没有安装 KS 活跃 tree。
- 使用原 B14 parent，准入前：5 models / 3 jobs / 1642872 ms，10 operations 全终，无 active/fence。新 request 有限额 0 models / 1 evaluator job / 900000 ms active；worker timeout 600 秒。

## 最小执行账适配

脚本 `baseline_job.py` SHA256：`966623fa071bccf25c0e658485a716a4d85ec34f7526323393bcb0e3d0828b77`。

原脚本的 `Client.from_profile()` 默认不载入 AK/SK，且 `submit_eval()` 仍会在 overload/兼容端点下再 POST。修订使用官方 `profiles.load_config`、`AkSkAuth`，单次 SDK `_request(POST, /v1/jobs/eval)`；不打印或复制私有认证。每次 SDK 响应保留完整 decoded payload、phase、HTTP 状态与耗时，ACK 独立保留，查询只用原 job ID，每次 HTTP 不超过 30 秒。默认不根据一般 4xx 退回额度；接受后查询错误仍计 1 job。终止复用 AKA `remote_termination`，cleanup/quarantine 不视为完成。没有引入评测器。

少量纯合成测试通过：一次 submit；重复 submit 拒绝；ACK 和两次响应独立保存；accepted 后 404 不退款；普通 4xx 保留 unknown；明确 no-job overload 分类；原 native cleanup/terminal 解释。测试只用临时合成 parent，无网络或模型。

## 首次真实提交：明确入队前拒绝

原请求 SHA256：`51724eeacc11f61faa3c33c7d8b24fb90d5b8b64030ec02af3c77e58b6283dd8`；key：`schema-b31-reference-da572e069737cce07d3d700751e952b9`。仅 POST 一次。

网关返回 `400 validation/invalid_source`，`source validation failed`，trace `req-ca7bed9e0aa8`。三个明确错误均为：shape ID `small` / `medium` / `large` 必须能转整数，因为 worker 用 `int(shape_id)` 设置 seed。无 ACK/job ID，无 accepted GPU job，无模型。

原响应：`responses/0001.json`。首次保守记录 unknown，随后依据单次 POST 阶段、完整 source validation 错误、没有任何 job identity 与原 AKA `protocol_failure(source_validation_failed/request)` 追加 `reconciliation.json` 和第二条 usage receipt，明确为未入队，未改原 unknown 历史。

结算后 parent：5 models / 3 jobs / 1693121 ms；11 operations 全终；active=null，无 fence。该尝试新增 active 50248 ms，accepted jobs 0，models 0。GPU 毫秒和货币费用未报告，未推算。

## 数字 shape ID 修正

AKA 原示例 `skills/atrex-aka-light/examples/native-atrex-bench-vector-add/shapes.json` 仅将三个键改为 `0/1/2`，README 说明其原因。alpha、numel、reference.py、input.py 不变；不加入新的资格门。

独立修正目录：`/tmp/schema-b32-standard-real-20261121/numeric-shapes`。

- request SHA256：`6ee97cb4d6625fa78a5201c964f13ba3db53cc33b9ba2259c6c4f9cd888e2369`。
- 新 key：`schema-b32-reference-130af1893a0b13d6b49c6007467c1ac5`。没有以旧 key 更改 payload。
- 脚本 SHA256：`d505f69a6e8e8dfb41284851fa32d9a2f21a269fcbe3a97e907c70836f93fc07`；仅独立 run ID 与首个脚本不同。
- 新 shapes.json SHA256：`314314866671d5b1ac7655178a23ec23a0d6d533afb0c8364b2f134529145d34`。
- CPU 原 native Task load/freeze 通过，且确认完整请求只改变 shape IDs 和随之需要更改的 idempotency key；结果 `cpu-preparation.json`。

根明确授权新的 SHA/key 后，仅 POST 一次；原拒绝 key/payload/两条 receipts 原样保留。B33 已收到数字 Task 的路径及变更说明。

## 真实 L20D baseline 成功

唯一 accepted job：`ev_1861beac3248`；`succeeded`，`execution_state=stopped`，`cleanup_pending=false`，`execution_quarantined=false`。原 AKA `remote_termination` 返回 true。原 ACK 保留为 `numeric-shapes/baseline-ack.json`；单次原 job GET 的完整响应保留为 `numeric-shapes/responses/0002.json`，未覆盖 `0001.json`。

原 AKA `_optimizer_result_from_eval`：`all_pass=true`，failures 空，max_abs_err/max_rel_err 均为 0。实测 median end-to-end latency：

| Shape ID | numel | alpha | 微秒 |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0 | 9.456000290811062 |
| 1 | 65536 | 0.5 | 11.215999722480774 |
| 2 | 1048576 | 2.0 | 11.264000087976456 |

这是 reference 自身作为 candidate 的测量，speedup=1.0，不证明优化收益。metadata 内 provenance 明确为 measured PyTorch reference，不声称 production kernel。正常 gateway native correctness/performance 已真实执行，未增加自己的评测判据。

网关记录的实际环境：NVIDIA L20D，torch 2.9.0+cu130，Triton 3.7.1，Python 3.12.3，driver 580.105.08。clock_locked=false。原 `request_digest.atrex_bench_runtime` 明确远端为 gateway_default 的 `atrex-bench==0.1.0.post20261111022558`，`source=job_install`，原因 `image_version_unavailable`；evaluator SHA256 为 `183b6288b85f0cd7ee3828c1a2a9db922f8971774e0122f5fceccfb58e09456e`。本地固定 bench commit 是 Task/runtime 布局来源，不推导为远端相同 Git commit。远端 candidate/reference/input 哈希与请求逐字节匹配；shapes 的网关 JSON 序列化哈希亦已核对。

证据均在数字目录下：

- `baseline-job.json` SHA256：`a20143155567e52e82ba0ae63d13f20bdec8f79c6da0845f599a0da081122ca7`。
- `baseline-native-observation.json`：原 native 解释结果。
- Task `metadata.json` SHA256：`17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740`。
- `baseline-summary.json`：原环境与耗用摘要。
- `native-prepared-full-request.json`：metadata 导入后，原 `gateway_request.prepare_request` full-performance 准备通过；没有再次提交。
- `native-task-with-baseline`：原 LegacyProblemProvider 新建带 metadata 的完整 snapshot。
- `parent-after.json`：原 parent 累计 5 models / 4 jobs / 1716864 ms；12 operations 全终，无 active/fence。

B32 合计新增 0 models / 1 accepted GPU job / 73992 ms parent active；其中明确拒绝尝试 50248 ms，成功尝试 23743 ms，parent 总数与两项相差 1 ms 来自统一纳秒累计取整。服务端 created→finished 为 22.825820446 秒；GPU 毫秒和货币费用未报告，未用 wall time 推算。

已将 GPU slot 交还根。下一阶段使用 B33 实际写出的 Schema 运行原 AKA standard implementation，需根另行分配 slot/小额度；本阶段没有启动模型。示例两文件修改冻结在 `/tmp/schema-b32-standard-real-20261121/aka-example-fix.json`，待根精确提交。
