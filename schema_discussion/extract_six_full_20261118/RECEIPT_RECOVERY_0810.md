# 08:10 首次回执拒收与原 run 恢复

两次拒收均为模型回执缺项，不是原生调用失败或工具误拒。实际冻结 prompt 已提供对应要求；不修改校验规则、prompt、冻结快照或本次产物。以下计数与文件状态取自本次恢复前诊断。

| 项目 | GDN | FMHA |
| --- | --- | --- |
| Run | `68edadab-4371-402c-8acf-f73eb5ec9f92` | `fb8b3758-92d7-4093-aa18-c2e5028f0a43` |
| 拒收 attempt | `87b62d5b-1886-4b8e-a3f7-92c75fcc344d` | `ff6a95e9-68ce-40e6-a28d-7ed938a7a3f4` |
| 范围 | P1 reviewer | P1 worker |
| 已用/总 calls | 2/32 | 1/32 |
| `invalid_receipts` / `receipt_repairs` | 1 / 0 | 1 / 0 |
| 可续用原生 session | `c8ac1806-772f-493c-9a6c-0177a042d738` | `c84fb4d3-05b3-4058-93dd-79ac88e4057d` |

GDN 的 `response.json` 返回 `approve`，但 `conclusions` 只有 `contract`、`coverage`，缺 `document_self_contained`、`document_layering`、`document_consistency`、`document_relevance`。虽有 10 条 `artifact_evidence`，不能代替四项明确判断。原 reviewer 必须完成并提交这些判断，不自动补值。

FMHA 的 `response.json` 返回 `complete`，但 `implementation` 仅引用 CUDA/CUH helper；绑定入口 `candidate/solution.py::Model.forward` 的三条证据均标为 `host`。`implementation` 必须先覆盖该绑定入口。原 worker 需据实际阅读补齐正确分类的源码证据。已有 draft 仍须重新提交并接受正常独立审阅。

这两项校验均抛出普通 `ExtractError`，并非可自动格式修复的 `ReceiptFormatError`，因此没有用掉两次自动 repair 配额。总余额分别为 30、31 次，后续审阅、裁决和签收仍按原规则预留；无需追加 calls。

证据位于 `.schema_extract/v2/runs/<run>/attempts/<attempt>/` 的 `response.json`、`receipt.json`、`rejected.json`、`result.json`、`exit.json` 和 `prompt.txt`。两次均无 `validated.json`，不得将旧 `approve/complete` 当成批准。只读 `verified_result` 已复核真实 events、exit、schema、nonce：`exit_code=0`、唯一成功 native result、`schema_valid/nonce_valid/native_output_matches=true`、session `ready`。对应原生 transcript 文件仍存在。两项 `active_attempts/unconfirmed_attempts/unsettled_attempts` 均为空，`scoped_restart_evidence(P1)` 均通过且 `settlement_required=false`。

## 原稿与恢复路径

GDN 当前 checkpoint 为 `296820593de68897d5c3fd04a8c6af1d90000e0904c77475fffdb55b735eb327`，包含此前合法提交的 10225 字节 `root.yaml`；新 reviewer attempt 会从这里复制工作目录，沿用原 reviewer session。

FMHA 当前 checkpoint 为 `92242bee13c03090f7003c77be1bed515204c089184333d14d2de156dafa4196`，仍是 1522 字节 starter。被拒 attempt 的 `original/root.yaml`、`workspace/root.yaml` 各 16151 字节，`normalized/root.yaml` 为 14644 字节，均保留；它们尚未成为 checkpoint。普通恢复不会自动将拒收稿合入新 workspace。原 worker 精确会话保留此前上下文，收到拒因后在新授权 workspace 继续提交；不能由 Supervisor 复制稿件入 run 或代填证据。

在冻结身份、当前范围和控制状态仍匹配、旧调用已结清的前提下，优先运行公开 whole-run resume：

```sh
python3 schema_extract/extract_yaml.py resume 68edadab-4371-402c-8acf-f73eb5ec9f92 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
python3 schema_extract/extract_yaml.py resume fb8b3758-92d7-4093-aa18-c2e5028f0a43 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

这两项各只有当前 P1，whole-run resume 保留原模型 `claude / qwen3.8-max[1m] / max`、会话、checkpoint 和已用次数，并将拒因加入下一调用的 feedback。两条命令由 Supervisor 分别在独立后台进程运行即可；`subprocess.Popen` 的 `start_new_session=True`、`stdin=DEVNULL`、独立 stdout/stderr 日志不改变公开 CLI 的 launch 身份协议。CLI 自行登记并转入 frozen executor，不设置内部入口环境变量，不调用内部 `_background`。

## 当前源码的独立小修复

本批 frozen executor 的 `resume --scope P1` 虽满足恢复条件，却会在消费请求时清除 `reason` 而不传入 `feedback`；因而本批选 whole-run resume。发现的这一反馈缺失已在当前 `schema_extract/tools/runner.py` 修复：清除原因前按普通恢复相同的去重规则保留反馈。只增加 3 行，不改变接纳、计数、进程 fence 或审阅要求；不回写本批 frozen executor。

新增定向测试先复现拒因丢失，再验证反馈持久化、重放不重复、calls/receipt counters 和批准状态不变。`python3 -m unittest schema_extract.tools.tests.test_scoped_recovery -v` 的 4 项全部通过；`git diff --check` 通过。未修改 `maintenance.py`，未自行执行暂停、取消或恢复，未启动真实模型、提交或推送。
