# GEMM 交付核对

结论：`fp8_scaled_grouped_gemm` 已完整交付，本次核对未发现交付阻塞。正式目录包含 [root.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml)、[wide.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/wide.yaml)、[中文 dispatch.md](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) 和 [.extraction/receipt.json](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/receipt.json)。本次仅核对既有交付、审批绑定与调用结清，没有重新提取或进行新的语义全审。

Run：`06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31`；state root：`.schema_extract/v2`；最终 checkpoint：`36bab47cdeb78ff47c8fb083bf663eb69838ecc29ad40636c54c08000302f57d`。核对时间与精简证据见 [DELIVERY_GEMM_EVIDENCE.json](DELIVERY_GEMM_EVIDENCE.json)。

## 交付与审批

- 正式目录恰有上述四个普通文件。三份正文与最终 checkpoint 对应文件逐字节相同，SHA-256 同时匹配交付 receipt、checkpoint manifest 和 P6 approval 的 artifact manifest。交付 receipt 内容等于 state 与 publication journal 保存的 proof。
- P6 reviewer attempt `14300e28-6e2d-4626-ba15-14ee9cdae004` 的决定为 `approve`，`issues=[]`、`changed_files=[]`；attempt 与 receipt 的规范 JSON 摘要均匹配发布绑定。摘要口径为 Extract 的 `digest`，不是格式化 JSON 文件的原始字节摘要。最终 checkpoint、merged checkpoint、P6 approval 与 P6 输入 checkpoint 一致。
- 当前 `ready / P6`、`published=true`，P1–P6 均为 `approved`。P2 批准选择 root（`umma_gemm<0>`）与 wide（`umma_wide<4,32768,0>`），P4 批准安排为 `[{id: wide, wait_for: []}]`，与当前状态及发布选择一致。两节点均已批准、合入，完成 2/2；中文目录登记恰好两份完整设计，无“仅计划”或“记录计划”残留，参数与局部记录不计作完整设计。
- 24 个登记调用与 24 条账目 ID 一致，全部 `settled=true`、`outcome=completed`；`active_attempts`、`unconfirmed_attempts`、`unsettled_attempts` 与 `pending_rechecks` 均为空。

| 正式文件 | SHA-256 |
| --- | --- |
| root.yaml | `4e28944a053c0338038e828db46034fd032a73b6fb918c221ced20d3095a60ba` |
| wide.yaml | `9ba4026d93789ffc32ebf1d99bdff5650c0c74a96329dedd47dba05b29f3882a` |
| dispatch.md | `33988a787f53fef80a62f4401e92de49e67c5bc0f3666e51bee59ecba8219d3e` |

## 只读检查及限制

当前 Template 执行一次：`python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml --read-only --skip-hardware`。退出 0，2 YAML、0 规范化改动、没有写回；结构、文本预算、`from`、关联与 dispatch 检查通过。硬件区和依赖硬件的关系显式跳过，未运行 GPU、有限图样本、数值或性能验证。

公开 audit 仅对本 run 执行一次：`python3 -B schema_extract/extract_yaml.py audit 06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31 --state-root .schema_extract/v2 --verify --json`。退出 0，但 `verification.status=partial`、`errors=[]`，不能写为全部证据已验证。范围为 24 个登记调用及其输入/结果和当前/最终 checkpoint；五条 `unavailable` 均为 `receipt_not_accepted; artifact validation not completed`：

| 历史调用 | 审计保留的拒绝原因 |
| --- | --- |
| P2 reviewer `d9917f38-62b6-4bcd-b4a4-ffffe9841c1f` | `receipt_repair`：approval 的短交接超过 1200 字符或 20 非空行限制 |
| P3 worker `5e3e8c82-308b-4a75-9550-4681820bf830` | `proposal_rejected`：`missing bound implementation entry anchor` |
| P3 reviewer `34fb5656-f047-4f76-bcad-576ada9f3aa0` | `receipt_repair`：同上短交接长度限制 |
| P4 reviewer `4572493b-53ce-40a3-b067-15085f32d723` | `receipt_repair`：同上短交接长度限制 |
| P5 wide worker `c23a0d66-d4e5-439e-98c9-f1ed95a23a7b` | `proposal_rejected`：`missing bound implementation entry anchor` |

相关阶段后续均已取得批准；上述历史证据缺口仍然保留，不等于当前调用未结清。最终批准与发布绑定已另行核对。公开 audit 的权限为 `evidence_only; no approval, dispatch or publication`，本次检查没有产生新的批准。

## 框架候选

公开 `candidates RUN --state-root .schema_extract/v2` 返回两项，均已获 reviewer 确认且登记为 `pending`：

- [non-tensor-parameter-registration](../../schema_extract/issues/archive/schema_docs/non-tensor-parameter-registration.json)：Python 固定长度序列及框架 dtype 参数的登记示例建议。
- [location-storage-level-registration](../../schema_extract/issues/archive/extract_workflow/location-storage-level-registration.json)：P1 最小执行单元登记与 `Storage.location` 存储层级名称的说明口径。

两项均属已登记、尚未解决的非阻塞建议；`confirmed` 表示提报已被确认，`pending` 表示处置未决，不能因 GEMM 已发布而写成已解决。本次仅新增本报告与证据 JSON，未修改已发布产物、运行状态、回执、批准或工具，未启动模型、维护、恢复或新 run，未提交或 push。
