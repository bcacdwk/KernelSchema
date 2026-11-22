# 2026-11-19 十二项并行 Extract（已交付完成）

截至北京时间 2026-11-20 20:56，本轮 **12/12 已交付**，实际核实 **33 个 YAML 设计、12 个 dispatch、12 份正式回执**；全部 run 为 `ready / P6 / published=true`，活动、未确认及未结清调用均为空。正式文件与发布清单的 SHA 一致。

最终入口：[交付报告](FINAL_REPORT.md)、[交付索引](DELIVERY_INDEX.json)、[逐项交付核验](deliveries/)、[整批核验](closeout/delivery_verification.json)。mixed Sparse 的来源回执缺口、性能用户豁免及采样证明边界均保留，详见报告。

本轮巡检已由主代理关闭：`enabled=false`、`process_alive=false`、`process_status=stopped`，见 [关闭记录](monitoring/MONITOR_CLOSED.json) 和 [停止回执](monitoring/stop.stdout)。保留的 21:22:10 到期记录已停用，不再安排检查。

## 启动历史（2026-11-19）

以下是 2026-11-19 的启动时记录，其中 running、尚未等待完成等表述仅描述当时快照；当前交付状态以上方最终入口为准。

本轮按用户授权，在同一批命令中逐项登记并启动 12 个 detached Runner；六类各两项。所有人读时间为北京时间（UTC+8）。Supervisor 已完成两个只读预检，本执行记录只复核 CLI 参数、精确清理占位空叶目录并启动观察。

配置：Claude Code；模型 `qwen3.8-max[1m]`；worker/reviewer effort 均为 `max`；revision-limit=5；max-calls=32（每阶段或 P5 节点累计额度）。Python 为 `/opt/venv/python3.13.12/bin/python3`，后端真实 binary 为 `/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe`。

选题、输出映射和排除的已交付六项见 `selection.json`。GEMM：dense W4A16 与 masked FP8×MXFP4；MoE：BF16 与 FP8 block128 QDQ；FMHA：分页 BF16 与每 token/head FP8 scale；MLA：absorbed D576 与 NVFP4 D512 分页 decode；Sparse：物理索引分页 FP8 与混合 MXFP8 滑窗/NVFP4 压缩 KV；Linear：卷积 MTP 递推与 KDA 逐通道衰减。

来源例外：`paged_latent_mxfp8_nvfp4` 缺最新版原始题目验证回执，最终题目哈希确认尚未完成。其性能为人工接受例外，自动性能未通过；NCU 6/10 达标，超限 shape 0/1/3/9。因混合缓存设计差异显著且题目/源码齐全，按用户允许的普通取舍保留；不视为验证通过。`USER_ACCEPTANCE.md`、`performance/ACCEPTANCE.json`、`PRECISION.md` 由 ID 绑定冻结并进入 P1 acceptance；`PROBLEM_NOTES.md`、`kernel_wiki/EVIDENCE.md`、算子 README、`performance/SUMMARY.json`、catalog 也已绑定可供 P1 阅读。`PROBLEM_AUDIT.json` 本身不在冻结集合。

只删除经整体验证的十二个空 `.gitkeep` 及其空叶目录，正式成果只由正常 Runner 发布。不恢复旧任务，不修改源码/工具/规则，不运行 GPU，不提交或 push，不启动或停止 monitor。

启动原始 stdout/stderr 分别保存在 `launch_stdout.txt` / `launch_stderr.txt`，准确命令及时间结果保存在 `launch_execution.json`，12 项 run UUID/配置映射保存在 `launch.json`。初始观察保存在 `initial_inspect.json`。

Supervisor 按用户“继续自动化 extract、普通问题按推荐”的授权采用既有同线程巡检，由主代理单独配置并记录到 `monitoring/`。既定工具修复续接政策为：原进度继续、原预算保留、疑点打标、正常审阅门禁，末尾再知会；本次启动不涉及工具修复。历史临近结束的五分钟手工轮询仅针对上轮收尾，本轮不恢复旧任务。

启动结果：北京时间 2026-11-19 21:59:31 至 22:00:35，单条批量命令成功（exit 0），12 项均为新 run，无复用。初次 inspect 及进程/原生事件复核截至 22:01:21：12/12 running，均在 P1 worker；Runner 存活且持锁，12 项均有 active attempt；无 reason、anomalies、unconfirmed attempt 或 global hold。活动调用的 unsettled=dispatched 是尚未完成结算的正常状态。

每个调用均已产生原生 assistant 和 tool_use 事件，真实后端交互已启动；原生事件未观察到 error/is_error 标记。共同 stderr 为 Claude Code 的 `unrecognized_model` 提示，所指请求模型是 `qwen3.8-max[1m]`，调用仍正常继续，原生消息 model 字段为 `qwen3.8-max`。这不代表服务端上下文窗口或 effort 已核实。原始 stderr 和统计见 `initial_observation_summary.json`、`native_event_checks.json`；未修改模型或重试策略。

| Catalog ID | 新 run UUID |
| --- | --- |
| `bf16_nvfp4_gemm` | `2b64ba84-64b3-4d10-86fe-65c351ce571c` |
| `fp8_mxfp4_grouped_gemm_masked` | `c8dd5678-2935-4efa-943a-02e848d03611` |
| `bf16_fused_moe` | `9dbb114f-af3c-46e9-8f70-95049c768c35` |
| `fp8_scaled_fused_moe` | `5be021ee-1bb1-4554-bd82-88cc2599d792` |
| `fmha_bf16` | `e9af332e-dbf4-476b-8283-a88e3cd3cd98` |
| `fmha_fp8_per_token_head` | `1a493588-28a0-4097-849f-85cac55e5b16` |
| `mla_absorbed_fp8` | `e03cbc5e-da09-41bf-a981-757c7a46a4c1` |
| `mla_nvfp4` | `af01989c-c712-4fcf-8ecc-3a88f10541eb` |
| `paged_indexed_latent_fp8` | `fd1d34e0-80f0-4a48-a64f-e3952b68d456` |
| `paged_latent_mxfp8_nvfp4` | `d138394a-20a7-4397-896e-ebac43aa10bb` |
| `gdn_conv_mtp_bf16_state_bf16` | `e0311f19-50e5-4760-8331-5805dbb23a09` |
| `kda_raw_prefill_state_bf16` | `bda2a4ba-47f7-4ae2-8a9b-9e0081c41cf9` |

本次执行到初始启动验证即结束，不等待提取完成；后续巡检与正常 Runner 审阅/发布继续。工作区可见改动仅本轮 discussion 记录和已授权的十二个空 `.gitkeep` 删除，未提交或 push。
