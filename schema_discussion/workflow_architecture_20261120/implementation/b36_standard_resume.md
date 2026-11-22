# B36：修复版本的标准 Schema campaign 验证

2026-11-21 UTC。**没有交付 Triton kernel。** 固定 AKA `621481a9b8685f4880e32a9ddce56f4e3d94cd07` 的新标准 campaign 完成必要新 V0；唯一模型 invocation 在 900 秒超时，episode 为 `invalid_handoff`、changed_paths 为空、candidate_commit 为 null。没有 candidate full Evaluate 或 ABBA。标准返回的 `completed` 及 v1 继承的 PASS/latency 不能当作实现成功。没有再次启动 campaign。

## 原 campaign 调查与最终选择

原 B35 campaign `5a0862e9-76cd-4ccc-9f63-3693ff1760ef` 保持原终态：workspace `/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d`，HEAD `b08ae6d7dbc95632a2b2071d94c0ac385ad9b4a6`，episodes1/tokens539414，原 `invalid_handoff`、V0、费用和 unknown 不改。新阶段开始前 writer stopped、无 active episode。

原 LongHorizon 在 latest_version>=0 时可从同 workspace 的累计 state 继续且不重跑 V0；但公开 checkpoint.resume 固定旧 selection/资源，WorkflowStartup 拒绝非 frozen request，record_context 也拒绝改变 selection/argv。不能把新源码和提高后的 max_iters/token admission 直接冒充原普通 resume。标准 API 的终态缓存不是本例主阻塞，实际 B35 workspace 没有 schema-result.json。

根明确拒绝增加 checkpoint amendment/迁移机制，选择一次**新的标准验证 campaign**，明确承担修复版本的新 V0 成本。新 request 使用同 B14 parent，不改旧 request、不重置旧余额。本阶段没有修改公共源码或旧源码快照，也没有开发新的恢复平台。

原 12 项 context/state/version/result/source 的只读快照与 SHA 在 [original-manifest.json](/tmp/schema-b36-standard-resume-20261121/original-manifest.json)。收尾逐项校验一致；B35 request 的限额、hash、root、active_ms 也与新阶段前完全一致。

## 固定来源和零调用预检

- AKA 独立完整 archive 位于 `/tmp/schema-b36-standard-resume-20261121/aka-source`，commit `621481a9b8685f4880e32a9ddce56f4e3d94cd07`，archive SHA256 `5c5299de37ea5165cc038de954fc7eda46b55bc34cdf5c6a2de201e34c6b8f5e`。
- Runner Python `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`；AKA imports 来自新 archive，SDK 为 B32 固定 `atrex_gateway_client 0.14.37`，KS budget 来自 B32 固定源码。见 [fixed-execution-manifest.json](/tmp/schema-b36-standard-resume-20261121/fixed-execution-manifest.json)。
- 原 B33 root.yaml 单文件原文 8076 bytes/SHA256 `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c`；session 首个用户 prompt 确实含逐字原文，见 [schema-context-observed.json](/tmp/schema-b36-standard-resume-20261121/schema-context-observed.json)。不使用误标 Markdown 的 dispatch.yaml。
- Task/reference/input/shapes/metadata 均与 B35/B32 固定输入逐项一致，metadata SHA256 `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740`。见 [input-manifest.json](/tmp/schema-b36-standard-resume-20261121/input-manifest.json)。
- 原 `implement(prepare_only=True)`、真实 composition boot、原 Executor 的 L20D/Triton environment GET 通过；预检前后 parent 账完全一致，模型/GPU调用0。仍为 LightWorkflow、原 Agent、LegacyExecutor、LegacyEvaluatorFactory、LegacyProblemProvider、LegacySourceProvider、GitCandidateWorkspace。
- 独立 archive 上 `tests/test_shell_guard_snapshot.py tests/test_schema_standard.py tests/test_schema_off.py`：**16 passed in 8.43s**。CPU检查没有被当作新的模型/GPU验证。

## 新 campaign 参数

新 campaign `2f944067-0477-4741-a3b9-9f2c51627a6f`；精确 workspace [native](/tmp/schema-b36-standard-resume-20261121/native)。`exact_workspace` 修复生效，实际目录与请求一致。新有限 request `schema-b36-fixed-standard-one-episode` 归同 `/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json`，上界 coder1/model1/GPUjobs3/active1200000ms/tokens=null。三 job 上界明确包含新 V0、一个 candidate full Evaluate 和原 native ABBA，不是复用旧 V0。

模型为原 `qwen3.8-max[1m]`，实际 effort=`max`：原 Light 为固定常量，没有公开 episode effort override，未为改 high 而更改固定源码。agent900s/episode1100s，max_iters1/handoff_resumes0/framework_baseline never/convert_after0/max_stall1；native token admission600000 **不是硬 token/USD 上限**。原 evaluation 为三 shape 单批、native-abba、measurement_repetitions1；verify_repeats2/per-run120s。

真实 argv：

```text
/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python /tmp/schema-b36-standard-resume-20261121/run_standard.py run
```

认证仅由 Supervisor/原 SDK 读取已有私有配置，未放入模型材料。继承 BASH_FUNC_* 在环境复制边界过滤，显式环境校验与原 Python/delete 保护保留。完整参数和组成为 [preflight.json](/tmp/schema-b36-standard-resume-20261121/preflight.json)。

## 新 V0 与唯一 session 的真实结果

新 V0 job `ev_9690cfb0f05a`，succeeded、execution_state=stopped、cleanup_pending=false、execution_quarantined=false，原远端终止事实成立。三 shape all_pass，max_abs_err/max_rel_err0。

| Shape | N | alpha | 新 V0 latency_us |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1 | 11.264000087976456 |
| 1 | 65536 | 0.5 | 11.31199998781085 |
| 2 | 1048576 | 2 | 11.19999960064888 |

原 SDK 只读 `get_job(include_spec=True)` 回执保存在 [native-v0-job.json](/tmp/schema-b36-standard-resume-20261121/native-v0-job.json)。摘要明确 candidate/kernel.py SHA 为原 reference `d55b1beb21b3e27bbe7a91c2b2558ea74187c5e3f798df48a0efb084be051c0c`；实际 job 安装 `atrex-bench==0.1.0.post20261111022558`，selection_reason=image_version_unavailable、source=job_install。run_evaluate.py SHA `4cca1ba1352740e0bc9934c120e3558f4635a43e923da2599751056c34cc4711`。这些是新 V0 的回执，不套用为候选证据。

唯一 session `89427adc-5d08-4cc8-ba93-81290e2c6899`：原 CLI exit=-9、timed_out=true，resume_count0。原 [attempt.json](/tmp/schema-b36-standard-resume-20261121/native/.atrex_long_horizon/episodes/e0001/attempt.json) 明确 `session failed: exit=-9 timeout=True; handoff file is missing`，changed_paths=[]，verification=null。Outcome commit `0f7c0162ccef3a47be818a90046ddfa9bfe91d78` 记录失败 episode，非优化源码。原 stop reason 的“3 episodes without promotion”是原计数/文案；实际只跑了1 episode。

原 [v1.json](/tmp/schema-b36-standard-resume-20261121/native/.atrex_versions/v1.json) 的 correctness PASS 和 latency 属于 incumbent carry-forward，quality_gate FAIL；[kernel.py](/tmp/schema-b36-standard-resume-20261121/native/kernel.py) 仍为原 reference。没有 Triton solution 可交付。

## 工具错误与后续可定位诊断

只统计准确 session 归档中的公开工具调用/结果，不输出模型思维。去重后 28 个调用：13 Bash、15 Read，Write/Edit 为0。记录见 [native-bash-tools-summary.json](/tmp/schema-b36-standard-resume-20261121/native-bash-tools-summary.json)。

1. 普通 `python3 tools/sandbox.py --kind env` 首次因 scratch 目录未创建而重定向失败；创建 scratch 后同一受保护命令 **exit=0**。后续普通 python/python3 也能执行。没有 `_atrex_guarded_python: command not found` 或 readonly-function 错误；本次不是 B35 缺 helper 故障。未读取实际 shell snapshot 正文，不把 CPU 快照测试冒称该文件逐字审计。
2. 一次 records.md 路径读取失败，随后从已安装公开资源找到文件。环境 response 内的 `import_error: ModuleNotFoundError` 是环境探测数据，不能冒称本地 wrapper 执行失败。
3. 后续 `python -m aka.resources.cli list` 和 `gpu-wiki.query` 均返回 `plugin_changed: discovered plugins changed; restore the locked inputs or use a new workspace`。模型停留在文档/工具准备，未写 kernel 或提交评测。相关 Bash 使用 echo 打印 exit，外层 is_error=false 不能覆盖实际 CLI exit1 和 JSON error。

收尾做了**只读 CPU snapshot 比较**，无新模型/GPU调用：同一个固定 plugin 的路径、fingerprint 完全相同；唯一差异是 `plugins[0].commands.query[0]` 的解释器路径。

- 锁定值：`/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`。
- 普通 `python` 发现值：`/usr/local/bin/python`。
- 使用 runner Python 发现时，完整 snapshot 与原锁完全一致；使用普通 Python 时只替换上述一项即可完整相等。

这为后续标准资源发现兼容修复提供可复现证据；没有删除或重写锁、放松其校验、修改固定源码，亦未再收费重试。详见 [plugin-lock-diagnosis.json](/tmp/schema-b36-standard-resume-20261121/plugin-lock-diagnosis.json)、[host Python 比较](/tmp/schema-b36-standard-resume-20261121/plugin-lock-host-python.json)、[runner Python 比较](/tmp/schema-b36-standard-resume-20261121/plugin-lock-runner-python.json)。复现脚本 [inspect_plugin_lock.py](/tmp/schema-b36-standard-resume-20261121/inspect_plugin_lock.py) 只调用 PluginRegistry.snapshot。

下一阶段应检查的准确文件：

- 本次 Agent workspace 的 [.atrex_plugins/lock.json](/tmp/schema-b36-standard-resume-20261121/.atrex-supervisor-runtime/169f5199d9d31be4/journals/episodes/e00000001/agent/workspace/.atrex_plugins/lock.json)，原 native workspace 的锁内容相同。
- [plugin_runtime/execution.py](/tmp/schema-b36-standard-resume-20261121/aka-source/src/aka/resources/plugin_runtime/execution.py:27)：`{python}` 替换为 sys.executable。
- [plugin_runtime/registry.py](/tmp/schema-b36-standard-resume-20261121/aka-source/src/aka/resources/plugin_runtime/registry.py:243)：snapshot 保存 resolved_command；check_lock 在289行比较整个 snapshot。
- [gpu-wiki/plugin.json](/tmp/schema-b36-standard-resume-20261121/aka-source/plugins/gpu-wiki/plugin.json)：query command 使用 `{python}`。
- 本次准确 session 的 [原归档 manifest](/tmp/schema-b36-standard-resume-20261121/native/.atrex_long_horizon/episodes/e0001/.aka-agent-sessions/claude-307945b8e276886cf15cf4b722e1d7df472b095c62150a4f116a249517cbe033/session.json)。原始事件文件为同目录 `0`，只按工具 type/tool_use_id 提取，不输出思维内容。

## 同父结算

新增实际为 **1 host model invocation / 1 accepted GPU job / 954716ms active**。候选 full0/ABBA0；剩余两个 GPU reservation 已按原桥释放。

本次原 model receipt：input_tokens347207、output_tokens0、cache_read/cache_write=null、total_tokens347207、measurement=partial。超时没有完整 terminal 账单；真实 output/cache 总量和 USD unknown，不能把 output0 或347207当完整供应商用量。原 B35 partial539414及历史 token unknown 仍保留。

[Parent-after](/tmp/schema-b36-standard-resume-20261121/parent-after.json)：累计 **7 models / 6 jobs / 3315640ms / 21 operations**；active=null，无 unterminated/uncertain。原 parent 上限24/16/10800000ms不变，剩17models/10jobs/7484360ms；没有新余额。原 B35 request 仍active644060ms、原2models/3jobs/1800000ms限额和原请求hash。

未人工发送 SIGINT；由原900s timeout终止 CLI。没有新公共源码改动、提交、推送、派生 agent。完整收口见 [stage-summary.json](/tmp/schema-b36-standard-resume-20261121/stage-summary.json)、[implementation-result.json](/tmp/schema-b36-standard-resume-20261121/implementation-result.json)。

## 范围失误记录

初始化时误在 home 根执行一次递归 `rg --files` 定位，精确排除了 AKA_experiment 目录但未覆盖 archive 变体，输出枚举到 AKA_experiment_archive 内 AGENTS.md 文件名。没有读取这些文件内容、修改或运行；立即向根报告并停止。此后仅访问已知 AKA、指定报告、指定 /tmp 和本次 owned 进程 locator，未再从 home 递归搜索。

根后来允许本次精确 session 的只读诊断；我通过原 runner 精确 argv 找到 owned CLI PID/UUID，因没有可靠打开的日志 FD 而没有猜路径或枚举 host projects，最终只读原 Agent 在本次 episode 下正常归档的 session 文件，没有直接读取 host .claude 的其他文件或凭据。
