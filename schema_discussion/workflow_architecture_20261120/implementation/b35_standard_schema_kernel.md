# B35：真实标准 Schema-guided AKA 实现阶段

2026-11-21 UTC。本阶段真正进入了原 Light / Agent / Executor / Evaluator，完成必要 V0，并运行一次真实 Schema-guided CLI。**没有交付 Triton kernel**：原 session 在 600 秒超时，候选未改变，episode 为 `invalid_handoff`；没有候选完整评测或 ABBA。不能把返回值 `completed` 或 v1 继承的 `PASS` 当作实现成功。模型/GPU slot 已交回根，未自行启动第二 campaign、planner 或 optimize。

## 固定来源与零调用预检

- 执行源码是 AKA `13424e75eee012475ab3cbee8ae763056efb173f` 的完整 `git archive`，位于 `/tmp/schema-b35-standard-kernel-20261121/aka-source`，包含根已固定的 B32 数字 shape 示例修正。archive SHA256 `c4a0b15a8ab04dbbb808dc10ebbc8db0430ebc28a5404c2393e2d91221739a65`。本次运行没有热改该快照。
- Python：`/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`。AKA 实际 imports 全部来自上述快照；SDK 为 B32 独立固定 `atrex_gateway_client 0.14.37`，KS 只使用 B32 固定 budget 源码，不导入 B34 活跃树。
- 原 Claude CLI `2.1.316`，可执行文件 SHA256 `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`。使用原 `qwen3.8-max[1m]`，effort 为 Light 原配置 `max`；没有套用 B33 write-only 的 xhigh/USD2 profile。
- 原始 Schema 是 B33 提取的 `root.yaml` **单文件**，完整 8076 bytes/SHA256 `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c`。不使用误标 Markdown 的 dispatch.yaml，不改其正文，也不声称 B33 外层 API 已成功。
- Task 从 B32 数字目录复制；本地 bench 来源 `alibaba/atrex-bench@1fa1f0d8a663093a22946270e7d6accb00c33698`。reference/input/shapes/metadata 全部匹配原哈希，metadata SHA256 `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740`。完整清单见 [input-manifest.json](/tmp/schema-b35-standard-kernel-20261121/input-manifest.json)。
- `standard.implement(..., prepare_only=True)` 与真实 composition boot 均通过；组成保持 LightWorkflow、原 Agent、LegacyExecutor、LegacyEvaluatorFactory、LegacyProblemProvider、LegacySourceProvider、GitCandidateWorkspace。见 [preflight.json](/tmp/schema-b35-standard-kernel-20261121/preflight.json)、[prepare-only.json](/tmp/schema-b35-standard-kernel-20261121/prepare-only.json)。
- 原会话中确实收到完整不改字的 root.yaml。记录的用户 prompt 为 30224 bytes，SHA256 `38e3ceeb87f5b06a0dc6409bd4a859f2684dc368e3007ce819a7aaab6c36d79b`；[schema-context-observed.json](/tmp/schema-b35-standard-kernel-20261121/schema-context-observed.json)。同一文本在本地 session 日志重复记录，不据此推断第二次 CLI 调用。

真实 argv：

```text
/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python /tmp/schema-b35-standard-kernel-20261121/run_standard.py run
```

原参数为 `max_iters=1`、framework_baseline=never、convert_after=0、agent timeout600s、episode1200s、handoff_resumes1、native token admission60000、max_stall1。这是原 native admission 阈值，不是硬 token/USD 上限。Evaluation 用原公开配置 shape_batch_size3、native-abba、measurement_repetitions1；原 verify_repeats2/per-run120s。这样必要 V0/full/ABBA 最小路径各一个 GPU job，不伪造旧 baseline 为新 V0。

同 B14 parent 注册有限 request `schema-b35-native-one-episode`，limits 为 coder2/model2/jobs3/active1800000ms/tokens=null，通过 B31 `owner_run_id`/group 桥结算，没有新增账户或余额。认证仅由 Supervisor/原 SDK 从已有私有配置读取，不放入 Schema、Task 或反馈材料。

## 原启动和会话结果

前两次启动均在模型/GPU准入前被原环境验证拒绝：依次为继承的 `BASH_FUNC_which%%`、`BASH_FUNC_module%%`。原始结果保留于 [attempt-1](/tmp/schema-b35-standard-kernel-20261121/attempt-1) 和 [attempt-2](/tmp/schema-b35-standard-kernel-20261121/attempt-2)，合计活动6563ms。剔除继承的 BASH_FUNC_* 后复用同一已注册 request；没有新账户、job 或重投。

实际 session `ad5fb136-27c1-46bd-b8b6-4497d4d21d41`，模型先读取原 Task/skills，再遇到 Claude shell snapshot 只保留 `python3` wrapper、遗漏 `_atrex_guarded_python` 等 helper 的错误。原报错为 `...snapshot-bash-1795243197916-8sxl3h.sh: line 71: _atrex_guarded_python: command not found`。模型自行检查了该 snapshot，也搜索过 shell-snapshots 和 shell rc 中的 helper 定义；没有实现任务所需的代码。

原日志还显示模型自行用 `command python3 tools/sandbox.py --kind env` 查询环境。该行为绕过损坏的 shell function，不能宣称这一轮恢复了原 wrapper 语义或获得文件系统隔离保证；Supervisor 未提供或采用此做法作为修复。它没有提交候选 GPU job。根要求保持原受控包装后，已准备按准确 PID/session 发 SIGINT，但进程在发信号前已因原600秒 timeout退出，因此**实际未发送信号**。没有读/枚举/修改/运行禁止的实验仓，没有人为替写 kernel。

最终原 CLI `exit_status=-9`、`timed_out=true`，原 token admission 禁止自动 continuation，resume_count0。原 native stop reason 为 `stall: 3 episodes without promotion`；真实 `episodes=1`，这是原计数/停止文案，不能称跑了三个 episode。

## 唯一 GPU 结果：原生 V0

Job `ev_61c7e12b7ce7`，succeeded，execution_state=stopped，cleanup_pending=false，execution_quarantined=false；原 `remote_termination=true`。原 native all_pass=true，max_abs_err/max_rel_err0。该 job 是 PyTorch reference incumbent，不是 Schema 生成的 Triton 候选。

| Shape | N | alpha | 原 V0 latency_us |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0 | 11.231999844312668 |
| 1 | 65536 | 0.5 | 11.28000020980835 |
| 2 | 1048576 | 2.0 | 11.264000087976456 |

远端 result.environment：NVIDIA L20D、runtime arch sm_103、torch2.9.0+cu130、Triton3.7.1、Python3.12.3、driver580.105.08、clock_locked=false。此次原标准回执 request_digest=null，因此本次远端 bench wheel/evaluator SHA 不可用；不从 B32 或本地 Git commit 冒推。原回执和摘要见 [native-v0-job.json](/tmp/schema-b35-standard-kernel-20261121/native-v0-job.json)、[native-v0-summary.json](/tmp/schema-b35-standard-kernel-20261121/native-v0-summary.json)。

## 准确的产物和反馈

B31 `implement(workspace=.../native)` 存在入口缺陷：原 Startup 将它改为自动工作区，但 API collector 仍读不存在的请求目录，导致原 [implementation-result.json](/tmp/schema-b35-standard-kernel-20261121/implementation-result.json) 为 completed/空反馈。保留原错误结果，不覆盖它或宣称它正确反映候选成功。

实际 workspace 为 [/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d)。用同一原 `collect_result` 只读投影该真实目录，得到 [native-observed-result.json](/tmp/schema-b35-standard-kernel-20261121/native-observed-result.json)，包含原 versions/state/artifacts/feedback_files。没有追加收费调用。

- [v0.json](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/.atrex_versions/v0.json)：原 reference 测量。
- [v1.json](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/.atrex_versions/v1.json)：long_horizon.status=invalid_handoff，candidate_commit=null；correctness PASS 和性能字段明确为 `canonical_incumbent_carry_forward`/subject incumbent，从 v0 继承，不能解释为新候选通过。
- [attempt.json](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/.atrex_long_horizon/episodes/e0001/attempt.json)：accepted=false、changed_paths=[]、verification=null、timeout和token停止原文。
- [public-candidate.json](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/.atrex_long_horizon/episodes/e0001/public-candidate.json) 与 [schema-native-result.json](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/schema-native-result.json) 同样保留。

当前 [kernel.py](/tmp/schema-b35-standard-kernel-20261121/kernel_opt_native-atrex-bench-vector-add_triton_l20d/kernel.py) 仍是原 reference，SHA256 `d55b1beb21b3e27bbe7a91c2b2558ea74187c5e3f798df48a0efb084be051c0c`。base commit `f481a78342d87ed76f8bdd6e35126ab78f56c57a`；原 outcome commit `b08ae6d7dbc95632a2b2071d94c0ac385ad9b4a6` 记录失败 episode，不是优化源码提交。没有 Triton solution 文件或优化性能结果可交付。

## 同父结算

[postrun-account.json](/tmp/schema-b35-standard-kernel-20261121/postrun-account.json) 是当前有效结算，不能使用 runner 早期 `parent-after.json` 的旧快照。运行结束后 supervisor 已 `account.end`；打印摘要时私有脚本对未启动 operation 的可选 uncertain 字段用了直接索引而报 KeyError。已将该摘要工具改为 `.get` 并重新只读导出当前账户；没有改变账本记录。

- 本阶段：**1 native CLI invocation、1 accepted GPU job、644060ms parent active**，含两次零调用启动错误；model的内部工具/供应商turn不是一次HTTP请求。
- 原回执：input_tokens539414、output_tokens0、cache fields=null、total_tokens539414、measurement=partial。CLI被超时终止，没有完整 terminal 费用；USD、真实输出/cache总量不可用，不能把 partial 计数视为完整账单。
- 父累计：**6 models / 5 jobs / 2360924ms / 17 operations**；active=null，全部终止或未启动释放，无 uncertain/fence。原历史 token unknown 保留。
- 阶段上界未用满，余1 model/2 jobs/1155940ms；这些未用槽已按原桥释放，没有为继续执行重置账户或自动注册新 campaign。

## 三项公开最小修复（未提交，未用于本次已终态运行）

1. `WorkflowLaunch.exact_workspace` 默认false，标准文件 API设true；原 preparation保留指定path/name，仍经过原runtime检查、arch解析、lease与评测。普通CLI自动目录行为保持原约定。
2. `evaluation/tool_requests.py` 只在继承host环境复制边界过滤BASH_FUNC_*；显式用户 `--env` 继续严格校验。
3. 原 `session_shell_guard.sh` helper改用`atrex_`前缀，使Claude public-function snapshot保留完整helper依赖；policy targets在函数内部读既有受控环境值。保留原Python import、gateway path与tracked path检查，没有替换为直通解释器。

改动文件与SHA见 [public-fixes-manifest.json](/tmp/schema-b35-standard-kernel-20261121/public-fixes-manifest.json)，tracked diff见 [public-fixes.patch](/tmp/schema-b35-standard-kernel-20261121/public-fixes.patch)。新 `tests/test_shell_guard_snapshot.py` 被仓库原 tests/ ignore 规则忽略，根提交时需 `git add -f`；其副本已保留在临时根。其他修改包括 tests/test_schema_standard.py 和 docs/schema-standard-workflow.md。

针对性验证：`tests/test_shell_guard_snapshot.py tests/test_schema_standard.py tests/test_schema_off.py`，**16 passed in 15.83s**，git diff --check通过。新真实CPU Bash快照回归同时验证正常python可执行、`import kernel`与受保护gateway删除仍被126拒绝。没有将CPU测试称为新模型/GPU验证，没有重跑旧大套测试、安装服务、提交或推送。

全阶段摘要见 [stage-summary.json](/tmp/schema-b35-standard-kernel-20261121/stage-summary.json)。这些是一次真实执行失败的反馈，B33后续写作只能据此讨论实现环境/工具问题，不能据此推断 root.yaml 的 kernel 数值或性能好坏。
