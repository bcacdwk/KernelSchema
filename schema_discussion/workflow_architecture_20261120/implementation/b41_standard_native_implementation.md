# B41：标准 Schema 入口的原生 AKA 实现

2026-11-21 UTC。**阶段已终态，没有交付 Triton kernel。** 已用 B40 原 Schema 通过 KS API 进入原 Light，完成新 V0；运行中确认原环境身份缺失会阻断候选标准测量资格，主代理据此授权停止第一调用，不启动第二 continuation。原结果为 `invalid_handoff`，candidate=null、changed_paths=[]，无候选 full Evaluate、六 seed 验收或 ABBA。真实用量 **1 host model invocation / 1 accepted V0 GPU job / 661725ms**，父账已 idle、所有 operation 已终止或释放。未改旧 campaign、锁、失败记录或余额。

## 固定来源与入口

- AKA 固定 `ef8dadadd9867f298826d52ea159cf70bef93060`，独立源码 `/tmp/schema-b41-standard-native-20261121/aka-source`，包含 B38 当前配对与 B39 插件 Python 兼容修复；snapshot SHA256 `5095ad2bdee8fc0554c93e5400790cc69f5aab327764edd192f6a2fe5ff9d28b`。
- KS 固定 `907c99aa1589bddf73e26079f2304b83d68a8e51`，独立源码 `/tmp/schema-b41-standard-native-20261121/ks-source`；snapshot SHA256 `ab3f5ac347f501a6075357909d035f152e5f9ddae14ce318996a175d2c2a9afa`。不使用 KS main，也不修改算子提取。
- 复用 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python` 和 B32 固定 SDK `0.14.37`；不安装或下载 Python，不热装活跃仓库。
- 运行入口预备为 `kernelschema.api.implement`，其原代码直接委托 `aka.workflow.schema.standard.implement`；由原 Light、Agent、Executor、Evaluator、CandidateGate、ABBA 完成实现与测量。没有自定义 coder/评测/严格资格或迁移框架。
- Task 来自 B36 固定数字目录，reference/input/shapes/metadata 已逐项验证 SHA256 与 B32/B36 一致。bf16 `x + alpha * residual`，`(N, alpha)` 为 `(1024, 1.0)`、`(65536, 0.5)`、`(1048576, 2.0)`；metadata 是真实 PyTorch reference 测量，不是生产 kernel 声明。
- B33 root 只在临时目录保留为 `prepare-only-b33-root.yaml`，未选为实现输入。实际 B40 原模型 root 位于 `/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/6dcd8f97b77f4bbd8ed96a01758feff6/raw/root.yaml`，完整 9106 bytes，SHA256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`；已逐字复制到本次 `materials/root.yaml`。

来源和本地配置见 [fixed-execution-manifest.json](/tmp/schema-b41-standard-native-20261121/fixed-execution-manifest.json)、[input-manifest.json](/tmp/schema-b41-standard-native-20261121/input-manifest.json)、[run_standard.py](/tmp/schema-b41-standard-native-20261121/run_standard.py)。实际 B40 Schema 的 KS `api.implement(prepare_only=True)`、真实 composition boot 和原 SDK/Executor L20D/Triton environment GET 均通过；预检前后父账完全一致、工作区未创建，预检模型/GPU调用0。模块实际来自上述固定源码和 B32 SDK，原组成保持 LightWorkflow、Agent、LegacyExecutor、LegacyEvaluatorFactory、LegacyProblemProvider、LegacySourceProvider、GitCandidateWorkspace。证据见 [preflight.json](/tmp/schema-b41-standard-native-20261121/preflight.json) 与 [sdk-environment-preflight.json](/tmp/schema-b41-standard-native-20261121/sdk-environment-preflight.json)。

## 已批准的最小运行参数

主代理批准 `max_iters=1`、`handoff_resumes=1`，使用原生同 episode continuation/repair。首个有效终端报告即结束，不因有第二个模型槽而新跑 episode。agent timeout `900s`、episode timeout `2100s`，parent 新有限 request 为最多 **2 host model invocations / 5 accepted GPU jobs / 2400000ms active time**。`max_stall=1`、原 token admission `600000`、`output_limit_recovery_retries=0`、`setup_timeout=600`。原 token admission 不是硬 token 或美元上限；Light effort 仍为原固定 `max`、模型 `qwen3.8-max[1m]`。

保留原 `framework_baseline=never`、`convert_after=0`、`shape_batch_size=3`、`comparison_method=native-abba`、`measurement_repetitions=1`、`verify_repeats=2`、每 run `120s`。`docs=()`；Schema 原文作为设计输入，不作为评测证据。

资源使用 B39 已有 patch，仅本次调用设置 `plugins=[]`、bundles 为 `measurement-session` 和 `research-records`；保留 gpu-measurement、runtime-records、`tools/sandbox.py`，原 `wiki=false` / `research_records=false`。相较 B39 patch 只把 episode timeout 从 1100 改为获批 2100。没有改原 AKA 默认功能。

正常成功路径实际预计为 **4 个 accepted GPU jobs**：新 V0、候选 full Evaluate、原 CandidateGate 的六 seed 正确性验收、原 native ABBA。此前 B35/B36 参数说明中的 V0/full/ABBA 三 job 估算遗漏了原生 CandidateGate；这不同于已撤下的旧 KS 自定义六-job资格，必须保留。第五个槽仅容纳一次必要失败重测，不做探索性 compile/profile 或分 shape 提交。

本次 notes 明确要求先读一次公开测量 skill 后实现所选 root 候选，使用普通受保护 Python 和原 sandbox，按原 episode-report 提交真实测量；访问范围保持在该 managed workspace 和已安装公开资料。不开展无关主机/项目/凭证排查。

## 同父结算

复用原 `/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json`，B41 run id 为 `schema-b41-standard-native`；实际执行前仍使用原 `register/begin` 和现有 optional finite-group 适配，结束使用原 `end` 与收据。认证只由 Supervisor/SDK 读取原私有配置，不复制入源码或模型材料。

只读准备及实际启动前父累计为 **7 models / 6 jobs / 3315640ms / 21 operations**，active 为 null，未终止/uncertain 均为空，历史 `tokens unknown` 保留。主代理在 B40 真实终态后移交独占 slot，B41 只注册了这一个新 request。

最终父累计 **8 models / 7 jobs / 3977366ms / 28 operations**，active=null、unterminated=[]、uncertain=[]。B41 request active 为 661725ms，父累计差为 661726ms（原整数毫秒结算取整差 1ms）。未启动的 **1 model / 4 evaluator slots** 由原 group 释放；B35/B36 原 request 的完整字段与 B36 终态存留值逐项一致。见 [parent-after.json](/tmp/schema-b41-standard-native-20261121/parent-after.json)、[operation-settlement.json](/tmp/schema-b41-standard-native-20261121/operation-settlement.json)、[old-requests-unchanged.json](/tmp/schema-b41-standard-native-20261121/old-requests-unchanged.json)。收尾再次确认 owned CLI PID 已不存在、active_episode 文件已清除、原 writer-state=stopped。

原 native usage 为 input/total `239774`、output `0`、cache fields=null、measurement=`partial`。这是被停止调用的原不完整回执，**不是完整 token 账单**；真实 output/cache 总量及 USD unknown，历史 token unknown 保留。没有将 B40 的独立写作费用混入本次 B14 parent 统计。

## 唯一 GPU 结果与实际终态

新 V0 job `ev_b58cf6b2a519` 为 PyTorch reference；原 native all_pass=true、误差0。结束后通过同 SDK 只读 GET 确认 succeeded、execution_state=stopped、cleanup_pending=false、execution_quarantined=false，父账不变。

| Shape | N | alpha | V0 reference latency_us |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0 | 11.247999966144562 |
| 1 | 65536 | 0.5 | 11.296000331640244 |
| 2 | 1048576 | 2.0 | 11.296000331640244 |

原远端环境为 NVIDIA L20D/sm_103、torch2.9.0+cu130、Triton3.7.1、Python3.12.3、driver580.105.08，clock_locked=false。原命令回执与收尾 GET 分别保留在 [native-v0-original-job.json](/tmp/schema-b41-standard-native-20261121/native-v0-original-job.json)、[native-v0-job.json](/tmp/schema-b41-standard-native-20261121/native-v0-job.json)，指标见 [native-v0-measurement.json](/tmp/schema-b41-standard-native-20261121/native-v0-measurement.json)。没有据这些 reference 数字推断 B40 Schema 的数值或性能效果。

本阶段 CLI session `700da1ff-a1fb-4956-a839-0ac5f0b80888`。模型确实收到完整原 B40 root，user prompt 中原文字节校验见 [schema-context-observed.json](/tmp/schema-b41-standard-native-20261121/schema-context-observed.json)。精确 owned JSONL 中可去重确认 13 Bash / 6 Read、0 Write/Edit；重复 session 记录不解释为新模型 invocation。仅提取公开工具请求与结果，没有导出模型思维或秘密。

确认环境身份是确定性阻塞后，主代理明确授权立即停止且禁止第二 continuation。Supervisor 验证 PID/session/cwd 后，使用原 `aka.agent.execution.process.signal_process_groups` 向该 owned CLI process group 发送 **SIGKILL**，保留原 runner 自行回收和结账，没有修改现场配置、保护规则、锁或账。选定非 timeout 的原进程终止返回，避免进入已取消的 timeout/terminal-handoff continuation。完整操作见 [supervisor-stop.json](/tmp/schema-b41-standard-native-20261121/supervisor-stop.json)。

原 [attempt.json](/tmp/schema-b41-standard-native-20261121/native/.atrex_long_horizon/episodes/e0001/attempt.json) 为 exit=-9、timeout=false、resume_count=0、candidate_commit=null、changed_paths=[]、verification=null、handoff missing。base commit `17b0e3ab00517c0d7286358f97b0f65fa957d877`，outcome commit `a2d3807e79f83d95cdc6791c1cafd285b23c33a9` 只记录失败 episode；主工作区 `kernel.py` 仍是 reference，SHA256 `d55b1beb21b3e27bbe7a91c2b2558ea74187c5e3f798df48a0efb084be051c0c`。

原 API 返回 `completed`、stop reason 为 `stall: 3 episodes without promotion`；实际只运行 **1 episode**。保留此原文，不称跑过三个 episode，也不把 v1 的 incumbent carry-forward PASS 视为候选成功。原结果 [implementation-result.json](/tmp/schema-b41-standard-native-20261121/implementation-result.json)、[schema-native-result.json](/tmp/schema-b41-standard-native-20261121/native/schema-native-result.json) 未重写为成功。

## 确定性阻塞：环境身份与原测量资格

原 environment GET 能返回 L20D/Triton 的真实支持信息，所以 `require_supported` 和启动预检通过。但当前 Gateway/SDK 的 detail 与 capabilities 均没有 `environment_identity`、`worker_id`/`worker`、`image_digest`/`image`。原 `execution/environment_query.py:_snapshot` 只接受显式身份，或由 worker/image 派生身份；因此生成 identity=null。收尾用原 SDK 再做两次只读 GET，并把真实返回送回同一个原 parser，仍得到 null；未提交 GPU、未改变 parent。

- [sdk-env-detail.json](/tmp/schema-b41-standard-native-20261121/sdk-env-detail.json)：原 SDK `get_env('L20D')` 字段。
- [sdk-env-capabilities.json](/tmp/schema-b41-standard-native-20261121/sdk-env-capabilities.json)：原 SDK `get_capabilities('L20D')` 字段。
- [sdk-environment-parser-check.json](/tmp/schema-b41-standard-native-20261121/sdk-environment-parser-check.json)：原 parser 与 identity 输入字段的只读检查。
- [environment-identity-diagnosis.json](/tmp/schema-b41-standard-native-20261121/environment-identity-diagnosis.json)：V0 原环境 snapshot、实际字段、代码链与判断。

这会挡住新候选，不只是 V0 的显示限制：原 `LegacyExecutor._run` 将 `support.environment.identity` 写入所有 JobResult；原 `evaluation/measurement.py` 在无身份时把 comparability 置 unknown。原 Light `measurements.py` 的 known 条件要求 comparable，于是 all_pass/correct=null、execution_status=unknown、standard_full_evaluation=false、typed_six_case_correctness=false，ABBA 也不能 complete。原 `host/journal.py:_passing_evaluate` 必须取得 passing full Evaluate，原 CandidateGate 还要求 completed 的 typed-six-case evidence；因此在当前固定环境中，远端结果即使数值通过也不能完成原 `candidate_ready` 与 retention。

模型通过原受保护 Python 读取的 V0 `kernel-records` 和 `record-read` 已实际显示 `qualification: environment-identity-missing`。同 episode continuation 不能修复该冻结的 support snapshot。下一修复应在原 Execution 环境发现/身份字段与当前 Gateway SDK 的兼容边界核对真实证据；本阶段没有编造身份、降低 gate、另造资格规则或热修运行现场。

## EXIT147 的可定位证据与不确定性

本次 `env`、`kernel-records`、`kernel-pareto-frontier` 三种公开查询均出现：工具已经输出正常 JSON，但 Bash 层打印 `EXIT=147` 与 `[1]+ Stopped ...`。`kernel-read` 成功 exit0，后续 `record-read` 仍能读到原 V0；未出现 B36 的 `plugin_changed` 或 shell helper missing。

`147 = 128 + SIGSTOP(19)`，原 `aka.agent.execution.control.SessionControl.request` 确实用 SIGSTOP/SIGCONT 暂停/恢复 owned Agent tree，因此作业控制兼容是一个具体调查方向。但本阶段的最小 CPU fixture 使用同一原 SessionControl，在 Bash monitor on/off 各一次都得到 child_exit0，**没有复现**；不能把这一关联写成已证明根因，也不能断言它独立阻断 candidate。

精确工具请求/公开结果见 [owned-tools-summary.json](/tmp/schema-b41-standard-native-20261121/owned-tools-summary.json)，原 CPU fixture 与观察见 [reproduce_job_control.py](/tmp/schema-b41-standard-native-20261121/reproduce_job_control.py)、[job-control-cpu.json](/tmp/schema-b41-standard-native-20261121/job-control-cpu.json)。fixture 没有启动模型/GPU、未编辑 live workspace，不用于掩盖真实失败。

完整终态和费用摘要：[stage-summary.json](/tmp/schema-b41-standard-native-20261121/stage-summary.json)。收费 slot 已交还根；本阶段不再启动任何模型、GPU、续跑或新 campaign。

使用官方 [atrex-gpu-gateway SKILL.md](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md)。全阶段只访问指定 AKA/KS 独立源码、精确历史证据和本临时目录；未读取、枚举、运行或修改任何禁止的 AKA_experiment 或其 archive 变体，未从 home 递归搜索。
