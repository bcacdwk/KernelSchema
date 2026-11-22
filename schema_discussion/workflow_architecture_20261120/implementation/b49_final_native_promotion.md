# B49：已有 Schema 的最终 native 验证

2026-11-21 UTC。**本轮唯一标准请求已完成真实 host promotion：`accepted=true`、verification `PASS`、`reused=false`，5个实际 GPU job 全部终态完成。**产物目录：[B49](/tmp/schema-b49-final-native-20261121)。旧 B45 的 `accepted=false` 与独立 writer `keep` 结果保持原样；本轮只补新的完整 native 验证，不运行优化写作或第二个 Schema 版本。

固定 AKA `55bc39f3239e2d9fa65d2a799182a26df413368c`，KS `13efd1554d112d56e4d767ba43b66257b2d75669`。两者从精确 Git 对象选集导出到新的 `/tmp` source，四个 AKA 包在独立 venv 本地构建安装；新源码和已安装的 B48 shell guard 均为 SHA256 `1aa43a2fd28b4e856cd09953e4c1cf603af0469c0918e820c940f9f9e0973106`。未修改公共源码、共享 Python 环境、旧运行或 KS 原 main 提取。

本轮通过现有 `docs` 参数传入 [B45 候选参考](/tmp/schema-b49-final-native-20261121/materials/candidate-reference.md)：原候选 commit `fe100e75632180c73a63f7e257f8eed36bd20a7a`，2089B/SHA256 `105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f`。它仅是实现输入，不计作本轮正确性或性能成功。Schema 仍为 B40/keep 原字节，9106B/SHA256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`。

Task 沿用 B32/B41/B45 的真实 Atrex-Bench bf16 `Model(alpha).forward(x,residual)=x+alpha*residual`，三个数字 shape：0 `N1024/a1`、1 `N65536/a0.5`、2 `N1048576/a2`，L20D/Triton；四个 Task 文件与37个固定 benchmark runtime 文件均按旧 manifest 校验后复制。没有新 Task 或假 baseline。

| 本轮边界 | 明确参数 |
| --- | --- |
| 入口 | 一次公开 `api.implement`，已有 Schema，`max_iters=1` |
| 原 B14 新请求 | `schema-b49-final-native`；coder/model≤2、accepted GPU jobs≤5、active≤3,000,000ms |
| 原预算与会话 | `handoff_resumes=1`、agent1200s、episode2700s、token admission600000、max_stall1、output_limit_recovery_retries0 |
| 标准检查 | V0、agent full、agent ABBA、原 CandidateGate 六case、host 独立 native ABBA |
| 原评测参数 | `native-abba`、measurement_repetitions1、verify_repeats2、verify_run_timeout120、3 shapes 一批 |
| 资源 | 原 measurement-session/research-records，recording=false、wiki=false，未使用存在悬空链接的可选 episode-report bundle |

600000 tokens 是原准入设置，不是硬 token 或费用上限。没有单独 planner/reviewer、额外算法探索测量或第二次请求。

[零费用预检](/tmp/schema-b49-final-native-20261121/preflight.json) 已完成真实 `prepare_only` 与 composition boot，仍为原 Light/Agent/LegacyExecutor/LegacyEvaluator/Problem/Source/Candidate；[typed 准备](/tmp/schema-b49-final-native-20261121/typed-preparation.json) 使用本轮真实 Task，full 与同allocation ABBA 各一个 step；[安装态](/tmp/schema-b49-final-native-20261121/installed-package-check.json) 7个模块与 guard 字节匹配，AKA/KS help exit0。[SDK 只读环境](/tmp/schema-b49-final-native-20261121/sdk-environment-preflight.json) 与 CLI auth/version/hash 通过。环境 identity=null、availability unknown 保留，不据此捏造跨record可比性。

预检前后 B14 完全一致：9 models / 11 jobs / 4,961,837ms / 33 operations，active=null，无unterminated/uncertain；剩余 coder2/planner4/reviewer15/model15/jobs5/active5,838,163ms。tokens 总数仍 unknown，已知下界2,461,538。新请求注册前再次精确检查同一账目，旧 request、operation、descriptor 摘要用于终态对照，不改任何旧上限。

已遵循官方 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md)。认证仅经原私有配置交给 SDK，不写入模型材料、包或报告。全部实际 job 由原 AKA runtime/Supervisor 提交，原 guard、CandidateGate 与 comparator 未改。

准备阶段两处本地问题已在新目录修正并留痕：pip 起初把旧依赖环境里的同版本 AKA 当作已安装，随后用 `--ignore-installed --no-deps` 将新四包安装到 B49 venv；预检证据导出最初误用 `dataclasses.asdict` 处理 mappingproxy，改用原类型 `as_dict()` 后通过。两者发生于注册和收费前，不是 native 验证失败，也未触发模型/GPU。

## 实际 native 结果

一次 `api.implement` 返回 `completed` / exit0，原 workflow 单 episode 后以 `budget: max-iters` 正常停止。原 [attempt](/tmp/schema-b49-final-native-20261121/native/.atrex_long_horizon/episodes/e0001/attempt.json) 记录 `accepted=true`、host verification `PASS`、`reused=false`；[终态摘要](/tmp/schema-b49-final-native-20261121/native-terminal-summary.json) 保留完整投影。

实际 [kernel.py](/tmp/schema-b49-final-native-20261121/artifacts/native/kernel.py) 为2089B、SHA256 `105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f`，与本轮 public-candidate、candidate Git revision、最终 workspace 和 B45 参考逐字一致。**本轮 candidate_commit 为 `21563597f24e9ab4aca4ffcc2a812e9da8d18363`**；promotion commit 是 `4b41a98516f7ba3681e75d18410c47e09bb385ed`，不混用这两个身份。原 base commit 为 `cd179dc5da0b942b56d3634e2e3ed94564113f1b`。

| 实际 job | 用途与原结果 |
| --- | --- |
| `ev_4c605098b3bc` | 新 V0，原 PyTorch reference；full，3shape各6case全pass |
| `ev_7102dba05dfd` | 本轮候选 full，3shape各6case全pass、max_abs_err0；7.104/7.168/7.408µs，GeoMean7.225490µs，原performance_score1.472111 |
| `ev_7840da47d4aa` | agent同allocation ABBA，完整有效；baseline11.258554µs / candidate7.233772µs，原AKA投影1.556388× |
| `ev_9e107207e34c` | 原CandidateGate correctness_only，3shape各6case全pass、max_abs_err0 |
| `ev_53c06a9e4f27` | **host独立ABBA**，完整有效、原gate PASS；baseline11.226588µs / candidate7.466858µs，原AKA投影**1.503522×**，延迟降低33.489512% |

[五个原始job回执](/tmp/schema-b49-final-native-20261121/native-evidence)、[逐job用途与完整状态](/tmp/schema-b49-final-native-20261121/native-jobs.json)、[原record字段汇总](/tmp/schema-b49-final-native-20261121/native-records-summary.json) 均保留。每个record只含一次attempt，五个job均 `succeeded`、typed completeness=`complete` / validity=`valid`；未重投或另开任务。

host record `3ea86742e7fbd66078184d11901469be19ed5932dffd0ae970e6aa9817b8d1a6` 来自新 `trusted-71809c3c2bedb332e87030fc1e87b591940d8219a23779bea028ee4594071ee6` request；agent ABBA 则是另一record/request。host `reused=false`，其baseline/candidate source SHA均与本轮受托输入匹配。两次ABBA各为1个完整SDK block、每shape各2 baseline/2 candidate采样；逐个完整run的compile/correctness/performance均覆盖shape0/1/2且通过。原 `verify_repeats=2` 对应这一完整ABBA block，不伪称2个SDK blocks。ABBA每完整run使用原1个correctness case；full和原CandidateGate分别保留独立6case结果。

两个ABBA数字各属于自己的同allocation比较。所有record的environment_identity仍null、comparability仍unknown；没有启用跨record复用、构造环境身份或把两次不同allocation混成一个性能结论。eager时延收据含高离散/离群采样，本报告使用原AKA投影，不宣称统计显著性。

## 用量与结账

本轮实际 **1 host coder invocation、5 accepted GPU jobs、657,102 exact tokens**；无planner/reviewer、无writer、无handoff resume、无output-limit recovery。第2个coder reservation未启动，已由原 `release_unstarted` 释放。token组成：input96 + cache-read593875 + cache-write51952 + output11179。原coder receipt exit0、timed_out=false、observation_errors=[]；独立telemetry仍有 `usage_delta_exceeds_terminal` 并标measurement unavailable，原警告保留，不以它改写terminal exact receipt。

新request active **811,673ms**（原 `active_ns=811673646365`），小于3,000,000ms上限。父累计active前后差为811,674ms；原request与parent各自整毫秒值逐字保留，不重写账目以消除1ms投影差异。原native receipt没有USD字段，本轮USD及全局历史总费用仍 **unknown**，未从token换算账单。

B14最终累计 **10 models / 16 jobs / 5,773,511ms / 40 operations**；tokens历史总数仍unknown，已知下界 **3,118,640**。剩余 coder1/planner4/reviewer15/model14、**GPU jobs0**、active5,026,489ms。active=null，unterminated=[]，uncertain=[]。本轮7个原reservation中6个实际启动并终态、1个未启动已释放；见[operations](/tmp/schema-b49-final-native-20261121/native-operations-summary.json)。原15个旧request、旧33个operation、parent descriptor摘要全部未变，见[历史完整性](/tmp/schema-b49-final-native-20261121/historical-unchanged.json)和[parent-after](/tmp/schema-b49-final-native-20261121/parent-after.json)。

只读检查了本次owned会话的公开tool_use/tool_result白名单，没有选择/输出thinking块。7次sandbox相关命令均无报错，未见147/Stopped/plugin_changed/shell_helper_missing；GPU等待期间原heartbeat的active计时冻结。见[公开工具终态](/tmp/schema-b49-final-native-20261121/owned-tools-terminal.json)。这是55bc39新guard的本次真实成功证据，不扩展为对其他Bash版本或绕过wrapper调用的保证。

[最终完整性](/tmp/schema-b49-final-native-20261121/final-integrity.json) 校验全部固定AKA/KS源码、原输入、37个benchmark runtime文件与本轮配置未变；[stage summary](/tmp/schema-b49-final-native-20261121/stage-summary.json) 汇总结果。没有访问、枚举、执行或修改禁止的实验仓及其archive；没有home递归搜索，没有改KS原main、公共源码、旧运行或旧账。费用slot已交回根代理，本轮不再允许任何收费调用。

旧B45 `accepted=false` 和budget exhaustion仍是旧事实；旧writer 34155tokens/USD0.364455/active188933ms、旧native965010tokens/USD未报告均保持原样。本轮成功仅属于新B49请求，不回填旧任务为成功。
