# B44：当前 Agate 环境元数据与原 AKA 标准结果的最小兼容

2026-11-21 UTC。完成 CPU 修复与检查；**模型调用 0、GPU job 0、网络请求 0**。没有提交或推送。没有交付或声称交付候选 kernel；B41 原失败终态、旧 run、锁、结果、费用与父账均未修改。EXIT147 没有确定复现，未改 SessionControl。

## 判断与已批准语义

基于 `simplified_execution_20261121.md`、B41 完整报告及其保存的原 SDK 响应，问题不是 CLI 截断，也不是 SDK DTO 丢字段。B32 固定 SDK 0.14.37 的 `client.py:get_env/get_capabilities` 直接返回 HTTP response body；B41 原 detail/capabilities 均没有 environment_identity、worker(_id)、image(_digest)。原 `_snapshot` 则只接受显式身份，或从 worker/image 等原字段派生身份。因此 identity=null 是忠实结果。

官方 [atrex-gpu-gateway SKILL.md](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md) 及其 [evaluation.md](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/reference/evaluation.md:41) 明确：

- “Each group performs complete evaluations in A→B→B→A order within one GPU allocation.” 基线与候选共享输入、依赖和 evaluator version；SDK 返回每次完整评测及 ABBA verdict。
- “Reading a GPU environment alias shows the selected member's cache, not image consistency across the pool.” 环境别名的缓存不证明 worker/image 一致，不能拿 route/toolchain hash 冒充它。

原 Light 把 measurement.comparability=comparable 作为报告正确性、full Evaluate、typed-six-case 和 ABBA complete 的共同前提，因而当前服务的所有真实通过结果都被显示为 unknown。这个跨记录环境前提比单次标准服务评测与原 SDK 同 allocation ABBA 所需证据更强。

修复前已向主代理说明语义改变；主代理明确批准：当前 typed Eval 用原终态、源绑定、完整测量和数值结果；native ABBA 额外用原 SDK 完整 schedule/每 shape 计时/verdict。环境 identity 与 comparability 保留 null/unknown；跨记录 minima、正确性与通用测量复用继续严格拒绝。没有恢复旧 KS 严格环境资格或增加新的资格平台。

另确认一个尚未走到的确定性阻塞：真实 capabilities 也没有 operations。原两个执行器对 same_allocation_abba 一律 require_known=True，故 operation=unknown 即拒绝。主代理批准对 Agate 已文档化 typed Eval ABBA 使用原支持/拒绝检查与原 receipt 校验，不再要求这个接口未公开的声明；显式 unsupported 仍提交前拒绝，非 Agate 的 SSH 未知能力仍要求 affirmative support。

## 改动

仅修改 `/home/shaohanyong.shy/atrex-kernel-agent` 的以下 6 个文件。基于分支 `feat/schema-workflows-v1`、公共 HEAD `ef8dadadd9867f298826d52ea159cf70bef93060`；与 B43 的 standard/CLI/iterations 改动分离。

| 文件 | 作用 | SHA256 |
| --- | --- | --- |
| `src/aka/workflow/light/measurements.py` | 原 typed Atrex-Bench eval/ABBA 当前记录的 truth 不再取决于跨记录 comparability；保留原 measurement/source/terminal/SDK schedule 校验。公开结果明确带 environment_identity 和 comparability。 | `dfa50da773dab7e62b2ea9e378c21a06e0c62ffb0b825dfcb234b0cd2bc9d082` |
| `src/aka/execution/legacy/executor.py` | Agate typed ABBA 在缺 operations 声明时不作确定性拒绝；SSH 仍严格。原 immutable prepared envelope 验证保持。 | `16548339d99bff640428d2ab26f1ab75d6a66ca002abd9ca646fb17658e08a36` |
| `src/aka/execution/gateway/executor.py` | 同上，适用于只提供 Agate typed protocol 的 GatewayExecutor。显式 unsupported 保持拒绝。 | `e5a07d4c2970215a6b812627de402f924bb8906f0c6a9e10045ac96f68768a30` |
| `src/aka/execution/job_recovery.py` | 原 Agate replay 使用相同支持语义，并保留/补齐同 first submission 的 immutable typed ABBA envelope 校验。 | `abbd1623c4ea68be24cd19267dac58dd5c2a123ffb718791ec34ce1cc641e279` |
| `tests/test_native_environment_compat.py` | 28 项 CPU 正反例，使用原 CommandEvaluator/Experiment/Light/CandidateGate，执行器只返回标明 synthetic 的本地回执。 | `9671a5a7701ac4e38f0e9da2a814dc5a39fdc1745457060b71c45f8ffe69028a` |
| `tests/fixtures/agate_b41_environment.json` | B41 真实 detail/capabilities 字段及原文件 SHA256；仅省略无关 capabilities.profilers counter catalog。其余字段未改。 | `d83651fca74055e0b96d9d7c76377be165821846f6d5bc60ac311aab2d7e09c8` |

产品代码合计 33 行增加、9 行删除。没有改 environment snapshot/parser、JobResult 或 MeasurementAssessment 合同，也没有改 Evaluation 的数值/完整性/case/ABBA codec、CandidateGate、Experiment 的 reuse/minima 实现。没有新增开关、模型、服务探针、harness、OwnerAccount 或 baseline。

**注意：仓库 `.gitignore:19` 忽略 `tests/`，两个新测试文件需主代理固定时显式 `git add -f`。B44 未暂存任何文件。**

## CPU 检查

```bash
PYTHONPATH=/home/shaohanyong.shy/atrex-kernel-agent/src \
  /tmp/schema-b18-model-budget-usage-20261121/venv/bin/python \
  -m pytest -q tests/test_native_environment_compat.py
```

最终结果 **28 passed in 3.15s**；`git diff --check` 通过。复用原 Python 3.13.12/pytest，无安装。检查覆盖：

- 真实 SDK 字段进入原 parser 后 identity=null；run 支持，ABBA operation 声明未知，availability 仍 unknown。
- 原 FileExperimentRecords→GatewayCommandEvaluator→Light 当前记录成功投影；full Evaluate 和 six-case correctness 正例进入原 Journal/CandidateGate。
- 原 SDK ABBA 完整 A→B→B→A 的三 shape 回执可以 complete；仍为 comparability=unknown，environment_identity=null。
- 缺 shape、缺真实 timing、明确 correctness failure、evaluator error 均不能成为通过。
- ABBA 缺 run、顺序错误、某 shape 无 timing、SDK 明确拒绝 GPU/runtime 环境差异均保持 unknown/incomplete。
- 源绑定缺失、measurement 缺失、identity 不一致、记录未终态、非 typed-native 方法继续拒绝资格。
- 一 case correctness-only 请求即使通过也不获得 typed-six-case，原 CandidateGate 拒绝它。
- identity 未知时，原 reusable、correctness_reuse 和 shape_minima 对 None 或无关已知环境都拒绝跨记录复用。
- Legacy/Gateway/replay 对缺省和显式支持的 typed ABBA 可进入模拟 transport，原支持观测不变；显式不支持零 transport。SSH 未知 capability 零 transport。

## B41 唯一真实结果的只读复核

额外使用当前源码的 `FileExperimentRecords(..., create=False)` 和 `RecordedMeasurements.project` 读取 B41 原 namespace，没有运行或重写记录。原 experiment id `673495144dca5533f3303ba4580dc519ce4ab750550a5ae30141d2f992296e26`，原 job `ev_b58cf6b2a519`。

| 字段 | 当前只读投影 |
| --- | --- |
| 原 record outcome / gate | completed / PASS |
| execution_status / all_pass | completed / true |
| environment_identity / comparability | null / unknown |
| standard_full_evaluation / typed_six_case_correctness | true / true |
| source SHA256 | `d55b1beb21b3e27bbe7a91c2b2558ea74187c5e3f798df48a0efb084be051c0c` |
| shape 0 / 1 / 2 latency_us | 11.247999966144562 / 11.296000331640244 / 11.296000331640244 |

这仍是 **PyTorch reference V0**，没有新候选，不能归作 Schema 实现成功。旧 B41 stage-summary / implementation-result / failed episode 保持原样。

six-case=true 不是补造 case：只读确认原冻结 measurement identity 的 method=atrex-bench/eval、mode=full、workload_scope=full-task、options.num_correctness_cases=6，原 correctness_reuse_policy 已是 typed-six-case/1。真实 `native-v0-original-job.json` 的 shape_selection.correctness_case_count=6，selected_shape_ids=[0,1,2]；每个 correctness.shapes[sid].cases 长度 6，每个 input_seeds[sid].correctness 也长 6。

原字段表示该记录按六 case typed 请求完成测量，**不表示另跑过候选 CandidateGate**。CandidateGate 对所报候选仍独立要求精确 source_sha256 绑定与它的六 case 测量；V0 reference 不能替代不同源码候选。原 Eval codec 负责服务正确性 verdict 和 shape/timing 完整性，六 case 策略由原冻结请求 identity 绑定；本次没有增加或减少这套 case 解释规则。

同时把 B41 保存的**完整** detail/capabilities（含 profiler catalog）重新送入原 `_snapshot` 和 match_environment：run=支持、ABBA operation=unknown、hardware/Triton/num_gpus=支持、identity=null、availability=unknown。原 require_supported 可以接受这种 Agate 未声明状态；没有将 unknown 改成观察到的事实。

## 下一真实阶段的最小预检

先由主代理固定上述 6 文件与 B43 接点，再使用固定源码做准备；不热改活跃工作区，不重跑 B41，也不改旧账。

1. 沿用标准 prepare-only/composition boot，验证所选 Schema、任务三 shape、原 Light/Executor/Evaluator、预算与必需资源；这一步不启动 Agent 或 V0。
2. 用选定 Executor 的一次环境发现（必要时原 SDK 两个只读 GET）检查 run 和 same_allocation_abba 的实际硬件/Triton/requirements/明确拒绝；Agate operations 缺失可按本修复保留 unknown，显式 unsupported 则在任何模型/GPU费用前停止。不要因为 identity=null 重跑 baseline。
3. 用原 Evaluator.prepare / CommandEvaluator.prepare 冻结标准 full、六 case correctness 与 native ABBA 请求；确认原 typed envelope、完整三 shape、六 case 策略和 ABBA blocks/repeats。可复用本 CPU 检查，不提交额外探针 job。
4. 真实实现仍由原 native 流程产生所需 V0、candidate full Evaluate、候选 gate 和 ABBA；只有当这些实际回执完整时才能称候选完成。当前没有任何新远端 ABBA 或候选结果，不能由 CPU fixture 或文档推定性能通过。

发现接口缺字段不再是永远无法完成的资格死锁；真实服务器拒绝请求、未知远端终态、缺 case/计时/schedule、候选数值失败仍按原 native 错误路径处理。跨记录环境可比性未知的限制保留。

访问范围仅指定 AKA 源码、其测试、官方 skill、B32 固定 SDK、B41 精确证据和本报告。未读取、枚举、运行、修改任何 AKA_experiment/archive；未从 home 递归搜索；未读取或复制认证秘密。
