# 简化实现与交付记录

当前按 [2026-11-21 用户简化要求](../simplified_execution_20261121.md) 交付：design/optimize 只写 Schema；implement 复用原 AKA 标准流程。**两个公开入口目标均已实现；B45 实际有限循环产生候选、保留 host 复核失败，并将真实反馈交给 write-only optimize，以 keep 自动停止。B49 独立新 campaign 已完成 host promotion：accepted=true、verification PASS、reused=false，五个实际 GPU job 全终态。**

最终交付配对为 KS `7be9263fe9a6f5fba97ed7ad7685de13e4d6146f` / AKA `87404feb703891efa9d03221d2ae4fdb5068efdc`，均已推送到 `feat/schema-workflows-v1`；具名子模块与 301 文件 release 一致且验证通过，两个开发工作树干净。真实运行使用 KS `13efd1554d112d56e4d767ba43b66257b2d75669` / AKA `55bc39f3239e2d9fa65d2a799182a26df413368c`；此后只发布原字节例子、使用说明和固定配对，工作流运行代码与原运行快照未改。

## 现在可用的路径

| 能力 | 入口与边界 |
| --- | --- |
| 直接写作 | `api.design/optimize` 与 CLI；任意 UTF-8 task，一次 planner，无独立 reviewer/GPU/效果门；保留原稿和诊断 |
| KS 标准实现 | `api.implement` 直接调用 AKA standard，保留原 Light/Agent/Executor/Evaluator/CandidateGate/ABBA |
| KS 有限版本循环 | 唯一 `api.iterate` / `kernelschema iterate`：design 或已有 Schema → native implement → 真实反馈 optimize → revise 后下一版；keep 停止 |
| AKA 有限版本循环 | `--schema --schema-iterate` 只适配参数到同一 KS 循环；保留原 WorkflowLaunch/defaults/patches |
| 固定 Schema 与 off | `--schema --schema-path` 为固定 Schema 下原多轮 coder；默认 off/`--no-schema` 保留原路径；legacy `--schema-config` 分开 |
| 版本与停止 | `rounds` 计版本（默认 2），`max_iters` 计每版原 coder 轮数；每版新 workspace 不自动继承前版 kernel 为 incumbent，也不保证性能单调改善 |
| 反馈与重放 | settled failed 可交 writer；blocked/unsettled/interrupted/无产物/缺反馈停止；已有输出目录拒绝自动重放 |

公开 [vector-add 示例](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/vector_add/README.md) 提供安装链接、静态实现、KS/AKA 有限入口、结果读取及真实边界。B40 root 仍 9,106 B、SHA `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`，生成来源仍 KS907c99。原 native Task/reference metadata 由 AKA 示例提供。新增kernel为2089B/SHA105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f，results只保留公开事实。

## 当前真实证据

- B32 为 PyTorch reference 三 shape PASS；B33 原 API failed、B35/B36 无候选失败、B41 无候选停止均保留。
- B40 一次 `api.optimize` completed/revise：planner1/reviewer0/GPU0、40303 tokens、USD 约 0.58199875，写作完成无效果判定。
- B45 在 KS13efd/AKA88d773 真正执行 `api.iterate`：新 Triton 候选 full 三 shape、每 shape 六 case 正确性通过且 max_abs_err=0；Agent 同 allocation ABBA 通过，原投影 speedup 2.350190×。host 独立 ABBA 因原 4-job group 余额不足而无 job、verification ERROR、accepted=false；未伪造成 promotion。
- B45 将此已结账失败及五份原 feedback 逐字交 optimize，completed/keep 后自动停止，没有第二版。实际 native1coder/4jobs/965010tokens/984471ms；独立 writer1planner/0GPU/34155tokens/188933ms/USD0.364455。合计2次 host 模型/4jobs/999165tokens/1173404ms；native USD 未报告，总 USD unknown。
- B48 45项 CPU 与自然调度验证通过，55bc 已应用原 wrapper 的临时 monitor-off/恢复真实0、7、147。原暂停/保护保持；绕开 wrapper 和任意外层 pipeline 不属覆盖范围。
- B49 一次 `api.implement` 在55bc/13efd完成，host独立ABBA `ev_53c06a9e4f27` PASS、accepted=true/reused=false；baseline GeoMean11.226588µs、candidate7.466858µs，原AKA投影1.503522×。新full和六case gate均三shape各6case/max_abs_err0；本轮通过原docs显式提供B45候选为参考，模型逐字materialize，非第二次纯从Schema生成。1coder/5jobs/657102 exact tokens/request active811673ms、无writer/新Schema，USD未报告。

B49后原parent最终累计10models/16jobs/5773511ms/40operations，idle、unterminated=[]、uncertain=[]；剩coder1/model14/jobs0/active5026489ms。B49 request811673ms与parent增量811674ms的1ms投影差异保留，旧15requests/33operations/descriptor摘要未变，收费slot已释放，后续0调用。历史tokens/全会话USD仍unknown。identity=null/comparability=unknown 保留，单次完整 typed 结果和同 allocation 完整 ABBA 可报告，跨记录 reuse/minima 仍拒绝。

## 简化阶段索引

| 批次 | 记录 |
| --- | --- |
| B30 / B31 | [write-only](b30_simplify_writing.md)；[原 AKA standard](b31_standard_implementation.md) |
| B32 / B33 / B34 | [reference baseline](b32_standard_real_execution.md)；[design 原失败](b33_write_only_real_demo.md)；[原稿保留](b34_write_only_retention.md) |
| B35 / B36 / B39 | [首次标准失败](b35_standard_schema_kernel.md)；[原终态](b36_standard_resume.md)；[插件 Python 兼容](b39_native_plugin_python.md) |
| B40 / B41 / B42 | [真实 optimize](b40_write_only_optimize.md)；[标准原终态](b41_standard_native_implementation.md)；[历史阶段](b42_simplified_delivery_record.md) |
| B43 / B44 / B46 | [有限版本循环](b43_lightweight_schema_iterations.md)；[环境兼容](b44_native_environment_compat.md)；[配对与未结算修复](b46_iteration_delivery_pair.md) |
| B45 / B48 | [真实有限迭代](b45_finite_native_iteration_demo.md)；[原 shell 等待修复](b48_native_shell_wait_fix.md) |
| B49 | [独立完整 native host promotion](b49_final_native_promotion.md) |
| B50 | [当前例子与审计](b50_final_example_and_audit.md)；[交付审计](delivery_audit.md)；[机器进度](progress.json)；[后续](next_steps.md) |

## B01–B13 历史工程基线

以下表格与缺项按原阶段保留。当时的“当前”、`1c9e592 / 804e26f`、GPU0、401 及严格资格要求均为历史状态；Agate 认证/L20D 已在后续打通。旧六 job、fidelity、transport/reviewer 资格不能成为现行默认路径的前置。原失败和 unknown 用量仍保留，历史 [B12审计](b12_final_delivery_audit.md) 不能替代简化入口的最终审计。

### 历史开发位置

- KernelSchema：`/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`，起点 `484902855fe7be82a74fe7c394503d1d7d222dbb`。中断的 checkout 已补全、索引恢复，解锁前验证无 staged/unstaged/untracked 修改。
- AKA：`/home/shaohanyong.shy/atrex-kernel-agent`，分支 `feat/schema-workflows-v1`，起点 `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`，跟踪 `schema-fork` 同名远端分支。
- 原 KernelSchema main 的提取工作保持原位置和分支；实现代理不写原提取文件或计划正文。
- [原准备基线](baseline.json)、[本次开工基线](resume_baseline.json)、[机器可读进度](progress.json)。

### 历史交付表

| 批次/子交付 | 代理 | 状态与证据 |
| --- | --- | --- |
| B01 / K1.materials | b01_k1_materials，gpt-6-astra / xhigh | [已签收 E0 材料](b01_k1_materials.md)，commit `35443d2`；具体设备/profile 与完整 evaluator 尚未资格 |
| B01 / K7.materials | b01_k7_fa4_materials，gpt-6-astra / xhigh | [已签收材料预审](b01_k7_fa4_materials.md)，commit `0c57ef8`；K8/原栈/真实资格未通过 |
| B02 / K2.interface/cpu | b02_k2_core，gpt-6-astra / xhigh | [已签收](b02_k2_core.md)，`c1d7bd9`；59 新 CPU / Template51 / extract6，非业务运行资格 |
| B02 / A1.cpu | b02_a1_episode_seams，gpt-6-astra / xhigh | [已签收](b02_a1_episode_seams.md)，`a668f02`；18 CPU + 上游/off trace 一致 |
| B03 / K1.evaluator | b03_k1_evaluator，gpt-6-astra / xhigh | [已签收工程/CPU](b03_k1_evaluator.md)，`1a6c201`；GPU/服务未资格 |
| B03 / K3 | b03_k3_claude_runtime，gpt-6-astra / xhigh | [已签收](b03_k3_claude_runtime.md)，`5d497db`；46 CPU、三角色CLI profile真实资格；增量只CPU |
| B04 / runtime补丁 | b03_k3_claude_runtime，gpt-6-astra / xhigh | [期限/缺PID费用修复](b04_runtime_admission_fix.md)，`ca2cea4`；26+7项针对性记录 |
| B04 / K4.interface/cpu | b04_k4_reference_loop，gpt-6-astra / xhigh | [已签收](b04_k4_reference_loop.md)，`3a60cf9`；52项由完整49和增量轮覆盖，真实未签 |
| B04 / A2.interface/cpu | b04_a2_task_adapter，gpt-6-astra / xhigh | [已签收](b04_a2_task_adapter.md)，`c984d14`；原生Task/完整CPU evaluator与off验证 |
| B05 / K5.interface/cpu | b05_k5_writing，gpt-6-astra / xhigh | [已签收CPU](b05_k5_writing.md)，`878b1d2`；46项分轮覆盖，环境/行动整合继续归K6 |
| B05 / A3.interface/cpu | b05_a3_optional_module，gpt-6-astra / xhigh | [已签收CPU](b05_a3_optional_module.md)，`03a3f10`；联合53、focused20及末次1项 |
| B06 / A4.cpu | b06_a4_seed_lifecycle，gpt-6-astra / xhigh | [已签收CPU](b06_a4_seed_lifecycle.md)，`106400d`；35 A4 cases分轮覆盖，A1/off18、A2相关6通过 |
| B07 / A5.interface/cpu | b07_a5_fixed_backend，gpt-6-astra / xhigh | [已签收CPU](b07_a5_fixed_backend.md)，`4be322b`；23主集成+4最终入口定点，真实未签 |
| B06 / K6整合 | b06_k6_standalone_integration，gpt-6-astra / xhigh | [工程已交付](b06_k6_standalone_integration.md)，`d0f2b04`；真实MODEL/GPU为0 |
| B07 / active budget | b04_k4_reference_loop，gpt-6-astra / xhigh | [check_active已交付](b07_budget_active_check.md)，`067b37d`；43预算测试通过 |
| B07 / KS独立审阅 | b07_ks_independent_review，gpt-6-astra / xhigh | [已完成并修复](b07_ks_independent_review.md)，`b1049e1`；完整320+最终9定点 |
| B08 / K8初始工程 | b08_k8_fa4_engineering，gpt-6-astra / xhigh | [Source/case已交付](b08_k8_fa4_engineering.md)，`80e006e`；后续KS fixed业务接线见B09/B10 |
| B08 / AKA独立审阅 | b08_aka_fixed_review，gpt-6-astra / xhigh | [已完成工程审阅](b08_aka_fixed_review.md)，`879cfea`；119 CPU+最后5边界，真实probe一次timeout未资格 |
| B09 / A6.cpu | b09_a6_dynamic_workflow，gpt-6-astra / xhigh | [已交付CPU](b09_a6_dynamic_workflow.md)，`04fab50`；26动态/23兼容/6审计新配对通过 |
| B10 / AKA动态最终审阅 | b10_aka_dynamic_final_review，gpt-6-astra / xhigh | [已签收工程](b10_aka_dynamic_final_review.md)，`e867333`；155完整/26+5配对增量/10原入口；真实coder1次超时、费用unknown |
| B09 / K8 KS接线 | b09_k8_workflow_bridge，gpt-6-astra / xhigh | [KS fixed链已交付](b09_k8_workflow_bridge.md)，`8e846d7`；277稳定组及后续定点，真实门未签 |
| B10 / KS最终独立审阅 | b10_ks_final_review，gpt-6-astra / xhigh | [已签收](b10_ks_final_review.md)，`76ef9b1`；修前380、修后21新增分轮/56兼容及最后4定点、独立源码安装/正式FA4 blocked幂等 |
| B10 / 独立design试验 | b10_k5_design_trial，gpt-6-astra / xhigh | [实际试验报告](b10_k5_design_trial.md)：unbound预检通过；planner1次120秒超时，费用未知、reviewer0/GPU0，原终态零调用恢复已核 |
| B11 / 业务期限配置修复 | b11_runtime_deadline_config，gpt-6-astra / xhigh | [已签收](b11_runtime_deadline_config.md)，`1c9e592`；默认120/显式长配置仍取原deadline，34初轮/9修后/4写作兼容 |
| B12 / 最后交付核对 | b12_final_delivery_audit，gpt-6-astra / xhigh | [最终KS1c9/AKA804已核](b12_final_delivery_audit.md)，无待修工程阻断；版本/源码/工件/旧算子/证据边界一致 |
| B13 / 资格scope配置 | b13_probe_scope_limits，gpt-6-astra / xhigh | [已签收](b13_probe_scope_limits.md)，`804e26f`；11scope/10恢复/3owner缺失/3最终安装分轮通过，无新外部调用 |

派工不等于 gate 通过。子代理交付后按实际文件、必要验证及缺项签收，固定可供下一批消费的源码版本；真实资格单列，不用 CPU 记录代签。

当前 CPU 开发与测试使用隔离的 Python 3.13.12。可选的 Python 3.11 下载已停止，兼容性验证后置；不能据当前测试宣称 3.11 已通过。未替换系统解释器，未影响活动提取。


### 当时的缺项与交接

- Agate：用户确认已配置；本会话已按现成 skill/SDK 连接生产端，但未找到认证入口，GET env 返回401。常规可见MCP资源为空，Claude MCP list也没有配置server。不能据此说用户服务不可用；目前0个GPU作业。
- K6已提供内置Agate channel、6-job资格准备、typed target Contract和实际driver，无需用户编写胶水。实际认证/ACL/只读执行和完整结果回收仍须后续资格，KS最后工程审阅已完成，实际资格仍未通过。
- API入口统一 `kernelschema.api`，避免同名backend模块导入覆盖包根函数。`14a7246`已修复并通过12项API测试。
- 452份CLI完整私有capture原位保留但不提交/打包；源码只发布必要profile和说明摘要。历史K3调用4次/15291 tokens、reported USD约0.232673；另AKA旧coder一次超时，tokens/费用unknown，再加本轮KS design planner与AKA新coder各一次超时，因此目前总调用7次、总用量费用未知。B10新增试验以progress/最终报告据实更新，非核实账单。
- A3首发沿用原Agent/NativeSession，采用宿主完整资料投递/no-tools和宿主源码安装；当前模型不自行调用read/effective/check。AKA native profile真实资格不能由KS transport记录代签。
