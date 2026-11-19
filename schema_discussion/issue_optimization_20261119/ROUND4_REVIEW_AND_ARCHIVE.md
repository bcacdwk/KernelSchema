# 第四轮删减复核与 15 项归档

2026-11-19。独立复核代理 `/root/issue_r4_reduction_review_xhigh`，`gpt-6-astra / xhigh`。本轮依当前 AGENTS.md，只复核第三轮删减及 MLA/GEMM 已有限域修订，整理原 15 个稳定 ID；没有新增一轮架构或算子全树语义审计。

**删减通过，当前代码稳定，无须恢复或修改生产、测试文件。** 15 项已归档为 **12 resolved、2 declined、1 partially_adopted**；已解决/已采纳改动均属当前 **2.5.1**，不发布新版本。Supervisor 统一全 extract 离线套件运行中；下文既有定向结果不冒充本轮全套通过。

## 删减判断

直接比较第二轮 `after_implementation`、第三轮 `after_reduction` 的 manifest 与当前文件。两轮快照只相差下列 11 个文件；开始本轮复核时，第三轮 manifest 的 1116 个文件与当前字节全部一致。快照位于 `/tmp/kernel-schema-issue-opt-20261119-zbocalki/`，不作为 issue 的唯一长期依据。

| 范围 | 实际删减与判断 |
| --- | --- |
| `schema_extract/tools/publication.py` | 删除 markers 临时变量及空列表分支。`execution_binding` 非空须对应非空且一致的 binding history；`verify_bindings` 逐项核验唯一的 marked 授权，`delivery_audit_markers` 逐项生成标记，因此合法发布结果等价。未恢复无标记授权支线。 |
| `schema_extract/tools/snapshots.py` | 两个格式分支的同一 `Path(lock["snapshot"])` 赋值前移；分支、身份、manifest 和固定 Git 字节检查原样保留。 |
| `schema_extract/tools/artifacts.py`、`receipts.py` | 只缩短注释和 docstring。完整 inventory、原角色重交以及身份/来源/权限硬错误分类不变，手册仍明确这些职责。 |
| `schema_extract/tools/tests/test_executor_repairs.py`、`test_current_recovery.py` | 删除未使用的 lock 读取、重复注释及随后被覆盖的局部赋值。真实终止确认仍调用，后续确认结果及所有断言保留，没有删独立覆盖。 |
| `schema_extract/README.md`、`tools/README.md` | 命令和完整授权字段保留，执行协议链接 Supervisor 手册；原已哈希 scope、保稿来源、未完成诊断、提交前重核及严格封印复用各有明确落点。没有删正常阶段/P6、控制、维护或在途屏障。 |
| `schema_docs/1.contract.algorithm_requirement.md` | 去掉 meaning 和开放类型的泛义复述。对象/元素原则、三个例子以及 None/default/nullable 和不凭样本收窄 range 的边界仍清楚。 |
| `schema_docs/5.design.tile_pipeline.md` | 去掉纯存储无需独立执行单元键的重复附句；DOC2 仍直接定义该规则，location 继续引用所属执行单元摘要。 |
| `schema_extract/prompt/phases/06_final.md` | 合并 Version 更新重复说明；获准非 root 内容的隐含更新权、不单列 scope、Runner 校验与 root 只读仍明确。 |

续接保留来源/身份/权限硬核验与未完成诊断的分工；普通、恢复及裁决 P6 保存稿采用同一最新 draft 规则。原批准、旧 response、调用计数不重写，pause/cancel/hold、维护和在途 fence 不弱化。并发 capsule 只复用完整、无链接、已封印且内容一致的结果。审计点仍绑定真实 stage/attempt，未登记调用用当前阶段和 null。

monitor 的代码及测试在第三轮未变：仅重试锁进入，真实采样与完成时间分开，逐 run 旧观察不覆盖新值，unknown owner 不抢占；北京时间仅影响输出副本。normalizer、双角色 I/O、开放 dtype、存储名和 Version 边界保留。上述范围没有需要最小恢复的实错。

## MLA/GEMM 限域复核

本轮读取当前 root 差异及两份作者修订记录，并核对 after manifest 和原 receipt 摘要；直接复核改动含义。固定来源逐字比对和精确 4/2 字段的独立核对由[第三轮](ROUND3_MONITOR_SCHEMA.md)实际完成，本轮没有将其重述为一次新的源码全量审计。

| 实例 | 当前修订与结论 | 当前证据 |
| --- | --- | --- |
| MLA | Version 4→5；`sage_attn_sfs` 的整体 `python_tuple` 对应 `shape: []`，meaning 不把默认成员当固定域；相邻整数 tuple 保留四元素轴、`python_int`，只澄清容器 meaning。四字段修订必要，默认值和其余内容保持。 | [人工修订记录](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/issue-review-20261119.json)；当前 root SHA-256 `367704f500ce6078cc488b6e0ee809ddf61a4b862024e549f156557a01d061d8` |
| GEMM | Version 3→4；既有 thread 摘要补 thread-local `register`，为 root/wide 已有 Storage.location 提供名称与范围。两字段修订必要，不推导编译后分配、容量或无 spill；wide 和 dispatch 不变。 | [人工修订记录](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/issue-review-20261119.json)；当前 root SHA-256 `cba4ea9d28e94f3541ed0a60e255ff9b188bcaa5635d897d68475b4d42c3f92d` |

第三、四轮构成对上述小修及说明的独立人工复核，范围限于列出的字段。原 P6 及原 receipt 仍只覆盖原 manifest；两份作者修订记录、原 receipt 和正式 YAML 本轮均未改写。FMHA/Sparse 的专项语义复核尚未在本轮完成，旧待审计事实保留。

## 实际验证

| 验证 | 结果与责任 |
| --- | --- |
| R2→R3 manifest、R3→本轮开始时字节核对 | 11 文件删减；1116 文件与 R3 manifest 相符。第四轮实际执行。 |
| 删减文件 AST | 6 个 Python 文件均可解析；结合调用点复核，没有需要代码修复的差异。第四轮实际执行。 |
| MLA/GEMM 当前 after manifest、原 receipt 摘要及 root 文本差异 | 全部相符，仅记录所列 4/2 字段变化。第四轮实际执行。 |
| 15 项台账与导航完整性 | 检查结果见 [ROUND4_CHECKS.json](ROUND4_CHECKS.json)：原 ID、类别、问题、提案、必要性不变，原 evidence 保留；全部仅有一份正式记录并处于对应 archive，新增引用可定位。 |
| 前三轮定向验证 | [R2 运行](ROUND2_RUNTIME.md)的 30 个独立用例取得通过结果；[R2 monitor/Schema](ROUND2_MONITOR_SCHEMA.md)的 27 项 monitor、76 项 Template 回归及两算子全目录只读检查通过；[R3 运行](ROUND3_RUNTIME.md)的 5 项定向检查通过。这些是各轮实际结果，本轮没有重复执行或扩大其结论。 |
| Supervisor 全 extract 离线套件 | 运行中，由根代理继续汇总并另写统一验证记录；本轮删减复核与归档职责已完成，没有提前宣称全套通过。 |

代码与测试已在向 Supervisor 报告“稳定”后冻结；本轮未再运行重复的全套或无关语义检查。未操作真实 extract/resume/cancel/monitor、模型或 GPU；未写入 `.schema_extract/v2`、历史 run、Schema 或原 receipt，未 commit/push，AGENTS.md 保留外部更新。

## 15 项最终处置

| 稳定 ID | 最终结果 | 最小处置与保留边界 |
| --- | --- | --- |
| [maintenance-empty-poll-control-lock](../../schema_extract/issues/archive/extract_tools/maintenance-empty-poll-control-lock.json) | `resolved` | 保留空轮询快路径与控制屏障 |
| [failed-node-recovery](../../schema_extract/issues/archive/extract_tools/failed-node-recovery.json) | `resolved` | 保留 scoped 拒因反馈及原恢复边界 |
| [p6-retained-scope-inventory](../../schema_extract/issues/archive/extract_tools/p6-retained-scope-inventory.json) | `resolved` | 完整 inventory 校验保留范围 |
| [receipt-evidence-repair](../../schema_extract/issues/archive/extract_tools/receipt-evidence-repair.json) | `resolved` | 原角色有限重交，硬错误不降级 |
| [progress-preserving-continuation](../../schema_extract/issues/archive/extract_workflow/progress-preserving-continuation.json) | `partially_adopted` | 工具修复续接完成；语义/来源迁移不支持 |
| [monitor-registration-lock-retry](../../schema_extract/issues/archive/extract_tools/monitor-registration-lock-retry.json) | `resolved` | 仅锁进入重试，等待可见且可中断 |
| [monitor-wait-stale-observation](../../schema_extract/issues/archive/extract_tools/monitor-wait-stale-observation.json) | `resolved` | 真实采样不倒退，unknown owner 不抢占 |
| [monitor-beijing-time-display](../../schema_extract/issues/archive/extract_tools/monitor-beijing-time-display.json) | `resolved` | UTC+8 仅用于输出副本 |
| [normalizer-diff-not-visible](../../schema_extract/issues/archive/schema_template/normalizer-diff-not-visible.json) | `resolved` | 保留首差异位置与短上下文 |
| [inplace-output-input-name-diagnostic](../../schema_extract/issues/archive/schema_template/inplace-output-input-name-diagnostic.json) | `resolved` | 保留双方定位及双角色指引 |
| [maintenance-version-bump-outside-field-cap](../../schema_extract/issues/archive/extract_workflow/maintenance-version-bump-outside-field-cap.json) | `resolved` | Version 隐含更新权和 Runner 校验清楚 |
| [tuple-abi-input-typing](../../schema_extract/issues/archive/schema_docs/tuple-abi-input-typing.json) | `declined` | 拒绝新增类型机制；说明及 MLA 实例已修 |
| [inplace-output-input-registration](../../schema_extract/issues/archive/schema_docs/inplace-output-input-registration.json) | `declined` | 拒绝重复义务；现有双角色规则保留 |
| [non-tensor-parameter-registration](../../schema_extract/issues/archive/schema_docs/non-tensor-parameter-registration.json) | `resolved` | 对象/元素、None/default 说明及 MLA 修正 |
| [location-storage-level-registration](../../schema_extract/issues/archive/extract_workflow/location-storage-level-registration.json) | `resolved` | 统一存储名来源并补 GEMM thread 摘要 |

三项原 pending 已移到对应 archive；另外十二项原归档记录就地更新，没有复制新 ID。`partially_adopted` 只保留广义迁移等未支持范围，既定工具续接方案不再是等待用户决策的未完成事项。两项 `declined` 保留反对新增机制/重复义务的原决定，相关说明和实例修正分别如实记录。

当前入口已更新：issues README、archive README/TRIAGE 和本轮 README/覆盖表。六份旧批次报告只替换移动后的 issue 导航路径，原 pending、维护和批准结论保持历史时点；覆盖表的 `original_path` 也继续保留原路径，以 `final_path` 定位当前记录。FMHA/Sparse 专项与当前 wiki 索引由 Supervisor 按原分工继续。

## Supervisor 最终验证补记

第四轮报告完成后，稳定代码的25个 extract 测试模块共221项全部通过。汇总见 [FINAL_EXTRACT_TESTS.json](FINAL_EXTRACT_TESTS.json)。此结果补齐报告记录时尚在执行的全套验证；后续 FMHA/Sparse 专项只修改各自 Schema 与审修记录。
